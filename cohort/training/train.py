"""Train the cohort with masked PPO.

Usage:
    python -m cohort.training.train --scenario fireteam --total-steps 500000
    python -m cohort.training.train --scenario squad --n-envs 8 --device cpu

Outputs land in runs/<run-name>/: metrics.csv, training_curves.png,
checkpoints (latest/best), TensorBoard logs, and a post-training eval GIF.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from cohort.config import get_scenario
from cohort.core.missions import MissionType
from cohort.core.world import dist
from cohort.env.actions import N_ACTIONS
from cohort.env.cohort_env import CohortEnv, make_env
from cohort.env.observations import obs_dim
from cohort.env.rewards import COMPONENTS, RewardConfig
from cohort.metrics import (
    COMM_MODEL_MARKER_WAIVERS,
    ROOT_REPORT_CLOSE_FLOOR,
    SUCCESS_RATE_FLOOR,
)
from cohort.training import provenance
from cohort.training.ppo import PolicyNet, PPOConfig, RolloutBuffer, ppo_update, trajectory_config

METRIC_FIELDS = [
    "iteration",
    "env_steps",
    "ep_return",
    "ep_length",
    "success_rate",
    "success_rate_rolling",
    "entropy",
    "policy_loss",
    "value_loss",
    "approx_kl",
    # Optimizer diagnostics (v1.11). Every collapse investigated on this
    # project so far (issues #16-#19, the D4 thread) was diagnosed without a
    # single one of these, which is why the search ran through done_false,
    # contact_redundant, learning rate and observation width before anyone
    # scored the reward economics.
    #   grad_norm           — the pre-clip norm. With unnormalized returns the
    #                         value head held 95-99% of it and max_grad_norm
    #                         then scaled the POLICY update down ~5x.
    #   clipfrac            — fraction of the batch hitting the PPO clip.
    #   explained_variance  — 1 - Var(R-V)/Var(R). Negative means the critic is
    #                         worse than the batch mean and every advantage in
    #                         the update is noise.
    #   value/return scale  — the units the critic is being asked to fit.
    #   epochs_used         — how many of update_epochs ran before target_kl.
    "grad_norm",
    "clipfrac",
    "explained_variance",
    "value_mean",
    "value_std",
    "return_mean",
    "return_std",
    "epochs_used",
    "lr",
    "n_episodes",
    "sps",
    "tx_per_agent_step",
    # refs issue #18: tx counts LEARNED transmissions — every call that reaches
    # `_charge_transmission`, voice included, because #18 made SYNC PROPOSE/GO
    # pay airtime like the rest. messages_per_agent_step counts everything on
    # the transcript, learned or automatic; the pair is the composition.
    #
    # This comment said "voice is free by design, so a policy that stops
    # commanding and starts talking reads as radio silence here" until
    # 2026-08-13. That was the PRE-#18 world and it inverts the instrument: the
    # stall #18 closed would now show up in tx as extra volume, not as silence.
    # messages_per_agent_step counts everything said; the pair is the
    # composition. timeout_rate_rolling is the stall itself, over the same
    # window as success_rate_rolling: a run whose episodes start ending at
    # the step ceiling is farming the clock, and it says so 3M steps before
    # anyone evaluates a checkpoint.
    "messages_per_agent_step",
    "timeout_rate_rolling",
    # refs v1.20: did the COMMANDER close the operation, or did the grace
    # window expire? ckpt_best is selected partly on this now (best_save_gate),
    # and the property is learned late, so the curve is worth having on record.
    "root_report_close_rolling",
    # refs assurance #57: and over HOW MANY episodes. The rate above is
    # success-conditioned, so its denominator is the thing that made a 2%
    # window look like a reporting one — and until now it was the one input to
    # best_save_gate that metrics.csv did not record, which is why
    # checkpoint_selection.py can replay the gate exactly but can only reason
    # about the thin-sample edge as a mechanism.
    "root_report_close_n",
    # lightweight behavioral tracking (B2), over the episodes completed this
    # iteration: fraction whose human root died (issue #9: rolling success is
    # blind to exposure re-learning), and rejected DONE / total DONE claims
    "human_death_rate",
    "false_complete_rate",
    # positional regression gate for DEFEND roots (issue #11): where the unit
    # fights when the enemy is on it. Blank on every other root mission — the
    # per-step scan is the only measurement in this loop that is not already
    # in memory, so it is not paid for by runs the gate does not govern.
    "cover_under_threat",
    "objective_dist_under_threat",
    *[f"comp_{c}" for c in COMPONENTS],
]


def is_reporting(root_report_close: float | None, rolling: float) -> bool:
    """Is the commander closing its own operations in this window?

    None — no ENDEX inside the window — is *unmeasured*, and unmeasured is not
    reporting. It is also not a refusal: see ``best_save_gate``.

    ``rolling`` is the window's rolling success, and a window below
    ``SUCCESS_RATE_FLOOR`` cannot be reporting **whatever rate it shows**
    (refs assurance #57). The reporting rate is conditioned on winning —
    ``recent_root_closed`` is appended only on episodes that sent an ENDEX,
    which ``cohort_env`` sends only in the success branch — so its denominator
    shrinks exactly as success collapses, and a policy winning 2 episodes in
    100 can read 0.500 off one or two of them. Reusing the project's own
    success floor rather than inventing a second threshold keeps this one
    statement: *a window that would not pass the run's own success gate does
    not get to claim the reporting promotion.*
    """
    if root_report_close is None or rolling < SUCCESS_RATE_FLOOR:
        return False
    return root_report_close >= ROOT_REPORT_CLOSE_FLOOR


def best_save_gate(
    episodes_seen: int,
    window: int,
    rolling: float,
    best_so_far: float,
    root_report_close: float | None = None,
    best_was_reporting: bool = False,
    *,
    report_gate_waived: bool = False,
) -> bool:
    """Should the rolling-best checkpoint be (re)written this iteration?

    D4 fix (the fine-tune degeneracy): ``ckpt_best`` may only be written once
    the rolling-outcome window contains ONLY post-start episodes — i.e. the
    deque has fully turned over (``episodes_seen >= window``, its maxlen).
    The old gate (>= 20 episodes in a 100-episode window) let a fine-tune's
    strong parent pin rolling success at ~1.0 within the first ~20 episodes,
    freezing ``ckpt_best`` at ~3-4k steps for the rest of the run (observed
    on fireteam_v4d and squad_v3d/v3e — see ROADMAP D4/A4). Requiring full
    turnover means every eligible save reflects a full window of episodes
    played under *this* run's training, at its statistical resolution.

    **v1.20 (owner's decision): success alone may not select the checkpoint.**
    ``ckpt_best`` was chosen on ``success_rate_rolling`` and nothing else, and
    on this environment the completion report is learned LATE — measured across
    six squad arms, ``ckpt_best`` sat at a closed-on-root-report of 0.00-0.01
    while the FINAL policy of the same run reported normally (0.82-0.92). The
    project publishes the FINAL policy, so no shipped number was ever wrong,
    but ``ckpt_best`` is what ``cohort.play`` and every spot-check load by
    default, and what an outside reader would reasonably take as the run's best
    work. Selecting a mute commander as "best" is the same overstatement the
    gates elsewhere exist to refuse.

    **Selection is lexicographic, not a veto**, because a veto is too brittle to
    ship: refusing every mute save outright leaves a run that never learns to
    report with NO ``ckpt_best`` at all, which fails ``baseline.py``'s
    "every checkpoint loadable" and makes ``publish_baseline`` report a missing
    artifact. Verified on a 120k-step smoke run, which wrote none. So:

    * a reporting window ALWAYS supersedes a mute best, whatever the success
      numbers say — otherwise a mute 0.95 recorded early would lock out the
      reporting 0.90 that follows it, which is the exact inversion this is
      here to prevent;
    * once the best is reporting, a mute window may never take it back, however
      well it scores;
    * among windows of the same kind, higher rolling success wins, as before.

    A run that never reports still gets a ``ckpt_best`` — and is then caught
    where it should be, by ``metrics.regression_gates``'
    ``closed_on_root_report_rate`` at evaluation time. Training prefers; the
    gate refuses.

    ``root_report_close`` is None until the window contains an episode that
    actually sent an ENDEX; unmeasured counts as not-reporting for ordering.

    **v1.21 (refs assurance #57): the promotion requires a window that is
    winning.** The rule above is lexicographic over a rate whose denominator is
    success-conditioned, so as written it let a *collapsing* window outrank a
    working policy: ``patrol_brique_v19_rdb3_seed13`` wrote its only
    ``ckpt_best`` at iteration 25 of 2930 — 25,600 steps of 3,000,320 — off a
    window at **2% rolling success**, whose handful of wins read 0.500, cleared
    the floor, and then locked the absorbing flag against the 99%-success
    policy that followed. ``is_reporting`` now refuses that comparison below
    ``SUCCESS_RATE_FLOOR``. Replayed over the whole corpus with
    ``scripts/checkpoint_selection.py``, this moves **1 of 104 runs** — that
    one, from 0.020 @ iter 25 to 1.000 @ iter 550 — and no other.

    **v1.23 (owner-decided 2026-08-25): ``report_gate_waived`` drops the
    reporting key entirely.** Where a scenario's comm model waives
    ``closed_on_root_report_rate`` (``metrics.COMM_MODEL_MARKER_WAIVERS`` — today
    ``comm_model="jammed"``), the rate measures the net rather than the
    commander, and selection reverts to rolling success alone.

    Waiving the *gate* alone would have left this preference chasing the waived
    quantity, and under jamming that is not neutral. The rate is
    success-conditioned and the outage makes it sparse, so the lexicographic
    rule would promote whichever window happened to catch a close — an artefact
    of when the net was up — and the "a mute window may never take it back"
    clause then makes that artefact absorbing. `squad_jammed_control` seed 13
    shows the same hazard from the other side: it reads exactly 0.000 at every
    checkpoint, so the reporting key never fires and contributes nothing but the
    risk. Lifting the requirement while keeping the preference would have been
    the incoherent half-measure.
    """
    if episodes_seen < window:
        return False
    if report_gate_waived:
        # The comm model puts the reporting rate out of reach, so it carries no
        # information about which window holds the better policy — fall back to
        # rolling success alone.
        return rolling > best_so_far
    reporting = is_reporting(root_report_close, rolling)
    if reporting != best_was_reporting:
        return reporting  # the first reporting window wins; a mute one cannot
    return rolling > best_so_far


def collapse_stop_gate(
    streak: int,
    rolling: float,
    peak: float,
    *,
    window_full: bool,
    floor: float,
    margin: float,
    patience: int,
) -> tuple[int, bool]:
    """Should the run end here because its policy has collapsed?

    The D4 passive attractor (ROADMAP 2026-08-19) captures a converged policy
    and holds it: rolling success falls from a learned peak to ~0 and never
    returns, while the run spends its remaining budget entrenching the
    collapsed policy — the platoon_hard cycle put 6/6 hierarchy runs there
    from peaks of 75-93%, each finishing at 0/300. ``ckpt_best`` preserves
    the peak either way; what stopping buys is that ``ckpt_latest`` (the
    "final policy" every honest publication must also score) stops drifting
    further from it, and the compute goes back to the queue.

    Pure so scripts/collapse_replay.py can run the exact shipped rule over
    any metrics.csv. The guard arms only once the run has recorded a
    full-window rolling success of at least ``floor`` (a run that never
    learned has nothing to protect), then counts consecutive iterations at
    or below ``peak - margin``; ``patience`` of them ends the run. A single
    window back above the line resets the count, which is what spares
    dip-and-recover runs (platoon_hard_flat seed 12 recovered from a
    mid-run dip to finish at 91%). ``patience <= 0`` disables the guard.
    """
    if patience <= 0 or not window_full or peak < floor:
        return 0, False
    if rolling <= peak - margin:
        streak += 1
    else:
        streak = 0
    return streak, streak >= patience


def rescue_gate(
    streak: int,
    rescues_used: int,
    *,
    patience: int,
    max_rescues: int,
) -> bool:
    """Should the run roll back to ckpt_best here, instead of dying later?

    Companion to ``collapse_stop_gate`` and fed the same streak: the number
    of consecutive iterations rolling success has spent at or below
    ``peak - collapse_margin``. The stop ends a captured run to stop wasting
    budget; the rescue (``Trainer.rescue``) spends that budget on another
    attempt from the best policy the run ever had. It must fire before the
    stop to fire at all, so ``rescue_patience < collapse_patience`` — the
    defaults (700 vs 1200) keep that ordering and a config that breaks it
    simply never rescues. Pure, like the stop gate, so
    scripts/collapse_replay.py can replay the pair over any metrics.csv.
    """
    return max_rescues > 0 and rescues_used < max_rescues and 0 < patience <= streak


def _load_compatible(net: PolicyNet, state: dict) -> list[str]:
    """Load ``state`` into ``net``, tolerating only the v1.11 additions.

    A pre-v1.11 checkpoint has no ``critic_torso.*`` (shared torso) and no
    ``value_norm.*`` (no return normalization). Curriculum-initializing a
    v1.11 network from one is legitimate — the actor transfers, the new critic
    starts fresh — but "tolerate missing keys" must not become a blanket
    ``strict=False`` that silently swallows a real architecture mismatch. So
    only those two prefixes may be absent; anything else still raises.
    """
    missing, unexpected = net.load_state_dict(state, strict=False)
    allowed = ("critic_torso.", "value_norm.")
    hard = [k for k in missing if not k.startswith(allowed)]
    if hard or unexpected:
        msg = (
            f"checkpoint does not match this network: missing {hard}, "
            f"unexpected {list(unexpected)}"
        )
        raise RuntimeError(msg)
    return list(missing)


class Trainer:
    """Vectorized rollout collection + PPO updates for CohortEnv."""

    def __init__(
        self,
        scenario: str,
        cfg: PPOConfig,
        run_dir: Path,
        seed: int = 1,
        *,
        tensorboard: bool = True,
        init_from: str | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.scenario = scenario
        self.reward_config = reward_config or RewardConfig()

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.envs: list[CohortEnv] = [
            make_env(scenario, reward_config=self.reward_config) for _ in range(cfg.n_envs)
        ]
        self.agent_ids = list(self.envs[0].possible_agents)
        self.slot = {a: i for i, a in enumerate(self.agent_ids)}
        self.n_agents = len(self.agent_ids)
        self.current_obs: list[dict] = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=seed + i * 1000)
            self.current_obs.append(obs)

        self.device = torch.device(cfg.device)
        # read the width off the env, not the module: a scenario may present
        # the pre-v1.10 `core` observation (ScenarioSpec.observation_profile)
        self.obs_dim = obs_dim(self.envs[0].spec_cfg.observation_profile)
        self.net = PolicyNet(
            self.obs_dim,
            N_ACTIONS,
            cfg.hidden,
            separate_critic=cfg.separate_critic,
            normalize_value=cfg.normalize_value,
        ).to(self.device)
        if init_from is not None:
            ckpt = torch.load(init_from, map_location=self.device, weights_only=True)
            _load_compatible(self.net, ckpt["model"])
            print(f"initialized weights from {init_from} (scenario {ckpt.get('scenario')}, {ckpt.get('env_steps')} steps)")
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)

        # issue #11: the positional gate governs DEFEND roots, so only those
        # runs pay for the per-step disposition scan.
        spec = self.envs[0].spec_cfg
        self._track_disposition = spec.root_mission is MissionType.DEFEND
        # Does this scenario's comm model waive the root-report gate? If so the
        # same quantity is dropped from checkpoint SELECTION too (best_save_gate).
        # Read from the same table the gate reads, so the two can never disagree
        # about which scenarios are waived.
        self._report_gate_waived = "closed_on_root_report_rate" in (
            COMM_MODEL_MARKER_WAIVERS.get(spec.comm_model, {})
        )
        self._threat_radius = float(spec.combat.weapon_range)

        self.env_steps = 0
        self.iteration = 0
        self.recent_outcomes: deque[str] = deque(maxlen=100)
        # refs v1.20: did the commander's own report close the operation? Same
        # definition as metrics._endex_close — denominator is episodes that
        # actually sent an ENDEX, so success drift does not move it.
        self.recent_root_closed: deque[bool] = deque(maxlen=100)
        self.episodes_seen = 0  # episodes completed since training start (D4 best-gate)
        self.best_rolling_success = -1.0
        # D4 collapse stop: the highest FULL-window rolling success seen, and
        # the current run of iterations spent >= collapse_margin below it.
        # Tracked apart from best_rolling_success, which is not monotone (a
        # reporting window may supersede a higher mute one, see best_save_gate).
        self.peak_rolling_success = -1.0
        self.collapse_streak = 0
        self.rescues_used = 0  # D4 rescue: rollbacks performed so far (see rescue())
        # refs v1.20: whether the recorded ckpt_best came from a window whose
        # commander was reporting. A mute window may never take it back.
        self.best_was_reporting = False
        self._ep_return = [0.0] * cfg.n_envs
        self._ep_len = [0] * cfg.n_envs

        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(run_dir / "tb"))
        self.metrics_path = run_dir / "metrics.csv"
        if not self.metrics_path.exists():
            with self.metrics_path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=METRIC_FIELDS).writeheader()

    # ------------------------------------------------------------------ #

    def _forward_present(
        self, rows: list[tuple[int, str]], *, greedy: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = np.stack([self.current_obs[e][a]["observation"] for e, a in rows])
        mask = np.stack([self.current_obs[e][a]["action_mask"] for e, a in rows])
        t_obs = torch.as_tensor(obs, device=self.device)
        t_mask = torch.as_tensor(mask, device=self.device)
        action, logp, value = self.net.act(t_obs, t_mask, greedy=greedy)
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()

    @torch.no_grad()
    def _values_of(self, obs_list: list[dict]) -> np.ndarray:
        obs = torch.as_tensor(np.stack([o["observation"] for o in obs_list]), device=self.device)
        mask = torch.as_tensor(np.stack([o["action_mask"] for o in obs_list]), device=self.device)
        _, value = self.net.dist_value(obs, mask)
        return value.cpu().numpy()

    def _disposition(self, env: CohortEnv) -> tuple[int, int, float]:
        """(threatened agent-steps, of which in cover, summed distance to OBJ).

        The training-time half of the positional gate (issue #11): the same
        population ``cohort.metrics._fight_disposition`` scores at eval time —
        living soldiers with a living enemy inside weapon range — read
        straight off the live environment. Read-only; consumes no randomness.
        """
        enemies = [e.pos for e in env.enemies if e.alive]
        if not enemies:
            return 0, 0, 0.0
        obj = env.world.objective_by_name(env.spec_cfg.root_objective or "")
        pairs = 0
        cover = 0
        obj_dist = 0.0
        for s in env.roster.soldiers:
            if not s.alive or min(dist(s.pos, p) for p in enemies) > self._threat_radius:
                continue
            pairs += 1
            cover += env.world.cover_at(s.pos)
            if obj is not None:
                obj_dist += dist(s.pos, obj.pos)
        return pairs, cover, obj_dist

    def collect(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Fill the buffer with cfg.horizon steps from every env."""
        cfg = self.cfg
        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        outcomes: list[str] = []
        comp_sums = dict.fromkeys(COMPONENTS, 0.0)
        agent_steps = 0
        tx_total = 0
        message_total = 0
        human_deaths = 0
        done_claims = 0
        done_rejected = 0
        threat_pairs = 0
        threat_cover = 0
        threat_obj_dist = 0.0

        for t in range(cfg.horizon):
            rows = [(e, a) for e, env in enumerate(self.envs) for a in env.agents]
            if rows:
                actions, logps, values = self._forward_present(rows)
            row_of = {key: i for i, key in enumerate(rows)}

            for e, env in enumerate(self.envs):
                present = list(env.agents)
                if not present:
                    continue
                act_dict = {a: int(actions[row_of[(e, a)]]) for a in present}
                obs_next, rewards, terms, truncs, infos = env.step(act_dict)

                trunc_rows = []
                for a in present:
                    i = row_of[(e, a)]
                    s = self.slot[a]
                    buffer.obs[t, e, s] = self.current_obs[e][a]["observation"]
                    buffer.masks[t, e, s] = self.current_obs[e][a]["action_mask"]
                    buffer.actions[t, e, s] = actions[i]
                    buffer.logprobs[t, e, s] = logps[i]
                    buffer.values[t, e, s] = values[i]
                    buffer.rewards[t, e, s] = rewards[a]
                    buffer.dones[t, e, s] = float(terms[a] or truncs[a])
                    buffer.valid[t, e, s] = True
                    if truncs[a] and not terms[a]:
                        trunc_rows.append((a, s))
                    for comp, val in infos[a]["components"].items():
                        comp_sums[comp] += val
                    self._ep_return[e] += rewards[a]
                # LIVING agent-steps only. Since v1.11 the fallen stay in the
                # episode (to be paid the team terminal), so counting every
                # present agent here would silently dilute every per-agent-step
                # figure in metrics.csv — the reward components, tx rate and
                # message rate — by however many casualties a run takes, and
                # nothing in the record would be comparable across the change.
                agent_steps += sum(1 for a in present if self.envs[e].roster.by_callsign[a].alive)
                tx_total += env.transmissions_last_step
                # every message, not only the charged ones (refs #18)
                message_total += len(env.last_messages)
                self._ep_len[e] += 1
                if self._track_disposition:
                    pairs, cover, obj_dist = self._disposition(env)
                    threat_pairs += pairs
                    threat_cover += cover
                    threat_obj_dist += obj_dist

                # truncation: bootstrap the final state's value into the reward
                if trunc_rows:
                    vals = self._values_of([obs_next[a] for a, _ in trunc_rows])
                    for (_a, s), v in zip(trunc_rows, vals, strict=True):
                        buffer.rewards[t, e, s] += cfg.gamma * float(v)

                if env.agents:
                    self.current_obs[e] = obs_next
                else:  # episode over
                    ep_returns.append(self._ep_return[e] / self.n_agents)
                    ep_lengths.append(self._ep_len[e])
                    outcomes.append(env.outcome or "timeout")
                    self.recent_outcomes.append(outcomes[-1])
                    self.episodes_seen += 1
                    # B2 behavioral tracking, from state already in memory
                    human_deaths += any(s.human and not s.alive for s in env.roster.soldiers)
                    sent_endex = False
                    for m in env.transcript.messages:
                        done_claims += m.kind.value == "done"
                        done_rejected += m.kind.value == "done_reject"
                        sent_endex |= m.kind.value == "endex"
                    if sent_endex:
                        self.recent_root_closed.append(env.root_close_step is not None)
                    self._ep_return[e] = 0.0
                    self._ep_len[e] = 0
                    obs0, _ = env.reset()
                    self.current_obs[e] = obs0
            self.env_steps += cfg.n_envs

        # bootstrap values for streams still alive at the horizon boundary
        next_values = np.zeros((cfg.n_envs, self.n_agents), dtype=np.float32)
        next_valid = np.zeros((cfg.n_envs, self.n_agents), dtype=bool)
        rows = [(e, a) for e, env in enumerate(self.envs) for a in env.agents]
        if rows:
            vals = self._values_of([self.current_obs[e][a] for e, a in rows])
            for (e, a), v in zip(rows, vals, strict=True):
                next_values[e, self.slot[a]] = v
                next_valid[e, self.slot[a]] = True
        self._bootstrap = (next_values, next_valid)

        n_eps = max(1, len(ep_returns))
        stats = {
            "n_episodes": len(ep_returns),
            "ep_return": float(np.mean(ep_returns)) if ep_returns else float("nan"),
            "ep_length": float(np.mean(ep_lengths)) if ep_lengths else float("nan"),
            "success_rate": sum(o == "success" for o in outcomes) / n_eps if outcomes else 0.0,
            "success_rate_rolling": (
                sum(o == "success" for o in self.recent_outcomes) / len(self.recent_outcomes)
                if self.recent_outcomes
                else 0.0
            ),
            # refs v1.20: the share of closed operations the COMMANDER closed.
            # Blank (nan) until an ENDEX has been sent inside the window —
            # unmeasured, which best_save_gate treats as "does not block".
            "root_report_close_rolling": (
                sum(self.recent_root_closed) / len(self.recent_root_closed)
                if self.recent_root_closed
                else float("nan")
            ),
            "root_report_close_n": len(self.recent_root_closed),
            # refs #18: the same window, asking how the episodes were LOST.
            # A rising clock-expiry rate is the stall signature at training
            # time — squad_screen_v4 and squad_recon_v6 each spent their last
            # million steps at 1.0 here while nothing else in the row said so.
            "timeout_rate_rolling": (
                sum(o == "timeout" for o in self.recent_outcomes) / len(self.recent_outcomes)
                if self.recent_outcomes
                else 0.0
            ),
        }
        stats["tx_per_agent_step"] = tx_total / max(1, agent_steps)
        stats["messages_per_agent_step"] = message_total / max(1, agent_steps)
        stats["human_death_rate"] = human_deaths / n_eps if outcomes else 0.0
        stats["false_complete_rate"] = done_rejected / max(1, done_claims)
        if self._track_disposition:
            # NaN, not 0, when nothing was threatened this iteration: "no
            # firefight" must not read as "fought in the open on the objective"
            stats["cover_under_threat"] = (
                threat_cover / threat_pairs if threat_pairs else float("nan")
            )
            stats["objective_dist_under_threat"] = (
                threat_obj_dist / threat_pairs if threat_pairs else float("nan")
            )
        for comp in COMPONENTS:
            stats[f"comp_{comp}"] = comp_sums[comp] / max(1, agent_steps)
        return stats

    # ------------------------------------------------------------------ #

    def train(self, total_steps: int) -> None:
        """Main loop: collect → GAE → update → log, until total_steps."""
        cfg = self.cfg
        start = time.time()
        while self.env_steps < total_steps:
            self.iteration += 1
            lr_now = cfg.lr
            if cfg.anneal_lr:
                frac = 1.0 - min(1.0, self.env_steps / total_steps)
                lr_now = cfg.lr * max(0.05, frac)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr_now

            buffer = RolloutBuffer(cfg.horizon, cfg.n_envs, self.n_agents, self.obs_dim, N_ACTIONS)
            t0 = time.time()
            stats = self.collect(buffer)
            next_values, next_valid = self._bootstrap
            advantages, returns = buffer.compute_gae(next_values, next_valid, cfg.gamma, cfg.gae_lambda)
            losses = ppo_update(self.net, self.optimizer, buffer, advantages, returns, cfg)
            sps = cfg.horizon * cfg.n_envs / max(1e-9, time.time() - t0)

            row = {
                "iteration": self.iteration,
                "env_steps": self.env_steps,
                "sps": round(sps),
                "lr": round(lr_now, 8),
                **{k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()},
                **{k: round(v, 6) for k, v in losses.items()},
            }
            with self.metrics_path.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore").writerow(row)
            if self.writer is not None:
                for key in (
                    "ep_return", "success_rate_rolling", "entropy", "policy_loss",
                    "value_loss", "grad_norm", "clipfrac", "explained_variance",
                ):
                    val = row.get(key, stats.get(key, losses.get(key)))
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        self.writer.add_scalar(key, float(val), self.env_steps)

            if self.iteration % 5 == 0 or self.env_steps >= total_steps:
                elapsed = time.time() - start
                print(
                    f"iter {self.iteration:>4} | steps {self.env_steps:>8,} | "
                    f"return {stats['ep_return']:>7.2f} | success {stats['success_rate_rolling']:.0%} | "
                    f"len {stats['ep_length']:>5.0f} | ent {losses['entropy']:.2f} | "
                    f"ev {losses['explained_variance']:>5.2f} | gnorm {losses['grad_norm']:.2f} | "
                    f"sps {sps:>5.0f} | {elapsed:>5.0f}s"
                )
            self.save_checkpoint("ckpt_latest.pt")
            root_close = (
                sum(self.recent_root_closed) / len(self.recent_root_closed)
                if self.recent_root_closed
                else None
            )
            if best_save_gate(
                self.episodes_seen,
                self.recent_outcomes.maxlen or 0,
                stats["success_rate_rolling"],
                self.best_rolling_success,
                root_close,
                self.best_was_reporting,
                report_gate_waived=self._report_gate_waived,
            ):
                self.best_rolling_success = stats["success_rate_rolling"]
                self.best_was_reporting = is_reporting(
                    root_close, stats["success_rate_rolling"]
                )
                self.save_checkpoint("ckpt_best.pt")

            # D4 collapse stop — see collapse_stop_gate. The peak is tracked
            # here and not inside the gate so the gate stays pure (replayable
            # over any metrics.csv by scripts/collapse_replay.py).
            window_full = self.episodes_seen >= (self.recent_outcomes.maxlen or 0)
            if window_full:
                self.peak_rolling_success = max(
                    self.peak_rolling_success, stats["success_rate_rolling"]
                )
            self.collapse_streak, collapsed = collapse_stop_gate(
                self.collapse_streak,
                stats["success_rate_rolling"],
                self.peak_rolling_success,
                window_full=window_full,
                floor=cfg.collapse_floor,
                margin=cfg.collapse_margin,
                patience=cfg.collapse_patience,
            )
            if rescue_gate(
                self.collapse_streak,
                self.rescues_used,
                patience=cfg.rescue_patience,
                max_rescues=cfg.rescue_max,
            ) and self.rescue(stats["success_rate_rolling"]):
                collapsed = False
            if collapsed:
                # The marker is what train_status.py reads to call the run
                # EARLY-STOP rather than STOPPED (which reads as a crash).
                (self.run_dir / "early_stop.json").write_text(
                    json.dumps(
                        {
                            "reason": "collapse",
                            "env_steps": self.env_steps,
                            "iteration": self.iteration,
                            "rolling_success": stats["success_rate_rolling"],
                            "peak_rolling_success": self.peak_rolling_success,
                            "patience": cfg.collapse_patience,
                            "margin": cfg.collapse_margin,
                            "floor": cfg.collapse_floor,
                        },
                        indent=2,
                    )
                )
                print(
                    f"COLLAPSE STOP at step {self.env_steps:,} (iter {self.iteration}): "
                    f"rolling {stats['success_rate_rolling']:.0%} has sat >= "
                    f"{cfg.collapse_margin:.0%} below peak {self.peak_rolling_success:.0%} "
                    f"for {cfg.collapse_patience} iterations — the D4 attractor. "
                    f"ckpt_best holds the peak; ending the run here."
                )
                break
        if self.writer is not None:
            self.writer.close()

    def rescue(self, rolling: float) -> bool:
        """Roll a captured run back to its best policy and keep training.

        Restores ckpt_best's weights (value_norm statistics ride along in the
        state dict, so the critic stays consistent with its returns), rebuilds
        the optimizer from scratch — Adam's second-moment estimates encode the
        migration's direction, and a restored policy stepping with the old
        moments resumes the same walk — and tightens ``target_kl`` by
        ``rescue_kl_scale`` (compounding across rescues). The rolling windows
        are deliberately NOT cleared: they refill with the restored policy's
        episodes within ~50 iterations, a tax the 700-iteration patience
        absorbs, and clearing them would blind best_save_gate and the peak
        tracker to real history. Each event appends to ``rescues.json`` so the
        run's record shows every rollback, not just the final curve.

        Returns False (no rescue performed) if ckpt_best does not exist —
        possible in principle while best_save_gate withholds writes — in which
        case the collapse stop proceeds as if the rescue never existed.
        """
        best = self.run_dir / "ckpt_best.pt"
        if not best.exists():
            return False
        ckpt = torch.load(best, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["model"])
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        if self.cfg.target_kl is not None:
            self.cfg.target_kl *= self.cfg.rescue_kl_scale
        self.collapse_streak = 0
        self.rescues_used += 1
        events_path = self.run_dir / "rescues.json"
        events = json.loads(events_path.read_text()) if events_path.exists() else []
        events.append(
            {
                "rescue": self.rescues_used,
                "iteration": self.iteration,
                "env_steps": self.env_steps,
                "rolling_success": rolling,
                "peak_rolling_success": self.peak_rolling_success,
                "restored_from_iteration": ckpt.get("iteration"),
                "target_kl_after": self.cfg.target_kl,
            }
        )
        events_path.write_text(json.dumps(events, indent=2))
        print(
            f"RESCUE {self.rescues_used}/{self.cfg.rescue_max} at step {self.env_steps:,} "
            f"(iter {self.iteration}): rolling {rolling:.0%} vs peak "
            f"{self.peak_rolling_success:.0%} — restored ckpt_best "
            f"(iter {ckpt.get('iteration')}), fresh optimizer, "
            f"target_kl now {self.cfg.target_kl}"
        )
        return True

    def save_checkpoint(self, name: str) -> Path:
        """Persist model weights + everything needed to reload them."""
        path = self.run_dir / name
        torch.save(
            {
                "model": self.net.state_dict(),
                "obs_dim": self.obs_dim,
                "n_actions": N_ACTIONS,
                "hidden": self.cfg.hidden,
                # v1.11 architecture flags. Absent on every earlier checkpoint,
                # and load_policy defaults them to False, so the published
                # fleet keeps reconstructing its exact original network.
                "separate_critic": self.cfg.separate_critic,
                "normalize_value": self.cfg.normalize_value,
                "scenario": self.scenario,
                "iteration": self.iteration,
                "env_steps": self.env_steps,
                "ppo_config": asdict(self.cfg),
                # v1.12: the PRICES this policy learned under. Absent on every
                # earlier checkpoint, where the defaults were the only prices
                # there were — so the published fleet is unaffected. Stored so
                # evaluate() can score a run under its own economics: with
                # rewards on the CLI, a checkpoint and RewardConfig() are no
                # longer the same thing, and `mean_return` measured against the
                # wrong prices is a number that looks comparable and is not.
                "reward_config": asdict(self.reward_config),
            },
            path,
        )
        return path


def load_policy(checkpoint: str | Path, device: str = "cpu") -> tuple[PolicyNet, dict]:
    """Load a trained policy from a checkpoint file.

    The architecture is reconstructed from flags stored in the file, which
    default to the pre-v1.11 shape when absent — so every checkpoint the fleet
    published before the split critic landed still loads exactly as it trained.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    net = PolicyNet(
        ckpt["obs_dim"],
        ckpt["n_actions"],
        ckpt["hidden"],
        separate_critic=ckpt.get("separate_critic", False),
        normalize_value=ckpt.get("normalize_value", False),
    ).to(device)
    # ValueNorm's buffers exist even when it is disabled, so a pre-v1.11 file
    # is missing exactly those three keys and nothing else.
    _load_compatible(net, ckpt["model"])
    net.eval()
    return net, ckpt


def _git_commit() -> str | None:
    """HEAD at launch, so a run dir can be tied back to the code that made it.

    Re-exported from :mod:`cohort.training.provenance` since #39, which needed
    the same thing in ``evaluate.py``. Kept under this name because
    ``economics.json:git_commit`` means "the tree this RUN trained against" and
    that meaning must not drift when the helper moves.
    """
    return provenance.git_commit()


def _spec_economics(scenario: str) -> dict:
    """The scenario knobs that change what a policy is being paid to do."""
    spec = get_scenario(scenario)
    keys = (
        "root_mission", "root_objective", "max_steps", "grace_window",
        "done_cooldown", "order_cooldown", "assault_h_hour", "defend_horizon",
        "sitrep_cadence",
        "ablation", "opfor_mode", "comm_model", "n_enemies",
        # degraded communications (§8 provenance): the run must say which
        # communications/acoustic regime produced it
        "sound_model", "voice_range", "comm_range", "liaison_enabled",
    )
    return {k: getattr(spec, k, None) for k in keys}


def _warn_if_stalling_pays(rewards: RewardConfig, scenario: str, spec, cfg_gamma: float) -> None:
    """Shout if the requested prices make stalling competitive with winning.

    This is the v1.11 finding turned into a pre-flight check. The invariant
    that decides whether a run collapses is DISCOUNTED terminal dominance, and
    it is now reachable from the CLI: `--reward success_team=10` is one
    keystroke away from the economics that produced 21 collapsed runs. On a
    DEFEND/DENY scenario the bar is checked at the WORST-CASE terminal, since
    `defend_survivor_scale` can only scale the payout down.

    Deliberately a warning, not an error: ablating the economics below the bar
    is a legitimate experiment here (`test_the_discounted_invariant_would_have
    _caught_the_shipped_collapse` depends on being able to express it). What is
    not legitimate is doing it by accident and reading the wreckage as a
    finding about something else — so it goes to stdout, which train.sh tees
    into the run's log where a post-mortem will find it.
    """
    scale = rewards.terminal_scale_floor(spec.root_mission)
    ratio = rewards.win_beats_stall(cfg_gamma, spec.max_steps, terminal_scale=scale)
    if ratio >= 2.0:
        return
    worst = " (worst case: whole force lost)" if scale < 1.0 else ""  # defend scaling
    print(
        f"\n!! WARNING: on {scenario}, winning is worth {ratio:.2f}x stalling"
        f"{worst} at gamma={cfg_gamma} — the bar is 2.0.\n"
        f"!! Below ~1.0 the policy is expected to learn to farm shaping and "
        f"never finish (v1.11: 21/69 runs). Proceeding anyway.\n"
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the cohort with masked PPO.")
    parser.add_argument("--scenario", default="fireteam", help="scenario preset name")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    # The rest of PPOConfig, exposed (v1.11). These were code-only defaults, so
    # a campaign could not sweep them without editing the tree mid-run — and
    # `gamma`, the one that turned out to decide whether a run collapses, was
    # among them. Defaults mirror PPOConfig, so omitting a flag changes nothing.
    parser.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    parser.add_argument("--gae-lambda", type=float, default=PPOConfig.gae_lambda)
    parser.add_argument("--clip-coef", type=float, default=PPOConfig.clip_coef)
    parser.add_argument("--vf-coef", type=float, default=PPOConfig.vf_coef)
    parser.add_argument("--max-grad-norm", type=float, default=PPOConfig.max_grad_norm)
    parser.add_argument("--update-epochs", type=int, default=PPOConfig.update_epochs)
    parser.add_argument("--num-minibatches", type=int, default=PPOConfig.num_minibatches)
    parser.add_argument("--target-kl", type=float, default=PPOConfig.target_kl,
                        help="early-stop the update epochs above this approx KL (<=0 disables)")
    parser.add_argument("--collapse-patience", type=int, default=PPOConfig.collapse_patience,
                        help="end the run after this many consecutive iterations spent "
                             ">= collapse-margin below the peak rolling success "
                             "(the D4 attractor; 0 disables — do that when the collapse "
                             "IS the experiment)")
    parser.add_argument("--collapse-margin", type=float, default=PPOConfig.collapse_margin,
                        help="how far below the peak counts as collapsed")
    parser.add_argument("--collapse-floor", type=float, default=PPOConfig.collapse_floor,
                        help="the guard arms only once peak rolling success reaches this")
    parser.add_argument("--rescue-max", type=int, default=PPOConfig.rescue_max,
                        help="roll a captured run back to ckpt_best (fresh optimizer, "
                             "tightened target-kl) up to this many times before the "
                             "collapse stop is allowed to end it (0 disables, the default)")
    parser.add_argument("--rescue-patience", type=int, default=PPOConfig.rescue_patience,
                        help="iterations spent >= collapse-margin below the peak before "
                             "a rescue fires; must stay below collapse-patience to fire at all")
    parser.add_argument("--rescue-kl-scale", type=float, default=PPOConfig.rescue_kl_scale,
                        help="multiply target-kl by this on every rescue (compounding)")
    parser.add_argument("--hidden", type=int, default=PPOConfig.hidden)
    parser.add_argument("--normalize-value", action=argparse.BooleanOptionalAction,
                        default=PPOConfig.normalize_value,
                        help="fit the critic against standardized returns")
    parser.add_argument("--separate-critic", action=argparse.BooleanOptionalAction,
                        default=PPOConfig.separate_critic,
                        help="give the critic its own torso and its own gradient clip")
    # Reward weights, exposed (v1.12). Repeatable KEY=VALUE against any
    # RewardConfig field: `--reward done_false=-2.0 --reward death=-3.0`.
    # Until this existed, an experiment about a price meant editing the tree,
    # which is both unrecorded and unsafe mid-campaign.
    parser.add_argument("--reward", action="append", default=[], metavar="KEY=VALUE",
                        help="override a RewardConfig weight; repeatable")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--init-from", default=None, help="checkpoint to initialize weights from (curriculum)")
    parser.add_argument("--no-tb", action="store_true", help="disable TensorBoard logging")
    parser.add_argument("--no-eval", action="store_true", help="skip post-training eval + GIF")
    args = parser.parse_args()

    spec = get_scenario(args.scenario)  # fail fast on typos
    try:  # ...and on bad prices, as a usage error rather than a traceback
        rewards = RewardConfig.from_overrides(args.reward, base=RewardConfig.from_scenario(spec))
    except ValueError as exc:
        parser.error(str(exc))
    if args.reward:
        print(f"reward overrides: {' '.join(args.reward)}")
    _warn_if_stalling_pays(rewards, args.scenario, spec, cfg_gamma=args.gamma)
    run_name = args.run_name or f"{args.scenario}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_name
    cfg = PPOConfig(
        n_envs=args.n_envs,
        horizon=args.horizon,
        lr=args.lr,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        target_kl=args.target_kl if args.target_kl and args.target_kl > 0 else None,
        collapse_patience=args.collapse_patience,
        collapse_margin=args.collapse_margin,
        collapse_floor=args.collapse_floor,
        rescue_patience=args.rescue_patience,
        rescue_max=args.rescue_max,
        rescue_kl_scale=args.rescue_kl_scale,
        hidden=args.hidden,
        normalize_value=args.normalize_value,
        separate_critic=args.separate_critic,
        device=args.device,
    )
    print(f"training scenario={args.scenario} → {run_dir}")
    (run_dir / ".").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        # Only the fields that determine what the run learns: the shape rules
        # (collapse guard out, rescue knobs only when enabled) live in
        # trajectory_config, shared with the preflight duplicate matcher.
        json.dumps({"scenario": args.scenario, "seed": args.seed, "total_steps": args.total_steps,
                    **trajectory_config(cfg)},
                   indent=2)
    )
    # Reward and scenario economics are what a run is actually an experiment
    # ABOUT, and they live in code defaults rather than CLI flags — so without
    # this a run directory cannot say which prices produced it, and two runs a
    # reward commit apart are indistinguishable after the fact. Written next to
    # config.json rather than inside it so the PPO hyper-parameter block that
    # every existing reader parses keeps its exact shape.
    (run_dir / "economics.json").write_text(
        json.dumps(
            {
                "scenario": args.scenario,
                "git_commit": _git_commit(),
                # The prices this run ACTUALLY used, not the tree's defaults —
                # with --reward those diverge, and the confound audit in
                # run_report.py --vs diffs this file to decide whether an A/B
                # is single-variable. A defaults dump here would make every
                # override-driven experiment read as a no-op change.
                "rewards": asdict(rewards),
                "reward_overrides": list(args.reward),
                "spec": _spec_economics(args.scenario),
            },
            indent=2,
            default=str,
        )
    )
    # Import every post-training entry point BEFORE the long run starts, so the
    # process holds ONE consistent snapshot of the code for its whole life.
    # These used to be imported lazily at the end, which meant a run that began
    # at commit A and finished after commit B landed would mix A's
    # already-imported modules with B's freshly-read ones. fireteam_defend_v10
    # died exactly that way — it started before `is_done_admissible` existed, so
    # its in-memory cohort.env.actions had no such name, and the newer
    # cohort.metrics it then read off disk failed to import it. Three and a half
    # million steps produced no evaluation. Editing the tree during a run is
    # normal here; losing a finished run to it is not.
    # Hoisting the ENTRY POINTS is not enough, and assuming it was cost three
    # more runs on 2026-08-06: evaluate() defers `cohort.metrics` to call time
    # (evaluate.py:135, :168), so metrics.py landed outside this snapshot and was
    # still read fresh off disk at the end of training. It imports
    # `order_options` / `is_done_admissible` from cohort.env.actions at module
    # level, so squad_v7, squad_recon_v6 and platoon_v4 each finished their full
    # step budget and then died importing a name their in-memory actions did not
    # have. Import the deferred modules HERE, by hand: the snapshot has to cover
    # what the entry points reach, not just the entry points.
    # That correction was itself still one level too shallow. "What the entry
    # points reach" is a TRANSITIVE property: cohort.metrics defers
    # cohort.env.cohort_env, which defers cohort.core.oracle, which no run has
    # ever held in memory — measured 2026-08-07, `cohort.core.oracle` is absent
    # from sys.modules after this whole block runs. Nothing on today's artifact
    # path calls env.oracle(), so no run has died of it yet; the moment an
    # oracle-backed behavior metric joins evaluate() — which is where this
    # repo's diagnose-first rule keeps pointing — it would resume killing runs
    # at 3M steps apiece. The invariant is therefore stated over the CLOSURE:
    # nothing reachable from the snapshot may be read fresh off disk later.
    # test_import_snapshot.py computes that closure and fails if it is open.
    import cohort.core  # for the closure below, not for a name used here
    import cohort.core.acoustics  # deferred by cohort.config (briefing)
    import cohort.core.language  # deferred by cohort.config
    import cohort.core.liaison  # deferred by cohort.config (briefing)
    import cohort.core.oracle  # deferred by CohortEnv.oracle()
    import cohort.metrics  # imported for the snapshot, not for a name used here
    import cohort.viz.render  # noqa: F401  # same; reached only when --gif is set
    from cohort.training.evaluate import evaluate
    from cohort.viz.plots import plot_training

    trainer = Trainer(
        args.scenario, cfg, run_dir, seed=args.seed, tensorboard=not args.no_tb,
        init_from=args.init_from, reward_config=rewards,
    )
    trainer.train(args.total_steps)

    # Post-training artifacts are INDEPENDENT: a run that spent 40 minutes of
    # CPU must not lose its evaluation because the plotter tripped over a
    # column. Each step is attempted, failures are collected and reported
    # together, and the process still exits non-zero so train_status.py keeps
    # calling the run what it is.
    failures: list[tuple[str, BaseException]] = []

    def artifact(name: str, fn) -> None:
        try:
            fn()
        except BaseException as exc:
            failures.append((name, exc))
            traceback.print_exc()
            print(f"post-training artifact FAILED: {name} ({type(exc).__name__}: {exc})")

    def _curves() -> None:
        print(f"curves → {plot_training(run_dir)}")

    def _eval() -> None:
        ckpt = run_dir / ("ckpt_best.pt" if (run_dir / "ckpt_best.pt").exists() else "ckpt_latest.pt")
        evaluate(
            str(ckpt),
            episodes=20,
            gif_path=str(run_dir / "eval.gif"),
            transcript_path=str(run_dir / "eval_transcript.txt"),
        )

    def _eval_final() -> None:
        """Score the policy the run ENDED with, not only its best window.

        ``ckpt_best`` captures the best rolling window over the whole run, so
        on an unstable run it measures a peak. Measured across 18 runs the gap
        between peak rolling success and the published N=100 number averages
        +8.2 points, and three published policies — squad_recon_v5/v6 and
        squad_v7 — come from runs whose rolling success ENDED AT 0.00/0.41.
        Those numbers are real for that checkpoint and say nothing about
        whether the recipe reproduces. A run cannot be honestly published
        without both, so both are now measured, always, by default.
        """
        latest = run_dir / "ckpt_latest.pt"
        if not latest.exists() or not (run_dir / "ckpt_best.pt").exists():
            return  # only one checkpoint: _eval already scored it
        evaluate(
            str(latest),
            episodes=20,
            behavior_path=str(run_dir / "behavior_final.json"),
        )

    artifact("training_curves.png", _curves)
    if not args.no_eval:
        artifact("evaluate", _eval)
        artifact("evaluate_final", _eval_final)

    if failures:
        names = ", ".join(name for name, _ in failures)
        msg = f"{len(failures)} post-training artifact(s) failed: {names}"
        raise RuntimeError(msg) from failures[0][1]


if __name__ == "__main__":
    main()
