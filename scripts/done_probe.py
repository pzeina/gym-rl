#!/usr/bin/env python
"""MISSION COMPLETE channel probe — is a truthful DONE reachable, and at what rate?

The v1.10 defend cycle priced ``done_false`` at -2.0 to stop premature COMPLETE
claims, and fireteam_defend_v8 then emitted *zero* DONE reports in 100 episodes.
Two very different stories fit that observation:

* **pricing** — claiming is admissible but -EV, so the policy learned silence
  (the failure ``rewards.py`` predicts in its own comment: "over-pricing a
  speech act suppresses the HONEST one too");
* **reachability** — the act is hard-masked, and no price was ever consulted.

They call for opposite responses, so this measures which one is true before any
reward is touched. The central quantity is a **golden step**: an agent-step
where MISSION COMPLETE is admissible by the action mask *and* the claim would
be adjudicated truthful. Golden steps are the opportunities a policy could have
taken. Zero golden steps means the channel is unreachable and repricing it is
meaningless; many golden steps means the opportunity was there and declined.

Three regimes over one seeded episode block:

* ``observe`` — the policy runs untouched. Counts admissible and golden steps.
  This is the only regime whose trajectories are unperturbed, so it is the one
  to quote for opportunity.
* ``oracle``  — force DONE on every golden step. Accept rate should come out at
  1.00 by construction; it is an instrument check on this script's replication
  of the env's own truth test, and a count of how much reward silence forgoes.
* ``naive``   — claim with probability ``--naive-rate`` whenever admissible,
  ignoring truth. This is the accept rate an *uninformed* claimant sees, i.e.
  what a fresh policy meets during early training, before it can time the act.

Read-only: no training, no checkpoint is written. Prints facts and exits; it
does not interpret or recommend — that is the caller's job.

    scripts/done_probe.py runs/<run>/ckpt_best.pt --episodes 30 --seed 500
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cohort.core.missions import COMPLETABLE, is_complete
from cohort.core.orders import HQ_ID, MessageKind
from cohort.env.actions import CATALOG
from cohort.env.cohort_env import make_env
from cohort.training.evaluate import _pick_actions
from cohort.training.train import load_policy

REGIMES = ("observe", "oracle", "naive")

DONE_IX = next(spec.index for spec in CATALOG if spec.kind == "done")


def _would_be_truthful(env, soldier, root_obj) -> bool:
    """Replicate the env's own DONE adjudication (cohort_env.py::_report_done).

    Kept deliberately as a mirror rather than a call into a shared helper: if
    the two ever drift, the ``oracle`` regime's accept rate falls below 1.00
    and this script reports its own breakage instead of a finding.
    """
    mission = soldier.mission
    if mission is None:
        return False
    is_root_claim = (
        soldier is env.roster.root()
        and mission.issuer_id == HQ_ID
        and mission.type is env.spec_cfg.root_mission
        and mission.objective_id == (root_obj.id if root_obj is not None else None)
    )
    if is_root_claim:
        return bool(env._check_success(root_obj))
    ctx = env._compliance_ctx(soldier, None, env._make_view(soldier))
    return bool(is_complete(mission, ctx))


class Accum:
    """Running counters for one regime's episode block."""

    def __init__(self) -> None:
        self.episodes = 0
        self.steps = 0
        # agent-steps, split root / subordinate
        self.admissible: Counter[str] = Counter()
        self.golden: Counter[str] = Counter()
        self.mission_steps: Counter[str] = Counter()      # living agent-steps by mission
        self.admissible_by_mission: Counter[str] = Counter()
        self.golden_by_mission: Counter[str] = Counter()
        # transcript-adjudicated outcomes
        self.claims = 0
        self.confirms = 0
        self.rejects = 0
        self.root_claims = 0
        self.eps_with_golden = 0
        self.eps_with_confirm = 0
        self.success_steps = 0        # episodes where the root-mission condition was met
        self.golden_after_t0 = 0

    def rate(self, num: int, den: int) -> str:
        return f"{num / den:.3f}" if den else "—"


def probe(checkpoint, scenario, episodes, first_seed, greedy, regime, naive_rate):
    net = None
    if checkpoint is not None:
        net, ckpt = load_policy(checkpoint)
        scenario = scenario or ckpt.get("scenario")
    if scenario is None:
        raise SystemExit("need --scenario when probing without a checkpoint")

    env = make_env(scenario)
    acc = Accum()
    root_name = env.spec_cfg.root_objective

    for k in range(episodes):
        ep_seed = first_seed + k
        torch.manual_seed(ep_seed)
        rng = np.random.default_rng(ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        root_obj = env.world.objective_by_name(root_name) if root_name else None
        by_cs = {s.callsign: s for s in env.roster.soldiers}
        acc.episodes += 1
        ep_golden = 0
        msgs_before = len(env.transcript.messages)
        # succession moves the root mid-episode, so "who was root" is a
        # function of the step — resolving it once at reset misattributes a
        # promoted rifleman's earlier claims to the root.
        root_id_at_step: dict[int, int | None] = {}

        while env.agents:
            actions = _pick_actions(env, obs, net, rng, greedy=greedy)
            t0_open = env._success_step is not None
            root_now = env.roster.root()
            root_cs = root_now.callsign if root_now else None
            root_id_at_step[env._step_count] = root_now.id if root_now else None
            for cs in list(actions):
                soldier = by_cs.get(cs)
                if soldier is None or not soldier.alive:
                    continue
                role = "root" if cs == root_cs else "sub"
                mission_name = soldier.mission.type.name if soldier.mission else "NONE"
                acc.mission_steps[mission_name] += 1

                if not obs[cs]["action_mask"][DONE_IX]:
                    continue
                acc.admissible[role] += 1
                acc.admissible_by_mission[mission_name] += 1

                truthful = _would_be_truthful(env, soldier, root_obj)
                if truthful:
                    acc.golden[role] += 1
                    acc.golden_by_mission[mission_name] += 1
                    ep_golden += 1
                    if t0_open:
                        acc.golden_after_t0 += 1

                forced = (regime == "oracle" and truthful) or (
                    regime == "naive" and rng.random() < naive_rate
                )
                if forced:
                    actions[cs] = DONE_IX

            obs, _, _, _, _ = env.step(actions)
            acc.steps += 1

        # adjudicate from the transcript: the env's verdict, not ours
        new = env.transcript.messages[msgs_before:]
        confirms = sum(1 for m in new if m.kind is MessageKind.DONE_CONFIRM)
        rejects = sum(1 for m in new if m.kind is MessageKind.DONE_REJECT)
        claims = sum(1 for m in new if m.kind is MessageKind.DONE)
        acc.claims += claims
        acc.confirms += confirms
        acc.rejects += rejects
        acc.root_claims += sum(
            1 for m in new
            if m.kind is MessageKind.DONE and m.sender_id == root_id_at_step.get(m.step)
        )
        if ep_golden:
            acc.eps_with_golden += 1
        if confirms:
            acc.eps_with_confirm += 1
        if env._success_step is not None:
            acc.success_steps += 1

    return acc, env


def report(accs: dict, env, episodes: int) -> None:
    spec = env.spec_cfg
    root_mission = spec.root_mission
    reachable = root_mission in COMPLETABLE

    print(f"done probe: {spec.name}  episodes={episodes}")
    print()
    print("STRUCTURE (scenario constants, no policy involved)")
    print(f"  root mission                  {root_mission.name}")
    print(f"  root mission COMPLETABLE      {reachable}")
    print(f"  completable missions          {', '.join(sorted(m.name for m in COMPLETABLE))}")
    print(f"  grace_window / done_cooldown  {spec.grace_window} / {spec.done_cooldown}")
    print()

    head = "".join(f"{r:>12}" for r in REGIMES)
    print(f"{'':30}{head}")
    print("-" * (30 + 12 * len(REGIMES)))

    def row(label, fn):
        cells = "".join(f"{fn(accs[r]):>12}" for r in REGIMES)
        print(f"  {label:<28}{cells}")

    print("OPPORTUNITY (agent-steps; quote the observe column — it is unperturbed)")
    row("DONE admissible [root]", lambda a: a.admissible["root"])
    row("DONE admissible [sub]", lambda a: a.admissible["sub"])
    row("golden steps [root]", lambda a: a.golden["root"])
    row("golden steps [sub]", lambda a: a.golden["sub"])
    row("golden after T0", lambda a: a.golden_after_t0)
    row("eps with >=1 golden", lambda a: f"{a.eps_with_golden}/{a.episodes}")
    print("CLAIMS (adjudicated by the env, read off the transcript)")
    row("claims transmitted", lambda a: a.claims)
    row("  ...by the root", lambda a: a.root_claims)
    row("confirmed", lambda a: a.confirms)
    row("rejected", lambda a: a.rejects)
    row("accept rate", lambda a: a.rate(a.confirms, a.claims))
    row("eps with >=1 confirm", lambda a: f"{a.eps_with_confirm}/{a.episodes}")
    print("CONTEXT")
    row("eps reaching T0 (success)", lambda a: f"{a.success_steps}/{a.episodes}")

    obs_acc = accs["observe"]
    print()
    print("BY MISSION HELD (observe regime: living agent-steps / admissible / golden)")
    for name, steps in obs_acc.mission_steps.most_common():
        adm = obs_acc.admissible_by_mission[name]
        gold = obs_acc.golden_by_mission[name]
        print(f"  {name:<16} {steps:>8} {adm:>10} {gold:>8}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--scenario")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=500, help="first episode seed")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--naive-rate", type=float, default=0.05,
                    help="per-step claim probability in the naive regime")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    accs, env = {}, None
    for regime in REGIMES:
        accs[regime], env = probe(
            args.checkpoint, args.scenario, args.episodes,
            args.seed, args.greedy, regime, args.naive_rate,
        )
    report(accs, env, args.episodes)

    if args.json_out:
        blob = {
            r: {
                "admissible": dict(a.admissible),
                "golden": dict(a.golden),
                "claims": a.claims,
                "confirms": a.confirms,
                "rejects": a.rejects,
                "root_claims": a.root_claims,
                "episodes": a.episodes,
            }
            for r, a in accs.items()
        }
        Path(args.json_out).write_text(json.dumps(blob, indent=2))


if __name__ == "__main__":
    main()
