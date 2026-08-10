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

**Self-audited.** This script has already been wrong once, in a way that looked
like a finding: it keyed "who held the root at this step" one step early and
lost every claim made on an episode's last step, which is exactly where the
confirmed ones live (55 claims / 0 confirmed, against 87 / 32 once fixed). So
each episode's traffic is now checked against two guards before it is counted —
see ``_audit_root_claims`` — and the probe raises rather than reporting a number
it cannot stand behind. The invariant half of that is the assurance layer's
standing negative control from issue #33, pinned in
``tests/test_confirmed_claim_is_last.py``.

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


class RootClaimAuditError(Exception):
    """The probe's root attribution disagrees with the env's own structure."""


def _audit_root_claims(new, root_id_at_step: dict[int, int | None]) -> int:
    """Count this episode's ROOT claims, refusing to count them unsoundly.

    Two guards, because they catch different failures and this script has met
    one of them:

    * **coverage** — every adjudicated message must fall on a step whose root
      is known. A missing key silently turns a claim into a non-claim, which is
      how the pre-fix keying erased 32 confirmations without changing any
      number that looked wrong. A ratio check cannot see this: dropping a step
      removes the claim AND its confirmation together.
    * **the invariant** — a confirmed root claim is the last root claim of its
      episode (assurance layer, issue #33: 0 violations across 86 corpora). It
      holds structurally, because confirmation closes the grace window and the
      terminal check ends the episode in the same step. So at most one root
      claim per episode is confirmed, and nothing root-claims after it.

    Raising is deliberate. A probe that reports an impossible corpus is worse
    than one that stops: the impossible corpus gets quoted.
    """
    adjudicated = (MessageKind.DONE, MessageKind.DONE_CONFIRM, MessageKind.DONE_REJECT)
    claims = confirms = rejects = 0
    closed = False
    for msg in new:
        if msg.kind not in adjudicated:
            continue
        if msg.step not in root_id_at_step:
            raise RootClaimAuditError(
                f"step {msg.step} carries a {msg.kind.name} but no root was recorded for it — "
                "the step keying and the message stamping have drifted apart again"
            )
        root_id = root_id_at_step[msg.step]
        if msg.kind is MessageKind.DONE and msg.sender_id == root_id:
            if closed:
                raise RootClaimAuditError(
                    f"root claimed at step {msg.step}, after a confirmed claim had closed the "
                    "operation — attribution is wrong, or the terminal branch no longer ends it"
                )
            claims += 1
        elif msg.kind is MessageKind.DONE_CONFIRM and msg.recipient_id == root_id:
            confirms += 1
            closed = True
        elif msg.kind is MessageKind.DONE_REJECT and msg.recipient_id == root_id:
            rejects += 1
    if claims - rejects != confirms or confirms not in (0, 1):
        raise RootClaimAuditError(
            f"{claims} root claims, {rejects} rejected, {confirms} confirmed — every DONE is "
            "answered exactly once, and at most one root claim per episode is confirmed"
        )
    return claims


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
        #
        # Keyed by the step the messages will be STAMPED with, not by the
        # counter's value when the actions are chosen: ``CohortEnv.step``
        # increments ``_step_count`` first and ``_say`` stamps the incremented
        # value. Keying it pre-increment silently dropped every claim made on
        # an episode's LAST step — which is exactly the confirmed ones, since a
        # confirmed root claim ends the episode. Measured on defend_brique_v13
        # /ckpt_latest, 40 episodes from seed 500: 55 root claims and 0
        # confirmed under the old keying, 87 and 32 under this one.
        # ``_audit_root_claims`` now refuses to count an episode whose keys do
        # not cover its traffic, so the same slip fails instead of measuring.
        root_id_at_step: dict[int, int | None] = {}

        while env.agents:
            actions = _pick_actions(env, obs, net, rng, greedy=greedy)
            t0_open = env._success_step is not None
            root_now = env.roster.root()
            root_cs = root_now.callsign if root_now else None
            root_id_at_step[env._step_count + 1] = root_now.id if root_now else None
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
        acc.root_claims += _audit_root_claims(new, root_id_at_step)
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
