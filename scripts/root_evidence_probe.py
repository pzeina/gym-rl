#!/usr/bin/env python
"""ROOT EVIDENCE probe — does a reporting root close on its own eyes, or on traffic?

The 2026-09-09 fireteam seed spread split four same-config draws into two
modes: MUTE roots (seeds 12, 14 — ``closed_on_root_report_rate`` 0.000, root
death 0.00, human-forward 0.00) and REPORTING roots (seeds 13, 15 — 0.948 and
1.000, root death 0.15 and 0.25, human-forward 0.85+). The cost travels with
the mode: every draw that closes on its own report also walks its human TL
into the assault and buries it in 15-25% of episodes.

**The hypothesis this measures.** The root only acquires the evidence it
needs to file a creditable MISSION COMPLETE by walking into its own sight of
the objective — subordinate DONE/SITREP traffic either does not arrive or
does not suffice — so forward exposure is instrumentally coupled to closing,
and the fix (if the owner wants one) is evidence flow, not a reward on root
safety.

**What would refute it, stated before the run.** Three pre-registered checks:

1. ``own sight at claim`` — at the step each root claim is filed: the root's
   distance to the objective and whether ``world.can_spot`` says the root
   sees the objective centre with its own vision model. The mechanism
   REQUIRES confirmed claims to be filed from own sight. **If most confirmed
   claims are filed blind or from standoff distance, the own-eyes story is
   dead.**
2. ``subordinate channel at claim`` — steps since a subordinate DONE (and,
   separately, since ANY evidence-bearing message: DONE, SITREP, CONTACT,
   ACOUSTIC CONTACT) last landed on the root, at the moment of claiming
   (fresh = within 5 steps). **If most confirmed claims ride on a fresh
   subordinate DONE, the policy already closes on reported evidence and the
   forward walk is not evidence-seeking.**
3. ``exposure`` — the root's mean distance from the objective over the
   episode, per arm. Run a mute draw as ``--vs`` and the two arms bound the
   behaviour: the mechanism predicts the reporting arm's root spends its
   episode near the objective, the mute arm's does not. **No separation
   refutes the coupling being positional at all.**

A refuted mechanism is a result. This script prints the numbers and a
verdict line per check; it recommends nothing and writes nothing to any run
directory. Attribution follows the two rules ``jam_evidence_probe.py`` was
bitten into pinning: a claim is the ROOT's by SENDER identity against who
held the root at that step (never by recipient — a root's DONE is addressed
to its leader, not HQ), and "who is root" is a step function under
succession. Both are pinned by ``tests/test_root_evidence_probe.py``.

    scripts/root_evidence_probe.py runs/fireteam_v19_seed15/ckpt_latest.pt \
        --vs runs/fireteam_v18_seed14/ckpt_latest.pt --episodes 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cohort.core.orders import MessageKind
from cohort.core.world import dist
from cohort.env.cohort_env import make_env
from cohort.training.evaluate import _pick_actions
from cohort.training.train import load_policy

#: Traffic that carries the root evidence that a mission is finishing.
EVIDENCE_KINDS = (
    MessageKind.DONE, MessageKind.SITREP,
    MessageKind.CONTACT, MessageKind.ACOUSTIC_CONTACT,
)

#: "Fresh" evidence at the moment of claiming: landed within this many steps.
FRESH_WINDOW = 5


def pair_root_claims(messages, root_id_at_step):
    """Each ROOT claim with its own verdict, as (transcript_index, claim, verdict).

    Sender identity against who held the root at the claim's step decides
    whose claim it is; verdicts are paired in transcript order so each claim
    keeps the answer it actually received.
    """
    pending: list[tuple[int, object]] = []
    out: list[tuple[int, object, str]] = []
    for i, m in enumerate(messages):
        if m.kind is MessageKind.DONE:
            pending.append((i, m))
        elif m.kind in (MessageKind.DONE_CONFIRM, MessageKind.DONE_REJECT):
            if not pending:
                continue
            idx, claim = pending.pop(0)
            if claim.sender_id != root_id_at_step.get(claim.step):
                continue
            verdict = ("confirmed" if m.kind is MessageKind.DONE_CONFIRM
                       else "rejected")
            out.append((idx, claim, verdict))
    return out


def staleness_before(messages, claim_idx, claim_step, root_id_at_step, kinds):
    """Steps since a message of ``kinds`` last LANDED on the root, before the claim.

    Transcript order is delivery order, so "before" is transcript position,
    not just step number — a report landing earlier in the same step still
    counts as in hand. Only messages whose recipient held the root at their
    own step count (succession moves the target), and the root's own traffic
    never evidences itself. Returns None if nothing ever landed.
    """
    last: int | None = None
    for m in messages[:claim_idx]:
        if (m.kind in kinds
                and m.recipient_id == root_id_at_step.get(m.step)
                and m.sender_id != m.recipient_id):
            last = m.step
    return None if last is None else max(0, claim_step - last)


class Tally:
    def __init__(self, label: str):
        self.label = label
        self.episodes = 0
        self.success = 0
        self.root_died = 0
        self.root_step_dists: list[float] = []     # every alive root step, all eps
        self.death_dists: list[float] = []         # root dist at its last alive step
        # per claim: (verdict, dist, in_sight, done_stale, evidence_stale)
        self.claims: list[tuple[str, float | None, bool | None,
                                int | None, int | None]] = []

    def split(self, verdict: str):
        return [c for c in self.claims if c[0] == verdict]


def _fmt(x, spec: str = ".2f") -> str:
    return "—" if x is None else format(x, spec)


def _frac(vals: list[bool]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def run_arm(checkpoint: str, episodes: int, first_seed: int, greedy: bool) -> Tally:
    net, ckpt = load_policy(checkpoint)
    scenario = ckpt.get("scenario")
    env = make_env(scenario)
    if not env.spec_cfg.root_objective:
        raise SystemExit(f"{scenario} has no root_objective; nothing to measure")
    t = Tally(f"{Path(checkpoint).parent.name} [{scenario}]")

    for k in range(episodes):
        ep_seed = first_seed + k
        torch.manual_seed(ep_seed)
        rng = np.random.default_rng(ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        objective = env.world.objective_by_name(env.spec_cfg.root_objective)
        t.episodes += 1
        msgs_before = len(env.transcript.messages)
        initial_root = env.roster.root()
        # Both keyed by the step messages will be STAMPED with: `step()`
        # increments `_step_count` before `_say` stamps it, so key
        # post-increment — the slip that cost done_probe.py its last-step
        # claims (see that header).
        root_id_at_step: dict[int, int | None] = {}
        root_pos_at_step: dict[int, tuple[int, int] | None] = {}
        last_root_dist: float | None = None

        while env.agents:
            root_now = env.roster.root()
            stamped = env._step_count + 1
            root_id_at_step[stamped] = root_now.id if root_now else None
            root_pos_at_step[stamped] = tuple(root_now.pos) if root_now else None
            if root_now is not None:
                d = dist(root_now.pos, objective.pos)
                t.root_step_dists.append(d)
                last_root_dist = d
            actions = _pick_actions(env, obs, net, rng, greedy=greedy)
            obs, _, _, _, _ = env.step(actions)

        if env._check_success(objective):
            t.success += 1
        if initial_root is not None and not env.roster.by_id[initial_root.id].alive:
            t.root_died += 1
            if last_root_dist is not None:
                t.death_dists.append(last_root_dist)

        new = env.transcript.messages[msgs_before:]
        for idx, claim, verdict in pair_root_claims(new, root_id_at_step):
            pos = root_pos_at_step.get(claim.step)
            d = dist(pos, objective.pos) if pos is not None else None
            sight = (env.world.can_spot(pos, objective.pos,
                                        env.combat.vision_range,
                                        env.combat.forest_vision_range)
                     if pos is not None else None)
            done_stale = staleness_before(new, idx, claim.step, root_id_at_step,
                                          (MessageKind.DONE,))
            ev_stale = staleness_before(new, idx, claim.step, root_id_at_step,
                                        EVIDENCE_KINDS)
            t.claims.append((verdict, d, sight, done_stale, ev_stale))
    return t


def report(t: Tally) -> None:
    print(f"\n== {t.label} ==")
    print(f"  episodes {t.episodes}   success {t.success / t.episodes:.2f}   "
          f"root death {t.root_died / t.episodes:.2f}")
    print(f"  root mean dist from OBJ  {_fmt(float(np.mean(t.root_step_dists)) if t.root_step_dists else None)}"
          f"   at death {_fmt(float(np.mean(t.death_dists)) if t.death_dists else None)}"
          f" (n={len(t.death_dists)})")
    for verdict in ("confirmed", "rejected"):
        cs = t.split(verdict)
        if not cs:
            print(f"  {verdict:9s}  0 claims")
            continue
        dists = [c[1] for c in cs if c[1] is not None]
        sights = [c[2] for c in cs if c[2] is not None]
        fresh_done = [c[3] is not None and c[3] <= FRESH_WINDOW for c in cs]
        fresh_ev = [c[4] is not None and c[4] <= FRESH_WINDOW for c in cs]
        print(f"  {verdict:9s}  {len(cs)} claims   "
              f"dist median {_fmt(median(dists) if dists else None)}   "
              f"own sight {_fmt(_frac(sights))}   "
              f"fresh sub DONE (≤{FRESH_WINDOW}) {_fmt(_frac(fresh_done))}   "
              f"fresh evidence {_fmt(_frac(fresh_ev))}")


def verdicts(t: Tally) -> None:
    conf = t.split("confirmed")
    if not conf:
        print(f"  [{t.label}] no confirmed root claims — a mute arm bounds "
              f"check 3 only")
        return
    sights = _frac([c[2] for c in conf if c[2] is not None])
    fresh_done = _frac([c[3] is not None and c[3] <= FRESH_WINDOW for c in conf])
    own = (f"own-sight {sights:.2f} at confirmed claims — "
           + ("CONSISTENT with closing on own eyes"
              if sights is not None and sights > 0.5
              else "REFUTES the own-eyes story"))
    sub = (f"fresh subordinate DONE in hand at {fresh_done:.2f} of confirmed "
           "claims — "
           + ("the policy already closes on reported evidence; the forward "
              "walk is NOT evidence-seeking" if fresh_done is not None
              and fresh_done > 0.5
              else "subordinate traffic is not what it closes on"))
    print(f"  [{t.label}] {own}")
    print(f"  [{t.label}] {sub}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--vs", help="second checkpoint (e.g. a mute draw) as contrast")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    arms = [run_arm(args.checkpoint, args.episodes, args.seed, args.greedy)]
    if args.vs:
        arms.append(run_arm(args.vs, args.episodes, args.seed, args.greedy))
    for t in arms:
        report(t)
    print("\nverdicts (thresholds pre-registered in the header):")
    for t in arms:
        verdicts(t)


if __name__ == "__main__":
    main()
