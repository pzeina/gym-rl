#!/usr/bin/env python
"""JAMMING EVIDENCE probe — does an outage sever the root's *evidence*, or its *voice*?

The 2026-08-24 jamming arm produced one result that replicated at both seeds:
mission success survives a 35% duty cycle, but the root's MISSION COMPLETE
channel does not. Against matched clear-net controls on the same commit,
``closed_on_root_report_rate`` went 0.842 -> 0.211 (seed 12) and 0.800 -> 0.000
(seed 13). Both controls PASS the >= 0.5 gate; both jammed arms FAIL it. The two
seeds failed it in opposite ways — seed 12 filed 41 claims at first-claim
precision 0.067, seed 13 filed none at all.

**The hypothesis this measures.** ``comm_model="jammed"`` exempts HQ in both
directions, so the root can always SPEAK: its up-channel to HQ rides through
every outage by construction. What jamming takes away is lateral traffic — and
subordinate-to-leader reports are lateral. So the outage removes the root's
*evidence* that the mission is done while leaving its *voice* intact. A
commander with a voice and no evidence has two degenerate options, and the two
seeds took one each: claim without evidence (seed 12) or never claim (seed 13).

**What would refute it, stated before the run.** Three pre-registered checks:

1. ``evidence_to_root`` — evidence-bearing traffic (DONE, SITREP, CONTACT,
   ACOUSTIC CONTACT) that actually REACHES the root, per episode, counted off
   the transcript. The mechanism REQUIRES this to be materially below the
   control. **If the jammed root receives as much as the clear one, the
   mechanism is dead** and the gate failure is about something else.
   Counted off the transcript rather than off ``_audible_to`` deliberately: the
   env consults that predicate for action masks and ``useful=`` flags as well as
   for real transmissions, so an audibility rate overstates attempts and cannot
   carry a headline. A message enters the transcript only when it lands. The
   audibility rate is still printed, as corroboration.
2. ``precision | net UP`` vs ``precision | net DOWN`` — root claim precision
   conditioned on the jam state at the moment of claiming. The mechanism
   predicts claims made during an outage are the unevidenced ones and score
   worse. **Equal precision refutes the "claims blind" half.**
3. ``staleness`` — steps since the root last heard ANY subordinate transmission,
   at the moment it claims, split by verdict. The mechanism predicts rejected
   claims sit on staler evidence than confirmed ones. **No gap refutes it.**

A refuted mechanism is a result. This script prints the three numbers and the
verdict line for each; it does not recommend a reward change, and nothing here
writes to a run directory.

**Instrumentation is read-only by construction.** ``_audible_to`` is wrapped,
not replaced: the wrapper calls the real method, records the (sender, listener,
jam state, landed) tuple, and returns the real answer unchanged. Trajectories
should therefore be identical to an uninstrumented rollout, and that is not
taken on trust: ``--audit`` replays the first episode of each arm with no
wrapper installed and raises unless the transcript length and every message's
(step, kind, sender) match the instrumented replay exactly. A probe that
perturbs what it measures is the failure mode ``done_probe.py`` was bitten by,
so it fails loudly here rather than reporting a number it cannot stand behind.

    scripts/jam_evidence_probe.py runs/squad_jammed_control_v1_seed12/ckpt_latest.pt \
        --vs runs/squad_ctrl_v2_seed12/ckpt_latest.pt --episodes 30
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cohort.core.orders import HQ_ID, MessageKind
from cohort.env.cohort_env import make_env
from cohort.training.evaluate import _pick_actions
from cohort.training.train import load_policy

#: Traffic that carries the root its evidence that a mission is finishing.
EVIDENCE_KINDS = (
    MessageKind.DONE, MessageKind.SITREP,
    MessageKind.CONTACT, MessageKind.ACOUSTIC_CONTACT,
)


class Tally:
    """Everything one arm's rollout block records."""

    def __init__(self, label: str):
        self.label = label
        self.episodes = 0
        self.steps = 0
        self.jammed_steps = 0
        # transmission delivery, split by whether the listener is the sender's
        # leader (the up-channel that carries evidence) and whether that leader
        # is the root itself.
        self.up_attempt = 0
        self.up_landed = 0
        self.up_root_attempt = 0
        self.up_root_landed = 0
        self.lat_attempt = 0
        self.lat_landed = 0
        # root MISSION COMPLETE claims, keyed by the jam state when filed
        self.claims = Counter()          # (jam_state, verdict) -> n
        self.staleness = {"confirmed": [], "rejected": []}
        self.success = 0
        # Evidence that actually REACHED the root, counted off the transcript.
        # The audibility counters above are attempts as seen by `_audible_to`,
        # which the env also consults for action masks and for `useful=` flags
        # — so they overstate transmissions and cannot carry the headline. A
        # message only enters the transcript when it lands, so this count is
        # exactly "reports the root received", with no such confound.
        self.evidence_landed = 0

    # -- derived ---------------------------------------------------------
    @property
    def duty(self) -> float:
        return self.jammed_steps / self.steps if self.steps else 0.0

    def rate(self, num: int, den: int) -> float | None:
        return num / den if den else None

    def precision(self, jam: bool | None = None) -> tuple[float | None, int]:
        if jam is None:
            conf = sum(v for (_j, k), v in self.claims.items() if k == "confirmed")
            tot = sum(self.claims.values())
        else:
            conf = self.claims[(jam, "confirmed")]
            tot = self.claims[(jam, "confirmed")] + self.claims[(jam, "rejected")]
        return (conf / tot if tot else None), tot


def _fmt(x: float | None, spec: str = ".3f") -> str:
    return "—" if x is None else format(x, spec)


def _leader_id(env, sender_id: int) -> int | None:
    sender = env.roster.by_id.get(sender_id)
    if sender is None:
        return None
    leader = env.roster.leader_of(sender)
    return leader.id if leader is not None else None


def count_evidence_to_root(messages, root_id_at_step) -> int:
    """Evidence-bearing traffic that LANDED on whoever was root at that step."""
    return sum(
        1 for m in messages
        if m.kind in EVIDENCE_KINDS
        and m.recipient_id == root_id_at_step.get(m.step)
        and m.sender_id != m.recipient_id
    )


def attribute_root_claims(messages, root_id_at_step, jam_at_step, stale_at_step):
    """Pair each DONE with its own verdict and keep the ROOT's, as (jam, verdict, staleness).

    Two attribution rules, and this probe got both wrong on its first run —
    reporting **zero** root claims for a control the behaviour suite scores at
    ``closed_on_root_report_rate`` 0.842. A silent zero reads exactly like a
    finding ("the root never claims"), which is why the rules are pinned by
    ``tests/test_jam_evidence_probe.py`` rather than left inline:

    * **whose claim it is** is decided by SENDER identity against who held the
      root at that step, never by the recipient. A root's DONE is addressed to
      its ``leader_id``, which is not ``HQ_ID`` — filtering on the recipient
      discards every root claim there is.
    * **who the root was** is a step function. Succession promotes a
      subordinate mid-episode, so a single reset-time lookup misattributes the
      promoted agent's earlier claims.

    Verdicts are paired in transcript order rather than counted in aggregate,
    so each claim carries the verdict it actually received.
    """
    pending: list = []
    out: list[tuple[bool, str, int | None]] = []
    for m in messages:
        if m.kind is MessageKind.DONE:
            pending.append(m)
        elif m.kind in (MessageKind.DONE_CONFIRM, MessageKind.DONE_REJECT):
            if not pending:
                continue
            claim = pending.pop(0)
            if claim.sender_id != root_id_at_step.get(claim.step):
                continue
            verdict = "confirmed" if m.kind is MessageKind.DONE_CONFIRM else "rejected"
            out.append((jam_at_step.get(claim.step, False), verdict,
                        stale_at_step.get(claim.step)))
    return out


def run_arm(checkpoint: str, episodes: int, first_seed: int, greedy: bool,
            label: str) -> Tally:
    net, ckpt = load_policy(checkpoint)
    scenario = ckpt.get("scenario")
    env = make_env(scenario)
    t = Tally(f"{label} [{scenario}]")

    real_audible = env._audible_to
    # Per-step scratch: the wrapper fires many times per step, so it records
    # into these and the step loop drains them.
    state = {"root_id": None, "last_heard": None, "step": 0}

    def wrapped(listener, sender_id: int) -> bool:
        landed = real_audible(listener, sender_id)
        # HQ traffic and self-hearing are exempt by construction and say
        # nothing about the outage; count only cohort-to-cohort transmissions.
        if sender_id != HQ_ID and sender_id != listener.id:
            up = _leader_id(env, sender_id) == listener.id
            if up:
                t.up_attempt += 1
                t.up_landed += landed
                if listener.id == state["root_id"]:
                    t.up_root_attempt += 1
                    t.up_root_landed += landed
                    if landed:
                        state["last_heard"] = state["step"]
            else:
                t.lat_attempt += 1
                t.lat_landed += landed
        return landed

    env._audible_to = wrapped  # type: ignore[method-assign]

    for k in range(episodes):
        ep_seed = first_seed + k
        torch.manual_seed(ep_seed)
        rng = np.random.default_rng(ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        t.episodes += 1
        msgs_before = len(env.transcript.messages)
        state["last_heard"] = None
        # jam state and root identity are step functions: succession moves the
        # root mid-episode, and the outage flips under it. Both are keyed by
        # the step the messages will be STAMPED with — `CohortEnv.step`
        # increments `_step_count` before `_say` stamps it, and keying
        # pre-increment is the slip that cost done_probe.py every last-step
        # claim (see its header). Same keying here, for the same reason.
        jam_at_step: dict[int, bool] = {}
        stale_at_step: dict[int, int | None] = {}
        root_id_at_step: dict[int, int | None] = {}

        while env.agents:
            root_now = env.roster.root()
            state["root_id"] = root_now.id if root_now else None
            state["step"] = env._step_count
            root_id_at_step[env._step_count + 1] = state["root_id"]
            actions = _pick_actions(env, obs, net, rng, greedy=greedy)
            obs, _, _, _, _ = env.step(actions)
            t.steps += 1
            stamped = env._step_count
            jam_at_step[stamped] = bool(getattr(env, "_net_jammed", False))
            t.jammed_steps += jam_at_step[stamped]
            lh = state["last_heard"]
            stale_at_step[stamped] = None if lh is None else stamped - lh

        # Adjudicate from the transcript — the env's verdict, never ours.
        new = env.transcript.messages[msgs_before:]
        if env._check_success(env.world.objective_by_name(env.spec_cfg.root_objective)
                              if env.spec_cfg.root_objective else None):
            t.success += 1
        # A DONE at step s is answered by the DONE_CONFIRM / DONE_REJECT that
        # follows it; pair them in order rather than counting totals, so a
        # claim's own verdict is the one recorded against its jam state.
        t.evidence_landed += count_evidence_to_root(new, root_id_at_step)
        for jam, verdict, stale in attribute_root_claims(
                new, root_id_at_step, jam_at_step, stale_at_step):
            t.claims[(jam, verdict)] += 1
            if stale is not None:
                t.staleness[verdict].append(stale)

    env._audible_to = real_audible  # type: ignore[method-assign]
    return t


def audit_noninterference(checkpoint: str, first_seed: int, greedy: bool) -> None:
    """Replay one episode with and without the wrapper; the traffic must match.

    The wrapper only reads, so this should always pass — which is exactly why
    it is worth asserting. If a future edit makes the instrumentation consume
    RNG or short-circuit a predicate, every number this script prints becomes a
    measurement of the probe rather than of the policy, and that must fail
    loudly instead of looking like a finding.
    """
    def replay(instrumented: bool) -> list[tuple[int, str, int]]:
        net, ckpt = load_policy(checkpoint)
        env = make_env(ckpt.get("scenario"))
        if instrumented:
            real = env._audible_to
            seen: list = []

            def wrapped(listener, sender_id: int) -> bool:
                landed = real(listener, sender_id)
                seen.append(landed)     # read-only, as the real wrapper is
                return landed

            env._audible_to = wrapped  # type: ignore[method-assign]
        torch.manual_seed(first_seed)
        rng = np.random.default_rng(first_seed)
        obs, _ = env.reset(seed=first_seed)
        while env.agents:
            obs, _, _, _, _ = env.step(_pick_actions(env, obs, net, rng, greedy=greedy))
        return [(m.step, m.kind.value, m.sender_id) for m in env.transcript.messages]

    clean, hooked = replay(False), replay(True)
    if clean != hooked:
        raise SystemExit(
            "PROBE INVALID — instrumentation perturbed the rollout: "
            f"{len(clean)} messages clean vs {len(hooked)} instrumented"
        )
    print(f"  self-audit: instrumentation is non-interfering "
          f"({len(clean)} messages identical in both replays)")


def report(arm: Tally, ctrl: Tally | None) -> None:
    print(f"\n== {arm.label} ==")
    print(f"  episodes {arm.episodes}  steps {arm.steps}  "
          f"observed duty cycle {arm.duty:.3f}  success {arm.success}/{arm.episodes}")
    print(f"  up-channel delivery (sub -> its leader)   "
          f"{_fmt(arm.rate(arm.up_landed, arm.up_attempt))}  "
          f"({arm.up_landed}/{arm.up_attempt})")
    print(f"  up-channel delivery TO THE ROOT           "
          f"{_fmt(arm.rate(arm.up_root_landed, arm.up_root_attempt))}  "
          f"({arm.up_root_landed}/{arm.up_root_attempt})")
    print(f"  evidence REACHING the root, per episode   "
          f"{arm.evidence_landed / arm.episodes:.2f}  ({arm.evidence_landed} msgs)")
    print(f"  other cohort traffic delivery             "
          f"{_fmt(arm.rate(arm.lat_landed, arm.lat_attempt))}  "
          f"({arm.lat_landed}/{arm.lat_attempt})")
    tot_p, tot_n = arm.precision()
    up_p, up_n = arm.precision(False)
    dn_p, dn_n = arm.precision(True)
    print(f"  root claims {tot_n}   precision {_fmt(tot_p)}")
    print(f"    | net UP    {_fmt(up_p)}  (n={up_n})")
    print(f"    | net DOWN  {_fmt(dn_p)}  (n={dn_n})")
    for verdict in ("confirmed", "rejected"):
        vals = arm.staleness[verdict]
        mean = sum(vals) / len(vals) if vals else None
        print(f"  evidence staleness at claim, {verdict:9} "
              f"{_fmt(mean, '.1f')} steps  (n={len(vals)})")

    if ctrl is None:
        return
    print("\n  -- pre-registered checks against the control --")
    # Primary: evidence that reached the root, off the transcript. The
    # audibility rate is printed beside it as corroboration only — it counts
    # `_audible_to` calls, which include mask and `useful=` queries.
    a = arm.evidence_landed / arm.episodes
    c = ctrl.evidence_landed / ctrl.episodes
    aud_a = arm.rate(arm.up_root_landed, arm.up_root_attempt)
    aud_c = ctrl.rate(ctrl.up_root_landed, ctrl.up_root_attempt)
    if c == 0:
        print("  1. evidence_to_root: NOT MEASURABLE — the control's root receives none")
    elif a >= c:
        print(f"  1. evidence_to_root: REFUTED — jammed {a:.2f}/ep >= clear {c:.2f}/ep; "
              "the outage does not cost the root its evidence")
    else:
        print(f"  1. evidence_to_root: SUPPORTED — jammed {a:.2f}/ep vs clear {c:.2f}/ep "
              f"({(1 - a / c) * 100:.1f}% less evidence reaches the root); "
              f"audibility corroborates {_fmt(aud_a)} vs {_fmt(aud_c)}")

    if up_n == 0 or dn_n == 0:
        print(f"  2. precision by jam state: NOT MEASURABLE — n(UP)={up_n}, n(DOWN)={dn_n}")
    elif dn_p >= up_p:
        print(f"  2. precision by jam state: REFUTED — DOWN {dn_p:.3f} >= UP {up_p:.3f}; "
              "claims made during an outage are no worse")
    else:
        print(f"  2. precision by jam state: SUPPORTED — UP {up_p:.3f} vs DOWN {dn_p:.3f}")

    conf, rej = arm.staleness["confirmed"], arm.staleness["rejected"]
    if not conf or not rej:
        print(f"  3. staleness by verdict: NOT MEASURABLE — "
              f"n(confirmed)={len(conf)}, n(rejected)={len(rej)}")
    else:
        mc, mr = sum(conf) / len(conf), sum(rej) / len(rej)
        if mr <= mc:
            print(f"  3. staleness by verdict: REFUTED — rejected {mr:.1f} <= "
                  f"confirmed {mc:.1f} steps")
        else:
            print(f"  3. staleness by verdict: SUPPORTED — rejected {mr:.1f} vs "
                  f"confirmed {mc:.1f} steps")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="the jammed arm's checkpoint")
    ap.add_argument("--vs", dest="control", default=None,
                    help="matched clear-net control checkpoint")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--audit", action="store_true",
                    help="replay one episode with and without the wrapper first")
    args = ap.parse_args()

    if args.audit:
        print("== non-interference audit ==")
        audit_noninterference(args.checkpoint, args.seed, args.greedy)

    ctrl = None
    if args.control:
        ctrl = run_arm(args.control, args.episodes, args.seed, args.greedy, "CONTROL")
    arm = run_arm(args.checkpoint, args.episodes, args.seed, args.greedy, "JAMMED")
    if ctrl is not None:
        report(ctrl, None)
    report(arm, ctrl)


if __name__ == "__main__":
    main()
