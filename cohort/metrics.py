"""Behavioral metrics suite: measure what "behaves like its rank" means (B2).

Success rate says *whether* the cohort wins; this module measures *how it
behaves* while doing so, per evaluation run:

* obedience latency      — order received → first compliant action
* A5-2 staging           — orders staged (AT MY COMMAND / AT T PLUS n), how
  many were released and after how long, and how many were **abandoned**:
  staged and then never released (refs issue #15: obedience latency cannot
  see a staged order, because its recipient complies by standing still —
  which is exactly why counting the staged tick made staging read as instant
  obedience and pulled the latency mean toward 0)
* report precision/recall — CONTACT reports vs. enemies actually seen
  (the oracle-side ground truth: per-step visibility)
* doctrine-preference rate — share of issued orders that were the preferred
  derivation of the issuer's own mission, reported next to the doctrine
  *containment* rate and the ordered-task mix (refs issue #14: preference is
  ``allowed[0]``, so since A5 put ADVANCE in the DEFEND / SEIZE / RECON
  tables, a policy that adopts ADVANCE wholesale scores ~0 preference with
  zero doctrine violations — without the split, catalog adoption is
  indistinguishable from disregarding doctrine)
* false-COMPLETE rate    — MISSION COMPLETE claims rejected by the umpire
* COMPLETE claim rate    — claims transmitted over the agent-steps at which
  claiming was admissible (refs issue #13: zero DONE reports is either a shut
  channel or a declined opportunity, and only this denominator says which)
* succession recovery    — leader death → all orphaned subordinates re-tasked
* subordinate coverage   — share of steps every living subordinate is tasked
* human exposure         — the human root's distance to the nearest living
  enemy, its entries into the objective observation ring, and its death rate
  (refs issue #9: rolling success is blind to a policy re-learning to walk
  the commander into the ring, so checkpoint selection for preservation
  claims needs these numbers)
* fight disposition       — *where* the cohort fights once the enemy is on
  it: cover occupancy and distance from the root objective, measured only
  over the (soldier, step) pairs under threat, plus the pass/fail gate built
  on them (refs issue #11: an outside measurement showed the defend miss is
  positional, not mortal — halving the root-death rate bought no success,
  while every defend policy that ever cleared its bound fought <= 2.9 cells
  out with cover >= 0.79 and the one that missed fought 9.09 out at 0.06)

The pipeline has two halves, split so the math is unit-testable:

1. :class:`TraceRecorder` — hooks into an evaluation episode (see
   ``cohort.training.evaluate.run_episode``) and accumulates a plain-dict
   **trace**: per-step soldier/enemy state, per-step ground-truth visibility,
   per-soldier compliance scores (recomputed exactly as the environment
   scores them), and the radio transcript. Reading the environment consumes
   no randomness, so a recorded episode is bit-identical to an unrecorded
   one under the same seed.
2. Pure functions over the trace — :func:`episode_behavior` reduces one
   trace to event counts/latency lists, :func:`aggregate_behavior` pools
   episodes into the run-level summary written to ``behavior.json``.

Every metric's precise definition (formula and edge cases: no orders, no
contacts, no humans, no successions) lives in ``docs/metrics.md``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from cohort.core import language as lang
from cohort.core.missions import (
    IN_POSITION_RADIUS,
    MissionType,
    allowed_derivations,
    compliance,
    is_pending,
)
from cohort.core.units import CombatParams
from cohort.env.actions import is_done_admissible

if TYPE_CHECKING:
    from cohort.env.cohort_env import CohortEnv

#: Radius of the objective observation ring used for human ring entries —
#: RECON/SCREEN share it; it is where issue #9 measured the root's exposure.
RING_RADIUS: float = IN_POSITION_RADIUS[MissionType.RECON]

#: Fallback threat radius for traces recorded before the scenario's own
#: weapon range was written into them. A (soldier, step) pair counts as
#: "under threat" when a living enemy stands within this distance — i.e. when
#: the OpFor can actually shoot that soldier, which is what makes the pair
#: informative about *where the unit chose to fight*.
THREAT_RADIUS: float = CombatParams().weapon_range

#: Positional regression gate for DEFEND roots (refs issue #11). Bounds are
#: set from the measured record, not from taste: `_v5` (24/30) and
#: `defend_brique_v1` (27/30) held cover 0.79/0.96 at 2.90/1.99 cells, while
#: `_v6` (14/30) and `_v7` (12/30) sat at 0.496/0.060 and 3.46/9.09. The
#: floor and ceiling are placed in the empty band between the two groups, so
#: the gate separates every checkpoint on record without hair-splitting.
DEFEND_COVER_FLOOR: float = 0.40
DEFEND_OBJECTIVE_DIST_CEILING: float = 5.0


def _dist(a: list | tuple, b: list | tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------------- #
# trace recording
# ---------------------------------------------------------------------- #


class TraceRecorder:
    """Accumulates the behavioral trace of one episode.

    Wire into ``run_episode`` via its ``recorder`` argument: ``on_reset``
    right after ``env.reset``, ``before_step`` right before each
    ``env.step`` (snapshots mission-anchor distances exactly like the
    environment does), ``after_step`` right after. All reads are
    deterministic and consume no RNG, so recording never perturbs a seeded
    episode (covered by test).
    """

    def __init__(self) -> None:
        self.trace: dict[str, Any] = {}
        self._prev_dist: dict[str, float] = {}

    # -- hooks ---------------------------------------------------------- #

    def on_reset(self, env: CohortEnv) -> None:
        """Capture episode statics + the state after ``reset()`` (t=0)."""
        cfg = env.spec_cfg
        obj = env.world.objective_by_name(cfg.root_objective) if cfg.root_objective else None
        human = next((s.callsign for s in env.roster.soldiers if s.human), None)
        self.trace = {
            "scenario": cfg.name,
            "outcome": None,
            "length": 0,
            "root_mission": cfg.root_mission.name,
            "root_objective": list(obj.pos) if obj is not None else None,
            "ring_radius": RING_RADIUS,
            # issue #11: the scenario's own weapon range defines "under
            # threat", so the fight-disposition metrics travel with the
            # combat model instead of a hard-coded constant.
            "threat_radius": float(env.combat.weapon_range),
            "contact_refresh_age": env.rewards_cfg.contact_refresh_age,
            "knowledge_ttl": _knowledge_ttl(),
            "human": human,
            "reported": {},
            "steps": [self._step_record(env, initial=True)],
        }

    def before_step(self, env: CohortEnv) -> None:
        """Snapshot anchor distances (mirrors the env's own step snapshot)."""
        self._prev_dist = {
            s.callsign: env._anchor_distance(s) for s in env.roster.living if s.mission is not None
        }

    def after_step(self, env: CohortEnv) -> None:
        """Record the post-step state, this tick's messages, and compliance."""
        self.trace["steps"].append(self._step_record(env, initial=False))
        self.trace["length"] = len(self.trace["steps"]) - 1
        self.trace["outcome"] = env.outcome
        if not env.agents:  # episode over: freeze the per-soldier report sets
            self.trace["reported"] = {
                s.callsign: sorted(s.reported_enemy_ids) for s in env.roster.soldiers
            }

    # -- recording ------------------------------------------------------ #

    def _step_record(self, env: CohortEnv, *, initial: bool) -> dict:
        soldiers = []
        # succession moves the root mid-episode, so who holds the OPORD is a
        # function of the step, not of the roster at reset
        root = env.roster.root()
        for s in env.roster.soldiers:
            comp = None
            # A5-2: a pending order stages its recipient, and the environment
            # scores it as HOLD at the staging spot — not as the ordered task.
            # Scoring it as the ordered task here made a staged ADVANCE read
            # 0.5 ("in position") the tick it landed, which is the opposite of
            # what the recipient is doing: nothing, on purpose.
            pending = s.mission is not None and is_pending(s.mission, env._step_count)
            if not initial and s.alive and s.mission is not None:
                ctx = env._compliance_ctx(s, self._prev_dist.get(s.callsign), env._make_view(s))
                comp = compliance(MissionType.HOLD if pending else s.mission.type, ctx)
            leader = env.roster.leader_of(s)
            soldiers.append(
                {
                    "cs": s.callsign,
                    "alive": s.alive,
                    "pos": list(s.pos),
                    "mission": s.mission.type.name if s.mission is not None else None,
                    "since": s.mission.step_assigned if s.mission is not None else None,
                    # A5-2: is the standing order staged (AT MY COMMAND not yet
                    # EXECUTEd, AT T PLUS n not yet due)? A staged recipient is
                    # obeying by NOT executing, so obedience is not measurable
                    # until release — see _obedience and _staging.
                    "pending": pending,
                    "auth": s.effective_authority,
                    "subs": [x.callsign for x in s.living_subordinates(env.roster)],
                    "leader": leader.callsign if leader is not None else None,
                    "comp": comp,
                    # issue #11: terrain posture, the strongest correlate of
                    # defend success found so far. Read-only grid lookup.
                    "cover": bool(env.world.cover_at(s.pos)),
                    "fired": bool(s.fired_this_step) if s.alive else False,
                    "sees": [e.id for e in env._visible_enemies(s)] if s.alive else [],
                    # A5-3: the element stance set ON this soldier (leaders only)
                    "formation": s.formation.name if s.formation is not None else None,
                    # was MISSION COMPLETE admissible to this agent at this
                    # state? Read off the mask's own predicate, so a run's
                    # DONE silence is attributable: no admissible step means
                    # the channel was shut, admissible steps with no claim
                    # means the policy declined it (refs #13).
                    "done_ok": is_done_admissible(
                        s,
                        env.roster,
                        root_mission=env.spec_cfg.root_mission,
                        root_objective_id=env._root_objective_id(),
                        step=env._step_count,
                        done_cooldown=env.spec_cfg.done_cooldown,
                    ),
                    "root": s is root,
                }
            )
        messages = env.transcript.messages if initial else env.last_messages
        return {
            "t": env._step_count,
            "soldiers": soldiers,
            "enemies": [
                {"id": e.id, "alive": e.alive, "pos": list(e.pos)} for e in env.enemies
            ],
            "messages": [_message_record(env, m) for m in messages],
            # B5 order economics: this tick's re-task events, straight from
            # the environment's own adjudication (issuer rank, priced or
            # excepted and why, anchor rotation or same-anchor type change)
            "retasks": [] if initial else env.retask_events_last_step,
        }


def _knowledge_ttl() -> int:
    from cohort.env.cohort_env import KNOWLEDGE_TTL  # local: avoid a cycle at import time

    return KNOWLEDGE_TTL


def _message_record(env: CohortEnv, m) -> dict:
    def cs_of(agent_id: int | None) -> str:
        if agent_id is None:
            return "ALL"
        if agent_id == -1:
            return "HQ"
        soldier = env.roster.by_id.get(agent_id)
        return soldier.callsign if soldier is not None else f"#{agent_id}"

    mission = None
    if m.kind.value in ("order", "opord"):
        try:
            parsed_mission = lang.parse_order(m.text).mission
            # stance orders (A5-3, FORMATION X) carry no mission payload
            mission = parsed_mission.name if parsed_mission is not None else None
        except lang.OrderParseError:  # pragma: no cover - formats round-trip by invariant
            mission = None
    return {
        "kind": m.kind.value,
        "from": cs_of(m.sender_id),
        "to": cs_of(m.recipient_id),
        "mission": mission,
        "text": m.text,
    }


# ---------------------------------------------------------------------- #
# per-episode metrics (pure functions over a trace)
# ---------------------------------------------------------------------- #


def _soldier_at(step: dict, cs: str) -> dict | None:
    for rec in step["soldiers"]:
        if rec["cs"] == cs:
            return rec
    return None


def _obedience(trace: dict) -> tuple[list[int], int, dict[str, dict]]:
    """(latencies, censored, by_task): order applied → first step with compliance > 0.

    An *order event* is a step where an agent's standing mission carries
    ``step_assigned == t`` (OPORD at t=0 included). The event resolves at the
    first step, from the assignment step on, where the agent's per-step
    compliance score for that mission is positive; it is censored (counted,
    no latency) if the mission is replaced or cleared, the agent dies, or
    the episode ends first.

    ``by_task`` splits both by the ordered mission, because the pooled mean
    conflates "the cohort became disobedient" with "the cohort was ordered to
    do slower things". The defend line ran 1.19 / 1.26 steps at v6 / v8 and
    11.24 / 13.06 at v9 / v10, over the same stretch in which the ADVANCE share
    of orders went 0.69 → 0.99 — and an ADVANCE to a distant control measure
    cannot resolve as fast as a DEFEND in place. Which of those two it is
    cannot be read off the pooled number, so it is no longer only pooled.

    **Staged orders are not order events** (refs issue #15). A pending A5-2
    order (AT MY COMMAND before EXECUTE, AT T PLUS n before its tick) is one
    the recipient is obeying *by not executing*, and the environment scores it
    as HOLD at the staging spot — where the recipient already stands, so its
    compliance is positive from the tick the order lands. Booking that as an
    order event made every staged order resolve at latency **0**: an identical
    un-staged ADVANCE whose recipient never moved was censored, while the
    staged one read "obeyed instantly" (measured, fireteam seed 3). Because
    release restamps ``step_assigned``, the order books its real event at the
    release tick anyway — so the staged tick was a second, free, zero-latency
    copy of the same order, and the metric fell toward 0 in proportion to how
    much a policy staged. That is the *opposite* sign to cycle 8's hypothesis
    that staging inflates latency, and it is why the outside tap found
    incidence and duration both running backwards: v8 staged 0.878 of its
    ADVANCE orders and measured 1.01 steps, v10 staged 0.369 and measured
    16.21. Staged ticks are therefore skipped here and counted by
    :func:`_staging` instead, which also names the orders that are never
    released at all (61 of v8's 130) — those now leave no obedience event,
    which is correct: an order that never became effective was never binding.
    """
    steps = trace["steps"]
    latencies: list[int] = []
    censored = 0
    by_task: dict[str, dict] = {}

    def book(task: str, latency: int | None) -> None:
        slot = by_task.setdefault(task, {"latencies": [], "censored": 0})
        if latency is None:
            slot["censored"] += 1
        else:
            slot["latencies"].append(latency)
    for i, step in enumerate(steps):
        t0 = step["t"]
        for rec in step["soldiers"]:
            if not (rec["alive"] and rec["mission"] is not None and rec["since"] == t0):
                continue
            if rec.get("pending"):
                continue  # staged: not binding yet, and release restamps
            resolved = False
            for later in steps[max(i, 1):]:
                later_rec = _soldier_at(later, rec["cs"])
                if (
                    later_rec is None
                    or not later_rec["alive"]
                    or later_rec["mission"] != rec["mission"]
                    or later_rec["since"] != t0
                ):
                    break  # re-tasked, cleared, or dead before complying
                if later_rec.get("pending"):
                    continue  # compliance while staged is HOLD, not the task
                if later_rec["comp"] is not None and later_rec["comp"] > 0.0:
                    latencies.append(later["t"] - t0)
                    book(rec["mission"], later["t"] - t0)
                    resolved = True
                    break
            if not resolved:
                censored += 1
                book(rec["mission"], None)
    return latencies, censored, by_task


def _staging(trace: dict) -> dict[str, Any]:
    """A5-2 staging: how many orders staged, how many released, how long held.

    The counterpart to :func:`_obedience` skipping staged ticks (refs issue
    #15). Obedience cannot see a staged order — the recipient is complying by
    standing still — so without this the whole AT MY COMMAND / AT T PLUS n
    channel would be invisible in the behavior suite, and the interesting
    failure it can hide would be invisible with it: an order that is *issued
    and then abandoned*, staged until the episode ends because its issuer
    never sends EXECUTE. An outside tap found 61 of 130 of one checkpoint's
    staged orders in that state.

    A *staging span* runs from the tick a pending order lands to the tick it
    becomes effective — release restamps ``step_assigned`` to that tick, so a
    span is released exactly when the same soldier's next record carries the
    same mission, not pending, with ``since == t``. Anything else closes the
    span **abandoned**: re-tasked while staged, killed while staged, or still
    staged at the end of the episode. (A staged order superseded at its
    release tick by a fresh immediate order of the same task is
    indistinguishable from a release in the trace; it is a coincidence of one
    tick and one task, and it can only under-count abandonment.)
    """
    spans: dict[str, tuple[str, int]] = {}  # callsign -> (mission, staged-at)
    staged = 0
    released = 0
    gaps: list[int] = []
    for step in trace["steps"]:
        t = step["t"]
        for rec in step["soldiers"]:
            cs = rec["cs"]
            if rec["alive"] and rec["mission"] is not None and rec.get("pending"):
                key = (rec["mission"], rec["since"])
                if spans.get(cs) != key:  # a different staged order: the old one lapsed
                    spans[cs] = key
                    staged += 1
                continue
            span = spans.pop(cs, None)
            if span is None:
                continue
            mission, since = span
            if rec["alive"] and rec["mission"] == mission and rec["since"] == t:
                released += 1
                gaps.append(t - since)
    return {"orders_staged": staged, "staged_released": released, "staging_gaps": gaps}


def _reports(trace: dict) -> dict[str, int]:
    """CONTACT precision/recall counts against ground-truth visibility.

    Precision counts *informative* reports: a CONTACT whose content (the
    sender's ground-truth visible enemies at tick start) includes an enemy
    absent from the replayed team picture (new intel) or one whose picture
    entry has aged >= ``contact_refresh_age`` (a legitimate refresh). The
    picture replay mirrors the environment: entries expire after
    ``knowledge_ttl`` steps and dead enemies drop off. Recall counts enemy
    ids ever reported over enemy ids ever seen (union with reported, so the
    ratio is always <= 1).
    """
    steps = trace["steps"]
    refresh_age = trace["contact_refresh_age"]
    ttl = trace["knowledge_ttl"]
    picture: dict[int, int] = {}  # enemy id -> step of last report
    contacts = 0
    informative = 0
    seen: set[int] = set()
    for i, step in enumerate(steps):
        t = step["t"]
        for rec in step["soldiers"]:
            seen.update(rec["sees"])
        for msg in step["messages"]:
            if msg["kind"] != "contact":
                continue
            sender = _soldier_at(steps[i - 1], msg["from"]) if i > 0 else None
            content = list(sender["sees"]) if sender and sender["sees"] else []
            if not content:  # fallback: post-step visibility
                here = _soldier_at(step, msg["from"])
                content = list(here["sees"]) if here else []
            if not content:
                continue  # degenerate: content unknown, excluded entirely
            contacts += 1
            new = any(eid not in picture for eid in content)
            refresh = not new and any(t - picture[eid] >= refresh_age for eid in content)
            if new or refresh:
                informative += 1
            for eid in content:
                picture[eid] = t
        living = {e["id"] for e in step["enemies"] if e["alive"]}
        picture = {
            eid: last for eid, last in picture.items() if eid in living and t - last <= ttl
        }
    reported = {eid for ids in trace.get("reported", {}).values() for eid in ids}
    return {
        "contacts": contacts,
        "contacts_informative": informative,
        "enemies_seen": len(seen | reported),
        "enemies_reported": len(reported),
    }


#: Doctrine tiers an agent-issued order can fall into, in order of quality.
#: ``underivable`` is the issuer-had-no-mission case, which the order mask
#: makes unreachable in play but which a replayed / injected trace can show.
DOCTRINE_TIERS: tuple[str, ...] = ("preferred", "allowed", "violating", "underivable")


def _doctrine(trace: dict) -> dict[str, Any]:
    """Agent-issued orders split by doctrine tier, overall and by ordered task.

    An issued order's quality is judged against the issuer's mission at the
    moment of transmission: missions are read from the previous step's
    records and updated in within-step message order as orders are applied
    (an order counts as applied when the recipient's end-of-step mission
    matches it with ``step_assigned == t``). HQ traffic (OPORD, injected
    orders) is not an agent decision and is excluded from the rate.

    The tier split exists because the preference rate alone cannot tell
    *adopting a legal alternative* from *disregarding doctrine* — both merely
    fail to be ``allowed[0]`` (refs issue #14). Since A5 added ADVANCE to the
    DEFEND / SEIZE / RECON derivation tables, a policy that orders ADVANCE
    almost exclusively scores ~0 preference while committing zero doctrine
    violations, so the rate reads as a collapse when it is catalog adoption.
    ``orders_by_task`` makes that adoption visible: preference conditioned on
    the ordered task, against the task's share of all orders.
    """
    steps = trace["steps"]
    issued = 0
    tiers = dict.fromkeys(DOCTRINE_TIERS, 0)
    by_task: dict[str, dict[str, int]] = {}
    for i in range(1, len(steps)):
        step = steps[i]
        t = step["t"]
        cur: dict[str, str | None] = {
            rec["cs"]: rec["mission"] for rec in steps[i - 1]["soldiers"]
        }
        for msg in step["messages"]:
            if msg["kind"] not in ("order", "opord") or msg["mission"] is None:
                continue
            if msg["kind"] == "order" and msg["from"] != "HQ":
                issued += 1
                own = cur.get(msg["from"])
                allowed = allowed_derivations(MissionType[own]) if own is not None else ()
                task = MissionType[msg["mission"]]
                if not allowed:
                    tier = "underivable"
                elif task is allowed[0]:
                    tier = "preferred"
                elif task in allowed:
                    tier = "allowed"
                else:
                    tier = "violating"
                tiers[tier] += 1
                bucket = by_task.setdefault(task.name, dict.fromkeys(DOCTRINE_TIERS, 0))
                bucket[tier] += 1
            recipient = _soldier_at(step, msg["to"])
            applied = (
                recipient is not None
                and recipient["mission"] == msg["mission"]
                and recipient["since"] == t
            )
            if applied:
                cur[msg["to"]] = msg["mission"]
    return {
        "orders_issued": issued,
        "orders_preferred": tiers["preferred"],
        "orders_allowed": tiers["allowed"],
        "orders_violating": tiers["violating"],
        "orders_underivable": tiers["underivable"],
        "orders_by_task": by_task,
    }


def _retasks(trace: dict) -> dict[str, Any]:
    """Re-task economics counts (B5), read from the environment's own
    adjudication recorded per step: total re-tasks (orders replacing a
    standing mission — fresh taskings and identical reissues are not
    re-tasks), how many were priced vs. excepted (the tactical-picture
    carve-out: contact / element casualty / issuer intent change), how many
    rotated the anchor (vs. same-anchor mission-type changes), and the
    per-issuer-rank split of priced vs. excepted."""
    total = priced = excepted = rotations = 0
    by_rank: dict[str, dict[str, int]] = {}
    for step in trace["steps"]:
        for ev in step.get("retasks", []):
            total += 1
            bucket = by_rank.setdefault(ev["rank"], {"priced": 0, "excepted": 0})
            if ev["excepted"]:
                excepted += 1
                bucket["excepted"] += 1
            else:
                priced += 1
                bucket["priced"] += 1
            if not ev["same_anchor"]:
                rotations += 1
    return {
        "retasks": total,
        "retasks_priced": priced,
        "retasks_excepted": excepted,
        "retask_rotations": rotations,
        "retasks_by_rank": by_rank,
    }


def _vocabulary(trace: dict) -> dict[str, Any]:
    """A5 vocabulary usage: what share of the traffic uses the new forms.

    Counts ADVANCE orders (control-measure targets), timing-qualified orders
    (AT T PLUS / AT MY COMMAND) with their EXECUTE releases, FORMATION stance
    orders, and trinôme sync traffic (proposals / GO bounds); plus the share
    of living-agent steps governed by a formation stance (own, or the direct
    leader's — the geometry actually being shaped).
    """
    advance = timed = formation = executes = proposals = bounds = 0
    stance_steps = 0
    agent_steps = 0
    for step in trace["steps"]:
        stanced = {
            rec["cs"] for rec in step["soldiers"] if rec.get("formation") is not None
        }
        for rec in step["soldiers"]:
            if not rec["alive"]:
                continue
            agent_steps += 1
            if rec["cs"] in stanced or rec.get("leader") in stanced:
                stance_steps += 1
        for msg in step["messages"]:
            kind = msg["kind"]
            if kind == "execute":
                executes += 1
            elif kind == "sync_propose":
                proposals += 1
            elif kind == "sync_go":
                bounds += 1
            elif kind in ("order", "opord"):
                try:
                    parsed = lang.parse_order(msg["text"])
                except lang.OrderParseError:  # pragma: no cover
                    continue
                if parsed.formation is not None:
                    formation += 1
                if parsed.mission is MissionType.ADVANCE:
                    advance += 1
                if parsed.delay is not None or parsed.at_my_command:
                    timed += 1
    return {
        "advance_orders": advance,
        "timed_orders": timed,
        "execute_signals": executes,
        "formation_orders": formation,
        "sync_proposals": proposals,
        "sync_bounds": bounds,
        "stance_steps": stance_steps,
        "stance_agent_steps": agent_steps,
    }


def _false_complete(trace: dict) -> tuple[int, int]:
    """(DONE reports transmitted, of which rejected by the superior)."""
    dones = 0
    rejected = 0
    for step in trace["steps"]:
        for msg in step["messages"]:
            if msg["kind"] == "done":
                dones += 1
            elif msg["kind"] == "done_reject":
                rejected += 1
    return dones, rejected


def _done_opportunity(trace: dict) -> dict[str, int]:
    """Agent-steps at which MISSION COMPLETE was admissible (all / the root's).

    The denominator ``done_reports`` never had (refs #13). Zero DONE reports
    is two opposite findings wearing the same face: an admissible-step count
    of zero means the channel was shut and no price was ever consulted (the
    ``is_root_opord_claim`` mask bug, which sat unseen for a generation
    because the pipeline could not see this); a large count with no claims
    means the policy was *offered* the act and declined it, which is a
    statement about the price of a false claim, not about reachability.

    Counted over every recorded state but the last — the terminal state is
    followed by no action, so it is not an opportunity anybody had.
    """
    admissible = 0
    root_admissible = 0
    for step in trace["steps"][:-1]:
        for rec in step["soldiers"]:
            if not rec.get("done_ok"):
                continue
            admissible += 1
            root_admissible += bool(rec.get("root"))
    return {"done_admissible": admissible, "done_admissible_root": root_admissible}


def _succession(trace: dict) -> tuple[int, list[int], int]:
    """(leader-death events, recovery times, unrecovered events).

    A leader death is the death of an agent with living direct subordinates.
    The orphaned set is those subordinates minus the successor (the agent
    announcing "I AM ASSUMING COMMAND" for the fallen leader). The event
    recovers at the first step where every still-living orphan holds a
    mission assigned at or after the death step; dead orphans drop out of
    the requirement. Events not recovered by episode end are censored.
    """
    steps = trace["steps"]
    events = 0
    recovery: list[int] = []
    unrecovered = 0
    for i in range(1, len(steps)):
        step = steps[i]
        t_death = step["t"]
        for rec in step["soldiers"]:
            prev = _soldier_at(steps[i - 1], rec["cs"])
            if rec["alive"] or prev is None or not prev["alive"]:
                continue  # not a death this step
            orphans = list(prev["subs"])
            if not orphans:
                continue  # no command to devolve
            marker = f"{rec['cs']} IS DOWN. I AM ASSUMING COMMAND"
            successor = next(
                (
                    m["from"]
                    for m in step["messages"]
                    if m["kind"] == "taking_command" and marker in m["text"]
                ),
                None,
            )
            orphans = [cs for cs in orphans if cs != successor]
            events += 1
            recovered = None
            for later in steps[i:]:
                ok = True
                for cs in orphans:
                    o = _soldier_at(later, cs)
                    if o is None or not o["alive"]:
                        continue  # dead orphans drop out of the requirement
                    if o["mission"] is None or o["since"] is None or o["since"] < t_death:
                        ok = False
                        break
                if ok:
                    recovered = later["t"] - t_death
                    break
            if recovered is None:
                unrecovered += 1
            else:
                recovery.append(recovered)
    return events, recovery, unrecovered


def _coverage(trace: dict) -> tuple[int, int]:
    """(leader-step pairs with >=1 living subordinate, of which all tasked).

    Mirrors the environment's coverage scoring: a pair counts when a living
    leader (effective authority > 0) holds a mission and has at least one
    living direct subordinate; it is covered when every one of them is
    tasked that step.
    """
    pairs = 0
    covered = 0
    for step in trace["steps"][1:]:
        tasked = {rec["cs"]: rec["mission"] is not None for rec in step["soldiers"]}
        for rec in step["soldiers"]:
            if not (rec["alive"] and rec["auth"] > 0 and rec["mission"] is not None and rec["subs"]):
                continue
            pairs += 1
            if all(tasked.get(cs, False) for cs in rec["subs"]):
                covered += 1
    return pairs, covered


def _human_exposure(trace: dict) -> dict[str, Any]:
    """Exposure of the human root: enemy proximity, ring entries, death.

    Measured over the steps the human is alive. ``ring_entries`` counts
    outside→inside transitions of the objective observation ring
    (``ring_radius`` around the root objective; spawning inside counts as
    the first entry). All values are None when the scenario has no human
    or, for the distance means, when the relevant reference never exists
    (no living enemy while the human lives; no root objective).
    """
    human = trace.get("human")
    if human is None:
        return {
            "human_died": None,
            "human_mean_enemy_dist": None,
            "human_mean_objective_dist": None,
            "human_ring_entries": None,
        }
    obj = trace.get("root_objective")
    ring = trace.get("ring_radius", RING_RADIUS)
    enemy_dists: list[float] = []
    obj_dists: list[float] = []
    entries = 0
    inside_prev = False
    died = False
    for step in trace["steps"]:
        rec = _soldier_at(step, human)
        if rec is None:
            continue
        if not rec["alive"]:
            died = True
            break
        living = [e["pos"] for e in step["enemies"] if e["alive"]]
        if living:
            enemy_dists.append(min(_dist(rec["pos"], p) for p in living))
        if obj is not None:
            d = _dist(rec["pos"], obj)
            obj_dists.append(d)
            inside = d <= ring
            if inside and not inside_prev:
                entries += 1
            inside_prev = inside
    return {
        "human_died": died,
        "human_mean_enemy_dist": (sum(enemy_dists) / len(enemy_dists)) if enemy_dists else None,
        "human_mean_objective_dist": (sum(obj_dists) / len(obj_dists)) if obj_dists else None,
        "human_ring_entries": entries if obj is not None else None,
    }


def _fight_disposition(trace: dict) -> dict[str, Any]:
    """Where the cohort fights when the enemy is on it (refs issue #11).

    Scores the (living soldier, step) pairs that are **under threat** — a
    living enemy within ``threat_radius`` (the scenario's weapon range), i.e.
    pairs where the soldier's ground actually costs or saves it something.
    Two numbers come out of that population:

    * cover occupancy — share of those pairs spent on cover terrain;
    * distance from the root objective — mean over those pairs.

    Conditioning on threat is what makes the pair informative: averaged over
    a whole episode, an approach march and a prepared defense look alike.
    Both are None when the episode never came under threat; the distance is
    also None when the scenario has no root objective.
    """
    obj = trace.get("root_objective")
    radius = trace.get("threat_radius", THREAT_RADIUS)
    pairs = 0
    in_cover = 0
    dist_sum = 0.0
    dist_n = 0
    for step in trace["steps"]:
        living = [e["pos"] for e in step["enemies"] if e["alive"]]
        if not living:
            continue
        for rec in step["soldiers"]:
            if not rec["alive"]:
                continue
            if min(_dist(rec["pos"], p) for p in living) > radius:
                continue
            pairs += 1
            in_cover += bool(rec.get("cover"))
            if obj is not None:
                dist_sum += _dist(rec["pos"], obj)
                dist_n += 1
    return {
        "threat_pairs": pairs,
        "threat_cover_pairs": in_cover,
        "threat_objective_dist_sum": dist_sum,
        "threat_objective_dist_pairs": dist_n,
    }


def episode_behavior(trace: dict) -> dict[str, Any]:
    """Reduce one episode trace to its behavioral event counts and lists."""
    latencies, censored, obedience_by_task = _obedience(trace)
    dones, rejected = _false_complete(trace)
    events, recovery, unrecovered = _succession(trace)
    pairs, covered = _coverage(trace)
    return {
        "outcome": trace.get("outcome"),
        "length": trace.get("length"),
        # carried through so the aggregate knows which regression gates apply
        "root_mission": trace.get("root_mission"),
        "obedience_latencies": latencies,
        "obedience_censored": censored,
        "obedience_by_task": obedience_by_task,
        **_staging(trace),
        **_doctrine(trace),
        **_retasks(trace),
        **_reports(trace),
        "done_reports": dones,
        "done_rejected": rejected,
        **_done_opportunity(trace),
        "succession_events": events,
        "succession_recovery": recovery,
        "succession_unrecovered": unrecovered,
        "coverage_pairs": pairs,
        "coverage_covered": covered,
        **_vocabulary(trace),
        **_human_exposure(trace),
        **_fight_disposition(trace),
    }


# ---------------------------------------------------------------------- #
# run-level aggregation
# ---------------------------------------------------------------------- #


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate_behavior(episodes: list[dict]) -> dict[str, Any]:
    """Pool per-episode behavior dicts into the run-level metric summary.

    Event-level metrics pool events across episodes (a latency mean over all
    orders of the run, rates over total counts); the human-exposure means
    average per-episode values so long and short episodes weigh equally.
    Metrics whose denominator never occurred are None, with the counts kept
    alongside so a None is always explainable.
    """
    def total(key: str) -> int:
        return sum(ep.get(key, 0) for ep in episodes)

    n_eps = len(episodes)
    retasks_by_rank: dict[str, dict[str, int]] = {}
    for ep in episodes:
        for rank, bucket in ep.get("retasks_by_rank", {}).items():
            dst = retasks_by_rank.setdefault(rank, {"priced": 0, "excepted": 0})
            dst["priced"] += bucket.get("priced", 0)
            dst["excepted"] += bucket.get("excepted", 0)

    orders_by_task: dict[str, dict[str, int]] = {}
    for ep in episodes:
        for task, bucket in ep.get("orders_by_task", {}).items():
            dst = orders_by_task.setdefault(task, dict.fromkeys(DOCTRINE_TIERS, 0))
            for tier in DOCTRINE_TIERS:
                dst[tier] += bucket.get(tier, 0)
    # judgeable orders: the issuer held a mission to derive from
    derivable = total("orders_preferred") + total("orders_allowed") + total("orders_violating")

    latencies = [v for ep in episodes for v in ep["obedience_latencies"]]
    # refs the v9/v10 obedience regression: pooled latency cannot separate
    # "the cohort stopped obeying" from "the cohort was ordered to do slower
    # things", so it is also reported per ordered task
    obedience_by_task: dict[str, dict] = {}
    for ep in episodes:
        for task, bucket in ep.get("obedience_by_task", {}).items():
            dst = obedience_by_task.setdefault(task, {"latencies": [], "censored": 0})
            dst["latencies"].extend(bucket["latencies"])
            dst["censored"] += bucket["censored"]
    obedience_task_summary = {
        task: {
            "latency_mean": _mean(b["latencies"]),
            "orders": len(b["latencies"]) + b["censored"],
            "censored": b["censored"],
        }
        for task, b in sorted(obedience_by_task.items())
    }
    recovery = [v for ep in episodes for v in ep["succession_recovery"]]
    humans = [ep for ep in episodes if ep["human_died"] is not None]
    roots = {ep.get("root_mission") for ep in episodes} - {None}
    return {
        "episodes": n_eps,
        # the run's root mission when the pooled episodes agree on one; the
        # regression gates key off it (a mixed pool gates on nothing)
        "root_mission": roots.pop() if len(roots) == 1 else None,
        "success_rate": _ratio(
            sum(ep["outcome"] == "success" for ep in episodes), n_eps
        ),
        "obedience_latency_mean": _mean(latencies),
        "obedience_orders": len(latencies) + total("obedience_censored"),
        "obedience_censored": total("obedience_censored"),
        "obedience_by_task": obedience_task_summary,
        # A5-2 staging (refs #15): the channel obedience latency deliberately
        # cannot see. `staged_abandoned` is the one that reads as a fault —
        # an order issued AT MY COMMAND whose EXECUTE never came.
        "orders_staged": total("orders_staged"),
        "staged_released": total("staged_released"),
        "staged_abandoned": total("orders_staged") - total("staged_released"),
        "staging_gap_mean": _mean([g for ep in episodes for g in ep.get("staging_gaps", [])]),
        "report_precision": _ratio(total("contacts_informative"), total("contacts")),
        "report_recall": _ratio(total("enemies_reported"), total("enemies_seen")),
        "contact_reports": total("contacts"),
        "doctrine_preference_rate": _ratio(total("orders_preferred"), total("orders_issued")),
        # refs #14: preference is `allowed[0]`, so a policy that adopts a
        # legal alternative leg (ADVANCE, added to the DEFEND/SEIZE/RECON
        # tables in A5) scores identically to one that disregards doctrine.
        # The containment rate separates them; orders_by_task shows whether a
        # low preference rate is catalog adoption or command quality.
        "doctrine_allowed_rate": _ratio(
            total("orders_preferred") + total("orders_allowed"), derivable
        ),
        "orders_allowed": total("orders_allowed"),
        "orders_violating": total("orders_violating"),
        "orders_underivable": total("orders_underivable"),
        "orders_by_task": orders_by_task,
        "orders_issued": total("orders_issued"),
        "orders_per_episode": _ratio(total("orders_issued"), n_eps),
        "retasks": total("retasks"),
        "retasks_priced": total("retasks_priced"),
        "retasks_excepted": total("retasks_excepted"),
        "retask_rotations": total("retask_rotations"),
        "retasks_per_episode": _ratio(total("retasks"), n_eps),
        "retasks_priced_per_episode": _ratio(total("retasks_priced"), n_eps),
        "retasks_by_rank": retasks_by_rank,
        "false_complete_rate": _ratio(total("done_rejected"), total("done_reports")),
        "done_reports": total("done_reports"),
        "done_rejected": total("done_rejected"),
        # refs #13: the opportunity denominator. done_claim_rate is None only
        # when the channel was never open — which is itself the finding.
        "done_admissible": total("done_admissible"),
        "done_admissible_root": total("done_admissible_root"),
        "done_claim_rate": _ratio(total("done_reports"), total("done_admissible")),
        "succession_recovery_mean": _mean(recovery),
        "succession_events": total("succession_events"),
        "succession_unrecovered": total("succession_unrecovered"),
        "coverage_time": _ratio(total("coverage_covered"), total("coverage_pairs")),
        # A5 vocabulary usage
        "advance_orders_per_episode": _ratio(total("advance_orders"), n_eps),
        "timed_orders_per_episode": _ratio(total("timed_orders"), n_eps),
        "execute_signals_per_episode": _ratio(total("execute_signals"), n_eps),
        "formation_orders_per_episode": _ratio(total("formation_orders"), n_eps),
        "sync_proposals_per_episode": _ratio(total("sync_proposals"), n_eps),
        "sync_bounds_per_episode": _ratio(total("sync_bounds"), n_eps),
        "stance_share": _ratio(total("stance_steps"), total("stance_agent_steps")),
        "human_death_rate": _ratio(sum(ep["human_died"] for ep in humans), len(humans)),
        "human_mean_enemy_dist": _mean([ep["human_mean_enemy_dist"] for ep in humans]),
        "human_mean_objective_dist": _mean([ep["human_mean_objective_dist"] for ep in humans]),
        "human_ring_entries_mean": _mean([ep["human_ring_entries"] for ep in humans]),
        # fight disposition (issue #11): pooled over threatened (soldier,
        # step) pairs, so a long firefight weighs more than a brief brush —
        # which is the intent, the question being where the fighting happens
        "cover_occupancy_under_threat": _ratio(total("threat_cover_pairs"), total("threat_pairs")),
        "mean_distance_from_objective_under_threat": _ratio(
            total("threat_objective_dist_sum"), total("threat_objective_dist_pairs")
        ),
        "threat_pairs": total("threat_pairs"),
    }


# ---------------------------------------------------------------------- #
# regression gates
# ---------------------------------------------------------------------- #


def regression_gates(agg: dict[str, Any]) -> list[dict[str, Any]]:
    """Pass/fail bounds a retrain must clear, given an aggregated summary.

    The positional gate (issue #11) applies to DEFEND roots only: holding
    ground is the one root mission for which "fought here rather than there"
    is a correctness property rather than a style. It exists because success
    rate alone did not catch it — ``fireteam_defend_v7`` halved the
    root-death rate the ROADMAP had blamed, was measured firing on every
    threatened step (p(fire | threatened) 0.005 -> 1.000), and still missed,
    because it had walked off the position it was ordered to hold. Three
    million steps bought that lesson; these two numbers are ~free.

    Gates whose metric is None (never measured this run) are reported with
    ``passed=None`` — unmeasured is not the same as passed.
    """
    if agg.get("root_mission") != MissionType.DEFEND.name:
        return []
    return [
        _gate(
            "cover_occupancy_under_threat",
            agg.get("cover_occupancy_under_threat"),
            DEFEND_COVER_FLOOR,
            "min",
        ),
        _gate(
            "mean_distance_from_objective_under_threat",
            agg.get("mean_distance_from_objective_under_threat"),
            DEFEND_OBJECTIVE_DIST_CEILING,
            "max",
        ),
    ]


def _gate(name: str, value: float | None, bound: float, direction: str) -> dict[str, Any]:
    if value is None:
        passed = None
    else:
        passed = value >= bound if direction == "min" else value <= bound
    return {"name": name, "value": value, "bound": bound, "direction": direction, "passed": passed}


def format_gate_report(gates: list[dict[str, Any]]) -> str:
    """Human-readable gate verdicts; empty string when nothing gates."""
    if not gates:
        return ""
    lines = ["regression gates:"]
    for g in gates:
        mark = "—" if g["passed"] is None else ("PASS" if g["passed"] else "FAIL")
        value = "—" if g["value"] is None else f"{g['value']:.3f}"
        rel = ">=" if g["direction"] == "min" else "<="
        lines.append(f"  [{mark:^4}] {g['name']:<42} {value:>7}  ({rel} {g['bound']})")
    return "\n".join(lines)


#: (key, label, format) rows of the printed behavior table, in display order.
_TABLE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("obedience_latency_mean", "obedience latency (steps)", "{:.1f}"),
    ("orders_per_episode", "orders issued / ep", "{:.1f}"),
    ("retasks_per_episode", "re-tasks / ep", "{:.1f}"),
    ("retasks_priced_per_episode", "priced re-tasks / ep", "{:.2f}"),
    ("report_precision", "report precision", "{:.2f}"),
    ("report_recall", "report recall", "{:.2f}"),
    ("doctrine_preference_rate", "doctrine preference", "{:.2f}"),
    ("doctrine_allowed_rate", "doctrine containment", "{:.2f}"),
    ("false_complete_rate", "false-COMPLETE rate", "{:.2f}"),
    ("done_claim_rate", "COMPLETE claim rate", "{:.4f}"),
    ("succession_recovery_mean", "succession recovery (steps)", "{:.1f}"),
    ("coverage_time", "subordinate coverage time", "{:.2f}"),
    ("advance_orders_per_episode", "ADVANCE orders / ep", "{:.1f}"),
    ("timed_orders_per_episode", "timed orders / ep", "{:.2f}"),
    ("formation_orders_per_episode", "FORMATION orders / ep", "{:.2f}"),
    ("stance_share", "stance-governed step share", "{:.2f}"),
    ("sync_bounds_per_episode", "sync bounds (GO) / ep", "{:.2f}"),
    ("human_mean_enemy_dist", "human: mean dist to enemy", "{:.1f}"),
    ("human_ring_entries_mean", "human: ring entries / ep", "{:.2f}"),
    ("human_death_rate", "human: death rate", "{:.2f}"),
    ("cover_occupancy_under_threat", "fight: cover occupancy", "{:.3f}"),
    ("mean_distance_from_objective_under_threat", "fight: dist from OBJ", "{:.2f}"),
)


def format_obedience_by_task(agg: dict[str, Any], top: int = 3) -> str:
    """``TASK mean(n)`` obedience latency for the most-ordered tasks.

    The pooled mean cannot tell "the cohort stopped obeying" from "the cohort
    was ordered to do slower things" — the defend line went 1.26 steps at v8 to
    13.06 at v10 while the ADVANCE share of orders went 0.69 → 0.99, and an
    ADVANCE to a distant control measure resolves slower than a DEFEND in place
    no matter how obedient anyone is. Split per task, the two read differently:
    a rise concentrated in ADVANCE is a mix shift; a rise *within* DEFEND is a
    real obedience regression.
    """
    by_task = agg.get("obedience_by_task") or {}
    ranked = sorted(by_task.items(), key=lambda kv: -kv[1]["orders"])[:top]
    parts = []
    for task, b in ranked:
        mean = b["latency_mean"]
        shown = f"{mean:.1f}" if mean is not None else "—"
        parts.append(f"{task} {shown}({b['orders']})")
    return ", ".join(parts)


def format_order_task_mix(agg: dict[str, Any], top: int = 3) -> str:
    """``TASK share/preference`` for the most-ordered tasks (refs #14).

    Reads the preference rate conditioned on the ordered task against that
    task's share of all agent-issued orders, so a low overall preference can
    be attributed: ``ADVANCE .96/.00`` is a policy that adopted one legal
    alternative leg wholesale, not one issuing bad orders.
    """
    by_task = agg.get("orders_by_task") or {}
    issued = sum(sum(b.values()) for b in by_task.values())
    if not issued:
        return ""
    ranked = sorted(by_task.items(), key=lambda kv: -sum(kv[1].values()))[:top]
    return ", ".join(
        f"{task} {sum(b.values()) / issued:.2f}/{b.get('preferred', 0) / sum(b.values()):.2f}"
        for task, b in ranked
    )


def format_staging(agg: dict[str, Any]) -> str:
    """``staged N, released N (gap X), abandoned N`` — the A5-2 staging channel.

    Empty when nothing was ever staged. ``abandoned`` is the number worth
    reading: those orders were transmitted, staged a recipient, and never
    released, so they bought a held agent and no execution (refs #15).
    """
    staged = agg.get("orders_staged") or 0
    if not staged:
        return ""
    gap = agg.get("staging_gap_mean")
    shown = f"{gap:.1f}" if gap is not None else "—"
    return (
        f"staged {staged}, released {agg.get('staged_released', 0)} "
        f"(mean gap {shown} steps), abandoned {agg.get('staged_abandoned', 0)}"
    )


def format_behavior_table(agg: dict[str, Any]) -> str:
    """Human-readable table of an aggregated behavior summary."""
    by_rank = ", ".join(
        f"{rank} {b['priced']}p/{b['excepted']}e"
        for rank, b in sorted(agg.get("retasks_by_rank", {}).items())
    )
    task_mix = format_order_task_mix(agg)
    notes = {
        "obedience_latency_mean": (
            f"n={agg['obedience_orders']}, censored {agg['obedience_censored']}"
            + (f", staged {agg['orders_staged']} excluded" if agg.get("orders_staged") else "")
        ),
        "timed_orders_per_episode": format_staging(agg),
        "retasks_per_episode": (
            f"priced {agg.get('retasks_priced', 0)}, excepted {agg.get('retasks_excepted', 0)}, "
            f"rotations {agg.get('retask_rotations', 0)}"
            + (f"; by rank: {by_rank}" if by_rank else "")
        ),
        "report_precision": f"n={agg['contact_reports']}",
        "doctrine_preference_rate": f"n={agg['orders_issued']}" + (f"; {task_mix}" if task_mix else ""),
        "doctrine_allowed_rate": (
            f"{agg.get('orders_allowed', 0)} allowed, {agg.get('orders_violating', 0)} violating"
        ),
        "false_complete_rate": f"n={agg['done_reports']}",
        "done_claim_rate": (
            f"{agg.get('done_admissible', 0)} admissible agent-steps "
            f"({agg.get('done_admissible_root', 0)} the root's)"
        ),
        "succession_recovery_mean": (
            f"n={agg['succession_events']}, unrecovered {agg['succession_unrecovered']}"
        ),
        "cover_occupancy_under_threat": f"n={agg.get('threat_pairs', 0)} threatened agent-steps",
    }
    notes = {k: v for k, v in notes.items() if v}
    lines = [f"behavior over {agg['episodes']} episodes:"]
    for key, label, fmt in _TABLE_ROWS:
        value = agg.get(key)
        text = "—" if value is None else fmt.format(value)
        note = f"   ({notes[key]})" if key in notes else ""
        lines.append(f"  {label:<28} {text:>7}{note}")
    return "\n".join(lines)
