"""Behavioral metrics suite: measure what "behaves like its rank" means (B2).

Success rate says *whether* the cohort wins; this module measures *how it
behaves* while doing so, per evaluation run:

* obedience latency      — order received → first compliant action
* report precision/recall — CONTACT reports vs. enemies actually seen
  (the oracle-side ground truth: per-step visibility)
* doctrine-preference rate — share of issued orders that were the preferred
  derivation of the issuer's own mission
* false-COMPLETE rate    — MISSION COMPLETE claims rejected by the umpire
* succession recovery    — leader death → all orphaned subordinates re-tasked
* subordinate coverage   — share of steps every living subordinate is tasked
* human exposure         — the human root's distance to the nearest living
  enemy, its entries into the objective observation ring, and its death rate
  (refs issue #9: rolling success is blind to a policy re-learning to walk
  the commander into the ring, so checkpoint selection for preservation
  claims needs these numbers)

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
from cohort.core.missions import IN_POSITION_RADIUS, MissionType, allowed_derivations, compliance

if TYPE_CHECKING:
    from cohort.env.cohort_env import CohortEnv

#: Radius of the objective observation ring used for human ring entries —
#: RECON/SCREEN share it; it is where issue #9 measured the root's exposure.
RING_RADIUS: float = IN_POSITION_RADIUS[MissionType.RECON]


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
        for s in env.roster.soldiers:
            comp = None
            if not initial and s.alive and s.mission is not None:
                ctx = env._compliance_ctx(s, self._prev_dist.get(s.callsign), env._make_view(s))
                comp = compliance(s.mission.type, ctx)
            leader = env.roster.leader_of(s)
            soldiers.append(
                {
                    "cs": s.callsign,
                    "alive": s.alive,
                    "pos": list(s.pos),
                    "mission": s.mission.type.name if s.mission is not None else None,
                    "since": s.mission.step_assigned if s.mission is not None else None,
                    "auth": s.effective_authority,
                    "subs": [x.callsign for x in s.living_subordinates(env.roster)],
                    "leader": leader.callsign if leader is not None else None,
                    "comp": comp,
                    "fired": bool(s.fired_this_step) if s.alive else False,
                    "sees": [e.id for e in env._visible_enemies(s)] if s.alive else [],
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
            mission = lang.parse_order(m.text).mission.name
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


def _obedience(trace: dict) -> tuple[list[int], int]:
    """(latencies, censored): order applied → first step with compliance > 0.

    An *order event* is a step where an agent's standing mission carries
    ``step_assigned == t`` (OPORD at t=0 included). The event resolves at the
    first step, from the assignment step on, where the agent's per-step
    compliance score for that mission is positive; it is censored (counted,
    no latency) if the mission is replaced or cleared, the agent dies, or
    the episode ends first.
    """
    steps = trace["steps"]
    latencies: list[int] = []
    censored = 0
    for i, step in enumerate(steps):
        t0 = step["t"]
        for rec in step["soldiers"]:
            if not (rec["alive"] and rec["mission"] is not None and rec["since"] == t0):
                continue
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
                if later_rec["comp"] is not None and later_rec["comp"] > 0.0:
                    latencies.append(later["t"] - t0)
                    resolved = True
                    break
            if not resolved:
                censored += 1
    return latencies, censored


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


def _doctrine(trace: dict) -> tuple[int, int]:
    """(orders issued by agents, of which doctrine-preferred).

    An issued order's quality is judged against the issuer's mission at the
    moment of transmission: missions are read from the previous step's
    records and updated in within-step message order as orders are applied
    (an order counts as applied when the recipient's end-of-step mission
    matches it with ``step_assigned == t``). HQ traffic (OPORD, injected
    orders) is not an agent decision and is excluded from the rate.
    """
    steps = trace["steps"]
    issued = 0
    preferred = 0
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
                if allowed and MissionType[msg["mission"]] is allowed[0]:
                    preferred += 1
            recipient = _soldier_at(step, msg["to"])
            applied = (
                recipient is not None
                and recipient["mission"] == msg["mission"]
                and recipient["since"] == t
            )
            if applied:
                cur[msg["to"]] = msg["mission"]
    return issued, preferred


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


def episode_behavior(trace: dict) -> dict[str, Any]:
    """Reduce one episode trace to its behavioral event counts and lists."""
    latencies, censored = _obedience(trace)
    issued, preferred = _doctrine(trace)
    dones, rejected = _false_complete(trace)
    events, recovery, unrecovered = _succession(trace)
    pairs, covered = _coverage(trace)
    return {
        "outcome": trace.get("outcome"),
        "length": trace.get("length"),
        "obedience_latencies": latencies,
        "obedience_censored": censored,
        "orders_issued": issued,
        "orders_preferred": preferred,
        **_reports(trace),
        "done_reports": dones,
        "done_rejected": rejected,
        "succession_events": events,
        "succession_recovery": recovery,
        "succession_unrecovered": unrecovered,
        "coverage_pairs": pairs,
        "coverage_covered": covered,
        **_human_exposure(trace),
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
        return sum(ep[key] for ep in episodes)

    latencies = [v for ep in episodes for v in ep["obedience_latencies"]]
    recovery = [v for ep in episodes for v in ep["succession_recovery"]]
    humans = [ep for ep in episodes if ep["human_died"] is not None]
    return {
        "episodes": len(episodes),
        "success_rate": _ratio(
            sum(ep["outcome"] == "success" for ep in episodes), len(episodes)
        ),
        "obedience_latency_mean": _mean(latencies),
        "obedience_orders": len(latencies) + total("obedience_censored"),
        "obedience_censored": total("obedience_censored"),
        "report_precision": _ratio(total("contacts_informative"), total("contacts")),
        "report_recall": _ratio(total("enemies_reported"), total("enemies_seen")),
        "contact_reports": total("contacts"),
        "doctrine_preference_rate": _ratio(total("orders_preferred"), total("orders_issued")),
        "orders_issued": total("orders_issued"),
        "false_complete_rate": _ratio(total("done_rejected"), total("done_reports")),
        "done_reports": total("done_reports"),
        "done_rejected": total("done_rejected"),
        "succession_recovery_mean": _mean(recovery),
        "succession_events": total("succession_events"),
        "succession_unrecovered": total("succession_unrecovered"),
        "coverage_time": _ratio(total("coverage_covered"), total("coverage_pairs")),
        "human_death_rate": _ratio(sum(ep["human_died"] for ep in humans), len(humans)),
        "human_mean_enemy_dist": _mean([ep["human_mean_enemy_dist"] for ep in humans]),
        "human_mean_objective_dist": _mean([ep["human_mean_objective_dist"] for ep in humans]),
        "human_ring_entries_mean": _mean([ep["human_ring_entries"] for ep in humans]),
    }


#: (key, label, format) rows of the printed behavior table, in display order.
_TABLE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("obedience_latency_mean", "obedience latency (steps)", "{:.1f}"),
    ("report_precision", "report precision", "{:.2f}"),
    ("report_recall", "report recall", "{:.2f}"),
    ("doctrine_preference_rate", "doctrine preference", "{:.2f}"),
    ("false_complete_rate", "false-COMPLETE rate", "{:.2f}"),
    ("succession_recovery_mean", "succession recovery (steps)", "{:.1f}"),
    ("coverage_time", "subordinate coverage time", "{:.2f}"),
    ("human_mean_enemy_dist", "human: mean dist to enemy", "{:.1f}"),
    ("human_ring_entries_mean", "human: ring entries / ep", "{:.2f}"),
    ("human_death_rate", "human: death rate", "{:.2f}"),
)


def format_behavior_table(agg: dict[str, Any]) -> str:
    """Human-readable table of an aggregated behavior summary."""
    notes = {
        "obedience_latency_mean": f"n={agg['obedience_orders']}, censored {agg['obedience_censored']}",
        "report_precision": f"n={agg['contact_reports']}",
        "doctrine_preference_rate": f"n={agg['orders_issued']}",
        "false_complete_rate": f"n={agg['done_reports']}",
        "succession_recovery_mean": (
            f"n={agg['succession_events']}, unrecovered {agg['succession_unrecovered']}"
        ),
    }
    lines = [f"behavior over {agg['episodes']} episodes:"]
    for key, label, fmt in _TABLE_ROWS:
        value = agg.get(key)
        text = "—" if value is None else fmt.format(value)
        note = f"   ({notes[key]})" if key in notes else ""
        lines.append(f"  {label:<28} {text:>7}{note}")
    return "\n".join(lines)
