"""Transparency probe (B4): predict behavior from the radio net alone.

The founding promise of this project is that the radio net *explains* the
cohort's behavior: every command decision is on the transcript in NATO voice
procedure, so a reader following the traffic should be able to say what each
agent is about to do. This module makes that promise measurable.

Given ONLY the transcript-so-far plus **static briefing material** — the map's
objective coordinates, the spawn area, and the org chart, i.e. what any
observer holds before the operation starts — a deterministic rule engine
(:class:`NetPredictor`) predicts, for every living agent at every step:

* **destination** — which anchor the agent is heading for or holding over
  the next ``K`` steps: one class per named objective, ``LEADER`` (holding
  formation on / rallying to its direct leader), or ``HOLD`` (staying where
  it is). Ground truth: a stationary agent belongs to the region it
  occupies; a moving one to the anchor it closes on most (see
  :func:`destination_truth`);
* **posture** — ``STATIC`` / ``MOVING`` / ``FIRING`` over the same window.

The predictor holds **no private state**: its whole world model is the
standing order per station derived from ORDER/OPORD traffic (doctrine:
``core/missions.py``), grid references parsed from SITREP/TRAP messages,
CONTACT reports as the only enemy picture, DONE + confirmation clearing
missions, SUPPORT ENDED notices, and CASUALTY/succession broadcasts replayed
through the same devolution rules the roster uses. Everything it consumes is
literal transcript text (parsed with the shipped ``core/language.py``
parser) — if a fact is not on the net, the predictor does not know it.

Ground truth comes from a recorded episode trace (``cohort.metrics.
TraceRecorder``); accuracy is scored per *(step x living agent)* pair and
reported next to two honest baselines (majority class and uniform random).
The number that matters is the **gap** between the net-derived prediction and
those baselines. Method, results, and failure modes: ``docs/transparency.md``.

Usage:
    python -m cohort.probe runs/<run>/ckpt_best.pt --episodes 30 --seed 500
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from cohort.config import build_org, get_scenario
from cohort.core import language as lang
from cohort.core.missions import IN_POSITION_RADIUS, WEAPONS_TIGHT, MissionType

#: Prediction horizon: behavior is scored over the next K steps.
K = 15

#: Destination classes (besides one ``OBJ <NAME>`` class per objective).
LEADER = "LEADER"
HOLD = "HOLD"

#: Posture classes.
STATIC = "STATIC"
MOVING = "MOVING"
FIRING = "FIRING"
POSTURES: tuple[str, ...] = (STATIC, MOVING, FIRING)

# --- ground-truth region radii (grid cells) -------------------------------
#: A stationary agent whose mean window distance to an objective is within
#: this radius occupies that objective's region. Covers every objective-
#: anchored in-position radius (OBSERVE's ring, 9.0, is the widest).
OBJ_REGION = 9.0
#: Mean window distance to the direct leader for the LEADER class.
LEADER_REGION = 6.0
#: An agent that never leaves this radius of its position at prediction time
#: is stationary ("it stayed put").
HOLD_REGION = 3.0
#: Minimum net closure (cells) on an anchor over the window for a moving
#: agent to count as heading there; less is drift/dither noise.
CLOSURE_MIN = 2.0
#: Window fraction of moved steps at or above which the posture is MOVING.
MOVE_FRAC = 1 / 3

# --- predictor tunables (net-derived combat picture) ----------------------
#: A CONTACT report older than this no longer predicts an engagement.
CONTACT_FRESH = 10
#: Predicted-engagement radius around an agent's estimated position: weapon
#: range (8) plus a closing margin.
ENGAGE_RADIUS = 12.0


def obj_class(name: str) -> str:
    """Destination class label of a named objective."""
    return f"OBJ {name}"


def _euclid(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _manhattan(a, b) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------- #
# static briefing material
# ---------------------------------------------------------------------- #


@dataclass
class Briefing:
    """Public pre-mission knowledge: map, objectives, org chart. No positions,
    no oracle — exactly what an outside reader holds before step 0."""

    scenario: str
    objectives: dict[str, tuple[int, int]]  # name -> map position
    spawn: tuple[int, int]                  # friendly spawn anchor
    org: dict[str, str | None]              # callsign -> direct leader (None -> HQ)

    @property
    def dest_classes(self) -> list[str]:
        """All destination classes of this scenario, in stable order."""
        return [obj_class(n) for n in self.objectives] + [LEADER, HOLD]


def make_briefing(scenario: str) -> Briefing:
    """Build the briefing from a scenario preset (callsigns as the env names them)."""
    spec = get_scenario(scenario)
    slots = build_org(spec.org)
    counters: dict = {}
    callsigns: list[str] = []
    for slot in slots:
        counters[slot.rank] = counters.get(slot.rank, 0) + 1
        callsigns.append(f"{slot.rank.name}{counters[slot.rank]}")
    org = {
        cs: (callsigns[slot.leader] if slot.leader is not None else None)
        for cs, slot in zip(callsigns, slots, strict=True)
    }
    return Briefing(
        scenario=spec.name,
        objectives={name: pos for name, pos in spec.objectives},
        spawn=spec.spawn,
        org=org,
    )


# ---------------------------------------------------------------------- #
# the rule engine: net traffic -> per-station standing state
# ---------------------------------------------------------------------- #


@dataclass
class _Task:
    """A standing order as derivable from the net."""

    mission: MissionType
    objective: str | None            # objective name (objective-anchored missions)
    support: str | None              # supported callsign (SUPPORT)
    step: int                        # when it landed (or was inherited)
    origin: tuple[float, float]      # recipient's estimated position at that moment
    team: bool = False               # root OPORD RECON/SCREEN: team-adjudicated (#9)


_GRID_RE = re.compile(r"GRID (\d{2})(\d{2})")
_DOWN_RE = re.compile(r"ALL STATIONS: ([A-Za-z]{2,3}\d+) IS DOWN")
_TAKING_RE = re.compile(r"THIS IS ([A-Za-z]{2,3}\d+): ([A-Za-z]{2,3}\d+) IS DOWN\. I AM ASSUMING COMMAND")
_FILLING_RE = re.compile(r"THIS IS ([A-Za-z]{2,3}\d+): ASSUMING ([A-Za-z]{2,3}\d+)'S POSITION")
_TRAP_RE = re.compile(r"ALL STATIONS: ([A-Za-z]{2,3}\d+) HIT A DEVICE AT GRID (\d{2})(\d{2})")


class NetPredictor:
    """Deterministic rule engine over the transcript-so-far.

    Feed it each step's messages via :meth:`observe`, then ask
    :meth:`predict` for any callsign. All state is derived from the
    briefing plus message text; nothing else ever enters.
    """

    def __init__(self, brief: Briefing, k: int = K) -> None:
        self.brief = brief
        self.k = k
        self.t = 0
        self.alive: dict[str, bool] = {cs: True for cs in brief.org}
        self.leader: dict[str, str | None] = dict(brief.org)
        self.subs: dict[str, list[str]] = {
            cs: [c for c, ldr in brief.org.items() if ldr == cs] for cs in brief.org
        }
        self.task: dict[str, _Task | None] = {cs: None for cs in brief.org}
        self.sitrep: dict[str, tuple[int, tuple[int, int]]] = {}  # cs -> (step, grid)
        self.contacts: list[tuple[int, tuple[int, int]]] = []     # (step, grid) sightings
        #: org slots vacated by a promotion this succession, for the recursive
        #: "ASSUMING X'S POSITION" fills: mover -> (its old leader/subs/task)
        self._vacated: dict[str, tuple[str | None, list[str], _Task | None]] = {}

    # -- transcript consumption ----------------------------------------- #

    def observe(self, t: int, messages: list[dict]) -> None:
        """Consume one step's radio traffic (message dicts with kind/from/to/text)."""
        self.t = t
        for m in messages:
            self._consume(m)
        self.contacts = [(ts, g) for ts, g in self.contacts if t - ts <= CONTACT_FRESH]

    def _consume(self, m: dict) -> None:
        kind, text = m["kind"], m["text"]
        if kind in ("order", "opord"):
            try:
                parsed = lang.parse_order(text)
            except lang.OrderParseError:  # pragma: no cover - formatter output always parses
                return
            cs = parsed.recipient_callsign
            if not self.alive.get(cs):
                return
            team = (
                kind == "opord"
                and parsed.mission in (MissionType.RECON, MissionType.SCREEN)
                and self.leader.get(cs) is None
            )
            self.task[cs] = _Task(
                mission=parsed.mission,
                objective=parsed.objective_name,
                support=parsed.target_callsign,
                step=self.t,
                origin=self._est_pos(cs),
                team=team,
            )
        elif kind == "sitrep":
            grid = _GRID_RE.search(text)
            if grid:
                self.sitrep[m["from"]] = (self.t, (int(grid.group(1)), int(grid.group(2))))
        elif kind == "contact":
            grid = _GRID_RE.search(text)
            if grid:
                self.contacts.append((self.t, (int(grid.group(1)), int(grid.group(2)))))
        elif kind == "done_confirm":
            # verified MISSION COMPLETE: the mission is cleared on the net —
            # the claimant stands by for new orders (a DONE alone proves
            # nothing; DONE_REJECT means the mission stands)
            self.task[m["to"]] = None
        elif kind == "support_end":
            self.task[m["from"]] = None  # supported unit fell: standing by
        elif kind == "casualty":
            down = _DOWN_RE.search(text)
            if down:
                # death alone does not clear the standing order: succession
                # transfers it (the TAKING_COMMAND broadcast that follows)
                self.alive[down.group(1)] = False
        elif kind == "trap":
            hit = _TRAP_RE.search(text)
            if hit:  # the broadcast fixes the victim's position exactly
                self.sitrep[hit.group(1)] = (self.t, (int(hit.group(2)), int(hit.group(3))))
        elif kind == "taking_command":
            taking = _TAKING_RE.search(text)
            if taking:
                successor, dead = taking.group(1), taking.group(2)
                self.alive[dead] = False
                self._assume(
                    successor,
                    leader=self.leader.get(dead),
                    slot_subs=self.subs.get(dead, []),
                    slot_task=self.task.get(dead),
                )
                self.task[dead] = None
                self.subs[dead] = []
                return
            filling = _FILLING_RE.search(text)
            if filling:
                filler, moved = filling.group(1), filling.group(2)
                _, slot_subs, slot_task = self._vacated.pop(moved, (None, [], None))
                # the vacated slot reports to the agent who just moved up
                self._assume(filler, leader=moved, slot_subs=slot_subs, slot_task=slot_task)
                self.subs.setdefault(moved, []).append(filler)

    def _assume(
        self, cs: str, leader: str | None, slot_subs: list[str], slot_task: _Task | None
    ) -> None:
        """Replay one succession move: ``cs`` assumes a vacated org slot.

        Mirrors ``Roster.succeed``: the successor takes the slot's leader,
        its living subordinates, and — only if the slot held one — its
        standing mission (continuity), re-anchoring the transit estimate at
        the successor's own estimated position. The successor's old slot is
        remembered for the recursive "ASSUMING X'S POSITION" fill.
        """
        self._vacated[cs] = (self.leader.get(cs), list(self.subs.get(cs, [])), self.task.get(cs))
        self.leader[cs] = leader
        new_subs = [c for c in slot_subs if c != cs and self.alive.get(c)]
        self.subs[cs] = new_subs
        for c in new_subs:
            self.leader[c] = cs
        if slot_task is not None:
            self.task[cs] = replace(slot_task, step=self.t, origin=self._est_pos(cs))

    # -- net-derived position estimates ---------------------------------- #

    def _est_pos(self, cs: str, seen: frozenset = frozenset()) -> tuple[float, float]:
        """Best position estimate from the net: assumed progress toward the
        mission anchor from the latest evidence point (order receipt, or a
        later reported grid) at one cell per step, else the last reported
        grid, else the spawn area."""
        task = self.task.get(cs)
        sit = self.sitrep.get(cs)
        if task is not None and cs not in seen:
            anchor = self._anchor_est(cs, task, seen | {cs})
            if anchor is not None:
                step, pos = task.step, task.origin
                if sit is not None and sit[0] >= step:
                    step, pos = sit
                total = _manhattan(pos, anchor)
                if total <= 1e-9:
                    return anchor
                frac = max(0.0, min(1.0, (self.t - step) / total))
                return (
                    pos[0] + frac * (anchor[0] - pos[0]),
                    pos[1] + frac * (anchor[1] - pos[1]),
                )
        if sit is not None:
            return sit[1]
        return self.brief.spawn

    def _anchor_est(self, cs: str, task: _Task, seen: frozenset) -> tuple[float, float] | None:
        """Estimated anchor point of a standing order (None if unknowable)."""
        if task.objective is not None:
            return self.brief.objectives.get(task.objective)
        if task.mission is MissionType.RALLY:
            ldr = self.leader.get(cs)
            if ldr is None or not self.alive.get(ldr) or ldr in seen:
                return None
            return self._est_pos(ldr, seen | {ldr})
        if task.mission is MissionType.SUPPORT:
            sup = task.support
            if sup is None or not self.alive.get(sup) or sup in seen:
                return None
            return self._est_pos(sup, seen | {sup})
        return task.origin  # HOLD: where the order was received

    def _travel_remaining(self, cs: str, task: _Task, anchor: tuple[float, float]) -> int:
        """Estimated steps still needed to reach the mission station.

        From the latest evidence point (order receipt, or a later SITREP /
        TRAP grid), assume 1 cell per step of 4-neighbour movement toward
        the anchor until inside the mission's in-position radius.
        """
        step, pos = task.step, task.origin
        sit = self.sitrep.get(cs)
        if sit is not None and sit[0] >= step:
            step, pos = sit
        travel = max(0, math.ceil(_manhattan(pos, anchor) - IN_POSITION_RADIUS[task.mission]))
        return max(0, travel - (self.t - step))

    def _arrived(self, cs: str, task: _Task, anchor: tuple[float, float]) -> bool:
        """Transit estimate: has the agent had time to reach its station?"""
        return self._travel_remaining(cs, task, anchor) == 0

    # -- prediction ------------------------------------------------------ #

    def predict(self, cs: str) -> tuple[str, str]:
        """(destination class, posture class) for the next K steps."""
        return self._dest(cs, frozenset({cs})), self._posture(cs, frozenset({cs}))

    def _dest(self, cs: str, seen: frozenset) -> str:
        task = self.task.get(cs)
        if task is None:
            return HOLD  # untasked / completed: standing by where it is
        if task.team:
            # root OPORD RECON/SCREEN is team-adjudicated (#9): doctrine says
            # the commander observes through the squad and commands from cover
            return HOLD
        if task.mission is MissionType.SUPPORT:
            sup = task.support
            if sup is None or not self.alive.get(sup) or sup in seen:
                return HOLD
            # "pas un pas sans appui": the supporter moves with the supported
            # unit, so it inherits that unit's destination
            return self._dest(sup, seen | {sup})
        if task.mission is MissionType.RALLY:
            ldr = self.leader.get(cs)
            return LEADER if ldr is not None and self.alive.get(ldr) else HOLD
        if task.objective is not None:
            return obj_class(task.objective)
        return HOLD  # HOLD orders anchor where they were received

    def _posture(self, cs: str, seen: frozenset) -> str:
        task = self.task.get(cs)
        if task is None:
            return STATIC  # standing by
        if task.team:
            return STATIC  # commands from cover (#9)
        if task.mission is MissionType.SUPPORT:
            sup = task.support
            if sup is not None and self.alive.get(sup) and sup not in seen:
                # the supporter moves, halts, and engages with its supported
                # unit ("pas un pas sans appui"): mirror that unit's posture
                return self._posture(sup, seen | {sup})
            return STATIC
        anchor = self._anchor_est(cs, task, frozenset({cs}))
        remaining = self._travel_remaining(cs, task, anchor) if anchor is not None else 0
        if task.mission in WEAPONS_TIGHT:
            # SCREEN: weapons tight, never predicted firing
            return MOVING if remaining > 0 else STATIC
        if remaining == 0:
            return FIRING if self._contact_near(self._est_pos(cs)) else STATIC
        # in transit: an engagement is predicted when the route passes a hot
        # zone now, or the station itself is hot and reached inside the window
        if self._contact_near(self._est_pos(cs)):
            return FIRING
        if remaining <= self.k and anchor is not None and self._contact_near(anchor):
            return FIRING
        return MOVING

    def _contact_near(self, pos: tuple[float, float]) -> bool:
        """A fresh CONTACT grid within engagement radius of the estimate?"""
        return any(
            self.t - ts <= CONTACT_FRESH and _euclid(pos, grid) <= ENGAGE_RADIUS
            for ts, grid in self.contacts
        )


# ---------------------------------------------------------------------- #
# ground truth (pure functions over a recorded trace)
# ---------------------------------------------------------------------- #


def step_index(steps: list[dict]) -> list[dict[str, dict]]:
    """Per-step callsign -> soldier-record lookup (built once per episode)."""
    return [{rec["cs"]: rec for rec in s["soldiers"]} for s in steps]


def _truth_window(index: list[dict], i: int, cs: str, k: int) -> list[dict]:
    """The agent's records over steps i+1..i+k, truncated at death/episode end."""
    window: list[dict] = []
    for by_cs in index[i + 1 : i + 1 + k]:
        rec = by_cs.get(cs)
        if rec is None or not rec["alive"]:
            break
        window.append(rec)
    return window


def destination_truth(
    index: list[dict], i: int, cs: str, k: int, objectives: dict[str, tuple[int, int]]
) -> str | None:
    """Ground-truth destination class of agent ``cs`` at step index ``i``.

    Destination means *where the agent is going / what it holds*:

    * **Stationary** (never leaves ``HOLD_REGION`` of its position at
      prediction time): the region it occupies — the nearest objective by
      mean window distance if within ``OBJ_REGION``, else LEADER if it sits
      within ``LEADER_REGION`` of its leader, else HOLD.
    * **Moving**: the anchor it *closes on* most over the window (start
      minus end distance; anchors: every objective, the leader's concurrent
      position), if that closure reaches ``CLOSURE_MIN``. An agent moving
      *with* its leader toward an objective closes on the objective, not
      the leader — formation-keeping is not a destination.
    * **Moving but approaching nothing** (dither, retreat, wander): the
      region its endpoint occupies, HOLD if it ended near where it started,
      else the nearest anchor at the endpoint.

    None -> no window (dead next step or episode over): the pair is skipped.
    """
    window = _truth_window(index, i, cs, k)
    if not window:
        return None
    me = index[i].get(cs)
    p0 = me["pos"]
    leader_cs = me.get("leader")
    positions = [rec["pos"] for rec in window]
    n = len(window)
    p_end = positions[-1]

    # the leader's concurrent path over the window (None if it drops out)
    leader_path: list | None = None
    if leader_cs is not None and index[i].get(leader_cs) is not None:
        path = []
        for j in range(n):
            lrec = index[i + 1 + j].get(leader_cs)
            if lrec is None:
                path = None
                break
            path.append(lrec["pos"])
        leader_path = path

    if max(_euclid(p, p0) for p in positions) <= HOLD_REGION:  # stationary
        d_obj = {
            name: sum(_euclid(p, pos) for p in positions) / n
            for name, pos in objectives.items()
        }
        best = min(d_obj, key=lambda name: (d_obj[name], name)) if d_obj else None
        if best is not None and d_obj[best] <= OBJ_REGION:
            return obj_class(best)
        if leader_path is not None:
            d_led = sum(
                _euclid(p, q) for p, q in zip(positions, leader_path, strict=True)
            ) / n
            if d_led <= LEADER_REGION:
                return LEADER
        return HOLD

    # moving: the anchor it closes on most
    closures: list[tuple[float, str]] = [
        (_euclid(p0, pos) - _euclid(p_end, pos), obj_class(name))
        for name, pos in objectives.items()
    ]
    if leader_path is not None:
        l0 = index[i][leader_cs]["pos"]
        closures.append((_euclid(p0, l0) - _euclid(p_end, leader_path[-1]), LEADER))
    if closures:
        neg, label = min((-c, lbl) for c, lbl in closures)
        if -neg >= CLOSURE_MIN:
            return label

    # moved but approached nothing: where did it end up?
    d_end = {name: _euclid(p_end, pos) for name, pos in objectives.items()}
    best = min(d_end, key=lambda name: (d_end[name], name)) if d_end else None
    if best is not None and d_end[best] <= OBJ_REGION:
        return obj_class(best)
    if leader_path is not None and _euclid(p_end, leader_path[-1]) <= LEADER_REGION:
        return LEADER
    if _euclid(p_end, p0) <= HOLD_REGION:
        return HOLD
    candidates: list[tuple[float, str]] = [(d, obj_class(nm)) for nm, d in d_end.items()]
    if leader_path is not None:
        candidates.append((_euclid(p_end, leader_path[-1]), LEADER))
    if not candidates:
        return HOLD
    return min(candidates)[1]


def posture_truth(index: list[dict], i: int, cs: str, k: int) -> str | None:
    """Ground-truth posture over the window: FIRING if the agent fired on any
    window step, else MOVING if it changed cell on >= ``MOVE_FRAC`` of the
    window steps, else STATIC. None -> no window (pair skipped)."""
    window = _truth_window(index, i, cs, k)
    if not window:
        return None
    if any(rec.get("fired") for rec in window):
        return FIRING
    prev = index[i][cs]["pos"]
    moved = 0
    for rec in window:
        if rec["pos"] != prev:
            moved += 1
        prev = rec["pos"]
    return MOVING if moved / len(window) >= MOVE_FRAC else STATIC


# ---------------------------------------------------------------------- #
# scoring
# ---------------------------------------------------------------------- #


def probe_episode(trace: dict, brief: Briefing, k: int = K) -> dict[str, Any]:
    """Run the predictor along one recorded episode and score every pair.

    Returns per-episode confusion counts: ``{"pairs", "destination",
    "posture"}`` with confusions as nested dicts ``truth -> pred -> count``.
    """
    steps = trace["steps"]
    index = step_index(steps)
    predictor = NetPredictor(brief, k=k)
    dest_conf: dict[str, dict[str, int]] = {}
    post_conf: dict[str, dict[str, int]] = {}
    per_cs: dict[str, list[int]] = {}  # cs -> [dest_correct, post_correct, pairs]
    pairs = 0
    for i, s in enumerate(steps):
        predictor.observe(s["t"], s["messages"])
        for rec in s["soldiers"]:
            if not rec["alive"]:
                continue
            dest_t = destination_truth(index, i, rec["cs"], k, brief.objectives)
            if dest_t is None:
                continue  # no future window: nothing to predict
            post_t = posture_truth(index, i, rec["cs"], k)
            dest_p, post_p = predictor.predict(rec["cs"])
            pairs += 1
            dest_row = dest_conf.setdefault(dest_t, {})
            dest_row[dest_p] = dest_row.get(dest_p, 0) + 1
            post_row = post_conf.setdefault(post_t, {})
            post_row[post_p] = post_row.get(post_p, 0) + 1
            tally = per_cs.setdefault(rec["cs"], [0, 0, 0])
            tally[0] += dest_t == dest_p
            tally[1] += post_t == post_p
            tally[2] += 1
    return {
        "pairs": pairs,
        "destination": dest_conf,
        "posture": post_conf,
        "per_callsign": per_cs,
    }


def _merge(confusions: list[dict]) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = {}
    for conf in confusions:
        for truth, row in conf.items():
            dst = merged.setdefault(truth, {})
            for pred, count in row.items():
                dst[pred] = dst.get(pred, 0) + count
    return merged


def _summarize(conf: dict[str, dict[str, int]], classes: list[str]) -> dict[str, Any]:
    """Accuracy, per-class accuracy, honest baselines, and their gaps."""
    support = {c: sum(conf.get(c, {}).values()) for c in classes}
    total = sum(support.values())
    correct = sum(conf.get(c, {}).get(c, 0) for c in classes)
    accuracy = correct / total if total else None
    majority = max(support.values()) / total if total else None
    random = 1.0 / len(classes)
    per_class = {
        c: (conf.get(c, {}).get(c, 0) / support[c]) if support[c] else None for c in classes
    }
    return {
        "pairs": total,
        "accuracy": accuracy,
        "baseline_majority": majority,
        "baseline_random": random,
        "gap_vs_majority": (accuracy - majority) if total else None,
        "gap_vs_random": (accuracy - random) if total else None,
        "per_class_accuracy": per_class,
        "support": support,
        "confusion": {c: dict(conf.get(c, {})) for c in classes if conf.get(c)},
    }


def aggregate_probe(episodes: list[dict], dest_classes: list[str]) -> dict[str, Any]:
    """Pool per-episode confusions into the run-level probe summary."""
    per_cs: dict[str, list[int]] = {}
    for ep in episodes:
        for cs, (dest_ok, post_ok, n) in ep.get("per_callsign", {}).items():
            tally = per_cs.setdefault(cs, [0, 0, 0])
            tally[0] += dest_ok
            tally[1] += post_ok
            tally[2] += n
    return {
        "episodes": len(episodes),
        "pairs": sum(ep["pairs"] for ep in episodes),
        "destination": _summarize(_merge([ep["destination"] for ep in episodes]), dest_classes),
        "posture": _summarize(_merge([ep["posture"] for ep in episodes]), list(POSTURES)),
        "per_callsign": {
            cs: {
                "destination_accuracy": dest_ok / n,
                "posture_accuracy": post_ok / n,
                "pairs": n,
            }
            for cs, (dest_ok, post_ok, n) in per_cs.items()
            if n
        },
    }


def format_probe_table(agg: dict[str, Any]) -> str:
    """Human-readable summary of an aggregated probe result."""
    lines = [
        f"transparency probe over {agg['episodes']} episodes: "
        f"{agg['pairs']} (step x living agent) pairs"
    ]
    for name in ("destination", "posture"):
        s = agg[name]
        if not s["pairs"]:
            lines.append(f"  {name}: no pairs")
            continue
        lines.append(
            f"  {name}: accuracy {s['accuracy']:.3f}   "
            f"(majority {s['baseline_majority']:.3f} -> gap {s['gap_vs_majority']:+.3f}; "
            f"random {s['baseline_random']:.3f} -> gap {s['gap_vs_random']:+.3f})"
        )
        for cls, acc in s["per_class_accuracy"].items():
            if s["support"][cls] == 0:
                continue
            share = s["support"][cls] / s["pairs"]
            acc_txt = f"{acc:.3f}" if acc is not None else "—"
            lines.append(f"    {cls:<12} acc {acc_txt}   (share {share:.2f}, n={s['support'][cls]})")
    per_cs = agg.get("per_callsign", {})
    if per_cs:
        lines.append("  per callsign (dest / posture):")
        for cs, row in per_cs.items():
            lines.append(
                f"    {cs:<6} {row['destination_accuracy']:.3f} / "
                f"{row['posture_accuracy']:.3f}   (n={row['pairs']})"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #


def probe(
    checkpoint: str,
    scenario: str | None = None,
    episodes: int = 30,
    seed: int = 500,
    k: int = K,
    *,
    greedy: bool = False,
    out_path: str | None = None,
) -> dict[str, Any]:
    """Record ``episodes`` evaluation episodes and probe them.

    Uses the exact B2 evaluation protocol (per-episode self-contained
    seeding; defaults = the assurance seeds 500-529), so probed episodes are
    the same ones ``behavior.json`` measures.
    """
    from cohort.env.cohort_env import make_env
    from cohort.metrics import TraceRecorder
    from cohort.training.evaluate import _seeded_episode
    from cohort.training.train import load_policy

    net, ckpt = load_policy(checkpoint)
    scenario = scenario or ckpt["scenario"]
    brief = make_briefing(scenario)
    env = make_env(scenario)
    results = []
    for i in range(episodes):
        recorder = TraceRecorder()
        _seeded_episode(env, net, seed + i, greedy=greedy, recorder=recorder)
        results.append(probe_episode(recorder.trace, brief, k=k))
    agg = aggregate_probe(results, brief.dest_classes)
    print(f"probe [{checkpoint}] on {scenario} (K={k}, seed {seed}):")
    print(format_probe_table(agg))
    out = out_path
    if out is None:
        out = str(Path(checkpoint).parent / "probe.json")
    payload = {
        "checkpoint": checkpoint,
        "scenario": scenario,
        "episodes": episodes,
        "seed": seed,
        "k": k,
        "greedy": greedy,
        **agg,
    }
    Path(out).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"probe → {out}")
    return agg


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transparency probe (B4): predict behavior from the radio net alone."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--scenario", default=None, help="override the checkpoint's scenario")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=500, help="assurance protocol seeds: 500-529")
    parser.add_argument("--k", type=int, default=K, help="prediction horizon (steps)")
    parser.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    parser.add_argument("--out", default=None, help="output JSON path (default: probe.json next to the checkpoint)")
    args = parser.parse_args()
    probe(
        args.checkpoint,
        scenario=args.scenario,
        episodes=args.episodes,
        seed=args.seed,
        k=args.k,
        greedy=args.greedy,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
