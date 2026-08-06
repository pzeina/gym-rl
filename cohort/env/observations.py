"""Per-agent observation vectors.

Each agent sees: its own state (with *effective* rank, so a promoted acting
leader knows it now commands, and an is-human flag), its standing mission,
its direct leader (incl. whether the leader is human), its direct
subordinates, currently visible enemies, objectives, a comms summary, and a
local terrain patch. Enemy knowledge is deliberately split:

* ``enemy`` slots — what THIS agent can see right now (private).
* ``known enemy`` summary — the team picture, which only contains enemies
  someone has *reported* via CONTACT. Reporting is what turns a private
  sighting into shared knowledge, which is why reporting is worth doing,
  not just worth rewarding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cohort.core.missions import Formation, MissionType
from cohort.core.ranks import Rank

if TYPE_CHECKING:
    from cohort.core.units import Enemy, Roster, Soldier
    from cohort.core.world import World

RANK_ORDER: tuple[Rank, ...] = (Rank.RFN, Rank.TL, Rank.SL, Rank.PSG, Rank.PL, Rank.XO, Rank.CO)
MISSION_ORDER: tuple[MissionType, ...] = tuple(MissionType)
FORMATION_ORDER: tuple[Formation, ...] = tuple(Formation)

N_SUB_SLOTS = 4
N_ENEMY_SLOTS = 4
N_OBJECTIVE_SLOTS = 4
#: control-measure slots (A5): one per catalog name — 4 waypoints (GOLD /
#: SILVER / COPPER / IRON) + 3 phase lines (AMBER / COBALT / CRIMSON); a
#: scenario using fewer leaves the remaining slots zeroed (like objectives)
N_WAYPOINT_SLOTS = 4
N_PHASE_LINE_SLOTS = 3
PATCH_RADIUS = 2

#: mission block: one-hot over the 12 tasks (11 MICAT + ADVANCE) + has-mission
#: flag + 4 anchor fields (dx, dy, has-objective, age) + 2 pending fields
#: (A5-2: pending flag; time-to-effective — 1.0 while AT MY COMMAND, else
#: remaining/20 capped at 1) + 3 stance one-hot (A5-3: the governing element
#: formation — the agent's own stance if it leads one, else its direct
#: leader's; all zero when no stance applies)
_MISSION_BLOCK = len(MISSION_ORDER) + 1 + 4 + 2 + len(FORMATION_ORDER)

#: self block: x, y, health, ammo (4) + rank one-hot (7) + in-cover + is-human
_SELF_BLOCK = 4 + len(RANK_ORDER) + 1 + 1

#: leader block: present, dx, dy, mission index, leader-is-human
_LEADER_BLOCK = 5

#: 13 self + 22 mission/stance + 5 leader + 5*N_SUB + 4*N_ENEMY + 3*N_OBJ
#: + 3*N_WP + 3*N_PL (control measures: present, dx, dy — for a phase line
#: dx/dy point at its nearest segment point) + 5 comms + patch (50)
#: = 13 + 22 + 5 + 20 + 16 + 12 + 12 + 9 + 5 + 50 = 164
OBS_DIM = (
    _SELF_BLOCK + _MISSION_BLOCK + _LEADER_BLOCK
    + 5 * N_SUB_SLOTS
    + 4 * N_ENEMY_SLOTS
    + 3 * N_OBJECTIVE_SLOTS
    + 3 * N_WAYPOINT_SLOTS
    + 3 * N_PHASE_LINE_SLOTS
    + 5
    + (2 * PATCH_RADIUS + 1) ** 2 * 2
)


@dataclass
class AgentView:
    """Per-step, per-agent context the environment hands the obs builder."""

    visible_enemies: list[Enemy] = field(default_factory=list)  # sorted nearest-first
    known_enemies: list[tuple[float, float]] = field(default_factory=list)  # team picture
    step: int = 0
    #: SITREP due-ness in [0, 1] when the reporting doctrine
    #: (``ScenarioSpec.sitrep_cadence``) is active; None → doctrine off. When
    #: set, it replaces the comms-summary "known enemy present" flag — a slot
    #: fully redundant with the known-count field — so OBS_DIM is unchanged.
    sitrep_due: float | None = None


def _mission_idx(mission_type: MissionType | None) -> float:
    if mission_type is None:
        return 0.0
    return (MISSION_ORDER.index(mission_type) + 1) / len(MISSION_ORDER)


def build_observation(
    soldier: Soldier,
    roster: Roster,
    world: World,
    view: AgentView,
) -> np.ndarray:
    """Assemble the flat observation vector for one agent."""
    w, h = float(world.width), float(world.height)
    diag = float(np.hypot(w, h))
    x, y = float(soldier.pos[0]), float(soldier.pos[1])
    out = np.zeros(OBS_DIM, dtype=np.float32)
    i = 0

    # --- self (13) ---
    out[i : i + 4] = (x / w, y / h, soldier.health / 100.0, soldier.ammo / 30.0)
    i += 4
    out[i + RANK_ORDER.index(soldier.effective_rank)] = 1.0
    i += len(RANK_ORDER)
    out[i] = 1.0 if world.cover_at(soldier.pos) else 0.0
    i += 1
    out[i] = 1.0 if soldier.human else 0.0
    i += 1

    # --- mission (19) ---
    m = soldier.mission
    if m is not None:
        out[i + MISSION_ORDER.index(m.type)] = 1.0
    i += len(MISSION_ORDER)
    out[i] = 1.0 if m is not None else 0.0
    i += 1
    if m is not None:
        anchor = m.anchor
        if m.type is MissionType.RALLY:
            leader = roster.leader_of(soldier)
            if leader is not None:
                anchor = leader.pos
        elif m.type is MissionType.SUPPORT:
            supported = roster.by_id.get(m.extra.get("supported_id"))
            if supported is not None and supported.alive:
                anchor = supported.pos
        elif m.type is MissionType.ADVANCE and m.extra.get("control") is not None:
            cm = world.control_by_name(m.extra["control"])
            if cm is not None:
                anchor = cm.nearest_point(soldier.pos) if hasattr(cm, "nearest_point") else cm.pos
        out[i] = (anchor[0] - x) / w
        out[i + 1] = (anchor[1] - y) / h
        out[i + 2] = 1.0 if m.objective_id is not None else 0.0
        out[i + 3] = min(1.0, (view.step - m.step_assigned) / 50.0)
    i += 4
    # pending state (A5-2): staged until "AT T PLUS n" comes due or the
    # issuer's EXECUTE releases an "AT MY COMMAND" order
    if m is not None:
        pending = m.awaiting_signal or (
            m.effective_at is not None and view.step < m.effective_at
        )
        if pending:
            out[i] = 1.0
            out[i + 1] = (
                1.0
                if m.awaiting_signal
                else min(1.0, (m.effective_at - view.step) / 20.0)
            )
    i += 2

    # --- governing element stance (A5-3, 3) ---
    leader = roster.leader_of(soldier)
    stance = soldier.formation
    if stance is None and leader is not None:
        stance = leader.formation  # the member is shaped under its leader's stance
    if stance is not None:
        out[i + FORMATION_ORDER.index(stance)] = 1.0
    i += len(FORMATION_ORDER)

    # --- leader (5) ---
    if leader is not None:
        out[i] = 1.0
        out[i + 1] = (leader.pos[0] - x) / w
        out[i + 2] = (leader.pos[1] - y) / h
        out[i + 3] = _mission_idx(leader.mission.type if leader.mission else None)
        out[i + 4] = 1.0 if leader.human else 0.0
    i += 5

    # --- direct subordinates (5 each) ---
    subs = soldier.living_subordinates(roster)[:N_SUB_SLOTS]
    for k in range(N_SUB_SLOTS):
        if k < len(subs):
            s = subs[k]
            out[i] = 1.0
            out[i + 1] = (s.pos[0] - x) / w
            out[i + 2] = (s.pos[1] - y) / h
            out[i + 3] = _mission_idx(s.mission.type if s.mission else None)
            out[i + 4] = 1.0 if view.step - s.last_contact_report_step <= 10 else 0.0
        i += 5

    # --- visible enemies (4 each) ---
    for k in range(N_ENEMY_SLOTS):
        if k < len(view.visible_enemies):
            e = view.visible_enemies[k]
            d = float(np.hypot(e.pos[0] - x, e.pos[1] - y))
            out[i] = 1.0
            out[i + 1] = (e.pos[0] - x) / w
            out[i + 2] = (e.pos[1] - y) / h
            out[i + 3] = d / diag
        i += 4

    # --- objectives (3 each) ---
    for k in range(N_OBJECTIVE_SLOTS):
        if k < len(world.objectives):
            obj = world.objectives[k]
            out[i] = 1.0
            out[i + 1] = (obj.pos[0] - x) / w
            out[i + 2] = (obj.pos[1] - y) / h
        i += 3

    # --- control measures (A5): waypoints then phase lines (3 each) ---
    for k in range(N_WAYPOINT_SLOTS):
        if k < len(world.waypoints):
            wp = world.waypoints[k]
            out[i] = 1.0
            out[i + 1] = (wp.pos[0] - x) / w
            out[i + 2] = (wp.pos[1] - y) / h
        i += 3
    for k in range(N_PHASE_LINE_SLOTS):
        if k < len(world.phase_lines):
            pl = world.phase_lines[k]
            near = pl.nearest_point(soldier.pos)
            out[i] = 1.0
            out[i + 1] = (near[0] - x) / w
            out[i + 2] = (near[1] - y) / h
        i += 3

    # --- comms summary (5) ---
    out[i] = 1.0 if view.step - soldier.last_order_step <= 1 else 0.0
    out[i + 1] = min(1.0, len(view.known_enemies) / 4.0)
    if view.known_enemies:
        nearest = min(view.known_enemies, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
        out[i + 2] = 1.0
        out[i + 3] = (nearest[0] - x) / w
        out[i + 4] = (nearest[1] - y) / h
    if view.sitrep_due is not None:
        # reporting doctrine active: this slot (otherwise redundant — slot
        # i+1 > 0 says the same thing) carries SITREP due-ness instead
        out[i + 2] = view.sitrep_due
    i += 5

    # --- terrain patch ---
    patch = world.local_patch(soldier.pos, PATCH_RADIUS).reshape(-1)
    out[i : i + patch.shape[0]] = patch
    i += patch.shape[0]

    assert i == OBS_DIM, f"obs layout mismatch: wrote {i}, expected {OBS_DIM}"
    return np.clip(out, -1.0, 1.0)
