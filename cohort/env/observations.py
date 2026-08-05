"""Per-agent observation vectors.

Each agent sees: its own state (with *effective* rank, so a promoted acting
leader knows it now commands), its standing mission, its direct leader, its
direct subordinates, currently visible enemies, objectives, a comms summary,
and a local terrain patch. Enemy knowledge is deliberately split:

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

from cohort.core.missions import MissionType
from cohort.core.ranks import Rank

if TYPE_CHECKING:
    from cohort.core.units import Enemy, Roster, Soldier
    from cohort.core.world import World

RANK_ORDER: tuple[Rank, ...] = (Rank.RFN, Rank.TL, Rank.SL, Rank.PSG, Rank.PL, Rank.XO, Rank.CO)
MISSION_ORDER: tuple[MissionType, ...] = tuple(MissionType)

N_SUB_SLOTS = 4
N_ENEMY_SLOTS = 4
N_OBJECTIVE_SLOTS = 4
PATCH_RADIUS = 2

#: 12 self + 12 mission + 4 leader + 5*N_SUB + 4*N_ENEMY + 3*N_OBJ + 5 comms + patch
OBS_DIM = (
    12 + 12 + 4
    + 5 * N_SUB_SLOTS
    + 4 * N_ENEMY_SLOTS
    + 3 * N_OBJECTIVE_SLOTS
    + 5
    + (2 * PATCH_RADIUS + 1) ** 2 * 2
)


@dataclass
class AgentView:
    """Per-step, per-agent context the environment hands the obs builder."""

    visible_enemies: list[Enemy] = field(default_factory=list)  # sorted nearest-first
    known_enemies: list[tuple[float, float]] = field(default_factory=list)  # team picture
    step: int = 0


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

    # --- self (12) ---
    out[i : i + 4] = (x / w, y / h, soldier.health / 100.0, soldier.ammo / 30.0)
    i += 4
    out[i + RANK_ORDER.index(soldier.effective_rank)] = 1.0
    i += len(RANK_ORDER)
    out[i] = 1.0 if world.cover_at(soldier.pos) else 0.0
    i += 1

    # --- mission (12) ---
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
        out[i] = (anchor[0] - x) / w
        out[i + 1] = (anchor[1] - y) / h
        out[i + 2] = 1.0 if m.objective_id is not None else 0.0
        out[i + 3] = min(1.0, (view.step - m.step_assigned) / 50.0)
    i += 4

    # --- leader (4) ---
    leader = roster.leader_of(soldier)
    if leader is not None:
        out[i] = 1.0
        out[i + 1] = (leader.pos[0] - x) / w
        out[i + 2] = (leader.pos[1] - y) / h
        out[i + 3] = _mission_idx(leader.mission.type if leader.mission else None)
    i += 4

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

    # --- comms summary (5) ---
    out[i] = 1.0 if view.step - soldier.last_order_step <= 1 else 0.0
    out[i + 1] = min(1.0, len(view.known_enemies) / 4.0)
    if view.known_enemies:
        nearest = min(view.known_enemies, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
        out[i + 2] = 1.0
        out[i + 3] = (nearest[0] - x) / w
        out[i + 4] = (nearest[1] - y) / h
    i += 5

    # --- terrain patch ---
    patch = world.local_patch(soldier.pos, PATCH_RADIUS).reshape(-1)
    out[i : i + patch.shape[0]] = patch
    i += patch.shape[0]

    assert i == OBS_DIM, f"obs layout mismatch: wrote {i}, expected {OBS_DIM}"
    return np.clip(out, -1.0, 1.0)
