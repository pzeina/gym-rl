"""The global action catalog and per-agent legality masks.

Every agent shares one flat ``Discrete`` action catalog; what differs by rank
is the *mask*. Rank admissibility is a hard guarantee, not a learned habit:

* RFN (and any agent with no command authority) can never select an order
  action — those entries are masked off, permanently.
* Leaders can only order their own living direct subordinates (slots), and
  only missions that doctrine allows them to derive from their *own* current
  mission. A trained leader is therefore doctrine-conformant by construction.
* FIRE requires a visible enemy in weapon range and ammunition.
* CONTACT requires a currently visible enemy (you cannot report what you do
  not see). MISSION COMPLETE requires holding a completable mission —
  whether it is *true* is judged by reward, so honest reporting is learned.

Every entry has a stable human-readable name, so logged actions read like a
soldier's decision log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cohort.core.language import OBJECTIVE_NAMES
from cohort.core.missions import COMPLETABLE, NEEDS_OBJECTIVE, MissionType, allowed_derivations

if TYPE_CHECKING:
    from cohort.core.units import Roster, Soldier
    from cohort.core.world import World

#: Maximum direct-subordinate slots an agent's order vocabulary addresses.
MAX_SUB_SLOTS = 4

#: (dx, dy) for the four movement actions, grid y grows south.
MOVES: dict[str, tuple[int, int]] = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
}


@dataclass(frozen=True)
class ActionSpec:
    """One catalog entry."""

    index: int
    kind: str                       # stay | move | fire | contact | sitrep | done | order
    name: str
    move: tuple[int, int] | None = None
    order_slot: int | None = None
    order_mission: MissionType | None = None
    order_objective: str | None = None  # objective name, or None for RALLY/HOLD


def _build_catalog() -> list[ActionSpec]:
    specs: list[ActionSpec] = []

    def add(kind: str, name: str, **kw: object) -> None:
        specs.append(ActionSpec(index=len(specs), kind=kind, name=name, **kw))

    add("stay", "STAY")
    for direction, delta in MOVES.items():
        add("move", f"MOVE_{direction}", move=delta)
    add("fire", "FIRE")
    add("contact", "REPORT_CONTACT")
    add("sitrep", "REPORT_SITREP")
    add("done", "REPORT_MISSION_COMPLETE")
    for slot in range(MAX_SUB_SLOTS):
        for mission in MissionType:
            if mission in NEEDS_OBJECTIVE:
                for obj in OBJECTIVE_NAMES:
                    add(
                        "order",
                        f"ORDER_S{slot}_{mission.name}_OBJ_{obj}",
                        order_slot=slot,
                        order_mission=mission,
                        order_objective=obj,
                    )
            else:
                add(
                    "order",
                    f"ORDER_S{slot}_{mission.name}",
                    order_slot=slot,
                    order_mission=mission,
                    order_objective=None,
                )
    return specs


CATALOG: list[ActionSpec] = _build_catalog()
N_ACTIONS: int = len(CATALOG)

#: index lookups used by the mask builder
_STAY = next(s.index for s in CATALOG if s.kind == "stay")
_ORDER_SPECS = [s for s in CATALOG if s.kind == "order"]


def action_name(index: int) -> str:
    """Human-readable name of an action index."""
    return CATALOG[index].name


def compute_mask(
    soldier: Soldier,
    roster: Roster,
    world: World,
    visible_enemy_in_range: bool,
    visible_enemy: bool,
) -> np.ndarray:
    """Legality mask (int8, shape (N_ACTIONS,)) for one agent this step."""
    mask = np.zeros(N_ACTIONS, dtype=np.int8)
    mask[_STAY] = 1
    if not soldier.alive:
        return mask

    for spec in CATALOG:
        if spec.kind == "move":
            nxt = (soldier.pos[0] + spec.move[0], soldier.pos[1] + spec.move[1])
            if world.passable(nxt):
                mask[spec.index] = 1
        elif spec.kind == "fire":
            if soldier.ammo > 0 and visible_enemy_in_range:
                mask[spec.index] = 1
        elif spec.kind == "contact":
            if visible_enemy:
                mask[spec.index] = 1
        elif spec.kind == "sitrep":
            mask[spec.index] = 1
        elif spec.kind == "done":
            if soldier.mission is not None and soldier.mission.type in COMPLETABLE:
                mask[spec.index] = 1

    # Order vocabulary: command ranks only, doctrine-constrained.
    if soldier.effective_authority > 0 and soldier.mission is not None:
        allowed = allowed_derivations(soldier.mission.type)
        subs = soldier.living_subordinates(roster)
        objective_names = {o.name for o in world.objectives}
        for spec in _ORDER_SPECS:
            if spec.order_slot >= len(subs):
                continue
            if spec.order_mission not in allowed:
                continue
            if spec.order_objective is not None and spec.order_objective not in objective_names:
                continue
            mask[spec.index] = 1
    return mask
