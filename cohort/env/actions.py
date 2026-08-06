"""The global action catalog and per-agent legality masks.

Every agent shares one flat ``Discrete`` action catalog; what differs by rank
is the *mask*. Rank admissibility is a hard guarantee, not a learned habit:

* RFN (and any agent with no command authority) can never select an order
  action — those entries are masked off, permanently.
* Leaders can only order their own living direct subordinates (slots), and
  only missions that doctrine allows them to derive from their *own* current
  mission. A trained leader is therefore doctrine-conformant by construction.
* Per-echelon mission admissibility (manual p. 8 tableau récapitulatif):
  a mission with a minimum hold authority (DENY → section, authority >= 2)
  can never be ordered onto a subordinate below that authority.
* SUPPORT is unit-targeted: ``ORDER_S{i}_SUPPORT_U{j}`` tasks the subordinate
  in slot *i* to support the unit led by the subordinate in slot *j*.
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

from cohort.core.language import CONTROL_NAMES, OBJECTIVE_NAMES, control_phrase
from cohort.core.missions import (
    COMPLETABLE,
    NEEDS_CONTROL,
    NEEDS_OBJECTIVE,
    MissionType,
    allowed_derivations,
    is_pending,
    min_hold_authority,
)

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
    order_objective: str | None = None      # objective name, or None
    order_support_slot: int | None = None   # supported unit's slot (SUPPORT only)
    order_control: str | None = None        # control-measure name (ADVANCE only)
    order_amc: bool = False                 # "AT MY COMMAND" variant (A5-2, ADVANCE only)


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
    # A5-2: broadcast EXECUTE, releasing ALL of this issuer's pending
    # AT-MY-COMMAND orders at once (the COMMANDEMENT DU BOND's "EN AVANT !")
    add("execute", "EXECUTE_SIGNAL")
    for slot in range(MAX_SUB_SLOTS):
        for mission in MissionType:
            if mission is MissionType.SUPPORT:
                # unit-targeted: slot i supports the unit led by slot j (i != j)
                for other in range(MAX_SUB_SLOTS):
                    if other == slot:
                        continue
                    add(
                        "order",
                        f"ORDER_S{slot}_SUPPORT_U{other}",
                        order_slot=slot,
                        order_mission=mission,
                        order_support_slot=other,
                    )
            elif mission in NEEDS_CONTROL:
                # ADVANCE targets a control measure: WP GOLD ... PL CRIMSON.
                # Each target also has an AT-MY-COMMAND variant (A5-2): the
                # order stages the recipient until the issuer's EXECUTE —
                # the learnable half of the timing vocabulary ("AT T PLUS n"
                # is unbounded and stays human/inject-only).
                for cm in CONTROL_NAMES:
                    stem = f"ORDER_S{slot}_{mission.name}_{control_phrase(cm).replace(' ', '_')}"
                    add(
                        "order",
                        stem,
                        order_slot=slot,
                        order_mission=mission,
                        order_control=cm,
                    )
                    add(
                        "order",
                        f"{stem}_AMC",
                        order_slot=slot,
                        order_mission=mission,
                        order_control=cm,
                        order_amc=True,
                    )
            elif mission in NEEDS_OBJECTIVE:
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
    *,
    order_cooldown: int = 0,
    step: int = 0,
    net_contact_step: int | None = None,
    ablation: str = "full",
) -> np.ndarray:
    """Legality mask (int8, shape (N_ACTIONS,)) for one agent this step.

    ``order_cooldown`` > 0 masks re-tasking a subordinate within that many
    steps of its last received order, unless the leader's own mission changed
    since, or a CONTACT report hit the net since (``net_contact_step``).
    Untasked subordinates can always be ordered.

    ``ablation`` (ROADMAP B3, ``ScenarioSpec.ablation``) selects the
    hierarchy-ablation arm — masking-only changes, spaces frozen:

    * ``"full"`` (default) — the shipped system described above;
    * ``"nomask"`` — the doctrine-derivation constraint is dropped: a leader
      may issue any rank-admissible order regardless of its own mission
      (even with none). Rank admissibility, per-echelon hold authority and
      the cooldown stay;
    * ``"flat"`` — no ranks in effect: the order vocabulary is masked off
      for everyone; comms reduce to reports.
    """
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
            # a pending order (A5-2) is not yet executing: nothing to report
            if (
                soldier.mission is not None
                and soldier.mission.type in COMPLETABLE
                and not is_pending(soldier.mission, step)
            ):
                mask[spec.index] = 1
        elif spec.kind == "execute":
            # legal only while >= 1 living subordinate holds an AT-MY-COMMAND
            # order of THIS issuer still awaiting the signal
            if soldier.effective_authority > 0 and any(
                sub.mission is not None
                and sub.mission.awaiting_signal
                and sub.mission.issuer_id == soldier.id
                for sub in soldier.living_subordinates(roster)
            ):
                mask[spec.index] = 1

    # Order vocabulary: command ranks only, doctrine-constrained ("full").
    # B3 arms: "flat" removes the order vocabulary outright; "nomask" keeps
    # rank admissibility and the cooldown but drops the doctrine constraint.
    if ablation == "flat":
        return mask
    if soldier.effective_authority > 0 and (soldier.mission is not None or ablation == "nomask"):
        allowed = (
            allowed_derivations(soldier.mission.type) if ablation != "nomask" else None
        )
        subs = soldier.living_subordinates(roster)
        objective_names = {o.name for o in world.objectives}
        control_names = world.control_names
        for spec in _ORDER_SPECS:
            if spec.order_slot >= len(subs):
                continue
            if allowed is not None and spec.order_mission not in allowed:
                continue
            if spec.order_objective is not None and spec.order_objective not in objective_names:
                continue
            # ADVANCE needs its control measure on THIS map
            if spec.order_control is not None and spec.order_control not in control_names:
                continue
            # unit-targeted SUPPORT needs a living unit in the supported slot
            if spec.order_mission is MissionType.SUPPORT and (
                spec.order_support_slot is None or spec.order_support_slot >= len(subs)
            ):
                continue
            # per-echelon admissibility: the recipient must be able to HOLD it
            if subs[spec.order_slot].effective_authority < min_hold_authority(spec.order_mission):
                continue
            if order_cooldown > 0:
                sub = subs[spec.order_slot]
                recently_ordered = (
                    sub.mission is not None and step - sub.last_order_step < order_cooldown
                )
                intent_changed = (
                    soldier.mission is not None
                    and soldier.mission.step_assigned > sub.last_order_step
                )
                contact_since = net_contact_step is not None and net_contact_step > sub.last_order_step
                if recently_ordered and not intent_changed and not contact_since:
                    continue
            mask[spec.index] = 1
    return mask
