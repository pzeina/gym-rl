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
    Formation,
    MissionType,
    allowed_derivations,
    is_completable,
    is_pending,
    min_hold_authority,
)
from cohort.core.orders import HQ_ID

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
    order_formation: Formation | None = None  # element stance (A5-3, no mission payload)


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
    # A5-4: trinôme peer synchronization by VOICE (any rank; the manual's
    # bond par binôme, commanded "à la voix ou aux gestes", pp. 14-15)
    add("sync_propose", "SYNC_PROPOSE")
    add("sync_go", "SYNC_GO")
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
        # A5-3: element stance orders to the subordinate LEADER in this slot
        # ('TL1, FORMATION COLUMN') — a stance, not a mission: the recipient
        # keeps its task; its element's geometry is reward-shaped, never forced.
        for formation in Formation:
            add(
                "order",
                f"ORDER_S{slot}_FORMATION_{formation.name}",
                order_slot=slot,
                order_formation=formation,
            )
    return specs


CATALOG: list[ActionSpec] = _build_catalog()
N_ACTIONS: int = len(CATALOG)

#: index lookups used by the mask builder
_STAY = next(s.index for s in CATALOG if s.kind == "stay")
_ORDER_SPECS = [s for s in CATALOG if s.kind == "order"]
#: order entries that carry a mission payload (A5-3 stance orders do not), with
#: their ordered task — the vocabulary :func:`order_options` reports on.
_TASK_ORDER_SPECS: tuple[tuple[int, str], ...] = tuple(
    (s.index, s.order_mission.name) for s in _ORDER_SPECS if s.order_mission is not None
)


def action_name(index: int) -> str:
    """Human-readable name of an action index."""
    return CATALOG[index].name


def order_options(mask: np.ndarray) -> dict[str, int]:
    """Admissible mission-payload order entries in ``mask``, by ordered task.

    The *opportunity* side of the ordered-task mix (refs issue #16). An order
    share on its own cannot say whether a task is rare because the policy
    dislikes it or because the mask rarely offers it, and the two differ per
    task by construction: SUPPORT is unit-targeted, so it needs a second
    living subordinate slot and vanishes entirely from missions that cannot
    derive it (SCREEN), while OBSERVE is objective-targeted and admissible
    wherever it is derivable at all. Measured on the masked-random floor,
    `squad` offers OBSERVE 2.9x more entries than SUPPORT and
    `fireteam_defend` offers SUPPORT 1.9x *more* than OBSERVE — so the raw
    ratio flatters the policy in one family and slanders it in the other.

    Counting *entries* rather than tasks is what makes the baseline exact: a
    uniform-over-legal policy picks an entry, not a task, so the expected
    share of orders going to a task is its share of admissible entries.
    A5-3 stance orders (``FORMATION X``) are excluded because they carry no
    mission and never enter ``orders_by_task`` either — numerator and
    denominator must range over the same vocabulary.

    Read off the mask itself, deliberately: re-deriving "could this order have
    been issued?" beside the mask is how the ``is_root_opord_claim``
    divergence stayed invisible for a training generation.
    """
    options: dict[str, int] = {}
    for index, task in _TASK_ORDER_SPECS:
        if mask[index]:
            options[task] = options.get(task, 0) + 1
    return options


def is_root_opord_claim(
    soldier: Soldier,
    roster: Roster,
    root_mission: MissionType | None,
    root_objective_id: int | None,
    *,
    defend_horizon: int | None = None,
) -> bool:
    """Is this the root reporting the *operation* complete?

    The root's OPORD claim is judged against the team success condition, not
    against the claimant's personal end state — a commander reports the
    mission complete when the unit achieved it, wherever it stands. That makes
    it admissible even when the root's mission type is a continuous posture
    (DEFEND, DENY) that no individual can ever "finish", which is why this
    predicate exists instead of a plain ``type in COMPLETABLE`` test.

    Single source of truth on purpose. ``compute_mask`` and
    ``CohortEnv._report_done`` both call it, because when they disagreed the
    result was silent: the mask required ``COMPLETABLE`` while the
    adjudicator's root branch required ``type is root_mission``, so on every
    DEFEND- or DENY-rooted scenario the root branch was unreachable, the root
    never transmitted MISSION COMPLETE at all, ``root_done_bonus`` was dead
    reward, and the completion grace window could only ever expire by timeout.
    Measured on fireteam_defend_v8: 0 admissible root claims in 30 episodes.
    """
    mission = soldier.mission
    return (
        mission is not None
        and soldier is roster.root()
        and mission.issuer_id == HQ_ID
        and root_mission is not None
        # v1.13, owner's decision: a continuous posture has no end state that
        # its holder may declare. DEFEND/DENY run until a new order arrives,
        # so the root does not claim them complete — it reports the situation
        # and COMMAND transmits ENDEX. Without this clause the root could
        # declare its own DEFEND operation over, which is the one thing the
        # doctrine table (``COMPLETABLE``) says it must not do.
        # v1.14 refines it: a defense ordered to a horizon DOES have a
        # declarable end state (see ``missions.is_completable``).
        and is_completable(root_mission, defend_horizon=defend_horizon)
        and mission.type is root_mission
        and mission.objective_id == root_objective_id
    )


def is_done_admissible(
    soldier: Soldier,
    roster: Roster,
    *,
    root_mission: MissionType | None,
    root_objective_id: int | None,
    step: int,
    done_cooldown: int,
    defend_horizon: int | None = None,
) -> bool:
    """May this agent transmit MISSION COMPLETE *this step*?

    The DONE branch of :func:`compute_mask`, lifted out so that anything
    needing to know whether the completion channel was *open* reads the same
    predicate the mask admits on. ``cohort.metrics.TraceRecorder`` does: a run
    that emits zero DONE reports is either a channel that was never open
    (nothing was priced — the ``is_root_opord_claim`` bug) or a policy that
    declined an open one (the price suppressed the act), and only the
    admissible-step count separates the two. Re-deriving the condition there
    would let the two drift, which is exactly the failure mode that made the
    first silence invisible for a whole training generation.
    """
    mission = soldier.mission
    if not soldier.alive or mission is None:
        return False
    # a pending order (A5-2) is not yet executing: nothing to report;
    # and a rejected claim cannot be re-rolled every tick (v1.10).
    # Completable-by-type OR the root's own OPORD: a DEFEND/DENY root
    # can never "finish" its posture, but it can and must report that
    # the *operation* succeeded (see is_root_opord_claim).
    #
    # The type test stays the plain ``COMPLETABLE`` one, not the horizon-aware
    # predicate: the horizon belongs to the ROOT's operation order — the one
    # HQ gives, ``ScenarioSpec.defend_horizon`` — and only the root is ordered
    # to it. A subordinate tasked DEFEND by its leader holds an indefinite
    # posture that ends when that leader re-tasks it, exactly as in v1.13.
    # No clause of the transmitted OPORD *text* names the hour (issue #30);
    # it is published as briefing header material, not spoken on the net, so
    # do not read this comment as pointing at a wording in ``language.py``.
    claimable = mission.type in COMPLETABLE or is_root_opord_claim(
        soldier, roster, root_mission, root_objective_id, defend_horizon=defend_horizon
    )
    return bool(
        claimable
        and not is_pending(mission, step)
        and step - soldier.last_done_reject_step >= done_cooldown
    )


def compute_mask(
    soldier: Soldier,
    roster: Roster,
    world: World,
    visible_enemy_in_range: bool,
    visible_enemy: bool,
    *,
    order_cooldown: int = 0,
    done_cooldown: int = 0,
    root_mission: MissionType | None = None,
    root_objective_id: int | None = None,
    defend_horizon: int | None = None,
    step: int = 0,
    net_contact_step: int | None = None,
    ablation: str = "full",
    has_voice_peer: bool = False,
    has_pending_sync: bool = False,
) -> np.ndarray:
    """Legality mask (int8, shape (N_ACTIONS,)) for one agent this step.

    ``order_cooldown`` > 0 masks re-tasking a subordinate within that many
    steps of its last received order, unless the leader's own mission changed
    since, or a CONTACT report hit the net since (``net_contact_step``).

    ``done_cooldown`` > 0 masks MISSION COMPLETE within that many steps of a
    rejected claim: a premature claimant must wait before re-rolling.
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
            if is_done_admissible(
                soldier,
                roster,
                root_mission=root_mission,
                root_objective_id=root_objective_id,
                step=step,
                done_cooldown=done_cooldown,
                defend_horizon=defend_horizon,
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
        elif spec.kind == "sync_propose":
            # A5-4: any agent with >= 1 trinôme peer within voice range
            if has_voice_peer:
                mask[spec.index] = 1
        elif spec.kind == "sync_go":
            # A5-4: only the proposer of a still-live (unexpired) proposal
            if has_pending_sync:
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
            # A5-3 stance orders: any mission-holding leader may set a
            # subordinate LEADER's formation; the recipient must actually
            # lead an element. No doctrine derivation, no cooldown — a
            # stance is how the element moves, not what it does.
            if spec.order_formation is not None:
                if subs[spec.order_slot].living_subordinates(roster):
                    mask[spec.index] = 1
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
