"""Mission types, derivation doctrine, and mission execution semantics.

A *mission* is the payload of an order: what the recipient must do. Doctrine
constrains how a leader may decompose its own mission into subordinate
missions (preference-ordered tuples). The environment enforces doctrine as a
hard constraint through action masking, so a trained leader is *guaranteed*
to issue doctrine-valid orders; learning decides *which* valid order fits the
tactical situation.

Compliance and completion are pure functions over a small context struct the
environment computes each step, which keeps mission semantics unit-testable
without a live world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MissionType(Enum):
    """Missions an agent can hold / an order can carry."""

    RECON = "recon"          # observe an objective without engaging
    SEIZE = "seize"          # reach an objective and clear it of hostiles
    DEFEND = "defend"        # occupy an objective and repel hostiles
    OVERWATCH = "overwatch"  # static position with line of sight on an objective
    ENGAGE = "engage"        # close with and destroy hostiles at an objective
    REGROUP = "regroup"      # rally on the direct leader
    HOLD = "hold"            # hold current position

    @classmethod
    def from_str(cls, value: str) -> MissionType:
        """Parse from case-insensitive string."""
        return cls(value.lower())


#: Missions that target a named objective. REGROUP targets the leader,
#: HOLD targets the position where the order was received.
NEEDS_OBJECTIVE: frozenset[MissionType] = frozenset(
    {MissionType.RECON, MissionType.SEIZE, MissionType.DEFEND, MissionType.OVERWATCH, MissionType.ENGAGE}
)

#: Missions with a definite end state that can be reported COMPLETE.
#: DEFEND / OVERWATCH / HOLD are continuous postures — they end when a new
#: order arrives, so MISSION COMPLETE is inadmissible for them.
COMPLETABLE: frozenset[MissionType] = frozenset(
    {MissionType.RECON, MissionType.SEIZE, MissionType.ENGAGE, MissionType.REGROUP}
)

#: Derivation doctrine: own mission → subordinate missions allowed, in
#: preference order. Adapted from the legacy TACTICAL_DERIVATION_DOCTRINE.
DOCTRINE: dict[MissionType, tuple[MissionType, ...]] = {
    MissionType.RECON: (MissionType.RECON, MissionType.OVERWATCH, MissionType.HOLD),
    MissionType.SEIZE: (MissionType.SEIZE, MissionType.ENGAGE, MissionType.OVERWATCH),
    MissionType.DEFEND: (MissionType.DEFEND, MissionType.OVERWATCH, MissionType.HOLD),
    MissionType.OVERWATCH: (MissionType.OVERWATCH, MissionType.HOLD),
    MissionType.ENGAGE: (MissionType.ENGAGE, MissionType.OVERWATCH),
    MissionType.REGROUP: (MissionType.REGROUP, MissionType.HOLD),
    MissionType.HOLD: (MissionType.HOLD, MissionType.OVERWATCH),
}

#: Radius (grid cells) within which each mission counts as "in position".
IN_POSITION_RADIUS: dict[MissionType, float] = {
    MissionType.RECON: 7.0,      # observation ring, paired with LOS requirement
    MissionType.SEIZE: 2.5,
    MissionType.DEFEND: 3.5,
    MissionType.OVERWATCH: 9.0,  # paired with LOS requirement
    MissionType.ENGAGE: 3.5,
    MissionType.REGROUP: 2.5,
    MissionType.HOLD: 1.5,
}

#: Steps of cumulative observation required to complete a RECON.
RECON_OBSERVE_STEPS = 5


def allowed_derivations(own_mission: MissionType | None) -> tuple[MissionType, ...]:
    """Missions a leader may order subordinates given its own mission.

    A leader with no mission has nothing to derive from and may not order
    (the root agent always receives the OPORD at episode start).
    """
    if own_mission is None:
        return ()
    return DOCTRINE[own_mission]


def derivation_quality(own_mission: MissionType | None, proposed: MissionType) -> float:
    """Score how well a proposed subordinate mission fits doctrine.

    1.0 → preferred (first allowed), 0.5 → allowed, -0.5 → violates doctrine,
    0.0 → no superior mission to derive from.
    """
    allowed = allowed_derivations(own_mission)
    if not allowed:
        return 0.0
    if proposed not in allowed:
        return -0.5
    return 1.0 if proposed == allowed[0] else 0.5


@dataclass
class Mission:
    """Mutable per-agent mission state (the standing order being executed)."""

    type: MissionType
    objective_id: int | None       # index into world objectives, or None
    anchor: tuple[float, float]    # target point: objective center, leader pos, or hold pos
    issuer_id: int                 # agent id, or -1 for HQ/human
    step_assigned: int
    observe_steps: int = 0         # RECON progress
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ComplianceContext:
    """Everything mission semantics need to judge one agent for one step."""

    dist_prev: float               # distance to mission anchor before the step
    dist_now: float                # distance after the step
    in_position: bool              # within IN_POSITION_RADIUS (and LOS where required)
    stationary: bool               # agent did not change cell this step
    fired: bool                    # agent fired this step
    visible_enemies: int           # enemies currently visible to the agent
    enemies_at_objective: int      # living enemies within the objective radius
    dist_to_leader: float          # distance to the direct leader (inf if none)


def _progress(ctx: ComplianceContext) -> float:
    """Potential-based progress toward the mission anchor, in ~[-0.75, 0.75]."""
    delta = ctx.dist_prev - ctx.dist_now
    return 0.5 * max(-1.5, min(1.5, delta))


def compliance(mission: MissionType | None, ctx: ComplianceContext) -> float:
    """Per-step compliance score in [-1, 1]: is the agent executing its order?"""
    if mission is None:
        return 0.0
    if mission is MissionType.RECON:
        if ctx.fired:
            return -0.6  # recon is stealthy: do not engage
        return 0.6 if ctx.in_position else _progress(ctx)
    if mission is MissionType.SEIZE:
        return 0.5 if ctx.in_position else _progress(ctx)
    if mission is MissionType.DEFEND:
        return 0.5 if ctx.in_position else _progress(ctx)
    if mission is MissionType.OVERWATCH:
        if ctx.in_position:
            return 0.6 if ctx.stationary else 0.1
        return _progress(ctx)
    if mission is MissionType.ENGAGE:
        if ctx.fired:
            return 0.8
        return 0.0 if ctx.visible_enemies > 0 else _progress(ctx)
    if mission is MissionType.REGROUP:
        return 0.5 if ctx.in_position else _progress(ctx)
    # HOLD
    if ctx.in_position:
        return 0.5 if ctx.stationary else 0.1
    return _progress(ctx)


def is_complete(mission: Mission, ctx: ComplianceContext) -> bool:
    """True if the mission's end state is objectively reached."""
    if mission.type not in COMPLETABLE:
        return False
    if mission.type is MissionType.RECON:
        return mission.observe_steps >= RECON_OBSERVE_STEPS
    if mission.type is MissionType.SEIZE:
        return ctx.in_position and ctx.enemies_at_objective == 0
    if mission.type is MissionType.ENGAGE:
        return ctx.enemies_at_objective == 0
    # REGROUP
    return ctx.dist_to_leader <= IN_POSITION_RADIUS[MissionType.REGROUP]
