"""Mission tasks, derivation doctrine, and execution semantics.

The mission set is the full MICAT catalog of the French PROTERRE manual
(``docs/manuel-proterre.pdf``), carried under English names with PROTERRE
semantics (owner decision, 2026-08-05; see ``docs/missions.md`` for the
per-mission doctrine with manual page references):

    RECON   (RECONNAÎTRE)  get intel on an objective; MAY engage
    SCREEN  (ÉCLAIRER)     intel WITHOUT engaging; weapons tight
    OBSERVE (SURVEILLER)   static observation posture; detect and alert
    SUPPORT (APPUYER)      unit-targeted fire support ("pas un pas sans appui")
    COVER   (COUVRIR)      flank guard on an objective; free to fire in place
    DEFEND  (TENIR)        occupy and hold an objective
    DENY    (INTERDIRE)    section-level area denial (authority >= 2)
    SEIZE                  take possession of an objective and clear it
    CLEAR                  eliminate all hostiles at an objective
    RALLY                  assemble on the direct leader
    HOLD                   hold current position
    ADVANCE                move to / cross a control measure (WP / PL), then hold

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
    """MICAT tactical tasks an agent can hold / an order can carry.

    The declaration order is load-bearing: it defines the observation
    one-hot layout and the action-catalog layout. Changing it breaks
    every trained checkpoint.
    """

    RECON = "recon"      # RECONNAÎTRE: get intel on an objective, may engage
    SCREEN = "screen"    # ÉCLAIRER: intel without engaging (weapons tight)
    OBSERVE = "observe"  # SURVEILLER: static observation posture, alert role
    SUPPORT = "support"  # APPUYER: unit-targeted fire support
    COVER = "cover"      # COUVRIR: flank guard at an objective
    DEFEND = "defend"    # TENIR: occupy an objective and repel hostiles
    DENY = "deny"        # INTERDIRE: section-level area denial
    SEIZE = "seize"      # take possession of an objective and clear it
    CLEAR = "clear"      # eliminate all hostiles at an objective
    RALLY = "rally"      # assemble on the direct leader
    HOLD = "hold"        # hold current position
    ADVANCE = "advance"  # move to / cross a control measure (waypoint / phase line)

    @classmethod
    def from_str(cls, value: str) -> MissionType:
        """Parse from case-insensitive string."""
        return cls(value.lower())


class Formation(Enum):
    """Element-level movement stances (A5-3) — manual pp. 14-15.

    The PROTERRE group moves in three formations: EN COLONNE (speed, night,
    following a route), EN LIGNE (crossing a crest/road, assaulting a wood
    line), and EN COLONNE DOUBLE (teams abreast). COLUMN and LINE carry the
    manual's first two; WEDGE stands in for the two-directions-at-once role
    of the colonne double (owner scope). A stance is ordered to a LEADER
    ('TL1, FORMATION COLUMN'), persists until changed, and shapes — never
    forces — the element's geometry via a reward term.
    """

    COLUMN = "column"  # trail behind the leader within 1-cell lateral
    LINE = "line"      # abreast of the leader within 1-cell depth
    WEDGE = "wedge"    # V: diagonal offsets behind the leader


#: How far behind/beside its leader a formation slot may trail (cells).
FORMATION_DEPTH = 6.0


def in_formation(
    formation: Formation,
    leader_pos: tuple[float, float],
    heading: tuple[int, int],
    member_pos: tuple[float, float],
) -> bool:
    """Is a member at its formation station relative to the leader?

    Geometry in the leader's frame: ``along`` = signed distance along the
    leader's heading (negative = behind), ``lateral`` = signed distance
    across it. A leader that has never moved has no heading — no station
    exists, nothing is in formation.

    * COLUMN: behind (``-DEPTH <= along < 0``), within 1 cell of the axis;
    * LINE: abreast (``|along| <= 1``), 1..DEPTH cells to either side;
    * WEDGE: behind on the diagonals — ``|along|`` and ``|lateral|`` within
      1 cell of each other, at least 1 cell off-axis.
    """
    hx, hy = heading
    if hx == 0 and hy == 0:
        return False
    rx = member_pos[0] - leader_pos[0]
    ry = member_pos[1] - leader_pos[1]
    along = rx * hx + ry * hy
    lateral = -rx * hy + ry * hx
    if formation is Formation.COLUMN:
        return -FORMATION_DEPTH <= along < 0 and abs(lateral) <= 1
    if formation is Formation.LINE:
        return abs(along) <= 1 and 1 <= abs(lateral) <= FORMATION_DEPTH
    # WEDGE
    return (
        -FORMATION_DEPTH <= along < 0
        and abs(lateral) >= 1
        and abs(abs(along) - abs(lateral)) <= 1
    )


#: Missions that target a named objective. SUPPORT targets a friendly unit,
#: RALLY targets the leader, HOLD targets the position where the order was
#: received.
NEEDS_OBJECTIVE: frozenset[MissionType] = frozenset(
    {
        MissionType.RECON,
        MissionType.SCREEN,
        MissionType.OBSERVE,
        MissionType.COVER,
        MissionType.DEFEND,
        MissionType.DENY,
        MissionType.SEIZE,
        MissionType.CLEAR,
    }
)

#: Missions whose order names a friendly element instead of an objective.
UNIT_TARGETED: frozenset[MissionType] = frozenset({MissionType.SUPPORT})

#: Missions that target a named control measure (waypoint or phase line) —
#: the A5 vocabulary that puts route geometry on the net.
NEEDS_CONTROL: frozenset[MissionType] = frozenset({MissionType.ADVANCE})

#: Missions with a definite end state that can be reported COMPLETE.
#: OBSERVE / SUPPORT / COVER / DEFEND / DENY / HOLD are continuous postures —
#: they end when a new order arrives (SUPPORT also ends when the supported
#: unit dies), so MISSION COMPLETE is inadmissible for them.
COMPLETABLE: frozenset[MissionType] = frozenset(
    {
        MissionType.RECON,
        MissionType.SCREEN,
        MissionType.SEIZE,
        MissionType.CLEAR,
        MissionType.RALLY,
        MissionType.ADVANCE,  # completes on reaching/crossing the control measure
    }
)

#: Derivation doctrine: own mission → subordinate missions allowed, in
#: preference order. Rebuilt for the MICAT set from the manual's mission
#: definitions (docs/missions.md). Note DENY: a section holding INTERDIRE
#: tasks its groups with DEFEND/COVER/SUPPORT/OBSERVE — DENY itself is a
#: section-level mission no group can hold (tableau récapitulatif, manual
#: p. 8), so it is derivable to nobody.
DOCTRINE: dict[MissionType, tuple[MissionType, ...]] = {
    MissionType.RECON: (
        MissionType.RECON, MissionType.SUPPORT, MissionType.OBSERVE, MissionType.SCREEN,
        MissionType.ADVANCE,
    ),
    MissionType.SCREEN: (MissionType.SCREEN, MissionType.OBSERVE, MissionType.HOLD),
    MissionType.OBSERVE: (MissionType.OBSERVE, MissionType.COVER, MissionType.HOLD),
    MissionType.SUPPORT: (MissionType.SUPPORT, MissionType.OBSERVE, MissionType.HOLD),
    MissionType.COVER: (MissionType.COVER, MissionType.OBSERVE, MissionType.HOLD),
    MissionType.DEFEND: (
        MissionType.DEFEND, MissionType.SUPPORT, MissionType.OBSERVE, MissionType.HOLD,
        MissionType.ADVANCE,
    ),
    MissionType.DENY: (
        MissionType.DEFEND, MissionType.COVER, MissionType.SUPPORT, MissionType.OBSERVE,
        MissionType.ADVANCE,
    ),
    MissionType.SEIZE: (
        MissionType.SEIZE, MissionType.CLEAR, MissionType.SUPPORT, MissionType.OBSERVE,
        MissionType.ADVANCE,
    ),
    MissionType.CLEAR: (MissionType.CLEAR, MissionType.SUPPORT),
    MissionType.RALLY: (MissionType.RALLY, MissionType.HOLD),
    MissionType.HOLD: (MissionType.HOLD, MissionType.OBSERVE),
    # ADVANCE is a maneuver leg (actes élémentaires, manual pp. 14-15): it
    # decomposes into further legs, supported bounds, and watch postures.
    MissionType.ADVANCE: (
        MissionType.ADVANCE, MissionType.SUPPORT, MissionType.OBSERVE,
    ),
}

#: Per-echelon admissibility: minimum *effective* authority required to HOLD
#: a mission (manual p. 8, tableau récapitulatif: INTERDIRE is a section /
#: company mission, never a group's). Enforced in the order mask and in
#: ``inject_order`` validation.
MISSION_MIN_HOLD_AUTHORITY: dict[MissionType, int] = {MissionType.DENY: 2}


def min_hold_authority(mission: MissionType) -> int:
    """Minimum effective authority an agent needs to hold ``mission``."""
    return MISSION_MIN_HOLD_AUTHORITY.get(mission, 0)


#: Radius (grid cells) within which each mission counts as "in position".
IN_POSITION_RADIUS: dict[MissionType, float] = {
    MissionType.RECON: 7.0,    # observation ring, paired with LOS requirement
    MissionType.SCREEN: 7.0,   # same observation semantics as RECON
    MissionType.OBSERVE: 9.0,  # paired with LOS requirement
    MissionType.SUPPORT: 10.0,  # of the supported soldier, paired with LOS to it
    MissionType.COVER: 6.0,    # flank-guard station; no LOS requirement
    MissionType.DEFEND: 3.5,
    MissionType.DENY: 5.0,     # area denial: like DEFEND, wider footprint
    MissionType.SEIZE: 2.5,
    MissionType.CLEAR: 3.5,
    MissionType.RALLY: 2.5,
    MissionType.HOLD: 1.5,
    MissionType.ADVANCE: 2.5,  # of the waypoint / the phase line's nearest point
}

#: Missions whose "in position" additionally requires line of sight to the
#: anchor (the objective — or the supported soldier for SUPPORT).
LOS_REQUIRED: frozenset[MissionType] = frozenset(
    {MissionType.RECON, MissionType.SCREEN, MissionType.OBSERVE, MissionType.SUPPORT}
)

#: Fire-discipline classes (consumed by the env's combat-reward scaling):
#: WEAPONS_TIGHT missions earn nothing for fire, POSITION_ANCHORED_FIRE
#: missions earn combat rewards only when firing from the mission position.
WEAPONS_TIGHT: frozenset[MissionType] = frozenset({MissionType.SCREEN})
POSITION_ANCHORED_FIRE: frozenset[MissionType] = frozenset(
    {
        MissionType.OBSERVE,
        MissionType.SUPPORT,
        MissionType.COVER,
        MissionType.DEFEND,
        MissionType.DENY,
        MissionType.HOLD,
    }
)

#: Steps of cumulative observation required to complete a RECON / SCREEN.
RECON_OBSERVE_STEPS = 5

#: Steps of cumulative TEAM observation that complete a root-held (OPORD)
#: RECON / SCREEN — the campaign success condition. The operation's
#: observation is aggregated squad-wide: any member on the ring advances it,
#: so the commander can command from cover (refs issue #9 — the personal
#: adjudication measurably drove the human root into exposure).
TEAM_OBSERVE_STEPS = 2 * RECON_OBSERVE_STEPS


def allowed_derivations(own_mission: MissionType | None) -> tuple[MissionType, ...]:
    """Missions a leader may order subordinates given its own mission.

    A leader with no mission has nothing to derive from and may not order
    (the senior agent always receives the OPORD at episode start).
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
    """Mutable per-agent mission state (the standing order being executed).

    ``extra`` carries mission-specific state; SUPPORT stores the id of the
    supported soldier under ``extra["supported_id"]`` (the anchor tracks that
    soldier's position dynamically, like RALLY tracks the leader).

    ``team_observation`` marks a root-held (OPORD) RECON / SCREEN: the
    holder commands the *operation*, so observation is adjudicated on the
    squad's aggregated counter — the environment mirrors it into
    ``observe_steps`` and completion requires ``TEAM_OBSERVE_STEPS`` (the
    success condition), while in-position credit follows the team. A
    subordinate's RECON / SCREEN keeps personal ``observe_steps``: its own
    DONE reflects its own task.
    """

    type: MissionType
    objective_id: int | None       # index into world objectives, or None
    anchor: tuple[float, float]    # target point: objective center, leader pos, or hold pos
    issuer_id: int                 # agent id, or -1 for HQ/human
    step_assigned: int
    observe_steps: int = 0         # RECON / SCREEN progress (team-mirrored on OPORDs)
    team_observation: bool = False  # root OPORD RECON/SCREEN: team-adjudicated
    # --- timing qualifiers (A5-2): a pending order stages, then executes ---
    effective_at: int | None = None  # "AT T PLUS n": tick the order becomes effective
    awaiting_signal: bool = False    # "AT MY COMMAND": pending until the issuer's EXECUTE
    extra: dict = field(default_factory=dict)


def is_pending(mission: Mission, step: int) -> bool:
    """A pending order (A5-2) has been received but is not yet in effect.

    Until it is, the recipient's compliance is judged as HOLD near the
    position where the order landed (staging, ``extra["staging"]``), the
    mission cannot complete, and the pending state is observable.
    """
    return mission.awaiting_signal or (
        mission.effective_at is not None and step < mission.effective_at
    )


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
    #: did the mission ANCHOR move this step? False for a fixed anchor (an
    #: objective, a waypoint); meaningful for the anchors that are themselves
    #: soldiers — SUPPORT's supported unit, RALLY's leader. Displacing to hold
    #: station on a bounding element is execution, not drift, and only this
    #: flag can tell the two apart.
    anchor_moved: bool = False


def _progress(ctx: ComplianceContext) -> float:
    """Potential-based progress toward the mission anchor, in ~[-0.75, 0.75]."""
    delta = ctx.dist_prev - ctx.dist_now
    return 0.5 * max(-1.5, min(1.5, delta))


def compliance(mission: MissionType | None, ctx: ComplianceContext) -> float:
    """Per-step compliance score in [-1, 1]: is the agent executing its order?"""
    if mission is None:
        return 0.0
    if mission is MissionType.RECON:
        # RECONNAÎTRE may engage: no fire penalty (manual p. 30)
        return 0.6 if ctx.in_position else _progress(ctx)
    if mission is MissionType.SCREEN:
        # ÉCLAIRER: intel without engaging (manual p. 32) — weapons tight
        if ctx.fired:
            return -0.6
        return 0.6 if ctx.in_position else _progress(ctx)
    if mission is MissionType.OBSERVE:
        # a genuinely static posture: the anchor never moves, so settle on it
        if ctx.in_position:
            return 0.6 if ctx.stationary else 0.1
        return _progress(ctx)
    if mission is MissionType.SUPPORT:
        # APPUYER is overwatch of a MOVING element — "pas un pas sans appui".
        # It was scored as a static posture identical to OBSERVE, and that
        # inverted the doctrine: the anchor is the supported soldier, so a
        # supporter displacing to hold range and LOS on a bounding element
        # broke ``stationary`` and collected 0.1, while a supporter that let
        # its element walk away and stood still collected 0.6. Measured on the
        # squad map, six steps of a bounding element: following it paid 0.60
        # total, abandoning it paid 3.60 — a 6x premium for not supporting,
        # and OBSERVE offered the same 3.60 for watching a fixed point that
        # can never outrun you. Hence OBSERVE ordered 3.6-10x more than
        # SUPPORT across every corpus that used either.
        # Movement is excused exactly when the element itself moved: keeping
        # station is execution. If the element is holding, so should its
        # support — which is what stops this paying for aimless drift.
        if ctx.in_position:
            return 0.6 if (ctx.stationary or ctx.anchor_moved) else 0.1
        return _progress(ctx)
    if mission is MissionType.COVER:
        if ctx.in_position:
            return 0.5 if ctx.stationary else 0.1
        return _progress(ctx)
    if mission in (
        MissionType.SEIZE,
        MissionType.DEFEND,
        MissionType.DENY,
        MissionType.RALLY,
        MissionType.ADVANCE,  # reach the control measure, then hold on it
    ):
        return 0.5 if ctx.in_position else _progress(ctx)
    if mission is MissionType.CLEAR:
        if ctx.fired:
            return 0.8
        return 0.0 if ctx.visible_enemies > 0 else _progress(ctx)
    # HOLD
    if ctx.in_position:
        return 0.5 if ctx.stationary else 0.1
    return _progress(ctx)


def is_complete(mission: Mission, ctx: ComplianceContext) -> bool:
    """True if the mission's end state is objectively reached."""
    if mission.type not in COMPLETABLE:
        return False
    if mission.type in (MissionType.RECON, MissionType.SCREEN):
        goal = TEAM_OBSERVE_STEPS if mission.team_observation else RECON_OBSERVE_STEPS
        return mission.observe_steps >= goal
    if mission.type is MissionType.SEIZE:
        return ctx.in_position and ctx.enemies_at_objective == 0
    if mission.type is MissionType.CLEAR:
        return ctx.enemies_at_objective == 0
    if mission.type is MissionType.ADVANCE:
        # reached the control measure, or crossed the phase line (the env
        # flips extra["crossed"] when the agent's side of the line changes)
        return ctx.in_position or bool(mission.extra.get("crossed"))
    # RALLY
    return ctx.dist_to_leader <= IN_POSITION_RADIUS[MissionType.RALLY]
