"""The command language: format agent radio traffic, parse human orders.

Voice procedure follows NATO conventions (ACP 125 prowords: THIS IS, OVER,
OUT, WILCO, ALL STATIONS; four-digit GRID references; phonetic objective
names). Formatting and parsing are inverses over the order subset, so a human
can type exactly what agents say to each other:

    TL1, SEIZE OBJ ALPHA
    RFN2, RALLY ON ME
    RFN1, HOLD POSITION
    TL2, OBSERVE OBJ BRAVO
    TL2, SUPPORT TL1

Mission keywords accept common synonyms (take/capture → SEIZE, watch/
overwatch → OBSERVE, destroy/attack/engage → CLEAR, regroup/assemble →
RALLY, scout → RECON...) and the PROTERRE French doctrine names (éclairer →
SCREEN, surveiller → OBSERVE, appuyer → SUPPORT, couvrir → COVER, tenir →
DEFEND, interdire → DENY).

Backwards-friendly forms: ``support <callsign>`` / ``cover <callsign>`` /
``cover for <callsign>`` parse as the unit-targeted SUPPORT, while the plain
objective forms ``support obj X`` / ``cover obj X`` parse as OBSERVE at the
objective (the closest analog of the retired OVERWATCH task those phrases
used to mean). COVER's own canonical phrase is ``COVER FLANK OBJ X``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cohort.core.missions import NEEDS_CONTROL, NEEDS_OBJECTIVE, MissionType

#: Objective slot names, addressed as "OBJ ALPHA" etc. (NATO phonetic).
OBJECTIVE_NAMES: tuple[str, ...] = ("ALPHA", "BRAVO", "CHARLIE", "DELTA")

#: Control-measure names (A5). Waypoints take metal names ("WP GOLD"); phase
#: lines take mineral/color names ("PL AMBER"). Disjoint from objective names
#: and from each other, so a bare name resolves its kind unambiguously.
WAYPOINT_NAMES: tuple[str, ...] = ("GOLD", "SILVER", "COPPER", "IRON")
PHASE_LINE_NAMES: tuple[str, ...] = ("AMBER", "COBALT", "CRIMSON")
CONTROL_NAMES: tuple[str, ...] = WAYPOINT_NAMES + PHASE_LINE_NAMES


def control_phrase(name: str) -> str:
    """Spoken form of a control measure: 'WP GOLD' / 'PL AMBER'."""
    if name in WAYPOINT_NAMES:
        return f"WP {name}"
    if name in PHASE_LINE_NAMES:
        return f"PL {name}"
    msg = f"Unknown control measure {name!r} (known: {', '.join(CONTROL_NAMES)})"
    raise ValueError(msg)

_SYNONYMS: dict[str, MissionType] = {
    "recon": MissionType.RECON,
    "reconnoiter": MissionType.RECON,
    "reconnoitre": MissionType.RECON,
    "reconnaitre": MissionType.RECON,
    "scout": MissionType.RECON,
    "screen": MissionType.SCREEN,
    "eclairer": MissionType.SCREEN,
    "observe": MissionType.OBSERVE,
    "surveiller": MissionType.OBSERVE,
    "overwatch": MissionType.OBSERVE,
    "watch": MissionType.OBSERVE,
    "cover": MissionType.OBSERVE,      # 'cover obj X' → OBSERVE; 'cover <cs>' → SUPPORT
    "support": MissionType.OBSERVE,    # 'support obj X' → OBSERVE; 'support <cs>' → SUPPORT
    "seize": MissionType.SEIZE,
    "take": MissionType.SEIZE,
    "capture": MissionType.SEIZE,
    "assault": MissionType.SEIZE,
    "secure": MissionType.SEIZE,
    "defend": MissionType.DEFEND,
    "tenir": MissionType.DEFEND,
    "guard": MissionType.DEFEND,
    "retain": MissionType.DEFEND,
    "deny": MissionType.DENY,
    "interdict": MissionType.DENY,
    "interdire": MissionType.DENY,
    "clear": MissionType.CLEAR,
    "destroy": MissionType.CLEAR,
    "engage": MissionType.CLEAR,
    "attack": MissionType.CLEAR,
    "eliminate": MissionType.CLEAR,
    "neutralize": MissionType.CLEAR,
    "fix": MissionType.CLEAR,
    "rally": MissionType.RALLY,
    "regroup": MissionType.RALLY,
    "assemble": MissionType.RALLY,
    "return": MissionType.RALLY,
    "hold": MissionType.HOLD,
    "halt": MissionType.HOLD,
    "stop": MissionType.HOLD,
    "advance": MissionType.ADVANCE,
    "proceed": MissionType.ADVANCE,
}

#: Keywords that select the COVER flank guard (checked before the generic
#: synonym scan, so the canonical "COVER FLANK OBJ X" round-trips while a
#: plain "cover obj X" stays OBSERVE).
_COVER_WORDS = frozenset({"flank", "couvrir"})

#: Unit-targeted SUPPORT: "support TL1", "appuyer TL1", "cover [for] TL1".
_SUPPORT_RE = re.compile(r"\b(?:support|appuyer|cover(?:\s+for)?)\s+([a-z]{2,3}\d+)\b")

#: Control-measure reference: "wp gold", "pl amber", or a bare name.
_CONTROL_RE = re.compile(
    r"(?:\b(?:wp|waypoint|pl|phase\s*line)\s+)?\b("
    + "|".join(n.lower() for n in CONTROL_NAMES)
    + r")\b"
)


def grid_ref(pos: tuple[int, int]) -> str:
    """Four-digit NATO-style grid reference, e.g. (14, 7) → 'GRID 1407'."""
    return f"GRID {int(pos[0]):02d}{int(pos[1]):02d}"


@dataclass(frozen=True)
class ParsedOrder:
    """Result of parsing a human order line."""

    recipient_callsign: str
    mission: MissionType
    objective_name: str | None
    target_callsign: str | None = None  # supported unit (SUPPORT only)
    control_name: str | None = None     # control measure (ADVANCE only)


class OrderParseError(ValueError):
    """Raised when a human order line cannot be parsed; message explains why."""


def mission_phrase(mission: MissionType, target: str | None) -> str:
    """Canonical spoken form of a tasking.

    ``target`` is the objective name for objective-targeted missions, the
    supported unit's callsign for SUPPORT, or the control-measure name for
    ADVANCE: 'SEIZE OBJ ALPHA', 'SUPPORT TL1', 'COVER FLANK OBJ BRAVO',
    'ADVANCE TO WP GOLD', 'ADVANCE TO PL AMBER', 'RALLY ON ME',
    'HOLD POSITION'.
    """
    if mission is MissionType.SUPPORT:
        return f"SUPPORT {target}"
    if mission is MissionType.COVER:
        return f"COVER FLANK OBJ {target}"
    if mission in NEEDS_CONTROL:
        return f"ADVANCE TO {control_phrase(target)}"
    if mission in NEEDS_OBJECTIVE:
        return f"{mission.name} OBJ {target}"
    if mission is MissionType.RALLY:
        return f"{mission.name} ON ME"
    return f"{mission.name} POSITION"  # HOLD


def format_order(issuer_cs: str, recipient_cs: str, mission: MissionType, target: str | None) -> str:
    """Radio form of an order: 'TL1, THIS IS SL1: SEIZE OBJ ALPHA. OUT.'"""
    return f"{recipient_cs}, THIS IS {issuer_cs}: {mission_phrase(mission, target)}. OUT."


def format_opord(recipient_cs: str, mission: MissionType, target: str | None) -> str:
    """Initial operations order from higher HQ to the senior agent."""
    return f"{recipient_cs}, THIS IS HQ: OPORD — {mission_phrase(mission, target)}. OUT."


def format_ack(issuer_cs: str, recipient_cs: str) -> str:
    """Order acknowledgement (auto-emitted on receipt)."""
    return f"{issuer_cs}, THIS IS {recipient_cs}: WILCO. OUT."


def format_contact(leader_cs: str, sender_cs: str, n_hostiles: int, pos: tuple[int, int]) -> str:
    """Enemy sighting report (NATO contact report shape)."""
    return f"{leader_cs}, THIS IS {sender_cs}: CONTACT, {grid_ref(pos)}, {n_hostiles} x ENEMY. OVER."


def format_sitrep(leader_cs: str, sender_cs: str, health: int, ammo: int, pos: tuple[int, int]) -> str:
    """Situation report."""
    return f"{leader_cs}, THIS IS {sender_cs}: SITREP, {grid_ref(pos)}, HEALTH {health}%, AMMO {ammo}. OVER."


def format_done(leader_cs: str, sender_cs: str, mission: MissionType, target: str | None) -> str:
    """Mission-complete report."""
    return f"{leader_cs}, THIS IS {sender_cs}: {mission_phrase(mission, target)} — COMPLETE. OVER."


def format_done_confirm(
    claimant_cs: str, leader_cs: str, mission: MissionType, target: str | None
) -> str:
    """Superior confirms a truthful completion report."""
    return f"{claimant_cs}, THIS IS {leader_cs}: ROGER, {mission_phrase(mission, target)} CONFIRMED. OUT."


def format_done_reject(claimant_cs: str, leader_cs: str) -> str:
    """Superior rejects a false completion claim; the mission stands."""
    return f"{claimant_cs}, THIS IS {leader_cs}: NEGATIVE, CONTINUE MISSION. OUT."


def format_support_end(leader_cs: str, sender_cs: str, supported_cs: str) -> str:
    """Supporter reports its SUPPORT mission ended: the supported unit fell."""
    return (
        f"{leader_cs}, THIS IS {sender_cs}: SUPPORT ENDED, "
        f"{supported_cs} IS DOWN. STANDING BY. OVER."
    )


def format_casualty(callsign: str) -> str:
    """Broadcast when an agent goes down."""
    return f"ALL STATIONS: {callsign} IS DOWN. OUT."


def format_trap(callsign: str, pos: tuple[int, int]) -> str:
    """Broadcast when a friendly triggers a hidden device (mine / booby trap).

    BRIQUE harassment "y compris les mines et les pièges" (manual p. 9);
    umpire/net convention like CASUALTY: the report comes from HQ.
    """
    return f"ALL STATIONS: {callsign} HIT A DEVICE AT {grid_ref(pos)}. OUT."


def format_taking_command(new_cs: str, dead_cs: str) -> str:
    """Broadcast when succession occurs."""
    return f"ALL STATIONS, THIS IS {new_cs}: {dead_cs} IS DOWN. I AM ASSUMING COMMAND. OUT."


def format_assuming_position(new_cs: str, of_cs: str) -> str:
    """Broadcast when a recursive succession fill moves an agent up.

    The direct successor of the casualty says ``I AM ASSUMING COMMAND``
    (:func:`format_taking_command`); agents filling the vacancies that
    promotion leaves further down the chain use this form.
    """
    return f"ALL STATIONS, THIS IS {new_cs}: ASSUMING {of_cs}'S POSITION. OUT."


_ORDER_RE = re.compile(
    r"^\s*(?:(?P<issuer>[A-Za-z]{2,3}\d+)\s*(?:,|:)\s*)?"    # optional issuer prefix (ignored)
    r"(?:this\s+is\s+[A-Za-z]{2,3}\d+\s*(?::|,)\s*)?"        # optional 'THIS IS X:'
    r"(?P<recipient>[A-Za-z]{2,3}\d+)\s*(?:,|:)\s*"          # recipient callsign
    r"(?P<body>.+?)\.?\s*(?:out\.?|over\.?)?\s*$",           # order body
    re.IGNORECASE,
)


def parse_order(text: str) -> ParsedOrder:
    """Parse a human order line into (recipient, mission, objective/target).

    Accepts e.g. 'TL1, seize obj alpha', 'rfn2: rally on me',
    'RFN1, hold position', 'TL2, observe obj bravo. out.',
    'TL2, support TL1' (unit-targeted SUPPORT).
    """
    m = _ORDER_RE.match(text)
    if not m:
        msg = f"Cannot parse order: {text!r}. Expected '<CALLSIGN>, <MISSION> [OBJ <NAME>]'."
        raise OrderParseError(msg)
    recipient = m.group("recipient").upper()
    body = m.group("body").strip().lower().rstrip(".")

    # Unit-targeted SUPPORT: the order names a friendly callsign, not an
    # objective ('support tl1', 'appuyer tl1', 'cover for tl1').
    sup = _SUPPORT_RE.search(body)
    if sup:
        return ParsedOrder(
            recipient_callsign=recipient,
            mission=MissionType.SUPPORT,
            objective_name=None,
            target_callsign=sup.group(1).upper(),
        )

    words = re.findall(r"[a-z]+", body)
    mission: MissionType | None = None
    if any(w in _COVER_WORDS for w in words):
        mission = MissionType.COVER  # flank guard, before the generic scan
    else:
        for w in words:
            if w in _SYNONYMS:
                mission = _SYNONYMS[w]
                break
    if mission is None:
        known = ", ".join(sorted({s.upper() for s in _SYNONYMS} | {"FLANK"}))
        msg = f"No mission keyword in {text!r}. Known keywords: {known}."
        raise OrderParseError(msg)

    # 'hold obj X' / 'hold alpha' means DEFEND the objective, not HOLD in place.
    obj_match = re.search(r"(?:obj(?:ective)?\s+)?(alpha|bravo|charlie|delta)", body)
    objective = obj_match.group(1).upper() if obj_match else None
    if mission is MissionType.HOLD and objective is not None:
        mission = MissionType.DEFEND

    # ADVANCE targets a control measure: 'advance to wp gold' / 'advance pl amber'
    control = None
    cm_match = _CONTROL_RE.search(body)
    if cm_match:
        control = cm_match.group(1).upper()
    if mission in NEEDS_CONTROL and control is None:
        msg = (
            f"Mission {mission.name} needs a control measure, e.g. "
            f"'{recipient}, ADVANCE TO WP GOLD' or '{recipient}, ADVANCE TO PL AMBER'."
        )
        raise OrderParseError(msg)
    if mission not in NEEDS_CONTROL:
        control = None

    if mission in NEEDS_OBJECTIVE and objective is None:
        msg = f"Mission {mission.name} needs an objective, e.g. '{recipient}, {mission.name} OBJ ALPHA'."
        raise OrderParseError(msg)
    if mission not in NEEDS_OBJECTIVE:
        objective = None

    return ParsedOrder(
        recipient_callsign=recipient,
        mission=mission,
        objective_name=objective,
        control_name=control,
    )
