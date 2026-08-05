"""The command language: format agent radio traffic, parse human orders.

Voice procedure follows NATO conventions (ACP 125 prowords: THIS IS, OVER,
OUT, WILCO, ALL STATIONS; four-digit GRID references; phonetic objective
names). Formatting and parsing are inverses over the order subset, so a human
can type exactly what agents say to each other:

    TL1, SEIZE OBJ ALPHA
    RFN2, RALLY ON ME
    RFN1, HOLD POSITION
    TL2, OVERWATCH OBJ BRAVO

Mission keywords accept common synonyms (take/capture → SEIZE, cover/support
→ OVERWATCH, destroy/attack/engage → CLEAR, regroup/assemble → RALLY,
scout → RECON...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cohort.core.missions import NEEDS_OBJECTIVE, MissionType

#: Objective slot names, addressed as "OBJ ALPHA" etc. (NATO phonetic).
OBJECTIVE_NAMES: tuple[str, ...] = ("ALPHA", "BRAVO", "CHARLIE", "DELTA")

_SYNONYMS: dict[str, MissionType] = {
    "recon": MissionType.RECON,
    "reconnoiter": MissionType.RECON,
    "reconnoitre": MissionType.RECON,
    "scout": MissionType.RECON,
    "observe": MissionType.RECON,
    "seize": MissionType.SEIZE,
    "take": MissionType.SEIZE,
    "capture": MissionType.SEIZE,
    "assault": MissionType.SEIZE,
    "secure": MissionType.SEIZE,
    "defend": MissionType.DEFEND,
    "guard": MissionType.DEFEND,
    "retain": MissionType.DEFEND,
    "overwatch": MissionType.OVERWATCH,
    "cover": MissionType.OVERWATCH,
    "support": MissionType.OVERWATCH,
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
}


def grid_ref(pos: tuple[int, int]) -> str:
    """Four-digit NATO-style grid reference, e.g. (14, 7) → 'GRID 1407'."""
    return f"GRID {int(pos[0]):02d}{int(pos[1]):02d}"


@dataclass(frozen=True)
class ParsedOrder:
    """Result of parsing a human order line."""

    recipient_callsign: str
    mission: MissionType
    objective_name: str | None


class OrderParseError(ValueError):
    """Raised when a human order line cannot be parsed; message explains why."""


def mission_phrase(mission: MissionType, objective_name: str | None) -> str:
    """Canonical spoken form of a tasking, e.g. 'SEIZE OBJ ALPHA'."""
    name = mission.name
    if mission in NEEDS_OBJECTIVE:
        return f"{name} OBJ {objective_name}"
    if mission is MissionType.RALLY:
        return f"{name} ON ME"
    return f"{name} POSITION"  # HOLD


def format_order(issuer_cs: str, recipient_cs: str, mission: MissionType, objective_name: str | None) -> str:
    """Radio form of an order: 'TL1, THIS IS SL1: SEIZE OBJ ALPHA. OUT.'"""
    return f"{recipient_cs}, THIS IS {issuer_cs}: {mission_phrase(mission, objective_name)}. OUT."


def format_opord(recipient_cs: str, mission: MissionType, objective_name: str | None) -> str:
    """Initial operations order from higher HQ to the senior agent."""
    return f"{recipient_cs}, THIS IS HQ: OPORD — {mission_phrase(mission, objective_name)}. OUT."


def format_ack(issuer_cs: str, recipient_cs: str) -> str:
    """Order acknowledgement (auto-emitted on receipt)."""
    return f"{issuer_cs}, THIS IS {recipient_cs}: WILCO. OUT."


def format_contact(leader_cs: str, sender_cs: str, n_hostiles: int, pos: tuple[int, int]) -> str:
    """Enemy sighting report (NATO contact report shape)."""
    return f"{leader_cs}, THIS IS {sender_cs}: CONTACT, {grid_ref(pos)}, {n_hostiles} x ENEMY. OVER."


def format_sitrep(leader_cs: str, sender_cs: str, health: int, ammo: int, pos: tuple[int, int]) -> str:
    """Situation report."""
    return f"{leader_cs}, THIS IS {sender_cs}: SITREP, {grid_ref(pos)}, HEALTH {health}%, AMMO {ammo}. OVER."


def format_done(leader_cs: str, sender_cs: str, mission: MissionType, objective_name: str | None) -> str:
    """Mission-complete report."""
    return f"{leader_cs}, THIS IS {sender_cs}: {mission_phrase(mission, objective_name)} — COMPLETE. OVER."


def format_casualty(callsign: str) -> str:
    """Broadcast when an agent goes down."""
    return f"ALL STATIONS: {callsign} IS DOWN. OUT."


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
    """Parse a human order line into (recipient, mission, objective).

    Accepts e.g. 'TL1, seize obj alpha', 'rfn2: rally on me',
    'RFN1, hold position', 'TL2, cover obj bravo. out.'
    """
    m = _ORDER_RE.match(text)
    if not m:
        msg = f"Cannot parse order: {text!r}. Expected '<CALLSIGN>, <MISSION> [OBJ <NAME>]'."
        raise OrderParseError(msg)
    recipient = m.group("recipient").upper()
    body = m.group("body").strip().lower().rstrip(".")

    words = re.findall(r"[a-z]+", body)
    mission: MissionType | None = None
    for w in words:
        if w in _SYNONYMS:
            mission = _SYNONYMS[w]
            break
    if mission is None:
        known = ", ".join(sorted({s.upper() for s in _SYNONYMS}))
        msg = f"No mission keyword in {text!r}. Known keywords: {known}."
        raise OrderParseError(msg)

    # 'hold obj X' / 'hold alpha' means DEFEND the objective, not HOLD in place.
    obj_match = re.search(r"(?:obj(?:ective)?\s+)?(alpha|bravo|charlie|delta)", body)
    objective = obj_match.group(1).upper() if obj_match else None
    if mission is MissionType.HOLD and objective is not None:
        mission = MissionType.DEFEND

    if mission in NEEDS_OBJECTIVE and objective is None:
        msg = f"Mission {mission.name} needs an objective, e.g. '{recipient}, {mission.name} OBJ ALPHA'."
        raise OrderParseError(msg)
    if mission not in NEEDS_OBJECTIVE:
        objective = None

    return ParsedOrder(recipient_callsign=recipient, mission=mission, objective_name=objective)
