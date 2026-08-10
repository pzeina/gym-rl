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

from cohort.core.missions import (
    HOLDS_GROUND,
    NEEDS_CONTROL,
    NEEDS_OBJECTIVE,
    Formation,
    MissionType,
)

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

#: Timing qualifiers (A5-2): "at t plus 5" / "at t+5" / "at my command".
_T_PLUS_RE = re.compile(r"\bat\s+t\s*(?:plus\s+|\+\s*)(\d+)\b")
_AT_MY_COMMAND_RE = re.compile(r"\bat\s+my\s+command\b")

#: Element stance (A5-3): "formation column|line|wedge" (French names too).
_FORMATION_RE = re.compile(r"\bformation\s+(column|colonne|line|ligne|wedge)\b")
_FORMATION_SYNONYMS: dict[str, Formation] = {
    "column": Formation.COLUMN,
    "colonne": Formation.COLUMN,
    "line": Formation.LINE,
    "ligne": Formation.LINE,
    "wedge": Formation.WEDGE,
}


def grid_ref(pos: tuple[int, int]) -> str:
    """Four-digit NATO-style grid reference, e.g. (14, 7) → 'GRID 1407'."""
    return f"GRID {int(pos[0]):02d}{int(pos[1]):02d}"


@dataclass(frozen=True)
class ParsedOrder:
    """Result of parsing a human order line."""

    recipient_callsign: str
    mission: MissionType | None         # None for stance-only orders (FORMATION)
    objective_name: str | None
    target_callsign: str | None = None  # supported unit (SUPPORT only)
    control_name: str | None = None     # control measure (ADVANCE only)
    formation: Formation | None = None  # element stance (A5-3, FORMATION orders)
    # timing qualifiers (A5-2): at most one of the two is set
    delay: int | None = None            # "... AT T PLUS <n>": effective n ticks from now
    at_my_command: bool = False         # "... AT MY COMMAND": pending until EXECUTE


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


def timing_phrase(delay: int | None = None, at_my_command: bool = False) -> str:
    """Timing-qualifier suffix (A5-2): '' | ' AT T PLUS n' | ' AT MY COMMAND'."""
    if at_my_command:
        return " AT MY COMMAND"
    if delay is not None:
        return f" AT T PLUS {int(delay)}"
    return ""


def format_order(
    issuer_cs: str,
    recipient_cs: str,
    mission: MissionType,
    target: str | None,
    *,
    delay: int | None = None,
    at_my_command: bool = False,
) -> str:
    """Radio form of an order: 'TL1, THIS IS SL1: SEIZE OBJ ALPHA. OUT.'

    With a timing qualifier: '... SEIZE OBJ ALPHA AT T PLUS 5. OUT.' or
    '... ADVANCE TO PL AMBER AT MY COMMAND. OUT.'
    """
    phrase = mission_phrase(mission, target) + timing_phrase(delay, at_my_command)
    return f"{recipient_cs}, THIS IS {issuer_cs}: {phrase}. OUT."


def format_execute(issuer_cs: str) -> str:
    """The issuer releases all its pending AT-MY-COMMAND orders (A5-2)."""
    return f"ALL STATIONS, THIS IS {issuer_cs}: EXECUTE. OUT."


def format_formation_order(issuer_cs: str, recipient_cs: str, formation: Formation) -> str:
    """Element stance order (A5-3): 'TL1, THIS IS SL1: FORMATION COLUMN. OUT.'"""
    return f"{recipient_cs}, THIS IS {issuer_cs}: FORMATION {formation.name}. OUT."


def format_sync_propose(proposer_cs: str, peer_css: list[str]) -> str:
    """Trinôme bound proposal, by VOICE (A5-4, manual pp. 14-15):
    'RFN2 RFN3, THIS IS RFN1: PREPARE TO BOUND ON MY SIGNAL. OUT.'"""
    peers = " ".join(peer_css) if peer_css else "ALL NEARBY"
    return f"{peers}, THIS IS {proposer_cs}: PREPARE TO BOUND ON MY SIGNAL. OUT."


def format_sync_go(proposer_cs: str) -> str:
    """The bound signal, by voice: 'RFN1: GO! OUT.' (the manual's EN AVANT !)"""
    return f"{proposer_cs}: GO! OUT."


def format_opord(
    recipient_cs: str,
    mission: MissionType,
    target: str | None,
    announced_assault_step: int | None = None,
    defend_horizon: int | None = None,
) -> str:
    """Initial operations order from higher HQ to the senior agent.

    Two optional time clauses follow the task statement, both stated as
    ABSOLUTE step references — the same clock ``max_steps`` is counted on, so
    a listener holding one line needs no further arithmetic and no H-hour:

    ``announced_assault_step`` (v1.10) is the enemy-arrival ESTIMATE for a
    scenario with a preparation period: the assault arrives somewhere in the
    scenario's band, so the wording is "EXPECT", not a timetable. It was
    spoken as "AT H PLUS <n>" until v1.18, which reads as *n steps after
    H-hour* while the value was always the absolute step — tolerable while it
    was the only time-bearing clause, and not once a second sits beside it.

    ``defend_horizon`` (v1.18, refs #30) is the hour the root is ordered to
    hold to. It is an ORDER, not intelligence, so it does not borrow EXPECT's
    hedge: "HOLD UNTIL STEP <n>" is tasking. It is spoken only for a mission
    in :data:`~cohort.core.missions.HOLDS_GROUND` — exactly the missions whose
    horizon the environment adjudicates — because an hour said on the net that
    nothing scores would be worse than silence.

    Both clauses sit after the task statement, where ``parse_order`` ignores
    them: the task the OPORD assigns is unchanged by when the enemy is due or
    when the defense ends. :func:`parse_opord` reads the whole line back,
    clauses included.
    """
    warning = (
        f" EXPECT ASSAULT AT STEP {announced_assault_step}."
        if announced_assault_step is not None
        else ""
    )
    horizon = (
        f" HOLD UNTIL STEP {defend_horizon}."
        if defend_horizon is not None and mission in HOLDS_GROUND
        else ""
    )
    return (
        f"{recipient_cs}, THIS IS HQ: OPORD — {mission_phrase(mission, target)}."
        f"{warning}{horizon} OUT."
    )


def format_ack(issuer_cs: str, recipient_cs: str) -> str:
    """Order acknowledgement (auto-emitted on receipt)."""
    return f"{issuer_cs}, THIS IS {recipient_cs}: WILCO. OUT."


def format_contact(leader_cs: str, sender_cs: str, n_hostiles: int, pos: tuple[int, int]) -> str:
    """Enemy sighting report (NATO contact report shape)."""
    return f"{leader_cs}, THIS IS {sender_cs}: CONTACT, {grid_ref(pos)}, {n_hostiles} x ENEMY. OVER."


#: Spoken forms of the SITREP posture clause (v1.10, issue #10).
COVER_PHRASE, OPEN_PHRASE = "IN COVER", "IN THE OPEN"

#: Machine-readable shape of a SITREP, for monitors that read the net.
_SITREP_RE = re.compile(
    r"SITREP,\s*GRID\s*(\d{2})(\d{2}),\s*HEALTH\s*(\d+)%,\s*AMMO\s*(\d+)"
    rf",\s*(?P<posture>{COVER_PHRASE}|{OPEN_PHRASE})",
    re.IGNORECASE,
)


def format_sitrep(
    leader_cs: str,
    sender_cs: str,
    health: int,
    ammo: int,
    pos: tuple[int, int],
    *,
    in_cover: bool,
) -> str:
    """Situation report, including the sender's own terrain posture.

    The posture clause is **self-reported**, like grid, health and ammo: it
    is what the soldier says about the ground it is on, not a readout of the
    ground itself. That keeps it radio-legitimate — per-step cover remains
    ground truth in ``env.oracle()`` and never enters the observable stream
    by any other route — while making the strongest known correlate of
    defend performance measurable from the net alone (issue #10, and the
    fight-disposition metrics of #11 that motivated the request).
    """
    posture = COVER_PHRASE if in_cover else OPEN_PHRASE
    return (
        f"{leader_cs}, THIS IS {sender_cs}: SITREP, {grid_ref(pos)}, "
        f"HEALTH {health}%, AMMO {ammo}, {posture}. OVER."
    )


def parse_sitrep(text: str) -> dict | None:
    """Read a SITREP back into its reported fields, or None if it is not one.

    Shipped so a monitor never has to hand-roll a regex over the transcript —
    the failure mode issue #10 reported for objective coordinates, in
    miniature. Returns ``{"grid", "health", "ammo", "in_cover"}``; inverse of
    :func:`format_sitrep` over exactly the fields it formats.
    """
    m = _SITREP_RE.search(text)
    if m is None:
        return None
    return {
        "grid": (int(m.group(1)), int(m.group(2))),
        "health": int(m.group(3)),
        "ammo": int(m.group(4)),
        "in_cover": m.group("posture").upper() == COVER_PHRASE,
    }


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


def format_endex(root_cs: str) -> str:
    """COMMAND ends the operation.

    The counterpart of ``format_done`` for a mission that cannot be reported
    complete. A DEFEND/DENY holder occupies its ground until relieved or
    re-tasked, so nobody below COMMAND is in a position to say the operation
    is over — the root reports the situation, and the order to end it comes
    back down the net.
    """
    return f"{root_cs}, THIS IS HQ: ENDEX. OUT."


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


#: Machine-readable shape of an OPORD (issue #12): the task statement is read
#: by ``_ORDER_RE`` below, these pick out what is specific to an OPORD — that
#: it IS one, plus each of its two time clauses if the line carries them.
_OPORD_RE = re.compile(r"\bOPORD\b", re.IGNORECASE)
#: The assault estimate. ``H PLUS`` is the pre-v1.18 spoken form of the same
#: absolute step, accepted on the way IN only: every committed
#: ``runs/*/eval_transcript.txt`` says it that way, and a monitor pointed at
#: that corpus must not silently lose the announcement. Nothing emits it.
_ANNOUNCED_ASSAULT_RE = re.compile(
    r"\bEXPECT\s+ASSAULT\s+AT\s+(?:STEP|H\s+PLUS)\s+(?P<step>\d+)", re.I
)
#: The ordered horizon (v1.18, refs #30).
_DEFEND_HORIZON_RE = re.compile(r"\bHOLD\s+UNTIL\s+STEP\s+(?P<step>\d+)", re.I)

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

    # timing qualifiers (A5-2), stripped from the body before the mission scan
    delay: int | None = None
    at_my_command = False
    t_match = _T_PLUS_RE.search(body)
    if t_match:
        delay = int(t_match.group(1))
        body = body[: t_match.start()] + body[t_match.end() :]
    elif _AT_MY_COMMAND_RE.search(body):
        at_my_command = True
        body = _AT_MY_COMMAND_RE.sub("", body)

    # Element stance (A5-3): 'formation column' — no mission payload at all.
    fm = _FORMATION_RE.search(body)
    if fm:
        return ParsedOrder(
            recipient_callsign=recipient,
            mission=None,
            objective_name=None,
            formation=_FORMATION_SYNONYMS[fm.group(1)],
        )

    # Unit-targeted SUPPORT: the order names a friendly callsign, not an
    # objective ('support tl1', 'appuyer tl1', 'cover for tl1').
    sup = _SUPPORT_RE.search(body)
    if sup:
        return ParsedOrder(
            recipient_callsign=recipient,
            mission=MissionType.SUPPORT,
            objective_name=None,
            target_callsign=sup.group(1).upper(),
            delay=delay,
            at_my_command=at_my_command,
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
        delay=delay,
        at_my_command=at_my_command,
    )


def parse_opord(text: str) -> dict | None:
    """Read an OPORD back into its fields, or None if the line is not one.

    Inverse of :func:`format_opord` over exactly the fields it formats:
    ``{"recipient", "mission", "objective", "announced_assault_step",
    "defend_horizon"}``. The last two are the time clauses — the step HQ says
    to expect the assault at, and the step it orders the position held until —
    each None when the OPORD does not carry it. Both are absolute steps, and
    the keys are the briefing's, so a monitor reading the net and a monitor
    reading the header get the same two names for the same two numbers.

    It exists because the announcement is otherwise lost at the boundary: it
    is said on the net and then nowhere else, so a monitor reading a corpus
    either hand-rolls a regex over the transcript or drops the only
    forward-looking content the net carries (issue #12; issue #10 in
    miniature, and the same remedy as :func:`parse_sitrep`).

    The announced step is an estimate: the assault actually arrives somewhere
    in the scenario's band, and that draw is ground truth the cohort is never
    told (``env.oracle()["actual_assault_step"]``). What the radio says is all
    a listener knows — which is the point, since under ``comm_model="range"``
    whether a given subordinate heard this single broadcast at all is a
    genuine per-listener question.
    """
    if not _OPORD_RE.search(text):
        return None
    try:
        order = parse_order(text)
    except OrderParseError:
        return None
    announced = _ANNOUNCED_ASSAULT_RE.search(text)
    horizon = _DEFEND_HORIZON_RE.search(text)
    return {
        "recipient": order.recipient_callsign,
        "mission": order.mission.name if order.mission is not None else None,
        "objective": order.objective_name,
        "announced_assault_step": int(announced.group("step")) if announced else None,
        "defend_horizon": int(horizon.group("step")) if horizon else None,
    }
