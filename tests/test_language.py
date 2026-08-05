"""Command-language formatting and parsing round-trips."""

import pytest

from cohort.core.language import (
    OrderParseError,
    format_order,
    mission_phrase,
    parse_order,
)
from cohort.core.missions import NEEDS_OBJECTIVE, MissionType


def test_round_trip_all_missions():
    """Everything an agent can say as an order must parse back identically."""
    for mission in MissionType:
        obj = "BRAVO" if mission in NEEDS_OBJECTIVE else None
        text = format_order("SL1", "TL1", mission, obj)
        parsed = parse_order(text)
        assert parsed.recipient_callsign == "TL1"
        assert parsed.mission is mission
        assert parsed.objective_name == obj


@pytest.mark.parametrize(
    ("text", "recipient", "mission", "objective"),
    [
        ("TL1, seize obj alpha", "TL1", MissionType.SEIZE, "ALPHA"),
        ("tl2: take objective bravo", "TL2", MissionType.SEIZE, "BRAVO"),
        ("RFN2, rally on me", "RFN2", MissionType.RALLY, None),
        ("RFN1, hold position", "RFN1", MissionType.HOLD, None),
        ("RFN1, halt", "RFN1", MissionType.HOLD, None),
        ("TL1, cover obj charlie. out.", "TL1", MissionType.OVERWATCH, "CHARLIE"),
        ("TL1, attack obj delta", "TL1", MissionType.CLEAR, "DELTA"),
        ("RFN3, scout obj alpha", "RFN3", MissionType.RECON, "ALPHA"),
        # 'hold OBJ X' means defend the objective, not hold in place
        ("TL1, hold obj alpha", "TL1", MissionType.DEFEND, "ALPHA"),
    ],
)
def test_parse_variants(text, recipient, mission, objective):
    parsed = parse_order(text)
    assert parsed.recipient_callsign == recipient
    assert parsed.mission is mission
    assert parsed.objective_name == objective


def test_parse_rejects_garbage():
    with pytest.raises(OrderParseError):
        parse_order("do something useful")
    with pytest.raises(OrderParseError):
        parse_order("TL1, dance the tango")


def test_objective_required():
    with pytest.raises(OrderParseError, match="needs an objective"):
        parse_order("TL1, seize")


def test_mission_phrase_forms():
    assert mission_phrase(MissionType.SEIZE, "ALPHA") == "SEIZE OBJ ALPHA"
    assert mission_phrase(MissionType.RALLY, None) == "RALLY ON ME"
    assert mission_phrase(MissionType.HOLD, None) == "HOLD POSITION"


def test_succession_formatters():
    """Both succession shapes live in language.py, next to each other."""
    from cohort.core.language import format_assuming_position, format_taking_command

    assert (
        format_taking_command("RFN1", "TL1")
        == "ALL STATIONS, THIS IS RFN1: TL1 IS DOWN. I AM ASSUMING COMMAND. OUT."
    )
    assert (
        format_assuming_position("RFN2", "RFN1")
        == "ALL STATIONS, THIS IS RFN2: ASSUMING RFN1'S POSITION. OUT."
    )
