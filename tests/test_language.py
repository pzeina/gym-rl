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
        text = format_order("CDG1", "CAP1", mission, obj)
        parsed = parse_order(text)
        assert parsed.recipient_callsign == "CAP1"
        assert parsed.mission is mission
        assert parsed.objective_name == obj


@pytest.mark.parametrize(
    ("text", "recipient", "mission", "objective"),
    [
        ("CAP1, seize obj alpha", "CAP1", MissionType.SEIZE, "ALPHA"),
        ("cap2: take objective bravo", "CAP2", MissionType.SEIZE, "BRAVO"),
        ("SLD2, rally on me", "SLD2", MissionType.REGROUP, None),
        ("SLD1, hold position", "SLD1", MissionType.HOLD, None),
        ("SLD1, halt", "SLD1", MissionType.HOLD, None),
        ("CAP1, cover obj charlie. out.", "CAP1", MissionType.OVERWATCH, "CHARLIE"),
        ("CAP1, attack obj delta", "CAP1", MissionType.ENGAGE, "DELTA"),
        ("SLD3, scout obj alpha", "SLD3", MissionType.RECON, "ALPHA"),
        # 'hold OBJ X' means defend the objective, not hold in place
        ("CAP1, hold obj alpha", "CAP1", MissionType.DEFEND, "ALPHA"),
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
        parse_order("CAP1, dance the tango")


def test_objective_required():
    with pytest.raises(OrderParseError, match="needs an objective"):
        parse_order("CAP1, seize")


def test_mission_phrase_forms():
    assert mission_phrase(MissionType.SEIZE, "ALPHA") == "SEIZE OBJ ALPHA"
    assert mission_phrase(MissionType.REGROUP, None) == "REGROUP ON ME"
    assert mission_phrase(MissionType.HOLD, None) == "HOLD POSITION"
