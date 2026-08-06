"""Command-language formatting and parsing round-trips."""

import pytest

from cohort.core.language import (
    OrderParseError,
    format_order,
    mission_phrase,
    parse_order,
)
from cohort.core.missions import NEEDS_CONTROL, NEEDS_OBJECTIVE, MissionType


def test_round_trip_all_missions():
    """Everything an agent can say as an order must parse back identically."""
    for mission in MissionType:
        if mission is MissionType.SUPPORT:
            continue  # unit-targeted: covered by its own round-trip test
        if mission in NEEDS_CONTROL:
            continue  # control-targeted ADVANCE: covered by its own test
        obj = "BRAVO" if mission in NEEDS_OBJECTIVE else None
        text = format_order("SL1", "TL1", mission, obj)
        parsed = parse_order(text)
        assert parsed.recipient_callsign == "TL1"
        assert parsed.mission is mission, f"{mission}: {text!r} parsed as {parsed.mission}"
        assert parsed.objective_name == obj


def test_round_trip_advance_control_measures():
    """ADVANCE round-trips for every control-measure name, WP and PL alike."""
    from cohort.core.language import CONTROL_NAMES, control_phrase

    for name in CONTROL_NAMES:
        text = format_order("SL1", "TL1", MissionType.ADVANCE, name)
        assert f"ADVANCE TO {control_phrase(name)}" in text
        parsed = parse_order(text)
        assert parsed.recipient_callsign == "TL1"
        assert parsed.mission is MissionType.ADVANCE
        assert parsed.control_name == name
        assert parsed.objective_name is None


def test_advance_parse_variants():
    parsed = parse_order("TL1, advance to wp gold")
    assert parsed.mission is MissionType.ADVANCE
    assert parsed.control_name == "GOLD"
    parsed = parse_order("tl2: advance pl amber. out.")
    assert parsed.mission is MissionType.ADVANCE
    assert parsed.control_name == "AMBER"
    with pytest.raises(OrderParseError, match="needs a control measure"):
        parse_order("TL1, advance")


def test_round_trip_support_unit_target():
    """SUPPORT names a friendly element: 'TL2, THIS IS SL1: SUPPORT TL1. OUT.'"""
    text = format_order("SL1", "TL2", MissionType.SUPPORT, "TL1")
    assert text == "TL2, THIS IS SL1: SUPPORT TL1. OUT."
    parsed = parse_order(text)
    assert parsed.recipient_callsign == "TL2"
    assert parsed.mission is MissionType.SUPPORT
    assert parsed.objective_name is None
    assert parsed.target_callsign == "TL1"


@pytest.mark.parametrize(
    ("text", "recipient", "mission", "objective"),
    [
        ("TL1, seize obj alpha", "TL1", MissionType.SEIZE, "ALPHA"),
        ("tl2: take objective bravo", "TL2", MissionType.SEIZE, "BRAVO"),
        ("RFN2, rally on me", "RFN2", MissionType.RALLY, None),
        ("RFN1, hold position", "RFN1", MissionType.HOLD, None),
        ("RFN1, halt", "RFN1", MissionType.HOLD, None),
        # backwards-friendly: the retired OVERWATCH phrases now mean OBSERVE
        ("TL1, cover obj charlie. out.", "TL1", MissionType.OBSERVE, "CHARLIE"),
        ("TL1, support obj bravo", "TL1", MissionType.OBSERVE, "BRAVO"),
        ("TL1, overwatch obj bravo", "TL1", MissionType.OBSERVE, "BRAVO"),
        ("TL1, observe obj alpha", "TL1", MissionType.OBSERVE, "ALPHA"),
        ("TL1, surveiller obj alpha", "TL1", MissionType.OBSERVE, "ALPHA"),
        ("TL1, attack obj delta", "TL1", MissionType.CLEAR, "DELTA"),
        ("RFN3, scout obj alpha", "RFN3", MissionType.RECON, "ALPHA"),
        ("TL2, screen obj bravo", "TL2", MissionType.SCREEN, "BRAVO"),
        ("TL2, eclairer obj bravo", "TL2", MissionType.SCREEN, "BRAVO"),
        ("SL1, deny obj alpha", "SL1", MissionType.DENY, "ALPHA"),
        ("SL1, interdire obj alpha", "SL1", MissionType.DENY, "ALPHA"),
        # COVER (flank guard) keeps its own canonical keyword
        ("TL1, cover flank obj bravo", "TL1", MissionType.COVER, "BRAVO"),
        ("TL1, couvrir obj bravo", "TL1", MissionType.COVER, "BRAVO"),
        # 'hold OBJ X' means defend the objective, not hold in place
        ("TL1, hold obj alpha", "TL1", MissionType.DEFEND, "ALPHA"),
    ],
)
def test_parse_variants(text, recipient, mission, objective):
    parsed = parse_order(text)
    assert parsed.recipient_callsign == recipient
    assert parsed.mission is mission
    assert parsed.objective_name == objective


@pytest.mark.parametrize(
    ("text", "recipient", "target"),
    [
        ("TL2, support TL1", "TL2", "TL1"),
        ("tl2: appuyer tl1", "TL2", "TL1"),
        ("TL2, cover for TL1", "TL2", "TL1"),
        ("TL2, cover TL1. out.", "TL2", "TL1"),
        ("RFN2, support RFN1", "RFN2", "RFN1"),
    ],
)
def test_parse_support_unit_targets(text, recipient, target):
    parsed = parse_order(text)
    assert parsed.mission is MissionType.SUPPORT
    assert parsed.recipient_callsign == recipient
    assert parsed.target_callsign == target
    assert parsed.objective_name is None


def test_parse_rejects_garbage():
    with pytest.raises(OrderParseError):
        parse_order("do something useful")
    with pytest.raises(OrderParseError):
        parse_order("TL1, dance the tango")


def test_objective_required():
    with pytest.raises(OrderParseError, match="needs an objective"):
        parse_order("TL1, seize")
    with pytest.raises(OrderParseError, match="needs an objective"):
        parse_order("TL1, screen")


def test_mission_phrase_forms():
    assert mission_phrase(MissionType.SEIZE, "ALPHA") == "SEIZE OBJ ALPHA"
    assert mission_phrase(MissionType.SCREEN, "BRAVO") == "SCREEN OBJ BRAVO"
    assert mission_phrase(MissionType.OBSERVE, "BRAVO") == "OBSERVE OBJ BRAVO"
    assert mission_phrase(MissionType.SUPPORT, "TL1") == "SUPPORT TL1"
    assert mission_phrase(MissionType.COVER, "BRAVO") == "COVER FLANK OBJ BRAVO"
    assert mission_phrase(MissionType.DENY, "ALPHA") == "DENY OBJ ALPHA"
    assert mission_phrase(MissionType.RALLY, None) == "RALLY ON ME"
    assert mission_phrase(MissionType.HOLD, None) == "HOLD POSITION"


def test_support_end_formatter():
    from cohort.core.language import format_support_end

    assert (
        format_support_end("SL1", "TL2", "TL1")
        == "SL1, THIS IS TL2: SUPPORT ENDED, TL1 IS DOWN. STANDING BY. OVER."
    )


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
