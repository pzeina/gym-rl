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


@pytest.mark.skip(reason="needs cohort/core/language.py, frozen while the "
                         "baseline retrain campaign is in flight — train.py imports "
                         "the tree that exists when a job starts, so an edit under "
                         "cohort/ today would train the later fleet members against a "
                         "different environment than the earlier ones, and it would "
                         "date every best/final pair in the fleet as mixed-era under "
                         "publish_audit.era_gap. Patch is written out in ROADMAP.md's "
                         "2026-08-11 #40 entry; unskip when it lands.")
def test_a_succession_message_says_which_act_it_performs():
    """The durable half of #40: two acts share ``MessageKind.TAKING_COMMAND``.

    The root appointment ("I AM ASSUMING COMMAND", the command passes) and the
    backfill of the slot the successor just vacated ("ASSUMING X'S POSITION",
    the command does not) are the same kind and the same pair of callsigns.
    Only the prose tells them apart, so every consumer writes its own matcher:
    ``probe._TAKING_RE``/``_FILLING_RE``, ``metrics._succession``'s inline
    marker string, ``scenario_gallery.ACTS`` — which got it wrong — and any
    external monitor. #40 asks for a structured payload key; this repo's net is
    text-only by owner decision (``test_orders_flow.py::
    test_radio_messages_are_text_only``), so the answer that fits is the
    formatter's **inverse**, shipped beside it, which is the round-trip
    contract every other act on this net already has.
    """
    from cohort.core.language import (
        format_assuming_position,
        format_taking_command,
        parse_succession,
    )

    appointed = parse_succession(format_taking_command("RFN1", "TL1"))
    assert (appointed.successor, appointed.replaced) == ("RFN1", "TL1")
    assert appointed.assumes_command is True, "the root pointer moves to RFN1"

    backfill = parse_succession(format_assuming_position("RFN2", "RFN1"))
    assert (backfill.successor, backfill.replaced) == ("RFN2", "RFN1")
    assert backfill.assumes_command is False, "a slot was filled; no command passed"

    # the plain casualty broadcast is not a succession at all
    assert parse_succession("ALL STATIONS: TL1 IS DOWN. OUT.") is None


# ---------------------------------------------------------------------- #
# SITREP posture (issue #10)
# ---------------------------------------------------------------------- #


def test_sitrep_reports_its_own_terrain_posture():
    from cohort.core.language import format_sitrep

    assert (
        format_sitrep("TL1", "RFN1", 66, 24, (9, 12), in_cover=True)
        == "TL1, THIS IS RFN1: SITREP, GRID 0912, HEALTH 66%, AMMO 24, IN COVER. OVER."
    )
    assert format_sitrep("TL1", "RFN1", 66, 24, (9, 12), in_cover=False).endswith(
        "AMMO 24, IN THE OPEN. OVER."
    )


@pytest.mark.parametrize("in_cover", [True, False])
def test_sitrep_round_trips_through_its_parser(in_cover):
    """format_sitrep / parse_sitrep are inverses over the fields formatted."""
    from cohort.core.language import format_sitrep, parse_sitrep

    text = format_sitrep("SL1", "TL2", 33, 7, (18, 4), in_cover=in_cover)
    assert parse_sitrep(text) == {
        "grid": (18, 4),
        "health": 33,
        "ammo": 7,
        "in_cover": in_cover,
    }


def test_parse_sitrep_ignores_other_traffic():
    from cohort.core.language import format_contact, format_order, parse_sitrep

    assert parse_sitrep(format_contact("TL1", "RFN1", 2, (5, 5))) is None
    assert parse_sitrep(format_order("SL1", "TL1", MissionType.DEFEND, "ALPHA")) is None
