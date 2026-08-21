"""The gallery shows the net, and the net's two ends are the point.

A success rate cannot show that an OPORD came down, was acknowledged, produced
doctrine-valid subordinate tasks, survived its commander being killed, and ended
with HQ closing the operation. The transcript can, and that is the claim this
project actually makes — so the page that shows it has to be honest about which
parts of an episode it is dropping.

Pinned here: every baseline member gets a card; a long episode is elided in the
MIDDLE and never at the ends, because the OPORD cascade and the close are the
two things a reader came for; the elision says how much it hid; and the close
is colored as a close rather than as one more order.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import baseline, scenario_gallery

NET = """[t=  0] SL1, THIS IS HQ: OPORD — SEIZE OBJ ALPHA. OUT.
[t=  1] TL1, THIS IS SL1: ADVANCE TO PL AMBER. OUT.
[t=  1] SL1, THIS IS TL1: WILCO. OUT.
[t= 11] SL1, THIS IS TL1: CONTACT, GRID 2106, 1 x ENEMY. OVER.
[t= 87] ALL STATIONS: TL1 IS DOWN. OUT.
[t=112] HQ, THIS IS RFN1: SEIZE OBJ ALPHA — COMPLETE. OVER.
[t=112] RFN1, THIS IS HQ: ROGER, SEIZE OBJ ALPHA CONFIRMED. OUT.
[t=112] RFN1, THIS IS HQ: ENDEX. OUT.
"""


def test_each_act_on_the_net_is_colored_by_what_it_is(tmp_path):
    path = tmp_path / "eval_transcript.txt"
    path.write_text(NET)

    out = scenario_gallery._transcript(path)

    assert '<span class="opord">SL1, THIS IS HQ: OPORD' in out
    assert '<span class="order">TL1, THIS IS SL1: ADVANCE' in out
    assert '<span class="rep">SL1, THIS IS TL1: CONTACT' in out
    assert '<span class="cas">ALL STATIONS: TL1 IS DOWN' in out
    assert '<span class="close">RFN1, THIS IS HQ: ENDEX. OUT.</span>' in out
    # the report and the fact are both the close, not one order among many
    assert out.count('class="close"') == 3


def _one_of_every_act() -> dict[str, tuple[str, str]]:
    """One sample per shipped formatter: {formatter name: (text, css class)}.

    Built by CALLING ``cohort.core.language``, never by transcribing its prose
    into this file — transcribing is how the two misses below got in.
    """
    from cohort.core import language as lang
    from cohort.core.missions import Formation, MissionType

    return {
        "format_opord": (lang.format_opord("SL1", MissionType.SEIZE, "ALPHA"), "opord"),
        "format_order": (lang.format_order("SL1", "TL2", MissionType.OBSERVE, "ALPHA"), "order"),
        "format_formation_order": (
            lang.format_formation_order("SL1", "TL2", Formation.WEDGE), "order"),
        "format_ack": (lang.format_ack("SL1", "TL2"), "order"),
        "format_execute": (lang.format_execute("SL1"), "order"),
        "format_sync_propose": (lang.format_sync_propose("RFN1", ["RFN2"]), "order"),
        "format_sync_go": (lang.format_sync_go("RFN1"), "order"),
        "format_contact": (lang.format_contact("SL1", "TL2", 3, (12, 7)), "rep"),
        "format_acoustic_contact": (
            lang.format_acoustic_contact("SL1", "TL2", 0, 6, 1, 12), "rep"),
        "format_sitrep": (
            lang.format_sitrep("SL1", "TL2", 90, 24, (12, 7), in_cover=True), "rep"),
        "format_done": (lang.format_done("SL1", "TL2", MissionType.OBSERVE, "ALPHA"), "close"),
        "format_done_confirm": (
            lang.format_done_confirm("TL2", "SL1", MissionType.OBSERVE, "ALPHA"), "close"),
        "format_done_reject": (lang.format_done_reject("TL2", "SL1"), "close"),
        "format_endex": (lang.format_endex("SL1"), "close"),
        "format_casualty": (lang.format_casualty("TL1"), "cas"),
        # deliberate: SUPPORT ENDED is a report, but the fact it reports is a
        # death ("RFN3 IS DOWN"), and that is what a reader is looking for
        "format_support_end": (lang.format_support_end("SL1", "TL2", "RFN3"), "cas"),
        "format_trap": (lang.format_trap("RFN2", (9, 4)), "cas"),
        "format_taking_command": (lang.format_taking_command("RFN1", "TL1"), "cas"),
        "format_assuming_position": (lang.format_assuming_position("RFN2", "RFN1"), "cas"),
    }


def test_every_act_the_net_can_carry_is_colored_by_what_it_is():
    """Regression, refs #40: two acts fell through to the ORDER default.

    ``ASSUMING X'S POSITION`` is the backfill half of succession — the *other*
    act carried on ``MessageKind.TAKING_COMMAND``, told apart from the root
    appointment only by this prose — and the page's own standfirst promises to
    show "a rifleman took over a dead leader's fire team". It was colored as an
    order. So was the trap broadcast. Both were missed because ``ACTS`` was
    written from memory of the wording instead of from the formatters, and
    nothing checked the two against each other.

    Exhaustive on purpose: a new message kind, or a reworded formatter, fails
    here rather than quietly rendering as one more order.
    """
    from cohort.core import language as lang

    samples = _one_of_every_act()
    shipped = {name for name in dir(lang) if name.startswith("format_")}
    assert shipped == set(samples), (
        "a formatter in cohort/core/language.py has no expected color here: "
        f"{shipped ^ set(samples)}"
    )
    wrong = {
        name: (text, got, want)
        for name, (text, want) in samples.items()
        if (got := scenario_gallery._classify(text)) != want
    }
    assert not wrong, f"acts colored as something else: {wrong}"


def test_a_long_episode_is_elided_in_the_middle_never_at_the_ends(tmp_path):
    lines = NET.splitlines()
    filler = [f"[t={i:3d}] SL1, THIS IS TL2: SITREP, GRID 2029, HEALTH 100%. OVER."
              for i in range(200)]
    path = tmp_path / "eval_transcript.txt"
    path.write_text("\n".join(lines[:5] + filler + lines[5:]))

    out = scenario_gallery._transcript(path)

    assert "OPORD" in out, "the cascade that starts the episode was dropped"
    assert "ENDEX" in out, "the close was dropped — the one line the page is for"
    assert "more transmissions" in out
    assert "178 more transmissions" in out, "the elision must say how much it hid"


def test_an_episode_shorter_than_the_window_is_shown_whole(tmp_path):
    path = tmp_path / "eval_transcript.txt"
    path.write_text(NET)

    out = scenario_gallery._transcript(path)

    assert "more transmissions" not in out
    assert out.count("<span class=\"t\">") == len(NET.strip().splitlines())


def test_a_run_with_no_transcript_says_so_rather_than_rendering_an_empty_box(tmp_path):
    assert scenario_gallery._transcript(tmp_path / "absent.txt") == ""


def test_every_baseline_member_gets_a_card():
    page = scenario_gallery.render([])
    for scenario in baseline.DOCTRINE_SCENARIOS:
        assert f"<h3>{scenario}</h3>" in page, f"{scenario} has no card"


def test_the_page_explains_how_to_read_the_close():
    """The distinction the gallery exists to make legible."""
    page = scenario_gallery.render([])
    assert "won <i>without saying so</i>" in page
    assert "closed_on_root_report_rate" in page


@pytest.mark.parametrize("scenario", baseline.DOCTRINE_SCENARIOS)
def test_each_card_states_the_scenario_s_own_clock_and_root_mission(scenario):
    """Facts come from the ScenarioSpec, so a config change cannot leave the
    gallery describing a scenario that no longer exists."""
    from cohort.config import get_scenario

    spec = get_scenario(scenario)
    card = scenario_gallery._scenario(scenario, "whatever_v1", {})

    assert spec.root_mission.name in card
    assert f"<b>{spec.max_steps}</b> steps" in card
    assert Path(spec.description).name[:20] in card or spec.description[:20] in card
