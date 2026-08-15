"""A design that cannot reject must say so before the runs, not after.

The `rdb3_seeds` campaign is the case this exists for (assurance #56): six
3M-step runs, the right question, the right decision rule, and **one of its 64
possible outcomes** significant — the one requiring the comparison arm's last
seed to go mute, which is the branch the evidence argues against. The arithmetic
took four minutes and it bounds six runs.

What is pinned here is that arithmetic, against values derived from the exact
tests rather than from this module's own output:

* the campaign's ceilings and its single rejecting outcome;
* the asymmetry between the two readings — the unpaired one is capable at these
  sizes only because it throws the pairing away, and the paired one is not
  capable at five pairs no matter what the runs produce;
* concordant pairs are not evidence, so a comparison arm that already reports at
  a seed *removes* that seed from a paired design's power;
* an outcome is a LABELLING of the pending cells, not a count table — a campaign
  lands as five labels, and counting count-tables would understate the space.
"""

from __future__ import annotations

import pytest

from scripts import design_power as dp
from scripts.design_power import Pair, parse_pair, power, sizing


def _pairs(specs):
    return [parse_pair(s) for s in specs]


def test_the_live_campaign_can_reject_on_one_of_its_sixty_four_outcomes():
    result = power(_pairs(dp.CAMPAIGN))

    assert result["outcomes"] == 64, "five pending rdb3 cells and one pending rdb1 cell"
    assert result["unpaired"]["ceiling"] == pytest.approx(0.047619, rel=1e-4)
    assert result["unpaired"]["rejecting"] == 1
    assert result["unpaired"]["at"][0][0] == "first arm 5/5, second arm 1/5", \
        "and it needs the pending comparison cell to come back MUTE"


def test_the_campaigns_paired_reading_cannot_reject_at_all():
    """Conditioned on the arm as measured it is 0.125: seed 14 already reports.

    A concordant pair carries no information, so the seed the comparison arm
    already reports at is not evidence about direction however the new arm lands.
    """
    result = power(_pairs(dp.CAMPAIGN))

    assert result["paired"]["ceiling"] == pytest.approx(0.125)
    assert result["paired"]["rejecting"] == 0


def test_five_free_pairs_are_0625_paired_and_capable_unpaired():
    """The same five seeds with NOTHING measured — the ceiling is still 0.0625.

    This is the sizing fact: no five-pair matched design can reject at 0.05,
    including under perfect separation. The unpaired reading gets to 0.0079 on
    the same five seeds only by discarding the pairing.
    """
    free = _pairs([f"{seed}:?:?" for seed in range(12, 17)])

    result = power(free)

    assert result["paired"]["ceiling"] == pytest.approx(0.0625)
    assert result["paired"]["rejecting"] == 0
    assert result["unpaired"]["ceiling"] == pytest.approx(0.007937, rel=1e-4)


def test_a_concordant_pair_does_not_buy_paired_power():
    """Adding a seed both arms report at leaves the paired ceiling exactly where it was."""
    four = _pairs([f"{seed}:?:mute" for seed in range(12, 16)])
    plus_concordant = [*four, Pair("16", True, True)]

    assert power(four)["paired"]["ceiling"] == pytest.approx(0.125)
    assert power(plus_concordant)["paired"]["ceiling"] == pytest.approx(0.125)


def test_a_sixth_discordant_pair_does():
    six = _pairs([f"{seed}:?:mute" for seed in range(12, 18)])

    assert power(six)["paired"]["ceiling"] == pytest.approx(0.03125)
    assert power(six)["paired"]["rejecting"] > 0


@pytest.mark.parametrize(("seeds", "new_runs", "unpaired", "paired"), [
    # #56's sizing table, best case, comparison arm held at its measured 1-of-4
    (5, 6, 0.047619, 0.1250),
    (6, 8, 0.060606, 0.1250),   # worse than 5: the assumed count rounds 1 -> 2
    (7, 10, 0.020979, 0.0625),
    (8, 12, 0.006993, 0.03125),  # where both readings become capable
])
def test_sizing_reproduces_the_published_table(seeds, new_runs, unpaired, paired):
    row = sizing([seeds], measured=(1, 4))[0]

    assert row["new_runs"] == new_runs, "only the cells that do not exist are new runs"
    assert row["unpaired"] == pytest.approx(unpaired, rel=1e-4)
    assert row["paired"] == pytest.approx(paired, rel=1e-4)


def test_an_outcome_is_a_labelling_and_not_a_count_table():
    """Two pending cells in one arm are four outcomes, not three counts."""
    assert power(_pairs(["12:?:mute", "13:?:mute"]))["outcomes"] == 4


def test_a_design_with_nothing_pending_is_one_outcome():
    """Reading a finished design through the same instrument is legal and exact."""
    finished = _pairs(["12:reporting:mute", "13:reporting:mute", "14:reporting:mute",
                       "15:reporting:mute", "16:reporting:reporting"])

    result = power(finished)

    assert result["outcomes"] == 1
    assert result["unpaired"]["ceiling"] == pytest.approx(0.047619, rel=1e-4)
    assert result["paired"]["ceiling"] == pytest.approx(0.125)


def test_the_read_out_says_which_test_may_not_be_named(capsys):
    dp.report(_pairs(dp.CAMPAIGN), 0.05, None, (1, 4))
    out = capsys.readouterr().out

    assert "CANNOT REJECT — do not name this test in the read-out" in out
    assert "rejects on 1 of 64 outcomes" in out


def test_an_unreadable_cell_is_refused_rather_than_guessed():
    with pytest.raises(SystemExit):
        parse_pair("12:maybe:mute")
    with pytest.raises(SystemExit):
        parse_pair("12:mute")


def test_an_enumeration_too_big_to_be_the_right_instrument_is_refused():
    with pytest.raises(SystemExit):
        power(_pairs([f"{seed}:?:?" for seed in range(30)]))
