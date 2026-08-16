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


# --- after the runs: could the table that landed have rejected? (assurance #59)
#
# The eight-pair reading of the `patrol_brique` price campaign, as an external
# review filed it: 3 of 8 reporting either side, discordant 2/2. Both tests
# return p = 1.0000 and the two readings mean OPPOSITE things by it, which is
# the whole reason a realized ceiling exists. Labels are the pre-registered ones
# (`scripts/reporting_channel.py`), transcribed here only so the arithmetic is
# pinned; the same table comes off the artifacts with
# `reporting_channel.py --scenario patrol_brique --arms 1.0 3.0`.
EIGHT_PAIRS = ("12:mute:reporting", "13:mute:mute", "14:reporting:mute", "15:mute:reporting",
               "16:mute:mute", "17:mute:mute", "18:reporting:reporting", "19:reporting:mute")

#: The same design under the pre-registration's own rule — a run whose two
#: checkpoints disagree is SPLIT and dropped — which removes seeds 13, 16 and 17.
FIVE_PAIRS = ("12:mute:reporting", "14:reporting:mute", "15:mute:reporting",
              "18:reporting:reporting", "19:reporting:mute")


def test_the_eight_pair_table_is_a_null_on_one_reading_and_no_reading_on_the_other():
    """p = 1.0000 twice, and it means two different things (assurance #59).

    Fisher conditions on the margin: 6 reporters over 16 runs could have fallen
    8/8 vs 0/8, so p = 0.0070 was attainable and the observed 1.0000 IS evidence
    of absence. McNemar conditions on the discordant count: 4 discordant pairs
    cannot go below 0.1250 however they point, so the campaign's PRIMARY reading
    measured nothing at all — the pairing spent half its seeds on concordance.
    """
    result = dp.realized(_pairs(EIGHT_PAIRS))

    assert (result["first"], result["second"]) == (3, 3)
    assert (result["one_way"], result["other_way"]) == (2, 2)
    assert result["discordant"] == 4, "and the other four pairs are outside McNemar's sample"

    assert result["unpaired"]["p"] == pytest.approx(1.0)
    assert result["unpaired"]["ceiling"] == pytest.approx(0.006993, rel=1e-4)
    assert result["unpaired"]["capable"] is True

    assert result["paired"]["p"] == pytest.approx(1.0)
    assert result["paired"]["ceiling"] == pytest.approx(0.125)
    assert result["paired"]["capable"] is False


def test_dropping_the_split_cells_keeps_the_null_and_nearly_costs_the_power():
    """The pre-registered labelling rule gives 5 pairs, not 8 — same p, thinner margin."""
    result = dp.realized(_pairs(FIVE_PAIRS))

    assert (result["first"], result["second"]) == (3, 3)
    assert result["unpaired"]["p"] == pytest.approx(1.0)
    assert result["paired"]["p"] == pytest.approx(1.0), "the finding survives the stricter rule"
    assert result["unpaired"]["ceiling"] == pytest.approx(0.047619, rel=1e-4), \
        "capable by 0.002, where the eight-pair margin had 0.0070"
    assert result["paired"]["ceiling"] == pytest.approx(0.125)


def test_four_discordant_pairs_cannot_reject_however_they_point():
    """The realized paired ceiling is a fact about the count, not about the split."""
    for one_way, other_way in ((4, 0), (3, 1), (2, 2)):
        table = ([Pair(str(i), True, False) for i in range(one_way)]
                 + [Pair(str(i), False, True) for i in range(other_way)]
                 + [Pair("c", True, True)] * 4)

        assert dp.realized_paired_ceiling(table) == pytest.approx(0.125)


def test_concordant_pairs_do_not_move_the_realized_paired_ceiling():
    """Adding seeds both arms agree on grows the design and not the evidence."""
    four = [Pair(str(i), True, False) for i in range(4)]

    assert dp.realized_paired_ceiling(four) == pytest.approx(0.125)
    assert dp.realized_paired_ceiling([*four, Pair("x", True, True), Pair("y", False, False)]) \
        == pytest.approx(0.125), "eight pairs, still a four-pair paired reading"


def test_six_discordant_pairs_buy_a_paired_reading():
    six = [Pair(str(i), True, False) for i in range(6)]

    assert dp.realized_paired_ceiling(six) == pytest.approx(0.03125)
    assert dp.realized(six)["paired"]["capable"] is True


def test_the_realized_unpaired_ceiling_is_conditioned_on_the_observed_margin():
    """Fisher's best case is the margin's most separated split, not perfect separation.

    Restricted to the campaign's five planned seeds the margin is 3 reporters
    over 10 runs, and the most separated table THAT allows reaches only 0.1667 —
    so 'both tests p = 1.0000 on seeds 12-16' is two non-readings, not a
    robustness check.
    """
    planned = _pairs([s for s in EIGHT_PAIRS if int(s.split(":")[0]) <= 16])

    result = dp.realized(planned)

    assert (result["first"], result["second"]) == (1, 2)
    assert result["unpaired"]["ceiling"] == pytest.approx(0.166667, rel=1e-4)
    assert result["unpaired"]["capable"] is False
    assert result["paired"]["capable"] is False


def test_a_finished_designs_power_ceiling_is_just_its_observed_p():
    """The trap the realized reading exists for.

    `power` minimises over the outcomes a design can still produce, so once every
    cell is labelled it minimises over one outcome and reports the observed p
    under the name 'ceiling'. Read as a ceiling that says the design was capable
    whenever it happened to reject, and incapable whenever it did not.
    """
    landed = _pairs(EIGHT_PAIRS)

    assert power(landed)["outcomes"] == 1
    assert power(landed)["paired"]["ceiling"] == pytest.approx(1.0), "the observed p, not a ceiling"
    assert dp.realized(landed)["paired"]["ceiling"] == pytest.approx(0.125)


def test_a_realized_ceiling_is_refused_while_a_cell_is_pending():
    with pytest.raises(SystemExit):
        dp.realized(_pairs(dp.CAMPAIGN))


def test_the_finished_read_out_names_a_p_that_is_not_evidence_of_absence(capsys):
    dp.report(_pairs(EIGHT_PAIRS), 0.05, None, (1, 4))
    out = capsys.readouterr().out

    assert "discordant 2/2 (4 concordant)" in out
    assert "NOT A NULL — 4 discordant pairs could not go below 0.1250" in out
    assert "a null WITH power — margin 6/16 reporting could have shown p = 0.0070" in out
