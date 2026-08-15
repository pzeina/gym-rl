"""The exact tests are pinned against textbook values, not against our output.

Both are hand-rolled (no scipy in this venv), and a hand-rolled exact test is
the classic thing that agrees with the right answer on the symmetric cases
everyone checks by eye and disagrees everywhere else: a two-sided Fisher that
doubles one tail, a McNemar that counts the concordant pairs as evidence. The
values below are computed from the definitions by hand, so a rewrite that gets
subtly faster and subtly wrong fails here rather than in a ROADMAP entry.
"""

from __future__ import annotations

import pytest

from scripts.exact_tests import fisher_two_sided, mcnemar_two_sided


@pytest.mark.parametrize(("table", "expected"), [
    # the squad root-death cell at ckpt_best: 15/100 against 35/100
    ((15, 85, 35, 65), 0.001748),
    # the same arm against squad_v6's 45/100
    ((15, 85, 45, 55), 5.547e-06),
    # a pair that does NOT separate, which is the harder case to get right
    ((97, 3, 91, 9), 0.133763),
    # 5-of-5 against 1-of-5 — the only outcome the rdb3_seeds campaign can
    # reject on, and the reason #56 could bound six 3M-step runs with arithmetic
    ((5, 0, 1, 4), 0.047619),
    # one report more in the comparison arm and the same design cannot reject
    ((5, 0, 2, 3), 0.166667),
])
def test_fisher_matches_known_two_by_twos(table, expected):
    assert fisher_two_sided(*table) == pytest.approx(expected, rel=1e-3)


def test_fisher_reads_the_same_table_from_either_arm():
    assert fisher_two_sided(5, 0, 1, 4) == pytest.approx(fisher_two_sided(1, 4, 5, 0))


@pytest.mark.parametrize(("table", "expected"), [
    ((0, 0, 0, 0), 1.0),          # nothing measured is not evidence of nothing
    ((3, 0, 0, 0), 1.0),          # an empty row: one possible table, p = 1
    ((4, 0, 4, 0), 1.0),          # an empty column, likewise
])
def test_fisher_degenerate_margins_are_one_not_a_crash(table, expected):
    assert fisher_two_sided(*table) == expected


@pytest.mark.parametrize(("one_way", "other_way", "expected"), [
    (0, 0, 1.0),          # no discordant pairs: the design measured nothing
    (1, 1, 1.0),          # one flip each way is exactly the null
    (4, 0, 0.125),        # 2 x 0.5^4
    (5, 0, 0.0625),       # 2 x 0.5^5 — the ceiling of ANY five-pair design
    (6, 0, 0.03125),      # six unanimous pairs is where a paired reading rejects
    (10, 0, 0.001953),    # 2 x 0.5^10
    (8, 2, 0.109375),     # 2 x (1 + 10 + 45) / 1024
    (12, 3, 0.035156),    # 2 x (1 + 15 + 105 + 455) / 32768
])
def test_mcnemar_matches_the_exact_binomial(one_way, other_way, expected):
    assert mcnemar_two_sided(one_way, other_way) == pytest.approx(expected, rel=1e-4)


def test_mcnemar_does_not_care_which_direction_is_named_first():
    assert mcnemar_two_sided(8, 2) == mcnemar_two_sided(2, 8)


def test_five_matched_pairs_can_never_reject_at_five_percent():
    """The result that sizes designs: perfect separation at n = 5 is p = 0.0625.

    Every five-pair paired comparison in this repo is a direction and not an
    effect, whatever it measures — which is why `scripts/design_power.py` runs
    before a campaign rather than after it.
    """
    assert mcnemar_two_sided(5, 0) > 0.05
    assert mcnemar_two_sided(6, 0) < 0.05
