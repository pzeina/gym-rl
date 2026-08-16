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

from scripts.exact_tests import (
    fisher_two_sided,
    jackknife_rho,
    mcnemar_two_sided,
    spearman_rho,
)


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


# --- rank correlation: the instrument the "monotone spam" claim never had
#
# ROADMAP's 2026-08-16 entry listed nine reporting `patrol_brique` runs as
# "report rate -> false-DONE rate", sorted by false rate, and called the relation
# monotone. Sorted by the OTHER coordinate it is rho = +0.26 whose leave-one-out
# range straddles zero (refs assurance #59). The series is pinned here as
# published; it regenerates with `reporting_channel.py --spam`.
SPAM_SERIES = [(0.808, 0.223), (0.750, 0.348), (0.750, 0.500), (0.895, 0.320),
               (0.867, 0.375), (0.825, 0.481), (0.794, 0.503), (0.878, 0.561),
               (1.000, 0.750)]


def test_the_published_spam_series_is_not_monotone():
    xs = [x for x, _ in SPAM_SERIES]
    ys = [y for _, y in SPAM_SERIES]

    assert spearman_rho(xs, ys) == pytest.approx(0.2594, abs=5e-4)
    low, high = jackknife_rho(xs, ys)
    assert low == pytest.approx(-0.0599, abs=5e-4), "dropping the 1.000 endpoint flips the sign"
    assert high == pytest.approx(0.5150, abs=5e-4)
    assert low <= 0.0 <= high, "a relation carried by one point is not a relation"


def test_the_ninth_run_weakened_the_claim_it_was_added_to():
    """+0.381 over eight, +0.259 over nine — the run that 'confirmed' it cut it."""
    eight = [p for p in SPAM_SERIES if p != (0.750, 0.500)]

    assert spearman_rho([x for x, _ in eight], [y for _, y in eight]) \
        == pytest.approx(0.3810, abs=5e-4)


def test_a_tie_in_x_is_not_broken_by_input_order():
    """Two runs at report rate 0.750 with false rates 0.348 and 0.500."""
    forward = spearman_rho([0.750, 0.750, 0.900], [0.348, 0.500, 0.320])
    reversed_rows = spearman_rho([0.750, 0.750, 0.900], [0.500, 0.348, 0.320])

    assert forward == pytest.approx(reversed_rows)


def test_a_perfect_monotone_series_reads_plus_one_and_a_reversed_one_minus_one():
    assert spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert spearman_rho([1, 2, 3, 4], [1, 9, 25, 81]) == pytest.approx(1.0), \
        "rank correlation, so any increasing transform is the same +1"


def test_a_constant_column_has_no_rank_correlation():
    from math import isnan

    assert isnan(spearman_rho([1, 1, 1], [1, 2, 3]))


def test_paired_inputs_of_different_length_are_refused():
    with pytest.raises(ValueError):
        spearman_rho([1, 2, 3], [1, 2])
