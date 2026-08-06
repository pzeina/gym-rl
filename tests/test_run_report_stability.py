"""A peak must never be reported as a result.

Every published number in this repo comes from ``ckpt_best.pt``, which is
written on the best *rolling window* — not on the policy a run ended with. On a
run that spikes and falls back, evaluating ckpt_best measures the spike.

That is not hypothetical. Measured this session:

    squad_v6      best 98%  final 65%   evaluated 0.95 +/- 0.10 at N=20
    fireteam_v7   best 94%  final 26%   evaluated 0.95 +/- 0.04 at N=100

Both read as strong results. Neither run converged. The digest printed the two
numbers side by side and said nothing about the distance between them, and the
trap caught two consecutive sessions — so the verdict is now stated outright,
and pinned here.
"""

import math

from scripts.run_report import COLLAPSE_POINTS, stability


def test_converged_run_is_not_flagged():
    """squad_v5's shape: gave back 5 points."""
    line = stability(0.98, 0.93)
    assert "converged" in line
    assert "UNSTABLE" not in line and "COLLAPSED" not in line


def test_unstable_run_is_flagged():
    """fireteam_defend_v9's shape: gave back 17 points."""
    line = stability(0.96, 0.79)
    assert "UNSTABLE" in line
    assert "17 pts" in line


def test_collapsed_run_is_flagged_hardest():
    """fireteam_v7's shape: peaked at 94%, ended at 26%."""
    line = stability(0.94, 0.26)
    assert "COLLAPSED" in line
    assert "68 pts" in line
    assert "NOT a result" in line


def test_boundary_is_inclusive():
    assert "converged" in stability(1.0, 1.0 - (COLLAPSE_POINTS - 1) / 100)
    assert "UNSTABLE" in stability(1.0, 1.0 - COLLAPSE_POINTS / 100)
    assert "COLLAPSED" in stability(1.0, 1.0 - 2 * COLLAPSE_POINTS / 100)


def test_missing_rows_do_not_crash_or_claim_convergence():
    """No data must not silently read as a healthy run."""
    line = stability(math.nan, 0.5)
    assert "converged" not in line
    assert stability(0.9, math.nan) == line


def test_improving_run_is_converged():
    """A run that ends at its best gives back nothing."""
    assert "converged" in stability(0.90, 0.90)
