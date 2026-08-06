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

import csv
import json
import math

from scripts import run_report
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


def test_digest_names_the_checkpoint_it_scored(tmp_path, monkeypatch, capsys):
    """The same run reads 30/30 or 0/30 depending on the checkpoint (refs #18).

    ``squad_screen_v4`` succeeds in every episode from ``ckpt_best`` and times
    out in every episode from ``ckpt_latest``, on the same seeds. A behavior
    block printed under a curve that ended at 0% must therefore say which of
    the two it measured, and the clock-expiry number must travel with it.
    """
    run = tmp_path / "collapsed"
    run.mkdir()
    with (run / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iteration", "env_steps", "success_rate_rolling"])
        w.writeheader()
        for i in range(20):
            w.writerow({"iteration": i, "env_steps": i * 100, "success_rate_rolling": 1.0 if i < 10 else 0.0})
    (run / "behavior.json").write_text(json.dumps({
        "checkpoint": "runs/collapsed/ckpt_best.pt",
        "episodes": 30,
        "success_ci95": "1.00 ± 0.00",
        "metrics": {"success_rate": 1.0, "timeout_rate": 0.0, "messages_per_episode": 537.0,
                    "command_traffic_share": 0.155},
        "gates": [],
    }))
    monkeypatch.setattr(run_report, "RUNS", tmp_path)
    run_report.report("collapsed", show_components=False)
    out = capsys.readouterr().out
    assert "ckpt_best.pt" in out
    assert "COLLAPSED" in out          # the curve's own verdict, unchanged
    assert "ran the clock out" in out
    assert "of which command" in out
