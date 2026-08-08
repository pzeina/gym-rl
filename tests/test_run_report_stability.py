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


# The two checkpoints `defend_brique_v4` actually shipped, trimmed to the axes
# the v1.12 pre-registration (assurance #22) names. ckpt_best clears all three
# regression gates; the FINAL policy — the one the publishing standard calls
# the headline — fails the positional gate at 6.09 against a bound of 5.0, and
# kills the human root in 61 episodes in 100 against ckpt_best's 50 in 100.
V4_BEST = {
    "checkpoint": "runs/defend_brique_v4/ckpt_best.pt",
    "episodes": 20,
    "greedy": False,
    "success_ci95": "0.90 ± 0.13",
    "metrics": {
        "success_rate": 0.9, "human_death_rate": 0.5, "timeout_rate": 0.1,
        "cover_occupancy_under_threat": 0.4898,
        "mean_distance_from_objective_under_threat": 2.7497,
    },
    "gates": [
        {"name": "timeout_rate", "bound": 0.5, "direction": "max", "passed": True},
        {"name": "cover_occupancy_under_threat", "bound": 0.4, "direction": "min", "passed": True},
        {"name": "mean_distance_from_objective_under_threat", "bound": 5.0,
         "direction": "max", "passed": True},
    ],
}
V4_FINAL = {
    "checkpoint": "runs/defend_brique_v4/ckpt_latest.pt",
    "episodes": 100,
    "greedy": False,
    "success_ci95": "0.91 ± 0.06",
    "metrics": {
        "success_rate": 0.91, "human_death_rate": 0.61, "timeout_rate": 0.04,
        "cover_occupancy_under_threat": 0.4161,
        "mean_distance_from_objective_under_threat": 6.0907,
    },
    "gates": [
        {"name": "timeout_rate", "bound": 0.5, "direction": "max", "passed": True},
        {"name": "cover_occupancy_under_threat", "bound": 0.4, "direction": "min", "passed": True},
        {"name": "mean_distance_from_objective_under_threat", "bound": 5.0,
         "direction": "max", "passed": False},
    ],
}


def _write_run(tmp_path, name: str, best: dict, final: dict | None = None):
    run = tmp_path / name
    run.mkdir()
    with (run / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iteration", "env_steps", "success_rate_rolling"])
        w.writeheader()
        for i in range(20):
            w.writerow({"iteration": i, "env_steps": i * 100, "success_rate_rolling": 0.9})
    (run / "behavior.json").write_text(json.dumps(best))
    if final is not None:
        (run / "behavior_final.json").write_text(json.dumps(final))


def test_final_policy_gates_are_printed_not_only_ckpt_best_s(tmp_path, monkeypatch, capsys):
    """A gate the headline policy fails must not be hidden by ckpt_best's PASS.

    refs #22. `defend_brique_v4` is the case: three PASSes at ckpt_best, and
    the positional regression the whole v1.12 reward decision was taken over
    (`mean_distance_from_objective_under_threat` 6.09, bound 5.0) exists only
    at ckpt_latest. The digest used to print the suite plus every gate for
    ckpt_best and *one success number* for the final policy, so the FAIL that
    justified the retrain was absent from the artifact the verdict is read off.
    """
    _write_run(tmp_path, "defend_brique_v4", V4_BEST, V4_FINAL)
    monkeypatch.setattr(run_report, "RUNS", tmp_path)

    run_report.report("defend_brique_v4", show_components=False)
    out = capsys.readouterr().out

    assert "gate [FAIL] mean_distance_from_objective_under_threat" in out
    assert out.count("gate [PASS] timeout_rate") == 2      # both checkpoints gated
    assert "ckpt_latest.pt" in out                         # and both named
    assert "vs ckpt_best 0.90 → final 0.91" in out         # the old line survives


def test_root_death_rate_is_reported_for_both_checkpoints(tmp_path, monkeypatch, capsys):
    """The pre-registered PRIMARY axis has to be in the digest, at final.

    refs #22 pins root deaths at the final policy as the primary measurement
    of the survivor-scaled-terminal A/B. `human_death_rate` was aggregated on
    every behavior run and printed by `evaluate`'s table, but this digest — the
    only artifact a verdict is written against — dropped it, at either
    checkpoint. The summary keys matter as much as the printed lines: `--vs`
    deltas are built from them, so an axis missing here is an axis the A/B
    cannot compare.
    """
    _write_run(tmp_path, "defend_brique_v4", V4_BEST, V4_FINAL)
    monkeypatch.setattr(run_report, "RUNS", tmp_path)

    summary = run_report.report("defend_brique_v4", show_components=False)
    out = capsys.readouterr().out

    assert out.count("root death rate") == 2
    assert "0.500" in out and "0.610" in out
    assert summary["beh_human_death_rate"] == 0.5
    assert summary["final_human_death_rate"] == 0.61
    # the positional pair, at the checkpoint the prediction is stated for
    assert summary["final_cover_occupancy_under_threat"] == 0.4161
    assert summary["final_mean_distance_from_objective_under_threat"] == 6.0907
    assert summary["final_success"] == 0.91
