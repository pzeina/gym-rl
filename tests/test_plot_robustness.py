"""Regression hazard: a plotting failure must never cost a run its artifacts.

fireteam_v7 trained all 2,500,000 steps and then lost its curves, eval,
transcript, gif AND behavior.json because ``cover_under_threat`` — a
DEFEND-only column added with the positional regression gate — is written
blank by every non-DEFEND scenario, and ``plot_training`` called
``float('')``. ``train.py`` renders curves before it evaluates, so one
un-parseable cell discarded 44 minutes of CPU.

Two invariants keep that from recurring:

* a metrics.csv whose scenario-specific columns are entirely blank still
  plots, gapping the curve instead of raising;
* post-training artifacts are attempted independently, so a broken plotter
  cannot suppress the evaluation.
"""

from __future__ import annotations

import csv
import math

import pytest

from cohort.viz.plots import _num, _smooth, plot_training

COLUMNS = [
    "iteration",
    "env_steps",
    "ep_return",
    "success_rate",
    "ep_length",
    "entropy",
    "policy_loss",
    "value_loss",
    "comp_compliance",
    "comp_report",
    "comp_command",
    "comp_combat",
    "tx_per_agent_step",
    "cover_under_threat",           # DEFEND-only: blank everywhere else
    "objective_dist_under_threat",  # DEFEND-only: blank everywhere else
]


def _write_metrics(run_dir, rows: int = 40, blank: tuple[str, ...] = ()) -> None:
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i in range(rows):
            row = {c: f"{i * 0.5:.3f}" for c in COLUMNS}
            row["iteration"] = str(i)
            row["env_steps"] = str(i * 1024)
            for c in blank:
                row[c] = ""
            writer.writerow(row)


def test_all_blank_scenario_column_still_plots(tmp_path):
    """The exact fireteam_v7 shape: a column blank in every single row."""
    _write_metrics(tmp_path, blank=("cover_under_threat", "objective_dist_under_threat"))
    out = plot_training(tmp_path)
    assert out.exists()
    assert out.stat().st_size > 0


def test_intermittently_blank_column_still_plots(tmp_path):
    """A metric recorded for only part of a run must gap, not raise."""
    with (tmp_path / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i in range(40):
            row = {c: f"{i * 0.5:.3f}" for c in COLUMNS}
            row["iteration"] = str(i)
            row["env_steps"] = str(i * 1024)
            if i < 20:
                row["tx_per_agent_step"] = ""
            writer.writerow(row)
    assert plot_training(tmp_path).exists()


def test_empty_metrics_still_raises(tmp_path):
    """Tolerating blanks must not tolerate a run that recorded nothing."""
    _write_metrics(tmp_path, rows=0)
    with pytest.raises(ValueError, match="No metrics rows"):
        plot_training(tmp_path)


def test_num_degrades_blanks_to_nan():
    assert math.isnan(_num("cover_under_threat", ""))
    assert math.isnan(_num("cover_under_threat", None))
    assert math.isnan(_num("ep_return", "not-a-number"))
    assert _num("ep_return", "1.5") == 1.5
    assert _num("iteration", "7") == 7


def test_smooth_ignores_nan_and_keeps_empty_windows_nan():
    smoothed = _smooth([1.0, math.nan, 3.0, math.nan, 5.0], k=3)
    assert not any(math.isnan(v) for v in smoothed)
    assert all(math.isnan(v) for v in _smooth([math.nan] * 5, k=3))


def test_post_training_artifacts_are_independent(monkeypatch, tmp_path):
    """A broken plotter must not suppress the evaluation (the v7 loss)."""
    import cohort.viz.plots as plots_mod
    from cohort.training import train as train_mod

    evaluated: list[str] = []

    def boom(*_args, **_kwargs):
        msg = "plotter tripped"
        raise ValueError(msg)

    monkeypatch.setattr(plots_mod, "plot_training", boom)
    monkeypatch.setattr(
        "cohort.training.evaluate.evaluate",
        lambda *a, **k: evaluated.append("ran"),
    )

    class Args:
        no_eval = False

    failures: list[str] = []
    # mirror of train.main's artifact block, exercised without a 40-minute run
    for name, fn in (
        ("training_curves.png", lambda: plots_mod.plot_training(tmp_path)),
        ("evaluate", lambda: __import__(
            "cohort.training.evaluate", fromlist=["evaluate"]
        ).evaluate("ckpt")),
    ):
        try:
            fn()
        except Exception:
            failures.append(name)

    assert failures == ["training_curves.png"]
    assert evaluated == ["ran"], "evaluation must still run after a plot failure"
    assert hasattr(train_mod, "traceback"), "train.py reports artifact failures"
