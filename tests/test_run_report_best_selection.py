"""The digest says where ``ckpt_best`` came from, because nothing ever did.

``ckpt_best.pt`` is what ``cohort.play``, the scenario gallery and every
spot-check load by default, and since v1.20 it is selected lexicographically on
the REPORTING channel before rolling success. The window that wins that
comparison is therefore a fact about the run — and it was never printed.

What that cost is on the record (refs assurance #57):
`patrol_brique_v19_rdb3_seed13` wrote its **only** ``ckpt_best`` at iteration 25
of 2930, 25,600 steps into a 3,000,320-step run, on a window at 2% rolling
success, and shipped it as the run's best work. The run's final policy succeeds
in 99 episodes of 100. One digest line would have shown that on the day.
"""

from __future__ import annotations

import csv

from scripts import run_report
from scripts.run_report import BEST_SELECTION_GAP, best_selection_line


def _rows(spec):
    """metrics.csv rows as ``(iteration, env_steps, success_rolling, root_close)``."""
    return [{"iteration": str(i), "env_steps": str(s),
             "success_rate_rolling": f"{success:.5f}",
             "root_report_close_rolling": "nan" if close is None else f"{close:.5f}"}
            for i, s, success, close in spec]


V19 = _rows([(25, 25_600, 0.02, 0.5), (550, 563_200, 1.0, 0.0), (2930, 3_000_320, 1.0, 0.0)])


def _stamp(monkeypatch, stamp):
    monkeypatch.setattr(run_report, "checkpoint_stamp", lambda path: stamp)
    monkeypatch.setattr(run_report, "run_dir", lambda name: __import__("pathlib").Path("/nonexistent"))


def test_the_line_names_the_iteration_and_the_window_it_was_chosen_on(monkeypatch):
    _stamp(monkeypatch, {"iteration": 25, "env_steps": 25_600})
    line = best_selection_line("v19", V19, 1.0)
    assert "iteration 25" in line and "25,600 steps" in line
    assert "1% of the run" in line
    assert "success 2%" in line and "closed-on-root 0.500" in line


def test_a_checkpoint_far_below_the_final_window_is_flagged(monkeypatch):
    """The #57 signature: the digest must not print it as an ordinary line."""
    _stamp(monkeypatch, {"iteration": 25, "env_steps": 25_600})
    assert "⚠" in best_selection_line("v19", V19, 1.0)


def test_a_checkpoint_at_the_run_s_own_level_is_not_flagged(monkeypatch):
    """Otherwise the marker measures the threshold rather than the pathology.

    A checkpoint saved on the same window the run ended on, and one saved half a
    gap below it, are both ordinary — a rolling window is a noisy estimate and
    the digest already prints the give-back separately (``stability``).
    """
    _stamp(monkeypatch, {"iteration": 550, "env_steps": 563_200})
    assert "⚠" not in best_selection_line("run", V19, 1.0)
    _stamp(monkeypatch, {"iteration": 550, "env_steps": 563_200})
    assert "⚠" not in best_selection_line("run", V19, 1.0 + BEST_SELECTION_GAP / 2)


def test_an_unmeasured_reporting_window_prints_a_dash_not_a_zero(monkeypatch):
    """NaN means no ENDEX landed in the window; 0.000 means the commander was
    mute. The gate treats them differently and so must the digest."""
    rows = _rows([(10, 10_240, 0.9, None)])
    _stamp(monkeypatch, {"iteration": 10, "env_steps": 10_240})
    assert "closed-on-root —" in best_selection_line("run", rows, 0.9)


def test_an_unreadable_checkpoint_prints_nothing(monkeypatch):
    """A live run is mid-write at any moment; a digest line that cannot be
    trusted is worse than no digest line."""
    _stamp(monkeypatch, None)
    assert best_selection_line("run", V19, 1.0) is None


def test_an_iteration_outside_this_corpus_prints_nothing(monkeypatch):
    """A resumed or truncated metrics.csv cannot describe the window."""
    _stamp(monkeypatch, {"iteration": 7777, "env_steps": 1})
    assert best_selection_line("run", V19, 1.0) is None


def test_the_digest_prints_the_line_for_a_real_run_directory(tmp_path, monkeypatch, capsys):
    """Wired into ``report()``, not just available to it."""
    run = tmp_path / "run"
    run.mkdir()
    with (run / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(V19[0]))
        writer.writeheader()
        writer.writerows(V19)
    monkeypatch.setattr(run_report, "run_dir", lambda name: run)
    monkeypatch.setattr(run_report, "checkpoint_stamp",
                        lambda path: {"iteration": 25, "env_steps": 25_600})
    run_report.report("run", show_components=False)
    out = capsys.readouterr().out
    assert "ckpt_best  iteration 25" in out and "⚠" in out
