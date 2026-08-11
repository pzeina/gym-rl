"""The ablation is read on the axes it actually separates on.

The 2026-08-06 B3 result separated `full` from `flat` by 7 points of success and
by **2.2x the wipe rate**, and separated `full` from `nomask` not at all on
success while separating them completely on interpretability — 100% doctrine-
valid traffic against 33-48%, and 128 completion reports against ~0. A
replication that reads the success column alone concludes the hierarchy does
nothing, and is reading the wrong column.

So what is pinned here is the arithmetic that makes the other columns readable:

* **defeats per 100 is derived, not stored.** An episode that neither succeeded
  nor ran the clock out is a cohort that was killed. Getting that subtraction
  wrong would silently move the robustness cell — the one the original claim
  rests on.
* **overlapping intervals are not a difference**, and the report has to say so
  rather than leaving a reader to compare two point estimates.
"""

from __future__ import annotations

import json

import pytest

from scripts import ablation_report


def _arm(tmp_path, run: str, *, successes: int, timeout: float = 0.0, n: int = 100, **metrics):
    d = tmp_path / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "behavior_final.json").write_text(json.dumps({
        "episodes": n,
        "success_ci95": f"{successes / n:.2f} ± 0.05",
        "metrics": {"success_rate": successes / n, "successes": successes,
                    "timeout_rate": timeout, **metrics},
    }))


@pytest.fixture
def trio(tmp_path, monkeypatch):
    monkeypatch.setattr(ablation_report, "run_dir", lambda name: tmp_path / name)
    return tmp_path


def test_defeats_are_the_episodes_that_were_neither_won_nor_timed_out(trio):
    _arm(trio, "a", successes=85, timeout=0.04)

    facts = ablation_report._facts("a")

    assert facts["defeat_per_100"] == pytest.approx(11.0)


def test_a_perfect_arm_has_no_negative_defeats(trio):
    """Floating point on 1 - 1.0 - 0.0 must not print -0.0 wipes."""
    _arm(trio, "a", successes=100, timeout=0.0)

    assert ablation_report._facts("a")["defeat_per_100"] == pytest.approx(0.0)


def test_overlapping_intervals_are_reported_as_not_a_difference(trio, capsys):
    _arm(trio, "full_v1", successes=97)
    _arm(trio, "nomask_v1", successes=98)
    _arm(trio, "flat_v1", successes=96)

    ablation_report.report(["full_v1", "nomask_v1", "flat_v1"])
    out = capsys.readouterr().out

    assert out.count("intervals OVERLAP, not a difference") == 2
    assert "Fisher p = 1.000" in out


def test_a_real_separation_is_not_called_an_overlap(trio, capsys):
    _arm(trio, "full_v1", successes=97)
    _arm(trio, "nomask_v1", successes=96)
    _arm(trio, "flat_v1", successes=55)

    ablation_report.report(["full_v1", "nomask_v1", "flat_v1"])
    out = capsys.readouterr().out

    assert "— separated" in out


def test_the_report_leads_the_reader_to_the_axes_that_separate(trio, capsys):
    for run in ("full_v1", "nomask_v1", "flat_v1"):
        _arm(trio, run, successes=95, orders_per_episode=1.0,
             doctrine_allowed_rate=1.0, done_reports=10)

    ablation_report.report(["full_v1", "nomask_v1", "flat_v1"])
    out = capsys.readouterr().out

    assert "ROBUSTNESS" in out and "INTERPRET" in out
    assert "One seed per arm" in out, "the replication must state its own strength"


def test_an_unevaluated_arm_is_named_rather_than_silently_dropped(trio, capsys):
    _arm(trio, "full_v1", successes=97)

    ablation_report.report(["full_v1", "nomask_v1", "flat_v1"])
    out = capsys.readouterr().out

    assert "not evaluated yet: nomask_v1, flat_v1" in out


@pytest.mark.parametrize(("table", "expected"), [
    # the squad root-death cell at ckpt_best: 15/100 against 35/100
    ((15, 85, 35, 65), 0.001748),
    # the same arm against squad_v6's 45/100
    ((15, 85, 45, 55), 5.547e-06),
    # a pair that does NOT separate, which is the harder case to get right
    ((97, 3, 91, 9), 0.133763),
])
def test_fisher_matches_known_two_by_twos(table, expected):
    """No scipy in this venv, so the exact test is hand-rolled and pinned.

    A two-sided Fisher summing the wrong tail is the classic way to write one:
    it agrees with the right answer on symmetric tables and quietly disagrees
    everywhere else. The three cells here are asymmetric and span six orders of
    magnitude, which a tail error cannot survive.
    """
    assert ablation_report._fisher(*table) == pytest.approx(expected, rel=1e-3)
