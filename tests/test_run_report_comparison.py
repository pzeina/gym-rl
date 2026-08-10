"""An A/B table prints what the win cost, and says when the two sides differ in N.

Three cells, or the table flatters whichever axis was left out (refs #34):

* **success** alone is blind to the price — a cohort can win every episode over
  its commander's body;
* **root death rate** alone is gameable in the opposite direction, because a
  policy that never closes with the enemy buries nobody and achieves nothing;
* **timeout rate** is what closes that second gap — it separates "held the
  ground and everyone lived" from "rode the clock out and everyone lived".

The prompting case: the `squad_v8` → `squad_v9` A/B (`done_false` -0.5 -> -2.0)
was published with success and DONE volume alone. Its survival cell is a null
(p = 1.00 pooled over 200 episodes an arm), and the null is the finding, since
an earlier `done_false` change had once been *associated* with root deaths
moving 4/30 → 12/30 while success held. Nobody can infer a null from a missing
column, so the digest prints the pair instead of trusting the next author to.

The N is here for the same reason. The `squad_v8` comparator committed in this
repository is an **N=20** artifact and `squad_v9` publishes at N=100 — so the
A/B a reader can rebuild from the repo is 5x mismatched while looking exactly
like a matched one, once both sides are printed to three decimals.
"""

import csv
import json

import pytest

from scripts import run_report


def _write_run(tmp_path, name: str, *, best: dict | None = None, final: dict | None = None):
    """A run directory with just enough for ``report`` to read it."""
    run = tmp_path / name
    run.mkdir()
    with (run / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iteration", "env_steps", "success_rate_rolling"])
        w.writeheader()
        for i in range(20):
            w.writerow({"iteration": i, "env_steps": i * 100, "success_rate_rolling": 0.95})
    if best is not None:
        (run / "behavior.json").write_text(json.dumps(best))
    if final is not None:
        (run / "behavior_final.json").write_text(json.dumps(final))


def _behavior(ckpt: str, episodes: int, **metrics) -> dict:
    return {
        "checkpoint": f"runs/x/{ckpt}.pt",
        "episodes": episodes,
        "greedy": False,
        "success_ci95": "0.97 ± 0.03",
        "metrics": metrics,
        "gates": [],
    }


# The published squad_v9 A/B, at the N each arm was actually measured at.
V9_BEST = _behavior("ckpt_best", 100, success_rate=0.94, human_death_rate=0.19, timeout_rate=0.01)
V9_FINAL = _behavior("ckpt_latest", 100, success_rate=0.97, human_death_rate=0.18, timeout_rate=0.0)
V8_BEST = _behavior("ckpt_best", 100, success_rate=0.97, human_death_rate=0.15, timeout_rate=0.0)
V8_FINAL = _behavior("ckpt_latest", 100, success_rate=0.98, human_death_rate=0.23, timeout_rate=0.0)


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    monkeypatch.setattr(run_report, "RUNS", tmp_path)
    return tmp_path


#: metric key -> the label the comparison block prints it under
LABELS = {"success_rate": "success", "human_death_rate": "root death rate",
          "timeout_rate": "ran the clock out"}


def _run_vs(monkeypatch, capsys, run: str, baseline: str) -> str:
    """Drive the real ``--vs`` CLI path, not the helper underneath it."""
    monkeypatch.setattr(run_report.sys, "argv", ["run_report.py", run, "--vs", baseline])
    assert run_report.main() == 0
    return capsys.readouterr().out


def _rows(block: str, label: str) -> list[str]:
    """Every comparison ROW for one metric — prose mentioning it does not count."""
    return [ln for ln in block.splitlines() if ln.startswith(f"    {label:<20} ")]


def test_comparison_prints_survival_and_the_clock_beside_success(fleet, monkeypatch, capsys):
    """The pair #34 asks for, on both checkpoints, in the comparison block."""
    _write_run(fleet, "squad_v9", best=V9_BEST, final=V9_FINAL)
    _write_run(fleet, "squad_v8", best=V8_BEST, final=V8_FINAL)

    out = _run_vs(monkeypatch, capsys, "squad_v9", "squad_v8")
    block = out[out.index("== A/B:"):out.index("== delta:")]

    assert "== A/B: squad_v9 vs squad_v8 ==" in block
    for label in LABELS.values():
        assert len(_rows(block, label)) == 2, f"{label}: wanted ckpt_best AND the final policy"
    # baseline → run, matching the delta dump's own direction
    assert "0.970 →    0.940  (-0.030)" in block     # success, ckpt_best
    assert "0.150 →    0.190  (+0.040)" in block     # root deaths, ckpt_best
    assert "0.230 →    0.180  (-0.050)" in block     # root deaths, final policy


def test_matched_n_is_stated_on_both_checkpoints(fleet, monkeypatch, capsys):
    _write_run(fleet, "squad_v9", best=V9_BEST, final=V9_FINAL)
    _write_run(fleet, "squad_v8", best=V8_BEST, final=V8_FINAL)

    out = _run_vs(monkeypatch, capsys, "squad_v9", "squad_v8")

    assert out.count("N: squad_v8 100 · squad_v9 100   [matched]") == 2
    assert "MISMATCHED" not in out


def test_mismatched_n_is_labelled_not_printed_as_comparable(fleet, monkeypatch, capsys):
    """The repo-reconstructable squad A/B: an N=20 arm against an N=100 arm."""
    _write_run(fleet, "squad_v9", best=V9_BEST)
    _write_run(fleet, "squad_v8", best=_behavior(
        "ckpt_best", 20, success_rate=1.0, human_death_rate=0.05, timeout_rate=0.0))

    out = _run_vs(monkeypatch, capsys, "squad_v9", "squad_v8")

    assert "MISMATCHED N — 20 vs 100" in out
    assert "NOT an effect size" in out
    assert "[matched]" not in out
    # the numbers are still printed — the labelling is what makes them readable
    assert "0.050 →    0.190" in out


def test_unknown_n_is_not_reported_as_matched(fleet, monkeypatch, capsys):
    """An artifact without an episode count leaves matching unverified, not true."""
    no_count = _behavior("ckpt_best", 100, success_rate=0.9, human_death_rate=0.1, timeout_rate=0.0)
    del no_count["episodes"]
    _write_run(fleet, "squad_v9", best=V9_BEST)
    _write_run(fleet, "squad_v8", best=no_count)

    out = _run_vs(monkeypatch, capsys, "squad_v9", "squad_v8")

    assert "N UNKNOWN on one side" in out
    assert "[matched]" not in out and "MISMATCHED" not in out


@pytest.mark.parametrize("dropped", ["human_death_rate", "timeout_rate", "success_rate"])
def test_a_run_missing_either_metric_degrades_instead_of_crashing(
    fleet, monkeypatch, capsys, dropped
):
    """Pre-#18 runs carry no ``timeout_rate``; the comparison must still print.

    An unmeasured axis is not a passed one, so it reads as an em dash and a
    named absence rather than as a zero, a silently dropped row, or a
    traceback — and the axes that *were* measured still get compared.
    """
    old = {k: v for k, v in V8_BEST["metrics"].items() if k != dropped}
    _write_run(fleet, "squad_v9", best=V9_BEST)
    _write_run(fleet, "squad_v8", best=_behavior("ckpt_best", 100, **old))

    out = _run_vs(monkeypatch, capsys, "squad_v9", "squad_v8")
    block = out[out.index("== A/B:"):out.index("== delta:")]

    (row,) = _rows(block, LABELS[dropped])
    assert "—" in row                              # never a zero standing in for a gap
    assert "[not measured on squad_v8]" in row     # and it names which side is missing
    for label in LABELS.values():                  # the measured axes still compare
        assert len(_rows(block, label)) == 1


def test_a_run_with_no_behavior_suite_at_all_is_survivable(fleet, monkeypatch, capsys):
    """Nothing to compare is a quiet skip, not an exception."""
    _write_run(fleet, "squad_v9")
    _write_run(fleet, "squad_v8")

    out = _run_vs(monkeypatch, capsys, "squad_v9", "squad_v8")

    assert "== A/B: squad_v9 vs squad_v8 ==" in out
    assert "ckpt_best   N:" not in out


def test_episode_counts_do_not_masquerade_as_a_metric_delta(fleet, monkeypatch, capsys):
    """N belongs in the header. In the delta dump it reads as an axis that moved."""
    _write_run(fleet, "squad_v9", best=V9_BEST)
    _write_run(fleet, "squad_v8", best=_behavior(
        "ckpt_best", 20, success_rate=1.0, human_death_rate=0.05, timeout_rate=0.0))

    out = _run_vs(monkeypatch, capsys, "squad_v9", "squad_v8")
    delta = out[out.index("== delta:"):]

    assert "beh_episodes" not in delta
    assert "20.000 →  100.000" not in delta
