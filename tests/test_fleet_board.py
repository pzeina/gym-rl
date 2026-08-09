"""The fleet board must not overstate the evidence it is showing.

The board once captioned every column "success (N=100)" while the rows under it
were N=20 evaluations of the rolling-best checkpoint. That is the same class of
mistake the publish audit exists to catch, so it is pinned here: the collector
must say WHICH policy a number belongs to and over how many episodes, and the
renderer must print both on the row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import fleet_board, fleet_status


def _run(root: Path, name: str, *, scenario: str, final: dict | None, best: dict | None,
         economics: dict | None = None) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "metrics.csv").write_text("env_steps\n1000\n")
    (run / "config.json").write_text(json.dumps({"scenario": scenario}))
    (run / "ckpt_best.pt").write_text("stub")
    if final is not None:
        (run / "behavior_final.json").write_text(json.dumps(final))
    if best is not None:
        (run / "behavior.json").write_text(json.dumps(best))
    if economics is not None:
        (run / "economics.json").write_text(json.dumps(economics))
    return run


@pytest.fixture
def loadable(monkeypatch):
    """Every stub checkpoint reads as loadable under the current spaces."""
    from cohort.viz import dashboard

    monkeypatch.setattr(
        dashboard,
        "checkpoint_meta",
        lambda path: {"loadable": True, "obs_dim": 220, "env_steps": 2_000_000, "reason": ""},
    )


def test_collect_prefers_the_final_policy_and_records_which_it_used(tmp_path, loadable):
    _run(
        tmp_path,
        "fireteam_defend_v12",
        scenario="fireteam_defend",
        final={"episodes": 100, "success_ci95": "0.86 ± 0.07", "gates": []},
        best={"episodes": 20, "success_ci95": "0.85 ± 0.16", "gates": []},
    )
    (row,) = fleet_status.collect(tmp_path)

    assert row["policy"] == "final"
    assert (row["success"], row["episodes"]) == (0.86, 100)
    assert row["success_ci"] == pytest.approx(0.07)
    # the other evaluation is kept, not discarded — the row can show both
    assert (row["best_ci95"], row["best_episodes"]) == ("0.85 ± 0.16", 20)


def test_collect_falls_back_to_the_best_checkpoint_and_says_so(tmp_path, loadable):
    _run(
        tmp_path,
        "squad_v8",
        scenario="squad",
        final=None,
        best={"episodes": 20, "success_ci95": "1.00 ± 0.00", "gates": []},
    )
    (row,) = fleet_status.collect(tmp_path)

    assert row["policy"] == "best"
    assert row["episodes"] == 20


def test_the_board_prints_each_row_s_own_episode_count(tmp_path, loadable):
    _run(
        tmp_path,
        "squad_v8",
        scenario="squad",
        final=None,
        best={"episodes": 20, "success_ci95": "1.00 ± 0.00", "gates": []},
    )
    html = fleet_board.render(fleet_status.collect(tmp_path))

    # the honest caption is per-row; no page-wide claim of publication-grade N
    assert "success (N=100)" not in html
    assert "best ckpt" in html
    assert ">20<" in html


def test_a_failed_gate_is_visible_on_the_row(tmp_path, loadable):
    _run(
        tmp_path,
        "defend_brique_v4",
        scenario="defend_brique",
        final={
            "episodes": 100,
            "success_ci95": "0.91 ± 0.06",
            "gates": [
                {"name": "timeout_rate", "value": 0.04, "bound": 0.5,
                 "direction": "max", "passed": True},
                {"name": "mean_distance_from_objective_under_threat", "value": 6.09,
                 "bound": 5.0, "direction": "max", "passed": False},
            ],
        },
        best=None,
    )
    (row,) = fleet_status.collect(tmp_path)
    assert row["gates_failed"] == ["mean_distance_from_objective_under_threat"]

    html = fleet_board.render([row])
    assert "chip bad" in html
    assert "distance" in html


def test_reward_overrides_ride_the_run_so_an_a_b_is_attributable(tmp_path, loadable):
    _run(
        tmp_path,
        "defend_brique_v7",
        scenario="defend_brique",
        final={"episodes": 100, "success_ci95": "0.89 ± 0.06", "gates": []},
        best=None,
        economics={"reward_overrides": ["defend_survivor_scale=0"]},
    )
    (row,) = fleet_status.collect(tmp_path)

    assert row["overrides"] == ["defend_survivor_scale=0"]
    assert "defend_survivor_scale=0" in fleet_board.render([row])


def test_stale_runs_are_archived_rather_than_ranked_beside_current_ones(tmp_path, monkeypatch):
    from cohort.viz import dashboard

    monkeypatch.setattr(
        dashboard,
        "checkpoint_meta",
        lambda path: {"loadable": False, "obs_dim": 137, "env_steps": 1_000_000,
                      "reason": "obs_dim mismatch"},
    )
    _run(
        tmp_path,
        "fireteam_v5",
        scenario="fireteam",
        final=None,
        best={"episodes": 30, "success_ci95": "0.93 ± 0.09", "gates": []},
    )
    html = fleet_board.render(fleet_status.collect(tmp_path))

    assert "Superseded observation eras" in html
    assert "Current build — 0 runs" in html


def test_version_order_is_numeric_not_lexical():
    names = ["squad_v10", "squad_v2", "squad_v9"]
    assert sorted(names, key=fleet_board._nat) == ["squad_v2", "squad_v9", "squad_v10"]
