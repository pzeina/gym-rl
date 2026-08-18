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

from scripts import baseline as baseline_module
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
    assert "0 further runs on this build" in html


def test_the_board_leads_with_the_baseline_not_with_the_directory_listing(tmp_path, loadable,
                                                                         monkeypatch):
    """A hundred run names is a filing cabinet; the manifest says which eight ship.

    Membership comes from ``runs/BASELINE.json`` rather than from a name pattern,
    so the board and ``scripts/baseline.py``'s gate can never disagree about what
    the fleet is.
    """
    from scripts import fleet_board, fleet_status

    (tmp_path / "BASELINE.json").write_text(
        '{"version": "v9.9", "runs": {"squad": "squad_v10"}}'
    )
    _run(tmp_path, "squad_v10", scenario="squad",
         final={"episodes": 100, "success_ci95": "0.97 ± 0.03", "gates": []}, best=None)
    _run(tmp_path, "squad_v5", scenario="squad",
         final={"episodes": 100, "success_ci95": "0.93 ± 0.05", "gates": []}, best=None)

    rows = {r["run"]: r for r in fleet_status.collect(tmp_path)}
    assert rows["squad_v10"]["baseline"] == "squad"
    assert rows["squad_v5"]["baseline"] is None

    page = fleet_board.render(list(rows.values()))
    assert "chip base" in page
    # the member is not also listed among the runs below, or it reads as two runs
    assert page.count('<span class="r">squad_v10</span>') == 1
    assert '<span class="r">squad_v5</span>' in page
    # every doctrine scenario is accounted for, including the ones with no member
    assert page.count("no member on disk yet") == len(baseline_module.DOCTRINE_SCENARIOS) - 1


def test_version_order_is_numeric_not_lexical():
    names = ["squad_v10", "squad_v2", "squad_v9"]
    assert sorted(names, key=fleet_board._nat) == ["squad_v2", "squad_v9", "squad_v10"]


def test_an_unmeasured_gate_neither_crashes_nor_reads_as_a_failure(tmp_path, loadable):
    """A run that never completed an episode has nothing to read for some gates.

    `squad_v21_seed16` trained 3M steps at 0% success, so
    `closed_on_root_report_rate` was emitted with `value=None, passed=None`
    per `regression_gates`' contract. `_tip` formatted that value
    unconditionally and took the whole board refresh down with a TypeError —
    every landing run re-renders the boards, so one such run blocked all three.
    """
    _run(
        tmp_path,
        "squad_v21_seed16",
        scenario="squad",
        final={
            "episodes": 100,
            "success_ci95": "0.00 ± 0.00",
            "gates": [
                {"name": "timeout_rate", "value": 1.0, "bound": 0.5,
                 "direction": "max", "passed": False},
                {"name": "closed_on_root_report_rate", "value": None, "bound": 0.5,
                 "direction": "min", "passed": None},
            ],
        },
        best=None,
    )
    (row,) = fleet_status.collect(tmp_path)
    assert row["gates_failed"] == ["timeout_rate"]
    assert row["gates_unmeasured"] == ["closed_on_root_report_rate"]

    html = fleet_board.render([row])  # the crash
    assert "unmeasured" in html


def test_an_all_pass_row_does_not_count_an_unmeasured_gate_as_passing(tmp_path, loadable):
    _run(
        tmp_path,
        "squad_v30",
        scenario="squad",
        final={
            "episodes": 100,
            "success_ci95": "0.95 ± 0.04",
            "gates": [
                {"name": "timeout_rate", "value": 0.02, "bound": 0.5,
                 "direction": "max", "passed": True},
                {"name": "mean_distance_from_objective_under_threat", "value": None,
                 "bound": 5.0, "direction": "max", "passed": None},
            ],
        },
        best=None,
    )
    (row,) = fleet_status.collect(tmp_path)
    html = fleet_board.render([row])
    assert "2/2 pass" not in html
    assert "1/1 pass" in html
    assert "1 unmeasured" in html


# ------------------------------------------- the reporting-channel disclosure --


def _reporting_fleet(tmp_path, monkeypatch, *, search=None, rates=None):
    """A manifest plus the artifacts the disclosure is derived from."""
    runs = {s: f"{s}_m" for s in baseline_module.DOCTRINE_SCENARIOS}
    manifest = {"version": "test", "runs": runs}
    if search:
        manifest["seed_search"] = search
    (tmp_path / "BASELINE.json").write_text(json.dumps(manifest))
    names = set(runs.values()) | {r for rs in (search or {}).values() for r in rs}
    for name in names:
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps({"seed": (rates or {}).get(name, (12, 0.8))[0]}))
        (d / "economics.json").write_text(json.dumps({"git_commit": "a" * 40,
                                                      "reward_overrides": []}))
        (d / "behavior_final.json").write_text(json.dumps({
            "episodes": 100,
            "metrics": {"closed_on_root_report_rate": (rates or {}).get(name, (12, 0.8))[1]},
        }))
    monkeypatch.setattr(baseline_module, "RUNS", tmp_path)
    monkeypatch.setattr(baseline_module, "MANIFEST", tmp_path / "BASELINE.json")
    return manifest


def test_the_board_states_when_a_member_was_picked_from_several_seeds(tmp_path, monkeypatch):
    """The disclosure gate A rests on.

    A member chosen as the best of four and published as though it were the only
    one is exactly the overstatement the manifest's declaration exists to
    prevent — and a declaration nobody renders is not a disclosure.
    """
    rates = {"patrol_brique_s12": (12, 0.0), "patrol_brique_s18": (18, 0.8),
             "patrol_brique_s19": (19, 0.75), "patrol_brique_s14": (14, 0.0)}
    manifest = _reporting_fleet(
        tmp_path, monkeypatch,
        search={"patrol_brique": list(rates)}, rates=rates)
    manifest["runs"]["patrol_brique"] = "patrol_brique_s18"
    (tmp_path / "BASELINE.json").write_text(json.dumps(manifest))

    html = fleet_board.reporting_channel(baseline_module.load())

    assert "2 of 4 seeds report" in html
    assert "seeds 12, 18, 19, 14" in html


def test_an_unsearched_fleet_says_one_seed_rather_than_going_quiet(tmp_path, monkeypatch):
    """Silence would read as robustness. Each scenario says which it is."""
    _reporting_fleet(tmp_path, monkeypatch)
    html = fleet_board.reporting_channel(baseline_module.load())
    assert html.count("— one seed ·") == len(baseline_module.DOCTRINE_SCENARIOS)
    assert "each scenario ran one seed" in html


def test_a_mute_member_is_named_on_the_board(tmp_path, monkeypatch):
    """The v1.20b shape — every other axis fine, the commander never reports —
    must be legible to a reader who only looks at the board."""
    rates = {f"{s}_m": (12, 0.8) for s in baseline_module.DOCTRINE_SCENARIOS}
    rates["patrol_brique_m"] = (12, 0.0)
    _reporting_fleet(tmp_path, monkeypatch, rates=rates)
    assert "MUTE" in fleet_board.reporting_channel(baseline_module.load())


def test_a_member_with_no_done_claims_shows_a_dash_not_a_zero(tmp_path, monkeypatch):
    """``fireteam_defend_v23`` records ``false_complete_rate: null`` at N=100 —
    it filed no DONE claims, so there is nothing to measure precision over.
    A 0 on the board would read as a perfect claim record; the board must say
    absent, and say why, next to a member whose rate WAS measured."""
    _reporting_fleet(tmp_path, monkeypatch)
    absent = tmp_path / "fireteam_defend_m" / "behavior_final.json"
    data = json.loads(absent.read_text())
    data["metrics"]["false_complete_rate"] = None
    absent.write_text(json.dumps(data))
    measured = tmp_path / "squad_m" / "behavior_final.json"
    data = json.loads(measured.read_text())
    data["metrics"]["false_complete_rate"] = 0.372
    measured.write_text(json.dumps(data))

    html = fleet_board.reporting_channel(baseline_module.load())

    assert "false-DONE —" in html
    assert "false-DONE 0.37" in html
    assert "false-DONE 0.00" not in html
    assert "no DONE claims" in html  # the dash explains itself


def test_the_board_counts_spread_draws_beside_the_declared_search(tmp_path, monkeypatch):
    """The #63 gap, closed at the rendering: "2 of 2 seeds report" was true of
    the SEARCH while two more same-config draws sat outside it, one failing the
    gate on both checkpoints. The board now prints the spread beside the search
    — counts, never rates — and the honest denominator."""
    rates = {"squad_s12": (12, 0.8), "squad_s13": (13, 0.8),
             "squad_v29_seed14": (14, 0.0), "squad_v20_seed15": (15, 0.0)}
    manifest = _reporting_fleet(
        tmp_path, monkeypatch, search={"squad": ["squad_s12", "squad_s13"]}, rates=rates)
    manifest["runs"]["squad"] = "squad_s12"
    manifest["seed_spread"] = {"squad": ["squad_v29_seed14", "squad_v20_seed15"]}
    (tmp_path / "BASELINE.json").write_text(json.dumps(manifest))
    for name in ("squad_v29_seed14", "squad_v20_seed15"):
        d = tmp_path / name
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text(json.dumps({"seed": rates[name][0]}))
        (d / "economics.json").write_text(json.dumps({"git_commit": "a" * 40,
                                                      "reward_overrides": []}))
        (d / "behavior_final.json").write_text(json.dumps({
            "episodes": 100,
            "metrics": {"closed_on_root_report_rate": rates[name][1]}}))

    html = fleet_board.reporting_channel(baseline_module.load())

    assert "2 of 2 seeds report" in html          # the search claim stands...
    assert "spread +2 draws, 0 report" in html    # ...and its quiet half is beside it
    assert "<b>2 of 4 known</b>" in html          # the honest denominator
    assert "Counts, never rates." in html


def test_the_board_annotates_cross_tree_spread_draws(tmp_path, monkeypatch):
    """A draw from another cohort/ tree is evidence about the seed, not about
    the sealed environment — carried in the count, visibly annotated."""
    _reporting_fleet(tmp_path, monkeypatch)
    manifest = json.loads((tmp_path / "BASELINE.json").read_text())
    manifest["seed_spread"] = {"squad": ["squad_old_seed13"]}
    (tmp_path / "BASELINE.json").write_text(json.dumps(manifest))
    d = tmp_path / "squad_old_seed13"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"seed": 13}))
    (d / "economics.json").write_text(json.dumps({"git_commit": "b" * 40,
                                                  "reward_overrides": []}))
    (d / "behavior_final.json").write_text(json.dumps({
        "episodes": 100, "metrics": {"closed_on_root_report_rate": 0.9}}))
    monkeypatch.setattr(baseline_module, "cohort_tree",
                        lambda c: {"a" * 40: "sealed-tree"}.get(c, "older-tree"))

    html = fleet_board.reporting_channel(baseline_module.load())

    assert "spread +1 draws, 1 report" in html
    assert "(1 cross-tree)" in html
    assert "evidence about the seed, not about the sealed environment" in html


def test_an_undeclared_draw_changes_no_board_number(tmp_path, monkeypatch):
    """The board renders the DECLARED block only: an undeclared same-config
    draw is the audit's exit-1 to raise, not a number for the board to invent
    — a rendered count must never be truer than its manifest."""
    _reporting_fleet(tmp_path, monkeypatch)
    d = tmp_path / "squad_undeclared"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"seed": 14}))
    (d / "economics.json").write_text(json.dumps({"git_commit": "a" * 40,
                                                  "reward_overrides": []}))

    html = fleet_board.reporting_channel(baseline_module.load())

    assert "spread" not in html
