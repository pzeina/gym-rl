"""The price-dispersion bar is pinned, not merely written.

A pre-registration whose thresholds can be edited once the numbers are in is
decoration. These tests fail if a bound moves, if the frozen incumbents stop
matching the evaluations they were taken from, or if the rule loses the ability
to reach one of its own verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import prereg_dispersion as pd

ROOT = Path(__file__).resolve().parent.parent
FROZEN = json.loads(pd.FROZEN.read_text())


def _reading(*, stacked: float, nearest: float, successes: int = 100) -> dict:
    return {"run": "synthetic", "episodes": 100, "successes": successes,
            "success_rate": successes / 100, "stacked_rate": stacked,
            "mean_nearest_teammate_dist": nearest, "spatially_sound_rate": None,
            "checkpoint_sha256": None}


def _arms(**kwargs) -> dict:
    """A full primary pair, both members reading the same way."""
    return {s: _reading(**kwargs) for s in pd.PRIMARY}


def _frozen_row(scenario: str) -> dict:
    v = FROZEN["incumbents"][scenario]
    return {k: v[k] for k in ("run", "episodes", "successes", "success_rate",
                              "stacked_rate", "mean_nearest_teammate_dist",
                              "spatially_sound_rate", "checkpoint_sha256")}


def _fleet(**overrides) -> dict:
    """The whole nine: a synthetic DEFEND pair plus the seven bystanders.

    The guard family's SIZE is part of the rule — Holm at m = 7 is a different
    bar from Holm at m = 1 — so a test that stubs two bystanders is testing a
    design nobody is going to run.
    """
    rows = {s: _frozen_row(s) for s in FROZEN["incumbents"] if s not in pd.PRIMARY}
    return {**_arms(stacked=0.35, nearest=1.60), **rows, **overrides}


# ------------------------------------------------------- the bar cannot move ---

def test_thresholds_in_code_match_the_frozen_registration():
    t = FROZEN["thresholds"]
    assert t["stacked_bar"] == pd.STACKED_BAR
    assert t["stacked_move"] == pd.STACKED_MOVE
    assert t["nearest_rise"] == pd.NEAREST_RISE
    assert t["alpha"] == pd.ALPHA
    assert tuple(t["ladder"]) == pd.LADDER
    assert tuple(FROZEN["primary"]) == pd.PRIMARY


def test_frozen_incumbents_still_match_the_evaluations_they_came_from():
    """The snapshot is of committed files, so it must stay re-derivable.

    This is the check that would have caught the retracted seed-carry claim:
    a comparison is only a comparison if its baseline is the thing it says.
    """
    for scenario, frozen in FROZEN["incumbents"].items():
        blob = pd._final_eval(frozen["run"])
        assert blob is not None, f"{scenario}: {frozen['run']} lost its N=100 evaluation"
        live = pd._read(blob)
        for key in ("episodes", "successes", "stacked_rate",
                    "mean_nearest_teammate_dist", "checkpoint_sha256"):
            assert live[key] == pytest.approx(frozen[key]) if isinstance(
                live[key], float) else live[key] == frozen[key], \
                f"{scenario}.{key} drifted from the frozen registration"


def test_every_verdict_the_rule_emits_is_documented():
    assert set(pd.VERDICTS) >= {"SEPARATES", "WALKS", "DENOMINATOR", "PARTIAL",
                                "CEILING", "NO EFFECT AT THIS PRICE", "INCOMPLETE"}


# --------------------------------------------- every verdict is reachable ---

def test_the_incumbents_scored_against_themselves_move_nothing():
    rows = {s: {"run": v["run"], **{k: v[k] for k in
                ("episodes", "successes", "success_rate", "stacked_rate",
                 "mean_nearest_teammate_dist", "spatially_sound_rate",
                 "checkpoint_sha256")}}
            for s, v in FROZEN["incumbents"].items()}
    assert pd.decide(FROZEN, rows, "self", False)["verdict"] == "NO EFFECT AT THIS PRICE"
    assert pd.decide(FROZEN, rows, "self", True)["verdict"] == "CEILING"


def test_a_dispersed_cohort_that_keeps_winning_separates():
    out = pd.decide(FROZEN, _fleet(), "0.5", False)
    assert out["verdict"] == "SEPARATES"
    assert len(out["guard_p"]) == 7


def test_four_lost_episodes_out_of_a_hundred_still_separates():
    """96/100 against a perfect incumbent reads p = 0.0606 — not a loss."""
    out = pd.decide(FROZEN, _arms(stacked=0.35, nearest=1.60, successes=96), "0.5", False)
    assert out["verdict"] == "SEPARATES"
    assert all(v["p_non_inferiority"] == pytest.approx(0.0606, abs=1e-4)
               for v in out["primary"].values())


def test_five_lost_episodes_is_a_walk():
    out = pd.decide(FROZEN, _arms(stacked=0.35, nearest=1.60, successes=95), "0.5", False)
    assert out["verdict"] == "WALKS"


def test_bunching_bought_with_casualties_reads_denominator_not_separates():
    """The jamming mistake, encoded.

    Stacked collapses 0.96 -> 0.30, success is untouched, and the element is
    exactly as tightly packed as before: the rate fell because there were fewer
    living teammates to be near, not because anyone spread out.
    """
    out = pd.decide(FROZEN, _arms(stacked=0.30, nearest=0.22), "0.5", False)
    assert out["verdict"] == "DENOMINATOR"


def test_real_dispersion_short_of_the_bar_is_partial_then_ceiling_at_the_top():
    arms = _arms(stacked=0.78, nearest=1.10)
    assert pd.decide(FROZEN, arms, "0.5", False)["verdict"] == "PARTIAL"
    assert pd.decide(FROZEN, arms, "1.0", True)["verdict"] == "CEILING"


def test_one_defend_member_missing_is_incomplete_not_a_result():
    rows = {pd.PRIMARY[0]: _reading(stacked=0.35, nearest=1.60)}
    assert pd.decide(FROZEN, rows, "0.5", False)["verdict"] == "INCOMPLETE"


# ------------------------------------------------------------ the guard ---

def test_a_broken_bystander_scenario_turns_a_separation_into_a_walk():
    out = pd.decide(FROZEN, _fleet(fireteam=_reading(
        stacked=0.20, nearest=2.10, successes=40)), "0.5", False)
    assert out["verdict"] == "WALKS"
    assert out["guard_broken"]["fireteam"] is True


def test_the_guard_is_holm_corrected_so_one_soft_dip_does_not_convict():
    """A single member a few episodes down must not read as a broken fleet.

    squad_screen at 95/100 against a perfect incumbent is p = 0.0297 — under
    alpha on its own, and retained by Holm across the seven-member family. That
    gap is the whole point of correcting: a fleet of nine scenarios read
    uncorrected raises a false conviction better than a third of the time.
    """
    out = pd.decide(FROZEN, _fleet(squad_screen=_reading(
        stacked=0.24, nearest=1.70, successes=95)), "0.5", False)
    assert out["guard_p"]["squad_screen"] == pytest.approx(0.0297, abs=1e-4)
    assert not any(out["guard_broken"].values())
    assert out["verdict"] == "SEPARATES"


# ---------------------------------------------------- the reading itself ---

def test_the_bar_refuses_a_reading_that_is_not_at_n100(tmp_path, monkeypatch):
    """N=20 is not a smaller version of N=100, it is a different bar.

    The incumbents are frozen at 100 episodes. Letting a 20-episode reading in
    would widen every comparison silently — which is how the CI on a candidate
    quietly grows until it overlaps whatever it is being compared to.
    """
    member = FROZEN["incumbents"][pd.PRIMARY[0]]
    blob = json.loads((pd.find_run(member["run"], pd.RUNS)
                       / "behavior_final_n100.json").read_text())

    (tmp_path / "short_run").mkdir()
    (tmp_path / "short_run" / "behavior_final_n100.json").write_text(
        json.dumps({**blob, "episodes": 20}))
    monkeypatch.setattr(pd, "RUNS", tmp_path)
    assert pd._final_eval("short_run") is None

    (tmp_path / "full_run").mkdir()
    (tmp_path / "full_run" / "behavior_final_n100.json").write_text(json.dumps(blob))
    assert pd._final_eval("full_run") is not None


def test_a_run_with_only_a_best_checkpoint_evaluation_is_not_readable(tmp_path, monkeypatch):
    """``ckpt_best`` is a peak; publish_audit.py exists because of that."""
    (tmp_path / "peak_only").mkdir()
    (tmp_path / "peak_only" / "behavior.json").write_text(json.dumps({"episodes": 100}))
    monkeypatch.setattr(pd, "RUNS", tmp_path)
    assert pd._final_eval("peak_only") is None


def test_stacked_rate_is_read_from_the_marker_block_not_a_gate():
    """It was demoted on 2026-08-26; a reader must not resurrect it as a gate."""
    member = FROZEN["incumbents"][pd.PRIMARY[0]]["run"]
    blob = json.loads((pd.find_run(member, pd.RUNS) / "behavior_final_n100.json").read_text())
    markers = {m["name"]: m for m in blob["markers"]}
    assert "stacked_rate" in markers
    assert "passed" not in markers["stacked_rate"]
    assert "stacked_rate" not in {g["name"] for g in blob["gates"]}
