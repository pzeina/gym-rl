"""The baseline is a set with rules, and the rules are checkable.

Eight champions at seven commits, four of them reproducible only with a
``--reward`` override, one published with a flag saying it missed the bar — that
was the fleet before v1.19. Every number in it was honest and the set was still
not a system, because nothing anywhere asserted that its members belonged
together.

``scripts/baseline.py`` is that assertion. What is pinned here is the part that
rots quietest:

* a scenario added to ``cohort/config.py`` is either a baseline member or an
  explicitly excused non-member — silence is a failure, not a default;
* the manifest names a run for every doctrine scenario and nothing else;
* each individual check actually fails when violated. A gate nobody has seen
  fail is a gate nobody knows works, and this one is meant to be the last thing
  standing between an inconsistent fleet and a published claim.
"""

from __future__ import annotations

import json

import pytest

from cohort.config import SCENARIOS
from scripts import baseline


def test_every_scenario_is_either_in_the_baseline_or_excused():
    """The coverage guard. A new scenario cannot escape the fleet by silence."""
    accounted = set(baseline.DOCTRINE_SCENARIOS) | set(baseline.NOT_BASELINE)
    unaccounted = sorted(set(SCENARIOS) - accounted)
    assert not unaccounted, (
        f"{unaccounted} is neither a baseline member nor listed in NOT_BASELINE "
        "with a reason — decide which it is"
    )


def test_the_excusals_are_reasons_not_placeholders():
    for scenario, reason in baseline.NOT_BASELINE.items():
        assert scenario in SCENARIOS, f"{scenario} is excused but does not exist"
        assert len(reason) > 20, f"{scenario}'s excusal is not a reason"


def test_the_shipped_manifest_covers_the_doctrine_scenarios():
    manifest = baseline.load()
    assert set(manifest["runs"]) == set(baseline.DOCTRINE_SCENARIOS)
    for scenario, run in manifest["runs"].items():
        assert run.startswith(scenario), f"{run} does not look like a {scenario} run"


def _member(tmp_path, run: str, *, commit="a" * 40, overrides=(), episodes=100,
            gates=(), successes=97, announced=97):
    d = tmp_path / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "economics.json").write_text(json.dumps({
        "git_commit": commit, "reward_overrides": list(overrides),
    }))
    (d / "behavior_final.json").write_text(json.dumps({
        "episodes": episodes,
        "success_ci95": "0.97 ± 0.03",
        "metrics": {"success_rate": 0.97, "successes": successes,
                    "successes_announced": announced},
        "gates": [{"name": g, "passed": False} for g in gates],
    }))
    return d


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    """A manifest whose members all pass, so each test can break exactly one."""
    runs = {s: f"{s}_v1" for s in baseline.DOCTRINE_SCENARIOS}
    (tmp_path / "BASELINE.json").write_text(json.dumps({"version": "test", "runs": runs}))
    for run in runs.values():
        _member(tmp_path, run)
    monkeypatch.setattr(baseline, "RUNS", tmp_path)
    monkeypatch.setattr(baseline, "MANIFEST", tmp_path / "BASELINE.json")
    monkeypatch.setattr(baseline, "_loadable", lambda run: True)
    # commit -> cohort/ tree. "a"*40 and "t"*40 are the same environment a
    # tooling commit apart; "b"*40 is a different one.
    monkeypatch.setattr(baseline, "cohort_tree",
                        lambda c: {"a" * 40: "env1", "t" * 40: "env1"}.get(c, "env2"))
    # audit_run reads metrics.csv for the give-back; absent here, so no gap
    monkeypatch.setattr("scripts.publish_audit.audit_run", lambda d: None)
    return tmp_path


def _audit(capsys) -> tuple[int, str]:
    code = baseline.audit()
    return code, capsys.readouterr().out


def test_a_complete_consistent_fleet_passes(fleet, capsys):
    code, out = _audit(capsys)
    assert code == 0, out
    assert "BASELINE OK" in out


def test_a_second_environment_fails_the_fleet(fleet, capsys):
    """The check the old fleet could not have passed: eight runs, seven commits."""
    _member(fleet, "platoon_v1", commit="b" * 40)

    code, out = _audit(capsys)

    assert code == 1
    assert "2 distinct cohort/ trees" in out
    assert "not one environment" in out.lower()


def test_a_tooling_commit_between_launches_does_not_fail_the_fleet(fleet, capsys):
    """The lesson from this campaign's own first hour.

    `fireteam_v9` was launched three commits after its lane-mates, and all three
    were tooling — scripts, tests, a README table. The `cohort/` tree was
    byte-identical across every one of them, so the runs trained in the same
    environment. A commit-equality gate would have failed the fleet for a reason
    that has nothing to do with the runs, which is how a gate teaches people to
    ignore it.
    """
    _member(fleet, "platoon_v1", commit="t" * 40)

    code, out = _audit(capsys)

    assert code == 0, out
    assert "one environment" in out
    assert "tooling-only differences are expected" in out


def test_a_commit_this_clone_cannot_resolve_is_not_agreement(fleet, capsys, monkeypatch):
    """Unknown provenance must fail, not pass quietly."""
    monkeypatch.setattr(baseline, "cohort_tree",
                        lambda c: None if c == "b" * 40 else "env1")
    _member(fleet, "platoon_v1", commit="b" * 40)

    code, out = _audit(capsys)

    assert code == 1
    assert "cannot resolve cohort/ for platoon_v1" in out


def test_a_reward_override_fails_the_fleet(fleet, capsys):
    """What ships is what was trained — an override means those differ."""
    _member(fleet, "defend_brique_v1", overrides=["defend_survivor_scale=0.35"])

    code, out = _audit(capsys)

    assert code == 1
    assert "reward overrides: defend_survivor_scale=0.35" in out


def test_a_smoke_test_sized_evaluation_fails_the_fleet(fleet, capsys):
    _member(fleet, "squad_v1", episodes=20)

    code, out = _audit(capsys)

    assert code == 1
    assert "N=20, needs 100" in out


def test_a_failed_regression_gate_fails_the_fleet(fleet, capsys):
    _member(fleet, "defend_brique_v1", gates=["mean_distance_from_objective_under_threat"])

    code, out = _audit(capsys)

    assert code == 1
    assert "gate failed: mean_distance_from_objective_under_threat" in out


def test_an_unannounced_win_fails_the_fleet(fleet, capsys):
    """The v1.19 guarantee is structural, so a miss is a broken protocol.

    This is the check that would have caught platoon_v5's 0/100 at the fleet
    level rather than in a README column read three cycles later.
    """
    _member(fleet, "patrol_brique_v1", successes=99, announced=0)

    code, out = _audit(capsys)

    assert code == 1
    assert "announced 0/99" in out
    assert "guarantee is broken" in out


def test_a_missing_member_fails_the_fleet(fleet, capsys):
    manifest = json.loads((fleet / "BASELINE.json").read_text())
    del manifest["runs"]["platoon"]
    (fleet / "BASELINE.json").write_text(json.dumps(manifest))

    code, out = _audit(capsys)

    assert code == 1
    assert "coverage: no member for platoon" in out


def test_sealing_refuses_a_fleet_that_is_not_one_environment(fleet, capsys):
    _member(fleet, "platoon_v1", commit="b" * 40)

    assert baseline.seal("v1.19") == 1
    assert "refusing to seal" in capsys.readouterr().out


def test_sealing_records_the_environment_and_every_commit_in_it(fleet, capsys):
    _member(fleet, "platoon_v1", commit="t" * 40)  # same environment, later commit

    assert baseline.seal("v1.19") == 0

    manifest = json.loads((fleet / "BASELINE.json").read_text())
    assert manifest["cohort_tree"] == "env1"
    assert manifest["commits"] == sorted(["a" * 40, "t" * 40])
    assert manifest["commit"] is None, "two commits: there is no single one to name"
    assert manifest["version"] == "v1.19"


def test_a_sealed_manifest_detects_a_member_swapped_underneath_it(fleet, capsys):
    """Sealing is not a rubber stamp: it must notice the fleet moving after it."""
    baseline.seal("v1.19")
    for run in json.loads((fleet / "BASELINE.json").read_text())["runs"].values():
        _member(fleet, run, commit="c" * 40)  # a different environment entirely

    code, out = _audit(capsys)

    assert code == 1
    assert "sealed at cohort/ env1" in out


def test_an_uncommitted_final_policy_fails_the_fleet(fleet, capsys, monkeypatch):
    """Issue #44: a headline whose weights are not in the repository.

    `.gitignore` ignored `ckpt_latest.pt` fleet-wide, so every member shipped
    the number and withheld the policy that produces it. Nothing failed —
    `behavior_final.json` was present and complete, the gates were green, and
    the one artifact a reader needs to re-derive the figure was absent. That
    silence is why this is a gate and not a note.
    """
    monkeypatch.setattr(
        baseline, "_uncommitted",
        lambda run: ["ckpt_latest.pt"] if run == "squad_v1" else [],
    )

    code, out = _audit(capsys)

    assert code == 1
    assert "squad_v1: ckpt_latest.pt is not committed" in out
    assert "cannot re-derive it" in out


def test_a_tree_with_no_git_index_is_not_an_accusation(fleet):
    """The real `_uncommitted` against a directory git knows nothing about.

    A tarball export has no index, so it cannot distinguish "not committed"
    from "cannot tell" — and a gate that fires there fires for a reason that
    has nothing to do with the fleet, which is how a gate teaches people to
    ignore it. Silence is the only honest answer.
    """
    d = fleet / "squad_v1"
    (d / "ckpt_best.pt").write_text("stub")
    (d / "ckpt_latest.pt").write_text("stub")

    assert baseline._uncommitted("squad_v1") == []


def test_the_headline_checkpoint_is_the_final_policy_not_the_best_window():
    """What `_loadable` and `_uncommitted` are about, named once.

    The audit's `evidence` rule is explicit that the headline is the policy the
    run ended with. `_loadable` used to check `ckpt_best.pt` alone — the one
    checkpoint that rule says is not it.
    """
    assert baseline.HEADLINE_CKPT == "ckpt_latest.pt"
    assert set(baseline.CHECKPOINTS) == {"ckpt_best.pt", "ckpt_latest.pt"}
