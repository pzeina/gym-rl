"""An owner-decided exception is loud, validated, and re-verified — or it fails.

The pre-v1.19 fleet shipped one member "with a flag saying it did not clear the
bar", and the flag was the problem: nothing rendered it, nothing re-checked it,
and the fleet's own OK line contradicted it. v1.22 ships `platoon_hard` with a
FAILing reporting gate by explicit owner decision (2026-08-21), so the manifest
grew a first-class `exceptions` block. These tests pin the properties that keep
"disclosed" from rotting back into "assumed":

* an exception must carry a decision date, a decider, and reasons that are
  claims rather than placeholders;
* a waived gate still renders as a FAIL everywhere a reader looks (the README
  cell is bold and names the gate);
* a waiver whose gate no longer fails is itself a failure — dead waivers are
  removed, not accumulated;
* a tree waiver pins the member's exact cohort/ tree, and the pin moving is
  drift, not an exception.
"""

from __future__ import annotations

import json

import pytest

from scripts import baseline, results_table


def test_the_shipped_exceptions_block_is_valid_and_names_members():
    manifest = baseline.load()
    excs = baseline.exceptions(manifest)  # raises SystemExit if malformed
    for scenario in excs:
        assert scenario in manifest["runs"]


def test_an_exception_without_a_reason_is_refused():
    manifest = {
        "runs": {"squad": "squad_m"},
        "exceptions": {"squad": {"decided": "2026-08-21", "by": "owner",
                                 "waives": {"gate:some_gate": "too short"}}},
    }
    with pytest.raises(SystemExit, match="placeholder"):
        baseline.exceptions(manifest)


def test_an_exception_that_waives_nothing_is_refused():
    manifest = {
        "runs": {"squad": "squad_m"},
        "exceptions": {"squad": {"decided": "2026-08-21", "by": "owner", "waives": {}}},
    }
    with pytest.raises(SystemExit, match="waives nothing"):
        baseline.exceptions(manifest)


def test_a_tree_waiver_without_a_pin_is_refused():
    manifest = {
        "runs": {"squad": "squad_m"},
        "exceptions": {"squad": {
            "decided": "2026-08-21", "by": "owner",
            "waives": {"provenance:cohort_tree": "a reason long enough to be a real claim here"},
        }},
    }
    with pytest.raises(SystemExit, match="member_tree"):
        baseline.exceptions(manifest)


def test_a_waived_gate_is_absorbed_and_a_dead_waiver_is_a_problem(tmp_path, monkeypatch):
    """The two directions of the same guard, on one synthetic member."""
    d = tmp_path / "squad_m"
    d.mkdir()
    (d / "economics.json").write_text(json.dumps(
        {"git_commit": "a" * 40, "reward_overrides": []}))
    (d / "behavior_final.json").write_text(json.dumps({
        "episodes": 100,
        "metrics": {"success_rate": 0.9, "successes": 90, "successes_announced": 90},
        "gates": [{"name": "closed_on_root_report_rate", "passed": False},
                  {"name": "success_rate", "passed": True}],
    }))
    (d / "behavior.json").write_text(json.dumps({"episodes": 100}))
    monkeypatch.setattr(baseline, "RUNS", tmp_path)

    facts = baseline._run_facts(
        "squad_m", waived_gates=("closed_on_root_report_rate",))
    assert facts["waived_gates"] == ["closed_on_root_report_rate"]
    assert "closed_on_root_report_rate" not in facts["gates_failed"]
    assert not any("gate failed" in p for p in facts["problems"])

    # same member, waiver for a gate that PASSES: the dead waiver is a problem
    facts = baseline._run_facts("squad_m", waived_gates=("success_rate",))
    assert any("dead waiver" in p for p in facts["problems"])
    # and the real FAIL, being unwaived, still fails
    assert any("gate failed: closed_on_root_report_rate" in p for p in facts["problems"])


def test_an_unwaived_gate_failure_still_fails(tmp_path, monkeypatch):
    d = tmp_path / "squad_m"
    d.mkdir()
    (d / "economics.json").write_text(json.dumps(
        {"git_commit": "a" * 40, "reward_overrides": []}))
    (d / "behavior_final.json").write_text(json.dumps({
        "episodes": 100,
        "metrics": {"success_rate": 0.9, "successes": 90, "successes_announced": 90},
        "gates": [{"name": "closed_on_root_report_rate", "passed": False}],
    }))
    (d / "behavior.json").write_text(json.dumps({"episodes": 100}))
    monkeypatch.setattr(baseline, "RUNS", tmp_path)
    facts = baseline._run_facts("squad_m")
    assert any("gate failed: closed_on_root_report_rate" in p for p in facts["problems"])


def test_the_readme_renders_the_waived_gate_as_a_fail():
    """The README cell for the shipped exception is bold, names the gate, and
    names the decision — the reader sees the FAIL, never a softened pass."""
    manifest = baseline.load()
    excs = manifest.get("exceptions") or {}
    if "platoon_hard" not in excs:
        pytest.skip("no shipped exception to render")
    cell = results_table.row("platoon_hard", manifest["runs"]["platoon_hard"],
                             excs["platoon_hard"])
    assert "**FAIL: closed_on_root_report_rate**" in cell
    assert "owner decision 2026-08-21" in cell


def test_the_sealed_manifest_pins_the_excepted_members_tree():
    manifest = baseline.load()
    for scenario, exc in (manifest.get("exceptions") or {}).items():
        if "provenance:cohort_tree" not in (exc.get("waives") or {}):
            continue
        run = manifest["runs"][scenario]
        econ = json.loads((baseline.run_dir(run) / "economics.json").read_text())
        resolved = baseline.cohort_tree(econ.get("git_commit"))
        assert resolved == exc["member_tree"], (
            f"{run}: the excepted member's cohort/ tree moved off its pin — "
            "that is drift, not an exception")
