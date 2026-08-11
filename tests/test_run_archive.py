"""Archiving a run files it away; it must not make it unreadable.

100 run directories is not a fleet, it is a filing cabinet, and the eight that
constitute the current baseline were invisible in it. So the superseded
generations move to ``runs/archive/`` — the dead observation eras, the answered
experiment arms, the four-in-one-day churn of a family being tuned.

They are moved rather than deleted because they are the evidence behind
published claims: ROADMAP cites `squad_v7`'s collapse, `fireteam_defend_v6`'s
0.51, the D4 pair by name. A citation that stops resolving turns a documented
result into a story about one, which is the failure mode this whole repository
is built against.

So every reader resolves a run name through one place, and that place looks in
both. What is pinned here is that they all do — a new reader that goes back to
``RUNS / name`` would work perfectly until the day something got archived.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import fleet_status, run_report


def _run(root: Path, name: str, *, archived: bool = False, scenario: str = "squad") -> Path:
    d = (root / "archive" / name) if archived else (root / name)
    d.mkdir(parents=True)
    with (d / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iteration", "env_steps", "success_rate_rolling"])
        w.writeheader()
        for i in range(25):
            w.writerow({"iteration": i, "env_steps": i * 100, "success_rate_rolling": 0.9})
    (d / "config.json").write_text(json.dumps({"scenario": scenario}))
    (d / "economics.json").write_text(json.dumps({"git_commit": "a" * 40, "rewards": {}}))
    (d / "behavior.json").write_text(json.dumps(
        {"episodes": 100, "success_ci95": "0.90 ± 0.06", "gates": [], "metrics": {}}))
    (d / "ckpt_best.pt").write_text("stub")
    return d


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    _run(tmp_path, "squad_v10")
    _run(tmp_path, "squad_v5", archived=True)
    monkeypatch.setattr(run_report, "RUNS", tmp_path)
    return tmp_path


def test_run_dirs_walks_the_archive_as_well_as_the_fleet(fleet):
    names = [d.name for d in fleet_status.run_dirs(fleet)]
    assert names == ["squad_v10", "squad_v5"]


def test_the_archive_directory_is_not_itself_a_run(fleet):
    """``runs/archive`` has no metrics.csv, and must not be walked as a member."""
    assert "archive" not in [d.name for d in fleet_status.run_dirs(fleet)]


def test_a_row_says_whether_it_is_archived(fleet, monkeypatch):
    from cohort.viz import dashboard

    monkeypatch.setattr(dashboard, "checkpoint_meta",
                        lambda p: {"loadable": True, "obs_dim": 220, "env_steps": 1, "reason": ""})
    rows = {r["run"]: r for r in fleet_status.collect(fleet)}

    assert rows["squad_v10"]["archived"] is False
    assert rows["squad_v5"]["archived"] is True


def test_a_report_still_resolves_an_archived_run(fleet):
    """The citation guarantee: `run_report.py squad_v5` works after the move."""
    assert run_report.run_dir("squad_v5") == fleet / "archive" / "squad_v5"
    assert len(run_report.rows_of("squad_v5")) == 25


def test_a_current_run_is_preferred_over_a_same_named_archived_one(fleet):
    _run(fleet, "squad_v5")  # a name reused after archiving — current wins
    assert run_report.run_dir("squad_v5") == fleet / "squad_v5"


def test_an_absent_run_still_resolves_to_the_current_path_for_the_error_message(fleet):
    """A miss must name ``runs/x``, not ``runs/archive/x`` — the former is where
    the reader will look, and the latter would send them somewhere confusing."""
    assert run_report.run_dir("squad_v99") == fleet / "squad_v99"


def test_archiving_never_touches_a_manifest_run_or_a_live_one(fleet, monkeypatch):
    """The two refusals that matter: the fleet itself, and anything still writing.

    A run that is training has a process appending to its metrics.csv and its
    checkpoints; moving that directory out from under it corrupts the run and
    the campaign queue waiting on it.
    """
    from scripts import archive_runs, baseline, train_status

    _run(fleet, "squad_v9")
    _run(fleet, "platoon_v6")
    monkeypatch.setattr(archive_runs, "RUNS", fleet)
    monkeypatch.setattr(baseline, "MANIFEST", fleet / "BASELINE.json")
    (fleet / "BASELINE.json").write_text(json.dumps({
        "runs": {"squad": "squad_v10"},
        "referenced_history": {"squad_v5": "the arm the ablation rests on"},
    }))
    monkeypatch.setattr(
        train_status, "summarize",
        lambda d: {"state": "RUNNING" if d.name == "platoon_v6" else "DONE"},
    )
    monkeypatch.setattr(archive_runs, "summarize", train_status.summarize)

    moving, keep, live = archive_runs.candidates()

    assert sorted(keep) == ["squad_v10", "squad_v5"]
    assert live == ["platoon_v6"]
    assert [d.name for d in moving] == ["squad_v9"]


def test_every_reader_goes_through_the_resolver():
    """The guard that keeps this true.

    A new reader written as ``RUNS / name / "behavior.json"`` passes its own
    tests and silently stops seeing archived runs. There is one legitimate
    direct use — ``runs/.boards.json``, which is fleet state rather than a run —
    so the check is for the run-name pattern specifically.
    """
    repo = Path(__file__).resolve().parents[1]
    offenders = []
    # tests/ as well as scripts/. Six data-level invariants in
    # test_checkpoint_provenance.py and one in test_confirmed_claim_is_last.py
    # SKIPPED the moment 96 runs were filed away, because they resolved
    # `runs/<name>` directly — and a skip is not a pass. The guard scanning only
    # scripts/ is exactly why nobody noticed until the archive happened.
    sources = sorted(repo.glob("scripts/*.py")) + sorted(repo.glob("tests/*.py"))
    for path in sources:
        if path.name in {"fleet_status.py", "run_report.py", "test_run_archive.py"}:
            continue  # where the resolvers themselves live, and this guard
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "find_run(" in stripped:
                continue  # a fallback INSIDE a resolver is the resolver
            if "not-archive-aware:" in stripped:
                continue  # deliberate, and the line has to say why
            # Two shapes of the hazard, and only two. `RUNS / <var>` is the
            # module-level constant indexed by a run name. `ROOT / "runs" /` is
            # the repo's real runs directory reached by hand — as opposed to
            # `tmp_path / "runs"`, which is a fixture and reads nothing real.
            # Fleet state (.boards.json, BASELINE.json) is not a run and is
            # allowed to be addressed directly.
            # Not runs: fleet state, and the boards' own output files.
            fleet_state = ('.boards.json', 'BASELINE.json', '.html',
                           '_no_such_run_for_tests')
            hazard = (
                any(h in stripped for h in ('RUNS / run', 'RUNS / name', 'RUNS / a['))
                or ('ROOT / "runs" /' in stripped
                    and not any(f in stripped for f in fleet_state))
            )
            if hazard:
                offenders.append(f"{path.name}:{i}: {stripped}")
    assert not offenders, (
        "these read a run directory without the archive-aware resolver:\n  "
        + "\n  ".join(offenders)
    )
