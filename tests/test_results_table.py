"""The README's results table is generated, and this fails when it drifts.

Every published overstatement this repository has had to correct was a
transcription, not a lie: an N=20 row under an "N=100" caption, an em dash in
the announced column standing in for a zero, a headline read off `ckpt_best`
while the sentence beside it said "the policy this run ended with". Each was
caught by a person re-reading an artifact, sometimes cycles later.

So the table is generated from ``runs/BASELINE.json`` and the members' committed
evaluations, and this test asserts that what is in README.md is what
``scripts/results_table.py`` produces today. Drift becomes a red test instead of
a correction three sessions later.

Also pinned: the columns that make the table honest rather than flattering.
Success alone cannot be the whole row — a cohort can win every episode over its
commander's body — so root death and timeout travel with it, the peak is
labelled as a peak, and the announcement column keeps its denominator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import baseline, results_table

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_the_readme_table_matches_the_runs_on_disk():
    text = README.read_text()
    assert results_table.START in text and results_table.END in text, (
        "README.md has no generated-table markers — run scripts/results_table.py --write"
    )
    published = text.split(results_table.START, 1)[1].split(results_table.END, 1)[0].strip()
    assert published == results_table.table().strip(), (
        "README.md's baseline table no longer matches the committed evaluations.\n"
        "Regenerate it with:  scripts/results_table.py --write"
    )


def test_every_baseline_member_has_a_row():
    body = results_table.table()
    for scenario in baseline.DOCTRINE_SCENARIOS:
        assert f"`{scenario}`" in body, f"{scenario} is missing from the table"


def test_the_columns_that_keep_it_honest_are_present():
    header = results_table.HEADER
    for column in ("success (final, N)", "peak (best ckpt)", "give-back",
                   "root death", "timeout", "announced", "root-reported", "gates"):
        assert column in header, f"the {column!r} column has been dropped"


@pytest.fixture
def member(tmp_path, monkeypatch):
    """One fully-evaluated member, so the cell formatting can be checked."""
    run = tmp_path / "squad_v10"
    run.mkdir()
    (run / "behavior_final.json").write_text(json.dumps({
        "episodes": 100,
        "success_ci95": "0.97 ± 0.03",
        "metrics": {"success_rate": 0.97, "successes": 97, "successes_announced": 97,
                    "human_death_rate": 0.15, "timeout_rate": 0.0,
                    "closed_on_root_report_rate": 0.62},
        "gates": [{"name": "timeout_rate", "passed": True}],
    }))
    (run / "behavior.json").write_text(json.dumps({
        "episodes": 100, "success_ci95": "0.98 ± 0.03", "metrics": {}, "gates": []}))
    monkeypatch.setattr(baseline, "RUNS", tmp_path)
    monkeypatch.setattr(results_table, "audit_run", lambda d: {"gap": 2.4})
    return tmp_path


def test_a_row_carries_the_evidence_not_just_the_headline(member):
    cells = [c.strip() for c in results_table.row("squad", "squad_v10").strip("|").split("|")]

    assert cells[2] == "0.97 ± 0.03 (N=100)"     # final policy, with its N
    assert cells[3] == "0.98 ± 0.03 (N=100)"     # the peak, labelled by the column
    assert cells[4] == "2.4 pt"                  # give-back
    assert cells[5] == "15%"                     # root death
    assert cells[7] == "97/97"                   # announced, WITH its denominator
    assert cells[8] == "62%"                     # and what the root closed itself
    assert cells[9] == "pass"


def test_an_unannounced_win_is_visible_as_a_fraction_not_a_dash(member, monkeypatch):
    """The exact failure the README had: a `—` where a zero belonged.

    `platoon_v5` announced 0 of 100 wins and the column read as an em dash, so
    the gap survived three cycles of review. A zero must look like a zero.
    """
    path = member / "squad_v10" / "behavior_final.json"
    data = json.loads(path.read_text())
    data["metrics"]["successes_announced"] = 0
    path.write_text(json.dumps(data))

    cells = [c.strip() for c in results_table.row("squad", "squad_v10").strip("|").split("|")]

    assert cells[7] == "0/97"


def test_a_failed_gate_is_named_in_the_row(member):
    path = member / "squad_v10" / "behavior_final.json"
    data = json.loads(path.read_text())
    data["gates"] = [{"name": "timeout_rate", "passed": False}]
    path.write_text(json.dumps(data))

    assert "**timeout_rate**" in results_table.row("squad", "squad_v10")


def test_an_unevaluated_member_says_so_rather_than_going_blank(member):
    (member / "squad_v10" / "behavior_final.json").unlink()

    assert "not yet evaluated" in results_table.row("squad", "squad_v10")


def test_splice_replaces_the_block_instead_of_appending_a_second_one():
    readme = f"# t\n\n{results_table.START}\nold\n{results_table.END}\n\ntail\n"
    out = results_table.splice(readme, "new")

    assert out.count(results_table.START) == 1
    assert "old" not in out and "new" in out
    assert out.endswith("tail\n"), "content after the block must survive"
