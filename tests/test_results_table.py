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


def _campaign_in_flight() -> list[str]:
    """Baseline members whose training is live right now."""
    from scripts.train_status import summarize

    live = []
    for run in baseline.load().get("runs", {}).values():
        d = baseline.run_dir(run)
        if d.is_dir() and summarize(d).get("state") == "RUNNING":
            live.append(run)
    return live


def test_the_readme_table_matches_the_runs_on_disk():
    """Strict — but only when the fleet is at rest.

    The first version of this asserted unconditionally, and went red for the
    whole of a retrain campaign: each member landing changes the table, and the
    README cannot be regenerated into a state that stays true for more than the
    minutes until the next one lands. Hours of red is not a stale-README signal,
    it is a suite people learn to ignore — and it blocks the "never commit on
    red" rule every agent in this repo works under.

    Staleness during a campaign is expected and is somebody's job in progress.
    Staleness at rest is the defect this test exists for, and the gate that
    catches an unfinished campaign is scripts/baseline.py, which fails while any
    member is unscored.
    """
    live = _campaign_in_flight()
    if live:
        pytest.skip(f"campaign in flight ({', '.join(sorted(live))}) — the table is "
                    "regenerated when the fleet lands; scripts/baseline.py is the gate "
                    "that a campaign is finished")

    text = README.read_text()
    assert results_table.START in text and results_table.END in text, (
        "README.md has no generated-table markers — run scripts/results_table.py --write"
    )
    published = text.split(results_table.START, 1)[1].split(results_table.END, 1)[0].strip()
    assert published == results_table.table().strip(), (
        "README.md's baseline table no longer matches the committed evaluations.\n"
        "Regenerate it with:  scripts/results_table.py --write"
    )


def test_the_staleness_check_only_stands_down_for_a_live_campaign(monkeypatch):
    """The exemption must be narrow, or it becomes a way to never run the check.

    Pinned at the mechanism: with nothing training, the strict path runs. A
    future edit that widened the skip to "any run anywhere is training", or that
    left it permanently on, fails here.
    """
    from scripts import train_status

    monkeypatch.setattr(train_status, "summarize", lambda d: {"state": "DONE"})
    assert _campaign_in_flight() == []

    monkeypatch.setattr(train_status, "summarize", lambda d: {"state": "RUNNING"})
    members = set(baseline.load()["runs"].values())
    assert set(_campaign_in_flight()) <= members, "the check must look only at members"


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


def _patch_metrics(member, **metrics):
    path = member / "squad_v10" / "behavior_final.json"
    data = json.loads(path.read_text())
    data["metrics"].update(metrics)
    path.write_text(json.dumps(data))


def _root_reported_cell(member) -> str:
    return [c.strip() for c in
            results_table.row("squad", "squad_v10").strip("|").split("|")][8]


def test_a_mute_completable_root_prints_its_measured_zero(member):
    """Issue #62, correcting the guard #61 prescribed. On a completable root
    (SEIZE/RECON/…) a DONE claim is the only counted close, so zero claims
    *entails* a rate of exactly 0.000 — that zero is the mute-root finding,
    the one the seed-carry work is about, and 26 SEIZE evaluations of it were
    being replaced by a dash. A measured zero must print as a zero."""
    _patch_metrics(member, root_mission="SEIZE",
                   closed_on_root_report_rate=0.0, done_claim_episodes_root=0)

    cell = _root_reported_cell(member)

    assert cell == "0%"
    assert results_table.NO_CLAIMS not in cell


def test_a_defend_root_prints_the_sitrep_route_not_a_dash(member):
    """On a DEFEND/DENY root the DONE claim is masked shut by doctrine, so
    zero claims is a capability fact, not a behaviour — the SITREP route is
    that root's only completion channel and the rate measures it. The cell
    marks the route and quotes `closes_per_root_sitrep`, so a reader can tell
    timed reports (`fireteam_defend_v23`, 0.035/sitrep: volume) from timing."""
    _patch_metrics(member, root_mission="DEFEND",
                   closed_on_root_report_rate=1.0, done_claim_episodes_root=0,
                   closes_per_root_sitrep=0.035)

    assert _root_reported_cell(member) == "100% (sitrep, 0.035/sitrep)"


def test_the_two_defend_ends_of_the_metric_print_differently(member):
    """The collapse #62 caught in one line: `fireteam_defend_v23` closes on
    the root's report in every episode, `fireteam_defend_v18` in none — same
    mission, same capability, opposite behaviour — and the #61 dash printed
    the same cell for both. The column exists to measure that difference."""
    _patch_metrics(member, root_mission="DEFEND",
                   closed_on_root_report_rate=1.0, done_claim_episodes_root=0,
                   closes_per_root_sitrep=0.035)
    at_one = _root_reported_cell(member)

    _patch_metrics(member, closed_on_root_report_rate=0.0,
                   closes_per_root_sitrep=0.0)
    at_zero = _root_reported_cell(member)

    assert at_one != at_zero
    assert at_zero.startswith("0%")
    assert results_table.NO_CLAIMS not in (at_one, at_zero)


def test_a_defend_root_without_the_per_sitrep_ratio_still_marks_the_route(member):
    _patch_metrics(member, root_mission="DEFEND",
                   closed_on_root_report_rate=1.0, done_claim_episodes_root=0)

    assert _root_reported_cell(member) == "100% (sitrep)"


def test_an_unclassifiable_claimless_rate_keeps_the_dash(member):
    """The #61 guard survives only where the discriminator is unavailable: an
    evaluation that recorded no `root_mission` cannot say which route fed the
    rate, so a claimless 1.0 there still reads as a floor, not a measurement."""
    _patch_metrics(member, closed_on_root_report_rate=1.0,
                   done_claim_episodes_root=0)

    cell = _root_reported_cell(member)

    assert cell == results_table.NO_CLAIMS
    assert "100%" not in cell


def test_a_claiming_root_keeps_its_measured_rate(member):
    """With DONE claims on the record the rate is a measurement and must
    survive — the gate (`ROOT_REPORT_CLOSE_FLOOR`) keeps doing its job on the
    arms that do claim."""
    _patch_metrics(member, root_mission="SEIZE",
                   closed_on_root_report_rate=1.0, done_claim_episodes_root=96)

    assert _root_reported_cell(member) == "100%"


def test_an_evaluation_predating_the_claim_counter_keeps_its_rate(member):
    """Absence of `done_claim_episodes_root` is an old evaluation, not a mute
    root — the fixture omits the key and the rate must still print."""
    cells = [c.strip() for c in results_table.row("squad", "squad_v10").strip("|").split("|")]

    assert cells[8] == "62%"


def test_each_footnote_travels_with_its_cell_and_only_with_it(member, monkeypatch):
    """A note explains a marking, so it must appear when the marking does and
    stay out of the table when every cell is a plain measurement. And the
    sitrep note must not assert a mechanism for rows it does not cover — the
    #61 note claimed "every close came from a SITREP" under a table whose
    SEIZE rows forbid that route (issue #62)."""
    monkeypatch.setattr(baseline, "load",
                        lambda: {"runs": {"squad": "squad_v10"}})
    monkeypatch.setattr(baseline, "DOCTRINE_SCENARIOS", ["squad"])

    body = results_table.table()
    assert results_table.NO_CLAIMS not in body
    assert "(sitrep" not in body

    _patch_metrics(member, root_mission="DEFEND",
                   closed_on_root_report_rate=1.0, done_claim_episodes_root=0,
                   closes_per_root_sitrep=0.035)
    body = results_table.table()
    assert "(sitrep" in body
    assert "closes per root SITREP" in body  # the marking explains itself
    assert results_table.NO_CLAIMS not in body

    _patch_metrics(member, root_mission="(unrecorded)")
    body = results_table.table()
    assert results_table.NO_CLAIMS in body
    assert "no DONE claim" in body  # the dash explains itself


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
