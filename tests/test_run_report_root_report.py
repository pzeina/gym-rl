"""The digest must print the axis the fleet is now gated on (refs #48).

`closed_on_root_report_rate` — did the COMMANDER close its own operation, or did
the grace window simply expire — is the quantity the v1.20 `root_done_bonus`
default was decided on and the quantity `metrics.regression_gates` now refuses
runs on (floor 0.5). `run_report.py` is "the ONLY thing the big model reads",
and it printed report precision, recall and false-DONE while never printing
this one.

The omission has a measured cost. A handoff note scoped near-mute `ckpt_best` as
something to watch about the CHALLENGER price (`root_done_bonus=1.0`) when the
SHIPPED price does it too: `squad_v10b` files 0 root claims in 100 episodes at
`ckpt_best` against 307 at FINAL — 0.000 against 0.784 on this axis — so the
challenger was charged on an axis the incumbent was not graded on. The two
numbers sit in the same two artifacts the digest already opens; nothing printed
them side by side. This pins that they now are, at both checkpoints, together
with the two ways the rate is read wrong: as a claim count (the defend family
closes on a SITREP and reads ~1.00 on zero claims) and as a zero where it is
merely unmeasured.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.fleet_status import find_run
from scripts.run_report import _BEHAVIOR_ROWS, behavior_block

ROOT = Path(__file__).resolve().parents[1]
KEY = "closed_on_root_report_rate"


def _rate(run_name: str, artifact: str) -> float | None:
    """One committed corpus's rate, resolved rather than hard-pathed, so the pin
    survives the run being filed into ``runs/archive/``."""
    run = find_run(run_name, ROOT / "runs")
    assert run is not None, f"{run_name} is cited by ROADMAP's #48 entry and must resolve"
    return json.loads((run / artifact).read_text())["metrics"].get(KEY)


def test_the_digest_carries_the_gated_axis():
    """It is in the block rows, so it prints under BOTH checkpoints and `--vs`
    files it as a delta under both prefixes — the same treatment
    `human_death_rate` got when #22's pre-registered primary turned out to be
    unreadable from the digest the verdict was written against."""
    assert KEY in [key for key, _, _ in _BEHAVIOR_ROWS]


def test_the_shipped_price_goes_mute_at_ckpt_best_exactly_as_the_challenger_does():
    """The #48 correction, from the committed corpora rather than from the note.

    `squad_v10b` is `root_done_bonus=3.0` — the SHIPPED default, one of the two
    arms the whole pricing comparison is anchored on — and it is as mute at
    `ckpt_best` as any 1.0 seed. Below the 0.5 gate floor at `ckpt_best`: one of
    the two 3.0 seeds, three of four at 0, three of four at 1.0. Whatever else
    separates the prices, this does not.
    """
    from cohort.metrics import ROOT_REPORT_CLOSE_FLOOR

    for run, best, final in (
        ("squad_v10b", 0.000, 0.784),        # rdb=3.0, seed 13 — SHIPPED, and mute
        ("squad_v15b_bonus1", 0.000, 0.866),  # rdb=1.0, seed 13
        ("squad_v15c_bonus1", 0.010, 0.825),  # rdb=1.0, seed 14
        ("squad_v15d_bonus1", 0.000, 0.857),  # rdb=1.0, seed 15
    ):
        assert abs(_rate(run, "behavior.json") - best) < 0.005, run
        assert abs(_rate(run, "behavior_final.json") - final) < 0.005, run
        assert _rate(run, "behavior.json") < ROOT_REPORT_CLOSE_FLOOR, run
        assert _rate(run, "behavior_final.json") >= ROOT_REPORT_CLOSE_FLOOR, run

    # ...and the seeds that do NOT do it, at both prices, so this reads as an
    # instability rather than as a property of either economics.
    for run in ("squad_v10", "squad_v15_bonus1"):
        assert _rate(run, "behavior.json") >= ROOT_REPORT_CLOSE_FLOOR, run


def test_the_rate_is_not_the_root_claim_count():
    """A continuous-posture root closes the window with a SITREP and has MISSION
    COMPLETE masked shut, so the whole defend family reads ~1.00 here on ZERO
    root claims. Reading the claim column as the gate would call two published
    members mute; reading the gate as a claim count would call them talkative."""
    for run in ("fireteam_defend_v20", "defend_brique_v15"):
        for artifact in ("behavior.json", "behavior_final.json"):
            corpus = json.loads(
                (find_run(run, ROOT / "runs") / artifact).read_text()  # type: ignore[operator]
            )["metrics"]
            assert corpus["done_reports_root"] == 0, run
            assert corpus[KEY] > 0.98, run


def test_an_unmeasured_rate_prints_no_row_rather_than_a_zero(tmp_path, capsys):
    """`squad_v8` predates every scenario sending an ENDEX: 0 ENDEXes, so the
    rate has no denominator and the artifact records None. Printing 0.000 would
    publish "this commander never closed an operation" about a corpus that never
    asked the question — the em-dash rule, applied by the block's own
    `is not None` guard rather than by whoever reads the digest."""
    assert _rate("squad_v8", "behavior.json") is None

    artifact = tmp_path / "behavior.json"
    artifact.write_text(json.dumps({
        "checkpoint": "ckpt_best.pt", "episodes": 100, "greedy": False,
        "success_ci95": "0.97 ± 0.03",
        "metrics": {"success_rate": 0.97, KEY: None, "report_recall": 0.83},
        "per_episode": [],
    }))
    summary: dict = {}
    behavior_block(artifact, "behavior", summary, "beh_", diagnostics=False)

    assert "closed on root report" not in capsys.readouterr().out
    assert f"beh_{KEY}" not in summary


def test_both_checkpoints_file_the_rate_so_a_comparison_can_state_it(tmp_path, capsys):
    """The failure this whole entry is about is a best/final split, so a delta
    stated at one checkpoint is a delta between two unstated policies: the block
    files the rate under whichever prefix it was called with."""
    summary: dict = {}
    for prefix, rate in (("beh_", 0.0), ("final_", 0.784)):
        artifact = tmp_path / f"{prefix}behavior.json"
        artifact.write_text(json.dumps({
            "checkpoint": "ckpt.pt", "episodes": 100, "greedy": False,
            "metrics": {"success_rate": 0.9, KEY: rate}, "per_episode": [],
        }))
        behavior_block(artifact, prefix, summary, prefix, diagnostics=False)

    assert summary["beh_" + KEY] == 0.0
    assert summary["final_" + KEY] == 0.784
    assert capsys.readouterr().out.count("closed on root report") == 2
