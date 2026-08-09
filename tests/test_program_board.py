"""The program board must not read a family norm as a finding about one run.

Two threads on the published board described a level every generation of their
scenario shows — false-COMPLETE rates in the 0.7-0.9 band — as if it were a
regression the named run had invented (refs #24). The band mechanism is the
structural answer: a thread that leads with a *level* renders that level's
spread across the scenario's other generations beside it, read off disk, so the
claim can be checked on sight and widens by itself as runs land.

The second hazard pinned here is provenance. A reference row carries a number
that did not come from a run's own behavior file, and the board once printed
one — v12's checkpoint re-scored under the ENDEX rule — as a bare 0.22 naming
neither the checkpoint nor the N, for a metric that moves 2.5x between the two
checkpoints of that one run. A reference must say on its face where it came
from: its own N when something on disk backs it, "quoted" when nothing does.
"""

from __future__ import annotations

import json

import pytest

from scripts import program_board


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """A runs/ tree the board reads instead of the repo's own."""
    monkeypatch.setattr(program_board, "ROOT", tmp_path)
    (tmp_path / "runs").mkdir()

    def write(name: str, *, best: dict | None = None, final: dict | None = None) -> None:
        run = tmp_path / "runs" / name
        run.mkdir(parents=True, exist_ok=True)
        if best is not None:
            (run / "behavior.json").write_text(json.dumps({"episodes": 20, "metrics": best}))
        if final is not None:
            (run / "behavior_final.json").write_text(
                json.dumps({"episodes": 100, "metrics": final})
            )

    return write


def test_the_band_is_the_scenario_s_spread_not_the_run_s_number(runs):
    runs("fireteam_v5", best={"false_complete_rate": 0.88})
    runs("fireteam_v6", best={"false_complete_rate": 0.76})
    runs("fireteam_v8", best={"false_complete_rate": 0.84})

    band = program_board._family(
        {"prefix": "fireteam_v", "metric": "false_complete_rate", "exclude": ("fireteam_v8",)}
    )

    assert (band["lo"], band["hi"], band["n"]) == (0.76, 0.88, 2)
    # the run under discussion is inside its own family's band: that is the
    # whole point of printing it, and it is exactly what the thread now says
    assert band["lo"] <= 0.84 <= band["hi"]


def test_a_sibling_that_never_measured_the_metric_stays_out_of_the_band(runs):
    runs("platoon_v2", best={"false_complete_rate": 0.66})
    runs("platoon_v3", best={"false_complete_rate": None})  # filed no claims at all
    runs("platoon_v4", best={"false_complete_rate": 0.80})

    band = program_board._family({"prefix": "platoon_v", "metric": "false_complete_rate"})

    assert (band["lo"], band["hi"], band["n"]) == (0.66, 0.80, 2)


def test_one_sibling_is_an_anecdote_and_does_not_render_as_a_family(runs):
    runs("squad_v1", best={"false_complete_rate": 0.5})

    assert program_board._family({"prefix": "squad_v", "metric": "false_complete_rate"}) == {}


def test_a_prefix_does_not_reach_into_a_neighbouring_scenario(runs):
    runs("fireteam_v5", best={"false_complete_rate": 0.88})
    runs("fireteam_defend_v9", best={"false_complete_rate": 0.10})
    runs("fireteam_defend_v10", best={"false_complete_rate": 0.20})

    assert program_board._family({"prefix": "fireteam_v", "metric": "false_complete_rate"}) == {}


def test_a_rescored_baseline_names_its_checkpoint_and_its_n(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "fireteam_defend_v12"
    run.mkdir(parents=True)
    (run / "endex_rescore.json").write_text(
        json.dumps(
            {
                "checkpoints": {
                    "ckpt_latest.pt": {
                        "policy": "final",
                        "episodes": 100,
                        "closed_on_root_report_rate": 0.4651,
                    },
                    "ckpt_best.pt": {
                        "policy": "rolling-best",
                        "episodes": 100,
                        "closed_on_root_report_rate": 0.1852,
                    },
                }
            }
        )
    )
    monkeypatch.setattr(program_board, "ENDEX_RESCORE", run / "endex_rescore.json")

    baseline = program_board._endex_baseline()

    assert [r["policy"] for r in baseline] == ["rolling-best", "final"]  # weakest first
    assert all(r["episodes"] == 100 for r in baseline)
    # the row is a read, not a quote: it prints its N and the panel says so
    panel = program_board._panel(
        {
            "cap": "closed on the root's report",
            "metric": "closed_on_root_report_rate",
            "scale": 1.0,
            "runs": [],
            "references": baseline,
        }
    )
    assert "N=100" in panel and "quoted" not in panel
    assert "0.19" in panel and "0.47" in panel


def test_a_reference_with_nothing_on_disk_behind_it_still_reads_as_quoted():
    panel = program_board._panel(
        {
            "cap": "closed on the root's report",
            "metric": "closed_on_root_report_rate",
            "scale": 1.0,
            "runs": [],
            "references": [
                {"label": "some_run", "note": "measured elsewhere", "value": 0.22, "arm": "b"}
            ],
        }
    )

    assert "quoted" in panel and "N=" not in panel


def test_the_baseline_is_empty_when_nothing_on_disk_backs_it(tmp_path, monkeypatch):
    monkeypatch.setattr(program_board, "ENDEX_RESCORE", tmp_path / "absent.json")

    assert program_board._endex_baseline() == []
