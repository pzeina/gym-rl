"""The boards refresh themselves when a run lands — keep that wiring intact.

Two failure modes are pinned here. First, the wiring: if ``train.sh`` stops
going through the wrapper, every board silently goes back to being as stale as
the last time someone remembered. Second, the staleness signal: it has to mean
"what the board SAYS changed", not "the file was rewritten" — a digest that
ticks with a training percentage marks the artifacts stale every 30 seconds and
is then worth nothing.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from scripts import update_boards

ROOT = Path(__file__).resolve().parent.parent


def _row(**over) -> dict:
    row = {
        "run": "fireteam_defend_v12",
        "scenario": "fireteam_defend",
        "success_ci95": "0.86 ± 0.07",
        "episodes": 100,
        "policy": "final",
        "gates_failed": [],
        "overrides": ["defend_survivor_scale=0.35"],
        "env_steps": 3_500_000,
        "obs_dim": 220,
        "loadable": True,
        "state": None,
        # volatile: a live run's progress moves constantly and must not count
        "progress": 12.0,
        "rolling": 0.81,
        "eta": "17m12s",
    }
    row.update(over)
    return row


def test_a_training_run_s_progress_does_not_mark_the_artifacts_stale():
    early = [_row(state="RUNNING", progress=3.0, rolling=0.4, eta="41m")]
    later = [_row(state="RUNNING", progress=88.0, rolling=0.9, eta="4m")]

    assert update_boards.data_digest(early) == update_boards.data_digest(later)


def test_a_commit_alone_does_not_mark_the_artifacts_stale():
    """Same rule as a training percentage, and for the same reason.

    The program board prints the commits-ahead count, so a commit does change a
    corner of the page — but commits are pre-authorised now, and a republish
    flag that fires on every one of them is exactly the noise this digest
    exists to avoid. Fleet content is what the flag tracks.

    Checked at the mechanism, not the symptom: the digest went HEAD-dependent by
    shelling out to `git rev-parse`, so the guard is that it reads nothing but
    its argument. A test that merely called it twice would pass either way.
    """
    rows = [_row()]
    assert update_boards.data_digest(rows) == update_boards.data_digest(rows)

    src = (ROOT / "scripts" / "update_boards.py").read_text()
    body = src[src.index("def data_digest"):src.index("def read_state")]
    for reads_the_world in ("subprocess", "git", "Path(", "open("):
        assert reads_the_world not in body, (
            f"data_digest consults {reads_the_world!r}; it must be a pure "
            "function of the rows, or the publish flag fires on unrelated changes"
        )


def test_a_new_evaluation_does_mark_them_stale():
    before = [_row(success_ci95="0.74 ± 0.09")]
    after = [_row(success_ci95="0.86 ± 0.07")]

    assert update_boards.data_digest(before) != update_boards.data_digest(after)


def test_a_run_starting_or_landing_marks_them_stale():
    training = [_row(state="RUNNING")]
    landed = [_row(state=None)]

    assert update_boards.data_digest(training) != update_boards.data_digest(landed)


def test_a_failed_gate_appearing_marks_them_stale():
    passing = [_row(gates_failed=[])]
    failing = [_row(gates_failed=["mean_distance_from_objective_under_threat"])]

    assert update_boards.data_digest(passing) != update_boards.data_digest(failing)


def test_pending_lists_every_board_whose_published_copy_has_drifted():
    state = {
        "data_sha": "abc123",
        "boards": {
            "fleet": {"published_sha": "abc123"},
            "program": {"published_sha": "older99"},
        },
    }
    assert update_boards.pending(state) == ["program"]

    state["boards"]["fleet"]["published_sha"] = "older99"
    assert sorted(update_boards.pending(state)) == ["fleet", "program"]

    for board in state["boards"].values():
        board["published_sha"] = "abc123"
    assert update_boards.pending(state) == []


def test_a_never_published_board_reads_as_pending():
    state = {"data_sha": "abc123", "boards": {"fleet": {}, "program": {}}}
    assert sorted(update_boards.pending(state)) == ["fleet", "program"]


def test_every_board_the_refresher_writes_has_a_published_url():
    # a board with no URL cannot be updated in place — /boards would mint a new
    # artifact each time and the old link would rot
    for name, board in update_boards.BOARDS.items():
        assert board["url"].startswith("https://claude.ai/code/artifact/"), name
        assert board["path"].endswith(".html"), name


def test_train_sh_still_routes_through_the_board_refreshing_wrapper():
    wrapper = ROOT / "scripts" / "train_then_boards.sh"
    assert wrapper.is_file(), "the wrapper train.sh launches has gone missing"
    assert os.stat(wrapper).st_mode & stat.S_IXUSR, "wrapper is not executable"
    assert "update_boards.py" in wrapper.read_text()

    launch = ROOT / "scripts" / "train.sh"
    text = launch.read_text()
    assert "train_then_boards.sh" in text, (
        "train.sh no longer launches through the wrapper — runs would land "
        "without refreshing the boards"
    )
    assert "nohup" in text, "the launch must stay detached"


def test_the_state_file_records_where_each_board_was_published():
    """The shipped state file is the contract /boards reads; keep it honest."""
    state_path = ROOT / "runs" / ".boards.json"
    if not state_path.exists():  # not yet rendered on a fresh clone
        return
    state = json.loads(state_path.read_text())
    for name in update_boards.BOARDS:
        assert name in state.get("boards", {}), name
        assert state["boards"][name]["url"] == update_boards.BOARDS[name]["url"]


def test_a_manifest_declaration_block_changing_marks_the_artifacts_stale():
    """The spread block landed and the publish flag said "current" (2026-08-18):

    the digest read only the rows, and seed_search/seed_spread change what the
    fleet board says about a scenario without any run moving. The manifest
    blocks are an argument, not a read, so the purity guard above still holds.
    """
    rows = [_row()]
    before = {"version": "v1.21", "seed_search": {"squad": ["a", "b"]}, "seed_spread": None}
    after = dict(before, seed_spread={"squad": ["c"]})

    assert update_boards.data_digest(rows, before) != update_boards.data_digest(rows, after)
    assert update_boards.data_digest(rows, before) == update_boards.data_digest(rows, dict(before))
