"""Replaying ``best_save_gate`` off metrics.csv, and the shape #57 found with it.

The reader under test exists because `patrol_brique_v19_rdb3_seed13` ships a
``ckpt_best`` that succeeds in 17 episodes of 100 against its final policy's 99.
Three things have to hold for that observation to be worth anything, and each is
pinned here:

* **the replay is the shipped rule, not a copy of it** — it calls
  ``best_save_gate``, so a change to selection changes the reader;
* **a corpus that cannot be replayed says so** — the pre-``n_episodes`` runs
  cannot have ``episodes_seen`` reconstructed, and reporting them as "never
  saved" would invent a finding out of a missing column;
* **the mechanism reproduces synthetically** — a thin, early reporting window
  locks the checkpoint against a policy that later wins almost every episode,
  and a success floor is what releases it.

The last one is the regression hazard. It is written as a table rather than
against the real run so it keeps meaning after the corpus is archived.
"""

from __future__ import annotations

import json

import pytest

from cohort.metrics import ROOT_REPORT_CLOSE_FLOOR
from scripts import checkpoint_selection as cs


def _rows(spec, *, reporting_column=True, episodes_column=True):
    """A metrics.csv as rows: ``(n_episodes, success_rolling, root_close_rolling)``."""
    out = []
    for i, (n_episodes, success, close) in enumerate(spec, start=1):
        row = {
            "iteration": str(i),
            "env_steps": str(i * 1024),
            "success_rate_rolling": f"{success:.5f}",
            "false_complete_rate": "0.0",
        }
        if episodes_column:
            row["n_episodes"] = str(n_episodes)
        if reporting_column:
            row["root_report_close_rolling"] = "nan" if close is None else f"{close:.5f}"
        out.append(row)
    return out


# --------------------------------------------------------------- the replay --


def test_episodes_seen_is_the_running_sum_the_trainer_keeps():
    rows = _rows([(40, 0.1, None), (35, 0.2, None), (30, 0.3, 0.5)])
    assert [seen for seen, _, _ in cs.episodes_and_windows(rows)] == [40, 75, 105]


def test_an_unmeasured_reporting_window_reads_as_none_not_zero():
    """``root_report_close_rolling`` is NaN until an ENDEX lands in the window.

    The gate's contract is that unmeasured is not-reporting *and not a refusal*;
    parsing NaN as 0.0 would preserve that by accident and then break the moment
    the floor moved off zero.
    """
    rows = _rows([(100, 0.5, None)])
    assert cs.episodes_and_windows(rows)[0][2] is None


def test_the_gate_is_called_not_reimplemented(monkeypatch):
    seen = []

    def spy(episodes_seen, window, rolling, best_so_far, close=None, best_was_reporting=False):
        seen.append((episodes_seen, rolling, close, best_was_reporting))
        return False

    monkeypatch.setattr(cs, "best_save_gate", spy)
    cs.replay(_rows([(100, 0.4, 0.8), (10, 0.6, None)]))
    assert seen == [(100, 0.4, 0.8, False), (110, 0.6, None, False)]


def test_no_save_before_the_outcome_window_turns_over():
    """D4 composes with everything else: 99 episodes is never enough."""
    assert cs.replay(_rows([(99, 1.0, 1.0)])) == []
    assert cs.replay(_rows([(100, 1.0, 1.0)])) == [0]


# ------------------------------------------------------- the #57 mechanism --


#: The shape of `patrol_brique_v19_rdb3_seed13`, in four windows: the first
#: eligible iteration reports on a thin, success-conditioned sample at 2%
#: success, and the policy then learns to win and stops claiming.
V19_SHAPE = [
    (100, 0.02, ROOT_REPORT_CLOSE_FLOOR),  # first eligible window: reporting, barely
    (40, 0.55, 0.0),                       # winning more, no longer closing its own ops
    (40, 0.98, None),                      # winning nearly always, nothing measured
    (40, 1.00, 0.0),                       # the policy that actually ships
]


def test_a_thin_early_reporting_window_locks_the_checkpoint():
    """The absorbing flag, which is what makes one iteration decide a whole run.

    ``best_was_reporting`` is set by the first window at or above the floor and
    no mute window may ever take the best back — so the 2%-success iteration is
    the ONLY save, and the 100%-success policy that follows can never be
    selected however long the run continues.
    """
    saves = cs.replay(_rows(V19_SHAPE))
    assert saves == [0], "the first reporting window was the run's only ckpt_best"


def test_a_success_floor_releases_it_and_nothing_else_does():
    """#57's proposal, measured: the floor is what changes the selection.

    Below the floor a window may not claim the reporting side of the order, so
    it is compared on success like any mute one — and the run's best work
    becomes the policy that wins.
    """
    rows = _rows(V19_SHAPE)
    floored = cs.replay(rows, floor=0.5)
    assert floored, "a floored replay must still select something"
    assert cs.episodes_and_windows(rows)[floored[-1]][1] == 1.0
    # and floor 0.0 is byte-for-byte the shipped rule, which is what makes the
    # sweep readable as a delta rather than as a second implementation
    assert cs.replay(rows, floor=0.0) == cs.replay(rows)


def test_a_floor_does_not_disturb_a_run_that_reports_while_winning():
    """The floor must be inert where the two axes agree — otherwise the sweep
    measures the floor rather than the pathology."""
    healthy = _rows([(100, 0.90, 0.8), (40, 0.95, 0.85), (40, 0.99, 0.9)])
    assert cs.replay(healthy) == cs.replay(healthy, floor=0.5) == [0, 1, 2]


# --------------------------------------------------- reading a run directory --


def _run(root, name, spec, **kwargs):
    import csv

    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    rows = _rows(spec, **kwargs)
    with (d / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return d


def test_a_corpus_without_n_episodes_is_not_replayable_rather_than_empty(tmp_path):
    """The reader limitation that would otherwise read as a finding.

    Pre-``n_episodes`` runs have a perfectly good ``ckpt_best.pt`` on disk; a
    replay that cannot reconstruct ``episodes_seen`` must not report them as
    runs whose gate never fired.
    """
    d = _run(tmp_path, "old", [(100, 1.0, None)], episodes_column=False)
    facts = cs.run_facts(d)
    assert facts["replayable"] is False
    assert facts["missing"] == ["n_episodes"]
    assert cs.success_recovered(facts) is None


def test_a_pre_v120_corpus_replays_exactly_and_no_floor_can_move_it(tmp_path):
    """Before the reporting axis existed the gate WAS success-only, and a
    missing rate is the ``None`` the gate already treats as not-reporting — so
    the replay stays exact and the floor is inert by construction."""
    d = _run(tmp_path, "pre", [(100, 0.5, None), (40, 0.9, None)], reporting_column=False)
    facts = cs.run_facts(d)
    assert facts["replayable"] and not facts["has_reporting"]
    assert facts["sweep"][0.0]["success"] == facts["sweep"][0.5]["success"] == 0.9


def test_the_replay_is_checked_against_the_iteration_stamped_in_the_checkpoint(tmp_path, monkeypatch):
    """The reader never asks to be believed: ``ckpt_best.pt`` records the
    iteration it was written at, so agreement is reported per run."""
    d = _run(tmp_path, "run", V19_SHAPE)
    (d / "ckpt_best.pt").write_bytes(b"not a real checkpoint")
    monkeypatch.setattr(cs, "checkpoint_iteration", lambda path: 1)
    assert cs.run_facts(d)["agrees"] is True
    monkeypatch.setattr(cs, "checkpoint_iteration", lambda path: 999)
    assert cs.run_facts(d)["agrees"] is False


def test_an_unreadable_checkpoint_is_unverified_not_a_disagreement(tmp_path):
    """A run still training is halfway through writing this file at any moment."""
    d = _run(tmp_path, "run", V19_SHAPE)
    (d / "ckpt_best.pt").write_bytes(b"truncated")
    assert cs.checkpoint_iteration(d / "ckpt_best.pt") is None
    assert cs.run_facts(d)["agrees"] is None


def test_admitted_claims_are_emitted_minus_rejected(tmp_path):
    """#57's column, and the distinction it is NOT.

    ``admitted`` counts the root's accepted claims; ``closed_on_root_report_rate``
    asks whether a claim closed the *operation*. `v19`'s ``ckpt_best`` has two
    accepted claims and a close rate of 0.000, because a root can have a
    sub-mission COMPLETE accepted without ending anything.
    """
    d = tmp_path / "run"
    d.mkdir()
    (d / "behavior.json").write_text(json.dumps({
        "episodes": 100,
        "metrics": {"success_rate": 0.17, "done_reports_root": 94,
                    "done_rejected_root": 92, "done_claim_episodes_root": 40,
                    "closed_on_root_report_rate": 0.0},
    }))
    cell = cs.evaluation(d, "behavior.json")
    assert cell["emitted"] == 94
    assert cell["admitted"] == 2
    assert cell["close_rate"] == 0.0


def test_an_evaluation_without_the_claim_counters_reports_none_not_zero(tmp_path):
    """Pre-v1.13 corpora predate ``done_reports_root``; absent is not zero."""
    d = tmp_path / "run"
    d.mkdir()
    (d / "behavior.json").write_text(json.dumps({"episodes": 100, "metrics": {"success_rate": 0.9}}))
    cell = cs.evaluation(d, "behavior.json")
    assert cell["emitted"] is None and cell["admitted"] is None


@pytest.mark.parametrize("printer", [cs.print_run, lambda f: cs.print_fleet([f], 0.5)])
def test_every_run_shape_prints_without_raising(tmp_path, capsys, printer):
    """Including the two that carry no sweep at all."""
    for name, kwargs in (("full", {}), ("old", {"episodes_column": False}),
                         ("short", {})):
        spec = [(10, 0.5, None)] if name == "short" else V19_SHAPE
        printer(cs.run_facts(_run(tmp_path, name, spec, **kwargs)))
    assert capsys.readouterr().out
