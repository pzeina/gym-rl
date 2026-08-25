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
from cohort.training.train import best_save_gate
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

    def spy(episodes_seen, window, rolling, best_so_far, close=None,
            best_was_reporting=False, **kw):
        # **kw so this spy tracks the shipped signature rather than pinning it:
        # v1.23 added keyword-only `report_gate_waived`, and a spy that refuses
        # unknown keywords fails the call it is supposed to be observing.
        seen.append((episodes_seen, rolling, close, best_was_reporting))
        return False

    monkeypatch.setattr(cs, "best_save_gate", spy)
    cs.replay(_rows([(100, 0.4, 0.8), (10, 0.6, None)]))
    assert seen == [(100, 0.4, 0.8, False), (110, 0.6, None, False)]


def test_the_absorbing_flag_is_is_reporting_not_a_local_copy():
    """The half of the rule that is easy to re-implement, and was.

    ``replay`` must decide ``best_was_reporting`` with the shipped
    ``is_reporting``, not with its own ``rate >= FLOOR``. While it kept a local
    copy, v1.21's fix reached ``best_save_gate`` and not the flag, and the
    reader reported the PRE-fix selection under the POST-fix rule — a
    disagreement invisible in every test because both were wrong together.
    """
    calls = []
    real = cs.is_reporting

    def spy(close, rolling):
        calls.append((close, rolling))
        return real(close, rolling)

    original, cs.is_reporting = cs.is_reporting, spy
    try:
        cs.replay(_rows([(100, 0.02, ROOT_REPORT_CLOSE_FLOOR), (40, 1.0, 0.0)]))
    finally:
        cs.is_reporting = original
    assert (ROOT_REPORT_CLOSE_FLOOR, 0.02) in calls, "the flag must go through is_reporting"


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


def test_a_thin_early_reporting_window_locked_the_checkpoint_before_v121():
    """The absorbing flag, which is what made one iteration decide a whole run.

    Under the pre-v1.21 rule ``best_was_reporting`` was set by the first window
    at or above the floor whatever its success, and no mute window could take
    the best back — so the 2%-success iteration was the ONLY save, and the
    100%-success policy that followed could never be selected however long the
    run continued. Kept as the historical rule because every ``ckpt_best`` in
    the corpus today was written by it.
    """
    saves = cs.replay_pre_v121(_rows(V19_SHAPE))
    assert saves == [0], "the first reporting window was the run's only ckpt_best"


def test_the_shipped_rule_no_longer_locks_on_a_thin_window():
    """v1.21's fix, measured on the shape that motivated it.

    ``is_reporting`` refuses the reporting side of the order below
    ``SUCCESS_RATE_FLOOR``, so the 2% window is compared on success like any
    mute one — it still saves (nothing has beaten it yet, and selection is not
    a veto), and is then superseded by the policy that actually wins.
    """
    rows = _rows(V19_SHAPE)
    saves = cs.replay(rows)
    assert len(saves) > 1, "the thin window must not be the run's only save"
    assert cs.episodes_and_windows(rows)[saves[-1]][1] == 1.0


def test_an_extra_floor_above_the_shipped_one_changes_nothing_on_this_shape():
    """The sweep's meaning after v1.21: 0.0 already carries SUCCESS_RATE_FLOOR.

    Before the fix this shape was the one run in 104 a floor moved. Now the
    floor is inside the shipped rule, so sweeping a stricter one on top must be
    inert here — otherwise the sweep is measuring the floor rather than a
    pathology.
    """
    rows = _rows(V19_SHAPE)
    assert cs.replay(rows, floor=0.5) == cs.replay(rows, floor=0.0) == cs.replay(rows)


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
    selected = cs.replay(_rows(V19_SHAPE))[-1] + 1  # iterations are 1-based in _rows
    monkeypatch.setattr(cs, "checkpoint_stamp",
                        lambda path: {"iteration": selected, "env_steps": 1024})
    assert cs.run_facts(d)["agrees"] is True
    monkeypatch.setattr(cs, "checkpoint_stamp", lambda path: {"iteration": 999, "env_steps": 1024})
    assert cs.run_facts(d)["agrees"] is False


def test_a_pre_v121_checkpoint_reads_as_stale_rather_than_as_a_broken_replay(tmp_path, monkeypatch):
    """Every ``ckpt_best`` in the corpus predates the v1.21 fix.

    On the shape the fix targets the two rules select different iterations, so
    the artifact on disk cannot agree with the shipped replay — and reporting
    that as REPLAY DISAGREES would read as a broken reader rather than as a
    checkpoint written by the rule that has since been corrected.
    """
    d = _run(tmp_path, "run", V19_SHAPE)
    (d / "ckpt_best.pt").write_bytes(b"not a real checkpoint")
    legacy = cs.replay_pre_v121(_rows(V19_SHAPE))[-1] + 1
    monkeypatch.setattr(cs, "checkpoint_stamp",
                        lambda path: {"iteration": legacy, "env_steps": 1024})
    facts = cs.run_facts(d)
    assert facts["agrees"] is False, "the shipped rule selects a different window"
    assert facts["agrees_pre_v121"] is True, "and the pre-fix rule selects the one on disk"


def test_an_unreadable_checkpoint_is_unverified_not_a_disagreement(tmp_path):
    """A run still training is halfway through writing this file at any moment."""
    d = _run(tmp_path, "run", V19_SHAPE)
    (d / "ckpt_best.pt").write_bytes(b"truncated")
    assert cs.checkpoint_stamp(d / "ckpt_best.pt") is None
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


# --------------------------------------------------------------------------- #
# v1.23: a waived comm model drops the reporting key from SELECTION too
# --------------------------------------------------------------------------- #

def test_a_waived_comm_model_selects_on_success_alone():
    """The reporting key is dropped, so a mute higher-success window wins.

    Un-waived, the first REPORTING window takes ckpt_best whatever the success
    numbers say, and a mute one may never take it back. Waived, that ordering
    is gone: 0.90 mute must beat a 0.80 reporting best.
    """
    args = dict(episodes_seen=100, window=100)
    # un-waived: the reporting best is absorbing, so the better mute window loses
    assert not best_save_gate(
        **args, rolling=0.90, best_so_far=0.80,
        root_report_close=0.0, best_was_reporting=True,
    )
    # waived: success alone decides, so it wins
    assert best_save_gate(
        **args, rolling=0.90, best_so_far=0.80,
        root_report_close=0.0, best_was_reporting=True,
        report_gate_waived=True,
    )


def test_a_waived_comm_model_does_not_promote_a_worse_reporting_window():
    """The other direction: a reporting window no longer jumps the queue.

    This is the half that matters under jamming. The rate is
    success-conditioned and sparse, so an un-waived run promotes whichever
    window happened to catch a close — and the absorbing clause then pins it.
    """
    assert best_save_gate(
        episodes_seen=100, window=100, rolling=0.60, best_so_far=0.95,
        root_report_close=1.0, best_was_reporting=False,
    ), "un-waived, the first reporting window wins on the key alone"
    assert not best_save_gate(
        episodes_seen=100, window=100, rolling=0.60, best_so_far=0.95,
        root_report_close=1.0, best_was_reporting=False,
        report_gate_waived=True,
    ), "waived, a worse window may not take ckpt_best"


def test_the_turnover_guard_survives_the_waiver():
    """D4: ckpt_best still may not be written before the window turns over."""
    assert not best_save_gate(
        episodes_seen=20, window=100, rolling=1.0, best_so_far=-1.0,
        report_gate_waived=True,
    )


def test_selection_and_the_gate_read_the_same_waiver_table():
    """One table, so the two can never disagree about which scenarios are waived.

    If a comm model is ever waived in `metrics` but not honoured in selection
    (or vice versa), a run would be judged by one rule and selected by another.
    """
    from cohort.metrics import COMM_MODEL_GATE_WAIVERS
    assert "closed_on_root_report_rate" in COMM_MODEL_GATE_WAIVERS["jammed"]
    assert "global" not in COMM_MODEL_GATE_WAIVERS
