"""Scoring a fleet must never quietly replace evidence that already exists.

This repository has lost a committed N=100 artifact to a re-run twice. The first
time was `squad_v7`, caught by reading `git diff` before a commit. The second was
`scripts/publish_baseline.py` itself, on its first smoke test: pointed at
`squad_v9` — a finished run already scored at N=100 — it re-evaluated and
overwrote the artifact, the transcript and the GIF, because the guard read
``have > episodes`` and 100 is not greater than 100.

An equal-N re-evaluation is not an improvement. It is a different sample of the
same policy replacing a number that may already be published, for nothing. So the
rule is: **an existing evaluation at least as large as the one requested is left
exactly where it is**, and overwriting it takes an explicit ``--force``.

Also pinned: a run that is still training is never touched, because its process
owns that directory.
"""

from __future__ import annotations

import json

import pytest

from scripts import publish_baseline


@pytest.fixture
def run(tmp_path, monkeypatch):
    d = tmp_path / "squad_v10"
    d.mkdir()
    for ckpt, _artifact, _media in publish_baseline.TARGETS:
        (d / ckpt).write_text("stub")
    monkeypatch.setattr(publish_baseline, "RUNS", tmp_path)
    monkeypatch.setattr(publish_baseline.baseline, "RUNS", tmp_path)
    monkeypatch.setattr(publish_baseline, "summarize", lambda d: {"state": "DONE"})
    return d


def _existing(d, artifact: str, episodes: int):
    (d / artifact).write_text(json.dumps({"episodes": episodes, "success_ci95": "0.97 ± 0.03"}))


def _evaluator(calls):
    def fake(ckpt, **kw):
        calls.append((ckpt, kw["episodes"], kw["behavior_path"]))
        from pathlib import Path

        Path(kw["behavior_path"]).write_text(
            json.dumps({"episodes": kw["episodes"], "success_ci95": "0.50 ± 0.10"}))
        return {"success_ci95": "0.50 ± 0.10"}

    return fake


def test_an_equal_sized_evaluation_is_left_alone(run, monkeypatch, capsys):
    """The exact defect: 100 is not greater than 100, so it overwrote."""
    for _ckpt, artifact, _media in publish_baseline.TARGETS:
        _existing(run, artifact, 100)
    calls = []
    monkeypatch.setattr("cohort.training.evaluate.evaluate", _evaluator(calls))

    publish_baseline.publish("squad_v10", 100)

    assert calls == [], "it re-measured a run that was already scored at that N"
    assert "leaving it alone" in capsys.readouterr().out
    for _ckpt, artifact, _media in publish_baseline.TARGETS:
        assert json.loads((run / artifact).read_text())["success_ci95"] == "0.97 ± 0.03"


def test_a_larger_evaluation_is_never_shrunk(run, monkeypatch):
    for _ckpt, artifact, _media in publish_baseline.TARGETS:
        _existing(run, artifact, 100)
    calls = []
    monkeypatch.setattr("cohort.training.evaluate.evaluate", _evaluator(calls))

    publish_baseline.publish("squad_v10", 20)

    assert calls == []


def test_a_smoke_test_sized_artifact_is_replaced_by_the_real_one(run, monkeypatch):
    """The case this script exists for: training writes N=20, publication needs 100."""
    for _ckpt, artifact, _media in publish_baseline.TARGETS:
        _existing(run, artifact, 20)
    calls = []
    monkeypatch.setattr("cohort.training.evaluate.evaluate", _evaluator(calls))

    publish_baseline.publish("squad_v10", 100)

    assert len(calls) == len(publish_baseline.TARGETS)
    assert {c[1] for c in calls} == {100}


def test_force_is_the_only_way_past_the_guard(run, monkeypatch):
    for _ckpt, artifact, _media in publish_baseline.TARGETS:
        _existing(run, artifact, 100)
    calls = []
    monkeypatch.setattr("cohort.training.evaluate.evaluate", _evaluator(calls))

    publish_baseline.publish("squad_v10", 100, force=True)

    assert len(calls) == len(publish_baseline.TARGETS)


def test_a_training_run_is_never_touched(run, monkeypatch, capsys):
    """Its process owns that directory; evaluating into it races the trainer."""
    monkeypatch.setattr(publish_baseline, "summarize", lambda d: {"state": "RUNNING"})
    calls = []
    monkeypatch.setattr("cohort.training.evaluate.evaluate", _evaluator(calls))

    assert publish_baseline.publish("squad_v10", 100) == 0
    assert calls == []
    assert "still training, skipped" in capsys.readouterr().out


def test_the_final_policy_gets_the_transcript_and_the_gif(run, monkeypatch):
    """"Simulations available" has to mean files on disk, not an invitation."""
    calls = []

    def fake(ckpt, **kw):
        calls.append(kw)
        from pathlib import Path

        Path(kw["behavior_path"]).write_text(json.dumps({"episodes": kw["episodes"]}))
        return {}

    monkeypatch.setattr("cohort.training.evaluate.evaluate", fake)
    publish_baseline.publish("squad_v10", 100)

    with_media = [c for c in calls if "gif_path" in c]
    assert len(with_media) == 1, "exactly one checkpoint carries the media"
    assert with_media[0]["behavior_path"].endswith("behavior_final.json")
    assert with_media[0]["transcript_path"].endswith("eval_transcript.txt")
