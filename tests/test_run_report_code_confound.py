"""An A/B is single-variable only if the CODE matched too.

``run_report.py --vs`` has audited the prices since refs #20 and printed
"CLEAN — share every reward/spec value" whenever they matched. That sentence
was read, repeatedly, as "this is a single-variable A/B". It is not, and the
case that proves it was already published:

    squad_v7 -> squad_v8   prices: one key (done_false -2.0 -> -0.5)
                           code:   35 commits, 17 touching cohort/,
                                   among them d44ee8d "The fallen now share in
                                   the win they died taking"

`d44ee8d` is the change that ended the D4 collapse. Attributing that pair's
+7-point success move to a price is attributing it to the smaller of two
variables, and the instrument built to prevent exactly this class of mistake
could not see it — because a code change never touches ``economics.json``'s
prices, which is the only thing the audit read.

The commit was recorded all along (``economics.json:git_commit``, written by
``train.py``). Nothing consulted it. This file pins that something does:

* two runs at the same commit with one price apart are a single-variable A/B;
* two runs at the same commit with identical prices are the same setup — which
  is what a homogeneous baseline fleet must be able to assert;
* two runs at different commits are CONFOUNDED when anything under ``cohort/``
  moved between them, no matter how clean the prices are;
* commits that touch only ``scripts/`` or ``tests/`` are counted, not blamed —
  they cannot change what a policy learned;
* a missing or unknown commit reads as UNCHECKABLE rather than as clean, since
  "we could not tell" and "there is no difference" are opposite findings.
"""

from __future__ import annotations

import json

import pytest

from scripts import run_report


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    monkeypatch.setattr(run_report, "RUNS", tmp_path)
    return tmp_path


def _write_economics(root, name: str, *, commit: str | None, **rewards):
    run = root / name
    run.mkdir(exist_ok=True)
    payload = {"scenario": "squad", "rewards": {"done_false": -0.5, **rewards}, "spec": {}}
    if commit is not None:
        payload["git_commit"] = commit
    (run / "economics.json").write_text(json.dumps(payload))


def _diff(capsys, root, run: str, baseline: str) -> str:
    run_report.economics_diff(run, baseline)
    return capsys.readouterr().out


def test_same_commit_same_prices_reads_as_one_setup(fleet, capsys):
    """The assertion a baseline fleet needs to be able to make about itself."""
    _write_economics(fleet, "squad_v10", commit="a" * 40)
    _write_economics(fleet, "squad_v11", commit="a" * 40)

    out = _diff(capsys, fleet, "squad_v11", "squad_v10")

    assert "prices identical" in out
    assert "same commit aaaaaaaa" in out
    assert "IDENTICAL SETUP" in out


def test_same_commit_one_price_apart_is_the_single_variable_case(fleet, capsys):
    _write_economics(fleet, "squad_v10", commit="a" * 40)
    _write_economics(fleet, "squad_v11", commit="a" * 40, done_false=-2.0)

    out = _diff(capsys, fleet, "squad_v11", "squad_v10")

    assert "one price differs" in out
    assert "single-variable A/B" in out
    assert "CONFOUNDED" not in out


def test_a_code_change_confounds_a_price_clean_pair(fleet, capsys, monkeypatch):
    """The squad_v7 -> squad_v8 shape, with the git query stubbed.

    Stubbed rather than driven against a real repository: what is under test is
    that a non-empty ``cohort/`` log flips the verdict, not that git works.
    """
    _write_economics(fleet, "squad_v7", commit="0" * 40)
    _write_economics(fleet, "squad_v8", commit="1" * 40)
    monkeypatch.setattr(
        run_report,
        "_git",
        lambda argv: "35\n" if argv[0] == "rev-list" else
        "d44ee8d The fallen now share in the win they died taking\n",
    )

    out = _diff(capsys, fleet, "squad_v8", "squad_v7")

    assert "prices identical" in out          # the old audit's whole answer
    assert "CONFOUNDED" in out                # and the one it was missing
    assert "35 commits, 1 touching cohort/" in out
    assert "d44ee8d" in out


def test_commits_that_cannot_move_behaviour_are_counted_not_blamed(fleet, capsys, monkeypatch):
    """Twelve commits to scripts/ and tests/ do not confound an experiment."""
    _write_economics(fleet, "squad_v10", commit="0" * 40)
    _write_economics(fleet, "squad_v11", commit="1" * 40)
    monkeypatch.setattr(
        run_report, "_git", lambda argv: "12\n" if argv[0] == "rev-list" else "\n"
    )

    out = _diff(capsys, fleet, "squad_v11", "squad_v10")

    assert "12 commits, 0 touching cohort/" in out
    assert "CONFOUNDED" not in out


def test_an_unknown_commit_is_uncheckable_not_clean(fleet, capsys, monkeypatch):
    """A rebased or foreign sha must not read as agreement."""
    _write_economics(fleet, "squad_v10", commit="0" * 40)
    _write_economics(fleet, "squad_v11", commit="1" * 40)
    monkeypatch.setattr(run_report, "_git", lambda argv: None)

    out = _diff(capsys, fleet, "squad_v11", "squad_v10")

    assert "UNCHECKABLE" in out
    assert "CONFOUNDED" not in out


def test_a_run_predating_the_provenance_field_is_uncheckable(fleet, capsys):
    _write_economics(fleet, "squad_v5", commit=None)
    _write_economics(fleet, "squad_v10", commit="a" * 40)

    out = _diff(capsys, fleet, "squad_v10", "squad_v5")

    assert "UNCHECKABLE" in out
    assert "squad_v5" in out
