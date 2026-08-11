"""A "single-variable A/B" claim must be checked against economics.json, not asserted.

refs #20: the v1.11 fleet's own confound audit was done by hand — open
``economics.json`` for a run and its predecessor, eyeball the ``rewards``
dict, count what changed. It undercounted at least once. ROADMAP's audit
compared ``squad_v7`` -> ``squad_v8`` and found one differing key
(``done_false``); an independent review compared ``squad_v6`` -> ``squad_v8``
— an equally legitimate choice of "the run before this one" — and found two
(``done_false`` AND ``contact_redundant``). Nothing forced the manual diff to
be checked against the file that would have caught the discrepancy.

``economics.json`` exists specifically so this never has to be eyeballed
(train.py's own comment: "two runs a reward commit apart are indistinguishable
after the fact"). ``run_report.economics_diff`` is that check, callable
instead of transcribed — these tests pin it against the exact pair the issue
used, so a future edit cannot quietly go back to under- or over-counting.
"""

import json

from scripts import run_report
from scripts.run_report import economics_diff

# The two economics.json reward dicts squad_v6 and squad_v8 actually shipped
# with (trimmed to the keys that matter for this test; real files carry the
# full RewardConfig). squad_v6 predates BOTH the free-ride fix's contact
# reprice and the done_false revert; squad_v8 postdates both.
SQUAD_V6_REWARDS = {"done_false": -2.0, "contact_redundant": -0.02, "contact_new": 0.5}
SQUAD_V7_REWARDS = {"done_false": -2.0, "contact_redundant": -0.25, "contact_new": 0.5}
SQUAD_V8_REWARDS = {"done_false": -0.5, "contact_redundant": -0.25, "contact_new": 0.5}
SPEC = {"root_mission": "MissionType.SEIZE", "max_steps": 450}


def _write_economics(tmp_path, name: str, rewards: dict, commit: str | None = None) -> None:
    run_dir = tmp_path / name
    run_dir.mkdir()
    payload = {"rewards": rewards, "spec": SPEC}
    if commit is not None:
        payload["git_commit"] = commit
    (run_dir / "economics.json").write_text(json.dumps(payload))


def test_squad_v6_to_v8_is_confounded_by_two_keys(tmp_path, monkeypatch, capsys):
    """The exact pair issue #20 flagged: two reward keys differ, not one."""
    _write_economics(tmp_path, "squad_v6", SQUAD_V6_REWARDS)
    _write_economics(tmp_path, "squad_v8", SQUAD_V8_REWARDS)
    monkeypatch.setattr(run_report, "RUNS", tmp_path)

    economics_diff("squad_v8", "squad_v6")
    out = capsys.readouterr().out

    assert "CONFOUNDED" in out
    assert "2 prices differ" in out
    assert "rewards.done_false" in out
    assert "rewards.contact_redundant" in out
    # contact_new is unchanged across the pair and must not be reported as a diff
    assert "contact_new" not in out


def test_squad_v7_to_v8_is_one_price_and_an_unknown_amount_of_code(tmp_path, monkeypatch,
                                                                  capsys):
    """This test used to assert the pair was a single-variable A/B. It is not.

    ROADMAP published the `squad_v7` -> `squad_v8` move as attributable to
    `done_false`, this file agreed, and both were wrong in the same way: nothing
    had compared the CODE. `squad_v8` is the first squad run carrying `d44ee8d`
    ("The fallen now share in the win they died taking") — 17 commits touching
    ``cohort/`` separate the pair. The price diff is real and it is the smaller
    of two variables.

    These fixtures carry no ``git_commit`` at all (the runs they model predate
    the field being consulted), so the honest verdict is UNCHECKABLE. That is
    the point worth pinning: an unchecked axis must not read as an agreeing one.
    """
    _write_economics(tmp_path, "squad_v7", SQUAD_V7_REWARDS)
    _write_economics(tmp_path, "squad_v8", SQUAD_V8_REWARDS)
    monkeypatch.setattr(run_report, "RUNS", tmp_path)

    economics_diff("squad_v8", "squad_v7")
    out = capsys.readouterr().out

    assert "one price differs" in out
    assert "rewards.done_false" in out
    assert "contact_redundant" not in out
    assert "UNCHECKABLE" in out
    assert "single-variable A/B" not in out


def test_identical_prices_at_one_commit_are_the_same_setup(tmp_path, monkeypatch, capsys):
    """The claim a baseline fleet makes about its own members."""
    _write_economics(tmp_path, "squad_v10", SQUAD_V8_REWARDS, commit="a" * 40)
    _write_economics(tmp_path, "squad_v11", SQUAD_V8_REWARDS, commit="a" * 40)
    monkeypatch.setattr(run_report, "RUNS", tmp_path)

    economics_diff("squad_v11", "squad_v10")
    out = capsys.readouterr().out

    assert "prices identical" in out
    assert "IDENTICAL SETUP" in out
    assert "CONFOUNDED" not in out


def test_identical_prices_across_commits_are_not_clean(tmp_path, monkeypatch, capsys):
    """squad_screen_v9 -> fallen_v1: identical rewards, and a different environment.

    The pair this file previously called "clean". Its two runs differ by exactly
    the code change the D4 A/B was testing — reporting it as clean described the
    one axis that did not move and stayed silent about the one that did.
    """
    _write_economics(tmp_path, "squad_screen_v9", SQUAD_V8_REWARDS, commit="0" * 40)
    _write_economics(tmp_path, "fallen_v1", SQUAD_V8_REWARDS, commit="1" * 40)
    monkeypatch.setattr(run_report, "RUNS", tmp_path)
    monkeypatch.setattr(
        run_report, "_git",
        lambda argv: "3\n" if argv[0] == "rev-list" else "d44ee8d The fallen share the win\n",
    )

    economics_diff("fallen_v1", "squad_screen_v9")
    out = capsys.readouterr().out

    assert "prices identical" in out
    assert "CONFOUNDED" in out
    assert "d44ee8d" in out


def test_missing_economics_json_is_uncheckable_not_a_crash(tmp_path, monkeypatch, capsys):
    """fireteam_v7 predates economics.json entirely — say so, don't raise."""
    (tmp_path / "fireteam_v7").mkdir()
    _write_economics(tmp_path, "fireteam_v8", SQUAD_V8_REWARDS)
    monkeypatch.setattr(run_report, "RUNS", tmp_path)

    economics_diff("fireteam_v8", "fireteam_v7")
    out = capsys.readouterr().out

    assert "uncheckable" in out
    assert "fireteam_v7" in out
