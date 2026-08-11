"""Two evaluations taken in different environments cannot be differenced (refs #39).

``publish_audit --validate`` asks whether give-back predicts that ``ckpt_best``
overstates the policy a run ended with. It answered r = 0.889 (n = 19) — and it
built that answer out of ``behavior.json`` minus ``behavior_final.json``, two
files that for three runs entered the repository days and dozens of ``cohort/``
commits apart:

    fireteam_v7          best 703a6ac (2026-08-06)  final f18462d (2026-08-11)
                         36 commits under cohort/ between them, 21 in env/core/config
    fireteam_defend_v10  best 2bada50               final 2957ae8   (16, incl. d44ee8d)
    squad_v7             best 351eaca               final b1a6c0e   (14)

Those three carry the two largest give-backs in the fleet, so the gate's
validation rested on exactly the rows whose signed gap is a checkpoint
difference *plus* an environment difference. Taken over the 16 pairs measured at
one commit, the same gate reads r = 0.749 (p = 0.0008) — still real, and no
longer resting on the confound. The fleet's largest same-commit overstatement is
+3pt (``defend_brique_v10``), not the +17pt that ``fireteam_v7`` was published
with.

This is the repo's own rule for A/B pairs (``run_report.code_diff``, refs #36)
turned on the audit. The git query is stubbed here, as it is there: what is
under test is that a non-empty ``cohort/`` log excludes a pair from the
headline, not that git works.
"""

from __future__ import annotations

import json

import pytest

from scripts import publish_audit

pytest.importorskip("scipy")


def _pair(root, name: str, *, give_back: float, best: float, final: float, peak: float = 0.95):
    """A run carrying both checkpoints at N=100, and a curve with that give-back."""
    d = root / name
    d.mkdir()
    for value, filename in ((best, "behavior.json"), (final, "behavior_final.json")):
        (d / filename).write_text(json.dumps({
            "scenario": "squad", "episodes": 100, "metrics": {"success_rate": value},
        }))
    ended = peak - give_back / 100
    rolling = [peak] * 18 + [ended] * 2          # deciles() of 20 rows: last two are the last
    lines = ["step,success_rate_rolling"]
    lines += [f"{i * 1000},{v}" for i, v in enumerate(rolling)]
    (d / "metrics.csv").write_text("\n".join(lines) + "\n")
    return d


def _fleet(root):
    """Four pairs measured at one commit, plus one measured across an era."""
    _pair(root, "same_v1", give_back=1.0, best=0.95, final=0.97)
    _pair(root, "same_v2", give_back=4.0, best=0.97, final=0.98)
    _pair(root, "same_v3", give_back=8.0, best=0.91, final=0.88)
    _pair(root, "same_v4", give_back=12.0, best=0.82, final=0.80)
    _pair(root, "cross_v1", give_back=68.0, best=0.95, final=0.78)


def _stub_git(monkeypatch, *, cross_era_run: str | None = "cross_v1", commits: str = "36",
              unknown: str | None = None):
    """git as the audit consults it: an artifact's commit, and the span between two."""

    def fake_git(argv: list[str]) -> str | None:
        if argv[0] == "log":
            path = argv[-1]
            if unknown and unknown in path:
                return None                       # git could not answer
            if cross_era_run and f"{cross_era_run}/behavior_final.json" in path:
                return "b" * 40 + "\n"
            return "a" * 40 + "\n"
        if argv[0] == "rev-list":
            assert argv[-1] == publish_audit.EVALUATION_TREE, "only cohort/ can move a number"
            return commits + "\n"
        return None

    monkeypatch.setattr(publish_audit, "_git", fake_git)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(publish_audit, "RUNS", tmp_path)
    return tmp_path


def test_a_cross_era_pair_is_named_and_kept_out_of_the_headline(runs, monkeypatch, capsys):
    """The fireteam_v7 shape: the biggest give-back in the fleet, and the one
    pair whose two numbers were never measured in the same environment."""
    _fleet(runs)
    _stub_git(monkeypatch)

    assert publish_audit.validate_gate() == 0
    out = capsys.readouterr().out

    assert "+36 apart" in out and "same commit" in out
    assert "4/5 pairs were measured at one commit" in out
    assert "cross_v1" in out.split("are excluded (refs #39)")[1]

    headline = next(ln for ln in out.splitlines() if "one commit:" in ln)
    assert "n=4" in headline
    confounded = next(ln for ln in out.splitlines() if "CONFOUNDED" in ln)
    assert "n=5" in confounded
    # the give-back mean is a claim about checkpoints, so it drops the mixed pair too
    assert "over the 4 same-commit pairs" in out


def test_the_confound_is_the_environment_not_the_calendar(runs, monkeypatch, capsys):
    """Two artifacts committed apart with nothing under ``cohort/`` between them
    were still measured in the same environment, and stay in."""
    _fleet(runs)
    _stub_git(monkeypatch, commits="0")

    publish_audit.validate_gate()
    out = capsys.readouterr().out

    assert "are excluded" not in out
    assert "CONFOUNDED" not in out
    assert "n=5" in next(ln for ln in out.splitlines() if "one commit:" in ln)


def test_an_undatable_pair_reads_as_unknown_rather_than_as_agreement(runs, monkeypatch, capsys):
    """"We could not tell" and "there is no difference" are opposite findings —
    the same rule ``run_report.code_diff`` applies to a missing ``git_commit``."""
    _fleet(runs)
    _stub_git(monkeypatch, cross_era_run=None, unknown="same_v2/behavior.json")

    publish_audit.validate_gate()
    out = capsys.readouterr().out

    assert "unknown" in out
    assert "same_v2" in out.split("are excluded (refs #39)")[1]
    assert "cannot be dated in this clone" in out


def test_era_gap_dates_the_artifacts_not_the_repository(runs, monkeypatch):
    """A pair written at one commit is comparable however much ``cohort/`` has
    moved since — what matters is the span BETWEEN the two evaluations."""
    d = _pair(runs, "same_v1", give_back=1.0, best=0.95, final=0.97)
    _stub_git(monkeypatch, cross_era_run=None, commits="999")

    assert publish_audit.era_gap(d / "behavior.json", d / "behavior_final.json") == 0


@pytest.mark.skip(reason="needs cohort/training/evaluate.py, frozen while the "
                         "baseline retrain campaign is in flight — train.py imports "
                         "the tree that exists when a job starts, so an edit under "
                         "cohort/ today would train the later fleet members against a "
                         "different environment than the earlier ones. Patch is written "
                         "out in ROADMAP.md's 2026-08-11 #39 entry; unskip when it lands.")
def test_an_evaluation_records_the_tree_it_was_measured_against(tmp_path):
    """The durable half of #39: git provenance dates an artifact only from
    *outside*, and only while it stays committed and unmoved. The artifact
    should say it itself, next to the checkpoint digest #28 already put there."""
    from cohort.training.evaluate import evaluate

    out = tmp_path / "behavior.json"
    evaluate(None, scenario="fireteam", episodes=1, seed=17, behavior=True,
             behavior_path=str(out))

    assert json.loads(out.read_text())["eval_commit"], "an evaluation must date itself"


def test_an_uncommitted_evaluation_is_dated_at_head(runs, monkeypatch):
    """``behavior.json`` written by a run that has not been committed yet was
    measured against the working tree, not against whenever git last saw it."""
    d = _pair(runs, "same_v1", give_back=1.0, best=0.95, final=0.97)
    monkeypatch.setattr(publish_audit, "_git", lambda argv: "" if argv[0] == "log" else "0\n")

    assert publish_audit.evaluation_era(d / "behavior.json") == publish_audit.WORKTREE
    assert publish_audit.era_gap(d / "behavior.json", d / "behavior_final.json") == 0
