"""The mute/reporting label is a claim count, and it is read at both checkpoints.

Two ways to get this table wrong are on the record, and both are pinned here.

* **Reading the label off `closed_on_root_report_rate`.** That metric answers
  "did the root's report close the window", and the report that closes it is a
  SITREP wherever MISSION COMPLETE is masked shut — so the whole defend family
  scores 0.97-1.00 on ZERO root claims. It does not floor at zero on a
  completable root either: assurance #55 measured 0.020-0.104 on `patrol_brique`
  arms with no claims at all, and a 0.05 rate-cut calls those reporting.
* **Resolving a run whose checkpoints disagree.** `squad_v10c` claims in 18 of
  100 episodes at `ckpt_best` and 0 of 100 at `ckpt_latest`. Whichever
  checkpoint is picked, the pick is the finding — so the run is dropped and
  said to be dropped.

The transfer question itself is matched (same seed, same arm, two scenarios), so
the test is McNemar's over the discordant pairs and not Fisher's.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import reporting_channel as rc

UNITS = Path(__file__).resolve().parent.parent / "cohort" / "core" / "units.py"


def _run(root, name, *, scenario, seed, best, final, price=1.0, commit="deadbee",
         episodes=100, rate_best=None, rate_final=None, overrides=()):
    """One run directory, as `train.py` + `evaluate` leave it behind."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"scenario": scenario, "seed": seed}))
    (d / "economics.json").write_text(json.dumps({
        "git_commit": commit, "reward_overrides": list(overrides),
        "rewards": {"root_done_bonus": price},
    }))
    for filename, claims, rate in (("behavior.json", best, rate_best),
                                   ("behavior_final.json", final, rate_final)):
        if claims is None:
            continue
        (d / filename).write_text(json.dumps({
            "episodes": episodes,
            "metrics": {"done_claim_episodes_root": claims,
                        "closed_on_root_report_rate": rate},
        }))
    return d


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    """A runs/ directory whose trees resolve without touching git history."""
    monkeypatch.setattr(rc, "cohort_tree", lambda commit: f"tree-{commit}")
    monkeypatch.setattr(rc, "chart_link_state", lambda commit: "present")
    return tmp_path


def test_the_label_is_the_claim_count_not_the_close_rate(fleet):
    """A defend-shaped corpus: every operation closed on the root's SITREP, no claims."""
    _run(fleet, "defend", scenario="defend_brique", seed=12,
         best=0, final=0, rate_best=1.0, rate_final=0.99)

    row = rc.collect(fleet)[0]

    assert row["label"] == "mute", "0 claims in 100 episodes is a mute commander"
    assert row["best"]["rate"] == 1.0


def test_the_close_rate_does_not_floor_at_zero_and_the_report_says_where(fleet):
    """refs #55: a root SITREP landing on the ENDEX step enters that numerator.

    So a wholly mute arm reads 0.020-0.104, and a 0.05 rate-cut promotes it to
    reporting. The claim count says mute; the check names the disagreement
    rather than letting two tables of the same fleet quietly differ.
    """
    _run(fleet, "patrol_mute", scenario="patrol_brique", seed=14,
         best=0, final=0, rate_best=0.104, rate_final=0.020)

    disagreements = rc.rate_disagreements(rc.collect(fleet))

    assert [(run, ckpt) for run, ckpt, _ in disagreements] == [("patrol_mute", "best")]
    assert rc.rate_label(0.104) == "REPORTING" and rc.rate_label(0.020) == "mute"


def test_a_run_whose_checkpoints_disagree_is_dropped_not_resolved(fleet):
    """`squad_v10c`: 18 claiming episodes at best, 0 at latest."""
    _run(fleet, "squad_v10c", scenario="squad", seed=14, best=18, final=0)

    row = rc.collect(fleet)[0]

    assert row["label"] == "SPLIT"
    assert row["best"]["label"] == "REPORTING" and row["final"]["label"] == "mute"


def test_a_checkpoint_between_the_modes_is_undecided_not_rounded(fleet):
    """3 of 100 sits in the empty band; naming it is the honest reading."""
    _run(fleet, "between", scenario="squad", seed=14, best=3, final=97)

    row = rc.collect(fleet)[0]

    assert row["best"]["label"] == "undecided"
    assert row["label"] == "SPLIT", "an undecided checkpoint cannot carry a run's label"


def test_the_transfer_reading_is_mcnemar_over_the_discordant_pairs(fleet):
    """The #55 table: 5 unanimous discordant pairs and 2 concordant ones."""
    for seed, squad, patrol in ((12, 100, 0), (13, 100, 0), (14, 100, 0),
                                (15, 100, 0), (16, 100, 0),
                                (17, 100, 100), (18, 0, 0)):
        _run(fleet, f"squad_{seed}", scenario="squad", seed=seed, best=squad, final=squad)
        _run(fleet, f"patrol_{seed}", scenario="patrol_brique", seed=seed, best=patrol, final=patrol)

    result = rc.cross_scenario_pairs(rc.collect(fleet), "squad", "patrol_brique")

    assert len(result["pairs"]) == 7
    assert (result["one_way"], result["other_way"]) == (5, 0)
    assert result["p"] == pytest.approx(0.0625), "five unanimous pairs is a direction, not an effect"
    assert result["agree"] == 2


def test_a_pair_only_forms_at_one_fixed_arm(fleet):
    """Same seed, two prices: not a pair, because the arm is the treatment."""
    _run(fleet, "squad_rdb1", scenario="squad", seed=12, best=100, final=100, price=1.0)
    _run(fleet, "patrol_rdb3", scenario="patrol_brique", seed=12, best=0, final=0, price=3.0)

    result = rc.cross_scenario_pairs(rc.collect(fleet), "squad", "patrol_brique")

    assert result["pairs"] == []


def test_a_cell_that_replicates_is_kept_and_one_that_contradicts_is_dropped(fleet):
    """Two runs of one arm at one seed are a replication until they disagree."""
    _run(fleet, "squad_a", scenario="squad", seed=12, best=100, final=100)
    _run(fleet, "squad_b", scenario="squad", seed=12, best=99, final=98)
    _run(fleet, "patrol_a", scenario="patrol_brique", seed=12, best=0, final=0)

    kept = rc.cross_scenario_pairs(rc.collect(fleet), "squad", "patrol_brique")
    assert len(kept["pairs"]) == 1
    assert kept["pairs"][0]["runs"][0] == ["squad_a", "squad_b"]

    _run(fleet, "squad_b", scenario="squad", seed=12, best=0, final=0)
    dropped = rc.cross_scenario_pairs(rc.collect(fleet), "squad", "patrol_brique")
    assert dropped["pairs"] == [] and dropped["ambiguous"]


def test_a_pair_that_spans_two_trees_says_so(fleet, monkeypatch, capsys):
    """The arm is not the tree digest — but a comparison across two says which."""
    monkeypatch.setattr(rc, "cohort_tree", lambda commit: f"tree-{commit}")
    _run(fleet, "squad_a", scenario="squad", seed=12, best=100, final=100, commit="aaa")
    _run(fleet, "patrol_a", scenario="patrol_brique", seed=12, best=0, final=0, commit="bbb")

    rc.print_pairs(rc.collect(fleet), "squad", "patrol_brique")

    assert "TWO TREES" in capsys.readouterr().out


def test_an_evaluation_without_the_claim_counter_is_absent_not_zero(fleet):
    """The pre-v1.13 corpora predate `done_claim_episodes_root`."""
    d = fleet / "ancient"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"scenario": "squad", "seed": 3}))
    (d / "behavior.json").write_text(json.dumps({"episodes": 30, "metrics": {"success_rate": 0.9}}))

    assert rc.collect(fleet) == []


@pytest.mark.parametrize(("source", "state"), [
    ("if x not in parent.subordinate_ids:\n    parent.subordinate_ids.append(successor.id)", "present"),
    ("parent.subordinate_ids.append(successor.id)\nsuccessor.subordinate_ids.append(promoted.id)",
     "double-linked"),
    ("successor.subordinate_ids.append(promoted.id)", "absent"),
    (None, "unresolved"),
])
def test_the_three_chart_link_states_are_three(source, state):
    """`da24b42`'s double-linked tree is not the present arm and not the absent one.

    Between `56ada9a` and `da24b42` both appends were live and the promoted
    agent was filed under its leader twice. Pooling that tree with either arm
    would put a defective environment inside a treatment.
    """
    assert rc.classify_units_source(source) == state


def test_the_marker_the_arm_resolver_greps_for_still_exists():
    """A regression hazard: rename that line and every run silently reads `absent`.

    The arm a run belongs to is resolved by looking for this exact statement in
    the `cohort/core/units.py` of the commit the run recorded. Nothing fails if
    it is reworded — the tables just quietly become one arm.
    """
    assert rc.classify_units_source(UNITS.read_text()) == "present"
