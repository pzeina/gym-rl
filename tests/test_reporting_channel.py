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
         episodes=100, rate_best=None, rate_final=None, overrides=(),
         false_best=None, false_final=None):
    """One run directory, as `train.py` + `evaluate` leave it behind."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"scenario": scenario, "seed": seed}))
    (d / "economics.json").write_text(json.dumps({
        "git_commit": commit, "reward_overrides": list(overrides),
        "rewards": {"root_done_bonus": price},
    }))
    for filename, claims, rate, false in (("behavior.json", best, rate_best, false_best),
                                          ("behavior_final.json", final, rate_final, false_final)):
        if claims is None:
            continue
        (d / filename).write_text(json.dumps({
            "episodes": episodes,
            "metrics": {"done_claim_episodes_root": claims,
                        "closed_on_root_report_rate": rate,
                        "false_complete_rate_root": false},
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


# --- the price question is the same matched design as the transfer question
#
# `--arms 1.0 3.0` holds the scenario, the chart-link state and any other reward
# override, and varies `root_done_bonus`. It exists because assurance #59 read
# that table off the close rate at one checkpoint and got eight pairs where the
# campaign's own pre-registration ("SPLIT is dropped, not silently counted")
# gives five.


def test_the_price_table_pairs_by_seed_and_holds_everything_else(fleet):
    for seed, cheap, dear in ((12, 0, 100), (14, 100, 0), (15, 0, 100),
                              (18, 100, 100), (19, 100, 0)):
        _run(fleet, f"rdb1_{seed}", scenario="patrol_brique", seed=seed,
             best=cheap, final=cheap, price=1.0)
        _run(fleet, f"rdb3_{seed}", scenario="patrol_brique", seed=seed,
             best=dear, final=dear, price=3.0)

    result = rc.price_pairs(rc.collect(fleet), 1.0, 3.0)

    assert len(result["pairs"]) == 5
    assert (result["reporting_first"], result["reporting_second"]) == (3, 3)
    assert (result["one_way"], result["other_way"]) == (2, 2)
    assert result["p"] == pytest.approx(1.0)


def test_a_split_cell_at_either_price_drops_the_seed(fleet):
    """The pre-registered rule, and it is what separates five pairs from eight."""
    _run(fleet, "rdb1_12", scenario="patrol_brique", seed=12, best=0, final=0, price=1.0)
    _run(fleet, "rdb3_12", scenario="patrol_brique", seed=12, best=100, final=100, price=3.0)
    # seed 13's rdb=3.0 cell claims in 40 of 100 episodes at ckpt_best and none
    # at ckpt_latest — `patrol_brique_v19_rdb3_seed13`, the real one.
    _run(fleet, "rdb1_13", scenario="patrol_brique", seed=13, best=0, final=0, price=1.0)
    _run(fleet, "rdb3_13", scenario="patrol_brique", seed=13, best=40, final=0, price=3.0)

    result = rc.price_pairs(rc.collect(fleet), 1.0, 3.0)

    assert [p["seed"] for p in result["pairs"]] == [12]


def test_a_price_pair_needs_the_same_scenario_and_the_same_tree_state(fleet):
    _run(fleet, "patrol_rdb1", scenario="patrol_brique", seed=12, best=0, final=0, price=1.0)
    _run(fleet, "squad_rdb3", scenario="squad", seed=12, best=100, final=100, price=3.0)
    _run(fleet, "patrol_rdb3_override", scenario="patrol_brique", seed=12,
         best=100, final=100, price=3.0, overrides=("sitrep_bonus=0.5",))

    assert rc.price_pairs(rc.collect(fleet), 1.0, 3.0)["pairs"] == []


def test_the_matched_read_out_says_what_the_table_could_have_shown(fleet, capsys):
    """Two p values of 1.0000 that mean opposite things (assurance #59)."""
    for seed, cheap, dear in ((12, 0, 100), (14, 100, 0), (15, 0, 100),
                              (18, 100, 100), (19, 100, 0)):
        _run(fleet, f"rdb1_{seed}", scenario="patrol_brique", seed=seed,
             best=cheap, final=cheap, price=1.0)
        _run(fleet, f"rdb3_{seed}", scenario="patrol_brique", seed=seed,
             best=dear, final=dear, price=3.0)

    rc.print_arms(rc.collect(fleet), 1.0, 3.0)
    out = capsys.readouterr().out

    assert "NOT A NULL — 4 discordant pairs could not go below 0.1250" in out
    assert "a null WITH power — margin 6/10 reporting could have reached 0.0476" in out


def test_the_transfer_read_out_carries_the_same_ceiling(fleet, capsys):
    """One implementation, so #55's table gained the reading #59 was missing."""
    for seed in range(12, 17):
        _run(fleet, f"squad_{seed}", scenario="squad", seed=seed, best=100, final=100)
        _run(fleet, f"patrol_{seed}", scenario="patrol_brique", seed=seed, best=0, final=0)

    rc.print_pairs(rc.collect(fleet), "squad", "patrol_brique")
    out = capsys.readouterr().out

    assert "exact McNemar (paired)   p = 0.0625" in out
    assert "NOT A NULL — 5 discordant pairs could not go below 0.0625" in out, \
        "five unanimous pairs is the direction #55 reported and never a rejection"


def test_the_spam_series_reads_reporting_checkpoints_only(fleet):
    """A mute checkpoint has no false-DONE rate, so the relation lives in one mode."""
    _run(fleet, "reports", scenario="patrol_brique", seed=12, best=100, final=100,
         rate_final=0.80, false_final=0.22)
    _run(fleet, "mute_but_rated", scenario="patrol_brique", seed=13, best=0, final=0,
         rate_final=0.04, false_final=1.0)
    _run(fleet, "reports_more", scenario="patrol_brique", seed=14, best=100, final=100,
         rate_final=1.00, false_final=0.75)
    _run(fleet, "middling", scenario="patrol_brique", seed=15, best=100, final=100,
         rate_final=0.90, false_final=0.32)

    result = rc.spam_series(rc.collect(fleet))

    assert [p["run"] for p in result["points"]] == ["middling", "reports", "reports_more"]
    assert "mute_but_rated" not in [p["run"] for p in result["points"]], \
        "a single rejected claim is not an observation of the reporting mode"


def test_a_replicated_cell_is_one_observation_of_the_relation(fleet):
    for name in ("twin_a", "twin_b"):
        _run(fleet, name, scenario="patrol_brique", seed=12, best=100, final=100,
             rate_final=0.867, false_final=0.375)
    _run(fleet, "other", scenario="patrol_brique", seed=13, best=100, final=100,
         rate_final=1.0, false_final=0.75)
    _run(fleet, "third", scenario="patrol_brique", seed=14, best=100, final=100,
         rate_final=0.75, false_final=0.348)

    result = rc.spam_series(rc.collect(fleet))

    assert len(result["points"]) == 4 and len(result["distinct"]) == 3
    assert {p["run"] for p in result["points"] if p["replicate"]} == {"twin_b"}
    assert rc.spam_series(rc.collect(fleet))["rho"] == pytest.approx(1.0), \
        "and rho is taken over the three distinct points, not four rows"


def test_the_spam_read_out_prints_the_leave_one_out_range(fleet, capsys):
    """rho alone is what got 'monotone' written down; the range is what refutes it."""
    for seed, (rate, false) in enumerate([(0.750, 0.348), (0.794, 0.503), (0.808, 0.223),
                                          (0.825, 0.481), (0.867, 0.375), (0.878, 0.561),
                                          (0.895, 0.320), (1.000, 0.750), (0.750, 0.500)]):
        _run(fleet, f"r{seed}", scenario="patrol_brique", seed=seed, best=100, final=100,
             rate_final=rate, false_final=false)

    rc.print_spam(rc.collect(fleet))
    out = capsys.readouterr().out

    assert "Spearman rho = +0.259 over 9 distinct points from 9 checkpoints" in out
    assert "leave-one-out range -0.060 to +0.515" in out
    assert "the sign flips on leave-one-out, so this is not a relation" in out
