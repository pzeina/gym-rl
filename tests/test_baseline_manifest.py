"""The baseline is a set with rules, and the rules are checkable.

Eight champions at seven commits, four of them reproducible only with a
``--reward`` override, one published with a flag saying it missed the bar — that
was the fleet before v1.19. Every number in it was honest and the set was still
not a system, because nothing anywhere asserted that its members belonged
together.

``scripts/baseline.py`` is that assertion. What is pinned here is the part that
rots quietest:

* a scenario added to ``cohort/config.py`` is either a baseline member or an
  explicitly excused non-member — silence is a failure, not a default;
* the manifest names a run for every doctrine scenario and nothing else;
* each individual check actually fails when violated. A gate nobody has seen
  fail is a gate nobody knows works, and this one is meant to be the last thing
  standing between an inconsistent fleet and a published claim.
"""

from __future__ import annotations

import json
import re

import pytest

from cohort.config import SCENARIOS
from scripts import baseline


def test_every_scenario_is_either_in_the_baseline_or_excused():
    """The coverage guard. A new scenario cannot escape the fleet by silence."""
    accounted = set(baseline.DOCTRINE_SCENARIOS) | set(baseline.NOT_BASELINE)
    unaccounted = sorted(set(SCENARIOS) - accounted)
    assert not unaccounted, (
        f"{unaccounted} is neither a baseline member nor listed in NOT_BASELINE "
        "with a reason — decide which it is"
    )


def test_the_excusals_are_reasons_not_placeholders():
    for scenario, reason in baseline.NOT_BASELINE.items():
        assert scenario in SCENARIOS, f"{scenario} is excused but does not exist"
        assert len(reason) > 20, f"{scenario}'s excusal is not a reason"


def test_the_shipped_manifest_covers_the_doctrine_scenarios():
    manifest = baseline.load()
    assert set(manifest["runs"]) == set(baseline.DOCTRINE_SCENARIOS)
    for scenario, run in manifest["runs"].items():
        assert run.startswith(scenario), f"{run} does not look like a {scenario} run"


def _member(tmp_path, run: str, *, commit="a" * 40, overrides=(), episodes=100,
            gates=(), successes=97, announced=97, peak_episodes=100,
            seed=12, close_rate=0.85, scenario=None):
    d = tmp_path / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "economics.json").write_text(json.dumps({
        "git_commit": commit, "reward_overrides": list(overrides),
    }))
    # The config is the run's training identity: the seed_spread completeness
    # scan matches on it modulo seed, so every run carries a scenario key —
    # derived from the name unless a test says otherwise — or eight members
    # with the config {} would all read as draws of one another's lottery.
    scenario = scenario or re.split(r"_v\d", run)[0]
    (d / "config.json").write_text(json.dumps({"scenario": scenario, "seed": seed}))
    metrics = {"success_rate": 0.97, "successes": successes,
               "successes_announced": announced}
    if close_rate is not None:
        metrics["closed_on_root_report_rate"] = close_rate
    (d / "behavior_final.json").write_text(json.dumps({
        "episodes": episodes,
        "success_ci95": "0.97 ± 0.03",
        "metrics": metrics,
        "gates": [{"name": g, "passed": False} for g in gates],
    }))
    # The peak the README publishes. A member is not two files by accident: the
    # headline is scored from ckpt_latest and this from ckpt_best, and this one
    # is the half nothing used to check (issue #45).
    (d / "behavior.json").write_text(json.dumps({
        "episodes": peak_episodes,
        "success_ci95": "0.98 ± 0.03",
        "metrics": {"success_rate": 0.98},
    }))
    return d


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    """A manifest whose members all pass, so each test can break exactly one."""
    runs = {s: f"{s}_v1" for s in baseline.DOCTRINE_SCENARIOS}
    (tmp_path / "BASELINE.json").write_text(json.dumps({"version": "test", "runs": runs}))
    for run in runs.values():
        _member(tmp_path, run)
    monkeypatch.setattr(baseline, "RUNS", tmp_path)
    monkeypatch.setattr(baseline, "MANIFEST", tmp_path / "BASELINE.json")
    monkeypatch.setattr(baseline, "_loadable", lambda run: True)
    # commit -> cohort/ tree. "a"*40 and "t"*40 are the same environment a
    # tooling commit apart; "b"*40 is a different one.
    monkeypatch.setattr(baseline, "cohort_tree",
                        lambda c: {"a" * 40: "env1", "t" * 40: "env1"}.get(c, "env2"))
    # audit_run reads metrics.csv for the give-back; absent here, so no gap
    monkeypatch.setattr("scripts.publish_audit.audit_run", lambda d: None)
    return tmp_path


def _audit(capsys) -> tuple[int, str]:
    code = baseline.audit()
    return code, capsys.readouterr().out


def test_a_complete_consistent_fleet_passes(fleet, capsys):
    code, out = _audit(capsys)
    assert code == 0, out
    assert "BASELINE OK" in out


def test_a_second_environment_fails_the_fleet(fleet, capsys):
    """The check the old fleet could not have passed: eight runs, seven commits."""
    _member(fleet, "platoon_v1", commit="b" * 40)

    code, out = _audit(capsys)

    assert code == 1
    assert "2 distinct cohort/ trees" in out
    assert "not one environment" in out.lower()


def test_a_tooling_commit_between_launches_does_not_fail_the_fleet(fleet, capsys):
    """The lesson from this campaign's own first hour.

    `fireteam_v9` was launched three commits after its lane-mates, and all three
    were tooling — scripts, tests, a README table. The `cohort/` tree was
    byte-identical across every one of them, so the runs trained in the same
    environment. A commit-equality gate would have failed the fleet for a reason
    that has nothing to do with the runs, which is how a gate teaches people to
    ignore it.
    """
    _member(fleet, "platoon_v1", commit="t" * 40)

    code, out = _audit(capsys)

    assert code == 0, out
    assert "one environment" in out
    assert "tooling-only differences are expected" in out


def test_a_commit_this_clone_cannot_resolve_is_not_agreement(fleet, capsys, monkeypatch):
    """Unknown provenance must fail, not pass quietly."""
    monkeypatch.setattr(baseline, "cohort_tree",
                        lambda c: None if c == "b" * 40 else "env1")
    _member(fleet, "platoon_v1", commit="b" * 40)

    code, out = _audit(capsys)

    assert code == 1
    assert "cannot resolve cohort/ for platoon_v1" in out


def test_a_reward_override_fails_the_fleet(fleet, capsys):
    """What ships is what was trained — an override means those differ."""
    _member(fleet, "defend_brique_v1", overrides=["defend_survivor_scale=0.35"])

    code, out = _audit(capsys)

    assert code == 1
    assert "reward overrides: defend_survivor_scale=0.35" in out


def test_a_smoke_test_sized_evaluation_fails_the_fleet(fleet, capsys):
    _member(fleet, "squad_v1", episodes=20)

    code, out = _audit(capsys)

    assert code == 1
    assert "N=20, needs 100" in out


def test_a_spot_check_over_the_peak_evaluation_fails_the_fleet(fleet, capsys):
    """Issue #45, the exact incident.

    `cohort.training.evaluate` writes `behavior.json` by DEFAULT, so a review's
    `--episodes 5` functional spot-check overwrote `platoon_v6`'s N=100 peak
    with an N=5 one and `git add -A` committed it (`a321329`, repaired in
    `bcdbfab`). Reproduced against this gate on the real tree before the fix:
    control and treatment both exited 0 with `BASELINE OK`, and their output
    diffed empty. The environment digest and the checkpoint digest were both
    untouched and both correct — what moved was the number derived from them,
    and the README published it as `1.00 ± 0.00 (N=5)` in the peak column for
    the whole window.
    """
    _member(fleet, "platoon_v1", peak_episodes=5)

    code, out = _audit(capsys)

    assert code == 1
    assert "peak evaluated at N=5, needs 100" in out
    assert "README publishes this cell" in out


def test_an_absent_peak_evaluation_is_named_rather_than_silently_ungated(fleet, capsys):
    """The hole a "when present, require N>=100" rule would have left open.

    `publish_audit.audit_run` returns None when `behavior.json` is missing, and
    `_run_facts` only applies the give-back gate `if a:` — so deleting the file
    does not merely skip the new N check, it silently switches off the
    stability gate as well and prints a `—` in the give-back column. Two gates
    stood down for one absence and nothing said so.
    """
    (fleet / "squad_v1" / "behavior.json").unlink()

    code, out = _audit(capsys)

    assert code == 1
    assert "no behavior.json" in out
    assert "give-back gate cannot run" in out


def test_a_failed_regression_gate_fails_the_fleet(fleet, capsys):
    _member(fleet, "defend_brique_v1", gates=["mean_distance_from_objective_under_threat"])

    code, out = _audit(capsys)

    assert code == 1
    assert "gate failed: mean_distance_from_objective_under_threat" in out


def test_an_unannounced_win_fails_the_fleet(fleet, capsys):
    """The v1.19 guarantee is structural, so a miss is a broken protocol.

    This is the check that would have caught platoon_v5's 0/100 at the fleet
    level rather than in a README column read three cycles later.
    """
    _member(fleet, "patrol_brique_v1", successes=99, announced=0)

    code, out = _audit(capsys)

    assert code == 1
    assert "announced 0/99" in out
    assert "guarantee is broken" in out


def test_a_missing_member_fails_the_fleet(fleet, capsys):
    manifest = json.loads((fleet / "BASELINE.json").read_text())
    del manifest["runs"]["platoon"]
    (fleet / "BASELINE.json").write_text(json.dumps(manifest))

    code, out = _audit(capsys)

    assert code == 1
    assert "coverage: no member for platoon" in out


def test_sealing_refuses_a_fleet_that_is_not_one_environment(fleet, capsys):
    _member(fleet, "platoon_v1", commit="b" * 40)

    assert baseline.seal("v1.19") == 1
    assert "refusing to seal" in capsys.readouterr().out


def test_sealing_records_the_environment_and_every_commit_in_it(fleet, capsys):
    _member(fleet, "platoon_v1", commit="t" * 40)  # same environment, later commit

    assert baseline.seal("v1.19") == 0

    manifest = json.loads((fleet / "BASELINE.json").read_text())
    assert manifest["cohort_tree"] == "env1"
    assert manifest["commits"] == sorted(["a" * 40, "t" * 40])
    assert manifest["commit"] is None, "two commits: there is no single one to name"
    assert manifest["version"] == "v1.19"


def test_a_sealed_manifest_detects_a_member_swapped_underneath_it(fleet, capsys):
    """Sealing is not a rubber stamp: it must notice the fleet moving after it."""
    baseline.seal("v1.19")
    for run in json.loads((fleet / "BASELINE.json").read_text())["runs"].values():
        _member(fleet, run, commit="c" * 40)  # a different environment entirely

    code, out = _audit(capsys)

    assert code == 1
    assert "sealed at cohort/ env1" in out


def test_sealing_stamps_a_digest_of_every_published_evaluation(fleet):
    """What `cohort_tree` and `checkpoint_sha256` between them do not cover.

    Those two pin the environment and the weights. The numbers *derived* from
    them were undigested, which is how a fleet stayed byte-identical in its
    manifest across a corruption of one of its published cells (issue #45).
    """
    assert baseline.seal("v1.19") == 0

    stamped = json.loads((fleet / "BASELINE.json").read_text())["artifacts"]

    assert set(stamped) == set(baseline.load()["runs"].values())
    for run, files in stamped.items():
        assert set(files) == set(baseline.PUBLISHED_EVALUATIONS), run
        for name, digest in files.items():
            assert digest == baseline.artifact_digest(fleet / run / name)


def test_an_evaluation_rewritten_after_the_seal_fails_the_fleet(fleet, capsys):
    """The durable half of the fix: drift is detectable from the tree alone.

    No live campaign to diff against, no README to re-read, no host state — the
    manifest says what the numbers were and the files either still hash to it or
    they do not. The one check that caught the real incident,
    `test_the_readme_table_matches_the_runs_on_disk`, skips whenever any member
    is RUNNING, so it would have been silent had the fleet still been in flight.
    """
    baseline.seal("v1.19")
    # Re-scored at a full N — so the evidence bar alone cannot see this, and
    # only the digest can. A silent re-score is exactly how a published cell
    # stops describing the checkpoint the rest of the row describes.
    (fleet / "platoon_v1" / "behavior.json").write_text(json.dumps({
        "episodes": 100, "success_ci95": "1.00 ± 0.00", "metrics": {"success_rate": 1.0}}))

    code, out = _audit(capsys)

    assert code == 1
    assert "behavior.json changed since the seal" in out
    assert "re-seal" in out


def test_an_evaluation_deleted_after_the_seal_fails_the_fleet(fleet, capsys):
    baseline.seal("v1.19")
    (fleet / "squad_v1" / "behavior_final.json").unlink()

    code, out = _audit(capsys)

    assert code == 1
    assert "behavior_final.json was sealed at" in out
    assert "is not on disk" in out


def test_a_member_added_after_the_seal_is_not_covered_by_it(fleet, capsys):
    """Silence would make the stamp evadable by editing the members list."""
    baseline.seal("v1.19")
    manifest = json.loads((fleet / "BASELINE.json").read_text())
    manifest["runs"]["platoon"] = "platoon_v2"
    (fleet / "BASELINE.json").write_text(json.dumps(manifest))
    _member(fleet, "platoon_v2")

    code, out = _audit(capsys)

    assert code == 1
    assert "stamps no evaluation for platoon_v2" in out


def test_a_manifest_written_before_stamping_existed_is_not_an_accusation(fleet, capsys):
    """Unstamped is not the same finding as changed.

    Every gate in this file that fires on missing information rather than on
    wrong information has taught somebody to ignore it. A manifest with no
    `artifacts` key predates the stamp; it has no opinion about these files.
    """
    manifest = json.loads((fleet / "BASELINE.json").read_text())
    assert "artifacts" not in manifest

    code, out = _audit(capsys)

    assert code == 0, out
    assert "BASELINE OK" in out
    assert "sealed" not in out


def test_every_evaluation_the_shipped_manifest_sealed_is_unchanged_on_disk():
    """The tree-only detector, run against the real fleet.

    Deliberately keyed on the *sealed* run names rather than the manifest's
    current members, so a re-baseline campaign that re-points `runs` before its
    new members land does not turn this red for hours — the old runs are still
    on disk (archived or not, `run_dir` finds either) and still hash to their
    seal. `scripts/baseline.py` is the gate that a campaign is finished; this is
    the gate that a finished one has not moved since.
    """
    stamped = baseline.load().get("artifacts")
    assert stamped, (
        "the shipped manifest carries no evaluation digests — re-seal it with "
        "scripts/baseline.py --seal (issue #45)"
    )

    drift = []
    for run, files in sorted(stamped.items()):
        for name, want in sorted(files.items()):
            got = baseline.artifact_digest(baseline.run_dir(run) / name)
            if got != want:
                drift.append(f"{run}/{name}: sealed {want[:12]}, on disk "
                             f"{got[:12] if got else 'ABSENT'}")
    assert not drift, "sealed evaluations have changed:\n  " + "\n  ".join(drift)


def test_an_uncommitted_final_policy_fails_the_fleet(fleet, capsys, monkeypatch):
    """Issue #44: a headline whose weights are not in the repository.

    `.gitignore` ignored `ckpt_latest.pt` fleet-wide, so every member shipped
    the number and withheld the policy that produces it. Nothing failed —
    `behavior_final.json` was present and complete, the gates were green, and
    the one artifact a reader needs to re-derive the figure was absent. That
    silence is why this is a gate and not a note.
    """
    monkeypatch.setattr(
        baseline, "_uncommitted",
        lambda run: ["ckpt_latest.pt"] if run == "squad_v1" else [],
    )

    code, out = _audit(capsys)

    assert code == 1
    assert "squad_v1: ckpt_latest.pt is not committed" in out
    assert "cannot re-derive it" in out


def test_a_tree_with_no_git_index_is_not_an_accusation(fleet):
    """The real `_uncommitted` against a directory git knows nothing about.

    A tarball export has no index, so it cannot distinguish "not committed"
    from "cannot tell" — and a gate that fires there fires for a reason that
    has nothing to do with the fleet, which is how a gate teaches people to
    ignore it. Silence is the only honest answer.
    """
    d = fleet / "squad_v1"
    (d / "ckpt_best.pt").write_text("stub")
    (d / "ckpt_latest.pt").write_text("stub")

    assert baseline._uncommitted("squad_v1") == []


def test_the_headline_checkpoint_is_the_final_policy_not_the_best_window():
    """What `_loadable` and `_uncommitted` are about, named once.

    The audit's `evidence` rule is explicit that the headline is the policy the
    run ended with. `_loadable` used to check `ckpt_best.pt` alone — the one
    checkpoint that rule says is not it.
    """
    assert baseline.HEADLINE_CKPT == "ckpt_latest.pt"
    assert set(baseline.CHECKPOINTS) == {"ckpt_best.pt", "ckpt_latest.pt"}


# ----------------------------------------------- the declared seed search --
#
# v1.21 made `closed_on_root_report_rate` a per-run bar read off the FINAL
# policy, and allowed the member for a bimodal scenario to be chosen from
# several seeds. That is a selection procedure, and the whole of what keeps it
# honest is that the manifest declares it and the audit checks the declaration
# against the committed artifacts. These tests are the check on the check.


def _search(fleet, scenario, runs):
    """Point one scenario's seed_search at ``runs`` and re-write the manifest."""
    manifest = json.loads((fleet / "BASELINE.json").read_text())
    manifest.setdefault("seed_search", {})[scenario] = runs
    (fleet / "BASELINE.json").write_text(json.dumps(manifest))


def test_a_declared_search_publishes_the_count_and_names_the_member(fleet, capsys):
    """The number the board rests on: k of K, counted from the artifacts.

    Not a rate with an interval — at K=4 an interval would be a decoration over
    four coin flips.
    """
    for seed, rate in ((12, 0.0), (18, 0.80), (19, 0.75), (14, 0.0)):
        _member(fleet, f"patrol_brique_seed{seed}", seed=seed, close_rate=rate)
    _search(fleet, "patrol_brique", [f"patrol_brique_seed{s}" for s in (12, 18, 19, 14)])
    manifest = json.loads((fleet / "BASELINE.json").read_text())
    manifest["runs"]["patrol_brique"] = "patrol_brique_seed18"
    (fleet / "BASELINE.json").write_text(json.dumps(manifest))

    code, out = _audit(capsys)
    assert code == 0, out
    assert "2 of 4 seeds report" in out
    assert "patrol_brique_seed18" in out and "<- member" in out


def test_a_scenario_with_no_search_says_so_rather_than_going_quiet(fleet, capsys):
    """"1 seed" and "1 of 4 seeds" are different claims.

    A reader who only ever sees the count where it is flattering learns nothing
    from its absence, so the unsearched scenarios are labelled too.
    """
    _, out = _audit(capsys)
    assert out.count("1 seed, not searched") == len(baseline.DOCTRINE_SCENARIOS)


def test_a_mute_member_is_named_in_the_reporting_section(fleet, capsys):
    """The v1.20b shape: every other axis fine, the commander never reports."""
    _member(fleet, "patrol_brique_v1", close_rate=0.0)
    _, out = _audit(capsys)
    assert "MUTE" in out


def test_a_search_the_published_member_was_not_part_of_fails_the_fleet(fleet, capsys):
    """The selection this is here to refuse: a count over runs that do not
    include the one being shipped."""
    _member(fleet, "patrol_brique_other", seed=18, close_rate=0.8)
    _search(fleet, "patrol_brique", ["patrol_brique_other"])
    code, out = _audit(capsys)
    assert code == 1
    assert "does not contain the member" in out


def test_a_search_spanning_two_environments_fails_the_fleet(fleet, capsys):
    """A reporting rate pooled over two trees is a rate over nothing — the same
    provenance rule the fleet itself is held to."""
    _member(fleet, "patrol_brique_v1", seed=12, close_rate=0.0)
    _member(fleet, "patrol_brique_alt", seed=18, close_rate=0.8, commit="b" * 40)
    _search(fleet, "patrol_brique", ["patrol_brique_v1", "patrol_brique_alt"])
    code, out = _audit(capsys)
    assert code == 1
    assert "distinct cohort/ trees" in out


def test_a_candidate_trained_with_an_override_fails_the_fleet(fleet, capsys):
    """A seed that only reports at a price the fleet does not ship is not
    evidence about the fleet."""
    _member(fleet, "patrol_brique_v1", seed=12, close_rate=0.0)
    _member(fleet, "patrol_brique_priced", seed=18, close_rate=0.8,
            overrides=["root_done_bonus=3.0"])
    _search(fleet, "patrol_brique", ["patrol_brique_v1", "patrol_brique_priced"])
    code, out = _audit(capsys)
    assert code == 1
    assert "not the configuration the fleet ships" in out


def test_an_unscored_candidate_counts_in_neither_direction(fleet, capsys):
    """Unmeasured is not mute. A candidate with no measured rate would silently
    deflate k/K if it were counted as a failure."""
    _member(fleet, "patrol_brique_v1", seed=12, close_rate=0.8)
    _member(fleet, "patrol_brique_unscored", seed=18, close_rate=None)
    _search(fleet, "patrol_brique", ["patrol_brique_v1", "patrol_brique_unscored"])
    facts = baseline.seed_search_facts(
        json.loads((fleet / "BASELINE.json").read_text()), "patrol_brique", "patrol_brique_v1")
    assert facts["reporting"] == 1 and facts["total"] == 2
    code, out = _audit(capsys)
    assert code == 1
    assert "counts in neither direction" in out


def test_the_reporting_gate_reads_the_final_policy_not_the_best_window(fleet):
    """The v1.21 decision, pinned.

    Measured against the 0.5 floor the shipping v1.19 fleet fails its own bar on
    ``ckpt_best`` in two members of eight and passes on the final policy at a
    minimum of 0.808 — so the artifact the gate reads is the one the project
    publishes. A member whose peak is mute and whose final policy reports must
    pass, which is the case that decides it.
    """
    d = _member(fleet, "squad_v1", close_rate=0.85)
    (d / "behavior.json").write_text(json.dumps({
        "episodes": 100, "success_ci95": "0.98 ± 0.03",
        "metrics": {"success_rate": 0.98, "closed_on_root_report_rate": 0.0},
    }))
    assert baseline._reporting_gate("squad_v1") is True


# ------------------------------------------- policy reproductions (issue #60) --
#
# Training is bit-deterministic in (seed, scenario, steps, lr, price), so a
# re-launch across commits that never touch the trajectory reproduces its
# predecessor exactly — all TWELVE v1.21 campaign runs did, and the campaign's
# pre-registered seed-carry test therefore compared five checkpoints with
# themselves. The audit now discloses such reproductions before any claim about
# the pair can be written. A disclosure, never a gate: the sealed fleet it
# happened to is honest, and a check that failed it would be firing at
# determinism doing its job.


def _stub_digests(monkeypatch):
    """``policy_digest`` as a content read of stub files — no torch in the loop.

    Two stub checkpoints "hold the same tensors" iff their text matches, which
    is the only property the disclosure consumes.
    """
    monkeypatch.setattr("scripts.publish_audit.policy_digest",
                        lambda p: p.read_text() if p.is_file() else None)


def _checkpoints(fleet, run: str, *, best: str, latest: str):
    d = fleet / run
    d.mkdir(exist_ok=True)
    (d / "ckpt_best.pt").write_text(best)
    (d / "ckpt_latest.pt").write_text(latest)


def test_a_member_reproducing_an_existing_policy_is_disclosed_not_failed(
        fleet, capsys, monkeypatch):
    """The fireteam_v12 shape: a member that is a bit-for-bit re-execution of a
    run outside the manifest. Said out loud, exit still 0."""
    _stub_digests(monkeypatch)
    _checkpoints(fleet, "squad_v1", best="W1", latest="W2")
    _checkpoints(fleet, "squad_v0", best="W1", latest="W2")  # earlier run, not in the manifest

    code, out = _audit(capsys)

    assert code == 0, out
    assert "BASELINE OK" in out
    assert "policy reproductions" in out
    assert "squad_v1" in out and "ckpt_best + ckpt_latest  ==  squad_v0" in out
    assert "identity, not a measurement" in out


def test_a_fleet_of_fresh_policies_prints_no_reproduction_section(fleet, capsys, monkeypatch):
    """Silence means measured-and-new, so the section must not cry wolf."""
    _stub_digests(monkeypatch)
    _checkpoints(fleet, "squad_v1", best="W1", latest="W2")
    _checkpoints(fleet, "squad_v0", best="X1", latest="X2")

    code, out = _audit(capsys)

    assert code == 0, out
    assert "policy reproductions" not in out


# --- a collapsed seed search candidate: 0/0 is not a hole in the record ---
#
# `closed_on_root_report_rate` is wins-closed-on-a-root-report over WINS, so a
# run that won nothing divides by zero and the evaluator writes null rather than
# a fabricated 0.0 — a made-up zero would read as a measured refusal to report,
# which is the vanished-denominator error this repo has already retracted a claim
# over. But the seed_search check read every null as "no measured rate" and
# refused the fleet. The v1.24 campaign is the first search to contain a
# collapsed candidate (squad_v30, platoon_v15_seed12, both 0/100), so the
# distinction had never been needed: a run with NO evaluation is a hole and must
# still block; a run evaluated at N=100 that simply won nothing is a fact and is
# disclosed instead.


def _search_with_member(fleet, scenario, runs, member):
    _search(fleet, scenario, runs)
    manifest = json.loads((fleet / "BASELINE.json").read_text())
    manifest["runs"][scenario] = member
    (fleet / "BASELINE.json").write_text(json.dumps(manifest))


def _collapsed(fleet, run: str, seed: int):
    """A candidate that won nothing, in the evaluator's own shape: the reporting
    key is PRESENT and null — never absent, and never a fabricated 0.0."""
    _member(fleet, run, seed=seed, successes=0, announced=0, close_rate=None)
    d = fleet / run / "behavior_final.json"
    blob = json.loads(d.read_text())
    blob["metrics"]["success_rate"] = 0.0
    blob["metrics"]["closed_on_root_report_rate"] = None
    d.write_text(json.dumps(blob))


def test_a_candidate_that_won_nothing_is_disclosed_not_refused(fleet, capsys):
    """0 wins => the rate is undefined by arithmetic, and the search still seals."""
    _member(fleet, "patrol_brique_seed12", seed=12, close_rate=0.8)
    _collapsed(fleet, "patrol_brique_seed18", seed=18)
    _search_with_member(fleet, "patrol_brique",
                        ["patrol_brique_seed12", "patrol_brique_seed18"],
                        "patrol_brique_seed12")

    code, out = _audit(capsys)

    assert code == 0, out
    assert "no wins — rate undefined" in out
    # it leaves the denominator rather than quietly counting as a mute seed
    assert "1 of 1 seeds report" in out and "1 won nothing" in out


def test_a_candidate_with_no_evaluation_at_all_still_refuses(fleet, capsys):
    """The other direction: a genuine hole in the record must keep blocking."""
    _member(fleet, "patrol_brique_seed12", seed=12, close_rate=0.8)
    _member(fleet, "patrol_brique_seed18", seed=18, close_rate=None)  # key absent
    _search_with_member(fleet, "patrol_brique",
                        ["patrol_brique_seed12", "patrol_brique_seed18"],
                        "patrol_brique_seed12")

    code, out = _audit(capsys)

    assert code == 1, out
    assert "no measured closed_on_root_report_rate" in out


def test_a_seed_search_candidate_reproducing_a_policy_is_disclosed_too(
        fleet, capsys, monkeypatch):
    """Where the v1.21 identity actually lived: four of the twelve reproductions
    were seed_search candidates, not members, and the seed-carry claim was read
    off exactly those runs."""
    _stub_digests(monkeypatch)
    for seed in (12, 18):
        _member(fleet, f"patrol_brique_seed{seed}", seed=seed, close_rate=0.8)
    _search(fleet, "patrol_brique", ["patrol_brique_seed12", "patrol_brique_seed18"])
    manifest = json.loads((fleet / "BASELINE.json").read_text())
    manifest["runs"]["patrol_brique"] = "patrol_brique_seed12"
    (fleet / "BASELINE.json").write_text(json.dumps(manifest))
    _checkpoints(fleet, "patrol_brique_seed18", best="OLD1", latest="OLD2")
    _checkpoints(fleet, "patrol_brique_old", best="OLD1", latest="OLD2")

    code, out = _audit(capsys)

    assert code == 0, out
    assert "policy reproductions" in out
    assert "patrol_brique_seed18" in out and "==  patrol_brique_old" in out


def test_a_partial_reproduction_names_only_the_matching_checkpoint(fleet, capsys, monkeypatch):
    """Same training, selection moved: ckpt_latest matches, ckpt_best does not.

    The disclosure must say which checkpoint carries the identity, because
    "same trajectory, different best window" and "same policy end to end" are
    different findings (#60 §2 counts them separately)."""
    _stub_digests(monkeypatch)
    _checkpoints(fleet, "squad_v1", best="W1", latest="W2")
    _checkpoints(fleet, "squad_v0", best="OTHER", latest="W2")

    code, out = _audit(capsys)

    assert code == 0, out
    line = next(ln for ln in out.splitlines() if "==  squad_v0" in ln)
    assert "ckpt_latest" in line and "ckpt_best" not in line


def test_the_policy_identity_is_the_tensors_not_the_file(tmp_path):
    """The instrument itself, against real checkpoints (#60 §3, the rdb seed-16
    pair): one set of tensors under two `root_done_bonus` tags is ONE policy,
    and a file-level hash would split it in two."""
    torch = pytest.importorskip("torch")
    from scripts.publish_audit import policy_digest

    weights = {"pi.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    a, b, c = tmp_path / "a.pt", tmp_path / "b.pt", tmp_path / "c.pt"
    torch.save({"model": weights, "reward_config": {"root_done_bonus": 3.0}}, a)
    torch.save({"model": weights, "reward_config": {"root_done_bonus": 1.0}}, b)
    torch.save({"model": {"pi.weight": weights["pi.weight"] + 1},
                "reward_config": {"root_done_bonus": 3.0}}, c)

    assert a.read_bytes() != b.read_bytes(), "the files differ — that is the point"
    assert policy_digest(a) == policy_digest(b), "one policy, two price tags"
    assert policy_digest(a) != policy_digest(c), "different tensors, different policy"
    assert policy_digest(tmp_path / "absent.pt") is None, "absence, not a verdict"


# ------------------------------------------------ the seed spread (issue #63) --
#
# `seed_search` declares the seeds a member was CHOSEN from; `seed_spread`
# (owner decision 2026-08-18) declares every OTHER same-config draw the record
# holds — archived, cross-tree, mute or unmeasured. The board said "2 of 2
# seeds report" for squad, true of the search, while `squad_v29_seed14` failed
# the reporting gate on both checkpoints in the same directory listing and
# archived `squad_v20_seed15` sat at 0.000: the quiet half of a search. The
# audit dedupes draws on the final policy's tensors (a bit-identical
# re-execution is one draw) and FAILS on any same-config draw in neither block.


def _spread(fleet, scenario, runs):
    manifest = json.loads((fleet / "BASELINE.json").read_text())
    manifest.setdefault("seed_spread", {})[scenario] = runs
    (fleet / "BASELINE.json").write_text(json.dumps(manifest))


def test_a_same_config_draw_in_neither_block_fails_the_audit_by_name(fleet, capsys):
    """The completeness gate — the whole point of the block.

    The exact #63 shape: a same-config draw lands beside the member, fails the
    gate it would qualify, and no declaration anywhere says it exists.
    """
    _member(fleet, "squad_v2_seed14", seed=14, scenario="squad", close_rate=0.0)

    code, out = _audit(capsys)

    assert code == 1
    assert "neither seed_search nor seed_spread: squad_v2_seed14" in out
    assert "more draws than the manifest declares" in out


def test_a_declared_spread_draw_is_counted_and_does_not_fail(fleet, capsys):
    _member(fleet, "squad_v2_seed14", seed=14, scenario="squad", close_rate=0.0)
    _spread(fleet, "squad", ["squad_v2_seed14"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "spread: +1 distinct draws over 1 more runs — 0 report, 1 mute" in out
    assert "known same-config draws: 1 of 2 report" in out


def test_an_unmeasured_spread_draw_counts_in_neither_direction_but_is_disclosed(
        fleet, capsys):
    """Unlike a search candidate, an unmeasured spread draw does not fail the
    audit: the archive is not re-scored to make a count look complete. It is
    disclosed as unmeasured and stays out of both the numerator and the mute
    count."""
    _member(fleet, "squad_v2_seed14", seed=14, scenario="squad", close_rate=None)
    _spread(fleet, "squad", ["squad_v2_seed14"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "0 report, 0 mute, 1 unmeasured" in out


def test_bit_identical_spread_runs_count_as_one_draw(fleet, capsys, monkeypatch):
    """#60's lesson applied to counting: `squad_v29_seed14` == archived
    `squad_v10c`, so listing both must add ONE draw, disclosed as one."""
    _stub_digests(monkeypatch)
    for name in ("squad_v2_seed14", "squad_v2b_seed14"):
        _member(fleet, name, seed=14, scenario="squad", close_rate=0.0)
        _checkpoints(fleet, name, best="S14", latest="S14L")
    _spread(fleet, "squad", ["squad_v2_seed14", "squad_v2b_seed14"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "spread: +1 distinct draws over 2 more runs" in out
    assert "squad_v2_seed14 = squad_v2b_seed14" in out
    assert "known same-config draws: 1 of 2 report" in out


def test_a_spread_run_bit_identical_to_the_member_adds_no_draw(fleet, capsys, monkeypatch):
    """The patrol_brique_v11_seed14 shape: the member re-derived an archived
    run bit-for-bit, so the archived run is the member's own draw — carried
    for completeness, folded out of the count, and said out loud."""
    _stub_digests(monkeypatch)
    _checkpoints(fleet, "squad_v1", best="W1", latest="W2")
    _member(fleet, "squad_v0", seed=12, scenario="squad", close_rate=0.85)
    _checkpoints(fleet, "squad_v0", best="W1", latest="W2")
    _spread(fleet, "squad", ["squad_v0"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "spread: +0 distinct draws over 1 more runs" in out
    assert "squad_v0  ==  squad_v1 (the same draw, counted once)" in out
    assert "known same-config draws: 1 of 1 report" in out


# ----------------------------------- the dedupe key's availability (issue #65) --
#
# `runs/archive/` prunes `ckpt_latest.pt` and keeps `ckpt_best.pt`, and the
# dedupe keyed on exactly the pruned file — so 36 of the 56 declared spread
# runs could not be deduped at all, every one silently counted as a distinct
# draw, and the docstring's own example pair (`squad_v10c` == `squad_v29_seed14`,
# bit-identical at ckpt_best) printed as two rows. Absence-as-distinct inflates
# the independent-draw count, the exact failure the block exists to prevent.
# Identity now rests on every checkpoint both runs hold, and a run whose
# on-disk checkpoints yield no digest is a failure, not a quiet degrade.


def test_an_archived_draw_missing_its_final_is_deduped_on_the_checkpoint_it_holds(
        fleet, capsys, monkeypatch):
    """The #65 shape itself: the archive pruned the final, ckpt_best agrees —
    one draw, folded into the member, annotated with what settled it."""
    _stub_digests(monkeypatch)
    _checkpoints(fleet, "squad_v1", best="W1", latest="W2")
    _member(fleet, "squad_old", seed=14, scenario="squad", close_rate=0.85)
    (fleet / "squad_old" / "ckpt_best.pt").write_text("W1")  # no ckpt_latest.pt
    _spread(fleet, "squad", ["squad_old"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "spread: +0 distinct draws over 1 more runs" in out
    assert ("squad_old  ==  squad_v1 (the same draw, counted once — "
            "settled at ckpt_best)") in out
    assert "known same-config draws: 1 of 1 report" in out


def test_two_archived_draws_agreeing_on_the_checkpoint_both_hold_are_one_draw(
        fleet, capsys, monkeypatch):
    """Both sides pruned: two archived same-seed runs with only ckpt_best each,
    bit-identical there — one draw, not two."""
    _stub_digests(monkeypatch)
    for name in ("squad_v2_seed14", "squad_v2b_seed14"):
        _member(fleet, name, seed=14, scenario="squad", close_rate=0.0)
        (fleet / name / "ckpt_best.pt").write_text("S14")  # no ckpt_latest.pt
    _spread(fleet, "squad", ["squad_v2_seed14", "squad_v2b_seed14"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "spread: +1 distinct draws over 2 more runs" in out
    assert "squad_v2_seed14 = squad_v2b_seed14" in out
    assert "settled at ckpt_best" in out


def test_a_run_whose_checkpoints_yield_no_digest_is_a_loud_finding_not_a_quiet_distinct(
        fleet, capsys, monkeypatch):
    """The one change that would have caught #65 at authoring time: a key that
    is unavailable for a run whose checkpoint files ARE on disk is the dedupe
    going blind, and blind must not silently count as distinct."""
    monkeypatch.setattr("scripts.publish_audit.policy_digest", lambda p: None)
    _checkpoints(fleet, "squad_old", best="W1", latest="W2")
    _member(fleet, "squad_old", seed=14, scenario="squad", close_rate=0.0)
    _spread(fleet, "squad", ["squad_old"])

    code, out = _audit(capsys)

    assert code == 1
    assert "squad_old holds checkpoint files but none can be digested" in out
    assert "inflates the spread" in out


def test_a_run_with_no_checkpoints_at_all_is_disclosed_but_not_failed(
        fleet, capsys, monkeypatch):
    """Honest absence stays honest: nothing on disk to digest is disclosed as
    unknown identity — the archive is not failed for a file it never had."""
    _stub_digests(monkeypatch)
    _member(fleet, "squad_old", seed=14, scenario="squad", close_rate=0.0)
    _spread(fleet, "squad", ["squad_old"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "identity underived" in out
    assert "squad_old" in out


def test_runs_grouped_through_an_intermediary_but_disagreeing_are_flagged(
        fleet, capsys, monkeypatch):
    """A best-only run bridging two runs whose finals differ would launder two
    final policies into one draw — ambiguous identity is a human's call."""
    _stub_digests(monkeypatch)
    _checkpoints(fleet, "squad_v1", best="X", latest="L1")
    for name, latest in (("squad_a_seed14", None), ("squad_b_seed14", "L2")):
        _member(fleet, name, seed=14, scenario="squad", close_rate=0.0)
        (fleet / name / "ckpt_best.pt").write_text("X")
        if latest:
            (fleet / name / "ckpt_latest.pt").write_text(latest)
    _spread(fleet, "squad", ["squad_a_seed14", "squad_b_seed14"])

    code, out = _audit(capsys)

    assert code == 1
    assert "disagree at ckpt_latest" in out
    assert "identity is ambiguous" in out


def test_every_declared_draw_holds_a_deduplicable_checkpoint():
    """The population-level key-availability assertion, against the real
    manifest: every run in every block holds at least one checkpoint the
    dedupe can digest. 64% unavailable was a silent fact; now it is a red
    test the moment an archive prune takes a run's last checkpoint."""
    manifest = baseline.load()
    runs = set(manifest["runs"].values())
    for block in ("seed_search", "seed_spread"):
        for declared in (manifest.get(block) or {}).values():
            runs.update(declared)
    keyless = [r for r in sorted(runs)
               if not any((baseline.run_dir(r) / n).is_file()
                          for n in baseline.CHECKPOINTS)]
    assert not keyless, (
        "declared runs with no checkpoint on disk — the spread dedupe has no "
        f"key for them and would count each as distinct: {keyless}"
    )


# --------------------------- the pruned final's recorded identity (issue #67) --
#
# #65's every-checkpoint key still settled 12 of its 13 merged groups "at
# ckpt_best" — the block treated the final as unknowable because the archive
# pruned `ckpt_latest.pt`, while every run's own behavior_final.json IS the
# evaluation of that checkpoint and records its file sha256 beside the
# metrics, written at eval time and committed. The dedupe now falls back to
# that recorded hash — file hash against file hash, never against a tensor
# digest (#61's confound: one policy, two byte-strings) — and a final that is
# neither on disk nor recorded anywhere is disclosed by name, because it is
# precisely the case the recovery does not fix.


def _recorded_final(fleet, run: str, sha: str):
    d = fleet / run
    payload = json.loads((d / "behavior_final.json").read_text())
    payload["checkpoint"] = f"runs/{run}/ckpt_latest.pt"
    payload["checkpoint_sha256"] = sha
    (d / "behavior_final.json").write_text(json.dumps(payload))


def test_a_pruned_finals_recorded_hash_settles_the_final_not_just_best(
        fleet, capsys, monkeypatch):
    """The #67 shape: the archive pruned the final, but behavior_final.json
    still holds its file sha256 — the merge is settled at BOTH checkpoints,
    so no "settled at ckpt_best" understatement is printed."""
    _stub_digests(monkeypatch)
    _checkpoints(fleet, "squad_v1", best="W1", latest="W2")
    _recorded_final(fleet, "squad_v1", "f" * 64)
    _member(fleet, "squad_old", seed=14, scenario="squad", close_rate=0.85)
    (fleet / "squad_old" / "ckpt_best.pt").write_text("W1")  # no ckpt_latest.pt
    _recorded_final(fleet, "squad_old", "f" * 64)
    _spread(fleet, "squad", ["squad_old"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "squad_old  ==  squad_v1 (the same draw, counted once)" in out
    assert "settled at ckpt_best" not in out
    assert "final policy unrecoverable" not in out


def test_recorded_final_hashes_that_disagree_keep_the_draws_apart(
        fleet, capsys, monkeypatch):
    """ckpt_best agreement does not entail final agreement — two runs sharing
    a best-save and diverging afterwards is exactly the shape the spread
    exists to count correctly, and before #67 this pair merged silently."""
    _stub_digests(monkeypatch)
    _checkpoints(fleet, "squad_v1", best="W1", latest="W2")
    _recorded_final(fleet, "squad_v1", "1" * 64)
    _member(fleet, "squad_old", seed=14, scenario="squad", close_rate=0.0)
    (fleet / "squad_old" / "ckpt_best.pt").write_text("W1")  # no ckpt_latest.pt
    _recorded_final(fleet, "squad_old", "2" * 64)
    _spread(fleet, "squad", ["squad_old"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "spread: +1 distinct draws over 1 more runs — 0 report, 1 mute" in out


def test_disagreeing_recorded_finals_with_no_tensor_do_not_split_the_draw(
        fleet, capsys, monkeypatch):
    """The #68 negative control: two runs tensor-identical at the checkpoint
    both still hold, finals pruned on BOTH sides, recorded final file hashes
    differing. A file hash is the stronger discriminator — one policy lives in
    two byte-strings whenever only the serialized price differs (#61) — so
    with no tensor to appeal to the disagreement is UNRESOLVED, not distinct:
    the pair merges on ckpt_best, is counted once, and the audit says which
    checkpoint could not be adjudicated rather than quietly counting two."""
    _stub_digests(monkeypatch)
    (fleet / "squad_v1" / "ckpt_best.pt").write_text("W1")  # no ckpt_latest.pt
    _recorded_final(fleet, "squad_v1", "1" * 64)
    _member(fleet, "squad_old", seed=14, scenario="squad", close_rate=0.85)
    (fleet / "squad_old" / "ckpt_best.pt").write_text("W1")  # no ckpt_latest.pt
    _recorded_final(fleet, "squad_old", "2" * 64)
    _spread(fleet, "squad", ["squad_old"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "spread: +0 distinct draws over 1 more runs" in out
    assert ("squad_old  ==  squad_v1 (the same draw, counted once — "
            "settled at ckpt_best)") in out
    assert "unresolved at ckpt_latest for squad_v1 = squad_old" in out
    assert "unadjudicated, not a second draw" in out


def test_tensor_identity_outranks_the_recorded_file_hash(fleet, capsys, monkeypatch):
    """The caveat that bit the assurance layer, refused here: a checkpoint
    serializes its reward_config, so one policy can live in two byte-strings
    (#61). Where both finals are on disk the tensors decide, and differing
    recorded file hashes must not split one draw into two."""
    _stub_digests(monkeypatch)
    _checkpoints(fleet, "squad_v1", best="W1", latest="W2")
    _recorded_final(fleet, "squad_v1", "1" * 64)
    _member(fleet, "squad_old", seed=14, scenario="squad", close_rate=0.85)
    _checkpoints(fleet, "squad_old", best="W1", latest="W2")
    _recorded_final(fleet, "squad_old", "2" * 64)
    _spread(fleet, "squad", ["squad_old"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "squad_old  ==  squad_v1 (the same draw, counted once)" in out


def test_a_record_that_names_a_different_checkpoint_lends_no_identity(
        fleet, capsys, monkeypatch):
    """The guard on the fallback: a behavior_final.json whose `checkpoint`
    field does not name ckpt_latest.pt is a mis-keyed record, and its hash
    must not stand in for the final's identity."""
    _stub_digests(monkeypatch)
    _checkpoints(fleet, "squad_v1", best="W1", latest="W2")
    _recorded_final(fleet, "squad_v1", "f" * 64)
    _member(fleet, "squad_old", seed=14, scenario="squad", close_rate=0.85)
    (fleet / "squad_old" / "ckpt_best.pt").write_text("W1")  # no ckpt_latest.pt
    payload = json.loads((fleet / "squad_old" / "behavior_final.json").read_text())
    payload["checkpoint"] = "runs/squad_old/ckpt_best.pt"  # mis-keyed
    payload["checkpoint_sha256"] = "f" * 64
    (fleet / "squad_old" / "behavior_final.json").write_text(json.dumps(payload))
    _spread(fleet, "squad", ["squad_old"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "settled at ckpt_best" in out  # the final stays unsettled
    assert "final policy unrecoverable" in out


def test_a_final_no_record_can_recover_is_disclosed_by_name(fleet, capsys, monkeypatch):
    """The squad_screen_v12 shape: final pruned AND no recorded hash — the one
    case the recovery does not reach, said out loud rather than left to read
    as an ordinary independent draw."""
    _stub_digests(monkeypatch)
    _checkpoints(fleet, "squad_v1", best="W1", latest="W2")
    _member(fleet, "squad_old", seed=14, scenario="squad", close_rate=0.0)
    (fleet / "squad_old" / "ckpt_best.pt").write_text("OTHER")  # no ckpt_latest.pt
    _spread(fleet, "squad", ["squad_old"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "final policy unrecoverable for 1 run(s)" in out
    assert "squad_old" in out


def test_the_shipped_records_final_identities_are_recoverable_but_one():
    """The population-level #67 fact, pinned: for every run in every declared
    block, the FINAL policy's identity survives — the file itself on disk, or
    the file sha256 its evaluation recorded — except `squad_screen_v12`, the
    single known loss (final pruned, never evaluated, in no corpus). A second
    name appearing here is a new unrecoverable final, not an ordinary prune."""
    manifest = baseline.load()
    runs = set(manifest["runs"].values())
    for block in ("seed_search", "seed_spread"):
        for declared in (manifest.get(block) or {}).values():
            runs.update(declared)
    lost = [r for r in sorted(runs)
            if not (baseline.run_dir(r) / baseline.HEADLINE_CKPT).is_file()
            and not baseline._recorded_file_hash(r, baseline.HEADLINE_CKPT)]
    assert set(lost) <= {"squad_screen_v12"}, (
        f"final-policy identity unrecoverable for {lost} — neither the file "
        "nor a recorded checkpoint_sha256 survives"
    )


def test_a_cross_tree_spread_draw_is_annotated_not_failed(fleet, capsys):
    """Owner decision: cross-tree draws are carried — they are evidence about
    the seed, not about the sealed environment — and the audit says which."""
    _member(fleet, "squad_v0_seed13", seed=13, scenario="squad",
            commit="b" * 40, close_rate=0.0)
    _spread(fleet, "squad", ["squad_v0_seed13"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "cross-tree env2" in out


def test_a_spread_run_that_does_not_resolve_fails_the_fleet(fleet, capsys):
    _spread(fleet, "squad", ["squad_ghost"])

    code, out = _audit(capsys)

    assert code == 1
    assert "seed_spread[squad]: no run directory for squad_ghost" in out


def test_a_spread_run_of_a_different_config_fails_the_fleet(fleet, capsys):
    """A run that trained different hyper-parameters is a different experiment;
    counting it as a draw of the member's lottery would launder it."""
    d = _member(fleet, "squad_other", seed=14, scenario="squad", close_rate=0.0)
    (d / "config.json").write_text(json.dumps(
        {"scenario": "squad", "seed": 14, "total_steps": 999}))
    _spread(fleet, "squad", ["squad_other"])

    code, out = _audit(capsys)

    assert code == 1
    assert "not the member's modulo seed" in out


def test_a_spread_run_with_an_override_fails_the_fleet(fleet, capsys):
    _member(fleet, "squad_priced", seed=14, scenario="squad", close_rate=0.8,
            overrides=["root_done_bonus=3.0"])
    _spread(fleet, "squad", ["squad_priced"])

    code, out = _audit(capsys)

    assert code == 1
    assert "root_done_bonus=3.0" in out
    assert "not a draw of the shipped configuration" in out


def test_a_run_in_both_blocks_fails_the_fleet(fleet, capsys):
    """One draw, two blocks, two counts — the double-count the dedupe exists
    to refuse, at the declaration level."""
    _member(fleet, "squad_v0_seed13", seed=13, scenario="squad", close_rate=0.8)
    _search(fleet, "squad", ["squad_v1", "squad_v0_seed13"])
    _spread(fleet, "squad", ["squad_v0_seed13"])

    code, out = _audit(capsys)

    assert code == 1
    assert "counted in two blocks" in out


def test_an_override_run_is_not_an_undeclared_draw(fleet, capsys):
    """A different price is a different experiment (`overrides_match`), so a
    priced run at the member's config must not fail the completeness gate."""
    _member(fleet, "squad_priced", seed=14, scenario="squad",
            overrides=["root_done_bonus=3.0"])

    code, out = _audit(capsys)

    assert code == 0, out
    assert "seed_spread" not in out


def test_every_spread_run_the_shipped_manifest_declares_resolves_and_matches():
    """The real manifest's block, held to its own rules — every declared draw
    resolves (live or archived), trained the member's exact config modulo
    seed, and carries no reward override."""
    manifest = baseline.load()
    spread = manifest.get("seed_spread")
    assert spread, "the shipped manifest declares no seed_spread block"
    for scenario in spread:
        facts = baseline.seed_spread_facts(
            manifest, scenario, manifest["runs"][scenario], digest=lambda p: None)
        for r in facts["runs"]:
            assert r["exists"], f"seed_spread[{scenario}]: {r['run']} does not resolve"
            assert r["config_matches"], (
                f"seed_spread[{scenario}]: {r['run']} is not the member's config modulo seed")
            assert not r["overrides"], f"seed_spread[{scenario}]: {r['run']} carries overrides"


def test_every_run_the_shipped_manifest_declares_is_in_the_repository():
    """Issue #66, strengthened by #69: "declared" must mean the repository
    holds an artifact the dedupe can key on, not merely some tracked file.

    ``platoon_v10_seed12`` was declared with ZERO tracked files (#66); then
    ``platoon_v12_seed12`` was declared with exactly two — config.json and
    economics.json, the cheap metadata — while the identity-bearing artifacts
    (its checkpoints, or an evaluation recording their hashes) stayed
    untracked, so this gate reported "in the repository" for a run the two
    identity gates above reject in any clone. The predicate here is now the
    one those gates actually consume — at least one tracked checkpoint, or a
    tracked evaluation whose recorded ``checkpoint_sha256`` recovers one — so
    the three gates agree on what "declared" means. Skipped only where git
    cannot answer at all (a tarball export), which is silence, not a verdict.
    """
    manifest = baseline.load()
    declared: dict[str, str] = {r: "runs" for r in manifest["runs"].values()}
    for block in ("seed_search", "seed_spread"):
        for scenario, runs in (manifest.get(block) or {}).items():
            for r in runs:
                declared.setdefault(r, f"{block}[{scenario}]")
    unkeyed = []
    for run, where in sorted(declared.items()):
        tracked = baseline.tracked_files(run)
        if tracked is None:
            pytest.skip("git cannot answer here — no index, no verdict")
        keyed = (any(n in tracked for n in baseline.CHECKPOINTS)
                 or any(rec in tracked and baseline._recorded_file_hash(run, ckpt)
                        for ckpt, rec in baseline.RECORDED_EVAL.items()))
        if not keyed:
            unkeyed.append(f"{where}: {run} (tracked: {sorted(tracked) or 'nothing'})")
    assert not unkeyed, (
        "declared runs whose identity-bearing artifacts are not in the "
        "repository — no tracked checkpoint and no tracked evaluation "
        "recording one's hash — so the dedupe and identity gates fail in any "
        "clone:\n  " + "\n  ".join(unkeyed)
    )


def test_the_shipped_record_holds_no_draw_outside_the_declared_blocks():
    """The completeness gate against the real corpus — the actual claim the
    v1.21 board understated, now pinned. If a new same-config run lands, this
    fails until the manifest says so."""
    manifest = baseline.load()
    for scenario, member in manifest["runs"].items():
        facts = baseline.seed_spread_facts(manifest, scenario, member,
                                           digest=lambda p: None)
        if facts is None:
            # The scan ran and the record holds no other same-config draw —
            # true of `platoon_hard` at v1.22, whose only clean-config run is
            # the member itself (its bit-identical override twin is a different
            # recorded experiment, disclosed by the reproductions section).
            # The eight v1.21 members all have draws, so they still exercise
            # the branch below.
            continue
        assert not facts["undeclared"], (
            f"{scenario}: same-config draws in neither seed_search nor seed_spread: "
            f"{facts['undeclared']} — declare them in runs/BASELINE.json"
        )


# --- The price that ships as a default, not as a flag (autocycle 2026-08-31) ---
#
# Identity was split across two readings and a price could hide between them:
# `config_matches` reads config.json, which records the PPO hyperparameters and
# NOTHING about the reward, and `overrides_match` reads economics.json's
# `reward_overrides`, which is populated only from --reward flags. But CLAUDE.md
# forbids a --reward override in a baseline run, so the mandated way to change a
# price — editing the default in cohort/env/rewards.py — was invisible to both,
# and two runs either side of a reward change read as the same experiment.
#
# It reached a campaign. v1.24 armed `bunching_penalty = -0.05` as a default;
# campaign_preflight refused all 18 jobs as already-answered and the campaign
# launched under FORCE=1 over a populated record to get past the refusal.


def test_a_price_that_shipped_as_a_default_is_a_different_experiment(monkeypatch):
    """The hazard itself: no overrides on either side, same config, different
    default price. Before the fix this read as the same experiment."""
    monkeypatch.setattr(baseline, "reward_defaults", lambda c: {"bunching_penalty": -0.05})

    assert not baseline.prices_match("armed", [], {"bunching_penalty": 0.0}, [])
    assert baseline.prices_match("armed", [], {"bunching_penalty": -0.05}, [])


def test_an_unarmed_new_field_does_not_split_the_record(monkeypatch):
    """Adding `burst_fraction = 0.0` to the dataclass must not retroactively make
    every run in the record a different experiment from every other. A field on
    one side only counts as a difference when its default is non-zero."""
    monkeypatch.setattr(baseline, "reward_defaults", lambda c: {"time_penalty": -0.01})

    assert baseline.prices_match("old", [], {"time_penalty": -0.01, "burst_fraction": 0.0}, [])
    assert not baseline.prices_match("old", [], {"time_penalty": -0.01, "burst_fraction": 0.5}, [])


def test_an_unresolvable_price_stays_a_suspect(monkeypatch):
    """Unknown is not a finding — the convention `cohort_tree` and
    `overrides_match` already keep. A commit this clone cannot resolve must read
    as a possible match, because a false match is declared and read while a false
    difference is silent."""
    monkeypatch.setattr(baseline, "reward_defaults", lambda c: None)

    assert baseline.prices_match(None, [], {"bunching_penalty": -0.05}, [])
    assert baseline.prices_match("unknown-commit", [], None, [])
    # A recorded FLAG still separates them: only the defaults half is unknown.
    assert not baseline.prices_match("unknown-commit", ["a=1.0"], None, [])


def test_reward_defaults_reads_the_shipped_dataclass():
    """The reader is exercised against the real tree, not only against fakes —
    an ast walk that silently stopped finding RewardConfig would return an empty
    dict and make every run match every other."""
    current = baseline.current_reward_defaults()

    assert "time_penalty" in current and "bunching_penalty" in current
    resolved = baseline.reward_defaults("HEAD")
    if resolved is not None:  # a tarball export has no git; that is silence
        assert "time_penalty" in resolved
