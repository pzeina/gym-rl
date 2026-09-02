"""The campaign pre-flight refuses a job whose policy the record already holds.

Assurance #63: `squad_v29_seed14` spent 3M steps re-deriving archived
`squad_v10c` bit-for-bit — same (scenario, seed, steps, hyper-parameters,
prices), and the cohort/ tree transition between them moved no trajectory. The
N=100 evaluation the job was queued to produce was already on disk. These tests
pin the check that makes that impossible to do by accident: before
`train_queue.sh` detaches, every job's predicted `config.json` is looked up in
the record (live and archived runs alike) and a hit refuses the queue unless
FORCE=1 says the re-derivation is deliberate.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import baseline, campaign_preflight
from scripts.baseline import cohort_tree
from scripts.fleet_status import find_run

ROOT = Path(__file__).resolve().parents[1]

SQUAD_SEED14_ARGS = ["--scenario", "squad", "--total-steps", "3000000", "--seed", "14"]


def _record(runs_dir: Path, name: str, config: dict, *, archived: bool = False,
            git_commit: str | None = "HEAD", overrides: list[str] | None = None,
            economics: bool = True) -> None:
    d = (runs_dir / "archive" / name) if archived else (runs_dir / name)
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config))
    if economics:
        (d / "economics.json").write_text(
            json.dumps({"git_commit": git_commit, "reward_overrides": overrides or []})
        )


def test_predicted_config_reproduces_a_committed_runs_config_json():
    """The CLI mirror must not drift from what train.py actually writes.

    `squad_v29_seed14` is the run #63 is about; its committed config.json is
    exactly what `predicted_config` must reconstruct from the jobs-file args
    that launched it. If `cohort.training.train` grows or changes a flag, this
    is the tripwire that says the mirror in campaign_preflight.py is stale.
    """
    run = find_run("squad_v29_seed14", ROOT / "runs")
    assert run is not None
    committed = json.loads((run / "config.json").read_text())
    predicted, overrides, exemption = campaign_preflight.predicted_config(SQUAD_SEED14_ARGS)
    assert exemption is None
    assert overrides == []
    assert predicted == committed


def test_the_63_pair_is_caught_before_the_queue():
    """Queuing squad/seed-14/3M against the real record flags the #63 identity.

    The archived `squad_v10c` (pre-seal tree) and `squad_v29_seed14` (sealed
    tree) both hold this exact config, and #63 showed their checkpoints are
    bit-identical. The pre-flight must surface both — the same-tree hit as a
    certain DUPLICATE, the cross-tree hit as the record plausibly already
    containing the policy.

    The launch tree AND the launch price are pinned to the ones
    `squad_v29_seed14` trained on (the sealed tree, unpriced), not to HEAD and
    not to the working tree — both legitimately move whenever a cycle touches
    cohort/ or arms a price, and this test is about the record, not about where
    the current cycle happens to sit. Pinning only the tree was enough until
    v1.24 armed `bunching_penalty` as a default: the pair then correctly stopped
    being duplicates of a job launched TODAY, which is the fix working, not the
    #63 identity going unnoticed.

    The OBSERVATION WIDTH is pinned for the third time and for the same reason:
    the pair trained on a vector the current tree no longer presents (the v1.26
    cycle took OBS_DIM 351 -> 346), and a job launched TODAY genuinely could not
    re-derive a policy whose first layer is a different shape. That is the width
    check working, so this test asks the question in the pair's own world.
    """
    run = find_run("squad_v29_seed14", ROOT / "runs")
    sealed_commit = json.loads((run / "economics.json").read_text())["git_commit"]
    config, overrides, _ = campaign_preflight.predicted_config(SQUAD_SEED14_ARGS)
    matches = campaign_preflight.find_duplicates(
        config, overrides, "squad_v30_seed14", ROOT / "runs", cohort_tree(sealed_commit),
        current_prices=baseline.reward_defaults(sealed_commit),
        current_obs_dim=campaign_preflight.recorded_obs_dim(run))
    by_run = {m.run: m for m in matches}
    assert "squad_v10c" in by_run, "the archived original must be found through the archive"
    assert "squad_v29_seed14" in by_run
    assert by_run["squad_v29_seed14"].same_tree is True
    assert by_run["squad_v10c"].same_tree is False  # pre-seal tree c0f85409


def test_a_different_observation_width_is_not_a_duplicate(tmp_path):
    """The v1.26 refusal: 21 jobs blocked by runs no tree could re-derive.

    Same config, same prices, but the recorded run's policy takes a 351-wide
    input and the tree now presents 346. The checkpoints are not reproducible —
    they are not even loadable — so this is a different experiment, not another
    draw, and FORCE=1 must not be the price of saying so.
    """
    import torch

    runs = tmp_path / "runs"
    _record(runs, "old", campaign_preflight.predicted_config(SQUAD_SEED14_ARGS)[0])
    torch.save({"obs_dim": 351}, runs / "old" / "ckpt_best.pt")
    config, overrides, _ = campaign_preflight.predicted_config(SQUAD_SEED14_ARGS)
    assert campaign_preflight.find_duplicates(
        config, overrides, "new", runs, cohort_tree("HEAD"), current_obs_dim=346) == []
    # same width -> still a duplicate, so the rule cannot swallow the check
    assert [m.run for m in campaign_preflight.find_duplicates(
        config, overrides, "new", runs, cohort_tree("HEAD"), current_obs_dim=351)] == ["old"]


def test_an_unreadable_observation_width_leaves_the_run_a_suspect(tmp_path):
    """Fails CLOSED: unknown is not different — the same rule the price channel
    follows. A run with no checkpoint on disk stays a duplicate."""
    runs = tmp_path / "runs"
    _record(runs, "old", campaign_preflight.predicted_config(SQUAD_SEED14_ARGS)[0])
    config, overrides, _ = campaign_preflight.predicted_config(SQUAD_SEED14_ARGS)
    assert [m.run for m in campaign_preflight.find_duplicates(
        config, overrides, "new", runs, cohort_tree("HEAD"), current_obs_dim=346)] == ["old"]


def test_a_job_reproducing_a_recorded_config_refuses_the_queue(tmp_path):
    _record(tmp_path / "runs", "old",
            campaign_preflight.predicted_config(SQUAD_SEED14_ARGS)[0])
    jobs = [("new", SQUAD_SEED14_ARGS)]
    lines, flagged = campaign_preflight.preflight(jobs, tmp_path / "runs", cohort_tree("HEAD"))
    assert flagged == 1
    assert any("DUPLICATE" in line and "old" in line for line in lines)


def test_an_archived_run_counts_as_the_record_too(tmp_path):
    """#63's original sat in runs/archive/ — archiving is a move, not an exit."""
    _record(tmp_path / "runs", "old",
            campaign_preflight.predicted_config(SQUAD_SEED14_ARGS)[0],
            archived=True, git_commit="0" * 40)  # unresolvable commit: unknown tree
    lines, flagged = campaign_preflight.preflight(
        [("new", SQUAD_SEED14_ARGS)], tmp_path / "runs", cohort_tree("HEAD"))
    assert flagged == 1
    assert any("runs/archive/old" in line for line in lines)
    assert not any("DUPLICATE" in line for line in lines)  # unknown tree is not "same tree"


def test_a_different_seed_is_not_a_duplicate(tmp_path):
    _record(tmp_path / "runs", "old",
            campaign_preflight.predicted_config(SQUAD_SEED14_ARGS)[0])
    _, flagged = campaign_preflight.preflight(
        [("new", ["--scenario", "squad", "--total-steps", "3000000", "--seed", "99"])],
        tmp_path / "runs", cohort_tree("HEAD"))
    assert flagged == 0


def test_different_reward_overrides_are_a_different_experiment(tmp_path):
    """A --reward override changes the prices, so the config match is no identity."""
    args = [*SQUAD_SEED14_ARGS, "--reward", "done_false=-2.0"]
    config, overrides, _ = campaign_preflight.predicted_config(args)
    assert overrides == ["done_false=-2.0"]
    _record(tmp_path / "runs", "priced", config, overrides=["done_false=-4.0"])
    _record(tmp_path / "runs", "same_price", config, overrides=["done_false=-2.0"])
    lines, flagged = campaign_preflight.preflight([("new", args)], tmp_path / "runs",
                                                  cohort_tree("HEAD"))
    assert flagged == 1
    assert any("same_price" in line for line in lines)
    assert not any("priced:" in line or " priced " in line for line in lines)


def test_a_run_without_recorded_economics_stays_a_suspect(tmp_path):
    """Unknown prices are not different prices — pre-economics runs still flag."""
    _record(tmp_path / "runs", "ancient",
            campaign_preflight.predicted_config(SQUAD_SEED14_ARGS)[0], economics=False)
    _, flagged = campaign_preflight.preflight(
        [("new", SQUAD_SEED14_ARGS)], tmp_path / "runs", cohort_tree("HEAD"))
    assert flagged == 1


def test_a_warm_start_job_is_exempt(tmp_path):
    """--init-from makes the outcome depend on more than the config."""
    _record(tmp_path / "runs", "old",
            campaign_preflight.predicted_config(SQUAD_SEED14_ARGS)[0])
    lines, flagged = campaign_preflight.preflight(
        [("new", [*SQUAD_SEED14_ARGS, "--init-from", "runs/old/ckpt_best.pt"])],
        tmp_path / "runs", cohort_tree("HEAD"))
    assert flagged == 0
    assert any("skipped" in line for line in lines)


def test_a_force_rerun_of_the_same_name_is_the_queues_call(tmp_path):
    """The job's own directory is not a duplicate of itself — FORCE re-runs exist."""
    _record(tmp_path / "runs", "same_name",
            campaign_preflight.predicted_config(SQUAD_SEED14_ARGS)[0])
    _, flagged = campaign_preflight.preflight(
        [("same_name", SQUAD_SEED14_ARGS)], tmp_path / "runs", cohort_tree("HEAD"))
    assert flagged == 0


def test_jobs_file_parsing_skips_comments_and_blanks():
    text = "# a campaign\n\nrun_a --scenario squad --seed 12\n  # indented comment\nrun_b --seed 13\n"
    assert campaign_preflight.parse_jobs(text) == [
        ("run_a", ["--scenario", "squad", "--seed", "12"]),
        ("run_b", ["--seed", "13"]),
    ]


def test_cli_exits_nonzero_when_the_record_already_answers_a_job(tmp_path, monkeypatch, capsys):
    import pytest

    _record(tmp_path / "runs", "old",
            campaign_preflight.predicted_config(SQUAD_SEED14_ARGS)[0])
    jobs = tmp_path / "campaign.jobs"
    jobs.write_text("new --scenario squad --total-steps 3000000 --seed 14\n")
    monkeypatch.setattr("sys.argv", ["campaign_preflight.py", str(jobs),
                                     "--runs-dir", str(tmp_path / "runs")])
    with pytest.raises(SystemExit) as exc:
        campaign_preflight.main()
    assert exc.value.code == 1
    assert "FORCE=1" in capsys.readouterr().out

    jobs.write_text("new --scenario squad --total-steps 3000000 --seed 4242\n")
    campaign_preflight.main()  # returns without raising
    assert "no config already in the record" in capsys.readouterr().out
