"""Refuse to queue a training job whose policy the record already holds.

Assurance #63: `squad_v29_seed14` — one of the 13 runs of the 2026-08-18
measurement campaign — turned out to be archived `squad_v10c` re-executed
bit-for-bit, on both checkpoints. Training here is bit-deterministic in
(scenario, seed, steps, hyper-parameters, prices), the two runs shared all of
them, and no `cohort/` transition between their trees touched a trajectory —
so 3M steps of CPU re-derived a policy whose N=100 evaluation was already on
disk. The same phenomenon, one layer up, is #60: twelve v1.21 "re-rolls" that
were identities.

The cheap check both issues call for happens BEFORE the queue, not after the
hashes agree: parse each job into the exact ``config.json`` its training run
would write, and look that config up in the record (``runs/`` and
``runs/archive/`` alike, via ``fleet_status.run_dirs``). A hit means the job
would re-derive an existing run — certainly if the cohort/ trees match,
plausibly even across trees (#63's pair straddled one). Either way the queue
should stop and make the duplication a decision instead of a surprise;
``FORCE=1`` on the queue is the decision recorded.

``train_queue.sh`` runs this over the whole jobs file in its up-front
validation pass. Standalone:

    .venv/bin/python scripts/campaign_preflight.py <jobs-file>

Exit 0: no job's config is already in the record. Exit 1: at least one is
(details on stdout). Jobs with ``--init-from`` are exempt — a warm start makes
the outcome depend on more than the config — and jobs whose ``--reward``
overrides differ from a recorded run's are not duplicates of it (a different
price is a different experiment).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cohort.training.ppo import PPOConfig  # noqa: E402
from scripts.baseline import cohort_tree  # noqa: E402
from scripts.fleet_status import run_dirs  # noqa: E402


def parse_jobs(text: str) -> list[tuple[str, list[str]]]:
    """Jobs-file lines -> (run-name, train args), comments and blanks skipped."""
    jobs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, *args = line.split()
        jobs.append((name, args))
    return jobs


class _QuietParser(argparse.ArgumentParser):
    """Raise instead of printing usage + exiting — bad lines are the queue's to reject."""

    def error(self, message: str) -> None:  # argparse override
        raise ValueError(message)


def _train_arg_parser() -> argparse.ArgumentParser:
    """The determinism-relevant slice of ``cohort.training.train`` main()'s CLI.

    Defaults are read off ``PPOConfig`` exactly as train.py reads them, so an
    omitted flag predicts the same value train.py would record.
    ``tests/test_campaign_preflight.py`` pins this mirror against a committed
    run's ``config.json``; if train.py grows or changes a flag, that test is
    the tripwire.
    """
    p = _QuietParser(add_help=False)
    p.add_argument("--scenario", default="fireteam")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    p.add_argument("--gae-lambda", type=float, default=PPOConfig.gae_lambda)
    p.add_argument("--clip-coef", type=float, default=PPOConfig.clip_coef)
    p.add_argument("--vf-coef", type=float, default=PPOConfig.vf_coef)
    p.add_argument("--max-grad-norm", type=float, default=PPOConfig.max_grad_norm)
    p.add_argument("--update-epochs", type=int, default=PPOConfig.update_epochs)
    p.add_argument("--num-minibatches", type=int, default=PPOConfig.num_minibatches)
    p.add_argument("--target-kl", type=float, default=PPOConfig.target_kl)
    p.add_argument("--hidden", type=int, default=PPOConfig.hidden)
    p.add_argument("--normalize-value", action=argparse.BooleanOptionalAction,
                   default=PPOConfig.normalize_value)
    p.add_argument("--separate-critic", action=argparse.BooleanOptionalAction,
                   default=PPOConfig.separate_critic)
    p.add_argument("--reward", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu")
    # Accepted so a real jobs line parses; irrelevant to the prediction.
    p.add_argument("--run-name", default=None)
    p.add_argument("--init-from", default=None)
    p.add_argument("--no-tb", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    return p


def predicted_config(train_args: list[str]) -> tuple[dict | None, list[str], str | None]:
    """The ``config.json`` dict a job would write, its overrides, or why not.

    Returns ``(config, reward_overrides, exemption)``; ``config`` is None when
    the job is exempt (``--init-from``: a warm start's outcome is not a
    function of the config) or unparseable (left for the queue's own
    validation to reject with a better message).
    """
    parser = _train_arg_parser()
    try:
        args = parser.parse_args(train_args)
    except (ValueError, argparse.ArgumentError) as exc:
        return None, [], f"unparseable args ({exc})"
    if args.init_from:
        return None, [], "--init-from: warm start, not config-determined"
    cfg = PPOConfig(
        n_envs=args.n_envs,
        horizon=args.horizon,
        lr=args.lr,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        target_kl=args.target_kl if args.target_kl and args.target_kl > 0 else None,
        hidden=args.hidden,
        normalize_value=args.normalize_value,
        separate_critic=args.separate_critic,
        device=args.device,
    )
    config = {"scenario": args.scenario, "seed": args.seed,
              "total_steps": args.total_steps, **asdict(cfg)}
    # Round-trip through JSON so equality against a config.json read off disk
    # compares like with like (tuples->lists, float canonicalisation).
    return json.loads(json.dumps(config)), list(args.reward), None


@dataclass
class Match:
    """An existing run whose recorded config a queued job reproduces."""

    run: str
    path: Path
    tree: str | None  # cohort/ tree the existing run trained on, if resolvable
    same_tree: bool  # True only when both trees resolve and agree


def find_duplicates(config: dict, overrides: list[str], run_name: str,
                    runs_dir: Path, current_tree: str | None) -> list[Match]:
    """Every run in the record (live or archived) this job would re-derive."""
    matches = []
    for d in run_dirs(runs_dir):
        if d.name == run_name:  # a FORCE re-run of itself is the queue's call
            continue
        try:
            existing = json.loads((d / "config.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if existing != config:
            continue
        try:
            econ = json.loads((d / "economics.json").read_text())
        except (OSError, json.JSONDecodeError):
            econ = {}
        # Different recorded prices -> a different experiment, not a duplicate.
        # An unrecorded price (pre-economics run) stays a suspect: unknown is
        # not different.
        recorded = econ.get("reward_overrides")
        if recorded is not None and sorted(recorded) != sorted(overrides):
            continue
        tree = cohort_tree(econ.get("git_commit"))
        matches.append(Match(run=d.name, path=d, tree=tree,
                             same_tree=bool(tree and current_tree and tree == current_tree)))
    return matches


def preflight(jobs: list[tuple[str, list[str]]], runs_dir: Path,
              current_tree: str | None) -> tuple[list[str], int]:
    """Report lines and the number of jobs the record already contains."""
    lines: list[str] = []
    flagged = 0
    for name, train_args in jobs:
        config, overrides, exemption = predicted_config(train_args)
        if config is None:
            lines.append(f"  {name}: skipped ({exemption})")
            continue
        matches = find_duplicates(config, overrides, name, runs_dir, current_tree)
        if not matches:
            continue
        flagged += 1
        for m in matches:
            where = "runs/archive" if m.path.parent.name == "archive" else "runs"
            tree = (m.tree or "unknown")[:12]
            if m.same_tree:
                lines.append(
                    f"  {name}: DUPLICATE of {where}/{m.run} — same config, same prices, "
                    f"same cohort/ tree ({tree}). Training is bit-deterministic in these: "
                    f"the checkpoints this job would produce are already on disk."
                )
            else:
                lines.append(
                    f"  {name}: config already trained as {where}/{m.run} "
                    f"(cohort/ tree {tree}, current {(current_tree or 'unknown')[:12]}). "
                    f"Unless the tree transition moved this scenario's trajectories, the "
                    f"policy is already on disk — squad_v29_seed14 re-derived squad_v10c "
                    f"across exactly such a transition (assurance #63)."
                )
    return lines, flagged


def main() -> None:
    """CLI entry point: exit 1 when the record already contains a job's config."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("jobs_file", help="campaign jobs file, train_queue.sh format")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    jobs = parse_jobs(Path(args.jobs_file).read_text())
    lines, flagged = preflight(jobs, args.runs_dir, cohort_tree("HEAD"))
    for line in lines:
        print(line)
    if flagged:
        print(f"preflight: {flagged} of {len(jobs)} job(s) would re-derive a run the "
              f"record already holds (FORCE=1 queues them anyway)")
        raise SystemExit(1)
    print(f"preflight: {len(jobs)} job(s), no config already in the record")


if __name__ == "__main__":
    main()
