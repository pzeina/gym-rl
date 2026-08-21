#!/usr/bin/env python
"""Spatial consistency of ANY run, replayed against its OWN cohort/ tree.

`scripts/cohesion_probe.py` replays on the current tree, so an obs-layout
change orphans every checkpoint trained before it — the 2026-08-21 fleet
sweep could score 2 of 24 runs. This probe closes that gap the way
`BASELINE.json` doctrine says provenance works: each run's recorded
`git_commit` (economics.json) names the tree it trained against, a detached
git worktree materializes that tree under `.claude/worktrees/` (git-excluded,
reused across invocations), and a child process replays the checkpoint with
the OLD tree's own make_env / load_policy / run_episode.

The four spatial counts are computed live off the env — `env.reset`/`env.step`
are wrapped to snapshot each step the moment positions and the terrain grid
exist — because the old trees' recorders predate the cohesion booleans. The
predicates are the current `cohort/metrics.py::_cohesion` definitions, stated
in the header of every report:

    close   — a living teammate within the combat model's support umbrella
    seen    — a living teammate holds terrain line of sight
    stacked — 2+ living teammates within STACK_RADIUS (1.5) cells
    sound   — none of the three violated, unioned per agent-step

Read-only over the record (writes nothing to runs/); deterministic
(per-episode torch + numpy seeding); sampled policy (greedy=False), matching
how the fleet's behavior numbers are produced.

    scripts/spatial_probe_provenance.py                    # the baseline fleet
    scripts/spatial_probe_provenance.py platoon_v8 --episodes 5
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKTREE_CACHE = REPO / ".claude" / "worktrees"

#: The bunching radius — kept equal to cohort.metrics.STACK_RADIUS, restated
#: here because the child process must not import the CURRENT tree's cohort
#: while replaying against an old one (a mismatch is caught by test).
STACK_RADIUS = 1.5


# ---------------------------------------------------------------------- #
# child mode: runs inside the run's own tree, prints one JSON line
# ---------------------------------------------------------------------- #

def child(tree: str, ckpt_path: str, episodes: int, first_seed: int) -> None:
    sys.path.insert(0, tree)

    import numpy as np
    import torch

    from cohort.env.cohort_env import make_env
    from cohort.env.rewards import RewardConfig
    from cohort.training.evaluate import run_episode
    from cohort.training.train import load_policy

    net, ckpt = load_policy(ckpt_path)
    rewards = RewardConfig(**ckpt["reward_config"]) if ckpt.get("reward_config") else None
    env = make_env(ckpt["scenario"], reward_config=rewards)
    umbrella = float(getattr(getattr(env, "combat", None), "support_umbrella", 8.0))

    counts = {"agent_steps": 0, "no_close": 0, "unseen": 0, "stacked": 0, "sound": 0,
              "nn_sum": 0.0, "nn_steps": 0}

    def snapshot() -> None:
        living = [s for s in env.roster.soldiers if s.alive]
        for a in living:
            dists = [
                math.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1])
                for b in living if b is not a
            ]
            close = any(d <= umbrella for d in dists)
            seen = any(env.world.line_of_sight(a.pos, b.pos) for b in living if b is not a)
            crowded = sum(d <= STACK_RADIUS for d in dists) >= 2
            counts["agent_steps"] += 1
            counts["no_close"] += not close
            counts["unseen"] += not seen
            counts["stacked"] += crowded
            counts["sound"] += close and seen and not crowded
            if dists:
                counts["nn_sum"] += min(dists)
                counts["nn_steps"] += 1

    orig_reset, orig_step = env.reset, env.step

    def reset(*a, **kw):
        out = orig_reset(*a, **kw)
        snapshot()
        return out

    def step(*a, **kw):
        out = orig_step(*a, **kw)
        snapshot()
        return out

    env.reset, env.step = reset, step

    successes = 0
    for k in range(episodes):
        ep_seed = first_seed + k
        torch.manual_seed(ep_seed)
        out = run_episode(env, net, seed=ep_seed, rng=np.random.default_rng(ep_seed), greedy=False)
        successes += out["outcome"] == "success"

    n = counts["agent_steps"]
    print(json.dumps({
        "scenario": ckpt["scenario"],
        "no_close": counts["no_close"] / n,
        "unseen": counts["unseen"] / n,
        "stacked": counts["stacked"] / n,
        "sound": counts["sound"] / n,
        "nn_dist": counts["nn_sum"] / counts["nn_steps"] if counts["nn_steps"] else None,
        "agent_steps": n,
        "success": successes / episodes,
    }))


# ---------------------------------------------------------------------- #
# parent mode: resolve provenance, materialize worktrees, drive children
# ---------------------------------------------------------------------- #

def provenance_commit(d: Path) -> str | None:
    try:
        return json.loads((d / "economics.json").read_text()).get("git_commit")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def worktree_for(sha: str) -> Path:
    wt = WORKTREE_CACHE / f"provenance-{sha[:7]}"
    if not (wt / "cohort").is_dir():
        WORKTREE_CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt), sha],
            cwd=REPO, check=True, capture_output=True,
        )
    return wt


def main() -> int:
    sys.path.insert(0, str(REPO))
    from scripts.baseline import run_dir

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*",
                    help="run names (default: the BASELINE.json members)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=700,
                    help="first episode seed; episode k replays at seed+k")
    ap.add_argument("--ckpt", default="ckpt_best.pt",
                    help="checkpoint file to replay (default: ckpt_best.pt, what ships)")
    args = ap.parse_args()
    runs = args.runs or list(
        json.loads((REPO / "runs" / "BASELINE.json").read_text())["runs"].values()
    )

    print(
        f"spatial probe at provenance — N={args.episodes} episodes, {args.ckpt}, "
        f"greedy=False, seeds {args.seed}..{args.seed + args.episodes - 1}"
    )
    print("rates over living-agent-steps; close = within support umbrella, unseen = no "
          "teammate holds terrain LOS, stacked = 2+ teammates within 1.5 cells, "
          "sound = none of the three (unioned per step)")
    print()
    header = (f"{'run':<42} {'tree':<7} {'no_close':>8} {'unseen':>8} {'stacked':>8} "
              f"{'sound':>6} {'nn_dist':>7} {'agent-steps':>11} {'success':>7}")
    print(header)
    print("-" * len(header))
    for run in runs:
        d = run_dir(run)
        sha = provenance_commit(d) if d.is_dir() else None
        ckpt = d / args.ckpt
        if sha is None or not ckpt.is_file():
            print(f"{run:<42} no recorded provenance or missing {args.ckpt}")
            continue
        wt = worktree_for(sha)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--child", str(wt), str(ckpt), str(args.episodes), str(args.seed)],
            cwd=wt, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            reason = (proc.stderr.strip().splitlines() or ["unknown error"])[-1][:70]
            print(f"{run:<42} {sha[:7]:<7} replay failed: {reason}")
            continue
        r = json.loads(proc.stdout.strip().splitlines()[-1])
        print(f"{run:<42} {sha[:7]:<7} {r['no_close']:>8.3f} {r['unseen']:>8.3f} "
              f"{r['stacked']:>8.3f} {r['sound']:>6.3f} {r['nn_dist']:>7.2f} "
              f"{r['agent_steps']:>11,} {r['success']:>7.2f}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
        raise SystemExit(0)
    raise SystemExit(main())
