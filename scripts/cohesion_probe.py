#!/usr/bin/env python
"""Cohesion of the landed platoon-depth fleet — zero training, facts only.

The cohesion measurement (no_close_teammate_rate / unseen_by_any_teammate_rate,
owner's axis 2026-08-18, measured never enforced) landed after the nine
platoon-depth runs did, so their committed behavior.json files cannot carry it.
The traces can be replayed: this probe loads each run's checkpoints through the
existing evaluation machinery, replays N episodes with the TraceRecorder on the
current tree, and prints the two rates per run per checkpoint. It answers
"does the hierarchy keep the cohort together at standard difficulty" without a
single training step and WITHOUT touching any committed behavior.json — the
probe writes nothing to runs/.

Read-only over the record; deterministic (per-episode torch + numpy seeding,
the same scheme evaluate() uses); sampled policy (greedy=False), matching how
the fleet's behavior numbers are produced.

    scripts/cohesion_probe.py                       # the nine platoon-depth runs
    scripts/cohesion_probe.py platoon_v8 --episodes 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cohort.env.cohort_env import make_env
from cohort.env.rewards import RewardConfig
from cohort.metrics import TraceRecorder, aggregate_behavior, episode_behavior
from cohort.training.evaluate import run_episode
from cohort.training.train import load_policy
from scripts.baseline import run_dir

#: the platoon-depth ablation record (ROADMAP, owner-decided 2026-08-18):
#: three seeds of the shipped system and three of each ablation arm.
#: platoon_v10_seed12 is excluded on purpose — it is the lane-A neutrality
#: re-derivation, bit-identical to platoon_v8, so probing it would count one
#: policy twice.
PLATOON_DEPTH_RUNS = (
    "platoon_v8",
    "platoon_v9_seed13",
    "platoon_v11_seed14",
    "platoon_nomask_v1_seed12",
    "platoon_nomask_v2_seed13",
    "platoon_nomask_v3_seed14",
    "platoon_flat_v1_seed12",
    "platoon_flat_v2_seed13",
    "platoon_flat_v3_seed14",
)

CHECKPOINTS = ("ckpt_best.pt", "ckpt_latest.pt")


def probe_checkpoint(path: Path, episodes: int, first_seed: int) -> dict:
    """Replay ``episodes`` and pool the cohesion counts over them."""
    net, ckpt = load_policy(path)
    rewards = RewardConfig(**ckpt["reward_config"]) if ckpt.get("reward_config") else None
    env = make_env(ckpt["scenario"], reward_config=rewards)
    per_episode = []
    successes = 0
    for k in range(episodes):
        ep_seed = first_seed + k
        torch.manual_seed(ep_seed)
        rec = TraceRecorder()
        out = run_episode(
            env, net, seed=ep_seed, rng=np.random.default_rng(ep_seed),
            greedy=False, recorder=rec,
        )
        successes += out["outcome"] == "success"
        per_episode.append(episode_behavior(rec.trace))
    agg = aggregate_behavior(per_episode)
    return {
        "scenario": ckpt["scenario"],
        "no_close": agg["no_close_teammate_rate"],
        "unseen": agg["unseen_by_any_teammate_rate"],
        "stacked": agg["stacked_rate"],
        "sound": agg["spatially_sound_rate"],
        "nn_dist": agg["mean_nearest_teammate_dist"],
        "agent_steps": agg["cohesion_agent_steps"],
        "success": successes / episodes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*", default=None,
                    help="run names (default: the nine platoon-depth runs)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=700,
                    help="first episode seed; episode k replays at seed+k")
    args = ap.parse_args()
    runs = args.runs or list(PLATOON_DEPTH_RUNS)

    print(
        f"cohesion probe — N={args.episodes} episodes/checkpoint, greedy=False, "
        f"seeds {args.seed}..{args.seed + args.episodes - 1}"
    )
    print("rates over living-agent-steps; close = within support umbrella (8.0), "
          "unseen = no teammate holds terrain LOS, stacked = 2+ teammates within 1.5 "
          "cells, sound = none of the three (unioned per step)")
    print()
    header = (f"{'run':<26} {'ckpt':<7} {'no_close':>8} {'unseen':>8} {'stacked':>8} "
              f"{'sound':>6} {'nn_dist':>7} {'agent-steps':>11} {'success':>7}")
    print(header)
    print("-" * len(header))
    for run in runs:
        d = run_dir(run)
        if not d.is_dir():
            print(f"{run:<26} MISSING run directory")
            continue
        for ckpt_name in CHECKPOINTS:
            path = d / ckpt_name
            if not path.is_file():
                print(f"{run:<26} {ckpt_name.removeprefix('ckpt_').removesuffix('.pt'):<7} "
                      f"missing checkpoint")
                continue
            r = probe_checkpoint(path, args.episodes, args.seed)
            label = ckpt_name.removeprefix("ckpt_").removesuffix(".pt")
            print(f"{run:<26} {label:<7} {r['no_close']:>8.3f} {r['unseen']:>8.3f} "
                  f"{r['stacked']:>8.3f} {r['sound']:>6.3f} {r['nn_dist']:>7.2f} "
                  f"{r['agent_steps']:>11,} {r['success']:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
