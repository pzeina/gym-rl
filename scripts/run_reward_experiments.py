"""Simple experiment runner to test different reward functions.

Usage:
    python scripts/run_reward_experiments.py

This script loads the environment, iterates over registered reward functions,
runs a small number of episodes for each, and writes a JSONL report to
`debug/reward_experiments.jsonl`.
"""  # noqa: INP001
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from mili_env.envs.terrain_world import TerrainWorldEnv

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def run_experiments(
        env: TerrainWorldEnv,
        reward_names: list[str],
        episodes: int = 3,
        steps_per_episode: int = 50,
        seed: int | None = None
    ) -> list[dict[str, float | int | str]]:
    """Run experiments for each reward function and log results."""
    out_path = Path.cwd() / "debug" / "reward_experiments.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for name in reward_names:
        env.set_reward_function(name)
        env.archive_reward_function_details(name)

        seed_val = (seed or int(time.time()))
        total_reward = 0.0

        for ep in range(episodes):
            obs, _info = env.reset(seed=seed_val + ep)
            ep_reward = 0.0
            for _ in range(steps_per_episode):
                actions = dict.fromkeys(obs.keys(), 0)
                obs, r, terminated, truncated, _info = env.step(actions)
                ep_reward += float(r)
                if terminated or truncated:
                    break
            total_reward += ep_reward

        avg_reward = total_reward / episodes if episodes else 0.0
        rec = {
            "timestamp": time.time(),
            "reward_name": name,
            "seed": seed_val,
            "episodes": episodes,
            "steps_per_episode": steps_per_episode,
            "avg_reward": avg_reward,
        }
        results.append(rec)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    return results


def main() -> None:
    """Main entry point to run experiments and log results."""
    env = TerrainWorldEnv(render_mode=None, num_agents=3)
    reward_list = env.get_registered_reward_functions()
    logger.info("Registered reward functions: %s", reward_list)
    results = run_experiments(env, reward_list)
    logger.info("Results written to debug/reward_experiments.jsonl")
    logger.info("Experiment results: %s", results)


if __name__ == "__main__":
    main()
