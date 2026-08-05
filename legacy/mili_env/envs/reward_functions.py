"""Reward function utilities for TerrainWorldEnv.

This module provides example per-agent and team reward functions and a helper
that registers a small set of default reward functions on an environment
instance. Each per-agent function receives `prev_info: dict` and must return a
mapping of agent_id -> numeric reward. Team functions receive `(prev_info,
per_agent_rewards)` and return a scalar float.

These are intended to be small, well-documented building blocks you can copy
or extend for experiments.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mili_env.envs.terrain_world import TerrainWorldEnv


def per_agent_progress_reward(prev_info: dict) -> dict[str, float]:
    """Per-agent reward proportional to improvement in distance to target.

    Expects `prev_info[agent_key]["distance"]` to be a numeric or 1-D array
    containing the previous distance. The returned mapping uses the same agent
    keys as the environment (e.g. `"agent_0"`).
    """
    rewards: dict = {}
    # Current info will be fetched inside environment when needed; tests and the
    # default `_get_reward` implementation use the env's `_get_info()` to derive
    # current distances. Here we assume `prev_info` contains previous distances
    # and return a simple proxy reward (negative distance change is penalty).
    for k, v in prev_info.items():
        try:
            prev_distance = float(np.atleast_1d(v.get("distance", 0.0))[0])
        except (TypeError, ValueError):
            prev_distance = 0.0
        # Simple heuristic: reward = -prev_distance (encourage being closer)
        rewards[k] = float(-prev_distance)
    return rewards


def per_agent_energy_preservation(prev_info: dict) -> dict[str, float]:
    """Per-agent reward that encourages conserving energy.

    Returns a small positive reward proportional to remaining energy (if
    available in `prev_info`). This is intentionally simple and interpretable.
    """
    rewards: dict = {}
    for k, v in prev_info.items():
        try:
            energy = float(np.atleast_1d(v.get("energy", 0.0))[0])
        except (TypeError, ValueError):
            energy = 0.0
        rewards[k] = float(energy / 100.0)
    return rewards


def team_mean_of_per_agent(prev_info: dict, per_agent_rewards: dict) -> float:
    """Team reward defined as the mean of per-agent rewards."""
    try:
        vals = [float(x) for x in per_agent_rewards.values()]
        return float(sum(vals) / len(vals)) if vals else 0.0
    except (TypeError, ValueError):
        return 0.0


def team_cooperation_bonus(prev_info: dict, per_agent_rewards: dict) -> float:
    """Simple team reward that boosts performance if all agents have positive reward."""
    try:
        vals = [float(x) for x in per_agent_rewards.values()]
        if all(v > 0 for v in vals) and vals:
            return float(sum(vals) / len(vals) + 1.0)
        return float(sum(vals) / len(vals)) if vals else 0.0
    except (TypeError, ValueError):
        return 0.0


def register_default_rewards(env: TerrainWorldEnv) -> None:
    """Register a set of default reward functions on `env`.

    This helper registers a few sensible choices used by the test-suite and
    experiment runner. It is idempotent and safe to call multiple times.
    """
    env.register_reward_function(
        "per_agent_progress", per_agent_fn=per_agent_progress_reward, team_fn=team_mean_of_per_agent
    )
    env.register_reward_function(
        "energy_preservation",
        per_agent_fn=per_agent_energy_preservation,
        team_fn=team_mean_of_per_agent
    )
    env.register_reward_function(
        "cooperation", per_agent_fn=per_agent_progress_reward, team_fn=team_cooperation_bonus
    )


__all__ = [
    "per_agent_energy_preservation",
    "per_agent_progress_reward",
    "register_default_rewards",
    "team_cooperation_bonus",
    "team_mean_of_per_agent",
]
