"""Compatibility tests for RLlib multi-agent training.

Lightweight pytest tests adapted from the repository's top-level scripts.
"""

import logging

from ray.tune.registry import register_env

from hyperparameter_tuning_rllib import get_default_hyperparams
from train_rllib import TerrainWorldRLlibWrapper, create_algorithm_config

logger = logging.getLogger(__name__)


def test_environment_wrapper_smoke():
    env_config = {"num_agents": 4, "render_mode": None}
    env = TerrainWorldRLlibWrapper({"env_config": env_config})

    assert env.num_agents == 4
    if not isinstance(env.observation_space, dict):
        msg = "Expected observation_space to be a dict"
        raise TypeError(msg)
    # Adjusted to handle both Dict and MultiAgentDict observation spaces
    # Original line: assert set(env.agent_ids) == set(env.observation_space.spaces.keys()) --- IGNORE ---
    assert set(env.agent_ids) == set(env.observation_space.keys())

    obs, _info = env.reset()
    assert set(obs.keys()) == set(env.agent_ids)

    actions = dict.fromkeys(env.agent_ids, 0)
    _obs2, rewards, _terminated, _truncated, _info2 = env.step(actions)
    assert isinstance(rewards, dict)


def test_algorithm_config_creation():
    env_config = {"num_agents": 4, "render_mode": None}
    hyperparams = get_default_hyperparams("ppo")

    config = create_algorithm_config(
        algorithm="ppo",
        env_config=env_config,
        hyperparams=hyperparams,
        device="cpu",
    )

    # Basic sanity checks
    assert config is not None


def test_ray_register_env():
    # just ensure register_env accepts the wrapper callable/class
    register_env("terrain_world_ma", TerrainWorldRLlibWrapper)
