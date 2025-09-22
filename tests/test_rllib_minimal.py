from train_rllib import TerrainWorldRLlibWrapper

"""Minimal pytest tests for RLlib multi-agent training.

These replace the previous top-level scripts and live under `tests/` so pytest
discovers them in a single place.
"""

"""Minimal pytest tests for RLlib multi-agent training.

These replace the previous top-level scripts and live under `tests/` so pytest
discovers them in a single place.
"""


def test_minimal_rllib_setup_imports():
    """Sanity: can import the RLlib wrapper and build a minimal env instance."""
    env_config = {"num_agents": 2, "render_mode": None}

    env = TerrainWorldRLlibWrapper({"env_config": env_config})
    assert hasattr(env, "num_agents")
    assert env.num_agents == 2
    assert hasattr(env, "agent_ids")
    assert len(env.agent_ids) == 2


def test_minimal_rllib_reset_step():
    """Create environment, call reset and step using minimal actions dict."""

    env_config = {"num_agents": 2, "render_mode": None}
    env = TerrainWorldRLlibWrapper({"env_config": env_config})

    # gymnasium-style reset
    obs, _info = env.reset()
    assert isinstance(obs, dict)
    assert set(obs.keys()) == set(env.agent_ids)

    # step with zero actions (acceptable for minimal smoke test)
    actions = dict.fromkeys(env.agent_ids, 0)
    _obs2, rewards, _terminated, _truncated, _info2 = env.step(actions)
    assert isinstance(rewards, dict)
    assert set(rewards.keys()) == set(env.agent_ids)

