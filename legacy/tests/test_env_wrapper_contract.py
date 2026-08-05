import sys
import unittest
from pathlib import Path

# Ensure project root is importable
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from mili_env.envs.terrain_world import TerrainWorldEnv
from train import TerrainWorldRLlibWrapper


class TestEnvWrapperContract(unittest.TestCase):
    """Verify low-level env returns scalar reward/global dones and wrapper expands to per-agent dicts."""

    def test_env_returns_scalar_and_global_flags(self):
        env = TerrainWorldEnv(num_agents=3, render_mode=None)
        _obs, _info = env.reset()
        # Single-step with properly formatted nested actions
        actions = {
            f"agent_{i}": {
                "elementary_act": 0,
                "combat": {"move": 0, "fire_enemy": env.num_agents},
                "comm": {"types": [0, 0, 0], "targets": [env.num_agents, env.num_agents, env.num_agents]},
                "c2": {"give_order": 0, "orders": [env.MISSION_NO_CHANGE_SENTINEL] * env.num_agents}
            } for i in range(env.num_agents)
        }
        observation, reward, terminated, truncated, step_info = env.step(actions)

        # Low-level env should return a scalar reward and boolean dones
        self.assertIsInstance(reward, (int, float, np.floating))
        self.assertIsInstance(terminated, (bool, np.bool_))
        self.assertIsInstance(truncated, (bool, np.bool_))

        # Observations and info remain per-agent dicts
        self.assertIsInstance(observation, dict)
        self.assertIsInstance(step_info, dict)
        self.assertEqual(len(observation), env.num_agents)
        self.assertEqual(len(step_info), env.num_agents)

    def test_wrapper_expands_scalar_to_per_agent(self):
        wrapper = TerrainWorldRLlibWrapper({"env_config": {"num_agents": 3, "render_mode": None}})
        # Reset wrapper (which calls the low-level env.reset internally)
        _obs, _info = wrapper.reset()

        actions = {f"agent_{i}": 0 for i in range(wrapper.num_agents)}
        obs_step, rewards, terminated, truncated, step_info = wrapper.step(actions)

        # Wrapper should return per-agent dicts for rewards, terminated, and truncated
        self.assertIsInstance(rewards, dict)
        self.assertEqual(len(rewards), wrapper.num_agents)

        self.assertIsInstance(terminated, dict)
        self.assertIsInstance(truncated, dict)

        # per-agent terminated/truncated should be present
        for agent_id in wrapper.agent_ids:
            self.assertIn(agent_id, terminated)
            self.assertIn(agent_id, truncated)

        # All observations and infos map to agents
        self.assertEqual(len(obs_step), wrapper.num_agents)
        self.assertEqual(len(step_info), wrapper.num_agents)


if __name__ == "__main__":
    unittest.main()
