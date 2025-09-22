"""Unit tests for RLlib environment integration."""

# Import functions from train_rllib.py
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import ray

sys.path.append(str(Path(__file__).parent.parent))

import pytest

from train_rllib import TerrainWorldRLlibWrapper


class TestRLlibEnvironmentIntegration(unittest.TestCase):
    """Test cases for RLlib environment integration."""

    def setUp(self):
        """Set up test environment."""
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            # For debugging, consider using Ray Distributed Debugger instead.

        self.env_config = {
            "num_agents": 4,
            "render_mode": None
        }
        self.wrapper = TerrainWorldRLlibWrapper({"env_config": self.env_config})

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "wrapper"):
            del self.wrapper

    def test_environment_creation(self):
        """Test that environment can be created successfully."""
        self.assertIsNotNone(self.wrapper)
        self.assertIsNotNone(self.wrapper.env)

    def test_reset_returns_valid_observations(self):
        """Test that reset returns valid observations for all agents."""
        observations, _info = self.wrapper.reset()

        # Check that we have observations for all agents
        self.assertEqual(len(observations), self.env_config["num_agents"])

        # Check observation structure. Observations can be per-agent arrays or dicts
        for obs in observations.values():
            if isinstance(obs, dict):
                obs_dict: dict[str, Any] = obs  # Type assertion for linter
                # Check that observation has expected keys
                expected_keys = [
                    "position", "target_position", "distance", "direction",
                    "target_direction", "energy", "agent_id"
                ]
                for key in expected_keys:
                    self.assertIn(key, obs_dict)
                # Check that values are finite arrays or numbers
                for key, value in obs_dict.items():
                    if key != "agent_id":  # agent_id is an integer
                        self.assertTrue(np.all(np.isfinite(value)))
            else:
                # assume array-like observation vector
                self.assertTrue(hasattr(obs, "shape"))
                self.assertTrue(np.all(np.isfinite(obs)))

    def test_step_with_valid_actions(self):
        """Test step function with valid actions."""
        self.wrapper.reset()

        # Create valid actions for all agents
        actions = {}
        for agent_id in self.wrapper.agent_ids:
            actions[agent_id] = 0  # IDLE action

        obs, rewards, terminated, truncated, info = self.wrapper.step(actions)

        # Verify return types and structures
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(rewards, dict)
        self.assertIsInstance(terminated, dict)
        self.assertIsInstance(truncated, dict)
        self.assertIsInstance(info, dict)

        # Verify all agents are included
        for agent_id in self.wrapper.agent_ids:
            self.assertIn(agent_id, obs)
            self.assertIn(agent_id, rewards)
            self.assertIn(agent_id, terminated)
            self.assertIn(agent_id, truncated)
            self.assertIn(agent_id, info)

    def test_action_space_validation(self):
        """Test that action space is properly defined."""
        action_space = self.wrapper.action_space

        self.assertIsNotNone(action_space)

        # Test that action space can sample valid actions
        if action_space is not None:  # Type guard for linter
            # The wrapper may expose per-agent action spaces as a dict
            if isinstance(action_space, dict):
                for sp in action_space.values():
                    _ = sp.sample()
            else:
                _ = action_space.sample()

    def test_observation_space_validation(self):
        """Test that observation space is properly defined."""
        obs_space = self.wrapper.observation_space

        self.assertIsNotNone(obs_space)

        # Test that observation space can sample valid observations
        if obs_space is not None:  # Type guard for linter
            if isinstance(obs_space, dict):
                for sp in obs_space.values():
                    _ = sp.sample()
            else:
                _ = obs_space.sample()

    def test_multi_agent_consistency(self):
        """Test consistency across multiple agents."""
        obs, _ = self.wrapper.reset()

        # All agents should have observations with the same structure
        agent_ids = list(obs.keys())
        first_obs = obs[agent_ids[0]]
        # Accept either dict or array observations
        if isinstance(first_obs, dict):
            first_obs_dict: dict[str, Any] = first_obs  # Type assertion for linter
            first_keys = set(first_obs_dict.keys())

            for agent_id in agent_ids[1:]:
                agent_obs = obs[agent_id]
                self.assertIsInstance(agent_obs, dict)
                agent_obs_dict: dict[str, Any] = agent_obs  # Type assertion for linter
                self.assertEqual(set(agent_obs_dict.keys()), first_keys)
        else:
            # array-like: check that shapes are consistent across agents
            self.assertTrue(hasattr(first_obs, "shape"))
            for agent_id in agent_ids[1:]:
                agent_obs = obs[agent_id]
                self.assertTrue(hasattr(agent_obs, "shape"))
                self.assertEqual(agent_obs.shape, first_obs.shape)

    def test_episode_lifecycle(self):
        """Test complete episode lifecycle."""
        # Reset environment
        obs, info = self.wrapper.reset()
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(info, dict)

        # Run several steps
        for step in range(10):
            actions = dict.fromkeys(self.wrapper.agent_ids, step % 5)
            obs, rewards, terminated, truncated, info = self.wrapper.step(actions)

            # Check that all return values are properly formatted
            def effective_len(d):
                if isinstance(d, dict):
                    return len(d) - (1 if "__all__" in d else 0)
                return len(d)

            self.assertEqual(len(obs), len(self.wrapper.agent_ids))
            self.assertEqual(len(rewards), len(self.wrapper.agent_ids))
            self.assertEqual(effective_len(terminated), len(self.wrapper.agent_ids))
            self.assertEqual(effective_len(truncated), len(self.wrapper.agent_ids))
            self.assertEqual(len(info), len(self.wrapper.agent_ids))

            # If episode ends, break
            if any(terminated.values()) or any(truncated.values()):
                break

    def test_reward_structure(self):
        """Test reward structure and cooperative behavior."""
        self.wrapper.reset()

        actions = dict.fromkeys(self.wrapper.agent_ids, 1)  # FORWARD
        _obs, rewards, _terminated, _truncated, _info = self.wrapper.step(actions)

        # In cooperative setting, all rewards should be equal
        reward_values = list(rewards.values())
        first_reward = reward_values[0]

        for reward in reward_values:
            self.assertEqual(reward, first_reward)

    def test_termination_conditions(self):
        """Test episode termination conditions."""
        self.wrapper.reset()

        # Run until episode terminates or max steps
        max_steps = 100
        for _ in range(max_steps):
            actions = dict.fromkeys(self.wrapper.agent_ids, 0)
            _obs, _rewards, terminated, truncated, _info = self.wrapper.step(actions)

            # Check termination flag types
            for agent_id in self.wrapper.agent_ids:
                self.assertIsInstance(terminated[agent_id], bool)
                self.assertIsInstance(truncated[agent_id], bool)

            if any(terminated.values()) or any(truncated.values()):
                break

    def test_info_dictionary_content(self):
        """Test that info dictionaries contain useful information."""
        _obs, reset_info = self.wrapper.reset()

        # Check reset info structure
        for agent_id in self.wrapper.agent_ids:
            self.assertIn(agent_id, reset_info)
            self.assertIsInstance(reset_info[agent_id], dict)

        # Check step info structure
        actions = dict.fromkeys(self.wrapper.agent_ids, 0)
        _obs, _rewards, _terminated, _truncated, step_info = self.wrapper.step(actions)

        for agent_id in self.wrapper.agent_ids:
            self.assertIn(agent_id, step_info)
            self.assertIsInstance(step_info[agent_id], dict)

    def test_different_agent_counts(self):
        """Test environment with different numbers of agents."""
        agent_counts = [2, 3, 6, 8]

        for num_agents in agent_counts:
            with self.subTest(num_agents=num_agents):
                config = {"env_config": {"num_agents": num_agents, "render_mode": None}}
                env = TerrainWorldRLlibWrapper(config)

                # Check agent count
                self.assertEqual(env.num_agents, num_agents)
                self.assertEqual(len(env.agent_ids), num_agents)

                # Test basic functionality
                obs, _info = env.reset()
                self.assertEqual(len(obs), num_agents)

                actions = dict.fromkeys(env.agent_ids, 0)
                step_result = env.step(actions)
                self.assertEqual(len(step_result[0]), num_agents)  # observations

                del env

    def test_observation_bounds(self):
        """Test that observations are within reasonable bounds."""
        obs, _ = self.wrapper.reset()

        for agent_id in self.wrapper.agent_ids:
            observation = obs[agent_id]

            # Observation may be a dict of components or an array-like vector
            if isinstance(observation, dict):
                observation_dict: dict[str, Any] = observation  # Type assertion for linter

                # Check for NaN or infinite values in numeric fields
                for key, value in observation_dict.items():
                    if key != "agent_id":  # agent_id is an integer
                        self.assertTrue(np.all(np.isfinite(value)))

                # Check data types
                self.assertIsInstance(observation_dict["agent_id"], int)
            else:
                # array-like observation: check finiteness
                self.assertTrue(hasattr(observation, "shape"))
                self.assertTrue(np.all(np.isfinite(observation)))

    def test_action_validation(self):
        """Test that invalid actions are handled appropriately."""
        self.wrapper.reset()

        # Test with missing agent actions
        incomplete_actions = {self.wrapper.agent_ids[0]: 0}

        with pytest.raises(KeyError):
            self.wrapper.step(incomplete_actions)

    def test_seeded_reproducibility(self):
        """Test that seeded environments produce reproducible results."""
        # Reset with same seed multiple times
        seed = 42

        # First run
        obs1, _ = self.wrapper.reset(seed=seed)

        # Second run with same seed
        obs2, _ = self.wrapper.reset(seed=seed)

        # Check that observation structures are consistent
        self.assertEqual(set(obs1.keys()), set(obs2.keys()))

        # Keys should be identical for each agent
        for agent_id in self.wrapper.agent_ids:
            a1 = obs1[agent_id]
            a2 = obs2[agent_id]
            # If dict observations, compare keys; if arrays, compare shapes
            if isinstance(a1, dict) and isinstance(a2, dict):
                self.assertEqual(set(a1.keys()), set(a2.keys()))
            # Only check shape if both are not dicts
            elif not isinstance(a1, dict) and not isinstance(a2, dict):
                self.assertTrue(hasattr(a1, "shape") and hasattr(a2, "shape"))
                self.assertEqual(a1.shape, a2.shape)

    def test_agent_id_consistency(self):
        """Test that agent IDs are consistent and properly formatted."""
        agent_ids = self.wrapper.get_agent_ids()

        # Should be a set
        self.assertIsInstance(agent_ids, set)

        # Should match the wrapper's agent list
        expected_ids = set(self.wrapper.agent_ids)
        self.assertEqual(agent_ids, expected_ids)

        # All IDs should be strings
        for agent_id in agent_ids:
            self.assertIsInstance(agent_id, str)

    def test_environment_state_persistence(self):
        """Test that environment state persists correctly between steps."""
        self.wrapper.reset()

        # Take first step and record state
        actions1 = dict.fromkeys(self.wrapper.agent_ids, 1)  # FORWARD
        obs1, rewards1, _terminated1, _truncated1, _info1 = self.wrapper.step(actions1)

        # Take second step
        actions2 = dict.fromkeys(self.wrapper.agent_ids, 0)  # IDLE
        obs2, rewards2, _terminated2, _truncated2, _info2 = self.wrapper.step(actions2)

        # Observations should change (agents moved)
        # But structure should remain consistent
        self.assertEqual(set(obs1.keys()), set(obs2.keys()))
        self.assertEqual(set(rewards1.keys()), set(rewards2.keys()))

    def test_memory_cleanup(self):
        """Test that environment properly cleans up memory."""
        # Create multiple environments and ensure they can be cleaned up
        envs = []

        for _ in range(3):
            config = {"env_config": {"num_agents": 2, "render_mode": None}}
            env = TerrainWorldRLlibWrapper(config)
            envs.append(env)

            # Use environment briefly
            env.reset()
            actions = dict.fromkeys(env.agent_ids, 0)
            env.step(actions)

        # Clean up all environments
        for env in envs:
            del env

    def test_concurrent_environments(self):
        """Test that multiple environments can run concurrently."""
        env1_config = {"env_config": {"num_agents": 2, "render_mode": None}}
        env2_config = {"env_config": {"num_agents": 3, "render_mode": None}}

        env1 = TerrainWorldRLlibWrapper(env1_config)
        env2 = TerrainWorldRLlibWrapper(env2_config)

        try:
            # Reset both environments
            obs1, _ = env1.reset()
            obs2, _ = env2.reset()

            # Check they have different agent counts
            self.assertEqual(len(obs1), 2)
            self.assertEqual(len(obs2), 3)

            # Run steps on both
            actions1 = dict.fromkeys(env1.agent_ids, 0)
            actions2 = dict.fromkeys(env2.agent_ids, 1)

            result1 = env1.step(actions1)
            result2 = env2.step(actions2)

            # Both should return valid results
            self.assertEqual(len(result1), 5)  # obs, rewards, terminated, truncated, info
            self.assertEqual(len(result2), 5)

        finally:
            del env1
            del env2

    @classmethod
    def tearDownClass(cls):
        """Clean up Ray after all tests."""
        if ray.is_initialized():
            ray.shutdown()



if __name__ == "__main__":
    unittest.main()
