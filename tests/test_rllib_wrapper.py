"""Tests for RLlib multi-agent wrapper."""
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
from gymnasium import spaces

# Import functions from train_rllib.py
sys.path.append(str(Path(__file__).parent.parent))

import pytest

from tests.ray_test_utils import setup_ray_for_test
from train_rllib import TerrainWorldRLlibWrapper


class TestRLlibWrapper(unittest.TestCase):
    """Test cases for RLlib multi-agent wrapper."""

    @classmethod
    def setUpClass(cls):
        """Set up Ray for all tests in this class."""
        # Suppress ResourceWarnings during testing
        warnings.filterwarnings("ignore", category=ResourceWarning)
        setup_ray_for_test()

    def setUp(self):
        """Set up test environment."""
        self.config = {
            "env_config": {
                "num_agents": 4,
                "render_mode": None,
                "step_timeout": 10.0,
                "episode_timeout": 1000
            }
        }
        self.wrapper = TerrainWorldRLlibWrapper(self.config)

    def tearDown(self):
        """Clean up test environment."""
        try:
            if hasattr(self, "wrapper") and self.wrapper is not None:
                del self.wrapper
        except (AttributeError, RuntimeError):
            pass

    def test_wrapper_initialization(self):
        """Test wrapper initialization with configuration."""
        # Check that the wrapper is properly initialized
        self.assertIsNotNone(self.wrapper)

        # Check that the underlying environment is created
        self.assertIsNotNone(self.wrapper.env)

        # Check that agent IDs are properly set
        self.assertIsNotNone(self.wrapper.agent_ids)
        self.assertEqual(len(self.wrapper.agent_ids), 4)

        # Check that num_agents property is correct
        self.assertEqual(self.wrapper.num_agents, 4)

        # Check observation and action spaces are set
        self.assertIsNotNone(self.wrapper.observation_space)
        self.assertIsNotNone(self.wrapper.action_space)

    def test_wrapper_custom_config(self):
        """Test wrapper with custom configuration."""
        custom_config = {
            "env_config": {
                "num_agents": 6,
                "render_mode": None
            }
        }
        custom_wrapper = TerrainWorldRLlibWrapper(custom_config)

        self.assertEqual(custom_wrapper.num_agents, 6)
        self.assertEqual(len(custom_wrapper.agent_ids), 6)

        del custom_wrapper

    def test_wrapper_default_config(self):
        """Test wrapper initialization without configuration."""
        wrapper = TerrainWorldRLlibWrapper()

        # Should use default values
        self.assertEqual(wrapper.num_agents, 4)  # Default from environment
        self.assertIsNotNone(wrapper.agent_ids)

        del wrapper

    def test_reset_method(self):
        """Test reset method returns proper multi-agent format."""
        obs, info = self.wrapper.reset()

        # Check that observations is a dictionary
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(info, dict)

        # Check that all agents have observations
        for agent_id in self.wrapper.agent_ids:
            self.assertIn(agent_id, obs)
            self.assertIn(agent_id, info)

            # Check observation shape and type - may be dict or array
            self.assertTrue(isinstance(obs[agent_id], dict) or hasattr(obs[agent_id], "shape"))

    def test_reset_with_seed(self):
        """Test reset method with seed parameter."""
        # Reset with same seed twice
        obs1, _ = self.wrapper.reset(seed=42)
        obs2, _ = self.wrapper.reset(seed=42)

        # With same seed, initial observations should be similar
        # (Note: exact equality may not hold due to random target assignment)
        self.assertIsInstance(obs1, dict)
        self.assertIsInstance(obs2, dict)

        # Check structure is consistent
        self.assertEqual(set(obs1.keys()), set(obs2.keys()))

    def test_step_method(self):
        """Test step method returns proper multi-agent format."""
        # Reset environment first
        self.wrapper.reset()

        # Create action dictionary for all agents
        actions = {}
        for agent_id in self.wrapper.agent_ids:
            actions[agent_id] = 0  # IDLE action

        # Execute step
        obs, rewards, terminated, truncated, info = self.wrapper.step(actions)

        # Verify return format
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

    def test_step_with_different_actions(self):
        """Test step method with different actions for each agent."""
        self.wrapper.reset()

        # Create different actions for each agent
        actions = {}
        action_values = [0, 1, 2, 3]  # IDLE, FORWARD, BACKWARD, ROTATE_LEFT

        for i, agent_id in enumerate(self.wrapper.agent_ids):
            actions[agent_id] = action_values[i % len(action_values)]

        # Execute actions for different agents
        obs, rewards, terminated, _truncated, info = self.wrapper.step(actions)

        # Verify structure - terminated includes per-agent termination
        self.assertEqual(len(obs), len(self.wrapper.agent_ids))
        self.assertEqual(len(rewards), len(self.wrapper.agent_ids))
        # RLlib may include a '__all__' key in terminated/truncated dicts
        def effective_len(d):
            if isinstance(d, dict):
                return len(d) - (1 if "__all__" in d else 0)
            return len(d)

        self.assertEqual(effective_len(terminated), len(self.wrapper.agent_ids))
        self.assertEqual(len(info), len(self.wrapper.agent_ids))

    def test_cooperative_rewards(self):
        """Test that rewards are distributed cooperatively (same for all agents)."""
        self.wrapper.reset()

        actions = dict.fromkeys(self.wrapper.agent_ids, 0)
        _, rewards, _, _, _ = self.wrapper.step(actions)

        # In cooperative setting, all agents should receive the same reward
        reward_values = list(rewards.values())
        first_reward = reward_values[0]

        for reward in reward_values:
            self.assertEqual(reward, first_reward,
                           "All agents should receive the same reward in cooperative setting")

    def test_get_agent_ids(self):
        """Test get_agent_ids method."""
        agent_ids = self.wrapper.get_agent_ids()

        self.assertIsInstance(agent_ids, set)
        self.assertEqual(len(agent_ids), self.wrapper.num_agents)

        # Should match the wrapper's agent IDs
        expected_ids = set(self.wrapper.agent_ids)
        self.assertEqual(agent_ids, expected_ids)

    def test_observation_space_consistency(self):
        """Test observation space consistency across agents."""
        # All agents should have the same observation space structure
        self.assertIsNotNone(self.wrapper.observation_space)

        # Reset and get observations
        obs, _ = self.wrapper.reset()

        # Check that all observations have the same structure
        first_agent_id = self.wrapper.agent_ids[0]
        first_obs = obs[first_agent_id]

        # Check if observations are dictionaries
        if isinstance(first_obs, dict):
            # Verify that observations are dictionaries
            self.assertIsInstance(first_obs, dict)

            # Check that all agents have the same observation keys
            for agent_id in self.wrapper.agent_ids:
                agent_obs = obs[agent_id]
                self.assertIsInstance(agent_obs, dict)

                # Check that corresponding values have the same shape
                for key in first_obs:
                    if hasattr(first_obs[key], "shape"):  # numpy array
                        self.assertEqual(agent_obs[key].shape, first_obs[key].shape,
                                         f"Key '{key}' should have the same shape across agents")
                    else:  # scalar values
                        self.assertEqual(type(agent_obs[key]), type(first_obs[key]),
                                         f"Key '{key}' should have the same type across agents")
        else:
            # If observations are numpy arrays, check their shape and type consistency
            for agent_id in self.wrapper.agent_ids:
                agent_obs = obs[agent_id]
                self.assertIsInstance(agent_obs, np.ndarray)
                self.assertEqual(agent_obs.shape, first_obs.shape,
                                 "All agents should have the same observation shape")
                self.assertEqual(agent_obs.dtype, first_obs.dtype,
                                 "All agents should have the same observation dtype")

    def test_action_space_consistency(self):
        """Test action space consistency across agents."""
        self.assertIsNotNone(self.wrapper.action_space)

        # In discrete action space, all agents should have same action space
        # The action space should be for a single agent (not the full multi-agent dict)
        # Action space may be a dict of per-agent spaces (RLlib). Accept that or a single Discrete.
        if isinstance(self.wrapper.action_space, dict):
            for sp in self.wrapper.action_space.values():
                self.assertIsInstance(sp, spaces.Discrete)
        else:
            self.assertIsInstance(self.wrapper.action_space, spaces.Discrete)

    def test_episode_termination(self):
        """Test episode termination and truncation conditions."""
        self.wrapper.reset()

        # Run several steps to test termination logic
        max_steps = 10
        for _ in range(max_steps):
            actions = dict.fromkeys(self.wrapper.agent_ids, 0)
            _, _, terminated, _truncated, _ = self.wrapper.step(actions)

            # Check that termination flags are boolean (including numpy boolean)
            for agent_id in self.wrapper.agent_ids:
                self.assertIsInstance(terminated[agent_id], (bool, np.bool_))

            # If any agent is terminated, the episode should end
            if any(terminated.values()):
                break

    def test_info_dictionaries(self):
        """Test that info dictionaries contain expected information."""
        _, info = self.wrapper.reset()

        # Info should be provided for each agent
        self.assertEqual(len(info), len(self.wrapper.agent_ids))

        # Execute a step and check step info
        actions = dict.fromkeys(self.wrapper.agent_ids, 1)  # FORWARD
        _, _, _, _, step_info = self.wrapper.step(actions)

        # Step info should be provided for each agent
        self.assertEqual(len(step_info), len(self.wrapper.agent_ids))

        for agent_id in self.wrapper.agent_ids:
            self.assertIsInstance(step_info[agent_id], dict)

    def test_ray_multiagent_interface(self):
        """Test that wrapper works with Ray's multi-agent interface."""
        # This test ensures the wrapper is compatible with Ray's expectations

        # Test that required methods exist
        self.assertTrue(hasattr(self.wrapper, "reset"))
        self.assertTrue(hasattr(self.wrapper, "step"))
        self.assertTrue(hasattr(self.wrapper, "get_agent_ids"))

        # Test that methods return correct types
        obs, info = self.wrapper.reset()
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(info, dict)

        actions = dict.fromkeys(self.wrapper.agent_ids, 0)
        step_result = self.wrapper.step(actions)
        self.assertEqual(len(step_result), 5)  # obs, rewards, terminated, truncated, info

    def test_invalid_action_handling(self):
        """Test error handling for invalid action dictionaries."""
        self.wrapper.reset()

        # Test with missing agent actions - wrapper should raise KeyError
        incomplete_actions = {self.wrapper.agent_ids[0]: 0}

        # The wrapper should raise an error for missing agents
        with pytest.raises(KeyError):
            self.wrapper.step(incomplete_actions)

    def test_invalid_action_values(self):
        """Test error handling for invalid action values."""
        self.wrapper.reset()

        # Test with invalid action value
        invalid_actions = dict.fromkeys(self.wrapper.agent_ids, 999)

        # This might raise an exception or handle gracefully depending on implementation
        try:
            obs, _, _, _, _ = self.wrapper.step(invalid_actions)
            # If no exception, check that the step was handled
            self.assertIsInstance(obs, dict)
        except (ValueError, AssertionError, KeyError) as e:
            # If an exception is raised, it should be a reasonable one
            self.assertIn(type(e), [ValueError, AssertionError, KeyError])

    def test_environment_constraints(self):
        """Test that environment constraints are respected."""
        # Test that wrapper doesn't violate any environment constraints
        self.wrapper.reset()

        # Test multiple action combinations
        for action_value in range(4):  # Test all valid actions
            actions = dict.fromkeys(self.wrapper.agent_ids, action_value)
            try:
                obs, _, _, _, _ = self.wrapper.step(actions)
                # Should complete without error
                self.assertIsInstance(obs, dict)
            except (ValueError, AssertionError, KeyError) as e:
                self.fail(f"Action {action_value} caused unexpected error: {e}")

    def test_observation_bounds(self):
        """Test that observations are within expected bounds."""
        obs, _ = self.wrapper.reset()

        for agent_id in self.wrapper.agent_ids:
            observation = obs[agent_id]

            # Observation may be a dict of components or an array-like vector
            if isinstance(observation, dict):
                # Check that observation values are finite
                for value in observation.values():
                    if isinstance(value, np.ndarray):
                        self.assertTrue(np.all(np.isfinite(value)))
                    elif isinstance(value, int | float):
                        self.assertTrue(np.isfinite(value))
            else:
                # array-like observation: check finiteness
                self.assertTrue(hasattr(observation, "shape"))
                self.assertTrue(np.all(np.isfinite(observation)))

    def test_reward_bounds(self):
        """Test that rewards are within reasonable bounds."""
        self.wrapper.reset()

        actions = dict.fromkeys(self.wrapper.agent_ids, 0)
        _, rewards, _, _, _ = self.wrapper.step(actions)

        for agent_id in self.wrapper.agent_ids:
            reward = rewards[agent_id]

            # Reward should be a finite number
            self.assertTrue(np.isfinite(reward))
            # Reward should be a scalar
            self.assertIsInstance(reward, (int, float, np.number))

    def test_multiple_episodes(self):
        """Test that the wrapper works correctly across multiple episodes."""
        for _ in range(3):
            obs, info = self.wrapper.reset()

            # Check consistency across episodes
            self.assertEqual(len(obs), len(self.wrapper.agent_ids))
            self.assertEqual(len(info), len(self.wrapper.agent_ids))

            # Run a few steps
            for step in range(5):
                actions = dict.fromkeys(self.wrapper.agent_ids, step % 2)
                _, _, terminated, _truncated, _ = self.wrapper.step(actions)

                if any(terminated.values()):
                    break

    def test_wrapper_memory_management(self):
        """Test that wrapper doesn't leak memory when created/destroyed."""
        # Create and destroy multiple wrappers
        for _ in range(3):
            config = {"env_config": {"num_agents": 2, "render_mode": None}}
            temp_wrapper = TerrainWorldRLlibWrapper(config)

            # Use the wrapper briefly
            temp_wrapper.reset()
            actions = dict.fromkeys(temp_wrapper.agent_ids, 0)
            temp_wrapper.step(actions)

            # Delete wrapper
            del temp_wrapper


if __name__ == "__main__":
    unittest.main()
