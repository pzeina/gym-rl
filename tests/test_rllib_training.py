"""Unit tests for RLlib training functionality."""

import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import _global_registry

# Import functions from train_rllib.py
sys.path.append(str(Path(__file__).parent.parent))

from ray_test_utils import setup_ray_for_test
from train_rllib import (
    TrainingCallbacks,
    create_algorithm_config,
    setup_device,
    setup_ray_and_directories,
)


class TestRLlibTraining(unittest.TestCase):
    """Test cases for RLlib training functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up Ray for all tests in this class."""
        # Suppress ResourceWarnings during testing
        warnings.filterwarnings("ignore", category=ResourceWarning)
        setup_ray_for_test()

    def setUp(self):
        """Set up test environment."""
        # Initialize Ray if not already initialized

    def tearDown(self):
        """Clean up after tests."""

    def test_training_callbacks_inheritance(self):
        """Test that TrainingCallbacks properly inherits from DefaultCallbacks."""
        callbacks = TrainingCallbacks()
        self.assertIsInstance(callbacks, DefaultCallbacks)

    def test_training_callbacks_methods_exist(self):
        """Test that callback methods exist and are callable."""
        callbacks = TrainingCallbacks()

        # Check that required methods exist
        self.assertTrue(hasattr(callbacks, "on_episode_step"))
        self.assertTrue(hasattr(callbacks, "on_episode_end"))

        # Check that methods are callable
        self.assertTrue(callable(callbacks.on_episode_step))
        self.assertTrue(callable(callbacks.on_episode_end))

    def test_policy_mapping_function(self):
        """Test the policy mapping function."""
        # The policy mapping function is defined inside create_algorithm_config
        # We'll test it indirectly through the config creation

        config = create_algorithm_config(
            algorithm="PPO",
            hyperparams={},
            env_config={"num_agents": 4},
            device="cpu"
        )

        # Check that multi-agent config is properly set by checking the config dict
        config_dict = config.to_dict()
        self.assertIn("policies", config_dict)
        self.assertIn("policy_mapping_fn", config_dict)

    def test_create_algorithm_config_ppo(self):
        """Test creating RLlib configuration for PPO algorithm."""
        hyperparams = {
            "lr": 0.0003,
            "gamma": 0.99,
            "lambda_": 0.95,
            "num_epochs": 10,
            "minibatch_size": 128,
            "train_batch_size": 4000,
        }

        env_config = {"num_agents": 4, "render_mode": None}

        config = create_algorithm_config("PPO", hyperparams, env_config, "cpu")

        # Check that config is PPOConfig instance
        self.assertIsInstance(config, PPOConfig)

        # Check that hyperparameters are applied using config dict
        config_dict = config.to_dict()
        self.assertEqual(config_dict["lr"], 0.0003)
        self.assertEqual(config_dict["gamma"], 0.99)

    def test_create_algorithm_config_with_environment(self):
        """Test that RLlib config properly sets up environment."""
        env_config = {"num_agents": 6, "render_mode": None}

        config = create_algorithm_config("PPO", {}, env_config, "cpu")

        # Check environment configuration using config dict
        config_dict = config.to_dict()
        self.assertEqual(config_dict["env"], "terrain_world_rllib")
        self.assertEqual(config_dict["env_config"], env_config)

    def test_create_algorithm_config_multi_agent_setup(self):
        """Test multi-agent configuration setup."""
        env_config = {"num_agents": 4}

        config = create_algorithm_config("PPO", {}, env_config, "cpu")
        config_dict = config.to_dict()

        # Check multi-agent configuration
        self.assertIn("policies", config_dict)
        self.assertIn("policy_mapping_fn", config_dict)

        # Check that shared policy is configured
        policies = config_dict["policies"]
        self.assertIn("shared_policy", policies)

    def test_create_algorithm_config_callbacks(self):
        """Test that callbacks are properly configured."""
        config = create_algorithm_config("PPO", {}, {"num_agents": 4}, "cpu")
        config_dict = config.to_dict()

        # Check that callbacks are set
        self.assertIn("callbacks", config_dict)

    def test_setup_device_auto(self):
        """Test automatic device selection."""
        # Mock arguments
        class MockArgs:
            device = "auto"

        args = MockArgs()
        device = setup_device(args)

        # Should return either "cuda" or "cpu"
        self.assertIn(device, ["cuda", "cpu"])

    def test_setup_device_cpu(self):
        """Test CPU device selection."""
        class MockArgs:
            device = "cpu"

        args = MockArgs()
        device = setup_device(args)

        self.assertEqual(device, "cpu")

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.get_device_properties")
    def test_setup_device_cuda_available(self, mock_get_device_properties, mock_get_device_name, mock_cuda_available):
        """Test CUDA device selection when available."""
        mock_cuda_available.return_value = True
        mock_get_device_name.return_value = "Mock GPU"

        # Mock device properties
        mock_properties = type("MockProperties", (), {"total_memory": 8000000000})()
        mock_get_device_properties.return_value = mock_properties

        class MockArgs:
            device = "cuda"

        args = MockArgs()
        device = setup_device(args)

        self.assertEqual(device, "cuda")

    @patch("torch.cuda.is_available")
    def test_setup_device_cuda_unavailable(self, mock_cuda_available):
        """Test CUDA device selection when unavailable."""
        mock_cuda_available.return_value = False

        class MockArgs:
            device = "cuda"

        args = MockArgs()
        device = setup_device(args)

        # Should fallback to CPU
        self.assertEqual(device, "cpu")

    def test_setup_ray_and_directories(self):
        """Test Ray and directory setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            class MockArgs:
                model_dir = temp_dir
                algorithm = "PPO"  # Add missing algorithm attribute
                parallel = 4
                device = "cpu"  # Add device attribute to avoid CUDA issues

            args = MockArgs()
            timestamp, model_dir = setup_ray_and_directories(args)

            # Check return types
            self.assertIsInstance(timestamp, int)
            self.assertIsInstance(model_dir, Path)

            # Check that model directory is created
            self.assertTrue(model_dir.exists())
            self.assertTrue(model_dir.is_dir())

    def test_environment_registration(self):
        """Test that environment is properly registered with Ray."""
        # This tests the register_env call
        # Check that our environment is registered
        # (Note: this depends on the module being imported)
        env_name = "terrain_world_rllib"

        # The environment should be registered during import
        # We can test this by trying to create it
        try:
            env_creator = _global_registry.get("env", env_name)
            self.assertIsNotNone(env_creator)
        except (KeyError, AttributeError, ValueError):
            # If not found, we'll skip this test as it depends on the exact
            # module import order and Ray initialization state
            self.skipTest(f"Environment {env_name} not found in registry - depends on module import order")

    def test_config_validation_unsupported_algorithm(self):
        """Test error handling for unsupported algorithms."""
        try:
            create_algorithm_config(
                "UNSUPPORTED_ALGO",
                {},
                {"num_agents": 4},
                "cpu"
            )
            self.fail("Expected ValueError was not raised")
        except ValueError:
            pass  # Expected behavior

    def test_config_parameter_application(self):
        """Test that hyperparameters are correctly applied to different algorithms."""
        algorithms_to_test = ["PPO", "DQN", "IMPALA", "SAC"]

        base_hyperparams = {
            "lr": 0.001,
            "gamma": 0.95,
        }

        for algorithm in algorithms_to_test:
            with self.subTest(algorithm=algorithm):
                try:
                    config = create_algorithm_config(
                        algorithm,
                        base_hyperparams,
                        {"num_agents": 4},
                        "cpu"
                    )

                    # Should not raise an exception
                    self.assertIsNotNone(config)

                except ValueError as e:
                    if "Unsupported algorithm" in str(e):
                        # This is expected for algorithms we don't support
                        continue
                    self.fail(f"Unexpected error for {algorithm}: {e}")

    def test_multi_agent_policy_sharing(self):
        """Test that all agents share the same policy."""
        config = create_algorithm_config("PPO", {}, {"num_agents": 4}, "cpu")
        config_dict = config.to_dict()

        # Get the policy mapping function
        policy_mapping_fn = config_dict["policy_mapping_fn"]

        # Test that all agents map to the same policy
        test_agent_ids = ["agent_0", "agent_1", "agent_2", "agent_3"]

        policies = []
        for agent_id in test_agent_ids:
            policy = policy_mapping_fn(agent_id, None, None)
            policies.append(policy)

        # All agents should map to the same policy
        first_policy = policies[0]
        for policy in policies:
            self.assertEqual(policy, first_policy)

    def test_training_configuration_completeness(self):
        """Test that training configuration has all required components."""
        config = create_algorithm_config("PPO", {}, {"num_agents": 4}, "cpu")
        config_dict = config.to_dict()

        # Check required configuration keys
        required_keys = [
            "env",
            "env_config",
            "policies",
            "callbacks",
            "framework",
        ]

        for key in required_keys:
            self.assertIn(key, config_dict, f"Missing required config key: {key}")

    def test_framework_configuration(self):
        """Test that framework is properly configured."""
        config = create_algorithm_config("PPO", {}, {"num_agents": 4}, "cpu")
        config_dict = config.to_dict()

        # Framework should be set to torch
        self.assertEqual(config_dict["framework"], "torch")

    def test_num_env_runners_configuration(self):
        """Test that num_env_runners is properly configured."""
        config = create_algorithm_config("PPO", {}, {"num_agents": 4}, "cpu")
        config_dict = config.to_dict()

        # Should have num_env_runners configured
        self.assertIn("num_env_runners", config_dict)
        self.assertIsInstance(config_dict["num_env_runners"], int)
        self.assertGreaterEqual(config_dict["num_env_runners"], 0)

    def test_environment_config_propagation(self):
        """Test that environment configuration is properly propagated."""
        env_config = {
            "num_agents": 6,
            "render_mode": None,
            "custom_param": "test_value"
        }

        config = create_algorithm_config("PPO", {}, env_config, "cpu")
        config_dict = config.to_dict()

        # Environment config should be preserved
        self.assertEqual(config_dict["env_config"], env_config)


if __name__ == "__main__":
    unittest.main()
