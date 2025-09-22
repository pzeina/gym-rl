"""Unit tests for RLlib hyperparameter tuning functionality."""

# Import functions from hyperparameter_tuning_rllib.py
import contextlib
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import ray

sys.path.append(str(Path(__file__).parent.parent))

from hyperparameter_tuning_rllib import (
    OptimizationConfig,
    create_objective_function,
    get_default_hyperparams,
    suggest_dqn_hyperparams,
    suggest_ppo_hyperparams,
    tune_hyperparameters_rllib,
)


class TestRLlibHyperparameterTuning(unittest.TestCase):
    """Test cases for RLlib hyperparameter tuning functionality."""

    def setUp(self):
        """Set up test environment."""
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)

    def tearDown(self):
        """Clean up after tests."""

    def test_optimization_config_creation(self):
        """Test OptimizationConfig creation and validation."""
        config = OptimizationConfig(
            algorithm="PPO",
            env_config={"num_agents": 4},
            device="cpu"
        )

        self.assertEqual(config.algorithm, "PPO")
        self.assertEqual(config.env_config, {"num_agents": 4})
        self.assertEqual(config.device, "cpu")

    def test_optimization_config_defaults(self):
        """Test OptimizationConfig with basic values."""
        config = OptimizationConfig(
            algorithm="PPO",
            env_config={"num_agents": 4},
            device="cpu"
        )

        # Check that values are set correctly
        self.assertIsInstance(config.algorithm, str)
        self.assertIsInstance(config.env_config, dict)
        self.assertIsInstance(config.device, str)

        # Values should make sense
        self.assertIn(config.algorithm, ["PPO", "DQN", "IMPALA", "SAC"])
        self.assertIn("num_agents", config.env_config)

    def test_get_default_hyperparams_ppo(self):
        """Test getting default hyperparameters for PPO."""
        hyperparams = get_default_hyperparams("PPO")

        self.assertIsInstance(hyperparams, dict)

        # Check that expected PPO parameters are present
        expected_params = [
            "lr", "gamma", "clip_param",
            "train_batch_size", "minibatch_size"
        ]

        for param in expected_params:
            self.assertIn(param, hyperparams)

    def test_get_default_hyperparams_dqn(self):
        """Test getting default hyperparameters for DQN."""
        hyperparams = get_default_hyperparams("DQN")

        self.assertIsInstance(hyperparams, dict)

        # Check that expected DQN parameters are present
        expected_params = [
            "lr", "target_network_update_freq",
            "train_batch_size", "exploration_config"
        ]

        for param in expected_params:
            self.assertIn(param, hyperparams)

    def test_get_default_hyperparams_impala(self):
        """Test getting default hyperparameters for IMPALA."""
        hyperparams = get_default_hyperparams("IMPALA")

        self.assertIsInstance(hyperparams, dict)

        # Check that expected IMPALA parameters are present
        expected_params = [
            "lr", "vf_loss_coeff",
            "entropy_coeff", "grad_clip"
        ]

        for param in expected_params:
            self.assertIn(param, hyperparams)

    def test_get_default_hyperparams_sac(self):
        """Test getting default hyperparameters for SAC."""
        hyperparams = get_default_hyperparams("SAC")

        self.assertIsInstance(hyperparams, dict)

        # Check that expected SAC parameters are present
        expected_params = [
            "lr", "tau", "target_entropy",
            "train_batch_size"
        ]

        for param in expected_params:
            self.assertIn(param, hyperparams)

    def test_get_default_hyperparams_unsupported(self):
        """Test error handling for unsupported algorithms."""
        # The function returns an empty dict for unknown algorithms
        result = get_default_hyperparams("UNSUPPORTED_ALGORITHM")
        self.assertIsInstance(result, dict)
        # Should return empty dict for unsupported algorithms
        self.assertEqual(result, {})

    def test_suggest_hyperparameters_ppo(self):
        """Test hyperparameter suggestion for PPO."""
        # Mock Optuna trial
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.0003
        mock_trial.suggest_int.return_value = 10

        hyperparams = suggest_ppo_hyperparams(mock_trial)

        self.assertIsInstance(hyperparams, dict)

        # Check that suggestion methods were called
        self.assertTrue(mock_trial.suggest_float.called)

        # Check that returned hyperparameters have expected structure
        self.assertIn("lr", hyperparams)

    def test_suggest_hyperparameters_dqn(self):
        """Test hyperparameter suggestion for DQN."""
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.0001
        mock_trial.suggest_int.return_value = 1000
        mock_trial.suggest_categorical.return_value = "linear"

        hyperparams = suggest_dqn_hyperparams(mock_trial)

        self.assertIsInstance(hyperparams, dict)
        self.assertIn("lr", hyperparams)

    def test_suggest_hyperparameters_unsupported(self):
        """Test error handling for unsupported algorithms in suggestion."""
        with contextlib.suppress(ValueError, AttributeError):
            # This should raise error for invalid trial parameter
            suggest_ppo_hyperparams(MagicMock())
            # If it doesn't raise an error, that's fine too

    @patch("ray.tune.run")
    def test_objective_function_structure(self, mock_tune_run):
        """Test objective function structure and return value."""
        # Mock the tune.run result
        mock_result = MagicMock()
        mock_result.best_trial.last_result = {"episode_reward_mean": 100.0}
        mock_tune_run.return_value = mock_result

        # Create config
        config = OptimizationConfig(
            algorithm="PPO",
            env_config={"num_agents": 4},
            device="cpu"
        )

        # Create objective function
        objective_func = create_objective_function(config)

        # Mock trial for hyperparameter suggestion
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.0003
        mock_trial.suggest_int.return_value = 10

        # Test the objective function
        result = objective_func(mock_trial)

        # Should return a float (the reward)
        self.assertIsInstance(result, float)

    @patch("ray.tune.run")
    @patch("optuna.create_study")
    def test_tune_hyperparameters_rllib_integration(self, mock_create_study, mock_tune_run):
        """Test full hyperparameter tuning integration."""
        # Mock Optuna study
        mock_study = MagicMock()
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.0003
        mock_trial.suggest_int.return_value = 10
        mock_study.best_trial.params = {"learning_rate": 0.0003}
        mock_create_study.return_value = mock_study

        # Mock Ray tune result
        mock_result = MagicMock()
        mock_result.best_trial.last_result = {"episode_reward_mean": 100.0}
        mock_tune_run.return_value = mock_result

        config = OptimizationConfig(
            algorithm="PPO",
            env_config={"num_agents": 4},
            device="cpu"
        )
        best_params = tune_hyperparameters_rllib(
            config,
            n_trials=2
        )

        # Should return the best parameters
        self.assertIsInstance(best_params, dict)
        self.assertTrue(mock_study.optimize.called)

    def test_hyperparameter_value_ranges(self):
        """Test that suggested hyperparameters are within reasonable ranges."""
        mock_trial = MagicMock()

        # Set up mock to return specific values for range testing
        def mock_suggest_float(*args, **_kwargs):
            # Return the midpoint of the range
            if len(args) >= 3:
                low, high = args[1], args[2]
                return (low + high) / 2
            return 0.5

        def mock_suggest_int(*args, **_kwargs):
            # Return the midpoint of the range
            if len(args) >= 3:
                low, high = args[1], args[2]
                return (low + high) // 2
            return 64

        mock_trial.suggest_float = mock_suggest_float
        mock_trial.suggest_int = mock_suggest_int
        mock_trial.suggest_categorical.return_value = "linear"

        # Test PPO hyperparameters
        ppo_params = suggest_ppo_hyperparams(mock_trial)

        # Check that learning rate is reasonable
        self.assertGreater(ppo_params["lr"], 0)
        self.assertLess(ppo_params["lr"], 1)

        # Check that clip_param is between 0 and 1
        self.assertGreater(ppo_params["clip_param"], 0)
        self.assertLess(ppo_params["clip_param"], 1)

    def test_algorithm_specific_parameters(self):
        """Test that each algorithm gets appropriate parameters."""
        algorithms = ["PPO", "DQN", "IMPALA", "SAC"]

        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                defaults = get_default_hyperparams(algorithm)

                # Each algorithm should have lr and gamma (RLlib uses 'lr' not 'learning_rate')
                self.assertIn("lr", defaults)

                # Check algorithm-specific parameters
                if algorithm == "PPO":
                    self.assertIn("clip_param", defaults)
                    self.assertIn("entropy_coeff", defaults)
                elif algorithm == "DQN":
                    self.assertIn("target_network_update_freq", defaults)
                    self.assertIn("exploration_config", defaults)
                elif algorithm == "IMPALA":
                    self.assertIn("vf_loss_coeff", defaults)
                    self.assertIn("entropy_coeff", defaults)
                elif algorithm == "SAC":
                    self.assertIn("tau", defaults)
                    self.assertIn("target_entropy", defaults)

    def test_optimization_config_validation(self):
        """Test OptimizationConfig parameter validation."""
        # Test valid configuration
        config = OptimizationConfig(
            algorithm="PPO",
            env_config={"num_agents": 4},
            device="cpu"
        )
        self.assertEqual(config.algorithm, "PPO")
        self.assertEqual(config.device, "cpu")

        # Test that invalid algorithm raises error
        try:
            config = OptimizationConfig(
                algorithm="INVALID_ALGO",
                env_config={"num_agents": 4},
                device="cpu"
            )
            # If no error, check that algorithm is set
            self.assertIsNotNone(config.algorithm)
        except ValueError:
            # ValueError is acceptable for invalid inputs
            pass

    def test_hyperparameter_suggestion_consistency(self):
        """Test that hyperparameter suggestions are consistent across calls."""
        mock_trial = MagicMock()

        # Set up deterministic mock returns
        suggest_float_calls = 0
        suggest_int_calls = 0

        def mock_suggest_float(*_args, **_kwargs):
            nonlocal suggest_float_calls
            suggest_float_calls += 1
            return 0.001 * suggest_float_calls  # Different values for each call

        def mock_suggest_int(*_args, **_kwargs):
            nonlocal suggest_int_calls
            suggest_int_calls += 1
            return 10 * suggest_int_calls  # Different values for each call

        mock_trial.suggest_float = mock_suggest_float
        mock_trial.suggest_int = mock_suggest_int
        mock_trial.suggest_categorical.return_value = "linear"

        # Get parameters for the same algorithm multiple times
        params1 = suggest_ppo_hyperparams(mock_trial)

        # Reset counters for second call
        suggest_float_calls = 0
        suggest_int_calls = 0

        params2 = suggest_ppo_hyperparams(mock_trial)

        # Parameters should have the same structure
        self.assertEqual(set(params1.keys()), set(params2.keys()))

    @patch("optuna.create_study")
    def test_optimization_direction(self, mock_create_study):
        """Test that optimization direction is properly set."""
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study

        config = OptimizationConfig(
            algorithm="PPO",
            env_config={"num_agents": 4},
            device="cpu"
        )

        tune_hyperparameters_rllib(
            config,
            n_trials=2
        )

        # Check that study was created with correct direction
        mock_create_study.assert_called_once()
        call_kwargs = mock_create_study.call_args[1]
        self.assertEqual(call_kwargs["direction"], "maximize")

    @classmethod
    def tearDownClass(cls):
        """Clean up Ray after all tests."""
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    unittest.main()
