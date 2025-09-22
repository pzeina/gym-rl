"""Hyperparameter tuning module using Optuna for RLlib multi-agent algorithms.

This module provides automated hyperparameter optimization functionality
for RLlib algorithms in multi-agent cooperative environments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from ray import tune

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    algorithm: str
    env_config: dict[str, Any]
    device: str


def get_default_hyperparams(algorithm: str) -> dict[str, Any]:
    """Get default hyperparameters for each RLlib algorithm."""
    defaults = {
        "ppo": {
            "lr": 3e-4,
            "train_batch_size": 4000,
            "minibatch_size": 128,
            "num_epochs": 10,
            "clip_param": 0.2,
            "gamma": 0.95,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.0,
            "grad_clip": 0.5,
            "use_gae": True,
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
            },
        },
        "dqn": {
            "lr": 1e-4,
            "train_batch_size": 32,
            "target_network_update_freq": 10000,
            "epsilon": [[0, 1.0], [200000, 0.02]],  # Epsilon schedule format
            # Use simple replay buffer to avoid prioritized buffer bug
            "replay_buffer_config": {
                "type": "EpisodeReplayBuffer",
                "capacity": 50000,
            },
        },
        "impala": {
            "lr": 3e-4,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "grad_clip": 40.0,
            "use_gae": True,
            "lambda_": 0.9,
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
            },
        },
        "sac": {
            "lr": 3e-4,
            "buffer_size": 1000000,
            "learning_starts": 10000,
            "train_batch_size": 256,
            "target_network_update_freq": 1,
            "tau": 0.005,
            "target_entropy": "auto",
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
        },
    }

    return defaults.get(algorithm.lower(), {})


def suggest_ppo_hyperparams(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest PPO hyperparameters for optimization."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "train_batch_size": trial.suggest_categorical("train_batch_size", [2000, 4000, 8000]),
        "minibatch_size": trial.suggest_categorical("minibatch_size", [64, 128, 256]),
        "num_epochs": trial.suggest_int("num_epochs", 5, 20),
        "clip_param": trial.suggest_float("clip_param", 0.1, 0.4),
        "lambda_": trial.suggest_float("lambda_", 0.9, 0.99),
        "vf_loss_coeff": trial.suggest_float("vf_loss_coeff", 0.1, 1.0),
        "entropy_coeff": trial.suggest_float("entropy_coeff", 0.0, 0.1),
        "grad_clip": trial.suggest_float("grad_clip", 0.1, 1.0),
        "use_gae": True,
        "model": {
            "fcnet_hiddens": trial.suggest_categorical("fcnet_hiddens",
                [[128, 128], [256, 256], [512, 512], [256, 256, 256]]),
            "fcnet_activation": trial.suggest_categorical("fcnet_activation", ["tanh", "relu"]),
        },
    }


def suggest_dqn_hyperparams(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest DQN hyperparameters for optimization."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [100000, 500000, 1000000]),
        "learning_starts": trial.suggest_int("learning_starts", 10000, 100000),
        "train_batch_size": trial.suggest_categorical("train_batch_size", [32, 64, 128]),
        "target_network_update_freq": trial.suggest_int("target_network_update_freq", 1000, 20000),
        "epsilon_timesteps": trial.suggest_int("epsilon_timesteps", 100000, 500000),
        "final_epsilon": trial.suggest_float("final_epsilon", 0.01, 0.1),
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": trial.suggest_float("exploration_final_epsilon", 0.01, 0.1),
            "epsilon_timesteps": trial.suggest_int("exploration_epsilon_timesteps", 100000, 500000),
        },
        "model": {
            "fcnet_hiddens": trial.suggest_categorical("fcnet_hiddens",
                [[128, 128], [256, 256], [512, 512]]),
            "fcnet_activation": trial.suggest_categorical("fcnet_activation", ["relu", "tanh"]),
        },
    }


def suggest_impala_hyperparams(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest IMPALA hyperparameters for optimization."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "vf_loss_coeff": trial.suggest_float("vf_loss_coeff", 0.1, 1.0),
        "entropy_coeff": trial.suggest_float("entropy_coeff", 0.0, 0.1),
        "grad_clip": trial.suggest_float("grad_clip", 10.0, 80.0),
        "use_gae": True,
        "lambda_": trial.suggest_float("lambda_", 0.8, 0.99),
        "model": {
            "fcnet_hiddens": trial.suggest_categorical("fcnet_hiddens",
                [[128, 128], [256, 256], [512, 512]]),
            "fcnet_activation": trial.suggest_categorical("fcnet_activation", ["tanh", "relu"]),
        },
    }


def suggest_sac_hyperparams(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest SAC hyperparameters for optimization."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [100000, 500000, 1000000]),
        "learning_starts": trial.suggest_int("learning_starts", 1000, 20000),
        "train_batch_size": trial.suggest_categorical("train_batch_size", [128, 256, 512]),
        "target_network_update_freq": 1,
        "tau": trial.suggest_float("tau", 0.001, 0.01),
        "target_entropy": "auto",
        "model": {
            "fcnet_hiddens": trial.suggest_categorical("fcnet_hiddens",
                [[128, 128], [256, 256], [512, 512]]),
            "fcnet_activation": trial.suggest_categorical("fcnet_activation", ["relu", "tanh"]),
        },
    }


def create_objective_function(config: OptimizationConfig) -> Callable[..., float]:
    """Create objective function for Optuna optimization."""

    def objective(trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization."""
        # Suggest hyperparameters based on algorithm
        if config.algorithm.lower() == "ppo":
            hyperparams = suggest_ppo_hyperparams(trial)
        elif config.algorithm.lower() == "dqn":
            hyperparams = suggest_dqn_hyperparams(trial)
        elif config.algorithm.lower() == "impala":
            hyperparams = suggest_impala_hyperparams(trial)
        elif config.algorithm.lower() == "sac":
            hyperparams = suggest_sac_hyperparams(trial)
        else:
            msg = f"Unsupported algorithm: {config.algorithm}"
            raise ValueError(msg)

        # Import here to avoid circular imports
        from train_rllib import create_algorithm_config

        try:
            # Create algorithm configuration with suggested hyperparameters
            algo_config = create_algorithm_config(
                algorithm=config.algorithm,
                env_config=config.env_config,
                hyperparams=hyperparams,
                device=config.device,
            )

            # Reduce training time for optimization (use fewer timesteps)
            stop_criteria = {
                "timesteps_total": 50000,  # Reduced for faster optimization
                "episodes_total": 500,
            }

            # Run training trial
            results = tune.run(
                config.algorithm.upper(),
                config=algo_config.to_dict(),
                stop=stop_criteria,
                verbose=0,  # Reduced verbosity for optimization
                progress_reporter=None,  # Disable progress reporting
                local_dir=f"/tmp/optuna_trials/trial_{trial.number}",
                name=f"trial_{trial.number}",
            )

            # Get the best result
            best_trial = results.get_best_trial("episode_reward_mean", "max")
            best_reward = best_trial.last_result["episode_reward_mean"]

            # Report intermediate values for pruning
            if hasattr(trial, "report"):
                trial.report(best_reward, step=best_trial.last_result["timesteps_total"])

            # Check if trial should be pruned
            if hasattr(trial, "should_prune") and trial.should_prune():
                raise optuna.TrialPruned

        except RuntimeError:
            logger.warning("Trial failed with error")
            # Return a very low reward for failed trials
            return -float("inf")
        else:
            return best_reward

    return objective


def tune_hyperparameters_rllib(
    config: OptimizationConfig,
    n_trials: int = 50,
    study_name: str | None = None,
) -> dict[str, Any]:
    """Optimize hyperparameters using Optuna for RLlib algorithms."""
    logger.info("Starting hyperparameter optimization for %s", config.algorithm.upper())
    logger.info("Number of trials: %s", n_trials)

    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name or f"{config.algorithm}_optimization",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    # Create objective function
    objective = create_objective_function(config)

    try:
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=None)

        # Log results
        logger.info("Optimization completed")
        logger.info("Best trial: %s", study.best_trial.number)
        logger.info("Best value: %s", study.best_value)
        logger.info("Best params: %s", study.best_params)

        # Convert best params back to hyperparameters format
        best_hyperparams = {}
        trial = study.best_trial

        if config.algorithm.lower() == "ppo":
            best_hyperparams = suggest_ppo_hyperparams(trial)
        elif config.algorithm.lower() == "dqn":
            best_hyperparams = suggest_dqn_hyperparams(trial)
        elif config.algorithm.lower() == "impala":
            best_hyperparams = suggest_impala_hyperparams(trial)
        elif config.algorithm.lower() == "sac":
            best_hyperparams = suggest_sac_hyperparams(trial)

    except RuntimeError:
        logger.exception("Hyperparameter optimization failed")
        logger.info("Falling back to default hyperparameters")
        return get_default_hyperparams(config.algorithm)
    else:
        return best_hyperparams
