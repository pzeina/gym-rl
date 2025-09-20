"""Hyperparameter tuning module using Optuna for RL algorithms.

This module provides automated hyperparameter optimization functionality
following the recommendations from the RL Zoo and stable-baselines3 guide.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.callbacks import EvalCallback

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecNormalize

logger = logging.getLogger(__name__)


def get_default_hyperparams(algorithm: str) -> dict[str, Any]:
    """Get default hyperparameters for each algorithm following RL Zoo recommendations."""
    defaults = {
        "ppo": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "clip_range": 0.2,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "ent_coef": 0.0,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "policy_kwargs": {
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                "activation_fn": torch.nn.Tanh,
            },
        },
        "dqn": {
            "learning_rate": 1e-4,
            "buffer_size": 1000000,
            "learning_starts": 50000,
            "batch_size": 32,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 10000,
            "exploration_fraction": 0.1,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "policy_kwargs": {"net_arch": [256, 256]},
        },
        "a2c": {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.0,
            "vf_coef": 0.25,
            "max_grad_norm": 0.5,
            "use_rms_prop": True,
            "policy_kwargs": {
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                "activation_fn": torch.nn.Tanh,
            },
        },
        "sac": {
            "learning_rate": 3e-4,
            "buffer_size": 1000000,
            "learning_starts": 100,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "policy_kwargs": {"net_arch": [256, 256]},
        },
        "td3": {
            "learning_rate": 3e-4,
            "buffer_size": 1000000,
            "learning_starts": 25000,
            "batch_size": 100,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": (1, "episode"),
            "gradient_steps": -1,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "policy_kwargs": {"net_arch": [400, 300]},
        },
        "ddpg": {
            "learning_rate": 1e-3,
            "buffer_size": 1000000,
            "learning_starts": 100,
            "batch_size": 100,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": (1, "episode"),
            "gradient_steps": -1,
            "policy_kwargs": {"net_arch": [400, 300]},
        },
    }
    return defaults.get(algorithm, {})


def get_hyperparameter_space(algorithm: str, trial: optuna.Trial) -> dict[str, Any]:
    """Define hyperparameter search spaces for Optuna optimization."""
    match algorithm:
        case "ppo":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "n_epochs": trial.suggest_int("n_epochs", 3, 20),
                "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
                "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
                "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 2.0),
                "policy_kwargs": {
                    "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                    "activation_fn": torch.nn.Tanh,
                },
            }
        case "dqn":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 500000, 1000000]),
                "learning_starts": trial.suggest_int("learning_starts", 1000, 50000),
                "target_update_interval": trial.suggest_int("target_update_interval", 1000, 20000),
                "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
                "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.3),
                "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
                "policy_kwargs": {"net_arch": [256, 256]},
            }
        case "a2c":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "n_steps": trial.suggest_categorical("n_steps", [5, 8, 16, 32]),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
                "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
                "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 2.0),
                "policy_kwargs": {
                    "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                    "activation_fn": torch.nn.Tanh,
                },
            }
        case "sac":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
                "buffer_size": trial.suggest_categorical("buffer_size", [100000, 500000, 1000000]),
                "learning_starts": trial.suggest_int("learning_starts", 100, 10000),
                "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
                "tau": trial.suggest_float("tau", 0.001, 0.02),
                "ent_coef": "auto",  # Keep auto for SAC
                "policy_kwargs": {"net_arch": [256, 256]},
            }
        case "td3" | "ddpg":
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [64, 100, 128, 256]),
                "buffer_size": trial.suggest_categorical("buffer_size", [100000, 500000, 1000000]),
                "learning_starts": trial.suggest_int("learning_starts", 1000, 25000),
                "tau": trial.suggest_float("tau", 0.001, 0.02),
                "train_freq": (1, "episode"),
                "gradient_steps": -1,
                "policy_kwargs": {"net_arch": [400, 300]},
            }
        case _:
            msg = f"Hyperparameter space not defined for algorithm: {algorithm}"
            raise ValueError(msg)


class OptimizationConfig:
    """Configuration for hyperparameter optimization."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        algorithm: str,
        env: VecNormalize,
        eval_env: VecNormalize,
        policy: str,
        tb_log: str,
        algorithm_factory_func: Callable[..., Any],
    ) -> None:
        """Initialize optimization configuration.

        Args:
            algorithm: RL algorithm name (e.g., 'ppo', 'dqn')
            env: Training environment
            eval_env: Evaluation environment
            policy: Policy type ('MlpPolicy' or 'MultiInputPolicy')
            tb_log: TensorBoard log directory
            algorithm_factory_func: Function to create algorithm instances
        """
        self.algorithm = algorithm
        self.env = env
        self.eval_env = eval_env
        self.policy = policy
        self.tb_log = tb_log
        self.algorithm_factory_func = algorithm_factory_func


def _objective(trial: optuna.Trial, config: OptimizationConfig) -> float:
    """Objective function for Optuna hyperparameter optimization."""
    try:
        hyperparams = get_hyperparameter_space(config.algorithm, trial)

        # Create model with suggested hyperparameters
        model = config.algorithm_factory_func(
            algorithm=config.algorithm,
            env=config.env,
            policy=config.policy,
            tensorboard_log=config.tb_log,
            hyperparams=hyperparams,
        )

        # Evaluation callback to get intermediate values for pruning
        eval_callback = EvalCallback(
            config.eval_env,
            n_eval_episodes=5,  # Fewer episodes for faster evaluation during tuning
            eval_freq=max(5000, 4 * 500),  # Simplified calculation
            deterministic=True,
            render=False,
            verbose=0,
        )

        # Train for shorter duration during hyperparameter search
        total_timesteps = 50000  # Reduced for faster trials

        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=False,
        )

    except Exception:
        logger.exception("Trial failed with error")
        return float("-inf")
    else:
        # Return mean reward from evaluation
        return eval_callback.last_mean_reward


def tune_hyperparameters(
    config: OptimizationConfig,
    n_trials: int = 100,
    study_name: str | None = None,
) -> dict[str, Any]:
    """Perform hyperparameter tuning using Optuna.

    Args:
        config: Configuration object containing all necessary parameters
        n_trials: Number of optimization trials to run
        study_name: Name for the Optuna study (optional)

    Returns:
        Best hyperparameters found by optimization
    """
    if study_name is None:
        study_name = f"{config.algorithm}_{int(time.time())}"

    # Create study with TPE sampler and median pruner (as recommended in the guide)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name=study_name,
    )

    logger.info("Starting hyperparameter optimization for %s with %d trials", config.algorithm, n_trials)

    # Optimize
    study.optimize(
        lambda trial: _objective(trial, config),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    logger.info("Hyperparameter optimization completed!")
    logger.info("Best trial: %d", study.best_trial.number)
    logger.info("Best value: %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    # Return best hyperparameters merged with defaults
    best_hyperparams = get_default_hyperparams(config.algorithm)
    best_hyperparams.update(study.best_params)

    return best_hyperparams
