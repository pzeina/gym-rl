from __future__ import annotations

import logging
import time
from multiprocessing import freeze_support
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import torch
from gymnasium.spaces import Dict as gym_Dict
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common import logger as sb3_logger
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import mili_env  # ensure the package is imported so Gymnasium sees registered envs  # noqa: F401
import utils
from hyperparameter_tuning import OptimizationConfig, get_default_hyperparams, tune_hyperparameters

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def make_env(env_id: str = "mili_env/TerrainWorld-v0", seed: int = 0) -> Callable[[], gym.Env]:
    """Create a single instance of the environment with proper wrappers.

    Following SB3 RL tips for environment setup:
    - Monitor wrapper for episode statistics
    - Potential for additional normalization wrappers
    """
    def _init() -> gym.Env:
        env = gym.make(env_id, render_mode=None)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def create_algorithm( # noqa: PLR0913
    algorithm: str,
    env: VecNormalize,
    policy: str,
    tensorboard_log: str,
    hyperparams: dict[str, Any] | None = None,
    device: str = "auto",
) -> PPO | DQN | A2C | SAC | TD3 | DDPG:
    """Create RL algorithm instance based on the specified algorithm type."""
    if hyperparams is None:
        hyperparams = get_default_hyperparams(algorithm)

    # Common arguments for all algorithms
    common_args = {
        "policy": policy,
        "env": env,
        "tensorboard_log": tensorboard_log,
        "device": device,
        "verbose": 1,
    }

    algorithm_classes = {
        "ppo": PPO,
        "dqn": DQN,
        "a2c": A2C,
        "sac": SAC,
        "td3": TD3,
        "ddpg": DDPG,
    }

    if algorithm not in algorithm_classes:
        msg = f"Unsupported algorithm: {algorithm}"
        raise ValueError(msg)

    algorithm_class = algorithm_classes[algorithm]

    # Merge common args with hyperparams
    final_args = {**common_args, **hyperparams}

    return algorithm_class(**final_args)


def main() -> None: # noqa: PLR0915
    """Train a RL agent following SB3 RL tips and best practices.

    Implements key recommendations from:
    https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

    - Proper environment normalization with VecNormalize
    - Tuned hyperparameters for different algorithms
    - Parallel environments for better sample efficiency
    - Proper evaluation setup
    - Optional automatic hyperparameter tuning with Optuna
    """
    args = utils.parse_args()

    timestamp = int(time.time())
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{args.algorithm}_mili_{timestamp}.zip"

    # Configure SB3 logger
    sb3_logger.configure(str(model_dir / "logs"))
    logger.info("Run 'tensorboard --logdir models/tb --bind_all' to see training progress on localhost:6006")

    # Environment setup following SB3 best practices
    n_envs = max(4, args.parallel)  # Use at least 4 environments for better sample collection
    logger.info("Creating %d parallel environments", n_envs)

    # Create vectorized environment with proper seeding
    env = make_vec_env(
        "mili_env/TerrainWorld-v0",
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
        wrapper_class=Monitor,
    )

    # Add VecNormalize for observation normalization (key SB3 recommendation)
    # This normalizes observations to have zero mean and unit variance
    env = VecNormalize(
        env,
        norm_obs=True,        # Normalize observations
        norm_reward=True,     # Normalize rewards
        clip_obs=10.0,       # Clip normalized observations
        clip_reward=10.0,    # Clip normalized rewards
    )

    # Create evaluation environment (without training normalization)
    eval_env = make_vec_env(
        "mili_env/TerrainWorld-v0",
        n_envs=1,
        seed=123,  # Different seed for evaluation
        vec_env_cls=DummyVecEnv,
        wrapper_class=Monitor,
    )
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    # Policy selection based on observation space
    policy = "MultiInputPolicy" if isinstance(env.observation_space, gym_Dict) else "MlpPolicy"

    # TensorBoard logging
    tb_log = str(model_dir / "tb")
    logger.info("TensorBoard logging enabled; writing logs to %s", tb_log)

    # Device selection and availability check
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Auto-selected device: %s", device)
    else:
        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"

    logger.info("Using device: %s", device)
    if device.startswith("cuda"):
        logger.info("CUDA device name: %s", torch.cuda.get_device_name())
        logger.info("CUDA memory available: %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Hyperparameter optimization or default training
    if args.tune_hyperparams:
        logger.info("Starting hyperparameter optimization for %s", args.algorithm)

        study_name = args.study_name or f"{args.algorithm}_{timestamp}"

        # Create optimization configuration
        config = OptimizationConfig(
            algorithm=args.algorithm,
            env=env,
            eval_env=eval_env,
            policy=policy,
            tb_log=tb_log,
            algorithm_factory_func=create_algorithm,
            device=device,
        )

        best_hyperparams = tune_hyperparameters(
            config=config,
            n_trials=args.n_trials,
            study_name=study_name,
        )

        logger.info("Using optimized hyperparameters for final training")
        hyperparams = best_hyperparams
    else:
        logger.info("Using default hyperparameters for %s", args.algorithm)
        hyperparams = get_default_hyperparams(args.algorithm)

    # Create the RL algorithm
    model = create_algorithm(
        algorithm=args.algorithm,
        env=env,
        policy=policy,
        tensorboard_log=tb_log,
        hyperparams=hyperparams,
        device=device,
    )

    logger.info("Created %s model with policy: %s", args.algorithm.upper(), policy)

    # Callbacks following SB3 best practices
    checkpoint_cb = CheckpointCallback(
        save_freq=max(10000, n_envs * 1000),  # Save every 10k steps or 1k episodes
        save_path=str(model_dir),
        name_prefix=f"{args.algorithm}_checkpoint"
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(model_dir / "eval_logs"),
        eval_freq=max(5000, n_envs * 500),  # Evaluate every 5k steps or 500 episodes per env
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Training duration - scale with number of environments
    total_timesteps = 1_000_000  # 1M timesteps is often good for many tasks

    logger.info("Starting training for %d timesteps with %d environments", total_timesteps, n_envs)
    logger.info("%s hyperparameters: %s", args.algorithm.upper(),
                {k: v for k, v in hyperparams.items() if k != "policy_kwargs"})

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb],
            progress_bar=True,
        )
    finally:
        # Save normalization statistics
        env.save(str(model_dir / "vec_normalize.pkl"))
        eval_env.close()
        env.close()

    # Save final model
    model.save(str(model_path))
    logger.info("Training complete. Model saved to %s", model_path)
    logger.info("Normalization stats saved to %s", model_dir / "vec_normalize.pkl")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    freeze_support()
    main()
