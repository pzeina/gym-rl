from __future__ import annotations

import logging
import time
from multiprocessing import freeze_support
from pathlib import Path
from typing import Callable

import gymnasium as gym
import torch
from gymnasium.spaces import Dict as gym_Dict
from stable_baselines3 import PPO
from stable_baselines3.common import logger as sb3_logger
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import mili_env  # ensure the package is imported so Gymnasium sees registered envs  # noqa: F401
import utils

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


def main() -> None:
    """Train a PPO agent following SB3 RL tips and best practices.

    Implements key recommendations from:
    https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

    - Proper environment normalization with VecNormalize
    - Tuned hyperparameters for PPO
    - Parallel environments for better sample efficiency
    - Proper evaluation setup
    """
    args = utils.parse_args()

    timestamp = int(time.time())
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"ppo_mili_{timestamp}.zip"

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

    # PPO with tuned hyperparameters following SB3 recommendations
    # Custom policy kwargs for better network architecture
    policy_kwargs = {
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},  # Separate networks for policy and value
        "activation_fn": torch.nn.Tanh,  # Tanh activation often works better than ReLU for continuous control
    }

    model = PPO(
        policy=policy,
        env=env,
        # Learning rate with linear schedule (common best practice)
        learning_rate=3e-4,
        # Number of steps to run for each environment per update
        n_steps=2048,  # Good default for most tasks
        # Minibatch size for SGD
        batch_size=64,  # Should be factor of n_steps * n_envs
        # Number of epochs when optimizing the surrogate loss
        n_epochs=10,
        # Clipping parameter for PPO
        clip_range=0.2,
        # GAE lambda parameter
        gae_lambda=0.95,
        # Value function coefficient for the loss calculation
        vf_coef=0.5,
        # Entropy coefficient for the loss calculation
        ent_coef=0.0,
        # Maximum value for the gradient clipping
        max_grad_norm=0.5,
        # Whether to use generalized advantage estimation (GAE)
        use_sde=False,
        # Custom network architecture
        policy_kwargs=policy_kwargs,
        # Log to tensorboard
        tensorboard_log=tb_log,
        verbose=1,
    )

    # Callbacks following SB3 best practices
    checkpoint_cb = CheckpointCallback(
        save_freq=max(10000, n_envs * 1000),  # Save every 10k steps or 1k episodes
        save_path=str(model_dir),
        name_prefix="ppo_checkpoint"
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
    logger.info("PPO hyperparameters: lr=3e-4, n_steps=%d, batch_size=%d", 2048, 64)

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
