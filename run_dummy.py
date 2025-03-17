import argparse
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from tqdm import tqdm

from mili_env.envs.classes.dummy_agent import DummyAgent

# Hyperparameters
n_episodes = 10_000
num_envs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# parse arguments and get visualization flag
parser = argparse.ArgumentParser(description="Train a Q-learning agent.")
parser.add_argument("--visualization", type=str, default="false", help="Enable gradient loss visualization.")
args = parser.parse_args()

# Convert the visualization argument to a boolean
VISUALIZATION = args.visualization.lower() in ("true", "1", "t", "y", "yes")


# Create vectorized environments
def make_env() -> gym.Env:
    """Create a new environment instance."""
    return gym.make("mili_env/TerrainWorld-v0", render_mode="rgb_array", visualization=VISUALIZATION)


envs = SyncVectorEnv([make_env for _ in range(num_envs)])


agent = DummyAgent(envs)

exp_timestamp = int(time.time())
log_file_path = Path(__file__).resolve().parent / f"model/dummy_agent_{exp_timestamp}.csv"


def write_log_entry(episode: int, avg_reward: np.floating[Any], avg_length: np.floating[Any]) -> None:
    """Write the episode statistics to a log file."""
    log_entry = f"{episode + 1},{avg_reward:.2f},{avg_length:.2f}\n"

    # Ensure directory exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write header if file does not exist
    if not log_file_path.exists():
        with log_file_path.open("w") as log_file:
            log_file.write("Episode,Average Reward,Average Length\n")

    # Append log entry
    with log_file_path.open("a") as log_file:
        log_file.write(log_entry)


# When using vectorized environments (like SyncVectorEnv), Gymnasium automatically
#  resets only the individual environments that terminate, while others keep
#  running. You do not need to manually reset them after each episode.
obs, info = envs.reset()

episode_rewards_tmp = np.zeros(num_envs)
episode_lengths_tmp = np.zeros(num_envs, dtype=int)

# Training loop
for episode in tqdm(range(n_episodes)):
    # Initialize episode timing metrics
    episode_timing = {
        "train_time": np.zeros(num_envs),
        "env_step_time": np.zeros(num_envs),
        "reward_time": np.zeros(num_envs),
        "terminal_time": np.zeros(num_envs),
        "action_selection_time": np.zeros(num_envs),
        "others_time": np.zeros(num_envs),
    }
    episode_logs = {
        "episode": episode,
        "episode_length": 0,
        "episode_avg_reward": 0.0,
    }

    # Play one episode (until first termination)
    while True:
        start_time = time.time()

        def process_obs(obs: np.ndarray) -> np.ndarray:
            """Process the observations into a 2D numpy array where each row is the state of an environment."""
            if isinstance(envs, SyncVectorEnv):
                return np.concatenate([obs[key].reshape(num_envs, -1) for key in obs], axis=1)
            env_processed = [value.flatten() for key in obs for value in obs[key]]
            return np.array(env_processed)

        # Action selection
        action_selection_start = time.time()
        states = process_obs(obs)
        actions, random_picks = agent.get_action(states)
        episode_timing["action_selection_time"] += time.time() - action_selection_start

        # Environment step
        env_step_start = time.time()
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        episode_timing["env_step_time"] += time.time() - env_step_start

        # Reward calculation
        reward_start = time.time()
        next_states = process_obs(next_obs)
        episode_timing["reward_time"] += time.time() - reward_start

        # Update observations
        obs = next_obs

        # Update episode statistics
        episode_rewards_tmp += rewards

        # Calculate others time
        episode_timing["others_time"] += (
            time.time()
            - start_time
            - (
                episode_timing["action_selection_time"]
                + episode_timing["env_step_time"]
                + episode_timing["reward_time"]
                + episode_timing["terminal_time"]
                + episode_timing["train_time"]
            )
        )

        # Check if all environments are terminated
        if any(terminated):
            terminated_envs = np.where(terminated)[0]
            for terminated_env in terminated_envs:
                episode_logs["episode_length"] = episode_lengths_tmp[terminated_env]
                episode_logs["episode_avg_reward"] = (
                    episode_rewards_tmp[terminated_env] / episode_lengths_tmp[terminated_env]
                )

                # Write episode statistics for each terminated environment
                write_log_entry(episode, episode_logs["episode_avg_reward"], episode_logs["episode_length"].item())

                # Reset the terminated environment's statistics
                episode_rewards_tmp[terminated_env] = 0.0
                episode_lengths_tmp[terminated_env] = 0

            break

    # Write episode statistics
    write_log_entry(episode, episode_logs["episode_avg_reward"], episode_logs["episode_length"].item())
