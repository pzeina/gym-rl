import argparse
import os
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from tqdm import tqdm

from mili_env.envs.classes.qlearning_agent import AgentConfig, QLearningAgent
from mili_env.envs.classes.robot_base import Actions
from mili_env.envs.visualization import GradientLossVisualization


def save_periodic_model(agent: QLearningAgent, episode: int, base_model_path: Path) -> None:  # noqa: ARG001
    """Save the model to a temporary file and replace the main model file."""
    temp_model_path = base_model_path.with_name(f"{base_model_path.stem}_temp{base_model_path.suffix}")

    # Ensure directory exists
    temp_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to temporary file
    agent.save_model(str(temp_model_path))

    # Replace the main model file
    temp_model_path.replace(base_model_path)


# Hyperparameters
n_episodes = 100_000
N_ENVS = os.cpu_count()  # print(f"Number of CPUs: {N_ENVS}")
if N_ENVS is None:
    N_ENVS = 4
num_envs = N_ENVS  # Number of parallel environments
start_epsilon = 0.25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AgentConfig(
    learning_rate=0.01,
    initial_epsilon=0.25,
    final_epsilon=0.05,
    discount_factor=0.95,
    memory_size=100_000,
    batch_size=64,
    batch_num=10,
    hidden_size=64,
    decay_factor=0.8,  # Add decay factor for exponential decay
    update_frequency=1,
    subsampling_fraction=0.2,
    optimization_steps=5,
    likelihood_ratio_clipping=0.2,
    estimate_terminal=False,
    exploration=0.0,
    variable_noise=0.0,
    l2_regularization=0.0,
    entropy_regularization=0.0,
    name="agent",
    device=device,
    parallel_interactions=1,
    seed=42,
    execution=None,
    saver=None,
    summarizer=None,
    recorder=None,
)

model_path = Path(__file__).resolve().parent / "model/qlearning_model.pth"

# parse arguments and get visualization flag
parser = argparse.ArgumentParser(description="Train a Q-learning agent.")
parser.add_argument("--visualization", action="store_true", help="Enable gradient loss visualization.")
args = parser.parse_args()

VISUALIZATION = args.visualization


# Create vectorized environments
def make_env() -> gym.Env:
    """Create a new environment instance."""
    return gym.make("mili_env/TerrainWorld-v0", render_mode="rgb_array", visualization=VISUALIZATION)


envs = SyncVectorEnv([make_env for _ in range(N_ENVS)])

favoured_actions = [Actions.FORWARD.value, Actions.ROTATE_LEFT.value, Actions.ROTATE_RIGHT.value]

# Initialize visualizations
visualization = GradientLossVisualization(512, 196) if VISUALIZATION else None

agent = QLearningAgent(envs, config, favoured_actions, visualization)
agent.policy_model.to(device)

# Load the model if it exists and is valid
if model_path.exists():
    with suppress(OSError, ValueError, RuntimeError):
        agent.load_model(str(model_path))


# Handle interruptions gracefully
def save_model_and_exit(_signum, _frame):  # noqa: ANN001, ANN201, D103
    save_periodic_model(agent, -1, model_path)
    sys.exit(0)


log_file_path = model_path.with_name(f"training_log_{int(time.time())}.csv")


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


signal.signal(signal.SIGINT, save_model_and_exit)
signal.signal(signal.SIGTERM, save_model_and_exit)

# Training loop
for episode in tqdm(range(n_episodes)):
    obs, info = envs.reset()
    episode_terminated = [False] * num_envs
    episode_rewards = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs)

    # Play one episode
    while not all(episode_terminated):

        def process_obs(obs: np.ndarray) -> np.ndarray:
            """Process the observations into a 2D numpy array where each row is the state of an environment."""
            if isinstance(envs, SyncVectorEnv):
                return np.concatenate([obs[key].reshape(num_envs, -1) for key in obs], axis=1)
            env_processed = [value.flatten() for key in obs for value in obs[key]]
            return np.array(env_processed)

        states = process_obs(obs)
        actions, random_picks = agent.get_action(states)
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        next_states = process_obs(next_obs)

        # Remember the experiences
        agent.remember(states, actions, rewards, next_states, terminated)

        # Train short memory
        agent.train_short_memory(states, actions, rewards, next_states, terminated)

        # Update observations
        obs = next_obs

        # Update episode statistics
        episode_rewards += rewards
        episode_lengths += 1

        # Update episode_terminated
        for i, env_terminated in enumerate(terminated):
            if env_terminated:
                episode_terminated[i] = True

    # Update model based on update frequency
    agent.update_model()

    # Decay epsilon
    agent.decay_epsilon()

    # Display episode statistics
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)

    # Write episode statistics
    write_log_entry(episode, avg_reward, avg_length)

    # Save periodically
    if (episode + 1) % 10 == 0:
        save_periodic_model(agent, episode, model_path)

# Save the final model
save_periodic_model(agent, n_episodes, model_path)
