import argparse
import signal
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from tqdm import tqdm

from mili_env.envs.classes.qlearning_agent import AgentConfig, QLearningAgent
from mili_env.envs.classes.robot_base import Actions
from mili_env.envs.visualization import GradientLossVisualization
from mili_env.wrappers.reacher_weighted_reward import ReacherRewardWrapper


def save_periodic_model(agent, episode: int, base_model_path: Path) -> None:  # noqa: ANN001
    """Save the model to a temporary file and replace the main model file."""
    temp_model_path = base_model_path.with_name(f"{base_model_path.stem}_temp{base_model_path.suffix}")

    # Ensure directory exists
    temp_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to temporary file
    agent.save_model(temp_model_path)

    # Replace the main model file
    temp_model_path.replace(base_model_path)

    # Annotate the file with the episode number
    with Path.open(base_model_path.with_suffix(".txt"), "a") as f:
        f.write(f"Model saved at episode {episode}\n")


# Hyperparameters
n_episodes = 1_000
num_envs = 4  # Number of parallel environments
start_epsilon = 0.25
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
    env = gym.make("mili_env/TerrainWorld-v0", render_mode="rgb_array", visualization=VISUALIZATION)
    return ReacherRewardWrapper(env, reward_dist_weight=1.0, reward_ctrl_weight=0.1)


envs = SyncVectorEnv([make_env for _ in range(num_envs)])

favoured_actions = [Actions.FORWARD.value, Actions.ROTATE_LEFT.value, Actions.ROTATE_RIGHT.value]

# Initialize visualizations
visualization = GradientLossVisualization(512, 196) if VISUALIZATION else None

agent = QLearningAgent(envs, config, favoured_actions, visualization)

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent.model.to(device)

# Load the model if it exists and is valid
if model_path.exists():
    try:
        agent.load_model(str(model_path))
        agent.model.to(device)  # Ensure the model is moved to the correct device after loading
    except (OSError, ValueError, RuntimeError):
        pass


# Handle interruptions gracefully
def save_model_and_exit(_signum, _frame):  # noqa: ANN001, ANN201, D103
    save_periodic_model(agent, -1, model_path)
    sys.exit(0)


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

    # Train long memory
    agent.train_long_memory()

    # Decay epsilon
    agent.decay_epsilon()

    # Display episode statistics
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    print(f"Episode {episode + 1}/{n_episodes} - Average Reward: {avg_reward:.2f}, Average Length: {avg_length:.2f}")  # noqa: T201

    # Save periodically
    if (episode + 1) % 10 == 0:
        save_periodic_model(agent, episode, model_path)

# Save the final model
save_periodic_model(agent, n_episodes, model_path)
