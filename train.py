import os
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from tqdm import tqdm

import utils
from mili_env.envs.classes.qlearning_agent import AgentConfig, QLearningAgent
from mili_env.envs.metrics import GradientLossTracker

# Empêche la mise en veille sur macOS et Linux
# Safe list of commands for shell execution
safe_commands = ["caffeinate", "xset"]


# Function to validate command
def is_safe_command(command: str) -> bool:
    """Check if the command is safe to execute."""
    return command in safe_commands


if os.name == "posix":
    import subprocess

    subprocess.Popen("caffeinate" if "darwin" in sys.platform else "xset s off -dpms", shell=True)  # noqa: S602

# Hyperparameters
n_episodes = 10_000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AgentConfig(
    learning_rate=0.1,
    final_lr=0.01,
    decay_lr=0.9997,
    initial_epsilon=0.75,
    final_epsilon=0.02,
    discount_factor=0.95,
    memory_size=100_000,
    batch_size=1024,
    batch_num=10,
    hidden_size=264,
    decay_epsilon=0.9997,  # Add decay factor for exponential decay of epsi
    update_frequency=5,
    subsampling_fraction=0.2,
    optimization_steps=5,
    likelihood_ratio_clipping=0.2,
    estimate_terminal=False,  # not used
    exploration=0.0,  # not used
    variable_noise=0.0,  # not used
    l2_regularization=0.0,  # not used
    entropy_regularization=0.0,  # not used
    dummy_phase=1000,
    dummy_policy_decay=0.95,
    name="agent",
    device=device,
    parallel_interactions=1,
    seed=42,
    execution=None,
    saver=None,
    summarizer=None,
    recorder=None,
)


args = utils.parse_args()

TRACK_GRAD = args.track_grad.lower() in ("true", "1", "t", "y", "yes")
PLOT_GRAD = args.plot_grad.lower() in ("true", "1", "t", "y", "yes")
N_ENVS = args.parallel
RENDER_MODE = args.render_mode.lower()
num_envs = N_ENVS  # Number of parallel environments
MODEL_PATH = args.pretrained_model if args.pretrained_model else None
if RENDER_MODE not in ("rgb_array", "human"):
    error_msg = "The render mode must be either 'rgb_array' or 'human'."
    raise ValueError(error_msg)


# Create vectorized environments
def make_env() -> gym.Env:
    """Create a new environment instance."""
    return gym.make("mili_env/TerrainWorld-v0", render_mode=RENDER_MODE, visualization=PLOT_GRAD)


envs = SyncVectorEnv([make_env for _ in range(N_ENVS)])


agent = QLearningAgent(
    envs,
    config,
    visualization=GradientLossTracker(512, 196, graphs=PLOT_GRAD) if TRACK_GRAD else None,
)
if MODEL_PATH:
    with suppress(OSError, ValueError, RuntimeError):
        agent.load_model(str(MODEL_PATH))
agent.policy_model.to(device)


# Handle interruptions gracefully
def save_model_and_exit(_signum, _frame):  # noqa: ANN001, ANN201, D103
    utils.save_periodic_model(agent, -1, model_path)
    sys.exit(0)


exp_timestamp = int(time.time())
model_path = Path(__file__).resolve().parent / f"model/qlearning_model_{exp_timestamp}.pth"
log_file_path = model_path.with_name(f"training_log_{exp_timestamp}.csv")
print(f"Model path: {model_path}")  # noqa: T201
print(f"Log file path: {log_file_path}")  # noqa: T201


signal.signal(signal.SIGINT, save_model_and_exit)
signal.signal(signal.SIGTERM, save_model_and_exit)

timing_metrics_path = model_path.with_name(f"timing_metrics_{exp_timestamp}.csv")

# When using vectorized environments (like SyncVectorEnv), Gymnasium automatically
#  resets only the individual environments that terminate, while others keep
#  running. You do not need to manually reset them after each episode.
obs, info = envs.reset()

episode_rewards_tmp = np.zeros(num_envs)
episode_lengths_tmp = np.zeros(num_envs, dtype=int)

# Training loop
for episode in tqdm(range(n_episodes)):
    # Initialize episode metrics
    episode_logs = {
        "episode": episode,
        "episode_length": 0,
        "episode_total_reward": 0.0,
        "episode_avg_reward": 0.0,
        "learning_rate": 0.0,
        "epsilon": 0.0,
        "grad_value": 0.0,
        "loss_value": 0.0,
        "train_time": 0.0,
        "env_step_time": 0.0,
        "reward_time": 0.0,
        "action_selection_time": 0.0,
        "others_time": 0.0,
    }

    step_count: int = 0

    def process_obs(obs: np.ndarray) -> np.ndarray:
        """Process the observations into a 2D numpy array where each row is the state of an environment."""
        if isinstance(envs, SyncVectorEnv):
            return np.concatenate([obs[key].reshape(num_envs, -1) for key in obs], axis=1)
        env_processed = [value.flatten() for key in obs for value in obs[key]]
        return np.array(env_processed)

    # Play one episode (until first termination)
    while True:
        start_time = time.time()

        # Action selection
        action_selection_start = time.time()
        states = process_obs(obs)
        actions = agent.get_action(states)
        episode_logs["action_selection_time"] += time.time() - action_selection_start

        # Environment step
        env_step_start = time.time()
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        episode_logs["env_step_time"] += time.time() - env_step_start

        # Reward calculation
        reward_start = time.time()
        next_states = process_obs(next_obs)
        episode_logs["reward_time"] += time.time() - reward_start

        # Remember the experiences
        remember_start = time.time()
        agent.remember(states, actions, rewards, next_states, terminated)
        episode_logs["others_time"] += time.time() - remember_start

        # Train short memory
        train_start = time.time()
        agent.train_short_memory(states, actions, rewards, next_states, terminated)
        episode_logs["train_time"] += time.time() - train_start

        # Update observations
        obs = next_obs

        # Record gradient and loss
        grad_value, loss_value = agent.get_grad_loss_values()
        episode_logs["grad_value"] += grad_value
        episode_logs["loss_value"] += loss_value

        # Update episode statistics
        episode_rewards_tmp += rewards
        episode_lengths_tmp += np.ones(num_envs, dtype=int)

        step_count += 1

        # Check if any environments has terminated
        if any(terminated):
            list_not_to_average = ["episode", "episode_length", "episode_total_reward", "episode_avg_reward"]
            for key in episode_logs:
                if key not in list_not_to_average:
                    episode_logs[key] /= step_count
            terminated_envs = np.where(terminated)[0]
            for terminated_env in terminated_envs:
                episode_logs["episode_length"] = episode_lengths_tmp[terminated_env]
                episode_logs["episode_total_reward"] = episode_rewards_tmp[terminated_env]
                episode_logs["episode_avg_reward"] = (
                    episode_rewards_tmp[terminated_env] / episode_lengths_tmp[terminated_env]
                )

                # Record learning rate and epsilon
                episode_logs["learning_rate"] = agent.get_learning_rate()
                episode_logs["epsilon"] = agent.get_epsilon()

                # Reset the terminated environment's statistics
                episode_rewards_tmp[terminated_env] = 0.0
                episode_lengths_tmp[terminated_env] = 0

            break

    # Train long memory
    agent.update_model()

    # Decay epsilon
    agent.decay_epsilon_f()

    # Write episode statistics
    utils.write_log_entry(log_file_path, episode_logs)

    # Save periodically
    if (episode + 1) % 10 == 0:
        utils.save_periodic_model(agent, episode, model_path)

# Save the final model
utils.save_periodic_model(agent, n_episodes, model_path)
