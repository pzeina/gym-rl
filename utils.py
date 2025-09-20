import argparse
from pathlib import Path


def write_log_entry(log_file_path: Path, metrics: dict) -> None:
    """Write the episode statistics to a log file."""
    required_keys = {
        "episode",
        "episode_length",
        "episode_final_reward",
        "episode_total_reward",
        "episode_avg_reward",
        "learning_rate",
        "epsilon",
        "grad_value",
        "loss_value",
        "train_time",
        "env_step_time",
        "reward_time",
        "action_selection_time",
        "others_time",
    }
    if not required_keys.issubset(metrics.keys()):
        missing_keys = required_keys - metrics.keys()
        error_msg = f"The metrics dictionary must contain the following keys: {', '.join(missing_keys)}."
        raise ValueError(error_msg)

    log_entry = (
        f"{metrics['episode'] + 1}, {metrics['episode_final_reward']:.2f}, {metrics['episode_total_reward']:.2f}, "
        f"{metrics['episode_avg_reward']:.4f}, {metrics['episode_length']}, "
        f"{metrics['learning_rate']:.6f}, {metrics['epsilon']:.6f}, "
        f"{metrics['grad_value']:.4f}, {metrics['loss_value']:.4f}, "
        f"{metrics['train_time']:.4f}, {metrics['env_step_time']:.4f}, "
        f"{metrics['reward_time']:.4f}, {metrics['action_selection_time']:.4f}, "
        f"{metrics['others_time']:.4f}\n"
    )

    # Ensure directory exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write header if file does not exist
    if not log_file_path.exists():
        with log_file_path.open("w") as log_file:
            log_file.write(
                "Episode, Final Reward, Total Reward, Average Reward, Length, "
                "Learning Rate, Epsilon, Gradient, Loss, "
                "Train Time, Env Step Time, Reward Time, Action Selection Time, Others Time\n"
            )

    # Append log entry
    with log_file_path.open("a") as log_file:
        log_file.write(log_entry)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent.")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "dqn", "a2c", "sac", "td3", "ddpg"],
        help="RL algorithm to use for training."
    )
    parser.add_argument(
        "--tune-hyperparams",
        action="store_true",
        help="Enable automatic hyperparameter tuning using Optuna."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of hyperparameter optimization trials (only used with --tune-hyperparams)."
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name for the Optuna study (default: algorithm_timestamp)."
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="Number of parallel environments to use for training."
    )

    parser.add_argument("--render-mode", type=str, default="rgb_array", help="Enable map visualization.")

    return parser.parse_args()  # print(f"Number of CPUs: {N_ENVS}")
