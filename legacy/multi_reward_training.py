"""Simple script to train with different reward functions."""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default reward functions to test
DEFAULT_REWARD_FUNCTIONS = [
    "per_agent_progress_reward",
    "per_agent_energy_preservation",
    "team_cooperation_bonus"
]


def run_training_with_reward(reward_function: str, output_dir: Path,
                           algorithm: str = "ppo", parallel: int = 4,
                           total_timesteps: int = 50000) -> bool:
    """Run training with a specific reward function."""
    logger.info("Starting training with reward function: %s", reward_function)

    # Set environment variable for reward function
    env = os.environ.copy()
    env["REWARD_FUNCTION"] = reward_function

    # Build command
    cmd = [
        sys.executable, "train.py",
        "--algorithm", algorithm,
        "--parallel", str(parallel),
        "--timesteps", str(total_timesteps)
    ]

    # Create output directory for this reward function
    reward_dir = output_dir / f"reward_{reward_function}"
    reward_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run training
        start_time = time.time()
        result = subprocess.run(  # noqa: S603
            cmd,
            env=env,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            check=False  # We handle errors manually
        )
        duration = time.time() - start_time

        # Save output
        log_file = reward_dir / "training.log"
        with log_file.open("w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Environment: REWARD_FUNCTION={reward_function}\n")
            f.write(f"Duration: {duration:.1f}s\n")
            f.write(f"Exit code: {result.returncode}\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)

        if result.returncode == 0:
            logger.info("✓ Training completed successfully for %s (%.1fs)",
                       reward_function, duration)
            return True

        logger.error("✗ Training failed for %s (exit code %d)",
                    reward_function, result.returncode)

    except subprocess.TimeoutExpired:
        logger.exception("✗ Training timed out for %s", reward_function)
        return False
    except (OSError, subprocess.SubprocessError):
        logger.exception("✗ Training error for %s", reward_function)
        return False
    else:
        return False


def main() -> None:
    """Main function to run training with multiple reward functions."""
    parser = argparse.ArgumentParser(description="Train with multiple reward functions")
    parser.add_argument(
        "reward_functions",
        nargs="*",
        default=DEFAULT_REWARD_FUNCTIONS,
        help="Reward functions to test (default: all available)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("multi_reward_results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--algorithm",
        default="ppo",
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total training timesteps"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting multi-reward training with %d reward functions",
               len(args.reward_functions))
    logger.info("Output directory: %s", args.output_dir)

    # Run training for each reward function
    results = []
    total_start = time.time()

    for reward_function in args.reward_functions:
        success = run_training_with_reward(
            reward_function=reward_function,
            output_dir=args.output_dir,
            algorithm=args.algorithm,
            parallel=args.parallel,
            total_timesteps=args.timesteps
        )
        results.append((reward_function, success))

    total_time = time.time() - total_start

    # Print summary
    logger.info("="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info("Total time: %.1f seconds", total_time)

    successful = sum(1 for _, success in results if success)
    logger.info("Successful runs: %d/%d", successful, len(results))

    for reward_function, success in results:
        status = "✓" if success else "✗"
        logger.info("%s %s", status, reward_function)

    # Save summary
    summary_file = args.output_dir / "summary.txt"
    with summary_file.open("w") as f:
        f.write("Multi-reward training summary\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time:.1f}s\n")
        f.write(f"Successful runs: {successful}/{len(results)}\n\n")

        for reward_function, success in results:
            status = "SUCCESS" if success else "FAILED"
            f.write(f"{reward_function}: {status}\n")

    logger.info("Summary saved to: %s", summary_file)


if __name__ == "__main__":
    main()
