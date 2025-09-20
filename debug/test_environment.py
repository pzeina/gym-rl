"""Environment Creation Test Script.

This script tests the environment creation and basic functionality
to ensure the make_env function works correctly.

Usage: python debug/test_environment.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_environment_creation() -> bool:
    """Test basic environment creation and functionality."""
    logger.info("🔍 Testing Environment Creation...")

    try:
        # Import the make_env function from train.py
        from train import make_env
        logger.info("✓ make_env function imported successfully")

        # Create environment factory
        env_fn = make_env()
        logger.info("✓ Environment factory created successfully")

        # Create actual environment
        env = env_fn()
        logger.info("✓ Environment created successfully")

        # Test environment properties
        logger.info("Action space: %s", env.action_space)
        logger.info("Observation space: %s", env.observation_space)

        # Test environment reset
        obs, _ = env.reset()
        logger.info("✓ Environment reset successfully")
        logger.info("Initial observation keys: %s", list(obs.keys()))

        # Test a random action
        action = env.action_space.sample()
        obs, reward, _, _, _ = env.step(action)
        logger.info("✓ Environment step completed - reward: %s", reward)

        env.close()
        logger.info("✓ Environment closed successfully")

    except ImportError:
        logger.exception("✗ Import error during environment testing")
        return False
    except RuntimeError:
        logger.exception("✗ Runtime error during environment testing")
        return False
    else:
        return True

def test_monitor_wrapper() -> bool:
    """Test that the Monitor wrapper is working correctly."""
    logger.info("")
    logger.info("🔍 Testing Monitor Wrapper...")

    try:
        import gymnasium as gym
        from stable_baselines3.common.monitor import Monitor

        # Test that Monitor wrapper works with our environment
        env = gym.make("mili_env/TerrainWorld-v0", render_mode=None)
        env = Monitor(env)
        env.reset(seed=42)

        logger.info("✓ Monitor wrapper applied successfully")

        # Test that it tracks episode stats
        for _ in range(5):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        env.close()
        logger.info("✓ Monitor wrapper functioning correctly")

    except ImportError:
        logger.exception("✗ Import error testing Monitor wrapper")
        return False
    except RuntimeError:
        logger.exception("✗ Runtime error testing Monitor wrapper")
        return False
    return True

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Environment Test Suite")
    logger.info("=" * 50)

    success = True

    # Run tests
    success &= test_environment_creation()
    success &= test_monitor_wrapper()

    logger.info("")
    logger.info("=" * 50)
    if success:
        logger.info("🎉 All environment tests PASSED!")
        logger.info("✅ The environment is ready for training!")
    else:
        logger.error("❌ Some environment tests FAILED!")
        sys.exit(1)
    logger.info("=" * 50)
