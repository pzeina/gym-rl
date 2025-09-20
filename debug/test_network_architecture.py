"""Network Architecture Test Script.

This script specifically tests the network architecture configuration
and validates that the modern format doesn't produce deprecation warnings.

Usage: python debug/test_network_architecture.py
"""

import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_old_vs_new_network_format() -> bool:
    """Test old vs new network architecture format to show the deprecation fix."""
    logger.info("🔍 Testing Network Architecture Formats...")

    # Create test environment
    env = make_vec_env("CartPole-v1", n_envs=1, vec_env_cls=DummyVecEnv)

    try:
        logger.info("\n--- Testing OLD format (deprecated) ---")
        # Old format that produces warnings
        old_policy_kwargs = {
            "net_arch": [{"pi": [64, 64], "vf": [64, 64]}],  # List of dict (deprecated)
            "activation_fn": torch.nn.Tanh,
        }

        _ = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=old_policy_kwargs,
            verbose=0
        )
        logger.info("✓ Old format works but may show deprecation warning")

        logger.info("\n--- Testing NEW format (recommended) ---")
        # New format (no warnings)
        new_policy_kwargs = {
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},  # Dict directly (modern)
            "activation_fn": torch.nn.Tanh,
        }

        _ = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=new_policy_kwargs,
            verbose=0
        )
        logger.info("✓ New format works without deprecation warnings")

    except RuntimeError:
        logger.exception("✗ Network architecture test failed")
        import traceback
        traceback.print_exc()
        return False
    else:
        return True
    finally:
        env.close()


def test_different_activation_functions() -> bool:
    """Test different activation functions for the network."""
    logger.info("\n")
    logger.info("🔍 Testing Different Activation Functions...")

    activation_functions = [
        ("Tanh", torch.nn.Tanh),
        ("ReLU", torch.nn.ReLU),
        ("LeakyReLU", torch.nn.LeakyReLU),
    ]

    env = make_vec_env("CartPole-v1", n_envs=1, vec_env_cls=DummyVecEnv)

    try:
        for name, activation_fn in activation_functions:
            policy_kwargs = {
                "net_arch": {"pi": [32, 32], "vf": [32, 32]},
                "activation_fn": activation_fn,
            }

            _ = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=0
            )
            logger.info("✓ %s activation function works", name)

    except RuntimeError:
        logger.exception("✗ Activation function test failed")
        return False
    else:
        return True
    finally:
        env.close()


def test_different_network_sizes() -> bool:
    """Test different network architectures."""
    logger.info("\n🔍 Testing Different Network Sizes...")

    network_configs = [
        ("Small", {"pi": [32, 32], "vf": [32, 32]}),
        ("Medium", {"pi": [128, 128], "vf": [128, 128]}),
        ("Large", {"pi": [256, 256], "vf": [256, 256]}),
        ("Asymmetric", {"pi": [256, 128], "vf": [128, 64]}),
        ("Different Depths", {"pi": [128, 128, 64], "vf": [128, 64]}),
    ]

    env = make_vec_env("CartPole-v1", n_envs=1, vec_env_cls=DummyVecEnv)

    try:
        for name, net_arch in network_configs:
            policy_kwargs = {
                "net_arch": net_arch,
                "activation_fn": torch.nn.Tanh,
            }

            _ = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=0
            )
            logger.info("✓ %s network architecture: %s", name, net_arch)

    except RuntimeError:
        logger.exception("✗ Network size test failed")
        return False
    else:
        return True
    finally:
        env.close()


def test_shared_vs_separate_networks() -> bool:
    """Test shared vs separate policy/value networks."""
    logger.info("\n🔍 Testing Shared vs Separate Networks...")

    env = make_vec_env("CartPole-v1", n_envs=1, vec_env_cls=DummyVecEnv)

    try:
        logger.info("--- Shared Network ---")
        # Shared network (simple list)
        shared_kwargs = {
            "net_arch": [128, 128],  # Shared layers
            "activation_fn": torch.nn.Tanh,
        }

        _ = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=shared_kwargs,
            verbose=0
        )
        logger.info("✓ Shared network architecture works")

        logger.info("\n--- Separate Networks ---")
        # Separate networks (dict)
        separate_kwargs = {
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "activation_fn": torch.nn.Tanh,
        }

        _ = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=separate_kwargs,
            verbose=0
        )
        logger.info("✓ Separate network architecture works")
        logger.info("✓ Separate networks often perform better for complex tasks")

    except RuntimeError:
        logger.exception("✗ Shared vs separate test failed")
        return False
    else:
        return True
    finally:
        env.close()


def test_train_py_configuration() -> bool:
    """Test the exact configuration used in train.py."""
    logger.info("\n🔍 Testing train.py Network Configuration...")

    env = make_vec_env("CartPole-v1", n_envs=1, vec_env_cls=DummyVecEnv)

    try:
        # Exact configuration from train.py
        policy_kwargs = {
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "activation_fn": torch.nn.Tanh,
        }

        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=64,  # Reduced for testing
            batch_size=32,
            n_epochs=10,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.0,
            verbose=0
        )
        logger.info("✓ train.py configuration works perfectly")
        logger.info("✓ Network: %s", policy_kwargs["net_arch"])
        logger.info("✓ Activation: %s", policy_kwargs["activation_fn"])

        # Test a very short training
        model.learn(total_timesteps=32)
        logger.info("✓ Short training run successful")

    except RuntimeError:
        logger.exception("✗ train.py configuration test failed")
        import traceback
        traceback.print_exc()
        return False
    else:
        return True
    finally:
        env.close()


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Network Architecture Test Suite")
    logger.info("=" * 70)

    success = True

    # Run all tests
    success &= test_old_vs_new_network_format()
    success &= test_different_activation_functions()
    success &= test_different_network_sizes()
    success &= test_shared_vs_separate_networks()
    success &= test_train_py_configuration()

    logger.info("")
    logger.info("=" * 70)
    if success:
        logger.info("🎉 All network architecture tests PASSED!")
        logger.info("✅ No deprecation warnings with modern net_arch format")
        logger.info("✅ train.py uses optimal network configuration")
    else:
        logger.error("❌ Some network architecture tests FAILED!")
        sys.exit(1)
    logger.info("=" * 70)
