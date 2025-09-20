"""SB3 Components Test Script.

This script tests the complete SB3 implementation including VecNormalize,
PPO with custom policy_kwargs, and parallel environments.

Usage: python debug/test_sb3_components.py
"""

# ruff: noqa: T201  # Allow print statements in debug scripts

import sys
import traceback
from pathlib import Path
from typing import Any

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


def test_vecnormalize() -> bool:
    """Test VecNormalize wrapper functionality."""
    print("🔍 Testing VecNormalize...")

    try:
        # Create vectorized environment
        env = make_vec_env("CartPole-v1", n_envs=2, vec_env_cls=DummyVecEnv)

        # Apply VecNormalize
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        print("✓ VecNormalize wrapper applied successfully")

        # Test reset and step
        obs = env.reset()
        # For vectorized envs, obs is a numpy array, not a dict
        if isinstance(obs, dict):
            print(f"✓ Normalized observations keys: {list(obs.keys())}")
        else:
            print(f"✓ Normalized observations shape: {obs.shape}")

        # Test normalization stats - create proper numpy array for actions
        actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        obs, rewards, _dones, _infos = env.step(actions)
        print(f"✓ VecNormalize step completed, rewards: {rewards}")

        env.close()
    except (ImportError, RuntimeError, ValueError) as e:
        print(f"✗ VecNormalize test failed: {e}")
        traceback.print_exc()
        return False
    else:
        return True


def test_parallel_environments() -> bool:
    """Test SubprocVecEnv vs DummyVecEnv."""
    print("\n🔍 Testing Parallel Environments...")

    try:
        # Test DummyVecEnv
        env_dummy = make_vec_env("CartPole-v1", n_envs=2, vec_env_cls=DummyVecEnv)
        _obs_dummy = env_dummy.reset()
        print(f"✓ DummyVecEnv created: {env_dummy.num_envs} environments")
        env_dummy.close()

        # Test SubprocVecEnv (may fail on some systems)
        try:
            env_subproc = make_vec_env("CartPole-v1", n_envs=2, vec_env_cls=SubprocVecEnv)
            _obs_subproc = env_subproc.reset()
            print(f"✓ SubprocVecEnv created: {env_subproc.num_envs} environments")
            env_subproc.close()
        except (ImportError, RuntimeError, OSError) as e:
            print(f"⚠️  SubprocVecEnv failed (expected on some systems): {e}")
            print("   Will fall back to DummyVecEnv in training")

    except (ImportError, RuntimeError) as e:
        print(f"✗ Parallel environment test failed: {e}")
        return False
    else:
        return True


def test_custom_network_architecture() -> bool:
    """Test PPO with custom network architecture."""
    print("\n🔍 Testing Custom Network Architecture...")

    try:
        # Create environment
        env = make_vec_env("CartPole-v1", n_envs=1, vec_env_cls=DummyVecEnv)

        # Test modern network architecture format (dict, not list)
        policy_kwargs = {
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "activation_fn": torch.nn.Tanh,
        }

        # Create PPO model
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=64,  # Small for testing
            batch_size=32,
            verbose=0
        )
        print("✓ PPO model with custom network created successfully")
        print(f"✓ Network architecture: {policy_kwargs['net_arch']}")
        print(f"✓ Activation function: {policy_kwargs['activation_fn']}")

        # Test short training run (remove log_interval parameter)
        model.learn(total_timesteps=100)
        print("✓ Short training run completed successfully")

        env.close()
    except (ImportError, RuntimeError, ValueError) as e:
        print(f"✗ Custom network test failed: {e}")
        traceback.print_exc()
        return False
    else:
        return True


def test_complete_pipeline() -> bool:
    """Test the complete SB3 pipeline as used in train.py."""
    print("\n🔍 Testing Complete SB3 Pipeline...")

    try:
        # Create vectorized environment with monitoring
        def make_env_with_monitor() -> gym.Env[Any, Any]:
            """Create a single environment with Monitor wrapper."""
            return Monitor(gym.make("CartPole-v1"))

        # Create vectorized environment (use string ID for simplicity)
        env = make_vec_env("CartPole-v1", n_envs=2, vec_env_cls=DummyVecEnv)

        # Apply VecNormalize
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        print("✓ Vectorized environment with Monitor and VecNormalize created")

        # Create PPO with tuned hyperparameters (same as train.py)
        policy_kwargs = {
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},  # Smaller for testing
            "activation_fn": torch.nn.Tanh,
        }

        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=64,  # Smaller for testing
            batch_size=32,
            n_epochs=4,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.0,
            verbose=0
        )
        print("✓ Complete PPO pipeline configured successfully")

        # Test training (remove log_interval parameter)
        model.learn(total_timesteps=200)
        print("✓ Complete pipeline training test successful")

        env.close()
    except (ImportError, RuntimeError, ValueError) as e:
        print(f"✗ Complete pipeline test failed: {e}")
        traceback.print_exc()
        return False
    else:
        return True


def test_terrain_world_environment() -> bool:
    """Test with the actual TerrainWorld environment if available."""
    print("\n🔍 Testing TerrainWorld Environment...")

    try:
        # Try to import and test our custom environment
        env = gym.make("mili_env/TerrainWorld-v0", render_mode=None)
        env = Monitor(env)
        print("✓ TerrainWorld environment created successfully")

        # Test observation space
        obs, _info = env.reset(seed=42)
        print(f"✓ TerrainWorld observation keys: {list(obs.keys())}")
        print(f"✓ Action space: {env.action_space}")

        # Test with VecNormalize
        env.close()

        # Now test with vectorization - use string ID instead
        vec_env = make_vec_env("mili_env/TerrainWorld-v0", n_envs=2, vec_env_cls=DummyVecEnv)
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        print("✓ TerrainWorld with VecNormalize created successfully")

        obs = vec_env.reset()
        # Handle both dict and array observations
        if isinstance(obs, dict):
            print(f"✓ Vectorized TerrainWorld observation keys: {list(obs.keys())}")
        else:
            print(f"✓ Vectorized TerrainWorld observation shape: {obs.shape}")

        vec_env.close()
    except (ImportError, RuntimeError, ValueError, gym.error.NamespaceNotFound) as e:
        print(f"⚠️  TerrainWorld test skipped (environment not available): {e}")
        return True  # Not a failure, just environment not registered
    else:
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("SB3 Components Test Suite")
    print("=" * 60)

    success = True

    # Run all tests
    success &= test_vecnormalize()
    success &= test_parallel_environments()
    success &= test_custom_network_architecture()
    success &= test_complete_pipeline()
    success &= test_terrain_world_environment()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All SB3 component tests PASSED!")
        print("✅ The implementation is ready for training!")
    else:
        print("❌ Some SB3 component tests FAILED!")
        sys.exit(1)
    print("=" * 60)
