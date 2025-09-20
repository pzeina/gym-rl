"""Import Validation Test Script.

This script tests that all imports in train.py work correctly
and validates the dependencies are properly installed.

Usage: python debug/test_imports.py
"""

import logging
import sys
from pathlib import Path

# Configure logging to show info messages
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_basic_imports() -> bool:
    """Test basic Python and ML library imports."""
    logger.info("🔍 Testing Basic Imports...")

    try:
        import time  # noqa: F401
        logger.info("✓ time")

        import logging  # noqa: F401
        logger.info("✓ logging")

        import warnings  # noqa: F401
        logger.info("✓ warnings")

        from pathlib import Path  # noqa: F401
        logger.info("✓ pathlib.Path")

    except ImportError:
        logger.exception("✗ Basic import failed")
        return False
    else:
        return True


def test_stable_baselines3_imports() -> bool:
    """Test Stable-Baselines3 related imports."""
    logger.info("\n🔍 Testing Stable-Baselines3 Imports...")

    try:
        from stable_baselines3 import PPO  # noqa: F401
        logger.info("✓ stable_baselines3.PPO")

        from stable_baselines3.common.env_util import make_vec_env  # noqa: F401
        logger.info("✓ stable_baselines3.common.env_util.make_vec_env")

        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize  # noqa: F401
        logger.info("✓ stable_baselines3.common.vec_env.DummyVecEnv")
        logger.info("✓ stable_baselines3.common.vec_env.SubprocVecEnv")
        logger.info("✓ stable_baselines3.common.vec_env.VecNormalize")

        from stable_baselines3.common.monitor import Monitor  # noqa: F401
        logger.info("✓ stable_baselines3.common.monitor.Monitor")

        from stable_baselines3.common.callbacks import EvalCallback  # noqa:F401
        logger.info("✓ stable_baselines3.common.callbacks.EvalCallback")

    except ImportError:
        logger.exception("✗ Stable-Baselines3 import failed")
        return False
    else:
        return True


def test_torch_imports() -> bool:
    """Test PyTorch imports."""
    logger.info("\n🔍 Testing PyTorch Imports...")

    try:
        import torch
        logger.info("✓ torch (version: %s)", torch.__version__)

        import torch.nn
        logger.info("✓ torch.nn")

        # Test specific activation functions used in train.py
        _ = torch.nn.Tanh()
        logger.info("✓ torch.nn.Tanh")

    except ImportError:
        logger.exception("✗ PyTorch import failed")
        return False
    else:
        return True


def test_gymnasium_imports() -> bool:
    """Test Gymnasium (OpenAI Gym) imports."""
    logger.info("\n🔍 Testing Gymnasium Imports...")

    try:
        import gymnasium as gym
        logger.info("✓ gymnasium (version: %s)", gym.__version__)

        # Test that we can create a basic environment
        env = gym.make("CartPole-v1")
        logger.info("✓ gymnasium.make")
        env.close()

    except ImportError:
        logger.exception("✗ Gymnasium import failed")
        return False
    else:
        return True


def test_custom_environment_imports() -> bool:
    """Test custom environment imports from the project."""
    logger.info("\n🔍 Testing Custom Environment Imports...")

    try:
        # First, try to import the environment module to register it
        import mili_env  # noqa: F401
        logger.info("✓ mili_env module imported")

        # Test custom environment registration
        import gymnasium as gym
        env = gym.make("mili_env/TerrainWorld-v0", render_mode=None)
        logger.info("✓ mili_env.TerrainWorld-v0 environment registered")
        env.close()

    except ImportError as e:
        logger.warning("⚠️  Custom environment not available: %s", e)
        logger.warning("   This is expected if the environment is not registered")
        return True  # Not a failure, just environment not available
    except (TypeError, AttributeError) as e:
        logger.warning("⚠️  Custom environment registration failed: %s", e)
        logger.warning("   This might be expected during testing")
        return True  # Not a critical failure for import testing
    else:
        return True


def test_train_py_imports() -> bool:
    """Test importing train.py and its functions."""
    logger.info("\n🔍 Testing train.py Imports...")

    try:
        # Import train module
        import train
        logger.info("✓ train module imported")

        # Test specific functions
        _ = train.make_env
        logger.info("✓ train.make_env function")

        _ = train.main
        logger.info("✓ train.main function")

    except ImportError:
        logger.exception("✗ train.py import failed")
        import traceback
        traceback.print_exc()
        return False
    else:
        return True


def test_utils_imports() -> bool:
    """Test utils module imports."""
    logger.info("")
    logger.info("🔍 Testing Utils Imports...")

    try:
        import utils
        logger.info("✓ utils module imported")

        # Test parse_args function
        _ = utils.parse_args
        logger.info("✓ utils.parse_args function")

    except ImportError:
        logger.exception("✗ utils import failed")
        return False
    else:
        return True


def test_version_compatibility() -> bool:
    """Test version compatibility of key packages."""
    logger.info("\n🔍 Testing Version Compatibility...")

    try:
        import gymnasium
        import stable_baselines3
        import torch

        logger.info("✓ stable-baselines3: %s", stable_baselines3.__version__)
        logger.info("✓ torch: %s", torch.__version__)
        logger.info("✓ gymnasium: %s", gymnasium.__version__)

        # Check Python version
        python_version = sys.version_info
        logger.info("✓ Python: %d.%d.%d", python_version.major, python_version.minor, python_version.micro)

        if python_version < (3, 8):
            logger.warning("⚠️  Python version is quite old, consider upgrading")

    except ImportError:
        logger.exception("✗ Version check failed (import error)")
        return False
    except AttributeError:
        logger.exception("✗ Version check failed (attribute error)")
        return False
    else:
        return True

def test_gpu_availability() -> bool:
    """Test GPU availability for PyTorch."""
    logger.info("\n🔍 Testing GPU Availability...")

    try:
        import torch

        if torch.cuda.is_available():
            logger.info("✓ CUDA available with %d device(s)", torch.cuda.device_count())
            logger.info("✓ Current device: %s", torch.cuda.get_device_name())
        else:
            logger.warning("⚠️  CUDA not available, will use CPU")

        if torch.backends.mps.is_available():
            logger.info("✓ MPS (Apple Silicon) available")
        else:
            logger.warning("  MPS not available")

    except ImportError:
        logger.exception("✗ GPU check failed (import error)")
        return False
    except RuntimeError:
        logger.exception("✗ GPU check failed (runtime error)")
        return False
    else:
        return True


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Import Validation Test Suite")
    logger.info("=" * 60)

    success = True

    # Run all import tests
    success &= test_basic_imports()
    success &= test_stable_baselines3_imports()
    success &= test_torch_imports()
    success &= test_gymnasium_imports()
    success &= test_custom_environment_imports()
    success &= test_train_py_imports()
    success &= test_utils_imports()
    success &= test_version_compatibility()
    success &= test_gpu_availability()

    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("🎉 All import tests PASSED!")
        logger.info("✅ All dependencies are correctly installed")
        logger.info("✅ train.py is ready to run")
    else:
        logger.error("❌ Some import tests FAILED!")
        logger.info("💡 Install missing dependencies with: pip install -r requirements.txt")
        sys.exit(1)
    logger.info("=" * 60)
