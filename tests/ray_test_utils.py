"""Ray management utilities for tests."""

import atexit
import warnings
from contextlib import contextmanager

import ray


class RayTestManager:
    """Manages Ray initialization and cleanup for tests."""

    _instance = None
    _ray_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def init_ray_for_tests(cls):
        """Initialize Ray for testing with proper configuration."""
        if not cls._ray_initialized and not ray.is_initialized():
            # Suppress Ray warnings during tests
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", category=ResourceWarning)

                ray.init(
                    local_mode=True,
                    ignore_reinit_error=True,
                    include_dashboard=False,
                    log_to_driver=False,
                    object_store_memory=100_000_000,  # 100MB
                    num_cpus=1,
                    _temp_dir="/tmp/ray_test"
                )
            cls._ray_initialized = True

            # Register cleanup function
            atexit.register(cls.shutdown_ray)

    @classmethod
    def shutdown_ray(cls):
        """Properly shutdown Ray and clean up resources."""
        if cls._ray_initialized or ray.is_initialized():
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ResourceWarning)
                    ray.shutdown()
            except RuntimeError:
                # Ignore shutdown errors during cleanup
                pass
            finally:
                cls._ray_initialized = False


@contextmanager
def ray_test_context():
    """Context manager for Ray tests with proper cleanup."""
    RayTestManager.init_ray_for_tests()
    try:
        yield
    finally:
        # Individual test cleanup is handled by the manager
        pass


def setup_ray_for_test():
    """Set up Ray for a single test."""
    RayTestManager.init_ray_for_tests()

    # Register the environment for tests
    try:
        import sys
        from pathlib import Path

        from ray.tune.registry import register_env

        # Add project root to path
        project_root = Path(__file__).parent.parent.resolve()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from train_rllib import TerrainWorldRLlibWrapper
        register_env("terrain_world_rllib", TerrainWorldRLlibWrapper)
    except (ImportError, ModuleNotFoundError):
        # If registration fails, continue - some tests may not need it
        pass


def teardown_ray_after_test():
    """Clean up after a test (no-op, managed by RayTestManager)."""
