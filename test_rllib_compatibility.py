"""Test script for RLlib multi-agent training compatibility."""

import logging

import ray
from ray.tune.registry import register_env

from hyperparameter_tuning_rllib import get_default_hyperparams
from train_rllib import TerrainWorldRLlibWrapper, create_algorithm_config

logger = logging.getLogger(__name__)


# Test the environment wrapper
def test_environment_wrapper() -> bool:
    """Test the RLlib environment wrapper."""
    logger.info("Testing RLlib environment wrapper...")

    # Test environment creation
    env_config = {"num_agents": 4, "render_mode": None}
    env = TerrainWorldRLlibWrapper({"env_config": env_config})

    logger.info("Environment created with %d agents", env.num_agents)
    logger.info("Agent IDs: %s", env.agent_ids)
    logger.info("Observation space: %s", env.observation_space)
    logger.info("Action space: %s", env.action_space)

    # Test reset
    obs, _info = env.reset()
    logger.info("Reset successful. Observation keys: %s", list(obs.keys()))

    # Test step
    actions = dict.fromkeys(env.agent_ids, 0)
    obs, rewards, terminated, truncated, _info = env.step(actions)
    logger.info("Step successful. Rewards: %s", rewards)
    logger.info("Terminated: %s", terminated)
    logger.info("Truncated: %s", truncated)

    logger.info("Environment wrapper test completed successfully!")
    return True


def test_algorithm_config() -> bool:
    """Test algorithm configuration creation."""
    logger.info("Testing algorithm configuration...")

    env_config = {"num_agents": 4, "render_mode": None}
    hyperparams = get_default_hyperparams("ppo")

    try:
        config = create_algorithm_config(
            algorithm="ppo",
            env_config=env_config,
            hyperparams=hyperparams,
            device="cpu",
        )
    except (ValueError, TypeError, ImportError):
        logger.exception("Algorithm configuration failed")
        return False
    else:
        logger.info("Algorithm configuration created successfully!")
        logger.info("Config type: %s", type(config))
        return True


def test_ray_initialization() -> bool:
    """Test Ray initialization."""
    logger.info("Testing Ray initialization...")

    try:
        if not ray.is_initialized():
            ray.init(num_cpus=2, num_gpus=0, ignore_reinit_error=True)

        logger.info("Ray initialized successfully!")

        # Test environment registration
        register_env("terrain_world_ma", TerrainWorldRLlibWrapper)
        logger.info("Environment registered successfully!")

        ray.shutdown()
    except (RuntimeError, ValueError, ImportError):
        logger.exception("Ray initialization failed")
        return False
    else:
        return True


def main() -> bool:
    """Run all tests."""
    logging.basicConfig(level=logging.INFO)

    logger.info("=== RLlib Multi-Agent Training Compatibility Test ===")

    tests = [
        ("Environment Wrapper", test_environment_wrapper),
        ("Algorithm Config", test_algorithm_config),
        ("Ray Initialization", test_ray_initialization),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except (RuntimeError, ValueError, ImportError, TypeError):
            logger.exception("%s test failed with exception", test_name)
            results.append((test_name, False))

    logger.info("=== Test Results ===")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info("%s: %s", test_name, status)

    all_passed = all(result for _, result in results)
    if all_passed:
        logger.info("🎉 All tests passed! RLlib integration is ready.")
    else:
        logger.warning("⚠️  Some tests failed. Please check the implementation.")

    return all_passed


if __name__ == "__main__":
    main()
