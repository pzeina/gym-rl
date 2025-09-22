#!/usr/bin/env python3
"""Minimal test script for RLlib multi-agent training."""

import logging

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_minimal_rllib():
    """Test minimal RLlib setup."""
    print("🔬 Testing minimal RLlib multi-agent setup...")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=2, num_gpus=0, ignore_reinit_error=True, log_to_driver=False)

    try:
        # Import our environment wrapper
        from train_rllib import TerrainWorldRLlibWrapper

        # Register environment
        register_env("terrain_world_ma", TerrainWorldRLlibWrapper)
        print("✅ Environment registered successfully")

        # Test environment creation
        env_config = {"num_agents": 2, "render_mode": None}
        env = TerrainWorldRLlibWrapper({"env_config": env_config})
        print(f"✅ Environment created with {env.num_agents} agents")

        # Test reset and step
        obs, info = env.reset()
        print(f"✅ Environment reset successful. Agents: {list(obs.keys())}")

        actions = dict.fromkeys(env.agent_ids, 0)
        obs, rewards, dones, info = env.step(actions)
        print(f"✅ Environment step successful. Rewards: {rewards}")

        # Create minimal PPO config
        config = (
            PPOConfig()
            .environment(
                env="terrain_world_ma",
                env_config=env_config,
                disable_env_checking=True,
            )
            .multi_agent(
                policies={
                    f"agent_{i}": (None, env.observation_space, env.action_space, {})
                    for i in range(env_config["num_agents"])
                },
                policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
            )
            .framework("torch")
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .training(
                lr=3e-4,
                train_batch_size=1000,  # Smaller batch size for quick test
                minibatch_size=64,
                num_epochs=3,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                vf_loss_coeff=0.5,
                entropy_coeff=0.0,
            )
            .env_runners(
                num_env_runners=1,  # Single worker for minimal test
                num_envs_per_env_runner=1,
            )
            .resources(
                num_gpus=0,
                num_cpus_per_worker=1,
            )
            .debugging(log_level="ERROR")  # Minimize logging for test
        )
        print("✅ PPO configuration created successfully")

        # Build algorithm (this tests configuration validity)
        algo = config.build()
        print("✅ PPO algorithm built successfully")

        # Run a few training steps
        print("🏃 Running 3 training iterations...")
        for i in range(3):
            result = algo.train()
            reward = result.get("episode_reward_mean", "N/A")
            timesteps = result.get("timesteps_total", 0)
            print(f"  Iteration {i+1}: Reward={reward}, Timesteps={timesteps}")

        # Clean up
        algo.stop()
        print("✅ Algorithm stopped successfully")

        ray.shutdown()
        print("✅ Ray shutdown successfully")

        print("\n🎉 ALL TESTS PASSED! RLlib multi-agent training is working!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        if ray.is_initialized():
            ray.shutdown()
        return False

if __name__ == "__main__":
    success = test_minimal_rllib()
    exit(0 if success else 1)
