"""Test script for multi-agent communication system."""

import logging

import numpy as np

from mili_env.envs.classes.robot_base import Actions
from mili_env.envs.terrain_world import TerrainWorldEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_communication() -> None:
    """Test the communication system between agents."""
    logger.info("Creating multi-agent environment with 3 agents...")

    # Create environment with 3 agents
    env = TerrainWorldEnv(num_agents=3, render_mode=None)

    # Reset environment
    obs, info = env.reset()
    logger.info("Environment reset successful!")
    logger.info("Number of agents: %d", len(env.agents))
    logger.info("Observation space keys: %s", list(obs.keys()))

    # Test basic communication
    logger.info("\nAgent initial positions:")
    for i, agent in enumerate(env.agents):
        pos = agent.get_position()
        logger.info("  Agent %d: position %s, health %.1f", i, pos, agent.get_health())

    # Test status broadcast from first agent
    logger.info("\nTesting status broadcast...")
    env.agents[0].broadcast_status()

    def check_inbox_outbox() -> None:
        """Check and log inbox/outbox sizes for all agents."""
        for i, agent in enumerate(env.agents):
            inbox_size = len(getattr(agent, "communication_inbox", []))
            outbox_size = len(getattr(agent, "communication_outbox", []))
            logger.info("  Agent %d: inbox=%d, outbox=%d", i, inbox_size, outbox_size)

    # Take a few steps to test communication processing
    rng = np.random.default_rng()  # Use NumPy Generator for random numbers
    for step in range(5):
        logger.info("Step %d:", step + 1)

        # Create random actions for all agents
        actions = {
            f"agent_{i}": rng.choice([Actions.IDLE.value, Actions.FORWARD.value, Actions.ROTATE_LEFT.value])
            for i in range(env.num_agents)
        }

        obs, rewards, terminated, truncated, info = env.step(actions)
        check_inbox_outbox()

        logger.info("  Step completed. Current step counter: %d", env.current_step)
        logger.info("  Rewards: %s", rewards)

    logger.info("\nCommunication test completed successfully!")

if __name__ == "__main__":
    test_communication()
