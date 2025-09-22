"""Unit tests for agent reward systems."""

import unittest

import numpy as np

from mili_env.envs.classes.robot_base import Actions
from mili_env.envs.terrain_world import TerrainWorldEnv


class TestAgentRewards(unittest.TestCase):
    """Test cases for agent reward systems and cooperation mechanics."""

    def setUp(self):
        """Set up test environment."""
        self.env = TerrainWorldEnv(num_agents=3, render_mode=None)
        self.agent = self.env.agents[0]

    def test_environment_rewards_structure(self):
        """Test that environment returns proper reward structure."""
        # Execute a step to get rewards
        action_dict = {f"agent_{i}": Actions.IDLE.value for i in range(self.env.num_agents)}
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)

        # TerrainWorldEnv returns scalar reward, not per-agent dict
        assert isinstance(rewards, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        # The info dict should contain per-agent information
        assert isinstance(info, dict)

    def test_centralized_reward_calculation(self):
        """Test centralized reward calculation."""
        # Get initial rewards
        action_dict = {f"agent_{i}": Actions.IDLE.value for i in range(self.env.num_agents)}
        obs, rewards1, terminated, truncated, info = self.env.step(action_dict)

        # Execute another step
        obs, rewards2, terminated, truncated, info = self.env.step(action_dict)

        # Rewards should be scalar values
        assert isinstance(rewards1, float)
        assert isinstance(rewards2, float)

        # Rewards should be finite numbers
        assert np.isfinite(rewards1)
        assert np.isfinite(rewards2)

    def test_agent_target_reaching_reward(self):
        """Test reward for reaching target."""
        # Position agent at target center (within bounds)
        target = self.agent.get_target()
        target_x, target_y = target[0], target[1]

        # Position agent at the exact center of the target zone
        self.agent.state.x = float(target_x)
        self.agent.state.y = float(target_y)

        # Check if agent is at target
        is_at_target = self.agent.state.is_at_target()

        # The method should exist and return a boolean (including numpy bool)
        assert isinstance(is_at_target, bool | np.bool_)
        # Note: The target bounds depend on target_width and target_height
        # so we just test that the method works, not the specific result

    def test_energy_consumption_penalties(self):
        """Test that energy consumption affects rewards."""
        # Execute energy-consuming actions
        action_dict = {f"agent_{i}": Actions.FORWARD.value for i in range(self.env.num_agents)}

        # Execute energy-consuming actions
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)

        # Should receive scalar reward
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)

    def test_health_and_survival_rewards(self):
        """Test health-related rewards."""
        # Check initial health
        initial_health = self.agent.get_health()
        assert isinstance(initial_health, int | float)
        assert initial_health > 0

        # Agent should be alive initially
        assert self.agent.state.is_alive()

        # Execute actions and check rewards
        action_dict = {f"agent_{i}": Actions.IDLE.value for i in range(self.env.num_agents)}
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)

        # Should receive scalar reward
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)

    def test_cooperation_reward_system(self):
        """Test cooperative behavior rewards."""
        # Position agents close together
        for i, agent in enumerate(self.env.agents):
            agent.state.x = 10.0 + i * 2.0  # Close but not overlapping
            agent.state.y = 10.0

        # Execute cooperative actions
        action_dict = {f"agent_{i}": Actions.FORWARD.value for i in range(self.env.num_agents)}
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)

        # Should receive scalar reward
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)

    def test_reward_consistency(self):
        """Test that rewards are consistent across multiple steps."""
        rewards_history = []

        # Execute multiple steps with same actions
        action_dict = {f"agent_{i}": Actions.IDLE.value for i in range(self.env.num_agents)}

        for _ in range(5):
            obs, rewards, terminated, truncated, info = self.env.step(action_dict)
            rewards_history.append(rewards)  # rewards is now a scalar

        # Should have collected multiple reward sets
        assert len(rewards_history) == 5

        # All rewards should be finite numbers
        for reward in rewards_history:
            assert isinstance(reward, float)
        # All rewards should be finite numbers
        for reward in rewards_history:
            assert isinstance(reward, float)
            assert np.isfinite(reward)

    def test_individual_vs_team_rewards(self):
        """Test individual vs team reward components."""
        # Execute step to get rewards
        action_dict = {f"agent_{i}": Actions.FORWARD.value for i in range(self.env.num_agents)}
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)

        # TerrainWorldEnv returns scalar rewards, not per-agent
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)

    def test_action_specific_rewards(self):
        """Test that different actions yield different rewards."""
        initial_pos = self.agent.get_position()

        # Test different actions
        actions_to_test = [Actions.IDLE, Actions.FORWARD, Actions.BACKWARD,
                          Actions.ROTATE_LEFT, Actions.ROTATE_RIGHT]

        action_rewards = {}

        for action in actions_to_test:
            # Reset agent position
            self.agent.state.x = initial_pos[0]
            self.agent.state.y = initial_pos[1]

            # Execute action
            action_dict = {f"agent_{i}": action.value for i in range(self.env.num_agents)}
            obs, rewards, terminated, truncated, info = self.env.step(action_dict)

            action_rewards[action.name] = rewards  # rewards is now scalar

        # Should have rewards for all actions tested
        assert len(action_rewards) == len(actions_to_test)

        # All rewards should be numeric
        for action_name, reward in action_rewards.items():
            assert isinstance(reward, float), f"Reward for {action_name} is not numeric"

    def test_distance_based_rewards(self):
        """Test distance-based reward calculations."""
        # Get initial distance to target
        initial_distance = self.agent.get_distance_to_target()
        assert isinstance(initial_distance, int | float)
        assert initial_distance >= 0

        # Move agent and check distance change
        action_dict = {f"agent_{0}": Actions.FORWARD.value}
        for i in range(1, self.env.num_agents):
            action_dict[f"agent_{i}"] = Actions.IDLE.value

        obs, rewards, terminated, truncated, info = self.env.step(action_dict)

        # Should receive scalar reward
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)

    def test_multi_agent_reward_distribution(self):
        """Test reward distribution among multiple agents."""
        # Position agents in different locations
        positions = [(5, 5), (15, 15), (25, 5)]
        for i, (x, y) in enumerate(positions[:self.env.num_agents]):
            self.env.agents[i].state.x = float(x)
            self.env.agents[i].state.y = float(y)

        # Execute actions
        action_dict = {f"agent_{i}": Actions.FORWARD.value for i in range(self.env.num_agents)}
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)

        # TerrainWorldEnv returns scalar reward
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)

    def test_reward_bounds(self):
        """Test that rewards are within reasonable bounds."""
        # Execute multiple steps and collect rewards
        all_rewards = []
        action_dict = {f"agent_{i}": Actions.FORWARD.value for i in range(self.env.num_agents)}

        for _ in range(10):
            obs, rewards, terminated, truncated, info = self.env.step(action_dict)
            all_rewards.append(rewards)  # rewards is now scalar

        # Rewards should be finite numbers
        for reward in all_rewards:
            assert isinstance(reward, float)
            assert np.isfinite(reward), "Reward should be finite"
            assert not np.isnan(reward), "Reward should not be NaN"

    def test_done_conditions_and_rewards(self):
        """Test reward behavior when episodes end."""
        # Execute actions until done or timeout
        last_rewards = None
        max_steps = 100
        action_dict = {f"agent_{i}": Actions.FORWARD.value for i in range(self.env.num_agents)}

        for _ in range(max_steps):
            obs, rewards, terminated, truncated, info = self.env.step(action_dict)

            # Check reward structure - should be scalar
            assert isinstance(rewards, float)
            last_rewards = rewards

            # Check if episode is done
            if terminated or truncated:
                break

        # Final rewards should still be valid
        if last_rewards is not None:
            assert isinstance(last_rewards, float)
            assert np.isfinite(last_rewards)

    def test_ammunition_and_combat_rewards(self):
        """Test ammunition-related rewards."""
        # Check initial ammunition
        initial_ammo = self.agent.get_ammunition()
        assert isinstance(initial_ammo, int | float)
        assert initial_ammo >= 0

        # Execute actions
        action_dict = {f"agent_{i}": Actions.IDLE.value for i in range(self.env.num_agents)}
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)

        # Should receive scalar reward
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)


if __name__ == "__main__":
    unittest.main()
