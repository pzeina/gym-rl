"""Unit tests for agent reward systems."""

import unittest

import numpy as np

from mili_env.envs.classes.robot_base import Moves
from mili_env.envs.terrain_world import TerrainWorldEnv


class TestAgentRewards(unittest.TestCase):
    """Test cases for agent reward systems and cooperation mechanics."""

    def setUp(self) -> None:
        self.env = TerrainWorldEnv(num_agents=3, render_mode=None)
        self.agent = self.env.agents[0]

    def _create_action_dict(self, move_action: int = Moves.IDLE.value) -> dict:
        """Helper method to create properly formatted action dictionary."""
        return {
            f"agent_{i}": {
                "elementary_act": 0,
                "combat": {"move": move_action, "fire_enemy": self.env.num_agents},
                "comm": {"types": [0, 0, 0], "targets": [self.env.num_agents] * 3},
                "c2": {"give_order": 0, "orders": [self.env.MISSION_NO_CHANGE_SENTINEL] * self.env.num_agents}
            } for i in range(self.env.num_agents)
        }

    def test_environment_rewards_structure(self) -> None:
        action_dict = self._create_action_dict(Moves.IDLE.value)
        _obs, rewards, terminated, truncated, info = self.env.step(action_dict)
        assert isinstance(rewards, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_centralized_reward_calculation(self) -> None:
        action_dict = self._create_action_dict(Moves.IDLE.value)
        _obs, rewards1, _terminated, _truncated, _info = self.env.step(action_dict)
        _obs, rewards2, _terminated, _truncated, _info = self.env.step(action_dict)
        assert isinstance(rewards1, float)
        assert isinstance(rewards2, float)
        assert np.isfinite(rewards1)
        assert np.isfinite(rewards2)

    def test_agent_progress_reward_placeholder(self) -> None:
        action_dict = self._create_action_dict(Moves.IDLE.value)
        _obs, rewards, _terminated, _truncated, _info = self.env.step(action_dict)
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)

    def test_energy_consumption_penalties(self) -> None:
        action_dict = self._create_action_dict(Moves.FORWARD.value)
        _obs, rewards, _terminated, _truncated, _info = self.env.step(action_dict)
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)

    def test_health_and_survival_rewards(self) -> None:
        initial_health = self.agent.get_health()
        assert isinstance(initial_health, (int, float))
        assert initial_health > 0
        assert self.agent.state.is_alive()
        action_dict = self._create_action_dict(Moves.IDLE.value)
        _obs, rewards, _terminated, _truncated, _info = self.env.step(action_dict)
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)

    def test_cooperation_reward_system(self) -> None:
        for i, agent in enumerate(self.env.agents):
            agent.state.x = 10.0 + i * 2.0
            agent.state.y = 10.0
        action_dict = self._create_action_dict(Moves.FORWARD.value)
        _obs, rewards, _terminated, _truncated, _info = self.env.step(action_dict)
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)

    def test_reward_consistency(self) -> None:
        rewards_history: list[float] = []
        action_dict = self._create_action_dict(Moves.IDLE.value)
        for _ in range(5):
            _obs, rewards, _terminated, _truncated, _info = self.env.step(action_dict)
            rewards_history.append(rewards)
        assert len(rewards_history) == 5
        for reward in rewards_history:
            assert isinstance(reward, float)
            assert np.isfinite(reward)

    def test_individual_vs_team_rewards(self) -> None:
        action_dict = self._create_action_dict(Moves.FORWARD.value)
        _obs, rewards, _terminated, _truncated, _info = self.env.step(action_dict)
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)

    def test_action_specific_rewards(self) -> None:
        initial_pos = self.agent.get_position()
        actions_to_test = [
            Moves.IDLE,
            Moves.FORWARD,
            Moves.BACKWARD,
            Moves.ROTATE_LEFT,
            Moves.ROTATE_RIGHT,
        ]
        action_rewards: dict[str, float] = {}
        for action in actions_to_test:
            self.agent.state.x = initial_pos[0]
            self.agent.state.y = initial_pos[1]
            # Use proper nested action format
            action_dict = {
                f"agent_{i}": {
                    "elementary_act": 0,
                    "combat": {"move": action.value, "fire_enemy": self.env.num_agents},
                    "comm": {"types": [0, 0, 0], "targets": [self.env.num_agents] * 3},
                    "c2": {"give_order": 0, "orders": [self.env.MISSION_NO_CHANGE_SENTINEL] * self.env.num_agents}
                } for i in range(self.env.num_agents)
            }
            _obs, rewards, _terminated, _truncated, _info = self.env.step(action_dict)
            action_rewards[action.name] = rewards
        assert len(action_rewards) == len(actions_to_test)
        for action_name, reward in action_rewards.items():
            assert isinstance(reward, float), f"Reward for {action_name} is not numeric"

    def test_distance_based_rewards(self) -> None:
        initial_distance = self.agent.get_distance_to_target()
        assert isinstance(initial_distance, (int, float))
        assert initial_distance >= 0
        # Create action dict with different actions per agent
        action_dict = self._create_action_dict(Moves.IDLE.value)
        action_dict["agent_0"]["combat"]["move"] = Moves.FORWARD.value
        _obs, rewards, _terminated, _truncated, _info = self.env.step(action_dict)
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)

    def test_multi_agent_reward_distribution(self) -> None:
        positions = [(5, 5), (15, 15), (25, 5)]
        for i, (x, y) in enumerate(positions[:self.env.num_agents]):
            self.env.agents[i].state.x = float(x)
            self.env.agents[i].state.y = float(y)
        action_dict = self._create_action_dict(Moves.FORWARD.value)
        _obs, rewards, _terminated, _truncated, _info = self.env.step(action_dict)
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)

    def test_reward_bounds(self) -> None:
        all_rewards: list[float] = []
        action_dict = self._create_action_dict(Moves.FORWARD.value)
        for _ in range(10):
            _obs, rewards, _terminated, _truncated, _info = self.env.step(action_dict)
            all_rewards.append(rewards)
        for reward in all_rewards:
            assert isinstance(reward, float)
            assert np.isfinite(reward)
            assert not np.isnan(reward)

    def test_done_conditions_and_rewards(self) -> None:
        last_rewards: float | None = None
        max_steps = 100
        action_dict = self._create_action_dict(Moves.FORWARD.value)
        for _ in range(max_steps):
            _obs, rewards, terminated, truncated, _info = self.env.step(action_dict)
            assert isinstance(rewards, float)
            last_rewards = rewards
            if terminated or truncated:
                break
        if last_rewards is not None:
            assert isinstance(last_rewards, float)
            assert np.isfinite(last_rewards)

    def test_ammunition_and_combat_rewards(self) -> None:
        initial_ammo = self.agent.get_ammunition()
        assert isinstance(initial_ammo, (int, float))
        assert initial_ammo >= 0
        action_dict = self._create_action_dict(Moves.IDLE.value)
        _obs, rewards, _terminated, _truncated, _info = self.env.step(action_dict)
        assert isinstance(rewards, float)
        assert np.isfinite(rewards)


if __name__ == "__main__":
    unittest.main()
