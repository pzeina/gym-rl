from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mili_env.envs.classes.base_agent import BaseAgent
from mili_env.envs.classes.robot_base import Actions

if TYPE_CHECKING:
    import gymnasium as gym
    from gymnasium.vector import SyncVectorEnv


PI_OVER_8 = np.pi / 8


def get_action(state: np.ndarray) -> np.ndarray:
    """Get the action for the current state.

    1) Rotate until the target is in front, in the shortest direction
    2) Move forward
    """
    # Target direction is state[:, 5]
    # Robot angle is state[:, 0]
    diff_angle = (state[:, 0] - state[:, 5] + 2 * np.pi) % (2 * np.pi)

    # Where diff angle is less than pi, turn right, else turn left
    target_is_at_left = (diff_angle + PI_OVER_8 < np.pi) & (diff_angle > PI_OVER_8)
    # Note that target_is_at_right = (diff_angle - PI_OVER_8 > np.pi) & (diff_angle < 2 * np.pi - PI_OVER_8)
    target_is_in_front = (diff_angle <= PI_OVER_8) | (diff_angle >= 2 * np.pi - PI_OVER_8)

    return np.where(
        target_is_at_left,
        Actions.ROTATE_LEFT.value,
        np.where(target_is_in_front, Actions.FORWARD.value, Actions.ROTATE_RIGHT.value),
    )


class DummyAgent(BaseAgent):
    """Dummy agent using the shortest path to target (shortest distance)."""

    def __init__(self, env: gym.Env | SyncVectorEnv) -> None:
        """Initialize a Reinforcement Learning agent with a neural network model."""
        super().__init__(env)

    def get_action(self, states: np.ndarray) -> np.ndarray:
        """Get the action for the current state.

        1) Rotate until the target is in front, in the shortest direction
        2) Move forward
        """
        return get_action(states)
