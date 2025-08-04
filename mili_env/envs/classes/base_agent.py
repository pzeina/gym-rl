from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv

if TYPE_CHECKING:
    import numpy as np


class BaseAgent:
    """Base class for agents with common functionality."""

    def __init__(self, env: gym.Env | VectorEnv) -> None:
        """Initialize the base agent class."""
        self.env = env
        self._validate_observation_space(env)

    def _register_action(self, action: np.ndarray) -> None:
        """Register the action taken by the agent."""
        self.action = action

    def _validate_observation_space(self, env: gym.Env | VectorEnv) -> None:
        """Validate the observation space of the environment."""
        if not isinstance(env.observation_space, gym.spaces.Dict):
            msg = "Observation space must be of type gym.spaces.Dict"
            raise TypeError(msg)

    def _calculate_observation_size(self, env: gym.Env | VectorEnv) -> int:  # noqa: C901
        """Calculate the size of the observation space."""
        obs_size = 0
        if isinstance(env, VectorEnv):

            def get_obs_space(env: gym.Env | VectorEnv) -> gym.spaces.Space:
                """Recursively get the observation space."""
                if isinstance(env, SyncVectorEnv):
                    return env.envs[0].observation_space
                if isinstance(env, AsyncVectorEnv):
                    return env.get_attr("observation_space")[0]
                if hasattr(env, "env"):
                    return get_obs_space(env.env)  # type: ignore # noqa: PGH003
                return env.observation_space

            obs_space = get_obs_space(env)
            for key in obs_space:  # type: ignore # noqa: PGH003
                if hasattr(obs_space[key], "shape"):  # type: ignore # noqa: PGH003
                    if len(obs_space[key].shape) > 0:  # type: ignore # noqa: PGH003
                        obs_size += obs_space[key].shape[0]  # type: ignore # noqa: PGH003
                    else:
                        obs_size += 1
        else:
            for key in env.observation_space:  # type: ignore # noqa: PGH003
                if hasattr(env.observation_space[key], "shape"):  # type: ignore # noqa: PGH003
                    if len(env.observation_space[key].shape) > 0:  # type: ignore # noqa: PGH003
                        obs_size += env.observation_space[key].shape[0]  # type: ignore # noqa: PGH003
                    else:
                        obs_size += 1
        return obs_size

    def _get_action_space_size(self, env: gym.Env | VectorEnv) -> np.int64:
        """Get the size of the action space."""
        if isinstance(env.action_space, gym.spaces.Discrete):
            return env.action_space.n
        if isinstance(env, SyncVectorEnv) and isinstance(env.envs[0].action_space, gym.spaces.Discrete):
            return env.envs[0].action_space.n
        if isinstance(env, AsyncVectorEnv) and isinstance(env.get_attr("action_space")[0], gym.spaces.Discrete):
            return env.get_attr("action_space")[0].n
        if hasattr(env, "env"):
            return self._get_action_space_size(env.env)  # type: ignore  # noqa: PGH003
        msg = "Action space must be of type gym.spaces.Discrete"
        raise TypeError(msg)
