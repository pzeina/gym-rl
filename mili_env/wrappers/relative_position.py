import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class RelativePosition(gym.ObservationWrapper):
    """Observation wrapper that returns the relative position of the target."""

    def __init__(self, env: gym.Env) -> None:
        """Initialize the observation wrapper."""
        super().__init__(env)
        self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf)

    def observation(self, observation) -> np.ndarray:  # noqa: ANN001
        """Return the relative position of the target."""
        return observation["target"] - observation["position"]
