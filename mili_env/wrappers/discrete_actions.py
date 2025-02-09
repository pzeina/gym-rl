import gymnasium as gym
from gymnasium.spaces import Discrete


class DiscreteActions(gym.ActionWrapper):
    """Action wrapper that converts discrete actions to continuous actions."""

    def __init__(self, env: gym.Env, disc_to_cont) -> None:  # noqa: ANN001
        """Initialize the discrete action wrapper."""
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))

    def action(self, action) -> int:  # noqa: ANN001, D102
        return self.disc_to_cont[action]
