import gymnasium as gym
import numpy as np


class ClipReward(gym.RewardWrapper):
    """Reward wrapper that clips the reward to a specified range."""
    def __init__(self, env, min_reward, max_reward) -> None: # noqa: ANN001
        """Initialize the reward wrapper."""
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward) -> float: # noqa: ANN001, D102
        return np.clip(reward, self.min_reward, self.max_reward) # type: ignore # noqa: PGH003
