import gymnasium as gym
import numpy as np


class RewardNormalizeWrapper(gym.Wrapper):
    """Wrapper that scales rewards by a factor to keep them in a smaller range.

    This can help with learning stability in reinforcement learning algorithms.
    Compatible with vectorized environments using SyncVectEnv.
    """

    def __init__(self, env: gym.Env, scale_factor: float = 0.01, clip_range: tuple = (-1.0, 1.0)) -> None:
        """Initialize the reward scaling wrapper.

        Args:
            env (gym.Env): The environment to wrap (supports vectorized envs)
            scale_factor (float): Factor to scale rewards by
            clip_range (tuple): Range to clip rewards to, default (-1.0, 1.0)
        """
        super().__init__(env)
        self.scale_factor = scale_factor
        self.clip_min, self.clip_max = clip_range

    def step(self, action: np.ndarray) -> tuple:
        """Step the environment and return the scaled reward.

        Args:
            action: The action to take in the environment

        Returns:
            tuple: (observation, scaled_reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Convert to numpy array if it's not already (handles both scalar and array rewards)
        reward_array = np.asarray(reward)

        # Scale the reward
        scaled_reward = reward_array * self.scale_factor

        # Optional: clip to range
        scaled_reward = np.clip(scaled_reward, self.clip_min, self.clip_max)

        # Store the original reward in info
        if isinstance(info, dict):
            # Single environment case
            info["original_reward"] = reward
        elif isinstance(info, (list, tuple)):
            # Vectorized environment case
            for i, info_dict in enumerate(info):
                if isinstance(info_dict, dict):
                    if isinstance(reward, (list, tuple, np.ndarray)):
                        info_dict["original_reward"] = reward[i]
                    else:
                        info_dict["original_reward"] = reward

        return observation, scaled_reward, terminated, truncated, info
