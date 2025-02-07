import gymnasium as gym


class ReacherRewardWrapper(gym.Wrapper):
    """Reward wrapper that returns a weighted reward for the reacher environment."""
    def __init__(self, env: gym.Env, reward_dist_weight, reward_ctrl_weight) -> None: # noqa: ANN001
        """Initialize the reward wrapper."""
        super().__init__(env)
        self.reward_dist_weight = reward_dist_weight
        self.reward_ctrl_weight = reward_ctrl_weight

    def step(self, action) -> tuple: # noqa: ANN001
        """Step the environment and return the weighted reward."""
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.reward_dist_weight * info["reward_dist"] + self.reward_ctrl_weight * info["reward_ctrl"]
        return obs, reward, terminated, truncated, info
