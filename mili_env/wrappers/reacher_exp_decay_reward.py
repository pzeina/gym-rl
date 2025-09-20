import gymnasium as gym
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from mili_env.envs.terrain_world import TerrainWorldEnv


def process_obs(obs: dict) -> np.ndarray:
    """Process the observations into a 2D numpy array where each row is the state of an environment."""
    env_processed = [
        value.flatten() if isinstance(value, np.ndarray) else np.array(value).flatten() for key, value in obs.items()
    ]
    return np.concatenate(env_processed)


class ReacherRewardWrapper(gym.Wrapper):
    """Wrapper that tracks actions and assigns credit to past actions when target is reached."""

    def __init__(self, env: gym.Env, decay_factor: float = 0.9, history_length: int = 50) -> None:
        """Initialize the reward backpropagation wrapper.

        Args:
            env: The environment to wrap
            reward_dist_weight: Weight for the distance reward component
            reward_ctrl_weight: Weight for the control reward component
            decay_factor: Exponential decay factor (0-1), higher values decay slower
            history_length: Number of previous actions to consider for backpropagation
        """
        super().__init__(env)
        self.decay_factor = decay_factor
        self.history_length = history_length

        # Store state-action history and their corresponding rewards
        self.obs_history = []
        self.action_history = []
        self.reward_history = []
        self.terminated_history = []

        # Track environment steps
        self.step_counter = 0

    def step(self, action: int) -> tuple:
        """Step the environment, track actions, states, rewards, and handle backpropagation."""
        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Store experience for backpropagation
        self.obs_history.append(obs)
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.terminated_history.append(terminated)

        # Process target reached event if applicable
        if terminated:
            # Backpropagate the reward
            self._backpropagate_reward(np.asarray(reward))

        # Increment step counter
        self.step_counter += 1

        # Return modified reward
        return obs, reward, terminated, truncated, info

    def _backpropagate_reward(self, reward: np.ndarray, *, show_heatmap: bool = False) -> None:
        """Update backpropagated rewards and remember corresponding states."""
        current_step = self.step_counter

        # Calculate backpropagated rewards with exponential decay
        for past_step in range(len(self.obs_history)):
            # Calculate steps back from current step
            steps_back = current_step - past_step

            # Apply exponential decay based on how far back the action was
            if steps_back <= self.history_length:
                decay = self.decay_factor**steps_back

                # Add to agent's memory with updated reward
                obs = self.obs_history[past_step]
                action = self.action_history[past_step]
                self.reward_history[past_step] += float(reward) * decay
                terminated = self.terminated_history[past_step]
                if self.agent is not None:
                    state = process_obs(obs)
                    next_state = (
                        process_obs(self.obs_history[past_step + 1]) if past_step + 1 < len(self.obs_history) else state
                    )
                    self.agent.remember(state, action, reward, next_state, terminated)

        # Generate heatmap for debugging
        if show_heatmap:
            self._generate_heatmap()

    def _generate_heatmap(self) -> None:
        """Generate a heatmap of the environment based on the backpropagated rewards."""
        # Access the underlying environment
        env = self.env.unwrapped
        if not isinstance(env, TerrainWorldEnv):
            return

        # Create a map of the environment
        env_map = np.zeros((env.height, env.width))

        # Colorize the target position in red
        target_position = self.obs_history[0]["target_position"]
        target_position = np.round(target_position).astype(int)
        env_map[target_position[1], target_position[0]] = 1.0  # Red color for the target

        # Colorize the tiles based on the reward values
        for i, state in enumerate(self.obs_history):
            position = state["position"]
            reward = self.reward_history[i]
            env_map[int(position[1]), int(position[0])] = reward

        # Normalize the rewards for visualization
        norm = mcolors.Normalize(vmin=float(np.min(env_map)), vmax=float(np.max(env_map)))

        # Clear the current figure
        plt.clf()

        # Create the heatmap
        plt.imshow(env_map, cmap="hot", norm=norm)
        plt.colorbar()
        plt.title("Heatmap of Backpropagated Rewards")
        plt.pause(0.001)  # Use plt.pause to avoid blocking the script

    def reset(self, **kwargs) -> tuple:  # noqa: ANN003
        """Reset the environment and clear history."""
        # Clear history for the new episode
        self.obs_history = []
        self.action_history = []
        self.reward_history = []
        self.terminated_history = []
        self.step_counter = 0

        # Reset the environment
        return self.env.reset(**kwargs)
