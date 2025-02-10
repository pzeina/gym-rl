from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv

from mili_env.envs.classes.model import Linear_QNet, QTrainer
from mili_env.envs.visualization import GradientLossVisualization  # noqa: TC001


@dataclass
class AgentConfig:
    """Data class to store agent's configuration parameters."""

    learning_rate: float
    initial_epsilon: float
    final_epsilon: float
    discount_factor: float = 0.95
    memory_size: int = 100_000
    batch_size: int = 8  # 64
    batch_num: int = 1  # 10
    hidden_size: int = 256
    decay_factor: float = 0.995  # Add decay factor for exponential decay


class QLearningAgent:
    """Reinforcement Learning agent using Q-learning with a neural network model."""

    def __init__(
        self,
        env: gym.Env | SyncVectorEnv,
        config: AgentConfig,
        favored_actions: list[int] | None = None,
        visualization: GradientLossVisualization | None = None,
    ) -> None:
        """Initialize a Reinforcement Learning agent with a neural network model."""
        self.env = env
        self.config = config

        self._validate_observation_space(env)
        obs_size = self._calculate_observation_size(env)
        action_space_size = self._get_action_space_size(env)

        self.model = Linear_QNet(obs_size, config.hidden_size, int(action_space_size), favored_actions=favored_actions)

        self.trainer = QTrainer(self.model, config.learning_rate, config.discount_factor, visualization)
        self.memory = deque(maxlen=config.memory_size)
        self.epsilon = config.initial_epsilon
        self.final_epsilon = config.final_epsilon
        self.decay_factor = config.decay_factor

    def _validate_observation_space(self, env: gym.Env | SyncVectorEnv) -> None:
        """Validate the observation space of the environment."""
        if not isinstance(env.observation_space, gym.spaces.Dict):
            msg = "Observation space must be of type gym.spaces.Dict"
            raise TypeError(msg)

    def _calculate_observation_size(self, env: gym.Env | SyncVectorEnv) -> int:
        """Calculate the size of the observation space."""
        obs_size = 0
        if isinstance(env, SyncVectorEnv):
            obs_space: gym.spaces.Space = env.envs[0].observation_space
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

    def _get_action_space_size(self, env: gym.Env | SyncVectorEnv) -> np.int64:
        """Get the size of the action space."""
        if isinstance(env.action_space, gym.spaces.Discrete):
            return env.action_space.n
        if isinstance(env, SyncVectorEnv) and isinstance(env.envs[0].action_space, gym.spaces.Discrete):
            return env.envs[0].action_space.n
        msg = "Action space must be of type gym.spaces.Discrete"
        raise TypeError(msg)

    def get_action(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Choose actions based on the states using an epsilon-greedy policy."""
        rng = np.random.default_rng()

        if rng.random() < self.epsilon:
            actions = self.env.action_space.sample()
            random_picks = np.array([True] * states.shape[0])
        else:
            state_tensor = torch.tensor(states, dtype=torch.float)
            with torch.no_grad():
                output = self.model(state_tensor)
                actions = torch.argmax(output, dim=1).numpy()
                random_picks = np.array([False] * states.shape[0])

        return np.array(actions), np.array(random_picks)

    def remember(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray
    ) -> None:
        """Store the experiences in memory."""
        self.memory.append((states, actions, rewards, next_states, dones))

    def train_long_memory(self) -> None:
        """Train the model using a batch of experiences from memory."""
        if len(self.memory) < self.config.batch_size * self.config.batch_num:
            return
        # Train on multiple batches to improve learning
        for _ in range(self.config.batch_num):  # Train on multiple batches
            mini_batch = random.sample(self.memory, self.config.batch_size)

            states, actions, rewards, next_states, dones = zip(*mini_batch)
            states = np.concatenate(states)
            next_states = np.concatenate(next_states)
            actions = np.concatenate(actions)
            rewards = np.concatenate(rewards)
            dones = np.concatenate(dones)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Train the model using a batch of experiences."""
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def decay_epsilon(self) -> None:
        """Decay the epsilon value for the epsilon-greedy policy."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.decay_factor)

    def save_model(self, file_name: str = "model.pth") -> None:
        """Save the model to a file."""
        self.model.save(file_name)

    def load_model(self, file_name: str = "model.pth") -> None:
        """Load the model from a file."""
        self.model.load(file_name)
