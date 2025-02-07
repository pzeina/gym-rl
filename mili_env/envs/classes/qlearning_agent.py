from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from mili_env.envs.classes.model import Linear_QNet, QTrainer
from mili_env.envs.visualization import GradientLossVisualization  # noqa: TC001


@dataclass
class AgentConfig:
    """Data class to store agent's configuration parameters."""

    learning_rate: float
    initial_epsilon: float
    epsilon_decay: float
    final_epsilon: float
    discount_factor: float = 0.95
    memory_size: int = 100_000
    batch_size: int = 64
    hidden_size: int = 256


class QLearningAgent:
    """Reinforcement Learning agent using Q-learning with a neural network model."""
    def __init__(
            self,
            env: gym.Env,
            config: AgentConfig,
            visualization: GradientLossVisualization | None = None
    ) -> None:
        """Initialize a Reinforcement Learning agent with a neural network model."""
        self.env = env
        self.config = config

        position_shape = env.observation_space["position"].shape[0] # type: ignore # noqa: PGH003
        direction_shape = env.observation_space["direction"].shape[0] # type: ignore # noqa: PGH003
        target_shape = env.observation_space["target"].shape[0] # type: ignore # noqa: PGH003

        if isinstance(env.action_space, gym.spaces.Discrete):
            action_space_size = env.action_space.n
        else:
            msg = "Action space must be of type gym.spaces.Discrete"
            raise TypeError(msg)

        self.model = Linear_QNet(
            position_shape + direction_shape + target_shape,
            config.hidden_size,
            int(action_space_size)
        )

        self.trainer = QTrainer(self.model, config.learning_rate, config.discount_factor, visualization)
        self.memory = deque(maxlen=config.memory_size)
        self.epsilon = config.initial_epsilon
        self.epsilon_decay = config.epsilon_decay
        self.final_epsilon = config.final_epsilon

    def get_action(self, state: np.ndarray) -> int:
        """Choose an action based on the state using an epsilon-greedy policy."""
        rng = np.random.default_rng()
        if rng.random() < self.epsilon:
            return self.env.action_space.sample()
        state_tensor = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            action = torch.argmax(self.model(state_tensor)).item()
        return int(action)

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        *,
        done: bool = False
    ) -> None:
        """Store the experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        """Train the model using a batch of experiences from memory."""
        if len(self.memory) < self.config.batch_size:
            mini_batch = random.sample(self.memory, len(self.memory))
        else:
            mini_batch = random.sample(self.memory, self.config.batch_size)

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            *,
            done: bool = False
        ) -> None:
        """Train the model using a single experience."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def decay_epsilon(self) -> None:
        """Decay the epsilon value for the epsilon-greedy policy."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_model(self, file_name: str = "model.pth") -> None:
        """Save the model to a file."""
        self.model.save(file_name)

    def load_model(self, file_name: str = "model.pth") -> None:
        """Load the model from a file."""
        self.model.load(file_name)
