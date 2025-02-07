from dataclasses import dataclass
from mili_env.envs.classes.model import Linear_QNet, QTrainer
import numpy as np
import gymnasium as gym
from collections import deque
import random
import torch

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
    def __init__(self, env: gym.Env, config: AgentConfig, visualization=None) -> None:
        """Initialize a Reinforcement Learning agent with a neural network model."""
        self.env = env
        self.config = config
        self.model = Linear_QNet(env.observation_space['position'].shape[0] + env.observation_space['direction'].shape[0] + env.observation_space['target'].shape[0], config.hidden_size, env.action_space.n)

        self.trainer = QTrainer(self.model, config.learning_rate, config.discount_factor, visualization)
        self.memory = deque(maxlen=config.memory_size)
        self.epsilon = config.initial_epsilon
        self.epsilon_decay = config.epsilon_decay
        self.final_epsilon = config.final_epsilon

    def get_action(self, state: np.ndarray) -> int:
        """Choose an action based on the state using an epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                return torch.argmax(self.model(state)).item()

    def remember(self, state, action, reward, next_state, done) -> None:
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

    def train_short_memory(self, state, action, reward, next_state, done) -> None:
        """Train the model using a single experience."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def decay_epsilon(self) -> None:
        """Decay the epsilon value for the epsilon-greedy policy."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_model(self, file_name='model.pth') -> None:
        """Save the model to a file."""
        self.model.save(file_name)

    def load_model(self, file_name='model.pth') -> None:
        """Load the model from a file."""
        self.model.load(file_name)