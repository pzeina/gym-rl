from __future__ import annotations

import random
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn, optim

from mili_env.envs.classes.agent_config import AgentConfig  # noqa: TC001
from mili_env.envs.classes.base_agent import BaseAgent
from mili_env.envs.classes.dummy_agent import get_action as dummy_get_action
from mili_env.envs.classes.model import Linear_QNet
from mili_env.envs.metrics import GradientLossTracker  # noqa: TC001

if TYPE_CHECKING:
    import gymnasium as gym
    from gymnasium.vector import SyncVectorEnv


class QLearningAgent(BaseAgent):
    """Reinforcement Learning agent using Q-learning with a neural network model."""

    class QTrainer:
        """Q-learning trainer class to train the Q-learning model."""

        def __init__(
            self,
            policy_model: nn.Module,
            target_model: nn.Module,
            config: AgentConfig,
            visualization: GradientLossTracker | None = None,
        ) -> None:
            """Initialize the Q-learning trainer with a model, target model, learning rate, and discount factor."""
            self.lr: float = config.learning_rate
            self.policy_model: nn.Module = policy_model
            self.target_model: nn.Module = target_model
            self.optimizer = optim.Adam(policy_model.parameters(), lr=self.lr)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.decay_lr)
            self.criterion = nn.MSELoss()
            self.visualization: GradientLossTracker | None = visualization
            self.config: AgentConfig = config

            self.grad_value: float = 0.0
            self.loss_value: float = 0.0

            self.policy_model.to(self.config.device)
            self.target_model.to(self.config.device)

        def train_step(
            self,
            _state: np.ndarray,
            _action: np.ndarray,
            _reward: np.ndarray,
            _next_state: np.ndarray,
            _done: np.ndarray,
        ) -> None:
            """Train the model using a single experience."""
            device = next(self.policy_model.parameters()).device  # Get the device of the model
            state = torch.tensor(np.array(_state), dtype=torch.float).to(device)
            next_state = torch.tensor(np.array(_next_state), dtype=torch.float).to(device)
            action = torch.tensor(np.array(_action), dtype=torch.long).to(device)
            reward = torch.tensor(np.array(_reward), dtype=torch.float).to(device)
            done = torch.tensor(np.array(_done), dtype=torch.bool).to(device)

            pred = self.policy_model(state)
            target = pred.clone()
            with torch.no_grad():
                q_next = self.target_model(next_state)
                q_target = reward + self.config.discount_factor * torch.max(q_next, dim=1)[0] * (~done)
            target[range(len(action)), action] = (
                q_target  # Updating the Target Q-values only for the specific actions taken.
                # This ensures that the loss is computed only for the actions that were actually taken,
                #  and the Q-values for other actions remain unchanged.
            )

            self.optimizer.zero_grad()
            loss = self.criterion(pred, target)
            loss.backward()

            # Track metrics
            if self.visualization:
                self.grad_value = self.visualization.track_gradients(self.policy_model)
                self.loss_value = self.visualization.track_loss(loss.item())
                self.visualization.show()

            self.optimizer.step()

        def optimize(self, batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
            """Optimize the model using a batch of experiences."""
            states, actions, rewards, next_states, dones = batch

            # Subsampling
            if self.config.subsampling_fraction < 1.0:
                rng = np.random.default_rng()
                indices = rng.choice(len(states), int(len(states) * self.config.subsampling_fraction), replace=False)
                states: np.ndarray = states[indices]
                actions: np.ndarray = actions[indices]
                rewards: np.ndarray = rewards[indices]
                next_states: np.ndarray = next_states[indices]
                dones: np.ndarray = dones[indices]

            for _ in range(self.config.optimization_steps):
                self.train_step(states, actions, rewards, next_states, dones)

            # Step the scheduler after each optimization step
            self.scheduler.step()

    def __init__(
        self,
        env: gym.Env | SyncVectorEnv,
        config: AgentConfig,
        visualization: GradientLossTracker | None = None,
    ) -> None:
        """Initialize a Reinforcement Learning agent with a neural network model."""
        super().__init__(env)

        self.config = config
        obs_size = self._calculate_observation_size(env)
        action_space_size = self._get_action_space_size(env)

        self.policy_model = Linear_QNet(obs_size, config.hidden_size, int(action_space_size))
        self.target_model = Linear_QNet(obs_size, config.hidden_size, int(action_space_size))
        self.target_model.load_state_dict(
            self.policy_model.state_dict()
        )  # Initialize target model with policy model weights

        self.trainer = self.QTrainer(self.policy_model, self.target_model, config, visualization)
        self.memory = deque(maxlen=config.memory_size)
        self.epsilon = config.initial_epsilon
        self.final_epsilon = config.final_epsilon
        self.decay_epsilon = config.decay_epsilon
        self.dummy_frequency: float = 1.0
        self.update_counter = 0  # Initialize update counter

    def get_epsilon(self) -> float:
        """Get the epsilon value for the epsilon-greedy policy."""
        return self.epsilon

    def get_learning_rate(self) -> float:
        """Get the learning rate of the optimizer."""
        return self.trainer.lr

    def get_grad_loss_values(self) -> tuple[float, float]:
        """Get the gradient value."""
        return self.trainer.grad_value, self.trainer.loss_value

    def get_action(self, states: np.ndarray) -> np.ndarray:
        """Choose actions based on the states using an epsilon-greedy policy."""
        if self.config.dummy_phase > self.update_counter:
            return dummy_get_action(states)

        rng = np.random.default_rng()

        alea = rng.random()
        if alea < self.dummy_frequency:
            return dummy_get_action(states)

        alea = rng.random()
        if alea < self.epsilon:
            actions = self.env.action_space.sample()
        else:
            state_tensor = torch.tensor(states, dtype=torch.float).to(self.config.device)
            with torch.no_grad():
                output = self.policy_model(state_tensor)
                actions = torch.argmax(output, dim=1).cpu().numpy()  # Move to CPU before converting to numpy

        return np.array(actions)

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
            self.trainer.optimize((states, actions, rewards, next_states, dones))

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

    def decay_epsilon_f(self) -> None:
        """Decay the epsilon value for the epsilon-greedy policy."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.decay_epsilon)
        self.dummy_frequency *= self.config.dummy_policy_decay

    def save_model(self, file_name: str = "policy_model.pth") -> None:
        """Save the model to a file."""
        self.policy_model.save(file_name)
        self.target_model.save(file_name.replace("policy", "target"))

    def load_model(self, file_name: str = "policy_model.pth") -> None:
        """Load the model from a file."""
        self.policy_model.load(file_name)
        self.target_model.load(file_name.replace("policy", "target"))

        # Ensure the model is moved to the correct device after loading
        self.policy_model.to(self.config.device)
        self.target_model.to(self.config.device)

    def update_model(self) -> None:
        """Update the model based on the update frequency."""
        self.update_counter += 1
        self.train_long_memory()

        if self.update_counter % self.config.update_frequency == 0:
            self.target_model.load_state_dict(self.policy_model.state_dict())  # Update target model
