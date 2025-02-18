from __future__ import annotations

import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from torch import nn, optim

from mili_env.envs.classes.agent_config import AgentConfig  # noqa: TC001
from mili_env.envs.classes.model import Linear_QNet
from mili_env.envs.visualization import GradientLossVisualization  # noqa: TC001


class QLearningAgent:
    """Reinforcement Learning agent using Q-learning with a neural network model."""

    class QTrainer:
        """Q-learning trainer class to train the Q-learning model."""

        def __init__(
            self,
            policy_model: nn.Module,
            target_model: nn.Module,
            config: AgentConfig,
            visualization: GradientLossVisualization | None = None,
        ) -> None:
            """Initialize the Q-learning trainer with a model, target model, learning rate, and discount factor."""
            self.lr: float = config.learning_rate
            self.policy_model: nn.Module = policy_model
            self.target_model: nn.Module = target_model
            self.optimizer = optim.Adam(policy_model.parameters(), lr=self.lr)
            self.criterion = nn.MSELoss()
            self.visualization: GradientLossVisualization | None = visualization
            self.config: AgentConfig = config

        def train_step(
            self,
            state,  # noqa: ANN001
            action,  # noqa: ANN001
            reward,  # noqa: ANN001
            next_state,  # noqa: ANN001
            done,  # noqa: ANN001
        ) -> None:
            """Train the model using a single experience."""
            device = next(self.policy_model.parameters()).device  # Get the device of the model
            state = torch.tensor(np.array(state), dtype=torch.float).to(device)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
            action = torch.tensor(np.array(action), dtype=torch.long).to(device)
            reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)
            done = torch.tensor(np.array(done), dtype=torch.bool).to(device)

            pred = self.policy_model(state)
            target = pred.clone()
            q_new = reward + self.config.discount_factor * torch.max(self.target_model(next_state), dim=1)[0] * (~done)
            target[range(len(action)), action] = q_new

            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()

            # Track gradients
            if self.visualization:
                self.visualization.track_gradients(self.policy_model)

            self.optimizer.step()

            # Track loss
            if self.visualization:
                self.visualization.track_loss(loss.item())

        def optimize(self, batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
            """Optimize the model using a batch of experiences."""
            states, actions, rewards, next_states, dones = batch

            # Subsampling
            if self.config.subsampling_fraction < 1.0:
                rng = np.random.default_rng()
                indices = rng.choice(len(states), int(len(states) * self.config.subsampling_fraction), replace=False)
                states = states[indices]
                actions = actions[indices]
                rewards = rewards[indices]
                next_states = next_states[indices]
                dones = dones[indices]

            for _ in range(self.config.optimization_steps):
                self.train_step(states, actions, rewards, next_states, dones)

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

        self.policy_model = Linear_QNet(
            obs_size, config.hidden_size, int(action_space_size), favored_actions=favored_actions
        )
        self.target_model = Linear_QNet(
            obs_size, config.hidden_size, int(action_space_size), favored_actions=favored_actions
        )
        self.target_model.load_state_dict(
            self.policy_model.state_dict()
        )  # Initialize target model with policy model weights

        self.trainer = self.QTrainer(self.policy_model, self.target_model, config, visualization)
        self.memory = deque(maxlen=config.memory_size)
        self.epsilon = config.initial_epsilon
        self.final_epsilon = config.final_epsilon
        self.decay_factor = config.decay_factor
        self.update_counter = 0  # Initialize update counter

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
                output = self.policy_model(state_tensor)
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

    def decay_epsilon(self) -> None:
        """Decay the epsilon value for the epsilon-greedy policy."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.decay_factor)

    def save_model(self, file_name: str = "policy_model.pth") -> None:
        """Save the model to a file."""
        self.policy_model.save(file_name)
        self.target_model.save(file_name.replace("policy", "target"))

    def load_model(self, file_name: str = "policy_model.pth") -> None:
        """Load the model from a file."""
        self.policy_model.load(file_name)
        self.target_model.load(file_name.replace("policy", "target"))

    def update_model(self) -> None:
        """Update the model based on the update frequency."""
        self.update_counter += 1
        self.train_long_memory()

        if self.update_counter % self.config.update_frequency == 0:
            self.target_model.load_state_dict(self.policy_model.state_dict())  # Update target model
            self.update_counter = 0
