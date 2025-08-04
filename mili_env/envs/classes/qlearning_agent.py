from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn, optim

from mili_env.envs.classes.agent_config import AgentConfig  # noqa: TC001
from mili_env.envs.classes.base_agent import BaseAgent
from mili_env.envs.classes.dummy_agent import get_action as dummy_get_action
from mili_env.envs.classes.logger import Logger
from mili_env.envs.classes.model import Linear_QNet
from mili_env.envs.metrics import GradientLossTracker  # noqa: TC001

if TYPE_CHECKING:
    import gymnasium as gym
    from gymnasium.vector import VectorEnv


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
            logger_mode: str = "wp",
        ) -> None:
            """Initializes the Q-learning agent with the given models, configuration, visualization, logging options.

            Args:
                policy_model (nn.Module): The neural network model used for policy estimation (Q-network).
                target_model (nn.Module): The target neural network model for stable Q-learning updates.
                config (AgentConfig): Configuration object containing agent and training hyperparameters.
                visualization (GradientLossTracker | None, optional): Optional visualization tool for tracking gradients
                and loss.
                  Defaults to None.
                logger_mode (str, optional): Mode for the logger (e.g., write/append). Defaults to "wp".

            Initializes optimizer, learning rate scheduler, loss criterion, logging, and gradient tracking.
            Moves models to the specified device.
            """
            self.policy_model: nn.Module = policy_model
            self.target_model: nn.Module = target_model
            self.optimizer = optim.Adam(policy_model.parameters(), lr=config.learning_rate)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.decay_lr)
            self.criterion = nn.MSELoss()
            self.visualization: GradientLossTracker | None = visualization
            self.config: AgentConfig = config

            log_file_path = config.trainer_log_file
            self.logger: Logger | None = (
                None
                if config.disable_logs
                else Logger(
                    log_file_path,
                    keys=["Loss", "AvgGrad", "State", "Action", "Reward", "NextState", "Done", "QPred", "QTarget"],
                    mode=logger_mode,
                    graphs=[
                        {
                            "x": None,
                            "y": ["QPred"],
                            "vectorized": True,
                            "split": True,
                            "split_labels": ["Idle", "Forward", "Backward", "Left", "Right"],
                            "mark_occurrence": ["Done"],
                        },
                        {"x": None, "y": ["QTarget", "Reward"], "vectorized": True, "mark_occurrence": ["Done"]},
                        # {
                        #     "x": None,
                        #     "y": ["State"],
                        #     "vectorized": True,
                        #     "split": True,
                        #     "split_labels": [
                        #         "Pos-X",
                        #         "Pos-Y",
                        #         "Target-X",
                        #         "Target-Y",
                        #         "Distance",
                        #         "Direction",
                        #         "Target Angle",
                        #         "Energy",
                        #     ],
                        #     "mark_occurrence": ["Done"],
                        # },
                        {"x": None, "y": ["Action"], "vectorized": True, "mark_occurrence": ["Done"]},
                        {"x": None, "y": ["AvgGrad", "Loss"], "vectorized": False},
                    ],
                    static_keys={"Type": "QL-Agent"},
                )
            )

            self.grad_value: float = 0.0
            self.loss_value: float = 0.0

            self.policy_model.to(self.config.device)
            self.target_model.to(self.config.device)

            # Initialize gradient tracking
            self.gradients = deque(maxlen=config.grad_clip_window)
            self.default_max_grad = torch.tensor(config.grad_clip_max, dtype=torch.float, device=self.config.device)

        def train_step(
            self,
            _state: np.ndarray,
            _action: np.ndarray,
            _reward: np.ndarray,
            _next_state: np.ndarray,
            _done: np.ndarray,
            *,
            from_memory: bool = False,
        ) -> dict:
            """Train the model using a single experience."""
            device = next(self.policy_model.parameters()).device  # Get the device of the model
            state = torch.tensor(np.array(_state), dtype=torch.float).to(device)
            next_state = torch.tensor(np.array(_next_state), dtype=torch.float).to(device)
            reward = torch.tensor(np.array(_reward), dtype=torch.float).to(device)
            done = torch.tensor(np.array(_done), dtype=torch.bool).to(device)

            pred = self.policy_model(state)
            target = pred.clone().detach()
            with torch.no_grad():
                q_next = self.target_model(next_state)
                batched = q_next.ndim > 2  # noqa: PLR2004
                if batched:
                    sample_nb_total = q_next.shape[0]
                    q_target = torch.zeros(sample_nb_total, q_next.shape[1], device=device)
                    for sample_idx in range(sample_nb_total):
                        max_q = q_next[sample_idx].max(dim=-1)
                        done_sample = done[sample_idx].float()
                        q_target[sample_idx] = reward[sample_idx] + self.config.discount_factor * max_q.values * (
                            1 - done_sample
                        )

                        # Updating the Target Q-values only for the specific actions taken.
                        # This ensures that the loss is computed only for the actions that were actually taken,
                        #  and the Q-values for other actions remain unchanged.
                        # Update only the Q-value for the action taken
                        target[sample_idx, torch.arange(target.shape[1]), _action[sample_idx]] = q_target[sample_idx]
                else:
                    max_q = q_next.max(dim=1)
                    max_q = q_next.max(dim=1)
                    done = done.float()
                    q_target = reward + self.config.discount_factor * max_q.values * (1 - done)

                    # Updating the Target Q-values only for the specific actions taken.
                    # This ensures that the loss is computed only for the actions that were actually taken,
                    #  and the Q-values for other actions remain unchanged.
                    # Update only the Q-value for the action taken
                    target[torch.arange(target.shape[0]), _action] = q_target

            self.optimizer.zero_grad()
            loss = self.criterion(pred, target)
            loss.backward()

            # Track gradients
            self.gradients.append(self.grad_value)
            sliding_avg_grad = torch.tensor(np.mean(self.gradients), dtype=torch.float, device=self.config.device)
            max_grad = torch.max(self.default_max_grad, sliding_avg_grad).item()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=max_grad)

            # Track metrics
            if self.visualization:
                self.grad_value = self.visualization.track_gradients(self.policy_model)
                self.loss_value = self.visualization.track_loss(loss.item())
                self.visualization.show()

            # Get gradients for logging
            gradients = [
                param.grad.data.cpu().numpy() for param in self.policy_model.parameters() if param.grad is not None
            ]
            _avg_grad = float(np.mean([np.mean(np.abs(grad)) for grad in gradients]))
            _loss = loss.item()

            self.optimizer.step()

            if not from_memory:
                logs_dict = {
                    "Loss": _loss,
                    "AvgGrad": _avg_grad,
                    "State": _state,  # _state[i],
                    "Action": _action,  # _action[i],
                    "Reward": _reward,  # _reward[i],
                    "NextState": _next_state,  # _next_state[i],
                    "Done": _done,  # _done[i],
                    "QPred": pred.cpu().detach().numpy(),  # pred[i, _action[i]].item(),
                    "QTarget": q_target.cpu().detach().numpy(),  # q_target[i].item(), pred[i].cpu().detach().numpy(),
                }
                if self.logger:
                    self.logger.log_entry(logs_dict)

            return {"grad": _avg_grad, "loss": _loss}

        def optimize(self, batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
            """Optimize the model using a batch of experiences."""
            states, actions, rewards, next_states, dones = batch

            # Subsampling
            if self.config.subsampling_fraction < 1.0:
                rng = np.random.default_rng()
                subsample_indices = rng.choice(
                    len(states), int(len(states) * self.config.subsampling_fraction), replace=False
                )
                states = np.asarray([states[i] for i in subsample_indices])
                actions = np.asarray([actions[i] for i in subsample_indices])
                rewards = np.asarray([rewards[i] for i in subsample_indices])
                next_states = np.asarray([next_states[i] for i in subsample_indices])
                dones = np.asarray([dones[i] for i in subsample_indices])

            for _ in range(self.config.optimization_steps):
                self.train_step(states, actions, rewards, next_states, dones, from_memory=True)

            # Step the scheduler after each optimization step
            self.scheduler.step()

    def __init__(
        self,
        env: gym.Env | VectorEnv,
        config: AgentConfig,
        visualization: GradientLossTracker | None = None,
        logger_mode: str = "wp",
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

        self.trainer = self.QTrainer(
            self.policy_model, self.target_model, config, visualization, logger_mode=logger_mode
        )
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
        return self.trainer.optimizer.param_groups[0]["lr"]

    def get_grad_loss_values(self) -> tuple[float, float]:
        """Get the gradient value."""
        return self.trainer.grad_value, self.trainer.loss_value

    def get_action(self, states: np.ndarray) -> np.ndarray:
        """Choose actions based on the states using an epsilon-greedy policy."""
        if self.update_counter % self.config.dummy_recurrence < self.config.dummy_phase:
            action = dummy_get_action(states)
            self._register_action(action)
            return action

        rng = np.random.default_rng()

        alea = rng.random()
        if alea < self.dummy_frequency:
            action = dummy_get_action(states)
            self._register_action(action)
            return action

        alea = rng.random()
        if alea < self.epsilon:
            actions = self.env.action_space.sample()
        else:
            state_tensor = torch.tensor(states, dtype=torch.float).to(self.config.device)
            with torch.no_grad():
                output = self.policy_model(state_tensor)
                actions = torch.argmax(output, dim=1).cpu().numpy()  # Move to CPU before converting to numpy

        # Not used self._register_action(actions)
        return np.array(actions)

    def remember(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray
    ) -> None:
        """Store the experiences in memory."""
        self.memory.append((states, actions, rewards, next_states, dones))

    def train_long_memory(self) -> None:
        """Train the model using a batch of experiences from memory."""
        batch_num = min(self.config.batch_num, len(self.memory) // self.config.batch_size)
        # Train on multiple batches to improve learning
        for _ in range(batch_num):  # Train on multiple batches
            mini_batch = random.sample(self.memory, self.config.batch_size)

            states, actions, rewards, next_states, dones = map(np.array, zip(*mini_batch))
            self.trainer.optimize((states, actions, rewards, next_states, dones))

    def train_short_memory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> dict:
        """Train the model using a batch of experiences."""
        return self.trainer.train_step(states, actions, rewards, next_states, dones)

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

    def save_checkpoint(self, checkpoint_path: str = "checkpoint.pth") -> None:
        """Save a complete training checkpoint including models, optimizer state, and training parameters.

        Args:
            checkpoint_path (str): Path to save the checkpoint file. Defaults to "checkpoint.pth".
        """
        # Ensure directory exists
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "policy_model_state_dict": self.policy_model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "scheduler_state_dict": self.trainer.scheduler.state_dict(),
            "memory": list(self.memory),  # Convert deque to list for serialization
            "epsilon": self.epsilon,
            "final_epsilon": self.final_epsilon,
            "decay_epsilon": self.decay_epsilon,
            "dummy_frequency": self.dummy_frequency,
            "update_counter": self.update_counter,
            "gradients": list(self.trainer.gradients),  # Convert deque to list for serialization
            "grad_value": self.trainer.grad_value,
            "loss_value": self.trainer.loss_value,
            "config": self.config,  # Save the configuration as well
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")  # noqa: T201

    def save_checkpoint_silent(self, checkpoint_path: str = "checkpoint.pth") -> None:
        """Save a complete training checkpoint silently without output messages.

        Args:
            checkpoint_path (str): Path to save the checkpoint file. Defaults to "checkpoint.pth".
        """
        # Ensure directory exists
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "policy_model_state_dict": self.policy_model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "scheduler_state_dict": self.trainer.scheduler.state_dict(),
            "memory": list(self.memory),  # Convert deque to list for serialization
            "epsilon": self.epsilon,
            "final_epsilon": self.final_epsilon,
            "decay_epsilon": self.decay_epsilon,
            "dummy_frequency": self.dummy_frequency,
            "update_counter": self.update_counter,
            "gradients": list(self.trainer.gradients),  # Convert deque to list for serialization
            "grad_value": self.trainer.grad_value,
            "loss_value": self.trainer.loss_value,
            "config": self.config,  # Save the configuration as well
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str = "checkpoint.pth") -> None:
        """Load a complete training checkpoint to restore training state.

        Args:
            checkpoint_path (str): Path to the checkpoint file to load. Defaults to "checkpoint.pth".
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        # Restore model states
        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])

        # Restore optimizer and scheduler states
        self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training parameters
        self.memory = deque(checkpoint["memory"], maxlen=self.config.memory_size)
        self.epsilon = checkpoint["epsilon"]
        self.final_epsilon = checkpoint["final_epsilon"]
        self.decay_epsilon = checkpoint["decay_epsilon"]
        self.dummy_frequency = checkpoint["dummy_frequency"]
        self.update_counter = checkpoint["update_counter"]

        # Restore trainer state
        self.trainer.gradients = deque(checkpoint["gradients"], maxlen=self.config.grad_clip_window)
        self.trainer.grad_value = checkpoint["grad_value"]
        self.trainer.loss_value = checkpoint["loss_value"]

        # Ensure models are on the correct device
        self.policy_model.to(self.config.device)
        self.target_model.to(self.config.device)

        print(f"Checkpoint loaded from {checkpoint_path}")  # noqa: T201
        print(f"Resumed training at update counter: {self.update_counter}")  # noqa: T201
        print(f"Current epsilon: {self.epsilon:.4f}")  # noqa: T201
        print(f"Memory size: {len(self.memory)}")  # noqa: T201

    def load_checkpoint_silent(self, checkpoint_path: str = "checkpoint.pth") -> None:
        """Load a complete training checkpoint silently without output messages.

        Args:
            checkpoint_path (str): Path to the checkpoint file to load. Defaults to "checkpoint.pth".
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        # Restore model states
        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])

        # Restore optimizer and scheduler states
        self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training parameters
        self.memory = deque(checkpoint["memory"], maxlen=self.config.memory_size)
        self.epsilon = checkpoint["epsilon"]
        self.final_epsilon = checkpoint["final_epsilon"]
        self.decay_epsilon = checkpoint["decay_epsilon"]
        self.dummy_frequency = checkpoint["dummy_frequency"]
        self.update_counter = checkpoint["update_counter"]

        # Restore trainer state
        self.trainer.gradients = deque(checkpoint["gradients"], maxlen=self.config.grad_clip_window)
        self.trainer.grad_value = checkpoint["grad_value"]
        self.trainer.loss_value = checkpoint["loss_value"]

        # Ensure models are on the correct device
        self.policy_model.to(self.config.device)
        self.target_model.to(self.config.device)

    def save_periodic_checkpoint(self, episode: int, checkpoint_dir: str = "checkpoints") -> None:
        """Save a periodic checkpoint with episode information.

        Args:
            episode (int): Current episode number
            checkpoint_dir (str): Directory to save checkpoints. Defaults to "checkpoints".
        """
        # Create checkpoint directory if it doesn't exist
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_episode_{episode + 1}.pth"
        self.save_checkpoint(str(checkpoint_path))

        # Also save as latest checkpoint for easy resuming
        latest_checkpoint_path = Path(checkpoint_dir) / "latest_checkpoint.pth"
        self.save_checkpoint(str(latest_checkpoint_path))

    def update_model(self) -> None:
        """Update the model based on the update frequency."""
        self.update_counter += 1
        self.train_long_memory()

        if self.update_counter % self.config.update_frequency == 0:
            self.target_model.load_state_dict(self.policy_model.state_dict())  # Update target model

        if self.trainer.logger:
            self.trainer.logger.update_logs()
