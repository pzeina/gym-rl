from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from mili_env.envs.visualization import GradientLossVisualization  # noqa: TC001


class Linear_QNet(nn.Module):  # noqa: N801
    """Neural network model for Q-learning."""

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, favored_actions: list[int] | None = None
    ) -> None:
        """Initialize the neural network with input, hidden, and output layers."""
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        if favored_actions is not None:
            self.initialize_weights(favored_actions)

    def initialize_weights(self, favored_actions: list[int]) -> None:
        """Initialize the weights and biases of the network."""
        if favored_actions is not None:
            # Initialize weights and biases to favor specific actions
            nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)
            for action in favored_actions:
                self.linear2.bias.data[action] = 1.0  # Set higher bias for favored actions
        else:
            # Default initialization
            nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity="relu")
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)

    def save(self, file_name: str = "model.pth") -> None:
        """Save the model to a file."""
        model_folder_path = Path(__file__).resolve().parent.parent
        if not model_folder_path.exists():
            model_folder_path.mkdir(parents=True, exist_ok=True)

        file_path = model_folder_path / file_name
        torch.save(self.state_dict(), file_path)

    def load(self, file_name: str = "model.pth") -> None:
        """Load the model from a file."""
        model_folder_path = Path(__file__).resolve().parent.parent
        file_path = model_folder_path / file_name
        self.load_state_dict(torch.load(file_path))


class QTrainer:
    """Q-learning trainer class to train the Q-learning model."""

    def __init__(
        self, model: nn.Module, lr: float, gamma: float, visualization: GradientLossVisualization | None = None
    ) -> None:
        """Initialize the Q-learning trainer with a model, learning rate, and discount factor."""
        self.lr: float = lr
        self.gamma: float = gamma
        self.model: nn.Module = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.visualization: GradientLossVisualization | None = visualization

    def train_step(
        self,
        state,  # noqa: ANN001
        action,  # noqa: ANN001
        reward,  # noqa: ANN001
        next_state,  # noqa: ANN001
        done,  # noqa: ANN001
    ) -> None:
        """Train the model using a single experience."""
        device = next(self.model.parameters()).device  # Get the device of the model
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)
        done = torch.tensor(np.array(done), dtype=torch.bool).to(device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)

        pred = self.model(state)
        target = pred.clone()
        q_new = reward + self.gamma * torch.max(self.model(next_state), dim=1)[0] * (~done)
        target[range(len(action)), action] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # Track gradients
        if self.visualization:
            self.visualization.track_gradients(self.model)

        self.optimizer.step()

        # Track loss
        if self.visualization:
            self.visualization.track_loss(loss.item())
