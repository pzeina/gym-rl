from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


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
