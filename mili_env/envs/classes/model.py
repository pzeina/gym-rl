from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class Linear_QNet(nn.Module):  # noqa: N801
    """Neural network model for Q-learning."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """Initialize the neural network with input, hidden, and output layers."""
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.model(x)

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
