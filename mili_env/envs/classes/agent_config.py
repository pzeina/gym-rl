from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

TORCH_CPU = torch.device("cpu")


@dataclass
class AgentConfig:
    """Data class to store agent's configuration parameters."""

    learning_rate: float
    initial_epsilon: float
    final_epsilon: float
    discount_factor: float = 0.95
    memory_size: int = 100_000
    batch_size: int = 64
    batch_num: int = 8
    hidden_size: int = 256
    decay_factor: float = 0.995  # Add decay factor for exponential decay
    update_frequency: int = 2
    subsampling_fraction: float = 0.2
    optimization_steps: int = 5
    likelihood_ratio_clipping: float = 0.2
    estimate_terminal: bool = False
    critessing: Any = None
    exploration: float = 0.0
    variable_noise: float = 0.0
    l2_regularization: float = 0.0
    entropy_regularization: float = 0.0
    name: str = "agent"
    device: torch.device = TORCH_CPU
    parallel_interactions: int = 1
    seed: int = 42
    execution: Any = None
    saver: Any = None
    summarizer: Any = None
    recorder: Any = None
