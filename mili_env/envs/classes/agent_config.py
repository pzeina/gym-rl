from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

TORCH_CPU = torch.device("cpu")


@dataclass
class AgentConfig:
    """Data class to store agent's configuration parameters."""

    learning_rate: float
    final_lr: float
    decay_lr: float
    initial_epsilon: float
    final_epsilon: float
    discount_factor: float = 0.95
    memory_size: int = 10_000
    batch_size: int = 1024
    batch_num: int = 8
    hidden_size: int = 128
    decay_epsilon: float = 0.995  # Add decay factor for exponential decay
    update_frequency: int = 5  # Update frequency for target network
    subsampling_fraction: float = 0.2
    optimization_steps: int = 5
    grad_clip_window: int = 100
    grad_clip_max: float = 100.0
    likelihood_ratio_clipping: float = 0.2
    estimate_terminal: bool = False
    critessing: Any = None
    exploration: float = 0.0
    variable_noise: float = 0.0
    l2_regularization: float = 0.0
    entropy_regularization: float = 0.0
    dummy_phase: int = 10
    dummy_recurrence: int = 100
    dummy_policy_decay: float = 0.95
    name: str = "agent"
    device: torch.device = TORCH_CPU
    parallel_interactions: int = 1
    seed: int = 42
    execution: Any = None
    saver: Any = None
    summarizer: Any = None
    recorder: Any = None
    trainer_log_file: Path = Path()
    disable_logs: bool = True
