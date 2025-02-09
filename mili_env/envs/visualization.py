from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from numpy import floating

if TYPE_CHECKING:
    from torch import nn


class BaseVisualization:
    """Base class for visualizations with common functionality."""

    def __init__(self) -> None:
        """Initialize the base visualization class."""
        self.fig = None

    def set_window_position(self, x: int, y: int) -> None:
        """Set the position of the window on the screen."""
        backend = plt.get_backend()
        if self.fig and self.fig.canvas.manager and hasattr(self.fig.canvas.manager, "window"):
            if backend == "TkAgg":
                self.fig.canvas.manager.window.wm_geometry(f"+{x}+{y}")
            elif backend == "WXAgg":
                self.fig.canvas.manager.window.SetPosition((x, y))
            elif backend == "Qt5Agg":
                self.fig.canvas.manager.window.move(x, y)


class AgentEnvInteractionVisualization(BaseVisualization):
    """Visualization panel for tracking the agent-environment interaction."""

    def __init__(
        self, window_width: int, window_height: int, panel_width: int, random_flag: np.bool_, buffer_size: int = 200
    ) -> None:
        """Initialize the visualization panel."""
        super().__init__()
        self.window_width: int = window_width
        self.window_height: int = window_height
        self.panel_width: int = panel_width
        self.buffer_size: int = buffer_size
        self.random_flag: np.bool_ = random_flag

        self.fig, self.axs = plt.subplots(2, 1, figsize=(panel_width / 100, window_height / 100))
        self.fig.tight_layout(pad=2.0)

        self.rewards = deque(maxlen=buffer_size)
        self.distances = deque(maxlen=buffer_size)
        self.flags = deque(maxlen=buffer_size)
        self.action_counts = [0] * 5  # Initialize counters for 5 actions

        self.idx_ax = 0

    def update(
        self,
        reward: floating[Any] | np.ndarray,
        distance: floating[Any] | np.ndarray,
        action: np.int64 | np.ndarray,
        random_flag: np.bool_ | np.ndarray,
    ) -> None:
        """Update the visualization panel with the latest information."""
        if isinstance(reward, np.ndarray):
            self.rewards.extend(reward)
            self.distances.extend(-distance if isinstance(distance, np.ndarray) else [-distance])
            if isinstance(random_flag, np.ndarray):
                self.flags.extend(random_flag)
            else:
                self.flags.append(random_flag)
            self.action_counts[action] += 1
        else:
            self.rewards.append(reward)
            self.distances.append(-distance)
            self.flags.append(random_flag)
            if isinstance(action, np.ndarray):
                for a in action:
                    self.action_counts[a] += 1
            else:
                self.action_counts[action] += 1

        self.idx_ax = 0

        self.axs[self.idx_ax].cla()
        self.axs[self.idx_ax].plot(self.rewards, label="Reward")
        self.axs[self.idx_ax].plot(self.distances, label="- Distance")
        # add a red point to indicate the random action
        random_indices = [i for i, x in enumerate(self.flags) if x]
        random_rewards = [self.rewards[i] for i in random_indices]
        self.axs[self.idx_ax].plot(random_indices, random_rewards, "ro", markersize=4, label="Random Action")
        # put the legend outside the plot, below the x-axis
        self.axs[self.idx_ax].legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

        self.idx_ax += 1
        self.axs[self.idx_ax].cla()
        self._plot_actions()

        # self.axs[self.idx_ax].cla()                                # noqa: ERA001
        # self.axs[self.idx_ax].plot(self.errors, label="Errors")    # noqa: ERA001
        # self.axs[self.idx_ax].legend()                             # noqa: ERA001

        plt.draw()
        plt.pause(0.001)

    def _plot_actions(self) -> None:
        """Plot the frequency of actions taken."""
        self.axs[self.idx_ax].cla()
        self.axs[self.idx_ax].set_xlabel("Action")
        self.axs[self.idx_ax].set_ylabel("Frequency")

        action_labels = ["-", "↑", "↓", "↶", "↷"]
        action_colors = ["black", "green", "red", "blue", "purple"]

        self.axs[self.idx_ax].bar(action_labels, self.action_counts, color=action_colors)


class GradientLossVisualization(BaseVisualization):
    """A class to visualize the loss and gradients of a PyTorch model."""

    def __init__(self, window_width: int, window_height: int, panel_width: int = 0, buffer_size: int = 200) -> None:
        """Initialize the visualization panel."""
        super().__init__()
        self.window_width: int = window_width
        self.window_height: int = window_height
        self.panel_width: int = panel_width
        self.buffer_size: int = buffer_size

        self.fig, self.axs = plt.subplots(2, 1, figsize=(window_width / 100, window_height / 100))
        self.fig.tight_layout(pad=1.0)

        self.losses = deque(maxlen=buffer_size)
        self.gradients = deque(maxlen=buffer_size)

    def track_loss(self, loss: float | np.ndarray) -> None:
        """Track the loss of the model."""
        if isinstance(loss, np.ndarray):
            self.losses.extend(loss)
        else:
            self.losses.append(loss)
        self._update_plot()

    def track_gradients(self, model: nn.Module) -> None:
        """Track the gradients of the model."""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        self.gradients.append(total_norm)
        self._update_plot()

    def _update_plot(self) -> None:
        """Update the visualization panel with the latest information."""
        self.axs[0].cla()
        self.axs[0].plot(self.losses, label="Loss")
        self.axs[0].legend()

        self.axs[1].cla()
        self.axs[1].plot(self.gradients, label="Gradient Magnitude")
        self.axs[1].legend()

        plt.draw()
        plt.pause(0.001)
