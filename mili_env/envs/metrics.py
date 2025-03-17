from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import floating

if TYPE_CHECKING:
    from torch import nn


def get_latest_csv(directory: Path, filename: str) -> Path | None:
    """Finds the most recent CSV file based on timestamp in filename."""
    csv_files = list(directory.glob(filename))
    if not csv_files:
        return None  # No CSV files found

    # Extract timestamp from filename and sort by newest
    return max(csv_files, key=lambda x: int(x.stem.split("_")[-1]))


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


class GradientLossTracker(BaseVisualization):
    """A class to visualize the loss and gradients of a PyTorch model."""

    def __init__(
        self,
        window_width: int,
        window_height: int,
        panel_width: int = 0,
        buffer_size: int = 200,
        *,
        graphs: bool = False,
    ) -> None:
        """Initialize the visualization panel."""
        super().__init__()
        self.window_width: int = window_width
        self.window_height: int = window_height
        self.panel_width: int = panel_width
        self.buffer_size: int = buffer_size

        if graphs:
            self.fig, self.axs = plt.subplots(2, 1, figsize=(window_width / 100, window_height / 100))
            self.fig.tight_layout(pad=1.0)

        self.losses = deque(maxlen=buffer_size)
        self.gradients = deque(maxlen=buffer_size)

    def track_loss(self, loss: float | np.ndarray) -> float:
        """Track the loss of the model."""
        if isinstance(loss, np.ndarray):
            self.losses.extend(loss)
        else:
            self.losses.append(loss)
        return float(loss) if isinstance(loss, (int, float)) else float(np.mean(loss))

    def track_gradients(self, model: nn.Module) -> float:
        """Track the gradients of the model."""
        total_norm: float = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        self.gradients.append(total_norm)
        return total_norm

    def show(self) -> None:
        """Show the visualization panel."""
        if self.fig:
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


def plot_csv(directory: Path = Path("model/"), filename: str = "training_log_*.csv") -> None:
    """Continuously updates the plot with new CSV data."""
    plt.ion()  # Turn on interactive mode

    csv_file = get_latest_csv(directory, filename)
    if not csv_file:
        error_msg = f"No CSV files found in {directory} with filename {filename}."
        raise FileNotFoundError(error_msg)

    metric_df = pd.read_csv(csv_file)

    # Separate "time" columns from others
    time_columns = [col for col in metric_df.columns if "time" in col.lower()]
    other_columns = [col for col in metric_df.columns if col not in time_columns and col != "Episode"]

    # Group "time" metrics into one subplot
    all_columns = ["Time Metrics"] if time_columns else []
    all_columns += other_columns

    num_subplots = len(all_columns)
    ncols = 2  # Two columns
    nrows = math.ceil(num_subplots / ncols)  # Number of rows needed

    # Create a persistent figure with reduced width
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8), constrained_layout=True)
    axes = axes.flatten() if num_subplots > 1 else [axes]  # Ensure axes is always a list

    while True:
        csv_file = get_latest_csv(directory, filename)
        if not csv_file:
            time.sleep(2)
            continue

        metric_df = pd.read_csv(csv_file)

        subplot_idx = 0

        # 1️⃣ Plot cumulative time metrics as stacked bars
        if time_columns:
            ax = axes[subplot_idx]
            ax.clear()

            # Compute cumulative sum for stacked bars
            bottom = None
            colors = plt.colormaps.get_cmap("viridis")(np.linspace(0, 1, len(time_columns)))  # Generate unique colors

            for col, color in zip(time_columns, colors):
                ax.bar(metric_df["Episode"], metric_df[col], bottom=bottom, label=col, color=color, alpha=0.8)
                bottom = metric_df[col] if bottom is None else bottom + metric_df[col]  # Stack bars

            ax.set_title("Cumulative Time Metrics")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Time (s)")
            ax.legend()
            subplot_idx += 1

        # 2️⃣ Plot all other columns separately
        for col in other_columns:
            ax = axes[subplot_idx]
            ax.clear()
            ax.plot(metric_df["Episode"], metric_df[col], linestyle="-", color="tab:blue")  # No markers
            ax.set_title(col)
            ax.set_xlabel("Episode")
            ax.set_ylabel(col)
            subplot_idx += 1

        # Hide extra subplots (if any)
        for i in range(subplot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        fig.canvas.draw()  # Redraw figure without popping
        fig.canvas.flush_events()  # Process UI events smoothly
        time.sleep(2)  # Refresh every 2 seconds


if __name__ == "__main__":
    plot_csv()
