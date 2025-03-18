from __future__ import annotations

import argparse
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
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
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


def get_column_categories(metric_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Sépare les colonnes temporelles des autres métriques.

    Args:
        metric_df: DataFrame contenant les métriques

    Returns:
        tuple: (colonnes temporelles, autres colonnes)
    """
    time_columns = [col for col in metric_df.columns if "time" in col.lower()]
    other_columns = [col for col in metric_df.columns if col not in time_columns and col != "Episode"]
    return time_columns, other_columns


def setup_plot_layout(time_columns: list[str], other_columns: list[str]) -> tuple[Figure, list]:
    """Crée la mise en page des sous-graphiques.

    Args:
        time_columns: Liste des colonnes temporelles
        other_columns: Liste des autres colonnes

    Returns:
        tuple: (figure, axes)
    """
    all_columns = ["Time Metrics"] if time_columns else []
    all_columns += other_columns

    num_subplots = len(all_columns)
    ncols = 2  # Two columns
    nrows = math.ceil(num_subplots / ncols)  # Number of rows needed

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8), constrained_layout=True)
    return fig, axes.flatten() if num_subplots > 1 else [axes]


def plot_time_metrics(ax: Axes, metric_df: pd.DataFrame, time_columns: list[str], threshold: float) -> None:
    """Dessine les métriques temporelles sous forme de barres empilées.

    Args:
        ax: Axe matplotlib sur lequel dessiner
        metric_df: DataFrame contenant les métriques
        time_columns: Liste des colonnes temporelles
        threshold: Seuil pour clipper les valeurs
    """
    ax.clear()

    # Compute cumulative sum for stacked bars
    bottom = None
    colors = plt.colormaps.get_cmap("viridis")(np.linspace(0, 1, len(time_columns)))

    for col, color in zip(time_columns, colors):
        values = np.clip(metric_df[col], -threshold, threshold)
        ax.bar(metric_df["Episode"], values, bottom=bottom, label=col, color=color, alpha=0.8)
        bottom = values if bottom is None else bottom + values

    ax.set_title("Cumulative Time Metrics")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Time (s)")
    ax.legend(loc="upper right")  # Fixed legend location


def get_color_mapping(values: np.ndarray) -> list:
    """Crée un mapping de couleurs basé sur les valeurs (vert pour positif, rouge pour négatif).

    Args:
        values: Tableau des valeurs à représenter

    Returns:
        list: Liste des couleurs RGB pour chaque valeur
    """
    max_abs_val = np.max(np.abs(values)) if len(values) > 0 else 1
    max_abs_val = max(max_abs_val, 1)  # Éviter division par zéro

    colors = []
    for val in values:
        if val >= 0:
            intensity = min(abs(val) / max_abs_val * 0.8 + 0.2, 1.0)
            colors.append((0, intensity, 0))
        else:
            intensity = min(abs(val) / max_abs_val * 0.8 + 0.2, 1.0)
            colors.append((intensity, 0, 0))

    return colors


def plot_metric_scatter(ax: Axes, metric_df: pd.DataFrame, column: str, threshold: float) -> None:
    """Dessine une métrique sous forme de nuage de points avec gradient de couleur.

    Args:
        ax: Axe matplotlib sur lequel dessiner
        metric_df: DataFrame contenant les métriques
        column: Nom de la colonne à représenter
        threshold: Seuil pour clipper les valeurs
    """
    ax.clear()

    # Convertir en array numpy pour éviter les problèmes avec ExtensionArray
    values = np.clip(np.array(metric_df[column].tolist()), -threshold, threshold)
    episodes = np.array(metric_df["Episode"].tolist())

    # Obtenir les couleurs pour chaque point
    colors = get_color_mapping(values)

    # Tracer le nuage de points
    ax.scatter(episodes, values, c=colors, alpha=0.5, s=12, label=column)

    ax.set_title(column)
    ax.set_xlabel("Episode")
    ax.set_ylabel(column)
    ax.legend(loc="upper right")  # Fixed legend location


def plot_csv(
    directory: Path = Path("model/"),
    filename: str = "training_log_*.csv",
    *,
    once: bool = False,
    threshold: float = 1e3,
) -> None:
    """Continuously updates the plot with new CSV data."""
    plt.ion()  # Turn on interactive mode

    csv_file = get_latest_csv(directory, filename)
    if not csv_file:
        error_msg = f"No CSV files found in {directory} with filename {filename}."
        raise FileNotFoundError(error_msg)

    metric_df = pd.read_csv(csv_file)
    time_columns, other_columns = get_column_categories(metric_df)
    fig, axes = setup_plot_layout(time_columns, other_columns)

    while True:
        csv_file = get_latest_csv(directory, filename)
        if not csv_file:
            time.sleep(2)
            continue

        metric_df = pd.read_csv(csv_file)
        subplot_idx = 0

        # Plot time metrics
        if time_columns:
            plot_time_metrics(axes[subplot_idx], metric_df, time_columns, threshold)
            subplot_idx += 1

        # Plot other metrics
        for col in other_columns:
            plot_metric_scatter(axes[subplot_idx], metric_df, col, threshold)
            subplot_idx += 1

        # Hide extra subplots (if any)
        for i in range(subplot_idx, len(axes)):
            axes[i].set_visible(False)

        # plt.tight_layout() slow down the rendering
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Extract identifier from CSV filename
        identifier = csv_file.stem.split("_")[-1]

        # Save the plot with the same identifier in the same directory as the CSV file
        plot_path = csv_file.parent / f"plot_{identifier}.png"
        plt.savefig(plot_path)

        time.sleep(2)  # Refresh every 2 seconds

        if once:
            break


if __name__ == "__main__":
    # Parse args to get the once flag
    parser = argparse.ArgumentParser(description="Plot CSV data.")
    parser.add_argument("--once", action="store_true", help="Run the plotting once and exit.")
    args = parser.parse_args()

    once_flag = args.once
    plot_csv(once=once_flag)
