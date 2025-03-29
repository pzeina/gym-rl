from __future__ import annotations

import csv
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from matplotlib.axes import Axes


class Logger:
    """A class to log the details of the training steps."""

    def __init__(  # noqa: PLR0913
        self,
        log_file_path: Path,
        keys: list[str],
        graphs: list[dict[str, Any]] | None = None,
        static_keys: dict[str, Any] | None = None,
        mode: str = "wp",
        window_size: int = 2000,
    ) -> None:
        """Initialize the logger.

        Args:
            log_file_path: Path to the log file.
            keys: List of keys to log.
            graphs: List of dictionaries specifying graphs to plot.
            static_keys: Dictionary of static keys to include in every log entry.
            mode: Mode string, 'r' for read, 'w' for write, 'p' for plot.
            window_size: Maximum number of data points to display in plots. If None, all data is shown.
        """
        self.log_file_path = log_file_path
        self.logger_id: str = str(uuid.uuid4())
        self.log_mem: list[dict[str, Any]] = []
        self.log_tmp: list[dict[str, Any]] = []
        self.keys: list[str] = keys
        self.window_size = window_size

        # Check that the required graphs are valid
        if graphs:
            self._check_valid_keys(keys, graphs)

        for graph in graphs if graphs else []:
            # Optional vectorized flag
            graph.setdefault("vectorized", False)
            graph.setdefault("split", False)
            graph.setdefault("split_labels", [])

        self.graphs: list[dict[str, Any]] = graphs if graphs else []
        self.static_keys: dict[str, Any] = static_keys if static_keys else {}
        self.enable_read, self.enable_write, self.enable_plot = (x in mode for x in ["r", "w", "p"])

        if self.enable_read and self.enable_write:
            msg = "Cannot read and write to the same file at the same time."
            raise ValueError(msg)
        if self.enable_read or self.enable_write:
            mode = "r" if self.enable_read else "w"
            self._initialize_log_file(mode=mode)

        if self.enable_plot:
            self._initialize_graph_window()
        # thisisTODO self._initialize_debug_window()

    def _check_valid_keys(self, keys: list[str], graphs: list[dict[str, Any]]) -> None:
        for graph in graphs:
            # Validate required keys
            required_keys = ["x", "y"]
            for key in required_keys:
                if key not in graph:
                    msg = f"Graph must have '{key}' key"
                    raise ValueError(msg)

            # Validate x key
            if isinstance(graph["x"], str) and graph["x"] not in keys:
                msg = f"X key {graph['x']} not in keys: {keys}"
                raise ValueError(msg)

            for to_validate in ["y", "mark_occurrence"]:
                if to_validate not in graph:
                    continue
                # Validate y keys
                if not isinstance(graph[to_validate], list):
                    msg = f"{to_validate} key must be a list of keys"
                    raise TypeError(msg)

                if not all(isinstance(k, str) for k in graph[to_validate]):
                    msg = f"{to_validate} keys must be strings"
                    raise ValueError(msg)

                if not all(k in keys for k in graph[to_validate]):
                    msg = f"{to_validate} keys {graph[to_validate]} not in keys: {keys}"
                    raise ValueError(msg)

    def _initialize_log_file(self, mode: str) -> None:
        """Initialize the log file with headers or read existing logs."""
        if mode == "w":
            with self.log_file_path.open(mode, newline="") as log_file:
                writer = csv.writer(log_file)
                writer.writerow(list(self.static_keys.keys()) + self.keys)
        elif mode == "r":
            self.start_dynamic_reader(interval=5.0)
        else:
            msg = "Invalid mode. Use 'w' for write or 'r' for read."
            raise ValueError(msg)

    def _initialize_debug_window(self) -> None:
        """Initialize the debug window without threading issues."""
        self.text_fig, self.ax = plt.subplots(figsize=(6, 6))
        self.text_ani = FuncAnimation(self.text_fig, self._update_debug_info, interval=1000, cache_frame_data=False)  # type: ignore # noqa: PGH003
        plt.show(block=False)  # Non-blocking show

    def _initialize_graph_window(self) -> None:
        """Initialize the graph window without threading issues."""
        num_graphs = len(self.graphs)
        self.fig, self.axes = plt.subplots(num_graphs, 1, figsize=(20, 1.5 * num_graphs))
        self.ani = FuncAnimation(self.fig, self._plot_logs, interval=1000, cache_frame_data=False)  # type: ignore # noqa: PGH003
        plt.show(block=False)  # Non-blocking show

    def _dynamic_reader(self, interval: float = 1.0) -> None:
        """Dynamically read new data from the log file as it is written.

        Args:
            interval (float): Time interval (in seconds) to check for new data.
        """
        last_position = 0  # Track the last read position in the file

        while True:
            try:
                with self.log_file_path.open("r", newline="") as log_file:
                    # Move to the last read position
                    log_file.seek(last_position)

                    # Read new lines
                    reader = csv.DictReader(log_file)
                    for row in reader:
                        self.log_mem.append(dict(row))  # Append new data to log_mem
                    self.log_mem = self.log_mem[-self.window_size :]

                    # Update the last read position
                    last_position = log_file.tell()

            except FileNotFoundError:
                msg = "Log file not found. Ensure the file exists and the path is correct."
                raise RuntimeError(msg) from None
            except csv.Error as csv_err:
                msg = "Error parsing the log file. Ensure it is a valid CSV."
                raise RuntimeError(msg) from csv_err
            except Exception as exc:
                msg = "An unexpected error occurred while reading the log file."
                raise RuntimeError(msg) from exc

            # Wait for the specified interval before checking again
            time.sleep(interval)

    def start_dynamic_reader(self, interval: float = 1.0) -> None:
        """Start the dynamic reader in a separate thread.

        Args:
            interval (float): Time interval (in seconds) to check for new data.
        """
        reader_thread = threading.Thread(target=self._dynamic_reader, args=(interval,), daemon=True)
        reader_thread.start()

    def _merge_logs(self) -> None:
        """Merge the logs from the list to the current logs private memory."""
        self.log_mem.extend(self.log_tmp)
        self.log_mem = self.log_mem[-self.window_size :]  # Keep only the last window_size logs

    def log_entry(self, log_dict: dict[str, Any]) -> None:
        """Log the details of a training step to the log private memory."""
        # Check that the log_dict has the correct keys
        for key in log_dict:
            if key not in self.keys:
                msg = f"Key {key} not in keys: {self.keys}"
                raise ValueError(msg)

        log_dict.update(dict(self.static_keys))
        self.log_tmp.append(log_dict)

    def _write_logs(self) -> None:
        """Write the log tmp buffer to the CSV file."""
        with self.log_file_path.open("a", newline="") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=list(self.static_keys.keys()) + self.keys)
            writer.writerows(self.log_tmp)

    def update_logs(self) -> None:
        """Update the logs."""
        # First write the tmp logs to the file
        self._write_logs()

        # Then refresh window with the tmp logs
        self._refresh_window()

        # Finally merge tmp with mem
        self._merge_logs()

    def _show_debug_window(self) -> None:
        while True:
            plt.pause(0.1)  # Allow GUI updates

    def _update_debug_info(self, frame: Iterable[Any]) -> Axes:  # noqa: ARG002
        """Update the debug information in the window."""
        self.ax.clear()
        for log in self.log_tmp:
            info = "\n".join([f"{k}: {v}" for k, v in log.items()])
            self.ax.text(
                0.05,  # horizontal position (left)
                0.5,  # vertical position (center)
                info,
                ha="center",
                va="center",
                fontsize=12,
                transform=self.ax.transAxes,
            )
        self.ax.axis("off")
        return self.ax

    def _refresh_window(self) -> None:
        """Refresh the debug window."""
        if self.fig:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def _plot_logs(self, frame: Iterable[Any]) -> None:  # noqa: PLR0915, PLR0912, C901, ARG002
        """Plot the logs in the window using heatmaps for vectorized slice data.

        - Heatmap visualization for 2D slice data.
        - Vertical growth of heatmap as new samples are added.
        - Handling multiple plot keys and environments.

        Args:
            frame (Iterable[Any]): The frame data to be plotted.

        Returns:
            None
        """
        # Clear existing figure and recreate with GridSpec
        plt.clf()

        # Determine number of graphs and environments
        num_graphs = len(self.graphs)
        num_envs = 1  # Default to 1 environment

        # Check if any graph is vectorized
        for graph in self.graphs:
            if graph.get("vectorized", False) and self.log_mem:
                num_envs = len(self.log_mem[0][graph["y"][0]])
                break

        # Create GridSpec
        gs = gridspec.GridSpec(nrows=num_graphs, ncols=num_envs)

        # Process graphs
        for graph_idx, graph in enumerate(self.graphs):
            plot_keys = graph["y"]
            vertical_mark_keys = graph.get("mark_occurrence", [])
            is_vectorized = graph.get("vectorized", False)
            is_split = graph.get("split", False)

            if is_vectorized:
                for env_idx in range(num_envs):
                    ax_main_per_env = plt.subplot(gs[graph_idx, env_idx])

                    plt.sca(ax_main_per_env)

                    if is_split:
                        # Create a divider for the current axis
                        divider_slices = make_axes_locatable(ax_main_per_env)
                        split_labels = graph.get("split_labels", [])

                        ax_current = ax_main_per_env

                        if self.log_mem and self.log_mem[0][plot_keys[0]].ndim > 1:
                            # Split per slice
                            total_slices = self.log_mem[0][plot_keys[0]].shape[1]
                            for slice_idx in range(total_slices):
                                ax_current = (
                                    divider_slices.append_axes(
                                        "right", size="100%", sharex=ax_main_per_env, sharey=ax_main_per_env
                                    )
                                    if slice_idx > 0
                                    else ax_main_per_env
                                )

                                # Collect vectorized data for all plot keys for this slice
                                for plot_key in plot_keys:
                                    if self.log_mem[0][plot_key] is not None:
                                        # Collect slice data for the current environment
                                        y_data = [
                                            log[plot_key][env_idx, slice_idx]
                                            for log in self.log_mem
                                            if log[plot_key] is not None
                                        ]
                                        if y_data:
                                            x_data = range(len(y_data))
                                            ax_current.scatter(
                                                x_data,
                                                y_data,
                                                label=plot_key if len(plot_keys) > 1 else None,
                                                alpha=0.2,
                                                marker="d",
                                            )

                                for vertical_mark_key in vertical_mark_keys:
                                    vertical_mark_data = [
                                        log[vertical_mark_key][env_idx]
                                        for log in self.log_mem
                                        if log[vertical_mark_key] is not None
                                    ]
                                    vertical_mark_positions = [i for i, mark in enumerate(vertical_mark_data) if mark]

                                    for x_pos in vertical_mark_positions:
                                        ax_current.axvline(
                                            x=x_pos,
                                            color="red",
                                            linestyle="dotted",
                                            linewidth=1,
                                        )

                                ax_current.axhline(0, color="black", linestyle="--", linewidth=1)
                                ax_current.set_title(f"{split_labels[slice_idx] if split_labels else ''}", fontsize=6)
                                ax_current.legend(loc="upper right")

                            ax_main_per_env.text(
                                total_slices * 0.5,  # Center of the slice plots (horizontally)
                                1.2,  # Position above the row
                                f"{' & '.join(plot_keys)} (env:{env_idx + 1})",
                                fontsize=10,
                                ha="center",
                                transform=ax_main_per_env.transAxes,
                            )
                    else:
                        # Handle non-split data
                        for plot_key in plot_keys:
                            y_data = [log[plot_key][env_idx] for log in self.log_mem if log[plot_key] is not None]
                            if y_data:
                                x_data = range(len(y_data))

                                ax_main_per_env.scatter(
                                    x_data,
                                    y_data,
                                    label=plot_key if len(plot_keys) > 1 else None,
                                    alpha=0.2,
                                    marker="d",
                                )
                        for vertical_mark_key in vertical_mark_keys:
                            vertical_mark_data = [
                                log[vertical_mark_key][env_idx]
                                for log in self.log_mem
                                if log[vertical_mark_key] is not None
                            ]
                            vertical_mark_positions = [i for i, mark in enumerate(vertical_mark_data) if mark]

                            for x_pos in vertical_mark_positions:
                                ax_main_per_env.axvline(
                                    x=x_pos,
                                    color="red",
                                    linestyle="dotted",
                                    linewidth=1,
                                )

                        ax_main_per_env.axhline(0, color="black", linestyle="--", linewidth=1)
                        ax_main_per_env.set_title(f"{' & '.join(plot_keys)} (env:{env_idx + 1})", fontsize=10)
                        ax_main_per_env.legend(loc="upper right")
            else:
                ax_main_single_env = plt.subplot(gs[graph_idx, 0])
                plt.sca(ax_main_single_env)

                if is_split:
                    # Create a divider for the current axis
                    divider_slices = make_axes_locatable(ax_main_single_env)
                    split_labels = graph.get("split_labels", [])

                    if self.log_mem and isinstance(self.log_mem[0][plot_keys[0]], np.ndarray):
                        # Split per slice
                        total_slices = self.log_mem[0][plot_keys[0]].shape[0]
                        for slice_idx in range(total_slices):
                            ax_current = (
                                divider_slices.append_axes(
                                    "right", size="100%", sharex=ax_main_single_env, sharey=ax_main_single_env
                                )
                                if slice_idx > 0
                                else ax_main_single_env
                            )

                            # Collect slice data for all plot keys for this slice
                            for plot_key in plot_keys:
                                if self.log_mem[0][plot_key] is not None:
                                    # Collect slice data for the current environment
                                    y_data = [
                                        log[plot_key][slice_idx] for log in self.log_mem if log[plot_key] is not None
                                    ]
                                    if y_data:
                                        x_data = range(len(y_data))
                                        ax_current.scatter(
                                            x_data,
                                            y_data,
                                            label=plot_key if len(plot_keys) > 1 else None,
                                            alpha=0.2,
                                            marker="d",
                                        )
                            for vertical_mark_key in vertical_mark_keys:
                                vertical_mark_data = [
                                    log[vertical_mark_key] for log in self.log_mem if log[vertical_mark_key] is not None
                                ]
                                vertical_mark_positions = [i for i, mark in enumerate(vertical_mark_data) if mark]

                                for x_pos in vertical_mark_positions:
                                    ax_current.axvline(
                                        x=x_pos,
                                        color="red",
                                        linestyle="dotted",
                                        linewidth=1,
                                    )

                            ax_current.axhline(0, color="black", linestyle="--", linewidth=1)
                            ax_current.set_title(f"{split_labels[slice_idx] if split_labels else ''}", fontsize=6)
                            ax_main_single_env.text(
                                total_slices * 0.5,  # Center of the slice plots (horizontally)
                                1.2,  # Position above the row
                                f"{' & '.join(plot_keys)} (global)",
                                fontsize=10,
                                ha="center",
                                transform=ax_main_single_env.transAxes,
                            )
                            ax_main_single_env.legend(loc="upper right")
                else:
                    # Handle non-split data
                    for plot_key in plot_keys:
                        data = [log[plot_key] for log in self.log_mem if log[plot_key] is not None]
                        if data:
                            x_data = range(len(data))

                            ax_main_single_env.scatter(
                                x_data, data, label=plot_key if len(plot_keys) > 1 else None, alpha=0.2, marker="d"
                            )
                    for vertical_mark_key in vertical_mark_keys:
                        vertical_mark_data = [
                            log[vertical_mark_key] for log in self.log_mem if log[vertical_mark_key] is not None
                        ]
                        vertical_mark_positions = [i for i, mark in enumerate(vertical_mark_data) if mark]

                        for x_pos in vertical_mark_positions:
                            ax_main_single_env.axvline(
                                x=x_pos,
                                color="red",
                                linestyle="dotted",
                                linewidth=1,
                            )

                    ax_main_single_env.axhline(0, color="black", linestyle="--", linewidth=1)
                    ax_main_single_env.set_title(f"{' & '.join(plot_keys)} (global)", fontsize=10)
                    ax_main_single_env.legend(loc="upper right")

        plt.tight_layout()
