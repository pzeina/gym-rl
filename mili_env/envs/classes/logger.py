import csv
from pathlib import Path
from typing import Any

import numpy as np


class Logger:
    """A class to log the details of the training steps."""

    def __init__(self, log_file_path: Path) -> None:
        """Initialize the logger."""
        self.log_file_path = log_file_path
        self.logs: list[list[Any]] = []
        self._initialize_log_file()

    def _initialize_log_file(self) -> None:
        """Initialize the log file with headers."""
        with self.log_file_path.open("w", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(
                [
                    "State",
                    "Action",
                    "Reward",
                    "Next State",
                    "Done",
                    "Q_Predicted",
                    "Q_Target",
                    "Q_Values",
                    "Exploratory",
                ]
            )

    def log_step(  # noqa: PLR0913
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,  # noqa: FBT001
        q_predicted: float,
        q_target: float,
        q_values: np.ndarray,
        exploratory: bool,  # noqa: FBT001
    ) -> None:
        """Log the details of a training step."""
        self.logs.append(
            [
                state.tolist(),
                action,
                reward,
                next_state.tolist(),
                done,
                q_predicted,
                q_target,
                q_values.tolist(),
                exploratory,
            ]
        )

    def merge_logs(self, logs: list[list[Any]]) -> None:
        """Merge the logs from the list to the current logs."""
        self.logs.extend(logs)

    def write_logs(self) -> None:
        """Write the logs to the CSV file."""
        with self.log_file_path.open("a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerows(self.logs)
        self.logs = []  # Reset logs after writing
