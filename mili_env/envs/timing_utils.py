"""Utility helpers for lightweight timing statistics and a small dashboard launcher.

This module provides a simple global timing registry useful for profiling
short-lived sections of code. It is intentionally small and thread-safe.

Functions exposed:
- ``timing_start(name)``: start timing a named section
- ``timing_stop(name)``: stop timing and accumulate stats
- ``timing_log(log_file)``: write aggregated stats to CSV

Note: importing this module will attempt to launch a small dashboard process
the first time any timing function is used. The dashboard is optional and will
not be started if port 8080 is already in use.
"""

from __future__ import annotations

import csv
import socket
import threading
from pathlib import Path
from timeit import default_timer
from typing import Any


def _is_port_in_use(port: int) -> bool:
    """Return True if ``127.0.0.1:port`` is accepting TCP connections.

    This opens a short-lived TCP socket and checks connection availability.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


# Runtime timing storage and synchronization
TimingEntry = dict[str, Any]
TimingStats = dict[str, TimingEntry]

timing_stats: TimingStats = {}
timing_stats_lock = threading.Lock()
timing_stats_enabled: bool = True


def timing_start(name: str) -> None:
    """Start timing a named section.

    Args:
        name: A short identifier for the timed section. Multiple calls to
            ``timing_start`` with the same ``name`` will overwrite any
            previously recorded start time for that name.
    """
    if timing_stats_enabled:
        with timing_stats_lock:
            timing_stats.setdefault(name, {"count": 0, "total": 0.0})
            timing_stats[name]["_start"] = default_timer()
    # No dashboard in this trimmed version; keep timing only.


def timing_stop(name: str) -> None:
    """Stop timing the named section and accumulate statistics.

    If ``timing_start`` was not previously called for ``name`` this function
    is a no-op. The stats stored for each name include a running ``total`` and
    ``count`` which are used to compute an average by callers or by
    ``timing_log``.
    """
    if timing_stats_enabled:
        with timing_stats_lock:
            entry = timing_stats.get(name)
            if entry and "_start" in entry:
                elapsed = default_timer() - entry["_start"]
                entry["total"] += elapsed
                entry["count"] += 1
                del entry["_start"]
    # No dashboard in this trimmed version; keep timing only.


def timing_log(log_file: str = "timing_stats.csv") -> None:
    """Write aggregated timing statistics to CSV.

    The CSV columns are: ``function``, ``avg``, ``total``, ``count``. If the
    file does not exist a header row will be written. Only entries with a
    non-zero ``count`` are persisted.

    Args:
        log_file: Path to the CSV file to append statistics to.
    """
    stats: TimingStats = {}
    with timing_stats_lock:
        for k, v in timing_stats.items():
            if v.get("count", 0) > 0:
                stats[k] = {
                    "avg": v["total"] / v["count"],
                    "total": v["total"],
                    "count": v["count"],
                }
    file_path: Path = Path(log_file)
    write_header = not file_path.exists() or file_path.stat().st_size == 0
    # Use ``Path.open`` to follow ruff's recommendations and ensure proper
    # resource handling.
    with file_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["function", "avg", "total", "count"])
        for func, v in stats.items():
            writer.writerow([func, v["avg"], v["total"], v["count"]])
    # No dashboard in this trimmed version; keep timing only.


# Automatically launch dashboard if this file is run directly
if __name__ == "__main__":
    # Provide a tiny self-test when executed directly: record a short timing
    # and write to a temporary CSV in the current working directory.
    timing_start("self_test")
    import time

    time.sleep(0.001)
    timing_stop("self_test")
    timing_log("timing_stats.csv")
