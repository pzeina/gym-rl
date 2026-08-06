"""Training curve plots from a run's metrics.csv.

Also usable as a CLI, including while a training run is still going:

    python -m cohort.viz.plots runs/<run-name>
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _smooth(values: list[float], k: int = 9) -> list[float]:
    if len(values) < 3:
        return values
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - k // 2), min(len(values), i + k // 2 + 1)
        window = [v for v in values[lo:hi] if not math.isnan(v)]
        # a window with no finite sample stays NaN: matplotlib gaps the line
        # rather than inventing a value where the metric was never recorded
        out.append(sum(window) / len(window) if window else math.nan)
    return out


def _num(key: str, value: str | None) -> float | int:
    """Parse one metrics cell, tolerating columns a scenario never records.

    Scenario-specific metrics (the DEFEND positional gates, say) are written
    blank by runs they do not apply to. A blank must degrade to NaN and gap
    the curve — never abort the plot, because ``train.py`` renders curves
    before it evaluates, so a parse error here used to cost a *completed*
    run its eval, transcript, gif and behavior.json as well.
    """
    if key == "iteration":
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def plot_training(run_dir: str | Path, out_name: str = "training_curves.png") -> Path:
    """Render the standard 6-panel training dashboard for a run directory."""
    run_dir = Path(run_dir)
    rows: list[dict] = []
    with (run_dir / "metrics.csv").open() as f:
        rows = [
            {k: _num(k, v) for k, v in row.items() if k is not None}
            for row in csv.DictReader(f)
        ]
    if not rows:
        msg = f"No metrics rows in {run_dir / 'metrics.csv'}"
        raise ValueError(msg)

    steps = [r["env_steps"] for r in rows]
    panels = [
        ("episode return (team mean)", [("ep_return", "return")]),
        ("success rate", [("success_rate", "success")]),
        ("episode length", [("ep_length", "length")]),
        (
            "reward components (per agent-step)",
            [
                ("comp_compliance", "compliance"),
                ("comp_report", "report"),
                ("comp_command", "command"),
                ("comp_combat", "combat"),
                ("tx_per_agent_step", "transmissions"),  # absent in pre-A4 runs
            ],
        ),
        ("policy entropy", [("entropy", "entropy")]),
        ("losses", [("policy_loss", "policy"), ("value_loss", "value")]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), dpi=110)
    for ax, (title, series) in zip(axes.flat, panels, strict=True):
        for key, label in series:
            if key not in rows[0]:
                continue
            values = [r[key] for r in rows]
            if all(math.isnan(v) for v in values):
                continue  # column exists but this scenario never records it
            ax.plot(steps, _smooth(values), lw=1.6, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("env steps", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        if len(series) > 1:
            ax.legend(fontsize=8)
    fig.suptitle(f"cohort training — {run_dir.name}", fontsize=12)
    fig.tight_layout()
    out = run_dir / out_name
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    """CLI: (re)generate the training dashboard for one or more run dirs."""
    parser = argparse.ArgumentParser(description="Plot training curves from metrics.csv.")
    parser.add_argument("run_dirs", nargs="+", help="run directories, e.g. runs/fireteam_v2")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        print(f"curves → {plot_training(run_dir)}")


if __name__ == "__main__":
    main()
