#!/usr/bin/env python
"""Replay the D4 collapse-stop rule over recorded runs — the calibration tool.

The trainer's ``collapse_stop_gate`` is pure so this script can run the exact
shipped rule over any ``metrics.csv`` and report where it WOULD have fired.
That is how the defaults were chosen and how any retune must be justified:
the rule must fire on every run the attractor captured (their final thirds
are pure compute spent entrenching a dead policy) and must never fire on a
run that dipped and recovered or simply held.

    scripts/collapse_replay.py                     # every run with a metrics.csv
    scripts/collapse_replay.py platoon_hard_v2_seed13 platoon_hard_flat_v1_seed12
    scripts/collapse_replay.py --patience 100 --margin 0.4 --floor 0.5

One line per run: FIRE (at which step, wasting how much of the budget) or
HOLD (with the peak and the deepest post-peak drawdown, which is what bounds
how tight the margin could safely go).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cohort.training.train import collapse_stop_gate


def replay(rows: list[dict], *, floor: float, margin: float, patience: int) -> dict:
    """Run the shipped gate over one run's (iteration, rolling) sequence.

    metrics.csv does not record episodes_seen, so window-fullness is
    approximated by requiring the rolling figure to have been non-zero at
    least once — before any episode completes the column reads 0.0, and a
    peak of 0.0 can never arm a guard whose floor is positive, so the
    approximation cannot change an outcome, only skip dead air.
    """
    streak = 0
    peak = -1.0
    seen_signal = False
    deepest = 0.0  # largest (peak - rolling) that did NOT end the run
    for row in rows:
        rolling = float(row["success_rate_rolling"])
        seen_signal = seen_signal or rolling > 0.0
        if seen_signal:
            peak = max(peak, rolling)
            if peak >= floor:
                deepest = max(deepest, peak - rolling)
        streak, fire = collapse_stop_gate(
            streak, rolling, peak,
            window_full=seen_signal, floor=floor, margin=margin, patience=patience,
        )
        if fire:
            return {
                "fired": True,
                "at_step": int(float(row["env_steps"])),
                "at_iter": int(float(row["iteration"])),
                "peak": peak,
            }
    return {"fired": False, "peak": peak, "deepest_drawdown": deepest}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*", help="run names (default: all under runs/)")
    ap.add_argument("--patience", type=int, default=None, help="default: PPOConfig")
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--archive", action="store_true", help="include runs/archive")
    args = ap.parse_args()

    from cohort.training.ppo import PPOConfig

    patience = args.patience if args.patience is not None else PPOConfig.collapse_patience
    margin = args.margin if args.margin is not None else PPOConfig.collapse_margin
    floor = args.floor if args.floor is not None else PPOConfig.collapse_floor

    root = Path("runs")
    if args.runs:
        dirs = []
        for name in args.runs:
            hit = root / name
            if not hit.exists() and (root / "archive" / name).exists():
                hit = root / "archive" / name
            dirs.append(hit)
    else:
        dirs = sorted(d for d in root.iterdir() if (d / "metrics.csv").exists())
        if args.archive and (root / "archive").exists():
            dirs += sorted(
                d for d in (root / "archive").iterdir() if (d / "metrics.csv").exists()
            )

    print(f"rule: patience={patience} iters, margin={margin}, floor={floor}")
    fired = held = 0
    for d in dirs:
        path = d / "metrics.csv"
        if not path.exists():
            print(f"{d.name:44s} no metrics.csv")
            continue
        with path.open() as f:
            rows = [r for r in csv.DictReader(f) if r.get("success_rate_rolling")]
        if not rows:
            print(f"{d.name:44s} empty")
            continue
        total = int(float(rows[-1]["env_steps"]))
        r = replay(rows, floor=floor, margin=margin, patience=patience)
        if r["fired"]:
            fired += 1
            saved = 1.0 - r["at_step"] / total if total else 0.0
            print(
                f"{d.name:44s} FIRE @ {r['at_step']:>9,} of {total:,} "
                f"(peak {r['peak']:.0%}, saves {saved:.0%} of the budget)"
            )
        else:
            held += 1
            armed = r["peak"] >= floor
            note = (
                f"deepest post-peak drawdown {r['deepest_drawdown']:.0%}"
                if armed
                else "never armed (peak below floor)"
            )
            print(f"{d.name:44s} hold   (peak {r['peak']:.0%}, {note})")
    print(f"\n{fired} would fire, {held} hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
