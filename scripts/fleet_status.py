#!/usr/bin/env python
"""Fleet board: every run's headline result and whether it still loads.

After a breaking cycle the interesting question is not "how good is the fleet"
but "how much of it is still real". This walks ``runs/``, reads each run's
committed ``behavior.json`` (the N=100 evaluation — never ``metrics.csv``), and
checks each checkpoint's spaces against the current build.

    scripts/fleet_status.py                 # text table
    scripts/fleet_status.py --json          # machine-readable, for the board

Cheap by construction: no episodes are simulated and no training log is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cohort.env.actions import N_ACTIONS
from cohort.env.observations import OBS_DIM


def _behavior(run: Path) -> dict:
    path = run / "behavior.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def collect(runs_dir: Path) -> list[dict]:
    from cohort.viz.dashboard import checkpoint_meta

    rows = []
    for run in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        if not (run / "metrics.csv").is_file():
            continue
        cfg = {}
        cfg_file = run / "config.json"
        if cfg_file.is_file():
            try:
                cfg = json.loads(cfg_file.read_text())
            except (OSError, json.JSONDecodeError):
                cfg = {}
        best = run / "ckpt_best.pt"
        meta = checkpoint_meta(best) if best.is_file() else {"loadable": False, "reason": "no ckpt_best.pt"}
        beh = _behavior(run)
        metrics = beh.get("metrics", {})
        # behavior.json stores the headline as "0.51 ± 0.10"; keep the string
        # (the CI is the point) and parse the rate out for sorting/plotting
        ci = beh.get("success_ci95")
        rate = None
        if ci:
            try:
                rate = float(str(ci).split("±")[0].strip())
            except ValueError:
                rate = None
        rows.append(
            {
                "run": run.name,
                "scenario": cfg.get("scenario"),
                "success": rate,
                "success_ci95": ci,
                "episodes": beh.get("episodes"),
                "env_steps": meta.get("env_steps"),
                "obs_dim": meta.get("obs_dim"),
                "loadable": bool(meta.get("loadable")),
                "reason": meta.get("reason", ""),
                "human_death_rate": metrics.get("human_death_rate"),
                "false_complete_rate": metrics.get("false_complete_rate"),
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Fleet board: results + staleness.")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p.add_argument("--evaluated-only", action="store_true", help="only runs with behavior.json")
    args = p.parse_args()

    rows = collect(Path(args.runs_dir))
    if args.evaluated_only:
        rows = [r for r in rows if r["success"] is not None]

    if args.json:
        print(json.dumps(
            {"spaces": {"obs_dim": OBS_DIM, "n_actions": N_ACTIONS}, "runs": rows}, indent=2
        ))
        return

    live = sum(1 for r in rows if r["loadable"])
    print(f"build spaces: Discrete({N_ACTIONS})/Box({OBS_DIM})")
    print(f"runs: {len(rows)}   loadable: {live}   stale: {len(rows) - live}")
    print()
    print(f"{'run':<26}{'scenario':<18}{'success':>16}{'obs':>6}  {'':<1}")
    print("-" * 70)
    for r in rows:
        succ = r["success_ci95"] or ("—" if r["success"] is None else f"{r['success']:.2f}")
        flag = "" if r["loadable"] else "  ⚠ stale"
        print(
            f"{r['run']:<26}{(r['scenario'] or '?'):<18}{succ:>16}"
            f"{(r['obs_dim'] or '—')!s:>6}{flag}"
        )
    if live == 0 and rows:
        print()
        print("NOTE: no checkpoint on disk loads under the current spaces —")
        print("      an open breaking cycle. The fleet needs retraining.")


if __name__ == "__main__":
    main()
