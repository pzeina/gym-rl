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


def _json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _rate(ci95: str | None) -> float | None:
    """Parse the rate out of a ``"0.86 ± 0.07"`` headline."""
    if not ci95:
        return None
    try:
        return float(str(ci95).split("±")[0].strip())
    except ValueError:
        return None


def _half_width(ci95: str | None) -> float | None:
    if not ci95 or "±" not in str(ci95):
        return None
    try:
        return float(str(ci95).split("±")[1].strip())
    except ValueError:
        return None


def _live(run: Path) -> dict:
    """Training state, but only pay for it when a job is actually alive.

    ``summarize`` parses the whole metrics.csv; doing that for every run on
    disk would make the board cost what this repo's workflow exists to avoid.
    """
    from scripts.train_status import alive, job_of, summarize

    job = job_of(run)
    if not (job and alive(job.get("pid", -1))):
        return {"state": None, "progress": None, "rolling": None, "eta": ""}
    s = summarize(run)
    return {
        "state": "RUNNING",
        "progress": s["pct"],
        "rolling": s["rolling"],
        "eta": s["eta"],
        "steps_done": s["steps"],
        "steps_total": s["total"],
    }


def collect(runs_dir: Path) -> list[dict]:
    from cohort.viz.dashboard import checkpoint_meta

    rows = []
    for run in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        if not (run / "metrics.csv").is_file():
            continue
        cfg = _json(run / "config.json")
        best_ckpt = run / "ckpt_best.pt"
        meta = (
            checkpoint_meta(best_ckpt)
            if best_ckpt.is_file()
            else {"loadable": False, "reason": "no ckpt_best.pt"}
        )
        # Two evaluations live in a run dir and they answer different questions:
        # behavior_final.json is the FINAL policy (what publication quotes),
        # behavior.json the rolling-best checkpoint. The board must say which
        # it is showing and at what N — the old board printed one N=100 caption
        # over rows that were N=20.
        final, best = _json(run / "behavior_final.json"), _json(run / "behavior.json")
        head = final or best
        # ...and read the step count off the checkpoint the headline actually
        # describes. ckpt_best can be an early save — quoting its step count
        # beside a final-policy score reads as "trained for 0.28M" when the
        # policy scored there trained for 3M.
        latest_ckpt = run / "ckpt_latest.pt"
        env_steps = meta.get("env_steps")
        if final and latest_ckpt.is_file():
            env_steps = checkpoint_meta(latest_ckpt).get("env_steps") or env_steps
        gates = head.get("gates") or []
        econ = _json(run / "economics.json")
        live = _live(run)
        rows.append(
            {
                "run": run.name,
                "scenario": cfg.get("scenario"),
                "success": _rate(head.get("success_ci95")),
                "success_ci95": head.get("success_ci95"),
                "success_ci": _half_width(head.get("success_ci95")),
                "episodes": head.get("episodes"),
                "policy": ("final" if final else "best") if head else None,
                "final_ci95": final.get("success_ci95"),
                "final_episodes": final.get("episodes"),
                "best_ci95": best.get("success_ci95"),
                "best_episodes": best.get("episodes"),
                "gates": gates,
                "gates_failed": [g["name"] for g in gates if not g.get("passed")],
                "overrides": econ.get("reward_overrides") or [],
                "env_steps": env_steps,
                "obs_dim": meta.get("obs_dim"),
                "loadable": bool(meta.get("loadable")),
                "reason": meta.get("reason", ""),
                "human_death_rate": head.get("metrics", {}).get("human_death_rate"),
                "false_complete_rate": head.get("metrics", {}).get("false_complete_rate"),
                **live,
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
    print(f"{'run':<26}{'scenario':<18}{'success':>16}{'policy':>8}{'N':>5}{'obs':>6}  {'':<1}")
    print("-" * 82)
    for r in rows:
        succ = r["success_ci95"] or ("—" if r["success"] is None else f"{r['success']:.2f}")
        flags = "" if r["loadable"] else "  ⚠ stale"
        if r["gates_failed"]:
            flags += f"  ✕ {','.join(r['gates_failed'])}"
        if r["state"] == "RUNNING":
            flags += f"  ▶ training {r['progress']:.0f}%"
        print(
            f"{r['run']:<26}{(r['scenario'] or '?'):<18}{succ:>16}"
            f"{(r['policy'] or '—'):>8}{(r['episodes'] or '—')!s:>5}"
            f"{(r['obs_dim'] or '—')!s:>6}{flags}"
        )
    if live == 0 and rows:
        print()
        print("NOTE: no checkpoint on disk loads under the current spaces —")
        print("      an open breaking cycle. The fleet needs retraining.")


if __name__ == "__main__":
    main()
