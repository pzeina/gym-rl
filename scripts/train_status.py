#!/usr/bin/env python
"""One-screen status for detached training runs — the cheap way to check in.

Stdlib only (no torch/numpy import), and deliberately terse: reading this costs
a few hundred tokens instead of the ~200k a raw metrics.csv + log tail would.

    scripts/train_status.py              # all recent jobs, one line each
    scripts/train_status.py <run>        # detail for one run
    scripts/train_status.py --all        # include long-finished runs
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

KEY_METRICS = [
    ("success_rate_rolling", "rolling success", "{:.0%}"),
    ("ep_return", "ep return", "{:.2f}"),
    ("ep_length", "ep length", "{:.0f}"),
    ("entropy", "entropy", "{:.2f}"),
    ("human_death_rate", "human death", "{:.0%}"),
    ("false_complete_rate", "false DONE", "{:.0%}"),
]


def alive(pid: int, run: str | None = None) -> bool:
    """Is this pid still OUR training process for ``run``?

    ``kill(pid, 0)`` alone is not enough. Pids get recycled, and every finished
    run keeps its ``.job.json`` — so the moment the OS hands a stale pid to
    something unrelated, a run that ended days ago starts reading as RUNNING,
    then "ends" again when that stranger exits. That produced two spurious
    "training ended" notifications in one session, and it is not only cosmetic:
    ``train.sh`` refuses to launch a run whose recorded pid looks alive, so a
    recycled pid can block a launch outright.

    So when a run name is given, confirm the process is actually carrying
    ``--run-name <run>``. If ``ps`` cannot be consulted, trust the job file
    rather than declaring a live run dead.
    """
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False
    if run is None:
        return True
    try:
        out = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    tokens = out.split()
    return any(b == run for a, b in itertools.pairwise(tokens) if a == "--run-name")


def read_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def fnum(row: dict, key: str) -> float | None:
    """Parse a metric cell; NaN (iterations with no completed episode) reads as missing."""
    try:
        v = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return v if v == v else None


def job_of(run_dir: Path) -> dict | None:
    path = run_dir / ".job.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def total_steps_of(run_dir: Path, job: dict | None) -> int:
    if job and job.get("total_steps"):
        return int(job["total_steps"])
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            return int(json.loads(cfg.read_text()).get("total_steps", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 0


def dur(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def state_of(run_dir: Path, job: dict | None, rows: list[dict], total: int) -> str:
    steps = fnum(rows[-1], "env_steps") if rows else 0
    if job and alive(job.get("pid", -1), run_dir.name):
        return "RUNNING"
    log = Path(job["log"]) if job and job.get("log") else None
    if log and log.exists() and "Traceback" in log.read_text()[-4000:]:
        return "FAILED"
    if total and steps and steps >= total * 0.999:
        return "DONE"
    if job:
        return "STOPPED"
    return "done" if steps else "empty"


def summarize(run_dir: Path) -> dict:
    job = job_of(run_dir)
    rows = read_rows(run_dir)
    total = total_steps_of(run_dir, job)
    steps = int(fnum(rows[-1], "env_steps") or 0) if rows else 0
    sps = fnum(rows[-1], "sps") if rows else None
    state = state_of(run_dir, job, rows, total)
    eta = ""
    if state == "RUNNING" and sps and total > steps:
        eta = dur((total - steps) / sps)
    return {
        "run": run_dir.name,
        "state": state,
        "steps": steps,
        "total": total,
        "pct": (steps / total * 100) if total else 0.0,
        "rolling": fnum(rows[-1], "success_rate_rolling") if rows else None,
        "eta": eta,
        "job": job,
        "rows": rows,
        "run_dir": run_dir,
        "mtime": (run_dir / "metrics.csv").stat().st_mtime if (run_dir / "metrics.csv").exists() else 0,
    }


def line(s: dict) -> str:
    roll = f"{s['rolling']:.0%}" if s["rolling"] is not None else "  - "
    prog = f"{s['steps']:>9,}/{s['total']:<9,}" if s["total"] else f"{s['steps']:>9,}"
    eta = f"  eta {s['eta']}" if s["eta"] else ""
    return f"  {s['state']:<8} {s['run']:<26} {prog} {s['pct']:>3.0f}%  succ {roll}{eta}"


def overview(include_all: bool) -> None:
    # not-archive-aware: this is the live/recent training board. Nothing in
    # runs/archive/ is training, and parsing 96 finished metrics.csv to say so
    # is the cost this whole check-in exists to avoid.
    runs = [summarize(d) for d in RUNS.iterdir() if d.is_dir()]
    live = [s for s in runs if s["state"] == "RUNNING"]
    recent = sorted(
        (s for s in runs if s["state"] != "RUNNING" and (include_all or s["mtime"] > time.time() - 3 * 86400)),
        key=lambda s: s["mtime"],
        reverse=True,
    )
    print(f"training status — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nlive ({len(live)}):" if live else "\nlive (0): nothing training right now")
    for s in sorted(live, key=lambda s: s["run"]):
        print(line(s))
    cap = None if include_all else 8
    print(f"\nrecent ({'all' if include_all else 'last 3 days'}):")
    for s in recent[:cap]:
        print(line(s))
    if cap and len(recent) > cap:
        print(f"  … {len(recent) - cap} more (--all)")
    print(boards_line())
    print("\ndetail: scripts/train_status.py <run>   |   analysis: scripts/run_report.py <run>")


def boards_line() -> str:
    """One line on the boards — read from the state file, never re-rendered here.

    train.sh refreshes the HTML the moment a run lands, but only a session can
    push it to claude.ai. This is where that gap becomes visible instead of
    silently going stale.
    """
    state_path = RUNS / ".boards.json"
    if not state_path.exists():
        return "\nboards:   never rendered — scripts/update_boards.py"
    try:
        state = json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return "\nboards:   state file unreadable — scripts/update_boards.py"
    sha = state.get("data_sha")
    stale = [n for n, b in state.get("boards", {}).items() if b.get("published_sha") != sha]
    when = (state.get("rendered_at") or "?").replace("T", " ")
    if stale:
        return f"\nboards:   refreshed {when} · PUBLISH PENDING ({', '.join(sorted(stale))}) → /boards"
    return f"\nboards:   refreshed {when} · published artifacts current"


def detail(name: str) -> int:
    from scripts.fleet_status import find_run

    run_dir = find_run(name, RUNS)
    if run_dir is None:
        print(f"no such run: runs/{name}", file=sys.stderr)
        return 2
    if run_dir.parent.name == "archive":
        print(f"(archived — runs/archive/{name})")
    s = summarize(run_dir)
    job, rows = s["job"], s["rows"]
    print(f"{name}: {s['state']}")
    if job:
        el = time.time() - job.get("started", time.time())
        print(f"  pid {job.get('pid')}  started {job.get('started_human','?')}  elapsed {dur(el)}")
        print(f"  args {' '.join(job.get('args', []))}")
        print(f"  log  {Path(job['log']).name if job.get('log') else '-'}")
    if not rows:
        print("  no metrics yet")
    else:
        print(f"  steps {s['steps']:,}/{s['total']:,} ({s['pct']:.0f}%)" + (f"  eta {s['eta']}" if s["eta"] else ""))
        last = rows[-1]
        print("  latest:", "  ".join(
            f"{label} {fmt.format(v)}"
            for key, label, fmt in KEY_METRICS
            if (v := fnum(last, key)) is not None
        ))
        # coarse trend: first / middle / last fifth of the run so far
        n = len(rows)
        if n >= 10:
            def avg(seg, key):
                vals = [v for r in seg if (v := fnum(r, key)) is not None]
                return sum(vals) / len(vals) if vals else float("nan")
            fifth = max(1, n // 5)
            segs = [("start", rows[:fifth]), ("mid", rows[n // 2 - fifth // 2: n // 2 + fifth // 2 + 1]), ("end", rows[-fifth:])]
            print("  trend:  " + "   ".join(
                f"{tag} succ {avg(seg,'success_rate_rolling'):.0%} ret {avg(seg,'ep_return'):.1f}" for tag, seg in segs
            ))
    log = Path(job["log"]) if job and job.get("log") else None
    if log and log.exists():
        tail = [ln.rstrip() for ln in log.read_text().splitlines() if ln.strip()][-6:]
        print("  log tail:")
        for ln in tail:
            print(f"    {ln[:150]}")
    beh = run_dir / "behavior.json"
    if beh.exists():
        try:
            b = json.loads(beh.read_text())
            m = b.get("metrics", {})
            print(f"  behavior.json: success {b.get('success_ci95','?')} over {b.get('episodes','?')} eps"
                  f" | obey_lat {m.get('obedience_latency_mean', float('nan')):.1f}"
                  f" | report P/R {m.get('report_precision', 0):.2f}/{m.get('report_recall', 0):.2f}")
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 0


def main() -> int:
    # --is-running <run>: exit 0 if that run is genuinely training. The shell
    # scripts used bare `kill -0` and inherited the recycled-pid bug with it,
    # so they ask here instead of each keeping their own copy of the check.
    if "--is-running" in sys.argv[1:]:
        rest = sys.argv[sys.argv.index("--is-running") + 1:]
        if not rest:
            print("usage: train_status.py --is-running <run>", file=sys.stderr)
            return 2
        job = job_of(RUNS / rest[0])
        return 0 if job and alive(job.get("pid", -1), rest[0]) else 1
    args = [a for a in sys.argv[1:] if a != "--all"]
    if args:
        return detail(args[0])
    overview("--all" in sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
