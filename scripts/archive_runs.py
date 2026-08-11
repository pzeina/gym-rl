#!/usr/bin/env python
"""File superseded runs into ``runs/archive/`` — a move, never a delete.

    scripts/archive_runs.py                # dry run: what would move, and why
    scripts/archive_runs.py --apply        # do it, with git mv so it reads as renames

``runs/`` accumulated 100 directories, and the eight that constitute the current
baseline were invisible among them. Everything not named by ``runs/BASELINE.json``
— neither a member nor a deliberately kept reference — moves down a level.

**Why moved and not deleted.** These runs are the evidence behind published
claims. ROADMAP cites `squad_v7`'s collapse, `fireteam_defend_v6`'s 0.51, the D4
pair, the B3 ablation arms; the publish audit reads their curves; `--series`
reads a metric across every generation of a scenario. Delete them and every one
of those becomes a story about a measurement instead of the measurement. Readers
resolve through ``fleet_status.find_run`` / ``run_report.run_dir``, so a name
that resolved before still resolves after.

Three refusals, each one a way this could go wrong:

* a run that is **training right now** never moves — its process is writing into
  that directory;
* a run named in the manifest never moves, member or reference alike;
* nothing moves without ``--apply``, and what would move is printed first.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT))

from scripts import baseline  # noqa: E402
from scripts.train_status import summarize  # noqa: E402


def keep_set() -> dict[str, str]:
    """run name -> why it stays in ``runs/``."""
    manifest = baseline.load()
    keep = {run: f"baseline member ({scenario})"
            for scenario, run in (manifest.get("runs") or {}).items()}
    for run, why in (manifest.get("referenced_history") or {}).items():
        keep.setdefault(run, f"kept reference — {why}")
    return keep


def candidates() -> tuple[list[Path], dict[str, str], list[str]]:
    keep = keep_set()
    moving, live = [], []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name == "archive" or not (d / "metrics.csv").is_file():
            continue
        if d.name in keep:
            continue
        if summarize(d).get("state") == "RUNNING":
            live.append(d.name)
            continue
        moving.append(d)
    return moving, keep, live


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="actually move them")
    args = ap.parse_args()

    moving, keep, live = candidates()
    print(f"keeping {len(keep)} run(s) in runs/:")
    for run, why in sorted(keep.items()):
        # A kept run sitting in the archive is a fault to flag, not a location
        # to resolve through — so this one asks about runs/ and only runs/.
        mark = " " if (RUNS / run).is_dir() else "!"  # not-archive-aware: see above
        print(f"  {mark} {run:<26} {why}")
    if live:
        print(f"\nnot touching {len(live)} training right now: {', '.join(live)}")
    print(f"\n{len(moving)} run(s) would move to runs/archive/")
    if not args.apply:
        for d in moving:
            print(f"    {d.name}")
        print("\n(dry run — pass --apply to move them)")
        return 0

    archive = RUNS / "archive"
    archive.mkdir(exist_ok=True)
    moved = 0
    for d in moving:
        dest = archive / d.name
        if dest.exists():
            print(f"  ! {d.name}: already in the archive, leaving both alone")
            continue
        r = subprocess.run(["git", "mv", str(d), str(dest)], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode != 0:
            # untracked runs are not in the index; a plain rename is the same move
            d.rename(dest)
        moved += 1
    print(f"moved {moved} run(s) into runs/archive/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
