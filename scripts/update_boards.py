#!/usr/bin/env python
"""Refresh both boards, and remember whether the published copies still match.

    scripts/update_boards.py                   # re-render fleet + program boards
    scripts/update_boards.py --mark-published   # record that they were published
    scripts/update_boards.py --quiet            # one line, for the training log

Called automatically when a run lands (``scripts/train_then_boards.sh``, which
``scripts/train.sh`` wraps every launch in), so the HTML on disk is never stale.
Costs no model tokens: it reads the same committed evaluations the boards read.

**What this cannot do.** Publishing to claude.ai needs the Artifact tool, which
only exists inside a session — no shell can do it. So the automation stops at
"the local boards are current", and records a content digest of what was last
published. When they diverge, ``scripts/train_status.py`` says so and ``/boards``
republishes in one step. That divergence is the whole point of the state file:
without it, "is the artifact stale?" is a question nobody can answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.fleet_status import collect

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "runs" / ".boards.json"

# The published artifacts these files back. Keep in step with /boards.
BOARDS = {
    "fleet": {
        "path": "runs/fleet_board.html",
        "url": "https://claude.ai/code/artifact/a8b68d1c-93b6-4216-82bb-62e08f8c96d5",
        "title": "cohort · fleet board",
    },
    "program": {
        "path": "runs/program_board.html",
        "url": "https://claude.ai/code/artifact/1713413b-62c6-4576-8e25-db280a798cd8",
        "title": "cohort · program board",
    },
}

# Fields that change what a board SAYS about the FLEET. Deliberately excludes
# the render timestamp and a live run's percentage — otherwise every tick of a
# training run would mark the artifacts stale and the signal would mean nothing.
#
# Git HEAD is excluded for the same reason, decided once commits stopped being
# gated one at a time (owner's call, 2026-08-09). The program board prints the
# commits-ahead count and the last tag, so a commit genuinely does change a
# corner of the page — but a "republish me" flag that fires on every commit is
# the noisy signal this digest exists to avoid. The trade: those two header
# numbers can sit stale on the published page until the next fleet change.
# Anything about a RUN — a new evaluation, a gate flipping, a run starting or
# landing — still marks the artifacts stale immediately.
STABLE = (
    "run", "scenario", "success_ci95", "episodes", "policy", "gates_failed",
    "overrides", "env_steps", "obs_dim", "loadable", "state",
)


def data_digest(rows: list[dict]) -> str:
    """A digest of what the boards say about the fleet, not of the rendered bytes."""
    stable = [{k: r.get(k) for k in STABLE} for r in rows]
    payload = json.dumps({"rows": stable}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def read_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def pending(state: dict) -> list[str]:
    """Boards whose published copy no longer matches what is on disk."""
    sha = state.get("data_sha")
    if not sha:
        return []
    return [
        name
        for name, board in state.get("boards", {}).items()
        if board.get("published_sha") != sha
    ]


def render_all(rows: list[dict]) -> dict:
    from scripts import fleet_board, program_board

    renderers = {"fleet": fleet_board.render, "program": program_board.render}
    written = {}
    for name, board in BOARDS.items():
        out = ROOT / board["path"]
        out.write_text(renderers[name](rows))
        written[name] = out
    return written


def main() -> int:
    p = argparse.ArgumentParser(description="Refresh the fleet and program boards.")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument(
        "--mark-published",
        action="store_true",
        help="record the current content as published (run this AFTER publishing)",
    )
    p.add_argument("--quiet", action="store_true", help="one line of output")
    args = p.parse_args()

    state = read_state()

    if args.mark_published:
        sha = state.get("data_sha")
        if not sha:
            print("nothing rendered yet — run scripts/update_boards.py first", file=sys.stderr)
            return 2
        stamp = datetime.now().isoformat(timespec="seconds")
        for board in state.setdefault("boards", {}).values():
            board["published_sha"] = sha
            board["published_at"] = stamp
        write_state(state)
        print(f"marked published: {', '.join(BOARDS)} @ {sha}")
        return 0

    rows = collect(Path(args.runs_dir))
    sha = data_digest(rows)
    written = render_all(rows)

    boards = state.setdefault("boards", {})
    for name, board in BOARDS.items():
        entry = boards.setdefault(name, {})
        entry.update(path=board["path"], url=board["url"], title=board["title"])
    state["data_sha"] = sha
    state["rendered_at"] = datetime.now().isoformat(timespec="seconds")
    write_state(state)

    stale = pending(state)
    if args.quiet:
        note = f"publish pending ({', '.join(stale)})" if stale else "published copies current"
        print(f"boards refreshed @ {sha} — {note}")
        return 0

    for name, path in written.items():
        print(f"{name:<8} → {path.relative_to(ROOT)}")
    evaluated = sum(1 for r in rows if r["success"] is not None)
    live = sum(1 for r in rows if r["loadable"])
    print(f"{len(rows)} runs · {evaluated} evaluated · {live} loadable · digest {sha}")
    if stale:
        print(f"PUBLISH PENDING: {', '.join(stale)} — run /boards in a session to republish")
    else:
        print("published artifacts are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
