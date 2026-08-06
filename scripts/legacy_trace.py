#!/usr/bin/env python
"""Record a replayable episode trace for a checkpoint from ANY era — no retrain.

A breaking cycle makes an old checkpoint unloadable: the net's first layer wants
the observation width it was trained on. The policy is not lost, though — the
code that produced it is still in git. So instead of retraining or writing a
shim that would silently mean the wrong thing (both the observation layout AND
the action indices moved between eras), this checks out the run's own revision
in a throwaway git worktree, records the episode THERE, and writes the trace as
plain JSON.

A trace is data, not a model: once written it replays in the current dashboard
forever, whatever the spaces do next.

    scripts/legacy_trace.py squad_v4                 # → runs/squad_v4/traces/…
    scripts/legacy_trace.py squad_v4 --seed 7 --scenario squad

Cost is one episode of simulation (seconds) plus a worktree checkout. No
training, no GPU, nothing downloaded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cohort.env.actions import N_ACTIONS  # noqa: E402
from cohort.env.observations import OBS_DIM  # noqa: E402

#: Observation width → the tag whose code produced it. Each entry is a release
#: whose `cohort/env/observations.py` computes that OBS_DIM; verified by
#: checking the tag out and importing it.
ERA_REF: dict[int, str] = {
    137: "v1.8.0",   # v1.4-v1.8: MICAT set + humans, before control measures
    166: "v1.9.0",   # A5: control measures, order timing, formations, sync
}

#: Fields the current frontend expects that older traces predate. A trace is
#: replayed, never re-simulated, so filling them with empties is honest: the
#: era genuinely had no phase lines, no formations.
TRACE_DEFAULTS: dict = {"waypoints": [], "phase_lines": []}

RECORDER = r"""
import json, os, sys
WT, REPO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
scenario, ckpt, seed = sys.argv[4], sys.argv[5], int(sys.argv[6])
# the worktree's cohort package must win over the checked-out one
sys.path = [p for p in sys.path if p not in ("", REPO, os.getcwd())]
sys.path.insert(0, WT)
import cohort
assert cohort.__file__.startswith(WT), "resolved the wrong cohort: " + cohort.__file__
from cohort.env.observations import OBS_DIM
from cohort.env.actions import N_ACTIONS
import cohort.viz.dashboard as D
trace = D.record_episode(scenario, ckpt, seed)
trace["recorded_spaces"] = {"obs_dim": OBS_DIM, "n_actions": N_ACTIONS}
with open(OUT, "w") as f:
    json.dump(trace, f)
print(json.dumps({"outcome": trace["outcome"], "length": trace["length"],
                  "obs_dim": OBS_DIM, "n_actions": N_ACTIONS}))
"""


def checkpoint_spaces(path: Path) -> tuple[int, int]:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return int(ckpt["obs_dim"]), int(ckpt["n_actions"])


def record(run: str, seed: int, scenario: str | None, kind: str, runs_dir: Path) -> Path:
    run_dir = runs_dir / run
    ckpt = run_dir / f"ckpt_{kind}.pt"
    if not ckpt.is_file():
        raise SystemExit(f"no ckpt_{kind}.pt in {run_dir}")
    if scenario is None:
        cfg = json.loads((run_dir / "config.json").read_text())
        scenario = cfg["scenario"]

    obs_dim, n_actions = checkpoint_spaces(ckpt)
    out_dir = run_dir / "traces"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{scenario}_{kind}_seed{seed}.json"

    if (obs_dim, n_actions) == (OBS_DIM, N_ACTIONS):
        # current era: no worktree needed
        from cohort.viz.dashboard import record_episode

        trace = record_episode(scenario, str(ckpt), seed)
        trace["recorded_spaces"] = {"obs_dim": OBS_DIM, "n_actions": N_ACTIONS}
        out.write_text(json.dumps(trace))
        print(f"recorded in-process: {trace['outcome']} in {trace['length']} steps")
        return out

    ref = ERA_REF.get(obs_dim)
    if ref is None:
        raise SystemExit(
            f"{run}: Box({obs_dim}) is not a known era "
            f"(known: {', '.join(f'{k}→{v}' for k, v in ERA_REF.items())}). "
            "Add its release tag to ERA_REF."
        )

    with tempfile.TemporaryDirectory(prefix="cohort-era-") as tmp:
        wt = Path(tmp) / "wt"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt), ref],
            cwd=REPO, check=True, capture_output=True,
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", RECORDER, str(wt), str(REPO), str(out),
                 scenario, str(ckpt.resolve()), str(seed)],
                cwd=str(wt), check=True, capture_output=True, text=True,
            )
            info = json.loads(proc.stdout.strip().splitlines()[-1])
            print(
                f"recorded at {ref} (Discrete({info['n_actions']})/Box({info['obs_dim']})): "
                f"{info['outcome']} in {info['length']} steps"
            )
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt)],
                cwd=REPO, check=False, capture_output=True,
            )

    # fill in fields the era predates, so the current player can read it
    trace = json.loads(out.read_text())
    for key, default in TRACE_DEFAULTS.items():
        trace.setdefault(key, default)
    trace["recorded_at_ref"] = ref
    out.write_text(json.dumps(trace))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Record a replayable trace from any era.")
    p.add_argument("run", nargs="+", help="run directory name(s) under runs/")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--scenario", default=None, help="override config.json")
    p.add_argument("--kind", default="best", choices=("best", "latest"))
    p.add_argument("--runs-dir", default=str(REPO / "runs"))
    args = p.parse_args()
    if args.scenario and len(args.run) > 1:
        raise SystemExit("--scenario applies to a single run")
    for run in args.run:
        out = record(run, args.seed, args.scenario, args.kind, Path(args.runs_dir))
        print(f"trace → {out}")


if __name__ == "__main__":
    main()
