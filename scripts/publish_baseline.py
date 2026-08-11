#!/usr/bin/env python
"""Score every baseline member at publication size, and write its artifacts.

    scripts/publish_baseline.py                  # every member that needs it
    scripts/publish_baseline.py squad_v10 …      # named runs only
    scripts/publish_baseline.py --episodes 100

Training writes an N=20 evaluation of each checkpoint when a run exits — a smoke
test, not evidence. This takes each member of ``runs/BASELINE.json`` to N=100 on
BOTH checkpoints and, on the final policy, also writes the episode GIF and the
full radio transcript, because "the simulations are available" should mean a
file on disk and not an instruction to go and generate one.

Two refusals, both learned the hard way:

* **Never overwrite a larger evaluation with a smaller one.** `squad_v7`'s
  committed N=100 artifact was once replaced by an N=20 re-run and caught only
  by reading `git diff` before the commit. An artifact whose ``episodes`` is
  greater than what we are about to write is left exactly where it is.
* **Never touch a run that is still training.** Its process owns that directory.

Run it detached (``nohup``) or in a background shell: it simulates thousands of
episodes and costs no model tokens while it does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT))

from scripts import baseline  # noqa: E402
from scripts.train_status import summarize  # noqa: E402

#: (checkpoint file, artifact file, also write GIF + transcript)
TARGETS = (
    ("ckpt_latest.pt", "behavior_final.json", True),
    ("ckpt_best.pt", "behavior.json", False),
)


def _episodes(path: Path) -> int:
    try:
        return int(json.loads(path.read_text()).get("episodes", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def publish(run: str, episodes: int, *, force: bool = False) -> int:
    from cohort.training.evaluate import evaluate

    d = baseline.run_dir(run)
    if not d.is_dir():
        print(f"  ! {run}: no run directory")
        return 1
    if summarize(d).get("state") == "RUNNING":
        print(f"  · {run}: still training, skipped")
        return 0

    problems = 0
    for ckpt_name, artifact, with_media in TARGETS:
        ckpt = d / ckpt_name
        out = d / artifact
        if not ckpt.is_file():
            print(f"  ! {run}: no {ckpt_name}")
            problems += 1
            continue
        have = _episodes(out)
        if have >= episodes and not force:
            # >= and not >: an equal-N re-evaluation is a DIFFERENT SAMPLE
            # replacing a committed, possibly published number, and it buys
            # nothing. Caught by smoke-testing this script on squad_v9, which
            # duly overwrote its N=100 artifact, its transcript and its GIF —
            # the squad_v7 incident, reproduced by the tool built to avoid it.
            print(f"  · {run}/{artifact}: already N={have} — leaving it alone "
                  f"(--force to re-measure at N={episodes})")
            continue
        extra = {}
        if with_media:
            extra = {"gif_path": str(d / "eval.gif"),
                     "transcript_path": str(d / "eval_transcript.txt")}
        try:
            stats = evaluate(str(ckpt), episodes=episodes, behavior=True,
                             behavior_path=str(out), **extra)
        except Exception as exc:
            print(f"  ! {run}/{ckpt_name}: {type(exc).__name__}: {exc}")
            problems += 1
            continue
        print(f"  ✓ {run}/{artifact}: {stats.get('success_ci95')} at N={episodes}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run names (default: every baseline member)")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--force", action="store_true",
                    help="allow replacing a larger evaluation (you had better mean it)")
    args = ap.parse_args()

    names = args.runs or list(baseline.load().get("runs", {}).values())
    print(f"publishing {len(names)} run(s) at N={args.episodes}")
    problems = sum(publish(r, args.episodes, force=args.force) for r in names)
    print("done" if not problems else f"done with {problems} problem(s)")
    # Writing an evaluation is exactly what invalidates the seal (issue #45), and
    # it is meant to: the manifest digests these files so that a spot-check
    # cannot quietly replace a published number. Say so here, or the next
    # `baseline.py` reads as sixteen mysterious digest mismatches.
    if baseline.load().get("artifacts"):
        print("re-seal the manifest over the new artifacts:  "
              "scripts/baseline.py --seal")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
