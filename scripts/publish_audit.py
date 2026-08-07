#!/usr/bin/env python
"""Apply the publishing standard to every run that has ever been published.

    scripts/publish_audit.py              # the whole fleet, worst first
    scripts/publish_audit.py --evaluate   # also measure any missing FINAL policy

Every headline number in README/ROADMAP comes from ``ckpt_best.pt``, which
captures the best rolling WINDOW seen during training rather than the policy
the run ended with. On a stable run the two agree. On an unstable one they do
not, and the published number is then a measurement of a transient:

    squad_recon_v6   rolling success ENDED at 0.00   published 0.91 +/- 0.06
    squad_recon_v5   rolling success ENDED at 0.00   published 0.77 +/- 0.08
    squad_v7         rolling success ENDED at 0.41   published 0.92 +/- 0.05

Those numbers are not fabricated — that checkpoint really does score that on
100 held-out episodes. What they are not is a description of a system anyone
can retrain, and the README does not currently say which of the two it means.
Across 18 published runs the peak-vs-published gap averages +8.2 points.

So the standard is: a run publishes only if it gave back less than
``PUBLISH_STABILITY_POINTS`` between its peak and its final decile, the
headline number is the FINAL policy, and ``ckpt_best`` may be quoted beside it
as a peak, labelled as one. This script says which existing runs clear that bar.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT))

from scripts.run_report import PUBLISH_STABILITY_POINTS, deciles, fnum, mean  # noqa: E402


def audit_run(run_dir: Path) -> dict | None:
    """Peak/final rolling success and the published number, or None if unscored."""
    metrics = run_dir / "metrics.csv"
    behavior = run_dir / "behavior.json"
    if not metrics.exists() or not behavior.exists():
        return None
    with metrics.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 20:
        return None
    roll = [v for r in rows if (v := fnum(r, "success_rate_rolling")) is not None]
    if not roll:
        return None
    beh = json.loads(behavior.read_text())
    final_path = run_dir / "behavior_final.json"
    final_eval = None
    if final_path.exists():
        final_eval = json.loads(final_path.read_text()).get("metrics", {}).get("success_rate")
    peak = max(roll)
    final = mean(deciles(rows)[-1], "success_rate_rolling")
    return {
        "run": run_dir.name,
        "peak": peak,
        "final": final,
        "gap": (peak - final) * 100,
        "episodes": beh.get("episodes", 0),
        "published": beh.get("metrics", {}).get("success_rate"),
        "published_ci": beh.get("success_ci95", "?"),
        "final_eval": final_eval,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluate", action="store_true",
                    help="measure the FINAL policy for any run missing behavior_final.json")
    ap.add_argument("--min-episodes", type=int, default=100,
                    help="only audit runs published at this eval size or larger")
    args = ap.parse_args()

    audits = [a for d in sorted(RUNS.iterdir()) if d.is_dir() and (a := audit_run(d))]
    audits = [a for a in audits if a["episodes"] >= args.min_episodes]
    if not audits:
        print(f"no runs published at N>={args.min_episodes}")
        return 0

    if args.evaluate:
        from cohort.training.evaluate import evaluate

        for a in (x for x in audits if x["final_eval"] is None):
            ckpt = RUNS / a["run"] / "ckpt_latest.pt"
            if not ckpt.exists():
                continue
            try:
                s = evaluate(str(ckpt), episodes=args.min_episodes, behavior=True,
                             behavior_path=str(RUNS / a["run"] / "behavior_final.json"))
                a["final_eval"] = s["success_rate"]
            except Exception as exc:  # a space break, a missing scenario — say so, keep going
                print(f"  ! {a['run']}: final policy not scorable ({type(exc).__name__}: {exc})")

    audits.sort(key=lambda a: -a["gap"])
    blocked = [a for a in audits if a["gap"] >= PUBLISH_STABILITY_POINTS]
    # An unstable run is not automatically an OVERSTATED one. fireteam_defend_v7
    # gives back 22 points and still publishes 0.35 against a final decile of
    # 0.42 — its number is if anything pessimistic. Separating the two matters:
    # "unstable, do not publish" is a process verdict, "the number on the README
    # is higher than the policy that run ended with" is a claim about a specific
    # figure, and only the second one calls for a correction.
    overstated = [
        a for a in blocked
        if a["published"] is not None and a["published"] - a["final"] >= 0.10
    ]

    print(f"publishing standard: best-final gap < {PUBLISH_STABILITY_POINTS} pts, "
          f"headline = the FINAL policy\n")
    print(f"{'run':<26}{'peak':>7}{'final':>7}{'gap':>6}  {'published(best)':>16}"
          f"{'final policy':>14}   verdict")
    over = {a["run"] for a in overstated}
    for a in audits:
        fe = f"{a['final_eval']:.2f}" if a["final_eval"] is not None else "not measured"
        if a["run"] in over:
            verdict = "OVERSTATED — correct the number"
        elif a["gap"] >= PUBLISH_STABILITY_POINTS:
            verdict = "unstable — do not headline"
        else:
            verdict = "publishable"
        print(f"{a['run']:<26}{a['peak']:>7.2f}{a['final']:>7.2f}{a['gap']:>6.0f}"
              f"{a['published_ci']:>17}{fe:>14}   {verdict}")

    gaps = [a["gap"] for a in audits]
    print(f"\n{len(blocked)}/{len(audits)} published runs fail the stability gate; "
          f"mean give-back {sum(gaps)/len(gaps):.1f} pts")
    if overstated:
        print(f"\nOf those, {len(overstated)} carry a headline number at least 10 points above "
              "the policy\ntheir run ended with — these describe a transient, not a "
              "reproducible result:")
        for a in overstated:
            print(f"  {a['run']:<26} published {a['published_ci']:<14} "
                  f"but rolling success ended at {a['final']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
