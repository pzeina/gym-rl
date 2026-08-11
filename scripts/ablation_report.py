#!/usr/bin/env python
"""The B3 ablation, on whatever build the three arms were actually trained at.

    scripts/ablation_report.py                                  # the v1.19 trio
    scripts/ablation_report.py squad_v10 squad_nomask_v1 squad_flat_v1

Does the chain of command earn its keep? The 2026-08-06 answer — 3 arms x 3
seeds on Box(137) — was **yes, but not on the axis people expect**:

    N=100 success   full 0.92 ± 0.01   nomask 0.91 ± 0.03   flat 0.85 ± 0.06
    defeats / 100        5.0                4.7                 11.0
    doctrine-valid       100% (by construction)  33-48%          n/a, no orders
    DONE claims / 30 eps 128                ~0                   n/a

So success barely separates `full` from `nomask` and separates `flat` by 7
points; what separates them is **robustness** (the flat arm wipes 2.2x as often)
and **interpretability** (doctrine-valid traffic by construction, and completion
reporting surviving at all). A replication that reads only the success column
will conclude the hierarchy does nothing, and will be reading the wrong column.

Those runs are on a dead observation era — they no longer load and cannot be
re-evaluated. This prints the same axes for a trio on the current build, with
each rate's Wilson interval and Fisher's exact test between arms, so the
replication can be read at the strength it actually has: **one seed per arm**,
against the original's three.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.baseline import run_dir  # noqa: E402

#: (run-name default, arm label, what the arm removes)
ARMS = (
    ("squad_v10", "full", "orders + doctrine masks — the shipped system"),
    ("squad_nomask_v1", "nomask", "orders, doctrine masks removed"),
    ("squad_flat_v1", "flat", "no orders at all; every agent gets the OPORD"),
)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact on [[a,b],[c,d]] — no scipy in this venv."""
    from math import comb

    n = a + b + c + d
    row1, col1 = a + b, a + c

    def prob(x: int) -> float:
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)

    observed = prob(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return min(1.0, sum(p for x in range(lo, hi + 1)
                        if (p := prob(x)) <= observed * (1 + 1e-9)))


def _facts(run: str) -> dict | None:
    path = run_dir(run) / "behavior_final.json"
    try:
        b = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    m = b.get("metrics", {})
    n = b.get("episodes") or 0
    success = m.get("success_rate")
    timeout = m.get("timeout_rate") or 0.0
    # The original's headline robustness cell. Not stored directly: an episode
    # that neither succeeded nor ran the clock out is a cohort that was killed.
    defeat = None if success is None else max(0.0, 1.0 - success - timeout)
    return {
        "run": run,
        "n": n,
        "successes": m.get("successes"),
        "success": success,
        "ci": b.get("success_ci95"),
        "defeat_per_100": None if defeat is None else defeat * 100,
        "timeout": timeout,
        "root_death": m.get("human_death_rate"),
        "orders": m.get("orders_per_episode"),
        "doctrine_allowed": m.get("doctrine_allowed_rate"),
        "doctrine_preferred": m.get("doctrine_preference_rate"),
        "done_reports": m.get("done_reports"),
        "done_rejected": m.get("done_rejected"),
        "announced": m.get("successes_announced"),
        "closed_by_root": m.get("closed_on_root_report_rate"),
    }


def _fmt(v, spec="{:.3f}") -> str:
    return "—" if v is None else spec.format(v)


def report(runs: list[str]) -> int:
    rows = [(_facts(r), label, what) for r, (_, label, what) in zip(runs, ARMS, strict=False)]
    missing = [runs[i] for i, (f, _, _) in enumerate(rows) if f is None]
    if missing:
        print(f"not evaluated yet: {', '.join(missing)}")
        print("(scripts/publish_baseline.py <run> scores a landed run at N=100)")
    rows = [(f, label, what) for f, label, what in rows if f]
    if not rows:
        return 1

    print("B3 ablation — does the chain of command earn its keep?\n")
    for f, label, what in rows:
        print(f"  {label:<7} {f['run']:<18} {what}")
    print()

    print(f"{'axis':<28}" + "".join(f"{label:>16}" for _, label, _ in rows))
    print("-" * (28 + 16 * len(rows)))

    def line(name: str, key: str, spec="{:.3f}"):
        print(f"{name:<28}" + "".join(f"{_fmt(f[key], spec):>16}" for f, _, _ in rows))

    line("success (N=100)", "success")
    print(f"{'  95% CI':<28}" + "".join(f"{f['ci'] or '—':>16}" for f, _, _ in rows))
    line("defeats / 100  ROBUSTNESS", "defeat_per_100", "{:.1f}")
    line("ran the clock out", "timeout")
    line("root death rate", "root_death")
    print()
    line("orders / episode", "orders", "{:.2f}")
    line("doctrine-valid  INTERPRET", "doctrine_allowed")
    line("of which preferred", "doctrine_preferred")
    line("DONE reports (N=100)", "done_reports", "{:.0f}")
    line("  of which rejected", "done_rejected", "{:.0f}")
    print()
    line("wins announced", "announced", "{:.0f}")
    line("closed by the root itself", "closed_by_root")

    print()
    base = rows[0][0]
    for f, label, _ in rows[1:]:
        if base["successes"] is None or f["successes"] is None:
            continue
        p = _fisher(base["successes"], base["n"] - base["successes"],
                    f["successes"], f["n"] - f["successes"])
        lo1, hi1 = _wilson(base["successes"], base["n"])
        lo2, hi2 = _wilson(f["successes"], f["n"])
        overlap = not (hi1 < lo2 or hi2 < lo1)
        print(f"full vs {label:<7} success {base['successes']}/{base['n']} vs "
              f"{f['successes']}/{f['n']}, Fisher p = {p:.3f}"
              + ("  — intervals OVERLAP, not a difference" if overlap else "  — separated"))

    print("\nOne seed per arm. The 2026-08-06 result is three, and its separation was in\n"
          "robustness and interpretability rather than in success — read those rows first.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="full, nomask, flat (defaults to the v1.19 trio)")
    args = ap.parse_args()
    runs = args.runs or [r for r, _, _ in ARMS]
    if len(runs) != 3:
        print("give exactly three runs: full nomask flat")
        return 2
    return report(runs)


if __name__ == "__main__":
    raise SystemExit(main())
