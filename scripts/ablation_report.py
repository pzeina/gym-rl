#!/usr/bin/env python
"""The B3 ablation, on whatever build the three arms were actually trained at.

    scripts/ablation_report.py                                  # the v1.19 trio
    scripts/ablation_report.py squad_v10 squad_nomask_v1 squad_flat_v1
    scripts/ablation_report.py full_a full_b full_c nomask_a nomask_b nomask_c \\
                               flat_a flat_b flat_c              # 3 seeds per arm

Does the chain of command earn its keep? The 2026-08-06 answer — 3 arms x 3
seeds on Box(137) — was **yes, but not on the axis people expect**:

    N=100 success   full 0.92 ± 0.01   nomask 0.91 ± 0.03   flat 0.85 ± 0.06
    defeats / 100        5.0                4.7                 11.0
    doctrine-valid       100% (by construction)  0.395 ± 0.079   n/a, no orders

So success barely separates `full` from `nomask` and separates `flat` by 7
points; what separates them is **robustness** (the flat arm wipes 2.2x as often)
and **interpretability** (doctrine-valid traffic by construction, and completion
reporting surviving at all). A replication that reads only the success column
will conclude the hierarchy does nothing, and will be reading the wrong column.

The completion cell is deliberately absent from that summary. It is the one the
original's own three seeds disagree on, so a mean is the wrong way to carry it;
it is printed per seed at the foot of every report, straight off the committed
corpora rather than transcribed here.

Those runs are on a dead observation era — they no longer load and cannot be
re-evaluated — but their N=30 behavior corpora are committed, which is what lets
the per-seed block be computed rather than quoted. This prints the same axes for
a trio on the current build, with each rate's Wilson interval and Fisher's exact
test between arms, so the replication can be read at the strength it actually
has: **one seed per arm** against three — enough for the cells the original's
seeds agree on, not enough for the one they do not.
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
from scripts.exact_tests import fisher_two_sided as _fisher  # noqa: E402

#: (run-name default, arm label, what the arm removes)
ARMS = (
    ("squad_v10", "full", "orders + doctrine masks — the shipped system"),
    ("squad_nomask_v1", "nomask", "orders, doctrine masks removed"),
    ("squad_flat_v1", "flat", "no orders at all; every agent gets the OPORD"),
)

ORIGINAL_SEEDS = (3, 5, 7)
ORIGINAL_EPISODES = 30

#: The 2026-08-06 original, per seed, off the committed behavior corpora in
#: ``runs/squad_abl_{arm}_s{seed}/behavior.json`` — N=30 protocol episodes each.
#: Every overstatement this repo has corrected was a hand-kept number, so
#: ``tests/test_ablation_report.py`` recomputes every cell below from those nine
#: files: if a constant here or a corpus there drifts, the claim breaks loudly.
ORIGINAL: dict[str, dict[str, tuple | None]] = {
    "full": {
        "successes": (30, 26, 28),
        "done": (173, 210, 2),
        "doctrine_preferred": (0.4517, 0.5378, 0.3869),
    },
    "nomask": {
        "successes": (28, 26, 28),
        "done": (1, 0, 0),
        "doctrine_preferred": (0.1478, 0.2082, 0.1410),
    },
    "flat": {
        # DONE is a *report*, not an order, so the flat arm can transmit it and
        # does — 0/2/1 across the seeds. This cell is measured, not unavailable;
        # calling it n/a would hide the arm where completion reporting is the
        # only C2 channel left and still does not happen.
        "successes": (27, 26, 27),
        "done": (0, 2, 1),
        "doctrine_preferred": None,  # no orders exist to judge
    },
}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _facts(run: str) -> dict | None:
    path = run_dir(run) / "behavior_final.json"
    try:
        b = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    m = b.get("metrics", {})
    n = b.get("episodes") or 0
    try:
        seed = json.loads((run_dir(run) / "config.json").read_text()).get("seed")
    except (OSError, json.JSONDecodeError):
        seed = None
    success = m.get("success_rate")
    timeout = m.get("timeout_rate") or 0.0
    # The original's headline robustness cell. Not stored directly: an episode
    # that neither succeeded nor ran the clock out is a cohort that was killed.
    defeat = None if success is None else max(0.0, 1.0 - success - timeout)
    return {
        "run": run,
        "n": n,
        "seed": seed,
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
        # The axis a monitor on the radio can see without the environment's
        # internals: of the operations that were won, how many said so.
        "announced_rate": m.get("successes_announced_rate"),
        "closed_by_root": m.get("closed_on_root_report_rate"),
    }


def _fmt(v, spec="{:.3f}") -> str:
    return "—" if v is None else spec.format(v)


def _mean_sd(xs: tuple[float, ...]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def original_block() -> str:
    """The original's cells one seed at a time, and which of them one seed settles.

    Two of them behave differently across seeds, and the difference is the whole
    instruction for reading a one-seed replication:

    * **doctrine-preferred separates seed by seed** — the full arm's worst seed
      (0.387) is above the nomask arm's best (0.208). Any single seed reproduces
      the ordering, so one seed can settle it.
    * **the completion cell does not** — the full arm's mean of 128.3 is 173 and
      210 and a 2, and that 2 is the flat arm's own maximum. One seed of the full
      arm therefore lands on "the channel is alive" twice in three and on a number
      indistinguishable from the ablated arms once in three. A one-seed DONE cell
      neither confirms nor refutes the original, whichever way it falls.

    Carrying a bimodal cell as ``128 ± 111`` invites exactly the misreading that
    a mean is supposed to prevent, so the seeds are printed and the mean follows
    them rather than standing in for them.
    """
    seeds = "/".join(str(s) for s in ORIGINAL_SEEDS)
    lines = [f"the 2026-08-06 original, per seed — seeds {seeds}, "
             f"N={ORIGINAL_EPISODES} protocol episodes each",
             "",
             f"  {'arm':<8}{'successes':>11}{'DONE claims':>24}{'doctrine-preferred':>30}"]
    for _, label, _ in ARMS:
        cells = ORIGINAL[label]
        succ_s = " ".join(f"{v:3d}" for v in cells["successes"])
        done_m, _ = _mean_sd(cells["done"])
        done_s = " ".join(f"{v:4d}" for v in cells["done"]) + f"  ({done_m:.1f})"
        pref = cells["doctrine_preferred"]
        if pref is None:
            pref_s = "— no orders exist"
        else:
            pref_m, _ = _mean_sd(pref)
            pref_s = " ".join(f"{v:.3f}" for v in pref) + f"  ({pref_m:.3f})"
        lines.append(f"  {label:<8}{succ_s:>11}{done_s:>24}{pref_s:>30}")
    lines += [
        "",
        "  doctrine-preferred SEPARATES SEED BY SEED: the full arm's worst seed",
        f"    ({min(ORIGINAL['full']['doctrine_preferred']):.3f}) is above the nomask "
        f"arm's best ({max(ORIGINAL['nomask']['doctrine_preferred']):.3f}).",
        "  the completion cell DOES NOT: the full arm's seed 7 claimed "
        f"{ORIGINAL['full']['done'][-1]}, which is the",
        f"    flat arm's own maximum ({max(ORIGINAL['flat']['done'])}). "
        "Its 128.3 mean is 173 and 210 and a 2.",
    ]
    return "\n".join(lines)


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

    ns = sorted({f["n"] for f, _, _ in rows})
    n_label = f"N={ns[0]}" if len(ns) == 1 else "N varies: " + "/".join(map(str, ns))
    line(f"success ({n_label})", "success")
    print(f"{'  95% CI':<28}" + "".join(f"{f['ci'] or '—':>16}" for f, _, _ in rows))
    line("defeats / 100  ROBUSTNESS", "defeat_per_100", "{:.1f}")
    line("ran the clock out", "timeout")
    line("root death rate", "root_death")
    print()
    line("orders / episode", "orders", "{:.2f}")
    line("doctrine-valid  INTERPRET", "doctrine_allowed")
    line("of which preferred", "doctrine_preferred")
    line(f"DONE reports ({n_label})", "done_reports", "{:.0f}")
    line("  of which rejected", "done_rejected", "{:.0f}")
    print()
    line("wins announced", "announced", "{:.0f}")
    line("  as a rate of wins", "announced_rate", "{:.3f}")
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

    print()
    print(original_block())

    print("\nOne seed per arm. The 2026-08-06 result is three, its separation was in\n"
          "robustness and interpretability rather than in success — read those rows\n"
          "first — and its three seeds do not agree cell by cell. On doctrine they do,\n"
          "so one seed settles that row. On DONE they do not: read a one-seed\n"
          "completion number as one draw from 173/210/2, not as a replication of 128.")
    return 0


def seed_report(runs: list[str]) -> int:
    """Nine runs, three per arm: the strength the 2026-08-06 original ran at.

    Per-seed cells are printed for every axis and the mean follows them rather
    than standing in for them — the original's completion cell is bimodal
    (173/210/2) and averaging such a cell into ``128 ± 111`` is the misreading
    the per-seed layout exists to prevent. Success is additionally pooled
    across seeds for Fisher's exact test between arms, which the one-seed
    replication could not support.
    """
    arm_runs = {label: runs[i * 3:i * 3 + 3] for i, (_, label, _) in enumerate(ARMS)}
    facts = {label: [_facts(r) for r in rs] for label, rs in arm_runs.items()}
    missing = [r for label, rs in arm_runs.items()
               for r, f in zip(rs, facts[label], strict=True) if f is None]
    if missing:
        print(f"not evaluated yet: {', '.join(missing)}")
        print("(scripts/publish_baseline.py <run> scores a landed run at N=100)")
        return 1

    print("B3 ablation at three seeds per arm — does the chain of command earn its keep?\n")
    for _, label, what in ARMS:
        seeds = "/".join(str(f["seed"]) if f["seed"] is not None else "?" for f in facts[label])
        names = ", ".join(f["run"] for f in facts[label])
        print(f"  {label:<7} seeds {seeds:<9} {what}\n          {names}")
    print()

    ns = sorted({f["n"] for fs in facts.values() for f in fs})
    n_label = f"N={ns[0]} each" if len(ns) == 1 else "N varies: " + "/".join(map(str, ns))
    width = 26
    print(f"per-seed, mean follows the seeds ({n_label})")
    print(f"{'axis':<26}" + "".join(f"{label:>{width}}" for _, label, _ in ARMS))
    print("-" * (26 + width * len(ARMS)))

    def cell(fs: list[dict], key: str, spec: str) -> str:
        vals = [f[key] for f in fs]
        if all(v is None for v in vals):
            return "—"
        shown = " ".join("—" if v is None else spec.format(v) for v in vals)
        known = [v for v in vals if v is not None]
        return f"{shown}  ({sum(known) / len(known):{spec[2:-1]}})"

    def line(name: str, key: str, spec="{:.2f}"):
        print(f"{name:<26}" + "".join(f"{cell(facts[label], key, spec):>{width}}"
                                      for _, label, _ in ARMS))

    line("success", "success")
    line("defeats / 100  ROBUSTNESS", "defeat_per_100", "{:.1f}")
    line("root death rate", "root_death")
    print()
    line("orders / episode", "orders", "{:.1f}")
    line("doctrine-valid  INTERPRET", "doctrine_allowed")
    line("of which preferred", "doctrine_preferred")
    line("DONE reports", "done_reports", "{:.0f}")
    line("closed by the root itself", "closed_by_root")
    print()

    def pooled(label: str) -> tuple[int, int]:
        fs = facts[label]
        return sum(f["successes"] or 0 for f in fs), sum(f["n"] for f in fs)

    k0, n0 = pooled("full")
    for _, label, _ in ARMS[1:]:
        k1, n1 = pooled(label)
        p = _fisher(k0, n0 - k0, k1, n1 - k1)
        lo0, hi0 = _wilson(k0, n0)
        lo1, hi1 = _wilson(k1, n1)
        overlap = not (hi0 < lo1 or hi1 < lo0)
        print(f"full vs {label:<7} success pooled over seeds {k0}/{n0} vs {k1}/{n1}, "
              f"Fisher p = {p:.3f}"
              + ("  — intervals OVERLAP, not a difference" if overlap else "  — separated"))

    print()
    print(original_block())
    print("\nThree seeds per arm — the original's own strength, on one tree. Bimodal\n"
          "cells (the original's DONE column, nomask's false-claim behaviour) are\n"
          "readable here per seed; a mean printed after three seeds is a summary,\n"
          "a mean printed instead of them is a hiding place.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=f"{__doc__}\n{original_block()}\n",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*",
                    help="full nomask flat (defaults to the v1.19 trio), or nine runs "
                         "— three per arm in that order — for the per-seed report")
    args = ap.parse_args()
    runs = args.runs or [r for r, _, _ in ARMS]
    if len(runs) == 9:
        return seed_report(runs)
    if len(runs) != 3:
        print("give three runs (full nomask flat) or nine (three per arm in that order)")
        return 2
    return report(runs)


if __name__ == "__main__":
    raise SystemExit(main())
