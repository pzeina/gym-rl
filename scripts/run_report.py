#!/usr/bin/env python
"""Compact post-hoc digest of a finished run — the ONLY thing the big model reads.

A run's metrics.csv is ~3000 rows x 20 columns. Feeding that to Opus/Fable at
150k context is what makes a training campaign expensive. This collapses it to
~30 lines: config, learning curve by decile, reward-component drift, the
behavioral suite, and (optionally) deltas against a baseline run.

    scripts/run_report.py <run>
    scripts/run_report.py <run> --vs <baseline-run>
    scripts/run_report.py <run> --components      # per-component reward decile table
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

sys.path.insert(0, str(ROOT))
from cohort.metrics import format_order_task_mix  # noqa: E402


def rows_of(run: str) -> list[dict]:
    path = RUNS / run / "metrics.csv"
    if not path.exists():
        raise SystemExit(f"no metrics for run '{run}' ({path})")
    with path.open() as f:
        return list(csv.DictReader(f))


def fnum(row: dict, key: str) -> float | None:
    """Parse a metric cell; NaN (iterations with no completed episode) reads as missing."""
    try:
        v = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return v if v == v else None


def mean(rows: list[dict], key: str) -> float:
    vals = [v for r in rows if (v := fnum(r, key)) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def deciles(rows: list[dict], n: int = 10) -> list[list[dict]]:
    size = max(1, len(rows) // n)
    return [rows[i * size: (i + 1) * size] for i in range(n) if rows[i * size:]]


#: best-minus-final rolling success, in points, past which a run is not a
#: converged result. Set at 15 because squad_v5 — the last pre-v1.10 run that
#: trained cleanly — gave back 5, while every post-v1.10 run measured so far
#: gave back 10 (defend_v8), 17 (defend_v9), 33 (squad_v6) and 68 (fireteam_v7).
COLLAPSE_POINTS = 15


def stability(best: float, final: float) -> str:
    """One line separating "this run converged" from "this run peaked".

    Every published number comes from ``ckpt_best``, which captures the best
    ROLLING WINDOW, not the policy the run ended with. On a run that spikes and
    falls back, evaluating ckpt_best measures the peak and says nothing about
    convergence — squad_v6 read 0.95 at N=20 off a policy whose final decile
    was 65%, and fireteam_v7 evaluates at 0.95 off a final decile of 26%.
    Nothing in the digest used to say so, and the trap caught two consecutive
    sessions, so it is stated on its own line.
    """
    if best != best or final != final:  # NaN
        return "stability  —  (no rolling-success rows)"
    drop = (best - final) * 100
    if drop >= COLLAPSE_POINTS * 2:
        verdict = "COLLAPSED — ckpt_best is a peak, NOT a result; quote both numbers"
    elif drop >= COLLAPSE_POINTS:
        verdict = "UNSTABLE — ckpt_best overstates this run; quote both numbers"
    else:
        verdict = "converged"
    return f"stability  best-final gap {drop:.0f} pts  [{verdict}]"


def curve(rows: list[dict]) -> str:
    """Ten-bucket sparkline of rolling success, plus the numbers that matter."""
    blocks = "▁▂▃▄▅▆▇█"
    vals = [mean(seg, "success_rate_rolling") for seg in deciles(rows)]
    clean = [v for v in vals if v == v]
    lo, hi = (min(clean), max(clean)) if clean else (0.0, 1.0)
    span = (hi - lo) or 1.0
    spark = "".join(blocks[min(7, int((v - lo) / span * 7))] if v == v else " " for v in vals)
    return f"{spark}  ({lo:.0%}→{hi:.0%})"


def report(run: str, show_components: bool) -> dict:
    rows = rows_of(run)
    cfg_path = RUNS / run / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    first, last = deciles(rows)[0], deciles(rows)[-1]
    best = max((v for r in rows if (v := fnum(r, "success_rate_rolling")) is not None), default=float("nan"))

    print(f"== {run} ==")
    print(f"  scenario {cfg.get('scenario','?')}  seed {cfg.get('seed','?')}  "
          f"steps {int(fnum(rows[-1],'env_steps') or 0):,}/{cfg.get('total_steps',0):,}  "
          f"lr {cfg.get('lr','?')}  ent {cfg.get('ent_coef','?')}  n_envs {cfg.get('n_envs','?')}")
    final_roll = mean(last, "success_rate_rolling")
    print(f"  curve   {curve(rows)}   best rolling {best:.0%}   final {final_roll:.0%}")
    print(f"  {stability(best, final_roll)}")
    print("  final decile vs first decile:")
    summary = {}
    for key, label in [
        ("success_rate_rolling", "success (rolling)"),
        ("ep_return", "ep return"),
        ("ep_length", "ep length"),
        ("entropy", "entropy"),
        ("human_death_rate", "human death rate"),
        ("cover_under_threat", "cover under threat"),
        ("objective_dist_under_threat", "dist from OBJ (threat)"),
        ("false_complete_rate", "false DONE rate"),
        ("tx_per_agent_step", "tx / agent-step"),
        ("approx_kl", "approx KL"),
    ]:
        a, b = mean(first, key), mean(last, key)
        if a != a and b != b:
            continue
        summary[key] = b
        print(f"    {label:<20} {a:>8.3f} → {b:>8.3f}   ({b - a:+.3f})")

    comps = [k for k in rows[0] if k.startswith("comp_")]
    if comps:
        drift = sorted(((mean(last, c) - mean(first, c), c) for c in comps), key=lambda t: -abs(t[0]))
        print("  reward components (final decile, biggest drift first):")
        for d, c in drift[:6] if not show_components else drift:
            print(f"    {c[5:]:<20} {mean(last, c):>8.4f}   ({d:+.4f})")

    beh_path = RUNS / run / "behavior.json"
    if beh_path.exists():
        b = json.loads(beh_path.read_text())
        m = b.get("metrics", {})
        print(f"  behavior ({b.get('episodes','?')} eps, greedy={b.get('greedy')}): "
              f"success {b.get('success_ci95','?')}")
        for key, label, fmt in [
            ("obedience_latency_mean", "obedience latency", "{:.2f}"),
            ("report_precision", "report precision", "{:.2f}"),
            ("report_recall", "report recall", "{:.2f}"),
            ("doctrine_preference_rate", "doctrine preference", "{:.3f}"),
            ("doctrine_allowed_rate", "doctrine containment", "{:.3f}"),
            ("orders_per_episode", "orders / episode", "{:.2f}"),
            ("retasks_per_episode", "retasks / episode", "{:.2f}"),
            ("false_complete_rate", "false DONE", "{:.3f}"),
            ("cover_occupancy_under_threat", "cover under threat", "{:.3f}"),
            ("mean_distance_from_objective_under_threat", "dist from OBJ", "{:.2f}"),
        ]:
            if (v := m.get(key)) is not None:
                print(f"    {label:<20} {fmt.format(v)}")
                summary[f"beh_{key}"] = v
        # refs #14: a low preference rate is only a command-quality finding if
        # the ordered-task mix says it is not just adoption of one legal leg
        if mix := format_order_task_mix(m):
            print(f"    {'order task mix':<20} {mix}   (share/preference)")
        summary["beh_success"] = m.get("success_rate")
        for g in b.get("gates", []):
            mark = "—" if g["passed"] is None else ("PASS" if g["passed"] else "FAIL")
            print(f"    gate [{mark}] {g['name']} ({'>=' if g['direction'] == 'min' else '<='} {g['bound']})")
    else:
        print("  behavior: none — run `evaluate --behavior` for the B2 suite")
    return summary


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    show_components = "--components" in args
    args = [a for a in args if a != "--components"]
    run = args[0]
    baseline = None
    if "--vs" in args:
        baseline = args[args.index("--vs") + 1]

    a = report(run, show_components)
    if baseline:
        print()
        b = report(baseline, show_components)
        print(f"\n== delta: {run} - {baseline} ==")
        for key in sorted(set(a) & set(b)):
            va, vb = a.get(key), b.get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                mark = "  <-- " if abs(va - vb) > max(0.05 * abs(vb or 1), 0.02) else "      "
                print(f"  {key:<28} {vb:>8.3f} → {va:>8.3f}  ({va - vb:+.3f}){mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
