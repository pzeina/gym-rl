#!/usr/bin/env python
"""Compact post-hoc digest of a finished run — the ONLY thing the big model reads.

A run's metrics.csv is ~3000 rows x 20 columns. Feeding that to Opus/Fable at
150k context is what makes a training campaign expensive. This collapses it to
~30 lines: config, learning curve by decile, reward-component drift, the
behavioral suite, and (optionally) deltas against a baseline run — including
whether the two runs are actually a single-variable A/B (refs #20).

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
from cohort.metrics import (  # noqa: E402
    format_obedience_by_task,
    format_order_availability,
    format_order_task_mix,
    format_staging,
)


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

#: The PUBLISHING bar, tighter than the collapse warning above: a run may not
#: contribute a headline number to README/ROADMAP unless it gave back less than
#: this. A warning that a digest prints is advice; this is a gate, because the
#: advice was already there and three collapsed runs got published anyway
#: (squad_recon_v5 and v6 both ended at 0.00 rolling and are published at 0.77
#: and 0.91; squad_v7 ended at 0.41 and is published at 0.92).
PUBLISH_STABILITY_POINTS = 10


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
    gate = "PUBLISHABLE" if drop < PUBLISH_STABILITY_POINTS else "NOT PUBLISHABLE"
    return (
        f"stability  best-final gap {drop:.0f} pts  [{verdict}]\n"
        f"  publish gate  [{gate}]  (bar: gap < {PUBLISH_STABILITY_POINTS} pts, "
        f"and the headline number is the FINAL policy)"
    )


def optimizer_line(first: list[dict], last: list[dict]) -> str | None:
    """The PPO diagnostics, or None on a run that predates them (v1.11).

    ``explained_variance`` is the one to read first: below ~0 the critic is
    worse than predicting the batch mean, so every advantage in the update is
    noise and the run is not learning from what it looks like it is learning
    from. ``grad_norm`` next — if it sits far above max_grad_norm the update is
    being scaled down wholesale, which is how the value head used to swallow
    95-99% of the policy's gradient budget.
    """
    if not any(k in (first[0] if first else {}) for k in ("explained_variance", "grad_norm")):
        return None
    out = []
    for key, label, fmt in [
        ("explained_variance", "explained var", "{:+.3f}"),
        ("grad_norm", "grad norm", "{:.3f}"),
        ("clipfrac", "clip frac", "{:.3f}"),
        ("value_std", "value scale", "{:.2f}"),
        ("return_std", "return scale", "{:.2f}"),
        ("epochs_used", "epochs used", "{:.2f}"),
    ]:
        a, b = mean(first, key), mean(last, key)
        if b != b:
            continue
        out.append(f"{label} {fmt.format(b)} ({b - a:+.3f})")
    return "  optimizer  " + "  ".join(out) if out else None


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
    if opt := optimizer_line(first, last):
        print(opt)
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
        # refs #18: tx is the CHARGED traffic only. Read the two together —
        # tx down with messages up is the stall signature (the cohort stopped
        # commanding and started talking for free), and tx alone called that
        # "the whole radio goes quiet" when the net had got 2.5x louder.
        ("messages_per_agent_step", "messages / agent-step"),
        ("timeout_rate_rolling", "ran clock out (rolling)"),
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
        # WHICH checkpoint produced these numbers, always: on squad_screen_v4
        # ckpt_best evaluates 30/30 and ckpt_latest 0/30 on the same seeds, so
        # a behavior block that does not name its checkpoint is unreadable
        # next to a curve that ended at 0% (refs #18)
        scored = Path(b.get("checkpoint") or "?").name or "?"
        print(f"  behavior ({b.get('episodes','?')} eps, greedy={b.get('greedy')}, "
              f"{scored}): success {b.get('success_ci95','?')}")
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
            # refs #18: the clock and what the net carried while it ran out
            ("timeout_rate", "ran the clock out", "{:.2f}"),
            ("messages_per_episode", "messages / episode", "{:.0f}"),
            ("command_traffic_share", "of which command", "{:.3f}"),
        ]:
            if (v := m.get(key)) is not None:
                print(f"    {label:<20} {fmt.format(v)}")
                summary[f"beh_{key}"] = v
        # refs #14: a low preference rate is only a command-quality finding if
        # the ordered-task mix says it is not just adoption of one legal leg
        if mix := format_order_task_mix(m):
            print(f"    {'order task mix':<20} {mix}   (share/preference)")
        # refs #16: an order share is availability-confounded — the mask offers
        # the tasks in unequal numbers, and in opposite directions per scenario
        # family. The lift is the share over the masked-random floor: 1.00 is
        # no preference, and it is the number a fix has to move.
        if avail := format_order_availability(m):
            print(f"    {'order availability':<20} {avail}   (share/avail (xlift))")
        # a pooled obedience mean cannot separate disobedience from a shift to
        # slower-resolving tasks — see format_obedience_by_task
        if obey := format_obedience_by_task(m):
            print(f"    {'obey latency/task':<20} {obey}   (mean(orders))")
        # refs #15: staged orders are excluded from obedience by construction
        # (a staged agent complies by holding), so the AT MY COMMAND channel —
        # and orders staged and then never released — reports separately
        if staging := format_staging(m):
            print(f"    {'A5-2 staging':<20} {staging}")
        summary["beh_success"] = m.get("success_rate")
        for g in b.get("gates", []):
            mark = "—" if g["passed"] is None else ("PASS" if g["passed"] else "FAIL")
            print(f"    gate [{mark}] {g['name']} ({'>=' if g['direction'] == 'min' else '<='} {g['bound']})")
    else:
        print("  behavior: none — run `evaluate --behavior` for the B2 suite")

    # The FINAL policy's own number, side by side with ckpt_best's. On a run
    # that peaked and fell back these differ enormously (squad_screen_v4:
    # ckpt_best 30/30, ckpt_latest 0/30 on the same seeds), and the publishing
    # standard is that both are quoted — the headline is this one.
    final_path = RUNS / run / "behavior_final.json"
    if final_path.exists():
        fb = json.loads(final_path.read_text())
        fm = fb.get("metrics", {})
        print(f"  FINAL policy ({fb.get('episodes','?')} eps, ckpt_latest.pt): "
              f"success {fb.get('success_ci95','?')}")
        summary["final_success"] = fm.get("success_rate")
        if (bs := summary.get("beh_success")) is not None and (fs := fm.get("success_rate")) is not None:
            print(f"    vs ckpt_best {bs:.2f} → final {fs:.2f}  ({fs - bs:+.2f})")
    elif beh_path.exists():
        print("  FINAL policy: not measured — re-run `evaluate ckpt_latest.pt "
              "--behavior-out behavior_final.json` (the headline number)")
    return summary


def economics_diff(run: str, baseline: str) -> None:
    """Which reward/scenario prices actually differ between two runs.

    refs #20: this campaign's own confound audit was done by hand — open
    economics.json for each run, eyeball the ``rewards`` dict, count what
    changed — and it undercounted at least once. `squad_v7` -> `squad_v8`
    reads as a single-variable A/B (`done_false` only, the pair ROADMAP's
    audit used); `squad_v6` -> `squad_v8` — an equally legitimate choice of
    "the run before this one" — differs by TWO keys, `done_false` AND
    `contact_redundant`. Nothing forced the choice of baseline to be checked
    against the file that would have caught it: `economics.json` was written
    (train.py) *specifically* so "two runs a reward commit apart are
    indistinguishable after the fact" could not happen, and the manual
    process re-created the failure the file exists to prevent.

    This makes the check a function call instead of a transcription exercise,
    so the next campaign's "the only difference is X" claim is verified
    against the actual JSON, not asserted from memory of which commit landed
    when. Diffs both ``rewards`` (the price list) and ``spec`` (the scenario
    knobs) — `train.py::_spec_economics` calls scenario knobs part of the same
    "what is this run actually an experiment about" question.
    """
    a_path, b_path = RUNS / run / "economics.json", RUNS / baseline / "economics.json"
    if not a_path.exists() or not b_path.exists():
        missing = run if not a_path.exists() else baseline
        print(f"\n  economics: uncheckable — {missing} predates economics.json")
        return
    a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    diffs = [
        (section, key, b.get(section, {}).get(key), a.get(section, {}).get(key))
        for section in ("rewards", "spec")
        for key in sorted(set(a.get(section, {})) | set(b.get(section, {})))
        if a.get(section, {}).get(key) != b.get(section, {}).get(key)
    ]
    if not diffs:
        print(f"\n  economics: CLEAN — {run} and {baseline} share every reward/spec value")
        return
    label = ("single-variable A/B" if len(diffs) == 1 else
             f"CONFOUNDED — {len(diffs)} keys differ, NOT a single-variable A/B")
    print(f"\n  economics: {label} ({run} vs {baseline})")
    for section, key, bv, av in diffs:
        print(f"    {section}.{key:<24} {bv!r} → {av!r}")


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
        economics_diff(run, baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
