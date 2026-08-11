#!/usr/bin/env python
"""Compact post-hoc digest of a finished run — the ONLY thing the big model reads.

A run's metrics.csv is ~3000 rows x 20 columns. Feeding that to Opus/Fable at
150k context is what makes a training campaign expensive. This collapses it to
~30 lines: config, learning curve by decile, reward-component drift, the
behavioral suite, and (optionally) a comparison against a baseline run — the
success / root-survival / clock triple at each side's N (refs #34), the full
delta, and whether the two runs are actually a single-variable A/B (refs #20).

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


#: (metric key, label, format) rows of a behavior block, in display order.
#:
#: ``human_death_rate`` — the rate at which the human root (the commander the
#: cohort exists to keep alive) dies — is here because of assurance #22, whose
#: pre-registration names *root deaths at the final policy* as the PRIMARY axis
#: of the v1.12 survivor-scaled-terminal A/B. The metric was measured on every
#: behavior run and printed by `evaluate`'s own table, but this digest — "the
#: ONLY thing the big model reads" — dropped it, so the pre-registered primary
#: could not be read from the artifact the verdict is written against.
_BEHAVIOR_ROWS: tuple[tuple[str, str, str], ...] = (
    ("obedience_latency_mean", "obedience latency", "{:.2f}"),
    ("report_precision", "report precision", "{:.2f}"),
    ("report_recall", "report recall", "{:.2f}"),
    ("doctrine_preference_rate", "doctrine preference", "{:.3f}"),
    ("doctrine_allowed_rate", "doctrine containment", "{:.3f}"),
    ("orders_per_episode", "orders / episode", "{:.2f}"),
    ("retasks_per_episode", "retasks / episode", "{:.2f}"),
    ("false_complete_rate", "false DONE", "{:.3f}"),
    ("human_death_rate", "root death rate", "{:.3f}"),
    ("cover_occupancy_under_threat", "cover under threat", "{:.3f}"),
    ("mean_distance_from_objective_under_threat", "dist from OBJ", "{:.2f}"),
    # refs #18: the clock and what the net carried while it ran out
    ("timeout_rate", "ran the clock out", "{:.2f}"),
    ("messages_per_episode", "messages / episode", "{:.0f}"),
    ("command_traffic_share", "of which command", "{:.3f}"),
)


#: The three cells every A/B comparison prints, in display order (refs #34).
#:
#: They travel together because each one alone can be read to flatter a policy
#: the other two would convict. Success says nothing about what the success
#: cost — a cohort can win every episode over its commander's body. Root
#: survival on its own is gameable in the opposite direction, since a policy
#: that never closes with the enemy buries nobody and achieves nothing, which
#: is exactly what ``timeout_rate`` catches: it separates "kept everyone alive
#: by holding the ground" from "kept everyone alive by riding the clock out".
#:
#: The prompting case is the `squad_v8` → `squad_v9` A/B (`done_false` -0.5 ->
#: -2.0), first published with success and DONE volume alone. Its survival cell
#: is a **null** — p = 1.00 pooled over 200 episodes an arm — and the null is
#: the result, because an earlier `done_false` change had once been *associated*
#: with root deaths moving 4/30 → 12/30 while success held. A reader cannot
#: infer a null from a column that is not there, so the pair is printed by the
#: instrument rather than by whoever remembers to.
_COMPARISON_ROWS: tuple[tuple[str, str, str], ...] = (
    ("success", "success", "{:.3f}"),
    ("human_death_rate", "root death rate", "{:.3f}"),
    ("timeout_rate", "ran the clock out", "{:.3f}"),
)

#: (summary prefix, heading) of the checkpoints a comparison covers. Both, for
#: the same reason ``report`` prints both: on `squad_screen_v4` the two evaluate
#: 30/30 and 0/30 on the same seeds, so an A/B stated at one checkpoint is an
#: A/B between two unstated policies.
_COMPARISON_CHECKPOINTS: tuple[tuple[str, str], ...] = (
    ("beh_", "ckpt_best"),
    ("final_", "FINAL policy"),
)

#: Summary keys that are sample sizes rather than measurements. They belong in
#: the comparison header, where an N is read as an N, and not in the delta dump,
#: where "100.000 → 20.000  (-80.000)" reads as a metric that moved.
_EPISODE_COUNT_KEYS: frozenset[str] = frozenset(
    f"{prefix}episodes" for prefix, _ in _COMPARISON_CHECKPOINTS
)


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


def behavior_block(path: Path, header: str, summary: dict, prefix: str, *, diagnostics: bool) -> dict:
    """Print one evaluated checkpoint's suite, and file it under ``prefix``.

    Called twice per run — once for ``ckpt_best`` and once for the FINAL
    policy — because a single behavior block is not a run's behavior. On
    ``squad_screen_v4`` the two checkpoints evaluate 30/30 and 0/30 on the same
    seeds, and on ``defend_brique_v4`` the whole positional regression lives in
    the gap: ckpt_best passes all three gates, ckpt_latest FAILS
    ``mean_distance_from_objective_under_threat`` at 6.09. Printing the suite
    for ckpt_best and a single success number for the final policy meant the
    digest showed three PASSes and hid the FAIL on the checkpoint the
    publishing standard calls the headline (refs #22).

    ``diagnostics`` adds the four command-quality composites (task mix,
    availability lift, per-task obedience, staging). They stay on the ckpt_best
    block alone: they diagnose how the cohort commands rather than whether the
    run cleared its bars, and the digest's whole purpose is to stay short.
    """
    b = json.loads(path.read_text())
    m = b.get("metrics", {})
    # WHICH checkpoint produced these numbers, always: on squad_screen_v4
    # ckpt_best evaluates 30/30 and ckpt_latest 0/30 on the same seeds, so
    # a behavior block that does not name its checkpoint is unreadable
    # next to a curve that ended at 0% (refs #18)
    scored = Path(b.get("checkpoint") or "?").name or "?"
    print(f"  {header} ({b.get('episodes','?')} eps, greedy={b.get('greedy')}, "
          f"{scored}): success {b.get('success_ci95','?')}")
    for key, label, fmt in _BEHAVIOR_ROWS:
        if (v := m.get(key)) is not None:
            print(f"    {label:<20} {fmt.format(v)}")
            summary[f"{prefix}{key}"] = v
    if diagnostics:
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
    summary[f"{prefix}success"] = m.get("success_rate")
    # How many episodes these numbers are made of, so `--vs` can say whether the
    # two sides of a comparison were measured at the same N (refs #34)
    summary[f"{prefix}episodes"] = b.get("episodes")
    for g in b.get("gates", []):
        mark = "—" if g["passed"] is None else ("PASS" if g["passed"] else "FAIL")
        print(f"    gate [{mark}] {g['name']} ({'>=' if g['direction'] == 'min' else '<='} {g['bound']})")
    return m


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
        behavior_block(beh_path, "behavior", summary, "beh_", diagnostics=True)
    else:
        print("  behavior: none — run `evaluate --behavior` for the B2 suite")

    # The FINAL policy's own suite, side by side with ckpt_best's. On a run
    # that peaked and fell back these differ enormously (squad_screen_v4:
    # ckpt_best 30/30, ckpt_latest 0/30 on the same seeds), and the publishing
    # standard is that both are quoted — the headline is this one. It gets the
    # same rows and the same gates for that reason (refs #22): every prediction
    # in the v1.12 pre-registration is stated at the final policy, and until
    # this printed them the digest could not settle one of them.
    final_path = RUNS / run / "behavior_final.json"
    if final_path.exists():
        fm = behavior_block(final_path, "FINAL policy", summary, "final_", diagnostics=False)
        if (bs := summary.get("beh_success")) is not None and (fs := fm.get("success_rate")) is not None:
            print(f"    vs ckpt_best {bs:.2f} → final {fs:.2f}  ({fs - bs:+.2f})")
    elif beh_path.exists():
        print("  FINAL policy: not measured — re-run `evaluate ckpt_latest.pt "
              "--behavior-out behavior_final.json` (the headline number)")
    return summary


def _cell(value: float | None, fmt: str) -> str:
    """One right-aligned number, or an em dash where the run never measured it."""
    return f"{fmt.format(value):>8}" if value is not None else f"{'—':>8}"


def _sample_sizes(run: str, baseline: str, n_run: int | None, n_base: int | None) -> str:
    """``N`` for both sides, and — loudly — whether they are the same N.

    A delta between a 20-episode arm and a 100-episode arm is not an effect
    size, and the difference is invisible once both are printed to three
    decimals. This is not hypothetical: the `squad_v8` comparator committed in
    this repository is an N=20 artifact and `squad_v9` publishes at N=100, so
    the A/B a reader can reconstruct from the repo is 5x mismatched while
    looking exactly like a matched one.
    """
    shown = f"N: {baseline} {n_base if n_base is not None else '?'} · {run} {n_run if n_run is not None else '?'}"
    if n_run is None or n_base is None:
        return f"{shown}   [N UNKNOWN on one side — matching unverified]"
    if n_run != n_base:
        return (f"{shown}   [MISMATCHED N — {n_base} vs {n_run}; the deltas below are "
                "NOT an effect size]")
    return f"{shown}   [matched]"


def comparison(run: str, baseline: str, a: dict, b: dict) -> None:
    """Success, root survival and the clock, side by side, with each side's N.

    This is the block an A/B verdict gets written against, so the pair that
    #34 asks for is printed here — by the instrument, on every comparison —
    rather than left to whoever writes the next ROADMAP table. See
    ``_COMPARISON_ROWS`` for why the three cells are inseparable.

    Everything degrades to an em dash and a named absence: a run that predates a
    metric, or was evaluated without ``--behavior``, prints "not measured on
    <run>" instead of a number, a zero, or a traceback. An unmeasured axis is
    not a passed one, and the comparison is still worth printing for the axes
    that were measured.
    """
    print(f"\n== A/B: {run} vs {baseline} ==")
    print("  success + root survival + the clock, together: a policy that never fights "
          "buries no\n  commanders, and one that wins can still bury them (refs #34)")
    for prefix, header in _COMPARISON_CHECKPOINTS:
        keys = [f"{prefix}{key}" for key, _, _ in _COMPARISON_ROWS]
        if all(a.get(k) is None and b.get(k) is None for k in keys):
            continue  # neither side evaluated this checkpoint; nothing to compare
        sizes = _sample_sizes(run, baseline, a.get(f"{prefix}episodes"), b.get(f"{prefix}episodes"))
        print(f"  {header}   {sizes}")
        for key, label, fmt in _COMPARISON_ROWS:
            va, vb = a.get(f"{prefix}{key}"), b.get(f"{prefix}{key}")
            line = f"    {label:<20} {_cell(vb, fmt)} → {_cell(va, fmt)}"
            if va is None or vb is None:
                absent = " and ".join(n for v, n in ((vb, baseline), (va, run)) if v is None)
                print(f"{line}   [not measured on {absent}]")
            else:
                print(f"{line}  ({va - vb:+.3f})")


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
        print(f"\n  economics: prices identical — {run} and {baseline} share every reward/spec value")
    else:
        label = ("one price differs" if len(diffs) == 1 else
                 f"{len(diffs)} prices differ")
        print(f"\n  economics: {label} ({run} vs {baseline})")
        for section, key, bv, av in diffs:
            print(f"    {section}.{key:<24} {bv!r} → {av!r}")
    code_diff(run, baseline, price_diffs=len(diffs))


def code_diff(run: str, baseline: str, price_diffs: int = 0) -> None:
    """The other half of "is this a single-variable A/B" — the code.

    ``economics_diff`` above compares prices, and for a year it printed
    "CLEAN — share every reward/spec value" as though that settled the
    question. It does not. `squad_v7` -> `squad_v8` reads price-clean on one
    key and was reported as a single-variable A/B, but `squad_v8` is also the
    first squad run carrying `d44ee8d` — a change to the environment itself.
    A code change never touches `economics.json`'s prices, so the instrument
    built to catch confounds was blind to half the class it was built for.

    The commit is already recorded (``economics.json:git_commit``); nothing
    consulted it. Now the two runs' commits are compared and, when they differ,
    the intervening commits that touched ``cohort/`` are listed — those are the
    ones that can move behaviour. Commits touching only ``scripts/``, ``tests/``
    or docs are counted and not listed: they cannot change what a policy learned.
    """
    commits = {}
    for name in (baseline, run):
        path = RUNS / name / "economics.json"
        try:
            commits[name] = json.loads(path.read_text()).get("git_commit")
        except (OSError, json.JSONDecodeError):
            commits[name] = None

    code_moved: bool | None  # None = we could not tell, which is not "no"
    if not all(commits.values()):
        missing = [n for n, c in commits.items() if not c]
        print(f"    code: UNCHECKABLE — no git_commit recorded for {', '.join(missing)}")
        code_moved = None
    elif commits[run] == commits[baseline]:
        print(f"    code: same commit {commits[run][:8]}")
        code_moved = False
    else:
        span = _git(["rev-list", "--count", f"{commits[baseline]}..{commits[run]}"])
        touching = _git(
            ["log", "--oneline", f"{commits[baseline]}..{commits[run]}", "--", "cohort/"]
        )
        if span is None or touching is None:
            print(f"    code: UNCHECKABLE — {commits[baseline][:8]} is not an ancestor "
                  "of this run in this clone")
            code_moved = None
        else:
            lines = [ln for ln in touching.splitlines() if ln.strip()]
            print(f"    code: {commits[baseline][:8]} → {commits[run][:8]}  "
                  f"({int(span or 0)} commits, {len(lines)} touching cohort/)")
            for ln in lines[:8]:
                print(f"      {ln}")
            if len(lines) > 8:
                print(f"      … {len(lines) - 8} more")
            code_moved = bool(lines)

    # One verdict over both axes, because the failure this whole function exists
    # for was reading a clean verdict on one axis as a clean verdict overall.
    # Order matters: a known confound outranks an unknown one — two prices apart
    # is CONFOUNDED whether or not the code can be checked — and only a pair
    # that is clean on one axis and unreadable on the other is UNCHECKABLE.
    if price_diffs > 1:
        extra = " (and the code is unknown)" if code_moved is None else (
            " (and the environment moved too)" if code_moved else "")
        print(f"    verdict: CONFOUNDED — {price_diffs} prices differ{extra}")
    elif code_moved:
        print("    verdict: CONFOUNDED — the environment moved between these runs, "
              "whatever the prices say")
    elif code_moved is None:
        prices = "prices identical" if price_diffs == 0 else "one price differs"
        print(f"    verdict: UNCHECKABLE — {prices}, code unknown. "
              "Not the same finding as 'no difference'.")
    elif price_diffs == 1:
        print("    verdict: single-variable A/B — one price, same code")
    else:
        print("    verdict: IDENTICAL SETUP — same code, same prices")


def _git(argv: list[str]) -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", *argv], cwd=RUNS.parent, capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


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
        comparison(run, baseline, a, b)
        print(f"\n== delta: {run} - {baseline} ==")
        for key in sorted(set(a) & set(b)):
            if key in _EPISODE_COUNT_KEYS:
                continue  # a sample size, reported as one in the A/B block above
            va, vb = a.get(key), b.get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                mark = "  <-- " if abs(va - vb) > max(0.05 * abs(vb or 1), 0.02) else "      "
                print(f"  {key:<28} {vb:>8.3f} → {va:>8.3f}  ({va - vb:+.3f}){mark}")
        economics_diff(run, baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
