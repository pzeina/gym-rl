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
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT))

from scripts.fleet_status import run_dirs  # noqa: E402
from scripts.run_report import (  # noqa: E402
    PUBLISH_STABILITY_POINTS,
    _git,
    deciles,
    fnum,
    mean,
    run_dir,
)

# What a re-evaluation can move. A checkpoint's measured score depends on the
# environment it is scored in, the policy class that loads it and the evaluator
# that scores it — all of which live under ``cohort/``. Commits touching only
# ``scripts/``, ``tests/`` or docs cannot move a number, and are not counted.
EVALUATION_TREE = "cohort/"

# An artifact that is not committed yet was measured against the tree as it
# stands, i.e. HEAD or later.
WORKTREE = "WORKTREE"


def policy_digest(path: Path) -> str | None:
    """sha256 of a checkpoint's model TENSORS — the policy's identity, not the file's.

    The file bytes are the wrong identity in both directions. A checkpoint
    serializes its ``reward_config``, so a file-level hash splits one policy
    into two whenever only the price differs — precisely the comparison a price
    experiment makes: the rdb campaign's seed-16 "pair" is one set of tensors
    under two ``root_done_bonus`` tags, and its two arms were one run (#60 §3).
    File identity is sufficient for policy identity and never necessary.

    And tensor identity is common, not exotic: training is bit-deterministic in
    (seed, scenario, steps, lr, price), so a re-launch across commits that never
    touch the trajectory reproduces its predecessor exactly. All twelve v1.21
    campaign runs did (#60 §1). ``validate_gate`` uses this to keep such
    re-executions from inflating n, and ``baseline.py`` uses it to disclose
    them before anyone describes a re-run as an independent draw.

    None for a path that is not a readable checkpoint — absence, not a verdict.
    """
    if not path.is_file():
        return None
    try:
        import torch

        model = torch.load(path, map_location="cpu", weights_only=False)["model"]
    except Exception:
        return None
    h = hashlib.sha256()
    for key in sorted(model):
        h.update(key.encode())
        h.update(model[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


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


def evaluation_era(path: Path) -> str | None:
    """Which tree an evaluation artifact was measured against (refs #39).

    ``behavior.json`` records the checkpoint (``checkpoint_sha256``, refs #28),
    the seed, N and the sampling mode — everything about the measurement except
    *when* it was taken, which is the one field that decides whether two
    evaluations can be differenced at all.

    **Since v1.20 the artifact says it itself**: ``evaluate.py`` stamps
    ``eval_commit`` with HEAD at scoring time, which is the exact quantity —
    git's answer only ever bounded it from above, and only while the file
    stayed committed and unmoved. The git fallback is kept, and will be needed
    for as long as the artifacts written before v1.20 are on disk, none of
    which will ever carry the field.

    Returns a sha, ``WORKTREE`` for an artifact that is not committed yet, or
    None when git could not answer — which is *unknown*, not agreement.
    """
    try:
        stamped = json.loads(path.read_text()).get("eval_commit")
    except (OSError, json.JSONDecodeError):
        stamped = None
    if stamped:
        return str(stamped)
    out = _git(["log", "-1", "--format=%H", "--", str(path)])
    if out is None:
        return None
    return out.strip() or WORKTREE


def era_gap(best: Path, final: Path) -> int | None:
    """Commits under ``cohort/`` between two evaluations; None if unknowable.

    Zero means the pair can be differenced: both numbers describe policies
    scored in the same environment by the same instrument, so their difference
    is a property of the checkpoints. Anything above zero means the difference
    also contains however much the environment moved, and the pair is not
    evidence about checkpoints.
    """
    a, b = evaluation_era(best), evaluation_era(final)
    if a is None or b is None:
        return None
    if a == b:
        return 0
    span = _git(["rev-list", "--count", f"{'HEAD' if a == WORKTREE else a}..."
                 f"{'HEAD' if b == WORKTREE else b}", "--", EVALUATION_TREE])
    if span is None:
        return None
    return int(span.strip() or 0)


def series(metric: str, scenario: str | None = None) -> int:
    """One metric across every generation of each scenario, both checkpoints.

    The README's missing ``_family`` (refs #36). ``program_board.py`` renders a
    metric's spread across a scenario's other generations beside any thread that
    leads with a level, because a level a whole family shows is a property of
    the family and not a finding about the run being discussed. The README had
    no equivalent, and published ``squad_v8``'s root-death rate of 0.23 as "the
    highest in the fleet, and no gate covers it" — both true — with nothing to
    say that the same lineage read 0.45 and 0.35 in the two generations before
    it. Seven numbers have now been read as regressions against their
    predecessor and as ordinary-or-better against their series.

    Read off committed ``behavior*.json`` only: no re-scoring, no checkpoint
    load. Runs whose evaluation predates a metric simply do not appear for it,
    and a missing ``behavior_final.json`` prints as ``—`` rather than silently
    letting a ``ckpt_best`` number stand in for a final one.

    **Enumerated through ``run_dirs``, which walks ``runs/archive/`` too**
    (refs #58). This scan is the repository's answer to "what does the corpus
    actually show", and for a while it answered on 71 of 167 runs: the archive
    move filed 96 generations away and this walked ``runs/`` directly, so the
    README's own instruction to regenerate the `squad` root-death family from
    committed artifacts printed a table with `squad_v6`, `v7`, `v8` and `v9` —
    every row that table is built from — silently missing. An enumerator that
    quietly drops the older half of the evidence is worse than no enumerator,
    because a bound read off it looks measured.
    """
    rows = []
    for d in run_dirs(RUNS):
        cells = {}
        for tag, name in (("best", "behavior.json"), ("final", "behavior_final.json")):
            path = d / name
            if not path.is_file():
                continue
            doc = json.loads(path.read_text())
            cells[tag] = (doc.get("metrics", {}).get(metric), doc.get("episodes"))
            cells["scenario"] = doc.get("scenario")
        if not cells or all(cells.get(t, (None,))[0] is None for t in ("best", "final")):
            continue
        if scenario and cells.get("scenario") != scenario:
            continue
        rows.append((cells.get("scenario") or "?", d.name, cells.get("best"), cells.get("final")))

    if not rows:
        print(f"no committed evaluation carries {metric!r}"
              + (f" on scenario {scenario!r}" if scenario else ""))
        return 1

    def cell(v: tuple | None) -> str:
        if not v or v[0] is None:
            return "—"
        return f"{v[0]:.3f} (N={v[1]})"

    rows.sort()   # run_dirs yields the fleet then the archive; read by generation
    print(f"\n{metric}, every generation, both checkpoints — from committed artifacts\n")
    for family in sorted({r[0] for r in rows}):
        print(f"  {family}")
        for _, run, best, final in [r for r in rows if r[0] == family]:
            print(f"    {run:<26}best {cell(best):>16}   final {cell(final):>16}")
    print("\na level the whole family shows is a property of the family, not a "
          "finding about one run")
    _checkpoint_extremes(rows)
    return 0


def _checkpoint_extremes(rows: list[tuple]) -> None:
    """The metric's range at EACH checkpoint, and the widest disagreement.

    Printed because the table above was read down one column (refs #58). The
    commander-survival ceiling was placed "in the middle of an empty band" —
    nothing between 0.38 and 1.00 — which is true of the `ckpt_latest` column of
    the unarchived fleet and false of the table as a whole: `ckpt_best` carries
    0.98, 0.95, 0.85 and 0.60, and four runs land on opposite sides of the 0.5
    ceiling at their own two checkpoints. A bound is a claim about the corpus,
    so the corpus prints its own extremes rather than leaving them to be
    eyeballed off eighty rows.
    """
    got = {
        tag: [(v[0], run) for _, run, best, final in rows
              if (v := (best if tag == "best" else final)) and v[0] is not None]
        for tag in ("best", "final")
    }
    if not any(got.values()):
        return
    print("\nrange at each checkpoint — a bound read off one column is not a bound "
          "on the corpus:")
    for tag, label in (("best", "ckpt_best"), ("final", "ckpt_latest")):
        vals = got[tag]
        if not vals:
            print(f"  {label:<12} — (no run carries it here)")
            continue
        lo, hi = min(vals), max(vals)
        print(f"  {label:<12} min {lo[0]:.3f} ({lo[1]})   max {hi[0]:.3f} ({hi[1]})"
              f"   n={len(vals)}")
    both = [(abs(best[0] - final[0]), run, best[0], final[0])
            for _, run, best, final in rows
            if best and final and best[0] is not None and final[0] is not None]
    if both:
        gap, run, b, f = max(both)
        print(f"  widest best-vs-final disagreement: {run} {b:.3f} -> {f:.3f} "
              f"({gap:.3f}, over {len(both)} paired runs)")


def _era_label(gap: int | None) -> str:
    if gap is None:
        return "unknown"
    return "same commit" if gap == 0 else f"+{gap} apart"


def _announcement_axis(ann: list[tuple[str, float | None, float | None]]) -> None:
    """The same policies on `successes_announced_rate`, printed here so the
    success-axis result cannot be quoted about this one (refs #38).

    Everything above is measured on success, and the small spreads it reports
    are a fact about success. The assurance layer re-tapped one pair at one
    commit and found `squad_v8` announcing **0 of 97** at `ckpt_best` and
    **91 of 98** at `ckpt_latest` — one point apart on success, ninety-three on
    the announcement. So a bound established here says nothing about a column
    published there, and any between-checkpoint claim about the announcement
    has to be measured at both checkpoints or not made.
    """
    pairs = [(n, b, f) for n, b, f in ann if b is not None and f is not None]
    if not pairs:
        return
    worst = max(pairs, key=lambda z: abs(z[1] - z[2]))
    swings = sorted(((abs(b - f) * 100, n, b, f) for n, b, f in pairs), reverse=True)
    print(f"\n{len(pairs)} of those carry the ANNOUNCEMENT at both checkpoints "
          "(successes announced / successes):")
    print(f"{'run':<26}{'best':>7}{'final':>7}{'|best-final|':>14}")
    for swing, n, b, f in swings:
        print(f"{n:<26}{b:>7.2f}{f:>7.2f}{swing:>13.0f}pt")
    print(f"largest swing {abs(worst[1] - worst[2]) * 100:.0f}pt ({worst[0]}) — this gate is "
          "validated on SUCCESS; do not carry its bound to the announcement column")


def validate_gate() -> int:
    """Does give-back predict that ckpt_best OVERSTATES the final policy?

    The gate exists on the premise that a best-rolling-window checkpoint can be
    a measurement of a transient. That premise is checkable now that runs carry
    both checkpoints at N=100, and it must be checked against the SIGNED
    quantity: the gate claims the published figure is too HIGH, not merely that
    the two checkpoints differ. Reading it against |best - final| says the
    opposite of the truth, because absolute divergence is dominated by runs near
    the ceiling where neither checkpoint can move far.

    Runs whose ``ckpt_latest`` hashes identically are one policy, not several —
    v1.16/v1.17 produced three bit-identical fireteam_defend arms and two
    defend_brique ones, and counting them separately would inflate n by 17%.

    **This is a statement about the SUCCESS axis only.** It is measured on
    ``success_rate``, and nothing it finds transfers to another published
    column: the same checkpoint pair that agrees to one point on success can
    disagree by ninety-three on the announcement. ``_announcement_axis`` prints
    that axis underneath rather than leaving the scope to be assumed (refs #38).

    **And only pairs measured at one commit are evidence** (refs #39). ``best``
    comes from ``behavior.json`` and ``final`` from ``behavior_final.json``, and
    for three runs those two files entered the repository days and dozens of
    ``cohort/`` commits apart — ``fireteam_v7``'s best was scored at ``703a6ac``
    and its final at ``f18462d``, 36 ``cohort/`` commits later (21 of them under
    ``env``/``core``/``config``), among them the fallen-share-the-win fix and two
    rewrites of when a mission counts as done. Differencing those two numbers
    measures the
    checkpoint *and* the environment. That is the same rule this repo already
    applies to A/B pairs (``run_report.code_diff``), applied to the audit
    itself; the headline correlation is now taken over same-commit pairs only,
    with the all-pairs figure printed underneath and labelled as confounded.

    Enumerated through ``run_dirs`` so the archive counts (refs #58): the
    give-back premise is a claim about every run that ever published, and the
    runs that motivated it — ``squad_recon_v5``/``v6``, ``squad_v7`` — are all
    filed under ``runs/archive/``.
    """
    # Deduplicate on the WEIGHTS, not the file (``policy_digest``): the v1.15
    # revert and the v1.16 ENDEX restoration each produced arms whose tensors
    # match to 0.000e+00 and whose files do not. Hashing the file silently
    # fails to deduplicate and then reports the survivors as "distinct
    # policies", which is the kind of quiet false claim this audit exists to
    # catch.
    seen: dict[str, str] = {}
    rows = []
    ann: list[tuple[str, float | None, float | None]] = []
    for d in run_dirs(RUNS):
        best, final = d / "behavior.json", d / "behavior_final.json"
        if not (best.is_file() and final.is_file()):
            continue
        b, f = json.loads(best.read_text()), json.loads(final.read_text())
        if b.get("episodes") != 100 or f.get("episodes") != 100:
            continue          # mismatched N is not a comparison (refs #34)
        a = audit_run(d)
        if not a:
            continue
        digest = policy_digest(d / "ckpt_latest.pt")
        if digest and digest in seen:
            print(f"  (skipping {d.name}: weights identical to {seen[digest]})")
            continue
        if digest:
            seen[digest] = d.name
        rows.append((d.name, a["gap"], b["metrics"]["success_rate"],
                     f["metrics"]["success_rate"], era_gap(best, final)))
        ann.append((d.name,
                    b["metrics"].get("successes_announced_rate"),
                    f["metrics"].get("successes_announced_rate")))

    if len(rows) < 4:
        print(f"only {len(rows)} distinct policies carry both checkpoints at N=100 — "
              "not enough to say anything")
        return 0

    signed = [(bs - fs) * 100 for _, _, bs, fs, _ in rows]      # + = best overstates
    one_commit = [r[4] == 0 for r in rows]
    print(f"\n{len(rows)} distinct policies, both checkpoints at N=100\n")
    print(f"{'run':<26}{'give-back':>10}{'best':>7}{'final':>7}{'best-final':>12}"
          f"   {'evaluations':<13}")
    for (n, g, bs, fs, eg), sg in sorted(zip(rows, signed, strict=True), key=lambda z: -z[0][1]):
        print(f"{n:<26}{g:>10.2f}{bs:>7.2f}{fs:>7.2f}{sg:>+11.0f}pt   {_era_label(eg):<13}")

    cross = [(n, eg) for (n, _, _, _, eg) in rows if eg != 0]
    if cross:
        print(f"\n{sum(one_commit)}/{len(rows)} pairs were measured at one commit. "
              f"{len(cross)} were not, and are excluded (refs #39):")
        for n, eg in sorted(cross, key=lambda z: -(z[1] or 0)):
            what = (f"{eg} commits touched {EVALUATION_TREE} between the two evaluations"
                    if eg else "the two evaluations cannot be dated in this clone")
            print(f"  {n:<26} {what}")
        print("  their best-final is a checkpoint difference PLUS an environment "
              "difference, and\n  says nothing about either on its own")

    kept = [(r, s) for r, s, ok in zip(rows, signed, one_commit, strict=True) if ok]
    over = sum(1 for _, s in kept if s > 0)
    if kept:
        print(f"\nover the {len(kept)} same-commit pairs: ckpt_best overstates the final "
              f"policy in {over}/{len(kept)} runs; mean "
              f"{sum(s for _, s in kept) / len(kept):+.1f}pt")
    _announcement_axis([a for a, ok in zip(ann, one_commit, strict=True) if ok])
    try:
        from scipy.stats import pearsonr
    except ImportError:
        print("scipy not available — correlation skipped")
        return 0

    def correlate(pairs: list[tuple[tuple, float]], label: str) -> None:
        if len(pairs) < 4:
            print(f"only {len(pairs)} {label} — not enough to say anything")
            return
        r, p = pearsonr([r[1] for r, _ in pairs], [s for _, s in pairs])
        verdict = ("the gate PREDICTS overstatement" if r > 0 and p < 0.05
                   else "no significant relationship at this n")
        print(f"give-back vs signed (best - final), {label}: "
              f"n={len(pairs)}, Pearson r={r:.3f}, p={p:.3f}  ->  {verdict}")

    print()
    correlate(kept, "pairs measured at one commit")
    if cross:
        correlate(list(zip(rows, signed, strict=True)),
                  "ALL pairs (CONFOUNDED — mixes environment drift into the difference)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluate", action="store_true",
                    help="measure the FINAL policy for any run missing behavior_final.json")
    ap.add_argument("--validate", action="store_true",
                    help="ask whether the give-back gate predicts what it is used to predict")
    ap.add_argument("--min-episodes", type=int, default=100,
                    help="only audit runs published at this eval size or larger")
    ap.add_argument("--series", metavar="METRIC",
                    help="print one metric across every generation of each scenario, "
                         "both checkpoints (the README's missing _family, refs #36)")
    ap.add_argument("--scenario", help="restrict --series to one scenario")
    args = ap.parse_args()

    if args.series:
        return series(args.series, args.scenario)
    if args.validate:
        return validate_gate()

    audits = [a for d in run_dirs(RUNS) if (a := audit_run(d))]
    audits = [a for a in audits if a["episodes"] >= args.min_episodes]
    if not audits:
        print(f"no runs published at N>={args.min_episodes}")
        return 0

    if args.evaluate:
        from cohort.training.evaluate import evaluate

        for a in (x for x in audits if x["final_eval"] is None):
            ckpt = run_dir(a["run"]) / "ckpt_latest.pt"
            if not ckpt.exists():
                continue
            try:
                s = evaluate(str(ckpt), episodes=args.min_episodes, behavior=True,
                             behavior_path=str(run_dir(a["run"]) / "behavior_final.json"))
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
