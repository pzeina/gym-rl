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

from scripts.fleet_status import run_dirs  # noqa: E402
from scripts.run_report import (  # noqa: E402
    PUBLISH_STABILITY_POINTS,
    deciles,
    fnum,
    mean,
    run_dir,
)


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
    """
    rows = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
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

    print(f"\n{metric}, every generation, both checkpoints — from committed artifacts\n")
    for family in sorted({r[0] for r in rows}):
        print(f"  {family}")
        for _, run, best, final in [r for r in rows if r[0] == family]:
            print(f"    {run:<26}best {cell(best):>16}   final {cell(final):>16}")
    print("\na level the whole family shows is a property of the family, not a "
          "finding about one run")
    return 0


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
    """
    import hashlib

    def sha(path: Path) -> str | None:
        """Hash the WEIGHTS, not the file.

        The file bytes differ between policies that are bit-identical as
        policies: a checkpoint embeds its ``reward_config``, so the v1.15 revert
        and the v1.16 ENDEX restoration each produced arms whose tensors match
        to 0.000e+00 and whose files do not. Hashing the file silently fails to
        deduplicate and then reports the survivors as "distinct policies", which
        is the kind of quiet false claim this whole audit exists to catch.
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

    seen: dict[str, str] = {}
    rows = []
    ann: list[tuple[str, float | None, float | None]] = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        best, final = d / "behavior.json", d / "behavior_final.json"
        if not (best.is_file() and final.is_file()):
            continue
        b, f = json.loads(best.read_text()), json.loads(final.read_text())
        if b.get("episodes") != 100 or f.get("episodes") != 100:
            continue          # mismatched N is not a comparison (refs #34)
        a = audit_run(d)
        if not a:
            continue
        digest = sha(d / "ckpt_latest.pt")
        if digest and digest in seen:
            print(f"  (skipping {d.name}: weights identical to {seen[digest]})")
            continue
        if digest:
            seen[digest] = d.name
        rows.append((d.name, a["gap"],
                     b["metrics"]["success_rate"], f["metrics"]["success_rate"]))
        ann.append((d.name,
                    b["metrics"].get("successes_announced_rate"),
                    f["metrics"].get("successes_announced_rate")))

    if len(rows) < 4:
        print(f"only {len(rows)} distinct policies carry both checkpoints at N=100 — "
              "not enough to say anything")
        return 0

    signed = [(bs - fs) * 100 for _, _, bs, fs in rows]      # + = best overstates
    gaps = [g for _, g, _, _ in rows]
    over = sum(1 for x in signed if x > 0)
    print(f"\n{len(rows)} distinct policies, both checkpoints at N=100\n")
    print(f"{'run':<26}{'give-back':>10}{'best':>7}{'final':>7}{'best-final':>12}")
    for (n, g, bs, fs), sg in sorted(zip(rows, signed, strict=True), key=lambda z: -z[0][1]):
        print(f"{n:<26}{g:>10.2f}{bs:>7.2f}{fs:>7.2f}{sg:>+11.0f}pt")
    print(f"\nckpt_best overstates the final policy in {over}/{len(rows)} runs; "
          f"mean {sum(signed) / len(signed):+.1f}pt")
    _announcement_axis(ann)
    try:
        from scipy.stats import pearsonr
    except ImportError:
        print("scipy not available — correlation skipped")
        return 0
    r, p = pearsonr(gaps, signed)
    verdict = ("the gate PREDICTS overstatement" if r > 0 and p < 0.05
               else "no significant relationship at this n")
    print(f"give-back vs signed (best - final): Pearson r={r:.3f}, p={p:.3f}  ->  {verdict}")
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
