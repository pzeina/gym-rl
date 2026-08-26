#!/usr/bin/env python
"""The price-dispersion cycle's bar, written down before the cycle launches.

    scripts/prereg_dispersion.py --freeze          # once, before job 1 (committed)
    scripts/prereg_dispersion.py --setting burst_fraction=0.5 \
        --run fireteam_defend=fireteam_defend_v26 --run defend_brique=defend_brique_v20
    scripts/prereg_dispersion.py --setting burst_fraction=1.0 --top-of-ladder --manifest

**Why this file exists at all.** ``docs/next-cycles.md`` proposed the bar in
prose: separates if ``stacked_rate`` falls below 0.70 in both DEFEND members
"with success inside the incumbents' CI". Prose bars get read after the numbers
land, and this repo has already retracted one claim (assurance #60) that a
written-down rule would have caught. So the rule is code, the incumbents it
compares against are frozen to a file, and ``tests/test_prereg_dispersion.py``
pins both — a threshold edited after a run lands fails the suite.

**Two things the prose bar got wrong, fixed here rather than silently.**

1. *"Success inside the incumbents' CI (1.00 +/- 0.00)"* is unusable. A
   zero-width CI refuses one lost episode in a hundred, and the exact test says
   one lost episode out of 200 across two arms is a coin flip (p = 0.5). The
   bar is a one-sided Fisher non-inferiority test instead, which against a
   100/100 incumbent tolerates down to 96/100 and convicts at 95/100.

2. *The marker can be bought with casualties.* ``stacked_rate`` is the share of
   agent-steps with >= 2 LIVING teammates inside 1.5. Turning AREA FIRE on kills
   teammates, and dead teammates lower the rate without one agent having learned
   to disperse. That is the exact shape of this project's jamming finding — the
   human's death rate "improved" because its denominator vanished — so the bar
   requires ``mean_nearest_teammate_dist`` to RISE alongside, and names the
   failure mode DENOMINATOR rather than letting it read as a win.

**What is deliberately NOT in the bar.** The reporting markers
(``closed_on_root_report_rate``) and ``human_in_action_rate``. They are bimodal
across seeds — the record has patrol_brique at 0.43 over 14 runs and three
scenarios landing at 0.750-1.000 or exactly 0.000 with nothing between — so
scoring the cycle on them charges the mechanism for a seed draw. They are
measured, printed, and read separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FROZEN = ROOT / "docs" / "prereg-price-dispersion.json"
sys.path.insert(0, str(ROOT))

from scripts.exact_tests import fisher_one_sided_less, holm_reject  # noqa: E402
from scripts.fleet_status import find_run  # noqa: E402

# ---------------------------------------------------------------- the bar ---
# Every constant below is part of the pre-registration. Changing one after a
# run has landed is the thing this file exists to make visible, so the test
# suite pins each of them against the frozen JSON.

PRIMARY = ("fireteam_defend", "defend_brique")   # the two DEFEND-root members

STACKED_BAR = 0.70          # the marker bound the cycle is trying to clear
STACKED_MOVE = 0.10         # absolute fall that counts as the mechanism doing anything
NEAREST_RISE = 0.50         # cells the element must actually spread, not just thin out
ALPHA = 0.05                # one-sided, non-inferiority
LADDER = ("0.5", "0.75", "1.0")   # declared burst_fraction ladder, weakest first

VERDICTS = {
    "SEPARATES": "bunching cleared the bar, the element genuinely spread, and no "
                 "scenario lost episodes it was winning",
    "WALKS": "the element dispersed and the cohort lost — a cohort that "
             "disperses and loses is a worse cohort",
    "DENOMINATOR": "the marker fell because teammates died, not because anyone "
                   "spread out — the jamming mistake, caught before publishing",
    "PARTIAL": "dispersion is real but the marker did not reach the bar — "
               "climb the ladder",
    "NO EFFECT AT THIS PRICE": "the mechanism moved nothing here; the ladder is "
                               "not exhausted, so this is not a ceiling",
    "CEILING": "nothing moved at the strongest declared price — DEFEND cannot be "
               "held dispersed on these maps and the marker is documentation, "
               "not a defect",
    "INCOMPLETE": "a member the bar names has no N=100 final evaluation",
}


def _final_eval(run: str) -> dict | None:
    """The FINAL policy at N=100 — the only reading this bar accepts.

    ``ckpt_best`` is the best rolling window, and ``publish_audit.py`` exists
    because that is a measurement of a transient. N=20 is refused outright: the
    incumbents are frozen at N=100 and comparing across N is how a bar quietly
    becomes easier.
    """
    d = find_run(run, RUNS)
    if d is None:
        return None
    path = d / "behavior_final_n100.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    return blob if blob.get("episodes") == 100 else None


def _read(blob: dict) -> dict:
    metrics = blob["metrics"]
    markers = {m["name"]: m["value"] for m in blob.get("markers", [])}
    successes = round(metrics["success_rate"] * blob["episodes"])
    return {
        "episodes": blob["episodes"],
        "successes": successes,
        "success_rate": metrics["success_rate"],
        "stacked_rate": markers.get("stacked_rate"),
        "mean_nearest_teammate_dist": metrics.get("mean_nearest_teammate_dist"),
        "spatially_sound_rate": metrics.get("spatially_sound_rate"),
        "checkpoint_sha256": blob.get("checkpoint_sha256"),
    }


def freeze() -> dict:
    """Snapshot the incumbents the bar is scored against, once, before job 1."""
    members = json.loads((RUNS / "BASELINE.json").read_text())["runs"]
    incumbents = {}
    for scenario, run in members.items():
        blob = _final_eval(run)
        if blob is None:
            raise SystemExit(f"cannot freeze: {scenario}/{run} has no N=100 final evaluation")
        incumbents[scenario] = {"run": run, **_read(blob)}
    return {
        "cycle": "price-dispersion",
        "frozen_against": "v1.23",
        "primary": list(PRIMARY),
        "thresholds": {
            "stacked_bar": STACKED_BAR,
            "stacked_move": STACKED_MOVE,
            "nearest_rise": NEAREST_RISE,
            "alpha": ALPHA,
            "ladder": list(LADDER),
        },
        "incumbents": incumbents,
        "note": "Written before the first job of the price-dispersion cycle. The "
                "bar is scored by scripts/prereg_dispersion.py against these "
                "numbers and no others; tests/test_prereg_dispersion.py fails if "
                "either the thresholds or these incumbent readings drift.",
    }


def non_inferior(arm: dict, incumbent: dict) -> tuple[float, bool]:
    """One-sided Fisher: did this arm lose episodes the incumbent was winning?"""
    p = fisher_one_sided_less(
        arm["successes"], arm["episodes"] - arm["successes"],
        incumbent["successes"], incumbent["episodes"] - incumbent["successes"],
    )
    return p, p > ALPHA


def evaluate(frozen: dict, arms: dict[str, str], setting: str,
             top_of_ladder: bool) -> dict:
    """Read each candidate's committed evaluation, then apply the rule."""
    rows, missing = {}, []
    for scenario, run in arms.items():
        blob = _final_eval(run)
        if blob is None:
            missing.append(f"{scenario}/{run}")
            continue
        rows[scenario] = {"run": run, **_read(blob)}
    return decide(frozen, rows, setting, top_of_ladder, missing)


def decide(frozen: dict, rows: dict[str, dict], setting: str,
           top_of_ladder: bool, missing: list[str] | None = None) -> dict:
    """The rule itself, over readings already in hand — no disk, no rollouts.

    Kept separate from ``evaluate`` so the suite can drive every branch with
    synthetic readings. A decision rule nobody has shown can reach its own
    verdicts is not a pre-registration, it is a hope: ``design_power.py`` was
    written for the same reason after a six-run campaign turned out able to
    reject on 1 of its 64 possible outcomes.
    """
    incumbents = frozen["incumbents"]
    missing = list(missing or [])

    if any(s not in rows for s in PRIMARY):
        return {"verdict": "INCOMPLETE", "setting": setting, "rows": rows,
                "missing": missing or [s for s in PRIMARY if s not in rows]}

    # --- the two DEFEND members, each condition separately so the read-out can
    #     say WHICH one failed rather than just that the cycle did.
    primary = {}
    for scenario in PRIMARY:
        arm, inc = rows[scenario], incumbents[scenario]
        fell = inc["stacked_rate"] - arm["stacked_rate"]
        rose = arm["mean_nearest_teammate_dist"] - inc["mean_nearest_teammate_dist"]
        p, ok = non_inferior(arm, inc)
        primary[scenario] = {
            "stacked": arm["stacked_rate"], "stacked_was": inc["stacked_rate"],
            "stacked_fell": fell, "moves": fell >= STACKED_MOVE,
            "clears_bar": arm["stacked_rate"] < STACKED_BAR,
            "nearest": arm["mean_nearest_teammate_dist"],
            "nearest_was": inc["mean_nearest_teammate_dist"],
            "nearest_rose": rose, "real": rose >= NEAREST_RISE,
            "success": f"{arm['successes']}/{arm['episodes']}",
            "success_was": f"{inc['successes']}/{inc['episodes']}",
            "p_non_inferiority": p, "non_inferior": ok,
        }

    # --- the fleet guard: the other seven, Holm-corrected as one family, so a
    #     cycle is not convicted of breaking a scenario by multiplicity alone.
    guard_p = {s: non_inferior(rows[s], incumbents[s])[0]
               for s in rows if s not in PRIMARY and s in incumbents}
    guard_broken = holm_reject(guard_p, ALPHA) if guard_p else {}

    moves = all(v["moves"] for v in primary.values())
    real = all(v["real"] for v in primary.values())
    clears = all(v["clears_bar"] for v in primary.values())
    kept = (all(v["non_inferior"] for v in primary.values())
            and not any(guard_broken.values()))

    if not moves:
        verdict = "CEILING" if top_of_ladder else "NO EFFECT AT THIS PRICE"
    elif not real:
        verdict = "DENOMINATOR"
    elif not clears:
        verdict = "CEILING" if top_of_ladder else "PARTIAL"
    else:
        verdict = "SEPARATES" if kept else "WALKS"

    return {"verdict": verdict, "setting": setting, "top_of_ladder": top_of_ladder,
            "primary": primary, "guard_p": guard_p, "guard_broken": guard_broken,
            "rows": rows, "missing": missing}


def render(result: dict) -> str:
    out = [f"price-dispersion @ {result['setting']}",
           f"  VERDICT: {result['verdict']} — {VERDICTS[result['verdict']]}", ""]
    if result["verdict"] == "INCOMPLETE":
        out.append("  missing N=100 final evaluations: " + ", ".join(result["missing"]))
        return "\n".join(out)

    out.append(f"  {'DEFEND member':18s} {'stacked':>17s} {'nearest':>17s} "
               f"{'success':>17s} {'p':>7s}")
    for scenario, v in result["primary"].items():
        mark = lambda ok: "ok " if ok else "MISS"  # noqa: E731
        out.append(
            f"  {scenario:18s} "
            f"{v['stacked_was']:.3f}->{v['stacked']:.3f} {mark(v['moves'] and v['clears_bar'])} "
            f"{v['nearest_was']:.2f}->{v['nearest']:.2f} {mark(v['real'])} "
            f"{v['success_was']:>7s}->{v['success']:<7s} {v['p_non_inferiority']:.4f}"
            f"{'' if v['non_inferior'] else ' LOST'}")
    if result["guard_p"]:
        out += ["", "  fleet guard (Holm, family alpha 0.05):"]
        for scenario, p in sorted(result["guard_p"].items(), key=lambda kv: kv[1]):
            state = "BROKEN" if result["guard_broken"].get(scenario) else "held"
            out.append(f"    {scenario:18s} p={p:.4f}  {state}")
    if result["missing"]:
        out += ["", "  not read (no N=100 final evaluation): " + ", ".join(result["missing"])]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", action="store_true",
                    help="write the incumbent snapshot the bar scores against")
    ap.add_argument("--run", action="append", default=[], metavar="SCENARIO=RUN",
                    help="a candidate run for one scenario (repeatable)")
    ap.add_argument("--manifest", action="store_true",
                    help="take every candidate from runs/BASELINE.json instead")
    ap.add_argument("--setting", default="unstated",
                    help="the price this arm was trained at, e.g. burst_fraction=0.5")
    ap.add_argument("--top-of-ladder", action="store_true",
                    help="this is the strongest declared price — allows a CEILING verdict")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.freeze:
        blob = freeze()
        FROZEN.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"froze {len(blob['incumbents'])} incumbents -> {FROZEN.relative_to(ROOT)}")
        return

    if not FROZEN.exists():
        raise SystemExit(f"no frozen incumbents at {FROZEN.relative_to(ROOT)}; run --freeze first")
    frozen = json.loads(FROZEN.read_text())

    arms = dict(pair.split("=", 1) for pair in args.run)
    if args.manifest:
        arms = {**json.loads((RUNS / "BASELINE.json").read_text())["runs"], **arms}
    if not arms:
        raise SystemExit("nothing to score: pass --run SCENARIO=RUN or --manifest")

    result = evaluate(frozen, arms, args.setting, args.top_of_ladder)
    print(json.dumps(result, indent=2) if args.json else render(result))
    sys.exit(0 if result["verdict"] == "SEPARATES" else 1)


if __name__ == "__main__":
    main()
