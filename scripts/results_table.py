#!/usr/bin/env python
"""Generate the README's baseline results table from the runs themselves.

    scripts/results_table.py            # print it
    scripts/results_table.py --write    # splice it into README.md between markers

Every published overstatement this repository has had to correct was a hand-kept
number that stopped matching its artifact: an N=20 row captioned N=100, a `—` in
the announced column standing in for a zero, a headline read off `ckpt_best`
while the prose said "the policy". None of them were dishonest and all of them
were transcription.

So the table is not transcribed. It is read at generation time from
``runs/BASELINE.json`` and each member's ``behavior_final.json`` /
``behavior.json``, and ``tests/test_results_table.py`` fails if what is in
README.md is not what this script produces today. A number can still be wrong —
but it can no longer be wrong in a way the runs would have contradicted.

Columns, and why each one is there rather than success alone:

* **success (N, final)** — the headline, on the policy the run ENDED with.
* **peak** — ``ckpt_best``, labelled as a peak, because on an unstable run the
  two differ by up to 17 points and quoting the higher one silently is the
  single most common way this project has overstated itself.
* **give-back** — peak minus final decile, the publishing gate's own statistic.
* **root death / timeout** — what the success was bought with. Success alone is
  blind to a cohort that wins over its commander's body, and root survival alone
  is gameable by never closing with the enemy; the timeout column separates them.
* **announced** — wins that went out on the net. Complete by construction since
  v1.19; the column stays because a miss here means the guarantee broke.
* **reported** — of those, the ones the ROOT closed itself rather than leaving to
  HQ. This is where the agent behaviour the announcement column used to carry
  now lives, and it is the honest way to keep measuring it. The rate has two
  routes into its numerator — a root DONE claim, and a root SITREP landing on
  the final step — and *which* route is live is decided by the root's mission,
  not by the claim count (issue #62, correcting the #61 guard): on a
  completable root (SEIZE/RECON/…) the claim is the only counted close, so
  0% with zero claims is the mute-root finding and prints as the measurement
  it is; on a DEFEND/DENY root the claim is masked shut by doctrine, so the
  cell prints the rate marked as the SITREP route, with ``closes_per_root_sitrep``
  saying whether those closes were timed or bought with volume.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cohort.core.missions import MissionType, is_completable  # noqa: E402
from cohort.metrics import split_gates  # noqa: E402
from scripts import baseline  # noqa: E402
from scripts.publish_audit import audit_run  # noqa: E402

START = "<!-- BASELINE-TABLE:START -->"
END = "<!-- BASELINE-TABLE:END -->"

HEADER = (
    "| scenario | run | success (final, N) | peak (best ckpt) | give-back | "
    "root death | timeout | announced | root-reported | gates |"
)
RULE = "|---|---|---|---|---|---|---|---|---|---|"


NO_CLAIMS = "— (no claims)"


def _pct(v: float | None, places: int = 0) -> str:
    return "—" if v is None else f"{v * 100:.{places}f}%"


def _root_reported(m: dict) -> str:
    """The root-reported cell, classified by the root's mission (issue #62).

    ``closed_on_root_report_rate`` has two routes into its numerator — a root
    DONE claim, and a root SITREP landing on the final step — and the thing
    that decides which route is live is the root's *mission*
    (``is_completable``, the same predicate the env gates on), not the claim
    count. The #61 guard keyed on ``done_claim_episodes_root == 0`` and so
    dashed both ends of the metric at once: a SEIZE root's measured 0.000 (the
    mute-root finding — on a completable root a claim is the only counted
    close, so zero claims *entails* the zero) and a DEFEND root's 1.000 (the
    claim is masked shut by doctrine, so the SITREP route is that root's only
    completion channel, and ``fireteam_defend_v23`` at 1.000 vs
    ``fireteam_defend_v18`` at 0.000 is exactly the difference the column
    exists to measure).

    So: completable root — print the rate, zero included. Non-completable root
    (DEFEND/DENY) — print the rate marked as the SITREP route, with
    ``closes_per_root_sitrep`` beside it so a reader can tell timed reports
    from closes bought with SITREP volume (the #35 saturation point). Only an
    evaluation whose mission went unrecorded falls back to the #61 dash, and
    only when its claim record is empty — there the route cannot be
    classified, and printing a floor as a rate is the misreading #61 caught.
    """
    rate = m.get("closed_on_root_report_rate")
    if rate is None:
        return "—"
    try:
        mission = MissionType[str(m["root_mission"])]
    except (KeyError, TypeError):
        # Mission unrecorded (old or mixed-scenario evaluation): the route
        # cannot be classified, so keep the #61 guard for the claimless case.
        if m.get("done_claim_episodes_root") == 0:
            return NO_CLAIMS
        return _pct(rate)
    if is_completable(mission):
        return _pct(rate)
    per_sitrep = m.get("closes_per_root_sitrep")
    detail = "sitrep" if per_sitrep is None else f"sitrep, {per_sitrep:.3f}/sitrep"
    return f"{_pct(rate)} ({detail})"


def row(scenario: str, run: str, exception: dict | None = None) -> str:
    d = baseline.run_dir(run)
    final = _json(d / "behavior_final.json")
    best = _json(d / "behavior.json")
    if not final:
        return f"| `{scenario}` | `{run}` | not yet evaluated | — | — | — | — | — | — | — |"
    m = final.get("metrics", {})
    audit = audit_run(d)
    give_back = f"{audit['gap']:.1f} pt" if audit else "—"
    peak = best.get("success_ci95") or "—"
    peak_n = f" (N={best['episodes']})" if best.get("episodes") else ""
    gates = final.get("gates") or []
    failed, unmeasured = split_gates(gates)
    # A waived gate renders AS a FAIL — bold, named — plus who accepted it and
    # when. The waiver changes what blocks the fleet, never what the reader
    # sees: rendering it as anything softer is the pre-v1.19 flag again.
    waived_names = {k.removeprefix("gate:")
                    for k in ((exception or {}).get("waives") or {}) if k.startswith("gate:")}
    waived = [g for g in failed if g in waived_names]
    failed = [g for g in failed if g not in waived_names]
    if failed:
        gate_cell = "**" + ", ".join(failed) + "**"
    elif waived:
        exc = exception or {}
        gate_cell = (f"**FAIL: {', '.join(waived)}** (shipped by {exc.get('by', '?')} "
                     f"decision {exc.get('decided', '?')})")
    elif unmeasured:
        gate_cell = f"pass ({len(unmeasured)} unmeasured)" if len(gates) > len(unmeasured) else "—"
    else:
        gate_cell = "pass" if gates else "—"
    announced = m.get("successes_announced")
    successes = m.get("successes")
    ann_cell = "—" if announced is None or not successes else f"{announced}/{successes}"
    return (
        f"| `{scenario}` | `{run}` | {final.get('success_ci95', '—')} "
        f"(N={final.get('episodes', '?')}) | {peak}{peak_n} | {give_back} | "
        f"{_pct(m.get('human_death_rate'))} | {_pct(m.get('timeout_rate'))} | "
        f"{ann_cell} | {_root_reported(m)} | {gate_cell} |"
    )


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def table() -> str:
    manifest = baseline.load()
    members = manifest.get("runs", {})
    lines = [HEADER, RULE]
    excs = manifest.get("exceptions") or {}
    for scenario in baseline.DOCTRINE_SCENARIOS:
        run = members.get(scenario)
        lines.append(row(scenario, run, excs.get(scenario)) if run
                     else f"| `{scenario}` | — | no baseline member | — | — | — | — | — | — | — |")
    tree = manifest.get("cohort_tree")
    tree_excepted = sorted(sc for sc, e in excs.items()
                           if "provenance:cohort_tree" in (e.get("waives") or {}))
    if tree and tree_excepted:
        plain = len(members) - len(tree_excepted)
        parts = []
        for sc in tree_excepted:
            e = excs[sc]
            parts.append(f"`{sc}` (`{members.get(sc)}`) trained on "
                         f"`{str(e.get('member_tree'))[:8]}` — a disclosed exception "
                         f"({e.get('by', '?')} decision {e.get('decided', '?')}): "
                         f"{e['waives']['provenance:cohort_tree']}")
        stamp = (f"\n{plain} of {len(members)} runs trained against the same `cohort/` "
                 f"tree (`{tree[:8]}`) on the shipped reward defaults, no `--reward` "
                 f"overrides. {' '.join(parts)} Generated by `scripts/results_table.py`; "
                 "`tests/test_results_table.py` fails if this table and the runs disagree.")
    else:
        stamp = (f"\nAll {len(members)} runs trained against the same `cohort/` tree "
                 f"(`{tree[:8]}`) on the shipped reward defaults, no `--reward` overrides. "
                 "Generated by `scripts/results_table.py`; `tests/test_results_table.py` "
                 "fails if this table and the runs disagree." if tree else
                 "\nGenerated by `scripts/results_table.py`.")
    if any("(sitrep" in line for line in lines):
        stamp += (
            "\n\n`(sitrep, …)` in root-reported: a DEFEND/DENY root cannot file a "
            "DONE claim — doctrine masks MISSION COMPLETE shut on a continuous "
            "posture — so the SITREP route is that root's only counted way to "
            "close the operation, and the rate measures it. The number beside it "
            "is closes per root SITREP emitted: low means the closes were bought "
            "with report volume rather than timing.")
    if any(NO_CLAIMS in line for line in lines):
        stamp += (
            f"\n\n`{NO_CLAIMS}` in root-reported: the run's root filed no DONE claim "
            "in any episode and the evaluation did not record the root's mission, "
            "so the route behind the rate cannot be classified. The dash forbids "
            "reading the cell as \"reports reliably\".")
    return "\n".join(lines) + "\n" + stamp


def splice(readme: str, body: str) -> str:
    block = f"{START}\n{body}\n{END}"
    if START in readme and END in readme:
        return re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block,
                      readme, flags=re.S)
    return readme.rstrip() + "\n\n" + block + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="splice into README.md")
    args = ap.parse_args()
    body = table()
    if not args.write:
        print(body)
        return 0
    readme = ROOT / "README.md"
    readme.write_text(splice(readme.read_text(), body))
    print(f"wrote the baseline table into {readme.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
