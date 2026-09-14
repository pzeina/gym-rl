#!/usr/bin/env python
"""Score the read-back cycle against its pre-registered read (docs/readback-cycle.md).

The read was written before job 1 of the v1.27 campaign; this scorer only
does the arithmetic. Two halves:

1. **The fireteam repair.** For every fireteam candidate whose FINAL-policy
   evaluation at N>=100 clears the 0.5 reporting floor
   (``closed_on_root_report_rate >= 0.5``): root death
   (``human_death_rate`` — fireteam's root IS the human TL) must be
   ``<= 0.05``, and the root-evidence probe's fresh-subordinate-DONE share
   at confirmed claims must RISE above the 0.29 measured on v17 (the
   stricter of the pre-cycle 0.29/0.15 band). The probe is a 50-episode
   rollout, not a committed artifact, so its numbers are passed in via
   ``--probe run=value`` and echoed into the record. Per-draw verdicts:
   REPAIRED (floor + root death + probe), EVIDENCE-UNUSED (floor + root
   death, probe below the bar — the trade moved without the channel: look
   at the claim-discipline line before celebrating), NO-REPAIR (floor held,
   root death above 0.05), MUTE (floor missed).
2. **The fleet guard.** Each non-fireteam candidate against its v1.26
   incumbent: one-sided Fisher non-inferiority on FINAL-policy success at
   N>=100, Holm-corrected as one family (helpers shared with the dispersion
   prereg in scripts/exact_tests.py). The guard convicts the CYCLE, so a
   Holm rejection reads BROKEN for that scenario.

Claim discipline (``false_complete_rate_root``, claims per claiming
episode) is printed, never gated — the prereg did not gate it, and the
2026-09-14 probe found a rear-spam mode the gate would have been written
against had it been known. Printing it is how the next prereg learns.

    scripts/readback_read.py fireteam_v20_seed12 ... squad_v38_seed12 ... \
        --probe fireteam_v20_seed12=0.02 --probe fireteam_v21_seed13=0.26
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.exact_tests import fisher_one_sided_less, holm_reject
from scripts.fleet_status import find_run

RUNS = Path(__file__).resolve().parent.parent / "runs"
ALPHA = 0.05
REPORT_FLOOR = 0.5
ROOT_DEATH_BAR = 0.05
PROBE_BAR = 0.29  # the stricter pre-cycle measurement (v17)


def load_final(name: str) -> dict | None:
    d = find_run(name, RUNS)
    if d is None or not (d / "behavior_final.json").is_file():
        return None
    return json.loads((d / "behavior_final.json").read_text())


def marker(b: dict, name: str) -> float | None:
    for m in b.get("markers", []):
        if m["name"] == name:
            return m["value"]
    return b["metrics"].get(name)


def successes(b: dict) -> tuple[int, int]:
    eps = b["episodes"]
    per = b.get("per_episode") or []
    if per and all("success" in e for e in per):
        return sum(1 for e in per if e["success"]), eps
    return round(b["metrics"]["success_rate"] * eps), eps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("candidates", nargs="+", help="campaign run names, all scenarios")
    ap.add_argument("--probe", action="append", default=[],
                    metavar="RUN=FRESH", help="root_evidence_probe fresh-sub-DONE "
                    "share at confirmed claims, per fireteam candidate")
    args = ap.parse_args()
    probe = {}
    for spec in args.probe:
        run, _, val = spec.partition("=")
        probe[run] = float(val)

    manifest = json.loads((RUNS / "BASELINE.json").read_text())
    incumbents = manifest["runs"]

    incomplete: list[str] = []
    fire: list[tuple[str, dict]] = []
    guard: list[tuple[str, str, dict]] = []
    for name in args.candidates:
        b = load_final(name)
        if b is None:
            incomplete.append(f"{name}: no behavior_final.json anywhere")
            continue
        if b["episodes"] < 100:
            incomplete.append(f"{name}: N={b['episodes']} < 100")
            continue
        scenario = b["metrics"].get("scenario") or b.get("scenario")
        if scenario == "fireteam":
            fire.append((name, b))
        else:
            guard.append((scenario, name, b))

    print("== the fireteam repair ==")
    for name, b in fire:
        rep = marker(b, "closed_on_root_report_rate")
        death = b["metrics"].get("human_death_rate")
        fc = b["metrics"].get("false_complete_rate_root")
        cpc = b["metrics"].get("done_claims_per_claiming_episode_root")
        pr = probe.get(name)
        if rep is None or rep < REPORT_FLOOR:
            verdict = "MUTE"
        elif death is None or death > ROOT_DEATH_BAR:
            verdict = "NO-REPAIR"
        elif pr is None or pr <= PROBE_BAR:
            verdict = "EVIDENCE-UNUSED"
        else:
            verdict = "REPAIRED"
        pfmt = "—" if pr is None else f"{pr:.2f}"
        print(f"  {name:24s} report {rep:.3f}  root death {death:.3f}  "
              f"probe fresh-DONE {pfmt} (bar >{PROBE_BAR})   [{verdict}]")
        print(f"  {'':24s} claim discipline (printed, not gated): "
              f"false-complete {fc}  claims/claiming-ep {cpc}")

    print("\n== the fleet guard (one-sided Fisher, Holm over the family) ==")
    pvals: dict[str, float] = {}
    rows: dict[str, str] = {}
    for scenario, name, b in guard:
        inc_name = incumbents.get(scenario)
        inc = load_final(inc_name) if inc_name else None
        if inc is None or inc["episodes"] < 100:
            incomplete.append(f"{scenario}: incumbent {inc_name} lacks N>=100")
            continue
        a_s, a_n = successes(b)
        i_s, i_n = successes(inc)
        p = fisher_one_sided_less(a_s, a_n - a_s, i_s, i_n - i_s)
        pvals[scenario] = p
        rows[scenario] = (f"  {scenario:16s} {name:26s} {a_s}/{a_n}  vs "
                          f"{inc_name} {i_s}/{i_n}   p={p:.4f}")
    broken = holm_reject(pvals, ALPHA)
    for scenario, row in rows.items():
        print(f"{row}   [{'BROKEN' if broken.get(scenario) else 'OK'}]")

    if incomplete:
        print("\nINCOMPLETE — the read cannot conclude:")
        for line in incomplete:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
