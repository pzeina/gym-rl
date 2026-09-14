#!/usr/bin/env python
"""Score the interrogative cycle against its pre-registered read
(docs/interrogative-cycle.md §"The read, pre-registered before job 1").

The read was written before job 1 of the v1.28 campaign; this scorer only
does the arithmetic, extending ``scripts/readback_read.py``'s pattern. Six
checks, thresholds pinned by ``tests/test_interrogative_read.py`` BEFORE the
campaign's results exist:

1. **The exchange is used** — REQUEST_STATUS >= 1 per won episode on the
   reporting draws (``--usage run=value``, from a rollout or trace), or the
   cycle is an honest NO EFFECT regardless of anything else that moved.
2. **The close becomes evidence-based** — the extended
   ``root_evidence_probe``'s fresh status-COMPLETE share at confirmed claims
   (``--evidence run=value``) >= 0.50 on at least one reporting draw per
   SEIZE scenario, against the 0.00-0.26 measured everywhere to date.
3. **The original repair bar, unchanged from the read-back cycle** —
   reporting floor >= 0.5 with root death <= 0.05 at N=100 on at least one
   fireteam draw; printed for platoon/patrol too where reporting forms
   (root death read as ``human_death_rate``, the fireteam convention — the
   root IS the human commander in every scenario this read scores).
4. **Formation at full price** — patrol_brique and platoon each form
   sustained reporting (floor >= 0.5, FINAL policy, N=100) in >= 1 of 4
   seeds at default prices, vs 0-of-8 sustained on the v1.27 tree.
5. **Precision guard** — root false-complete <= 0.35 on any draw claimed as
   a repair; a draw above it is a failure named SPAM, not a win.
6. **Fleet success guard** — every scenario non-inferior to its v1.27-read
   candidate (one-sided Fisher, Holm as ONE family; helpers shared with
   ``scripts/exact_tests.py``). For a multi-seed scenario the best
   FINAL-success candidate enters the guard — the draw that would ship.

Inputs are the campaign candidates' committed FINAL evaluations at N>=100
(``behavior_final_n100.json`` preferred, ``behavior_final.json`` accepted);
probe and usage numbers are rollout products, passed in and echoed like
readback_read's ``--probe``. The incumbents are the v1.27-READ candidates —
the runs the read-back verdict was scored on — named here explicitly, NOT
runs/BASELINE.json: the v1.27 seal is still on the owner's desk, and this
read must not move when that decision lands.

    scripts/interrogative_read.py fireteam_v24_seed12 ... squad_v42_seed12 ... \
        --usage fireteam_v24_seed12=1.4 --evidence fireteam_v24_seed12=0.55
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
#: check 1: REQUEST_STATUS per won episode on a reporting draw
USAGE_BAR = 1.0
#: check 2: fresh status-COMPLETE share at confirmed root claims
EVIDENCE_BAR = 0.50
#: check 3 (unchanged from the read-back cycle's registered bars)
REPORT_FLOOR = 0.5
ROOT_DEATH_BAR = 0.05
#: check 4: reporting seeds required per formation scenario (of 4)
FORMATION_MIN_SEEDS = 1
FORMATION_SCENARIOS = ("patrol_brique", "platoon")
#: check 5: root false-complete above this on a claimed repair reads SPAM
FALSE_COMPLETE_BAR = 0.35
#: the SEIZE-family scenarios check 2 ranges over
SEIZE_SCENARIOS = ("fireteam", "squad", "patrol_brique", "platoon", "platoon_hard")

#: check 6 incumbents: the v1.27-READ candidates (the runs readback_read
#: scored), pinned by test — deliberately not runs/BASELINE.json
V127_READ_CANDIDATES = {
    "fireteam": "fireteam_v21_seed13",
    "squad": "squad_v38_seed12",
    "patrol_brique": "patrol_brique_v54_seed12",
    "platoon": "platoon_v24_seed13",
    "platoon_hard": "platoon_hard_v17_seed13",
    "defend_brique": "defend_brique_v22",
    "fireteam_defend": "fireteam_defend_v28",
    "squad_recon": "squad_recon_v16",
    "squad_screen": "squad_screen_v22",
}


def load_final(name: str) -> dict | None:
    """FINAL-policy evaluation at the largest committed N (n100 preferred)."""
    d = find_run(name, RUNS)
    if d is None:
        return None
    for fname in ("behavior_final_n100.json", "behavior_final.json"):
        if (d / fname).is_file():
            return json.loads((d / fname).read_text())
    return None


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


def scenario_of(b: dict) -> str | None:
    return b["metrics"].get("scenario") or b.get("scenario")


def _kv(pairs: list[str], what: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for spec in pairs:
        run, sep, val = spec.partition("=")
        if not sep:
            raise SystemExit(f"--{what} takes RUN=VALUE, got {spec!r}")
        out[run] = float(val)
    return out


def _fmt(x, spec: str = ".2f") -> str:
    return "—" if x is None else format(x, spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("candidates", nargs="+", help="campaign run names, all scenarios")
    ap.add_argument("--usage", action="append", default=[], metavar="RUN=N",
                    help="REQUEST_STATUS per WON episode for a candidate "
                    "(rollout/trace product, echoed into the record)")
    ap.add_argument("--evidence", action="append", default=[], metavar="RUN=F",
                    help="extended root_evidence_probe fresh status-COMPLETE "
                    "share at confirmed claims, per candidate")
    args = ap.parse_args()
    usage = _kv(args.usage, "usage")
    evidence = _kv(args.evidence, "evidence")

    incomplete: list[str] = []
    by_scenario: dict[str, list[tuple[str, dict]]] = {}
    for name in args.candidates:
        b = load_final(name)
        if b is None:
            incomplete.append(f"{name}: no behavior_final(_n100).json anywhere")
            continue
        if b["episodes"] < 100:
            incomplete.append(f"{name}: N={b['episodes']} < 100")
            continue
        scenario = scenario_of(b)
        if scenario is None:
            incomplete.append(f"{name}: no scenario recorded")
            continue
        by_scenario.setdefault(scenario, []).append((name, b))

    reporting: dict[str, list[tuple[str, dict]]] = {
        sc: [(n, b) for n, b in cands
             if (marker(b, "closed_on_root_report_rate") or 0.0) >= REPORT_FLOOR]
        for sc, cands in by_scenario.items()
    }

    print("== 1. the exchange is used (bar: >= "
          f"{USAGE_BAR:.0f} REQUEST_STATUS per won episode, reporting draws) ==")
    any_reporting = False
    for _sc, cands in sorted(reporting.items()):
        for name, _b in cands:
            any_reporting = True
            u = usage.get(name)
            verdict = ("USED" if u is not None and u >= USAGE_BAR
                       else "NO EFFECT" if u is not None else "usage not supplied")
            print(f"  {name:28s} requests/won-ep {_fmt(u)}   [{verdict}]")
            if u is None:
                incomplete.append(f"{name}: reporting draw without --usage")
    if not any_reporting:
        print("  no reporting draws anywhere — the cycle is an honest NO "
              "EFFECT regardless of anything else that moved")

    print("\n== 2. the close becomes evidence-based (bar: fresh "
          f"status-COMPLETE >= {EVIDENCE_BAR} at confirmed claims, >= 1 "
          "reporting draw per SEIZE scenario) ==")
    for sc in SEIZE_SCENARIOS:
        rows = reporting.get(sc, [])
        if not rows:
            print(f"  {sc:16s} no reporting draw — nothing to score")
            continue
        best = None
        for name, _b in rows:
            e = evidence.get(name)
            print(f"  {sc:16s} {name:28s} fresh status-COMPLETE {_fmt(e)}")
            if e is None:
                incomplete.append(f"{name}: reporting draw without --evidence")
            elif best is None or e > best:
                best = e
        if best is not None:
            print(f"  {sc:16s} [{'EVIDENCE-BASED' if best >= EVIDENCE_BAR else 'FLAGS DARK'}]")

    print("\n== 3. the repair bar (floor >= "
          f"{REPORT_FLOOR}, root death <= {ROOT_DEATH_BAR}, N=100) ==")
    repaired: list[str] = []
    for sc in ("fireteam", "platoon", "patrol_brique"):
        for name, b in by_scenario.get(sc, []):
            rep = marker(b, "closed_on_root_report_rate")
            death = b["metrics"].get("human_death_rate")
            fc = b["metrics"].get("false_complete_rate_root")
            if rep is None or rep < REPORT_FLOOR:
                verdict = "MUTE"
            elif death is None or death > ROOT_DEATH_BAR:
                verdict = "NO-REPAIR"
            elif fc is not None and fc > FALSE_COMPLETE_BAR:
                verdict = "SPAM"          # check 5: precision guard on a claimed repair
            else:
                verdict = "REPAIRED"
                repaired.append(name)
            print(f"  {name:28s} report {_fmt(rep, '.3f')}  root death "
                  f"{_fmt(death, '.3f')}  false-complete {_fmt(fc, '.3f')} "
                  f"(bar <= {FALSE_COMPLETE_BAR})   [{verdict}]")
    print(f"  fireteam repair: {'MET' if any(n for n in repaired if scenario_of(load_final(n)) == 'fireteam') else 'NOT MET'}")

    print("\n== 4. formation at full price (>= "
          f"{FORMATION_MIN_SEEDS} of 4 seeds reporting; v1.27 tree was 0-of-8) ==")
    for sc in FORMATION_SCENARIOS:
        n_rep = len(reporting.get(sc, []))
        n_all = len(by_scenario.get(sc, []))
        print(f"  {sc:16s} {n_rep} of {n_all} seeds reporting   "
              f"[{'FORMED' if n_rep >= FORMATION_MIN_SEEDS else 'NOT FORMED'}]")

    print("\n== 5. precision guard (root false-complete <= "
          f"{FALSE_COMPLETE_BAR} on any claimed repair; above = SPAM) ==")
    print("  applied per-draw inside check 3 — a SPAM verdict there is this "
          "guard firing; false-complete is echoed on every scored row")

    print("\n== 6. fleet guard vs the v1.27-read candidates "
          "(one-sided Fisher, Holm as one family) ==")
    pvals: dict[str, float] = {}
    rows: dict[str, str] = {}
    for sc, cands in sorted(by_scenario.items()):
        inc_name = V127_READ_CANDIDATES.get(sc)
        if inc_name is None:
            incomplete.append(f"{sc}: no v1.27-read incumbent registered")
            continue
        inc = load_final(inc_name)
        if inc is None or inc["episodes"] < 100:
            incomplete.append(f"{sc}: incumbent {inc_name} lacks N>=100")
            continue
        # the best FINAL-success draw is the one that would ship
        name, b = max(cands, key=lambda nb: successes(nb[1])[0])
        a_s, a_n = successes(b)
        i_s, i_n = successes(inc)
        p = fisher_one_sided_less(a_s, a_n - a_s, i_s, i_n - i_s)
        pvals[sc] = p
        rows[sc] = (f"  {sc:16s} {name:28s} {a_s}/{a_n}  vs "
                    f"{inc_name} {i_s}/{i_n}   p={p:.4f}")
    broken = holm_reject(pvals, ALPHA)
    for sc, row in rows.items():
        print(f"{row}   [{'BROKEN' if broken.get(sc) else 'OK'}]")

    if incomplete:
        print("\nINCOMPLETE — the read cannot conclude:")
        for line in incomplete:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
