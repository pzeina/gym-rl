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
  now lives, and it is the honest way to keep measuring it. The rate has a
  second route into its numerator — a root SITREP landing on the final step
  closes the episode with no DONE claim anywhere — so on a run whose root never
  claims it reads a floor, not a zero. Where the claim record is empty the cell
  says so instead of printing the floor (issue #61).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
    """The root-reported cell — the rate where a claim record exists, else absent.

    ``closed_on_root_report_rate`` counts closes by root DONE claim *and* closes
    by a root SITREP landing on the final step, so on a run whose root files no
    DONE claim at all the rate reads a floor, not a zero: ``fireteam_defend_v23``
    and ``defend_brique_v17`` sit at 1.000 with ``done_claim_episodes_root: 0``.
    Printing 100% there, next to ``announced 99/99``, reads as a completion
    channel working perfectly when the root filed nothing — the same misreading
    the fleet board's false-DONE dash forbids (8537c2c), one column over. Where
    the claim record is empty the cell says so, and the note under the table
    says which reading that forbids. Older evaluations that never recorded
    ``done_claim_episodes_root`` keep their rate: absence of the counter is not
    evidence of a mute root.
    """
    rate = m.get("closed_on_root_report_rate")
    if rate is not None and m.get("done_claim_episodes_root") == 0:
        return NO_CLAIMS
    return _pct(rate)


def row(scenario: str, run: str) -> str:
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
    if failed:
        gate_cell = "**" + ", ".join(failed) + "**"
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
    for scenario in baseline.DOCTRINE_SCENARIOS:
        run = members.get(scenario)
        lines.append(row(scenario, run) if run
                     else f"| `{scenario}` | — | no baseline member | — | — | — | — | — | — | — |")
    tree = manifest.get("cohort_tree")
    stamp = (f"\nAll {len(members)} runs trained against the same `cohort/` tree "
             f"(`{tree[:8]}`) on the shipped reward defaults, no `--reward` overrides. "
             "Generated by `scripts/results_table.py`; `tests/test_results_table.py` "
             "fails if this table and the runs disagree." if tree else
             "\nGenerated by `scripts/results_table.py`.")
    if any(NO_CLAIMS in line for line in lines):
        stamp += (
            f"\n\n`{NO_CLAIMS}` in root-reported: the run's root filed no DONE claim "
            "in any episode — every close came from a SITREP landing on the final "
            "step, a route that reads as a floor on a claimless root. The dash "
            "forbids reading the cell as \"reports reliably\"; the operations "
            "closed, but not because the root claimed them complete.")
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
