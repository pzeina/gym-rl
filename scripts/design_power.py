#!/usr/bin/env python
"""Before the campaign: can this design reject at all, and on which outcomes?

    scripts/design_power.py                                  # the live rdb3_seeds design
    scripts/design_power.py --pair 12:?:mute --pair 13:?:? --alpha 0.05
    scripts/design_power.py --size 5 6 7 8 --measured 1/4    # how many seeds buy a test

**Why this runs first** (refs assurance #56). A 3M-step run costs ~30 minutes of
CPU and no tokens, so the cheap thing about a campaign is launching it and the
expensive thing is discovering afterwards that no outcome it could have produced
would have settled anything. That happened here: the six-run `rdb3_seeds`
campaign asks the right question with the right decision rule, and **it can
produce a significant result on exactly 1 of its 64 possible outcomes** — the
one that needs seed 16 to go mute at `rdb=1.0`, which is the branch its own
evidence says is unlikely. Every number in that sentence is arithmetic over a
table that existed before the first job launched.

**The two readings, and why the answer differs between them.** Two arms that
share their seeds are matched, so:

* **unpaired (Fisher)** treats the arms as independent — k of n runs report here
  against j of m there. It ignores the pairing, which is why it can look more
  powerful than the paired reading at these sizes;
* **paired (McNemar)** uses the discordant pairs only. Concordant pairs carry no
  information, so a comparison arm that already reports at a seed *removes* that
  seed from the evidence. At five pairs the smallest attainable p is 0.0625 over
  every outcome including perfect separation; conditioned on a comparison arm
  that reports at one of them, 0.125.

**What "ceiling" means here.** The smallest p the design can attain, minimised
over every outcome it could possibly produce — the best case, not the expected
case. A design whose ceiling is above alpha cannot reject, so the honest read-out
must not name that test at all: a p = 0.17 from a design that could never have
gone below 0.17 is not a null result, and writing it down as one is how a
campaign that answered its question descriptively gets read as having failed to.

Descriptive settlement needs none of this. "3.0 splits across seeds too" is
settled by the first mute cell and wants no test; it is the other branch — "the
price is real" — that needs inference, and that is the branch to size for.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.exact_tests import fisher_two_sided, mcnemar_two_sided  # noqa: E402

#: The design `scripts/campaigns/patrol_brique_incumbent_rdb3_seeds.jobs`
#: launched: five pending `rdb=3.0` cells against the `rdb=1.0` arm, whose four
#: measured cells are seeds 12/13/15 mute and seed 14 reporting, plus its one
#: pending cell (seed 16, `patrol_brique_v23`). The labels are not transcribed
#: from a note — re-derive them with
#: ``scripts/reporting_channel.py --scenario patrol_brique --price 1.0``.
CAMPAIGN: tuple[str, ...] = ("12:?:mute", "13:?:mute", "14:?:reporting", "15:?:mute", "16:?:?")

#: The comparison arm as measured when the campaign was sized: 1 of 4 reporting.
CAMPAIGN_MEASURED: tuple[int, int] = (1, 4)

LABELS: dict[str, bool | None] = {"reporting": True, "mute": False, "?": None}
MAX_PENDING = 22  # 4M outcomes; past this the enumeration is the wrong instrument


@dataclass(frozen=True)
class Pair:
    """One seed, and what each arm did at it — ``None`` for a cell not yet run."""

    seed: str
    first: bool | None
    second: bool | None


def parse_pair(spec: str) -> Pair:
    """``12:?:mute`` — seed, first arm, second arm; ``?`` for a pending cell."""
    try:
        seed, first, second = spec.split(":")
        return Pair(seed, LABELS[first.lower()], LABELS[second.lower()])
    except (ValueError, KeyError):
        raise SystemExit(f"cannot read design cell {spec!r}: "
                         f"expected SEED:{'|'.join(LABELS)}:{'|'.join(LABELS)}") from None


def pending_cells(pairs: list[Pair]) -> list[tuple[int, str]]:
    return [(i, side) for i, pair in enumerate(pairs)
            for side in ("first", "second") if getattr(pair, side) is None]


def fill(pairs: list[Pair], assignment: dict[tuple[int, str], bool]) -> list[Pair]:
    return [Pair(p.seed,
                 assignment.get((i, "first"), p.first),
                 assignment.get((i, "second"), p.second))
            for i, p in enumerate(pairs)]


def unpaired_p(pairs: list[Pair]) -> float:
    """Fisher exact on the two arms' totals, pairing discarded."""
    n = len(pairs)
    first = sum(1 for p in pairs if p.first)
    second = sum(1 for p in pairs if p.second)
    return fisher_two_sided(first, n - first, second, n - second)


def paired_p(pairs: list[Pair]) -> float:
    """Exact McNemar over the discordant pairs, which is all the evidence there is."""
    one_way = sum(1 for p in pairs if p.first and not p.second)
    other_way = sum(1 for p in pairs if p.second and not p.first)
    return mcnemar_two_sided(one_way, other_way)


def describe(pairs: list[Pair]) -> str:
    first = sum(1 for p in pairs if p.first)
    second = sum(1 for p in pairs if p.second)
    return f"first arm {first}/{len(pairs)}, second arm {second}/{len(pairs)}"


def power(pairs: list[Pair], alpha: float = 0.05) -> dict:
    """Enumerate every outcome the design can produce and score both readings.

    Returns each reading's ceiling (the smallest attainable p), how many of the
    possible outcomes reject at ``alpha``, and what those outcomes look like. An
    outcome here is a full labelling of the pending cells — 2 per cell — because
    that is the unit a campaign actually produces: five runs land as five
    labels, not as a count.
    """
    pending = pending_cells(pairs)
    if len(pending) > MAX_PENDING:
        raise SystemExit(f"{len(pending)} pending cells is {2 ** len(pending)} outcomes; "
                         "size this design analytically instead")
    readings = {"unpaired": unpaired_p, "paired": paired_p}
    out: dict = {"pairs": len(pairs), "pending": len(pending), "outcomes": 2 ** len(pending),
                 "alpha": alpha}
    for name in readings:
        out[name] = {"ceiling": 1.0, "rejecting": 0, "at": []}
    for values in product((False, True), repeat=len(pending)):
        filled = fill(pairs, dict(zip(pending, values, strict=True)))
        for name, test in readings.items():
            p = test(filled)
            record = out[name]
            record["ceiling"] = min(record["ceiling"], p)
            if p < alpha:
                record["rejecting"] += 1
                if len(record["at"]) < 6:
                    record["at"].append((describe(filled), p))
    return out


def sizing(seeds: list[int], measured: tuple[int, int], alpha: float = 0.05) -> list[dict]:
    """Best case per seed count: the new arm reports everywhere, the old one holds.

    ``measured`` is the comparison arm as already run (reporting, total); its
    rate is assumed to continue, so the reporting count it contributes is
    ``round(rate x seeds)`` and ``new runs`` counts only the cells that do not
    exist yet. The Fisher column is NOT monotone in the seed count — the rounded
    comparison count steps up between sizes — which is an artifact of the
    assumption, not a property of the designs.
    """
    reporting, total = measured
    rate = reporting / total if total else 0.0
    rows = []
    for n in seeds:
        held = round(rate * n)
        rows.append({
            "seeds": n,
            "new_runs": n + max(0, n - total),
            "unpaired": fisher_two_sided(n, 0, held, n - held),
            "paired": mcnemar_two_sided(n - held, 0),
            "assumed": f"{held}/{n}",
        })
    return rows


def report(pairs: list[Pair], alpha: float, sizes: list[int] | None,
           measured: tuple[int, int]) -> None:
    result = power(pairs, alpha)
    print(f"design: {result['pairs']} seeds, {result['pending']} pending cells, "
          f"{result['outcomes']} possible outcomes, alpha = {alpha}\n")
    for pair in pairs:
        cells = [("pending" if v is None else "REPORTING" if v else "mute") for v in (pair.first, pair.second)]
        print(f"    seed {pair.seed:>4}   {cells[0]:<10} {cells[1]:<10}")
    print()
    for name, label in (("unpaired", "unpaired (Fisher)"), ("paired", "paired (McNemar)")):
        record = result[name]
        verdict = ("CANNOT REJECT — do not name this test in the read-out"
                   if record["rejecting"] == 0 else
                   f"rejects on {record['rejecting']} of {result['outcomes']} outcomes")
        print(f"    {label:<20} ceiling p = {record['ceiling']:.4f}   {verdict}")
        for description, p in record["at"]:
            print(f"        {description}, p = {p:.4f}")
    print()
    if sizes:
        print(f"sizing, best case (new arm reports at every seed, comparison arm holds "
              f"{measured[0]}/{measured[1]})\n")
        print(f"    {'seeds':>5} {'new runs':>9} {'assumed':>8} {'unpaired':>9} {'paired':>8}")

        def capable(p: float) -> str:
            return "OK " if p < alpha else "   "

        for row in sizing(sizes, measured, alpha):
            print(f"    {row['seeds']:>5} {row['new_runs']:>9} {row['assumed']:>8} "
                  f"{row['unpaired']:>9.4f}{capable(row['unpaired'])}"
                  f"{row['paired']:>8.4f}{capable(row['paired'])}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pair", action="append", metavar="SEED:FIRST:SECOND",
                        help="one seed of the design; label is reporting, mute or ? (repeatable)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--size", nargs="*", type=int, default=None,
                        help="also print the sizing table for these seed counts")
    parser.add_argument("--measured", default=f"{CAMPAIGN_MEASURED[0]}/{CAMPAIGN_MEASURED[1]}",
                        help="the comparison arm as already measured, REPORTING/TOTAL")
    args = parser.parse_args()

    specs = args.pair or list(CAMPAIGN)
    sizes = args.size if args.size is not None else ([5, 6, 7, 8] if not args.pair else None)
    reporting, _, total = args.measured.partition("/")
    report([parse_pair(s) for s in specs], args.alpha, sizes, (int(reporting), int(total)))


if __name__ == "__main__":
    main()
