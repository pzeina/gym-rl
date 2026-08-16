#!/usr/bin/env python
"""Which commanders report a truthful MISSION COMPLETE, and does the seed carry it?

    scripts/reporting_channel.py                       # every run that can be labelled
    scripts/reporting_channel.py --scenario patrol_brique
    scripts/reporting_channel.py --pairs squad patrol_brique   # the transfer question
    scripts/reporting_channel.py --arms 1.0 3.0                # the price question
    scripts/reporting_channel.py --spam                        # does reporting mean lying?

**The question this exists for** (refs assurance #55). The 2026-08-15 retraction
left the reporting channel *seed-determined*: whether a commander ever files a
truthful completion is a property of the optimisation path, not of the reward or
the chart. One of the three routes that leaves open for v1.21 is a DECLARED
seed-selection policy — train k seeds and ship one that reports — and that route
only works if a seed is GLOBALLY good. If the seed matters only in interaction
with the scenario, the policy has to be declared per scenario, and the
declaration costs a fleet per scenario rather than a fleet.

Those two worlds are separated by runs already on disk, because the squad and
`patrol_brique` cells share seeds at a fixed arm. This pairs them and prints the
exact McNemar over the discordant pairs. It is a READER over committed
evaluations: it launches nothing, evaluates nothing, and re-derives every arm
from `config.json` / `economics.json` rather than trusting a run's name.

**How a run is labelled, and why not on the rate.** The mute/reporting split is
a binary property of a policy, so it is read off ``done_claim_episodes_root`` —
the number of episodes in which whoever held the root filed a MISSION COMPLETE —
as a share of episodes, at BOTH published checkpoints. Two rules do the work:

* **``closed_on_root_report_rate`` is the wrong instrument for this cut, in both
  directions.** It answers "did the root's report close the window", and on a
  continuous-posture root the report that closes it is a SITREP, not a claim —
  so the whole defend family reads 0.97-1.00 on ZERO root claims. In the other
  direction it does not floor at zero either: a root SITREP landing on the ENDEX
  step enters the numerator, and #55 measured 0.020-0.104 on `patrol_brique`
  arms with no root claims at all, where a naive 0.05 rate-cut would read
  "reporting". The report prints both and NAMES every run the two disagree on.
* **A run whose two checkpoints disagree is dropped, not resolved.** `squad_v10c`
  claims in 18 of 100 episodes at `ckpt_best` and in 0 of 100 at `ckpt_latest`;
  no single label describes that policy, and picking the checkpoint that suits
  the argument is how a 4-of-4 gets built out of a 2-of-4.

The cuts sit in a wide empty band — on the record the highest mute arm claims in
2% of episodes and the lowest reporting arm in 18% — so any cut inside the gap
gives the same table. What falls INSIDE the band is reported as `undecided`
rather than rounded to the nearer mode.

**Every reading here prints what it could have shown, not only what it did**
(refs assurance #59). The matched tables carry the realized ceilings out of
`design_power` — an exact test conditions on a statistic the runs decided, so a
p of 1.0000 off four discordant pairs is not the same object as a p of 1.0000
off a margin that could have reached 0.0070 — and `--spam` carries the
leave-one-out range of its rank correlation, because the nine-run series it
reads was called monotone after being sorted and before being measured.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT))

from scripts import design_power  # noqa: E402
from scripts.baseline import cohort_tree  # noqa: E402
from scripts.exact_tests import jackknife_rho, mcnemar_two_sided, spearman_rho  # noqa: E402
from scripts.fleet_status import run_dirs  # noqa: E402

#: Share of episodes carrying at least one root MISSION COMPLETE, above which a
#: checkpoint is REPORTING and below which it is mute. The 9x gap between the
#: modes is what makes a two-sided cut honest: everything measured is either at
#: or under 0.02 or at or over 0.18, so the band between them is empty on the
#: record and a run that lands in it is a new observation, not a rounding
#: decision.
REPORTING_CUT: float = 0.10
MUTE_CUT: float = 0.02

#: The cut a reader would put on ``closed_on_root_report_rate`` if they used it
#: for this. Kept only so the report can show where it disagrees with the claim
#: count — it is never used to label anything.
RATE_CUT: float = 0.05

#: How the `cohort/` tree of a run is read for the one structural difference the
#: `patrol_brique`/squad seed arms were built around: #42's chart link, which
#: files a promoted successor under its new superior.
#:
#: Resolved from the run's recorded ``git_commit`` by reading that commit's own
#: ``cohort/core/units.py``, never from the run's name — a name says what
#: somebody meant to launch and a tree says what trained. Three states exist and
#: they are not two: between `56ada9a` and `da24b42` BOTH appends were live and
#: the promoted agent was linked into its leader twice, which is a defective
#: tree rather than a treatment arm (see `da24b42`), so it is named as its own
#: arm and never pooled with either side.
CHART_LINK = "parent.subordinate_ids.append(successor.id)"
PRE42_APPEND = "successor.subordinate_ids.append(promoted.id)"

_TREE_CACHE: dict[str, str] = {}


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def classify_units_source(source: str | None) -> str:
    """`present` / `double-linked` / `absent` / `unresolved` for one `units.py`.

    ``unresolved`` for source that could not be read at all. It is its own arm,
    so a run whose tree cannot be resolved never silently joins one.
    """
    if source is None:
        return "unresolved"
    if CHART_LINK in source:
        return "double-linked" if PRE42_APPEND in source else "present"
    return "absent"


def chart_link_state(commit: str | None) -> str:
    """The chart-link state of the tree a run was trained against.

    Read out of the recorded commit with ``git show``, which is why a run whose
    commit is not in this clone (an experiment branch that was never pushed, a
    run predating the provenance field) reads ``unresolved`` rather than being
    guessed from its name.
    """
    if not commit:
        return "unresolved"
    if commit not in _TREE_CACHE:
        try:
            out = subprocess.run(["git", "show", f"{commit}:cohort/core/units.py"],
                                 cwd=ROOT, capture_output=True, text=True, timeout=30)
            source = out.stdout if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            source = None
        _TREE_CACHE[commit] = classify_units_source(source)
    return _TREE_CACHE[commit]


def checkpoint_facts(path: Path) -> dict | None:
    """One published evaluation, reduced to the claim share and the close rate."""
    data = _json(path)
    episodes = data.get("episodes")
    metrics = data.get("metrics") or {}
    claims = metrics.get("done_claim_episodes_root")
    if not episodes or claims is None:
        return None
    share = claims / episodes
    return {
        "episodes": episodes,
        "claim_episodes": claims,
        "share": share,
        "label": ("REPORTING" if share >= REPORTING_CUT
                  else "mute" if share <= MUTE_CUT else "undecided"),
        "rate": metrics.get("closed_on_root_report_rate"),
        "false": metrics.get("false_complete_rate_root"),
    }


def rate_label(rate: float | None) -> str | None:
    """What a cut on ``closed_on_root_report_rate`` would have said."""
    return None if rate is None else ("REPORTING" if rate >= RATE_CUT else "mute")


def run_facts(run: Path) -> dict | None:
    """One run: its arm, both checkpoints, and the label the two agree on.

    ``None`` for a run with no published evaluation carrying the claim counter —
    the pre-v1.13 corpora predate it, and an absent counter is not a zero.
    """
    config = _json(run / "config.json")
    scenario = config.get("scenario")
    if scenario is None:
        return None
    best = checkpoint_facts(run / "behavior.json")
    final = checkpoint_facts(run / "behavior_final.json")
    if best is None and final is None:
        return None
    economics = _json(run / "economics.json")
    commit = economics.get("git_commit")
    overrides = tuple(o for o in (economics.get("reward_overrides") or [])
                      if not o.startswith("root_done_bonus="))
    labels = {c["label"] for c in (best, final) if c}
    if best is None or final is None:
        label = "one checkpoint"
    elif labels == {"REPORTING"} or labels == {"mute"}:
        label = labels.pop()
    else:
        label = "SPLIT"
    return {
        "run": run.name,
        "scenario": scenario,
        "seed": config.get("seed"),
        "commit": commit,
        "tree": cohort_tree(commit),
        "chart_link": chart_link_state(commit),
        "price": (economics.get("rewards") or {}).get("root_done_bonus"),
        "overrides": overrides,
        "best": best,
        "final": final,
        "label": label,
    }


def arm_of(facts: dict) -> tuple:
    """What has to match for two runs to be the same experimental arm.

    The price actually paid (override or default), any OTHER override, and the
    chart-link state of the tree. Deliberately NOT the `cohort/` tree digest:
    that moves on a docstring commit, and two runs either side of one are the
    same experiment. The digests are printed per pair instead, so a comparison
    that does span two trees says so rather than being silently dropped or
    silently pooled.
    """
    return (facts["price"], facts["overrides"], facts["chart_link"])


def arm_label(arm: tuple) -> str:
    price, overrides, chart = arm
    price_text = "rdb —" if price is None else f"rdb {price:g}"
    extra = ("+" + ",".join(overrides)) if overrides else ""
    return f"{price_text}, chart {chart}{extra}"


def collect(runs_dir: Path = RUNS) -> list[dict]:
    """Every run on disk that carries a labellable evaluation, archives included."""
    rows = [f for run in run_dirs(runs_dir) if (f := run_facts(run))]
    return sorted(rows, key=lambda r: (r["scenario"], arm_label(arm_of(r)), r["seed"] or 0))


def rate_disagreements(rows: list[dict]) -> list[tuple[str, str, dict]]:
    """Runs where a cut on the close rate would contradict the claim count.

    This is the instrument check, and it fires on the repo's own corpus in both
    directions: the defend family reads ~1.00 with zero claims, and a mute
    `patrol_brique` arm can read above the rate cut on a SITREP that happened to
    land on the ENDEX step. Whenever this list is non-empty, a mute/reporting
    table built on the rate is a different table from this one.
    """
    out = []
    for row in rows:
        for name in ("best", "final"):
            checkpoint = row[name]
            if checkpoint is None:
                continue
            other = rate_label(checkpoint["rate"])
            if other is not None and checkpoint["label"] != "undecided" and other != checkpoint["label"]:
                out.append((row["run"], name, checkpoint))
    return out


def matched_pairs(rows: list[dict], held_of, side_of, first, second) -> dict:
    """The generic matched design: hold ``held_of`` fixed, vary ``side_of``.

    Two questions in this repo are the same arithmetic over different cuts —
    "does a reporting seed transfer across scenarios" holds the arm and varies
    the scenario, and "does the price buy the channel" holds the scenario and
    varies `root_done_bonus` — so they share one implementation and one set of
    rules about which cells are allowed to enter a table.

    The pairing is matched by construction, so the primary reading is McNemar
    over the discordant pairs. Fisher is computed too and reported beside it,
    never instead of it: which one a design named PRIMARY is a property of the
    design, and picking the one that clears afterwards is test-shopping.
    """
    by_cell: dict[tuple, list[dict]] = {}
    for row in rows:
        if side_of(row) in (first, second) and row["label"] in ("REPORTING", "mute"):
            by_cell.setdefault((held_of(row), row["seed"], side_of(row)), []).append(row)
    pairs, ambiguous = [], []
    for (arm, seed, side), members in sorted(by_cell.items(), key=lambda kv: str(kv[0])):
        if side != first:
            continue
        others = by_cell.get((arm, seed, second), [])
        if not others:
            continue
        # A cell holding two runs is only ambiguous if they DISAGREE. Two runs
        # of one arm at one seed that carry the same label are a replication,
        # and dropping the cell would throw a confirmed observation away.
        if len({m["label"] for m in members}) > 1 or len({o["label"] for o in others}) > 1:
            ambiguous.append((arm, seed))
            continue
        # Where a cell replicates, show the pair through the two runs that share
        # a `cohort/` tree if any do — same label either way, but a comparison
        # that CAN be made on one tree should not be reported as spanning two.
        same_tree = [(m, o) for m in members for o in others if m["tree"] == o["tree"]]
        one, two = same_tree[0] if same_tree else (members[0], others[0])
        pairs.append({"arm": arm, "seed": seed, "first": one, "second": two,
                      "runs": ([m["run"] for m in members], [o["run"] for o in others])})
    one_way = sum(1 for p in pairs if p["first"]["label"] == "REPORTING" and p["second"]["label"] == "mute")
    other_way = sum(1 for p in pairs if p["first"]["label"] == "mute" and p["second"]["label"] == "REPORTING")
    agree = sum(1 for p in pairs if p["first"]["label"] == p["second"]["label"])
    n = len(pairs)
    reporting_first = sum(1 for p in pairs if p["first"]["label"] == "REPORTING")
    reporting_second = sum(1 for p in pairs if p["second"]["label"] == "REPORTING")
    expected = (None if not n else
                (reporting_first * reporting_second + (n - reporting_first) * (n - reporting_second)) / n**2 * n)
    result = {
        "pairs": pairs,
        "ambiguous": sorted(set(ambiguous), key=str),
        "agree": agree,
        "expected_agree": expected,
        "one_way": one_way,
        "other_way": other_way,
        "reporting_first": reporting_first,
        "reporting_second": reporting_second,
        "p": mcnemar_two_sided(one_way, other_way),
        "realized": None,
    }
    if n:
        # Both p values AND what each could have attained on this table. A table
        # is not a null until its own ceiling says it could have been one, and
        # the two readings disagree about that more often than they disagree
        # about p (refs assurance #59).
        result["realized"] = design_power.realized(
            [design_power.Pair(str(p["seed"]), p["first"]["label"] == "REPORTING",
                               p["second"]["label"] == "REPORTING")
             for p in pairs])
    return result


def cross_scenario_pairs(rows: list[dict], first: str, second: str) -> dict:
    """Seeds carrying a definite label on BOTH scenarios at one fixed arm."""
    return matched_pairs(rows, arm_of, lambda row: row["scenario"], first, second)


def price_pairs(rows: list[dict], first: float, second: float) -> dict:
    """Seeds carrying a definite label at BOTH prices, everything else held.

    The `patrol_brique` `rdb=3.0` campaign's design. What is held is the
    scenario, the chart-link state of the trained tree and any OTHER reward
    override — everything ``arm_of`` holds except the price, which is the
    treatment. A seed whose run at either price is SPLIT does not enter the
    table, because the campaign pre-registered that rule before it was scored.
    """
    def held(row: dict) -> tuple:
        return (row["scenario"], row["overrides"], row["chart_link"])

    return matched_pairs(rows, held, lambda row: row["price"], first, second)


def _checkpoint_cell(checkpoint: dict | None) -> str:
    if checkpoint is None:
        return f"{'—':>22}"
    rate = "—" if checkpoint["rate"] is None else f"{checkpoint['rate']:.3f}"
    return (f"{checkpoint['claim_episodes']:>3}/{checkpoint['episodes']:<3} "
            f"{checkpoint['label']:<9} rate {rate:>5}")


def print_table(rows: list[dict]) -> None:
    print(f"{len(rows)} runs with a labellable evaluation "
          f"(claim share >= {REPORTING_CUT:.2f} reports, <= {MUTE_CUT:.2f} mute)\n")
    arm = None
    for row in rows:
        here = (row["scenario"], arm_label(arm_of(row)))
        if here != arm:
            arm = here
            print(f"  {here[0]} · {here[1]}")
        tree = (row["tree"] or "unresolved")[:8]
        print(f"    {row['run']:<32} seed {row['seed']!s:>3}  {tree}  "
              f"best {_checkpoint_cell(row['best'])}  final {_checkpoint_cell(row['final'])}  "
              f"=> {row['label']}")
    print()


def print_rate_check(rows: list[dict]) -> None:
    disagreements = rate_disagreements(rows)
    print(f"closed_on_root_report_rate vs the claim count: {len(disagreements)} checkpoints disagree "
          f"(a {RATE_CUT:.2f} rate-cut would label these the other way)")
    for run, checkpoint, facts in disagreements:
        print(f"    {run:<32} {checkpoint:<6} {facts['claim_episodes']:>3}/{facts['episodes']:<4} claims "
              f"=> {facts['label']:<9} but rate {facts['rate']:.3f} => {rate_label(facts['rate'])}")
    print()


def held_label(held: tuple) -> str:
    """What `price_pairs` holds fixed: the scenario, the tree state, any other override."""
    scenario, overrides, chart = held
    extra = ("+" + ",".join(overrides)) if overrides else ""
    return f"{scenario}, chart {chart}{extra}"


def print_matched(result: dict, title: str, held: str, first: str, second: str,
                  dropped: list[str], empty: str, label_of=arm_label) -> None:
    """One matched table, both readings, and what each COULD have shown.

    The last line is the one that stops a table being over-read: an exact test
    conditions on a statistic the runs decided — Fisher on the margins, McNemar
    on the discordant count — so p alone never says whether a null was measured
    or merely permitted (refs assurance #59).
    """
    pairs = result["pairs"]
    print(f"{title}\n")
    if not pairs:
        print(f"    {empty}\n")
        return
    print(f"    {held:<34} {'seed':>4}  {first:<14} {second:<14} trees")
    for pair in pairs:
        trees = "one tree" if pair["first"]["tree"] == pair["second"]["tree"] else "TWO TREES"
        runs = " · ".join("/".join(side) for side in pair["runs"])
        print(f"    {label_of(pair['arm']):<34} {pair['seed']:>4}  "
              f"{pair['first']['label']:<14} {pair['second']['label']:<14} {trees:<9}  {runs}")
    print(f"\n    {first} {result['reporting_first']}/{len(pairs)} reporting, "
          f"{second} {result['reporting_second']}/{len(pairs)}; "
          f"agreement {result['agree']} of {len(pairs)}, "
          f"{result['expected_agree']:.2f} expected under independence")
    print(f"    discordant {result['one_way']} ({first} reports, {second} mute) "
          f"vs {result['other_way']} (the other way)")
    realized = result["realized"]
    for name, label, conditioned in (
        ("paired", "exact McNemar (paired)", f"{realized['discordant']} discordant pairs"),
        ("unpaired", "Fisher (unpaired)",
         f"margin {realized['first'] + realized['second']}/{2 * len(pairs)} reporting"),
    ):
        record = realized[name]
        verdict = ("rejects" if record["p"] < realized["alpha"] else
                   f"a null WITH power — {conditioned} could have reached {record['ceiling']:.4f}"
                   if record["capable"] else
                   f"NOT A NULL — {conditioned} could not go below {record['ceiling']:.4f}")
        print(f"    {label:<24} p = {record['p']:.4f}   {verdict}")
    if result["ambiguous"]:
        print(f"    {len(result['ambiguous'])} cells dropped as ambiguous (more than one run at the same arm and seed)")
    if dropped:
        print(f"    {len(dropped)} runs dropped, checkpoints disagree: {', '.join(dropped)}")
    print()


def spam_series(rows: list[dict], checkpoint: str = "final") -> dict:
    """Does a policy that reports more also lie more? — as a measurement, not a sort.

    The series is every checkpoint that REPORTS and carries
    ``false_complete_rate_root``; a mute checkpoint has no rate, so the two
    quantities are undefined exactly where the other is informative and the
    correlation can only ever be read over the reporting mode.

    Rho is taken over DISTINCT points, because a replicated cell (the same policy
    trained twice at one seed and one arm) is one observation of the relation and
    counting it twice inflates n without adding evidence. The leave-one-out range
    is returned beside it and is not optional: this series' rho is +0.26 over
    nine points and its jackknife straddles zero, which is what "monotone" was
    read off before it was measured (refs assurance #59).
    """
    points, seen = [], set()
    for row in sorted(rows, key=lambda r: r["run"]):
        facts = row[checkpoint]
        if facts is None or facts["label"] != "REPORTING" or facts["false"] is None \
                or facts["rate"] is None:
            continue
        point = (facts["rate"], facts["false"])
        points.append({"run": row["run"], "rate": point[0], "false": point[1],
                       "replicate": point in seen})
        seen.add(point)
    distinct = sorted({(p["rate"], p["false"]) for p in points})
    xs = [x for x, _ in distinct]
    ys = [y for _, y in distinct]
    return {"points": points, "distinct": distinct,
            "rho": spearman_rho(xs, ys), "jackknife": jackknife_rho(xs, ys)}


def print_spam(rows: list[dict], checkpoint: str = "final") -> None:
    result = spam_series(rows, checkpoint)
    points = result["points"]
    print(f"is reporting monotone in lying? {checkpoint} checkpoints that REPORT, "
          "close rate vs false-DONE rate\n")
    if len(result["distinct"]) < 3:
        print(f"    {len(result['distinct'])} distinct points — no relation to read\n")
        return
    for point in sorted(points, key=lambda p: p["rate"]):
        mark = "  (replicate, not a second observation)" if point["replicate"] else ""
        print(f"    {point['run']:<34} report {point['rate']:.3f} -> false {point['false']:.3f}{mark}")
    low, high = result["jackknife"]
    print(f"\n    Spearman rho = {result['rho']:+.3f} over {len(result['distinct'])} distinct "
          f"points from {len(points)} checkpoints")
    verdict = ("carried by single points — the sign flips on leave-one-out, so this is not a relation"
               if low <= 0.0 <= high else "sign holds under leave-one-out")
    print(f"    leave-one-out range {low:+.3f} to {high:+.3f}: {verdict}\n")


def print_pairs(rows: list[dict], first: str, second: str) -> None:
    print_matched(
        cross_scenario_pairs(rows, first, second),
        f"does a reporting seed transfer? {first} vs {second}, matched on arm and seed",
        "arm", first, second,
        [r["run"] for r in rows if r["scenario"] in (first, second) and r["label"] == "SPLIT"],
        "no seed carries a definite label on both scenarios at one arm")


def print_arms(rows: list[dict], first: float, second: float) -> None:
    print_matched(
        price_pairs(rows, first, second),
        f"does root_done_bonus buy the channel? rdb {first:g} vs rdb {second:g}, "
        "matched on scenario, chart state and seed",
        "scenario, chart, other overrides", f"rdb {first:g}", f"rdb {second:g}",
        [r["run"] for r in rows if r["price"] in (first, second) and r["label"] == "SPLIT"],
        "no seed carries a definite label at both prices", held_label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=RUNS, help="runs directory to read")
    parser.add_argument("--scenario", action="append", default=[],
                        help="restrict to this scenario (repeatable)")
    parser.add_argument("--price", type=float, default=None,
                        help="restrict to arms trained at this root_done_bonus")
    parser.add_argument("--pairs", nargs=2, metavar=("FIRST", "SECOND"),
                        help="pair two scenarios by arm and seed, and test the transfer")
    parser.add_argument("--arms", nargs=2, type=float, metavar=("FIRST", "SECOND"),
                        help="pair two root_done_bonus values by seed, and test the price")
    parser.add_argument("--spam", action="store_true",
                        help="close rate against false-DONE rate over the reporting checkpoints")
    args = parser.parse_args()

    rows = collect(args.runs)
    if args.scenario:
        rows = [r for r in rows if r["scenario"] in args.scenario]
    if args.price is not None:
        rows = [r for r in rows if r["price"] == args.price]
    print_table(rows)
    print_rate_check(rows)
    if args.pairs:
        print_pairs(rows, *args.pairs)
    if args.arms:
        print_arms(rows, *args.arms)
    if args.spam:
        print_spam(rows)


if __name__ == "__main__":
    main()
