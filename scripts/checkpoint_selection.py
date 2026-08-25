#!/usr/bin/env python
"""Which iteration wrote ``ckpt_best``, what selecting it cost, and what a floor changes.

    scripts/checkpoint_selection.py <run>            # one run, with the floor sweep
    scripts/checkpoint_selection.py                  # every run that has a metrics.csv
    scripts/checkpoint_selection.py --floor 0.5      # fleet-wide, one floor

**The question this exists for** (refs assurance #57). `patrol_brique_v19_rdb3_seed13`
ships a `ckpt_best` that succeeds in **17** episodes of 100 while its FINAL policy
succeeds in **99** — and files 94 root MISSION COMPLETEs of which 92 are rejected,
against the final policy's silence. `786ce93` already recorded that this is
``best_save_gate`` working as designed rather than a selection bug: since v1.20 a
reporting window supersedes a mute best lexicographically, whatever the success
numbers say. What was missing is the size of that side effect across the corpus,
and whether the two changes #57 proposes — a success floor before the reporting
comparison, or reading *admitted* rather than *emitted* claims — would move
anything. This is a READER: it launches nothing, evaluates nothing, and writes
nothing. Both changes are `cohort/` decisions and stay the owner's.

**The mechanism, which is not the one #57 names.** #57 reads the pathology as
"reports is measured on claims emitted, not on claims admitted". The training
signal the gate actually reads is ``root_report_close_rolling`` =
``env.root_close_step is not None`` over the recent window, and
``root_close_step`` is set only by a *truthful* close (an accepted root-mission
COMPLETE, or a timely SITREP on a continuous-posture root). So the gate's input
is already truth-conditioned and reading admitted claims would change nothing
about it — the 92 rejected claims are visible on the net and were never what
selected the checkpoint.

What does the selecting is the **denominator**. ``recent_root_closed`` is
appended once per episode **that sent an ENDEX**, and `cohort_env` transmits
ENDEX only in the success branch — so the reporting rate is conditioned on
winning, and its sample shrinks exactly as success collapses. A policy that wins
17 times in 100 and closes its own report in 9 of those 17 reads 0.53, clears
``ROOT_REPORT_CLOSE_FLOOR``, and lexicographically outranks a policy that wins 99
times and never reports. Two further edges follow from the same line:

* the two deques are **misaligned**. ``best_save_gate``'s D4 turnover check is
  ``episodes_seen >= window`` on ``recent_outcomes`` only; ``recent_root_closed``
  has no turnover requirement at all, so the first eligible save can compare a
  100-episode success window against a reporting window holding **one** win.
* the flag is **absorbing**. Once ``best_was_reporting`` is set no mute window may
  take the best back, so a tiny-denominator reporting window early in training
  locks the checkpoint against every later policy that stops claiming.

That is why the floor sweep below is the useful read-out: a floor is the one
proposal in #57 that touches the mechanism, because it refuses the comparison
where the denominator is thinnest.

**How the replay works, and how it is checked.** ``metrics.csv`` records every
input ``best_save_gate`` takes — ``success_rate_rolling``,
``root_report_close_rolling`` (blank when unmeasured, which the gate treats as
not-reporting) and ``n_episodes``, whose running sum is the trainer's own
``episodes_seen``. The replay calls the SHIPPED gate function, never a copy of
it. It is then **verified against the artifact**: ``ckpt_best.pt`` stores the
``iteration`` it was written at, so every run prints ``replay agrees`` or
``REPLAY DISAGREES`` rather than asking to be believed.

Two honest limits. ``metrics.csv`` rounds to five decimals, so a save whose
margin over the incumbent was under 1e-5 can land one iteration either side of
the real one; and the *denominator* of ``root_report_close_rolling`` is never
logged, so the thin-sample edge above is visible as a mechanism and is not
recoverable per iteration after the fact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT))

from cohort.metrics import (  # noqa: E402
    COMM_MODEL_GATE_WAIVERS,
    ROOT_REPORT_CLOSE_FLOOR,
)
from cohort.training.train import best_save_gate, is_reporting  # noqa: E402
from scripts.fleet_status import run_dirs  # noqa: E402
from scripts.run_report import checkpoint_stamp, fnum, run_dir  # noqa: E402

#: Success floors swept per run, applied ON TOP of the shipped rule. 0.0 is the
#: shipped rule itself and is always first so every other row reads as a delta
#: against it.
#:
#: **This changed meaning in v1.21.** When this reader was written the shipped
#: gate had no success condition at all, so 0.0 meant "no floor" and the sweep
#: asked what adding one would do. The answer — one run of 104 — is now IN
#: ``is_reporting`` at ``SUCCESS_RATE_FLOOR``, so 0.0 already carries a floor of
#: 0.5 and the rows at 0.25 and below can no longer move anything by
#: construction. What the sweep still asks is whether a floor STRICTER than the
#: shipped one would buy anything further.
FLOORS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9)

#: The trainer's rolling-outcome window (``Trainer.recent_outcomes`` maxlen).
#: Only used to reproduce the D4 turnover check; a run trained with a different
#: window would replay wrong, and none has been.
WINDOW = 100

#: Columns without which the gate cannot be replayed at all. ``n_episodes`` is
#: the one that dates a corpus: runs before it was logged cannot have
#: ``episodes_seen`` reconstructed, so the D4 turnover check never passes and the
#: replay would report "never saved" for a run that has a perfectly good
#: ``ckpt_best.pt`` on disk. That reads as a finding and is a reader limitation,
#: so those runs are reported as NOT REPLAYABLE and excluded from every count.
REQUIRED = ("n_episodes", "success_rate_rolling", "iteration", "env_steps")

#: The reporting axis itself. Absent on every pre-v1.20 corpus, where the gate
#: really was success-only — so the replay stays exact there (a missing rate is
#: the ``None`` the gate already treats as not-reporting) and a floor can move
#: nothing, which is a property of the run's era and not a result.
REPORTING_COLUMN = "root_report_close_rolling"


def rows_of(run: Path) -> list[dict]:
    """A run's ``metrics.csv``, or an empty list when it has none.

    Read here rather than through ``run_report.rows_of`` because a missing
    metrics file is normal for this reader (a run that died before its first
    iteration, an archived corpus kept for its evaluations) and must not exit.
    """
    import csv

    path = run / "metrics.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def episodes_and_windows(rows: list[dict]) -> list[tuple[int, float, float | None]]:
    """Per iteration: ``episodes_seen``, rolling success, rolling root-close.

    ``episodes_seen`` is the running sum of ``n_episodes``, which is exactly
    what ``Trainer`` counts — the trainer increments it once per finished
    episode and logs the same count per iteration.
    """
    out, seen = [], 0
    for row in rows:
        seen += int(fnum(row, "n_episodes") or 0)
        out.append((seen, fnum(row, "success_rate_rolling") or 0.0, fnum(row, "root_report_close_rolling")))
    return out


def replay(rows: list[dict], floor: float = 0.0, window: int = WINDOW,
           report_gate_waived: bool = False) -> list[int]:
    """Indices of the rows that would have written ``ckpt_best``, under one floor.

    ``report_gate_waived`` must match the run's scenario (v1.23): a jammed run
    is SELECTED without the reporting key, so replaying it with the key would
    report a selection the trainer never made. Resolved by ``run_facts`` from
    the run's own recorded comm model, never defaulted per-run — the default
    here is for the historical sweeps, whose runs all predate the waiver.

    ``floor`` is applied in the one place that makes it a floor rather than a
    veto: a window whose rolling success is below it may not claim the REPORTING
    side of the lexicographic order, so it is compared on success like any mute
    window. At ``floor=0.0`` this is byte-for-byte the shipped rule, which is
    what makes every other row of the sweep readable as a delta.

    Neither half of the comparison is re-implemented: ``best_save_gate`` decides
    the save and ``is_reporting`` decides the absorbing flag, exactly as
    ``Trainer`` wires them. That mattered more than it looked — while the flag
    update WAS a local copy here, v1.21's fix to ``is_reporting`` reached the
    gate but not the flag, and the reader reported the pre-fix selection under
    the post-fix rule.
    """
    saves: list[int] = []
    best_success, best_reporting = -1.0, False
    for i, (seen, success, close) in enumerate(episodes_and_windows(rows)):
        rate = None if (floor > 0.0 and success < floor) else close
        if best_save_gate(seen, window, success, best_success, rate, best_reporting,
                          report_gate_waived=report_gate_waived):
            best_success = success
            best_reporting = is_reporting(rate, success)
            saves.append(i)
    return saves


def replay_pre_v121(rows: list[dict], window: int = WINDOW) -> list[int]:
    """The same replay under the gate as it stood BEFORE the v1.21 #57 fix.

    A frozen historical copy, and the only rule in this reader that is written
    out rather than imported — it no longer exists in ``cohort/`` to import.
    It earns its place by making the artifact check honest: every ``ckpt_best``
    on disk today was written by THIS rule, so a run whose stamped iteration
    disagrees with the shipped replay but agrees with this one is not a broken
    replay, it is a pre-fix artifact, and the printer says so.
    """
    saves: list[int] = []
    best_success, best_reporting = -1.0, False
    for i, (seen, success, close) in enumerate(episodes_and_windows(rows)):
        if seen < window:
            continue
        reporting = close is not None and close >= ROOT_REPORT_CLOSE_FLOOR
        if reporting != best_reporting:
            if not reporting:
                continue
        elif not success > best_success:
            continue
        best_success, best_reporting = success, reporting
        saves.append(i)
    return saves


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def evaluation(run: Path, name: str) -> dict | None:
    """One committed evaluation, reduced to the columns #57 argues from.

    ``admitted`` is emitted minus rejected, which ``cohort.metrics`` guarantees
    is exact: every DONE is adjudicated on the step it is transmitted, so there
    is no third outcome. It is a count of the root's *accepted claims*, and NOT
    the same quantity as ``closed_on_root_report_rate`` — a root can have a
    sub-mission COMPLETE accepted without closing the operation, which is why
    `v19`'s `ckpt_best` reads 2 admitted claims and a close rate of 0.000.
    """
    data = _json(run / name)
    metrics = data.get("metrics") or {}
    if not data.get("episodes"):
        return None
    emitted = metrics.get("done_reports_root")
    rejected = metrics.get("done_rejected_root")
    return {
        "episodes": data["episodes"],
        "success": metrics.get("success_rate"),
        "emitted": emitted,
        "admitted": None if emitted is None or rejected is None else emitted - rejected,
        "claim_episodes": metrics.get("done_claim_episodes_root"),
        "close_rate": metrics.get("closed_on_root_report_rate"),
    }


def missing_columns(rows: list[dict]) -> list[str]:
    """Which of ``REQUIRED`` this corpus does not carry."""
    return [] if not rows else [c for c in REQUIRED if c not in rows[0]]


def _report_gate_waived_for(run: Path) -> bool:
    """Was this run SELECTED without the reporting key? (v1.23)

    Read from the run's own recorded `comm_model` rather than from the current
    scenario table: a run is replayed as it was trained, and a scenario whose
    comm model changed later must not have its history re-selected under the
    new rule.
    """
    try:
        econ = json.loads((run / "economics.json").read_text())
    except (OSError, ValueError):
        return False
    comm = (econ.get("spec") or {}).get("comm_model") or econ.get("comm_model")
    return "closed_on_root_report_rate" in COMM_MODEL_GATE_WAIVERS.get(comm, {})


def run_facts(run: Path) -> dict | None:
    """One run's selection story, or None if it has no iterations to replay."""
    rows = rows_of(run)
    if not rows:
        return None
    if gaps := missing_columns(rows):
        return {"run": run.name, "rows": len(rows), "replayable": False, "missing": gaps}
    windows = episodes_and_windows(rows)
    waived = _report_gate_waived_for(run)
    sweep = {}
    for floor in FLOORS:
        saves = replay(rows, floor, report_gate_waived=waived)
        sweep[floor] = None if not saves else {
            "row": saves[-1],
            "iteration": int(fnum(rows[saves[-1]], "iteration") or 0),
            "env_steps": int(fnum(rows[saves[-1]], "env_steps") or 0),
            "success": windows[saves[-1]][1],
            "close": windows[saves[-1]][2],
            "false_complete": fnum(rows[saves[-1]], "false_complete_rate"),
            "saves": len(saves),
        }
    shipped = sweep[0.0]
    legacy_saves = replay_pre_v121(rows)
    legacy_iter = (
        None if not legacy_saves else int(fnum(rows[legacy_saves[-1]], "iteration") or 0)
    )
    stamp = checkpoint_stamp(run / "ckpt_best.pt")
    stamped = None if stamp is None else stamp["iteration"]
    return {
        "legacy_iteration": legacy_iter,
        "agrees_pre_v121": None if stamped is None or legacy_iter is None else stamped == legacy_iter,
        "run": run.name,
        "rows": len(rows),
        "replayable": True,
        "missing": [],
        "has_reporting": REPORTING_COLUMN in rows[0],
        "episodes": windows[-1][0],
        "last": {"success": windows[-1][1], "close": windows[-1][2],
                 "env_steps": int(fnum(rows[-1], "env_steps") or 0)},
        "sweep": sweep,
        "stamped_iteration": stamped,
        "agrees": None if stamped is None or shipped is None else stamped == shipped["iteration"],
        "best": evaluation(run, "behavior.json"),
        "final": evaluation(run, "behavior_final.json"),
    }


def success_recovered(facts: dict) -> float | None:
    """Best rolling success any floor reaches, minus what the shipped rule took.

    The single number that says whether a floor would have changed this run:
    0.0 means every floor selects the same window the shipped rule did.
    """
    if not facts["replayable"]:
        return None
    shipped = facts["sweep"][0.0]
    if shipped is None:
        return None
    others = [s["success"] for f, s in facts["sweep"].items() if f > 0.0 and s]
    return None if not others else max(others) - shipped["success"]


# ---------------------------------------------------------------- printing --


def _pct(value: float | None) -> str:
    return "  —  " if value is None else f"{value:.3f}"


def print_run(facts: dict) -> None:
    if not facts["replayable"]:
        print(f"{facts['run']}  —  {facts['rows']} iterations, NOT REPLAYABLE "
              f"(metrics.csv predates {', '.join(facts['missing'])})\n")
        return
    shipped = facts["sweep"][0.0]
    verify = ("unverified" if facts["agrees"] is None else
              "replay agrees" if facts["agrees"] else
              f"pre-#57 artifact (ckpt_best.pt says iter {facts['stamped_iteration']}, "
              f"which is what the pre-v1.21 gate selected)" if facts["agrees_pre_v121"] else
              f"REPLAY DISAGREES (ckpt_best.pt says iter {facts['stamped_iteration']})")
    print(f"{facts['run']}  —  {facts['rows']} iterations, {facts['episodes']:,} episodes, window {WINDOW}")
    if shipped is None:
        print("    no iteration ever passed the gate (the run never filled its outcome window)\n")
        return
    print(f"    ckpt_best written at iteration {shipped['iteration']} "
          f"({shipped['env_steps']:,} steps, {shipped['saves']} saves) — {verify}")
    print(f"    that window     success {_pct(shipped['success'])}  closed-on-root {_pct(shipped['close'])}"
          f"  false-COMPLETE {_pct(shipped['false_complete'])}")
    print(f"    last window     success {_pct(facts['last']['success'])}  "
          f"closed-on-root {_pct(facts['last']['close'])}  ({facts['last']['env_steps']:,} steps)")
    print(f"\n    success floor applied BEFORE the reporting comparison (floor {ROOT_REPORT_CLOSE_FLOOR:g} on the rate):")
    for floor, save in facts["sweep"].items():
        tag = " (shipped)" if floor == 0.0 else ""
        if save is None:
            print(f"      {floor:.2f}{tag:<10}  no save")
            continue
        print(f"      {floor:.2f}{tag:<10}  iter {save['iteration']:>5}  {save['env_steps']:>10,} steps  "
              f"success {_pct(save['success'])}  closed-on-root {_pct(save['close'])}")
    recovered = success_recovered(facts)
    if recovered:
        print(f"\n    a floor recovers up to {recovered:+.3f} rolling success against the shipped rule")
    print("\n    committed evaluations (root MISSION COMPLETEs; admitted = emitted - rejected):")
    for name, label in (("best", "ckpt_best "), ("final", "ckpt_final")):
        cell = facts[name]
        if cell is None:
            print(f"      {label}  not evaluated")
            continue
        print(f"      {label}  N={cell['episodes']:<4} success {_pct(cell['success'])}  "
              f"claims {cell['emitted']!s:>4} emitted / {cell['admitted']!s:>4} admitted / "
              f"{cell['claim_episodes']!s:>4} claim-eps   closed-on-root {_pct(cell['close_rate'])}")
    print()


def print_fleet(rows: list[dict], floor: float) -> None:
    """One line per replayable run: does a floor move its ckpt_best, and by how much."""
    replayable = [f for f in rows if f["replayable"]]
    skipped = len(rows) - len(replayable)
    pre_v120 = sum(1 for f in replayable if not f["has_reporting"])
    print(f"{len(replayable)} runs replayed, floor {floor:g} against the shipped rule "
          f"(reporting floor {ROOT_REPORT_CLOSE_FLOOR:g})\n")
    print(f"    {'run':<38} {'shipped':>18}  {'with floor':>18}  {'delta':>7}  verify")
    movers = disagree = verified = pre_fix = 0
    for facts in replayable:
        shipped, floored = facts["sweep"][0.0], facts["sweep"].get(floor)
        era = "" if facts["has_reporting"] else "  pre-v1.20"
        if shipped is None:
            print(f"    {facts['run']:<38} {'never saved':>18}{era}")
            continue
        if floored is None:
            cell, delta = "never saved", None
        else:
            cell = f"{floored['success']:.3f} @ {floored['iteration']:>5}"
            delta = floored["success"] - shipped["success"]
        moved = delta is not None and abs(delta) > 1e-9
        movers += moved
        stale = facts["agrees"] is False and facts["agrees_pre_v121"] is True
        pre_fix += stale
        disagree += facts["agrees"] is False and not stale
        verified += facts["agrees"] is True
        verify = ("?" if facts["agrees"] is None else "ok" if facts["agrees"]
                  else "pre-#57" if stale else "DISAGREES")
        mark = " <-" if moved else ""
        print(f"    {facts['run']:<38} {shipped['success']:.3f} @ {shipped['iteration']:>5}  "
              f"{cell:>18}  {'' if delta is None else f'{delta:+.3f}':>7}  {verify}{mark}{era}")
    print(f"\n    {movers} of {len(replayable)} runs select a different checkpoint under this floor")
    print(f"    {pre_v120} predate the reporting axis (no {REPORTING_COLUMN}), where no floor can move anything")
    print(f"    {skipped} runs not replayable (metrics.csv predates {REQUIRED[0]}), excluded from both counts")
    print(f"    {verified} of {len(replayable)} replays verified against the iteration in ckpt_best.pt")
    if pre_fix:
        print(f"    {pre_fix} carry a pre-#57 ckpt_best — the artifact on disk is the one the "
              f"pre-v1.21 gate selected, and that replay verifies")
    if disagree:
        print(f"    ⚠ {disagree} replays DISAGREE with the iteration stamped in ckpt_best.pt")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", help="one run to report in full")
    parser.add_argument("--runs", type=Path, default=RUNS, help="runs directory to read")
    parser.add_argument("--floor", type=float, default=0.5,
                        help="success floor used for the fleet table (default 0.5)")
    args = parser.parse_args()

    if args.run:
        facts = run_facts(run_dir(args.run))
        if facts is None:
            raise SystemExit(f"no metrics.csv to replay for run '{args.run}'")
        print_run(facts)
        return
    rows = [f for run in run_dirs(args.runs) if (f := run_facts(run))]
    rows.sort(key=lambda f: -(success_recovered(f) or 0.0))
    print_fleet(rows, args.floor)


if __name__ == "__main__":
    main()
