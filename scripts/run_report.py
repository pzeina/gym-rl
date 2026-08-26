#!/usr/bin/env python
"""Compact post-hoc digest of a finished run — the ONLY thing the big model reads.

A run's metrics.csv is ~3000 rows x 20 columns. Feeding that to Opus/Fable at
150k context is what makes a training campaign expensive. This collapses it to
~30 lines: config, learning curve by decile, reward-component drift, the
behavioral suite, and (optionally) a comparison against a baseline run — the
success / root-survival / clock triple at each side's N (refs #34) plus root
deaths within successes (refs #47), the full delta, and whether the two runs
are actually a single-variable A/B (refs #20).

    scripts/run_report.py <run>
    scripts/run_report.py <run> --vs <baseline-run>
    scripts/run_report.py <run> --components      # per-component reward decile table
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

sys.path.insert(0, str(ROOT))
from cohort.metrics import (  # noqa: E402
    format_obedience_by_task,
    format_order_availability,
    format_order_task_mix,
    format_staging,
    gate_mark,
)


def run_dir(name: str) -> Path:
    """Where a run lives — ``runs/<name>``, or ``runs/archive/<name>``.

    Archiving a superseded generation must not break the citations that made it
    worth keeping. Every reader here goes through this, so a run named in
    ROADMAP three cycles ago still reports after it has been filed away.
    """
    current = RUNS / name
    if not current.is_dir() and (RUNS / "archive" / name).is_dir():
        return RUNS / "archive" / name
    return current


def rows_of(run: str) -> list[dict]:
    path = run_dir(run) / "metrics.csv"
    if not path.exists():
        raise SystemExit(f"no metrics for run '{run}' ({path})")
    with path.open() as f:
        return list(csv.DictReader(f))


#: How far below the run's final rolling success the window that WROTE
#: ``ckpt_best`` may sit before the digest flags it. Not a gate and not a
#: threshold on anything published — a digest line, because until #57 nothing
#: printed where a run's "best work" came from and a checkpoint written at 0.9%
#: of a 3M-step run went unremarked for a cycle.
BEST_SELECTION_GAP = 0.10


def checkpoint_stamp(path: Path) -> dict | None:
    """The iteration and env_steps a checkpoint was written at, or None.

    ``train.py::save_checkpoint`` stores both, so a digest can say WHERE in a
    run its ``ckpt_best`` came from — the one thing ``best_save_gate`` decides
    and nothing ever reported. Every failure mode gives the same answer: a live
    run may be halfway through writing the file, an archived run may have had
    its weights pruned, a pre-v1.x checkpoint may not carry the fields. None
    means "cannot say", never "iteration zero".
    """
    if not path.exists():
        return None
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        return {"iteration": int(ckpt["iteration"]), "env_steps": int(ckpt["env_steps"])}
    except Exception:  # unreadable is a report, not a crash
        return None


def fnum(row: dict, key: str) -> float | None:
    """Parse a metric cell; NaN (iterations with no completed episode) reads as missing."""
    try:
        v = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return v if v == v else None


def mean(rows: list[dict], key: str) -> float:
    vals = [v for r in rows if (v := fnum(r, key)) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def deciles(rows: list[dict], n: int = 10) -> list[list[dict]]:
    size = max(1, len(rows) // n)
    return [rows[i * size: (i + 1) * size] for i in range(n) if rows[i * size:]]


#: best-minus-final rolling success, in points, past which a run is not a
#: converged result. Set at 15 because squad_v5 — the last pre-v1.10 run that
#: trained cleanly — gave back 5, while every post-v1.10 run measured so far
#: gave back 10 (defend_v8), 17 (defend_v9), 33 (squad_v6) and 68 (fireteam_v7).
COLLAPSE_POINTS = 15

#: The PUBLISHING bar, tighter than the collapse warning above: a run may not
#: contribute a headline number to README/ROADMAP unless it gave back less than
#: this. A warning that a digest prints is advice; this is a gate, because the
#: advice was already there and three collapsed runs got published anyway
#: (squad_recon_v5 and v6 both ended at 0.00 rolling and are published at 0.77
#: and 0.91; squad_v7 ended at 0.41 and is published at 0.92).
PUBLISH_STABILITY_POINTS = 10


#: (metric key, label, format) rows of a behavior block, in display order.
#:
#: ``human_death_rate`` — the rate at which the human root (the commander the
#: cohort exists to keep alive) dies — is here because of assurance #22, whose
#: pre-registration names *root deaths at the final policy* as the PRIMARY axis
#: of the v1.12 survivor-scaled-terminal A/B. The metric was measured on every
#: behavior run and printed by `evaluate`'s own table, but this digest — "the
#: ONLY thing the big model reads" — dropped it, so the pre-registered primary
#: could not be read from the artifact the verdict is written against.
#:
#: ``closed_on_root_report_rate`` is here for the same reason and was found the
#: same way (refs #48). It is the axis the v1.20 `root_done_bonus` default was
#: chosen on and the axis `metrics.regression_gates` now refuses runs on (floor
#: 0.5) — and this digest printed report precision, recall and false-DONE while
#: never printing it. The cost of the omission is on the record: a handoff note
#: scoped near-mute `ckpt_best` as a property of the challenger price when the
#: SHIPPED one does it too (`squad_v10b`, 0.000 at `ckpt_best` against 0.784 at
#: FINAL, 0 root claims in 100 episodes against 307), which is exactly the
#: comparison a row printed under both blocks makes at a glance. Note it is NOT
#: the root-claim count: on a continuous-posture root the window closes on a
#: SITREP and MISSION COMPLETE is masked shut, so the whole defend family reads
#: ~1.00 here on zero claims.
_BEHAVIOR_ROWS: tuple[tuple[str, str, str], ...] = (
    ("obedience_latency_mean", "obedience latency", "{:.2f}"),
    ("report_precision", "report precision", "{:.2f}"),
    ("report_recall", "report recall", "{:.2f}"),
    ("closed_on_root_report_rate", "closed on root report", "{:.3f}"),
    ("doctrine_preference_rate", "doctrine preference", "{:.3f}"),
    ("doctrine_allowed_rate", "doctrine containment", "{:.3f}"),
    ("orders_per_episode", "orders / episode", "{:.2f}"),
    ("retasks_per_episode", "retasks / episode", "{:.2f}"),
    ("false_complete_rate", "false DONE", "{:.3f}"),
    ("human_death_rate", "root death rate", "{:.3f}"),
    ("cover_occupancy_under_threat", "cover under threat", "{:.3f}"),
    ("mean_distance_from_objective_under_threat", "dist from OBJ", "{:.2f}"),
    # refs #18: the clock and what the net carried while it ran out
    ("timeout_rate", "ran the clock out", "{:.2f}"),
    ("messages_per_episode", "messages / episode", "{:.0f}"),
    ("command_traffic_share", "of which command", "{:.3f}"),
)


#: The three cells every A/B comparison prints, in display order (refs #34).
#:
#: They travel together because each one alone can be read to flatter a policy
#: the other two would convict. Success says nothing about what the success
#: cost — a cohort can win every episode over its commander's body. Root
#: survival on its own is gameable in the opposite direction, since a policy
#: that never closes with the enemy buries nobody and achieves nothing, which
#: is exactly what ``timeout_rate`` catches: it separates "kept everyone alive
#: by holding the ground" from "kept everyone alive by riding the clock out".
#:
#: The prompting case is the `squad_v8` → `squad_v9` A/B (`done_false` -0.5 ->
#: -2.0), first published with success and DONE volume alone. Its survival cell
#: is a **null** — p = 1.00 pooled over 200 episodes an arm — and the null is
#: the result, because an earlier `done_false` change had once been *associated*
#: with root deaths moving 4/30 → 12/30 while success held. A reader cannot
#: infer a null from a column that is not there, so the pair is printed by the
#: instrument rather than by whoever remembers to.
#: The fourth cell closes the loophole the first three leave open between them
#: (refs #47): a policy can zero the RAW death rate by the exact conversion
#: ``timeout_rate`` flags — `squad_v12b` reads 0/100 root deaths while turning
#: every defeat into a timeout, and in that corpus defeat and root death are the
#: same event, so part of its 0 is declining the fight. Deaths counted over
#: successful episodes alone cannot be moved that way: those episodes achieved
#: the mission either way, and the cell drops only when commanders stop dying
#: inside wins (on that axis `squad_v12b` is 0/96 and 0/86 against its
#: control's 14/93 and 14/88, p < 1e-4 — real, and invisible in the raw rate
#: next to the timeout column that discounts it). Runs evaluated before
#: per-episode outcomes read as an em dash, never as a zero.
_COMPARISON_ROWS: tuple[tuple[str, str, str], ...] = (
    ("success", "success", "{:.3f}"),
    ("human_death_rate", "root death rate", "{:.3f}"),
    ("timeout_rate", "ran the clock out", "{:.3f}"),
    ("root_death_in_success", "root death in success", "{:.3f}"),
)

#: (summary prefix, heading) of the checkpoints a comparison covers. Both, for
#: the same reason ``report`` prints both: on `squad_screen_v4` the two evaluate
#: 30/30 and 0/30 on the same seeds, so an A/B stated at one checkpoint is an
#: A/B between two unstated policies.
_COMPARISON_CHECKPOINTS: tuple[tuple[str, str], ...] = (
    ("beh_", "ckpt_best"),
    ("final_", "FINAL policy"),
)

#: Summary keys that are sample sizes rather than measurements. They belong in
#: the comparison header, where an N is read as an N, and not in the delta dump,
#: where "100.000 → 20.000  (-80.000)" reads as a metric that moved.
_EPISODE_COUNT_KEYS: frozenset[str] = frozenset(
    f"{prefix}episodes" for prefix, _ in _COMPARISON_CHECKPOINTS
)


def stability(best: float, final: float) -> str:
    """One line separating "this run converged" from "this run peaked".

    Every published number comes from ``ckpt_best``, which captures the best
    ROLLING WINDOW, not the policy the run ended with. On a run that spikes and
    falls back, evaluating ckpt_best measures the peak and says nothing about
    convergence — squad_v6 read 0.95 at N=20 off a policy whose final decile
    was 65%, and fireteam_v7 evaluates at 0.95 off a final decile of 26%.
    Nothing in the digest used to say so, and the trap caught two consecutive
    sessions, so it is stated on its own line.
    """
    if best != best or final != final:  # NaN
        return "stability  —  (no rolling-success rows)"
    drop = (best - final) * 100
    if drop >= COLLAPSE_POINTS * 2:
        verdict = "COLLAPSED — ckpt_best is a peak, NOT a result; quote both numbers"
    elif drop >= COLLAPSE_POINTS:
        verdict = "UNSTABLE — ckpt_best overstates this run; quote both numbers"
    else:
        verdict = "converged"
    gate = "PUBLISHABLE" if drop < PUBLISH_STABILITY_POINTS else "NOT PUBLISHABLE"
    return (
        f"stability  best-final gap {drop:.0f} pts  [{verdict}]\n"
        f"  publish gate  [{gate}]  (bar: gap < {PUBLISH_STABILITY_POINTS} pts, "
        f"and the headline number is the FINAL policy)"
    )


def optimizer_line(first: list[dict], last: list[dict]) -> str | None:
    """The PPO diagnostics, or None on a run that predates them (v1.11).

    ``explained_variance`` is the one to read first: below ~0 the critic is
    worse than predicting the batch mean, so every advantage in the update is
    noise and the run is not learning from what it looks like it is learning
    from. ``grad_norm`` next — if it sits far above max_grad_norm the update is
    being scaled down wholesale, which is how the value head used to swallow
    95-99% of the policy's gradient budget.
    """
    if not any(k in (first[0] if first else {}) for k in ("explained_variance", "grad_norm")):
        return None
    out = []
    for key, label, fmt in [
        ("explained_variance", "explained var", "{:+.3f}"),
        ("grad_norm", "grad norm", "{:.3f}"),
        ("clipfrac", "clip frac", "{:.3f}"),
        ("value_std", "value scale", "{:.2f}"),
        ("return_std", "return scale", "{:.2f}"),
        ("epochs_used", "epochs used", "{:.2f}"),
    ]:
        a, b = mean(first, key), mean(last, key)
        if b != b:
            continue
        out.append(f"{label} {fmt.format(b)} ({b - a:+.3f})")
    return "  optimizer  " + "  ".join(out) if out else None


def curve(rows: list[dict]) -> str:
    """Ten-bucket sparkline of rolling success, plus the numbers that matter."""
    blocks = "▁▂▃▄▅▆▇█"
    vals = [mean(seg, "success_rate_rolling") for seg in deciles(rows)]
    clean = [v for v in vals if v == v]
    lo, hi = (min(clean), max(clean)) if clean else (0.0, 1.0)
    span = (hi - lo) or 1.0
    spark = "".join(blocks[min(7, int((v - lo) / span * 7))] if v == v else " " for v in vals)
    return f"{spark}  ({lo:.0%}→{hi:.0%})"


class ClaimOrdinalError(Exception):
    """The artifact's per-episode root claims cannot be split by ordinal."""


def root_claim_ordinal(per_episode: list[dict]) -> dict | None:
    """Split the root's MISSION COMPLETE claims into the episode's FIRST and the rest.

    **Why a pooled precision is the wrong instrument here** (refs #46). The
    lever now under test on ``squad`` — ``root_done_bonus_first_claim_only`` —
    is a rule about the first claim versus later ones, and its expected value
    was pre-registered from ``squad_v10``'s POOLED claim precision, 77/178 =
    0.433. That number describes neither ordinal: on the same corpus the first
    claim is accepted at 0.543 and later ones at 0.314. On the same run's
    ``ckpt_best`` the split INVERTS (0.474 / 0.547), so a precision computed on
    one checkpoint does not describe the other either. **The missing quantity
    is the one the existing metric cannot represent** — the same shape as
    ``done_reports`` without ``done_admissible`` and order share without
    availability. So the digest carries the split, not just the pool.

    **The derivation is exact, not a heuristic**, and it rests on one pinned
    invariant: a confirmed root claim is the LAST root claim of its episode
    (``tests/test_confirmed_claim_is_last.py``; ``cohort_env`` closes the
    operation in the same ``step`` that confirms it). So per episode there is at
    most one acceptance, and ``done_reports_root - done_rejected_root`` says
    which ordinal it fell on: 1 acceptance with 1 claim means the FIRST claim
    was accepted; 1 acceptance with n > 1 claims means a LATER one was, and the
    first was rejected.

    ``first_rejected`` / ``closed_after_rejected_first`` are the two halves of
    what ``rewards.py`` calls burning the bonus: under the first-claim rule a
    rejected opening probe forfeits ``root_done_bonus`` for the whole episode,
    so its real price is ``done_false - root_done_bonus x P(the episode later
    closes by a root claim)``. That P measured **1.000** on
    ``defend_brique_v11`` and is what reverted the rule at v1.16 — a run's own
    corpus can say in advance whether the rule would price its honest first
    report into silence.

    **Succession episodes are excluded, because that is exactly as far as the
    proxy is exact** — the same scope, and for the same reason, as
    ``test_confirmed_claim_is_last.py``. The invariant is about the root's
    *OPORD* claim: ``cohort_env._report_done`` closes the operation only when
    ``is_root_opord_claim`` holds, while ``metrics._done_traffic`` counts
    ``done_reports_root`` as *any* DONE whose sender held the root at that step.
    The two agree while the root is one soldier for the whole episode and
    diverge the moment a successor is promoted: the promoted commander still
    carries its personal SEIZE/ADVANCE mission, may truthfully complete **that**,
    and the completion is confirmed and counted here while the operation
    correctly runs on. So a succession episode can carry two confirmed root
    claims without anything being wrong — ``fireteam_v10`` ep19 is one (4 claims,
    2 rejected, 2 successions, and ``endex_on_root_report`` still 1, the
    operation closing exactly once).

    That is a limit of the recorded quantity, not of the invariant, so the
    exclusion is narrow and **counted, never silent**: ``excluded`` rides in the
    result and the digest prints it. An ordinal split quoted over 18 of 20
    episodes while captioned as 20 is the same overstatement as an N=20 row
    captioned N=100. The honest fix upstream is a root-*mission* claim counter;
    until one exists, this keeps the derivation exact where it is sound instead
    of loosening the bound everywhere.

    Returns ``None`` for an artifact that predates the root-split fields (the
    2026-08-07 era has ``done_reports`` only), because an absent measurement is
    not a zero. Still raises for a NON-succession episode, on ``done_probe.py``'s
    rule: the impossible number is the one that gets quoted.
    """
    first = first_accepted = later = later_accepted = excluded = 0
    measured = False
    for i, ep in enumerate(per_episode):
        claims, rejected = ep.get("done_reports_root"), ep.get("done_rejected_root")
        if claims is None or rejected is None:
            return None
        measured = True
        # A missing field is not a "no succession" — but the 2026-08-07 era that
        # lacks it is already returned above, so absent here means a corpus that
        # records successions and had none. Strict is the right default anyway.
        if ep.get("succession_events"):
            excluded += 1
            continue
        accepted = claims - rejected
        if accepted not in (0, 1):
            raise ClaimOrdinalError(
                f"episode {i}: {claims} root claims, {rejected} rejected, no succession — "
                "every DONE is answered exactly once and, with the root held by one soldier "
                "for the whole episode, at most one root claim is confirmed, so this artifact "
                "is a broken measurement rather than a strange policy"
            )
        if not claims:
            continue
        first += 1
        later += claims - 1
        if accepted and claims == 1:
            first_accepted += 1
        elif accepted:
            later_accepted += 1
    if not measured:
        return None
    return {
        "claims": first + later,
        "first": first,
        "first_accepted": first_accepted,
        "later": later,
        "later_accepted": later_accepted,
        # the two halves of the burn: opening probes that were rejected, and how
        # many of those episodes went on to close by a root claim anyway
        "first_rejected": first - first_accepted,
        "closed_after_rejected_first": later_accepted,
        # succession episodes the root-sender proxy cannot split; printed, so the
        # split is never read as covering episodes it was not derived from
        "excluded": excluded,
    }


def root_death_in_success(per_episode: list[dict]) -> tuple[int, int] | None:
    """``(root deaths, episodes)`` counted over successful episodes only (refs #47).

    **Why the raw rate is not enough.** ``human_death_rate`` divides by every
    episode, so it improves when a policy converts defeats — where the commander
    dies — into timeouts, where nobody does. That is not hypothetical:
    `squad_v12b` is 0/100 root deaths at both checkpoints, and part of that zero
    is taking zero defeats and riding the clock instead (its control's defeats
    are root deaths 5/5 and 10/10). A success achieved the mission either way,
    so a death rate conditioned on success cannot be bought by declining the
    engagement — and on `squad_v12b` it still separates the arms (0/96 and 0/86
    against 14/93 and 14/88, p < 1e-4): the reduction inside wins is real, it
    was measured on every run, and the rejection that ignored it is why this is
    now printed by the instrument rather than by whoever remembers to look.

    Derivable from what every behavior corpus already records — ``outcome`` and
    ``human_died`` per episode — so no ``cohort/`` change and no re-evaluation.
    Returns ``None`` for an artifact predating those fields, because an absent
    measurement is not a zero.
    """
    deaths = successes = 0
    measured = False
    for ep in per_episode:
        outcome, died = ep.get("outcome"), ep.get("human_died")
        if outcome is None or died is None:
            return None
        measured = True
        if outcome == "success":
            successes += 1
            deaths += 1 if died else 0
    if not measured:
        return None
    return deaths, successes


def format_claim_ordinal(o: dict) -> tuple[str, str | None]:
    """The ordinal split as digest lines: the claim line, and the burn line or None."""
    parts = [str(o["claims"])]
    for label, num, den in (
        ("first", o["first_accepted"], o["first"]),
        ("later", o["later_accepted"], o["later"]),
    ):
        if den:
            parts.append(f"{label} {num}/{den} = {num / den:.3f}")
    if excluded := o.get("excluded", 0):
        # never a bare split: the reader must see it is over a subset, and why
        parts.append(f"({excluded} succession ep{'s' if excluded > 1 else ''} not splittable)")
    burn_den, burn_num = o["first_rejected"], o["closed_after_rejected_first"]
    burn = (
        f"{burn_num}/{burn_den} = {burn_num / burn_den:.3f}   "
        "(rejected first claims whose episode still closed)"
    ) if burn_den else None
    return "   ".join(parts), burn


def behavior_block(path: Path, header: str, summary: dict, prefix: str, *, diagnostics: bool) -> dict:
    """Print one evaluated checkpoint's suite, and file it under ``prefix``.

    Called twice per run — once for ``ckpt_best`` and once for the FINAL
    policy — because a single behavior block is not a run's behavior. On
    ``squad_screen_v4`` the two checkpoints evaluate 30/30 and 0/30 on the same
    seeds, and on ``defend_brique_v4`` the whole positional regression lives in
    the gap: ckpt_best passes all three gates, ckpt_latest FAILS
    ``mean_distance_from_objective_under_threat`` at 6.09. Printing the suite
    for ckpt_best and a single success number for the final policy meant the
    digest showed three PASSes and hid the FAIL on the checkpoint the
    publishing standard calls the headline (refs #22).

    ``diagnostics`` adds the four command-quality composites (task mix,
    availability lift, per-task obedience, staging). They stay on the ckpt_best
    block alone: they diagnose how the cohort commands rather than whether the
    run cleared its bars, and the digest's whole purpose is to stay short.
    """
    b = json.loads(path.read_text())
    m = b.get("metrics", {})
    # WHICH checkpoint produced these numbers, always: on squad_screen_v4
    # ckpt_best evaluates 30/30 and ckpt_latest 0/30 on the same seeds, so
    # a behavior block that does not name its checkpoint is unreadable
    # next to a curve that ended at 0% (refs #18)
    scored = Path(b.get("checkpoint") or "?").name or "?"
    print(f"  {header} ({b.get('episodes','?')} eps, greedy={b.get('greedy')}, "
          f"{scored}): success {b.get('success_ci95','?')}")
    for key, label, fmt in _BEHAVIOR_ROWS:
        if (v := m.get(key)) is not None:
            print(f"    {label:<20} {fmt.format(v)}")
            summary[f"{prefix}{key}"] = v
    # refs #47: the death rate above, conditioned on success — the raw rate is
    # gameable by converting defeats into timeouts (which is where squad_v12b's
    # 0/100 partly comes from); this one only moves when commanders stop dying
    # inside wins. Filed so `--vs` carries it at both checkpoints.
    if (in_success := root_death_in_success(b.get("per_episode", []))) is not None:
        deaths, successes = in_success
        if successes:
            print(f"    {'root death in success':<20} {deaths}/{successes} = {deaths / successes:.3f}")
            summary[f"{prefix}root_death_in_success"] = deaths / successes
        else:  # an undefined rate is a named absence, never a 0.000
            print(f"    {'root death in success':<20} —/0 (no successful episodes)")
    # refs #46: the ordinal split, because the rule being priced is ordinal and
    # `false_complete_rate` above is a pool. Filed into the summary so `--vs`
    # prints it as a delta — reading squad_v12 against squad_v10 on a pooled
    # precision is the mistake this line exists to make impossible.
    if (ordinal := root_claim_ordinal(b.get("per_episode", []))) is not None:
        claims_line, burn_line = format_claim_ordinal(ordinal)
        print(f"    {'root claims':<20} {claims_line}")
        if burn_line:
            print(f"    {'first claim burned':<20} {burn_line}")
        summary[f"{prefix}root_claims"] = ordinal["claims"]
        for name, num, den in (
            ("first_claim_precision", ordinal["first_accepted"], ordinal["first"]),
            ("later_claim_precision", ordinal["later_accepted"], ordinal["later"]),
        ):
            if den:  # an undefined rate is left absent, never printed as 0.000
                summary[f"{prefix}{name}"] = num / den
    if diagnostics:
        # refs #14: a low preference rate is only a command-quality finding if
        # the ordered-task mix says it is not just adoption of one legal leg
        if mix := format_order_task_mix(m):
            print(f"    {'order task mix':<20} {mix}   (share/preference)")
        # refs #16: an order share is availability-confounded — the mask offers
        # the tasks in unequal numbers, and in opposite directions per scenario
        # family. The lift is the share over the masked-random floor: 1.00 is
        # no preference, and it is the number a fix has to move.
        if avail := format_order_availability(m):
            print(f"    {'order availability':<20} {avail}   (share/avail (xlift))")
        # a pooled obedience mean cannot separate disobedience from a shift to
        # slower-resolving tasks — see format_obedience_by_task
        if obey := format_obedience_by_task(m):
            print(f"    {'obey latency/task':<20} {obey}   (mean(orders))")
        # refs #15: staged orders are excluded from obedience by construction
        # (a staged agent complies by holding), so the AT MY COMMAND channel —
        # and orders staged and then never released — reports separately
        if staging := format_staging(m):
            print(f"    {'A5-2 staging':<20} {staging}")
    summary[f"{prefix}success"] = m.get("success_rate")
    # How many episodes these numbers are made of, so `--vs` can say whether the
    # two sides of a comparison were measured at the same N (refs #34)
    summary[f"{prefix}episodes"] = b.get("episodes")
    for g in b.get("gates", []):
        mark = gate_mark(g)
        print(f"    gate [{mark}] {g['name']} ({'>=' if g['direction'] == 'min' else '<='} {g['bound']})")
        if g.get("waived"):
            print(f"           waived — {g['waived']}")
    for mk in b.get("markers", []):
        val = "—" if mk["value"] is None else f"{mk['value']:.3f}"
        print(f"    marker   {mk['label']:<34} {val}")
        if mk.get("not_attributable"):
            print(f"             not attributable — {mk['not_attributable']}")
    return m


def best_selection_line(run: str, rows: list[dict], final_roll: float) -> str | None:
    """Where ``ckpt_best`` came from, and on what window it was chosen.

    ``ckpt_best`` is what ``cohort.play``, the gallery and every spot-check load
    by default, and since v1.20 it is selected lexicographically on the
    REPORTING channel before success. That makes the selecting window a fact
    about the run, and it was invisible: `patrol_brique_v19_rdb3_seed13` wrote
    its only ``ckpt_best`` at iteration 25 of 2930 — 25,600 of 3,000,320 steps —
    on a window at 2% success, and shipped it as the run's best work (refs #57).

    None when the checkpoint cannot be read (a live run mid-write) or its
    iteration is not in this metrics.csv (a resumed or truncated corpus): a
    digest line that cannot be trusted is worse than no digest line.
    """
    stamp = checkpoint_stamp(run_dir(run) / "ckpt_best.pt")
    if stamp is None:
        return None
    row = next((r for r in rows if fnum(r, "iteration") == stamp["iteration"]), None)
    if row is None:
        return None
    total = fnum(rows[-1], "env_steps") or 0
    share = stamp["env_steps"] / total if total else 0.0
    success, close = fnum(row, "success_rate_rolling"), fnum(row, "root_report_close_rolling")
    line = (f"  ckpt_best  iteration {stamp['iteration']} / {stamp['env_steps']:,} steps "
            f"({share:.0%} of the run)   that window: success "
            f"{'—' if success is None else f'{success:.0%}'}, closed-on-root "
            f"{'—' if close is None else f'{close:.3f}'}")
    if success is not None and final_roll == final_roll and final_roll - success > BEST_SELECTION_GAP:
        line += f"\n    ⚠ selected {final_roll - success:.0%} below the run's final window — see scripts/checkpoint_selection.py"
    return line


def rescue_line(run: str) -> str | None:
    """One line per run that was rolled back mid-training, or None.

    A rescued run's curve shows dips the collapse stop would once have ended
    the run at; without this line the digest reads them as ordinary recoveries
    and overstates the policy's own stability. States each rollback's step and
    the target_kl the run finished under so the intervention is on the record
    wherever the numbers are.
    """
    path = run_dir(run) / "rescues.json"
    if not path.exists():
        return None
    events = json.loads(path.read_text())
    if not events:
        return None
    at = ", ".join(f"{int(e['env_steps']):,}" for e in events)
    return (
        f"  rescues  {len(events)} rollback(s) to ckpt_best at step {at}  "
        f"(final target_kl {events[-1]['target_kl_after']}) — dips to the line "
        f"were interrupted, not outgrown"
    )


def report(run: str, show_components: bool) -> dict:
    rows = rows_of(run)
    cfg_path = run_dir(run) / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    first, last = deciles(rows)[0], deciles(rows)[-1]
    best = max((v for r in rows if (v := fnum(r, "success_rate_rolling")) is not None), default=float("nan"))

    print(f"== {run} ==")
    print(f"  scenario {cfg.get('scenario','?')}  seed {cfg.get('seed','?')}  "
          f"steps {int(fnum(rows[-1],'env_steps') or 0):,}/{cfg.get('total_steps',0):,}  "
          f"lr {cfg.get('lr','?')}  ent {cfg.get('ent_coef','?')}  n_envs {cfg.get('n_envs','?')}")
    final_roll = mean(last, "success_rate_rolling")
    print(f"  curve   {curve(rows)}   best rolling {best:.0%}   final {final_roll:.0%}")
    print(f"  {stability(best, final_roll)}")
    if rescues := rescue_line(run):
        print(rescues)
    if selection := best_selection_line(run, rows, final_roll):
        print(selection)
    if opt := optimizer_line(first, last):
        print(opt)
    print("  final decile vs first decile:")
    summary = {}
    for key, label in [
        ("success_rate_rolling", "success (rolling)"),
        ("ep_return", "ep return"),
        ("ep_length", "ep length"),
        ("entropy", "entropy"),
        ("human_death_rate", "human death rate"),
        ("cover_under_threat", "cover under threat"),
        ("objective_dist_under_threat", "dist from OBJ (threat)"),
        ("false_complete_rate", "false DONE rate"),
        ("tx_per_agent_step", "tx / agent-step"),
        # refs #18: tx is the CHARGED traffic only. Read the two together —
        # tx down with messages up is the stall signature (the cohort stopped
        # commanding and started talking for free), and tx alone called that
        # "the whole radio goes quiet" when the net had got 2.5x louder.
        ("messages_per_agent_step", "messages / agent-step"),
        ("timeout_rate_rolling", "ran clock out (rolling)"),
        ("approx_kl", "approx KL"),
    ]:
        a, b = mean(first, key), mean(last, key)
        if a != a and b != b:
            continue
        summary[key] = b
        print(f"    {label:<20} {a:>8.3f} → {b:>8.3f}   ({b - a:+.3f})")

    comps = [k for k in rows[0] if k.startswith("comp_")]
    if comps:
        drift = sorted(((mean(last, c) - mean(first, c), c) for c in comps), key=lambda t: -abs(t[0]))
        print("  reward components (final decile, biggest drift first):")
        for d, c in drift[:6] if not show_components else drift:
            print(f"    {c[5:]:<20} {mean(last, c):>8.4f}   ({d:+.4f})")

    beh_path = run_dir(run) / "behavior.json"
    if beh_path.exists():
        behavior_block(beh_path, "behavior", summary, "beh_", diagnostics=True)
    else:
        print("  behavior: none — run `evaluate --behavior` for the B2 suite")

    # The FINAL policy's own suite, side by side with ckpt_best's. On a run
    # that peaked and fell back these differ enormously (squad_screen_v4:
    # ckpt_best 30/30, ckpt_latest 0/30 on the same seeds), and the publishing
    # standard is that both are quoted — the headline is this one. It gets the
    # same rows and the same gates for that reason (refs #22): every prediction
    # in the v1.12 pre-registration is stated at the final policy, and until
    # this printed them the digest could not settle one of them.
    final_path = run_dir(run) / "behavior_final.json"
    if final_path.exists():
        fm = behavior_block(final_path, "FINAL policy", summary, "final_", diagnostics=False)
        if (bs := summary.get("beh_success")) is not None and (fs := fm.get("success_rate")) is not None:
            print(f"    vs ckpt_best {bs:.2f} → final {fs:.2f}  ({fs - bs:+.2f})")
    elif beh_path.exists():
        print("  FINAL policy: not measured — re-run `evaluate ckpt_latest.pt "
              "--behavior-out behavior_final.json` (the headline number)")
    return summary


def _cell(value: float | None, fmt: str) -> str:
    """One right-aligned number, or an em dash where the run never measured it."""
    return f"{fmt.format(value):>8}" if value is not None else f"{'—':>8}"


def _sample_sizes(run: str, baseline: str, n_run: int | None, n_base: int | None) -> str:
    """``N`` for both sides, and — loudly — whether they are the same N.

    A delta between a 20-episode arm and a 100-episode arm is not an effect
    size, and the difference is invisible once both are printed to three
    decimals. This is not hypothetical: the `squad_v8` comparator committed in
    this repository is an N=20 artifact and `squad_v9` publishes at N=100, so
    the A/B a reader can reconstruct from the repo is 5x mismatched while
    looking exactly like a matched one.
    """
    shown = f"N: {baseline} {n_base if n_base is not None else '?'} · {run} {n_run if n_run is not None else '?'}"
    if n_run is None or n_base is None:
        return f"{shown}   [N UNKNOWN on one side — matching unverified]"
    if n_run != n_base:
        return (f"{shown}   [MISMATCHED N — {n_base} vs {n_run}; the deltas below are "
                "NOT an effect size]")
    return f"{shown}   [matched]"


def comparison(run: str, baseline: str, a: dict, b: dict) -> None:
    """Success, root survival and the clock, side by side, with each side's N.

    This is the block an A/B verdict gets written against, so the pair that
    #34 asks for is printed here — by the instrument, on every comparison —
    rather than left to whoever writes the next ROADMAP table. See
    ``_COMPARISON_ROWS`` for why the three cells are inseparable.

    Everything degrades to an em dash and a named absence: a run that predates a
    metric, or was evaluated without ``--behavior``, prints "not measured on
    <run>" instead of a number, a zero, or a traceback. An unmeasured axis is
    not a passed one, and the comparison is still worth printing for the axes
    that were measured.
    """
    print(f"\n== A/B: {run} vs {baseline} ==")
    print("  success + root survival + the clock, together: a policy that never fights "
          "buries no\n  commanders, and one that wins can still bury them (refs #34)")
    for prefix, header in _COMPARISON_CHECKPOINTS:
        keys = [f"{prefix}{key}" for key, _, _ in _COMPARISON_ROWS]
        if all(a.get(k) is None and b.get(k) is None for k in keys):
            continue  # neither side evaluated this checkpoint; nothing to compare
        sizes = _sample_sizes(run, baseline, a.get(f"{prefix}episodes"), b.get(f"{prefix}episodes"))
        print(f"  {header}   {sizes}")
        for key, label, fmt in _COMPARISON_ROWS:
            va, vb = a.get(f"{prefix}{key}"), b.get(f"{prefix}{key}")
            line = f"    {label:<20} {_cell(vb, fmt)} → {_cell(va, fmt)}"
            if va is None or vb is None:
                absent = " and ".join(n for v, n in ((vb, baseline), (va, run)) if v is None)
                print(f"{line}   [not measured on {absent}]")
            else:
                print(f"{line}  ({va - vb:+.3f})")


def economics_diff(run: str, baseline: str) -> None:
    """Which reward/scenario prices actually differ between two runs.

    refs #20: this campaign's own confound audit was done by hand — open
    economics.json for each run, eyeball the ``rewards`` dict, count what
    changed — and it undercounted at least once. `squad_v7` -> `squad_v8`
    reads as a single-variable A/B (`done_false` only, the pair ROADMAP's
    audit used); `squad_v6` -> `squad_v8` — an equally legitimate choice of
    "the run before this one" — differs by TWO keys, `done_false` AND
    `contact_redundant`. Nothing forced the choice of baseline to be checked
    against the file that would have caught it: `economics.json` was written
    (train.py) *specifically* so "two runs a reward commit apart are
    indistinguishable after the fact" could not happen, and the manual
    process re-created the failure the file exists to prevent.

    This makes the check a function call instead of a transcription exercise,
    so the next campaign's "the only difference is X" claim is verified
    against the actual JSON, not asserted from memory of which commit landed
    when. Diffs both ``rewards`` (the price list) and ``spec`` (the scenario
    knobs) — `train.py::_spec_economics` calls scenario knobs part of the same
    "what is this run actually an experiment about" question.
    """
    a_path, b_path = run_dir(run) / "economics.json", run_dir(baseline) / "economics.json"
    if not a_path.exists() or not b_path.exists():
        missing = run if not a_path.exists() else baseline
        print(f"\n  economics: uncheckable — {missing} predates economics.json")
        return
    a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    diffs = [
        (section, key, b.get(section, {}).get(key), a.get(section, {}).get(key))
        for section in ("rewards", "spec")
        for key in sorted(set(a.get(section, {})) | set(b.get(section, {})))
        if a.get(section, {}).get(key) != b.get(section, {}).get(key)
    ]
    if not diffs:
        print(f"\n  economics: prices identical — {run} and {baseline} share every reward/spec value")
    else:
        label = ("one price differs" if len(diffs) == 1 else
                 f"{len(diffs)} prices differ")
        print(f"\n  economics: {label} ({run} vs {baseline})")
        for section, key, bv, av in diffs:
            print(f"    {section}.{key:<24} {bv!r} → {av!r}")
    code_diff(run, baseline, price_diffs=len(diffs))


def code_diff(run: str, baseline: str, price_diffs: int = 0) -> None:
    """The other half of "is this a single-variable A/B" — the code.

    ``economics_diff`` above compares prices, and for a year it printed
    "CLEAN — share every reward/spec value" as though that settled the
    question. It does not. `squad_v7` -> `squad_v8` reads price-clean on one
    key and was reported as a single-variable A/B, but `squad_v8` is also the
    first squad run carrying `d44ee8d` — a change to the environment itself.
    A code change never touches `economics.json`'s prices, so the instrument
    built to catch confounds was blind to half the class it was built for.

    The commit is already recorded (``economics.json:git_commit``); nothing
    consulted it. Now the two runs' commits are compared and, when they differ,
    the intervening commits that touched ``cohort/`` are listed — those are the
    ones that can move behaviour. Commits touching only ``scripts/``, ``tests/``
    or docs are counted and not listed: they cannot change what a policy learned.
    """
    commits = {}
    for name in (baseline, run):
        path = run_dir(name) / "economics.json"
        try:
            commits[name] = json.loads(path.read_text()).get("git_commit")
        except (OSError, json.JSONDecodeError):
            commits[name] = None

    code_moved: bool | None  # None = we could not tell, which is not "no"
    if not all(commits.values()):
        missing = [n for n, c in commits.items() if not c]
        print(f"    code: UNCHECKABLE — no git_commit recorded for {', '.join(missing)}")
        code_moved = None
    elif commits[run] == commits[baseline]:
        print(f"    code: same commit {commits[run][:8]}")
        code_moved = False
    else:
        span = _git(["rev-list", "--count", f"{commits[baseline]}..{commits[run]}"])
        touching = _git(
            ["log", "--oneline", f"{commits[baseline]}..{commits[run]}", "--", "cohort/"]
        )
        if span is None or touching is None:
            print(f"    code: UNCHECKABLE — {commits[baseline][:8]} is not an ancestor "
                  "of this run in this clone")
            code_moved = None
        else:
            lines = [ln for ln in touching.splitlines() if ln.strip()]
            print(f"    code: {commits[baseline][:8]} → {commits[run][:8]}  "
                  f"({int(span or 0)} commits, {len(lines)} touching cohort/)")
            for ln in lines[:8]:
                print(f"      {ln}")
            if len(lines) > 8:
                print(f"      … {len(lines) - 8} more")
            code_moved = bool(lines)

    # One verdict over both axes, because the failure this whole function exists
    # for was reading a clean verdict on one axis as a clean verdict overall.
    # Order matters: a known confound outranks an unknown one — two prices apart
    # is CONFOUNDED whether or not the code can be checked — and only a pair
    # that is clean on one axis and unreadable on the other is UNCHECKABLE.
    if price_diffs > 1:
        extra = " (and the code is unknown)" if code_moved is None else (
            " (and the environment moved too)" if code_moved else "")
        print(f"    verdict: CONFOUNDED — {price_diffs} prices differ{extra}")
    elif code_moved:
        print("    verdict: CONFOUNDED — the environment moved between these runs, "
              "whatever the prices say")
    elif code_moved is None:
        prices = "prices identical" if price_diffs == 0 else "one price differs"
        print(f"    verdict: UNCHECKABLE — {prices}, code unknown. "
              "Not the same finding as 'no difference'.")
    elif price_diffs == 1:
        print("    verdict: single-variable A/B — one price, same code")
    else:
        print("    verdict: IDENTICAL SETUP — same code, same prices")


def _git(argv: list[str]) -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", *argv], cwd=RUNS.parent, capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    show_components = "--components" in args
    args = [a for a in args if a != "--components"]
    run = args[0]
    baseline = None
    if "--vs" in args:
        baseline = args[args.index("--vs") + 1]

    a = report(run, show_components)
    if baseline:
        print()
        b = report(baseline, show_components)
        comparison(run, baseline, a, b)
        print(f"\n== delta: {run} - {baseline} ==")
        for key in sorted(set(a) & set(b)):
            if key in _EPISODE_COUNT_KEYS:
                continue  # a sample size, reported as one in the A/B block above
            va, vb = a.get(key), b.get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                mark = "  <-- " if abs(va - vb) > max(0.05 * abs(vb or 1), 0.02) else "      "
                print(f"  {key:<28} {vb:>8.3f} → {va:>8.3f}  ({va - vb:+.3f}){mark}")
        economics_diff(run, baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
