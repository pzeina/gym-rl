"""A raw death rate cannot represent commander survival inside wins (refs #47).

`squad_v12b` was rejected on success, replication and the timeout mode — all
correct, none disputed — and it is also the only squad arm on record with ZERO
root deaths: 0/100 at both checkpoints against its control's 20/100 and 25/100.
The rejection weighed none of that, and the raw rate could not have been
trusted anyway: `squad_v12b` takes zero defeats, converting them into timeouts,
and in its control every defeat IS a root death (5/5 and 10/10), so part of the
zero is the policy declining the fight — the exact conversion ``timeout_rate``
exists to flag.

The quantity that survives that objection is **root deaths within successful
episodes**: a success achieved the mission either way, so the conditioned rate
cannot be bought by riding the clock out — and it still separates the arms at
p < 1e-4 (0/96 and 0/86 vs 14/93 and 14/88). Same shape as ``done_reports``
without ``done_admissible``, order share without availability, the pooled claim
precision of #46: the missing quantity is the one the existing metric cannot
represent. So the digest now derives it (``run_report.root_death_in_success``)
from the ``outcome`` / ``human_died`` fields every behavior corpus already
records — no ``cohort/`` change, no re-evaluation, baseline seal untouched —
and it is pinned here against the committed corpora the finding was made on.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.fleet_status import find_run
from scripts.run_report import root_death_in_success

ROOT = Path(__file__).resolve().parents[1]


def _episodes(*pairs: tuple[str, bool]) -> list[dict]:
    """``(outcome, human_died)`` per episode, in the shape an artifact records."""
    return [{"outcome": outcome, "human_died": died} for outcome, died in pairs]


def test_only_successful_episodes_enter_either_side_of_the_rate():
    """Deaths in defeats and timeouts belong to the raw rate, not to this one."""
    counted = root_death_in_success(_episodes(
        ("success", True),    # the one this metric exists to count
        ("success", False),
        ("defeat", True),     # a defeat's death must not inflate the numerator
        ("timeout", False),   # and a timeout must not inflate the denominator
    ))

    assert counted == (1, 2)


def test_declining_the_fight_cannot_move_the_conditioned_rate():
    """The gaming this metric is immune to, in miniature (the v12b shape).

    Two corpora with identical behavior inside their wins; one converts its
    losing episodes from defeats-with-deaths into clean timeouts. The raw death
    rate halves — the conditioned rate does not move at all.
    """
    fights = _episodes(("success", True), ("success", False), ("defeat", True))
    declines = _episodes(("success", True), ("success", False), ("timeout", False))

    assert root_death_in_success(fights) == root_death_in_success(declines) == (1, 2)


def test_a_corpus_predating_per_episode_outcomes_reads_as_absent_not_as_zero():
    """Returning 0 would publish "no commander ever died in a win" about a corpus
    that never measured it — the em-dash rule ("an unmeasured axis is not a
    passed one") applies to this metric like every other."""
    assert root_death_in_success([{"length": 200, "messages": 90}]) is None
    assert root_death_in_success([]) is None


def test_a_corpus_with_no_wins_has_no_rate_rather_than_a_perfect_one():
    """0 deaths over 0 successes is undefined, not 0.000 — a policy that never
    wins must not read as one that keeps its commander alive while winning."""
    assert root_death_in_success(_episodes(("defeat", True), ("timeout", False))) == (0, 0)


def _corpus(run_name: str, artifact: str) -> list[dict]:
    """One committed behavior corpus, resolved rather than hard-pathed, so the
    pin survives the run being filed into ``runs/archive/``."""
    run = find_run(run_name, ROOT / "runs")
    assert run is not None, f"{run_name} is cited by ROADMAP's #47 entry and must resolve"
    return json.loads((run / artifact).read_text())["per_episode"]


def test_the_number_the_v12b_rejection_did_not_weigh_at_both_checkpoints():
    """The four counts the finding rests on, from the committed artifacts.

    Both checkpoints, because a survival number stated at one checkpoint is a
    number about an unstated policy — and here both agree, which is exactly
    what made the axis worth putting on the record.
    """
    for artifact, control, flag in (
        ("behavior.json", (14, 93), (0, 96)),        # ckpt_best
        ("behavior_final.json", (14, 88), (0, 86)),  # FINAL policy, the headline
    ):
        assert root_death_in_success(_corpus("squad_v10b", artifact)) == control
        assert root_death_in_success(_corpus("squad_v12b", artifact)) == flag

    # and the part of v12b's raw 0/100 that this metric deliberately discounts:
    # its control's defeats are commander deaths, every one, at both checkpoints
    for artifact, defeats in (("behavior.json", 5), ("behavior_final.json", 10)):
        lost = [ep for ep in _corpus("squad_v10b", artifact) if ep["outcome"] == "defeat"]
        assert [ep["human_died"] for ep in lost] == [True] * defeats
        assert not any(ep["outcome"] == "defeat" for ep in _corpus("squad_v12b", artifact))
