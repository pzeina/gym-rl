"""A pooled claim precision cannot represent an ordinal rule (refs #46).

``root_done_bonus_first_claim_only`` prices the episode's FIRST root claim
differently from every later one. Its expected value was pre-registered for
``squad_v12`` off ``squad_v10``'s **pooled** claim precision — 77/178 = 0.433 —
and that single number describes neither ordinal: on the same corpus the first
claim is accepted at 0.543 and the later ones at 0.314. Worse, the split
INVERTS across the checkpoint (0.474 / 0.547 at ``ckpt_best``), so a precision
read off one policy does not describe the other. This is the third instance of
one shape in this project — ``done_reports`` without ``done_admissible``, order
share without availability, ``tx/agent-step`` blind to the uncharged voice
channel: **the missing quantity is the one the existing metric cannot
represent.**

So the split is now measured by the digest (``run_report.root_claim_ordinal``)
rather than by whoever remembers to, and pinned here against the committed
corpus the pre-registration was written from.

**The derivation is exact.** It rests on the invariant in
``test_confirmed_claim_is_last.py``: a confirmed root claim is the LAST root
claim of its episode, because confirming it closes the operation in the same
``step``. So ``done_reports_root - done_rejected_root`` is 0 or 1 per episode,
and where it is 1 the number of claims says which ordinal collected it.

**And the arithmetic that goes with it.** The EV pins below carry ``done_true``,
which the ROADMAP entries of 2026-08-11 dropped: they quote break-evens of
0.143 and 0.400 where ``rewards.py``'s own comment says 1/9 and 1/3. The
difference decides the question — at the pooled 0.433 a later claim under the
flag is worth **+0.139** and the spam does not stop paying; only at the measured
later-claim rate of 0.314 does it go negative.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cohort.env.rewards import RewardConfig
from scripts.fleet_status import find_run
from scripts.run_report import ClaimOrdinalError, format_claim_ordinal, root_claim_ordinal

ROOT = Path(__file__).resolve().parents[1]


def _episodes(*pairs: tuple[int, int]) -> list[dict]:
    """``(root claims, rejected)`` per episode, in the shape an artifact records."""
    return [{"done_reports_root": c, "done_rejected_root": r} for c, r in pairs]


def _claim_ev(p: float, cfg: RewardConfig, *, bonus: bool, burn: float = 0.0) -> float:
    """What one root claim is worth in expectation at acceptance rate ``p``.

    The canonical form, from ``rewards.py``: an accepted claim pays
    ``done_true`` and — while the bonus is still on the table — ``root_done_bonus``
    with the terminal; a rejected one pays ``done_false``; every claim pays
    ``transmission_cost`` for the airtime. ``burn`` is the v1.15 correction:
    under the first-claim rule a rejected opening probe forfeits the bonus for
    the whole episode, so it also costs ``root_done_bonus x P(the episode later
    closes by a root claim)``.
    """
    win = cfg.done_true + (cfg.root_done_bonus if bonus else 0.0)
    lose = cfg.done_false - (cfg.root_done_bonus * burn if bonus else 0.0)
    return p * win + (1 - p) * lose + cfg.transmission_cost


def test_the_acceptance_count_says_which_ordinal_collected_it():
    """One acceptance per episode; the claim count says whether it was the first."""
    split = root_claim_ordinal(_episodes((1, 0), (3, 2), (2, 2), (0, 0)))

    assert split == {
        "claims": 6,
        "first": 3,            # three episodes opened the channel
        "first_accepted": 1,   # only the single-claim one closed on its first
        "later": 3,
        "later_accepted": 1,   # the 3-claim episode closed on a later claim
        "first_rejected": 2,
        "closed_after_rejected_first": 1,
    }


def test_an_episode_that_never_claims_is_not_a_rejected_first_claim():
    """A silent root has no ordinal at all — it must not inflate any denominator."""
    assert root_claim_ordinal(_episodes((0, 0), (0, 0))) == {
        "claims": 0, "first": 0, "first_accepted": 0, "later": 0,
        "later_accepted": 0, "first_rejected": 0, "closed_after_rejected_first": 0,
    }


def test_a_corpus_predating_the_root_split_reads_as_absent_not_as_zero():
    """The 2026-08-07 era recorded ``done_reports`` only, pooled over all agents.

    Returning 0 there would publish "this policy never root-claimed" about a
    corpus that never measured it — the failure mode ``run_report``'s em-dash
    cells exist to prevent ("an unmeasured axis is not a passed one").
    """
    assert root_claim_ordinal([{"done_reports": 55, "done_rejected": 44}]) is None
    assert root_claim_ordinal([]) is None


def test_two_confirmed_root_claims_in_one_episode_raise_rather_than_report():
    """The impossible corpus is the one that gets quoted, so refuse to count it."""
    with pytest.raises(ClaimOrdinalError, match="episode 1"):
        root_claim_ordinal(_episodes((1, 0), (3, 1)))


def test_the_digest_line_carries_its_own_denominators():
    """A rate printed without its counts cannot be checked or pooled afterwards."""
    claims, burn = format_claim_ordinal(root_claim_ordinal(_episodes((1, 0), (3, 2), (2, 2))))

    assert claims == "6   first 1/3 = 0.333   later 1/3 = 0.333"
    assert burn is not None and burn.startswith("1/2 = 0.500")
    # nothing was rejected first, so there is nothing that could have burned
    assert format_claim_ordinal(root_claim_ordinal(_episodes((1, 0))))[1] is None


def _squad_v10(artifact: str) -> dict:
    """One committed ``squad_v10`` corpus, resolved rather than hard-pathed.

    ``squad_v10`` is the ``squad`` baseline member and the control arm of the
    price A/B; a data-level pin that switches itself off when its run is filed
    into ``runs/archive/`` is not a pin at all.
    """
    run = find_run("squad_v10", ROOT / "runs")
    assert run is not None, "squad_v10 is a sealed baseline member and must resolve"
    return json.loads((run / artifact).read_text())["per_episode"]


def test_squad_v10s_pool_describes_neither_ordinal_and_inverts_across_the_checkpoint():
    """The measurement the #46 correction rests on, at BOTH checkpoints.

    Reading one checkpoint here produces a confident wrong finding in either
    direction: the FINAL policy says first claims are the precise ones (0.543 vs
    0.314) and ``ckpt_best`` says the opposite (0.474 vs 0.547).
    """
    final = root_claim_ordinal(_squad_v10("behavior_final.json"))
    best = root_claim_ordinal(_squad_v10("behavior.json"))

    assert (final["claims"], final["first_accepted"], final["first"]) == (178, 50, 92)
    assert (final["later_accepted"], final["later"]) == (27, 86)
    assert (best["claims"], best["first_accepted"], best["first"]) == (170, 45, 95)
    assert (best["later_accepted"], best["later"]) == (41, 75)

    pooled_final = (final["first_accepted"] + final["later_accepted"]) / final["claims"]
    assert pooled_final == pytest.approx(0.433, abs=0.001)   # the pre-registered number

    def gap(o):
        return o["first_accepted"] / o["first"] - o["later_accepted"] / o["later"]

    assert gap(final) > 0.2 > 0 > gap(best), "the ordinal split no longer inverts"

    # what a spent first claim would forgo: episodes whose opening probe was
    # rejected and which closed by a later root claim anyway (rewards.py's P)
    assert final["closed_after_rejected_first"] / final["first_rejected"] == pytest.approx(
        0.643, abs=0.001
    )


def test_a_pooled_precision_cannot_price_the_rule_the_split_can():
    """The EV arithmetic, with ``done_true`` in it — the term the entries dropped.

    ``rewards.py`` states both break-evens itself: 1/9 with the bonus on the
    table, 1/3 once the slot is spent. Anything that reproduces 0.143 or 0.400
    has dropped ``done_true``, and at the pooled rate that error flips the sign
    of the whole pre-registration.
    """
    cfg = RewardConfig()
    assert -cfg.done_false / (cfg.done_true + cfg.root_done_bonus - cfg.done_false) == pytest.approx(1 / 9)
    assert -cfg.done_false / (cfg.done_true - cfg.done_false) == pytest.approx(1 / 3)

    pooled, later = 77 / 178, 27 / 86

    # a later claim under the first-claim rule: the bonus is gone, done_true is not
    assert _claim_ev(pooled, cfg, bonus=False) == pytest.approx(+0.139, abs=0.001)
    assert _claim_ev(later, cfg, bonus=False) == pytest.approx(-0.039, abs=0.001)
    assert pooled > 1 / 3 > later, "the pool and the ordinal fall on opposite sides of break-even"

    # and the honest first report stays worth filing on this corpus, burn included
    assert _claim_ev(50 / 92, cfg, bonus=True, burn=27 / 42) == pytest.approx(+1.055, abs=0.001)
