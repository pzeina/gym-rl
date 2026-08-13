"""SYNC PROPOSE / GO pay airtime, and the docs must not say otherwise (#18).

**Why this file exists, which is not the same as why the charge exists.** The
charge landed with #18, after ``squad_screen_v4``/``ckpt_latest`` poured 93% of
its traffic into the then-free voice channel — 1173 messages an episode — to run
the clock out. A speech act nobody pays for is an action sink.

What went wrong on 2026-08-13 is that **four separate comments still described
the pre-#18 world**, and one of them was read as current fact off the docstring
rather than the code. It produced a confident, wrong diagnosis of
``patrol_brique_v7`` — 2,098 voice messages read as a free-channel exploit when
the policy had been paying ``transmission_cost`` for every one of them — and a
recommendation to make a change that had already been made. The behaviour was
never wrong; only the record of it was.

So this pins the *claim*, not just the mechanism. A prose sweep fixes today's
copies; an assertion fails the next time one drifts. That is the same bargain as
the other regression-hazard tests in this suite: they each encode a real exploit
and are kept green so the exploit cannot come back quietly.

Two things are asserted, because the pair is what was misread:

1. **Voice is charged**, and charged to ``report`` — SYNC is speech between
   peers, not authority over a subordinate, and the ``flat`` ablation arm must
   still show command reward of exactly 0.0 while being able to say GO.
2. **Voice counts as a learned transmission** (``_tx_count``). Both halves of
   the old comment were wrong: it claimed voice was free *and* that a talking,
   non-commanding policy would therefore read as radio silence in ``tx``. The
   stall #18 closed now shows up in ``tx`` as volume, which is the opposite.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from cohort.env.cohort_env import CohortEnv
from cohort.env.rewards import RewardLedger

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _proposer(env: CohortEnv):
    """A soldier with at least one peer inside ``voice_range``, or skip.

    Picked from live state rather than hard-coded: the callsign that has a peer
    is a scenario detail, and pinning it would make this test about the roster.
    """
    from cohort.env.cohort_env import voice_peers

    for s in env.roster.living:
        if voice_peers(s, env.roster, env.spec_cfg.voice_range):
            return s
    pytest.skip("no soldier has a voice peer in this scenario's start state")


def test_sync_propose_pays_airtime_and_counts_as_a_transmission():
    """The mechanism, driven rather than read."""
    env = CohortEnv(scenario="squad")
    env.reset(seed=0)
    soldier = _proposer(env)
    ledger = RewardLedger()

    before = env._tx_count
    env._sync_propose(soldier, ledger)

    assert env._tx_count == before + 1, "voice must count as a learned transmission"
    charged = ledger.breakdown(soldier.callsign)["report"]
    assert charged == pytest.approx(env.rewards_cfg.transmission_cost), (
        "SYNC PROPOSE must pay transmission_cost, charged to `report` — "
        f"got {charged} against {env.rewards_cfg.transmission_cost}"
    )
    # and nothing lands in `command`: the flat ablation arm must read exactly 0.0
    assert ledger.breakdown(soldier.callsign)["command"] == 0.0


def test_sync_go_pays_airtime_too():
    """GO is charged like PROPOSE — but only when it actually lands.

    A GO with no live proposal says nothing, so it costs nothing; that early
    return is deliberate and is why the propose comes first here.
    """
    env = CohortEnv(scenario="squad")
    env.reset(seed=0)
    soldier = _proposer(env)
    ledger = RewardLedger()

    env._sync_propose(soldier, ledger)
    tx_after_propose = env._tx_count
    env._sync_go(soldier, ledger)

    assert env._tx_count == tx_after_propose + 1, "a landed GO is a transmission"
    assert ledger.breakdown(soldier.callsign)["report"] == pytest.approx(
        2 * env.rewards_cfg.transmission_cost
    )


def test_a_go_that_never_lands_says_nothing_and_costs_nothing():
    """The one thing that IS free, so the assertions above cannot overreach."""
    env = CohortEnv(scenario="squad")
    env.reset(seed=0)
    soldier = _proposer(env)
    ledger = RewardLedger()

    before = env._tx_count
    env._sync_go(soldier, ledger)  # no pending proposal

    assert env._tx_count == before
    assert ledger.breakdown(soldier.callsign)["report"] == 0.0


#: Files that describe the voice channel and must not resurrect the pre-#18
#: claim. Listed explicitly rather than globbed: a new file saying something
#: wrong is a new mistake, but these four are the ones that actually drifted,
#: and naming them says where the record lives.
_VOICE_DOCS = (
    "cohort/metrics.py",
    "cohort/training/train.py",
    "cohort/env/cohort_env.py",
    "docs/command_language.md",
)

#: Phrases that assert the channel is free. Each is matched only in the same
#: line as a voice word, so unrelated "no cost" comments (NET BUSY drops a
#: transmission that was never emitted — genuinely free) do not trip it.
_FREE = re.compile(
    r"(cost(s)? no airtime|is free|are free|for free|uncharged|not charged|free by design)",
    re.IGNORECASE,
)
_VOICE = re.compile(r"(voice|sync[_ ]propose|sync[_ ]go|VOICE_KINDS)", re.IGNORECASE)


def test_no_document_claims_the_voice_channel_is_free():
    """The claim, not just the mechanism — this is what actually went stale.

    Scoped to lines that mention voice AND assert freeness, so the many correct
    "no cost" comments elsewhere (a NET BUSY drop, a waived retask) are not
    swept up. A line that quotes the old world on purpose should say so in the
    past tense, which is what the fixed copies now do.
    """
    historical = re.compile(r"until 2026|pre-#18|#18 closed|it was,|this comment said|said ", re.I)
    offenders = []
    for rel in _VOICE_DOCS:
        path = ROOT / rel
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            if not (_VOICE.search(line) and _FREE.search(line)):
                continue
            # An explicit past-tense note is the correct way to quote the old
            # world, and it wraps across lines — so look at a small window, not
            # just the offending line.
            window = " ".join(lines[max(0, i - 4): i + 3])
            if historical.search(window):
                continue
            offenders.append(f"{rel}:{i}: {line.strip()}")

    assert not offenders, (
        "voice has paid airtime since #18; these lines still say otherwise, which "
        "is exactly the drift that produced a wrong diagnosis of patrol_brique_v7:\n  "
        + "\n  ".join(offenders)
    )
