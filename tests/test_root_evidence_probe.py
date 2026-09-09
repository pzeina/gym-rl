"""Pins for root_evidence_probe's attribution and staleness rules.

The jam probe's first run reported a silent zero root claims for a policy the
behaviour suite scores at 0.842 — because it filtered on recipient and looked
the root up once per episode. Those two rules are pinned there; this probe
re-implements the pairing (it needs the claim message itself, not just its
step), so the same two hazards are pinned again here, plus the one new rule:
staleness is transcript-order "in hand", not step arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cohort.core.orders import HQ_ID, Message, MessageKind
from scripts.root_evidence_probe import pair_root_claims, staleness_before

ROOT_ID = 1
SUB_ID = 2


def msg(step: int, kind: MessageKind, sender: int, recipient: int | None) -> Message:
    return Message(step=step, kind=kind, sender_id=sender, recipient_id=recipient,
                   text=f"{kind.value} {sender}->{recipient}")


def test_a_root_claim_is_identified_by_sender_not_by_recipient():
    """A root's DONE is addressed to its leader, never to HQ."""
    messages = [
        msg(5, MessageKind.DONE, ROOT_ID, HQ_ID + 7),
        msg(5, MessageKind.DONE_CONFIRM, HQ_ID, ROOT_ID),
    ]
    got = pair_root_claims(messages, {5: ROOT_ID})
    assert len(got) == 1
    idx, claim, verdict = got[0]
    assert (idx, claim.step, verdict) == (0, 5, "confirmed")


def test_who_the_root_is_is_a_step_function():
    """Succession promotes mid-episode; a reset-time lookup misattributes."""
    messages = [
        msg(5, MessageKind.DONE, SUB_ID, ROOT_ID),
        msg(5, MessageKind.DONE_REJECT, ROOT_ID, SUB_ID),
        msg(40, MessageKind.DONE, SUB_ID, HQ_ID),
        msg(40, MessageKind.DONE_CONFIRM, HQ_ID, SUB_ID),
    ]
    got = pair_root_claims(messages, {5: ROOT_ID, 40: SUB_ID})
    assert [(c.step, v) for _, c, v in got] == [(40, "confirmed")]


def test_each_claim_keeps_its_own_verdict():
    messages = [
        msg(10, MessageKind.DONE, ROOT_ID, HQ_ID + 7),
        msg(10, MessageKind.DONE_REJECT, HQ_ID, ROOT_ID),
        msg(30, MessageKind.DONE, ROOT_ID, HQ_ID + 7),
        msg(30, MessageKind.DONE_CONFIRM, HQ_ID, ROOT_ID),
    ]
    got = pair_root_claims(messages, {10: ROOT_ID, 30: ROOT_ID})
    assert [(c.step, v) for _, c, v in got] == [(10, "rejected"), (30, "confirmed")]


def test_staleness_is_transcript_order_so_same_step_evidence_counts():
    """A DONE landing earlier in the claim's own step is evidence in hand.

    Step arithmetic alone would need a strict "< claim step" cut and drop it;
    transcript position is delivery order and keeps it, at staleness 0.
    """
    messages = [
        msg(12, MessageKind.DONE, SUB_ID, ROOT_ID),      # lands on the root
        msg(12, MessageKind.DONE_REJECT, ROOT_ID, SUB_ID),  # the root answers it
        msg(12, MessageKind.DONE, ROOT_ID, HQ_ID + 7),   # the root's own claim
        msg(12, MessageKind.DONE_CONFIRM, HQ_ID, ROOT_ID),
    ]
    got = pair_root_claims(messages, {12: ROOT_ID})
    assert [(i, v) for i, _, v in got] == [(2, "confirmed")]
    stale = staleness_before(messages, 2, 12, {12: ROOT_ID}, (MessageKind.DONE,))
    assert stale == 0


def test_the_roots_own_traffic_never_evidences_itself():
    messages = [
        msg(3, MessageKind.SITREP, ROOT_ID, ROOT_ID),
        msg(9, MessageKind.DONE, ROOT_ID, HQ_ID + 7),
    ]
    stale = staleness_before(messages, 1, 9, {3: ROOT_ID, 9: ROOT_ID},
                             (MessageKind.SITREP,))
    assert stale is None
