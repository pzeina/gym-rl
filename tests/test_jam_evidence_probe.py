"""The jamming probe's attribution rules — pinned because both were wrong once.

`scripts/jam_evidence_probe.py` reported **zero** root MISSION COMPLETE claims
on its first run, for a control whose behaviour suite scores
`closed_on_root_report_rate` 0.842 with 10 root claims. A zero is the most
dangerous possible defect in a probe: "the root never claims" is a publishable
sentence, and nothing about the output said it was a bug.

Two independent rules have to hold, and each of these tests fails if one of
them is reverted to its broken form.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cohort.core.orders import HQ_ID, Message, MessageKind
from scripts.jam_evidence_probe import (
    EVIDENCE_KINDS,
    attribute_root_claims,
    count_evidence_to_root,
)

ROOT_ID = 1
SUB_ID = 2


def msg(step: int, kind: MessageKind, sender: int, recipient: int | None) -> Message:
    return Message(step=step, kind=kind, sender_id=sender, recipient_id=recipient,
                   text=f"{kind.value} {sender}->{recipient}")


def test_a_root_claim_is_identified_by_sender_not_by_recipient():
    """The original bug: filtering on ``recipient_id == HQ_ID`` finds nothing.

    A root's DONE is addressed to its ``leader_id``. That is not HQ — HQ only
    answers it — so a recipient-side filter discards every root claim there is
    and returns the silent zero.
    """
    messages = [
        msg(5, MessageKind.DONE, ROOT_ID, HQ_ID + 7),   # NOT addressed to HQ
        msg(5, MessageKind.DONE_CONFIRM, HQ_ID, ROOT_ID),
    ]
    got = attribute_root_claims(messages, {5: ROOT_ID}, {5: False}, {5: 3})
    assert got == [(False, "confirmed", 3)], (
        "a root claim addressed to its leader must still be attributed to the root"
    )


def test_who_the_root_is_is_a_step_function():
    """Succession moves the root mid-episode.

    The subordinate's step-5 claim is NOT a root claim; the same agent's
    step-40 claim IS, because it has been promoted by then. A single
    reset-time lookup of "the root" gets exactly one of these two wrong.
    """
    messages = [
        msg(5, MessageKind.DONE, SUB_ID, ROOT_ID),
        msg(5, MessageKind.DONE_REJECT, ROOT_ID, SUB_ID),
        msg(40, MessageKind.DONE, SUB_ID, HQ_ID),
        msg(40, MessageKind.DONE_CONFIRM, HQ_ID, SUB_ID),
    ]
    root_at = {5: ROOT_ID, 40: SUB_ID}          # SUB_ID promoted before step 40
    got = attribute_root_claims(messages, root_at, {5: False, 40: True}, {5: 1, 40: 9})
    assert got == [(True, "confirmed", 9)], (
        "only the claim made while holding the root counts, and it keeps the "
        "jam state and staleness of its own step"
    )


def test_each_claim_keeps_its_own_verdict():
    """Verdicts are paired in order, never counted in aggregate.

    Two claims, one rejected then one confirmed. Counting totals would report
    both at the same precision; pairing keeps them apart, which is the whole
    point of conditioning precision on the jam state at the moment of claiming.
    """
    messages = [
        msg(10, MessageKind.DONE, ROOT_ID, HQ_ID),
        msg(10, MessageKind.DONE_REJECT, HQ_ID, ROOT_ID),
        msg(20, MessageKind.DONE, ROOT_ID, HQ_ID),
        msg(20, MessageKind.DONE_CONFIRM, HQ_ID, ROOT_ID),
    ]
    got = attribute_root_claims(messages, {10: ROOT_ID, 20: ROOT_ID},
                                {10: True, 20: False}, {10: 30, 20: 2})
    assert got == [(True, "rejected", 30), (False, "confirmed", 2)]


def test_evidence_counts_only_what_landed_on_the_root():
    """Evidence is counted off the transcript, and only toward the root.

    A message enters the transcript only when it lands, which is what makes
    this measure free of the ``_audible_to`` confound (that predicate is also
    consulted for action masks and ``useful=`` flags). Traffic to someone else,
    and an agent's own echo, are not the root's evidence.
    """
    messages = [
        msg(1, MessageKind.CONTACT, SUB_ID, ROOT_ID),        # counts
        msg(2, MessageKind.SITREP, SUB_ID, ROOT_ID),         # counts
        msg(3, MessageKind.CONTACT, SUB_ID, SUB_ID + 1),     # to a peer, not the root
        msg(4, MessageKind.ORDER, ROOT_ID, SUB_ID),          # command, not evidence
        msg(5, MessageKind.CONTACT, ROOT_ID, ROOT_ID),       # own echo
    ]
    root_at = dict.fromkeys(range(1, 6), ROOT_ID)
    assert count_evidence_to_root(messages, root_at) == 2


def test_orders_are_not_evidence():
    """The evidence set is reports up the chain, never traffic down it.

    If ORDER or ENDEX ever enters ``EVIDENCE_KINDS``, the jammed arm's evidence
    count absorbs the root's own command traffic — which rises under jamming —
    and the measured evidence loss shrinks toward zero for the wrong reason.
    """
    assert MessageKind.ORDER not in EVIDENCE_KINDS
    assert MessageKind.ENDEX not in EVIDENCE_KINDS
    assert MessageKind.DONE in EVIDENCE_KINDS
