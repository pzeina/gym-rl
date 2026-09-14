"""root_done_false: the root-scoped OPORD-claim rejection price.

The knob exists so the 2026-09-14 SEIZE-collapse diagnosis can be tested by
experiment arm (learning WHEN a root claim lands costs done_false per probe;
the claim is that this tax, at fleet-map walk distances, is what kills the
reporting mode). Two hazards pinned: the default must be BIT-IDENTICAL to
pre-knob behavior, and the override must touch ONLY the root's OPORD-claim
rejection — a subordinate's false DONE keeps its price, or the knob quietly
loosens the whole fleet's claim discipline.
"""

from __future__ import annotations

from dataclasses import replace

from cohort.config import get_scenario
from cohort.core.orders import MessageKind
from cohort.env.actions import CATALOG
from tests.test_garble import _flat_env, _step_all

DONE_IDX = next(s.index for s in CATALOG if s.kind == "done")


def _false_root_claim_reward(overrides=()):
    """TL1 files its OPORD claim with the objective still garrisoned: rejected."""
    spec = replace(get_scenario("fireteam"), reward_overrides=tuple(overrides))
    env = _flat_env(spec)
    _, rewards, *_ = _step_all(env, {"TL1": DONE_IDX})
    assert env.transcript.messages[-1].kind is MessageKind.DONE_REJECT
    return rewards["TL1"]


def _false_sub_claim_reward(overrides=()):
    """RFN1's CLEAR claim with the defender alive: rejected, subordinate-priced."""
    spec = replace(get_scenario("fireteam"), reward_overrides=tuple(overrides))
    env = _flat_env(spec)
    env.inject_order("RFN1, clear obj bravo", issuer="TL1")
    env.enemies[0].pos = env.world.objective_by_name("BRAVO").pos
    env.enemies[0].home = env.enemies[0].pos
    _, rewards, *_ = _step_all(env, {"RFN1": DONE_IDX})
    assert env.transcript.messages[-1].kind is MessageKind.DONE_REJECT
    return rewards["RFN1"]


def test_default_is_bit_identical_to_pre_knob_pricing():
    assert _false_root_claim_reward() == _false_root_claim_reward(
        [("root_done_false", -0.5)]
    ), "an explicit -0.5 override must equal the None default (done_false is -0.5)"


def test_override_reprices_exactly_the_root_opord_rejection():
    base = _false_root_claim_reward()
    cheap = _false_root_claim_reward([("root_done_false", -0.1)])
    assert abs((cheap - base) - 0.4) < 1e-9, (
        "the root's rejected OPORD claim must cost -0.1 instead of -0.5"
    )


def test_subordinate_false_claims_keep_their_price():
    base = _false_sub_claim_reward()
    with_knob = _false_sub_claim_reward([("root_done_false", -0.1)])
    assert base == with_knob, (
        "a subordinate's rejected DONE must be untouched by the root-scoped knob"
    )
