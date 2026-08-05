"""Order issuing, human injection, acknowledgement, and rank enforcement."""

from dataclasses import replace

import pytest

from cohort import make_env
from cohort.config import get_scenario
from cohort.core.language import OrderParseError
from cohort.core.missions import MissionType
from cohort.env.actions import CATALOG

ORDER_INDICES = [s.index for s in CATALOG if s.kind == "order"]


def _order_spec(mission, slot=0, objective="ALPHA"):
    return next(
        s
        for s in CATALOG
        if s.kind == "order"
        and s.order_slot == slot
        and s.order_mission is mission
        and s.order_objective == objective
    )


def _no_cooldown_env(scenario="fireteam"):
    """Env with the order-mask cooldown off, for tests of the churn *rewards*."""
    return make_env(replace(get_scenario(scenario), order_cooldown=0))


def test_radio_messages_are_text_only():
    """Owner decision: the net carries voice-procedure text, nothing else.

    Structured payloads on messages are forbidden — the transcript is the
    single source of truth for what was said, and ground truth for external
    analysis lives in the oracle observer channel, not in the messages.
    This test pins the Message schema so payloads cannot quietly return.
    """
    import dataclasses

    from cohort.core.orders import Message

    assert {f.name for f in dataclasses.fields(Message)} == {
        "step",
        "kind",
        "sender_id",
        "recipient_id",
        "text",
    }


def test_human_hq_order_reaches_agent():
    env = make_env("fireteam")
    env.reset(seed=1)
    env.inject_order("RFN1, observe obj bravo", issuer="HQ")
    rfn1 = env.roster.by_callsign["RFN1"]
    assert rfn1.mission is not None
    assert rfn1.mission.type is MissionType.OBSERVE
    assert env.world.objectives[rfn1.mission.objective_id].name == "BRAVO"
    texts = [m.text for m in env.transcript.messages]
    assert any("RFN1, THIS IS HQ: OBSERVE OBJ BRAVO" in t for t in texts)
    assert any("WILCO" in t for t in texts), "orders are acknowledged on the net"


def test_per_echelon_admissibility_on_injection():
    """DENY is a section mission (manual p. 8): a TL or RFN can never hold it."""
    env = make_env("squad")
    env.reset(seed=1)
    env.inject_order("SL1, deny obj alpha", issuer="HQ")  # SL: authority 2 — fine
    assert env.roster.by_callsign["SL1"].mission.type is MissionType.DENY
    with pytest.raises(PermissionError, match="cannot hold DENY"):
        env.inject_order("TL1, deny obj alpha", issuer="HQ")
    with pytest.raises(PermissionError, match="cannot hold DENY"):
        env.inject_order("RFN1, deny obj alpha", issuer="HQ")


def test_agent_issuer_must_outrank_and_own_the_subordinate():
    env = make_env("squad")
    env.reset(seed=1)
    # TL1 may order their own rifleman
    env.inject_order("RFN1, hold position", issuer="TL1")
    assert env.roster.by_callsign["RFN1"].mission.type is MissionType.HOLD
    # ...but not the squad leader above them
    with pytest.raises(PermissionError):
        env.inject_order("SL1, hold position", issuer="TL1")
    # ...and not a rifleman from the other fire team
    with pytest.raises(PermissionError):
        env.inject_order("RFN3, hold position", issuer="TL1")


def test_unknown_station_rejected():
    env = make_env("fireteam")
    env.reset(seed=1)
    with pytest.raises(OrderParseError):
        env.inject_order("TL9, seize obj alpha")
    with pytest.raises(OrderParseError):
        env.inject_order("RFN1, seize obj zulu")


def test_learned_order_action_assigns_mission_and_logs():
    env = make_env("fireteam")
    env.reset(seed=21)
    # TL1 orders subordinate slot 0 (RFN1): SEIZE OBJ ALPHA — doctrine-preferred
    spec = next(
        s
        for s in CATALOG
        if s.kind == "order"
        and s.order_slot == 0
        and s.order_mission is MissionType.SEIZE
        and s.order_objective == "ALPHA"
    )
    _obs, _rewards, *_ , infos = env.step({a: (spec.index if a == "TL1" else 0) for a in env.agents})
    rfn1 = env.roster.by_callsign["RFN1"]
    assert rfn1.mission is not None and rfn1.mission.type is MissionType.SEIZE
    assert infos["TL1"]["components"]["command"] > 0, "doctrine-preferred order pays"
    order_msgs = [m for m in env.transcript.messages if m.kind.value == "order"]
    assert any("RFN1, THIS IS TL1: SEIZE OBJ ALPHA" in m.text for m in order_msgs)


def test_reissuing_standing_order_is_churn():
    # cooldown off: this test exercises the churn *penalty* on a back-to-back
    # re-order, which the default order-mask cooldown would forbid outright
    env = _no_cooldown_env()
    env.reset(seed=21)
    spec = _order_spec(MissionType.SEIZE)
    env.step({a: (spec.index if a == "TL1" else 0) for a in env.agents})
    _obs, _r, *_ , infos = env.step({a: (spec.index if a == "TL1" else 0) for a in env.agents})
    assert infos["TL1"]["components"]["command"] < 0, "identical re-order is penalized"


def test_alternating_orders_cannot_farm_bonuses():
    """Cycling two different valid orders must not pay; it is churn.

    Regression test: an early policy learned to alternate SEIZE/ENGAGE to the
    same rifleman every step, collecting the doctrine-preferred bonus while
    dodging the identical-reissue check and flooding the radio net.
    (Cooldown off: with the default order-mask cooldown this exploit is
    masked away entirely — see the cooldown tests below.)
    """
    env = _no_cooldown_env()
    env.reset(seed=21)

    # initial tasking pays
    *_, infos = env.step({a: (_order_spec(MissionType.SEIZE).index if a == "TL1" else 0) for a in env.agents})
    assert infos["TL1"]["components"]["command"] > 0
    # flipping to a different valid order right away is churn, not command
    *_, infos = env.step({a: (_order_spec(MissionType.CLEAR).index if a == "TL1" else 0) for a in env.agents})
    assert infos["TL1"]["components"]["command"] < 0
    # ...and flipping back is churn again
    *_, infos = env.step({a: (_order_spec(MissionType.SEIZE).index if a == "TL1" else 0) for a in env.agents})
    assert infos["TL1"]["components"]["command"] < 0


def test_hold_anchors_where_received():
    env = make_env("fireteam")
    env.reset(seed=2)
    sld = env.roster.by_callsign["RFN2"]
    env.inject_order("RFN2, hold position")
    assert sld.mission.type is MissionType.HOLD
    assert sld.mission.anchor == sld.pos


# ---------------------------------------------------------------------- #
# order-mask cooldown (churn prevention at the mask level)
# ---------------------------------------------------------------------- #


def _slot_order_indices(slot):
    return [s.index for s in CATALOG if s.kind == "order" and s.order_slot == slot]


def _flat_fireteam(seed=21, **spec_overrides):
    """Fireteam env on open terrain with enemies parked far away."""
    env = make_env(replace(get_scenario("fireteam"), **spec_overrides))
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (22, 1)
        e.home = e.pos
    return env


def test_order_cooldown_masks_prompt_retasking():
    """A just-tasked subordinate cannot be re-tasked until the cooldown expires."""
    env = _flat_fireteam()  # default order_cooldown = 8
    spec = _order_spec(MissionType.SEIZE)
    obs, *_ = env.step({a: (spec.index if a == "TL1" else 0) for a in env.agents})  # step 1
    slot0, slot1 = _slot_order_indices(0), _slot_order_indices(1)
    assert obs["TL1"]["action_mask"][slot0].sum() == 0, "slot 0 re-tasking must be masked"
    assert obs["TL1"]["action_mask"][slot1].sum() > 0, "untasked RFN2 stays orderable"
    for _ in range(7):  # steps 2..8: still inside the cooldown
        obs, *_ = env.step({a: 0 for a in env.agents})
        assert obs["TL1"]["action_mask"][slot0].sum() == 0
    obs, *_ = env.step({a: 0 for a in env.agents})  # step 9: cooldown expired
    assert obs["TL1"]["action_mask"][slot0].sum() > 0


def test_order_cooldown_lifts_on_contact():
    """A CONTACT report on the net re-opens the order vocabulary at once."""
    env = _flat_fireteam()
    spec = _order_spec(MissionType.SEIZE)
    obs, *_ = env.step({a: (spec.index if a == "TL1" else 0) for a in env.agents})
    slot0 = _slot_order_indices(0)
    assert obs["TL1"]["action_mask"][slot0].sum() == 0
    # RFN2 spots an enemy and reports: the situation changed, re-tasking allowed
    rfn2 = env.roster.by_callsign["RFN2"]
    enemy = env.enemies[0]
    enemy.pos = (rfn2.pos[0] + 2, rfn2.pos[1])
    enemy.home = enemy.pos
    contact_idx = next(s.index for s in CATALOG if s.kind == "contact")
    obs, *_ = env.step({a: (contact_idx if a == "RFN2" else 0) for a in env.agents})
    assert obs["TL1"]["action_mask"][slot0].sum() > 0, "CONTACT on the net lifts the cooldown"


def test_order_cooldown_lifts_when_superior_intent_changes():
    """A new order to the leader itself re-opens its subordinate vocabulary."""
    env = _flat_fireteam()
    spec = _order_spec(MissionType.SEIZE)
    env.step({a: (spec.index if a == "TL1" else 0) for a in env.agents})  # step 1
    obs, *_ = env.step({a: 0 for a in env.agents})  # step 2
    slot0 = _slot_order_indices(0)
    assert obs["TL1"]["action_mask"][slot0].sum() == 0
    env.inject_order("TL1, clear obj alpha", issuer="HQ")  # fresh superior intent
    obs, *_ = env.step({a: 0 for a in env.agents})
    assert obs["TL1"]["action_mask"][slot0].sum() > 0, "new own mission lifts the cooldown"


def test_order_cooldown_zero_restores_prompt_retasking():
    env = _flat_fireteam(order_cooldown=0)
    spec = _order_spec(MissionType.SEIZE)
    env.step({a: (spec.index if a == "TL1" else 0) for a in env.agents})
    obs, *_ = env.step({a: 0 for a in env.agents})
    assert obs["TL1"]["action_mask"][_slot_order_indices(0)].sum() > 0


def test_auto_ack_disabled_suppresses_wilco():
    """auto_ack=False: the ORDER lands and is applied, but no WILCO is emitted."""
    env = make_env(replace(get_scenario("fireteam"), auto_ack=False))
    env.reset(seed=1)
    env.inject_order("RFN1, seize obj alpha", issuer="HQ")
    kinds = [m.kind.value for m in env.transcript.messages]
    assert "order" in kinds
    assert "ack" not in kinds
    assert env.roster.by_callsign["RFN1"].mission is not None, "the order itself still lands"
