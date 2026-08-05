"""Order issuing, human injection, acknowledgement, and rank enforcement."""

import pytest

from cohort import make_env
from cohort.core.language import OrderParseError
from cohort.core.missions import MissionType
from cohort.env.actions import CATALOG


def test_human_hq_order_reaches_agent():
    env = make_env("fireteam")
    env.reset(seed=1)
    env.inject_order("RFN1, overwatch obj bravo", issuer="HQ")
    rfn1 = env.roster.by_callsign["RFN1"]
    assert rfn1.mission is not None
    assert rfn1.mission.type is MissionType.OVERWATCH
    assert env.world.objectives[rfn1.mission.objective_id].name == "BRAVO"
    texts = [m.text for m in env.transcript.messages]
    assert any("RFN1, THIS IS HQ: OVERWATCH OBJ BRAVO" in t for t in texts)
    assert any("WILCO" in t for t in texts), "orders are acknowledged on the net"


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
    env = make_env("fireteam")
    env.reset(seed=21)
    spec = next(
        s
        for s in CATALOG
        if s.kind == "order"
        and s.order_slot == 0
        and s.order_mission is MissionType.SEIZE
        and s.order_objective == "ALPHA"
    )
    env.step({a: (spec.index if a == "TL1" else 0) for a in env.agents})
    _obs, _r, *_ , infos = env.step({a: (spec.index if a == "TL1" else 0) for a in env.agents})
    assert infos["TL1"]["components"]["command"] < 0, "identical re-order is penalized"


def test_alternating_orders_cannot_farm_bonuses():
    """Cycling two different valid orders must not pay; it is churn.

    Regression test: an early policy learned to alternate SEIZE/ENGAGE to the
    same rifleman every step, collecting the doctrine-preferred bonus while
    dodging the identical-reissue check and flooding the radio net.
    """
    env = make_env("fireteam")
    env.reset(seed=21)

    def order(mission):
        return next(
            s
            for s in CATALOG
            if s.kind == "order"
            and s.order_slot == 0
            and s.order_mission is mission
            and s.order_objective == "ALPHA"
        )

    # initial tasking pays
    *_, infos = env.step({a: (order(MissionType.SEIZE).index if a == "TL1" else 0) for a in env.agents})
    assert infos["TL1"]["components"]["command"] > 0
    # flipping to a different valid order right away is churn, not command
    *_, infos = env.step({a: (order(MissionType.CLEAR).index if a == "TL1" else 0) for a in env.agents})
    assert infos["TL1"]["components"]["command"] < 0
    # ...and flipping back is churn again
    *_, infos = env.step({a: (order(MissionType.SEIZE).index if a == "TL1" else 0) for a in env.agents})
    assert infos["TL1"]["components"]["command"] < 0


def test_hold_anchors_where_received():
    env = make_env("fireteam")
    env.reset(seed=2)
    sld = env.roster.by_callsign["RFN2"]
    env.inject_order("RFN2, hold position")
    assert sld.mission.type is MissionType.HOLD
    assert sld.mission.anchor == sld.pos
