"""Order timing qualifiers (A5-2): AT T PLUS n / AT MY COMMAND + EXECUTE.

A pending order stages its recipient (compliance = HOLD near the position
where the order landed), is observable as pending, cannot be reported
COMPLETE, and starts binding — tenure included — only when it becomes
effective (its tick comes due, or the issuer broadcasts EXECUTE).
"""

import pytest

from cohort import make_env
from cohort.core import language as lang
from cohort.core.missions import MissionType, is_pending
from cohort.env.actions import CATALOG

STAY = 0
EXECUTE_IDX = next(s.index for s in CATALOG if s.kind == "execute")
DONE_IDX = next(s.index for s in CATALOG if s.kind == "done")


def _stay_all(env):
    return dict.fromkeys(env.agents, STAY)


# ---------------------------------------------------------------------- #
# language round-trips
# ---------------------------------------------------------------------- #


def test_round_trip_t_plus():
    text = lang.format_order("SL1", "TL1", MissionType.SEIZE, "ALPHA", delay=5)
    assert text == "TL1, THIS IS SL1: SEIZE OBJ ALPHA AT T PLUS 5. OUT."
    parsed = lang.parse_order(text)
    assert parsed.mission is MissionType.SEIZE
    assert parsed.objective_name == "ALPHA"
    assert parsed.delay == 5
    assert not parsed.at_my_command


def test_round_trip_at_my_command():
    text = lang.format_order("SL1", "TL1", MissionType.ADVANCE, "GOLD", at_my_command=True)
    assert text == "TL1, THIS IS SL1: ADVANCE TO WP GOLD AT MY COMMAND. OUT."
    parsed = lang.parse_order(text)
    assert parsed.mission is MissionType.ADVANCE
    assert parsed.control_name == "GOLD"
    assert parsed.at_my_command
    assert parsed.delay is None


def test_timing_parse_variants():
    assert lang.parse_order("TL1, seize obj alpha at t+7").delay == 7
    assert lang.parse_order("TL1, advance to pl amber at my command").at_my_command
    plain = lang.parse_order("TL1, seize obj alpha")
    assert plain.delay is None and not plain.at_my_command


def test_execute_formatter():
    assert lang.format_execute("SL1") == "ALL STATIONS, THIS IS SL1: EXECUTE. OUT."


# ---------------------------------------------------------------------- #
# pending semantics in the env
# ---------------------------------------------------------------------- #


def test_t_plus_stages_then_releases():
    env = make_env("fireteam")
    env.reset(seed=5)
    env.inject_order("TL1, seize obj bravo at t plus 3", issuer="HQ")
    tl = env.roster.by_callsign["TL1"]
    m = tl.mission
    assert m.effective_at == 3
    assert m.extra["staging"] == tl.pos
    assert is_pending(m, 0)
    # while pending: the anchor is the staging spot, not BRAVO
    assert env._mission_anchor(tl) == m.extra["staging"]
    env.step(_stay_all(env))  # t=1
    env.step(_stay_all(env))  # t=2
    assert is_pending(tl.mission, env._step_count)
    env.step(_stay_all(env))  # t=3: comes due
    assert not is_pending(tl.mission, env._step_count)
    assert tl.mission.effective_at is None
    assert tl.mission.step_assigned == 3, "binding (tenure) starts at release"
    bravo = env.world.objective_by_name("BRAVO")
    assert env._mission_anchor(tl) == bravo.pos


def test_pending_compliance_is_hold_at_staging():
    """Staying put while staged earns positive compliance; the SEIZE target
    does not pull the pending recipient."""
    env = make_env("fireteam")
    env.reset(seed=5)
    env.inject_order("TL1, seize obj bravo at t plus 30", issuer="HQ")
    env.step(_stay_all(env))
    # positive compliance for holding at the staging position
    tl = env.roster.by_callsign["TL1"]
    ctx = env._compliance_ctx(tl, None, env._make_view(tl))
    assert ctx.in_position, "staged agent holding its spot is in position"


def test_pending_mission_cannot_report_done():
    env = make_env("fireteam")
    env.reset(seed=5)
    env.inject_order("TL1, advance to wp gold at my command", issuer="HQ")
    tl = env.roster.by_callsign["TL1"]
    assert env._mask_for(tl)[DONE_IDX] == 0, "pending: nothing to report"
    env.inject_execute("HQ")
    assert not is_pending(tl.mission, env._step_count)
    assert env._mask_for(tl)[DONE_IDX] == 1


def test_pending_state_is_observable():
    env = make_env("fireteam")
    obs, _ = env.reset(seed=5)
    base = 13 + 12 + 1 + 4  # pending fields sit after the mission anchor block
    vec = obs["TL1"]["observation"]
    assert vec[base] == 0.0 and vec[base + 1] == 0.0  # OPORD is immediate
    env.inject_order("TL1, seize obj bravo at t plus 10", issuer="HQ")
    vec = env._all_observations()["TL1"]["observation"]
    assert vec[base] == 1.0
    assert vec[base + 1] == pytest.approx(0.5)  # 10 remaining / 20
    env.inject_order("TL1, seize obj bravo at my command", issuer="HQ")
    vec = env._all_observations()["TL1"]["observation"]
    assert vec[base] == 1.0
    assert vec[base + 1] == 1.0  # AT MY COMMAND: no clock


# ---------------------------------------------------------------------- #
# EXECUTE_SIGNAL (learned) + AMC order variants
# ---------------------------------------------------------------------- #


def test_learned_amc_order_then_execute_signal():
    env = make_env("fireteam")
    env.reset(seed=3)
    tl = env.roster.by_callsign["TL1"]
    amc_spec = next(
        s
        for s in CATALOG
        if s.kind == "order"
        and s.order_mission is MissionType.ADVANCE
        and s.order_slot == 0
        and s.order_control == "GOLD"
        and s.order_amc
    )
    assert env._mask_for(tl)[EXECUTE_IDX] == 0, "no pending AMC orders yet"
    actions = _stay_all(env)
    actions["TL1"] = amc_spec.index
    env.step(actions)
    rfn = tl.living_subordinates(env.roster)[0]
    assert rfn.mission is not None and rfn.mission.awaiting_signal
    assert is_pending(rfn.mission, env._step_count)
    order = next(m for m in env.transcript.messages if m.kind.value == "order")
    assert "ADVANCE TO WP GOLD AT MY COMMAND" in order.text

    # only the issuer may EXECUTE, and only while something is pending
    assert env._mask_for(tl)[EXECUTE_IDX] == 1
    for cs in ("RFN1", "RFN2", "RFN3"):
        soldier = env.roster.by_callsign[cs]
        assert env._mask_for(soldier)[EXECUTE_IDX] == 0

    actions = _stay_all(env)
    actions["TL1"] = EXECUTE_IDX
    env.step(actions)
    assert not rfn.mission.awaiting_signal
    assert rfn.mission.step_assigned == env._step_count, "binding starts at EXECUTE"
    exe = next(m for m in env.transcript.messages if m.kind.value == "execute")
    assert exe.text == "ALL STATIONS, THIS IS TL1: EXECUTE. OUT."
    assert env._mask_for(tl)[EXECUTE_IDX] == 0, "nothing pending anymore"


def test_execute_releases_all_pending_of_issuer_only():
    env = make_env("squad")
    env.reset(seed=3)
    env.inject_order("TL1, advance to wp gold at my command", issuer="SL1")
    env.inject_order("TL2, advance to wp silver at my command", issuer="SL1")
    tl1 = env.roster.by_callsign["TL1"]
    tl2 = env.roster.by_callsign["TL2"]
    assert tl1.mission.awaiting_signal and tl2.mission.awaiting_signal
    env.inject_execute("HQ")  # HQ issued nothing pending: releases nothing
    assert tl1.mission.awaiting_signal and tl2.mission.awaiting_signal
    env.inject_execute("SL1")  # ONE broadcast frees every staged recipient
    assert not tl1.mission.awaiting_signal
    assert not tl2.mission.awaiting_signal


# ---------------------------------------------------------------------- #
# probe honors timing
# ---------------------------------------------------------------------- #


def test_probe_pending_predicts_staging_then_target():
    from cohort.probe import HOLD, MOVING, STATIC, NetPredictor, cm_class, make_briefing

    p = NetPredictor(make_briefing("fireteam"))
    text = lang.format_order("HQ", "TL1", MissionType.ADVANCE, "GOLD", at_my_command=True)
    p.observe(0, [{"kind": "order", "from": "HQ", "to": "TL1", "text": text}])
    assert p.predict("TL1") == (HOLD, STATIC), "staged: holding where it stands"
    p.observe(4, [{"kind": "execute", "from": "HQ", "to": "ALL",
                   "text": lang.format_execute("HQ")}])
    assert p.predict("TL1") == (cm_class("GOLD"), MOVING), "released: en route"


def test_probe_t_plus_release_by_clock():
    from cohort.probe import HOLD, MOVING, STATIC, NetPredictor, cm_class, make_briefing

    p = NetPredictor(make_briefing("fireteam"))
    text = lang.format_order("HQ", "TL1", MissionType.ADVANCE, "GOLD", delay=6)
    p.observe(0, [{"kind": "order", "from": "HQ", "to": "TL1", "text": text}])
    assert p.predict("TL1") == (HOLD, STATIC)
    p.observe(5, [])
    assert p.predict("TL1") == (HOLD, STATIC), "one tick early: still staged"
    p.observe(6, [])
    assert p.predict("TL1") == (cm_class("GOLD"), MOVING)
