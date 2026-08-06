"""B5 binding-order economics: re-task pricing + standing-order tenure.

Re-tasking an already-tasked subordinate is an act of command with real
weight — priced by the issuer's rank, half price when only the mission type
changes on the same anchor — UNLESS the tactical picture changed since the
standing order landed (contact on the net, a casualty in the issuer's
element, the issuer's own mission changed, or the subordinate's truthful
DONE, which clears the mission and makes the next order fresh). Positive
compliance credit grows with standing-order tenure, so settled, executed
orders out-earn churned ones — and terminal dominance still holds.
"""

from dataclasses import replace

import pytest

from cohort import make_env
from cohort.config import get_scenario
from cohort.core.missions import Mission, MissionType
from cohort.core.units import Trap
from cohort.env.actions import CATALOG
from cohort.env.rewards import RewardConfig

STAY = 0
MOVE_EAST = next(s.index for s in CATALOG if s.name == "MOVE_EAST")
MOVE_WEST = next(s.index for s in CATALOG if s.name == "MOVE_WEST")
CONTACT = next(s.index for s in CATALOG if s.kind == "contact")
DONE = next(s.index for s in CATALOG if s.kind == "done")


def _order_spec(mission, slot=0, objective="ALPHA"):
    return next(
        s
        for s in CATALOG
        if s.kind == "order"
        and s.order_slot == slot
        and s.order_mission is mission
        and s.order_objective == objective
    )


def _flat_env(scenario="fireteam", seed=21, reward_config=None, **spec_overrides):
    """Env on open terrain, cooldown off (economics, not masks), enemies far."""
    spec_overrides.setdefault("order_cooldown", 0)
    env = make_env(replace(get_scenario(scenario), **spec_overrides), reward_config=reward_config)
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (env.world.width - 2, 1)
        e.home = e.pos
        e.last_seen_player = None
    return env


def _step_all(env, overrides=None):
    acts = {a: STAY for a in env.agents}
    acts.update(overrides or {})
    return env.step(acts)


def _retask_price(cfg: RewardConfig, authority: int, *, same_anchor=False) -> float:
    price = cfg.order_retask_cost_base * (1.0 + cfg.order_retask_rank_scale * authority)
    return price * 0.5 if same_anchor else price


# ---------------------------------------------------------------------- #
# re-task pricing
# ---------------------------------------------------------------------- #


def test_retask_to_new_objective_pays_full_rank_price():
    """Rotating a tasked subordinate to another objective, with no change in
    the tactical picture, costs the issuer the full rank-scaled price."""
    env = _flat_env()
    cfg = env.rewards_cfg
    _step_all(env, {"TL1": _order_spec(MissionType.SEIZE).index})
    *_, infos = _step_all(env, {"TL1": _order_spec(MissionType.SEIZE, objective="BRAVO").index})
    # TL authority 1: -0.5 x 1.5 = -0.75; plus airtime and the standing
    # coverage gap (RFN2/RFN3 untasked) — no fresh-tasking bonus applies
    assert infos["TL1"]["components"]["command"] == pytest.approx(
        _retask_price(cfg, 1) + cfg.transmission_cost + cfg.coverage_gap
    )
    # the order still APPLIES: commanding stays possible, just expensive
    rfn1 = env.roster.by_callsign["RFN1"]
    assert rfn1.mission.type is MissionType.SEIZE
    assert env.world.objectives[rfn1.mission.objective_id].name == "BRAVO"
    (event,) = env.retask_events_last_step
    assert event["issuer"] == "TL1" and event["recipient"] == "RFN1"
    assert event["excepted"] is False and event["same_anchor"] is False
    assert event["cost"] == pytest.approx(_retask_price(cfg, 1))


def test_same_objective_mission_change_is_half_price():
    """SEIZE→CLEAR on the same objective changes the task, not the anchor:
    half the rank-scaled price."""
    env = _flat_env()
    cfg = env.rewards_cfg
    _step_all(env, {"TL1": _order_spec(MissionType.SEIZE).index})
    *_, infos = _step_all(env, {"TL1": _order_spec(MissionType.CLEAR).index})
    assert infos["TL1"]["components"]["command"] == pytest.approx(
        _retask_price(cfg, 1, same_anchor=True) + cfg.transmission_cost + cfg.coverage_gap
    )
    (event,) = env.retask_events_last_step
    assert event["same_anchor"] is True and event["excepted"] is False


def test_retask_price_scales_with_issuer_rank():
    """The heavier the rank, the heavier the act of command: an SL pays more
    than a TL for the same whim."""
    env = _flat_env("squad")
    cfg = env.rewards_cfg
    # SL1 (authority 2) tasks TL1 (slot 0), then rotates it for no reason
    _step_all(env, {"SL1": _order_spec(MissionType.SEIZE).index})
    *_, infos = _step_all(env, {"SL1": _order_spec(MissionType.SEIZE, objective="BRAVO").index})
    (event,) = env.retask_events_last_step
    assert event["rank"] == "SL" and event["authority"] == 2
    assert event["cost"] == pytest.approx(_retask_price(cfg, 2))  # -1.0 at defaults
    assert _retask_price(cfg, 2) < _retask_price(cfg, 1) < 0
    assert infos["SL1"]["components"]["command"] == pytest.approx(
        _retask_price(cfg, 2) + cfg.transmission_cost + cfg.coverage_gap
    )


def test_contact_on_the_net_makes_the_retask_free():
    """A CONTACT since the standing order is the carve-out: intervening on a
    changed picture costs nothing beyond airtime."""
    env = _flat_env()
    cfg = env.rewards_cfg
    _step_all(env, {"TL1": _order_spec(MissionType.SEIZE).index})
    # RFN2 spots an enemy and reports: the tactical picture changed
    rfn2 = env.roster.by_callsign["RFN2"]
    enemy = env.enemies[0]
    enemy.pos = (rfn2.pos[0] + 2, rfn2.pos[1])
    enemy.home = enemy.pos
    _step_all(env, {"RFN2": CONTACT})
    *_, infos = _step_all(env, {"TL1": _order_spec(MissionType.SEIZE, objective="BRAVO").index})
    (event,) = env.retask_events_last_step
    assert event["excepted"] is True and event["reason"] == "contact"
    assert event["cost"] == 0.0
    assert infos["TL1"]["components"]["command"] == pytest.approx(
        cfg.transmission_cost + cfg.coverage_gap
    )


def test_casualty_in_the_element_makes_the_retask_free():
    """A death in the issuer's element since the standing order is news:
    re-tasking the survivors is free."""
    env = _flat_env()
    cfg = env.rewards_cfg
    _step_all(env, {"TL1": _order_spec(MissionType.SEIZE).index})
    # RFN2 (in TL1's element) steps on a device and dies
    rfn2 = env.roster.by_callsign["RFN2"]
    rfn2.health = 10
    env.traps.append(Trap(id=9, pos=(rfn2.pos[0] + 1, rfn2.pos[1])))
    _step_all(env, {"RFN2": MOVE_EAST})
    assert not rfn2.alive
    *_, infos = _step_all(env, {"TL1": _order_spec(MissionType.SEIZE, objective="BRAVO").index})
    (event,) = env.retask_events_last_step
    assert event["excepted"] is True and event["reason"] == "casualty"
    assert infos["TL1"]["components"]["command"] == pytest.approx(
        cfg.transmission_cost + cfg.coverage_gap
    )


def test_new_superior_intent_makes_the_retask_free_and_pays_propagation():
    """The issuer's own mission changed: rotating subordinates onto the new
    plan is free AND earns the fresh-tasking propagation credit, as before."""
    env = _flat_env()
    cfg = env.rewards_cfg
    _step_all(env, {"TL1": _order_spec(MissionType.SEIZE).index})
    _step_all(env)  # a tick passes, then HQ re-orients TL1: fresh superior intent
    env.inject_order("TL1, seize obj bravo", issuer="HQ")
    *_, infos = _step_all(env, {"TL1": _order_spec(MissionType.SEIZE, objective="BRAVO").index})
    (event,) = env.retask_events_last_step
    assert event["excepted"] is True and event["reason"] == "intent"
    # preferred derivation + objective match + airtime + coverage gap
    assert infos["TL1"]["components"]["command"] == pytest.approx(
        cfg.order_preferred + cfg.order_objective_match + cfg.transmission_cost + cfg.coverage_gap
    )


def test_truthful_done_clears_the_mission_so_the_next_order_is_fresh():
    """The subordinate's confirmed MISSION COMPLETE cleared its mission: the
    follow-up order is a fresh tasking — no re-task event, no price."""
    env = _flat_env()
    cfg = env.rewards_cfg
    _step_all(env, {"TL1": _order_spec(MissionType.SEIZE, objective="BRAVO").index})
    rfn1 = env.roster.by_callsign["RFN1"]
    rfn1.pos = env.world.objective_by_name("BRAVO").pos  # stands on BRAVO, no enemies near
    _step_all(env, {"RFN1": DONE})
    assert rfn1.mission is None, "truthful DONE clears the standing order"
    *_, infos = _step_all(env, {"TL1": _order_spec(MissionType.SEIZE, objective="ALPHA").index})
    assert env.retask_events_last_step == []
    # fresh tasking: preferred derivation + objective match (TL1's OPORD is ALPHA)
    assert infos["TL1"]["components"]["command"] == pytest.approx(
        cfg.order_preferred + cfg.order_objective_match + cfg.transmission_cost + cfg.coverage_gap
    )


def test_identical_reissue_stays_a_churn_noop_and_keeps_tenure():
    """Reissuing the standing order is still radio noise: order_churn, no
    re-task event, and the mission is NOT restamped (tenure keeps accruing)."""
    env = _flat_env()
    cfg = env.rewards_cfg
    spec = _order_spec(MissionType.SEIZE)
    _step_all(env, {"TL1": spec.index})
    rfn1 = env.roster.by_callsign["RFN1"]
    assigned = rfn1.mission.step_assigned
    *_, infos = _step_all(env, {"TL1": spec.index})
    assert infos["TL1"]["components"]["command"] == pytest.approx(
        cfg.order_churn + cfg.transmission_cost + cfg.coverage_gap
    )
    assert env.retask_events_last_step == []
    assert rfn1.mission.step_assigned == assigned, "no restamp: tenure is preserved"


def test_alternating_orders_are_priced_not_farmable():
    """The old alternation exploit under the new pricing: every flip between
    two valid orders is a priced re-task — strictly negative command."""
    env = _flat_env()
    *_, infos = _step_all(env, {"TL1": _order_spec(MissionType.SEIZE).index})
    assert infos["TL1"]["components"]["command"] > 0
    *_, infos = _step_all(env, {"TL1": _order_spec(MissionType.CLEAR).index})
    assert infos["TL1"]["components"]["command"] < -0.2
    *_, infos = _step_all(env, {"TL1": _order_spec(MissionType.SEIZE).index})
    assert infos["TL1"]["components"]["command"] < -0.2


# ---------------------------------------------------------------------- #
# standing-order tenure
# ---------------------------------------------------------------------- #


def _hold_in_place(env, callsign="RFN1", step_assigned=0):
    sld = env.roster.by_callsign[callsign]
    sld.mission = Mission(
        MissionType.HOLD, None, sld.pos, issuer_id=-1, step_assigned=step_assigned
    )
    return sld


def test_tenure_grows_positive_compliance_and_caps_at_horizon():
    env = _flat_env()
    cfg = env.rewards_cfg
    _hold_in_place(env)  # HOLD in position, stationary: raw compliance 0.5
    seen = {}
    for _ in range(cfg.tenure_horizon + 5):
        *_, infos = _step_all(env)
        seen[env._step_count] = infos["RFN1"]["components"]["compliance"]

    def expected(t):
        held = min(t, cfg.tenure_horizon)
        return 0.5 * cfg.compliance_weight * (1 + cfg.tenure_factor * held / cfg.tenure_horizon)

    assert seen[1] == pytest.approx(expected(1))
    assert seen[20] == pytest.approx(expected(20))
    assert seen[20] > seen[1], "credit grows with tenure"
    assert seen[cfg.tenure_horizon + 5] == pytest.approx(
        0.5 * cfg.compliance_weight * (1 + cfg.tenure_factor)
    ), "capped at the horizon"


def test_tenure_resets_on_retask():
    env = _flat_env()
    cfg = env.rewards_cfg
    alpha = env.world.objective_by_name("ALPHA")
    bravo = env.world.objective_by_name("BRAVO")
    rfn1 = env.roster.by_callsign["RFN1"]
    for e in env.enemies:  # well clear of BRAVO too: nobody shoots the holder
        e.pos = (1, env.world.height - 2)
        e.home = e.pos
    rfn1.pos = bravo.pos  # on BRAVO, holding SEIZE BRAVO: in position (0.5)
    rfn1.mission = Mission(
        MissionType.SEIZE, bravo.id, bravo.pos, issuer_id=-1, step_assigned=0
    )
    infos = None
    for _ in range(cfg.tenure_horizon + 2):  # ride the clock past the horizon
        *_, infos = _step_all(env)
    assert infos["RFN1"]["components"]["compliance"] == pytest.approx(
        0.5 * cfg.compliance_weight * (1 + cfg.tenure_factor)
    ), "full tenure before the re-task"
    _step_all(env, {"TL1": _order_spec(MissionType.SEIZE, objective="ALPHA").index})
    t0 = env._step_count
    assert rfn1.mission.step_assigned == t0, "re-tasking restamps the mission"
    rfn1.pos = alpha.pos  # in position for the NEW order
    *_, infos = _step_all(env)
    assert infos["RFN1"]["components"]["compliance"] == pytest.approx(
        0.5 * cfg.compliance_weight * (1 + cfg.tenure_factor * 1 / cfg.tenure_horizon)
    ), "tenure restarted from the re-task step"


def test_tenure_never_amplifies_negative_compliance():
    env = _flat_env()
    cfg = env.rewards_cfg
    obj = env.world.objective_by_name("ALPHA")
    sld = env.roster.by_callsign["RFN1"]
    sld.pos = (10, obj.pos[1])
    sld.mission = Mission(MissionType.SEIZE, 0, obj.pos, issuer_id=-1, step_assigned=0)
    for _ in range(cfg.tenure_horizon + 2):
        _step_all(env)
    *_, infos = _step_all(env, {"RFN1": MOVE_WEST})  # walks AWAY at full tenure
    assert infos["RFN1"]["components"]["compliance"] == pytest.approx(
        cfg.compliance_weight * -0.5
    ), "negative compliance is never tenure-scaled"


# ---------------------------------------------------------------------- #
# knobs off / dominance
# ---------------------------------------------------------------------- #


def test_knobs_at_zero_disable_the_new_economics():
    cfg = RewardConfig(order_retask_cost_base=0.0, tenure_factor=0.0)
    env = _flat_env(reward_config=cfg)
    # unexcepted rotation: no price (only airtime + the standing coverage gap)
    _step_all(env, {"TL1": _order_spec(MissionType.SEIZE).index})
    *_, infos = _step_all(env, {"TL1": _order_spec(MissionType.SEIZE, objective="BRAVO").index})
    assert infos["TL1"]["components"]["command"] == pytest.approx(
        cfg.transmission_cost + cfg.coverage_gap
    )
    (event,) = env.retask_events_last_step  # the event is still measured
    assert event["excepted"] is False and event["cost"] == 0.0
    # tenure off: flat compliance credit at any tenure
    env2 = _flat_env(reward_config=cfg)
    _hold_in_place(env2)
    for _ in range(45):
        *_, infos = _step_all(env2)
    assert infos["RFN1"]["components"]["compliance"] == pytest.approx(
        0.5 * cfg.compliance_weight
    )


def test_terminal_dominance_margin_survives_the_tenure_ceiling():
    """success_team must beat a full-tenure perfect farm on EVERY scenario cap
    (the main dominance regression test iterates all scenarios; this one pins
    the B5 arithmetic explicitly)."""
    from cohort.config import SCENARIOS
    from cohort.core.missions import RECON_OBSERVE_STEPS

    cfg = RewardConfig()
    assert cfg.max_step_farm() == pytest.approx(
        cfg.compliance_weight * 0.6 * (1 + cfg.tenure_factor)
        + cfg.coverage_bonus
        + cfg.time_penalty
    )
    longest = max(spec.max_steps for spec in SCENARIOS.values())
    observe_cap = cfg.observe_progress * 2 * RECON_OBSERVE_STEPS
    assert cfg.success_team > cfg.max_step_farm() * longest + observe_cap


# ---------------------------------------------------------------------- #
# metrics plumbing
# ---------------------------------------------------------------------- #


def test_retask_events_land_in_the_behavior_metrics():
    from cohort.metrics import TraceRecorder, aggregate_behavior, episode_behavior

    env = _flat_env()
    recorder = TraceRecorder()
    recorder.on_reset(env)

    def step(overrides=None):
        recorder.before_step(env)
        acts = {a: STAY for a in env.agents}
        acts.update(overrides or {})
        env.step(acts)
        recorder.after_step(env)

    step({"TL1": _order_spec(MissionType.SEIZE).index})                       # fresh
    step({"TL1": _order_spec(MissionType.CLEAR).index})                       # priced, same anchor
    step({"TL1": _order_spec(MissionType.SEIZE, objective="BRAVO").index})    # priced rotation
    ep = episode_behavior(recorder.trace)
    assert ep["orders_issued"] == 3
    assert ep["retasks"] == 2
    assert ep["retasks_priced"] == 2
    assert ep["retasks_excepted"] == 0
    assert ep["retask_rotations"] == 1
    assert ep["retasks_by_rank"] == {"TL": {"priced": 2, "excepted": 0}}
    agg = aggregate_behavior([ep])
    assert agg["retasks_per_episode"] == pytest.approx(2.0)
    assert agg["retasks_priced_per_episode"] == pytest.approx(2.0)
    assert agg["orders_per_episode"] == pytest.approx(3.0)
    assert agg["retasks_by_rank"] == {"TL": {"priced": 2, "excepted": 0}}
