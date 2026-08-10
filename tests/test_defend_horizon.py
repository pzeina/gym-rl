"""DEFEND success is conservation of the position, to a stated hour (v1.14).

The owner's decision, in one sentence: a defense succeeds by still being on
the ground when the hour comes, not by killing everyone. Written out —

    occupied(t)  a living friendly within root_obj.radius + 1 of the objective
    FAIL         permanently, at the first t >= H with occupied(t) false
    SUCCESS      at the first t >= H with the threat out of the fight
                 (``_band_neutralized`` — early release) or t >= the horizon

Three of those clauses are decisions rather than derivations, and each one has
a measurement or an argument behind it that a future edit should have to
answer:

* **Occupation, not safety.** The criterion is the ``manned`` half of
  ``_objective_held``, never the ``clear`` half. An enemy assaulting into
  contact on the position is the mission arriving, not the mission failing;
  scoring the strict conjunction instead costs 29 of 100 episodes on the
  committed checkpoints, and 26 of the 40 first breaks it counts are exactly
  "the assault got here".
* **No retake.** Once the position is not occupied the mission has failed and
  stays failed. A position handed over and walked back onto was not held.
* **A fixed step count, not H + D.** ``PolicyNet`` is a memoryless MLP whose
  only clock is the ``step / max_steps`` tempo feature, so an H-relative
  deadline is not merely hard to learn — it is unperceivable.

Scenarios with no ``defend_horizon`` are the pre-v1.14 object and keep the
pre-v1.14 criterion exactly; that is pinned here too, because "we only changed
the defend scenarios" is the kind of claim that quietly stops being true.

v1.18 (refs #30) puts the hour on the net — ``HOLD UNTIL STEP 210`` in the
OPORD — so the cohort is no longer adjudicated against an hour HQ never spoke.
The last section here pins that the spoken clause and the scored horizon are
the same number, gated by the same predicate.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from cohort import get_scenario, make_env
from cohort.core.missions import COMPLETABLE, MissionType, is_completable

STAY = 0

DEFEND_SCENARIOS = ("fireteam_defend", "defend_brique")


def _env(scenario="defend_brique", seed=3, *, horizon=...):
    spec = get_scenario(scenario)
    if horizon is not ...:
        spec = replace(spec, defend_horizon=horizon)
    env = make_env(spec)
    env.reset(seed=seed)
    return env


def _step(env):
    return env.step({a: STAY for a in env.agents})


def _root_obj(env):
    return env.world.objective_by_name(env.spec_cfg.root_objective)


def _man(env, obj):
    """Put the whole living force on the objective."""
    for s in env.roster.living:
        s.pos = obj.pos


def _abandon(env, obj):
    """Walk the whole living force off the objective footprint."""
    far = (obj.pos[0] + int(obj.radius) + 6, obj.pos[1])
    for s in env.roster.living:
        s.pos = far


# ---------------------------------------------------------------------- #
# the scenarios carry the horizon the owner set
# ---------------------------------------------------------------------- #

def test_the_defend_scenarios_are_ordered_to_half_the_step_budget():
    for name in DEFEND_SCENARIOS:
        spec = get_scenario(name)
        assert spec.defend_horizon == int(0.5 * spec.max_steps), name


def test_every_other_scenario_is_an_indefinite_posture():
    from cohort.config import SCENARIOS

    for name, spec in SCENARIOS.items():
        if name in DEFEND_SCENARIOS:
            continue
        assert spec.defend_horizon is None, name


# ---------------------------------------------------------------------- #
# occupation, and the permanence of losing it
# ---------------------------------------------------------------------- #

def test_occupation_is_the_manned_clause_not_the_whole_of_objective_held():
    """An enemy on the position does not take it from a force standing on it."""
    env = _env()
    obj = _root_obj(env)
    _man(env, obj)
    env.enemies[0].alive = True
    env.enemies[0].pos = obj.pos

    assert env._position_occupied(obj) is True
    assert env._objective_held(obj) is False, "the strict clause must still differ"


def test_an_assault_arriving_on_the_position_does_not_fail_the_mission():
    env = _env()
    env._h_hour = 0
    obj = _root_obj(env)
    _man(env, obj)
    for e in env.enemies:
        e.alive = True
        e.pos = obj.pos
    _step(env)
    assert env._defend_lost_step is None, "repelling an assault is the mission"


def test_one_unoccupied_step_after_h_fails_the_mission_permanently():
    env = _env()
    env._h_hour = 1
    obj = _root_obj(env)
    _abandon(env, obj)
    _step(env)
    assert env._defend_lost_step is not None, "the position was not occupied at H"

    # no retake: walking back onto the ground does not un-fail it, and the
    # early release cannot rescue it either
    lost_at = env._defend_lost_step
    _man(env, obj)
    for e in env.enemies:
        e.alive = False
    _step(env)
    assert env._defend_lost_step == lost_at
    assert env._check_success(obj) is False
    assert env.outcome != "success"


def test_ground_given_up_after_the_operation_is_won_fails_nothing():
    """The latch stops at T0: the grace window is aftermath, not the mission.

    It cannot change a verdict — success locks at T0 either way — but it
    decides whether ``_defend_lost_step`` means what its name says. Measured
    on defend_brique_v9 before the guard: 17 latched losses against 12 lost
    episodes, the 5 extra all after an early release had already won.
    """
    env = _env()
    env._h_hour = 1
    obj = _root_obj(env)
    _man(env, obj)
    for e in env.enemies:
        e.alive = False
    _step(env)
    assert env._success_step is not None

    _abandon(env, obj)
    _step(env)
    assert env._defend_lost_step is None
    assert env.outcome != "defeat"


def test_occupation_is_not_required_before_h():
    """The preparation period exists so the ground can be occupied at all."""
    env = _env()
    env._h_hour = 40
    obj = _root_obj(env)
    _abandon(env, obj)
    for _ in range(5):
        _step(env)
    assert env._step_count < env._h_hour
    assert env._defend_lost_step is None


def test_a_wiped_force_cannot_hold_and_so_never_succeeds():
    env = _env()
    env._h_hour = 1
    for s in env.roster.soldiers:
        s.alive = False
        s.health = 0
    for e in env.enemies:
        e.alive = False           # early release would otherwise fire
    _step(env)
    assert env._defend_lost_step is not None
    assert env.outcome == "defeat"


# ---------------------------------------------------------------------- #
# the two routes to success
# ---------------------------------------------------------------------- #

def test_early_release_fires_when_the_band_is_out_of_the_fight():
    env = _env()
    env._h_hour = 1
    obj = _root_obj(env)
    _man(env, obj)
    for e in env.enemies:
        e.alive = False
    _step(env)
    assert env._band_neutralized(obj) is True
    assert env._success_step is not None
    assert env._success_step < env.spec_cfg.defend_horizon, "released early, not by clock"


def test_the_horizon_fires_with_the_band_still_alive():
    """The backstop: held to the ordered hour, threat or no threat."""
    env = _env()
    env._h_hour = 1
    obj = _root_obj(env)
    horizon = env.spec_cfg.defend_horizon
    env._step_count = horizon - 1
    _man(env, obj)
    for e in env.enemies:                     # alive, so no early release
        e.alive = True
        e.pos = (obj.pos[0] + 8, obj.pos[1])
    _step(env)
    assert any(e.alive for e in env.enemies)
    assert env._band_neutralized(obj) is False
    assert env._success_step == horizon


def test_success_is_denied_before_h_even_with_the_field_clear():
    """t >= H is part of the criterion: the operation has not started yet."""
    env = _env()
    env._h_hour = 30
    obj = _root_obj(env)
    _man(env, obj)
    for e in env.enemies:
        e.alive = False
    _step(env)
    assert env._success_step is None
    assert env._check_success(obj) is False


def test_a_failed_defense_runs_out_the_clock_rather_than_ending_early():
    """v1.14 changes the criterion, deliberately not the termination rule.

    Changing when episodes end would be a second variable in the retrain that
    follows, and the two effects would not be separable afterwards.
    """
    spec = replace(get_scenario("defend_brique"), max_steps=30, defend_horizon=15)
    env = make_env(spec)
    env.reset(seed=5)
    env._h_hour = 1
    obj = _root_obj(env)
    _abandon(env, obj)
    steps = 0
    while env.agents and steps < 60:
        _step(env)
        steps += 1
    assert env._defend_lost_step is not None
    assert env.outcome == "timeout"
    assert steps == spec.max_steps


# ---------------------------------------------------------------------- #
# the indefinite posture is untouched
# ---------------------------------------------------------------------- #

def test_a_scenario_without_a_horizon_keeps_the_old_criterion_exactly():
    """The BRIQUE conjunction, unchanged: band neutralised AND objective held."""
    env = _env(horizon=None)
    obj = _root_obj(env)
    assert env._horizon_defense() is None

    # the old criterion is the strict one, and it is what still decides
    _man(env, obj)
    env.enemies[0].alive = True
    env.enemies[0].pos = obj.pos
    for e in env.enemies[1:]:
        e.alive = False
    assert env._check_success(obj) is False, "an enemy on the objective still blocks it"

    env.enemies[0].alive = False
    assert env._check_success(obj) is True

    # and nothing latches, at any point in the episode
    _abandon(env, obj)
    _step(env)
    assert env._defend_lost_step is None


def test_a_seize_root_is_not_touched_even_if_a_horizon_is_set():
    """The horizon is a DEFEND/DENY clause; other roots ignore it entirely."""
    spec = replace(get_scenario("fireteam"), defend_horizon=10)
    env = make_env(spec)
    env.reset(seed=2)
    assert env._horizon_defense() is None
    # ...including on the net: HQ does not order a hold it will not score
    assert "HOLD UNTIL" not in env.transcript.messages[0].text
    env._h_hour = 0
    obj = _root_obj(env)
    _abandon(env, obj)
    _step(env)
    assert env._defend_lost_step is None


# ---------------------------------------------------------------------- #
# the ordered hour is SAID (v1.18, refs #30)
#
# Until v1.18 the horizon was the success criterion and was nowhere on the
# net: the cohort was adjudicated against an hour HQ never spoke, and a
# monitor holding only the transcript could not state the standard the
# episode was being judged by. The clause closes that. It is transcript
# completeness, not capability — PolicyNet is a memoryless MLP whose only
# clock is step/max_steps, so it cannot use a spoken deadline; the
# beneficiary is the human commander and the transcript-only monitor.
# ---------------------------------------------------------------------- #

def test_the_opord_orders_the_hour_it_will_be_judged_by():
    from cohort.core import language as lang

    for name in DEFEND_SCENARIOS:
        env = _env(name)
        opord = env.transcript.messages[0]
        horizon = env.spec_cfg.defend_horizon
        assert f"HOLD UNTIL STEP {horizon}." in opord.text, name
        # what is said is what is scored
        assert lang.parse_opord(opord.text)["defend_horizon"] == env._horizon_defense(), name
        # and it agrees with the header route (issue #30's first half)
        assert env.briefing()["defend_horizon"] == horizon, name


def test_the_horizon_clause_is_an_order_and_the_assault_clause_an_estimate():
    """Different moods, on purpose: HOLD UNTIL tasks, EXPECT hedges.

    Both are absolute step references — the v1.18 half of the change that is
    not the new clause at all. "AT H PLUS 65" read as *65 steps after H-hour*
    while always meaning step 65, which was survivable only while it was the
    one time-bearing clause on the line.
    """
    env = _env("defend_brique")
    text = env.transcript.messages[0].text
    assert "EXPECT ASSAULT AT STEP 45." in text
    assert "HOLD UNTIL STEP 210." in text
    assert "EXPECT" not in text.split("HOLD UNTIL")[1], "the order must not hedge"
    assert "H PLUS" not in text


def test_an_indefinite_defend_orders_no_hour():
    """No horizon, no clause: HQ cannot name an hour the spec does not set."""
    env = _env("fireteam_defend", horizon=None)
    text = env.transcript.messages[0].text
    assert env._horizon_defense() is None
    assert "HOLD UNTIL" not in text
    assert "EXPECT ASSAULT AT STEP 65." in text, "the other clause is untouched"


def test_the_clauses_do_not_change_the_tasking():
    """`parse_order` reads the task; neither time clause perturbs it."""
    from cohort.core import language as lang

    for name in DEFEND_SCENARIOS:
        env = _env(name)
        parsed = lang.parse_order(env.transcript.messages[0].text)
        assert parsed.mission is MissionType.DEFEND, name
        assert parsed.objective_name == env.spec_cfg.root_objective, name
        assert parsed.delay is None and parsed.at_my_command is False, name


def test_the_horizon_round_trips_through_the_opord_line():
    """Formatter and parser stay inverses over the new clause too."""
    from cohort.core import language as lang

    for horizon in (0, 12, 210, 4321):
        text = lang.format_opord("TL1", MissionType.DEFEND, "ALPHA", 45, horizon)
        assert lang.parse_opord(text) == {
            "recipient": "TL1",
            "mission": "DEFEND",
            "objective": "ALPHA",
            "announced_assault_step": 45,
            "defend_horizon": horizon,
        }
    # a DENY root holds ground too, and is ordered to an hour the same way
    deny = lang.format_opord("SL1", MissionType.DENY, "BRAVO", None, 99)
    assert lang.parse_opord(deny)["defend_horizon"] == 99
    assert lang.parse_opord(deny)["announced_assault_step"] is None


def test_a_mission_that_does_not_hold_ground_is_never_ordered_to_an_hour():
    """The clause is gated on HOLDS_GROUND, not on the caller's discipline."""
    from cohort.core import language as lang
    from cohort.core.missions import HOLDS_GROUND

    for mission in (MissionType.SEIZE, MissionType.RECON, MissionType.SCREEN):
        assert mission not in HOLDS_GROUND
        text = lang.format_opord("SL1", mission, "ALPHA", None, 210)
        assert "HOLD UNTIL" not in text, mission.name
        assert lang.parse_opord(text)["defend_horizon"] is None, mission.name


# ---------------------------------------------------------------------- #
# COMPLETABLE, refined and then un-refined
# ---------------------------------------------------------------------- #

def test_the_horizon_adjudicates_success_and_grants_no_permission():
    """v1.17 (owner's decision) withdraws v1.14's second clause.

    v1.14 gave the horizon two jobs — the success criterion, and opening the
    root's MISSION COMPLETE bit. The second was measured over three cycles and
    bought nothing (early close pays no speed bonus and is capped at
    ``grace_window``; ``defend_brique`` claimed at 0.71 false; three prices
    moved volume without moving informedness), so it is gone and
    ``is_completable`` no longer takes a horizon at all. The FIRST job is
    untouched, which is what the rest of this file measures.
    """
    assert MissionType.DEFEND not in COMPLETABLE
    assert is_completable(MissionType.DEFEND) is False
    assert is_completable(MissionType.DENY) is False
    assert is_completable(MissionType.SEIZE) is True
    assert is_completable(MissionType.HOLD) is False
    assert is_completable(None) is False
    # the knob is removed rather than left doing nothing: a caller that still
    # passes a horizon gets an error, not a silently ignored argument
    with pytest.raises(TypeError):
        is_completable(MissionType.DEFEND, defend_horizon=210)
