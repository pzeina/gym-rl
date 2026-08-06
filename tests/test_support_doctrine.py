"""SUPPORT (APPUYER) is overwatch of a MOVING element — "pas un pas sans appui".

It was scored as a static posture sharing OBSERVE's branch outright. Two
consequences, both of which inverted the doctrine:

1. **``stationary`` was the wrong predicate.** SUPPORT's anchor is the supported
   *soldier*, so a supporter displacing to hold range and LOS on a bounding
   element broke ``stationary`` and collected 0.1, while a supporter that let
   its element walk away and stood still collected 0.6. Measured on the squad
   map over six steps of a bounding element: **following paid 0.60 total,
   abandoning paid 3.60** — a 6x premium for not supporting. OBSERVE offered
   that same 3.60 for watching a fixed point that can never outrun you, which
   is why OBSERVE was ordered 3.6-10x more than SUPPORT in every corpus that
   used either (defend_v8 0.098 vs 0.010; platoon_v4 0.057 vs 0.016).

2. **The station was wider than the umbrella.** ``IN_POSITION_RADIUS[SUPPORT]``
   was 10.0 while ``CombatConfig.support_umbrella`` is 8.0, so a supporter
   could draw full posture pay from 9-10 cells while ``_covered_by_support``
   protected nothing — the reward describing support the environment never
   delivered.

Both are pinned here. The third property matters just as much: movement is
excused *only* when the element itself moved, so this pays for keeping station
and never for aimless drift.
"""

import pytest

from cohort import make_env
from cohort.core.missions import (
    IN_POSITION_RADIUS,
    Mission,
    MissionType,
    compliance,
)

FULL = 0.6
LOITER = 0.1


@pytest.fixture
def field():
    """Flat ground, no OpFor: LOS holds everywhere so only the rules bite."""
    env = make_env("squad")
    env.reset(seed=3)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.alive = False
    living = env.roster.living
    supporter, supported = living[0], living[1]
    supporter.mission = Mission(
        MissionType.SUPPORT, None, (0, 0), issuer_id=-1,
        step_assigned=0, extra={"supported_id": supported.id},
    )
    return env, supporter, supported


def _place(soldier, pos):
    soldier.pos = soldier.prev_pos = pos


def _move(soldier, pos):
    soldier.prev_pos = soldier.pos
    soldier.pos = pos


def _comp(env, soldier, prev=None):
    return compliance(
        soldier.mission.type,
        env._compliance_ctx(soldier, prev, env._make_view(soldier)),
    )


def test_supporter_keeping_station_with_a_bounding_element_earns_full_pay(field):
    """The regression: following your element WAS worth 0.1 against 0.6."""
    env, supporter, supported = field
    _place(supporter, (20, 20))
    _place(supported, (22, 20))
    for _ in range(5):
        _move(supported, (supported.pos[0] + 1, supported.pos[1]))
        _move(supporter, (supporter.pos[0] + 1, supporter.pos[1]))
        assert _comp(env, supporter) == pytest.approx(FULL), (
            "displacing to hold station on a bounding element is execution, "
            "not drift — it must not be paid as loiter"
        )


def test_supporter_drifting_while_its_element_holds_is_not_paid_full(field):
    """Movement is excused only when the ELEMENT moved — no aimless drift."""
    env, supporter, supported = field
    _place(supporter, (20, 20))
    _place(supported, (22, 20))
    for i in range(4):
        supported.prev_pos = supported.pos          # element holds
        _move(supporter, (supporter.pos[0] + (1 if i % 2 == 0 else -1), supporter.pos[1]))
        assert _comp(env, supporter) == pytest.approx(LOITER)


def test_static_overwatch_of_a_holding_element_still_earns_full_pay(field):
    """The unchanged case: both stationary, in position."""
    env, supporter, supported = field
    _place(supporter, (20, 20))
    _place(supported, (22, 20))
    supported.prev_pos = supported.pos
    assert _comp(env, supporter) == pytest.approx(FULL)


def test_support_station_never_exceeds_the_umbrella_that_makes_it_real(field):
    """Full pay must not be earnable from outside the protective umbrella."""
    env, supporter, supported = field
    umbrella = float(env.combat.support_umbrella)
    assert IN_POSITION_RADIUS[MissionType.SUPPORT] >= umbrella, "fixture assumption"

    _place(supported, (20, 20))
    # just outside the umbrella: covers nothing, so must not read in position
    _place(supporter, (20 + int(umbrella) + 1, 20))
    assert not env._in_mission_position(supporter), (
        "a supporter outside combat.support_umbrella protects nobody and must "
        "not draw full posture pay"
    )
    # just inside it
    _place(supporter, (20 + int(umbrella) - 1, 20))
    assert env._in_mission_position(supporter)


def test_observe_is_untouched_and_still_requires_settling(field):
    """OBSERVE's anchor cannot move, so it keeps the stationary requirement."""
    env, supporter, _ = field
    obj = env.world.objectives[0]
    supporter.mission = Mission(
        MissionType.OBSERVE, obj.id, obj.pos, issuer_id=-1, step_assigned=0,
    )
    _place(supporter, (obj.pos[0] + 2, obj.pos[1]))
    assert _comp(env, supporter) == pytest.approx(FULL)
    _move(supporter, (obj.pos[0] + 3, obj.pos[1]))
    assert _comp(env, supporter) == pytest.approx(LOITER)


def test_anchor_moved_is_false_for_fixed_anchor_missions(field):
    """Missions anchored to terrain must be entirely unaffected."""
    env, supporter, _ = field
    obj = env.world.objectives[0]
    for mission_type in (MissionType.DEFEND, MissionType.OBSERVE, MissionType.SEIZE):
        supporter.mission = Mission(
            mission_type, obj.id, obj.pos, issuer_id=-1, step_assigned=0,
        )
        assert env._anchor_moved(supporter) is False


def test_dead_supported_unit_does_not_excuse_movement(field):
    env, supporter, supported = field
    supported.alive = False
    _place(supporter, (20, 20))
    assert env._anchor_moved(supporter) is False
