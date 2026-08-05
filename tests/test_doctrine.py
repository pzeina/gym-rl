"""Doctrine derivation table and mission semantics."""

from cohort.core.missions import (
    COMPLETABLE,
    DOCTRINE,
    NEEDS_OBJECTIVE,
    ComplianceContext,
    MissionType,
    allowed_derivations,
    compliance,
    derivation_quality,
)


def _ctx(**kw):
    base = {
        "dist_prev": 10.0,
        "dist_now": 10.0,
        "in_position": False,
        "stationary": True,
        "fired": False,
        "visible_enemies": 0,
        "enemies_at_objective": 0,
        "dist_to_leader": 5.0,
    }
    base.update(kw)
    return ComplianceContext(**base)


def test_every_mission_has_doctrine():
    for mission in MissionType:
        allowed = DOCTRINE[mission]
        assert allowed, f"{mission} has no derivations"
        assert mission is allowed[0], "a mission's own type should be its preferred derivation"


def test_no_derivation_without_a_mission():
    assert allowed_derivations(None) == ()
    assert derivation_quality(None, MissionType.SEIZE) == 0.0


def test_derivation_quality_scores():
    assert derivation_quality(MissionType.SEIZE, MissionType.SEIZE) == 1.0
    assert derivation_quality(MissionType.SEIZE, MissionType.CLEAR) == 0.5
    assert derivation_quality(MissionType.SEIZE, MissionType.RALLY) == -0.5


def test_compliance_progress_sign():
    """Moving toward the anchor pays; moving away costs — for every mission."""
    for mission in MissionType:
        toward = compliance(mission, _ctx(dist_prev=10.0, dist_now=9.0, stationary=False))
        away = compliance(mission, _ctx(dist_prev=10.0, dist_now=11.0, stationary=False))
        assert toward > 0, f"{mission}: approaching the anchor should score > 0"
        assert away < 0, f"{mission}: leaving the anchor should score < 0"


def test_recon_is_stealthy():
    assert compliance(MissionType.RECON, _ctx(fired=True, in_position=True)) < 0
    assert compliance(MissionType.RECON, _ctx(in_position=True)) > 0


def test_overwatch_and_hold_reward_being_static():
    static = compliance(MissionType.OVERWATCH, _ctx(in_position=True, stationary=True))
    moving = compliance(MissionType.OVERWATCH, _ctx(in_position=True, stationary=False))
    assert static > moving
    static_h = compliance(MissionType.HOLD, _ctx(in_position=True, stationary=True))
    moving_h = compliance(MissionType.HOLD, _ctx(in_position=True, stationary=False))
    assert static_h > moving_h


def test_engage_rewards_firing():
    assert compliance(MissionType.CLEAR, _ctx(fired=True)) > compliance(
        MissionType.CLEAR, _ctx(visible_enemies=2)
    )


def test_no_mission_no_compliance():
    assert compliance(None, _ctx()) == 0.0


def test_continuous_postures_are_not_completable():
    assert MissionType.DEFEND not in COMPLETABLE
    assert MissionType.OVERWATCH not in COMPLETABLE
    assert MissionType.HOLD not in COMPLETABLE
    assert MissionType.SEIZE in COMPLETABLE


def test_needs_objective_partition():
    assert MissionType.RALLY not in NEEDS_OBJECTIVE
    assert MissionType.HOLD not in NEEDS_OBJECTIVE
    assert MissionType.SEIZE in NEEDS_OBJECTIVE
