"""Doctrine derivation table and MICAT mission semantics."""

from cohort.core.missions import (
    COMPLETABLE,
    DOCTRINE,
    NEEDS_OBJECTIVE,
    UNIT_TARGETED,
    ComplianceContext,
    MissionType,
    allowed_derivations,
    compliance,
    derivation_quality,
    min_hold_authority,
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


def test_mission_set_is_the_micat_catalog():
    """The enum order defines obs one-hot + catalog layout — pinned.

    ADVANCE (A5) is appended AFTER the MICAT set so the earlier one-hot
    indices are stable.
    """
    assert [m.name for m in MissionType] == [
        "RECON", "SCREEN", "OBSERVE", "SUPPORT", "COVER",
        "DEFEND", "DENY", "SEIZE", "CLEAR", "RALLY", "HOLD", "ADVANCE",
    ]


def test_every_mission_has_doctrine():
    for mission in MissionType:
        allowed = DOCTRINE[mission]
        assert allowed, f"{mission} has no derivations"
        if mission is MissionType.DENY:
            # DENY is section-level: no group can hold it (manual p. 8), so a
            # section on INTERDIRE tasks its groups with DEFEND first
            assert allowed[0] is MissionType.DEFEND
            assert MissionType.DENY not in allowed
        else:
            assert mission is allowed[0], "a mission's own type should be its preferred derivation"


def test_deny_is_derivable_from_nothing():
    """No leader can pass DENY down: it enters only via HQ (OPORD/injection)."""
    for mission in MissionType:
        assert MissionType.DENY not in DOCTRINE[mission]


def test_per_echelon_admissibility():
    assert min_hold_authority(MissionType.DENY) == 2, "DENY: section level and above"
    for mission in MissionType:
        if mission is not MissionType.DENY:
            assert min_hold_authority(mission) == 0


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


def test_recon_may_engage_screen_may_not():
    """PROTERRE: RECONNAÎTRE may engage (p. 30); ÉCLAIRER may not (p. 32)."""
    assert compliance(MissionType.RECON, _ctx(fired=True, in_position=True)) > 0
    assert compliance(MissionType.SCREEN, _ctx(fired=True, in_position=True)) < 0
    assert compliance(MissionType.SCREEN, _ctx(in_position=True)) > 0


def test_static_postures_reward_being_static():
    for mission in (MissionType.OBSERVE, MissionType.SUPPORT, MissionType.COVER, MissionType.HOLD):
        static = compliance(mission, _ctx(in_position=True, stationary=True))
        moving = compliance(mission, _ctx(in_position=True, stationary=False))
        assert static > moving, f"{mission}: static in position must outscore shuffling"


def test_engage_rewards_firing():
    assert compliance(MissionType.CLEAR, _ctx(fired=True)) > compliance(
        MissionType.CLEAR, _ctx(visible_enemies=2)
    )


def test_no_mission_no_compliance():
    assert compliance(None, _ctx()) == 0.0


def test_continuous_postures_are_not_completable():
    for mission in (
        MissionType.OBSERVE, MissionType.SUPPORT, MissionType.COVER,
        MissionType.DEFEND, MissionType.DENY, MissionType.HOLD,
    ):
        assert mission not in COMPLETABLE, f"{mission} is a continuous posture"
    for mission in (
        MissionType.RECON, MissionType.SCREEN, MissionType.SEIZE,
        MissionType.CLEAR, MissionType.RALLY,
    ):
        assert mission in COMPLETABLE, f"{mission} has an end state"


def test_needs_objective_partition():
    assert MissionType.RALLY not in NEEDS_OBJECTIVE
    assert MissionType.HOLD not in NEEDS_OBJECTIVE
    assert MissionType.SUPPORT not in NEEDS_OBJECTIVE, "SUPPORT targets a unit"
    assert MissionType.SUPPORT in UNIT_TARGETED
    for mission in (
        MissionType.RECON, MissionType.SCREEN, MissionType.OBSERVE, MissionType.COVER,
        MissionType.DEFEND, MissionType.DENY, MissionType.SEIZE, MissionType.CLEAR,
    ):
        assert mission in NEEDS_OBJECTIVE
