"""The completion-report grace window: MISSION COMPLETE is load-bearing.

When the root-mission success condition is first met (T0) the episode stays
open for ``ScenarioSpec.grace_window`` steps so the root can transmit its
COMPLETE report. A truthful root DONE ends the episode that step and pays
``root_done_bonus``; policies that never report still succeed at
T0 + grace_window with the identical terminal reward (speed from T0).
"""

import pytest

from cohort import make_env
from cohort.core.orders import HQ_ID
from cohort.env.actions import CATALOG

STAY = 0
DONE = next(s.index for s in CATALOG if s.kind == "done")


def _step_all(env, overrides):
    acts = {a: STAY for a in env.agents}
    acts.update(overrides)
    return env.step(acts)


def _cleared_env(seed=1):
    """Fireteam env with the success condition one step away: enemies dead,
    the root (TL1, SEIZE OBJ ALPHA) standing on the objective."""
    env = make_env("fireteam")
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.alive = False
    env.roster.by_callsign["TL1"].pos = env.world.objectives[0].pos
    return env


def test_the_fallen_are_paid_the_team_terminal():
    """A soldier who dies before the win still shares in it (v1.11).

    THE regression test for the collapse the v1.11 bisect diagnosed. The
    terminal used to be paid to ``roster.living``, so a casualty forfeited all
    60 points, and the per-agent arithmetic on a 9-agent squad was:

        hanging back cuts P(die) 0.129 -> 0.022, worth +6.4 to that agent
        ...but ONE shared policy updates EVERY agent at once, so team success
        goes 1.00 -> 0.00, worth -52.3 to that agent

    A per-agent advantage only ever sees the first number. Three arms of the
    squad_screen bisect collapsed into the resulting basin and none escaped it
    in 350k+ further steps; the oracle measured the policy there at 19.96 cells
    from the objective against 10.39 before, and 0.20 friendly deaths per
    episode against 1.12. Dying must not forfeit the win.
    """
    from cohort.env.rewards import RewardConfig

    env = _cleared_env()
    dead = env.roster.by_callsign["RFN2"]
    dead.health = 0
    dead.alive = False
    _step_all(env, {})  # T0: the condition is met, the grace window opens
    assert "RFN2" in env.agents, "the fallen stay in the episode"
    _obs, _rewards, terms, _tr, infos = _step_all(env, {"TL1": DONE})
    assert env.outcome == "success"

    fallen = infos["RFN2"]["components"]
    survivor = infos["RFN3"]["components"]
    assert survivor["terminal"] >= RewardConfig().success_team
    assert fallen["terminal"] == pytest.approx(survivor["terminal"]), (
        "a casualty is paid the same team terminal as a survivor"
    )
    assert fallen["time"] == 0.0, "and accrues nothing per step while it waits"
    assert all(terms.values()), "success still terminates everyone, fallen included"


def test_root_report_ends_the_episode_that_step():
    env = _cleared_env()
    _step_all(env, {})  # T0 = 1: condition met, window opens
    assert env.outcome is None
    _obs, _rewards, terms, *_ , infos = _step_all(env, {"TL1": DONE})
    assert env.outcome == "success"
    assert all(terms.values())
    # the transcript ends with the root's COMPLETE answered by HQ, and then HQ
    # closing the operation — the claim is the REPORT, the ENDEX is the FACT,
    # and since v1.19 the FACT is transmitted on every root, not only a defence
    last_three = env.transcript.messages[-3:]
    assert [m.kind.value for m in last_three] == ["done", "done_confirm", "endex"]
    assert "— COMPLETE" in last_three[0].text
    assert last_three[1].sender_id == HQ_ID
    assert "CONFIRMED" in last_three[1].text
    assert last_three[2].sender_id == HQ_ID
    assert "ENDEX" in last_three[2].text
    # the reporter earns the root_done_bonus on top of the shared terminal
    assert infos["TL1"]["components"]["terminal"] == pytest.approx(
        infos["RFN1"]["components"]["terminal"] + env.rewards_cfg.root_done_bonus
    )


def test_unreported_success_ends_at_window_close():
    env = _cleared_env()
    steps = 0
    while env.agents:
        _step_all(env, {})
        steps += 1
        assert steps < 100, "episode must terminate"
    assert env.outcome == "success"
    assert steps == 1 + env.spec_cfg.grace_window, "T0=1, closes at T0 + grace_window"
    kinds = [m.kind.value for m in env.transcript.messages]
    assert "done" not in kinds, "nobody reported; success comes from the state check"


def test_speed_bonus_anchored_at_condition_step():
    """Reporting vs not reporting: identical shared terminal reward (from T0)."""
    env_a = _cleared_env()
    _step_all(env_a, {})
    *_, infos_a = _step_all(env_a, {"TL1": DONE})

    env_b = _cleared_env()
    infos_b = None
    while env_b.agents:
        *_, infos_b = _step_all(env_b, {})
    assert infos_a["RFN1"]["components"]["terminal"] == pytest.approx(
        infos_b["RFN1"]["components"]["terminal"]
    ), "old checkpoints (which never report) keep the exact terminal payout"


def test_root_reports_the_operation_not_its_own_position():
    """The root's OPORD claim is judged against the *team* success condition:
    a rifleman holds the objective, the commander reports from a distance."""
    env = make_env("fireteam")
    env.reset(seed=1)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.alive = False
    env.roster.by_callsign["RFN1"].pos = env.world.objectives[0].pos  # RFN holds ALPHA
    tl = env.roster.by_callsign["TL1"]
    tl.pos = (3, 3)  # commander far away
    _step_all(env, {})  # T0: condition met (clear + occupied by RFN1)
    assert env.outcome is None
    _obs, _r, terms, *_ = _step_all(env, {"TL1": DONE})
    assert env.outcome == "success"
    assert all(terms.values())
    assert [m.kind.value for m in env.transcript.messages[-2:]] == ["done_confirm", "endex"]


def test_false_root_claim_does_not_end_the_episode():
    env = make_env("fireteam")
    env.reset(seed=1)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.alive = False
    # condition NOT met: nobody stands on ALPHA (clear but not occupied)
    for s in env.roster.soldiers:
        s.pos = (3, 3)
    _obs, _r, terms, *_ = _step_all(env, {"TL1": DONE})
    assert not any(terms.values())
    assert env.outcome is None
    kinds = [m.kind.value for m in env.transcript.messages]
    assert "done_reject" in kinds, "the false claim is rejected on the net"


# --------------------------------------------------------------------- #
# Team adjudication of root RECON / SCREEN (refs #9): the commander's
# OPORD observation task completes on the squad's AGGREGATED observation,
# so the (human) root can command from cover instead of exposing itself.
# --------------------------------------------------------------------- #


def _recon_env(seed=1):
    """squad_recon env, flat ground, garrison removed: SL1 (root, human)
    holds RECON OBJ BRAVO from HQ; nobody starts on the observation ring."""
    env = make_env("squad_recon")
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.alive = False
    return env


def test_root_recon_opord_is_flagged_team_observation():
    env = _recon_env()
    assert env.roster.root().mission.team_observation is True
    env_screen = make_env("squad_screen")
    env_screen.reset(seed=1)
    assert env_screen.roster.root().mission.team_observation is True
    env_seize = make_env("fireteam")
    env_seize.reset(seed=1)
    assert env_seize.roster.root().mission.team_observation is False


def test_is_complete_team_threshold_vs_personal():
    from cohort.core.missions import (
        RECON_OBSERVE_STEPS,
        TEAM_OBSERVE_STEPS,
        ComplianceContext,
        Mission,
        MissionType,
        is_complete,
    )

    ctx = ComplianceContext(
        dist_prev=20.0, dist_now=20.0, in_position=False, stationary=True,
        fired=False, visible_enemies=0, enemies_at_objective=0, dist_to_leader=1.0,
    )
    personal = Mission(MissionType.RECON, 1, (0, 0), issuer_id=-1, step_assigned=0)
    personal.observe_steps = RECON_OBSERVE_STEPS
    assert is_complete(personal, ctx), "a subordinate's own 5 steps complete its task"
    team = Mission(
        MissionType.RECON, 1, (0, 0), issuer_id=-1, step_assigned=0, team_observation=True
    )
    team.observe_steps = RECON_OBSERVE_STEPS
    assert not is_complete(team, ctx), "the OPORD needs the team success threshold"
    team.observe_steps = TEAM_OBSERVE_STEPS
    assert is_complete(team, ctx)


def test_root_recon_completes_from_cover_via_team_observation():
    """A subordinate observes; the root, far from the objective the whole
    episode, still gets its COMPLETE confirmed — commanding from cover."""
    env = _recon_env()
    obj = env.world.objective_by_name("BRAVO")
    sl = env.roster.by_callsign["SL1"]
    env.roster.by_callsign["TL2"].pos = (obj.pos[0] - 5, obj.pos[1])  # on the ring
    sl.pos = (5, 21)  # commander in cover, ~30 cells out
    for _ in range(10):  # the team counter reaches the success threshold
        _step_all(env, {})
    assert env.outcome is None, "grace window open, awaiting the report"
    assert env._team_observe_steps >= 10
    assert sl.mission.observe_steps == env._team_observe_steps, "OPORD counter is team-mirrored"
    _obs, _r, terms, *_ = _step_all(env, {"SL1": DONE})
    assert env.outcome == "success"
    assert all(terms.values())
    assert [m.kind.value for m in env.transcript.messages[-2:]] == ["done_confirm", "endex"]


def test_early_root_claim_rejected_at_team_level():
    env = _recon_env()
    obj = env.world.objective_by_name("BRAVO")
    env.roster.by_callsign["TL2"].pos = (obj.pos[0] - 5, obj.pos[1])
    for _ in range(3):  # team observation short of the threshold
        _step_all(env, {})
    _obs, _r, terms, *_ = _step_all(env, {"SL1": DONE})
    assert not any(terms.values())
    assert env.outcome is None
    assert "done_reject" in [m.kind.value for m in env.transcript.messages]
    assert env.roster.by_callsign["SL1"].mission is not None, "false claimant keeps the task"


def test_subordinate_recon_done_stays_personal():
    """Team success does not let a subordinate that never observed claim its
    own RECON complete — its DONE reflects its own task."""
    env = _recon_env()
    obj = env.world.objective_by_name("BRAVO")
    env.roster.by_callsign["TL2"].pos = (obj.pos[0] - 5, obj.pos[1])  # TL2 observes
    env.inject_order("TL1, recon obj bravo", issuer="SL1")
    tl1 = env.roster.by_callsign["TL1"]
    tl1.pos = (5, 21)  # never in position
    for _ in range(10):
        _step_all(env, {})
    assert env._team_observe_steps >= 10, "team success reached by TL2"
    assert tl1.mission.observe_steps == 0, "personal counter untouched by the mirror"
    _step_all(env, {"TL1": DONE})
    reject = env.transcript.messages[-1]
    assert reject.kind.value == "done_reject"
    assert reject.recipient_id == tl1.id
    assert tl1.mission is not None, "the personal task still stands"


def test_root_compliance_pays_from_cover_while_team_observes():
    """In-position credit for the OPORD holder follows the team: the reward
    no longer pulls the commander's body onto the observation ring."""
    from cohort.core.missions import POSTURE_HOLD

    env = _recon_env()
    obj = env.world.objective_by_name("BRAVO")
    env.roster.by_callsign["TL2"].pos = (obj.pos[0] - 5, obj.pos[1])
    env.roster.by_callsign["SL1"].pos = (5, 21)
    *_, infos = _step_all(env, {})
    cfg = env.rewards_cfg
    # OPORD held since step 0: 1 step of standing-order tenure at step 1 (B5)
    tenure = 1.0 + cfg.tenure_factor * 1 / cfg.tenure_horizon
    assert infos["SL1"]["components"]["compliance"] == pytest.approx(
        POSTURE_HOLD * cfg.compliance_weight * tenure
    )

    env2 = _recon_env()  # control: nobody observes
    env2.roster.by_callsign["SL1"].pos = (5, 21)
    *_, infos2 = _step_all(env2, {})
    assert infos2["SL1"]["components"]["compliance"] == pytest.approx(0.0)


def test_grace_window_zero_restores_immediate_termination():
    from dataclasses import replace

    from cohort.config import get_scenario

    env = make_env(replace(get_scenario("fireteam"), grace_window=0))
    env.reset(seed=1)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.alive = False
    env.roster.by_callsign["TL1"].pos = env.world.objectives[0].pos
    _obs, _r, terms, *_ = _step_all(env, {})
    assert env.outcome == "success"
    assert all(terms.values())
