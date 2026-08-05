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


def test_root_report_ends_the_episode_that_step():
    env = _cleared_env()
    _step_all(env, {})  # T0 = 1: condition met, window opens
    assert env.outcome is None
    _obs, _rewards, terms, *_ , infos = _step_all(env, {"TL1": DONE})
    assert env.outcome == "success"
    assert all(terms.values())
    # the transcript ends with the root's COMPLETE answered by HQ
    last_two = env.transcript.messages[-2:]
    assert [m.kind.value for m in last_two] == ["done", "done_confirm"]
    assert "— COMPLETE" in last_two[0].text
    assert last_two[1].sender_id == HQ_ID
    assert "CONFIRMED" in last_two[1].text
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
    assert env.transcript.messages[-1].kind.value == "done_confirm"


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
