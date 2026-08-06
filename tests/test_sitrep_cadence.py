"""Reporting doctrine: ScenarioSpec.sitrep_cadence gives silence semantics.

When set, an agent out of contact owes a SITREP every ``sitrep_cadence``
steps; being overdue draws ``RewardConfig.sitrep_overdue`` per step and the
due-ness is surfaced in the comms-summary observation slot that is otherwise
redundant (the "known enemy present" flag, implied by the known-count field).
Default (None) changes nothing.
"""

from dataclasses import replace

from cohort import make_env
from cohort.config import get_scenario
from cohort.env.actions import CATALOG

STAY = 0
SITREP_IDX = next(s.index for s in CATALOG if s.kind == "sitrep")
CONTACT_IDX = next(s.index for s in CATALOG if s.kind == "contact")

#: comms summary block starts at 106 (A5-2 layout); slot 108 = known-present / sitrep-due.
DUE_FIELD = 108

CADENCE = 10


def _flat_env(spec, seed=1):
    env = make_env(spec)
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (22, 1)
        e.home = e.pos
    return env


def _cadence_env(seed=1):
    return _flat_env(replace(get_scenario("fireteam"), sitrep_cadence=CADENCE), seed=seed)


def _step_all(env, overrides=None):
    acts = {a: STAY for a in env.agents}
    acts.update(overrides or {})
    return env.step(acts)


def test_overdue_penalty_applies_exactly_past_the_cadence():
    env = _cadence_env()
    for step in range(1, CADENCE + 5):
        *_, infos = _step_all(env)
        overdue = infos["RFN1"]["components"]["report"]
        if step <= CADENCE:
            assert overdue == 0.0, f"step {step}: not yet overdue"
        else:
            assert overdue < 0.0, f"step {step}: overdue penalty must apply"


def test_sitrep_resets_the_clock_and_a_due_report_is_fresh():
    env = _cadence_env()
    for _ in range(CADENCE + 1):  # step CADENCE+1: first overdue step
        *_, infos = _step_all(env)
    assert infos["RFN1"]["components"]["report"] < 0.0
    # the mandated report is scored fresh (the cadence is the interval)...
    *_, infos = _step_all(env, {"RFN1": SITREP_IDX})
    assert infos["RFN1"]["components"]["report"] > 0.0
    # ...and resets the clock: no penalty for the next CADENCE steps
    for step in range(1, CADENCE):
        *_, infos = _step_all(env)
        assert infos["RFN1"]["components"]["report"] == 0.0, f"{step} steps after the report"


def test_agents_in_contact_are_exempt():
    env = _cadence_env()
    sld = env.roster.by_callsign["RFN1"]
    sld.pos = (7, 10)
    enemy = env.enemies[0]
    enemy.pos = (10, 10)  # visible: RFN1 is in contact
    enemy.home = enemy.pos
    for _ in range(CADENCE + 3):
        *_, infos = _step_all(env)
        if not sld.alive:  # the garrison may win the firefight; enough sampled
            break
        assert infos["RFN1"]["components"]["report"] == 0.0, "in contact → no SITREP owed"


def test_due_ness_is_surfaced_in_the_observation():
    env = _cadence_env()
    obs = None
    for step in range(1, CADENCE + 2):
        obs, *_ = _step_all(env)
        expected = min(1.0, step / CADENCE)
        assert abs(obs["RFN1"]["observation"][DUE_FIELD] - expected) < 1e-6, f"step {step}"
    # sending the report drops the due-ness back toward zero
    obs, *_ = _step_all(env, {"RFN1": SITREP_IDX})
    assert obs["RFN1"]["observation"][DUE_FIELD] < 0.2


def test_default_none_changes_nothing():
    env = _flat_env(get_scenario("fireteam"))
    for _ in range(30):
        *_, infos = _step_all(env)
        assert infos["RFN1"]["components"]["report"] == 0.0, "no doctrine, no penalty"
    # and the observation slot keeps its known-present semantics
    obs, *_ = _step_all(env)
    assert obs["RFN1"]["observation"][DUE_FIELD] == 0.0
    sld = env.roster.by_callsign["RFN2"]
    sld.pos = (7, 10)
    enemy = env.enemies[0]
    enemy.pos = (10, 10)
    enemy.home = enemy.pos
    obs, *_ = _step_all(env, {"RFN2": CONTACT_IDX})
    assert obs["RFN1"]["observation"][DUE_FIELD] == 1.0, "known-present flag as shipped"
