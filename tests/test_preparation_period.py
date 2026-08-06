"""The defend preparation period (v1.10): ScenarioSpec.assault_h_hour.

Before H the OpFor is on the map — spawned, oracle-visible, spottable — but
does not move, fire, or advance. A DEFEND mission presumes prepared positions;
the preparation period is the time to occupy them.

The OPORD announces the band's MIDPOINT as the nominal H while the assault
actually arrives anywhere in the band, so the habit the scenario rewards is
*being set early*, not *timing the tick*.
"""

from dataclasses import replace

import pytest

from cohort import make_env
from cohort.config import get_scenario
from cohort.env.observations import TEMPO_TIME_TO_CONTACT

BAND = (55, 75)


def _defend(seed=7, **overrides):
    spec = get_scenario("fireteam_defend")
    if overrides:
        spec = replace(spec, **overrides)
    env = make_env(spec)
    obs, _ = env.reset(seed=seed)
    return env, obs


def test_h_hour_is_drawn_inside_the_band_and_the_nominal_is_the_midpoint():
    for seed in range(12):
        env, _ = _defend(seed=seed)
        assert BAND[0] <= env._h_hour <= BAND[1], f"seed {seed}: H={env._h_hour}"
        assert env._h_hour_nominal == (BAND[0] + BAND[1]) // 2


def test_h_hour_is_deterministic_per_seed():
    """Determinism convention: all env randomness through the seeded _rng."""
    a, _ = _defend(seed=11)
    b, _ = _defend(seed=11)
    assert a._h_hour == b._h_hour
    assert [e.pos for e in a.enemies] == [e.pos for e in b.enemies]


def test_the_band_actually_varies_the_arrival():
    """A constant H would let a policy time the tick instead of standing to."""
    drawn = {_defend(seed=s)[0]._h_hour for s in range(30)}
    assert len(drawn) > 1, "H must jitter across seeds"


def test_the_opord_announces_the_nominal_h_on_the_net():
    env, _ = _defend()
    opord = env.transcript.messages[0]
    assert "EXPECT ASSAULT AT H PLUS 65" in opord.text
    # ...and the task statement still parses: the warning is not the order
    from cohort.core import language as lang
    from cohort.core.missions import MissionType

    parsed = lang.parse_order(opord.text)
    assert parsed.mission is MissionType.DEFEND
    assert parsed.objective_name == "ALPHA"


def test_opfor_is_frozen_until_h_then_advances():
    env, _ = _defend(seed=7)
    h = env._h_hour
    start = [e.pos for e in env.enemies]
    for _ in range(h - 1):  # the H-th step IS the assault: hold up to H-1
        env.step({a: 0 for a in env.agents})
    assert env._step_count == h - 1 and env._in_preparation()
    assert [e.pos for e in env.enemies] == start, "no OpFor movement before H"
    assert all(s.alive for s in env.roster.soldiers), "and no OpFor fire before H"
    for _ in range(15):
        env.step({a: 0 for a in env.agents})
    assert [e.pos for e in env.enemies] != start, "the assault begins at H"


def test_opfor_exists_from_step_zero_even_while_held():
    """Held, not absent: the oracle sees the assault forming from step 0, and a
    patrol that goes looking can spot it (the early warning a defense earns)."""
    env, _ = _defend()
    assert len(env.enemies) == env.spec_cfg.n_enemies
    assert all(e.alive for e in env.enemies)
    assert env._in_preparation()


def test_time_to_contact_counts_down_to_the_nominal_h():
    env, obs = _defend()
    nominal = env._h_hour_nominal
    assert obs["TL1"]["observation"][TEMPO_TIME_TO_CONTACT] == 1.0
    prev = 1.0
    for step in range(1, nominal + 5):
        obs, *_ = env.step({a: 0 for a in env.agents})
        ttc = obs["TL1"]["observation"][TEMPO_TIME_TO_CONTACT]
        if step < nominal:
            assert ttc == pytest.approx((nominal - step) / nominal, abs=1e-6)
            assert ttc < prev, "monotone countdown"
            prev = ttc
        else:
            assert ttc == 0.0, "0 once the announced hour passes"


def test_scenarios_without_a_preparation_period_are_untouched():
    for name in ("fireteam", "squad", "platoon", "defend_brique", "squad_recon"):
        env = make_env(name)
        env.reset(seed=3)
        assert env.spec_cfg.assault_h_hour is None
        assert env._h_hour is None
        assert not env._in_preparation()
        assert "EXPECT ASSAULT" not in env.transcript.messages[0].text


def test_no_preparation_period_consumes_no_randomness():
    """The draw is guarded, so every other scenario's seeds reproduce exactly."""
    a = make_env("fireteam")
    a.reset(seed=5)
    b = make_env(replace(get_scenario("fireteam"), max_steps=999))
    b.reset(seed=5)
    assert [e.pos for e in a.enemies] == [e.pos for e in b.enemies]
    assert [s.pos for s in a.roster.soldiers] == [s.pos for s in b.roster.soldiers]
