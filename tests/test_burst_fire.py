"""AREA FIRE (owner-decided 2026-08-21): a hit sprays the struck unit's
neighbors — the mechanic that makes dispersion tactically real. Shipped OFF
(``burst_fraction=0.0``); every test that turns it on does so on its own env
instance, so no shipped scenario is touched."""

from dataclasses import replace

from cohort.core.units import CombatParams
from cohort.env.cohort_env import make_env
from cohort.env.rewards import RewardLedger
from cohort.metrics import STACK_RADIUS


def _env(fraction: float):
    env = make_env("fireteam")
    env.reset(seed=5)
    env.combat = replace(env.combat, burst_fraction=fraction)
    return env


def test_shipped_default_is_off_and_the_radius_matches_the_metric():
    p = CombatParams()
    assert p.burst_fraction == 0.0
    # the fault the suite measures and the one the world prices must be the
    # same fault, or a run could pass the gate while dying to the mechanic
    assert p.burst_radius == STACK_RADIUS


def test_burst_off_touches_nobody():
    env = _env(0.0)
    a, b, *_ = env.roster.soldiers
    a.pos, b.pos = (5, 5), (5, 6)
    before = [s.health for s in env.roster.soldiers]
    env._burst_on_soldiers(a, 34, RewardLedger(), [])
    assert [s.health for s in env.roster.soldiers] == before


def test_burst_sprays_inside_the_radius_and_spares_the_struck():
    env = _env(0.5)
    a, b, c, d = env.roster.soldiers
    a.pos, b.pos, c.pos, d.pos = (5, 5), (5, 6), (6, 6), (5, 9)
    deaths: list = []
    env._burst_on_soldiers(a, 34, RewardLedger(), deaths)
    assert a.health == 100  # the struck soldier took the round; splash is for neighbors
    assert b.health == 100 - 17  # adjacent
    assert c.health == 100 - 17  # diagonal, 1.41 <= burst_radius
    assert d.health == 100  # 4.0 cells: outside the footprint
    assert deaths == []


def test_splash_kills_route_through_the_ordinary_casualty_path():
    env = _env(0.5)
    a, b, *_ = env.roster.soldiers
    a.pos, b.pos = (5, 5), (5, 6)
    b.health = 10
    deaths: list = []
    env._burst_on_soldiers(a, 34, RewardLedger(), deaths)
    assert not b.alive and b.health == 0
    assert deaths == [b]


def test_dead_neighbors_take_no_splash():
    env = _env(0.5)
    a, b, *_ = env.roster.soldiers
    a.pos, b.pos = (5, 5), (5, 6)
    b.alive = False
    b.health = 0
    env._burst_on_soldiers(a, 34, RewardLedger(), [])
    assert b.health == 0


def test_friendly_burst_credits_the_shooter_and_kills_stacked_enemies():
    env = _env(0.5)
    shooter = env.roster.soldiers[0]
    e1, e2, e3 = env.enemies
    e2.pos = e1.pos
    e2.health = 10
    e3.pos = (e1.pos[0] + 10, e1.pos[1])  # well outside the footprint
    kills: list = []
    env._burst_on_enemies(shooter, e1, 34, 1.0, RewardLedger(), kills)
    assert e1.health == 100  # the struck enemy's own damage is the caller's line
    assert not e2.alive
    assert e3.health == 100
    assert kills == [(shooter, e2)]
