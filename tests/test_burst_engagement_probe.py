"""The counter that refuted AREA FIRE has to count the right thing.

`scripts/burst_engagement_probe.py` produced the numbers that withdrew a
mechanism the owner had already approved, so its arithmetic is load-bearing.
The failure mode that would matter is silent: counting the DIRECT round as
splash would inflate every column and make a mechanic that never fires look
like one that fires constantly — the exact opposite of the finding.
"""

from __future__ import annotations

from dataclasses import replace

from cohort.env.cohort_env import make_env
from cohort.env.rewards import RewardLedger
from scripts.burst_engagement_probe import Counter


def _env(fraction: float):
    env = make_env("fireteam")
    env.reset(seed=5)
    env.combat = replace(env.combat, burst_fraction=fraction)
    return env


def test_the_struck_soldier_is_a_hit_and_never_a_spray():
    env = _env(0.5)
    c = Counter(env)
    a, b, c_far, d = env.roster.soldiers
    a.pos, b.pos, c_far.pos, d.pos = (5, 5), (5, 6), (20, 20), (5, 9)
    env._burst_on_soldiers(a, 34, RewardLedger(), [])
    assert c.hits_on_soldiers == 1
    assert c.bursts == 1
    assert c.sprayed == 1          # b only — d is 4.0 cells away
    assert c.splash_damage == 17   # int(34 * 0.5), the neighbour's share alone
    assert a.health == 100         # the probe must not bill the struck soldier


def test_a_landed_round_that_sprays_nobody_is_a_hit_but_not_a_burst():
    env = _env(0.5)
    c = Counter(env)
    a, b, cc, d = env.roster.soldiers
    a.pos = (5, 5)
    for s in (b, cc, d):
        s.pos = (20, 20)
    env._burst_on_soldiers(a, 34, RewardLedger(), [])
    assert (c.hits_on_soldiers, c.bursts, c.sprayed, c.splash_damage) == (1, 0, 0, 0)


def test_with_the_mechanic_off_hits_still_count_and_nothing_sprays():
    """The 0.0 row is the control the whole comparison rests on."""
    env = _env(0.0)
    c = Counter(env)
    a, b, *rest = env.roster.soldiers
    a.pos, b.pos = (5, 5), (5, 6)
    for s_ in rest:
        s_.pos = (20, 20)
    env._burst_on_soldiers(a, 34, RewardLedger(), [])
    assert c.hits_on_soldiers == 1
    assert (c.bursts, c.sprayed, c.splash_damage) == (0, 0, 0)
    assert b.health == 100


def test_direct_damage_outside_a_burst_is_not_counted_as_splash():
    env = _env(0.5)
    c = Counter(env)
    a = env.roster.soldiers[0]
    env._damage_soldier(a, 34, RewardLedger(), [])
    assert (c.sprayed, c.splash_damage, c.hits_on_soldiers) == (0, 0, 0)
    assert a.health == 100 - 34


def test_a_neighbour_killed_by_splash_is_attributed_to_splash():
    env = _env(1.0)
    c = Counter(env)
    a, b, *rest = env.roster.soldiers
    a.pos, b.pos = (5, 5), (5, 6)
    for s_ in rest:
        s_.pos = (20, 20)
    b.health = 20
    deaths: list = []
    env._burst_on_soldiers(a, 34, RewardLedger(), deaths)
    assert not b.alive
    assert c.splash_deaths == 1
    assert c.splash_damage == 20  # capped at the health actually on the soldier
