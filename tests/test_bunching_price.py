"""The threshold bunching price: what it charges, and what it must not.

Owner-decided 2026-08-26 after AREA FIRE was refuted — a casualty coupling
cannot reach a cohort nobody is shooting at, so the pile is charged directly.

The load-bearing property is that **the priced quantity is the measured one**.
``stacked_rate`` is the share of agent-steps with >= 2 living teammates within
``STACK_RADIUS``, so one free teammate makes the first charged step exactly the
first stacked step. If those two ever drift apart, a run could clear the marker
while paying for a different fault, or pay for one the board never shows.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from cohort.env.cohort_env import make_env
from cohort.env.rewards import COMPONENTS, RewardConfig
from cohort.metrics import STACK_RADIUS
from scripts.prereg_dispersion import LADDER


def _env(price: float, free: int = 1):
    env = make_env("fireteam")
    env.reset(seed=5)
    env.rewards_cfg = replace(env.rewards_cfg, bunching_penalty=price,
                              bunching_free_teammates=free)
    return env


def _charge(env) -> float:
    """One step's total bunching charge across every agent."""
    actions = {a: 0 for a in env.agents}
    _obs, _r, _t, _tr, infos = env.step(actions)
    return sum((i.get("components") or {}).get("bunching", 0.0) for i in infos.values())


# ------------------------------------------------- the tie to the metric ---

def test_the_priced_radius_is_the_measured_radius():
    assert RewardConfig().bunching_radius == STACK_RADIUS


def test_one_free_teammate_makes_the_first_charged_step_the_first_stacked_step():
    """cohort.metrics calls an agent stacked at >= 2 teammates inside the radius.

    So the free allowance must be exactly 1: at 1 teammate the metric says not
    stacked and the price must say nothing owed; at 2 both must fire.
    """
    assert RewardConfig().bunching_free_teammates == 1


def test_the_shipped_price_is_the_declared_first_rung():
    """Moving it is climbing the ladder: a decision, and a fleet retrain."""
    assert RewardConfig().bunching_penalty == -0.05
    assert LADDER[0] == "-0.05", "the fleet is not on rung 1 of the declared ladder"


def test_bunching_is_its_own_reward_component():
    # so run_report's component drift shows the price separately from the clock
    assert "bunching" in COMPONENTS


# ------------------------------------------------------- what it charges ---

def test_a_pair_inside_the_radius_pays_nothing():
    env = _env(-1.0)
    a, b, c, d = env.roster.soldiers
    a.pos, b.pos = (5, 5), (5, 6)
    c.pos, d.pos = (20, 20), (21, 21)   # a second, separate pair
    assert _charge(env) == 0.0


def test_the_third_teammate_is_the_first_one_billed():
    env = _env(-1.0)
    a, b, c, d = env.roster.soldiers
    a.pos, b.pos, c.pos = (5, 5), (5, 6), (6, 5)
    d.pos = (20, 20)
    # each of the three sees two teammates inside the radius -> one excess each
    assert _charge(env) == pytest.approx(-3.0)


def test_the_price_is_linear_in_the_excess():
    env = _env(-1.0)
    for s, pos in zip(env.roster.soldiers, [(5, 5), (5, 6), (6, 5), (6, 6)], strict=True):
        s.pos = pos
    # four in one clump: every agent has 3 teammates close, 2 excess each
    assert _charge(env) == pytest.approx(-8.0)


def test_a_teammate_outside_the_radius_is_not_billed():
    env = _env(-1.0)
    a, b, c, d = env.roster.soldiers
    a.pos, b.pos, c.pos = (5, 5), (5, 6), (5, 9)
    d.pos = (20, 20)
    assert _charge(env) == 0.0


def test_the_dead_are_not_billed_and_do_not_bill_others():
    """A price that counted corpses would charge an element for its casualties."""
    env = _env(-1.0)
    a, b, c, d = env.roster.soldiers
    a.pos, b.pos, c.pos = (5, 5), (5, 6), (6, 5)
    d.pos = (20, 20)
    c.alive, c.health = False, 0
    assert _charge(env) == 0.0   # a and b are now merely a pair


def test_off_charges_nothing_however_tight_the_pile():
    env = _env(0.0)
    for s, pos in zip(env.roster.soldiers, [(5, 5), (5, 6), (6, 5), (6, 6)], strict=True):
        s.pos = pos
    assert _charge(env) == 0.0


def test_the_free_allowance_is_a_knob_and_shifts_the_first_charge():
    env = _env(-1.0, free=2)
    a, b, c, d = env.roster.soldiers
    a.pos, b.pos, c.pos = (5, 5), (5, 6), (6, 5)
    d.pos = (20, 20)
    assert _charge(env) == 0.0   # two free teammates: the trio is now unbilled
