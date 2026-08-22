"""The D4 ledger probe's counterfactual is arithmetic, and it must stay honest.

The probe exists to satisfy the repo's diagnose-first rule: a reward change is
proposed only after the ledger shows what a policy actually earns. The pricing
step below is the part that turns a measurement into a recommendation, so it is
pinned here rather than trusted.
"""

import pytest

from scripts.d4_ledger_probe import DEFAULT_TIME_PENALTY, price_at


def test_repricing_at_the_default_is_the_identity():
    ledger = {"compliance": 0.0096, "command": 0.0042, "time": DEFAULT_TIME_PENALTY}
    total, _non_time, at_default = price_at(ledger, DEFAULT_TIME_PENALTY)
    assert at_default == pytest.approx(total)


def test_only_the_time_component_scales():
    # tripling the price must triple the time charge and touch nothing else
    ledger = {"compliance": 0.0096, "command": 0.0042, "time": -0.01}
    _total, non_time, tripled = price_at(ledger, -0.03)
    assert non_time == pytest.approx(0.0138)
    assert tripled == pytest.approx(0.0138 - 0.03)


def test_a_captured_ledger_flips_sign_at_the_platoon_depth_cure():
    # the squad_range_control_v1_seed14 capture, measured: the trickle beats the
    # default time price, and tripling it is what takes idling negative
    captured = {"compliance": 0.0107, "command": 0.0041, "combat": 0.0002,
                "report": 0.0001, "terminal": 0.0, "time": -0.0098}
    total, _non_time, tripled = price_at(captured, -0.03)
    assert total > 0, "the capture must be net-positive income, or it is not D4"
    assert tripled < 0, "tripling the price must make idling bleed"
