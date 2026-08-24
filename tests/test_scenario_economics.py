"""Per-scenario reward semantics (v1.21, owner-decided 2026-08-21).

platoon_hard prices idle time at -0.03 as part of its definition: the
anti-capture cycle showed the D4 attractor lives on idle income vs the time
price (0/8 survival at the default price, 5/6 at -0.03 with the rescue
armed). The price travels with the SCENARIO, not with a --reward flag, so a
clean run trains and ships under it with an empty economics.json override
list — baseline purity holds because the price is part of what the scenario
is, not a deviation from it.
"""

import pytest

from cohort.config import SCENARIOS, get_scenario
from cohort.env.cohort_env import make_env
from cohort.env.rewards import RewardConfig


def test_platoon_hard_prices_idle_time_and_its_parent_does_not():
    hard = dict(get_scenario("platoon_hard").reward_overrides)
    assert hard == {"time_penalty": -0.03}
    assert get_scenario("platoon").reward_overrides == ()


def test_env_defaults_to_scenario_economics():
    """An env built with no explicit RewardConfig runs the spec's prices —
    every consumer that omits the argument (play, probes) gets the scenario
    as defined, not the bare dataclass defaults."""
    assert make_env("platoon_hard").rewards_cfg.time_penalty == -0.03
    assert make_env("platoon").rewards_cfg.time_penalty == RewardConfig().time_penalty


def test_explicit_config_still_wins_over_the_spec():
    """A checkpoint's recorded economics (evaluate.py) must keep overriding
    the spec fallback, or old runs would silently rescore under new prices."""
    frozen = RewardConfig(time_penalty=-0.01)
    assert make_env("platoon_hard", reward_config=frozen).rewards_cfg.time_penalty == -0.01


def test_cli_overrides_layer_on_top_of_the_spec():
    """--reward flags beat the spec (experiments stay expressible), while
    unrelated spec prices survive underneath."""
    spec = get_scenario("platoon_hard")
    base = RewardConfig.from_scenario(spec)
    layered = RewardConfig.from_overrides(["time_penalty=-0.05"], base=base)
    assert layered.time_penalty == -0.05
    untouched = RewardConfig.from_overrides(["done_false=-2.0"], base=base)
    assert untouched.time_penalty == -0.03
    assert untouched.done_false == -2.0


def test_ablation_arms_inherit_the_parent_price():
    """nomask/flat are single-variable arms OF platoon_hard: they must share
    its economics or the ablation delta silently includes the price."""
    for arm in ("platoon_hard_nomask", "platoon_hard_flat"):
        assert dict(get_scenario(arm).reward_overrides) == {"time_penalty": -0.03}, arm


def test_a_typo_in_a_spec_fails_at_env_build():
    from dataclasses import replace

    broken = replace(get_scenario("platoon_hard"), reward_overrides=(("time_pnalty", -0.03),))
    with pytest.raises(ValueError, match="time_penalty"):
        RewardConfig.from_scenario(broken)


def test_no_other_shipping_scenario_gained_a_price():
    """The owner shipped the price for platoon_hard (and its arms inherit).
    Every other registered scenario keeps bare-default economics — a new
    spec override is a design decision and must land in this test."""
    priced = {n for n, s in SCENARIOS.items() if s.reward_overrides}
    assert priced == {
        "platoon_hard", "platoon_hard_nomask", "platoon_hard_flat",
        # degraded communications (docs/degraded-communications.md §3.3): a
        # voice-only root has no HQ channel, so the bonus for closing it is
        # structurally unearnable — priced at 0 by the scenario, never a flag
        "squad_voice_direct", "squad_voice_no_acoustic_ablation", "squad_voice_liaison",
        # owner-decided 2026-08-24: the same D4 idle-income attractor that
        # priced platoon_hard also captures this scenario at seed 14 (0.00
        # success, every episode to the clock). -0.03 removes it and is
        # neutral on the seeds that never captured; casualties flat
        # (46/400 -> 34/400, p = 0.19). Its sibling comm controls are NOT
        # priced — the capture is a property of the range-radio arm.
        "squad_range_control",
    }


def test_voice_only_presets_price_only_the_absent_hq_channel():
    for arm in ("squad_voice_direct", "squad_voice_no_acoustic_ablation", "squad_voice_liaison"):
        assert dict(get_scenario(arm).reward_overrides) == {"root_done_bonus": 0.0}, arm
