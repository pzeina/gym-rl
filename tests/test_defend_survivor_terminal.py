"""The defend terminal is scaled by the force that held (v1.12, option 4).

Where a decisive objective exists, engaging ends the episode sooner, so paying
the fallen the team terminal (``d44ee8d``) made cover and survival rise as
instruments of a win everyone could now collect: squad_screen went 165 -> 53
steps with deaths halved and cover x15, and both seeds went 0.00 -> 1.00.

A defend mission has no fast win — the mission is to still be there later — so
the same change made bodies cost 1.0 apiece with nothing to buy.
``defend_brique_v3 -> v4`` at N=100 held success flat (0.88 -> 0.91) while
commander death went 0.24 -> 0.61, cover 0.513 -> 0.416 and the fight moved
from 2.87 to 6.09 cells off the objective, failing the
``mean_distance_from_objective_under_threat`` regression gate.

Scaling the terminal by surviving rank-weighted strength restores the
preservation pressure that forfeiture used to supply. The reason it is not
simply forfeiture again is the whole design: the multiplier is IDENTICAL for
every agent, fallen included, so a death is a shared loss and not a private
one. That is what these tests pin.
"""

from __future__ import annotations

import pytest

from cohort import make_env
from cohort.config import SCENARIOS, get_scenario
from cohort.core.missions import MissionType
from cohort.core.ranks import AUTHORITY, Rank
from cohort.env.rewards import RewardConfig
from cohort.training.ppo import PPOConfig

STAY = 0
DEFEND_SCENARIOS = [n for n, s in SCENARIOS.items()
                    if s.root_mission in (MissionType.DEFEND, MissionType.DENY)]


# ---------------------------------------------------------------------- #
# the invariant that sets the constant
# ---------------------------------------------------------------------- #

def test_defend_terminal_scaling_preserves_dominance():
    """Winning must beat stalling 2x at the FLOOR, not merely at full payout.

    The multiplier can only reduce the terminal, so checking dominance at
    ``success_team`` checks a payout a defend policy may never see. The case
    that matters is precisely the one the scenario is about: hold, and be
    ground down doing it. If THAT is worth less than farming shaping, stalling
    is the better play exactly when the mission is hardest — which is the
    v1.11 collapse with extra steps.

    This is what fixes ``defend_survivor_scale`` at 0.35: fireteam_defend
    scores 3.42 undiminished, so the largest admissible scale is
    1 - 2.0/3.42 = 0.415.
    """
    cfg = RewardConfig()
    gamma = PPOConfig.gamma
    thin = {}
    for name in DEFEND_SCENARIOS:
        spec = get_scenario(name)
        floor = cfg.terminal_scale_floor(spec.root_mission)
        ratio = cfg.win_beats_stall(gamma, spec.max_steps, terminal_scale=floor)
        if ratio < 2.0:
            thin[name] = round(ratio, 2)
    assert not thin, (
        f"at defend_survivor_scale={cfg.defend_survivor_scale}, a wiped-but-holding "
        f"force scores {thin} against the 2x bar. Lower the scale — the ceiling is "
        f"1 - 2.0/min(undiminished ratio)."
    )


def test_the_scale_is_at_the_edge_of_what_dominance_allows():
    """Pin the headroom, so a later economics change cannot silently eat it.

    If someone lowers ``success_team`` or lengthens a defend scenario, the
    admissible ceiling moves and 0.35 may stop being safe. That would surface
    as the test above failing — but only if this margin is small enough to
    notice, so state it.
    """
    cfg = RewardConfig()
    gamma = PPOConfig.gamma
    tightest = min(
        cfg.win_beats_stall(gamma, get_scenario(n).max_steps) for n in DEFEND_SCENARIOS
    )
    ceiling = 1.0 - 2.0 / tightest
    assert cfg.defend_survivor_scale <= ceiling, (
        f"scale {cfg.defend_survivor_scale} exceeds the dominance ceiling {ceiling:.3f}"
    )
    assert cfg.defend_survivor_scale >= 0.25, (
        "below ~0.25 the multiplier is too weak to price the bodies the v1.11 "
        "defend regression was made of; if it must go lower, the fix is a "
        "different lever, not a homeopathic dose of this one"
    )


# ---------------------------------------------------------------------- #
# the multiplier itself
# ---------------------------------------------------------------------- #

def test_an_intact_force_is_paid_in_full():
    assert RewardConfig().survivor_multiplier(10.0, 10.0) == pytest.approx(1.0)


def test_a_wiped_force_is_paid_the_floor_not_nothing():
    """Zero would be forfeiture at team scale, and would break dominance."""
    cfg = RewardConfig()
    assert cfg.survivor_multiplier(0.0, 10.0) == pytest.approx(1.0 - cfg.defend_survivor_scale)
    assert cfg.survivor_multiplier(0.0, 10.0) > 0.0


def test_the_multiplier_is_monotone_in_surviving_strength():
    cfg = RewardConfig()
    values = [cfg.survivor_multiplier(w, 10.0) for w in range(11)]
    assert values == sorted(values)
    assert values[0] < values[-1], "it must actually vary, or it prices nothing"


def test_a_ratio_above_one_cannot_mint_reward():
    cfg = RewardConfig()
    assert cfg.survivor_multiplier(99.0, 10.0) == pytest.approx(1.0)


def test_scale_zero_restores_the_flat_terminal_exactly():
    """The escape hatch: `--reward defend_survivor_scale=0` is v1.11 behaviour.

    This is the revert path to the owner's option 1 (scope the payout by
    scenario) — it needs no code change, only a flag.
    """
    off = RewardConfig.from_overrides(["defend_survivor_scale=0"])
    assert off.survivor_multiplier(0.0, 10.0) == 1.0
    assert off.terminal_scale_floor(MissionType.DEFEND) == 1.0


def test_a_non_defend_root_is_never_scaled():
    cfg = RewardConfig()
    for mission in (MissionType.SEIZE, MissionType.RECON, MissionType.SCREEN):
        assert cfg.terminal_scale_floor(mission) == 1.0


# ---------------------------------------------------------------------- #
# the payout, in the env
# ---------------------------------------------------------------------- #

def _defend_env(seed=1):
    """fireteam_defend with every enemy dead: success is one step away."""
    env = make_env("fireteam_defend")
    env.reset(seed=seed)
    for e in env.enemies:
        e.alive = False
    return env


def _run_to_outcome(env, max_steps=60):
    infos = {}
    for _ in range(max_steps):
        if not env.agents:
            break
        _obs, _r, _t, _tr, infos = env.step({a: STAY for a in env.agents})
        if env.outcome is not None:
            break
    return infos


def test_an_intact_defend_force_collects_the_undiminished_terminal():
    env = _defend_env()
    infos = _run_to_outcome(env)
    assert env.outcome == "success"
    terminal = infos["TL1"]["components"]["terminal"]
    assert terminal >= RewardConfig().success_team


def test_holding_with_losses_pays_less_than_holding_intact():
    """The whole point, measured end to end."""
    intact = _run_to_outcome(_defend_env())["TL1"]["components"]["terminal"]

    env = _defend_env()
    env.roster.by_callsign["RFN2"].alive = False
    env.roster.by_callsign["RFN2"].health = 0
    reduced = _run_to_outcome(env)["TL1"]["components"]["terminal"]

    assert reduced < intact, "a body must cost some of the payout"
    assert reduced > 0.0, "but holding is still worth winning"


def test_the_fallen_are_paid_the_same_scaled_terminal_as_the_living():
    """D4 must not come back through this door.

    Forfeiture is what caused the collapse: the individual gain from hanging
    back is visible to a per-agent advantage while the collective cost is not.
    A scaled terminal only avoids re-creating that if the fallen are inside it
    — every agent seeing the SAME multiplier is what makes a death a shared
    loss rather than a private one.
    """
    env = _defend_env()
    dead = env.roster.by_callsign["RFN2"]
    dead.alive = False
    dead.health = 0
    infos = _run_to_outcome(env)
    assert env.outcome == "success"
    assert infos["RFN2"]["components"]["terminal"] == pytest.approx(
        infos["RFN3"]["components"]["terminal"]
    ), "a casualty is paid the same scaled terminal as a survivor"


def test_losing_the_commander_costs_more_of_the_payout_than_losing_a_rifleman():
    """Rank-weighted, matching how casualties are already priced.

    The measured half of the v1.11 defend regression was commander death
    0.24 -> 0.61, so the leader has to be worth more than the rifleman here.
    """
    env_rfn = _defend_env()
    env_rfn.roster.by_callsign["RFN2"].alive = False
    env_rfn.roster.by_callsign["RFN2"].health = 0
    lost_rifleman = _run_to_outcome(env_rfn)["RFN3"]["components"]["terminal"]

    env_tl = _defend_env()
    env_tl.roster.by_callsign["TL1"].alive = False
    env_tl.roster.by_callsign["TL1"].health = 0
    lost_leader = _run_to_outcome(env_tl)["RFN3"]["components"]["terminal"]

    assert lost_leader < lost_rifleman


def test_succession_cannot_inflate_the_surviving_force():
    """Intrinsic rank, not effective authority — the trap in this design.

    Succession promotes a survivor into the dead leader's slot. Summed over
    ``effective_authority`` the living force would get STRONGER by losing its
    commander, and a defend cohort could raise its own terminal by getting the
    leader killed. Weighted by intrinsic rank the numerator can only fall.
    """
    env = _defend_env()
    tl = env.roster.by_callsign["TL1"]
    tl.alive = False
    tl.health = 0
    after_loss = env._defend_terminal_scale()
    assert after_loss < 1.0, "the force is down a commander; the payout must reflect it"

    # promote a survivor into the empty slot, which is what succession does
    successor = env.roster.by_callsign["RFN1"]
    successor.acting_rank = Rank.TL
    assert AUTHORITY[successor.effective_rank] > AUTHORITY[successor.rank], (
        "the promotion has to be real, or this asserts nothing"
    )
    assert env._defend_terminal_scale() == pytest.approx(after_loss), (
        "an effective-authority sum would RISE here — a cohort could raise its "
        "own terminal by getting its commander killed"
    )


def test_the_weights_are_the_rank_casualty_convention():
    """Same weighting as `death`/`teammate_death`, computed by hand."""
    env = _defend_env()
    env.roster.by_callsign["RFN2"].alive = False
    cfg = env.rewards_cfg
    total = sum(1.0 + cfg.rank_casualty_scale * AUTHORITY[s.rank]
                for s in env.roster.soldiers)
    alive = sum(1.0 + cfg.rank_casualty_scale * AUTHORITY[s.rank]
                for s in env.roster.soldiers if s.alive)
    assert env._defend_terminal_scale() == pytest.approx(
        cfg.survivor_multiplier(alive, total)
    )


@pytest.mark.parametrize("scenario", ["fireteam", "squad", "squad_recon", "squad_screen"])
def test_the_converged_fleet_keeps_its_flat_terminal(scenario):
    """Five scenarios sit at 1.00 under the flat terminal; none of them move.

    This is why option 4 is scoped to defend roots rather than applied
    globally — the fix targets the scenarios that regressed and is a no-op on
    the ones that did not.
    """
    env = make_env(scenario)
    env.reset(seed=1)
    assert env._defend_terminal_scale() == 1.0
    env.roster.by_callsign["RFN2"].alive = False
    assert env._defend_terminal_scale() == 1.0, "still 1.0 with a casualty"
