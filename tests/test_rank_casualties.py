"""Rank-weighted casualties (P4): losing a leader costs more.

``death`` and ``teammate_death`` scale with the FALLEN agent's *effective*
authority: x (1 + rank_casualty_scale x authority). An RFN (authority 0)
costs the base penalty; a PL (authority 4) costs double at the default
scale of 0.25.
"""

from dataclasses import replace

import pytest

from cohort import make_env
from cohort.config import get_scenario
from cohort.core.ranks import Rank


def _kill(scenario, victim_cs, seed=5):
    """Park ``victim_cs`` (1 hp) next to a live enemy, everyone else far away;
    step until it dies. Returns (env, infos-of-the-killing-step).
    root_human=False so the -25 human penalty does not cloud the arithmetic."""
    env = make_env(replace(get_scenario(scenario), root_human=False))
    env.reset(seed=seed)
    env.world.grid[:] = 0
    victim = env.roster.by_callsign[victim_cs]
    enemy = next(e for e in env.enemies if e.alive)
    for e in env.enemies:  # one shooter only: keeps the ledger arithmetic exact
        if e is not enemy:
            e.alive = False
    for s in env.roster.soldiers:
        if s is not victim:
            s.pos = (1, 1)
    victim.health = 1
    victim.pos = (enemy.pos[0] + 1, enemy.pos[1])
    infos = None
    for _ in range(40):
        if not victim.alive:
            break
        *_, infos = env.step({a: 0 for a in env.agents})
    assert not victim.alive, f"{victim_cs} should have been killed"
    return env, infos


def test_rifleman_death_costs_the_base_penalty():
    env, infos = _kill("fireteam", "RFN1")
    cfg = env.rewards_cfg
    assert infos["RFN1"]["components"]["combat"] == pytest.approx(cfg.took_hit + cfg.death)
    assert infos["TL1"]["components"]["combat"] == pytest.approx(cfg.teammate_death)


def test_platoon_leader_death_costs_double():
    env, infos = _kill("platoon", "PL1")
    cfg = env.rewards_cfg
    weight = 1.0 + cfg.rank_casualty_scale * 4  # PL authority = 4 → 2.0
    assert weight == 2.0
    assert infos["PL1"]["components"]["combat"] == pytest.approx(
        cfg.took_hit + cfg.death * weight
    )
    for cs in ("SL1", "TL1", "RFN1"):
        assert infos[cs]["components"]["combat"] == pytest.approx(cfg.teammate_death * weight)


def test_scaling_uses_effective_authority():
    """A rifleman acting as TL after succession dies at the TL weight."""
    env, infos = _kill("fireteam", "TL1")  # TL death first: RFN1 assumes command
    cfg = env.rewards_cfg
    tl_weight = 1.0 + cfg.rank_casualty_scale * 1
    assert infos["TL1"]["components"]["combat"] == pytest.approx(
        cfg.took_hit + cfg.death * tl_weight
    )
    promoted = env.roster.root()
    assert promoted.rank is Rank.RFN and promoted.effective_rank is Rank.TL

    # now the acting TL falls: penalties carry the acting (effective) weight
    enemy = next(e for e in env.enemies if e.alive)
    promoted.health = 1
    promoted.pos = (enemy.pos[0] + 1, enemy.pos[1])
    infos = None
    for _ in range(40):
        if not promoted.alive:
            break
        *_, infos = env.step({a: 0 for a in env.agents})
    assert not promoted.alive
    assert infos[promoted.callsign]["components"]["combat"] == pytest.approx(
        cfg.took_hit + cfg.death * tl_weight
    ), "the acting rank, not the intrinsic one, sets the casualty weight"


def test_knob_zero_restores_flat_penalties():
    from cohort.env.rewards import RewardConfig

    env = make_env(replace(get_scenario("fireteam"), root_human=False),
                   reward_config=replace(RewardConfig(), rank_casualty_scale=0.0))
    env.reset(seed=5)
    env.world.grid[:] = 0
    tl = env.roster.by_callsign["TL1"]
    enemy = next(e for e in env.enemies if e.alive)
    for s in env.roster.soldiers:
        if s is not tl:
            s.pos = (1, 1)
    tl.health = 1
    tl.pos = (enemy.pos[0] + 1, enemy.pos[1])
    infos = None
    for _ in range(40):
        if not tl.alive:
            break
        *_, infos = env.step({a: 0 for a in env.agents})
    assert not tl.alive
    cfg = env.rewards_cfg
    assert infos["TL1"]["components"]["combat"] == pytest.approx(cfg.took_hit + cfg.death)
    assert infos["RFN1"]["components"]["combat"] == pytest.approx(cfg.teammate_death)
