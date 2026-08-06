"""Human agents (P3): the root commander is human, observable, and costly to lose."""

import pytest

from cohort import SCENARIOS, make_env
from cohort.core.ranks import Rank
from cohort.core.units import Soldier, validate_human_ranks

#: observation offsets (see env/observations.py layout)
SELF_HUMAN_FIELD = 12          # self block: 4 + 7 rank one-hot + cover, then is-human
LEADER_HUMAN_FIELD = 13 + 19 + 4  # leader block: present, dx, dy, mission, is-human


def test_root_is_human_in_every_preset():
    for name in SCENARIOS:
        env = make_env(name)
        env.reset(seed=1)
        root = env.roster.root()
        assert root.human, f"{name}: the root commander must be human by default"
        others = [s for s in env.roster.soldiers if s.id != root.id]
        assert all(not s.human for s in others), f"{name}: only the root is human"


def test_root_human_knob_off():
    from dataclasses import replace

    from cohort.config import get_scenario

    env = make_env(replace(get_scenario("fireteam"), root_human=False))
    env.reset(seed=1)
    assert all(not s.human for s in env.roster.soldiers)


def test_invariant_humans_outrank_all_non_humans():
    # valid: the TL is human, everyone below is not
    ok = [
        Soldier(id=0, callsign="TL1", rank=Rank.TL, pos=(0, 0), human=True),
        Soldier(id=1, callsign="RFN1", rank=Rank.RFN, pos=(1, 0), leader_id=0),
    ]
    validate_human_ranks(ok)  # no raise
    validate_human_ranks([])  # vacuous
    # invalid: a human rifleman below a non-human TL
    bad = [
        Soldier(id=0, callsign="TL1", rank=Rank.TL, pos=(0, 0)),
        Soldier(id=1, callsign="RFN1", rank=Rank.RFN, pos=(1, 0), leader_id=0, human=True),
    ]
    with pytest.raises(ValueError, match="must outrank all non-humans"):
        validate_human_ranks(bad)
    # invalid: human and non-human at the SAME intrinsic authority
    tie = [
        Soldier(id=0, callsign="TL1", rank=Rank.TL, pos=(0, 0), human=True),
        Soldier(id=1, callsign="TL2", rank=Rank.TL, pos=(1, 0)),
    ]
    with pytest.raises(ValueError, match="must outrank all non-humans"):
        validate_human_ranks(tie)


def test_human_flags_in_observations():
    env = make_env("squad")
    obs, _ = env.reset(seed=2)
    assert obs["SL1"]["observation"][SELF_HUMAN_FIELD] == 1.0, "the root knows it is human"
    assert obs["TL1"]["observation"][SELF_HUMAN_FIELD] == 0.0
    assert obs["TL1"]["observation"][LEADER_HUMAN_FIELD] == 1.0, "TL1's leader (SL1) is human"
    assert obs["RFN1"]["observation"][LEADER_HUMAN_FIELD] == 0.0, "RFN1's leader (TL1) is not"
    assert obs["SL1"]["observation"][LEADER_HUMAN_FIELD] == 0.0, "the root reports to HQ"


def test_human_death_penalty_hits_every_present_agent():
    env = make_env("fireteam")
    env.reset(seed=5)
    env.world.grid[:] = 0
    tl = env.roster.by_callsign["TL1"]  # the human root
    assert tl.human
    enemy = next(e for e in env.enemies if e.alive)
    tl.health = 1
    tl.pos = (enemy.pos[0] + 1, enemy.pos[1])
    # everyone else far from the fight
    for cs in ("RFN1", "RFN2", "RFN3"):
        env.roster.by_callsign[cs].pos = (1, 1)
    infos = None
    for _ in range(30):
        if not tl.alive:
            break
        *_, infos = env.step({a: 0 for a in env.agents})
    assert not tl.alive, "the adjacent enemy should have killed the human root"
    cfg = env.rewards_cfg
    for cs in ("RFN1", "RFN2", "RFN3"):
        combat = infos[cs]["components"]["combat"]
        assert combat <= cfg.human_death, f"{cs}: human_death must be included, got {combat}"
    # the fallen human pays it too (present that step), on top of its own death
    assert infos["TL1"]["components"]["combat"] <= cfg.human_death + cfg.death
    # ...and the episode continues: succession exercises
    assert env.outcome is None
    assert env.roster.root() is not None
    assert env.roster.root().effective_rank is Rank.TL


def test_non_human_death_costs_only_teammate_death():
    from dataclasses import replace

    from cohort.config import get_scenario

    env = make_env(replace(get_scenario("fireteam"), root_human=False))
    env.reset(seed=5)
    env.world.grid[:] = 0
    tl = env.roster.by_callsign["TL1"]
    assert not tl.human
    enemy = next(e for e in env.enemies if e.alive)
    tl.health = 1
    tl.pos = (enemy.pos[0] + 1, enemy.pos[1])
    for cs in ("RFN1", "RFN2", "RFN3"):
        env.roster.by_callsign[cs].pos = (1, 1)
    infos = None
    for _ in range(30):
        if not tl.alive:
            break
        *_, infos = env.step({a: 0 for a in env.agents})
    assert not tl.alive
    for cs in ("RFN1", "RFN2", "RFN3"):
        combat = infos[cs]["components"]["combat"]
        assert combat > env.rewards_cfg.human_death / 2, (
            f"{cs}: no human died — only the ordinary teammate penalty applies, got {combat}"
        )
