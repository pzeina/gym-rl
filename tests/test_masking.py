"""Rank admissibility is a hard guarantee: masks, not hopes."""

import numpy as np

from cohort import make_env
from cohort.core.missions import DOCTRINE, MissionType
from cohort.env.actions import CATALOG, MOVES, N_ACTIONS

ORDER_INDICES = [s.index for s in CATALOG if s.kind == "order"]


def test_catalog_is_stable():
    assert len(CATALOG) == N_ACTIONS
    assert CATALOG[0].kind == "stay"
    names = [s.name for s in CATALOG]
    assert len(names) == len(set(names)), "action names must be unique"


def test_riflemen_can_never_command():
    """Across many random steps, no rifleman ever has a legal order action."""
    env = make_env("fireteam")
    obs, _ = env.reset(seed=11)
    rng = np.random.default_rng(0)
    for _ in range(60):
        if not env.agents:
            obs, _ = env.reset()
        for agent in env.agents:
            if agent.startswith("RFN"):
                soldier = env.roster.by_callsign[agent]
                if soldier.effective_rank.name == "RFN":  # not promoted by succession
                    assert obs[agent]["action_mask"][ORDER_INDICES].sum() == 0
        acts = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
        obs, *_ = env.step(acts)


def test_leader_orders_are_doctrine_constrained():
    env = make_env("fireteam")
    obs, _ = env.reset(seed=3)
    allowed = DOCTRINE[MissionType.SEIZE]  # TL1 holds the SEIZE OPORD
    mask = obs["TL1"]["action_mask"]
    for spec in CATALOG:
        if spec.kind == "order" and mask[spec.index]:
            assert spec.order_mission in allowed, (
                f"{spec.name} legal but {spec.order_mission} not derivable from SEIZE"
            )
    # and at least the preferred derivation is available
    legal_missions = {
        spec.order_mission for spec in CATALOG if spec.kind == "order" and mask[spec.index]
    }
    assert allowed[0] in legal_missions


def test_agent_without_mission_cannot_order():
    env = make_env("squad")
    obs, _ = env.reset(seed=3)
    # fire-team leaders have no mission yet at t=0 (only the SL holds the OPORD)
    assert obs["TL1"]["action_mask"][ORDER_INDICES].sum() == 0
    assert obs["SL1"]["action_mask"][ORDER_INDICES].sum() > 0


def test_deny_never_reaches_group_level():
    """Per-echelon admissibility at the mask: a leader cannot give DENY to a
    TL/RFN. Doctrine already derives DENY to nobody; even if it did (patched
    here), the min-hold-authority check blocks recipients below section level."""
    env = make_env("squad")
    obs, _ = env.reset(seed=3)
    env.inject_order("SL1, deny obj alpha", issuer="HQ")
    obs = env._all_observations()
    legal = {
        spec.order_mission
        for spec in CATALOG
        if spec.kind == "order"
        and spec.order_mission is not None  # A5-3 stance orders carry no mission
        and obs["SL1"]["action_mask"][spec.index]
    }
    assert MissionType.DENY not in legal, "doctrine derives DENY to nobody"
    assert legal <= set(DOCTRINE[MissionType.DENY])

    # belt and braces: even with doctrine patched to allow DENY→DENY, the
    # admissibility check keeps it off the mask for sub-section recipients
    original = DOCTRINE[MissionType.DENY]
    DOCTRINE[MissionType.DENY] = (MissionType.DENY, *original)
    try:
        obs = env._all_observations()
        legal = {
            spec.order_mission
            for spec in CATALOG
            if spec.kind == "order"
            and spec.order_mission is not None
            and obs["SL1"]["action_mask"][spec.index]
        }
        assert MissionType.DENY not in legal, "TL recipients are below min hold authority"
    finally:
        DOCTRINE[MissionType.DENY] = original


def test_support_orders_need_a_living_supported_unit():
    """ORDER_S{i}_SUPPORT_U{j} is legal only when both slots hold living subs."""
    env = make_env("squad")
    obs, _ = env.reset(seed=3)
    sl = env.roster.by_callsign["SL1"]
    assert sl.mission is not None  # holds the OPORD (SEIZE → SUPPORT derivable)
    support_specs = [
        s for s in CATALOG if s.kind == "order" and s.order_mission is MissionType.SUPPORT
    ]
    mask = obs["SL1"]["action_mask"]
    n_subs = len(sl.living_subordinates(env.roster))  # 2 fire-team leaders
    for spec in support_specs:
        expected = spec.order_slot < n_subs and spec.order_support_slot < n_subs
        assert bool(mask[spec.index]) == expected, spec.name


def test_fire_requires_visible_enemy():
    env = make_env("fireteam")
    obs, _ = env.reset(seed=7)
    fire_idx = next(s.index for s in CATALOG if s.kind == "fire")
    for agent in env.agents:
        soldier = env.roster.by_callsign[agent]
        visible = env._visible_enemies(soldier)
        in_range = any(
            np.hypot(e.pos[0] - soldier.pos[0], e.pos[1] - soldier.pos[1]) <= env.combat.weapon_range
            for e in visible
        )
        assert bool(obs[agent]["action_mask"][fire_idx]) == in_range


def test_moves_blocked_by_walls():
    env = make_env("fireteam")
    obs, _ = env.reset(seed=13)
    for agent in env.agents:
        soldier = env.roster.by_callsign[agent]
        mask = obs[agent]["action_mask"]
        for spec in CATALOG:
            if spec.kind == "move":
                nxt = (soldier.pos[0] + spec.move[0], soldier.pos[1] + spec.move[1])
                assert bool(mask[spec.index]) == env.world.passable(nxt)


def test_stay_is_always_legal():
    env = make_env("squad")
    obs, _ = env.reset(seed=1)
    for agent in env.agents:
        assert obs[agent]["action_mask"][0] == 1
    assert set(MOVES) == {"NORTH", "SOUTH", "EAST", "WEST"}
