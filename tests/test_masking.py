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
    """Across many random steps, no SLD ever has a legal order action."""
    env = make_env("fireteam")
    obs, _ = env.reset(seed=11)
    rng = np.random.default_rng(0)
    for _ in range(60):
        if not env.agents:
            obs, _ = env.reset()
        for agent in env.agents:
            if agent.startswith("SLD"):
                soldier = env.roster.by_callsign[agent]
                if soldier.effective_rank.name == "SLD":  # not promoted by succession
                    assert obs[agent]["action_mask"][ORDER_INDICES].sum() == 0
        acts = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
        obs, *_ = env.step(acts)


def test_leader_orders_are_doctrine_constrained():
    env = make_env("fireteam")
    obs, _ = env.reset(seed=3)
    allowed = DOCTRINE[MissionType.SEIZE]  # CAP1 holds the SEIZE OPORD
    mask = obs["CAP1"]["action_mask"]
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
    # fire-team leaders have no mission yet at t=0 (only the CDG holds the OPORD)
    assert obs["CAP1"]["action_mask"][ORDER_INDICES].sum() == 0
    assert obs["CDG1"]["action_mask"][ORDER_INDICES].sum() > 0


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
