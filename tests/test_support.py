"""SUPPORT (APPUYER): the unit-targeted fire-support mission.

The order names a friendly element, not an objective: ``ORDER_S{i}_SUPPORT_U{j}``
tasks the subordinate in slot i to support the unit led by the subordinate in
slot j. The mission anchor is dynamic — it tracks the supported soldier's
position (like RALLY tracks the leader) — and the mission ends on re-tasking
or on the supported unit's death (auto-clear with a notice on the net).
"""

from cohort import make_env
from cohort.core.missions import MissionType
from cohort.env.actions import CATALOG

STAY = 0


def _support_spec(slot, unit):
    return next(
        s
        for s in CATALOG
        if s.kind == "order"
        and s.order_mission is MissionType.SUPPORT
        and s.order_slot == slot
        and s.order_support_slot == unit
    )


def _flat_squad(seed=1):
    """Squad env, open terrain, enemies parked far away."""
    env = make_env("squad")
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (26, 1)
        e.home = e.pos
        e.prev_pos = e.pos
    return env


def _step_all(env, overrides=None):
    acts = {a: STAY for a in env.agents}
    acts.update(overrides or {})
    return env.step(acts)


def test_support_order_lands_with_unit_target():
    """SL1 orders TL1 (slot 0) to support TL2's team (slot 1)."""
    env = _flat_squad()
    spec = _support_spec(0, 1)
    _step_all(env, {"SL1": spec.index})
    tl1, tl2 = env.roster.by_callsign["TL1"], env.roster.by_callsign["TL2"]
    assert tl1.mission is not None and tl1.mission.type is MissionType.SUPPORT
    assert tl1.mission.objective_id is None
    assert tl1.mission.extra["supported_id"] == tl2.id
    order = next(m for m in env.transcript.messages if m.kind.value == "order")
    assert order.text == "TL1, THIS IS SL1: SUPPORT TL2. OUT."


def test_support_anchor_tracks_the_supported_soldier():
    env = _flat_squad()
    spec = _support_spec(0, 1)
    _step_all(env, {"SL1": spec.index})
    tl1, tl2 = env.roster.by_callsign["TL1"], env.roster.by_callsign["TL2"]
    tl2.pos = (20, 20)
    assert env._mission_anchor(tl1) == (20, 20), "anchor follows the supported soldier"
    tl2.pos = (5, 9)
    assert env._mission_anchor(tl1) == (5, 9)


def test_support_in_position_needs_range_and_los():
    env = _flat_squad()
    spec = _support_spec(0, 1)
    _step_all(env, {"SL1": spec.index})
    tl1, tl2 = env.roster.by_callsign["TL1"], env.roster.by_callsign["TL2"]
    tl2.pos = (10, 10)
    tl1.pos = (14, 10)  # within 10 cells, open ground → LOS
    assert env._in_mission_position(tl1)
    tl1.pos = (24, 10)  # out of range
    assert not env._in_mission_position(tl1)
    # back in range but a wall blocks the line of sight
    from cohort.core.world import WALL

    tl1.pos = (14, 10)
    env.world.grid[10, 12] = WALL
    assert not env._in_mission_position(tl1), "no LOS to the supported unit"


def test_support_ends_when_supported_unit_dies():
    env = _flat_squad()
    spec = _support_spec(0, 1)
    _step_all(env, {"SL1": spec.index})
    tl1, tl2 = env.roster.by_callsign["TL1"], env.roster.by_callsign["TL2"]
    assert tl1.mission is not None
    # kill TL2: park it next to a live enemy with 1 hp
    enemy = env.enemies[0]
    enemy.pos = (26, 3)
    enemy.home = enemy.pos
    tl2.health = 1
    tl2.pos = (26, 2)
    for _ in range(30):
        if not tl2.alive:
            break
        _step_all(env)
    assert not tl2.alive, "the adjacent enemy should have killed TL2"
    assert tl1.mission is None, "SUPPORT auto-clears when the supported unit falls"
    notice = next(m for m in env.transcript.messages if m.kind.value == "support_end")
    assert "SUPPORT ENDED, TL2 IS DOWN" in notice.text
    assert notice.sender_id == tl1.id


def test_support_via_injection():
    env = _flat_squad()
    env.inject_order("TL2, support TL1", issuer="HQ")
    tl2 = env.roster.by_callsign["TL2"]
    assert tl2.mission is not None and tl2.mission.type is MissionType.SUPPORT
    assert tl2.mission.extra["supported_id"] == env.roster.by_callsign["TL1"].id
    assert any(
        m.text == "TL2, THIS IS HQ: SUPPORT TL1. OUT." for m in env.transcript.messages
    )


def test_support_observation_anchor_is_dynamic():
    """The mission-anchor direction in the observation tracks the supported unit."""
    import numpy as np

    env = _flat_squad()
    spec = _support_spec(0, 1)
    _step_all(env, {"SL1": spec.index})
    tl1, tl2 = env.roster.by_callsign["TL1"], env.roster.by_callsign["TL2"]
    tl2.pos = (tl1.pos[0] + 6, tl1.pos[1])  # due east of the supporter
    obs = env._all_observations()["TL1"]["observation"]
    # mission block: 12 self, then 11 one-hot + 1 flag, then anchor dx at +2
    dx = obs[12 + 11 + 1]
    dy = obs[12 + 11 + 2]
    assert dx > 0 and abs(dy) < 1e-6, "anchor direction must point at the supported unit"
    assert np.isclose(dx, 6 / env.world.width)
