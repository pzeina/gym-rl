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


# ---------------------------------------------------------------------- #
# P2 — SUPPORT mechanics: covered movement + focus fire
# ---------------------------------------------------------------------- #


def _give_support(env, supporter_cs, supported_cs):
    from cohort.core.missions import Mission

    supporter = env.roster.by_callsign[supporter_cs]
    supported = env.roster.by_callsign[supported_cs]
    supporter.mission = Mission(
        MissionType.SUPPORT, None, supported.pos, issuer_id=-1, step_assigned=0,
        extra={"supported_id": supported.id},
    )
    return supporter, supported


def _enemy_shots_at_supported(seed, supporter_in_position):
    """Fixed-seed run: an enemy shoots RFN3 (member of TL2's supported
    element) 60 times; count the hits. Same seed → identical RNG draws in
    both variants, so the only difference is the covered-movement debuff."""
    env = _flat_squad(seed=seed)
    tl1, tl2 = _give_support(env, "TL1", "TL2")
    rfn3 = env.roster.by_callsign["RFN3"]
    enemy = env.enemies[0]
    # park everyone far except the actors
    for s in env.roster.soldiers:
        s.pos = (2, 26)
    tl2.pos = (10, 10)
    rfn3.pos = (12, 10)
    enemy.pos = (13, 10)  # adjacent to RFN3: fires every step, RFN3 nearest
    enemy.home = enemy.pos
    enemy.prev_pos = enemy.pos
    # supporter: within 10+LOS of TL2 and within 8 of the enemy — or far away
    tl1.pos = (10, 12) if supporter_in_position else (2, 2)
    for e in env.enemies[1:]:
        e.alive = False

    hits = 0
    for _ in range(60):
        rfn3.health = 100  # reset so it survives every volley
        rfn3.alive = True
        _step_all(env)
        if rfn3.health < 100:
            hits += 1
    return hits


def test_covered_movement_makes_the_supported_element_safer():
    """Same seed, same draws: the umbrella (accuracy x0.7) must strictly
    reduce enemy hits on the supported element."""
    covered = _enemy_shots_at_supported(seed=11, supporter_in_position=True)
    uncovered = _enemy_shots_at_supported(seed=11, supporter_in_position=False)
    assert covered < uncovered, f"covered {covered} !< uncovered {uncovered}"
    assert uncovered - covered >= 5, "the x0.7 debuff should be clearly measurable"


def test_covered_movement_off_when_supporter_out_of_position():
    """Direct check of the umbrella predicate."""
    env = _flat_squad()
    tl1, tl2 = _give_support(env, "TL1", "TL2")
    rfn3 = env.roster.by_callsign["RFN3"]
    tl2.pos = (10, 10)
    rfn3.pos = (12, 10)
    tl1.pos = (10, 12)
    _step_all(env)  # snapshot the umbrella
    assert env._covered_by_support(rfn3, (13, 10)), "enemy inside the umbrella"
    assert not env._covered_by_support(rfn3, (25, 25)), "enemy outside the 8-cell umbrella"
    rfn1 = env.roster.by_callsign["RFN1"]
    assert not env._covered_by_support(rfn1, (13, 10)), "TL1's own team is not the supported element"
    tl1.pos = (26, 12)  # walk off the support station
    _step_all(env)
    assert not env._covered_by_support(rfn3, (13, 10)), "effects OFF when out of position"


def _volley_damage(seed, with_support):
    """Fixed-seed run: RFN1 and RFN2 both fire point-blank at an undying
    enemy each step; total damage dealt is returned. The target never dies
    and every step consumes the same number of RNG draws in both variants
    (two friendly shots + one enemy return shot), so the draw sequences are
    identical — the ONLY difference is the focus-fire x1.15 on the second
    shot, which can turn a miss into a hit but never the reverse."""
    env = _flat_squad(seed=seed)
    if with_support:
        tl1, tl2 = _give_support(env, "TL1", "TL2")
        tl2.pos = (10, 20)
        tl1.pos = (10, 22)  # in position: support active, focus fire enabled
    rfn1 = env.roster.by_callsign["RFN1"]
    rfn2 = env.roster.by_callsign["RFN2"]
    rfn1.pos = (11, 10)
    rfn2.pos = (13, 10)
    enemy = env.enemies[0]
    enemy.home = (12, 10)
    for e in env.enemies[1:]:
        e.alive = False

    fire = next(s.index for s in CATALOG if s.kind == "fire")
    damage = 0
    for _ in range(80):
        enemy.alive = True
        enemy.health = 1000  # never dies: keeps the RNG streams aligned
        enemy.pos = (12, 10)
        enemy.prev_pos = enemy.pos
        rfn1.ammo = rfn2.ammo = 30
        rfn1.health = rfn2.health = 100
        rfn1.alive = rfn2.alive = True
        _step_all(env, {"RFN1": fire, "RFN2": fire})
        damage += 1000 - enemy.health
    return damage


def test_focus_fire_bonus_on_the_second_shooter():
    """Same seed, same draws: with support active the second shooter's hit
    probability is x1.15 — hits are a strict superset, damage strictly more."""
    boosted = _volley_damage(seed=23, with_support=True)
    plain = _volley_damage(seed=23, with_support=False)
    assert boosted > plain, f"boosted {boosted} !> plain {plain}"


def test_resolve_fire_modifier_and_cap():
    """The modifier plumbs into the hit probability; the cap holds at 0.95."""
    from cohort.core.units import CombatParams, resolve_fire

    class Draw:
        def __init__(self, v):
            self.v = v

        def random(self):
            return self.v

    params = CombatParams()
    # point blank (d=0): p = 0.85; x1.15 = 0.9775 → capped at 0.95
    hit, _ = resolve_fire((0, 0), (0, 0), False, 0.0, params, Draw(0.949), modifier=1.15)
    assert hit, "0.949 < capped 0.95 must hit"
    hit, _ = resolve_fire((0, 0), (0, 0), False, 0.0, params, Draw(0.951), modifier=10.0)
    assert not hit, "the cap keeps any modifier below 0.95"
    # debuff: p(d=1) = 0.78 → x0.7 = 0.546
    hit, _ = resolve_fire((0, 0), (1, 0), False, 1.0, params, Draw(0.6), modifier=0.7)
    assert not hit, "0.6 >= 0.546: the debuffed shot misses"
    hit, _ = resolve_fire((0, 0), (1, 0), False, 1.0, params, Draw(0.6), modifier=1.0)
    assert hit, "0.6 < 0.78: the same draw hits without the debuff"


def test_support_observation_anchor_is_dynamic():
    """The mission-anchor direction in the observation tracks the supported unit."""
    import numpy as np

    env = _flat_squad()
    spec = _support_spec(0, 1)
    _step_all(env, {"SL1": spec.index})
    tl1, tl2 = env.roster.by_callsign["TL1"], env.roster.by_callsign["TL2"]
    tl2.pos = (tl1.pos[0] + 6, tl1.pos[1])  # due east of the supporter
    obs = env._all_observations()["TL1"]["observation"]
    # mission block: 13 self, then 11 one-hot + 1 flag, then anchor dx/dy
    dx = obs[13 + 12 + 1]
    dy = obs[13 + 12 + 2]
    assert dx > 0 and abs(dy) < 1e-6, "anchor direction must point at the supported unit"
    assert np.isclose(dx, 6 / env.world.width)
