"""Defensive-scenario terrain doctrine: prepared positions + early warning."""

from dataclasses import replace

import pytest

from cohort import make_env
from cohort.config import get_scenario
from cohort.core.world import FOREST, dist


def test_objective_cover_rings_the_objective():
    env = make_env("fireteam_defend")
    env.reset(seed=5)
    obj = env.world.objectives[0]
    ox, oy = obj.pos
    ring = [
        (ox + dx, oy + dy)
        for dx in range(-2, 3)
        for dy in range(-2, 3)
        if max(abs(dx), abs(dy)) == 2 and env.world.in_bounds((ox + dx, oy + dy))
    ]
    forest_cells = sum(env.world.grid[y, x] == FOREST for x, y in ring)
    assert forest_cells >= len(ring) // 2, "the objective ring must offer cover"
    assert env.world.grid[oy, ox] != FOREST, "the objective center stays clear"


def test_assault_spawns_respect_early_warning_distance():
    env = make_env("fireteam_defend")
    for seed in range(4):
        env.reset(seed=seed)
        obj = env.world.objectives[0]
        for e in env.enemies:
            assert dist(e.pos, obj.pos) >= env.spec_cfg.assault_spawn_min_dist - 1e-9


def test_objective_lost_pressure_on_defend_roots():
    """While a living enemy stands on the DEFEND root objective, every living
    agent bleeds objective_lost per step; the pressure stops when the ground
    is regained, and never applies to non-DEFEND/DENY roots."""
    env = make_env("fireteam_defend")
    env.reset(seed=3)
    env.world.grid[:] = 0
    obj = env.world.objectives[0]
    cfg = env.rewards_cfg
    for s in env.roster.soldiers:
        s.pos = (2, 2)  # defenders hiding in a corner
    intruder = env.enemies[0]
    for e in env.enemies:
        e.pos = (30, 30)
        e.home = e.pos
    intruder.pos = obj.pos  # the enemy owns the objective
    intruder.home = intruder.pos
    *_, infos = env.step({a: 0 for a in env.agents})
    for a in infos:
        assert infos[a]["components"]["compliance"] <= cfg.objective_lost, (
            f"{a}: hiding while the enemy holds the objective must bleed"
        )
    # ground regained → the pressure stops
    intruder.pos = (30, 30)
    intruder.home = intruder.pos
    *_, infos = env.step({a: 0 for a in env.agents})
    for a in infos:
        assert infos[a]["components"]["compliance"] > cfg.objective_lost

    # a SEIZE root is not a defense: no pressure even with enemies at the objective
    env2 = make_env("fireteam")
    env2.reset(seed=3)
    env2.world.grid[:] = 0
    for s in env2.roster.soldiers:
        s.pos = (2, 2)
        s.mission = None
    obj2 = env2.world.objectives[0]
    env2.enemies[0].pos = obj2.pos
    env2.enemies[0].home = obj2.pos
    for e in env2.enemies[1:]:
        e.pos = (2, 30)
        e.home = e.pos
    *_, infos = env2.step({a: 0 for a in env2.agents})
    for a in infos:
        assert infos[a]["components"]["compliance"] == 0.0


def test_knobs_default_off_elsewhere():
    env = make_env("fireteam")  # garrison scenario: no defensive terrain doctrine
    spec = get_scenario("fireteam")
    assert spec.objective_cover is False
    env.reset(seed=1)
    # and a defend spec with the knob off leaves the map untouched by the ring
    bare = replace(get_scenario("fireteam_defend"), objective_cover=False)
    env2 = make_env(bare)
    env2.reset(seed=5)
    obj = env2.world.objectives[0]
    ox, oy = obj.pos
    env3 = make_env(get_scenario("fireteam_defend"))
    env3.reset(seed=5)
    ring_bare = sum(
        env2.world.grid[oy + dy, ox + dx] == FOREST
        for dx in range(-2, 3) for dy in range(-2, 3) if max(abs(dx), abs(dy)) == 2
    )
    ring_prepared = sum(
        env3.world.grid[oy + dy, ox + dx] == FOREST
        for dx in range(-2, 3) for dy in range(-2, 3) if max(abs(dx), abs(dy)) == 2
    )
    assert ring_prepared > ring_bare


def test_observation_concealment_places_ops():
    from cohort.core.world import FOREST

    env = make_env("squad_recon")
    env.reset(seed=7)
    obj = env.world.objective_by_name("BRAVO")
    ox, oy = obj.pos
    ring_forest = sum(
        1
        for y in range(env.world.height)
        for x in range(env.world.width)
        if env.world.grid[y, x] == FOREST and 4.5 <= dist((x, y), (ox, oy)) <= 7.5
    )
    assert ring_forest >= 10, "concealed OPs must exist on the observation ring"


def test_exposed_under_threat_prices_static_tasks_out_of_cover():
    """The squad_screen diagnosis (16/16 deaths out of cover): a static-tasked
    soldier out of cover with a living enemy in weapon range bleeds
    exposed_under_threat per step; cover stops it, distance stops it, a moving
    task never pays it, and the default price is 0 (an experiment arm's price,
    not yet the environment's)."""
    from cohort.core.missions import Mission, MissionType
    from cohort.env.cohort_env import STATIC_EXPOSURE_MISSIONS
    from cohort.env.rewards import RewardConfig

    assert RewardConfig().exposed_under_threat == 0.0

    arm_price = -0.02

    def compliance_of(price: float, *, covered: bool, in_range: bool) -> float:
        """One step of a static OBSERVE watcher; returns its compliance bucket.

        Rebuilt per call so the ONLY difference between two calls is the
        (price, cover, range) triple — posture-compliance credit is identical
        across them, and the delta is exactly the exposure price or zero.
        """
        env = make_env("fireteam")
        env.reset(seed=5)
        env.rewards_cfg = replace(env.rewards_cfg, exposed_under_threat=price)
        env.world.grid[:] = 0
        watcher = env.roster.soldiers[-1]
        for s in env.roster.soldiers:
            s.pos = (2, 2)
            s.mission = None
        watcher.pos = (5, 5)
        if covered:
            env.world.grid[5, 5] = 1
            assert env.world.cover_at(watcher.pos)
        watcher.mission = Mission(
            type=MissionType.OBSERVE, objective_id=None, anchor=(5.0, 5.0),
            issuer_id=-1, step_assigned=0,
        )
        threatener = env.enemies[0]
        for e in env.enemies:
            e.pos = (35, 35)
            e.home = e.pos
        if in_range:
            threatener.pos = (5, 5 + int(env.combat.weapon_range))
            threatener.home = threatener.pos
        *_, infos = env.step({a: 0 for a in env.agents})
        return infos[watcher.callsign]["components"]["compliance"]

    baseline = compliance_of(0.0, covered=False, in_range=True)

    # static task, out of cover, enemy in weapon range: bleeds exactly the price
    # on top of whatever posture credit the mission earns
    exposed = compliance_of(arm_price, covered=False, in_range=True)
    assert exposed == pytest.approx(baseline + arm_price)

    # cover stops the bleed — the marginal value of the cover cell is the full price
    assert compliance_of(arm_price, covered=True, in_range=True) == pytest.approx(baseline)

    # ...and so does distance
    assert compliance_of(arm_price, covered=False, in_range=False) == pytest.approx(baseline)

    # the priced set is exactly the measured one
    assert {
        MissionType.OBSERVE, MissionType.SCREEN, MissionType.HOLD, MissionType.COVER,
    } == STATIC_EXPOSURE_MISSIONS
