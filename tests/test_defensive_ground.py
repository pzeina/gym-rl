"""Defensive-scenario terrain doctrine: prepared positions + early warning."""

from dataclasses import replace

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
