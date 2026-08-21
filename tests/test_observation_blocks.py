"""The v1.10 observation blocks: tempo, nearest cover, and the derived offsets.

The layout is indexed through the exported ``OFF_*`` constants rather than
magic numbers, so a future block change surfaces as an OBS_DIM assertion
failure (a real signal) instead of a scatter of silently-wrong offsets.
"""

import numpy as np

from cohort import make_env
from cohort.core.world import FOREST, OPEN, World
from cohort.env.observations import (
    COVER_PRESENT,
    OBS_DIM,
    OFF_PATCH,
    PATCH_RADIUS,
    TEMPO_PROGRESS,
    TEMPO_TIME_TO_CONTACT,
)


def _flat_world(w=20, h=20):
    return World(np.zeros((h, w), dtype=np.int8), [], [], [])


def test_nearest_cover_finds_the_closest_cell():
    world = _flat_world()
    world.grid[10, 14] = FOREST  # (x=14, y=10)
    world.grid[10, 5] = FOREST   # (x=5,  y=10) — closer to (8, 10)
    assert world.nearest_cover((8, 10), radius=8) == (5, 10)
    # standing in cover: distance 0, returns the agent's own cell
    assert world.nearest_cover((5, 10), radius=8) == (5, 10)
    # nothing within the radius
    assert world.nearest_cover((8, 10), radius=2) is None


def test_nearest_cover_is_deterministic_and_scan_order_free():
    """Equidistant cover must resolve identically every call (determinism)."""
    world = _flat_world()
    for pos in ((7, 10), (9, 10), (8, 9), (8, 11)):  # four cells, all distance 1
        world.grid[pos[1], pos[0]] = FOREST
    first = world.nearest_cover((8, 10), radius=8)
    assert all(world.nearest_cover((8, 10), radius=8) == first for _ in range(20))


def test_cover_block_points_at_the_nearest_cover():
    env = make_env("fireteam_defend")
    obs, _ = env.reset(seed=3)
    tl = env.roster.by_callsign["TL1"]
    vec = obs["TL1"]["observation"]
    cover = env.world.nearest_cover(tl.pos, 8)
    assert cover is not None, "objective_cover guarantees a ring near the spawn"
    assert vec[COVER_PRESENT] == 1.0
    assert vec[COVER_PRESENT + 1] == np.float32((cover[0] - tl.pos[0]) / env.world.width)
    assert vec[COVER_PRESENT + 2] == np.float32((cover[1] - tl.pos[1]) / env.world.height)


def test_cover_block_zeroed_when_no_cover_is_near():
    env = make_env("fireteam")
    env.reset(seed=3)
    env.world.grid[env.world.grid == FOREST] = OPEN  # strip every forest cell
    obs = env._all_observations()
    for cs in env.agents:
        assert obs[cs]["observation"][COVER_PRESENT] == 0.0, f"{cs}: no cover to point at"


def test_episode_progress_advances_with_the_step_count():
    env = make_env("fireteam")
    obs, _ = env.reset(seed=1)
    assert obs["TL1"]["observation"][TEMPO_PROGRESS] == 0.0
    prev = 0.0
    for _ in range(10):
        obs, *_ = env.step({a: 0 for a in env.agents})
        cur = obs["TL1"]["observation"][TEMPO_PROGRESS]
        assert cur > prev, "progress is monotone"
        prev = cur
    assert prev == np.float32(10 / env.spec_cfg.max_steps)


def test_time_to_contact_is_zero_without_a_preparation_period():
    for name in ("fireteam", "squad", "platoon"):
        env = make_env(name)
        obs, _ = env.reset(seed=1)
        for cs in env.agents:
            assert obs[cs]["observation"][TEMPO_TIME_TO_CONTACT] == 0.0, name


def test_patch_block_precedes_the_degraded_comms_blocks_and_is_sized_by_the_radius():
    """The patch used to be the last block; the degraded-communications cycle
    appended the acoustic and cohesion blocks after it (spec §5: append, so
    nothing before them moves)."""
    from cohort.env.observations import OFF_ACOUSTIC, OFF_COHESION

    env = make_env("fireteam")
    obs, _ = env.reset(seed=1)
    tl = env.roster.by_callsign["TL1"]
    vec = obs["TL1"]["observation"]
    patch = env.world.local_patch(tl.pos, PATCH_RADIUS).reshape(-1)
    assert OFF_PATCH + patch.shape[0] == OFF_ACOUSTIC < OFF_COHESION < OBS_DIM
    assert np.array_equal(vec[OFF_PATCH:OFF_ACOUSTIC], patch)
    # a radio scenario with sound off: the appended blocks are structurally
    # unavailable and read as zeros, except the cohesion link/perception
    # flags that are measured in every mode
    assert not vec[OFF_ACOUSTIC:OFF_COHESION].any()
