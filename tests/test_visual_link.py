"""Hierarchical visual-link graph and formation-station mechanics
(docs/degraded-communications.md §3.7, §8 "Visual-link and formation")."""

from dataclasses import replace

from cohort import make_env
from cohort.config import get_scenario
from cohort.core import cohesion
from cohort.core.world import WALL
from cohort.env.actions import CATALOG
from cohort.env.observations import COHESION_BREAK_AGE, COHESION_LINK

STAY = 0
MOVE_EAST = next(s.index for s in CATALOG if s.name == "MOVE_EAST")


def _env(comm="voice_only", seed=1):
    spec = replace(
        get_scenario("squad"), name="squad_link_test", comm_model=comm,
        voice_range=2.0 if comm == "voice_only" else 6.0,
    )
    env = make_env(spec)
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (40, 40)
        e.home = e.pos
    # a compact squad: SL1 with TL1 (RFN1, RFN2) and TL2 (RFN3, RFN4)
    for cs, pos in {"SL1": (10, 10), "TL1": (12, 10), "RFN1": (13, 10), "RFN2": (12, 11),
                    "TL2": (10, 12), "RFN3": (10, 13), "RFN4": (11, 13)}.items():
        env.roster.by_callsign[cs].pos = pos
    env._update_visual_links()
    # The bunching price is OFF in this fixture. These agents are placed on
    # adjacent cells to exercise a different channel entirely, and the tests
    # below assert an exact per-step total — a second always-on term would
    # make them assertions about two mechanisms at once. The price has its own
    # suite in tests/test_bunching_price.py.
    env.rewards_cfg = replace(env.rewards_cfg, bunching_penalty=0.0)
    return env


def _step(env, overrides=None):
    acts = {a: STAY for a in env.agents}
    if overrides:
        acts.update(overrides)
    return env.step(acts)


def test_friendly_visible_requires_finite_range_and_los():
    env = _env()
    a, b = env.roster.by_callsign["SL1"], env.roster.by_callsign["TL1"]
    assert cohesion.friendly_visible(env.world, a, b)
    b.pos = (19, 10)  # 9 cells: beyond VISUAL_LINK_RANGE=8
    assert not cohesion.friendly_visible(env.world, a, b)
    b.pos = (14, 10)
    env.world.grid[10, 12] = WALL
    assert not cohesion.friendly_visible(env.world, a, b), "a wall breaks the edge"


def test_sibling_relay_links_but_another_element_never_does():
    env = _env()
    # RFN2 can only see RFN1 (sibling), RFN1 sees TL1: linked through the sibling
    env.roster.by_callsign["TL1"].pos = (20, 10)
    env.roster.by_callsign["RFN1"].pos = (26, 10)
    env.roster.by_callsign["RFN2"].pos = (32, 10)
    env._update_visual_links()
    assert env._link_state["RFN2"][0] is True
    assert env._link_state["TL1"][0] is True  # element intact
    # now put TL2's rifleman RFN3 as the only bridge: a DIFFERENT element
    env.roster.by_callsign["RFN1"].pos = (40, 1)   # gone
    env.roster.by_callsign["RFN3"].pos = (26, 10)  # TL2's man sits where RFN1 was
    env._update_visual_links()
    assert env._link_state["RFN2"][0] is False, "a different element cannot be a relay"
    assert env._link_state["TL1"][0] is False


def test_break_is_measured_on_its_first_tick_and_the_element_cap_bounds_the_penalty():
    env = _env()
    cfg = env.rewards_cfg
    # all three of TL2's... TL1 has 2 members; make every member of BOTH teams break
    for cs, pos in {"RFN1": (30, 30), "RFN2": (35, 35), "RFN3": (30, 1), "RFN4": (35, 1)}.items():
        env.roster.by_callsign[cs].pos = pos
    _, rewards, *_ = _step(env)
    assert env._link_state["RFN1"] == (False, 1), "first tick counts"
    for cs in ("RFN1", "RFN2", "RFN3", "RFN4"):
        assert rewards[cs] <= cfg.time_penalty + cfg.visual_link_broken + 1e-9
    # the cap: four broken members under one leader cannot exceed the element cap
    spec = replace(get_scenario("platoon"), name="pl_link", comm_model="voice_only", voice_range=2.0)
    pl = make_env(spec)
    pl.reset(seed=2)
    pl.world.grid[:] = 0
    for e in pl.enemies:
        e.pos = (50, 50)
        e.home = e.pos
    root = pl.roster.root()
    subs = root.living_subordinates(pl.roster)
    assert len(subs) >= 3
    root.pos = (5, 5)
    for k, s in enumerate(subs):
        s.pos = (5 + 12 * (k + 1), 40)
    pl._update_visual_links()
    _, rewards, *_ = _step(pl)
    total = sum(rewards[s.callsign] - pl.rewards_cfg.time_penalty for s in subs)
    # subordinates of the root may hold no mission: their only other term is this one
    assert total >= pl.rewards_cfg.visual_link_broken_element_cap - 1e-9
    assert total < 0


def test_stay_earns_no_cohesion_reward_and_no_move_is_masked_for_formation():
    env = _env()
    _, rewards, *_ = _step(env)
    for cs in ("RFN1", "RFN2"):
        assert rewards[cs] == env.rewards_cfg.time_penalty, "intact link pays nothing"
    # every passable move stays legal whatever it does to the link
    rfn1 = env.roster.by_callsign["RFN1"]
    mask = env._mask_for(rfn1)
    assert mask[MOVE_EAST] == 1


def test_link_state_is_observable_and_ages_while_broken():
    env = _env()
    rfn1 = env.roster.by_callsign["RFN1"]
    obs = env._observe(rfn1, env._make_view(rfn1))["observation"]
    assert obs[COHESION_LINK] == 1.0 and obs[COHESION_BREAK_AGE] == 0.0
    rfn1.pos = (30, 30)
    for _ in range(4):
        obs, *_ = _step(env)
    assert obs["RFN1"]["observation"][COHESION_LINK] == 0.0
    assert abs(obs["RFN1"]["observation"][COHESION_BREAK_AGE] - 4 / 20) < 1e-6


def test_radio_modes_measure_the_link_but_never_price_it():
    env = _env(comm="global")
    env.roster.by_callsign["RFN1"].pos = (30, 30)
    _, rewards, *_ = _step(env)
    assert env._link_state["RFN1"][0] is False
    assert rewards["RFN1"] == env.rewards_cfg.time_penalty


def test_casualties_and_succession_rebuild_the_graph_without_dead_nodes():
    env = _env()
    tl1 = env.roster.by_callsign["TL1"]
    tl1.health = 1
    from cohort.core.units import Trap

    env.traps = [Trap(id=0, pos=(13, 10), damage=50)]
    # RFN1 stands on (13,10); move TL1 east onto the trap
    env.roster.by_callsign["RFN1"].pos = (14, 10)
    env._update_visual_links()
    _step(env, {"TL1": MOVE_EAST})
    assert not tl1.alive
    assert "TL1" not in env._link_state
    successor = env.roster.by_id[tl1.id] if tl1.alive else next(
        s for s in env.roster.living if s.effective_rank.name == "TL" and s.callsign.startswith("RFN")
    )
    assert env._link_state[successor.callsign][0] is not None


def test_formation_station_is_reported_every_tick_including_halted():
    env = _env()
    from cohort.core.missions import Formation

    tl1 = env.roster.by_callsign["TL1"]
    tl1.formation = Formation.COLUMN
    tl1.heading = (1, 0)
    env.roster.by_callsign["RFN1"].pos = (10, 10)  # 2 behind on the axis: station
    env.roster.by_callsign["RFN2"].pos = (12, 20)  # far off
    env.roster.by_callsign["SL1"].pos = (11, 11)
    env._update_visual_links()
    assert env._station["RFN1"] == (True, 0.0)
    station, err = env._station["RFN2"]
    assert station is False and 0.0 < err <= 1.0
    _step(env)  # halted: still measured
    assert env._station["RFN1"][0] is True
