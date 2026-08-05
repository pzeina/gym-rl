"""BRIQUE asymmetric OpFor: band intent machine, ambush discipline, traps.

The threat model is the PROTERRE manual's p. 9 "LA MENACE": armed bands of
5-20 with light weapons — coups de main, limited raids, harassment with
improvised means including mines and traps. Spaces stay FROZEN: BRIQUE is
environment-side only (enemy behavior, terrain hazards), so v1.4/v1.5
checkpoints keep loading.
"""

import json

import numpy as np

from cohort import make_env
from cohort.config import ScenarioSpec
from cohort.core import language as lang
from cohort.core.missions import MissionType
from cohort.core.ranks import Rank
from cohort.core.units import BriqueBandConfig, Soldier, Trap, select_band_target
from cohort.core.world import dist
from cohort.env.actions import CATALOG, N_ACTIONS
from cohort.env.observations import OBS_DIM

STAY = 0
MOVE_EAST = next(s.index for s in CATALOG if s.name == "MOVE_EAST")


def _spec(**kw) -> ScenarioSpec:
    base = dict(
        name="brique_test",
        description="BRIQUE test band",
        org="fireteam",
        map_size=(36, 36),
        objectives=(("ALPHA", (27, 27)),),
        spawn=(5, 5),
        n_enemies=4,
        opfor_mode="brique",
        root_mission=MissionType.SEIZE,
        root_objective="ALPHA",
        max_steps=300,
    )
    base.update(kw)
    return ScenarioSpec(**base)


def _flat_brique(seed=3, **kw):
    """Brique env on flattened terrain (deterministic geometry for tests)."""
    env = make_env(_spec(**kw))
    env.reset(seed=seed)
    env.world.grid[:] = 0
    return env


def _step_all(env, overrides=None):
    acts = {a: STAY for a in env.agents}
    acts.update(overrides or {})
    return env.step(acts)


def _post_band(env, pos):
    """Park every band member (and its post) on ``pos``."""
    for e in env.enemies:
        e.pos = pos
        e.home = pos
        e.prev_pos = pos
        env.band.posts[e.id] = pos


# ------------------------------------------------------------------ #
# spaces frozen
# ------------------------------------------------------------------ #


def test_spaces_frozen_under_brique():
    """BRIQUE is environment-side only: Discrete(157) / Box(137) unchanged."""
    assert N_ACTIONS == 157
    assert OBS_DIM == 137
    env = make_env(_spec(n_traps=3))
    obs, _ = env.reset(seed=0)
    assert env.action_space(env.possible_agents[0]).n == 157
    for a in env.agents:
        assert obs[a]["observation"].shape == (137,)
        assert obs[a]["action_mask"].shape == (157,)


def test_brique_scenario_presets():
    """The two shipped BRIQUE scenarios: geometry matches their non-brique
    parents (checkpoint transfer), band + traps configured per the manual."""
    from cohort.config import get_scenario

    patrol = get_scenario("patrol_brique")
    assert patrol.opfor_mode == "brique"
    assert patrol.band.initial_intent == "ambush"
    assert patrol.n_traps == 3
    assert patrol.root_mission is MissionType.SEIZE
    squad = get_scenario("squad")
    assert (patrol.map_size, patrol.objectives, patrol.spawn) == (
        squad.map_size, squad.objectives, squad.spawn,
    )

    defend = get_scenario("defend_brique")
    assert defend.opfor_mode == "brique"
    assert defend.band.initial_intent == "harass"
    assert defend.band.raid_period > 0
    assert defend.n_traps == 2
    assert defend.root_mission is MissionType.DEFEND
    ftd = get_scenario("fireteam_defend")
    assert (defend.map_size, defend.objectives, defend.spawn) == (
        ftd.map_size, ftd.objectives, ftd.spawn,
    )
    # both scenarios reset with a live band and the full trap count
    for name in ("patrol_brique", "defend_brique"):
        env = make_env(name)
        env.reset(seed=11)
        assert env.band is not None
        assert len(env.traps) == get_scenario(name).n_traps
        assert all(e.mode == "brique" for e in env.enemies)


# ------------------------------------------------------------------ #
# band intent machine
# ------------------------------------------------------------------ #


def test_ambush_holds_fire_outside_ambush_range():
    """Posted ambushers HOLD FIRE on blue inside weapon range (8) but outside
    ambush_range (5) — the discipline that distinguishes an ambush from a
    firefight."""
    env = _flat_brique(band=BriqueBandConfig(initial_intent="ambush", ambush_range=5.0))
    _post_band(env, (20, 5))
    tl1 = env.roster.by_callsign["TL1"]
    tl1.pos = (13, 5)  # dist 7: in weapon range, outside ambush range
    tl1.prev_pos = tl1.pos
    _step_all(env)
    assert env.band.sprung is False
    assert not any(e.fired_this_step for e in env.enemies)
    assert all(e.behavior == "posted" for e in env.enemies if e.alive)
    assert all(s.health == 100 for s in env.roster.soldiers)


def test_ambush_springs_and_volleys_then_goes_harass():
    """Blue within ambush_range springs the ambush (volley); after
    volley_steps the band dissolves into hit-and-run (HARASS)."""
    env = _flat_brique(
        band=BriqueBandConfig(initial_intent="ambush", ambush_range=5.0, volley_steps=2)
    )
    _post_band(env, (20, 5))
    tl1 = env.roster.by_callsign["TL1"]
    tl1.pos = (16, 5)  # dist 4 <= ambush_range
    tl1.prev_pos = tl1.pos
    _step_all(env)
    assert env.band.sprung is True
    assert env.band.spring_step == 1
    assert any(e.fired_this_step for e in env.enemies), "the volley fires immediately"
    assert any(e.behavior == "volleying" for e in env.enemies if e.alive)
    for _ in range(2):
        _step_all(env)
    assert env.band.intent == "harass", "the ambush dissolves after the volley window"


def test_compromised_ambush_springs_early():
    """An ambush taking effective fire is compromised and opens fire even
    with no blue unit inside ambush_range — it does not die posted."""
    env = _flat_brique(band=BriqueBandConfig(initial_intent="ambush", ambush_range=5.0))
    _post_band(env, (20, 5))
    tl1 = env.roster.by_callsign["TL1"]
    tl1.pos = (13, 5)  # dist 7: outside ambush range, inside weapon range
    tl1.prev_pos = tl1.pos
    env.enemies[0].health = 60  # the band has been hit
    _step_all(env)
    assert env.band.sprung is True
    assert any(e.fired_this_step for e in env.enemies), "the compromised ambush shoots back"


def test_lurk_posts_the_ambush_when_blue_approaches():
    env = _flat_brique(
        band=BriqueBandConfig(initial_intent="lurk", lurk_trigger=12.0, ambush_range=5.0)
    )
    _post_band(env, (25, 5))
    # blue far: the band lurks
    _step_all(env)
    assert env.band.intent == "lurk"
    assert all(e.behavior == "hiding" for e in env.enemies if e.alive)
    # blue closes within the trigger: the band posts its ambush
    tl1 = env.roster.by_callsign["TL1"]
    tl1.pos = (15, 5)  # dist 10 <= lurk_trigger
    tl1.prev_pos = tl1.pos
    _step_all(env)
    assert env.band.intent == "ambush"
    assert env.band.sprung is False, "posting is not springing: fire is still held"


def test_scatter_below_strength_threshold_breaks_contact():
    """The band scatters only under 30% strength (low self-preservation) and
    then flees toward the map edge without firing — even point-blank."""
    env = _flat_brique(band=BriqueBandConfig(initial_intent="harass", scatter_below=0.3))
    # 2 of 4 dead: strength 0.5 >= 0.3 → still fighting
    for e in env.enemies[:2]:
        e.alive = False
    survivor = env.enemies[3]
    survivor.pos = (20, 20)
    survivor.prev_pos = survivor.pos
    _step_all(env)
    assert env.band.intent != "scatter"
    # 3 of 4 dead: strength 0.25 < 0.3 → scatter
    env.enemies[2].alive = False
    tl1 = env.roster.by_callsign["TL1"]
    tl1.pos = (21, 20)  # adjacent: a self-preserving enemy would fire
    tl1.prev_pos = tl1.pos
    d_edge_before = dist(survivor.pos, (survivor.pos[0], 1))
    _step_all(env)
    assert env.band.intent == "scatter"
    assert survivor.behavior == "fleeing"
    assert not survivor.fired_this_step, "a scattering band breaks contact, it does not fight"
    d_edge_after = min(
        dist(survivor.pos, p)
        for p in [(1, survivor.pos[1]), (34, survivor.pos[1]), (survivor.pos[0], 1), (survivor.pos[0], 34)]
    )
    assert d_edge_after < d_edge_before


def test_harass_fires_then_displaces():
    """Hit-and-run: harass_shots from range, then displace to a new cell."""
    env = _flat_brique(
        n_enemies=1, band=BriqueBandConfig(initial_intent="harass", harass_shots=2)
    )
    member = env.enemies[0]
    member.pos = (13, 5)
    member.home = member.pos
    member.prev_pos = member.pos
    tl1 = env.roster.by_callsign["TL1"]
    tl1.pos = (6, 5)  # dist 7: max-range sniping
    tl1.prev_pos = tl1.pos
    behaviors = []
    for _ in range(3):
        _step_all(env)
        behaviors.append(member.behavior)
    assert behaviors[0] == "sniping" and behaviors[1] == "sniping"
    assert behaviors[2] == "displacing", "after harass_shots the member displaces"
    d0 = dist(member.pos, tl1.pos)
    _step_all(env)
    assert member.behavior == "displacing"
    assert dist(member.pos, tl1.pos) > d0, "displacement opens distance from the threat"


def test_raid_moves_on_objective_lingers_then_withdraws():
    """RAID: move fast onto the installation, linger k steps (sabotage), then
    the band reverts to HARASS (withdrawal)."""
    env = _flat_brique(
        n_enemies=1,
        band=BriqueBandConfig(initial_intent="raid", raid_linger=3),
    )
    member = env.enemies[0]
    member.pos = (22, 27)  # 5 cells west of OBJ ALPHA (27, 27)
    member.home = member.pos
    member.prev_pos = member.pos
    obj = env.world.objective_by_name("ALPHA").pos
    sabotage_steps = 0
    for _ in range(20):
        _step_all(env)
        if member.behavior == "sabotaging":
            sabotage_steps += 1
        if env.band.intent == "harass":
            break
    assert sabotage_steps >= 3, "the raider lingers on the objective (sabotage)"
    assert env.band.intent == "harass", "after the linger the raid ends"
    assert dist(member.pos, obj) <= 2.5 or sabotage_steps > 0


def test_target_selection_prefers_commander_wounded_isolated():
    def mk(i, pos, health=100, human=False):
        return Soldier(id=i, callsign=f"S{i}", rank=Rank.RFN, pos=pos, health=health, human=human)

    shooter = (15, 15)
    # the human commander outranks every other preference, even wounded
    human = mk(0, (25, 25), human=True)
    wounded = mk(1, (16, 15), health=30)
    both = [human, wounded]
    assert select_band_target(shooter, both, both) is human
    # wounded beats healthy
    healthy = mk(2, (16, 16))
    assert select_band_target(shooter, [wounded, healthy], [wounded, healthy]) is wounded
    # isolated beats grouped, even farther away
    a = mk(3, (14, 15))
    b = mk(4, (14, 16))
    iso = mk(5, (22, 22))
    blue = [a, b, iso]
    assert select_band_target(shooter, [a, iso], blue) is iso
    # distance breaks remaining ties
    assert select_band_target(shooter, [a, b], blue) is a
    assert select_band_target(shooter, [], blue) is None


# ------------------------------------------------------------------ #
# traps / mines
# ------------------------------------------------------------------ #


def test_trap_message_formatter():
    assert (
        lang.format_trap("RFN2", (11, 10))
        == "ALL STATIONS: RFN2 HIT A DEVICE AT GRID 1110. OUT."
    )


def _flat_garrison(seed=1):
    env = make_env("fireteam")
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (30, 1)
        e.home = e.pos
        e.prev_pos = e.pos
    return env


def test_trap_damages_first_friendly_and_reveals():
    env = _flat_garrison()
    env.traps.append(Trap(id=0, pos=(11, 10)))
    rfn1 = env.roster.by_callsign["RFN1"]
    rfn1.pos = (10, 10)
    rfn1.prev_pos = rfn1.pos
    *_, infos = _step_all(env, {"RFN1": MOVE_EAST})
    trap = env.traps[0]
    assert rfn1.health == 60, "the device does its 40 damage"
    assert trap.revealed and not trap.armed, "triggered devices are spent and revealed"
    assert infos["RFN1"]["components"]["combat"] < 0
    assert any(
        m.kind.value == "trap" and m.text == "ALL STATIONS: RFN1 HIT A DEVICE AT GRID 1110. OUT."
        for m in env.transcript.messages
    ), "the trigger lands on the net as a CASUALTY-style broadcast"
    # a second friendly on the same cell takes nothing: the device is spent
    rfn2 = env.roster.by_callsign["RFN2"]
    rfn2.pos = (10, 10)
    rfn2.prev_pos = rfn2.pos
    _step_all(env, {"RFN2": MOVE_EAST})
    assert rfn2.health == 100


def test_trap_kill_flows_through_casualty_processing():
    env = _flat_garrison()
    env.traps.append(Trap(id=0, pos=(11, 10)))
    rfn1 = env.roster.by_callsign["RFN1"]
    rfn1.pos = (10, 10)
    rfn1.prev_pos = rfn1.pos
    rfn1.health = 30
    _obs, _r, terms, _tr, infos = _step_all(env, {"RFN1": MOVE_EAST})
    assert not rfn1.alive
    assert terms["RFN1"] is True
    assert any(m.kind.value == "casualty" and "RFN1" in m.text for m in env.transcript.messages)
    assert infos["RFN2"]["components"]["combat"] < 0, "teammates pay the death penalty"


def test_traps_spawn_on_route_and_are_oracle_visible_from_reset():
    env = make_env(_spec(n_traps=3))
    env.reset(seed=5)
    assert len(env.traps) == 3
    snap = env.oracle()
    assert len(snap["traps"]) == 3
    for t in snap["traps"]:
        assert t["armed"] and not t["revealed"]
        assert env.world.passable(tuple(t["pos"]))
        assert dist(tuple(t["pos"]), env.spec_cfg.spawn) >= 6.0
    json.dumps(snap)


def test_traps_never_in_blue_observations():
    """The obs vector is bit-identical with and without traps on the map —
    the trap layer is the assurance layer's inference target, not the
    cohort's input."""
    env = make_env(_spec(n_traps=3))
    env.reset(seed=7)
    with_traps = {
        a: (o["observation"].copy(), o["action_mask"].copy())
        for a, o in env._all_observations().items()
    }
    env.traps = []
    without = env._all_observations()
    for a, (obs, mask) in with_traps.items():
        assert np.array_equal(obs, without[a]["observation"])
        assert np.array_equal(mask, without[a]["action_mask"])


# ------------------------------------------------------------------ #
# oracle exposure + terminal semantics + determinism
# ------------------------------------------------------------------ #


def test_band_state_exposed_in_oracle():
    env = _flat_brique(n_traps=2)
    _step_all(env)
    snap = env.oracle()
    band = snap["band"]
    assert band is not None
    assert band["intent"] in ("lurk", "ambush", "harass", "raid", "scatter")
    assert band["sprung"] in (True, False)
    assert len(band["posts"]) == len(env.enemies)
    assert 0.0 <= band["strength"] <= 1.0
    for e in snap["enemies"]:
        assert e["mode"] == "brique"
        assert e["behavior"] is not None
    json.dumps(snap)


def test_defend_brique_success_on_band_destroyed():
    env = _flat_brique(
        objectives=(("ALPHA", (18, 18)),),
        spawn=(17, 17),
        root_mission=MissionType.DEFEND,
        band=BriqueBandConfig(initial_intent="harass"),
    )
    for e in env.enemies:
        e.alive = False
    for s in env.roster.soldiers:
        s.pos = (18, 18)
        s.prev_pos = s.pos
    terms = {}
    for _ in range(env.spec_cfg.grace_window + 2):
        _obs, _r, terms, *_ = _step_all(env)
        if terms and all(terms.values()):
            break
    assert env.outcome == "success"


def test_defend_brique_success_on_scatter_with_contact_broken():
    env = _flat_brique(
        objectives=(("ALPHA", (18, 18)),),
        spawn=(17, 17),
        root_mission=MissionType.DEFEND,
        band=BriqueBandConfig(initial_intent="harass", break_contact_dist=12.0),
    )
    for s in env.roster.soldiers:
        s.pos = (18, 18)
        s.prev_pos = s.pos
    # a scattered survivor still near the position: contact NOT broken
    for e in env.enemies[1:]:
        e.alive = False
    survivor = env.enemies[0]
    survivor.pos = (10, 18)  # 8 cells: too close
    survivor.prev_pos = survivor.pos
    env.band.intent = "scatter"
    _step_all(env)
    assert env._success_step is None, "a nearby survivor is not broken contact"
    # far from every blue and the objective: the band is out of the fight
    survivor.pos = (1, 1)
    survivor.prev_pos = survivor.pos
    _step_all(env)
    assert env._success_step is not None, "scattered + contact broken + objective held = success"


def test_seize_brique_keeps_standard_seize_semantics():
    """Destroying the band does not seize anything: the OPORD objective must
    still be cleared and occupied."""
    env = _flat_brique()  # SEIZE ALPHA
    for e in env.enemies:
        e.alive = False
    _step_all(env)
    assert env._success_step is None, "band destroyed but objective not occupied: no success"
    for s in env.roster.soldiers:
        s.pos = (27, 27)
        s.prev_pos = s.pos
    _step_all(env)
    assert env._success_step is not None


def test_brique_rollout_deterministic_given_seed():
    def rollout(seed):
        env = make_env(_spec(n_traps=2, band=BriqueBandConfig(initial_intent="ambush")))
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(99)
        trace = []
        for _ in range(40):
            if not env.agents:
                break
            acts = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
            obs, *_ = env.step(acts)
            trace.append(
                (
                    tuple(e.pos for e in env.enemies),
                    tuple((e.behavior, e.alive) for e in env.enemies),
                    env.band.intent,
                    tuple((t.pos, t.armed) for t in env.traps),
                )
            )
        return trace, env.transcript.render()

    t1, log1 = rollout(1234)
    t2, log2 = rollout(1234)
    assert t1 == t2
    assert log1 == log2
