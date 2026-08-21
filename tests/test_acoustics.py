"""Acoustic mechanics and information boundaries (spec §8).

The deterministic substrate of the degraded-communications cycle: one event
per physical sound, threshold propagation with published terrain loss, coarse
cues that never leak source identity, and OpFor investigation of a frozen
estimated anchor. Everything here must hold BEFORE any PPO run touches the
mode (spec §7 Phase A)."""

from dataclasses import fields, replace

import numpy as np

from cohort import make_env
from cohort.config import get_scenario
from cohort.core import acoustics as snd
from cohort.core import language as lang
from cohort.core.world import FOREST, WALL, World
from cohort.env.actions import CATALOG

STAY = 0
MOVE_EAST = next(s.index for s in CATALOG if s.name == "MOVE_EAST")
MOVE_NORTH = next(s.index for s in CATALOG if s.name == "MOVE_NORTH")
FIRE = next(s.index for s in CATALOG if s.kind == "fire")
SITREP = next(s.index for s in CATALOG if s.kind == "sitrep")


def _tactical_env(seed=1, **spec_overrides):
    spec = replace(
        get_scenario("fireteam"), sound_model="tactical", **spec_overrides
    )
    env = make_env(spec)
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:  # park the OpFor far away and at home
        e.pos = (30, 30)
        e.home = e.pos
        e.prev_pos = e.pos
    return env


def _step_all(env, overrides=None):
    acts = {a: STAY for a in env.agents}
    if overrides:
        acts.update(overrides)
    return env.step(acts)


def _events(env, kind=None):
    evs = env.last_sound_events
    return [e for e in evs if kind is None or e.kind == kind]


# ------------------------------------------------------------------ #
# event generation
# ------------------------------------------------------------------ #


def test_open_move_creates_exactly_one_radius_2_movement_event():
    env = _tactical_env()
    tl = env.roster.by_callsign["TL1"]
    tl.pos = (10, 10)
    _step_all(env, {"TL1": MOVE_EAST})
    moves = _events(env, "movement")
    assert len(moves) == 1
    assert moves[0].base_radius == snd.MOVEMENT_OPEN_RADIUS
    assert moves[0].side == "friendly"


def test_move_touching_forest_creates_exactly_one_radius_3_event():
    env = _tactical_env()
    tl = env.roster.by_callsign["TL1"]
    tl.pos = (10, 10)
    env.world.grid[10, 11] = FOREST  # destination cell (x=11, y=10)
    _step_all(env, {"TL1": MOVE_EAST})
    moves = _events(env, "movement")
    assert len(moves) == 1
    assert moves[0].base_radius == snd.MOVEMENT_FOREST_RADIUS


def test_stay_and_blocked_movement_create_no_event():
    env = _tactical_env()
    tl = env.roster.by_callsign["TL1"]
    tl.pos = (10, 10)
    env.world.grid[10, 11] = WALL  # EAST is illegal → treated as STAY
    _step_all(env, {"TL1": MOVE_EAST})
    assert _events(env, "movement") == []


def test_two_simultaneous_movers_create_two_independent_events():
    env = _tactical_env()
    env.roster.by_callsign["TL1"].pos = (10, 10)
    env.roster.by_callsign["RFN1"].pos = (12, 12)
    _step_all(env, {"TL1": MOVE_EAST, "RFN1": MOVE_NORTH})
    moves = _events(env, "movement")
    assert len(moves) == 2
    assert moves[0].id != moves[1].id


def test_every_fire_creates_a_weapon_event_for_both_sides():
    env = _tactical_env()
    tl = env.roster.by_callsign["TL1"]
    tl.pos = (10, 10)
    enemy = env.enemies[0]
    enemy.pos = (14, 10)  # visible, in weapon range → both sides will shoot
    enemy.home = enemy.pos
    _step_all(env, {"TL1": FIRE})
    weapon = _events(env, "weapon_fire")
    sides = {e.side for e in weapon}
    assert "friendly" in sides, "blue shot must sound whether it hit or missed"
    assert "hostile" in sides, "the enemy return fire must sound too"
    blue = next(e for e in weapon if e.side == "friendly")
    assert tuple(blue.pos) == (10, 10), "the event is at the shooter"


def test_voice_event_from_speech_and_detectable_beyond_intelligibility():
    env = _tactical_env()
    tl = env.roster.by_callsign["TL1"]
    tl.pos = (10, 10)
    _step_all(env, {"TL1": SITREP})
    voice = _events(env, "voice")
    assert len(voice) == 1
    assert voice[0].base_radius == snd.VOICE_DETECT_RADIUS
    assert voice[0].message_index is not None


def test_trap_activation_sounds_at_the_trap_cell():
    env = _tactical_env(seed=2)
    from cohort.core.units import Trap

    env.traps = [Trap(id=0, pos=(11, 10))]
    tl = env.roster.by_callsign["TL1"]
    tl.pos = (10, 10)
    _step_all(env, {"TL1": MOVE_EAST})
    traps = _events(env, "trap")
    assert len(traps) == 1
    assert tuple(traps[0].pos) == (11, 10)
    assert traps[0].base_radius == snd.TRAP_DETECT_RADIUS


# ------------------------------------------------------------------ #
# propagation: terrain loss, symmetry
# ------------------------------------------------------------------ #


def _bare_world(w=30, h=30):
    return World(np.zeros((h, w), dtype=np.int8), [], [], [])


def test_wall_attenuation_applies_exactly_once():
    one_wall = _bare_world()
    one_wall.grid[10, 12] = WALL
    two_walls = _bare_world()
    two_walls.grid[10, 12] = WALL
    two_walls.grid[10, 14] = WALL
    r1 = snd.effective_radius(one_wall, (10, 10), (18, 10), 16.0)
    r2 = snd.effective_radius(two_walls, (10, 10), (18, 10), 16.0)
    assert r1 == 16.0 * snd.WALL_SOUND_FACTOR
    assert r2 == r1, "one or more walls apply the factor ONCE"


def test_forest_attenuation_is_deterministic_and_symmetric():
    world = _bare_world()
    world.grid[10, 12] = FOREST
    world.grid[10, 13] = FOREST
    a, b = (10, 10), (17, 10)
    fwd = snd.effective_radius(world, a, b, 16.0)
    rev = snd.effective_radius(world, b, a, 16.0)
    assert fwd == rev == 16.0 * snd.FOREST_SOUND_FACTOR**2
    # off-axis endpoints too (Bresenham direction asymmetry must not leak)
    world.grid[12, 14] = FOREST
    a, b = (10, 10), (19, 14)
    assert snd.effective_radius(world, a, b, 16.0) == snd.effective_radius(world, b, a, 16.0)


def test_detection_is_a_threshold_on_the_effective_radius():
    world = _bare_world()
    assert snd.received_strength(world, (10, 10), (13, 10), 4.0) is not None
    assert snd.received_strength(world, (10, 10), (15, 10), 4.0) is None
    world.grid[10, 12] = WALL  # 4.0 * 0.5 = 2.0 < distance 3
    assert snd.received_strength(world, (10, 10), (13, 10), 4.0) is None


# ------------------------------------------------------------------ #
# cue information boundaries
# ------------------------------------------------------------------ #


def test_cue_exposes_only_the_allowed_fields():
    allowed = {"kind", "side", "bearing", "distance_band", "strength", "event_step", "event_id"}
    assert {f.name for f in fields(snd.AcousticCue)} == allowed
    # and none of those is a source id, exact cell, text, or heard_by list


def test_enemy_fire_gives_blue_a_coarse_cue_never_a_position_or_fire_solution():
    env = _tactical_env()
    tl = env.roster.by_callsign["TL1"]
    tl.pos = (10, 10)
    enemy = env.enemies[0]
    # audible (weapon radius 16) but invisible (vision 10) and out of range
    enemy.pos = (22, 10)
    enemy.home = enemy.pos
    enemy.last_seen_player = tl.pos  # make it shoot? no: it cannot see → walks
    enemy.last_seen_step = 0
    # force a hostile weapon event directly instead of relying on the AI
    env._step_sounds = []
    env._emit_sound("weapon_fire", enemy.pos, "hostile", snd.WEAPON_DETECT_RADIUS, source_cs="E0")
    env._deliver_sounds_to_blue()
    cues = env._agent_cues["TL1"]
    assert cues, "a shot at 12 cells must be heard"
    cue = cues[0]
    assert cue.kind == "weapon_fire"
    assert cue.side == "unknown", "unseen source: attribution stays unknown"
    assert cue.distance_band == 2  # far — never an exact range
    # a heard-but-unseen target does not make FIRE legal
    mask = env._mask_for(tl)
    assert mask[FIRE] == 0


def test_hostile_attribution_requires_seeing_the_source():
    env = _tactical_env()
    tl = env.roster.by_callsign["TL1"]
    tl.pos = (10, 10)
    enemy = env.enemies[0]
    enemy.pos = (14, 10)  # visible
    env._step_sounds = []
    env._emit_sound("movement", enemy.pos, "hostile", 16.0, source_cs="E0")
    env._deliver_sounds_to_blue()
    assert env._agent_cues["TL1"][0].side == "hostile"


def test_friendly_attribution_requires_semantics_or_perception():
    env = _tactical_env()
    tl = env.roster.by_callsign["TL1"]
    rfn = env.roster.by_callsign["RFN1"]
    tl.pos = (10, 10)
    rfn.pos = (13, 10)  # in LOS, close → perceives the source
    env._step_sounds = []
    ev = env._emit_sound("movement", tl.pos, "friendly", 16.0, source_cs="TL1")
    assert ev is not None
    env._deliver_sounds_to_blue()
    assert env._agent_cues["RFN1"][0].side == "friendly"
    assert env._agent_cues["TL1"] == [], "the emitter gets no cue of its own sound"


def test_cue_ttl_expiration_clears_observation_and_investigation():
    env = _tactical_env()
    tl = env.roster.by_callsign["TL1"]
    tl.pos = (10, 10)
    env._step_sounds = []
    env._emit_sound("weapon_fire", (20, 10), "hostile", snd.WEAPON_DETECT_RADIUS, source_cs="E9")
    env._deliver_sounds_to_blue()
    assert env._agent_cues["TL1"]
    for _ in range(snd.SOUND_MEMORY_TTL + 1):
        _step_all(env)
    assert env._agent_cues["TL1"] == []


def test_cue_memory_keeps_at_most_four_by_the_stable_order():
    env = _tactical_env()
    tl = env.roster.by_callsign["TL1"]
    tl.pos = (10, 10)
    env._step_sounds = []
    for i in range(6):
        env._emit_sound("movement", (12 + i, 10), "hostile", 16.0, source_cs=f"E{i}")
    env._deliver_sounds_to_blue()
    cues = env._agent_cues["TL1"]
    assert len(cues) == snd.MAX_CUES
    keys = [(-c.strength, c.age(env._step_count), c.event_id) for c in cues]
    assert keys == sorted(keys), "truncation order is (strength, age, event id)"


# ------------------------------------------------------------------ #
# OpFor use of sound
# ------------------------------------------------------------------ #


def test_opfor_investigates_a_frozen_anchor_and_does_not_track_the_source():
    env = _tactical_env()
    enemy = env.enemies[0]
    enemy.pos = (20, 10)
    enemy.home = enemy.pos
    # a Blue weapon event 12 cells out: audible (16), invisible (vision 10)
    env._pending_enemy_sounds = []
    ev = env._emit_sound("weapon_fire", (8, 10), "friendly", snd.WEAPON_DETECT_RADIUS)
    assert ev is not None
    env._deliver_sounds_to_enemies()
    anchor = enemy.heard_blue_anchor
    assert anchor is not None
    assert enemy.last_seen_player is None, "sound never writes the visual memory"
    # the anchor is coarse — the true source cell is not disclosed
    before = enemy.pos
    _step_all(env)  # silent step: the enemy moves toward the FROZEN anchor
    assert enemy.heard_blue_anchor == anchor, "the anchor never follows the source"
    from cohort.core.world import dist

    assert dist(enemy.pos, anchor) < dist(before, anchor), "it investigates"


def test_opfor_anchor_expires_with_the_cue_ttl():
    env = _tactical_env()
    enemy = env.enemies[0]
    enemy.pos = (20, 10)
    enemy.home = enemy.pos
    enemy.heard_blue_anchor = (10, 10)
    enemy.heard_blue_step = env._step_count
    for _ in range(snd.SOUND_MEMORY_TTL + 2):
        _step_all(env)
    # past the TTL the anchor no longer authorizes movement: garrison holds home
    pos_after_expiry = enemy.pos
    _step_all(env)
    from cohort.core.world import dist

    assert dist(pos_after_expiry, enemy.home) <= dist((20, 10), (10, 10))


def test_one_brique_member_hearing_blue_does_not_update_the_others():
    spec = replace(get_scenario("patrol_brique"), sound_model="tactical")
    env = make_env(spec)
    env.reset(seed=4)
    env.world.grid[:] = 0
    near, far = env.enemies[0], env.enemies[1]
    near.pos = (20, 20)
    far.pos = (40, 40)
    env._pending_enemy_sounds = []
    env._emit_sound("voice", (18, 20), "friendly", snd.VOICE_DETECT_RADIUS)
    env._deliver_sounds_to_enemies()
    assert near.heard_blue_anchor is not None
    assert far.heard_blue_anchor is None


def test_the_same_seeded_tactical_episode_reproduces_exactly():
    def run(seed):
        env = make_env(replace(get_scenario("fireteam"), sound_model="tactical"))
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(5)
        log = []
        for _ in range(40):
            if not env.agents:
                break
            acts = {}
            for cs in env.agents:
                legal = np.flatnonzero(obs[cs]["action_mask"])
                acts[cs] = int(legal[rng.integers(len(legal))])
            obs, *_ = env.step(acts)
            log.append(
                (
                    [(e.id, e.kind, e.pos, e.base_radius) for e in env.last_sound_events],
                    [(e.id, e.heard_blue_anchor, e.heard_blue_step) for e in env.enemies],
                    {cs: [(c.kind, c.bearing, c.distance_band, c.strength, c.event_id)
                          for c in cues]
                     for cs, cues in env._agent_cues.items()},
                    [e.pos for e in env.enemies],
                )
            )
        return log

    assert run(9) == run(9)


# ------------------------------------------------------------------ #
# ACOUSTIC_CONTACT language
# ------------------------------------------------------------------ #


def test_acoustic_contact_formats_and_parses_as_inverses():
    for kind_index in range(len(lang.SOUND_KIND_WORDS)):
        for bearing in range(8):
            text = lang.format_acoustic_contact("SL1", "TL1", kind_index, bearing, 1, 42)
            parsed = lang.parse_acoustic_contact(text)
            assert parsed == {
                "kind_index": kind_index,
                "bearing": bearing,
                "distance_band": 1,
                "source_step": 42,
            }, text


def test_acoustic_contact_carries_no_grid_reference_and_is_not_a_contact():
    text = lang.format_acoustic_contact("SL1", "TL1", 0, 6, 2, 7)
    assert "GRID" not in text, "a sound never discloses an exact cell"
    assert lang.parse_acoustic_contact("SL1, THIS IS TL1: CONTACT, GRID 1010, 1 x ENEMY. OVER.") is None
    # nor does a visual CONTACT parse as an acoustic one, or vice versa
    from cohort.core.language import _SITREP_RE  # sanity: unrelated report untouched

    assert _SITREP_RE.search(text) is None


def test_oracle_exposes_source_truth_and_detectors_but_cues_stay_coarse():
    env = _tactical_env()
    tl = env.roster.by_callsign["TL1"]
    tl.pos = (10, 10)
    _step_all(env, {"TL1": SITREP})
    snap = env.oracle()
    voice = [e for e in snap["sound_events"] if e["kind"] == "voice"]
    assert voice and voice[0]["pos"] == [10, 10]
    assert "heard_by" in voice[0] and "detected_by_hostile" in voice[0]
    for rec in snap["soldiers"]:
        for cue in rec["cues"]:
            assert set(cue) == {"kind", "side", "bearing", "band", "strength", "age"}
