"""patrol_brique_voice — the epistemic-monitor fixture scenario (HOST_REQUESTS 8).

The squad_voice_liaison fixture episode never moved the coarse-cue dimensions
(51/51 cues at distance band 0, age 0, zero contact reports), so a monitor's
range and staleness handling was exercised by nothing. This scenario exists to
move them; the test pins that it actually does."""

from dataclasses import fields

from cohort import make_env
from cohort.config import get_scenario
from cohort.env.actions import CATALOG


def test_spec_is_the_brique_patrol_under_the_voice_mode():
    """Differs from patrol_brique exactly by the degraded-comms knobs (and
    labels) — geometry untouched, so squad-family checkpoints transfer."""
    base = get_scenario("patrol_brique")
    voice = get_scenario("patrol_brique_voice")
    varied = {
        "name", "description", "comm_model", "sound_model", "voice_range",
        "liaison_enabled", "reward_overrides", "experiment_arm",
    }
    for f in fields(base):
        if f.name in varied:
            continue
        assert getattr(base, f.name) == getattr(voice, f.name), f.name
    assert voice.comm_model == "voice_only"
    assert voice.sound_model == "tactical"
    assert voice.liaison_enabled is True


def test_the_fixture_scenario_moves_the_cue_dimensions():
    """A scripted southeast patrol at the fixture seed walks into the ambush:
    the acoustic memory must span distance bands, hold cues stale, and hear
    more than one sound kind — and sightings must form visual contacts."""
    east = next(s.index for s in CATALOG if s.name == "MOVE_EAST")
    south = next(s.index for s in CATALOG if s.name == "MOVE_SOUTH")
    env = make_env(get_scenario("patrol_brique_voice"))
    env.reset(seed=100)
    bands, ages, kinds, visual_rows = set(), set(), set(), 0
    for t in range(300):
        if not env.agents:
            break
        env.step({a: (east if t % 2 else south) for a in env.agents})
        for cs in env.roster.by_callsign:
            p = env.perception(cs)
            for cue in p["cues"]:
                bands.add(cue.distance_band)
                ages.add(cue.age(p["step"]))
                kinds.add(cue.kind)
            visual_rows += len(p["visual_contacts"])
    assert len(bands) >= 2, f"cues never left band 0: {sorted(bands)}"
    assert max(ages) >= 3, f"no cue ever held stale: {sorted(ages)}"
    assert len(kinds) >= 3, f"sound kinds too uniform: {sorted(kinds)}"
    assert visual_rows > 0, "the patrol never sighted the band"
