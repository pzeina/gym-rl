"""Garble state (docs/readback-cycle.md §A): the radio analog of the voice cue.

Under ``comm_model="range"`` only, a transmission whose listener sits beyond
``comm_range`` but within ``GARBLE_RADIUS_FACTOR * comm_range`` produces a
listener-private garble ping — non-semantic, no content, no sender identity,
no bearing. ``global`` never garbles, ``voice_only`` already carries the state
as the voice cue, and ``jammed`` outages stay unobservable (owner decision
2026-08-24): a jammed transmission produces NO ping. Deterministic geometry,
no RNG; TTL discipline mirrors the acoustic cue memory.
"""

import json
from dataclasses import replace

from cohort import make_env
from cohort.config import get_scenario
from cohort.env.actions import CATALOG
from cohort.env.cohort_env import GARBLE_RADIUS_FACTOR, GARBLE_TTL

STAY = 0
SITREP_IDX = next(s.index for s in CATALOG if s.kind == "sitrep")


def _flat_env(spec, seed=1):
    env = make_env(spec)
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (34, 34)  # far corner: no contact during the scripted steps
        e.home = e.pos
        e.goal = e.pos
    return env


def _range_env(comm_range=5.0, seed=1):
    return _flat_env(
        replace(get_scenario("fireteam"), comm_model="range", comm_range=comm_range),
        seed=seed,
    )


def _place(env):
    """Sender RFN1 with one listener in each regime of the annulus geometry."""
    env.roster.by_callsign["RFN1"].pos = (7, 10)
    env.roster.by_callsign["RFN3"].pos = (10, 10)  # d=3 <= 5: hears, no ping
    env.roster.by_callsign["RFN2"].pos = (13, 10)  # d=6 in (5, 7.5]: garble
    env.roster.by_callsign["TL1"].pos = (20, 10)   # d=13 > 7.5: silence


def _step_all(env, overrides):
    acts = {a: STAY for a in env.agents}
    acts.update(overrides)
    return env.step(acts)


def test_garble_ping_forms_in_the_annulus_only():
    env = _range_env(comm_range=5.0)
    _place(env)
    _step_all(env, {"RFN1": SITREP_IDX})
    sender_id = env.roster.by_callsign["RFN1"].id
    assert env._garble["RFN2"] == [(sender_id, 1)], "annulus listener gets one ping"
    assert env._garble["RFN3"] == [], "an in-range listener heard the semantics"
    assert env._garble["TL1"] == [], "beyond the garble radius is silence"
    assert env._garble["RFN1"] == [], "the sender never garbles itself"


def test_garble_annulus_matches_the_published_factor():
    """The annulus edge is GARBLE_RADIUS_FACTOR * comm_range, inclusive."""
    env = _range_env(comm_range=4.0)
    env.roster.by_callsign["RFN1"].pos = (7, 10)
    env.roster.by_callsign["RFN2"].pos = (13, 10)   # d=6.0 == 1.5 * 4.0: pings
    env.roster.by_callsign["RFN3"].pos = (14, 10)   # d=7.0 > 6.0: silence
    env.roster.by_callsign["TL1"].pos = (11, 10)    # d=4.0 == comm_range: hears
    _step_all(env, {"RFN1": SITREP_IDX})
    assert GARBLE_RADIUS_FACTOR == 1.5
    assert len(env._garble["RFN2"]) == 1
    assert env._garble["RFN3"] == []
    assert env._garble["TL1"] == []


def test_no_garble_under_global():
    env = _flat_env(get_scenario("fireteam"))
    _place(env)
    _step_all(env, {"RFN1": SITREP_IDX})
    assert env._garble == {}, "global never garbles: everything lands"


def test_no_garble_under_voice_only():
    """voice_only already carries this state as the voice cue — no duplicate."""
    env = _flat_env(replace(get_scenario("fireteam"), comm_model="voice_only"))
    # keep the leader in low-voice range so the SITREP is legal at all
    env.roster.by_callsign["RFN1"].pos = (7, 10)
    env.roster.by_callsign["TL1"].pos = (8, 10)
    _step_all(env, {"RFN1": SITREP_IDX})
    assert env._garble == {}


def test_no_garble_under_jammed_even_during_an_outage():
    """A jammed transmission produces NO ping (owner decision 2026-08-24:
    outages stay unobservable — the cohort learns the net is down by not
    being answered, never by a signal that says so)."""
    env = _flat_env(replace(get_scenario("fireteam"), comm_model="jammed"))
    _place(env)
    env._net_jammed = True
    _step_all(env, {"RFN1": SITREP_IDX})
    assert env._garble == {}


def test_garble_ttl_expires_and_a_casualty_holds_none():
    env = _range_env(comm_range=5.0)
    _place(env)
    _step_all(env, {"RFN1": SITREP_IDX})
    assert len(env._garble["RFN2"]) == 1
    # fresh through the TTL window, expired one step past it
    for _ in range(GARBLE_TTL):
        _step_all(env, {})
        _place(env)  # hold the geometry still
    assert len(env._garble["RFN2"]) == 1, "a ping is held for GARBLE_TTL steps"
    _step_all(env, {})
    assert env._garble["RFN2"] == [], "and expires after it"
    # a casualty's memory clears, like the cue memory
    _place(env)
    _step_all(env, {"RFN1": SITREP_IDX})
    assert len(env._garble["RFN2"]) == 1
    env.roster.by_callsign["RFN2"].alive = False
    _step_all(env, {})
    assert env._garble["RFN2"] == []


def test_garble_is_deterministic_geometry_and_consumes_no_rng():
    env = _range_env(comm_range=5.0)
    _place(env)
    before = json.dumps(env._rng.bit_generator.state, default=str, sort_keys=True)
    env._register_garble(env.roster.by_callsign["RFN1"].id)
    after = json.dumps(env._rng.bit_generator.state, default=str, sort_keys=True)
    assert before == after, "garble registration must consume no RNG"
    assert env._garble["RFN2"] == [(env.roster.by_callsign["RFN1"].id, 0)]
    # and two identically seeded episodes hold identical records
    env1, env2 = _range_env(comm_range=5.0, seed=9), _range_env(comm_range=5.0, seed=9)
    for e in (env1, env2):
        _place(e)
        _step_all(e, {"RFN1": SITREP_IDX})
    assert env1._garble == env2._garble


def test_perception_exposes_garble_without_identity():
    env = _range_env(comm_range=5.0)
    _place(env)
    _step_all(env, {"RFN1": SITREP_IDX})
    records = env.perception("RFN2")["garble"]
    assert records == [{"step": 1, "ttl_remaining": GARBLE_TTL}]
    # no sender, no bearing, no content — a ping says only that it happened
    assert set(records[0]) == {"step", "ttl_remaining"}
    # copies, never live references
    records.clear()
    assert len(env.perception("RFN2")["garble"]) == 1
    assert env.perception("TL1")["garble"] == []
    # ttl_remaining counts down as the ping ages
    _step_all(env, {})
    assert env.perception("RFN2")["garble"][0]["ttl_remaining"] == GARBLE_TTL - 1


def test_garble_feeds_only_the_say_again_mask_bit_and_never_a_reward():
    """The pin, narrowed exactly once (docs/readback-cycle.md): the phase-2
    version said garble moves NO mask and NO reward; phase 4 gave it its one
    sanctioned reader — the SAY_AGAIN legality bit. With the records wiped,
    every other mask entry and every reward stays bit-identical."""
    import numpy as np

    from cohort.env.actions import CATALOG

    say_again_idx = next(s.index for s in CATALOG if s.kind == "say_again")
    env1, env2 = _range_env(comm_range=5.0, seed=3), _range_env(comm_range=5.0, seed=3)
    for e in (env1, env2):
        _place(e)
        _step_all(e, {"RFN1": SITREP_IDX})
    assert env1._garble["RFN2"], "precondition: a ping exists"
    # wipe one twin's garble state entirely
    for records in env2._garble.values():
        records.clear()
    for cs in env1.agents:
        m1 = env1._mask_for(env1.roster.by_callsign[cs])
        m2 = env2._mask_for(env2.roster.by_callsign[cs])
        diff = set(np.flatnonzero(m1 != m2))
        assert diff <= {say_again_idx}, f"garble state moved {cs}'s mask beyond SAY_AGAIN"
    assert env1._mask_for(env1.roster.by_callsign["RFN2"])[say_again_idx] == 1
    assert env2._mask_for(env2.roster.by_callsign["RFN2"])[say_again_idx] == 0
    _, r1, _, _, _ = _step_all(env1, {})
    _, r2, _, _, _ = _step_all(env2, {})
    assert r1 == r2, "garble state moved a reward"
