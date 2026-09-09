"""Read-back cycle observation blocks (docs/readback-cycle.md): +11 appended.

Layout: garble (2) + say-again-pending (1) + read-back-CORRECT-heard per
subordinate slot (4) + DONE-confirmed-heard per subordinate slot (4), all
appended after the liaison block for BOTH profiles, zero-filled wherever the
state is structurally unavailable. The garble pair is the only one wired in
the observations phase; the other nine slots stay zero until the env wiring
phase fills their AgentView fields.
"""

from dataclasses import replace

import numpy as np

from cohort.config import get_scenario
from cohort.env.cohort_env import GARBLE_TTL
from cohort.env.observations import (
    _DONE_HEARD_BLOCK,
    _GARBLE_BLOCK,
    _LIAISON_BLOCK,
    _READBACK_HEARD_BLOCK,
    _SAY_AGAIN_BLOCK,
    N_SUB_SLOTS,
    OBS_DIM,
    OFF_DONE_HEARD,
    OFF_GARBLE,
    OFF_LIAISON,
    OFF_READBACK_HEARD,
    OFF_SAY_AGAIN,
    obs_dim,
)
from tests.test_garble import SITREP_IDX, _flat_env, _place, _range_env, _step_all


def test_readback_blocks_close_the_layout():
    """The four blocks sit after liaison, in spec order, and end the vector."""
    assert OFF_GARBLE == OFF_LIAISON + _LIAISON_BLOCK
    assert OFF_SAY_AGAIN == OFF_GARBLE + _GARBLE_BLOCK
    assert OFF_READBACK_HEARD == OFF_SAY_AGAIN + _SAY_AGAIN_BLOCK
    assert OFF_DONE_HEARD == OFF_READBACK_HEARD + _READBACK_HEARD_BLOCK
    assert OFF_DONE_HEARD + _DONE_HEARD_BLOCK == OBS_DIM
    assert (_GARBLE_BLOCK, _SAY_AGAIN_BLOCK) == (2, 1)
    assert _READBACK_HEARD_BLOCK == _DONE_HEARD_BLOCK == N_SUB_SLOTS == 4


def test_both_profiles_carry_the_appended_blocks():
    """+11 lands on `full` AND `core`, so the profile delta stays the v1.10
    bisect's single variable (54) — the appended-block precedent."""
    assert obs_dim("full") - obs_dim("core") == 54


def test_all_eleven_slots_zero_where_structurally_unavailable():
    """Under comm_model="global" no garble exists, and before the env wiring
    phase nothing sets the other nine — every agent reads eleven zeros."""
    env = _flat_env(get_scenario("fireteam"))
    obs, *_ = _step_all(env, {"RFN1": SITREP_IDX})
    for cs in env.agents:
        tail = obs[cs]["observation"][OFF_GARBLE:]
        assert tail.shape == (11,)
        assert not tail.any(), f"{cs} observed read-back state that cannot exist"


def test_garble_slots_read_the_ping_and_its_freshness():
    env = _range_env(comm_range=5.0)
    _place(env)
    obs, *_ = _step_all(env, {"RFN1": SITREP_IDX})
    garbled = obs["RFN2"]["observation"]
    assert garbled[OFF_GARBLE] == 1.0
    assert garbled[OFF_GARBLE + 1] == 1.0, "a ping formed this step is fully fresh"
    # nobody else holds a ping: in-range heard it, out-of-annulus heard nothing
    for cs in ("RFN1", "RFN3", "TL1"):
        assert obs[cs]["observation"][OFF_GARBLE] == 0.0
        assert obs[cs]["observation"][OFF_GARBLE + 1] == 0.0
    # freshness decays on the cue TTL clock as the ping ages
    _place(env)
    obs, *_ = _step_all(env, {})
    aged = obs["RFN2"]["observation"]
    assert aged[OFF_GARBLE] == 1.0
    assert np.isclose(aged[OFF_GARBLE + 1], (GARBLE_TTL - 1) / GARBLE_TTL)
    # ...and both slots return to zero once the ping expires
    for _ in range(GARBLE_TTL + 1):
        _place(env)
        obs, *_ = _step_all(env, {})
    assert obs["RFN2"]["observation"][OFF_GARBLE] == 0.0
    assert obs["RFN2"]["observation"][OFF_GARBLE + 1] == 0.0


def test_garble_slots_zero_under_jammed_and_voice_only():
    """The two non-range degraded models: a jammed outage is unobservable and
    voice_only speaks through the voice cue — the garble pair stays zero."""
    jammed = _flat_env(replace(get_scenario("fireteam"), comm_model="jammed"))
    _place(jammed)
    jammed._net_jammed = True
    obs, *_ = _step_all(jammed, {"RFN1": SITREP_IDX})
    for cs in jammed.agents:
        assert not obs[cs]["observation"][OFF_GARBLE:OFF_SAY_AGAIN].any()

    voice = _flat_env(replace(get_scenario("fireteam"), comm_model="voice_only"))
    voice.roster.by_callsign["RFN1"].pos = (7, 10)
    voice.roster.by_callsign["TL1"].pos = (8, 10)
    obs, *_ = _step_all(voice, {"RFN1": SITREP_IDX})
    for cs in voice.agents:
        assert not obs[cs]["observation"][OFF_GARBLE:OFF_SAY_AGAIN].any()


def test_nine_event_slots_stay_zero_without_their_traffic():
    """Say-again-pending and the two per-subordinate blocks fill only from
    traffic actually heard (a SAY AGAIN request, a READBACK_CORRECT answer,
    a received-and-confirmed DONE) — a held garble ping alone moves none of
    them."""
    env = _range_env(comm_range=5.0)
    _place(env)
    obs, *_ = _step_all(env, {"RFN1": SITREP_IDX})
    assert obs["RFN2"]["observation"][OFF_GARBLE] == 1.0
    for cs in env.agents:
        assert not obs[cs]["observation"][OFF_SAY_AGAIN:].any()
