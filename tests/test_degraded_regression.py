"""Regression safety for the degraded-communications cycle (spec §8).

The voice-only/acoustics work is an EXPLICITLY AUTHORIZED space break, but the
existing modes must not move: ``global`` and ``range`` seeded episodes stay
bit-identical in everything that matters — positions, health, rewards, the
transcript, and the RNG stream (``sound_model="off"`` consumes no new RNG) —
and the pre-existing action indices do not shift.

The golden digests below were computed on the tree at commit 58732ae, BEFORE
any Phase A/B/C code landed, by hashing 80 masked-random steps (driver rng
seeded separately from the env) of each episode: soldier positions/health/
rewards, enemy positions/health/liveness, the full transcript, and the final
``env._rng`` bit-generator state. Raw observation bytes are deliberately NOT
hashed — appended zero-filled blocks are the authorized break — and the
masked-random driver samples over LEGAL indices, which stay identical while
every appended action is masked off outside its mode.
"""

import hashlib
import json
from dataclasses import replace

import numpy as np

from cohort.config import get_scenario
from cohort.env.actions import CATALOG
from cohort.env.cohort_env import make_env

#: digests computed on the pre-cycle tree (commit 58732ae) — see module doc
GOLDEN = {
    "squad_global_seed123": "6829cb8ca349dc051b750d530f864c0e9db72200c61ccaebe0464aede891a2ab",
    "squad_global_seed7": "438af72106f968c2e63eb8e4114597163f1ac2119db574081eada02306e2e65f",
    "squad_range_seed123": "8dbdca06f67068cdc9eb55803c2286c783df6ad5e4a8ca98aa0feb5fc63fa29f",
    "fireteam_defend_seed5": "394c1729784c85f8adb9c121427649a9bf759eecc4c0f50b827a2aca379d3840",
    "patrol_brique_seed11": "73b20a0e72b87f54138213a0783f1f72d3585365a0113982e249cd8b957c1c6d",
}

#: the 228 pre-cycle action names, pinned by digest: appended actions may only
#: ever FOLLOW them — an index that moves silently breaks every checkpoint
PRE_CYCLE_N_ACTIONS = 228
PRE_CYCLE_ACTION_NAME_DIGEST = (
    "8f4b85d954f297373362288102a23428cd5f3f9d3bb957e7ff1180418cd2fff1"
)


def episode_digest(spec, seed: int, steps: int = 80) -> str:
    env = make_env(spec)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(7)
    h = hashlib.sha256()
    for _ in range(steps):
        if not env.agents:
            break
        actions = {}
        for cs in env.agents:
            mask = obs[cs]["action_mask"]
            legal = np.flatnonzero(mask)
            actions[cs] = int(legal[rng.integers(len(legal))])
        obs, rewards, _term, _trunc, _ = env.step(actions)
        for cs in sorted(rewards):
            s = env.roster.by_callsign[cs]
            h.update(f"{cs}:{s.pos}:{s.health}:{rewards[cs]:.9f};".encode())
        for e in env.enemies:
            h.update(f"E{e.id}:{e.pos}:{e.health}:{e.alive};".encode())
    for m in env.transcript.messages:
        h.update(f"[{m.step}]{m.text}|".encode())
    h.update(
        json.dumps(env._rng.bit_generator.state, default=str, sort_keys=True).encode()
    )
    return h.hexdigest()


def test_global_seeded_episodes_are_bit_identical():
    squad = get_scenario("squad")
    assert episode_digest(squad, 123) == GOLDEN["squad_global_seed123"]
    assert episode_digest(squad, 7) == GOLDEN["squad_global_seed7"]


def test_range_seeded_episodes_are_bit_identical():
    spec = replace(get_scenario("squad"), name="squad", comm_model="range")
    assert episode_digest(spec, 123) == GOLDEN["squad_range_seed123"]


def test_defend_and_brique_seeded_episodes_are_bit_identical():
    """The H-hour draw and the band/trap machinery consume RNG at reset; the
    acoustic layer must consume none of it under sound_model='off'."""
    assert episode_digest(get_scenario("fireteam_defend"), 5) == GOLDEN["fireteam_defend_seed5"]
    assert episode_digest(get_scenario("patrol_brique"), 11) == GOLDEN["patrol_brique_seed11"]


def test_sound_off_creates_no_behavioral_state():
    env = make_env("squad")
    env.reset(seed=3)
    for _ in range(5):
        env.step({a: 0 for a in env.agents})
    assert env.last_sound_events == []
    assert all(not cues for cues in env._agent_cues.values())
    assert all(e.heard_blue_anchor is None for e in env.enemies)


def test_action_indices_before_the_appended_actions_do_not_move():
    names = [s.name for s in CATALOG[:PRE_CYCLE_N_ACTIONS]]
    digest = hashlib.sha256("|".join(names).encode()).hexdigest()
    assert len(CATALOG) >= PRE_CYCLE_N_ACTIONS
    assert digest == PRE_CYCLE_ACTION_NAME_DIGEST, (
        "a pre-cycle action index moved — appended actions must only ever "
        "FOLLOW the existing catalog"
    )
