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
import pytest

from cohort.config import get_scenario
from cohort.env.actions import CATALOG
from cohort.env.cohort_env import make_env

#: digests computed on the pre-cycle tree (commit 58732ae) — see module doc
#: Re-golded 2026-08-26 for the price-dispersion cycle: `episode_digest` hashes
#: `rewards[cs]` alongside positions, health, enemies, transcript and RNG state,
#: and `RewardConfig.bunching_penalty` was armed at -0.05. Only the reward half
#: moved — `test_a_reward_price_cannot_move_the_world` below pins that, at a
#: price 100x the shipped one, and it is the assertion that makes re-golding
#: safe. Re-golding because a digest changed, without showing WHICH half
#: changed, is how a dynamics regression gets waved through as a reward edit.
GOLDEN = {
    "squad_global_seed123": "98a00d07b3b31147b442f7589a56ec159c3e553b459c878b1e160c73910034a7",
    "squad_global_seed7": "aec5e28507417a9850f6e78735b302ba8c1997c9fa9c37ff2c54b4618299b4de",
    "squad_range_seed123": "1bd398508482ef4ba50cc6f66e694bfce65941ef201483ad0c355f8543650c6b",
    "fireteam_defend_seed5": "9bb7c3e634bd8d87e27a230aa7f6cd9320690fa7d44fadced870b11a6f857059",
    "patrol_brique_seed11": "c7bf8d7cbcd40f56fe4006f178c844445c3a30f557a1e55f982ab42c0dfabbaa",
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


def dynamics_digest(spec, seed: int, price: float, steps: int = 80) -> str:
    """`episode_digest` minus the rewards: the WORLD, and nothing that pays.

    The counterpart the re-golding above depends on. `episode_digest` hashes
    rewards together with the world, so any reward edit moves it and a real
    dynamics regression landing in the same commit would be indistinguishable
    from the edit. This one cannot be moved by a price at all.
    """
    env = make_env(spec)
    obs, _ = env.reset(seed=seed)
    env.rewards_cfg = replace(env.rewards_cfg, bunching_penalty=price)
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
            h.update(f"{cs}:{s.pos}:{s.health};".encode())   # deliberately no reward
        for e in env.enemies:
            h.update(f"E{e.id}:{e.pos}:{e.health}:{e.alive};".encode())
    for m in env.transcript.messages:
        h.update(f"[{m.step}]{m.text}|".encode())
    h.update(
        json.dumps(env._rng.bit_generator.state, default=str, sort_keys=True).encode()
    )
    return h.hexdigest()


@pytest.mark.parametrize("name, seed", [
    ("squad", 123), ("squad", 7), ("fireteam_defend", 5), ("patrol_brique", 11),
])
def test_a_reward_price_cannot_move_the_world(name, seed):
    """A price changes what a step is WORTH, never what a step DOES.

    Tested at -5.0 as well as the shipped -0.05 — a hundred times the real
    price — because a term that leaked into dynamics through a rounding path or
    an early return would leak proportionally, and a test run only at the
    shipped magnitude could miss it inside the same trajectory.
    """
    spec = get_scenario(name)
    off = dynamics_digest(spec, seed, 0.0)
    assert dynamics_digest(spec, seed, -0.05) == off
    assert dynamics_digest(spec, seed, -5.0) == off


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
