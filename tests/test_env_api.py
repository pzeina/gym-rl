"""PettingZoo API contract, determinism, and observation sanity."""

import numpy as np
import pytest

from cohort import SCENARIOS, make_env
from cohort.env.observations import OBS_DIM


def test_pettingzoo_parallel_api():
    from pettingzoo.test import parallel_api_test

    env = make_env("fireteam")
    parallel_api_test(env, num_cycles=300)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_all_scenarios_reset_and_step(scenario):
    env = make_env(scenario)
    obs, infos = env.reset(seed=0)
    assert set(obs) == set(env.possible_agents)
    for agent in env.agents:
        assert obs[agent]["observation"].shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs[agent]["observation"]))
        assert np.all(np.abs(obs[agent]["observation"]) <= 1.0)
        assert obs[agent]["action_mask"].sum() >= 1
    rng = np.random.default_rng(0)
    for _ in range(10):
        acts = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
        obs, rewards, _terms, _truncs, infos = env.step(acts)
        assert set(rewards) == set(acts)
        for a in acts:
            assert np.isfinite(rewards[a])
            assert set(infos[a]["components"]) or infos[a]["components"] == {}


def test_deterministic_given_seed():
    def rollout(seed):
        env = make_env("fireteam")
        obs, _ = env.reset(seed=seed)
        trace = []
        rng = np.random.default_rng(99)
        for _ in range(40):
            if not env.agents:
                break
            acts = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
            obs, rewards, *_ = env.step(acts)
            trace.append((tuple(sorted(rewards.items())), tuple(s.pos for s in env.roster.soldiers)))
        world_fingerprint = (env.world.grid.tobytes(), tuple(e.pos for e in env.enemies))
        return trace, env.transcript.render(), world_fingerprint

    t1, log1, w1 = rollout(1234)
    t2, log2, w2 = rollout(1234)
    assert t1 == t2
    assert log1 == log2
    assert w1 == w2
    *_, w3 = rollout(4321)
    assert w3 != w1, "different seeds should generate different worlds"


def test_episode_terminates_at_max_steps():
    env = make_env("fireteam")
    env.reset(seed=2)
    steps = 0
    while env.agents:
        acts = {a: 0 for a in env.agents}  # everyone freezes: guaranteed timeout or defeat
        _obs, _r, _t, _truncs, _i = env.step(acts)
        steps += 1
        assert steps <= env.spec_cfg.max_steps
    assert env.outcome in ("timeout", "defeat")


def test_outcome_is_public_and_none_while_running():
    env = make_env("fireteam")
    env.reset(seed=2)
    assert env.outcome is None
    env.step({a: 0 for a in env.agents})
    assert env.outcome is None, "outcome stays None until the episode ends"


def test_transcript_starts_with_opord():
    env = make_env("squad")
    env.reset(seed=8)
    first = env.transcript.messages[0]
    assert first.kind.value == "opord"
    assert "OPORD" in first.text
    assert "SEIZE OBJ ALPHA" in first.text
