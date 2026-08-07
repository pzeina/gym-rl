"""PettingZoo API contract, determinism, and observation sanity."""

import numpy as np
import pytest

from cohort import SCENARIOS, make_env
from cohort.env.observations import obs_dim


def test_pettingzoo_parallel_api():
    from pettingzoo.test import parallel_api_test

    env = make_env("fireteam")
    parallel_api_test(env, num_cycles=300)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_all_scenarios_reset_and_step(scenario):
    env = make_env(scenario)
    obs, infos = env.reset(seed=0)
    assert set(obs) == set(env.possible_agents)
    # width is a property of the scenario's observation profile, not a
    # global: the bisect arms present 166 where the fleet presents 220
    width = obs_dim(env.spec_cfg.observation_profile)
    assert env.observation_space(env.agents[0])["observation"].shape == (width,)
    for agent in env.agents:
        assert obs[agent]["observation"].shape == (width,)
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


# ---------------------------------------------------------------------- #
# static briefing / observable-stream header (issue #10)
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_briefing_is_static_json_ready_and_scenario_derived(scenario):
    """Header material: identical across episodes, valid before reset, and
    carrying every anchor a monitor would otherwise hand-pin in a table."""
    import json

    from cohort.config import briefing, get_scenario

    env = make_env(scenario)
    before = env.briefing()                       # available with no world built
    json.dumps(before)                            # JSON-ready: it is a stream header
    assert before == briefing(scenario)           # same from the scenario name alone

    spec = get_scenario(scenario)
    assert before["map_size"] == list(spec.map_size)
    assert before["root_mission"] == spec.root_mission.name
    assert set(before["objectives"]) == {name for name, _ in spec.objectives}
    assert set(before["waypoints"]) == {name for name, _ in spec.waypoints}
    assert set(before["phase_lines"]) == {name for name, *_ in spec.phase_lines}
    assert before["terrain_static"] is False, "the grid is regenerated every reset"

    # the coordinates are the ones the episode actually uses, and they do not
    # drift between episodes — the era-sensitivity issue #10 reported
    env.reset(seed=1)
    for name, pos in before["objectives"].items():
        assert list(env.world.objective_by_name(name).pos) == pos
    for name, pos in before["waypoints"].items():
        assert list(env.world.control_by_name(name).pos) == pos
    env.reset(seed=99)
    assert env.briefing() == before


def test_briefing_carries_the_defend_objective_coordinates():
    """The concrete case from issue #10: OBJ ALPHA of `fireteam_defend`."""
    from cohort.config import briefing

    assert briefing("fireteam_defend")["objectives"] == {"ALPHA": [18, 18]}
    assert briefing("fireteam_defend")["objective_cover"] is True


def test_sitrep_posture_matches_the_ground_the_sender_stands_on():
    """The self-report must be true: what the soldier says about its cover is
    what `world.cover_at` says (issue #10 — the correlate is only worth
    measuring if the net's version of it is honest)."""
    from cohort.core.language import parse_sitrep
    from cohort.env.actions import CATALOG

    sitrep_idx = next(s.index for s in CATALOG if s.kind == "sitrep")
    env = make_env("fireteam_defend")
    obs, _ = env.reset(seed=4)
    seen = set()
    for _ in range(40):
        acts = {}
        for a in env.agents:
            mask = obs[a]["action_mask"]
            acts[a] = sitrep_idx if mask[sitrep_idx] else int(np.flatnonzero(mask)[0])
        obs, *_ = env.step(acts)
        for m in env.last_messages:
            if m.kind.value != "sitrep":
                continue
            reported = parse_sitrep(m.text)
            assert reported is not None, f"every SITREP must parse: {m.text!r}"
            sender = env.roster.by_id[m.sender_id]
            assert reported["grid"] == tuple(sender.pos)
            assert reported["in_cover"] == env.world.cover_at(sender.pos)
            seen.add(reported["in_cover"])
        if not env.agents:
            break
    assert seen, "no SITREP was transmitted — the test asserted nothing"
