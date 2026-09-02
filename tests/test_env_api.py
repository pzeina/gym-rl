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
    # global: the bisect arms present the narrow profile, the fleet the wide
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
    from cohort.core.missions import MissionType

    assert set(before["admissible_sub_missions"]) == {mission.name for mission in MissionType}
    assert before["admissible_sub_missions"]["SEIZE"] == [
        "SEIZE", "CLEAR", "SUPPORT", "OBSERVE", "ADVANCE",
    ]
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


def test_briefing_states_the_cohesion_metrics_own_constant():
    """support_umbrella is the constant no_close_teammate_rate is defined by,
    and it equals weapon_range today only by coincidence of defaults — the
    overlay must publish it itself, or a reader borrows the decoy (issue #70)."""
    from cohort.config import SCENARIOS, briefing, get_scenario

    for name in sorted(SCENARIOS):
        brief = briefing(name)
        spec = get_scenario(name)
        assert brief["support_umbrella"] == spec.combat.support_umbrella
        # both constants are published independently; neither stands in for
        # the other the day one of them is tuned alone
        assert "weapon_range" in brief


def test_briefing_carries_the_ordered_hour_the_defense_holds_to():
    """Issue #30: the horizon IS what DEFEND success means, so leaving it
    unpublished made the adjudication undecidable from outside the environment.

    v1.14 also made it the gate on the root's MISSION COMPLETE bit; v1.17
    withdrew that, so the header now carries one meaning rather than two —
    and the claim question below is answerable from the root mission alone.

    Header material by the same argument as ``announced_assault_step``: a pure
    function of the spec, so it is available before ``reset()``, identical in
    every episode, and it never enters a rollout."""
    import json

    from cohort.config import briefing, get_scenario
    from cohort.core.missions import MissionType, is_completable

    for name, spec in SCENARIOS.items():
        brief = briefing(name)
        json.dumps(brief)
        assert brief["defend_horizon"] == spec.defend_horizon, name
        # published means an outside monitor can now decide the same question
        # the mask decides — with nothing but the header. Since v1.17 the
        # horizon is not an input to that question at all: a stated hour on a
        # DEFEND root buys no permission, so a monitor that reads one must not
        # expect a root claim to follow.
        if spec.root_mission in (MissionType.DEFEND, MissionType.DENY):
            assert is_completable(spec.root_mission) is False, name

    # the defended scenarios state an hour; every other posture is indefinite
    assert briefing("fireteam_defend")["defend_horizon"] == 225
    assert briefing("defend_brique")["defend_horizon"] == 210
    assert briefing("fireteam")["defend_horizon"] is None

    # and it is the hour the episode is actually adjudicated against, before
    # and after reset, from the name alone or from the built env
    env = make_env("fireteam_defend")
    assert env.briefing()["defend_horizon"] == get_scenario("fireteam_defend").defend_horizon
    before = env.briefing()
    env.reset(seed=7)
    assert env.briefing() == before
    assert env.briefing()["defend_horizon"] == env.spec_cfg.defend_horizon


def test_briefing_carries_the_interval_a_sitrep_is_priced_against():
    """Issue #37: `closed_on_cadence_report_rate` is *defined* against
    `sitrep_interval`, and the value was spoken nowhere — not in the words
    (correctly: HQ orders an hour, it does not read out a reward weight) and
    not in the overlay. A monitor holding only the radio had to assume 25, and
    the cadence finding reverses below ~12 on the fireteam pair, so the
    assumption was load-bearing.

    Same treatment `defend_horizon` got in #30: pure function of the spec,
    identical every episode, available before `reset()`, never in a rollout.
    """
    import json
    from dataclasses import replace

    from cohort.config import SCENARIOS, briefing, get_scenario, sitrep_interval
    from cohort.env.rewards import RewardConfig

    for name, spec in SCENARIOS.items():
        brief = briefing(name)
        json.dumps(brief)
        assert brief["sitrep_interval"] == sitrep_interval(spec), name
        assert brief["sitrep_interval"] == (spec.sitrep_cadence or RewardConfig().sitrep_interval)
        assert isinstance(brief["sitrep_interval"], int), name

    # the shipped default, and the per-scenario override that is the reason
    # this cannot be a constant a monitor pins once and forgets
    assert briefing("fireteam_defend")["sitrep_interval"] == 25
    doctrine = replace(get_scenario("fireteam"), sitrep_cadence=8)
    assert briefing(doctrine)["sitrep_interval"] == 8

    # static: same before and after reset, and the same number the recorder
    # writes into the trace the cadence metric is actually computed from
    env = make_env("fireteam_defend")
    before = env.briefing()
    env.reset(seed=11)
    assert env.briefing() == before
    assert env.briefing()["sitrep_interval"] == (
        env.spec_cfg.sitrep_cadence or env.rewards_cfg.sitrep_interval
    )


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
