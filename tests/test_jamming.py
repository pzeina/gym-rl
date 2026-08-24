"""comm_model="jammed" — the TIME axis of degraded communications.

`range` degrades comms by *where you are*; a policy fixes it by closing
distance. Jamming degrades them by *when it is*, and cannot be walked out of.
The three properties the owner decided on 2026-08-24 are each pinned here,
because each one is a thing a later change could silently take away:
unobservable, HQ-exempt, and no failure receipt for the sender.
"""

from dataclasses import replace

import numpy as np
import pytest

from cohort.config import get_scenario
from cohort.core.orders import HQ_ID
from cohort.env.cohort_env import CohortEnv
from cohort.env.observations import OBS_DIM


def _env(seed: int = 3, scenario: str = "squad_jammed_control") -> CohortEnv:
    env = CohortEnv(get_scenario(scenario))
    env.reset(seed=seed)
    return env


def test_the_scenario_is_the_time_axis_and_nothing_else():
    """Jamming layers on `global`, not on `range`: one degradation at a time."""
    jam, squad = get_scenario("squad_jammed_control"), get_scenario("squad")
    assert jam.comm_model == "jammed"
    # sound off and no range limit — the outage is the only degradation
    assert jam.sound_model == "off"
    assert jam.comm_range == squad.comm_range
    from dataclasses import fields

    for f in fields(squad):
        if f.name in ("name", "description", "comm_model", "experiment_arm"):
            continue
        assert getattr(squad, f.name) == getattr(jam, f.name), f.name


def test_unknown_comm_model_still_raises_and_jam_knobs_are_validated():
    base = get_scenario("squad")
    with pytest.raises(ValueError, match="Unknown comm model"):
        replace(base, comm_model="smoke_signals")
    with pytest.raises(ValueError, match="jam_duty_cycle"):
        replace(base, comm_model="jammed", jam_duty_cycle=1.0)
    with pytest.raises(ValueError, match="jam_mean_outage_steps"):
        replace(base, comm_model="jammed", jam_mean_outage_steps=0.5)


def test_an_episode_always_opens_with_the_net_up():
    """The OPORD goes out at step 0; every scenario must brief identically."""
    for seed in range(20):
        assert _env(seed)._net_jammed is False


def test_who_can_hear_whom_during_an_outage():
    env = _env()
    a, b = env.roster.living[0], env.roster.living[1]

    env._net_jammed = False
    assert env._audible_to(a, b.id) is True
    assert env._audible_to(a, HQ_ID) is True

    env._net_jammed = True
    # (i) lateral cohort traffic goes dark
    assert env._audible_to(a, b.id) is False
    # (ii) HQ is exempt — the up-channel survives (owner-decided 2026-08-24)
    assert env._audible_to(a, HQ_ID) is True
    # (iii) a sender always hears itself, and is never told it reached no-one
    assert env._audible_to(a, a.id) is True


def test_the_outage_chain_hits_the_duty_cycle_and_length_it_advertises():
    """The two configured knobs must be the two observed properties — a chain
    whose stationary duty cycle drifts from `jam_duty_cycle` makes the scenario
    description a lie."""
    env = _env(seed=1)
    env._net_jammed = False
    seen = []
    for _ in range(60_000):
        env._advance_jamming()
        seen.append(env._net_jammed)

    runs, current = [], 0
    for jammed in seen:
        if jammed:
            current += 1
        elif current:
            runs.append(current)
            current = 0

    spec = get_scenario("squad_jammed_control")
    assert sum(seen) / len(seen) == pytest.approx(spec.jam_duty_cycle, abs=0.02)
    assert sum(runs) / len(runs) == pytest.approx(spec.jam_mean_outage_steps, rel=0.10)


def test_outages_are_deterministic_under_the_seed():
    """Determinism rule: all env randomness flows through `env._rng`."""

    def sequence(seed: int) -> list[bool]:
        env = _env(seed)
        out = []
        for _ in range(300):
            env._advance_jamming()
            out.append(env._net_jammed)
        return out

    assert sequence(11) == sequence(11)
    assert sequence(11) != sequence(12)


def test_jamming_is_unobservable():
    """THE contract. No agent input may encode the outage — a cohort learns the
    net is down by not being answered, not by reading a flag. If a future
    change adds a "net is down" bit, this fails and `OBS_DIM` moves with it."""
    env = _env()
    # the jammed scenario did not widen the observation
    assert env.observation_space(env.possible_agents[0])["observation"].shape[0] == OBS_DIM

    def snapshot(jammed: bool) -> dict[str, np.ndarray]:
        env._net_jammed = jammed
        # _make_view is where a leak would surface: it is what reads the
        # per-listener picture and everything else the agent is told.
        return {
            s.callsign: env._observe(s, env._make_view(s))["observation"].copy()
            for s in env.roster.living
        }

    clear, jammed = snapshot(False), snapshot(True)
    for cs in clear:
        assert np.array_equal(clear[cs], jammed[cs]), (
            f"{cs}'s observation changed with the jam state — jamming is meant "
            "to be unobservable"
        )


def test_a_contact_report_lost_to_an_outage_leaves_other_pictures_stale():
    """Per-listener pictures are why the outage bites: a globally shared
    picture would hand back the situational awareness jamming removes."""
    env = _env()
    assert env._local_pictures is True

    from cohort.env.rewards import RewardLedger

    # Enemies spawn far from the squad and neither idle nor masked-random play
    # reaches them, so put one in front of a soldier rather than hunt for a seed
    # that happens to make contact — the delivery rule is what is under test.
    reporter = env.roster.living[0]
    env.enemies[0].pos = tuple(reporter.pos)
    env.enemies[0].alive = True
    assert env._visible_enemies(reporter), "the planted enemy should be visible"

    env._net_jammed = True
    env._report_contact(reporter, RewardLedger())

    others = [s for s in env.roster.living if s.id != reporter.id]
    for s in others:
        picture = env._agent_known.get(s.callsign, {})
        for enemy in env._visible_enemies(reporter):
            assert enemy.id not in picture, (
                f"{s.callsign} learned of enemy {enemy.id} through a jammed net"
            )
