"""The defend preparation period (v1.10): ScenarioSpec.assault_h_hour.

Before H the OpFor is on the map — spawned, oracle-visible, spottable — but
does not move, fire, or advance. A DEFEND mission presumes prepared positions;
the preparation period is the time to occupy them.

The OPORD announces the band's MIDPOINT as the nominal H while the assault
actually arrives anywhere in the band, so the habit the scenario rewards is
*being set early*, not *timing the tick*.
"""

from dataclasses import replace

import pytest

from cohort import make_env
from cohort.config import get_scenario
from cohort.env.observations import TEMPO_TIME_TO_CONTACT

BAND = (55, 75)


def _defend(seed=7, **overrides):
    spec = get_scenario("fireteam_defend")
    if overrides:
        spec = replace(spec, **overrides)
    env = make_env(spec)
    obs, _ = env.reset(seed=seed)
    return env, obs


def test_h_hour_is_drawn_inside_the_band_and_the_nominal_is_the_midpoint():
    for seed in range(12):
        env, _ = _defend(seed=seed)
        assert BAND[0] <= env._h_hour <= BAND[1], f"seed {seed}: H={env._h_hour}"
        assert env._h_hour_nominal == (BAND[0] + BAND[1]) // 2


def test_h_hour_is_deterministic_per_seed():
    """Determinism convention: all env randomness through the seeded _rng."""
    a, _ = _defend(seed=11)
    b, _ = _defend(seed=11)
    assert a._h_hour == b._h_hour
    assert [e.pos for e in a.enemies] == [e.pos for e in b.enemies]


def test_the_band_actually_varies_the_arrival():
    """A constant H would let a policy time the tick instead of standing to."""
    drawn = {_defend(seed=s)[0]._h_hour for s in range(30)}
    assert len(drawn) > 1, "H must jitter across seeds"


def test_the_opord_announces_the_nominal_h_on_the_net():
    env, _ = _defend()
    opord = env.transcript.messages[0]
    assert "EXPECT ASSAULT AT H PLUS 65" in opord.text
    # ...and the task statement still parses: the warning is not the order
    from cohort.core import language as lang
    from cohort.core.missions import MissionType

    parsed = lang.parse_order(opord.text)
    assert parsed.mission is MissionType.DEFEND
    assert parsed.objective_name == "ALPHA"


def test_opfor_is_frozen_until_h_then_advances():
    env, _ = _defend(seed=7)
    h = env._h_hour
    start = [e.pos for e in env.enemies]
    for _ in range(h - 1):  # the H-th step IS the assault: hold up to H-1
        env.step({a: 0 for a in env.agents})
    assert env._step_count == h - 1 and env._in_preparation()
    assert [e.pos for e in env.enemies] == start, "no OpFor movement before H"
    assert all(s.alive for s in env.roster.soldiers), "and no OpFor fire before H"
    for _ in range(15):
        env.step({a: 0 for a in env.agents})
    assert [e.pos for e in env.enemies] != start, "the assault begins at H"


def test_opfor_exists_from_step_zero_even_while_held():
    """Held, not absent: the oracle sees the assault forming from step 0, and a
    patrol that goes looking can spot it (the early warning a defense earns)."""
    env, _ = _defend()
    assert len(env.enemies) == env.spec_cfg.n_enemies
    assert all(e.alive for e in env.enemies)
    assert env._in_preparation()


def test_time_to_contact_counts_down_to_the_nominal_h():
    env, obs = _defend()
    nominal = env._h_hour_nominal
    assert obs["TL1"]["observation"][TEMPO_TIME_TO_CONTACT] == 1.0
    prev = 1.0
    for step in range(1, nominal + 5):
        obs, *_ = env.step({a: 0 for a in env.agents})
        ttc = obs["TL1"]["observation"][TEMPO_TIME_TO_CONTACT]
        if step < nominal:
            assert ttc == pytest.approx((nominal - step) / nominal, abs=1e-6)
            assert ttc < prev, "monotone countdown"
            prev = ttc
        else:
            assert ttc == 0.0, "0 once the announced hour passes"


def test_scenarios_without_a_preparation_period_are_untouched():
    for name in ("fireteam", "squad", "platoon", "squad_recon"):
        env = make_env(name)
        env.reset(seed=3)
        assert env.spec_cfg.assault_h_hour is None
        assert env._h_hour is None
        assert not env._in_preparation()
        assert "EXPECT ASSAULT" not in env.transcript.messages[0].text


def test_the_brique_band_is_frozen_before_h_like_any_other_opfor():
    """v1.12: `defend_brique` earned a preparation period too.

    The spec had asked for a DEFEND root and `objective_cover=True` and then
    let the band run from step 0, so the fire team was never given a moment to
    occupy the ground the scenario had built for it. The gate in `step()` is
    mode-agnostic, so the fix is a spec value — but the protection worth
    holding is behavioral: the band does not move before H, and does after.
    """
    spec = get_scenario("defend_brique")
    assert spec.assault_h_hour == (35, 55)
    # the preparation is bought, not taken out of the fight it precedes
    assert spec.max_steps - ((35 + 55) // 2) == 375

    env = make_env("defend_brique")
    env.reset(seed=11)
    assert env.band is not None, "still the BRIQUE band, not a formed assault"
    h = env._h_hour
    assert 35 <= h <= 55

    start = [e.pos for e in env.enemies]
    actions = {c: 0 for c in env.agents}
    for _ in range(h - 1):
        env.step(actions)
    assert env._in_preparation()
    assert [e.pos for e in env.enemies] == start, "band moved during preparation"

    for _ in range(25):
        env.step(actions)
    assert not env._in_preparation()
    assert [e.pos for e in env.enemies] != start, "band never started after H"


def test_no_preparation_period_consumes_no_randomness():
    """The draw is guarded, so every other scenario's seeds reproduce exactly."""
    a = make_env("fireteam")
    a.reset(seed=5)
    b = make_env(replace(get_scenario("fireteam"), max_steps=999))
    b.reset(seed=5)
    assert [e.pos for e in a.enemies] == [e.pos for e in b.enemies]
    assert [s.pos for s in a.roster.soldiers] == [s.pos for s in b.roster.soldiers]


# ---------------------------------------------------------------------- #
# preparation-period occupancy pay (v1.10)
# ---------------------------------------------------------------------- #


def _place(env, callsign, pos):
    env.roster.by_callsign[callsign].pos = pos


def test_occupancy_pays_only_in_cover_at_the_objective():
    """Cover is required, not proximity: bare ground at the objective is not a
    prepared position (the v1.2 terrain lesson)."""
    from cohort.core.missions import IN_POSITION_RADIUS, MissionType
    from cohort.core.world import FOREST, OPEN

    env, _ = _defend()
    obj = env.world.objective_by_name("ALPHA")
    radius = IN_POSITION_RADIUS[MissionType.DEFEND]
    covered, bare = (obj.pos[0] + 1, obj.pos[1]), (obj.pos[0] - 1, obj.pos[1])
    env.world.grid[covered[1], covered[0]] = FOREST
    env.world.grid[bare[1], bare[0]] = OPEN
    far = (obj.pos[0], obj.pos[1] + int(radius) + 4)
    env.world.grid[far[1], far[0]] = FOREST  # in cover, but off the position
    _place(env, "RFN1", covered)
    _place(env, "RFN2", bare)
    _place(env, "RFN3", far)
    *_, infos = env.step({a: 0 for a in env.agents})
    cfg = env.rewards_cfg
    assert infos["RFN1"]["components"]["compliance"] >= cfg.prep_in_position
    base = infos["RFN2"]["components"]["compliance"]
    assert infos["RFN1"]["components"]["compliance"] > base, "cover at the position pays"
    assert infos["RFN3"]["components"]["compliance"] < base + cfg.prep_in_position, (
        "cover off the position does not"
    )


def test_occupancy_stops_paying_at_h():
    """Bounded by H — it cannot be farmed for the length of the episode."""
    from cohort.core.world import FOREST

    env, _ = _defend()
    obj = env.world.objective_by_name("ALPHA")
    covered = (obj.pos[0] + 1, obj.pos[1])
    env.world.grid[covered[1], covered[0]] = FOREST
    cfg = env.rewards_cfg
    for _ in range(env._h_hour - 1):
        _place(env, "RFN1", covered)
        *_, infos = env.step({a: 0 for a in env.agents})
        during = infos["RFN1"]["components"]["compliance"]
    assert during >= cfg.prep_in_position, "paid throughout the preparation period"
    for _ in range(3):  # past H: the assault has begun
        _place(env, "RFN1", covered)
        *_, infos = env.step({a: 0 for a in env.agents})
    assert infos["RFN1"]["components"]["compliance"] < during, "no pay after H"


def test_occupancy_is_absent_without_a_preparation_period():
    from cohort.core.world import FOREST

    env = make_env("fireteam")
    env.reset(seed=4)
    obj = env.world.objective_by_name("ALPHA")
    covered = (obj.pos[0] + 1, obj.pos[1])
    env.world.grid[covered[1], covered[0]] = FOREST
    _place(env, "RFN1", covered)
    env.step({a: 0 for a in env.agents})
    assert not env._in_preparation()
    # whatever compliance it earns, none of it is prep occupancy
    assert env.spec_cfg.assault_h_hour is None


# ---------------------------------------------------------------------- #
# the announcement on the observable surface (issue #12)
#
# The OPORD's "EXPECT ASSAULT AT H PLUS 65" is the first forward-looking
# content the net has ever carried, and it was reaching an outside monitor
# nowhere: said once, then dropped at the boundary. Two routes now carry it —
# language.parse_opord (read it back off the transcript) and briefing()
# (hold it as header material, before reset and for corpora that predate the
# clause). What stays behind is the ACTUAL drawn arrival: announced != actual
# is the inference problem, so the truth lives only in env.oracle().
# ---------------------------------------------------------------------- #

ANNOUNCED = (BAND[0] + BAND[1]) // 2  # the OPORD says the band's midpoint


def _announced_steps_in(transcript):
    """Every "H PLUS <n>" said on the net, as a set of ints."""
    import re

    return {int(n) for m in transcript.messages for n in re.findall(r"H PLUS (\d+)", m.text)}


def test_the_announced_step_round_trips_out_of_the_opord_text():
    """Formatter and parser stay inverses over the clause, not just the task."""
    from cohort.core import language as lang
    from cohort.core.missions import MissionType

    for step in (0, 7, 65, 1234):
        text = lang.format_opord("TL1", MissionType.DEFEND, "ALPHA", step)
        assert lang.parse_opord(text) == {
            "recipient": "TL1",
            "mission": "DEFEND",
            "objective": "ALPHA",
            "announced_assault_step": step,
        }
    # an OPORD with no preparation period parses too, with the clause absent
    plain = lang.format_opord("SL1", MissionType.SEIZE, "ALPHA")
    assert lang.parse_opord(plain)["announced_assault_step"] is None
    assert lang.parse_opord(plain)["mission"] == "SEIZE"


def test_the_announced_step_is_recoverable_from_the_episode_transcript():
    """The path a monitor actually takes: read the net, get the deadline."""
    from cohort.core import language as lang

    env, _ = _defend()
    opord = env.transcript.messages[0]
    assert lang.parse_opord(opord.text)["announced_assault_step"] == ANNOUNCED


def test_parse_opord_declines_traffic_that_is_not_an_opord():
    from cohort.core import language as lang

    env, _ = _defend()
    env.step({a: 0 for a in env.agents})
    for msg in env.transcript.messages[1:]:
        assert lang.parse_opord(msg.text) is None, msg.text
    assert lang.parse_opord("TL1, THIS IS RFN1: SITREP, GRID 0912, "
                            "HEALTH 66%, AMMO 24, IN COVER. OVER.") is None


def test_the_briefing_carries_the_announced_step_before_reset():
    """Header material: a pure function of the scenario, so a corpus that
    predates the clause — or a listener that never heard the broadcast —
    still gives a monitor the deadline."""
    import json

    from cohort.config import briefing

    brief = briefing("fireteam_defend")
    json.dumps(brief)
    assert brief["announced_assault_step"] == ANNOUNCED
    env, _ = _defend()
    assert env.briefing()["announced_assault_step"] == env._h_hour_nominal
    for name in ("fireteam", "squad", "squad_recon"):
        assert briefing(name)["announced_assault_step"] is None


def test_the_actual_assault_step_never_reaches_an_observable_payload():
    """The announcement is public; the draw is the answer, and stays hidden."""
    from cohort.core import language as lang

    jittered = [s for s in range(12) if _defend(seed=s)[0]._h_hour != ANNOUNCED]
    assert jittered, "the band must jitter, or this asserts nothing"
    for seed in jittered:
        env, _ = _defend(seed=seed)
        actual = env._h_hour
        assert env.briefing()["announced_assault_step"] == ANNOUNCED != actual
        # the briefing cannot carry the draw at all: it is the same dict in
        # every episode, whatever this episode's H turned out to be
        assert env.briefing() == _defend(seed=seed + 100)[0].briefing()
        for msg in env.transcript.messages:
            parsed = lang.parse_opord(msg.text)
            if parsed is not None:
                assert parsed["announced_assault_step"] == ANNOUNCED
        assert env.oracle()["actual_assault_step"] == actual, "the truth is oracle-side"


def test_no_message_ever_names_the_actual_assault_step():
    """Including after H: the assault arriving is not itself announced."""
    seed = next(s for s in range(12) if _defend(seed=s)[0]._h_hour != ANNOUNCED)
    env, _ = _defend(seed=seed)
    for _ in range(env._h_hour + 5):
        env.step({a: 0 for a in env.agents})
    assert _announced_steps_in(env.transcript) == {ANNOUNCED}


def test_the_oracle_carries_both_the_announcement_and_the_truth():
    env, _ = _defend()
    snap = env.oracle()
    assert snap["announced_assault_step"] == ANNOUNCED
    assert BAND[0] <= snap["actual_assault_step"] <= BAND[1]
    plain = make_env("fireteam")
    plain.reset(seed=2)
    assert plain.oracle()["announced_assault_step"] is None
    assert plain.oracle()["actual_assault_step"] is None
