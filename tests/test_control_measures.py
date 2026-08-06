"""Control measures + ADVANCE (A5-1): waypoints, phase lines, orders, obs.

The A5 vocabulary names the terrain an operation maneuvers through: WAYPOINTS
(points, phonetic-metal names) and PHASE LINES (straight segments). The new
MissionType.ADVANCE anchors on one — move to / cross it, completable on
reach/cross, then hold.
"""

import numpy as np
import pytest

from cohort import make_env
from cohort.core.language import CONTROL_NAMES, PHASE_LINE_NAMES, WAYPOINT_NAMES
from cohort.core.missions import (
    COMPLETABLE,
    DOCTRINE,
    IN_POSITION_RADIUS,
    NEEDS_CONTROL,
    MissionType,
)
from cohort.core.world import PhaseLine
from cohort.env.actions import CATALOG
from cohort.env.observations import (
    N_OBJECTIVE_SLOTS,
    N_PHASE_LINE_SLOTS,
    N_WAYPOINT_SLOTS,
    OBS_DIM,
)

STAY = 0


def _stay_all(env):
    return dict.fromkeys(env.agents, STAY)


# ---------------------------------------------------------------------- #
# geometry
# ---------------------------------------------------------------------- #


def test_phase_line_geometry():
    pl = PhaseLine(id=0, name="AMBER", a=(10, 0), b=(10, 10))
    assert pl.nearest_point((0, 5)) == (10.0, 5.0)
    assert pl.nearest_point((14, 20)) == (10.0, 10.0)  # clamped to the endpoint
    assert pl.distance_to((13, 4)) == pytest.approx(3.0)
    assert pl.side((5, 5)) != pl.side((15, 5))  # opposite sides
    assert pl.side((10, 3)) == 0  # on the line


def test_names_are_disjoint_vocabularies():
    assert not set(WAYPOINT_NAMES) & set(PHASE_LINE_NAMES)
    assert not set(CONTROL_NAMES) & {"ALPHA", "BRAVO", "CHARLIE", "DELTA"}


# ---------------------------------------------------------------------- #
# doctrine + catalog
# ---------------------------------------------------------------------- #


def test_advance_doctrine():
    assert MissionType.ADVANCE in COMPLETABLE
    assert MissionType.ADVANCE in NEEDS_CONTROL
    for parent in (MissionType.SEIZE, MissionType.RECON, MissionType.DEFEND, MissionType.DENY):
        assert MissionType.ADVANCE in DOCTRINE[parent], parent
    assert DOCTRINE[MissionType.ADVANCE] == (
        MissionType.ADVANCE, MissionType.SUPPORT, MissionType.OBSERVE,
    )
    assert IN_POSITION_RADIUS[MissionType.ADVANCE] == 2.5


def test_catalog_has_advance_orders_per_slot_and_name():
    advance_specs = [
        s for s in CATALOG if s.kind == "order" and s.order_mission is MissionType.ADVANCE
    ]
    # each slot x control name exists plain AND as an AT-MY-COMMAND variant (A5-2)
    assert len(advance_specs) == 4 * len(CONTROL_NAMES) * 2
    names = {s.order_control for s in advance_specs}
    assert names == set(CONTROL_NAMES)
    assert sum(s.order_amc for s in advance_specs) == 4 * len(CONTROL_NAMES)


def test_mask_offers_only_this_maps_control_measures():
    """fireteam has WP GOLD + PL AMBER: no other ADVANCE target is legal."""
    env = make_env("fireteam")
    obs, _ = env.reset(seed=3)
    mask = obs["TL1"]["action_mask"]  # TL1 holds SEIZE: ADVANCE derivable
    legal = {
        s.order_control
        for s in CATALOG
        if s.kind == "order" and s.order_mission is MissionType.ADVANCE and mask[s.index]
    }
    assert legal == {"GOLD", "AMBER"}


# ---------------------------------------------------------------------- #
# env flow
# ---------------------------------------------------------------------- #


def test_inject_advance_to_waypoint_and_complete():
    env = make_env("fireteam")
    env.reset(seed=5)
    env.inject_order("TL1, advance to wp gold", issuer="HQ")
    tl = env.roster.by_callsign["TL1"]
    assert tl.mission is not None
    assert tl.mission.type is MissionType.ADVANCE
    assert tl.mission.extra["control"] == "GOLD"
    wp = env.world.control_by_name("GOLD")
    assert tuple(tl.mission.anchor) == tuple(wp.pos)
    # teleport onto the waypoint: the mission's end state is reached
    tl.pos = wp.pos
    tl.prev_pos = wp.pos
    from cohort.core.missions import is_complete

    ctx = env._compliance_ctx(tl, None, env._make_view(tl))
    assert ctx.in_position
    assert is_complete(tl.mission, ctx)


def test_advance_to_phase_line_completes_on_cross():
    """Crossing the line (side flip) completes even without stopping on it."""
    env = make_env("fireteam")
    env.reset(seed=5)
    env.inject_order("TL1, advance to pl amber", issuer="HQ")
    tl = env.roster.by_callsign["TL1"]
    pl = env.world.control_by_name("AMBER")
    side0 = tl.mission.extra["side"]
    assert side0 == pl.side(tl.pos)
    # move the agent to the far side of the line and tick the bookkeeping
    far = (int(2 * pl.nearest_point(tl.pos)[0] - tl.pos[0] + 6), int(2 * pl.nearest_point(tl.pos)[1] - tl.pos[1] + 6))
    tl.pos = far
    assert pl.side(far) != side0, "test setup: the agent must actually cross"
    env._update_crossing(tl)
    assert tl.mission.extra.get("crossed") is True
    from cohort.core.missions import is_complete

    ctx = env._compliance_ctx(tl, None, env._make_view(tl))
    assert is_complete(tl.mission, ctx)


def test_phase_line_anchor_is_dynamic():
    """The PL anchor follows the agent: the nearest point of the segment."""
    env = make_env("fireteam")
    env.reset(seed=5)
    env.inject_order("TL1, advance to pl amber", issuer="HQ")
    tl = env.roster.by_callsign["TL1"]
    pl = env.world.control_by_name("AMBER")
    tl.pos = (20, 30)
    a1 = env._mission_anchor(tl)
    tl.pos = (30, 18)
    a2 = env._mission_anchor(tl)
    assert a1 != a2
    assert a1 == pl.nearest_point((20, 30))
    assert a2 == pl.nearest_point((30, 18))


def test_advance_order_on_the_net_round_trips():
    env = make_env("fireteam")
    env.reset(seed=5)
    env.inject_order("TL1, advance to wp gold", issuer="HQ")
    msg = next(m for m in env.transcript.messages if m.kind.value == "order")
    assert msg.text == "TL1, THIS IS HQ: ADVANCE TO WP GOLD. OUT."
    from cohort.core import language as lang

    parsed = lang.parse_order(msg.text)
    assert parsed.mission is MissionType.ADVANCE
    assert parsed.control_name == "GOLD"


def test_advance_unknown_control_rejected():
    env = make_env("fireteam")
    env.reset(seed=5)
    from cohort.core.language import OrderParseError

    with pytest.raises(OrderParseError, match="No control measure"):
        env.inject_order("TL1, advance to wp iron", issuer="HQ")  # not on this map


def test_learned_advance_order_lands(monkeypatch=None):
    """A leader issuing ORDER_S*_ADVANCE_WP_GOLD tasks the subordinate."""
    env = make_env("fireteam")
    env.reset(seed=3)
    spec = next(
        s
        for s in CATALOG
        if s.kind == "order"
        and s.order_mission is MissionType.ADVANCE
        and s.order_slot == 0
        and s.order_control == "GOLD"
    )
    tl = env.roster.by_callsign["TL1"]
    assert env._mask_for(tl)[spec.index] == 1
    actions = _stay_all(env)
    actions["TL1"] = spec.index
    env.step(actions)
    rfn = tl.living_subordinates(env.roster)[0]
    assert rfn.mission is not None
    assert rfn.mission.type is MissionType.ADVANCE
    assert rfn.mission.extra["control"] == "GOLD"
    order_msgs = [m for m in env.transcript.messages if m.kind.value == "order"]
    assert any("ADVANCE TO WP GOLD" in m.text for m in order_msgs)


# ---------------------------------------------------------------------- #
# observations
# ---------------------------------------------------------------------- #


def test_control_measure_obs_slots():
    """Waypoint/phase-line slots carry presence + direction; empties zeroed."""
    env = make_env("fireteam")
    obs, _ = env.reset(seed=7)
    tl = env.roster.by_callsign["TL1"]
    vec = obs["TL1"]["observation"]
    base = OBS_DIM - 50 - 5 - 3 * N_PHASE_LINE_SLOTS - 3 * N_WAYPOINT_SLOTS
    # waypoint slot 0 = GOLD, present, dx/dy point at it
    wp = env.world.waypoints[0]
    assert vec[base] == 1.0
    assert vec[base + 1] == pytest.approx((wp.pos[0] - tl.pos[0]) / env.world.width)
    assert vec[base + 2] == pytest.approx((wp.pos[1] - tl.pos[1]) / env.world.height)
    # waypoint slots 1..3 empty on fireteam
    for k in range(1, N_WAYPOINT_SLOTS):
        assert vec[base + 3 * k] == 0.0
    # phase-line slot 0 = AMBER: present, dx/dy at the segment's nearest point
    plbase = base + 3 * N_WAYPOINT_SLOTS
    pl = env.world.phase_lines[0]
    near = pl.nearest_point(tl.pos)
    assert vec[plbase] == 1.0
    assert vec[plbase + 1] == pytest.approx((near[0] - tl.pos[0]) / env.world.width)
    assert vec[plbase + 2] == pytest.approx((near[1] - tl.pos[1]) / env.world.height)
    for k in range(1, N_PHASE_LINE_SLOTS):
        assert vec[plbase + 3 * k] == 0.0


def test_every_preset_has_control_measures():
    from cohort.config import SCENARIOS

    for name, spec in SCENARIOS.items():
        assert spec.waypoints or spec.phase_lines, f"{name} has no control measures"


def test_obs_dim_math():
    # 13 self + 17 mission + 5 leader + 20 subs + 16 enemies + 12 obj
    # + 12 wp + 9 pl + 5 comms + 50 patch = 166 (mission 22 + sync 2)
    assert OBS_DIM == 166
    assert N_OBJECTIVE_SLOTS == 4
    assert N_WAYPOINT_SLOTS == 4
    assert N_PHASE_LINE_SLOTS == 3


def test_waypoints_are_standable():
    """Map generation keeps waypoint cells passable, like objectives."""
    for scenario in ("fireteam", "squad", "platoon"):
        env = make_env(scenario)
        env.reset(seed=11)
        for wp in env.world.waypoints:
            assert env.world.passable(wp.pos), f"{scenario} WP {wp.name}"


# ---------------------------------------------------------------------- #
# determinism
# ---------------------------------------------------------------------- #


def test_episode_determinism_with_advance_orders():
    def run(seed):
        env = make_env("fireteam")
        env.reset(seed=seed)
        env.inject_order("TL1, advance to wp gold", issuer="HQ")
        rng = np.random.default_rng(seed)
        trail = []
        for _ in range(40):
            if not env.agents:
                break
            acts = {}
            obs = env._all_observations()
            for a in env.agents:
                legal = np.flatnonzero(obs[a]["action_mask"])
                acts[a] = int(rng.choice(legal))
            env.step(acts)
            trail.append(tuple(s.pos for s in env.roster.soldiers))
        return trail

    assert run(21) == run(21)
