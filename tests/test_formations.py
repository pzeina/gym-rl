"""Formations (A5-3): element stances COLUMN / LINE / WEDGE.

Manual pp. 14-15 (2.2 Les actes elementaires du groupe): the group moves in
three formations — en colonne, en ligne, en colonne double; WEDGE stands in
for the colonne double's two-directions role (owner scope). A stance is
ordered to a LEADER, persists until changed, shapes (never forces) the
element's geometry, and is visible in obs and to the probe.
"""

import pytest

from cohort import make_env
from cohort.core import language as lang
from cohort.core.missions import Formation, MissionType, in_formation
from cohort.env.actions import CATALOG
from cohort.env.observations import FORMATION_ORDER

STAY = 0


def _stay_all(env):
    return dict.fromkeys(env.agents, STAY)


def _formation_spec(slot, formation):
    return next(
        s
        for s in CATALOG
        if s.kind == "order" and s.order_slot == slot and s.order_formation is formation
    )


# ---------------------------------------------------------------------- #
# geometry
# ---------------------------------------------------------------------- #


HEADING_EAST = (1, 0)


def test_column_geometry():
    """COLUMN: trail behind the leader within 1-cell lateral."""
    leader = (10, 10)
    assert in_formation(Formation.COLUMN, leader, HEADING_EAST, (8, 10))
    assert in_formation(Formation.COLUMN, leader, HEADING_EAST, (7, 11))  # 1 off-axis ok
    assert not in_formation(Formation.COLUMN, leader, HEADING_EAST, (12, 10))  # ahead
    assert not in_formation(Formation.COLUMN, leader, HEADING_EAST, (8, 13))  # too wide
    assert not in_formation(Formation.COLUMN, leader, HEADING_EAST, (1, 10))  # too far back


def test_line_geometry():
    """LINE: abreast within 1-cell depth."""
    leader = (10, 10)
    assert in_formation(Formation.LINE, leader, HEADING_EAST, (10, 12))
    assert in_formation(Formation.LINE, leader, HEADING_EAST, (11, 8))  # 1 ahead ok
    assert not in_formation(Formation.LINE, leader, HEADING_EAST, (13, 10))  # on-axis ahead
    assert not in_formation(Formation.LINE, leader, HEADING_EAST, (10, 10))  # on the leader
    assert not in_formation(Formation.LINE, leader, HEADING_EAST, (10, 18))  # too wide


def test_wedge_geometry():
    """WEDGE: V at diagonal offsets behind the leader."""
    leader = (10, 10)
    assert in_formation(Formation.WEDGE, leader, HEADING_EAST, (9, 11))
    assert in_formation(Formation.WEDGE, leader, HEADING_EAST, (8, 8))
    assert not in_formation(Formation.WEDGE, leader, HEADING_EAST, (8, 10))  # on-axis
    assert not in_formation(Formation.WEDGE, leader, HEADING_EAST, (11, 11))  # ahead
    assert not in_formation(Formation.WEDGE, leader, HEADING_EAST, (9, 5))  # off the V


def test_no_heading_no_formation():
    assert not in_formation(Formation.COLUMN, (10, 10), (0, 0), (8, 10))


# ---------------------------------------------------------------------- #
# language
# ---------------------------------------------------------------------- #


def test_formation_order_round_trip():
    for formation in Formation:
        text = lang.format_formation_order("SL1", "TL1", formation)
        assert text == f"TL1, THIS IS SL1: FORMATION {formation.name}. OUT."
        parsed = lang.parse_order(text)
        assert parsed.recipient_callsign == "TL1"
        assert parsed.mission is None
        assert parsed.formation is formation


def test_formation_parse_variants():
    assert lang.parse_order("TL1, formation column").formation is Formation.COLUMN
    assert lang.parse_order("tl2: formation ligne. out.").formation is Formation.LINE
    assert lang.parse_order("TL1, formation wedge").formation is Formation.WEDGE


# ---------------------------------------------------------------------- #
# orders + masks
# ---------------------------------------------------------------------- #


def test_formation_orderable_to_leaders_only():
    """SL1 may stance its TLs (they lead elements); a TL may not stance an RFN."""
    env = make_env("squad")
    env.reset(seed=3)
    sl = env.roster.by_callsign["SL1"]
    spec = _formation_spec(0, Formation.COLUMN)
    assert env._mask_for(sl)[spec.index] == 1
    # give TL1 a mission so its order block opens; its subs are RFNs (lead nobody)
    env.inject_order("TL1, seize obj alpha", issuer="SL1")
    tl = env.roster.by_callsign["TL1"]
    for formation in Formation:
        for slot in range(2):
            assert env._mask_for(tl)[_formation_spec(slot, formation).index] == 0


def test_stance_set_persists_and_transcript():
    env = make_env("squad")
    env.reset(seed=3)
    spec = _formation_spec(0, Formation.WEDGE)
    acts = _stay_all(env)
    acts["SL1"] = spec.index
    env.step(acts)
    tl = env.roster.by_callsign["TL1"]
    assert tl.formation is Formation.WEDGE
    order = next(m for m in env.transcript.messages if "FORMATION" in m.text)
    assert order.text == "TL1, THIS IS SL1: FORMATION WEDGE. OUT."
    # persists across steps until changed
    for _ in range(5):
        env.step(_stay_all(env))
    assert tl.formation is Formation.WEDGE
    # reissuing the standing stance is churn (no message, order_churn cost)
    before = len(env.transcript)
    acts = _stay_all(env)
    acts["SL1"] = spec.index
    env.step(acts)
    assert tl.formation is Formation.WEDGE
    assert not any(
        "FORMATION" in m.text for m in env.transcript.messages[before:]
    ), "identical stance reissue is a churn no-op"


def test_inject_formation_order():
    env = make_env("squad")
    env.reset(seed=3)
    env.inject_order("TL2, formation line", issuer="SL1")
    assert env.roster.by_callsign["TL2"].formation is Formation.LINE
    with pytest.raises(PermissionError, match="leads no element"):
        env.inject_order("RFN1, formation column", issuer="HQ")


# ---------------------------------------------------------------------- #
# shaping economics
# ---------------------------------------------------------------------- #


def _straight_env():
    """Flat squad map; SL1 stanced COLUMN, tasked SEIZE ALPHA far east."""
    env = make_env("squad")
    env.reset(seed=3)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (41, 1)
        e.home = e.pos
    sl = env.roster.by_callsign["SL1"]
    tl = env.roster.by_callsign["TL1"]
    rfn = env.roster.by_callsign["RFN1"]
    # geometry: TL1 leads RFN1/RFN2; stance on TL1; RFN1 trails in column
    env.inject_order("TL1, seize obj alpha", issuer="SL1")
    env.inject_order("TL1, formation column", issuer="SL1")
    sl.pos = (5, 20)
    tl.pos = (10, 20)
    tl.prev_pos = (10, 20)
    tl.heading = (1, 0)
    rfn.pos = (8, 20)  # 2 behind on the axis
    rfn.prev_pos = (8, 20)
    rfn2 = env.roster.by_callsign["RFN2"]
    rfn2.pos = (10, 26)  # far off station
    rfn2.prev_pos = (10, 26)
    return env, tl, rfn, rfn2


MOVE_EAST = next(s.index for s in CATALOG if s.name == "MOVE_EAST")


def test_formation_bonus_paid_in_station_while_leader_advances():
    env, _tl, _rfn, _rfn2 = _straight_env()
    acts = _stay_all(env)
    acts["TL1"] = MOVE_EAST  # leader closes on ALPHA: watermark set on 1st move
    env.step(acts)
    acts = _stay_all(env)
    acts["TL1"] = MOVE_EAST
    acts["RFN1"] = MOVE_EAST  # keeps station 2 behind
    _, rewards, *_ = env.step(acts)
    # RFN1 was in COLUMN station while the leader set a new watermark
    assert rewards["RFN1"] > rewards["RFN2"], "in-station member out-earns the straggler"


def test_formation_bonus_watermark_blocks_circling():
    """Pacing back and forth pays at most once per cell of NEW closure."""
    env, tl, rfn, _rfn2 = _straight_env()
    cfg = env.rewards_cfg
    move_w = next(s.index for s in CATALOG if s.name == "MOVE_WEST")
    total_bonus = 0.0
    # east (new ground), west (retreat), east (old ground: no pay), east (new)
    for a in (MOVE_EAST, move_w, MOVE_EAST, MOVE_EAST):
        acts = _stay_all(env)
        acts["TL1"] = a
        rfn.pos = (tl.pos[0] - 2, 20)  # keep the member glued to its station
        rfn.prev_pos = rfn.pos
        _, rewards, *_ = env.step(acts)
        base = -0.01 - 0.01 * 0  # time penalty (no other rfn income is formation-sized)
        del base
        total_bonus += max(0.0, rewards["RFN1"] - rewards["RFN2"])
    # 3 east moves but only 2 cells of NEW closure past the first watermark set
    assert total_bonus <= cfg.formation_bonus * 2 + 1e-6


def test_stance_dies_with_the_leader():
    env = make_env("squad")
    env.reset(seed=3)
    env.inject_order("TL1, formation column", issuer="SL1")
    tl = env.roster.by_callsign["TL1"]
    assert tl.formation is Formation.COLUMN
    ledger_deaths = []
    from cohort.env.rewards import RewardLedger

    env._damage_soldier(tl, 1000, RewardLedger(), ledger_deaths)
    for dead in ledger_deaths:
        for successor, _replaced in env.roster.succeed(dead):
            assert successor.formation is None, "stance does not transfer"


# ---------------------------------------------------------------------- #
# observations
# ---------------------------------------------------------------------- #


def test_stance_one_hot_in_obs():
    env = make_env("squad")
    env.reset(seed=3)
    env.inject_order("TL1, formation wedge", issuer="SL1")
    obs = env._all_observations()
    base = 13 + 12 + 1 + 4 + 2  # stance block after mission one-hot/anchor/pending
    idx = FORMATION_ORDER.index(Formation.WEDGE)
    # the leader itself shows its element's stance
    assert obs["TL1"]["observation"][base + idx] == 1.0
    # its members show the governing (leader's) stance
    assert obs["RFN1"]["observation"][base + idx] == 1.0
    # unrelated element shows nothing
    assert obs["TL2"]["observation"][base : base + 3].sum() == 0.0


# ---------------------------------------------------------------------- #
# probe
# ---------------------------------------------------------------------- #


def test_probe_stanced_members_follow_leader():
    from cohort.probe import LEADER, MOVING, NetPredictor, make_briefing, obj_class

    p = NetPredictor(make_briefing("squad"))
    p.observe(0, [
        {"kind": "opord", "from": "HQ", "to": "SL1",
         "text": lang.format_opord("SL1", MissionType.SEIZE, "ALPHA")},
        {"kind": "order", "from": "SL1", "to": "TL1",
         "text": lang.format_order("SL1", "TL1", MissionType.SEIZE, "ALPHA")},
        {"kind": "order", "from": "SL1", "to": "TL1",
         "text": lang.format_formation_order("SL1", "TL1", Formation.COLUMN)},
    ])
    # untasked members of TL1's element now follow TL1 instead of HOLD
    assert p.predict("RFN1") == (obj_class("ALPHA"), MOVING)
    # once the leader has arrived, members hold formation ON the leader
    p.observe(3, [{"kind": "sitrep", "from": "TL1", "to": "SL1",
                   "text": lang.format_sitrep("SL1", "TL1", 100, 30, (33, 33), in_cover=False)}])
    p.observe(4, [])
    dest, _post = p.predict("RFN1")
    assert dest == LEADER


def test_probe_without_stance_untasked_is_hold():
    from cohort.probe import HOLD, NetPredictor, make_briefing

    p = NetPredictor(make_briefing("squad"))
    p.observe(0, [
        {"kind": "order", "from": "SL1", "to": "TL1",
         "text": lang.format_order("SL1", "TL1", MissionType.SEIZE, "ALPHA")},
    ])
    assert p.predict("RFN1")[0] == HOLD, "no stance on the net: B4 behavior stands"
