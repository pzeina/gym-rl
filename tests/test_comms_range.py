"""comm_model="range": per-listener audibility on the radio net.

Default ("global") is exactly the shipped behavior: every station hears every
message. Under "range", a message is heard only by stations within
``comm_range`` of the sender (euclidean; the sender hears itself; HQ is a
high-power station). CONTACT reports feed only the pictures of stations in
earshot, and an unheard ORDER assigns nothing — silence is the only clue.
"""

from dataclasses import replace

from cohort import make_env
from cohort.config import get_scenario
from cohort.core.missions import MissionType
from cohort.env.actions import CATALOG

STAY = 0
CONTACT_IDX = next(s.index for s in CATALOG if s.kind == "contact")
ORDER_SEIZE_S0 = next(
    s.index
    for s in CATALOG
    if s.kind == "order"
    and s.order_slot == 0
    and s.order_mission is MissionType.SEIZE
    and s.order_objective == "ALPHA"
)

#: Offset of the comms-summary "known enemy count" field in the observation:
#: 13 self + 16 mission + 5 leader + 5*4 subs + 4*4 enemies + 3*4 obj = 82,
#: comms block = [new-order flag, known count, known present, dx, dy].
KNOWN_COUNT_FIELD = 83


def _flat_env(spec, seed=1):
    env = make_env(spec)
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (22, 1)
        e.home = e.pos
    return env


def _range_env(comm_range=5.0, seed=1):
    return _flat_env(
        replace(get_scenario("fireteam"), comm_model="range", comm_range=comm_range), seed=seed
    )


def _step_all(env, overrides):
    acts = {a: STAY for a in env.agents}
    acts.update(overrides)
    return env.step(acts)


def test_out_of_range_contact_stays_private():
    """An out-of-range rifleman's CONTACT does not update a distant leader."""
    env = _range_env(comm_range=5.0)
    env.roster.by_callsign["RFN1"].pos = (7, 10)
    env.roster.by_callsign["RFN2"].pos = (9, 10)   # within 5 of the sender
    env.roster.by_callsign["TL1"].pos = (20, 20)   # far out of earshot
    env.roster.by_callsign["RFN3"].pos = (3, 3)
    enemy = env.enemies[0]
    enemy.pos = (10, 10)
    enemy.home = enemy.pos
    obs, *_ = _step_all(env, {"RFN1": CONTACT_IDX})
    assert env._agent_known["RFN1"], "the sender hears itself"
    assert env._agent_known["RFN2"], "an in-range teammate heard the report"
    assert not env._agent_known["TL1"], "the out-of-range leader heard nothing"
    assert obs["TL1"]["observation"][KNOWN_COUNT_FIELD] == 0.0
    assert obs["RFN2"]["observation"][KNOWN_COUNT_FIELD] > 0.0


def test_in_range_contact_updates_leader():
    env = _range_env(comm_range=15.0)
    env.roster.by_callsign["RFN1"].pos = (7, 10)
    env.roster.by_callsign["TL1"].pos = (12, 10)   # 5 cells: audible
    enemy = env.enemies[0]
    enemy.pos = (10, 10)
    enemy.home = enemy.pos
    obs, *_ = _step_all(env, {"RFN1": CONTACT_IDX})
    assert env._agent_known["TL1"], "in-range leader's picture updates"
    assert obs["TL1"]["observation"][KNOWN_COUNT_FIELD] > 0.0


def test_global_mode_shares_the_picture():
    """Default comm_model: the shipped behavior, one picture for everyone."""
    env = _flat_env(get_scenario("fireteam"))
    env.roster.by_callsign["RFN1"].pos = (7, 10)
    env.roster.by_callsign["TL1"].pos = (20, 20)   # distance is irrelevant
    enemy = env.enemies[0]
    enemy.pos = (10, 10)
    enemy.home = enemy.pos
    obs, *_ = _step_all(env, {"RFN1": CONTACT_IDX})
    assert obs["TL1"]["observation"][KNOWN_COUNT_FIELD] > 0.0, "global net: everyone hears"
    assert env._agent_known == {}, "no per-agent bookkeeping in global mode"


def test_out_of_range_order_is_not_received():
    env = _range_env(comm_range=5.0)
    rfn1 = env.roster.by_callsign["RFN1"]
    rfn1.pos = (20, 20)  # far from TL1 at the spawn
    *_, infos = _step_all(env, {"TL1": ORDER_SEIZE_S0})
    kinds = [m.kind.value for m in env.transcript.messages]
    assert kinds.count("order") == 1, "the transmission itself is on the transcript"
    assert "ack" not in kinds, "no WILCO comes back from the void"
    assert rfn1.mission is None, "an unheard order assigns nothing"
    # no order-quality credit — only the standing coverage-gap penalty remains
    assert infos["TL1"]["components"]["command"] <= 0.0, "no command credit either"


def test_in_range_order_lands_with_wilco():
    env = _range_env(comm_range=5.0)  # spawn cells are adjacent: audible
    rfn1 = env.roster.by_callsign["RFN1"]
    _step_all(env, {"TL1": ORDER_SEIZE_S0})
    kinds = [m.kind.value for m in env.transcript.messages]
    assert rfn1.mission is not None and rfn1.mission.type is MissionType.SEIZE
    assert "ack" in kinds


def test_hq_traffic_always_heard():
    env = _range_env(comm_range=2.0)
    rfn2 = env.roster.by_callsign["RFN2"]
    rfn2.pos = (20, 20)
    env.inject_order("RFN2, hold position")  # HQ: high-power station
    assert rfn2.mission is not None and rfn2.mission.type is MissionType.HOLD
