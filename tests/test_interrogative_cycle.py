"""Interrogative-cycle env wiring (docs/interrogative-cycle.md §A/§B).

REQUEST_STATUS is an APPENDED broadcast action (index pinned): a leader with
>= 1 living direct subordinate asks its element for status instead of
staking a claim. Every living direct subordinate that HEARS it auto-answers
in slot order with its OWN mission state (is_complete on its own context —
never the root's success condition), subject to audibility both ways. The
request pays ordinary airtime at SITREP priority; a repeat inside
``request_status_cooldown`` is masked, not priced — no verdict, no penalty,
anywhere. Regression hazards pinned here: existing action indices never
moved, a deaf station stays silent, a dead station answers nothing, the
answer is the subordinate's truth and not the operation's.
"""

from dataclasses import replace

import numpy as np

from cohort.config import get_scenario
from cohort.core.language import parse_status_reply
from cohort.core.missions import MissionType
from cohort.core.orders import MessageKind
from cohort.env.actions import CATALOG, N_ACTIONS
from cohort.env.cohort_env import _TX_PRIORITY
from tests.test_garble import SITREP_IDX, _flat_env, _range_env, _step_all

STAY = 0
REQUEST_STATUS_IDX = next(s.index for s in CATALOG if s.kind == "request_status")


def _mask(env, cs):
    return env._mask_for(env.roster.by_callsign[cs])


def _new_kinds(env, before):
    return [m.kind for m in env.transcript.messages[before:]]


# ------------------------------------------------------------------ #
# the appended catalog entry
# ------------------------------------------------------------------ #


def test_request_status_is_appended_at_the_pinned_index():
    """+1 appended after every read-back-cycle entry. 239 was N_ACTIONS for
    the whole read-back era — nothing moved."""
    assert N_ACTIONS == 240
    assert (REQUEST_STATUS_IDX, CATALOG[239].name) == (239, "REQUEST_STATUS")
    assert (CATALOG[237].name, CATALOG[238].name) == ("SAY_AGAIN", "READBACK")


def test_the_request_rides_at_sitrep_priority():
    """§B: net arbitration at SITREP priority — same class, not a new one."""
    assert _TX_PRIORITY["request_status"] == _TX_PRIORITY["sitrep"]


# ------------------------------------------------------------------ #
# the mask (§B): >= 1 living direct subordinate, cooldown masked not priced
# ------------------------------------------------------------------ #


def test_mask_requires_a_living_direct_subordinate():
    env = _flat_env(get_scenario("fireteam"))
    assert _mask(env, "TL1")[REQUEST_STATUS_IDX] == 1, "TL1 leads an element"
    for cs in ("RFN1", "RFN2", "RFN3"):
        assert not _mask(env, cs)[REQUEST_STATUS_IDX], "a rifleman has no element to ask"
    for cs in ("RFN1", "RFN2", "RFN3"):
        env.roster.by_callsign[cs].alive = False
    assert not _mask(env, "TL1")[REQUEST_STATUS_IDX], "a dead element cannot be asked"


def test_it_generalizes_down_the_chain():
    """Any agent with a living direct subordinate may ask — SL and TL alike."""
    env = _flat_env(get_scenario("squad"))
    assert _mask(env, "SL1")[REQUEST_STATUS_IDX] == 1
    assert _mask(env, "TL1")[REQUEST_STATUS_IDX] == 1
    assert not _mask(env, "RFN1")[REQUEST_STATUS_IDX]


def test_a_repeat_inside_the_cooldown_is_masked_not_priced():
    env = _flat_env(get_scenario("fireteam"))
    cooldown = env.spec_cfg.request_status_cooldown
    assert cooldown == 8, "the done_cooldown figure, per the spec"
    _step_all(env, {"TL1": REQUEST_STATUS_IDX})  # spoken at step 1
    assert not _mask(env, "TL1")[REQUEST_STATUS_IDX], "the window just opened"
    for _ in range(cooldown - 1):  # steps 2..8: inside the window
        before = len(env.transcript.messages)
        _, _, _, _, infos = _step_all(env, {"TL1": REQUEST_STATUS_IDX})
        # the masked attempt is treated as STAY: nothing spoken, nothing paid
        assert MessageKind.REQUEST_STATUS not in _new_kinds(env, before)
        assert infos["TL1"]["components"].get("report", 0.0) == 0.0
    before = len(env.transcript.messages)
    _step_all(env, {"TL1": REQUEST_STATUS_IDX})  # step 9: the window elapsed
    assert MessageKind.REQUEST_STATUS in _new_kinds(env, before)


def test_masks_still_admit_nothing_for_the_dead():
    env = _flat_env(get_scenario("fireteam"))
    env.roster.by_callsign["TL1"].alive = False
    assert not _mask(env, "TL1")[REQUEST_STATUS_IDX]


# ------------------------------------------------------------------ #
# the exchange (§A): ask -> answers in slot order, the subordinate's truth
# ------------------------------------------------------------------ #


def test_full_chain_every_living_subordinate_answers_in_slot_order():
    env = _flat_env(get_scenario("fireteam"))
    env.inject_order("RFN1, seize obj alpha", issuer="TL1")
    before = len(env.transcript.messages)
    _step_all(env, {"TL1": REQUEST_STATUS_IDX})
    new = env.transcript.messages[before:]
    assert [m.kind for m in new] == [
        MessageKind.REQUEST_STATUS,
        MessageKind.STATUS_REPLY,
        MessageKind.STATUS_REPLY,
        MessageKind.STATUS_REPLY,
    ]
    ask = new[0]
    assert ask.text == "ALL STATIONS, THIS IS TL1: REPORT STATUS. OVER."
    assert ask.recipient_id is None, "a broadcast"
    # slot order: TL1's living subordinates, first to last
    ids = [env.roster.by_callsign[cs].id for cs in ("RFN1", "RFN2", "RFN3")]
    assert [m.sender_id for m in new[1:]] == ids
    assert all(m.recipient_id == env.roster.by_callsign["TL1"].id for m in new[1:])
    # RFN1 holds a live tasking; RFN2/RFN3 hold nothing
    assert new[1].text == "TL1, THIS IS RFN1: SEIZE OBJ ALPHA — IN PROGRESS. OVER."
    assert new[2].text == "TL1, THIS IS RFN2: AWAITING ORDERS. OVER."
    assert new[3].text == "TL1, THIS IS RFN3: AWAITING ORDERS. OVER."


def test_a_complete_answer_is_the_subordinates_own_truth():
    """RFN1 ordered CLEAR OBJ BRAVO with BRAVO's garrison dead: ITS mission
    is complete (the same is_complete a DONE adjudication would use on its
    own context) even though the OPERATION (root SEIZE ALPHA) is not — the
    reply carries what the element knows, never the success condition."""
    env = _flat_env(get_scenario("fireteam"))
    env.inject_order("RFN1, clear obj bravo", issuer="TL1")
    for e in env.enemies:
        e.alive = False
    root_obj = env.world.objective_by_name(env.spec_cfg.root_objective)
    assert not env._check_success(root_obj), "the operation itself is NOT complete"
    before = len(env.transcript.messages)
    _step_all(env, {"TL1": REQUEST_STATUS_IDX})
    reply = env.transcript.messages[before + 1]
    assert reply.kind is MessageKind.STATUS_REPLY
    assert reply.text == "TL1, THIS IS RFN1: CLEAR OBJ BRAVO — COMPLETE. OVER."
    parsed = parse_status_reply(reply.text)
    assert parsed["status"] == "COMPLETE"
    assert parsed["mission"] is MissionType.CLEAR
    # a reply is never adjudicated: no verdict followed, the mission stands
    assert env.transcript.messages[before + 1].kind is MessageKind.STATUS_REPLY
    assert env.roster.by_callsign["RFN1"].mission.type is MissionType.CLEAR


def test_an_incomplete_mission_answers_in_progress():
    env = _flat_env(get_scenario("fireteam"))
    env.inject_order("RFN1, clear obj bravo", issuer="TL1")  # garrison alive
    env.enemies[0].pos = env.world.objective_by_name("BRAVO").pos
    env.enemies[0].home = env.enemies[0].pos
    before = len(env.transcript.messages)
    _step_all(env, {"TL1": REQUEST_STATUS_IDX})
    reply = env.transcript.messages[before + 1]
    assert "CLEAR OBJ BRAVO — IN PROGRESS" in reply.text


def test_a_dead_station_answers_nothing():
    env = _flat_env(get_scenario("fireteam"))
    env.roster.by_callsign["RFN2"].alive = False
    before = len(env.transcript.messages)
    _step_all(env, {"TL1": REQUEST_STATUS_IDX})
    new = env.transcript.messages[before:]
    replies = [m for m in new if m.kind is MessageKind.STATUS_REPLY]
    senders = {m.sender_id for m in replies}
    assert env.roster.by_callsign["RFN2"].id not in senders
    assert len(replies) == 2, "only the living element answers"


def test_a_station_deaf_to_the_request_stays_silent():
    """Audibility both ways, the inbound half: under comm_model="range" a
    subordinate beyond earshot never heard the ask and answers nothing —
    asking into a dead zone is the outage made legible."""
    env = _range_env(comm_range=5.0)
    env.roster.by_callsign["TL1"].pos = (20, 10)
    env.roster.by_callsign["RFN2"].pos = (17, 10)  # d=3 <= 5: hears the ask
    env.roster.by_callsign["RFN1"].pos = (5, 10)   # d=15: silence
    env.roster.by_callsign["RFN3"].pos = (32, 10)  # d=12: silence
    before = len(env.transcript.messages)
    _step_all(env, {"TL1": REQUEST_STATUS_IDX})
    new = env.transcript.messages[before:]
    replies = [m for m in new if m.kind is MessageKind.STATUS_REPLY]
    rfn2 = env.roster.by_callsign["RFN2"].id
    assert [m.sender_id for m in replies] == [rfn2], (
        "exactly the one station in earshot answered"
    )


def test_the_ask_pays_airtime_and_contends_like_a_sitrep():
    """§B price: transmission_cost only, arbitrated on the single frequency —
    a learned transmission is never free air. The replies are protocol and
    ride regardless (WILCO / DONE_CONFIRM precedent)."""
    env = _flat_env(get_scenario("fireteam"))
    before = len(env.transcript.messages)
    _, _, _, _, infos = _step_all(
        env, {"TL1": REQUEST_STATUS_IDX, "RFN1": SITREP_IDX}
    )
    assert infos["RFN1"]["net_busy"], "one frequency: the later station loses"
    new = env.transcript.messages[before:]
    assert [m.kind for m in new if m.kind is MessageKind.SITREP] == []
    assert sum(1 for m in new if m.kind is MessageKind.REQUEST_STATUS) == 1
    assert sum(1 for m in new if m.kind is MessageKind.STATUS_REPLY) == 3, (
        "the auto-replies are never arbitrated away"
    )
    assert np.isclose(
        infos["TL1"]["components"].get("report", 0.0),
        env.rewards_cfg.transmission_cost,
    ), "the asker paid ordinary airtime and NOTHING else — no stake, no verdict"
    assert infos["RFN1"]["components"].get("report", 0.0) == 0.0


def test_the_replies_carry_no_reward_or_penalty_for_the_answerers():
    env = _flat_env(get_scenario("fireteam"))
    env.inject_order("RFN1, clear obj bravo", issuer="TL1")
    for e in env.enemies:
        e.alive = False
    _, _, _, _, infos = _step_all(env, {"TL1": REQUEST_STATUS_IDX})
    for cs in ("RFN1", "RFN2", "RFN3"):
        assert infos[cs]["components"].get("report", 0.0) == 0.0


def test_voice_only_asks_by_voice_within_earshot():
    """Degraded comms apply unchanged: under voice_only the ask and answers
    ride the low-voice audibility rule like every other utterance."""
    env = _flat_env(replace(get_scenario("fireteam"), comm_model="voice_only"))
    # everyone within voice range of TL1 except RFN3, moved out of earshot
    tl = env.roster.by_callsign["TL1"]
    env.roster.by_callsign["RFN1"].pos = (tl.pos[0] + 1, tl.pos[1])
    env.roster.by_callsign["RFN2"].pos = (tl.pos[0] - 1, tl.pos[1])
    env.roster.by_callsign["RFN3"].pos = (tl.pos[0] + 20, tl.pos[1])
    before = len(env.transcript.messages)
    _step_all(env, {"TL1": REQUEST_STATUS_IDX})
    replies = [
        m for m in env.transcript.messages[before:]
        if m.kind is MessageKind.STATUS_REPLY
    ]
    senders = [m.sender_id for m in replies]
    assert env.roster.by_callsign["RFN1"].id in senders
    assert env.roster.by_callsign["RFN2"].id in senders
    assert env.roster.by_callsign["RFN3"].id not in senders


# ------------------------------------------------------------------ #
# the observation block (§C): +4 appended, [status COMPLETE heard recently]
# ------------------------------------------------------------------ #


def test_obs_dim_grows_by_exactly_the_status_block():
    """OBS_DIM 357 -> 361: the +4 block closes the vector after DONE-heard.
    Every v1.27-tree checkpoint is orphaned by this — stated in the spec."""
    from cohort.env.observations import (
        _DONE_HEARD_BLOCK,
        _STATUS_HEARD_BLOCK,
        N_SUB_SLOTS,
        OBS_DIM,
        OFF_DONE_HEARD,
        OFF_STATUS_HEARD,
        obs_dim,
    )

    assert OBS_DIM == 361
    assert _STATUS_HEARD_BLOCK == N_SUB_SLOTS == 4
    assert OFF_STATUS_HEARD == OFF_DONE_HEARD + _DONE_HEARD_BLOCK
    assert OFF_STATUS_HEARD + _STATUS_HEARD_BLOCK == OBS_DIM
    # appended to BOTH profiles: the core/full delta stays the v1.10 bisect's
    assert obs_dim("full") - obs_dim("core") == 54


def _sub_complete_env():
    """RFN1 ordered CLEAR OBJ BRAVO with the garrison dead: a COMPLETE
    answer one request away (mirror of test_readback_cycle._sub_done_env)."""
    env = _flat_env(get_scenario("fireteam"))
    env.inject_order("RFN1, clear obj bravo", issuer="TL1")
    for e in env.enemies:
        e.alive = False
    return env


def test_a_landed_complete_reply_lights_the_askers_slot_for_a_window():
    from cohort.env.cohort_env import DONE_HEARD_WINDOW
    from cohort.env.observations import OBS_DIM, OFF_STATUS_HEARD

    env = _sub_complete_env()
    obs, *_ = _step_all(env, {"TL1": REQUEST_STATUS_IDX})
    assert obs["TL1"]["observation"][OFF_STATUS_HEARD] == 1.0, "RFN1 is slot 0"
    assert not obs["TL1"]["observation"][OFF_STATUS_HEARD + 1 : OBS_DIM].any()
    # nobody else carries it — the flag is the ASKER's knowledge
    for cs in ("RFN1", "RFN2", "RFN3"):
        assert not obs[cs]["observation"][OFF_STATUS_HEARD:].any()
    # age <= WINDOW holds, the DONE-heard clock this block mirrors
    for _ in range(DONE_HEARD_WINDOW):
        obs, *_ = _step_all(env, {})
    assert obs["TL1"]["observation"][OFF_STATUS_HEARD] == 1.0
    obs, *_ = _step_all(env, {})
    assert obs["TL1"]["observation"][OFF_STATUS_HEARD] == 0.0


def test_in_progress_and_awaiting_orders_set_nothing():
    """The flag is closing evidence, not a presence ping."""
    from cohort.env.observations import OFF_STATUS_HEARD

    env = _flat_env(get_scenario("fireteam"))
    env.inject_order("RFN1, seize obj alpha", issuer="TL1")  # IN PROGRESS
    obs, *_ = _step_all(env, {"TL1": REQUEST_STATUS_IDX})
    kinds = [m.kind for m in env.transcript.messages]
    assert kinds.count(MessageKind.STATUS_REPLY) == 3, "all three answered"
    assert env._status_complete_heard == {}
    assert not obs["TL1"]["observation"][OFF_STATUS_HEARD:].any()


def test_an_unlanded_complete_reply_sets_no_flag():
    """Audibility both ways, the return half: a station the asker cannot
    hear answers into the void — under "range" the transcript records the
    reply, but nothing LANDED on the asker and the slot stays dark."""
    from cohort.env.observations import OFF_STATUS_HEARD

    env = _range_env(comm_range=5.0)
    env.inject_order("RFN1, clear obj bravo", issuer="TL1")  # in earshot: lands
    for e in env.enemies:
        e.alive = False
    env.roster.by_callsign["TL1"].pos = (7, 10)
    env.roster.by_callsign["RFN1"].pos = (30, 10)  # far out of earshot
    obs, *_ = _step_all(env, {"TL1": REQUEST_STATUS_IDX})
    replies = [
        m for m in env.transcript.messages if m.kind is MessageKind.STATUS_REPLY
    ]
    assert env.roster.by_callsign["RFN1"].id not in {m.sender_id for m in replies}, (
        "out of earshot both ways: RFN1 never even heard the ask"
    )
    assert env._status_complete_heard == {}
    assert obs["TL1"]["observation"][OFF_STATUS_HEARD] == 0.0


def test_zero_filled_for_agents_with_no_element():
    from cohort.env.observations import OFF_STATUS_HEARD

    env = _sub_complete_env()
    obs, *_ = _step_all(env, {"TL1": REQUEST_STATUS_IDX})
    for cs in ("RFN1", "RFN2", "RFN3"):
        assert not obs[cs]["observation"][OFF_STATUS_HEARD:].any(), (
            f"{cs} has no element: the block stays zero"
        )


def test_the_done_heard_channel_stays_beside_the_new_one():
    """The v1.27 DONE-heard flags STAY — this cycle's bet is that a root
    taught to ask starts reading the passive channel too; removing them
    would destroy the measurement. A confirmed DONE still lights DONE-heard
    and never the status block."""
    from cohort.env.observations import OFF_DONE_HEARD, OFF_STATUS_HEARD

    env = _sub_complete_env()
    done_idx = next(s.index for s in CATALOG if s.kind == "done")
    obs, *_ = _step_all(env, {"RFN1": done_idx})
    assert obs["TL1"]["observation"][OFF_DONE_HEARD] == 1.0
    assert obs["TL1"]["observation"][OFF_STATUS_HEARD] == 0.0, (
        "a DONE is a claim, not a status reply — the two channels never blur"
    )
