"""Read-back cycle env wiring (docs/readback-cycle.md §B/§C/§D).

SAY_AGAIN and READBACK are APPENDED actions (indices pinned); SAY_AGAIN is
legal on a fresh garble ping or a fresh unintelligible voice cue and flags
the garbled senders that hear the request; READBACK reads the held order
back and is auto-answered CORRECT/WRONG by the superior, WRONG restating
the order actually in force; a leader that receives a subordinate's DONE
and answers DONE_CONFIRM carries closing evidence for a window. Regression
hazards pinned here: the restated order is the issuer's book, existing
action indices never moved, rejected DONEs carry nothing.
"""

from dataclasses import replace

import numpy as np

from cohort.config import get_scenario
from cohort.core import acoustics as snd
from cohort.core.missions import MissionType
from cohort.core.orders import HQ_ID, MessageKind
from cohort.env.actions import CATALOG, N_ACTIONS
from cohort.env.cohort_env import (
    DONE_HEARD_WINDOW,
    READBACK_HEARD_WINDOW,
    SAY_AGAIN_TTL,
)
from cohort.env.observations import (
    OFF_DONE_HEARD,
    OFF_READBACK_HEARD,
    OFF_SAY_AGAIN,
)
from tests.test_garble import SITREP_IDX, _flat_env, _place, _range_env, _step_all

STAY = 0
SAY_AGAIN_IDX = next(s.index for s in CATALOG if s.kind == "say_again")
READBACK_IDX = next(s.index for s in CATALOG if s.kind == "readback")
DONE_IDX = next(s.index for s in CATALOG if s.kind == "done")


def _mask(env, cs):
    return env._mask_for(env.roster.by_callsign[cs])


# ------------------------------------------------------------------ #
# the appended catalog entries
# ------------------------------------------------------------------ #


def test_new_actions_are_appended_at_pinned_indices():
    """+2 appended: SAY_AGAIN then READBACK, after every pre-existing entry.
    237 was N_ACTIONS for the whole degraded-comms era — nothing moved."""
    assert N_ACTIONS == 239
    assert (SAY_AGAIN_IDX, CATALOG[237].name) == (237, "SAY_AGAIN")
    assert (READBACK_IDX, CATALOG[238].name) == (238, "READBACK")


# ------------------------------------------------------------------ #
# SAY AGAIN (§B)
# ------------------------------------------------------------------ #


def test_say_again_masked_without_either_spelling():
    env = _flat_env(get_scenario("fireteam"))
    for cs in env.agents:
        assert not _mask(env, cs)[SAY_AGAIN_IDX]


def test_say_again_opens_on_a_fresh_garble_ping_and_closes_with_it():
    env = _range_env(comm_range=5.0)
    _place(env)
    _step_all(env, {"RFN1": SITREP_IDX})
    assert _mask(env, "RFN2")[SAY_AGAIN_IDX] == 1
    for cs in ("RFN1", "RFN3", "TL1"):
        assert not _mask(env, cs)[SAY_AGAIN_IDX]
    env._garble["RFN2"].clear()
    assert not _mask(env, "RFN2")[SAY_AGAIN_IDX]


def test_say_again_opens_on_a_fresh_unintelligible_voice_cue():
    """The other spelling: a held `voice` cue attributed to nobody. A voice
    heard as friendly was understood or seen; a hostile one is an enemy, not
    a station to answer — only `unknown` opens the request."""
    env = _flat_env(get_scenario("fireteam"))
    cues = env._agent_cues.setdefault("RFN2", [])

    def cue(kind, side, step):
        return snd.AcousticCue(
            kind=kind, side=side, bearing=2, distance_band=1,
            strength=0.5, event_step=step, event_id=0,
        )

    cues.append(cue("voice", "friendly", 0))
    assert not _mask(env, "RFN2")[SAY_AGAIN_IDX]
    cues.append(cue("movement", "unknown", 0))
    assert not _mask(env, "RFN2")[SAY_AGAIN_IDX]
    cues.append(cue("voice", "unknown", -snd.SOUND_MEMORY_TTL - 1))
    assert not _mask(env, "RFN2")[SAY_AGAIN_IDX], "an expired cue opens nothing"
    cues.append(cue("voice", "unknown", 0))
    assert _mask(env, "RFN2")[SAY_AGAIN_IDX] == 1


def test_say_again_flags_the_garbled_sender_that_hears_the_request():
    """ping -> SAY AGAIN -> sender sees it: the learnable loop. The env never
    echoes a re-transmission — the flag is all the sender gets."""
    env = _range_env(comm_range=5.0)
    _place(env)
    _step_all(env, {"RFN1": SITREP_IDX})  # RFN2 garbled at d=6
    # the requester closes distance so the sender can actually hear it
    env.roster.by_callsign["RFN2"].pos = (10, 10)  # d=3 from RFN1
    obs, *_ = _step_all(env, {"RFN2": SAY_AGAIN_IDX})
    last = env.transcript.messages[-1]
    assert last.kind is MessageKind.SAY_AGAIN
    assert last.text == "STATION CALLING, THIS IS RFN2: SAY AGAIN. OVER."
    assert env._say_again_pending == {"RFN1": 2}
    assert obs["RFN1"]["observation"][OFF_SAY_AGAIN] == 1.0
    assert obs["RFN2"]["observation"][OFF_SAY_AGAIN] == 0.0
    # TTL discipline: the flag holds SAY_AGAIN_TTL steps, then clears
    for _ in range(SAY_AGAIN_TTL):
        obs, *_ = _step_all(env, {})
    assert obs["RFN1"]["observation"][OFF_SAY_AGAIN] == 1.0
    obs, *_ = _step_all(env, {})
    assert obs["RFN1"]["observation"][OFF_SAY_AGAIN] == 0.0


def test_say_again_flags_nobody_out_of_earshot():
    """A sender that cannot hear the request learns nothing — with symmetric
    ranges the garbled station usually cannot hear the reply either, which
    is exactly the physics: the requester's own SAY AGAIN garbles back."""
    env = _range_env(comm_range=5.0)
    _place(env)
    _step_all(env, {"RFN1": SITREP_IDX})
    obs, *_ = _step_all(env, {"RFN2": SAY_AGAIN_IDX})  # still at d=6
    assert env._say_again_pending == {}
    assert obs["RFN1"]["observation"][OFF_SAY_AGAIN] == 0.0
    # ...and the request itself pinged RFN1's garble state (annulus, d=6)
    assert env.perception("RFN1")["garble"], "the reply garbles back"


# ------------------------------------------------------------------ #
# READBACK (§C)
# ------------------------------------------------------------------ #


def test_readback_mask_needs_a_live_mission_and_a_superior():
    env = _flat_env(get_scenario("fireteam"))
    assert _mask(env, "TL1")[READBACK_IDX] == 1, "the root reads back to HQ"
    assert not _mask(env, "RFN1")[READBACK_IDX], "no mission held yet"
    env.inject_order("RFN1, seize obj alpha", issuer="TL1")
    assert _mask(env, "RFN1")[READBACK_IDX] == 1
    # voice_only: a root has no HQ channel after the briefing — masked
    voice = _flat_env(replace(get_scenario("fireteam"), comm_model="voice_only"))
    assert not _mask(voice, "TL1")[READBACK_IDX]


def test_readback_correct_flow_and_the_leader_side_window():
    env = _flat_env(get_scenario("fireteam"))
    env.inject_order("RFN1, seize obj alpha", issuer="TL1")
    obs, *_ = _step_all(env, {"RFN1": READBACK_IDX})
    rb, verdict = env.transcript.messages[-2:]
    assert rb.kind is MessageKind.READBACK
    assert rb.text == "TL1, THIS IS RFN1: I READ BACK — SEIZE OBJ ALPHA. OVER."
    assert verdict.kind is MessageKind.READBACK_CORRECT
    assert verdict.text == "RFN1, THIS IS TL1: CORRECT. OUT."
    # leader-side flag: RFN1 is TL1's slot 0
    assert obs["TL1"]["observation"][OFF_READBACK_HEARD] == 1.0
    assert not obs["TL1"]["observation"][OFF_READBACK_HEARD + 1:OFF_DONE_HEARD].any()
    # the mission is untouched — a read-back is verification, not a report
    assert env.roster.by_callsign["RFN1"].mission.type is MissionType.SEIZE
    # window mirrors the recent-contact-report flag: age <= WINDOW holds
    for _ in range(READBACK_HEARD_WINDOW):
        obs, *_ = _step_all(env, {})
    assert obs["TL1"]["observation"][OFF_READBACK_HEARD] == 1.0
    obs, *_ = _step_all(env, {})
    assert obs["TL1"]["observation"][OFF_READBACK_HEARD] == 0.0


def test_root_readback_is_answered_by_hq_and_flags_nobody():
    env = _flat_env(get_scenario("fireteam"))
    _step_all(env, {"TL1": READBACK_IDX})
    rb, verdict = env.transcript.messages[-2:]
    assert rb.text == "HQ, THIS IS TL1: I READ BACK — SEIZE OBJ ALPHA. OVER."
    assert verdict.kind is MessageKind.READBACK_CORRECT
    assert verdict.sender_id == HQ_ID
    assert env._readback_correct_heard == {}, "HQ is not an agent"


def test_readback_wrong_restates_the_order_actually_in_force():
    """THE regression pin: the WRONG branch restates the order the superior
    actually issued (its book), never an echo of the claimant's error. Here
    HQ re-tasks RFN1 over TL1's head, so RFN1's holding diverges from what
    TL1 last issued — TL1 answers NEGATIVE and repeats its own order."""
    env = _flat_env(get_scenario("fireteam"))
    env.inject_order("RFN1, seize obj alpha", issuer="TL1")
    env.inject_order("RFN1, observe obj bravo")  # HQ, over TL1's head
    assert env.roster.by_callsign["RFN1"].mission.type is MissionType.OBSERVE
    obs, *_ = _step_all(env, {"RFN1": READBACK_IDX})
    rb, verdict = env.transcript.messages[-2:]
    assert rb.text == "TL1, THIS IS RFN1: I READ BACK — OBSERVE OBJ BRAVO. OVER."
    assert verdict.kind is MessageKind.READBACK_WRONG
    assert verdict.text == "RFN1, THIS IS TL1: NEGATIVE, I SAY AGAIN — SEIZE OBJ ALPHA. OUT."
    # no CORRECT flag, and the correction is words on the net, not a re-issue
    assert obs["TL1"]["observation"][OFF_READBACK_HEARD] == 0.0
    assert env.roster.by_callsign["RFN1"].mission.type is MissionType.OBSERVE


def test_readback_surfaces_an_unlanded_range_order():
    """The cycle's premise end-to-end under comm_model="range": an order that
    never landed leaves the sub holding the OLD mission; its read-back is
    answered WRONG with the order in force on the issuer's book — the
    mismatch is now ON THE NET instead of silent."""
    env = _range_env(comm_range=5.0)
    env.inject_order("RFN1, seize obj alpha", issuer="TL1")  # in earshot: lands
    # sit out the order cooldown, then re-task RFN1 once it is out of earshot
    env.roster.by_callsign["TL1"].pos = (7, 10)
    env.roster.by_callsign["RFN1"].pos = (20, 10)
    for _ in range(9):
        _step_all(env, {})
        env.roster.by_callsign["TL1"].pos = (7, 10)
        env.roster.by_callsign["RFN1"].pos = (20, 10)
    order = next(
        s.index for s in CATALOG
        if s.kind == "order" and s.order_slot == 0
        and s.order_mission is MissionType.OBSERVE and s.order_objective == "BRAVO"
    )
    _step_all(env, {"TL1": order})
    assert env.roster.by_callsign["RFN1"].mission.type is MissionType.SEIZE, "unheard"
    # back in range, RFN1 reads back what it holds
    env.roster.by_callsign["RFN1"].pos = (9, 10)
    _step_all(env, {"RFN1": READBACK_IDX})
    verdict = env.transcript.messages[-1]
    assert verdict.kind is MessageKind.READBACK_WRONG
    assert "NEGATIVE, I SAY AGAIN — OBSERVE OBJ BRAVO" in verdict.text


# ------------------------------------------------------------------ #
# DONE heard (§D — option (a), the closing-evidence channel)
# ------------------------------------------------------------------ #


def _sub_done_env():
    """RFN1 ordered CLEAR OBJ BRAVO with BRAVO's garrison dead: a truthful,
    confirmable subordinate DONE one action away."""
    env = _flat_env(get_scenario("fireteam"))
    env.inject_order("RFN1, clear obj bravo", issuer="TL1")
    for e in env.enemies:
        e.alive = False
    return env


def test_confirmed_done_sets_the_leaders_heard_flag_for_a_window():
    env = _sub_done_env()
    obs, *_ = _step_all(env, {"RFN1": DONE_IDX})
    assert env.transcript.messages[-1].kind is MessageKind.DONE_CONFIRM
    assert obs["TL1"]["observation"][OFF_DONE_HEARD] == 1.0, "RFN1 is slot 0"
    assert not obs["TL1"]["observation"][OFF_DONE_HEARD + 1:].any()
    # nobody else carries it — the flag is the RECEIVING leader's knowledge
    for cs in ("RFN1", "RFN2", "RFN3"):
        assert not obs[cs]["observation"][OFF_DONE_HEARD:].any()
    # age <= WINDOW holds, like the recent-contact-report flag it mirrors
    for _ in range(DONE_HEARD_WINDOW):
        obs, *_ = _step_all(env, {})
    assert obs["TL1"]["observation"][OFF_DONE_HEARD] == 1.0
    obs, *_ = _step_all(env, {})
    assert obs["TL1"]["observation"][OFF_DONE_HEARD] == 0.0


def test_rejected_done_carries_no_closing_evidence():
    env = _flat_env(get_scenario("fireteam"))
    env.inject_order("RFN1, clear obj bravo", issuer="TL1")
    env.enemies[0].pos = env.world.objective_by_name("BRAVO").pos  # defender alive: false claim
    env.enemies[0].home = env.enemies[0].pos
    obs, *_ = _step_all(env, {"RFN1": DONE_IDX})
    assert env.transcript.messages[-1].kind is MessageKind.DONE_REJECT
    assert env._done_heard == {}
    assert obs["TL1"]["observation"][OFF_DONE_HEARD] == 0.0


def test_roots_own_confirmed_claim_sets_no_flag():
    """HQ answers the root's OPORD claim; HQ is not an agent and carries no
    observation — the channel is leader-received traffic only."""
    env = _flat_env(get_scenario("fireteam"))
    for e in env.enemies:
        e.alive = False
    env.roster.by_callsign["TL1"].pos = env.world.objectives[0].pos
    _step_all(env, {})  # let success latch with the root in position
    _step_all(env, {"TL1": DONE_IDX})
    kinds = [m.kind for m in env.transcript.messages]
    assert MessageKind.DONE_CONFIRM in kinds
    assert env._done_heard == {}


def test_out_of_earshot_done_confirm_leaves_the_leader_blind():
    """Under "range" the umpire still adjudicates an out-of-earshot claim —
    but a leader that did not RECEIVE the DONE learned nothing, and its
    flag stays down (leader-received traffic only)."""
    env = _range_env(comm_range=5.0)
    env.inject_order("RFN1, clear obj bravo", issuer="TL1")
    for e in env.enemies:
        e.alive = False
    env.roster.by_callsign["TL1"].pos = (7, 10)
    env.roster.by_callsign["RFN1"].pos = (30, 10)  # far out of earshot
    obs, *_ = _step_all(env, {"RFN1": DONE_IDX})
    assert env.transcript.messages[-1].kind is MessageKind.DONE_CONFIRM
    assert env._done_heard == {}
    assert obs["TL1"]["observation"][OFF_DONE_HEARD] == 0.0


# ------------------------------------------------------------------ #
# regression hazards
# ------------------------------------------------------------------ #


def test_verification_traffic_is_arbitrated_and_charged_like_speech():
    """A learned transmission is never free air (issue #18): SAY_AGAIN and
    READBACK contend on the single-frequency net and pay transmission_cost.
    A CONTACT outranks them; the loser's tick is dropped with NET BUSY."""
    env = _flat_env(get_scenario("fireteam"))
    env.inject_order("RFN1, seize obj alpha", issuer="TL1")
    before = len(env.transcript.messages)
    _, _, _, _, infos = _step_all(env, {"TL1": READBACK_IDX, "RFN1": READBACK_IDX})
    assert infos["RFN1"]["net_busy"], "one frequency: the later station loses"
    spoken = [m for m in env.transcript.messages[before:] if m.kind is MessageKind.READBACK]
    assert len(spoken) == 1 and spoken[0].sender_id == env.roster.by_callsign["TL1"].id
    assert np.isclose(
        infos["TL1"]["components"].get("report", 0.0), env.rewards_cfg.transmission_cost
    ), "the speaker paid airtime"
    assert infos["RFN1"]["components"].get("report", 0.0) == 0.0, (
        "the blocked station paid nothing")


def test_masks_still_admit_nothing_for_the_dead_or_carrying():
    env = _flat_env(get_scenario("fireteam"))
    env.inject_order("RFN1, seize obj alpha", issuer="TL1")
    dead = env.roster.by_callsign["RFN1"]
    dead.alive = False
    mask = _mask(env, "RFN1")
    assert not mask[SAY_AGAIN_IDX] and not mask[READBACK_IDX]
