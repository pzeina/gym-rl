# ruff: noqa: RUF059  — the shared fixture unpacks five handles; each test uses the ones it needs
"""Packet and liaison hazards (docs/degraded-communications.md §4, §8
"Packet and liaison hazards").

A MessagePacket is one immutable line of voice procedure plus routing
bookkeeping; an agent of liaison is a temporary carrying duty, not a MICAT
mission. Nothing is communicated, charged or rewarded until the line is
spoken at the recipient."""

from dataclasses import replace

import numpy as np
import pytest

from cohort import make_env
from cohort.config import get_scenario
from cohort.core import liaison as lia
from cohort.core.missions import MissionType
from cohort.core.orders import MessageKind
from cohort.env.actions import CATALOG
from cohort.env.observations import (
    LIAISON_ANCHOR,
    LIAISON_CAN_DELIVER,
    LIAISON_CARRYING,
    LIAISON_ENABLED,
    LIAISON_OUTBOX,
    LIAISON_RECEIPT,
    LIAISON_RETURNING,
)
from cohort.env.rewards import RewardConfig

STAY = 0
SITREP = next(s.index for s in CATALOG if s.kind == "sitrep")
CONTACT = next(s.index for s in CATALOG if s.kind == "contact")
DONE = next(s.index for s in CATALOG if s.kind == "done")
DELIVER = next(s.index for s in CATALOG if s.kind == "deliver")
CANCEL = next(s.index for s in CATALOG if s.kind == "cancel")
DISPATCH_S0 = next(s.index for s in CATALOG if s.name == "DISPATCH_LIAISON_S0")
DISPATCH_S1 = next(s.index for s in CATALOG if s.name == "DISPATCH_LIAISON_S1")
MOVE_EAST = next(s.index for s in CATALOG if s.name == "MOVE_EAST")
MOVE_WEST = next(s.index for s in CATALOG if s.name == "MOVE_WEST")
ORDER_S0_OBSERVE = next(
    s.index for s in CATALOG
    if s.kind == "order" and s.order_slot == 0 and s.order_mission is MissionType.OBSERVE
    and s.order_objective == "ALPHA"
)
ORDER_S0_CLEAR = next(
    s.index for s in CATALOG
    if s.kind == "order" and s.order_slot == 0 and s.order_mission is MissionType.CLEAR
    and s.order_objective == "ALPHA"
)


def _env(seed=1, **over):
    spec = replace(get_scenario("squad_voice_liaison"), name="squad_liaison_test", **over)
    env = make_env(spec)
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (40, 40)
        e.home = e.pos
    return env


def _place(env, **positions):
    for cs, pos in positions.items():
        env.roster.by_callsign[cs].pos = pos
    env._update_visual_links()
    env._refresh_friendly_perception()


def _step(env, overrides=None):
    acts = {a: STAY for a in env.agents}
    if overrides:
        acts.update(overrides)
    return env.step(acts)


def _mask(env, cs):
    return env._mask_for(env.roster.by_callsign[cs])


def _walk(env, cs, action, n):
    for _ in range(n):
        _step(env, {cs: action})


# ------------------------------------------------------------------ #
# preparation
# ------------------------------------------------------------------ #


def test_preparing_a_packet_emits_no_message_costs_nothing_and_has_no_remote_effect():
    env = _env()
    _place(env, SL1=(10, 10), TL1=(30, 10), RFN1=(11, 10))
    sl1, tl1 = env.roster.by_callsign["SL1"], env.roster.by_callsign["TL1"]
    assert _mask(env, "SL1")[ORDER_S0_OBSERVE] == 1, "out of range but a courier is possible"
    before = len(env.transcript)
    _, rewards, *_ = _step(env, {"SL1": ORDER_S0_OBSERVE})
    assert len(env.transcript) == before
    assert tl1.mission is None
    assert rewards["SL1"] == pytest.approx(env.rewards_cfg.time_penalty + env.rewards_cfg.coverage_gap)
    packet = env._outbox[sl1.id]
    assert packet.kind == "order" and packet.status == "held" and packet.holder_id == sl1.id
    assert "OBSERVE OBJ ALPHA" in packet.text
    # the outbox is occupied: further out-of-range acts are masked, CANCEL is not
    assert _mask(env, "SL1")[ORDER_S0_CLEAR] == 0
    assert _mask(env, "SL1")[CANCEL] == 1
    # a radio scenario never prepares anything
    radio = make_env("squad")
    radio.reset(seed=1)
    assert all(radio._mask_for(s)[DISPATCH_S0] == 0 and radio._mask_for(s)[DELIVER] == 0
               for s in radio.roster.living)
    with pytest.raises(ValueError, match="liaison_enabled requires"):
        replace(get_scenario("squad"), liaison_enabled=True)


def test_delivery_is_impossible_outside_voice_range_and_self_carry_delivers_at_receipt():
    env = _env()
    _place(env, SL1=(10, 10), TL1=(14, 10), RFN1=(11, 10), TL2=(9, 10))
    sl1, tl1 = env.roster.by_callsign["SL1"], env.roster.by_callsign["TL1"]
    _step(env, {"SL1": ORDER_S0_OBSERVE})
    packet = env._outbox[sl1.id]
    assert _mask(env, "SL1")[DELIVER] == 0, "TL1 at 4 cells: not in voice range"
    _walk(env, "SL1", MOVE_EAST, 2)  # now 2 cells away
    assert _mask(env, "SL1")[DELIVER] == 1
    assert tl1.mission is None, "tenure has not started"
    _, _rewards, _, _, infos = _step(env, {"SL1": DELIVER})
    assert tl1.mission is not None and tl1.mission.type is MissionType.OBSERVE
    assert tl1.mission.step_assigned == env._step_count, "tenure starts at delivery"
    assert tl1.last_order_step == env._step_count
    cfg = env.rewards_cfg
    assert infos["SL1"]["components"]["command"] == pytest.approx(
        cfg.order_allowed + cfg.order_objective_match + cfg.transmission_cost + cfg.coverage_gap
    ), "order credit at receipt (OBSERVE is an allowed derivation of SEIZE on the same objective)"
    order = next(m for m in env.last_messages if m.kind is MessageKind.ORDER)
    assert order.text == packet.text, "the canonical text is what is spoken"
    assert any(m.kind is MessageKind.ACK for m in env.last_messages)
    assert sl1.id not in env._outbox and packet.status == "delivered"


# ------------------------------------------------------------------ #
# dispatch and the courier
# ------------------------------------------------------------------ #


def _dispatch_setup():
    """SL1 writes an order for distant TL1 and hands it to TL2 (slot 1)."""
    env = _env()
    _place(env, SL1=(10, 10), TL1=(30, 10), TL2=(11, 10), RFN3=(11, 11), RFN4=(12, 11), RFN1=(31, 10))
    sl1, tl1, tl2 = (env.roster.by_callsign[c] for c in ("SL1", "TL1", "TL2"))
    # SL1 last saw TL1 heading to (30, 10): that is what a courier gets
    env._friendly_state["SL1"][tl1.id][0] = (30, 10)
    _step(env, {"SL1": ORDER_S0_OBSERVE})
    packet = env._outbox[sl1.id]
    assert packet.recipient_id == tl1.id
    assert _mask(env, "SL1")[DISPATCH_S0] == 0, "the recipient cannot carry its own order"
    assert _mask(env, "SL1")[DISPATCH_S1] == 1
    return env, sl1, tl1, tl2, packet


def test_dispatch_moves_the_packet_suspends_the_mission_and_detaches_the_courier():
    env, sl1, tl1, tl2, packet = _dispatch_setup()
    # TL2 already holds a task: it is suspended, not lost
    from cohort.core.missions import Mission

    tl2.mission = Mission(type=MissionType.HOLD, objective_id=None, anchor=tl2.pos,
                          issuer_id=sl1.id, step_assigned=1)
    _, rewards, *_ = _step(env, {"SL1": DISPATCH_S1})
    assert sl1.id not in env._outbox, "dispatch removes the issuer copy"
    task = env._liaison[tl2.id]
    assert task.packet is packet and packet.holder_id == tl2.id and packet.status == "dispatched"
    assert tl2.mission is None and task.suspended_mission.type is MissionType.HOLD
    assert task.anchor == (30, 10), "the fixed last-known anchor, not a live beacon"
    assert any(m.kind is MessageKind.DISPATCH for m in env.last_messages)
    assert rewards["SL1"] < env.rewards_cfg.time_penalty, "dispatch speech is charged"
    # detached: excluded from cohesion, masked to movement / fire / delivery
    assert tl2.id in env._detached_ids()
    mask = _mask(env, "TL2")
    legal = {CATALOG[i].kind for i in np.flatnonzero(mask)}
    assert legal <= {"stay", "move", "deliver", "cancel", "fire"}
    assert mask[DELIVER] == 0
    # the courier counts as tasked for the leader's coverage (TL1, whose
    # order is still in transit, is the untasked one — so task it directly)
    tl1.mission = Mission(type=MissionType.HOLD, objective_id=None, anchor=tl1.pos,
                          issuer_id=sl1.id, step_assigned=1)
    _, _r, _, _, infos = _step(env)
    assert infos["SL1"]["components"]["command"] == pytest.approx(env.rewards_cfg.coverage_bonus)


def test_courier_progress_is_watermarked_and_delivery_pays_only_on_acceptance():
    env, sl1, tl1, tl2, packet = _dispatch_setup()
    _step(env, {"SL1": DISPATCH_S1})
    cfg = env.rewards_cfg
    # walk east: each new best cell pays once
    _, r1, *_ = _step(env, {"TL2": MOVE_EAST})
    assert r1["TL2"] == pytest.approx(cfg.time_penalty + cfg.liaison_progress)
    # walk back and forth: the walked ground cannot be re-earned
    _step(env, {"TL2": MOVE_WEST})
    _, r2, *_ = _step(env, {"TL2": MOVE_EAST})
    assert r2["TL2"] == pytest.approx(cfg.time_penalty)
    # close to TL1 (at 30,10): TL2 at 12 → 28 is 16 moves
    _walk(env, "TL2", MOVE_EAST, 16)
    assert _mask(env, "TL2")[DELIVER] == 1
    _, r3, *_ = _step(env, {"TL2": DELIVER})
    assert tl1.mission is not None and tl1.mission.type is MissionType.OBSERVE
    assert r3["TL2"] >= cfg.liaison_delivery - 0.02, "accepted order: courier credit"
    assert r3["SL1"] > 0.0, "the ORIGIN gets the order credit"
    # the WILCO becomes a receipt carried back; the order already stands
    task = env._liaison[tl2.id]
    assert task.leg == "returning" and packet.receipt is True and packet.status == "returning"
    obs = env._observe(tl2, env._make_view(tl2))["observation"]
    assert obs[LIAISON_RETURNING] == 1.0 and obs[LIAISON_RECEIPT] == 1.0 and obs[LIAISON_CARRYING] == 1.0
    _walk(env, "TL2", MOVE_WEST, 16)
    assert _mask(env, "TL2")[DELIVER] == 1
    _, r4, *_ = _step(env, {"TL2": DELIVER})
    assert r4["TL2"] == pytest.approx(
        cfg.liaison_receipt_return + cfg.time_penalty + cfg.transmission_cost
    )
    assert tl2.id not in env._liaison, "the cycle completed"
    assert any(m.kind is MessageKind.RECEIPT for m in env.last_messages)


def test_courier_death_loses_the_packet_with_no_backup():
    env, sl1, tl1, tl2, packet = _dispatch_setup()
    _step(env, {"SL1": DISPATCH_S1})
    tl2.health = 0
    tl2.alive = False
    _step(env)
    assert packet.status == "lost" and packet.holder_id is None
    assert tl2.id not in env._liaison and sl1.id not in env._outbox
    assert tl1.mission is None
    assert not any(p is not packet for p in env.packets), "no invisible copy"


def test_delivery_reaches_the_succeeded_position_and_a_vacant_one_returns_negative():
    env, sl1, tl1, tl2, packet = _dispatch_setup()
    rfn1 = env.roster.by_callsign["RFN1"]  # TL1's man, beside it at (31,10)
    _step(env, {"SL1": DISPATCH_S1})
    # TL1 dies; RFN1 succeeds to TL1's position
    tl1.health = 1
    from cohort.core.units import Trap

    env.traps = [Trap(id=0, pos=(30, 11), damage=50)]
    tl1_move_south = next(s.index for s in CATALOG if s.name == "MOVE_SOUTH")
    _step(env, {"TL1": tl1_move_south})
    assert not tl1.alive and rfn1.effective_rank.name == "TL"
    target = lia.resolve_position(packet.recipient_id, env.roster, env._successions)
    assert target is rfn1, "the position's current holder, not a dead id"
    _walk(env, "TL2", MOVE_EAST, 18)
    assert _mask(env, "TL2")[DELIVER] == 1
    _step(env, {"TL2": DELIVER})
    assert rfn1.mission is not None and rfn1.mission.type is MissionType.OBSERVE
    # a vacant position: the whole team dead
    env2, sl1b, tl1b, tl2b, packet2 = _dispatch_setup()
    _step(env2, {"SL1": DISPATCH_S1})
    for cs in ("TL1", "RFN1", "RFN2"):
        s = env2.roster.by_callsign[cs]
        s.alive = False
        s.health = 0
    _step(env2)
    task = env2._liaison[tl2b.id]
    assert task.leg == "returning" and packet2.receipt is False and packet2.status == "undeliverable"


def test_packet_text_and_captured_coordinates_never_change_in_transit():
    env = _env()
    enemy = env.enemies[0]
    enemy.pos = (18, 10)
    enemy.home = enemy.pos
    _place(env, RFN1=(10, 10), TL1=(30, 30), SL1=(32, 30), RFN2=(1, 22))
    _step(env)  # RFN1 sights it
    enemy.pos = (5, 40)
    assert _mask(env, "RFN1")[CONTACT] == 1, "out of range, courier possible: prepares"
    _step(env, {"RFN1": CONTACT})
    packet = env._outbox[env.roster.by_callsign["RFN1"].id]
    assert packet.kind == "contact" and "GRID 1810" in packet.text
    text0, payload0 = packet.text, packet.payload
    enemy.pos = (6, 41)  # the enemy walks on; the packet does not follow it
    for _ in range(5):
        _step(env)
    assert (packet.text, packet.payload) == (text0, payload0)
    # RFN1 self-carries to TL1 and delivers: the superior learns the CAPTURED fix
    _place(env, RFN1=(29, 30), TL1=(30, 30), SL1=(30, 31))
    assert _mask(env, "RFN1")[DELIVER] == 1
    _, rewards, *_ = _step(env, {"RFN1": DELIVER})
    assert env._agent_known["TL1"][enemy.id][:2] == (18.0, 10.0)
    assert rewards["RFN1"] > 0.3, "contact credit to the origin on delivery"
    assert env._agent_known["SL1"][enemy.id][:2] == (18.0, 10.0), "overheard at delivery"


def test_cancel_is_churn_not_speech_and_restores_the_courier_mission():
    env, sl1, tl1, tl2, packet = _dispatch_setup()
    cfg = env.rewards_cfg
    before = len(env.transcript)
    _, rewards, *_ = _step(env, {"SL1": CANCEL})
    assert len(env.transcript) == before and packet.status == "cancelled"
    assert rewards["SL1"] == pytest.approx(cfg.time_penalty + cfg.coverage_gap + cfg.order_churn)
    assert env.order_pay_events_last_step[0]["outcome"] == "churn"
    # a courier cancelling its duty gets its mission back
    env2, sl1b, tl1b, tl2b, packet2 = _dispatch_setup()
    from cohort.core.missions import Mission

    tl2b.mission = Mission(type=MissionType.HOLD, objective_id=None, anchor=tl2b.pos,
                           issuer_id=sl1b.id, step_assigned=1)
    _step(env2, {"SL1": DISPATCH_S1})
    assert tl2b.mission is None
    _step(env2, {"TL2": CANCEL})
    assert tl2b.mission is not None and tl2b.mission.type is MissionType.HOLD
    assert tl2b.id not in env2._liaison


def test_an_obsolete_carried_order_is_rejected_aloud_with_a_negative_receipt():
    env, sl1, tl1, tl2, packet = _dispatch_setup()
    _step(env, {"SL1": DISPATCH_S1})
    # the chain of command changes: SL1 dies, TL1 becomes root — the order's
    # origin position is now held by TL1 itself, which cannot order itself
    sl1.alive = False
    sl1.health = 0
    _step(env)
    _walk(env, "TL2", MOVE_EAST, 17)
    if _mask(env, "TL2")[DELIVER]:
        _step(env, {"TL2": DELIVER})
        assert tl1.mission is None or tl1.mission.type is not MissionType.OBSERVE
        assert any("NEGATIVE" in m.text for m in env.last_messages)
    else:
        # the origin position resolved onto the recipient: vacant from the
        # carrier's point of view, the duty turned around instead
        task = env._liaison.get(tl2.id)
        assert task is None or task.leg == "returning" or packet.status in ("lost", "expired", "undeliverable")


def test_stale_done_claim_is_adjudicated_only_at_delivery():
    env = _env()
    from cohort.core.missions import Mission

    sl1, tl1 = env.roster.by_callsign["SL1"], env.roster.by_callsign["TL1"]
    _place(env, SL1=(10, 10), TL1=(25, 10), RFN1=(26, 10), RFN2=(25, 11))
    obj = env.world.objective_by_name("ALPHA")
    tl1.mission = Mission(type=MissionType.SEIZE, objective_id=obj.id, anchor=obj.pos,
                          issuer_id=sl1.id, step_assigned=1)
    assert _mask(env, "TL1")[DONE] == 1
    _step(env, {"TL1": DONE})
    packet = env._outbox[tl1.id]
    assert packet.kind == "done" and tl1.mission is not None, "not adjudicated yet"
    _place(env, SL1=(10, 10), TL1=(12, 10), RFN1=(26, 10), RFN2=(25, 11))
    _, rewards, *_ = _step(env, {"TL1": DELIVER})
    assert any(m.kind is MessageKind.DONE_REJECT for m in env.last_messages), "false claim rejected"
    assert rewards["TL1"] < 0


def test_suspended_compliance_does_not_pay_during_duty_and_expiry_ends_it():
    env, sl1, tl1, tl2, packet = _dispatch_setup()
    from cohort.core.missions import Mission

    tl2.mission = Mission(type=MissionType.HOLD, objective_id=None, anchor=tl2.pos,
                          issuer_id=sl1.id, step_assigned=1)
    _step(env, {"SL1": DISPATCH_S1})
    _, rewards, *_ = _step(env)
    assert rewards["TL2"] == pytest.approx(env.rewards_cfg.time_penalty), "no compliance while detached"
    for _ in range(lia.PACKET_TTL + 1):
        _step(env)
    assert packet.status == "expired" and tl2.id not in env._liaison
    assert tl2.mission is not None and tl2.mission.type is MissionType.HOLD


def test_liaison_block_carries_packet_and_anchor_state_and_no_live_target():
    env, sl1, tl1, tl2, packet = _dispatch_setup()
    obs = env._observe(sl1, env._make_view(sl1))["observation"]
    assert obs[LIAISON_ENABLED] == 1.0 and obs[LIAISON_OUTBOX] == 1.0
    _step(env, {"SL1": DISPATCH_S1})
    w = float(env.world.width)
    tl1.pos = (38, 10)  # the recipient moves away unseen: the anchor does not follow
    env._update_visual_links()
    obs = env._observe(tl2, env._make_view(tl2))["observation"]
    assert obs[LIAISON_CARRYING] == 1.0
    assert abs(obs[LIAISON_ANCHOR] * w - (30 - tl2.pos[0])) < 1e-4, "fixed anchor at (30,10)"
    assert obs[LIAISON_CAN_DELIVER] == 0.0
    radio = make_env("squad")
    radio.reset(seed=1)
    s = radio.roster.living[0]
    assert radio._observe(s, radio._make_view(s))["observation"][LIAISON_ENABLED] == 0.0


def test_liaison_cycle_income_stays_far_below_the_terminal():
    """Terminal dominance (§6): a courier cycle — progress across the whole
    map plus delivery plus receipt — is one-shot per packet and bounded by
    the diagonal, so the sum of the three terms over the longest conceivable
    chain is a small fraction of success_team."""
    cfg = RewardConfig()
    spec = get_scenario("squad_voice_liaison")
    diagonal = float(np.hypot(*spec.map_size))
    one_cycle = cfg.liaison_progress * 2 * diagonal + cfg.liaison_delivery + cfg.liaison_receipt_return
    assert one_cycle < 0.1 * cfg.success_team
    # and the per-step stall bound is untouched by any of the new terms
    assert cfg.max_step_farm() == RewardConfig(
        liaison_progress=0.0, liaison_delivery=0.0, liaison_receipt_return=0.0,
        visual_link_broken=0.0, acoustic_contact_new=0.0,
    ).max_step_farm()


def test_presets_and_provenance():
    direct = get_scenario("squad_voice_direct")
    liaison = get_scenario("squad_voice_liaison")
    from dataclasses import fields

    for f in fields(direct):
        if f.name in ("name", "description", "liaison_enabled", "experiment_arm"):
            continue
        assert getattr(direct, f.name) == getattr(liaison, f.name), f.name
    squad = get_scenario("squad")
    ga = get_scenario("squad_global_acoustic_control")
    rc = get_scenario("squad_range_control")
    assert (ga.comm_model, ga.sound_model) == ("global", "tactical")
    assert (rc.comm_model, rc.sound_model, rc.comm_range) == ("range", "tactical", 12.0)
    for f in fields(squad):
        if f.name in ("name", "description", "sound_model", "comm_model", "experiment_arm"):
            continue
        # `reward_overrides` is the ONE disclosed asymmetry (owner-decided
        # 2026-08-24): squad_range_control prices idle time at -0.03 because the
        # D4 attractor captures it at the default price, and its two controls do
        # not. The three arms are therefore NO LONGER matched on economics, and
        # any comm-regime comparison across them must say so — see
        # docs/degraded-communications.md. Everything else still must match.
        if f.name == "reward_overrides":
            continue
        assert getattr(squad, f.name) == getattr(ga, f.name) == getattr(rc, f.name), f.name
    assert squad.reward_overrides == ga.reward_overrides == ()
    assert rc.reward_overrides == (("time_penalty", -0.03),)
    b = make_env("squad_voice_liaison").briefing()
    assert b["liaison_enabled"] is True and b["packet_ttl"] == lia.PACKET_TTL
    snap = make_env("squad_voice_liaison")
    snap.reset(seed=1)
    o = snap.oracle()
    assert "packets" in o and "liaison" in o
