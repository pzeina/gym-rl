"""Voice-only degraded communications — mechanics and information boundaries
(docs/degraded-communications.md §3, §8 "Voice-only mechanics").

No radio after the briefing: every utterance is low voice within
``voice_range`` with LOS, there is no HQ station, no global net and no
arbitration; orders need proximity, reports need a superior in earshot, and
pictures are per listener. Gestures and pre-arranged signals are the two
silent/loud alternatives to speech."""

from dataclasses import replace

import numpy as np
import pytest

from cohort import make_env
from cohort.config import get_scenario
from cohort.core import acoustics as snd
from cohort.core.missions import MissionType
from cohort.core.orders import MessageKind
from cohort.core.world import WALL
from cohort.env.actions import CATALOG
from cohort.env.observations import (
    COHESION_LEADER_AGE,
    COHESION_LEADER_SEEN,
    OFF_LEADER,
    OFF_SUBS,
)

STAY = 0
SITREP = next(s.index for s in CATALOG if s.kind == "sitrep")
CONTACT = next(s.index for s in CATALOG if s.kind == "contact")
DONE = next(s.index for s in CATALOG if s.kind == "done")
EXECUTE = next(s.index for s in CATALOG if s.kind == "execute")
GESTURE_EXECUTE = next(s.index for s in CATALOG if s.kind == "gesture_execute")
GESTURE_GO = next(s.index for s in CATALOG if s.kind == "gesture_sync_go")
ACOUSTIC = next(s.index for s in CATALOG if s.kind == "acoustic_contact")
SYNC_PROPOSE = next(s.index for s in CATALOG if s.kind == "sync_propose")
SYNC_GO = next(s.index for s in CATALOG if s.kind == "sync_go")
MOVE_EAST = next(s.index for s in CATALOG if s.name == "MOVE_EAST")


def _order(slot: int, mission: str, amc: bool = False) -> int:
    suffix = "_AMC" if amc else ""
    return next(
        s.index for s in CATALOG
        if s.kind == "order" and s.order_slot == slot and s.order_mission is not None
        and s.order_mission.name == mission and bool(s.order_amc) == amc
        and (s.name.endswith(suffix) or not amc)
    )


ORDER_S0_OBSERVE = next(
    s.index for s in CATALOG
    if s.kind == "order" and s.order_slot == 0 and s.order_mission is MissionType.OBSERVE
    and s.order_objective == "ALPHA"
)


def _voice_env(seed=1, sound="tactical", **over):
    spec = replace(
        get_scenario("squad"), name="squad_voice_test", comm_model="voice_only",
        sound_model=sound, voice_range=2.0, reward_overrides=(("root_done_bonus", 0.0),),
        **over,
    )
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


def _blue_sounds(env):
    return [e for e in env.last_sound_events if e.side == "friendly"]


def _step(env, overrides=None):
    acts = {a: STAY for a in env.agents}
    if overrides:
        acts.update(overrides)
    return env.step(acts)


def _mask(env, cs):
    return env._mask_for(env.roster.by_callsign[cs])


# ------------------------------------------------------------------ #
# audibility
# ------------------------------------------------------------------ #


def test_comm_model_is_validated():
    with pytest.raises(ValueError, match="voice_only"):
        replace(get_scenario("squad"), comm_model="shouting")


def test_nobody_beyond_voice_range_receives_any_utterance():
    env = _voice_env()
    _place(env, SL1=(10, 10), TL1=(11, 10), RFN1=(10, 11), TL2=(20, 10), RFN3=(12, 12))
    _step(env, {"TL1": SITREP})
    meta = env.last_message_meta[-1]
    assert meta["medium"] == "voice"
    assert set(meta["heard_by"]) == {"SL1", "RFN1"}, meta  # RFN3 at 2.24 cells is out


def test_a_wall_prevents_understanding_even_at_one_cell():
    env = _voice_env()
    _place(env, SL1=(10, 10), TL1=(12, 10))
    env.world.grid[10, 11] = WALL
    assert not env._audible_to(env.roster.by_callsign["SL1"], env.roster.by_callsign["TL1"].id)


def test_no_hq_exception_after_reset():
    env = _voice_env()
    from cohort.core.orders import HQ_ID

    for s in env.roster.living:
        assert not env._audible_to(s, HQ_ID)
    with pytest.raises(PermissionError, match="no remote HQ"):
        env.inject_order("TL1, HOLD", issuer="HQ")
    with pytest.raises(PermissionError, match="no remote HQ"):
        env.inject_execute("HQ")
    # the briefing is delivered to its addressee only and labelled as such
    opord = env.transcript.messages[0]
    assert opord.kind is MessageKind.OPORD
    assert env.briefing()["hq_available"] is False


def test_root_to_hq_reports_are_masked_not_emitted_into_the_void():
    env = _voice_env()
    m = _mask(env, "SL1")
    assert m[SITREP] == 0 and m[CONTACT] == 0 and m[DONE] == 0
    # and a subordinate out of its leader's earshot is equally silent
    _place(env, SL1=(10, 10), TL1=(20, 20))
    assert _mask(env, "TL1")[SITREP] == 0
    _place(env, SL1=(10, 10), TL1=(11, 10))
    assert _mask(env, "TL1")[SITREP] == 1


def test_two_distant_speakers_are_both_heard_locally_in_one_tick():
    """No global net: no NET BUSY, both utterances land on their audiences."""
    env = _voice_env()
    _place(env, SL1=(10, 10), TL1=(11, 10), TL2=(30, 30), RFN4=(31, 30), RFN3=(12, 10))
    _, _, _, _, infos = _step(env, {"TL1": SITREP, "RFN4": SITREP})
    assert not any(i["net_busy"] for i in infos.values())
    sitreps = [(m, meta) for m, meta in zip(env.last_messages, env.last_message_meta, strict=False)
               if m.kind is MessageKind.SITREP]
    assert len(sitreps) == 2
    heard = {env.roster.by_id[m.sender_id].callsign: set(meta["heard_by"]) for m, meta in sitreps}
    assert heard["TL1"] == {"SL1", "RFN3"}
    assert heard["RFN4"] == {"TL2"}


# ------------------------------------------------------------------ #
# orders
# ------------------------------------------------------------------ #


def test_out_of_range_order_is_masked_changes_no_mission_and_produces_no_wilco():
    env = _voice_env()
    _place(env, SL1=(10, 10), TL1=(25, 25))
    assert _mask(env, "SL1")[ORDER_S0_OBSERVE] == 0
    before = len(env.transcript)
    _step(env, {"SL1": ORDER_S0_OBSERVE})  # illegal → STAY
    assert env.roster.by_callsign["TL1"].mission is None
    assert not any(m.kind in (MessageKind.ORDER, MessageKind.ACK) for m in env.transcript.messages[before:])


def test_in_range_order_lands_with_a_local_wilco_and_is_overheard_without_authority():
    env = _voice_env()
    _place(env, SL1=(10, 10), TL1=(11, 10), TL2=(10, 11), RFN1=(30, 30))
    assert _mask(env, "SL1")[ORDER_S0_OBSERVE] == 1
    _step(env, {"SL1": ORDER_S0_OBSERVE})
    tl1, tl2 = env.roster.by_callsign["TL1"], env.roster.by_callsign["TL2"]
    assert tl1.mission is not None and tl1.mission.type is MissionType.OBSERVE
    assert tl2.mission is None, "overhearing an order is information, not authority"
    order_meta = next(meta for m, meta in zip(env.last_messages, env.last_message_meta, strict=False)
                      if m.kind is MessageKind.ORDER)
    assert set(order_meta["heard_by"]) == {"TL1", "TL2"}
    assert any(m.kind is MessageKind.ACK for m in env.last_messages)
    # the overhearing sibling now KNOWS the recipient's mission (semantic refresh)
    assert env._friendly_state["SL1"][tl1.id][1] is MissionType.OBSERVE
    assert env._friendly_state["TL2"].get(tl1.id) is None, "TL2 is not related to TL1"


def test_execute_releases_only_pending_recipients_it_reaches():
    env = _voice_env()
    amc = _order(0, "ADVANCE", amc=True)
    _place(env, SL1=(10, 10), TL1=(11, 10), TL2=(10, 11))
    _step(env, {"SL1": amc})
    tl1 = env.roster.by_callsign["TL1"]
    assert tl1.mission is not None and tl1.mission.awaiting_signal
    # a second pending recipient, then TL1 walks beyond signal range (6)
    amc2 = _order(1, "ADVANCE", amc=True)
    _step(env, {"SL1": amc2})
    tl2 = env.roster.by_callsign["TL2"]
    assert tl2.mission is not None and tl2.mission.awaiting_signal
    _place(env, SL1=(10, 10), TL1=(20, 10), TL2=(13, 10))
    assert _mask(env, "SL1")[EXECUTE] == 1
    _step(env, {"SL1": EXECUTE})
    assert not tl2.mission.awaiting_signal, "within signal range: released"
    assert tl1.mission.awaiting_signal, "beyond signal range: still staged"
    meta = next(meta for m, meta in zip(env.last_messages, env.last_message_meta, strict=False)
                if m.kind is MessageKind.EXECUTE)
    assert meta["medium"] == "signal" and meta["heard_by"] == ["TL2"]
    # the signal is a louder sound than voice
    sig = [e for e in env.last_sound_events if e.kind == "signal"]
    assert sig and sig[0].base_radius == snd.SIGNAL_DETECT_RADIUS


def test_gesture_execute_needs_a_visual_edge_emits_no_sound_and_releases_only_its_audience():
    env = _voice_env()
    amc = _order(0, "ADVANCE", amc=True)
    _place(env, SL1=(10, 10), TL1=(11, 10), TL2=(10, 11))
    _step(env, {"SL1": amc})
    _step(env, {"SL1": _order(1, "ADVANCE", amc=True)})
    tl1, tl2 = env.roster.by_callsign["TL1"], env.roster.by_callsign["TL2"]
    # TL2 in gesture range with LOS, TL1 behind a wall at the same distance
    _place(env, SL1=(10, 10), TL1=(14, 10), TL2=(10, 14))
    env.world.grid[10, 12] = WALL
    assert _mask(env, "SL1")[GESTURE_EXECUTE] == 1
    _step(env, {"SL1": GESTURE_EXECUTE})
    assert not tl2.mission.awaiting_signal
    assert tl1.mission.awaiting_signal, "a wall blocks a gesture"
    assert _blue_sounds(env) == [], "a gesture makes no sound"
    meta = next(meta for m, meta in zip(env.last_messages, env.last_message_meta, strict=False)
                if m.kind is MessageKind.EXECUTE)
    assert meta["medium"] == "gesture" and meta["heard_by"] == ["TL2"]
    # gestures are not a radio-mode action
    radio = make_env("squad")
    radio.reset(seed=1)
    assert all(radio._mask_for(s)[GESTURE_EXECUTE] == 0 for s in radio.roster.living)


def test_gesture_go_synchronizes_only_visible_registered_peers():
    env = _voice_env()
    _place(env, TL1=(20, 20), RFN1=(21, 20), RFN2=(20, 21))
    assert _mask(env, "RFN1")[SYNC_PROPOSE] == 1
    _step(env, {"RFN1": SYNC_PROPOSE})
    # RFN2 still visible (5 cells), TL1 walled off
    _place(env, TL1=(24, 20), RFN1=(21, 20), RFN2=(21, 25))
    env.world.grid[20, 23] = WALL
    assert _mask(env, "RFN1")[GESTURE_GO] == 1
    _step(env, {"RFN1": GESTURE_GO})
    assert env._synchronized(env.roster.by_callsign["RFN2"]) is not None
    assert env._synchronized(env.roster.by_callsign["TL1"]) is None
    assert _blue_sounds(env) == []


# ------------------------------------------------------------------ #
# reports and pictures
# ------------------------------------------------------------------ #


def test_contact_from_held_intel_updates_only_listeners_and_pays_on_the_superiors_novelty():
    env = _voice_env()
    rfn1 = env.roster.by_callsign["RFN1"]
    enemy = env.enemies[0]
    enemy.pos = (18, 10)
    enemy.home = enemy.pos
    _place(env, RFN1=(10, 10), TL1=(30, 30), SL1=(32, 30), RFN2=(1, 22))
    _step(env)  # RFN1 sights the enemy: its OWN picture only
    assert enemy.id in env._agent_known["RFN1"]
    assert enemy.id not in env._agent_known["TL1"]
    assert _mask(env, "RFN1")[CONTACT] == 0, "no superior in earshot"
    # withdraw to the leader: the enemy is no longer visible, the intel is held
    enemy.pos = (5, 40)
    _place(env, RFN1=(31, 30), TL1=(30, 30), SL1=(32, 30), RFN2=(1, 22))
    assert not env._visible_enemies(rfn1)
    assert _mask(env, "RFN1")[CONTACT] == 1
    _, rewards, *_ = _step(env, {"RFN1": CONTACT})
    assert rewards["RFN1"] > 0.3, "novel to the intended superior: contact_new"
    assert env._agent_known["TL1"][enemy.id][:2] == (18.0, 10.0), "reported coords, not live"
    assert env._agent_known["SL1"][enemy.id][:2] == (18.0, 10.0), "overheard: picture updated"
    assert enemy.id not in env._agent_known["RFN2"], "out of earshot: stale"
    # the leader's picture clock moved; the force-wide clock did not
    assert env._picture_changed_step["TL1"] == env._step_count
    assert env._last_net_contact_step is None
    # a repeat to the same superior is redundant
    _, rewards, *_ = _step(env, {"RFN1": CONTACT})
    assert rewards["RFN1"] < 0


def test_acoustic_contact_preserves_coarse_fields_through_relay_and_never_becomes_a_contact():
    env = _voice_env()
    _place(env, RFN1=(10, 10), TL1=(11, 10), SL1=(30, 30))
    env._step_sounds = []
    env._emit_sound("weapon_fire", (22, 10), "hostile", snd.WEAPON_DETECT_RADIUS, source_cs="E0")
    env._deliver_sounds_to_blue()
    assert _mask(env, "RFN1")[ACOUSTIC] == 1
    _, rewards, *_ = _step(env, {"RFN1": ACOUSTIC})
    report = next(m for m in env.last_messages if m.kind is MessageKind.ACOUSTIC_CONTACT)
    from cohort.core.language import parse_acoustic_contact

    parsed = parse_acoustic_contact(report.text)
    assert parsed is not None and parsed["kind_index"] == 3 and "GRID" not in report.text
    assert rewards["RFN1"] > 0.2
    assert env._agent_known["TL1"] == {}, "a cue never enters the exact enemy picture"
    # TL1 carries it to SL1: same fields, same source step
    _place(env, RFN1=(10, 10), TL1=(29, 30), SL1=(30, 30))
    for _ in range(3):
        _step(env)
    assert _mask(env, "TL1")[ACOUSTIC] == 1
    _step(env, {"TL1": ACOUSTIC})
    relayed = next(m for m in env.last_messages if m.kind is MessageKind.ACOUSTIC_CONTACT)
    assert parse_acoustic_contact(relayed.text) == parsed
    assert "CONTACT, GRID" not in relayed.text, "carriage never upgrades it to a CONTACT"
    assert env._agent_known["SL1"] == {}


def test_casualty_is_news_only_to_a_witness():
    env = _voice_env()
    sl1, tl1, rfn1 = (env.roster.by_callsign[c] for c in ("SL1", "TL1", "RFN1"))
    _place(env, SL1=(30, 30), TL1=(10, 10), RFN1=(11, 10))
    rfn1.health = 1
    from cohort.core.units import Trap

    env.traps = [Trap(id=0, pos=(12, 10), damage=50)]
    _step(env, {"RFN1": MOVE_EAST})
    assert not rfn1.alive
    assert env._element_casualty_step.get(tl1.id) == env._step_count, "TL1 saw it"
    assert sl1.id not in env._element_casualty_step, "SL1 did not"
    casualty_meta = next(meta for m, meta in zip(env.last_messages, env.last_message_meta, strict=False)
                         if m.kind is MessageKind.CASUALTY)
    assert casualty_meta["medium"] == "external" and casualty_meta["heard_by"] == []


# ------------------------------------------------------------------ #
# friendly telemetry gating
# ------------------------------------------------------------------ #


def test_a_non_visible_leader_stops_updating_live_deltas_and_last_known_ages():
    env = _voice_env()
    sl1 = env.roster.by_callsign["SL1"]
    _place(env, SL1=(10, 10), TL1=(12, 10))
    obs = env._observe(env.roster.by_callsign["TL1"], env._make_view(env.roster.by_callsign["TL1"]))["observation"]
    w = float(env.world.width)
    assert obs[OFF_LEADER] == 1.0 and abs(obs[OFF_LEADER + 1] - (-2 / w)) < 1e-6
    assert obs[COHESION_LEADER_SEEN] == 1.0 and obs[COHESION_LEADER_AGE] == 0.0
    # the leader walks away out of sight (beyond 8 cells), over several steps
    for x in range(11, 24):
        sl1.pos = (x, 10)
        _step(env)
    tl1 = env.roster.by_callsign["TL1"]
    obs = env._observe(tl1, env._make_view(tl1))["observation"]
    assert obs[COHESION_LEADER_SEEN] == 0.0
    assert obs[COHESION_LEADER_AGE] > 0.0
    # the delta is the LAST PERCEIVED position (within 8 cells), not the live one
    assert obs[OFF_LEADER + 1] * w < 9.0, "live delta would be 11 cells"
    # a radio scenario keeps live telemetry
    radio = make_env("squad")
    radio.reset(seed=1)
    r_sl1, r_tl1 = radio.roster.by_callsign["SL1"], radio.roster.by_callsign["TL1"]
    r_sl1.pos, r_tl1.pos = (10, 10), (30, 30)
    radio._update_visual_links()
    o = radio._observe(r_tl1, radio._make_view(r_tl1))["observation"]
    assert abs(o[OFF_LEADER + 1] * float(radio.world.width) - (-20)) < 1e-4


def test_subordinate_slots_are_gated_the_same_way():
    env = _voice_env()
    sl1 = env.roster.by_callsign["SL1"]
    _place(env, SL1=(10, 10), TL1=(30, 10))
    for _ in range(3):
        _step(env)
    obs = env._observe(sl1, env._make_view(sl1))["observation"]
    # TL1 is SL1's slot 0: present (known at the briefing), last-known delta small
    assert obs[OFF_SUBS] == 1.0
    assert abs(obs[OFF_SUBS + 1] * float(env.world.width)) < 9.0


# ------------------------------------------------------------------ #
# presets, provenance, determinism
# ------------------------------------------------------------------ #


def test_presets_mirror_squad_exactly_except_the_documented_fields():
    squad = get_scenario("squad")
    direct = get_scenario("squad_voice_direct")
    abl = get_scenario("squad_voice_no_acoustic_ablation")
    differ = {"name", "description", "comm_model", "sound_model", "voice_range",
              "reward_overrides", "experiment_arm"}
    from dataclasses import fields

    for f in fields(squad):
        if f.name in differ:
            continue
        assert getattr(squad, f.name) == getattr(direct, f.name) == getattr(abl, f.name), f.name
    assert (direct.comm_model, direct.sound_model, direct.voice_range) == ("voice_only", "tactical", 2.0)
    assert abl.sound_model == "off" and abl.comm_model == "voice_only"
    b = make_env("squad_voice_direct").briefing()
    for key in ("comm_model", "voice_range", "hq_available", "sound_model", "acoustics",
                "acoustic_report_ttl", "liaison_enabled", "visual_link_priced"):
        assert key in b, key
    assert b["liaison_enabled"] is False and b["visual_link_priced"] is True
    assert make_env("squad_voice_direct").rewards_cfg.root_done_bonus == 0.0


def test_voice_only_episode_is_deterministic_under_seed():
    def run(seed):
        env = make_env("squad_voice_direct")
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(3)
        log = []
        for _ in range(60):
            if not env.agents:
                break
            acts = {cs: int(rng.choice(np.flatnonzero(obs[cs]["action_mask"]))) for cs in env.agents}
            obs, rewards, *_ = env.step(acts)
            log.append((sorted(rewards.items()), [s.pos for s in env.roster.soldiers],
                        [m.text for m in env.last_messages]))
        return log

    assert run(4) == run(4)
