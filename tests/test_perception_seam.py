"""The public perception seam (epistream HOST_REQUESTS items 1, 2, 6 + accessor).

Visual-contact records are sight's analog of the acoustic cue — coarse,
listener-attributed, bounded, carrying no ground truth — and, unlike cues,
they are host-side telemetry ONLY: the boundary test at the bottom pins that
maintaining them changes no agent's observation or reward."""

from dataclasses import replace

import numpy as np
import pytest

from cohort import make_env
from cohort.config import get_scenario
from cohort.core import liaison as lia
from cohort.core import perception as per


def _env_with_enemy_east(seed=1, gap=3, **spec_overrides):
    """Fireteam on a flattened map, every enemy parked ``gap`` cells east of
    TL1 (in sight for gap <= vision range)."""
    spec = replace(get_scenario("fireteam"), sound_model="tactical", **spec_overrides)
    env = make_env(spec)
    env.reset(seed=seed)
    env.world.grid[:] = 0
    tl = env.roster.by_callsign["TL1"]
    for e in env.enemies:
        e.pos = (tl.pos[0] + gap, tl.pos[1])
        e.home = e.pos
        e.prev_pos = e.pos
    return env


def _stay(env):
    return {a: 0 for a in env.agents}


# --------------------------------------------------------------------- #
# the record (unit level)
# --------------------------------------------------------------------- #


def test_visual_contact_carries_only_bounded_observer_fields():
    """No true cell, no enemy ids, no fabricated confidence — the fields ARE
    the coarseness, exactly like the acoustic cue's discipline."""
    fields = set(per.VisualContact.__dataclass_fields__)
    assert fields == {"bearing", "distance_band", "count_band", "formed_step"}


def test_observe_contacts_groups_by_sector_and_band():
    contacts = per.observe_contacts((5, 5), [(8, 5), (11, 5), (5, 15)], step=3)
    assert [(c.bearing, c.distance_band, c.count_band) for c in contacts] == [
        (0, 0, 0),   # (8,5): east, near (dist 3), lone
        (0, 1, 0),   # (11,5): east, medium (dist 6), lone
        (2, 1, 0),   # (5,15): south, medium (dist 10), lone
    ]
    # a pair in one (sector, band) cell reads as one contact, count band 1
    pair = per.observe_contacts((5, 5), [(8, 5), (9, 5), (7, 8)], step=3)
    assert [(c.bearing, c.distance_band, c.count_band) for c in pair] == [(0, 0, 1), (1, 0, 0)]


def test_merge_contacts_refreshes_expires_and_truncates():
    old = per.VisualContact(bearing=0, distance_band=0, count_band=0, formed_step=0)
    memory = [old]
    # same (sector, band) cell refreshes in place
    fresh = [per.VisualContact(bearing=0, distance_band=0, count_band=1, formed_step=5)]
    merged = per.merge_contacts(memory, fresh, step=5)
    assert len(merged) == 1 and merged[0].count_band == 1 and merged[0].formed_step == 5
    # TTL expiry
    assert per.merge_contacts([old], [], step=per.VISUAL_CONTACT_TTL) == []
    assert per.merge_contacts([old], [], step=per.VISUAL_CONTACT_TTL - 1) == [old]
    # cap at the freshest MAX, stable by (sector, band)
    many = [
        per.VisualContact(bearing=b, distance_band=0, count_band=0, formed_step=b)
        for b in range(6)
    ]
    kept = per.merge_contacts(many, [], step=6)
    assert len(kept) == per.MAX_VISUAL_CONTACTS
    assert [c.bearing for c in kept] == [5, 4, 3, 2]  # freshest first


# --------------------------------------------------------------------- #
# the env memory
# --------------------------------------------------------------------- #


def test_contact_memory_forms_on_sighting_and_expires_out_of_sight():
    env = _env_with_enemy_east()
    env.step(_stay(env))
    contacts = env.perception("TL1")["visual_contacts"]
    assert contacts, "an enemy in the vision range must form a contact"
    assert contacts[0].bearing == 0 and contacts[0].distance_band == 0
    # enemies leave sight: the memory persists, then expires by TTL
    for e in env.enemies:
        e.pos = (30, 30)
        e.home = e.pos
        e.prev_pos = e.pos
    seen = []
    for _ in range(per.VISUAL_CONTACT_TTL + 1):
        env.step(_stay(env))
        seen.append(len(env.perception("TL1")["visual_contacts"]))
    assert seen[0] >= 1, "memory must outlive the sighting"
    assert seen[-1] == 0, "memory must expire by the TTL"


def test_dead_observer_holds_no_contacts():
    env = _env_with_enemy_east()
    env.step(_stay(env))
    tl = env.roster.by_callsign["TL1"]
    tl.health = 0
    tl.alive = False
    env.step(_stay(env))
    p = env.perception("TL1")
    assert p["alive"] is False and p["visual_contacts"] == []


# --------------------------------------------------------------------- #
# the public accessor
# --------------------------------------------------------------------- #


def test_perception_covers_the_former_private_seams():
    env = _env_with_enemy_east()
    env.step(_stay(env))
    p = env.perception("TL1")
    assert p["cues"] == env._agent_cues["TL1"]
    assert p["visual_contacts"] == env._visual_contacts["TL1"]
    # copies, not the live lists: a monitor cannot mutate env state
    p["cues"].clear()
    p["visual_contacts"].clear()
    assert env._visual_contacts["TL1"] != []


def test_perception_friendly_is_coarse_and_in_the_observer_frame():
    env = _env_with_enemy_east()
    env.step(_stay(env))
    friendly = env.perception("TL1")["friendly"]
    assert friendly, "a fireteam leader has related stations"
    for record in friendly.values():
        assert set(record) == {"seen_now", "bearing", "range_band", "age"}
        assert 0 <= record["bearing"] <= 7
        assert record["range_band"] in (0, 1, 2)
        assert record["age"] >= 0


def test_perception_states_self_knowledge_explicitly():
    env = _env_with_enemy_east()
    env.step(_stay(env))
    own = env.perception("TL1")["self"]
    assert own == {"alive": True, "pos": tuple(env.roster.by_callsign["TL1"].pos),
                   "health": 100, "ammo": 30}


def test_perception_rejects_unknown_callsigns():
    env = _env_with_enemy_east()
    with pytest.raises(KeyError):
        env.perception("NOSUCH9")


def test_packet_events_survive_the_per_step_clear():
    env = _env_with_enemy_east()
    packet = lia.MessagePacket(
        id=0, kind="contact", origin_id=1, origin_cs="TL1", recipient_id=2,
        recipient_cs="RFN1", text="x", created_step=0, source_step=0,
        ack_required=False, payload=(), holder_id=1,
    )
    env._log_packet("prepared", packet)
    env.step(_stay(env))  # clears _packet_log at the top of the step
    assert env._packet_log == []
    events = env.packet_events()
    assert len(events) == 1 and events[0]["event"] == "prepared"
    # copies: mutating the returned entry never touches the log
    events[0]["event"] = "tampered"
    assert env.packet_events()[0]["event"] == "prepared"
    assert env.packet_events(since_step=1) == []


# --------------------------------------------------------------------- #
# the boundary: telemetry only
# --------------------------------------------------------------------- #


def test_visual_contact_memory_never_touches_behavior(monkeypatch):
    """The promise in the docstring, pinned: removing the maintenance call
    changes no observation and no reward — the record is host telemetry,
    not a new sense."""
    def run(disabled):
        env = _env_with_enemy_east(seed=7)
        if disabled:
            monkeypatch.setattr(
                type(env), "_update_visual_contacts", lambda self: None
            )
        out = []
        for _ in range(5):
            obs, rew, _term, _trunc, _info = env.step(_stay(env))
            out.append((
                {a: np.asarray(o["observation"]).copy() for a, o in obs.items()},
                dict(rew),
            ))
            if not env.agents:
                break
        monkeypatch.undo()
        return out

    on, off = run(disabled=False), run(disabled=True)
    assert len(on) == len(off)
    for (obs_a, rew_a), (obs_b, rew_b) in zip(on, off, strict=True):
        assert rew_a == rew_b
        assert obs_a.keys() == obs_b.keys()
        for agent in obs_a:
            np.testing.assert_array_equal(obs_a[agent], obs_b[agent])
