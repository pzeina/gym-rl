"""Message.payload carries the structured facts behind every radio message.

The text form stays the source of presentation; the payload is the source of
truth for consumers (dashboard, evaluation, external tools) — no regexing the
NATO text back apart. Payload and text must always agree.
"""

import numpy as np

from cohort import make_env
from cohort.core.language import grid_ref, parse_order
from cohort.core.missions import Mission, MissionType
from cohort.env.actions import CATALOG

STAY = 0
CONTACT = next(s.index for s in CATALOG if s.kind == "contact")
SITREP = next(s.index for s in CATALOG if s.kind == "sitrep")
DONE = next(s.index for s in CATALOG if s.kind == "done")

#: Minimum payload keys per message kind.
EXPECTED_KEYS = {
    "opord": {"issuer", "recipient", "mission", "objective"},
    "order": {"issuer", "recipient", "mission", "objective"},
    "ack": {"issuer", "recipient"},
    "contact": {"sender", "recipient", "grid", "count"},
    "sitrep": {"sender", "recipient", "grid", "health", "ammo"},
    "done": {"sender", "recipient", "mission", "objective"},
    "casualty": {"callsign"},
    "taking_command": {"successor", "replaced", "assumed_command"},
}


def _flat_env(seed=1):
    """Fireteam env with all-open terrain and enemies parked far away."""
    env = make_env("fireteam")
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (22, 1)
        e.home = e.pos
    return env


def _step_all(env, overrides):
    acts = {a: STAY for a in env.agents}
    acts.update(overrides)
    return env.step(acts)


def _last(env, kind):
    return next(m for m in reversed(env.transcript.messages) if m.kind.value == kind)


def test_opord_payload_matches_text():
    env = make_env("fireteam")
    env.reset(seed=1)
    msg = env.transcript.messages[0]
    assert msg.kind.value == "opord"
    assert msg.payload == {
        "issuer": "HQ",
        "recipient": "TL1",
        "mission": "SEIZE",
        "objective": "ALPHA",
    }
    parsed = parse_order(msg.text)
    assert parsed.recipient_callsign == msg.payload["recipient"]
    assert parsed.mission.name == msg.payload["mission"]
    assert parsed.objective_name == msg.payload["objective"]


def test_order_and_ack_payloads_match_text():
    env = make_env("fireteam")
    env.reset(seed=1)
    env.inject_order("RFN1, seize obj alpha", issuer="TL1")
    order = _last(env, "order")
    assert order.payload == {
        "issuer": "TL1",
        "recipient": "RFN1",
        "mission": "SEIZE",
        "objective": "ALPHA",
    }
    parsed = parse_order(order.text)
    assert parsed.recipient_callsign == order.payload["recipient"]
    assert parsed.mission.name == order.payload["mission"]
    assert parsed.objective_name == order.payload["objective"]
    ack = _last(env, "ack")
    assert ack.payload == {"issuer": "TL1", "recipient": "RFN1"}
    assert f"{ack.payload['issuer']}, THIS IS {ack.payload['recipient']}: WILCO" in ack.text


def test_contact_payload_matches_text():
    env = _flat_env()
    sld = env.roster.by_callsign["RFN1"]
    enemy = env.enemies[0]
    enemy.pos = (10, 10)
    sld.pos = (7, 10)
    _step_all(env, {"RFN1": CONTACT})
    msg = _last(env, "contact")
    assert msg.payload["sender"] == "RFN1"
    assert msg.payload["recipient"] == "TL1"
    assert msg.payload["grid"] == [10, 10]
    assert msg.payload["count"] >= 1
    assert grid_ref(tuple(msg.payload["grid"])) in msg.text
    assert f"{msg.payload['count']} x ENEMY" in msg.text


def test_sitrep_payload_matches_text():
    env = _flat_env()
    sld = env.roster.by_callsign["RFN2"]
    _step_all(env, {"RFN2": SITREP})
    msg = _last(env, "sitrep")
    assert msg.payload["sender"] == "RFN2"
    assert msg.payload["grid"] == [sld.pos[0], sld.pos[1]]
    assert msg.payload["health"] == sld.health
    assert msg.payload["ammo"] == sld.ammo
    assert grid_ref(tuple(msg.payload["grid"])) in msg.text
    assert f"HEALTH {msg.payload['health']}%" in msg.text
    assert f"AMMO {msg.payload['ammo']}" in msg.text


def test_done_payload_matches_text():
    env = _flat_env()
    sld = env.roster.by_callsign["RFN1"]
    obj = env.world.objectives[1]  # BRAVO, no enemies near after _flat_env
    for e in env.enemies:
        e.pos = (1, 22)
        e.home = e.pos
    sld.pos = obj.pos
    sld.mission = Mission(MissionType.SEIZE, 1, obj.pos, issuer_id=-1, step_assigned=0)
    _step_all(env, {"RFN1": DONE})
    msg = _last(env, "done")
    assert msg.payload["sender"] == "RFN1"
    assert msg.payload["mission"] == "SEIZE"
    assert msg.payload["objective"] == "BRAVO"
    assert "SEIZE OBJ BRAVO — COMPLETE" in msg.text


def test_casualty_and_succession_payloads():
    env = make_env("fireteam")
    env.reset(seed=5)
    cap = env.roster.by_callsign["TL1"]
    cap.health = 1
    enemy = next(e for e in env.enemies if e.alive)
    cap.pos = (enemy.pos[0] + 1, enemy.pos[1])
    for _ in range(30):
        if not cap.alive:
            break
        env.step({a: 0 for a in env.agents})
    assert not cap.alive
    casualty = _last(env, "casualty")
    assert casualty.payload == {"callsign": "TL1"}
    take = _last(env, "taking_command")
    assert take.payload["replaced"] == "TL1"
    assert take.payload["assumed_command"] is True
    assert f"THIS IS {take.payload['successor']}" in take.text


def test_every_message_carries_its_payload():
    """Random rollout: every kind that appears ships its structured payload."""
    env = make_env("squad")
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(80):
        if not env.agents:
            break
        acts = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
        obs, *_ = env.step(acts)
    kinds_seen = set()
    for m in env.transcript.messages:
        kinds_seen.add(m.kind.value)
        assert set(m.payload) >= EXPECTED_KEYS[m.kind.value], (m.kind, m.payload)
    assert {"opord", "order", "ack"} <= kinds_seen


def test_dashboard_trace_exposes_payloads():
    from cohort.viz.dashboard import record_episode

    trace = record_episode("fireteam", None, seed=5, max_steps=10)
    msgs = [m for step in trace["steps"] for m in step["messages"]]
    assert msgs and all("data" in m for m in msgs)
    opord = next(m for m in msgs if m["kind"] == "opord")
    assert opord["data"]["mission"] == "SEIZE"
    assert opord["data"]["objective"] == "ALPHA"
