"""Ground-truth oracle: correct tags, and provably invisible to the cohort."""

import json

import numpy as np

from cohort import make_env
from cohort.core.oracle import WOUNDED_BELOW, Observable, unit_observables
from cohort.env.actions import CATALOG

STAY = 0
FIRE = next(s.index for s in CATALOG if s.kind == "fire")


def _flat_env(seed=3):
    env = make_env("fireteam")
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (22, 1)
        e.home = e.pos
        e.prev_pos = e.pos
    return env


def _tags(**kw):
    base = {
        "alive": True,
        "health": 100,
        "pos": (5, 5),
        "prev_pos": (5, 5),
        "fired": False,
        "in_cover": False,
        "seen_by_any_opponent": True,
        "opponents": [(10, 5)],
        "weapon_range": 8.0,
        "has_los": lambda a, b: True,
    }
    base.update(kw)
    return unit_observables(**base)


def test_vocabulary_is_stable():
    assert {o.value for o in Observable} == {
        "attacking", "advancing", "retreating", "covering",
        "holding", "hidden", "wounded", "supporting", "supported", "down",
    }


def test_pure_tag_semantics():
    assert _tags(alive=False) == [Observable.DOWN]
    assert Observable.ATTACKING in _tags(fired=True)
    assert Observable.ADVANCING in _tags(prev_pos=(4, 5), pos=(5, 5))   # closing on (10,5)
    assert Observable.RETREATING in _tags(prev_pos=(6, 5), pos=(5, 5))  # opening distance
    assert Observable.COVERING in _tags()                                # static, LOS, in range
    assert Observable.HOLDING in _tags(opponents=[(30, 30)])             # static, nothing covered
    assert Observable.HIDDEN in _tags(in_cover=True, seen_by_any_opponent=False)
    assert Observable.HIDDEN not in _tags(in_cover=True, seen_by_any_opponent=True)
    assert Observable.WOUNDED in _tags(health=WOUNDED_BELOW - 1)
    assert Observable.WOUNDED not in _tags(health=WOUNDED_BELOW)
    assert Observable.SUPPORTING in _tags(supporting=True)
    assert Observable.SUPPORTED in _tags(supported=True)
    assert Observable.SUPPORTING not in _tags()
    assert _tags(alive=False, supporting=True) == [Observable.DOWN]


def test_support_tags_in_snapshot():
    """An in-position supporter and its supported element carry the tags;
    both disappear the moment the supporter leaves its position."""
    from cohort.core.missions import Mission, MissionType

    env = make_env("squad")
    env.reset(seed=3)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (26, 1)
        e.home = e.pos
        e.prev_pos = e.pos
    tl1 = env.roster.by_callsign["TL1"]
    tl2 = env.roster.by_callsign["TL2"]
    tl2.pos = (10, 10)
    tl1.pos = (14, 10)  # in range + LOS on open ground
    tl1.mission = Mission(
        MissionType.SUPPORT, None, tl2.pos, issuer_id=-1, step_assigned=0,
        extra={"supported_id": tl2.id},
    )
    snap = env.oracle()
    by_cs = {s["cs"]: s for s in snap["soldiers"]}
    assert "supporting" in by_cs["TL1"]["tags"]
    assert "supported" in by_cs["TL2"]["tags"]
    assert "supported" in by_cs["RFN3"]["tags"], "the supported element includes the team"
    assert "supported" not in by_cs["RFN1"]["tags"], "TL1's own riflemen are not covered"

    tl1.pos = (26, 10)  # out of SUPPORT range → effects off
    snap = env.oracle()
    by_cs = {s["cs"]: s for s in snap["soldiers"]}
    assert "supporting" not in by_cs["TL1"]["tags"]
    assert "supported" not in by_cs["TL2"]["tags"]


def test_oracle_snapshot_tags_and_enemy_internals():
    env = _flat_env()
    rfn = env.roster.by_callsign["RFN1"]
    enemy = env.enemies[0]
    enemy.pos = (rfn.pos[0] + 2, rfn.pos[1])
    enemy.prev_pos = enemy.pos
    enemy.health = WOUNDED_BELOW - 10

    env.step({a: (FIRE if a == "RFN1" else STAY) for a in env.agents})
    snap = env.oracle()

    me = next(s for s in snap["soldiers"] if s["cs"] == "RFN1")
    assert "attacking" in me["tags"] or me["ammo"] == 30  # fired unless target died first
    foe = next(e for e in snap["enemies"] if e["id"] == enemy.id)
    if foe["alive"]:
        assert "wounded" in foe["tags"]
    # OpFor AI internals are exposed to the oracle...
    for key in ("mode", "home", "goal", "last_seen_player", "last_seen_step", "seen_by"):
        assert key in foe
    assert foe["mode"] in ("garrison", "assault")
    # ...and the whole snapshot is JSON-serializable for external consumers
    json.dumps(snap)


def test_oracle_is_invisible_to_observations():
    """Calling the oracle changes no agent observation, mask, or env state."""
    env = make_env("fireteam")
    obs_before, _ = env.reset(seed=9)
    before = {a: (obs_before[a]["observation"].copy(), obs_before[a]["action_mask"].copy())
              for a in env.agents}
    env.oracle()
    obs_after = env._all_observations()
    for a in before:
        assert np.array_equal(before[a][0], obs_after[a]["observation"])
        assert np.array_equal(before[a][1], obs_after[a]["action_mask"])


def test_oracle_consumes_no_randomness():
    """Two identical seeded rollouts stay identical when one calls the oracle."""
    def rollout(with_oracle):
        env = make_env("fireteam")
        env.reset(seed=77)
        trace = []
        for _ in range(25):
            if not env.agents:
                break
            if with_oracle:
                env.oracle()
            env.step({a: STAY for a in env.agents})
            trace.append(tuple(e.pos for e in env.enemies) + tuple(s.pos for s in env.roster.soldiers))
        return trace, env.transcript.render()

    t1, log1 = rollout(with_oracle=False)
    t2, log2 = rollout(with_oracle=True)
    assert t1 == t2
    assert log1 == log2


def test_dead_units_are_down_only():
    env = _flat_env()
    env.enemies[1].alive = False
    env.step({a: STAY for a in env.agents})
    snap = env.oracle()
    foe = next(e for e in snap["enemies"] if e["id"] == env.enemies[1].id)
    assert foe["tags"] == ["down"]


def test_sighting_sets_match_the_environments_own_visibility():
    """``sees`` is the per-agent sighting set the truth stream promises (#17).

    It must equal ``env._visible_enemies`` exactly — same members, same
    nearest-first order — for every living soldier, at every step of a real
    rollout. Consumers previously had to transpose ``seen_by`` to get this,
    which disagreed with the simulation on 36-47% of agent-steps because
    ``seen_by`` was also computed for the dead.
    """
    env = make_env("squad")
    obs, _ = env.reset(seed=11)
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(60):
        if not env.agents:
            break
        snap = env.oracle()
        by_cs = {s["cs"]: s for s in snap["soldiers"]}
        for s in env.roster.soldiers:
            if not s.alive:
                continue
            assert by_cs[s.callsign]["sees"] == [e.id for e in env._visible_enemies(s)]
            checked += 1
        env.step({
            a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents
        })
        obs = env._all_observations()
    assert checked > 100, "the rollout must actually exercise living agents"


def test_the_dead_neither_see_nor_are_seen():
    """A corpse is not an observer and is not an observation (#17).

    Without this, a dead unit kept reporting sightings from its last position,
    and those entries dominated the stream: 8,901 of 9,647 enemy ``seen_by``
    entries over eight ``squad`` episodes named dead enemies.
    """
    env = _flat_env()
    rfn = env.roster.by_callsign["RFN1"]
    foe = env.enemies[0]
    foe.pos = (rfn.pos[0] + 1, rfn.pos[1])  # point blank, open ground: mutually visible
    foe.prev_pos = foe.pos
    env.step({a: STAY for a in env.agents})

    snap = env.oracle()
    live = next(e for e in snap["enemies"] if e["id"] == foe.id)
    me = next(s for s in snap["soldiers"] if s["cs"] == "RFN1")
    assert rfn.callsign in live["seen_by"], "precondition: the live enemy is spotted"
    assert foe.id in me["sees"], "precondition: RFN1 sees the live enemy"

    foe.alive = False
    snap = env.oracle()
    dead = next(e for e in snap["enemies"] if e["id"] == foe.id)
    me = next(s for s in snap["soldiers"] if s["cs"] == "RFN1")
    assert dead["seen_by"] == [], "a dead enemy is not being observed"
    assert foe.id not in me["sees"], "a dead enemy is not a sighting"

    rfn.alive = False
    snap = env.oracle()
    me = next(s for s in snap["soldiers"] if s["cs"] == "RFN1")
    assert me["seen_by"] == [] and me["sees"] == [], "a dead soldier neither sees nor is seen"
