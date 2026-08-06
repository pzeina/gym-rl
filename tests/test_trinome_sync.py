"""Trinôme peer synchronization (A5-4): SYNC_PROPOSE / SYNC_GO by voice.

The manual's bond par binôme (pp. 14-15) — "les bonds sont exécutés au
commandement donné à la voix ou aux gestes". Any agent may propose a bound
to peers within voice range; GO synchronizes proposer + registered living
peers for the next 8 steps: movers closing new ground toward their target
under a COVERING group-mate earn the bound bonus and the covered-movement
accuracy debuff vs attackers (the B5/P2 support machinery at binôme scale).
"""

import numpy as np

from cohort import make_env
from cohort.core.units import voice_peers
from cohort.env.actions import CATALOG
from cohort.env.cohort_env import SYNC_PROPOSE_TTL, SYNC_WINDOW

STAY = 0
PROPOSE_IDX = next(s.index for s in CATALOG if s.kind == "sync_propose")
GO_IDX = next(s.index for s in CATALOG if s.kind == "sync_go")
MOVE_EAST = next(s.index for s in CATALOG if s.name == "MOVE_EAST")


def _stay_all(env):
    return dict.fromkeys(env.agents, STAY)


def _flat_squad(seed=3):
    env = make_env("squad")
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (41, 1)
        e.home = e.pos
    return env


# ---------------------------------------------------------------------- #
# peers
# ---------------------------------------------------------------------- #


def test_voice_peers_same_element_and_adjacent_trinome():
    env = _flat_squad()
    r = env.roster
    rfn1 = r.by_callsign["RFN1"]  # element: TL1 + RFN1 + RFN2
    tl1 = r.by_callsign["TL1"]
    rfn3 = r.by_callsign["RFN3"]  # adjacent trinôme (TL2's element)
    sl = r.by_callsign["SL1"]
    # co-locate everyone tightly
    for i, s in enumerate(r.soldiers):
        s.pos = (10 + i, 10)
    names = {p.callsign for p in voice_peers(rfn1, r, env.spec_cfg.voice_range)}
    assert tl1.callsign in names, "own leader is a peer"
    assert "RFN2" in names, "sibling is a peer"
    assert rfn3.callsign in names or "TL2" in names, "adjacent trinôme members are peers"
    assert sl.callsign not in names, "the SL is two echelons up, not a trinôme peer"


def test_voice_peers_range_gated():
    env = _flat_squad()
    r = env.roster
    rfn1 = r.by_callsign["RFN1"]
    rfn2 = r.by_callsign["RFN2"]
    rfn1.pos = (10, 10)
    rfn2.pos = (30, 30)  # far out of voice range
    for s in r.soldiers:
        if s.callsign not in ("RFN1", "RFN2"):
            s.pos = (35, 5)
    assert voice_peers(rfn1, r, env.spec_cfg.voice_range) == []
    rfn2.pos = (13, 10)
    assert [p.callsign for p in voice_peers(rfn1, r, env.spec_cfg.voice_range)] == ["RFN2"]


# ---------------------------------------------------------------------- #
# propose / GO windows + masks
# ---------------------------------------------------------------------- #


def test_propose_then_go_flow_and_masks():
    env = _flat_squad()
    r = env.roster
    rfn1 = r.by_callsign["RFN1"]
    for i, s in enumerate(r.soldiers):
        s.pos = (10 + i, 10)
    assert env._mask_for(rfn1)[PROPOSE_IDX] == 1, "peers in range: may propose"
    assert env._mask_for(rfn1)[GO_IDX] == 0, "nothing proposed yet"
    acts = _stay_all(env)
    acts["RFN1"] = PROPOSE_IDX
    env.step(acts)
    assert rfn1.id in env._sync_pending
    prop = next(m for m in env.transcript.messages if m.kind.value == "sync_propose")
    assert prop.voice, "spoken, not transmitted"
    assert "THIS IS RFN1: PREPARE TO BOUND ON MY SIGNAL. OUT." in prop.text
    assert env._mask_for(rfn1)[GO_IDX] == 1
    acts = _stay_all(env)
    acts["RFN1"] = GO_IDX
    env.step(acts)
    go = next(m for m in env.transcript.messages if m.kind.value == "sync_go")
    assert go.text == "RFN1: GO! OUT." and go.voice
    assert env._synchronized(rfn1) is not None
    assert env._mask_for(rfn1)[GO_IDX] == 0, "proposal consumed"
    # the window closes after SYNC_WINDOW steps
    for _ in range(SYNC_WINDOW):
        env.step(_stay_all(env))
    assert env._synchronized(rfn1) is None


def test_stale_proposal_expires():
    env = _flat_squad()
    r = env.roster
    rfn1 = r.by_callsign["RFN1"]
    for i, s in enumerate(r.soldiers):
        s.pos = (10 + i, 10)
    acts = _stay_all(env)
    acts["RFN1"] = PROPOSE_IDX
    env.step(acts)
    for _ in range(SYNC_PROPOSE_TTL + 1):
        env.step(_stay_all(env))
    assert env._mask_for(rfn1)[GO_IDX] == 0, "the moment has passed"


def test_peers_registered_at_propose_time_not_go_time():
    env = _flat_squad()
    r = env.roster
    rfn1, rfn2 = r.by_callsign["RFN1"], r.by_callsign["RFN2"]
    rfn1.pos = (10, 10)
    rfn2.pos = (12, 10)
    for s in r.soldiers:
        if s.callsign not in ("RFN1", "RFN2"):
            s.pos = (35, 5)
    acts = _stay_all(env)
    acts["RFN1"] = PROPOSE_IDX
    env.step(acts)
    rfn2.pos = (30, 30)  # walks away AFTER the proposal
    acts = _stay_all(env)
    acts["RFN1"] = GO_IDX
    env.step(acts)
    assert env._synchronized(rfn2) is not None, "was in range at propose time: bound holds"


# ---------------------------------------------------------------------- #
# bound bonus + cover interplay
# ---------------------------------------------------------------------- #


def _bounding_pair(env):
    """RFN1 tasked toward ALPHA (east), RFN2 static covering it."""
    r = env.roster
    rfn1, rfn2 = r.by_callsign["RFN1"], r.by_callsign["RFN2"]
    env.inject_order("TL1, seize obj alpha", issuer="SL1")
    env.inject_order("RFN1, seize obj alpha", issuer="TL1")
    rfn1.pos = (10, 10)
    rfn1.prev_pos = rfn1.pos
    rfn2.pos = (12, 10)
    rfn2.prev_pos = rfn2.pos
    for s in r.soldiers:
        if s.callsign not in ("RFN1", "RFN2"):
            s.pos = (35, 40)
    return rfn1, rfn2


def test_bound_bonus_requires_covering_peer():
    env = _flat_squad()
    rfn1, _rfn2 = _bounding_pair(env)
    acts = _stay_all(env)
    acts["RFN1"] = PROPOSE_IDX
    env.step(acts)
    acts = _stay_all(env)
    acts["RFN1"] = GO_IDX
    env.step(acts)
    # covered bound: RFN2 static with LOS to the mover
    acts = _stay_all(env)
    acts["RFN1"] = MOVE_EAST
    _, rewards, *_ = env.step(acts)
    covered_reward = rewards["RFN1"]
    assert env._covered_by_sync(rfn1)

    # same geometry WITHOUT a sync window: no bonus, no cover
    env2 = _flat_squad()
    rfn1b, _rfn2b = _bounding_pair(env2)
    acts = _stay_all(env2)
    acts["RFN1"] = MOVE_EAST
    _, rewards2, *_ = env2.step(acts)
    assert not env2._covered_by_sync(rfn1b)
    assert covered_reward > rewards2["RFN1"], "the covered bound out-earns the lone rush"


def test_bound_watermark_blocks_refarming():
    """Retreating and re-bounding the same ground pays nothing new."""
    env = _flat_squad()
    _rfn1, _rfn2 = _bounding_pair(env)
    cfg = env.rewards_cfg

    def sync_cycle():
        acts = _stay_all(env)
        acts["RFN1"] = PROPOSE_IDX
        env.step(acts)
        acts = _stay_all(env)
        acts["RFN1"] = GO_IDX
        env.step(acts)

    move_w = next(s.index for s in CATALOG if s.name == "MOVE_WEST")
    sync_cycle()
    gains = []
    for a in (MOVE_EAST, MOVE_EAST):  # two cells of new closure, covered
        acts = _stay_all(env)
        acts["RFN1"] = a
        _, rw, *_ = env.step(acts)
        gains.append(rw["RFN1"])
    # walk back two, re-propose, walk the same ground again
    for a in (move_w, move_w):
        acts = _stay_all(env)
        acts["RFN1"] = a
        env.step(acts)
    sync_cycle()
    regains = []
    for a in (MOVE_EAST, MOVE_EAST):
        acts = _stay_all(env)
        acts["RFN1"] = a
        _, rw, *_ = env.step(acts)
        regains.append(rw["RFN1"])
    # the re-walked ground carries no bound bonus: strictly poorer steps
    # (margin: half a bonus — tenure growth slightly lifts later compliance)
    assert sum(regains) < sum(gains) - cfg.bound_bonus / 2


def test_covered_bound_degrades_attacker_accuracy():
    """Fixed-seed parity: same volleys, fewer hits on a covered bounder."""
    from cohort.core.units import CombatParams, resolve_fire

    params = CombatParams()
    hits_plain = 0
    hits_covered = 0
    for seed in range(300):
        rng = np.random.default_rng(seed)
        hit, _ = resolve_fire((0, 0), (5, 0), False, 5.0, params, rng)
        hits_plain += hit
        rng = np.random.default_rng(seed)
        hit, _ = resolve_fire(
            (0, 0), (5, 0), False, 5.0, params, rng,
            modifier=params.support_cover_accuracy,
        )
        hits_covered += hit
    assert hits_covered < hits_plain


def test_sync_pending_and_window_observable():
    env = _flat_squad()
    r = env.roster
    for i, s in enumerate(r.soldiers):
        s.pos = (10 + i, 10)
    base = 13 + 22  # sync block right after the mission/stance block
    obs = env._all_observations()
    assert obs["RFN1"]["observation"][base] == 0.0
    acts = _stay_all(env)
    acts["RFN1"] = PROPOSE_IDX
    obs, *_ = env.step(acts)
    assert obs["RFN1"]["observation"][base] == 1.0, "proposer sees the pending bound"
    assert obs["TL1"]["observation"][base] == 1.0, "registered peer sees it too"
    acts = _stay_all(env)
    acts["RFN1"] = GO_IDX
    obs, *_ = env.step(acts)
    assert obs["RFN1"]["observation"][base + 1] > 0.9, "window just opened"
    for _ in range(SYNC_WINDOW):
        obs, *_ = env.step(_stay_all(env))
    assert obs["RFN1"]["observation"][base + 1] == 0.0, "window closed"


def test_sync_determinism():
    def run(seed):
        env = make_env("squad")
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        trail = []
        obs = env._all_observations()
        for _ in range(50):
            if not env.agents:
                break
            acts = {}
            for a in env.agents:
                legal = np.flatnonzero(obs[a]["action_mask"])
                acts[a] = int(rng.choice(legal))
            # force some sync traffic into the mix
            for a in env.agents:
                soldier = env.roster.by_callsign[a]
                if env._mask_for(soldier)[GO_IDX]:
                    acts[a] = GO_IDX
                elif env._mask_for(soldier)[PROPOSE_IDX] and rng.random() < 0.3:
                    acts[a] = PROPOSE_IDX
            obs, rewards, *_ = env.step(acts)
            trail.append((tuple(s.pos for s in env.roster.soldiers), tuple(sorted(rewards.items()))))
        return trail

    assert run(33) == run(33)


# ---------------------------------------------------------------------- #
# airtime (issue #18): voice is no longer free
# ---------------------------------------------------------------------- #


def test_voice_pays_airtime_like_every_other_transmission():
    """The owner's call on #18: no channel is free.

    While SYNC was uncharged it was the only action a policy could emit at
    no cost, and `squad_screen_v4/ckpt_latest` poured 93% of its traffic
    into it — 1173 messages an episode against its own best checkpoint's
    170 — to run the clock out at 0% success. An action sink is not a
    doctrine.
    """
    env = _flat_squad()
    for i, s in enumerate(env.roster.soldiers):
        s.pos = (10 + i, 10)
    cost = env.rewards_cfg.transmission_cost

    acts = _stay_all(env)
    acts["RFN1"] = PROPOSE_IDX
    *_, infos = env.step(acts)
    assert infos["RFN1"]["components"]["report"] == cost, "PREPARE TO BOUND pays airtime"
    assert env.transmissions_last_step == 1, "and it is counted as a transmission"

    acts = _stay_all(env)
    acts["RFN1"] = GO_IDX
    *_, infos = env.step(acts)
    assert infos["RFN1"]["components"]["report"] == cost, "GO pays airtime"
    assert env.transmissions_last_step == 1

    # charged to report, never command: the flat arm has no chain of command
    # and its command reward must stay exactly 0.0 (tests/test_ablation.py)
    assert infos["RFN1"]["components"]["command"] == 0.0


def test_a_sync_that_says_nothing_costs_nothing():
    """Airtime is paid for what is SAID. A GO with no live proposal, or one
    past its TTL, emits no message and so must not be charged."""
    env = _flat_squad()
    rfn1 = env.roster.by_callsign["RFN1"]
    for i, s in enumerate(env.roster.soldiers):
        s.pos = (10 + i, 10)

    # a GO the mask would refuse: nothing pending, nothing said, nothing paid
    acts = _stay_all(env)
    acts["RFN1"] = GO_IDX
    *_, infos = env.step(acts)
    assert infos["RFN1"]["components"]["report"] == 0.0
    assert env.transmissions_last_step == 0

    # a proposal that goes stale: the PROPOSE was said and paid, the expired
    # GO is not
    acts = _stay_all(env)
    acts["RFN1"] = PROPOSE_IDX
    env.step(acts)
    for _ in range(SYNC_PROPOSE_TTL + 1):
        env.step(_stay_all(env))
    before = len(env.transcript.messages)
    acts = _stay_all(env)
    acts["RFN1"] = GO_IDX
    *_, infos = env.step(acts)
    assert infos["RFN1"]["components"]["report"] == 0.0, "the moment passed; nothing said"
    assert len(env.transcript.messages) == before
    assert env._synchronized(rfn1) is None
