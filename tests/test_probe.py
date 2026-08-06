"""Transparency probe (B4): rule engine, ground truth, and scoring.

The predictor is tested on hand-built transcripts (real ``core/language.py``
formatter output — nothing enters it that was not radio text), the ground
truth on hand-built traces with known classes, and the scoring end-to-end on
a synthetic episode where every pair's outcome is known, plus a recorded
real-environment episode for structure and determinism.
"""

import json

import numpy as np

from cohort.core import language as lang
from cohort.core.missions import MissionType
from cohort.probe import (
    FIRING,
    HOLD,
    LEADER,
    MOVING,
    STATIC,
    Briefing,
    K,
    NetPredictor,
    aggregate_probe,
    destination_truth,
    format_probe_table,
    make_briefing,
    obj_class,
    posture_truth,
    probe_episode,
    step_index,
)

# ---------------------------------------------------------------------- #
# transcript / trace construction helpers
# ---------------------------------------------------------------------- #


def m(kind, frm, to, text=""):
    return {"kind": kind, "from": frm, "to": to, "mission": None, "text": text}


def order(issuer, recipient, mission, target):
    return m("order", issuer, recipient, lang.format_order(issuer, recipient, mission, target))


def opord(recipient, mission, target):
    return m("opord", "HQ", recipient, lang.format_opord(recipient, mission, target))


def rec(cs, pos, *, alive=True, leader=None, fired=False):
    return {"cs": cs, "alive": alive, "pos": list(pos), "leader": leader, "fired": fired}


def gt_step(t, soldiers, messages=()):
    return {"t": t, "soldiers": soldiers, "enemies": [], "messages": list(messages)}


# ---------------------------------------------------------------------- #
# briefing (static scenario knowledge)
# ---------------------------------------------------------------------- #


def test_briefing_matches_env_org():
    from cohort.env.cohort_env import make_env

    brief = make_briefing("squad")
    env = make_env("squad")
    env.reset(seed=1)
    assert list(brief.org) == env.possible_agents
    for s in env.roster.soldiers:
        ldr = env.roster.leader_of(s)
        assert brief.org[s.callsign] == (ldr.callsign if ldr is not None else None)
    assert brief.objectives == dict(env.spec_cfg.objectives)
    assert brief.spawn == env.spec_cfg.spawn
    assert brief.dest_classes == ["OBJ ALPHA", "OBJ BRAVO", "OBJ CHARLIE", LEADER, HOLD]


# ---------------------------------------------------------------------- #
# the rule engine on hand-built transcripts
# ---------------------------------------------------------------------- #


def test_opord_sets_destination_untasked_stand_by():
    p = NetPredictor(make_briefing("squad"))
    p.observe(0, [opord("SL1", MissionType.SEIZE, "ALPHA")])
    assert p.predict("SL1") == (obj_class("ALPHA"), MOVING)  # in transit from spawn
    assert p.predict("RFN1") == (HOLD, STATIC)  # never tasked: standing by


def test_transit_time_estimates_arrival():
    # spawn (5,5) -> ALPHA (33,33): manhattan 56, SEIZE radius 2.5 -> 54 steps
    p = NetPredictor(make_briefing("squad"))
    p.observe(0, [opord("SL1", MissionType.SEIZE, "ALPHA")])
    p.observe(53, [])
    assert p.predict("SL1")[1] == MOVING
    p.observe(54, [])
    assert p.predict("SL1")[1] == STATIC  # assumed on station, no contact on the net


def test_sitrep_evidence_shortcuts_arrival():
    p = NetPredictor(make_briefing("squad"))
    p.observe(0, [order("SL1", "TL1", MissionType.OBSERVE, "BRAVO")])
    assert p.predict("TL1") == (obj_class("BRAVO"), MOVING)
    sitrep = lang.format_sitrep("SL1", "TL1", 100, 30, (34, 10))  # 2 cells off BRAVO
    p.observe(5, [m("sitrep", "TL1", "SL1", sitrep)])
    assert p.predict("TL1") == (obj_class("BRAVO"), STATIC)


def test_hold_and_rally_classes():
    p = NetPredictor(make_briefing("squad"))
    p.observe(1, [order("TL1", "RFN1", MissionType.HOLD, None)])
    assert p.predict("RFN1") == (HOLD, STATIC)
    p.observe(2, [order("TL1", "RFN2", MissionType.RALLY, None)])
    assert p.predict("RFN2")[0] == LEADER
    # the leader falls (no successor announced): nobody left to rally on
    p.observe(3, [m("casualty", "HQ", "ALL", lang.format_casualty("TL1"))])
    assert p.predict("RFN2")[0] == HOLD


def test_done_needs_confirmation_to_clear():
    p = NetPredictor(make_briefing("squad"))
    p.observe(0, [order("SL1", "TL1", MissionType.SEIZE, "ALPHA")])
    done = lang.format_done("SL1", "TL1", MissionType.SEIZE, "ALPHA")
    # rejected claim: the mission stands
    p.observe(3, [m("done", "TL1", "SL1", done),
                  m("done_reject", "SL1", "TL1", lang.format_done_reject("TL1", "SL1"))])
    assert p.predict("TL1")[0] == obj_class("ALPHA")
    # confirmed claim: mission cleared, standing by for new orders
    confirm = lang.format_done_confirm("TL1", "SL1", MissionType.SEIZE, "ALPHA")
    p.observe(9, [m("done", "TL1", "SL1", done), m("done_confirm", "SL1", "TL1", confirm)])
    assert p.predict("TL1") == (HOLD, STATIC)


def test_support_follows_the_supported_unit():
    p = NetPredictor(make_briefing("squad"))
    p.observe(0, [opord("SL1", MissionType.SEIZE, "ALPHA")])
    p.observe(1, [order("SL1", "TL1", MissionType.SEIZE, "ALPHA"),
                  order("SL1", "TL2", MissionType.SUPPORT, "TL1")])
    # "pas un pas sans appui": destination and posture mirror the supported unit
    assert p.predict("TL2") == (obj_class("ALPHA"), MOVING)
    # the supported unit is re-tasked: the umbrella moves with it
    p.observe(2, [order("SL1", "TL1", MissionType.CLEAR, "BRAVO")])
    assert p.predict("TL2")[0] == obj_class("BRAVO")
    # the supported unit falls: SUPPORT ENDED, standing by
    p.observe(3, [m("casualty", "HQ", "ALL", lang.format_casualty("TL1")),
                  m("support_end", "TL2", "SL1", lang.format_support_end("SL1", "TL2", "TL1"))])
    assert p.predict("TL2") == (HOLD, STATIC)


def test_succession_inherits_mission_and_rewires_org():
    p = NetPredictor(make_briefing("squad"))
    p.observe(0, [opord("SL1", MissionType.SEIZE, "ALPHA")])
    p.observe(1, [order("SL1", "TL1", MissionType.OBSERVE, "BRAVO")])
    p.observe(2, [
        m("casualty", "HQ", "ALL", lang.format_casualty("SL1")),
        m("taking_command", "TL1", "ALL", lang.format_taking_command("TL1", "SL1")),
        m("taking_command", "RFN1", "ALL", lang.format_assuming_position("RFN1", "TL1")),
    ])
    assert not p.alive["SL1"]
    assert p.predict("TL1")[0] == obj_class("ALPHA")   # mission continuity up the chain
    assert p.predict("RFN1")[0] == obj_class("BRAVO")  # the fill inherits the vacated task
    assert p.leader["TL2"] == "TL1"                    # squad rewired under the successor
    assert p.leader["RFN1"] == "TL1"
    assert p.leader["RFN2"] == "RFN1"


def test_contact_predicts_firing_and_ages_out_screen_stays_tight():
    p = NetPredictor(make_briefing("squad"))
    p.observe(0, [order("SL1", "TL1", MissionType.DEFEND, "ALPHA"),
                  order("SL1", "TL2", MissionType.SCREEN, "BRAVO")])
    p.observe(1, [m("sitrep", "TL1", "SL1", lang.format_sitrep("SL1", "TL1", 100, 30, (33, 33))),
                  m("sitrep", "TL2", "SL1", lang.format_sitrep("SL1", "TL2", 100, 30, (35, 9)))])
    assert p.predict("TL1")[1] == STATIC  # in position, net quiet
    p.observe(2, [m("contact", "RFN1", "SL1", lang.format_contact("SL1", "RFN1", 2, (33, 30)))])
    assert p.predict("TL1")[1] == FIRING  # fresh contact 3 cells from its station
    p.observe(3, [m("contact", "RFN3", "SL1", lang.format_contact("SL1", "RFN3", 1, (35, 12)))])
    assert p.predict("TL2")[1] == STATIC  # SCREEN is weapons tight: never predicted firing
    p.observe(13, [])  # the defender's contact aged past CONTACT_FRESH
    assert p.predict("TL1")[1] == STATIC


def test_team_opord_root_commands_from_cover():
    # #9: a root OPORD RECON/SCREEN is team-adjudicated — doctrine says the
    # commander observes through the squad, not from the ring.
    p = NetPredictor(make_briefing("squad_screen"))
    p.observe(0, [opord("SL1", MissionType.SCREEN, "BRAVO")])
    assert p.predict("SL1") == (HOLD, STATIC)
    p.observe(1, [order("SL1", "TL1", MissionType.SCREEN, "BRAVO")])
    assert p.predict("TL1") == (obj_class("BRAVO"), MOVING)  # a subordinate's is personal


# ---------------------------------------------------------------------- #
# ground truth on hand-built traces
# ---------------------------------------------------------------------- #

OBJS = {"ALPHA": (30, 30), "BRAVO": (30, 5)}


def test_destination_truth_objective_beats_hold_in_region():
    # parked 1 cell off ALPHA: the objective region claims it, not HOLD
    steps = [gt_step(t, [rec("A", (30, 29))]) for t in range(6)]
    idx = step_index(steps)
    assert destination_truth(idx, 0, "A", 5, OBJS) == obj_class("ALPHA")


def test_destination_truth_hold_when_far_from_everything():
    steps = [gt_step(t, [rec("A", (5, 5))]) for t in range(6)]
    idx = step_index(steps)
    assert destination_truth(idx, 0, "A", 5, OBJS) == HOLD


def test_destination_truth_leader_class():
    # trailing 2 cells behind a moving leader, far from both objectives
    steps = [
        gt_step(t, [rec("L", (12 + t, 15)), rec("A", (10 + t, 15), leader="L")])
        for t in range(6)
    ]
    idx = step_index(steps)
    assert destination_truth(idx, 0, "A", 5, OBJS) == LEADER


def test_destination_truth_transit_classes_by_nearest_anchor():
    # marching from (5,5) toward BRAVO: far from every region, nearest wins
    steps = [gt_step(t, [rec("A", (5 + t, 5))]) for t in range(16)]
    idx = step_index(steps)
    assert destination_truth(idx, 0, "A", 15, OBJS) == obj_class("BRAVO")


def test_destination_truth_window_ends():
    steps = [
        gt_step(0, [rec("A", (5, 5))]),
        gt_step(1, [rec("A", (5, 5), alive=False)]),
    ]
    idx = step_index(steps)
    assert destination_truth(idx, 0, "A", 5, OBJS) is None  # dead next step
    assert destination_truth(idx, 1, "A", 5, OBJS) is None  # episode over


def test_posture_truth_classes():
    static = [gt_step(t, [rec("A", (5, 5))]) for t in range(6)]
    assert posture_truth(step_index(static), 0, "A", 5) == STATIC
    moving = [gt_step(t, [rec("A", (5 + t, 5))]) for t in range(6)]
    assert posture_truth(step_index(moving), 0, "A", 5) == MOVING
    fired = [gt_step(t, [rec("A", (5, 5), fired=(t == 3))]) for t in range(6)]
    assert posture_truth(step_index(fired), 0, "A", 5) == FIRING


def test_posture_truth_move_fraction_threshold():
    # 1 move in 5 steps (0.2 < 1/3) -> STATIC; 2 in 5 (0.4 >= 1/3) -> MOVING
    one = [(5, 5), (6, 5), (6, 5), (6, 5), (6, 5), (6, 5)]
    two = [(5, 5), (6, 5), (7, 5), (7, 5), (7, 5), (7, 5)]
    for positions, expected in ((one, STATIC), (two, MOVING)):
        steps = [gt_step(t, [rec("A", positions[t])]) for t in range(6)]
        assert posture_truth(step_index(steps), 0, "A", 5) == expected


# ---------------------------------------------------------------------- #
# scoring end-to-end
# ---------------------------------------------------------------------- #


def test_probe_episode_synthetic_known_score():
    """A 2-agent episode where every pair's truth and prediction is known.

    TL1 is OPORD'd to SEIZE ALPHA at t=0 and walks there (arrives t=10);
    RFN1 is never tasked and never moves. 22 pairs. Destination is perfect;
    posture misses exactly the i=8 and i=9 windows (the predictor's transit
    estimate says arrived, the tail of the walk is still MOVING).
    """
    brief = Briefing("mini", {"ALPHA": (10, 0)}, (0, 0), {"TL1": None, "RFN1": "TL1"})
    steps = [
        gt_step(
            t,
            [rec("TL1", (min(t, 10), 0)), rec("RFN1", (0, 1))],
            messages=[opord("TL1", MissionType.SEIZE, "ALPHA")] if t == 0 else [],
        )
        for t in range(12)
    ]
    ep = probe_episode({"scenario": "mini", "steps": steps}, brief, k=5)
    assert ep["pairs"] == 22
    agg = aggregate_probe([ep], brief.dest_classes)
    dest, post = agg["destination"], agg["posture"]
    assert dest["accuracy"] == 1.0
    assert dest["support"] == {obj_class("ALPHA"): 11, LEADER: 0, HOLD: 11}
    assert dest["baseline_majority"] == 0.5
    assert dest["baseline_random"] == 1 / 3
    assert dest["gap_vs_majority"] == 0.5
    assert post["accuracy"] == 20 / 22
    assert post["confusion"][MOVING][STATIC] == 2  # the two late-transit misses
    table = format_probe_table(agg)
    assert "destination" in table and "posture" in table and "gap" in table


def test_aggregate_baseline_math():
    ep = {
        "pairs": 10,
        "destination": {"OBJ ALPHA": {"OBJ ALPHA": 6, "HOLD": 2}, "HOLD": {"HOLD": 2}},
        "posture": {"STATIC": {"STATIC": 5, "MOVING": 1}, "MOVING": {"MOVING": 4}},
    }
    agg = aggregate_probe([ep, ep], ["OBJ ALPHA", "LEADER", "HOLD"])
    d = agg["destination"]
    assert d["pairs"] == 20
    assert d["accuracy"] == 16 / 20
    assert d["baseline_majority"] == 16 / 20  # truth is 80% OBJ ALPHA
    assert d["baseline_random"] == 1 / 3
    assert abs(d["gap_vs_majority"]) < 1e-12
    assert d["per_class_accuracy"] == {"OBJ ALPHA": 12 / 16, "LEADER": None, "HOLD": 1.0}
    p = agg["posture"]
    assert p["accuracy"] == 18 / 20
    assert p["baseline_majority"] == 12 / 20
    assert p["baseline_random"] == 1 / 3


def test_probe_on_recorded_episode_structure_and_determinism():
    from cohort.env.cohort_env import make_env
    from cohort.metrics import TraceRecorder
    from cohort.training.evaluate import run_episode

    brief = make_briefing("fireteam")

    def one():
        env = make_env("fireteam")
        recorder = TraceRecorder()
        run_episode(env, None, seed=31, rng=np.random.default_rng(31), recorder=recorder)
        return recorder.trace, probe_episode(recorder.trace, brief, k=K)

    trace_a, ep_a = one()
    _trace_b, ep_b = one()
    assert ep_a == ep_b, "same seed -> identical probe result"
    assert ep_a["pairs"] > 0
    # the recorder carries the probe's ground-truth fields
    soldier = trace_a["steps"][1]["soldiers"][0]
    assert "fired" in soldier and "leader" in soldier
    agg = aggregate_probe([ep_a], brief.dest_classes)
    assert 0.0 <= agg["destination"]["accuracy"] <= 1.0
    assert 0.0 <= agg["posture"]["accuracy"] <= 1.0
    json.dumps(agg)  # the whole summary is JSON-serializable
