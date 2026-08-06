"""Behavioral metrics suite (B2): pure metric math + trace recording.

The metric functions are tested against constructed mini-episodes where the
value of every metric is known by hand; the recorder is tested against the
real environment for structure, determinism, and the guarantee that
recording never perturbs a seeded episode.
"""

import json

import numpy as np

from cohort.env.cohort_env import make_env
from cohort.metrics import (
    TraceRecorder,
    aggregate_behavior,
    episode_behavior,
    format_behavior_table,
    format_gate_report,
    format_order_task_mix,
    regression_gates,
)
from cohort.training.evaluate import evaluate, run_episode

# ---------------------------------------------------------------------- #
# constructed-trace helpers
# ---------------------------------------------------------------------- #


def sold(
    cs, *, alive=True, pos=(0, 0), mission=None, since=None, auth=0, subs=(), comp=None, sees=(),
    cover=False,
):
    return {
        "cs": cs,
        "alive": alive,
        "pos": list(pos),
        "mission": mission,
        "since": since,
        "auth": auth,
        "subs": list(subs),
        "comp": comp,
        "sees": list(sees),
        "cover": cover,
    }


def enemy(eid, *, alive=True, pos=(30, 30)):
    return {"id": eid, "alive": alive, "pos": list(pos)}


def msg(kind, frm, to, mission=None, text=""):
    return {"kind": kind, "from": frm, "to": to, "mission": mission, "text": text}


def step(t, soldiers, enemies=(), messages=()):
    return {"t": t, "soldiers": soldiers, "enemies": list(enemies), "messages": list(messages)}


def trace(
    steps, *, human=None, root_objective=None, reported=None, outcome="success", refresh_age=20,
    ttl=40, root_mission="SEIZE", threat_radius=8.0,
):
    return {
        "scenario": "test",
        "outcome": outcome,
        "length": steps[-1]["t"],
        "root_mission": root_mission,
        "root_objective": list(root_objective) if root_objective else None,
        "ring_radius": 7.0,
        "threat_radius": threat_radius,
        "contact_refresh_age": refresh_age,
        "knowledge_ttl": ttl,
        "human": human,
        "reported": reported or {},
        "steps": steps,
    }


# ---------------------------------------------------------------------- #
# obedience latency
# ---------------------------------------------------------------------- #


def test_obedience_latency_by_hand():
    # RFN1 ordered at t=2, first positive compliance at t=5 -> latency 3.
    # TL1 ordered at t=2 already in position (comp>0 immediately) -> latency 0.
    steps = [
        step(0, [sold("TL1"), sold("RFN1")]),
        step(1, [sold("TL1"), sold("RFN1")]),
        step(2, [sold("TL1", mission="HOLD", since=2, comp=0.5), sold("RFN1", mission="HOLD", since=2, comp=0.0)]),
        step(3, [sold("TL1", mission="HOLD", since=2, comp=0.5), sold("RFN1", mission="HOLD", since=2, comp=0.0)]),
        step(4, [sold("TL1", mission="HOLD", since=2, comp=0.5), sold("RFN1", mission="HOLD", since=2, comp=-0.2)]),
        step(5, [sold("TL1", mission="HOLD", since=2, comp=0.5), sold("RFN1", mission="HOLD", since=2, comp=0.4)]),
    ]
    ep = episode_behavior(trace(steps))
    assert sorted(ep["obedience_latencies"]) == [0, 3]
    assert ep["obedience_censored"] == 0


def test_obedience_opord_counts_from_t0():
    steps = [
        step(0, [sold("TL1", mission="SEIZE", since=0)]),
        step(1, [sold("TL1", mission="SEIZE", since=0, comp=0.0)]),
        step(2, [sold("TL1", mission="SEIZE", since=0, comp=0.3)]),
    ]
    ep = episode_behavior(trace(steps))
    assert ep["obedience_latencies"] == [2]


def test_obedience_censored_on_retask_and_death():
    # RFN1's first order is superseded at t=3 before compliance; the second
    # is never complied with before the episode ends -> both censored.
    # RFN2 dies at t=3 without complying -> censored.
    steps = [
        step(0, [sold("RFN1"), sold("RFN2")]),
        step(1, [sold("RFN1", mission="HOLD", since=1, comp=0.0), sold("RFN2", mission="HOLD", since=1, comp=0.0)]),
        step(2, [sold("RFN1", mission="HOLD", since=1, comp=0.0), sold("RFN2", mission="HOLD", since=1, comp=0.0)]),
        step(3, [sold("RFN1", mission="OBSERVE", since=3, comp=0.0), sold("RFN2", alive=False)]),
        step(4, [sold("RFN1", mission="OBSERVE", since=3, comp=0.0), sold("RFN2", alive=False)]),
    ]
    ep = episode_behavior(trace(steps))
    assert ep["obedience_latencies"] == []
    assert ep["obedience_censored"] == 3


# ---------------------------------------------------------------------- #
# report precision / recall
# ---------------------------------------------------------------------- #


def test_report_precision_recall_by_hand():
    # Enemy 0 reported new (t=2), re-reported fresh (t=3, redundant), then
    # re-reported at age >= refresh_age (t=6, refresh). Enemy 1 is seen but
    # never reported. -> precision 2/3, recall 1/2.
    enemies = [enemy(0), enemy(1)]

    def rec(sees):
        return [sold("RFN1", sees=sees), sold("RFN2", sees=[1])]

    steps = [
        step(0, rec([]), enemies),
        step(1, rec([0]), enemies),
        step(2, rec([0]), enemies, [msg("contact", "RFN1", "TL1")]),
        step(3, rec([0]), enemies, [msg("contact", "RFN1", "TL1")]),
        step(4, rec([0]), enemies),
        step(5, rec([0]), enemies),
        step(6, rec([0]), enemies, [msg("contact", "RFN1", "TL1")]),
    ]
    ep = episode_behavior(trace(steps, reported={"RFN1": [0]}, refresh_age=3, ttl=10))
    assert ep["contacts"] == 3
    assert ep["contacts_informative"] == 2
    assert ep["enemies_seen"] == 2
    assert ep["enemies_reported"] == 1


def test_report_picture_expires_after_ttl():
    # After knowledge_ttl steps without a refresh the entry expires, so a
    # later report of the same enemy is new intel again.
    enemies = [enemy(0)]
    steps = [step(t, [sold("RFN1", sees=[0])], enemies) for t in range(10)]
    steps[2] = step(2, [sold("RFN1", sees=[0])], enemies, [msg("contact", "RFN1", "TL1")])
    steps[9] = step(9, [sold("RFN1", sees=[0])], enemies, [msg("contact", "RFN1", "TL1")])
    ep = episode_behavior(trace(steps, reported={"RFN1": [0]}, refresh_age=2, ttl=4))
    assert ep["contacts"] == 2
    assert ep["contacts_informative"] == 2  # the picture expired in between


def test_report_edges_no_contacts_no_enemies():
    ep = episode_behavior(trace([step(0, [sold("TL1")]), step(1, [sold("TL1")])]))
    assert ep["contacts"] == 0 and ep["enemies_seen"] == 0
    agg = aggregate_behavior([ep])
    assert agg["report_precision"] is None
    assert agg["report_recall"] is None


# ---------------------------------------------------------------------- #
# doctrine preference
# ---------------------------------------------------------------------- #


def test_doctrine_preference_within_step_ordering():
    # SL1 (SEIZE) orders SEIZE (preferred) at t=1; at t=2 SL1 orders OBSERVE
    # (allowed, not preferred) to TL1 and TL1 — under its NEW mission, applied
    # earlier in the same step — orders OBSERVE to RFN1 (preferred for an
    # OBSERVE holder). The HQ OPORD is never counted.
    steps = [
        step(0, [sold("SL1", mission="SEIZE", since=0, auth=2, subs=["TL1"]), sold("TL1", subs=["RFN1"]), sold("RFN1")],
             messages=[msg("opord", "HQ", "SL1", "SEIZE")]),
        step(1, [sold("SL1", mission="SEIZE", since=0, auth=2, subs=["TL1"]),
                 sold("TL1", mission="SEIZE", since=1, subs=["RFN1"]), sold("RFN1")],
             messages=[msg("order", "SL1", "TL1", "SEIZE")]),
        step(2, [sold("SL1", mission="SEIZE", since=0, auth=2, subs=["TL1"]),
                 sold("TL1", mission="OBSERVE", since=2, subs=["RFN1"]),
                 sold("RFN1", mission="OBSERVE", since=2)],
             messages=[msg("order", "SL1", "TL1", "OBSERVE"), msg("order", "TL1", "RFN1", "OBSERVE")]),
    ]
    ep = episode_behavior(trace(steps))
    assert ep["orders_issued"] == 3
    assert ep["orders_preferred"] == 2
    assert aggregate_behavior([ep])["doctrine_preference_rate"] == 2 / 3


def _defend_leader_orders(task):
    """One agent-issued order of ``task`` from a DEFEND-holding TL1."""
    return trace([
        step(0, [sold("TL1", mission="DEFEND", since=0, auth=1, subs=["RFN1"]), sold("RFN1")]),
        step(1, [sold("TL1", mission="DEFEND", since=0, auth=1, subs=["RFN1"]),
                 sold("RFN1", mission=task, since=1)],
             messages=[msg("order", "TL1", "RFN1", task)]),
    ], root_mission="DEFEND")


def test_doctrine_tiers_separate_catalog_adoption_from_violation():
    """refs #14: preference is `allowed[0]`, so ADVANCE under DEFEND — a legal
    maneuver leg A5 put in `DOCTRINE[DEFEND]` — scores exactly like RALLY,
    which is not derivable from DEFEND at all. Both read 0.0 preference; only
    the tier split says one is contained doctrine and the other is a breach.
    (The order mask makes the RALLY case unreachable in play, but the B3
    `nomask` arm and injected orders both produce it.)
    """
    adopted = aggregate_behavior([episode_behavior(_defend_leader_orders("ADVANCE"))])
    breached = aggregate_behavior([episode_behavior(_defend_leader_orders("RALLY"))])

    # the rate the record has always reported cannot tell them apart
    assert adopted["doctrine_preference_rate"] == 0.0
    assert breached["doctrine_preference_rate"] == 0.0

    assert adopted["doctrine_allowed_rate"] == 1.0
    assert adopted["orders_allowed"] == 1 and adopted["orders_violating"] == 0
    assert adopted["orders_by_task"]["ADVANCE"]["allowed"] == 1

    assert breached["doctrine_allowed_rate"] == 0.0
    assert breached["orders_allowed"] == 0 and breached["orders_violating"] == 1
    assert breached["orders_by_task"]["RALLY"]["violating"] == 1


def test_order_task_mix_attributes_a_low_preference_rate():
    """The share/preference pair per ordered task: three ADVANCE orders (all
    merely allowed) and one DEFEND (preferred) is 0.25 preference overall, but
    the mix shows it as ADVANCE adoption rather than degraded command."""
    eps = [episode_behavior(_defend_leader_orders(t)) for t in ("ADVANCE",) * 3 + ("DEFEND",)]
    agg = aggregate_behavior(eps)
    assert agg["doctrine_preference_rate"] == 0.25
    assert agg["doctrine_allowed_rate"] == 1.0
    assert format_order_task_mix(agg) == "ADVANCE 0.75/0.00, DEFEND 0.25/1.00"
    assert "ADVANCE 0.75/0.00" in format_behavior_table(agg)


def test_doctrine_underivable_issuer_is_not_counted_as_violating():
    """An issuer with no mission has nothing to derive from: it stays in the
    preference denominator (unchanged semantics) but must not be booked as a
    doctrine breach, which would make containment unreadable."""
    steps = [
        step(0, [sold("TL1", auth=1, subs=["RFN1"]), sold("RFN1")]),
        step(1, [sold("TL1", auth=1, subs=["RFN1"]), sold("RFN1", mission="HOLD", since=1)],
             messages=[msg("order", "TL1", "RFN1", "HOLD")]),
    ]
    agg = aggregate_behavior([episode_behavior(trace(steps))])
    assert agg["orders_issued"] == 1
    assert agg["doctrine_preference_rate"] == 0.0
    assert agg["orders_violating"] == 0 and agg["orders_underivable"] == 1
    assert agg["doctrine_allowed_rate"] is None


def test_doctrine_no_orders_is_none():
    ep = episode_behavior(trace([step(0, [sold("TL1")]), step(1, [sold("TL1")])]))
    assert ep["orders_issued"] == 0
    assert aggregate_behavior([ep])["doctrine_preference_rate"] is None


# ---------------------------------------------------------------------- #
# false COMPLETE
# ---------------------------------------------------------------------- #


def test_false_complete_rate():
    steps = [
        step(0, [sold("TL1")]),
        step(1, [sold("TL1")], messages=[msg("done", "TL1", "HQ"), msg("done_reject", "HQ", "TL1")]),
        step(2, [sold("TL1")], messages=[msg("done", "TL1", "HQ"), msg("done_confirm", "HQ", "TL1")]),
    ]
    ep = episode_behavior(trace(steps))
    assert ep["done_reports"] == 2 and ep["done_rejected"] == 1
    assert aggregate_behavior([ep])["false_complete_rate"] == 0.5


# ---------------------------------------------------------------------- #
# succession recovery
# ---------------------------------------------------------------------- #


def _succession_steps(retask_step=None):
    tl_alive = [sold("TL1", auth=1, mission="SEIZE", since=0, subs=["RFN1", "RFN2"], comp=0.1)]
    riflemen = [sold("RFN1", mission="HOLD", since=1, comp=0.1), sold("RFN2", mission="HOLD", since=1, comp=0.1)]
    steps = [
        step(0, [sold("TL1", auth=1, mission="SEIZE", since=0, subs=["RFN1", "RFN2"]), sold("RFN1"), sold("RFN2")]),
        step(1, tl_alive + riflemen),
        step(2, tl_alive + riflemen),
        step(3, tl_alive + riflemen),
    ]
    down = "ALL STATIONS, THIS IS RFN1: TL1 IS DOWN. I AM ASSUMING COMMAND. OUT."
    for t in range(4, 9):
        rfn2_mission = ("OBSERVE", retask_step) if retask_step is not None and t >= retask_step else ("HOLD", 1)
        steps.append(
            step(
                t,
                [
                    sold("TL1", alive=False),
                    sold("RFN1", auth=1, mission="SEIZE", since=0, subs=["RFN2"], comp=0.1),
                    sold("RFN2", mission=rfn2_mission[0], since=rfn2_mission[1], comp=0.1),
                ],
                messages=[msg("taking_command", "RFN1", "ALL", text=down)] if t == 4 else [],
            )
        )
    return steps


def test_succession_recovery_by_hand():
    ep = episode_behavior(trace(_succession_steps(retask_step=7)))
    assert ep["succession_events"] == 1
    assert ep["succession_recovery"] == [3]  # died t=4, RFN2 re-tasked t=7
    assert ep["succession_unrecovered"] == 0


def test_succession_unrecovered_is_censored():
    ep = episode_behavior(trace(_succession_steps(retask_step=None)))
    assert ep["succession_events"] == 1
    assert ep["succession_recovery"] == []
    assert ep["succession_unrecovered"] == 1


def test_death_without_subordinates_is_no_event():
    steps = [
        step(0, [sold("TL1", auth=1, mission="SEIZE", since=0, subs=[]), sold("RFN1")]),
        step(1, [sold("TL1", auth=1, mission="SEIZE", since=0, subs=[], comp=0.1), sold("RFN1", alive=False)]),
    ]
    ep = episode_behavior(trace(steps))
    assert ep["succession_events"] == 0
    assert aggregate_behavior([ep])["succession_recovery_mean"] is None


# ---------------------------------------------------------------------- #
# subordinate coverage
# ---------------------------------------------------------------------- #


def test_coverage_time_by_hand():
    steps = [
        step(0, [sold("TL1", auth=1, mission="SEIZE", since=0, subs=["RFN1", "RFN2"]), sold("RFN1"), sold("RFN2")]),
        # one subordinate untasked -> gap
        step(1, [sold("TL1", auth=1, mission="SEIZE", since=0, subs=["RFN1", "RFN2"], comp=0.1),
                 sold("RFN1", mission="HOLD", since=1, comp=0.1), sold("RFN2")]),
        # everyone tasked -> covered
        step(2, [sold("TL1", auth=1, mission="SEIZE", since=0, subs=["RFN1", "RFN2"], comp=0.1),
                 sold("RFN1", mission="HOLD", since=1, comp=0.1), sold("RFN2", mission="HOLD", since=2, comp=0.1)]),
    ]
    ep = episode_behavior(trace(steps))
    assert (ep["coverage_pairs"], ep["coverage_covered"]) == (2, 1)
    assert aggregate_behavior([ep])["coverage_time"] == 0.5


# ---------------------------------------------------------------------- #
# human exposure
# ---------------------------------------------------------------------- #


def test_human_exposure_by_hand():
    # Objective at (10,10), ring radius 7. The human spawns inside the ring
    # (entry #1), leaves, and re-enters (entry #2). Enemy distance is only
    # averaged over steps with a living enemy.
    e = [enemy(0, pos=(10, 14))]
    steps = [
        step(0, [sold("TL1", pos=(10, 10))], e),                      # dist 4, inside
        step(1, [sold("TL1", pos=(20, 10))], e),                      # dist 10.77, outside
        step(2, [sold("TL1", pos=(10, 12))], [enemy(0, alive=False, pos=(10, 14))]),  # inside again, enemy dead
    ]
    ep = episode_behavior(trace(steps, human="TL1", root_objective=(10, 10)))
    assert ep["human_died"] is False
    assert ep["human_ring_entries"] == 2
    assert abs(ep["human_mean_enemy_dist"] - (4 + np.hypot(10, 4)) / 2) < 1e-9
    assert ep["human_mean_objective_dist"] is not None


def test_human_death_and_no_human_edges():
    e = [enemy(0)]
    died = [
        step(0, [sold("TL1", pos=(0, 0))], e),
        step(1, [sold("TL1", alive=False)], e),
    ]
    ep = episode_behavior(trace(died, human="TL1", root_objective=(10, 10)))
    assert ep["human_died"] is True
    ep2 = episode_behavior(trace([step(0, [sold("TL1")]), step(1, [sold("TL1")])]))
    assert ep2["human_died"] is None and ep2["human_ring_entries"] is None
    agg = aggregate_behavior([ep2])
    assert agg["human_death_rate"] is None


# ---------------------------------------------------------------------- #
# fight disposition + positional regression gate (issue #11)
# ---------------------------------------------------------------------- #


def test_fight_disposition_scores_only_threatened_pairs():
    # Threat radius 8. RFN1 sits in cover on the objective the whole time;
    # RFN2 stands in the open 10 cells out. An enemy is only within reach on
    # steps 1 and 2, and only of RFN1 — so RFN2's open ground never enters
    # the numbers, and neither does the unthreatened step 0.
    obj = (10, 10)
    far, near = enemy(0, pos=(30, 30)), enemy(0, pos=(14, 10))
    steps = [
        step(0, [sold("RFN1", pos=(10, 10), cover=True), sold("RFN2", pos=(25, 10))], [far]),
        step(1, [sold("RFN1", pos=(10, 10), cover=True), sold("RFN2", pos=(25, 10))], [near]),
        step(2, [sold("RFN1", pos=(12, 10), cover=False), sold("RFN2", pos=(25, 10))], [near]),
    ]
    ep = episode_behavior(trace(steps, root_objective=obj))
    assert ep["threat_pairs"] == 2                       # RFN1 at t=1 and t=2
    assert ep["threat_cover_pairs"] == 1                 # only t=1 was on cover
    assert ep["threat_objective_dist_pairs"] == 2
    assert ep["threat_objective_dist_sum"] == 2.0        # 0 + 2 cells
    agg = aggregate_behavior([ep])
    assert agg["cover_occupancy_under_threat"] == 0.5
    assert agg["mean_distance_from_objective_under_threat"] == 1.0


def test_fight_disposition_is_none_without_threat():
    # Enemies exist but never close; nothing is measured rather than 0.0 —
    # "no firefight" must not read as "fought in the open".
    steps = [step(t, [sold("RFN1", pos=(0, 0))], [enemy(0, pos=(30, 30))]) for t in range(3)]
    agg = aggregate_behavior([episode_behavior(trace(steps, root_objective=(10, 10)))])
    assert agg["cover_occupancy_under_threat"] is None
    assert agg["mean_distance_from_objective_under_threat"] is None
    assert agg["threat_pairs"] == 0


def _defend_agg(*, cover, dist_from_obj):
    """Aggregate with a DEFEND root at a chosen disposition (via one pair)."""
    obj = (10, 10)
    here = sold("RFN1", pos=(10 + dist_from_obj, 10), cover=cover)
    assault = [enemy(0, pos=(10 + dist_from_obj, 10))]  # on top of the soldier: always in reach
    steps = [step(0, [here], assault), step(1, [here], assault)]
    ep = episode_behavior(trace(steps, root_objective=obj, root_mission="DEFEND"))
    return aggregate_behavior([ep])


def test_positional_gate_fails_the_v7_disposition():
    # fireteam_defend_v7: cover 0.060 at 9.09 cells — both bounds broken.
    gates = regression_gates(_defend_agg(cover=False, dist_from_obj=9))
    assert [g["name"] for g in gates] == [
        "cover_occupancy_under_threat",
        "mean_distance_from_objective_under_threat",
    ]
    assert [g["passed"] for g in gates] == [False, False]
    report = format_gate_report(gates)
    assert "FAIL" in report and "PASS" not in report


def test_positional_gate_passes_a_prepared_defense():
    # fireteam_defend_v5 / defend_brique_v1 shape: on cover, on the position.
    gates = regression_gates(_defend_agg(cover=True, dist_from_obj=2))
    assert all(g["passed"] for g in gates)
    assert "PASS" in format_gate_report(gates)


def test_positional_gate_applies_to_defend_roots_only():
    # The same disposition under a SEIZE root gates on nothing: an assault is
    # supposed to leave its start point and cross open ground.
    seize = _defend_agg(cover=False, dist_from_obj=9)
    seize["root_mission"] = "SEIZE"
    assert regression_gates(seize) == []
    assert format_gate_report([]) == ""


def test_unmeasured_gate_is_not_a_pass():
    steps = [step(t, [sold("RFN1", pos=(0, 0))], [enemy(0, pos=(30, 30))]) for t in range(3)]
    agg = aggregate_behavior(
        [episode_behavior(trace(steps, root_objective=(10, 10), root_mission="DEFEND"))]
    )
    gates = regression_gates(agg)
    assert gates and all(g["passed"] is None for g in gates)
    assert "FAIL" not in format_gate_report(gates)


def test_recorder_records_cover_and_threat_radius():
    env = make_env("fireteam_defend")
    rec = TraceRecorder()
    run_episode(env, None, seed=7, rng=np.random.default_rng(7), recorder=rec)
    assert rec.trace["threat_radius"] == env.combat.weapon_range
    assert rec.trace["root_mission"] == "DEFEND"
    assert all("cover" in s for st in rec.trace["steps"] for s in st["soldiers"])
    ep = episode_behavior(rec.trace)
    agg = aggregate_behavior([ep])
    # a defend episode fought on a real map produces a measurable disposition
    # and a well-formed gate verdict
    for g in regression_gates(agg):
        assert g["passed"] in (True, False, None)
        assert g["bound"] > 0


# ---------------------------------------------------------------------- #
# aggregation + table
# ---------------------------------------------------------------------- #


def test_aggregate_pools_events_across_episodes():
    a = episode_behavior(trace(_succession_steps(retask_step=7)))
    b = episode_behavior(trace(_succession_steps(retask_step=5), outcome="timeout"))
    agg = aggregate_behavior([a, b])
    assert agg["episodes"] == 2
    assert agg["success_rate"] == 0.5
    assert agg["succession_events"] == 2
    assert agg["succession_recovery_mean"] == 2.0  # (3 + 1) / 2
    table = format_behavior_table(agg)
    assert "succession recovery" in table and "obedience latency" in table
    assert "—" in table  # undefined metrics render as em dash, never 0


# ---------------------------------------------------------------------- #
# recorder integration (real environment)
# ---------------------------------------------------------------------- #


def test_recorder_does_not_perturb_the_episode():
    env = make_env("fireteam")
    plain = run_episode(env, None, seed=17, rng=np.random.default_rng(17))
    rec = TraceRecorder()
    recorded = run_episode(env, None, seed=17, rng=np.random.default_rng(17), recorder=rec)
    assert plain == recorded, "recording must not consume RNG or alter the episode"


def test_recorder_trace_structure_and_determinism():
    def record(seed):
        env = make_env("fireteam")
        rec = TraceRecorder()
        run_episode(env, None, seed=seed, rng=np.random.default_rng(seed), recorder=rec)
        return rec.trace

    a, b = record(23), record(23)
    assert json.dumps(a) == json.dumps(b), "same seed -> identical trace"
    assert a["scenario"] == "fireteam"
    assert a["human"] == "TL1"
    assert a["steps"][0]["t"] == 0
    assert a["outcome"] in ("success", "defeat", "timeout")
    assert a["length"] == len(a["steps"]) - 1
    # the OPORD is on the t=0 record and parsed into a mission
    opord = [m for m in a["steps"][0]["messages"] if m["kind"] == "opord"]
    assert opord and opord[0]["mission"] == "SEIZE"
    # every metric computes on a real trace and the result is serializable
    ep = episode_behavior(a)
    json.dumps(ep)
    agg = aggregate_behavior([ep])
    assert agg["episodes"] == 1
    assert agg["coverage_time"] is None or 0.0 <= agg["coverage_time"] <= 1.0


def test_evaluate_writes_behavior_json(tmp_path, capsys):
    out = tmp_path / "behavior.json"
    summary = evaluate(
        None, scenario="fireteam", episodes=1, seed=41, behavior=True, behavior_path=str(out)
    )
    assert "behavior" in summary
    payload = json.loads(out.read_text())
    assert payload["scenario"] == "fireteam"
    assert payload["episodes"] == 1
    assert set(payload["metrics"]) >= {
        "obedience_latency_mean",
        "report_precision",
        "report_recall",
        "doctrine_preference_rate",
        "false_complete_rate",
        "succession_recovery_mean",
        "coverage_time",
        "human_death_rate",
        "human_mean_enemy_dist",
        "human_ring_entries_mean",
    }
    assert len(payload["per_episode"]) == 1
    assert "behavior over 1 episodes" in capsys.readouterr().out


# ---------------------------------------------------------------------- #
# A5 vocabulary usage
# ---------------------------------------------------------------------- #


def test_vocabulary_usage_counts():
    """ADVANCE / timed / FORMATION orders, sync traffic, and stance share."""
    def msg(kind, text="", frm="SL1", to="TL1"):
        return {"kind": kind, "from": frm, "to": to, "mission": None, "text": text}

    s0 = {
        "t": 0,
        "soldiers": [sold("SL1", auth=2, subs=["TL1"]), sold("TL1", auth=1)],
        "enemies": [],
        "messages": [msg("opord", "SL1, THIS IS HQ: OPORD — SEIZE OBJ ALPHA. OUT.", "HQ", "SL1")],
    }
    s1 = {
        "t": 1,
        "soldiers": [
            {**sold("SL1", auth=2, subs=["TL1"]), "formation": None},
            {**sold("TL1", auth=1), "formation": "COLUMN", "leader": "SL1"},
            {**sold("RFN1"), "leader": "TL1"},
        ],
        "enemies": [],
        "messages": [
            msg("order", "TL1, THIS IS SL1: ADVANCE TO WP GOLD AT MY COMMAND. OUT."),
            msg("order", "TL1, THIS IS SL1: FORMATION COLUMN. OUT."),
            msg("execute", "ALL STATIONS, THIS IS SL1: EXECUTE. OUT."),
            msg("sync_propose", "RFN1, THIS IS TL1: PREPARE TO BOUND ON MY SIGNAL. OUT.", "TL1", "ALL"),
            msg("sync_go", "TL1: GO! OUT.", "TL1", "ALL"),
        ],
    }
    ep = episode_behavior(trace([s0, s1]))
    assert ep["advance_orders"] == 1
    assert ep["timed_orders"] == 1
    assert ep["formation_orders"] == 1
    assert ep["execute_signals"] == 1
    assert ep["sync_proposals"] == 1
    assert ep["sync_bounds"] == 1
    # stance share: step 1 has TL1 (own stance) + RFN1 (leader's stance)
    # governed out of 5 living agent-steps across both steps
    assert ep["stance_steps"] == 2
    assert ep["stance_agent_steps"] == 5
    agg = aggregate_behavior([ep])
    assert agg["advance_orders_per_episode"] == 1
    assert agg["stance_share"] == 2 / 5
    assert agg["sync_bounds_per_episode"] == 1
