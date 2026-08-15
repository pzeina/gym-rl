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
    ROOT_REPORT_CLOSE_FLOOR,
    SUCCESS_RATE_FLOOR,
    TraceRecorder,
    aggregate_behavior,
    episode_behavior,
    format_behavior_table,
    format_gate_report,
    format_order_task_mix,
    format_root_claim_shape,
    regression_gates,
    split_gates,
)
from cohort.training.evaluate import evaluate, run_episode

# ---------------------------------------------------------------------- #
# constructed-trace helpers
# ---------------------------------------------------------------------- #


def sold(
    cs, *, alive=True, pos=(0, 0), mission=None, since=None, auth=0, subs=(), comp=None, sees=(),
    cover=False, rank=None,
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
        "rank": rank,
    }


def enemy(eid, *, alive=True, pos=(30, 30)):
    return {"id": eid, "alive": alive, "pos": list(pos)}


def msg(kind, frm, to, mission=None, text=""):
    return {"kind": kind, "from": frm, "to": to, "mission": mission, "text": text}


def step(t, soldiers, enemies=(), messages=()):
    return {"t": t, "soldiers": soldiers, "enemies": list(enemies), "messages": list(messages)}


def trace(
    steps, *, human=None, root_objective=None, reported=None, outcome="success", refresh_age=20,
    ttl=40, root_mission="SEIZE", threat_radius=8.0, max_steps=375, root_close_step=None,
    sitrep_interval=25, sitrep_clock_start=None,
):
    return {
        "scenario": "test",
        "outcome": outcome,
        "length": steps[-1]["t"],
        "root_mission": root_mission,
        "max_steps": max_steps,
        "root_close_step": root_close_step,
        "sitrep_interval": sitrep_interval,
        "sitrep_clock_start": sitrep_clock_start,
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


def _claim_episode(claims, *, rejected=0, claimant="TL1", root=True):
    """One episode in which ``claimant`` files ``claims`` MISSION COMPLETEs."""
    who = {**sold(claimant), "root": root}
    steps = [step(0, [who])]
    for i in range(claims):
        answer = "done_reject" if i < rejected else "done_confirm"
        steps.append(
            step(i + 1, [who], messages=[msg("done", claimant, "HQ"), msg(answer, "HQ", claimant)])
        )
    return episode_behavior(trace(steps))


def test_the_same_rejection_rate_separates_a_report_from_a_flood():
    """refs #23: a ratio cannot tell one-per-episode from spam. This can."""
    # thirteen episodes, one claim each, every claim accepted
    disciplined = aggregate_behavior([_claim_episode(1) for _ in range(13)])
    # the same acceptance would be unreachable, so match the *ratio* instead:
    # both policies are rejected on half their claims, at 1 and at 8 per episode
    reporting = aggregate_behavior([_claim_episode(2, rejected=1) for _ in range(13)])
    spamming = aggregate_behavior([_claim_episode(8, rejected=4) for _ in range(13)])

    assert reporting["false_complete_rate"] == spamming["false_complete_rate"] == 0.5
    assert reporting["done_claims_per_claiming_episode"] == 2.0
    assert spamming["done_claims_per_claiming_episode"] == 8.0
    assert disciplined["done_claims_per_claiming_episode"] == 1.0
    assert disciplined["false_complete_rate"] == 0.0


def test_the_denominator_is_claiming_episodes_not_episodes():
    # silence in nine episodes out of ten must not read as restraint in the
    # tenth: the episodes that carried no claim are not part of the ratio
    quiet = [episode_behavior(trace([step(0, [sold("TL1")]), step(1, [sold("TL1")])]))] * 9
    agg = aggregate_behavior([*quiet, _claim_episode(4)])

    assert agg["episodes"] == 10 and agg["done_claim_episodes"] == 1
    assert agg["done_claims_per_claiming_episode"] == 4.0
    assert agg["done_claim_rate"] is None  # nothing was ever admissible here


def test_nobody_claiming_leaves_the_concentration_undefined():
    agg = aggregate_behavior(
        [episode_behavior(trace([step(0, [sold("TL1")]), step(1, [sold("TL1")])]))]
    )

    assert agg["done_claim_episodes"] == 0
    assert agg["done_claims_per_claiming_episode"] is None
    assert agg["false_complete_rate"] is None


def test_the_root_s_channel_is_counted_apart_from_its_subordinates():
    # the pooled rate is carried by RFN1's four rejected claims; the root filed
    # one and it was accepted. Reading the pooled number as the root's is the
    # confusion this split exists to stop.
    root = {**sold("TL1"), "root": True}
    rifleman = {**sold("RFN1"), "root": False}
    steps = [
        step(0, [root, rifleman]),
        step(1, [root, rifleman],
             messages=[msg("done", "TL1", "HQ"), msg("done_confirm", "HQ", "TL1")]),
    ]
    for i in range(4):
        steps.append(
            step(i + 2, [root, rifleman],
                 messages=[msg("done", "RFN1", "TL1"), msg("done_reject", "TL1", "RFN1")])
        )
    agg = aggregate_behavior([episode_behavior(trace(steps))])

    assert agg["done_reports"] == 5 and agg["done_rejected"] == 4
    assert agg["false_complete_rate"] == 0.8
    assert agg["done_reports_root"] == 1 and agg["done_rejected_root"] == 0
    assert agg["false_complete_rate_root"] == 0.0
    assert agg["done_claims_per_claiming_episode_root"] == 1.0


def test_a_successor_s_claim_counts_as_the_root_s():
    # TL1 holds the root, dies, and SGT2 assumes command: the claim that closes
    # the operation is whoever is root when it is made, not a fixed callsign
    before = [{**sold("TL1"), "root": True}, {**sold("SGT2"), "root": False}]
    after = [{**sold("TL1", alive=False), "root": False}, {**sold("SGT2"), "root": True}]
    steps = [
        step(0, before),
        step(1, before, messages=[msg("done", "TL1", "HQ"), msg("done_reject", "HQ", "TL1")]),
        step(2, after, messages=[msg("done", "SGT2", "HQ"), msg("done_confirm", "HQ", "SGT2")]),
    ]
    agg = aggregate_behavior([episode_behavior(trace(steps))])

    assert agg["done_reports_root"] == 2 and agg["done_rejected_root"] == 1
    assert agg["done_claims_per_claiming_episode_root"] == 2.0


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
    # `_defend_agg` traces default to outcome="success" (timeout_rate 0.0), so
    # the success-axis gate (issue #21) is also in scope and passes (1.0);
    # it is excluded from `positional` below alongside timeout_rate.
    gates = regression_gates(_defend_agg(cover=False, dist_from_obj=9))
    assert [g["name"] for g in gates] == [
        "timeout_rate",
        "success_rate",
        "closed_on_root_report_rate",
        "cover_occupancy_under_threat",
        "mean_distance_from_objective_under_threat",
    ]
    positional = [
        g
        for g in gates
        if g["name"] not in ("timeout_rate", "success_rate", "closed_on_root_report_rate")
    ]
    assert [g["passed"] for g in positional] == [False, False]
    assert "FAIL" in format_gate_report(positional)
    assert "PASS" not in format_gate_report(positional)


def test_positional_gate_passes_a_prepared_defense():
    # fireteam_defend_v5 / defend_brique_v1 shape: on cover, on the position.
    # These synthetic traces send no ENDEX, so the v1.20 command-report gate has
    # no denominator and reads None — unmeasured, which is deliberately not a
    # pass. Nothing may FAIL; that is the claim this test makes.
    gates = regression_gates(_defend_agg(cover=True, dist_from_obj=2))
    assert [g["passed"] for g in gates if g["name"] != "closed_on_root_report_rate"] == [
        True,
        True,
        True,
        True,
    ]
    assert not any(g["passed"] is False for g in gates)
    assert "PASS" in format_gate_report(gates)


def test_positional_gate_applies_to_defend_roots_only():
    # The same disposition under a SEIZE root gates on position at all: an
    # assault is supposed to leave its start point and cross open ground.
    # Only the universal gates remain: clock-expiry (issue #18) and the
    # success axis (issue #21), both root-mission-agnostic.
    seize = _defend_agg(cover=False, dist_from_obj=9)
    seize["root_mission"] = "SEIZE"
    assert [g["name"] for g in regression_gates(seize)] == [
        "timeout_rate",
        "success_rate",
        "closed_on_root_report_rate",
    ]
    assert format_gate_report([]) == ""


def test_unmeasured_gate_is_not_a_pass():
    steps = [step(t, [sold("RFN1", pos=(0, 0))], [enemy(0, pos=(30, 30))]) for t in range(3)]
    agg = aggregate_behavior(
        [episode_behavior(trace(steps, root_objective=(10, 10), root_mission="DEFEND"))]
    )
    # success_rate IS measured here (outcome defaults to "success"), so it is
    # excluded alongside timeout_rate — this test is about the positional
    # gates (issue #11), which go unmeasured with no threat pairs.
    positional = [g for g in regression_gates(agg) if g["name"] not in ("timeout_rate", "success_rate")]
    assert positional and all(g["passed"] is None for g in positional)
    assert "FAIL" not in format_gate_report(positional)


def test_orders_by_rank_separates_the_commander_from_its_team_leaders():
    """refs #52: `orders_by_task` is team-wide, so two policies issuing the same
    task mix are indistinguishable when one commands from the objective and the
    other from the rear. The mute-commander diagnosis stalled exactly there —
    only re-tasks were rank-resolved, and they said opposite things on the two
    mute seeds. Rank is the issuer's EFFECTIVE rank, so a promoted TL holding
    the squad counts as the SL it is acting as.
    """
    steps = [
        step(0, [sold("SL1", mission="SEIZE", since=0, auth=2, subs=["TL1"], rank="SL"),
                 sold("TL1", mission=None, auth=1, subs=["RFN1"], rank="TL"),
                 sold("RFN1", rank="RFN")]),
        step(1, [sold("SL1", mission="SEIZE", since=0, auth=2, subs=["TL1"], rank="SL"),
                 sold("TL1", mission="SEIZE", since=1, auth=1, subs=["RFN1"], rank="TL"),
                 sold("RFN1", rank="RFN")],
             messages=[msg("order", "SL1", "TL1", "SEIZE")]),
        step(2, [sold("SL1", mission="SEIZE", since=0, auth=2, subs=["TL1"], rank="SL"),
                 sold("TL1", mission="SEIZE", since=1, auth=1, subs=["RFN1"], rank="TL"),
                 sold("RFN1", mission="ADVANCE", since=2, rank="RFN")],
             messages=[msg("order", "TL1", "RFN1", "ADVANCE")]),
    ]
    agg = aggregate_behavior([episode_behavior(trace(steps))])

    by_rank = agg["orders_by_rank"]
    assert sum(by_rank["SL"].values()) == 1
    assert sum(by_rank["TL"].values()) == 1
    assert "RFN" not in by_rank  # issued nothing

    # the tier split survives the rank split: the SL's SEIZE is preferred off
    # its own SEIZE, the TL's ADVANCE is merely allowed
    assert by_rank["SL"]["preferred"] == 1
    assert by_rank["TL"]["allowed"] == 1

    # and the team-wide mix cannot make that distinction on its own
    assert sum(sum(b.values()) for b in agg["orders_by_task"].values()) == 2


def test_orders_by_rank_accounts_for_every_agent_issued_order():
    """The invariant that makes the metric trustworthy on a real map: no order
    is dropped or double-counted by the rank split. It also fails loudly if the
    recorder ever stops writing `rank`, which would otherwise show up as a
    silently empty dict beside a healthy `orders_issued`.
    """
    env = make_env("squad")
    rec = TraceRecorder()
    run_episode(env, None, seed=5, rng=np.random.default_rng(5), recorder=rec)
    assert all("rank" in s for st in rec.trace["steps"] for s in st["soldiers"])

    agg = aggregate_behavior([episode_behavior(rec.trace)])
    assert agg["orders_issued"] > 0
    assert sum(sum(b.values()) for b in agg["orders_by_rank"].values()) == agg["orders_issued"]


def test_order_pay_by_rank_accounts_for_every_order_and_agrees_with_the_rank_split():
    """refs #52: the fresh/churn/re-task split must cover exactly the orders
    `orders_by_rank` counts — every agent-issued mission order takes one of the
    three outcomes. A drift between the two would mean one of them is reading a
    channel the other is not, which is the failure that made `retasks_by_rank`
    misleading for this question in the first place.
    """
    env = make_env("squad")
    rec = TraceRecorder()
    run_episode(env, None, seed=5, rng=np.random.default_rng(5), recorder=rec)
    agg = aggregate_behavior([episode_behavior(rec.trace)])

    pay = agg["order_pay_by_rank"]
    assert pay, "a squad episode issues orders"

    # The identity is fresh + retask, NOT + churn: an identical reissue is
    # charged `order_churn` and returns without `_say`, so it never reaches the
    # transcript and no transcript-derived count can see it. Everything the
    # record says about order volume therefore UNDERCOUNTS a commander that
    # reissues, by exactly the quantity it is being charged for.
    on_the_net = sum(b["fresh"] + b["retask"] for b in pay.values())
    assert on_the_net == agg["orders_issued"]

    for rank, bucket in pay.items():
        assert bucket["fresh"] + bucket["retask"] == sum(agg["orders_by_rank"][rank].values())


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
# clock expiry + traffic composition (issue #18)
# ---------------------------------------------------------------------- #


def _talk(*kinds):
    """One step whose traffic is exactly ``kinds``."""
    return step(1, [sold("TL1")], messages=[msg(k, "TL1", "ALL") for k in kinds])


def test_traffic_composition_has_a_denominator():
    # 8 messages: 2 command (order, execute), 4 voice (SYNC PROPOSE/GO),
    # and 2 that are neither (a report and its automatic acknowledgement).
    steps = [
        step(0, [sold("TL1")]),
        _talk("order", "ack", "execute", "sync_propose", "sync_go", "sync_propose", "sync_go",
              "sitrep"),
    ]
    ep = episode_behavior(trace(steps))
    assert ep["messages"] == 8
    assert ep["messages_command"] == 2
    assert ep["messages_voice"] == 4
    agg = aggregate_behavior([ep])
    assert agg["messages_per_episode"] == 8
    assert agg["command_traffic_share"] == 0.25
    assert agg["voice_traffic_share"] == 0.5
    assert "messages / ep" in format_behavior_table(agg)


def _clock_agg(timeouts, successes, *, root_mission="SEIZE"):
    """Aggregate over ``timeouts`` clock-expiry episodes and ``successes`` wins."""
    steps = [step(0, [sold("TL1")]), step(1, [sold("TL1")])]
    eps = [
        episode_behavior(trace(steps, outcome=out, root_mission=root_mission))
        for out in ["timeout"] * timeouts + ["success"] * successes
    ]
    return aggregate_behavior(eps)


def test_clock_expiry_gate_fails_a_stall_under_any_root_mission():
    # squad_screen_v4 / squad_recon_v6 / squad_screen_v5 at ckpt_latest:
    # 30/30 episodes pinned at max_steps. The root missions are SCREEN and
    # RECON, which the positional gate never looked at.
    agg = _clock_agg(30, 0, root_mission="RECON")
    assert agg["timeout_rate"] == 1.0
    gate = next(g for g in regression_gates(agg) if g["name"] == "timeout_rate")
    assert gate["passed"] is False
    assert "FAIL" in format_gate_report(regression_gates(agg))


def test_clock_expiry_gate_passes_the_healthy_record():
    # The worst healthy checkpoint measured (fireteam_defend_v9/best,
    # fireteam_defend_v8/latest): 2 timeouts in 10 episodes.
    agg = _clock_agg(2, 8)
    assert agg["timeout_rate"] == 0.2
    gate = next(g for g in regression_gates(agg) if g["name"] == "timeout_rate")
    assert gate["passed"] is True


def test_traffic_composition_is_diagnosis_not_a_gate():
    """A command-share bound would flag healthy runs and pass stalled ones.

    Measured at 10 episodes/checkpoint, seeds 500-509: the healthy
    ``fireteam_defend_v10/ckpt_best`` (8/10 success) carries a command share
    of 0.026, *below* the collapsed ``squad_recon_v6/ckpt_latest`` (0/10) at
    0.022. Order rate orders them no better: ``fireteam_v7/ckpt_latest``
    issues 1.5 orders/episode at 8/10 success. Composition is scenario
    idiom, so it is reported and never gated — the clock is the separator.
    """
    healthy = _clock_agg(0, 10)
    stalled = _clock_agg(10, 0)
    for agg, share in ((healthy, 0.026), (stalled, 0.022)):
        agg["command_traffic_share"] = share
    verdicts = [
        next(g for g in regression_gates(a) if g["name"] == "timeout_rate")["passed"]
        for a in (healthy, stalled)
    ]
    assert verdicts == [True, False]
    assert healthy["command_traffic_share"] > stalled["command_traffic_share"]
    assert "command_traffic_share" not in [g["name"] for g in regression_gates(stalled)]


# ---------------------------------------------------------------------- #
# success axis: the defeat-shaped collapse (issue #21)
# ---------------------------------------------------------------------- #


def _shape_agg(successes, defeats, timeouts, *, root_mission="DEFEND"):
    """Aggregate built from an outcome mix — the corpora shapes issue #21 cites."""
    steps = [step(0, [sold("TL1")]), step(1, [sold("TL1")])]
    outcomes = ["success"] * successes + ["defeat"] * defeats + ["timeout"] * timeouts
    eps = [episode_behavior(trace(steps, outcome=out, root_mission=root_mission)) for out in outcomes]
    return aggregate_behavior(eps)


def test_success_axis_fires_on_documented_defeat_shaped_corpora():
    """refs issue #21: the premise check found four measured collapses that
    are DEFEAT-shaped, not STALL-shaped — none within an order of magnitude
    of the D4 stall signature (>= 28/30 timeout on record) — so the
    clock-expiry gate alone reads every one of them as healthy on the clock.
    `squad_screen_v7` is not a defend scenario, included to show the axis
    (like the clock-expiry gate) is root-mission-agnostic.
    """
    corpora = {
        "fireteam_defend_v6b": (1, 27, 2, "DEFEND"),  # 1/30 success, 2/30 timeout
        "fireteam_defend_v6": (14, 12, 4, "DEFEND"),  # 14/30 success, 4/30 timeout
        "fireteam_defend_v7": (12, 11, 7, "DEFEND"),  # 12/30 success, 7/30 timeout
        "squad_screen_v7": (6, 24, 0, "SCREEN"),  # 6/30 success, 0/30 timeout
    }
    for name, (successes, defeats, timeouts, root) in corpora.items():
        agg = _shape_agg(successes, defeats, timeouts, root_mission=root)
        gates = regression_gates(agg)
        timeout_gate = next(g for g in gates if g["name"] == "timeout_rate")
        success_gate = next((g for g in gates if g["name"] == "success_rate"), None)
        assert timeout_gate["passed"] is True, f"{name}: not stall-shaped by the clock"
        assert success_gate is not None, f"{name}: the success axis should apply"
        assert success_gate["passed"] is False, f"{name}: should read as a collapse"
        assert "FAIL" in format_gate_report([success_gate])


def test_success_axis_does_not_fire_on_the_healthy_fleet():
    """The v1.11 fleet the reward decision was measured against: neither the
    two lowest-success healthy checkpoints on record nor a clean 1.00 run
    should trip the new axis.
    """
    healthy = {
        "fireteam_v8": (27, 3, 0),  # 0.90
        "fireteam_defend_v11": (74, 26, 0),  # 0.74, N=100
        "squad_v8": (30, 0, 0),  # 1.00
    }
    for name, (successes, defeats, timeouts) in healthy.items():
        agg = _shape_agg(successes, defeats, timeouts)
        success_gate = next(g for g in regression_gates(agg) if g["name"] == "success_rate")
        assert success_gate["passed"] is True, name


def test_success_axis_is_silent_on_a_genuine_stall():
    """A stalled run (30/30 timeout, 0 success) must read as STALLED only —
    the point of gating the success axis on timeout_rate already passing is
    that a collapsed run fails exactly one axis, never both, so the report
    always says which shape it was.
    """
    agg = _shape_agg(0, 0, 30, root_mission="SEIZE")  # SEIZE: no positional gate in play
    gates = regression_gates(agg)
    # the command-report gate is unconditional (it is an axis, not a shape), but
    # a run that never won sends no ENDEX, so it reads unmeasured here
    assert [g["name"] for g in gates] == ["timeout_rate", "closed_on_root_report_rate"]
    assert gates[0]["passed"] is False
    assert gates[1]["passed"] is None
    report = format_gate_report(gates)
    assert "FAIL" in report
    assert "success_rate" not in report


def test_the_command_report_gate_fails_a_mute_commander_that_wins_everything():
    """v1.20's gate, on the exact corpora that motivated it.

    ``successes_announced_rate`` counts the ENDEX, not who claimed it, so it
    reads 1.00 for a commander that never transmits — and did, on three runs in
    one day. Each won 0.93-0.98 of its episodes and filed ZERO root claims:
    ``squad_v11``, ``squad_v14b_nobonus``, ``squad_v14c_nobonus``. Every other
    gate on the board passes them. This one must not.
    """
    for realised in (0.0, 0.01):  # the two mute values ever measured
        agg = _shape_agg(98, 1, 1, root_mission="SEIZE")
        agg["closed_on_root_report_rate"] = realised
        gates = {g["name"]: g for g in regression_gates(agg)}
        assert gates["success_rate"]["passed"] is True, "the run wins — that is the point"
        assert gates["timeout_rate"]["passed"] is True
        assert gates["closed_on_root_report_rate"]["passed"] is False
        assert "FAIL" in format_gate_report(list(gates.values()))

    # and the weakest non-mute corpus on record still passes: squad_v10b, 0.784
    agg = _shape_agg(88, 10, 2, root_mission="SEIZE")
    agg["closed_on_root_report_rate"] = 0.784
    gates = {g["name"]: g for g in regression_gates(agg)}
    assert gates["closed_on_root_report_rate"]["passed"] is True


def test_command_report_bound_sits_in_the_empty_band():
    # Nothing has ever been measured between the mute regime (0.000-0.01) and
    # the weakest reporting corpus on file (squad_v10b, 0.784). The floor
    # refuses a regime; it does not police good-vs-better.
    assert 0.01 < ROOT_REPORT_CLOSE_FLOOR < 0.784


def test_success_axis_bound_sits_in_the_measured_gap():
    # The floor (0.5) must separate every documented defeat-shaped corpus
    # (highest: fireteam_defend_v6 at 0.467) from the lowest healthy record
    # on file (fireteam_defend_v11 at 0.74) without hair-splitting.
    assert 0.467 < SUCCESS_RATE_FLOOR < 0.74


def test_recorder_records_the_step_ceiling():
    # "pinned at max_steps" is only a statement the trace can make if it
    # carries the ceiling the episode was played under.
    env = make_env("squad_screen")
    rec = TraceRecorder()
    run_episode(env, None, seed=11, rng=np.random.default_rng(11), recorder=rec)
    assert rec.trace["max_steps"] == env.spec_cfg.max_steps
    agg = aggregate_behavior([episode_behavior(rec.trace)])
    assert agg["max_steps"] == env.spec_cfg.max_steps
    assert agg["timeout_rate"] in (0.0, 1.0)
    assert agg["timeout_rate"] == float(rec.trace["outcome"] == "timeout")
    if agg["timeout_rate"]:
        assert agg["episode_length_mean"] == env.spec_cfg.max_steps


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


# ---------------------------------------------------------------------- #
# ENDEX: the completion signal on a continuous-posture root (v1.13)
# ---------------------------------------------------------------------- #


def test_closed_on_root_report_rate_separates_a_prompted_close_from_a_silent_one():
    """`false_complete_rate` is structurally 0 on a DEFEND root, so it stops
    being a reporting-quality signal. This is what replaces it: COMMAND sends
    ENDEX either way, and the question worth asking is whether the root's
    report is what closed the window."""
    prompted = episode_behavior(
        trace(
            [step(0, [sold("TL1")]), step(1, [sold("TL1")], messages=[msg("endex", "HQ", "TL1")])],
            root_mission="DEFEND",
            root_close_step=1,
        )
    )
    silent = episode_behavior(
        trace(
            [step(0, [sold("TL1")]), step(1, [sold("TL1")], messages=[msg("endex", "HQ", "TL1")])],
            root_mission="DEFEND",
            root_close_step=None,
        )
    )
    assert prompted["endex_sent"] == 1 and prompted["endex_on_root_report"] == 1
    assert silent["endex_sent"] == 1 and silent["endex_on_root_report"] == 0

    agg = aggregate_behavior([prompted, silent])
    assert agg["endex_sent"] == 2
    assert agg["closed_on_root_report_rate"] == 0.5


def test_a_completable_root_reports_no_endex_denominator_at_all():
    """No ENDEX means the rate is None, not 0 — a SEIZE root closes its own
    operation with MISSION COMPLETE, and reading that as "never reported"
    would be the same denominator confusion `false_complete_rate` fell into
    on fireteam_defend_v12 (one claim, rate 1.00)."""
    seize = episode_behavior(
        trace([step(0, [sold("TL1")]), step(1, [sold("TL1")])], root_mission="SEIZE")
    )
    agg = aggregate_behavior([seize])
    assert agg["endex_sent"] == 0
    assert agg["closed_on_root_report_rate"] is None


def test_successes_announced_counts_the_wins_that_said_so_on_the_net():
    """v1.16, the number #31 asked for and no existing metric could give.

    `closed_on_root_report_rate` has ENDEXes-sent for a denominator, so an
    operation that closed in total silence does not appear in it at all — which
    is how v1.14 could announce 0 of 57 successes on fireteam_defend without a
    single published figure moving. This one counts successes, and asks of each
    whether ANYTHING went out: COMMAND's ENDEX or the root's own confirmed
    claim (on a SEIZE root the claim is the announcement, and there is no ENDEX
    to want).
    """
    endex_only = episode_behavior(
        trace(
            [step(0, [sold("TL1")]), step(1, [sold("TL1")], messages=[msg("endex", "HQ", "TL1")])],
            root_mission="DEFEND",
            root_close_step=None,
        )
    )
    claim_only = episode_behavior(
        trace(
            [step(0, [sold("TL1")]), step(1, [sold("TL1")])],
            root_mission="SEIZE",
            root_close_step=1,
        )
    )
    silent = episode_behavior(
        trace([step(0, [sold("TL1")]), step(1, [sold("TL1")])], root_mission="DEFEND")
    )
    lost = episode_behavior(
        trace([step(0, [sold("TL1")]), step(1, [sold("TL1")])],
              root_mission="DEFEND", outcome="timeout")
    )

    assert endex_only["close_announced"] == 1
    assert claim_only["close_announced"] == 1
    assert silent["close_announced"] == 0

    agg = aggregate_behavior([endex_only, claim_only, silent, lost])
    assert agg["successes"] == 3, "a failed operation is not a win to announce"
    assert agg["successes_announced"] == 2
    assert agg["successes_announced_rate"] == 2 / 3


def test_a_zero_announcement_says_which_kind_of_silence_it_is():
    """refs #38: `successes_announced` is one integer, and a zero on it has
    three causes that want three different fixes.

    The published case: at N=100 on the final policy `patrol_brique_v5`
    announces 0 of 99 with its root **never claiming**, and `platoon_v5`
    announces 0 of 100 with its root **claiming five times and refused every
    time**. Identical on the integer, opposite on the radio — a silent policy
    and a rejected one — and the README grouped them as the same result.

    So the announcement line renders the root's own claim channel beside it,
    which is #13's argument about zero DONE reports carried one level up.
    """
    root_open = {**sold("TL1"), "root": True, "done_ok": True}
    root_shut = {**sold("TL1"), "root": True, "done_ok": False}

    declined = episode_behavior(
        trace([step(0, [root_open]), step(1, [root_open])], root_mission="SEIZE")
    )
    refused = episode_behavior(
        trace(
            [
                step(0, [root_open]),
                step(1, [root_open],
                     messages=[msg("done", "TL1", "HQ"), msg("done_reject", "HQ", "TL1")]),
                step(2, [root_open]),
            ],
            root_mission="SEIZE",
        )
    )
    shut = episode_behavior(
        trace(
            [step(0, [root_shut]),
             step(1, [root_shut], messages=[msg("endex", "HQ", "TL1")])],
            root_mission="DEFEND",
        )
    )

    # all three succeed; two of them announce nothing, for opposite reasons
    assert declined["close_announced"] == 0 and refused["close_announced"] == 0
    assert shut["close_announced"] == 1

    silent_agg = aggregate_behavior([declined])
    assert silent_agg["successes_announced"] == 0
    assert silent_agg["done_reports_root"] == 0
    assert silent_agg["done_admissible_root"] == 1
    assert format_root_claim_shape(silent_agg) == "root never claimed, 1 admissible step"

    refused_agg = aggregate_behavior([refused])
    assert refused_agg["successes_announced"] == 0
    assert refused_agg["done_reports_root"] == 1 == refused_agg["done_rejected_root"]
    assert format_root_claim_shape(refused_agg) == "root claimed 1, all refused"

    # the defend family's zero claims are the mask, not a policy — the
    # distinction that made v1.15's silence a failure and v1.17's a design
    shut_agg = aggregate_behavior([shut])
    assert shut_agg["done_admissible_root"] == 0
    assert format_root_claim_shape(shut_agg) == "root never claimed, channel shut"

    # and a partly-accepted channel reports both halves rather than a ratio
    accepted = episode_behavior(
        trace(
            [
                step(0, [root_open]),
                step(1, [root_open],
                     messages=[msg("done", "TL1", "HQ"), msg("done_reject", "HQ", "TL1")]),
                step(2, [root_open],
                     messages=[msg("done", "TL1", "HQ"), msg("done_confirm", "HQ", "TL1")]),
            ],
            root_mission="SEIZE",
            root_close_step=2,
        )
    )
    accepted_agg = aggregate_behavior([accepted])
    assert accepted_agg["successes_announced"] == 1
    assert format_root_claim_shape(accepted_agg) == "root claimed 2, 1 refused"

    # the shape travels with the rendered table, where the grouping went wrong
    table = format_behavior_table(refused_agg)
    assert "root claimed 1, all refused" in table


def test_successes_announced_rate_is_none_when_nothing_succeeded():
    """No wins, no denominator — 0.00 would read as a reporting failure."""
    lost = episode_behavior(
        trace([step(0, [sold("TL1")])], root_mission="DEFEND", outcome="timeout")
    )
    agg = aggregate_behavior([lost])
    assert agg["successes"] == 0
    assert agg["successes_announced_rate"] is None


# ---------------------------------------------------------------------- #
# the close route: timing vs volume (issue #35)
# ---------------------------------------------------------------------- #


def _sitrep_close_episode(
    sitrep_steps, *, close_step, interval=25, clock_start=None, endex=True
):
    """A DEFEND episode whose root SITREPs at ``sitrep_steps``.

    ``close_step`` is the step the root's report closed the window on, where
    COMMAND transmits ENDEX — the environment does both in the same tick.
    """
    root = {**sold("TL1"), "root": True}
    sitrep_steps = list(sitrep_steps)
    last = max([*sitrep_steps, close_step or 0, 1])
    steps = [step(0, [root])]
    for t in range(1, last + 1):
        messages = [msg("sitrep", "TL1", "HQ")] if t in sitrep_steps else []
        if endex and t == last:
            messages = [*messages, msg("endex", "HQ", "TL1")]
        steps.append(step(t, [root], messages=messages))
    return episode_behavior(
        trace(
            steps,
            root_mission="DEFEND",
            root_close_step=close_step,
            sitrep_interval=interval,
            sitrep_clock_start=clock_start,
        )
    )


def test_the_close_rate_saturates_and_the_new_denominator_separates_it():
    """refs #35: `closed_on_root_report_rate` reads 1.00 for two opposite
    policies — one that timed a single report to the closing moment, and one
    that transmitted ten and was closed by whichever landed last. The rate
    cannot separate timing from volume; these three can."""
    timed = aggregate_behavior([_sitrep_close_episode([30], close_step=30)])
    bought = aggregate_behavior([_sitrep_close_episode(range(3, 31, 3), close_step=30)])

    # the saturating reading: identical, and it is the one both sides publish
    assert timed["closed_on_root_report_rate"] == 1.0
    assert bought["closed_on_root_report_rate"] == 1.0

    assert timed["root_sitreps"] == 1 and bought["root_sitreps"] == 10
    assert timed["root_sitreps_per_episode"] == 1.0
    assert bought["root_sitreps_per_episode"] == 10.0
    # closes per report emitted: one buys one, ten buy the same one
    assert timed["closes_per_root_sitrep"] == 1.0
    assert bought["closes_per_root_sitrep"] == 0.1
    # and the close itself: a report the cadence would have produced anyway,
    # against one bought three steps after the last
    assert timed["closed_on_cadence_report_rate"] == 1.0
    assert bought["closed_on_cadence_report_rate"] == 0.0
    assert timed["root_sitrep_off_cadence_share"] == 0.0
    assert bought["root_sitrep_off_cadence_share"] == 0.9  # only the first was fresh


def test_cadence_compliance_is_the_environment_s_own_freshness_rule():
    """A report at exactly ``sitrep_interval`` is what the environment pays
    `sitrep_fresh` for; one step earlier is `sitrep_spam`. The metric must cut
    in the same place, or "off cadence" means something the reward does not."""
    on_cadence = aggregate_behavior([_sitrep_close_episode([1, 26], close_step=26)])
    off_cadence = aggregate_behavior([_sitrep_close_episode([1, 25], close_step=25)])

    assert on_cadence["closed_on_cadence_report_rate"] == 1.0
    assert on_cadence["root_sitreps_off_cadence"] == 0
    assert off_cadence["closed_on_cadence_report_rate"] == 0.0
    assert off_cadence["root_sitreps_off_cadence"] == 1
    assert off_cadence["root_sitrep_off_cadence_share"] == 0.5
    assert on_cadence["sitrep_interval"] == off_cadence["sitrep_interval"] == 25


def test_the_reporting_doctrine_starts_the_freshness_clock_at_zero():
    """Where `ScenarioSpec.sitrep_cadence` is on, the first report is *owed*
    within one interval, so an early one is off cadence. Where it is off, the
    first report of an episode is fresh whenever it comes — the two differ
    only in the clock the trace records, so the metric must read it."""
    doctrine = aggregate_behavior([_sitrep_close_episode([10], close_step=10, clock_start=0)])
    free = aggregate_behavior([_sitrep_close_episode([10], close_step=10)])

    assert doctrine["closed_on_cadence_report_rate"] == 0.0
    assert doctrine["root_sitreps_off_cadence"] == 1
    assert free["closed_on_cadence_report_rate"] == 1.0
    assert free["root_sitreps_off_cadence"] == 0


def test_a_subordinate_s_sitreps_are_not_the_root_s_volume():
    """The denominator is the root's channel. A rifleman transmitting every
    step must not make the root's one timed report read as spam — the
    environment's freshness clock is per soldier, and so is this."""
    root = {**sold("TL1"), "root": True}
    rifleman = {**sold("RFN1"), "root": False}
    steps = [step(0, [root, rifleman])]
    for t in range(1, 31):
        messages = [msg("sitrep", "RFN1", "TL1")]
        if t == 30:
            messages += [msg("sitrep", "TL1", "HQ"), msg("endex", "HQ", "TL1")]
        steps.append(step(t, [root, rifleman], messages=messages))
    agg = aggregate_behavior(
        [
            episode_behavior(
                trace(steps, root_mission="DEFEND", root_close_step=30)
            )
        ]
    )

    assert agg["root_sitreps"] == 1
    assert agg["closes_per_root_sitrep"] == 1.0
    assert agg["closed_on_cadence_report_rate"] == 1.0


def test_a_successor_s_sitrep_closes_the_operation_on_its_own_clock():
    """Succession moves the root, so the closing report is whoever holds it —
    and the freshness clock follows the *soldier*, exactly as the environment
    keeps it, so the successor's first report is fresh however loud its
    predecessor was."""
    before = [{**sold("TL1"), "root": True}, {**sold("SGT2"), "root": False}]
    after = [{**sold("TL1", alive=False), "root": False}, {**sold("SGT2"), "root": True}]
    steps = [step(0, before)]
    for t in range(1, 6):
        steps.append(step(t, before, messages=[msg("sitrep", "TL1", "HQ")]))
    steps.append(step(6, after))
    steps.append(
        step(7, after, messages=[msg("sitrep", "SGT2", "HQ"), msg("endex", "HQ", "SGT2")])
    )
    agg = aggregate_behavior(
        [episode_behavior(trace(steps, root_mission="DEFEND", root_close_step=7))]
    )

    assert agg["root_sitreps"] == 6
    assert agg["root_sitreps_off_cadence"] == 4  # TL1's 2nd-5th, at one-step gaps
    assert agg["closed_on_cadence_report_rate"] == 1.0
    assert agg["closes_per_root_sitrep"] == 1 / 6


def test_a_root_that_never_reported_leaves_the_ratio_undefined_not_zero():
    """The v1.13 denominator lesson: a rate with no events reads None. Zero
    closes per zero reports is not "the root bought nothing" — it is a
    question nobody asked."""
    silent = aggregate_behavior([_sitrep_close_episode([], close_step=None)])

    assert silent["root_sitreps"] == 0
    assert silent["root_sitreps_per_episode"] == 0.0  # a count, and it is real
    assert silent["closes_per_root_sitrep"] is None
    assert silent["root_sitrep_off_cadence_share"] is None
    assert silent["closed_on_cadence_report_rate"] == 0.0  # an ENDEX went out
    assert silent["closed_on_root_report_rate"] == 0.0


def test_a_completable_root_has_no_cadence_denominator_either():
    """No ENDEX, no operations COMMAND closed — the same None
    `closed_on_root_report_rate` reports on a SEIZE root, for the same
    reason. The volume count is still measured, because it is a count."""
    steps = [step(0, [{**sold("TL1"), "root": True}])]
    for t in (1, 2):
        steps.append(step(t, [{**sold("TL1"), "root": True}],
                          messages=[msg("sitrep", "TL1", "HQ")]))
    agg = aggregate_behavior([episode_behavior(trace(steps, root_mission="SEIZE"))])

    assert agg["endex_sent"] == 0
    assert agg["closed_on_cadence_report_rate"] is None
    assert agg["closed_on_root_report_rate"] is None
    assert agg["root_sitreps"] == 2 and agg["closes_per_root_sitrep"] == 0.0


def test_a_claim_route_close_counts_in_the_denominator_and_not_the_numerator():
    """v1.16's horizon defense closed on a confirmed MISSION COMPLETE while
    COMMAND still sent ENDEX. That close did not use the SITREP channel, so it
    cannot count as evidence the channel is timed — and it must not vanish
    from the denominator either, or the rate would flatter the policy."""
    root = {**sold("TL1"), "root": True}
    steps = [
        step(0, [root]),
        step(1, [root], messages=[msg("done", "TL1", "HQ"), msg("done_confirm", "HQ", "TL1")]),
        step(2, [root], messages=[msg("endex", "HQ", "TL1")]),
    ]
    agg = aggregate_behavior(
        [episode_behavior(trace(steps, root_mission="DEFEND", root_close_step=1))]
    )

    assert agg["endex_sent"] == 1
    assert agg["closed_on_root_report_rate"] == 1.0
    assert agg["closed_on_cadence_report_rate"] == 0.0
    assert agg["closes_per_root_sitrep"] is None  # no SITREP was ever sent


def test_the_table_prints_the_density_beside_the_rate():
    """The density has to sit next to the rate rather than be inferable from
    it: 1.00 with 30 reports an episode and 1.00 with one are the finding."""
    agg = aggregate_behavior([_sitrep_close_episode(range(3, 31, 3), close_step=30)])
    table = format_behavior_table(agg)

    assert "root SITREPs / ep" in table and "closes / root SITREP" in table
    assert "90% off cadence" in table and "interval 25" in table


def test_recorder_records_the_freshness_interval_it_was_played_under():
    """Off-cadence is not a statement the trace can support without the
    interval the environment priced the reports at — the same reason the
    contact refresh age and the step ceiling are recorded."""
    env = make_env("fireteam_defend")
    rec = TraceRecorder()
    run_episode(env, None, seed=5, rng=np.random.default_rng(5), recorder=rec)

    expected = env.spec_cfg.sitrep_cadence or env.rewards_cfg.sitrep_interval
    assert rec.trace["sitrep_interval"] == expected
    agg = aggregate_behavior([episode_behavior(rec.trace)])
    assert agg["sitrep_interval"] == expected
    assert agg["root_sitreps"] >= 0
    # every root SITREP is either on cadence or off it, and never both
    assert agg["root_sitreps_off_cadence"] <= agg["root_sitreps"]


def test_split_gates_keeps_unmeasured_out_of_the_failures():
    gates = [
        {"name": "timeout_rate", "value": 1.0, "bound": 0.5, "direction": "max", "passed": False},
        {"name": "success_rate", "value": 0.9, "bound": 0.5, "direction": "min", "passed": True},
        {"name": "closed_on_root_report_rate", "value": None, "bound": 0.5,
         "direction": "min", "passed": None},
    ]
    failed, unmeasured = split_gates(gates)
    assert failed == ["timeout_rate"]
    assert unmeasured == ["closed_on_root_report_rate"]
