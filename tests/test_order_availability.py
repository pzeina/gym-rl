"""Ordered-task mix against what the order mask actually offered (issue #16).

An order share is *availability-confounded*. The order vocabulary is not a
flat menu: SUPPORT is unit-targeted, so it needs a second living subordinate
slot and disappears entirely from missions that cannot derive it, while
OBSERVE is objective-targeted and admissible wherever it is derivable at all.
A task can therefore be rare because the policy declined it or because it was
barely on the menu, and the raw share cannot tell those apart.

Worse, the confound does not even point one way. Measured on the masked-random
floor, `squad` offers OBSERVE ~2.9x more entries than SUPPORT while
`fireteam_defend` offers SUPPORT ~1.9x more than OBSERVE — so reading a raw
OBSERVE-over-SUPPORT ratio as a preference *overstates* it in one scenario
family and *understates* it in the other. `fireteam_defend_v8` reads 0.10
OBSERVE against 0.01 SUPPORT, which looks like a strong OBSERVE preference and
is not: corrected for availability it is OBSERVE x0.92 (the floor, i.e. no
preference at all) and SUPPORT x0.04 (declining 96% of the opportunities it
held). The finding is SUPPORT avoidance, and only the denominator shows it.

These tests pin the denominator, its calibration (a masked-random policy must
measure at lift 1.00 by construction), and the two behaviours that make a
naive implementation wrong: an unavailable task must read as *no measurement*
rather than as zero, and an order the issuer's own mask never offered must
stay out of the matched control.
"""

import numpy as np

from cohort.env.actions import CATALOG, order_options
from cohort.env.cohort_env import make_env
from cohort.metrics import (
    TraceRecorder,
    aggregate_behavior,
    episode_behavior,
    format_order_availability,
    order_selection_lift,
)
from cohort.training.evaluate import _seeded_episode

# ---------------------------------------------------------------------- #
# constructed-trace helpers
# ---------------------------------------------------------------------- #


def sold(cs, *, order_opts=None, auth=0):
    return {
        "cs": cs,
        "alive": True,
        "pos": [0, 0],
        "mission": "DEFEND",
        "since": 0,
        "auth": auth,
        "subs": [],
        "comp": None,
        "sees": [],
        "cover": False,
        "order_opts": order_opts or {},
    }


def msg(frm, to, mission):
    return {"kind": "order", "from": frm, "to": to, "mission": mission, "text": ""}


def trace(steps):
    return {
        "scenario": "test",
        "outcome": "success",
        "length": steps[-1]["t"],
        "root_mission": "DEFEND",
        "root_objective": None,
        "ring_radius": 7.0,
        "threat_radius": 8.0,
        "contact_refresh_age": 20,
        "knowledge_ttl": 40,
        "human": None,
        "reported": {},
        "steps": steps,
    }


def step(t, soldiers, messages=()):
    return {"t": t, "soldiers": soldiers, "enemies": [], "messages": list(messages)}


# ---------------------------------------------------------------------- #
# the denominator itself
# ---------------------------------------------------------------------- #


def test_order_options_counts_mission_payload_entries_off_the_mask():
    """It reports the admissible order *entries*, never re-derives them."""
    env = make_env("squad")
    obs, _ = env.reset(seed=1)
    leader = env.roster.root()
    mask = obs[leader.callsign]["action_mask"]
    options = order_options(mask)

    by_hand = {}
    for spec in CATALOG:
        if spec.kind == "order" and spec.order_mission is not None and mask[spec.index]:
            by_hand[spec.order_mission.name] = by_hand.get(spec.order_mission.name, 0) + 1
    assert options == by_hand
    assert options, "a squad leader holding the OPORD must have orders on the menu"
    # A5-3 stance orders carry no mission and never enter orders_by_task, so
    # they must not enter its denominator either.
    assert "FORMATION" not in options


def test_order_options_of_a_soldier_with_no_command_authority_is_empty():
    env = make_env("squad")
    obs, _ = env.reset(seed=1)
    rifleman = next(s for s in env.roster.soldiers if s.effective_authority == 0)
    assert order_options(obs[rifleman.callsign]["action_mask"]) == {}


def test_the_menu_favours_support_and_observe_in_opposite_directions():
    """The confound has a sign, and it flips by scenario family.

    This is the fact that makes the raw share unreadable: a DEFEND root
    derives SUPPORT and OBSERVE both, but SUPPORT is unit-targeted (one entry
    per supported slot) while OBSERVE takes a single objective on these maps,
    so the defend menu leans SUPPORT — the exact opposite of the squad menu,
    where a SEIZE root's OBSERVE entries outnumber a two-slot SUPPORT's.
    """
    defend = make_env("fireteam_defend")
    obs, _ = defend.reset(seed=1)
    menu = order_options(obs[defend.roster.root().callsign]["action_mask"])
    assert menu["SUPPORT"] > menu["OBSERVE"]

    squad = make_env("squad")
    obs, _ = squad.reset(seed=1)
    menu = order_options(obs[squad.roster.root().callsign]["action_mask"])
    assert menu["OBSERVE"] > menu["SUPPORT"]


def test_a_screen_root_never_offers_support_at_all():
    """SCREEN cannot derive SUPPORT, so its share is not a preference reading.

    An external corpus measured `squad_screen` masked-random at SUPPORT 0.000
    and read the ratio as infinite policy bias. There is no policy in it: the
    mask offers nothing to select.
    """
    env = make_env("squad_screen")
    obs, _ = env.reset(seed=1)
    for callsign in env.agents:
        assert "SUPPORT" not in order_options(obs[callsign]["action_mask"])


# ---------------------------------------------------------------------- #
# lift: share over availability
# ---------------------------------------------------------------------- #


def test_lift_separates_a_declined_task_from_a_rarely_offered_one():
    """Identical raw shares, opposite findings.

    Both leaders issue 1 OBSERVE and 1 SUPPORT — a raw mix of 0.50/0.50 that
    says nothing. TL1 was offered SUPPORT nine times as often as OBSERVE and
    still split evenly (it declines SUPPORT); TL2 was offered the reverse.
    """
    steps = [
        step(0, [sold("TL1", order_opts={"OBSERVE": 1, "SUPPORT": 9}),
                 sold("TL2", order_opts={"OBSERVE": 9, "SUPPORT": 1})]),
        step(1, [sold("TL1", order_opts={"OBSERVE": 1, "SUPPORT": 9}),
                 sold("TL2", order_opts={"OBSERVE": 9, "SUPPORT": 1})],
             [msg("TL1", "RFN1", "OBSERVE"), msg("TL2", "RFN2", "SUPPORT")]),
        step(2, [sold("TL1", order_opts={"OBSERVE": 1, "SUPPORT": 9}),
                 sold("TL2", order_opts={"OBSERVE": 9, "SUPPORT": 1})],
             [msg("TL1", "RFN1", "SUPPORT"), msg("TL2", "RFN2", "OBSERVE")]),
    ]
    agg = aggregate_behavior([episode_behavior(trace(steps))])

    assert agg["orders_issued"] == 4
    assert agg["orders_matched"] == 4
    # each of the four decisions was made from a menu that was 0.1 one task
    # and 0.9 the other, in balanced pairs -> both tasks were offered 0.50
    assert agg["order_availability"] == {"OBSERVE": 0.5, "SUPPORT": 0.5}

    lift = order_selection_lift(agg)
    assert lift == {"OBSERVE": 1.0, "SUPPORT": 1.0}

    # ...and TL1 alone, the leader that declined the SUPPORT it was offered,
    # reads as declining it rather than as a 50/50 split.
    solo = [
        step(0, [sold("TL1", order_opts={"OBSERVE": 1, "SUPPORT": 9})]),
        step(1, [sold("TL1", order_opts={"OBSERVE": 1, "SUPPORT": 9})],
             [msg("TL1", "RFN1", "OBSERVE")]),
        step(2, [sold("TL1", order_opts={"OBSERVE": 1, "SUPPORT": 9})],
             [msg("TL1", "RFN1", "SUPPORT")]),
    ]
    lift = order_selection_lift(aggregate_behavior([episode_behavior(trace(solo))]))
    assert lift["OBSERVE"] == 5.0     # 0.50 share on a 0.10 menu
    assert abs(lift["SUPPORT"] - 0.5555555) < 1e-6   # 0.50 share on a 0.90 menu


def test_a_task_that_was_never_offered_reads_as_no_measurement():
    """Not as zero, and never as a division by zero."""
    steps = [
        step(0, [sold("TL1", order_opts={"OBSERVE": 2})]),
        step(1, [sold("TL1", order_opts={"OBSERVE": 2})], [msg("TL1", "RFN1", "OBSERVE")]),
    ]
    agg = aggregate_behavior([episode_behavior(trace(steps))])
    lift = order_selection_lift(agg)
    assert lift["OBSERVE"] == 1.0
    assert "SUPPORT" not in lift
    assert format_order_availability(agg) == "OBSERVE 1.00/1.00 (x1.00)"


def test_an_order_the_mask_never_offered_stays_out_of_the_control():
    """Injected / replayed orders have no opportunity set to compare against.

    They still count as issued — the gap between the two denominators is how
    an availability reading stays honest about what it could not match.
    """
    steps = [
        step(0, [sold("TL1", order_opts={"OBSERVE": 2})]),
        step(1, [sold("TL1", order_opts={"OBSERVE": 2})], [msg("TL1", "RFN1", "OBSERVE")]),
        step(2, [sold("TL1", order_opts={"OBSERVE": 2})], [msg("TL1", "RFN1", "DENY")]),
    ]
    agg = aggregate_behavior([episode_behavior(trace(steps))])
    assert agg["orders_issued"] == 2
    assert agg["orders_matched"] == 1
    assert agg["order_availability"] == {"OBSERVE": 1.0}


def test_no_orders_leaves_the_availability_reading_empty_not_wrong():
    steps = [step(0, [sold("TL1")]), step(1, [sold("TL1")])]
    agg = aggregate_behavior([episode_behavior(trace(steps))])
    assert agg["orders_matched"] == 0
    assert agg["order_availability"] == {}
    assert order_selection_lift(agg) == {}
    assert format_order_availability(agg) == ""


# ---------------------------------------------------------------------- #
# calibration against the real environment
# ---------------------------------------------------------------------- #


def test_masked_random_measures_at_lift_one_though_its_raw_mix_does_not():
    """The floor is the whole point: no reward pressure must read as no preference.

    A uniform-over-legal policy has no preferences by construction, yet its
    raw ordered-task mix on `fireteam_defend` is nowhere near flat — it orders
    SUPPORT well above OBSERVE purely because the menu offers more of it. The
    correction has to send that ratio back to 1 while the raw one stays put;
    if it does not, every "excess over the masked-random floor" reading built
    on it is measuring the mask.
    """
    env = make_env("fireteam_defend")
    recorders = [TraceRecorder() for _ in range(12)]
    for i, recorder in enumerate(recorders):
        _seeded_episode(env, None, 500 + i, recorder=recorder)
    agg = aggregate_behavior([episode_behavior(r.trace) for r in recorders])

    assert agg["orders_matched"] == agg["orders_issued"] > 200
    issued = {task: sum(b.values()) for task, b in agg["orders_by_task"].items()}
    lift = order_selection_lift(agg)

    # the raw mix is inverted relative to a flat menu: masked-random orders
    # SUPPORT clearly more often than OBSERVE, having chosen neither
    assert issued["SUPPORT"] > 1.4 * issued["OBSERVE"]
    # ...and the correction removes exactly that, task by task
    for task, count in issued.items():
        if count >= 25:
            assert 0.7 < lift[task] < 1.35, f"{task} lift {lift[task]:.2f} off the floor"
    assert 0.75 < lift["OBSERVE"] / lift["SUPPORT"] < 1.35


def test_recording_availability_does_not_perturb_a_seeded_episode():
    """The denominator is read off the same mask function the policy saw.

    Recomputing the mask must stay a pure read: it consumes no RNG, so a
    recorded episode is bit-identical to an unrecorded one (the invariant the
    whole trace pipeline rests on).
    """
    env = make_env("squad")
    plain = _seeded_episode(env, None, 4242)
    recorder = TraceRecorder()
    recorded = _seeded_episode(env, None, 4242, recorder=recorder)
    assert plain == recorded

    # ...and it recorded the menu the observation actually carried
    check = make_env("squad")
    obs, _ = check.reset(seed=4242)
    first = recorder.trace["steps"][0]
    for rec in first["soldiers"]:
        expected = order_options(np.asarray(obs[rec["cs"]]["action_mask"]))
        assert rec["order_opts"] == expected
