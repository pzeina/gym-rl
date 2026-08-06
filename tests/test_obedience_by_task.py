"""Pooled obedience latency conflates two different findings.

The defend line read 1.26 steps at v8 and 13.06 at v10, and I called that an
obedience regression caused by opening the root's MISSION COMPLETE. Split by
ordered task it is not one:

    v8    ADVANCE 1.01 (n=255)    DEFEND 0.68 (n=83)
    v10   ADVANCE 16.21 (n=286)   DEFEND 1.00 (n=40)

DEFEND — the mission the cohort is actually there to hold — barely moved. The
whole pooled rise lives in ADVANCE, whose share of orders also went 0.69 to
0.99. A mean over a changing task mix cannot separate "the cohort stopped
obeying" from "the cohort was ordered to do slower things", and an ADVANCE to a
distant control measure resolves slower than a DEFEND in place however obedient
the recipient is.

Same failure shape as issue #14's doctrine-preference metric: a pooled rate
that moves for a reason other than the one its name implies.
"""

from cohort.metrics import _obedience, aggregate_behavior, format_obedience_by_task


def _soldier(cs, mission, since, comp, alive=True):
    return {
        "cs": cs, "alive": alive, "pos": [0, 0], "mission": mission,
        "since": since, "auth": 0, "subs": [], "leader": None, "comp": comp,
        "cover": False, "fired": False, "sees": [], "formation": None,
        "done_ok": False, "root": False,
    }


def _step(t, soldiers):
    return {"t": t, "soldiers": soldiers, "enemies": [], "messages": [], "retasks": []}


def _trace(steps):
    return {
        "steps": steps,
        "outcome": "success",
        "length": len(steps),
        "root_mission": "DEFEND",
        "contact_refresh_age": 20,
        "knowledge_ttl": 40,
    }


def _one(cs, mission, comps, since=0):
    """A trace of one soldier holding `mission` with the given compliance series."""
    return _trace([_step(t, [_soldier(cs, mission, since, c)]) for t, c in enumerate(comps)])


def test_latency_is_booked_against_the_ordered_task():
    """A slow ADVANCE and a fast DEFEND must not average into one number."""
    trace = _trace([
        _step(0, [_soldier("RFN1", "ADVANCE", 0, 0.0), _soldier("RFN2", "DEFEND", 0, 0.9)]),
        _step(1, [_soldier("RFN1", "ADVANCE", 0, 0.0), _soldier("RFN2", "DEFEND", 0, 0.9)]),
        _step(2, [_soldier("RFN1", "ADVANCE", 0, 0.5), _soldier("RFN2", "DEFEND", 0, 0.9)]),
    ])
    latencies, censored, by_task = _obedience(trace)
    assert by_task["ADVANCE"]["latencies"] == [2]
    assert by_task["DEFEND"]["latencies"] == [1]  # scan starts at steps[1]
    assert censored == 0
    assert sorted(latencies) == [1, 2]
    # the pooled mean, 1.5, describes neither task
    assert sum(latencies) / len(latencies) == 1.5


def test_censored_orders_are_booked_by_task_too():
    """An order that never resolves must still be attributed."""
    _, censored, by_task = _obedience(_one("RFN1", "ADVANCE", [0.0, 0.0]))
    assert censored == 1
    assert by_task["ADVANCE"]["censored"] == 1
    assert by_task["ADVANCE"]["latencies"] == []


def test_aggregate_exposes_per_task_means_and_counts():
    from cohort.metrics import episode_behavior

    slow = _one("RFN1", "ADVANCE", [0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    fast = _one("RFN2", "DEFEND", [1.0, 1.0, 1.0])
    agg = aggregate_behavior([episode_behavior(slow), episode_behavior(fast)])
    by_task = agg["obedience_by_task"]
    assert by_task["ADVANCE"]["latency_mean"] == 5
    assert by_task["DEFEND"]["latency_mean"] == 1
    assert by_task["ADVANCE"]["orders"] == 1
    # the pooled mean sits between them and describes neither
    assert 1 < agg["obedience_latency_mean"] < 5


def test_formatter_ranks_by_order_volume_and_survives_all_censored():
    agg = {"obedience_by_task": {
        "ADVANCE": {"latency_mean": 16.21, "orders": 286, "censored": 106},
        "DEFEND": {"latency_mean": 1.0, "orders": 40, "censored": 0},
        "OBSERVE": {"latency_mean": None, "orders": 3, "censored": 3},
    }}
    line = format_obedience_by_task(agg)
    assert line.startswith("ADVANCE 16.2(286)")
    assert "DEFEND 1.0(40)" in line
    assert format_obedience_by_task({"obedience_by_task": {}}) == ""
    assert "OBSERVE —(3)" in format_obedience_by_task(agg, top=3)


def test_pooled_mean_is_unchanged_so_pinned_corpora_stay_comparable():
    """The split is additive: the old number must not move."""
    from cohort.metrics import episode_behavior

    agg = aggregate_behavior([episode_behavior(_one("RFN1", "ADVANCE", [0.0] * 4 + [1.0]))])
    assert agg["obedience_latency_mean"] == 4
    assert agg["obedience_orders"] == 1
