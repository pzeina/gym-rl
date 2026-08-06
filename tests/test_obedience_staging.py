"""A staged order is not an obeyed order (refs issue #15).

Cycle 8 blamed the within-task ADVANCE latency rise (1.01 → 16.21) on AT MY
COMMAND staging being counted as disobedience. An outside tap refuted it from
the net: v8 staged 0.878 of its ADVANCE orders and held them 44.4 steps, v10
staged 0.369 and held them 20.7 — the checkpoint that stages *more* and
*longer* measured 16x LOWER latency. Incidence and duration both ran backwards
against the hypothesis.

They ran backwards because the sign was backwards. The environment scores a
pending order as HOLD at the staging spot — where the recipient already
stands — so a staged agent's compliance is positive from the tick the order
lands, and the metric read that as "obeyed in 0 steps". Release then restamps
``step_assigned``, booking the same order a second time, so every staged order
donated a free zero and the mean fell in proportion to how much a policy
staged. The two tests below are the mechanism at its smallest: the same
ADVANCE, to a recipient that never moves, must not read as instant obedience
merely because it was staged.
"""

from cohort import make_env
from cohort.core.missions import MissionType
from cohort.env.actions import CATALOG
from cohort.metrics import (
    TraceRecorder,
    _obedience,
    _staging,
    aggregate_behavior,
    episode_behavior,
    format_staging,
)

STAY = 0
EXECUTE_IDX = next(s.index for s in CATALOG if s.kind == "execute")


def _advance_order_idx(amc: bool) -> int:
    return next(
        s.index
        for s in CATALOG
        if s.kind == "order"
        and s.order_mission is MissionType.ADVANCE
        and s.order_slot == 0
        and s.order_control == "GOLD"
        and bool(s.order_amc) is amc
    )


def _episode(*, amc: bool, execute_after: int | None, steps: int = 20) -> dict:
    """One fireteam episode: TL1 orders RFN1 to ADVANCE, everyone stands still.

    Nobody moves, so the ADVANCE is never executed by anyone — the only
    difference between the arms is the AT MY COMMAND qualifier and whether the
    EXECUTE that releases it is ever sent.
    """
    env = make_env("fireteam")
    env.reset(seed=3)
    rec = TraceRecorder()
    rec.on_reset(env)

    def tick(tl_action: int = STAY) -> None:
        rec.before_step(env)
        actions = dict.fromkeys(env.agents, STAY)
        if "TL1" in actions:
            actions["TL1"] = tl_action
        env.step(actions)
        rec.after_step(env)

    tick(_advance_order_idx(amc))
    for i in range(1, steps):
        if not env.agents:
            break
        tick(EXECUTE_IDX if execute_after is not None and i == execute_after else STAY)
    return rec.trace


def _advance(trace: dict) -> dict:
    _, _, by_task = _obedience(trace)
    return by_task.get("ADVANCE", {"latencies": [], "censored": 0})


def test_staged_advance_does_not_read_as_instant_obedience():
    """The bug, at its smallest: staging turned "never moved" into latency 0."""
    plain = _advance(_episode(amc=False, execute_after=None))
    staged = _advance(_episode(amc=True, execute_after=6))
    # the recipient stands still in both arms, so neither ADVANCE is ever obeyed
    assert plain["latencies"] == [], "control: an unmoved recipient never resolves"
    assert staged["latencies"] == [], (
        "a staged ADVANCE resolved instantly — compliance while pending is HOLD "
        "at the staging spot, not progress toward the ordered control measure"
    )
    # and staging must not double-book the order either: one order, one event
    assert staged["censored"] == plain["censored"] == 1


def test_staging_is_counted_where_obedience_cannot_see_it():
    released = _staging(_episode(amc=True, execute_after=6))
    assert released["orders_staged"] == 1
    assert released["staged_released"] == 1
    assert released["staging_gaps"] == [6], "order landed at t=1, EXECUTE at t=7"


def test_an_order_staged_and_never_released_is_abandoned_not_obeyed():
    """61 of one checkpoint's 130 staged orders never saw an EXECUTE."""
    trace = _episode(amc=True, execute_after=None)
    assert _advance(trace)["latencies"] == [], "never released, never binding"
    assert _staging(trace) == {"orders_staged": 1, "staged_released": 0, "staging_gaps": []}
    agg = aggregate_behavior([episode_behavior(trace)])
    assert agg["orders_staged"] == 1 and agg["staged_released"] == 0
    assert agg["staged_abandoned"] == 1
    assert "abandoned 1" in format_staging(agg)


def test_recorder_scores_a_pending_mission_the_way_the_environment_pays_it():
    """RECON in position pays 0.6; HOLD in position pays 0.5. A staged RECON is
    paid as HOLD by the environment, so the trace must record 0.5."""
    env = make_env("fireteam")
    env.reset(seed=5)
    rec = TraceRecorder()
    rec.on_reset(env)
    env.inject_order("RFN1, recon obj bravo at t plus 10", issuer="HQ")
    rec.before_step(env)
    env.step(dict.fromkeys(env.agents, STAY))
    rec.after_step(env)
    r = next(s for s in rec.trace["steps"][-1]["soldiers"] if s["cs"] == "RFN1")
    assert r["pending"] is True
    assert r["mission"] == "RECON"
    assert r["comp"] == 0.5, "staged: scored as HOLD at the staging spot, not as RECON"
