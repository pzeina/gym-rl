"""A4 comms discipline: transmission cost, net-busy arbitration, dedup credit.

The net is a single frequency: at most one LEARNED transmission (CONTACT /
DONE / order / SITREP, in that priority) per tick; losers are dropped with a
NET BUSY outcome (no cost, no effect). Every emitted learned transmission
costs ``RewardConfig.transmission_cost``; auto-traffic (WILCO, verdicts,
CASUALTY, succession) stays free. CONTACT credit is deduplicated: first
accurate report pays, an aging-intel refresh is exactly 0, pure noise draws
the small redundant penalty. Spaces are frozen — v1.4 checkpoints stay valid.
"""

import pytest

from cohort import make_env
from cohort.core.missions import MissionType
from cohort.env.actions import CATALOG, N_ACTIONS
from cohort.env.observations import OBS_DIM

STAY = 0
CONTACT = next(s.index for s in CATALOG if s.kind == "contact")
SITREP = next(s.index for s in CATALOG if s.kind == "sitrep")
DONE = next(s.index for s in CATALOG if s.kind == "done")
ORDER_SEIZE_S0 = next(
    s.index
    for s in CATALOG
    if s.kind == "order"
    and s.order_slot == 0
    and s.order_mission is MissionType.SEIZE
    and s.order_objective == "ALPHA"
)


def _flat_env(seed=1):
    """Fireteam env on open terrain with enemies parked far away."""
    env = make_env("fireteam")
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (22, 1)
        e.home = e.pos
    return env


def _step_all(env, overrides=None):
    acts = {a: STAY for a in env.agents}
    acts.update(overrides or {})
    return env.step(acts)


def _put_enemy_in_view(env, callsign, enemy_idx=0):
    sld = env.roster.by_callsign[callsign]
    enemy = env.enemies[enemy_idx]
    enemy.pos = (sld.pos[0] + 3, sld.pos[1])
    enemy.home = enemy.pos
    return enemy


# ---------------------------------------------------------------------- #
# frozen spaces
# ---------------------------------------------------------------------- #


def test_spaces_are_frozen_at_v14():
    """Spaces at the degraded-communications layout (authorized breaking
    cycle, docs/degraded-communications.md §5): the v1.10 Discrete(228) /
    Box(220) plus the appended acoustic-report and gesture actions and the
    acoustic + cohesion observation blocks. Phase C appends the liaison
    actions and message block on top; the first 228 indices never move
    (tests/test_degraded_regression.py pins them)."""
    assert N_ACTIONS == 231
    assert OBS_DIM == 328


# ---------------------------------------------------------------------- #
# transmission cost
# ---------------------------------------------------------------------- #


def test_sitrep_charges_transmission_cost():
    env = _flat_env()
    cfg = env.rewards_cfg
    *_, infos = _step_all(env, {"RFN1": SITREP})
    # the first SITREP is fresh (no prior report): fresh credit + airtime
    assert infos["RFN1"]["components"]["report"] == pytest.approx(
        cfg.sitrep_fresh + cfg.transmission_cost
    )
    assert env.transmissions_last_step == 1
    # an immediate repeat is spam — and still pays airtime
    *_, infos = _step_all(env, {"RFN1": SITREP})
    assert infos["RFN1"]["components"]["report"] == pytest.approx(
        cfg.sitrep_spam + cfg.transmission_cost
    )


def test_contact_charges_transmission_cost():
    env = _flat_env()
    cfg = env.rewards_cfg
    _put_enemy_in_view(env, "RFN1")
    *_, infos = _step_all(env, {"RFN1": CONTACT})
    assert infos["RFN1"]["components"]["report"] == pytest.approx(
        cfg.contact_new + cfg.transmission_cost
    )


def test_done_charges_transmission_cost():
    env = _flat_env()
    cfg = env.rewards_cfg
    # TL1 holds the OPORD (SEIZE ALPHA) far from done: a false claim, still airtime
    *_, infos = _step_all(env, {"TL1": DONE})
    assert infos["TL1"]["components"]["report"] == pytest.approx(
        cfg.done_false + cfg.transmission_cost
    )


def test_order_charges_transmission_cost_and_ack_is_free():
    env = _flat_env()
    cfg = env.rewards_cfg
    *_, infos = _step_all(env, {"TL1": ORDER_SEIZE_S0})
    # preferred derivation + objective match + coverage gap (2 untasked) + airtime
    assert infos["TL1"]["components"]["command"] == pytest.approx(
        cfg.order_preferred + cfg.order_objective_match + cfg.coverage_gap + cfg.transmission_cost
    )
    # the auto-WILCO is protocol: the recipient pays nothing
    assert infos["RFN1"]["components"]["report"] == 0.0
    assert infos["RFN1"]["components"]["command"] == 0.0
    kinds = [m.kind.value for m in env.transcript.messages]
    assert "ack" in kinds
    assert env.transmissions_last_step == 1, "the WILCO is not a learned transmission"


# ---------------------------------------------------------------------- #
# net-busy arbitration
# ---------------------------------------------------------------------- #


def test_one_transmission_per_tick_contact_beats_sitrep():
    env = _flat_env()
    _put_enemy_in_view(env, "RFN2")
    # RFN1 (earlier in agent order) tries SITREP; RFN2's CONTACT outranks it
    *_, infos = _step_all(env, {"RFN1": SITREP, "RFN2": CONTACT, "RFN3": SITREP})
    kinds = [m.kind.value for m in env.transcript.messages if m.step == env._step_count]
    assert kinds.count("contact") == 1
    assert kinds.count("sitrep") == 0, "one transmission per net per tick"
    assert env.transmissions_last_step == 1
    assert infos["RFN2"]["net_busy"] is False
    assert infos["RFN1"]["net_busy"] is True
    assert infos["RFN3"]["net_busy"] is True
    # NET BUSY: no cost, no effect for the losers
    assert infos["RFN1"]["components"]["report"] == 0.0
    assert env.roster.by_callsign["RFN1"].last_sitrep_step < 0, "blocked SITREP never sent"


def test_tie_breaks_by_agent_order_deterministically():
    env = _flat_env()
    *_, infos = _step_all(env, {"RFN1": SITREP, "RFN2": SITREP})
    assert infos["RFN1"]["net_busy"] is False
    assert infos["RFN2"]["net_busy"] is True
    sitreps = [m for m in env.transcript.messages if m.kind.value == "sitrep"]
    assert len(sitreps) == 1
    assert sitreps[0].text.split(": ")[0].endswith("RFN1"), "earlier agent wins the tie"


def test_contact_beats_order_and_blocked_order_has_no_effect():
    env = _flat_env()
    cfg = env.rewards_cfg
    _put_enemy_in_view(env, "RFN2")
    *_, infos = _step_all(env, {"TL1": ORDER_SEIZE_S0, "RFN2": CONTACT})
    rfn1 = env.roster.by_callsign["RFN1"]
    assert rfn1.mission is None, "the blocked ORDER assigns nothing"
    kinds = [m.kind.value for m in env.transcript.messages]
    assert "order" not in kinds and "ack" not in kinds
    assert infos["TL1"]["net_busy"] is True
    # no churn, no airtime — only the standing coverage gap remains
    assert infos["TL1"]["components"]["command"] == pytest.approx(cfg.coverage_gap)


def test_single_transmitter_is_never_blocked():
    env = _flat_env()
    *_, infos = _step_all(env, {"RFN1": SITREP})
    assert infos["RFN1"]["net_busy"] is False
    assert env.transmissions_last_step == 1


def test_illegal_attempt_does_not_contend():
    env = _flat_env()
    # RFN1 sees no enemy: its CONTACT is illegal (→ STAY) and must not block RFN2
    *_, infos = _step_all(env, {"RFN1": CONTACT, "RFN2": SITREP})
    assert infos["RFN2"]["net_busy"] is False
    assert [m.kind.value for m in env.transcript.messages if m.step == 1] == ["sitrep"]


def test_net_busy_is_surfaced_in_the_oracle():
    env = _flat_env()
    _step_all(env, {"RFN1": SITREP, "RFN2": SITREP})
    snap = env.oracle()
    by_cs = {s["cs"]: s for s in snap["soldiers"]}
    assert by_cs["RFN1"]["net_busy"] is False
    assert by_cs["RFN2"]["net_busy"] is True
    assert by_cs["TL1"]["net_busy"] is False


def test_arbitration_is_deterministic_across_reruns():
    def transcript_of():
        env = _flat_env(seed=7)
        _put_enemy_in_view(env, "RFN2")
        for _ in range(3):
            _step_all(env, {"RFN1": SITREP, "RFN2": CONTACT, "TL1": ORDER_SEIZE_S0})
        return env.transcript.render()

    assert transcript_of() == transcript_of()


# ---------------------------------------------------------------------- #
# dedup credit
# ---------------------------------------------------------------------- #


def test_first_report_pays_immediate_rereport_is_noise():
    env = _flat_env()
    cfg = env.rewards_cfg
    enemy = _put_enemy_in_view(env, "RFN1")
    *_, infos = _step_all(env, {"RFN1": CONTACT})
    assert infos["RFN1"]["components"]["report"] == pytest.approx(
        cfg.contact_new + cfg.transmission_cost
    )
    # next tick: same enemy, still fresh on the picture — pure noise
    *_, infos = _step_all(env, {"RFN1": CONTACT})
    assert infos["RFN1"]["components"]["report"] == pytest.approx(
        cfg.contact_redundant + cfg.transmission_cost
    )
    # ...but the picture entry was still refreshed
    assert env._known_enemies[enemy.id][2] == env._step_count


def test_rereport_of_aging_intel_is_exactly_zero_and_refreshes():
    env = _flat_env()
    cfg = env.rewards_cfg
    enemy = _put_enemy_in_view(env, "RFN1")
    _step_all(env, {"RFN1": CONTACT})
    enemy.pos = (22, 1)  # break contact while the intel ages (TTL keeps it)
    enemy.home = enemy.pos
    enemy.last_seen_player = None  # forget the sighting: no chase during aging
    for _ in range(cfg.contact_refresh_age):  # let the entry age past the threshold
        _step_all(env)
    _put_enemy_in_view(env, "RFN1")  # the enemy reappears where it can be seen
    *_, infos = _step_all(env, {"RFN1": CONTACT})
    assert infos["RFN1"]["components"]["report"] == pytest.approx(cfg.transmission_cost), (
        "an aging-intel refresh earns exactly 0 report credit (airtime only)"
    )
    assert env._known_enemies[enemy.id][2] == env._step_count, "the picture is refreshed"


def test_report_with_any_unknown_enemy_still_pays_full():
    env = _flat_env()
    cfg = env.rewards_cfg
    _put_enemy_in_view(env, "RFN1", enemy_idx=0)
    _step_all(env, {"RFN1": CONTACT})
    # a SECOND, unknown enemy walks into view next to the known one
    _put_enemy_in_view(env, "RFN1", enemy_idx=1)
    *_, infos = _step_all(env, {"RFN1": CONTACT})
    assert infos["RFN1"]["components"]["report"] == pytest.approx(
        cfg.contact_new + cfg.transmission_cost
    )


def test_refresh_age_is_below_knowledge_ttl():
    from cohort.env.cohort_env import KNOWLEDGE_TTL
    from cohort.env.rewards import RewardConfig

    assert RewardConfig().contact_refresh_age < KNOWLEDGE_TTL, (
        "refreshes must be possible while the entry is still on the picture"
    )
