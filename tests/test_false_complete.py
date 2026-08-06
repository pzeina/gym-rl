"""False-COMPLETE pricing (v1.10): done_false + ScenarioSpec.done_cooldown.

B2 measured 53-84% of DONE claims rejected as premature wherever DONE is
admissible. At the shipped prices that was *rational*, not a training failure:
claiming pays whenever ``p x done_true > (1-p) x |done_false|``, which at
+1.0/-0.5 means any p above one third. Two levers answer it — the price
(break-even moves to p > 0.67) and a structural re-claim cooldown, since a
rejected claim never cleared the mission and DONE was admissible on every step.

The regression hazard runs the OTHER way, and is encoded below: over-pricing a
speech act suppresses the HONEST one too. That is exactly what B5's re-task
pricing did to initial tasking ("an order that is never issued cannot bind"),
and it cost a retrain to undo. A cohort that stops transmitting DONE never
closes the grace window and never earns root_done_bonus — a worse failure than
one that over-claims.
"""

from dataclasses import replace

from cohort import make_env
from cohort.config import get_scenario
from cohort.core.missions import Mission, MissionType
from cohort.env.actions import CATALOG
from cohort.env.rewards import RewardConfig

STAY = 0
DONE = next(s.index for s in CATALOG if s.kind == "done")


def _flat_env(spec="fireteam", seed=1):
    env = make_env(spec)
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (1, 22)
        e.home = e.pos
    return env


def _step(env, overrides=None):
    acts = {a: STAY for a in env.agents}
    acts.update(overrides or {})
    return env.step(acts)


def _tasked_far_from_bravo(env):
    sld = env.roster.by_callsign["RFN1"]
    obj = env.world.objectives[1]  # BRAVO
    sld.pos = (2, 2)
    sld.mission = Mission(MissionType.SEIZE, 1, obj.pos, issuer_id=-1, step_assigned=0)
    return sld, obj


# ---------------------------------------------------------------------- #
# the price
# ---------------------------------------------------------------------- #


def test_claiming_needs_two_thirds_confidence_to_pay():
    """The break-even is the thing being changed, so assert it directly."""
    cfg = RewardConfig()
    break_even = abs(cfg.done_false) / (cfg.done_true + abs(cfg.done_false))
    assert break_even > 0.6, f"claiming is +EV from p={break_even:.2f} — too cheap"
    assert cfg.done_true > 0, "honesty must still pay"


def test_false_complete_costs_more_than_it_did():
    env = _flat_env()
    sld, _ = _tasked_far_from_bravo(env)
    *_, infos = _step(env, {"RFN1": DONE})
    assert infos["RFN1"]["components"]["report"] <= env.rewards_cfg.done_false
    assert sld.mission is not None, "a rejected claim does not clear the mission"


# ---------------------------------------------------------------------- #
# the cooldown
# ---------------------------------------------------------------------- #


def test_rejected_claim_cannot_be_re_rolled_every_step():
    env = _flat_env()
    sld, _ = _tasked_far_from_bravo(env)
    obs, *_ = _step(env, {"RFN1": DONE})
    cooldown = env.spec_cfg.done_cooldown
    assert cooldown > 0
    # the rejection lands on step R; DONE stays masked while step - R < cooldown
    for _ in range(cooldown):
        assert obs["RFN1"]["action_mask"][DONE] == 0, "DONE masked during the cooldown"
        obs, *_ = _step(env)
        sld.pos = (2, 2)
    assert obs["RFN1"]["action_mask"][DONE] == 1, "and legal again once it lapses"


def test_the_cooldown_delays_the_honest_claim_but_never_denies_it():
    """The muteness hazard: an honest DONE must still be reachable."""
    env = _flat_env()
    sld, obj = _tasked_far_from_bravo(env)
    _step(env, {"RFN1": DONE})  # premature: rejected, cooldown opens
    sld.pos = obj.pos           # now genuinely complete
    for _ in range(env.spec_cfg.done_cooldown):
        _step(env)
        sld.pos = obj.pos
    *_, infos = _step(env, {"RFN1": DONE})
    assert infos["RFN1"]["components"]["report"] > 0, "the honest claim still pays"
    assert sld.mission is None, "and still clears the mission"


def test_first_claim_is_never_rate_limited():
    """Only the RETRY is limited — the honest first claim pays immediately, so
    the cooldown cannot be what silences a cohort."""
    env = _flat_env()
    sld = env.roster.by_callsign["RFN1"]
    obj = env.world.objectives[1]
    sld.pos = obj.pos
    sld.mission = Mission(MissionType.SEIZE, 1, obj.pos, issuer_id=-1, step_assigned=0)
    obs = env._all_observations()
    assert obs["RFN1"]["action_mask"][DONE] == 1
    *_, infos = _step(env, {"RFN1": DONE})
    assert infos["RFN1"]["components"]["report"] > 0


def test_cooldown_off_restores_the_pre_v110_behavior():
    env = _flat_env(replace(get_scenario("fireteam"), done_cooldown=0))
    sld, _ = _tasked_far_from_bravo(env)
    obs, *_ = _step(env, {"RFN1": DONE})
    assert obs["RFN1"]["action_mask"][DONE] == 1, "0 → off: re-claim immediately legal"
    assert sld.mission is not None


def test_every_scenario_carries_the_cooldown():
    from cohort.config import SCENARIOS

    for name, spec in SCENARIOS.items():
        assert spec.done_cooldown > 0, f"{name}: false-COMPLETE spam left unpriced"
