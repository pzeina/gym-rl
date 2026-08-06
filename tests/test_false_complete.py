"""False-COMPLETE pricing: done_false + ScenarioSpec.done_cooldown.

B2 measured 53-84% of DONE claims rejected as premature wherever DONE is
admissible. At +1.0/-0.5 that was *rational*, not a training failure: claiming
pays whenever ``p x done_true > (1-p) x |done_false|``, i.e. any p above one
third. v1.10 pulled both levers — the price (-0.5 → -2.0, break-even p > 0.67)
and a structural re-claim cooldown, since a rejected claim never cleared the
mission and DONE was admissible on every step.

**The hazard this docstring predicted is the one that fired.** Over-pricing a
speech act suppresses the HONEST one too — B5's re-task pricing did it to
initial tasking ("an order that is never issued cannot bind"), and -2.0 did it
to DONE. Under -2.0 the final-decile false-DONE rate fell to ~0 across squad,
squad_recon, squad_screen and fireteam, and the two report-centric scenarios
lost their terminal income outright: squad_recon_v6 and squad_screen_v4 both
ended at 0% with terminal 0.0000 and episodes pinned at max_steps. A cohort
that stops transmitting DONE never closes the grace window and never earns
root_done_bonus — the worse failure, as written here before it happened.

The price is back at -0.5. The structural half (done_cooldown) stays: it cannot
suppress an honest claim, only a repeated one. The standing constraint, pinned
below, is that a claim's break-even must not exceed the confidence an agent can
actually form — and RECON/SCREEN completion is team-adjudicated through
``_team_observe_steps``, which no observation slot carries.
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


def test_the_claiming_break_even_stays_inside_estimable_confidence():
    """The break-even is the thing that broke the fleet, so assert it directly.

    Raising it past ~0.5 asks the agent for a confidence it has no observation
    to form: a RECON/SCREEN root completes on the TEAM observation counter, and
    that counter is in no obs slot. Silence is then correct play, and silence
    scores 0. Move this bound only together with an obs slot for mission
    completion progress — the two are one change, not two.
    """
    cfg = RewardConfig()
    break_even = abs(cfg.done_false) / (cfg.done_true + abs(cfg.done_false))
    assert break_even <= 0.5, (
        f"claiming needs p>{break_even:.2f} — more confidence than the "
        "observation supports; this is what muted squad_recon_v6/squad_screen_v4"
    )
    assert cfg.done_true > 0, "honesty must still pay"
    assert cfg.done_false < 0, "a false claim must still cost"


def test_a_rejected_claim_is_priced_and_does_not_clear_the_mission():
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
