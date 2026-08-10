"""``root_done_bonus`` belongs to the episode's FIRST root claim (v1.15).

**What was being farmed.** v1.14 reopened MISSION COMPLETE to horizon-DEFEND
roots, and ``defend_brique_v11``/``ckpt_latest`` immediately filed **321 root
claims in 100 episodes, 227 of them rejected** — a false-complete rate of 0.71
(``runs/defend_brique_v11/behavior_final.json``). That is not noise, it is the
arithmetic doing what it was told: an accepted claim pays ``done_true`` 1.0 +
``root_done_bonus`` 3.0 = 4.0 while a rejected one costs ``done_false`` -0.5, so
probing breaks even at p > 1/9 and, at v11's realised acceptance of 94/321 =
0.293, earned about **+262 per 100 episodes**. ``ScenarioSpec.done_cooldown``
(8) rate-limits the retries and never touches their sign — the same shape as the
pre-v1.10 re-roll exploit the cooldown was built against.

**The rule, and the half that carries it.** The bonus is on the table once per
episode, and *the first claim consumes it whether or not it is accepted*. A
rejected first claim burns the bonus. Were a rejection to leave the slot open,
nothing would change: spam until one lands, collect on "the first ACCEPTED
claim". Consuming it either way is what turns the first claim into a judgement
instead of a probe — with the slot spent, a further claim is worth 1.0 / -0.5,
break-even moves to p > 0.333, and probing at 0.293 goes negative.

**What must NOT change**, and is pinned below as hard as the mechanism: an
honest claim still pays ``done_true``, a truthful claim still closes the window
and ends the operation, a root that never claims is untouched, and subordinate
DONEs — which never earned this bonus — are not in scope. The ``done_false=-2.0``
episode in ``rewards.py`` is the standing warning: a price that buys silence
instead of precision loses the channel, and a policy that simply stops claiming
has not passed this test.

**v1.16: the default is REVERTED to False (owner's decision), and that warning
is why.** A confirmed root claim ends the episode, so at most one claim per
episode is ever confirmed and it is necessarily the last — which prices the
first probe at ``done_false - root_done_bonus x P(a later claim closes)``, and
that P measured 1.000 on ``defend_brique_v11``. -3.50, not -0.5. The measured
result was ``defend_brique_v12``: 321 root claims -> 0, and P(DONE | a true
claim is available) 0.401 -> 0.000083. So these tests now run the mechanism
under an EXPLICIT ``FIRST_CLAIM_RULE`` config, and the shipped default is
pinned separately as the pre-v1.15 rule. Nothing about the mechanism is
deleted: it is a measured price with a known effect, kept ready to revisit.
"""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from cohort import get_scenario, make_env
from cohort.core.orders import MessageKind
from cohort.env.actions import CATALOG
from cohort.env.rewards import RewardConfig

STAY = 0
DONE = next(s.index for s in CATALOG if s.kind == "done")

#: The v1.15 mechanism, switched on explicitly — it is no longer the default.
FIRST_CLAIM_RULE = RewardConfig(root_done_bonus_first_claim_only=True)


def _step_all(env, overrides=None):
    acts = {a: STAY for a in env.agents}
    acts.update(overrides or {})
    return env.step(acts)


def _seize_env(reward_config=None, seed=1):
    """fireteam (SEIZE root, TL1): enemies dead, flat ground, ALPHA unoccupied.

    The success condition is one body away, so a claim made now is FALSE and a
    claim made after TL1 steps onto ALPHA is TRUE — which is the whole probe /
    judgement distinction, in the fleet's simplest completable root.
    """
    env = make_env("fireteam", reward_config=reward_config)
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.alive = False
    for s in env.roster.soldiers:
        s.pos = (3, 3)
    return env


def _claim_true(env, actor="TL1"):
    """Put the root on the objective and claim; returns the step's infos."""
    env.roster.by_callsign[actor].pos = env.world.objectives[0].pos
    return _step_all(env, {actor: DONE})[-1]


def _wait_out_cooldown(env):
    """``done_cooldown`` masks a re-claim; a rejected root has to sit it out."""
    for _ in range(env.spec_cfg.done_cooldown):
        _step_all(env)


def _bonus_paid(infos, env, root="TL1", other="RFN1"):
    """Did the closer collect ``root_done_bonus`` on top of the shared terminal?

    Read as a difference against a subordinate's terminal rather than against a
    constant: the defend terminal is survivor-scaled, so the shared part is not
    a number this test may hard-code.
    """
    gap = infos[root]["components"]["terminal"] - infos[other]["components"]["terminal"]
    return gap == pytest.approx(env.rewards_cfg.root_done_bonus)


# --------------------------------------------------------------------- #
# The mechanism
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("cfg", [None, FIRST_CLAIM_RULE], ids=["default", "first_claim"])
def test_an_accepted_first_claim_is_paid_in_full(cfg):
    """The honest act is untouched by the rule, in either position of the flag:
    claim once, truthfully, collect 4.0."""
    env = _seize_env(reward_config=cfg)
    infos = _claim_true(env)
    assert env.outcome == "success"
    cfg = env.rewards_cfg
    assert infos["TL1"]["components"]["report"] == pytest.approx(
        cfg.done_true + cfg.transmission_cost
    )
    assert _bonus_paid(infos, env), "a first, true claim must still earn the bonus"


def test_a_rejected_first_claim_burns_the_bonus_for_the_episode():
    """THE test. The later claim is true, is confirmed, ends the operation —
    and is paid ``done_true`` and nothing else, because the probe spent the slot.

    If this ever asserts a paid bonus again, the exploit is back in full: the
    rejection cost -0.5, the eventual acceptance would repay 4.0, and spamming
    the channel would be worth +0.81 a claim exactly as it was in v11.
    """
    env = _seize_env(reward_config=FIRST_CLAIM_RULE)
    *_, infos = _step_all(env, {"TL1": DONE})  # ALPHA unoccupied: rejected
    cfg = env.rewards_cfg
    assert env.outcome is None
    assert env._root_claim_filed, "the probe did not register as a claim"
    assert infos["TL1"]["components"]["report"] == pytest.approx(
        cfg.done_false + cfg.transmission_cost
    )

    _wait_out_cooldown(env)
    infos = _claim_true(env)

    assert env.outcome == "success", "the true claim must still close the operation"
    assert MessageKind.DONE_CONFIRM in [m.kind for m in env.transcript.messages]
    assert infos["TL1"]["components"]["report"] == pytest.approx(
        cfg.done_true + cfg.transmission_cost
    ), "honest claiming must keep paying — the point is not to mute the channel"
    assert not _bonus_paid(infos, env), "the rejected probe did not burn the bonus"
    assert infos["TL1"]["components"]["terminal"] == pytest.approx(
        infos["RFN1"]["components"]["terminal"]
    )


def test_the_second_claim_is_no_better_off_than_the_third():
    """Two probes, then the truth: the slot is spent once, not per claim."""
    env = _seize_env(reward_config=FIRST_CLAIM_RULE)
    for _ in range(2):
        _step_all(env, {"TL1": DONE})
        assert env.outcome is None
        _wait_out_cooldown(env)
    infos = _claim_true(env)
    assert env.outcome == "success"
    assert not _bonus_paid(infos, env)


def test_a_root_that_never_claims_is_unaffected():
    """No claim, no slot spent, no behaviour change: success at T0 + grace with
    the identical shared terminal, exactly as before v1.15."""
    env = _seize_env()
    env.roster.by_callsign["TL1"].pos = env.world.objectives[0].pos
    infos = None
    while env.agents:
        *_, infos = _step_all(env)
    assert env.outcome == "success"
    assert not env._root_claim_filed
    assert env._root_close_step is None, "nobody reported; nothing closed early"
    assert infos["TL1"]["components"]["terminal"] == pytest.approx(
        infos["RFN1"]["components"]["terminal"]
    ), "the bonus was never earned by anyone, under either rule"


def test_a_subordinate_claim_does_not_spend_the_root_slot():
    """Scope: the ROOT's channel. A rifleman's DONE never earned this bonus and
    must not consume it — otherwise a subordinate could burn its commander's."""
    env = _seize_env(reward_config=FIRST_CLAIM_RULE)
    env.inject_order("RFN1, seize obj alpha", issuer="TL1")
    rfn = env.roster.by_callsign["RFN1"]
    assert rfn.mission is not None
    _step_all(env, {"RFN1": DONE})  # RFN1 is at (3,3): its own SEIZE is not done
    assert MessageKind.DONE_REJECT in [m.kind for m in env.transcript.messages]
    assert not env._root_claim_filed, "a subordinate's claim spent the root's slot"

    infos = _claim_true(env)
    assert env.outcome == "success"
    assert _bonus_paid(infos, env), "the root's first claim was still its first"


# --------------------------------------------------------------------- #
# The flag
# --------------------------------------------------------------------- #


def test_the_shipped_default_is_the_pre_v115_rule():
    """v1.16: the same script as the burns-the-bonus test, run on what the fleet
    actually trains under — and there the probe is repaid in full.

    This is the assertion that would have failed silently if the revert had
    touched only the flag's documentation: an arm that says "old economics" in
    ``economics.json`` and prices the first claim at -3.50 anyway."""
    env = _seize_env()
    _step_all(env, {"TL1": DONE})
    assert env.outcome is None
    _wait_out_cooldown(env)
    infos = _claim_true(env)
    assert env.outcome == "success"
    assert _bonus_paid(infos, env), "the default must be pre-v1.15 behaviour"


def test_the_flag_is_off_by_default_and_settable_from_the_cli():
    """The rule has to stay reproducible from ``--reward``, and
    ``economics.json`` (an ``asdict`` of this config) has to record which arm
    a run was — a bool read by truthiness would silently train the other one."""
    assert RewardConfig().root_done_bonus_first_claim_only is False
    on = RewardConfig.from_overrides(["root_done_bonus_first_claim_only=true"])
    assert on.root_done_bonus_first_claim_only is True
    assert asdict(on)["root_done_bonus_first_claim_only"] is True
    baseline = asdict(RewardConfig())
    changed = {k: v for k, v in asdict(on).items() if baseline[k] != v}
    assert changed == {"root_done_bonus_first_claim_only": True}
    env = make_env("fireteam", reward_config=on)
    assert env.rewards_cfg.root_done_bonus_first_claim_only is True


# --------------------------------------------------------------------- #
# The scenario the exploit was measured on
# --------------------------------------------------------------------- #


def _horizon_defend_env(reward_config=None, seed=12):
    """fireteam_defend past H, root on the objective, the band still alive —
    i.e. the exact state ``defend_brique_v11`` filed 227 rejected claims from."""
    env = make_env("fireteam_defend", reward_config=reward_config)
    env.reset(seed=seed)
    env._h_hour = 0  # the preparation period is over; the criterion is live
    root = env.roster.root()
    root.pos = env.world.objective_by_name(env.spec_cfg.root_objective).pos
    return env, root


def test_the_horizon_defend_probe_pays_for_itself_only_once():
    """The fleet-wide rule, exercised where the exploit was actually measured."""
    env, root = _horizon_defend_env(reward_config=FIRST_CLAIM_RULE)
    _step_all(env, {root.callsign: DONE})  # band alive, hour not up: rejected
    assert MessageKind.DONE_REJECT in [m.kind for m in env.transcript.messages]
    assert env.outcome is None
    assert env._root_claim_filed

    for e in env.enemies:  # the position is now released; the claim becomes true
        e.alive = False
    _wait_out_cooldown(env)
    other = next(s.callsign for s in env.roster.living if s is not root)
    *_, infos = _step_all(env, {root.callsign: DONE})

    assert env.outcome == "success", "the true claim must still close the defense"
    assert env.rewards_cfg.done_true == pytest.approx(1.0)
    gap = (
        infos[root.callsign]["components"]["terminal"]
        - infos[other]["components"]["terminal"]
    )
    assert gap == pytest.approx(0.0), "the spent slot did not stay spent"


def test_an_indefinite_defense_still_pays_its_endex_close():
    """The v1.13 route files no claim at all, so it cannot spend a slot. Pinned
    because losing the ENDEX bonus to this change would be silent: the root's
    SITREP closes the window and the payout is the only thing that says so."""
    spec = replace(get_scenario("fireteam_defend"), defend_horizon=None)
    env = make_env(spec)
    env.reset(seed=12)
    for e in env.enemies:
        e.alive = False
    root = env.roster.root()
    root.pos = env.world.objective_by_name(env.spec_cfg.root_objective).pos
    other = next(s.callsign for s in env.roster.living if s is not root)
    sitrep = next(s.index for s in CATALOG if s.kind == "sitrep")
    *_, infos = _step_all(env, {root.callsign: sitrep})

    assert env.outcome == "success"
    assert not env._root_claim_filed, "ENDEX closure files no MISSION COMPLETE"
    gap = (
        infos[root.callsign]["components"]["terminal"]
        - infos[other]["components"]["terminal"]
    )
    assert gap == pytest.approx(env.rewards_cfg.root_done_bonus)


def test_checkpoints_from_either_era_reconstruct_as_the_rule_they_trained_under():
    """``evaluate`` rebuilds the config with ``RewardConfig(**ckpt[
    "reward_config"])``, so what an old checkpoint loads AS is decided by this
    default. Both directions matter, and v1.16 makes them agree:

    * a **v1.14** dict has no key for this flag, so it falls to the default —
      which is once again False, i.e. the rule that run actually trained under.
      The v1.15 default silently re-scored those runs under a price they had
      never seen; that is why their published claim numbers went stale.
    * a **v1.15** dict carries the key explicitly (it was written by ``asdict``
      of a config with the flag on), so it keeps its own rule through the
      revert. An arm of a landed A/B must not change era underneath it.
    """
    v114 = {k: v for k, v in asdict(RewardConfig()).items()
            if k != "root_done_bonus_first_claim_only"}
    restored = RewardConfig(**v114)
    assert restored.root_done_bonus_first_claim_only is False, (
        "a pre-v1.15 checkpoint must reconstruct under its own era's rule"
    )
    assert asdict(restored) == asdict(RewardConfig())

    v115 = asdict(FIRST_CLAIM_RULE)
    assert RewardConfig(**v115).root_done_bonus_first_claim_only is True, (
        "a v1.15 arm must keep the rule it trained with"
    )
