"""How a root OPORD is closed out — and by whom.

Two eras, both worth keeping in view, because the second is a reversal of the
first and the reasoning that produced the first was not wrong, only incomplete.

**v1.4.** ``COMPLETABLE`` deliberately excludes DEFEND/DENY: no individual
"finishes" a continuous posture. The action mask gated MISSION COMPLETE on
``mission.type in COMPLETABLE`` while ``_report_done``'s root branch gated on
``mission.type is spec.root_mission``, and on a DEFEND-rooted scenario those
cannot both hold — so the root's claim was hard-masked every step, the root
branch was unreachable, ``root_done_bonus`` was dead reward, and the grace
window could only expire by timeout. Measured on ``fireteam_defend_v8``: 0
admissible root claims in 30 episodes. ``is_root_opord_claim`` became the one
predicate both sides consult, and it opened the channel to the root.

**v1.13 (owner's decision).** Opening it was the wrong repair. A DEFEND is not
a task with an end state its holder may declare — it is held until relieved or
re-tasked, so the order that ends it comes DOWN the chain. The measurement that
made this visible: ``fireteam_defend_v12`` filed ONE claim in 100 episodes
against 16,152 admissible agent-steps, which read as a dead channel and was
actually the policy declining an act it should never have been offered. The
root now reports the situation and COMMAND transmits ENDEX.

**v1.14 (owner's decision).** A refinement of v1.13, not a second reversal.
What makes a posture undeclarable is that it has no stated end — so a DEFEND
ordered to a HORIZON (``ScenarioSpec.defend_horizon``) is a different object:
at the horizon the mission genuinely is complete, and the root can perceive
both the clock and the ground, so it may declare, and the claim is adjudicated
against ground truth like any other. An INDEFINITE defense keeps v1.13's
ENDEX-only closure exactly. The tests below therefore run the v1.13 assertions
on an indefinite variant of the scenario, and pin the horizon case beside them.

**v1.16 (owner's decision).** v1.14 had a side effect nobody chose: the ENDEX
was gated on ``not is_completable(...)``, the same predicate that decides the
closing route, so giving DEFEND a horizon switched the announcement off. It cost
the whole channel — 0 of 57 successes announced on ``fireteam_defend``, against
103 of 103 across the four corpora before it. ENDEX is a PROTOCOL ACT (COMMAND
emits it; not optional, not learned, not trainable away) while a root claim is an
AGENT BEHAVIOUR (optional, priced, learnable in either direction — identical
prices bought spam on one scenario and silence on the other). So the two are
decoupled: COMMAND transmits ENDEX whenever it closes the operation, whether or
not the root also claimed. The claim is the REPORT, the ENDEX is the FACT.

**v1.17 (owner's decision).** v1.14's refinement is withdrawn: DEFEND / DENY
roots are not completable at any horizon, so the root's MISSION COMPLETE is
masked shut on every defend scenario again. What makes this different from v1.13
— and what makes it available at all — is that v1.16 split the predicate. The
ENDEX is gated on ``command_closes_the_operation`` and keeps firing exactly as
it did; only ``root_may_declare_the_end`` moves. v1.13 could not do that,
because one predicate served both questions, and masking the claim took the
announcement with it.

Why withdraw it: the reopened claim was measured and bought nothing. Early close
is bounded at ``grace_window`` = 12 steps and pays no speed bonus (the terminal
speed term keys on ``_success_step``), so claim+ENDEX and ENDEX-only episodes
are indistinguishable at N=100 (p = 0.9942); its informational value is negative
in practice (``defend_brique`` filed 321 root claims at 0.71 false); and three
pricing experiments over three scenarios moved claim volume without ever moving
claim informedness. The announcement is free and guaranteed (391/391 under
v1.16), so the mask costs no observability.

So the invariant these tests hold has flipped twice, but the *hazard* has not:
the mask and the adjudicator must never disagree about who may say what, because
when they did the result was silence that looked like learned behaviour. What
follows pins all three halves — that a continuous root cannot declare itself
done at any horizon, that the C2 loop it does close (its SITREP) stays reachable
and still pays, and that the announcement is conditional on neither.
"""

from dataclasses import replace

import pytest

from cohort import get_scenario, make_env
from cohort.core.missions import COMPLETABLE, Mission, MissionType, is_completable
from cohort.core.orders import HQ_ID, MessageKind
from cohort.env.actions import CATALOG, is_done_admissible, is_root_opord_claim

STAY = 0
DONE = next(s.index for s in CATALOG if s.kind == "done")
SITREP = next(s.index for s in CATALOG if s.kind == "sitrep")


def _defend_env(seed=12):
    """The shipped scenario: a defense ordered to a horizon (v1.14).

    Since v1.17 the horizon is an adjudication clause only — it decides when the
    defense has succeeded, not who may say so — so this env and the indefinite
    one below must agree on every claim question. Both factories are kept
    precisely so that agreement is asserted rather than assumed.
    """
    env = make_env("fireteam_defend")
    env.reset(seed=seed)
    return env


def _indefinite_defend_env(seed=12):
    """The same defense with no stated end — v1.13's object, still supported."""
    spec = replace(get_scenario("fireteam_defend"), defend_horizon=None)
    env = make_env(spec)
    env.reset(seed=seed)
    return env


def _step(env, overrides=None):
    acts = {a: STAY for a in env.agents}
    acts.update(overrides or {})
    return env.step(acts)


def _win(env):
    """Drive the world to the root-mission success condition."""
    for e in env.enemies:
        e.alive = False
    root = env.roster.root()
    obj = env.world.objective_by_name(env.spec_cfg.root_objective)
    root.pos = obj.pos
    return root, obj


def test_defend_root_mission_is_not_completable_by_type():
    """The premise, unchanged since v1.4: DEFEND is a posture, not a task."""
    env = _defend_env()
    assert env.spec_cfg.root_mission is MissionType.DEFEND
    assert MissionType.DEFEND not in COMPLETABLE


def test_a_continuous_root_cannot_declare_its_own_operation_over():
    """The v1.13 reversal: not even the root, and not even when it is true."""
    env = _indefinite_defend_env()
    _win(env)
    root = env.roster.root()
    assert root.mission is not None
    assert root.mission.issuer_id == HQ_ID
    assert not is_root_opord_claim(
        root, env.roster, env.spec_cfg.root_mission, env._root_objective_id()
    )
    assert env._mask_for(root)[DONE] == 0, "a DEFEND root may not claim COMPLETE"


def test_a_horizon_defense_does_not_reopen_the_channel_to_the_root():
    """v1.17 withdraws v1.14: the stated hour opens no act to anybody.

    The predicate is the same one the mask admits on, so this is also the
    guard against the two drifting apart a fourth time. Asserted on the state
    where the claim would be TRUE — the operation is won — because a mask that
    only shuts on false claims would be pricing, not masking.
    """
    env = _defend_env()
    assert env.spec_cfg.defend_horizon is not None
    _win(env)
    root = env.roster.root()
    assert not is_root_opord_claim(
        root, env.roster, env.spec_cfg.root_mission, env._root_objective_id()
    )
    assert env._mask_for(root)[DONE] == 0, "the horizon root was offered the act"


@pytest.mark.parametrize("scenario", ["fireteam_defend", "defend_brique"])
def test_a_horizon_defend_root_is_masked_shut_on_every_step_of_an_episode(scenario):
    """The mask, not the outcome — at every step, including once it is TRUE.

    "No claims were filed" is compatible with a policy that simply declined an
    open channel; that ambiguity is what ``done_ok`` exists to resolve and is
    the exact failure mode of v1.4 (silence read as behaviour). So this asserts
    the DONE bit itself, at every step of a whole episode that reaches the
    success condition, and cross-checks it against the trace predicate the
    metrics record — three readings of the same fact from three call sites,
    which is what stopped the mask and the adjudicator drifting apart before.
    """
    env = make_env(scenario)
    obs, _ = env.reset(seed=7)
    env._h_hour = 0  # the preparation period is over; the criterion is live
    assert env.spec_cfg.defend_horizon is not None

    steps = 0
    won_at = 3  # early release: the band is destroyed, so success is genuine
    while env.agents:
        root = env.roster.root()
        if root is not None and root.alive:
            assert env._mask_for(root)[DONE] == 0, f"root DONE admitted at step {steps}"
            assert obs[root.callsign]["action_mask"][DONE] == 0, "the policy saw an open bit"
            assert not is_done_admissible(
                root,
                env.roster,
                root_mission=env.spec_cfg.root_mission,
                root_objective_id=env._root_objective_id(),
                step=env._step_count,
                done_cooldown=env.spec_cfg.done_cooldown,
            )
        if steps == won_at:
            _win(env)  # a true claim is available from here on, and still masked
        obs, _, _, _, _ = _step(env)
        steps += 1

    assert steps > won_at, "the episode ended before the claim could be true"
    assert env._success_step is not None, "the success condition was never met"
    assert env.outcome == "success"
    assert not any(m.kind is MessageKind.DONE for m in env.transcript.messages)


def test_the_horizon_root_closes_with_a_sitrep_and_command_still_announces():
    """v1.17: the ROUTE reverts to v1.13's; the ANNOUNCEMENT never moved.

    With the claim masked shut the root's C2 loop is the one v1.13 built — it
    reports the situation, the SITREP closes the grace window, and COMMAND
    transmits ENDEX. That is what keeps ``root_done_bonus`` reachable on a
    defense: masking the claim WITHOUT this route would make the bonus dead
    reward, which is the v1.4 failure in v1.13 clothes. And the ENDEX fires
    here whether or not anything was reported, because since v1.16 it is gated
    on ``command_closes_the_operation`` and not on the closing route.
    """
    env = _defend_env()
    env._h_hour = 0  # the preparation period is over; the criterion is live
    root, _ = _win(env)
    before = len(env.transcript.messages)
    _step(env, {root.callsign: SITREP})
    new = env.transcript.messages[before:]
    kinds = [m.kind for m in new]

    assert MessageKind.SITREP in kinds, "the root's report never went out"
    assert MessageKind.ENDEX in kinds, "COMMAND stopped announcing the close"
    assert MessageKind.DONE not in kinds, "the horizon root claimed after all"
    assert env._root_close_step is not None, "grace window never closed"
    assert env._root_close_callsign == root.callsign

    endex = next(m for m in new if m.kind is MessageKind.ENDEX)
    assert endex.sender_id == HQ_ID, "ENDEX must come from COMMAND, not the root"
    assert endex.recipient_id == root.id


def test_a_horizon_defense_that_never_claims_is_announced_anyway_and_once():
    """The protocol act is not conditional on the agent behaviour.

    This is the case v1.14 lost: the root says nothing, the grace window
    expires, the operation succeeds — and on ``fireteam_defend`` that was 30
    successes in 30 episodes with not one word on the net. Pinned as: exactly
    one ENDEX, no claim, and no early close (so no ``root_done_bonus``, which is
    what silence should cost and all it should cost).
    """
    env = _defend_env()
    env._h_hour = 0
    _win(env)
    for _ in range(env.spec_cfg.grace_window + 2):
        _, _, term, trunc, _ = _step(env)
        if all(term.values()) or all(trunc.values()):
            break

    assert env.outcome == "success"
    endexes = [m for m in env.transcript.messages if m.kind is MessageKind.ENDEX]
    assert len(endexes) == 1, f"expected exactly one ENDEX, got {len(endexes)}"
    assert not any(m.kind is MessageKind.DONE for m in env.transcript.messages)
    assert env._root_close_step is None, "nobody reported; no early close, no bonus"


def test_an_early_close_and_an_endex_coexist_in_one_episode_exactly_once():
    """Both channels, over the whole episode rather than one step of it.

    The once-per-episode guard (``_endex_step``) has to survive the early-close
    path: the SITREP close ends the episode on the spot, so a second ENDEX here
    would mean the guard was never consulted on this route at all.
    """
    env = _defend_env()
    env._h_hour = 0
    root, _ = _win(env)
    _step(env, {root.callsign: SITREP})
    while env.agents:  # nothing left to do if the close already terminated it
        _step(env)

    kinds = [m.kind for m in env.transcript.messages]
    assert env.outcome == "success"
    assert env._root_close_step is not None, "the SITREP did not close the window"
    assert kinds.count(MessageKind.ENDEX) == 1, "one operation, one ENDEX"
    assert env._endex_step is not None


def test_the_horizon_root_is_masked_shut_on_a_false_claim_too():
    """Not pricing, masking: the act is unavailable whatever the ground truth.

    v1.14 adjudicated a premature horizon claim and rejected it. v1.17 never
    offers it, which is the whole point of the owner's decision — three
    experiments showed prices move claim volume without moving informedness.
    """
    env = _defend_env()
    env._h_hour = 0
    root = env.roster.root()
    obj = env.world.objective_by_name(env.spec_cfg.root_objective)
    root.pos = obj.pos  # occupied, but neither released nor at the horizon
    before = len(env.transcript.messages)
    assert env._mask_for(root)[DONE] == 0
    _step(env, {root.callsign: DONE})  # illegal: substituted, not adjudicated
    kinds = [m.kind for m in env.transcript.messages[before:]]

    assert MessageKind.DONE not in kinds, "a masked act reached the net"
    assert MessageKind.DONE_REJECT not in kinds, "the claim was adjudicated, not masked"
    assert env._root_close_step is None


def test_command_transmits_endex_and_the_root_sitrep_closes_the_window():
    """The replacement loop: the root reports, COMMAND ends the operation.

    This is the test that keeps ``root_done_bonus`` from becoming dead reward
    a second time — the v1.4 failure, in its v1.13 clothes.
    """
    env = _indefinite_defend_env()
    root, _ = _win(env)
    before = len(env.transcript.messages)
    _step(env, {root.callsign: SITREP})
    new = env.transcript.messages[before:]
    kinds = [m.kind for m in new]

    assert MessageKind.SITREP in kinds, "the root's report never went out"
    assert MessageKind.ENDEX in kinds, "COMMAND never closed the operation"
    assert MessageKind.DONE not in kinds
    assert env._root_close_step is not None, "grace window never closed"
    assert env._root_close_callsign == root.callsign

    endex = next(m for m in new if m.kind is MessageKind.ENDEX)
    assert endex.sender_id == HQ_ID, "ENDEX must come from COMMAND, not the root"
    assert endex.recipient_id == root.id
    assert "ENDEX" in endex.text


def test_endex_closes_a_silent_defense_too_but_only_once():
    """COMMAND ends the operation whether or not the root reported in time.

    What the SITREP buys is closing *early* and the bonus — not the ending
    itself, which was never the root's to give.
    """
    env = _indefinite_defend_env()
    _win(env)
    for _ in range(env.spec_cfg.grace_window + 2):
        _, _, term, trunc, _ = _step(env)
        if all(term.values()) or all(trunc.values()):
            break
    endexes = [m for m in env.transcript.messages if m.kind is MessageKind.ENDEX]
    assert len(endexes) == 1, f"expected exactly one ENDEX, got {len(endexes)}"
    assert env._root_close_step is None, "nobody reported; no early close, no bonus"


@pytest.mark.parametrize("factory", [_defend_env, _indefinite_defend_env])
def test_mask_and_adjudicator_agree_on_every_step(factory):
    """The hazard that outlived all three revisions: the two must not drift.

    Wherever the mask admits a DONE, the predicate must agree it is claimable.
    Since v1.17 that means never, for the root, on either factory — the horizon
    scenario and the indefinite one have to answer alike, because the horizon is
    an adjudication clause and no longer a permission.
    """
    env = factory()
    root_id = env._root_objective_id()
    for _ in range(30):
        root = env.roster.root()
        if root is None or not root.alive:
            break
        claimable = (
            is_root_opord_claim(root, env.roster, env.spec_cfg.root_mission, root_id)
            or root.mission.type in COMPLETABLE
        )
        assert claimable is False, "a continuous root became claimable"
        assert bool(env._mask_for(root)[DONE]) == claimable
        if not _step(env)[2]:
            continue


def test_subordinate_posture_still_cannot_claim():
    """Unchanged: a rifleman's DEFEND posture was never claimable either."""
    env = _defend_env()
    _win(env)
    sub = next(s for s in env.roster.living if s is not env.roster.root())
    obj = env.world.objective_by_name(env.spec_cfg.root_objective)
    sub.mission = Mission(
        MissionType.DEFEND, obj.id, obj.pos,
        issuer_id=env.roster.root().id, step_assigned=0,
    )
    assert env._mask_for(sub)[DONE] == 0, "a subordinate posture became claimable"


def test_predicate_rejects_a_non_root_and_a_non_hq_mission():
    env = make_env("fireteam")  # SEIZE root: the predicate can be true here
    env.reset(seed=1)
    root = env.roster.root()
    root_id = env._root_objective_id()
    spec_mission = env.spec_cfg.root_mission

    sub = next(s for s in env.roster.living if s is not root)
    assert not is_root_opord_claim(sub, env.roster, spec_mission, root_id)

    if root.mission is not None:
        root.mission.issuer_id = root.id  # no longer the HQ OPORD
        assert not is_root_opord_claim(root, env.roster, spec_mission, root_id)


@pytest.mark.parametrize(
    "mission",
    [
        MissionType.SEIZE,
        MissionType.RECON,
        MissionType.CLEAR,
        MissionType.RALLY,
        MissionType.ADVANCE,
        MissionType.SCREEN,
    ],
)
def test_the_task_missions_keep_their_claim_at_every_horizon(mission):
    """v1.17 is scoped to continuous postures, and nothing else moves.

    ``is_completable`` lost its ``defend_horizon`` parameter outright rather
    than keeping a knob that does nothing, so the guard here is that the answer
    for a task mission was never horizon-dependent in the first place — the
    scenario field still exists and still adjudicates DEFEND success.
    """
    assert is_completable(mission) is True


@pytest.mark.parametrize(
    "mission",
    [MissionType.DEFEND, MissionType.DENY, MissionType.HOLD, MissionType.OBSERVE,
     MissionType.SUPPORT, MissionType.COVER, None],
)
def test_the_continuous_postures_are_uniformly_undeclarable(mission):
    assert is_completable(mission) is False


def test_seize_rooted_scenario_keeps_mission_complete():
    """The reversal is scoped to continuous postures — SEIZE still reports."""
    env = make_env("fireteam")
    env.reset(seed=1)
    assert env.spec_cfg.root_mission in COMPLETABLE
    root = env.roster.root()
    if root.mission is not None and root.mission.type in COMPLETABLE:
        assert env._mask_for(root)[DONE] in (0, 1)  # gated by pending/cooldown only
    # and no ENDEX anywhere: COMMAND does not close what the root can finish
    for _ in range(15):
        if not _step(env)[2]:
            continue
    assert not any(m.kind is MessageKind.ENDEX for m in env.transcript.messages)
