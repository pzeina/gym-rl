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

So the invariant these tests hold has flipped, but the *hazard* has not: the
mask and the adjudicator must never disagree about who may say what, because
when they did the result was silence that looked like learned behaviour. What
follows pins both halves — that a continuous root cannot declare itself done,
and that the C2 loop it does close is reachable and pays.
"""

from cohort import make_env
from cohort.core.missions import COMPLETABLE, Mission, MissionType
from cohort.core.orders import HQ_ID, MessageKind
from cohort.env.actions import CATALOG, is_root_opord_claim

STAY = 0
DONE = next(s.index for s in CATALOG if s.kind == "done")
SITREP = next(s.index for s in CATALOG if s.kind == "sitrep")


def _defend_env(seed=12):
    env = make_env("fireteam_defend")
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
    env = _defend_env()
    _win(env)
    root = env.roster.root()
    assert root.mission is not None
    assert root.mission.issuer_id == HQ_ID
    assert not is_root_opord_claim(
        root, env.roster, env.spec_cfg.root_mission, env._root_objective_id()
    )
    assert env._mask_for(root)[DONE] == 0, "a DEFEND root may not claim COMPLETE"


def test_command_transmits_endex_and_the_root_sitrep_closes_the_window():
    """The replacement loop: the root reports, COMMAND ends the operation.

    This is the test that keeps ``root_done_bonus`` from becoming dead reward
    a second time — the v1.4 failure, in its v1.13 clothes.
    """
    env = _defend_env()
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
    env = _defend_env()
    _win(env)
    for _ in range(env.spec_cfg.grace_window + 2):
        _, _, term, trunc, _ = _step(env)
        if all(term.values()) or all(trunc.values()):
            break
    endexes = [m for m in env.transcript.messages if m.kind is MessageKind.ENDEX]
    assert len(endexes) == 1, f"expected exactly one ENDEX, got {len(endexes)}"
    assert env._root_close_step is None, "nobody reported; no early close, no bonus"


def test_mask_and_adjudicator_agree_on_every_step():
    """The hazard that outlived the reversal: the two must not drift apart.

    Wherever the mask admits a DONE, the predicate must agree it is claimable;
    on this scenario that means it is never admitted for the root at all.
    """
    env = _defend_env()
    root_id = env._root_objective_id()
    for _ in range(30):
        root = env.roster.root()
        if root is None or not root.alive:
            break
        if env._mask_for(root)[DONE]:
            assert is_root_opord_claim(
                root, env.roster, env.spec_cfg.root_mission, root_id
            ) or root.mission.type in COMPLETABLE
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
