"""The root must be able to report the OPORD complete — even a DEFEND OPORD.

``COMPLETABLE`` deliberately excludes DEFEND/DENY: no individual ever
"finishes" a continuous posture. The action mask gated MISSION COMPLETE on
``mission.type in COMPLETABLE``, while ``_report_done``'s root branch gated on
``mission.type is spec.root_mission``. On any DEFEND- or DENY-rooted scenario
those two conditions cannot both hold, so:

* the root's MISSION COMPLETE was hard-masked on every step of every episode;
* ``_report_done``'s root branch — the one whose comment says a commander
  reports the operation complete "wherever it stands" — was unreachable;
* ``root_done_bonus`` was dead reward;
* ``grace_window`` could only ever expire by timeout, never close on a report.

Measured on fireteam_defend_v8 with ``scripts/done_probe.py``: 0 admissible
root claims and 0 truthful-and-admissible agent-steps across 30 episodes. The
silence read as a policy that had learned not to claim; it was a mask.

``is_root_opord_claim`` is now the single predicate both sides consult. These
tests pin that they cannot drift apart again, and that opening the root's
claim did not open anyone else's.
"""

from cohort import make_env
from cohort.core.missions import COMPLETABLE, Mission, MissionType
from cohort.core.orders import HQ_ID, MessageKind
from cohort.env.actions import CATALOG, is_root_opord_claim

STAY = 0
DONE = next(s.index for s in CATALOG if s.kind == "done")


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
    """The premise: DEFEND is a posture, so type-based claiming can't work."""
    env = _defend_env()
    assert env.spec_cfg.root_mission is MissionType.DEFEND
    assert MissionType.DEFEND not in COMPLETABLE


def test_root_can_claim_its_defend_opord():
    """The regression: the mask must admit the root's OPORD claim anyway."""
    env = _defend_env()
    _win(env)
    root = env.roster.root()
    assert root.mission is not None
    assert root.mission.issuer_id == HQ_ID
    mask = env._mask_for(root)
    assert mask[DONE] == 1, "root's MISSION COMPLETE is hard-masked again"


def test_truthful_root_claim_is_confirmed_and_paid():
    """The whole point: a true claim closes the operation and earns the bonus."""
    env = _defend_env()
    root, _ = _win(env)
    before = len(env.transcript.messages)
    _step(env, {root.callsign: DONE})
    new = env.transcript.messages[before:]
    kinds = [m.kind for m in new]
    assert MessageKind.DONE in kinds
    assert MessageKind.DONE_CONFIRM in kinds, "truthful root claim was rejected"
    assert MessageKind.DONE_REJECT not in kinds
    assert env._root_done_step is not None, "grace window never closed on the report"
    assert env._root_done_callsign == root.callsign


def test_premature_root_claim_is_rejected_and_cooled_down():
    """Opening the channel must not make lying free."""
    env = _defend_env()
    root = env.roster.root()
    assert any(e.alive for e in env.enemies), "fixture needs a live enemy"
    before = len(env.transcript.messages)
    _step(env, {root.callsign: DONE})
    kinds = [m.kind for m in env.transcript.messages[before:]]
    assert MessageKind.DONE_REJECT in kinds
    assert MessageKind.DONE_CONFIRM not in kinds
    assert env._mask_for(root)[DONE] == 0, "rejected claim must cool down"


def test_mask_and_adjudicator_agree_on_every_admissible_root_step():
    """The drift that caused this: mask says yes, adjudicator routes elsewhere.

    Wherever the mask admits the root's DONE, the env must adjudicate it on
    the *team success* condition, not on the root's personal end state.
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
    """Opening the root's claim must not open a rifleman's DEFEND posture."""
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
    env = _defend_env()
    _win(env)
    root = env.roster.root()
    root_id = env._root_objective_id()
    spec_mission = env.spec_cfg.root_mission

    sub = next(s for s in env.roster.living if s is not root)
    assert not is_root_opord_claim(sub, env.roster, spec_mission, root_id)

    root.mission.issuer_id = root.id  # no longer the HQ OPORD
    assert not is_root_opord_claim(root, env.roster, spec_mission, root_id)


def test_seize_rooted_scenario_is_unaffected():
    """A SEIZE root was always claimable by type — keep it that way."""
    env = make_env("fireteam")
    env.reset(seed=1)
    root = env.roster.root()
    if root.mission is not None and root.mission.type in COMPLETABLE:
        assert env._mask_for(root)[DONE] in (0, 1)  # gated by pending/cooldown only
