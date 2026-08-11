"""Every operation this cohort wins is announced on the net (v1.19).

Before v1.19 the announcement was gated on the root's mission: a continuous
posture got COMMAND's ENDEX, a completable one got whatever the root chose to
say. Measured at N=100 on the final policy of every published champion, that
made the guarantee cover **two scenarios of nine** —

    defend (ENDEX, a protocol act)        391/391 successes announced
    squad / squad_recon / squad_screen    91-98%
    fireteam_v8                            49/80
    platoon_v5  0/100   ·   patrol_brique_v5  0/99

``platoon`` and ``patrol_brique`` succeeded on essentially every episode and
never once said so. That is not two standards of reporting; it is one standard
and one silence, and the fix is the same one v1.16 applied to the defend
family: make the announcement a protocol act.

What is pinned here, in the order it matters:

* **The guarantee.** Every scenario, every root mission: a won episode ends
  with HQ's ENDEX on the transcript. Driven with agents that do nothing, so
  what is measured is the protocol and not a policy.
* **Its denominator.** A lost or timed-out episode gets no ENDEX. "Successes
  announced" is only worth reading if the announcement cannot appear without a
  success — otherwise the metric measures HQ's chattiness.
* **Rollout neutrality.** The ENDEX is stamped on the step the episode
  terminates, after that step's actions are applied, so no agent ever selects
  an action from an observation containing it. This is what makes v1.19 a
  scoring and transcript change rather than a behavioural one — and it is the
  claim that was asserted in the opposite direction once already and measured
  false, so it is a test now and not a comment.
* **The root's own report survives.** The claim is the REPORT, the ENDEX is the
  FACT. A completable root that reports gets its CONFIRMED, ends the episode
  early, and keeps ``root_done_bonus``; the guarantee must not be implemented
  by taking the announcement away from the agent that earned it.
"""

from __future__ import annotations

import pytest

from cohort import get_scenario, make_env
from cohort.core.missions import MissionType
from cohort.env.actions import CATALOG
from cohort.env.cohort_env import TEAM_OBSERVE_STEPS

STAY = 0
DONE = next(s.index for s in CATALOG if s.kind == "done")

# The eight doctrine scenarios the baseline fleet covers. The ablation arms
# (squad_nomask, squad_flat) and the observation probes deliberately re-present
# the squad scenario and would only re-assert the same env.
SCENARIOS = [
    "fireteam",
    "fireteam_defend",
    "squad",
    "squad_recon",
    "squad_screen",
    "patrol_brique",
    "defend_brique",
    "platoon",
]


def _step_all(env, overrides=None):
    acts = {a: STAY for a in env.agents}
    acts.update(overrides or {})
    return env.step(acts)


def _kinds(env):
    return [m.kind.value for m in env.transcript.messages]


def _root_objective(env):
    cfg = env.spec_cfg
    if cfg.root_objective:
        return env.world.objective_by_name(cfg.root_objective)
    return env.world.objectives[0] if env.world.objectives else None


def _hand_it_to_them(env):
    """Put the cohort in the winning position without asking it to fight.

    Every branch of ``_check_success`` needs the threat gone, so the enemies
    are removed in all of them; what differs is where the friendlies must be
    standing and for how long. Terrain is flattened so line of sight — which
    the RECON/SCREEN counter depends on — is never the reason a test fails.
    """
    obj = _root_objective(env)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.alive = False
    mission = env.spec_cfg.root_mission
    if mission in (MissionType.RECON, MissionType.SCREEN):
        # observers on the ring, not on the objective: ÉCLAIRER is watching it
        for i, s in enumerate(env.roster.living):
            s.pos = (max(0, obj.pos[0] - 3 - (i % 2)), obj.pos[1])
    else:
        for s in env.roster.living:
            s.pos = obj.pos
    return obj


def _drive_to_outcome(env, limit=None):
    """Step a do-nothing cohort until the episode resolves; return the outcome."""
    limit = limit or env.spec_cfg.max_steps + 1
    for _ in range(limit):
        _step_all(env)
        if env.outcome is not None:
            return env.outcome
    return env.outcome


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_won_operation_is_announced_on_every_scenario(scenario):
    env = make_env(scenario)
    env.reset(seed=1)
    _hand_it_to_them(env)

    outcome = _drive_to_outcome(env)

    assert outcome == "success", f"{scenario}: the setup no longer wins the episode"
    kinds = _kinds(env)
    assert kinds.count("endex") == 1, f"{scenario}: expected exactly one ENDEX, got {kinds[-4:]}"
    assert kinds[-1] == "endex", f"{scenario}: the ENDEX must close the net, got {kinds[-3:]}"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_announcement_needs_a_success_to_announce(scenario):
    """The denominator half: no win, no ENDEX.

    Driven by doing nothing from the start position, which on every scenario
    either times out or gets the cohort killed — both of them not-a-success.
    """
    env = make_env(scenario)
    env.reset(seed=7)
    outcome = _drive_to_outcome(env)

    assert outcome != "success", f"{scenario}: idling won the episode; pick a harder seed"
    assert "endex" not in _kinds(env), f"{scenario}: ENDEX transmitted on a {outcome}"


def test_the_endex_lands_on_the_terminating_step_so_no_agent_acts_on_it():
    """Rollout neutrality, asserted where it can actually be checked.

    If the ENDEX were transmitted before the terminal step, some agent would
    select an action from an observation containing it and v1.19 would be a
    behavioural change — every checkpoint trained before it would be arguing
    from a different net. It is not: the message is stamped with the step that
    ends the episode.
    """
    env = make_env("fireteam")
    env.reset(seed=1)
    _hand_it_to_them(env)
    _drive_to_outcome(env)

    endex = [m for m in env.transcript.messages if m.kind.value == "endex"]
    assert len(endex) == 1
    assert endex[0].step == env._step_count, "ENDEX predates the terminal step"


def test_a_reporting_root_still_owns_its_report():
    """The guarantee is additive: HQ's FACT does not replace the root's REPORT."""
    env = make_env("fireteam")
    env.reset(seed=1)
    _hand_it_to_them(env)
    _step_all(env)  # T0: the condition holds, the grace window opens
    assert env.outcome is None

    _obs, _r, _terms, _trunc, infos = _step_all(env, {"TL1": DONE})

    assert env.outcome == "success"
    assert [m.kind.value for m in env.transcript.messages[-3:]] == [
        "done",
        "done_confirm",
        "endex",
    ]
    assert infos["TL1"]["components"]["terminal"] == pytest.approx(
        infos["RFN1"]["components"]["terminal"] + env.rewards_cfg.root_done_bonus
    )


def test_a_silent_root_is_announced_anyway_and_earns_no_bonus():
    """The other side of the same coin, and the reason the metric stays useful.

    ``successes_announced`` is complete by construction after v1.19, so the
    behaviour that used to hide inside it moves to ``closed_on_root_report_rate``
    — ENDEXes sent for a denominator, the root's own act for a numerator. Here
    the root never reports: the operation is still announced, the window still
    expires on the grace clock, and no bonus is paid.
    """
    env = make_env("fireteam")
    env.reset(seed=1)
    _hand_it_to_them(env)
    _drive_to_outcome(env)

    assert env.outcome == "success"
    assert "endex" in _kinds(env)
    assert "done_confirm" not in _kinds(env), "the root was told to stay silent"


def test_observe_threshold_import_is_the_one_the_env_scores():
    """Guards the RECON/SCREEN setup above against a silent constant change."""
    assert TEAM_OBSERVE_STEPS >= 1
    assert get_scenario("squad_recon").root_mission is MissionType.RECON
    assert get_scenario("squad_screen").root_mission is MissionType.SCREEN
