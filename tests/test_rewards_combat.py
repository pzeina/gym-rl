"""Reward semantics on a flattened (all-open) deterministic map."""


from cohort import make_env
from cohort.core.missions import Mission, MissionType
from cohort.env.actions import CATALOG

STAY = 0
MOVE_EAST = next(s.index for s in CATALOG if s.name == "MOVE_EAST")
MOVE_WEST = next(s.index for s in CATALOG if s.name == "MOVE_WEST")
FIRE = next(s.index for s in CATALOG if s.kind == "fire")
CONTACT = next(s.index for s in CATALOG if s.kind == "contact")
DONE = next(s.index for s in CATALOG if s.kind == "done")


def _flat_env(seed=1):
    """Fireteam env with all-open terrain and enemies parked far away."""
    env = make_env("fireteam")
    env.reset(seed=seed)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (22, 1)
        e.home = e.pos
    return env


def _step_all(env, overrides):
    acts = {a: STAY for a in env.agents}
    acts.update(overrides)
    return env.step(acts)


def test_compliance_progress_sign_in_env():
    env = _flat_env()
    sld = env.roster.by_callsign["RFN1"]
    obj = env.world.objectives[0]  # ALPHA at (18, 18)
    sld.pos = (10, 18)
    sld.mission = Mission(MissionType.SEIZE, 0, obj.pos, issuer_id=-1, step_assigned=0)

    *_, infos = _step_all(env, {"RFN1": MOVE_EAST})
    assert infos["RFN1"]["components"]["compliance"] > 0, "closing on the objective pays"

    *_, infos = _step_all(env, {"RFN1": MOVE_WEST})
    assert infos["RFN1"]["components"]["compliance"] < 0, "walking away from the objective costs"


def test_contact_report_new_then_redundant():
    env = _flat_env()
    sld = env.roster.by_callsign["RFN1"]
    enemy = env.enemies[0]
    enemy.pos = (10, 10)
    sld.pos = (7, 10)  # clear LOS on open ground, inside vision range

    *_, infos = _step_all(env, {"RFN1": CONTACT})
    first = infos["RFN1"]["components"]["report"]
    assert first > 0, "first CONTACT on an unknown enemy is rewarded"
    assert env._known_enemies, "report feeds the team picture"

    *_, infos = _step_all(env, {"RFN1": CONTACT})
    assert infos["RFN1"]["components"]["report"] < 0, "re-reporting known intel is spam"
    assert any(m.kind.value == "contact" for m in env.transcript.messages)


def test_mission_complete_truthful_vs_false():
    env = _flat_env()
    sld = env.roster.by_callsign["RFN1"]
    obj = env.world.objectives[1]  # BRAVO — no enemies parked there after _flat_env
    for e in env.enemies:
        e.pos = (1, 22)
        e.home = e.pos
    # false claim first: far from BRAVO, mission clearly not done
    sld.pos = (2, 2)
    sld.mission = Mission(MissionType.SEIZE, 1, obj.pos, issuer_id=-1, step_assigned=0)
    *_, infos = _step_all(env, {"RFN1": DONE})
    assert infos["RFN1"]["components"]["report"] < 0, "false MISSION COMPLETE is penalized"
    assert sld.mission is not None, "mission stands until actually complete"

    # now truthfully: stand on BRAVO with no enemies near it
    sld.pos = obj.pos
    *_, infos = _step_all(env, {"RFN1": DONE})
    assert infos["RFN1"]["components"]["report"] > 0
    assert sld.mission is None, "honest completion clears the mission"


def test_done_verdict_lands_on_the_net():
    """The superior answers every completion report: the verdict is traffic."""
    env = _flat_env()
    sld = env.roster.by_callsign["RFN1"]
    obj = env.world.objectives[1]  # BRAVO — no enemies near after repositioning
    for e in env.enemies:
        e.pos = (1, 22)
        e.home = e.pos
    # false claim → NEGATIVE from the leader, mission stands
    sld.pos = (2, 2)
    sld.mission = Mission(MissionType.SEIZE, 1, obj.pos, issuer_id=-1, step_assigned=0)
    _step_all(env, {"RFN1": DONE})
    reject = next(m for m in reversed(env.transcript.messages) if m.kind.value == "done_reject")
    assert reject.text == "RFN1, THIS IS TL1: NEGATIVE, CONTINUE MISSION. OUT."
    assert sld.mission is not None
    # truthful claim → ROGER ... CONFIRMED, mission cleared
    sld.pos = obj.pos
    _step_all(env, {"RFN1": DONE})
    confirm = next(m for m in reversed(env.transcript.messages) if m.kind.value == "done_confirm")
    assert confirm.text == "RFN1, THIS IS TL1: ROGER, SEIZE OBJ BRAVO CONFIRMED. OUT."
    assert sld.mission is None
    # the claim precedes its verdict on the transcript
    kinds = [m.kind.value for m in env.transcript.messages]
    assert kinds.index("done") < kinds.index("done_reject")


def test_root_done_is_answered_by_hq():
    from cohort.core.orders import HQ_ID

    env = _flat_env()
    # TL1 holds the OPORD (SEIZE ALPHA) and is far from it: a false claim
    _step_all(env, {"TL1": DONE})
    reject = next(m for m in reversed(env.transcript.messages) if m.kind.value == "done_reject")
    assert reject.sender_id == HQ_ID
    assert reject.text.startswith("TL1, THIS IS HQ: NEGATIVE")


def test_kill_rewards_and_team_share():
    env = _flat_env()
    sld = env.roster.by_callsign["RFN1"]
    enemy = env.enemies[0]
    enemy.pos = (8, 10)
    enemy.health = 1
    sld.pos = (7, 10)
    for e in env.enemies[1:]:
        e.alive = False

    killed = False
    for _ in range(8):  # point-blank shots; a miss is possible but rare
        *_, infos = _step_all(env, {"RFN1": FIRE})
        if not enemy.alive:
            killed = True
            assert infos["RFN1"]["components"]["combat"] > 0.5
            assert infos["RFN2"]["components"]["combat"] > 0, "teammates share the kill"
            break
    assert killed, "adjacent target with 1 hp should die within a few shots"


def test_time_penalty_always_applies():
    env = _flat_env()
    *_, infos = _step_all(env, {})
    for agent in infos:
        assert infos[agent]["components"]["time"] < 0


def test_terminal_dominates_stalling():
    """Winning must always beat farming shaping until timeout.

    Regression test: on the squad scenario the policy once learned to stall —
    return kept rising while success collapsed to 0% and every episode ran the
    full 300 steps. Success must be worth strictly more than a perfect farm
    over the longest episode any scenario allows.
    """
    from cohort.config import SCENARIOS
    from cohort.env.rewards import RewardConfig

    cfg = RewardConfig()
    for spec in SCENARIOS.values():
        best_farm = cfg.max_step_farm() * spec.max_steps
        assert cfg.success_team > best_farm, (
            f"{spec.name}: stalling for {spec.max_steps} steps yields {best_farm:.1f} "
            f">= success reward {cfg.success_team} — reward hacking is profitable"
        )


def test_success_pays_everyone():
    """Success pays the whole team. Since the completion-report grace window
    landed, the episode no longer ends the step the condition is met — it ends
    when the root reports COMPLETE, or at T0 + grace_window at the latest."""
    env = _flat_env()
    obj = env.world.objectives[0]
    for e in env.enemies:
        e.alive = False  # objective cleared
    env.roster.by_callsign["TL1"].pos = obj.pos
    _obs, rewards, terms, *_ = _step_all(env, {})
    assert env.outcome is None, "the grace window holds the episode open"
    assert not any(terms.values())
    for _ in range(env.spec_cfg.grace_window):
        _obs, rewards, terms, *_ = _step_all(env, {})
        if all(terms.values()):
            break
    assert env.outcome == "success"
    assert all(terms.values())
    for agent, r in rewards.items():
        assert r > 3.0, f"{agent} should share the terminal success reward"
