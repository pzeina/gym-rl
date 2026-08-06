"""Fire discipline by mission: combat rewards must serve the standing order.

Found via oracle diagnosis (A2): with flat combat rewards, recon elements
out-shot the static postures, and defenders left the objective to chase
kills — 32 of 37 defender deaths happened away from the position they were
ordered to hold. Combat pay is scaled by mission (core/missions.py): weapons
tight on SCREEN (ÉCLAIRER — intel *without* engaging), position-anchored pay
for the static postures, full pay for RECON (RECONNAÎTRE may engage),
assault tasks, and untasked agents.
"""

from cohort import make_env
from cohort.core.missions import Mission, MissionType
from cohort.env.actions import CATALOG
from cohort.env.rewards import RewardConfig

STAY = 0
FIRE = next(s.index for s in CATALOG if s.kind == "fire")


def _flat_env(**reward_overrides):
    from dataclasses import replace

    cfg = replace(RewardConfig(), **reward_overrides) if reward_overrides else None
    env = make_env("fireteam", reward_config=cfg)
    env.reset(seed=4)
    env.world.grid[:] = 0
    for e in env.enemies:
        e.pos = (22, 1)
        e.home = e.pos
        e.prev_pos = e.pos
    return env


def _point_blank_kill(env, mission=None):
    """Give RFN1 an adjacent 1-hp target (repeat until the dice land a hit)."""
    rfn = env.roster.by_callsign["RFN1"]
    enemy = env.enemies[0]
    for other in env.enemies[1:]:
        other.alive = False
    rfn.mission = mission
    combat = 0.0
    for _ in range(10):
        enemy.alive = True
        enemy.health = 1
        enemy.pos = (rfn.pos[0] + 1, rfn.pos[1])
        enemy.prev_pos = enemy.pos
        acts = {a: (FIRE if a == "RFN1" else STAY) for a in env.agents}
        *_, infos = env.step(acts)
        combat = infos["RFN1"]["components"]["combat"]
        if not enemy.alive:
            return combat
    raise AssertionError("no hit in 10 point-blank shots")


def test_untasked_shooter_paid_in_full():
    env = _flat_env()
    combat = _point_blank_kill(env, mission=None)
    assert combat >= 1.0, "untasked kill pays hit + kill rewards"


def test_screen_is_weapons_tight():
    env = _flat_env()
    rfn = env.roster.by_callsign["RFN1"]
    obj = env.world.objectives[1]
    mission = Mission(MissionType.SCREEN, 1, obj.pos, issuer_id=-1, step_assigned=0)
    combat = _point_blank_kill(env, mission=mission)
    assert combat <= 0.0, "SCREEN shooter earns nothing for the kill"
    # ...and compliance still punishes the shot itself
    assert rfn.mission is not None


def test_recon_may_engage():
    """PROTERRE RECONNAÎTRE engages when needed: full combat pay on RECON."""
    env = _flat_env()
    obj = env.world.objectives[1]
    mission = Mission(MissionType.RECON, 1, obj.pos, issuer_id=-1, step_assigned=0)
    combat = _point_blank_kill(env, mission=mission)
    assert combat >= 1.0, "RECON kill pays hit + kill rewards"


def test_defend_pays_only_from_position():
    env = _flat_env()
    rfn = env.roster.by_callsign["RFN1"]
    obj = env.world.objectives[0]

    # off-position: chasing kills far from the objective earns nothing
    rfn.pos = (3, 3)
    mission = Mission(MissionType.DEFEND, 0, obj.pos, issuer_id=-1, step_assigned=0)
    combat_off = _point_blank_kill(env, mission=mission)
    assert combat_off <= 0.0, "kill away from the defended objective earns nothing"

    # on-position: fighting from the objective pays in full
    rfn.pos = obj.pos
    mission2 = Mission(MissionType.DEFEND, 0, obj.pos, issuer_id=-1, step_assigned=0)
    combat_on = _point_blank_kill(env, mission=mission2)
    assert combat_on >= 1.0, "kill from the defended position pays hit + kill"


def test_defend_pays_against_assault_on_position():
    """Defense-of-the-position carve-out (v1.9 defend diagnosis): a defender
    pushed OFF its disc still earns full pay for fire at an enemy standing
    inside the position's engagement envelope (anchor distance <=
    IN_POSITION_RADIUS + weapon_range) — that enemy is assaulting the
    objective, and killing it IS the mission. The v6 oracle showed the human
    TL firing on 0.5% of threatened opportunities because its off-position
    fire earned nothing."""
    env = _flat_env()
    rfn = env.roster.by_callsign["RFN1"]
    obj = env.world.objectives[0]

    # shooter out of position (6 cells west of the anchor, > DEFEND radius
    # 3.5) kills an adjacent enemy standing 5 cells from the anchor — inside
    # the envelope (3.5 + weapon_range 8 = 11.5): full pay
    rfn.pos = (obj.pos[0] - 6, obj.pos[1])
    mission = Mission(MissionType.DEFEND, 0, obj.pos, issuer_id=-1, step_assigned=0)
    combat = _point_blank_kill(env, mission=mission)
    assert combat >= 1.0, "fire against the assault on the position pays in full"


def test_screen_stays_tight_despite_assault_carveout():
    """The carve-out applies to position-anchored missions only — SCREEN
    (weapons tight) earns nothing even for a target on the objective."""
    env = _flat_env()
    rfn = env.roster.by_callsign["RFN1"]
    obj = env.world.objectives[0]
    rfn.pos = (obj.pos[0] - 1, obj.pos[1])
    mission = Mission(MissionType.SCREEN, 0, obj.pos, issuer_id=-1, step_assigned=0)
    combat = _point_blank_kill(env, mission=mission)
    assert combat <= 0.0, "SCREEN stays weapons tight inside the envelope too"


def test_knob_off_restores_flat_combat_pay():
    env = _flat_env(fire_discipline=False)
    rfn = env.roster.by_callsign["RFN1"]
    obj = env.world.objectives[1]
    mission = Mission(MissionType.SCREEN, 1, obj.pos, issuer_id=-1, step_assigned=0)
    combat = _point_blank_kill(env, mission=mission)
    del rfn
    assert combat >= 1.0, "fire_discipline=False restores the old behavior"


def test_teammate_share_unscaled():
    env = _flat_env()
    obj = env.world.objectives[1]
    mission = Mission(MissionType.SCREEN, 1, obj.pos, issuer_id=-1, step_assigned=0)
    rfn = env.roster.by_callsign["RFN1"]
    enemy = env.enemies[0]
    for other in env.enemies[1:]:
        other.alive = False
    rfn.mission = mission
    for _ in range(10):
        enemy.alive = True
        enemy.health = 1
        enemy.pos = (rfn.pos[0] + 1, rfn.pos[1])
        enemy.prev_pos = enemy.pos
        *_, infos = env.step({a: (FIRE if a == "RFN1" else STAY) for a in env.agents})
        if not enemy.alive:
            assert infos["RFN2"]["components"]["combat"] > 0, "teammates still share the kill"
            return
    raise AssertionError("no hit in 10 point-blank shots")
