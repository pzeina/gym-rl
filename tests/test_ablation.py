"""B3 hierarchy-ablation arms (ScenarioSpec.ablation) behave as specified.

Three arms, masking-only changes, spaces frozen:

* full   — the shipped system (hierarchy + doctrine masks); the default.
* nomask — hierarchy without doctrine masks: any rank-admissible order,
  regardless of the issuer's own mission; cooldown + hold authority stay.
* flat   — no ranks in effect: no orders at all, everyone OPORD-tasked at
  reset, comms limited to reports, leader coverage reward neutralized.
"""

from dataclasses import replace

import numpy as np
import pytest

from cohort import make_env
from cohort.config import SCENARIOS, get_scenario
from cohort.core.missions import DOCTRINE, MissionType
from cohort.core.orders import HQ_ID, MessageKind
from cohort.env.actions import CATALOG, N_ACTIONS
from cohort.env.observations import OBS_DIM

ORDER_INDICES = [s.index for s in CATALOG if s.kind == "order"]
STAY = 0


def _random_legal(env, obs, rng):
    return {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}


# ------------------------------------------------------------------ #
# the knob itself
# ------------------------------------------------------------------ #


def test_every_preset_carries_its_declared_arm():
    """Shipped presets stay on "full"; only the *_nomask/*_flat variants differ."""
    for name, spec in SCENARIOS.items():
        if name.endswith("_nomask"):
            assert spec.ablation == "nomask"
        elif name.endswith("_flat"):
            assert spec.ablation == "flat"
        else:
            assert spec.ablation == "full", f"{name} must default to the shipped system"


def test_unknown_arm_rejected():
    with pytest.raises(ValueError, match="ablation"):
        replace(get_scenario("squad"), ablation="bogus")


def test_spaces_frozen_across_arms():
    """Same parameter count by construction: identical spaces in every arm."""
    for name in ("squad", "squad_nomask", "squad_flat"):
        env = make_env(name)
        obs, _ = env.reset(seed=5)
        assert env.action_space("SL1").n == N_ACTIONS == 185
        assert env.observation_space("SL1")["observation"].shape == (OBS_DIM,) == (159,)
        for a in env.agents:
            assert obs[a]["observation"].shape == (OBS_DIM,)
            assert obs[a]["action_mask"].shape == (N_ACTIONS,)


def test_full_arm_is_the_shipped_system():
    """An explicit ablation="full" env is bit-identical to the shipped squad."""
    env_a = make_env("squad")
    env_b = make_env(replace(get_scenario("squad"), ablation="full"))
    obs_a, _ = env_a.reset(seed=9)
    obs_b, _ = env_b.reset(seed=9)
    rng = np.random.default_rng(0)
    for _ in range(40):
        assert env_a.agents == env_b.agents
        for a in env_a.agents:
            np.testing.assert_array_equal(obs_a[a]["observation"], obs_b[a]["observation"])
            np.testing.assert_array_equal(obs_a[a]["action_mask"], obs_b[a]["action_mask"])
        if not env_a.agents:
            break
        acts = _random_legal(env_a, obs_a, rng)
        obs_a, ra, *_ = env_a.step(acts)
        obs_b, rb, *_ = env_b.step(dict(acts))
        assert ra == rb


# ------------------------------------------------------------------ #
# arm (ii): nomask — hierarchy without doctrine masks
# ------------------------------------------------------------------ #


def test_nomask_leader_may_order_beyond_doctrine():
    env = make_env("squad_nomask")
    obs, _ = env.reset(seed=3)
    allowed = set(DOCTRINE[MissionType.SEIZE])  # SL1 holds the SEIZE OPORD
    legal = {
        CATALOG[i].order_mission for i in ORDER_INDICES if obs["SL1"]["action_mask"][i]
    }
    assert legal - allowed, "nomask must open doctrine-invalid orders"
    assert MissionType.DEFEND in legal
    assert MissionType.RALLY in legal
    # per-echelon hold authority stays: a TL (authority 1) can never hold DENY
    assert MissionType.DENY not in legal


def test_nomask_riflemen_still_cannot_command():
    """Rank admissibility is untouched: RFN order vocabulary stays empty."""
    env = make_env("squad_nomask")
    obs, _ = env.reset(seed=11)
    rng = np.random.default_rng(0)
    for _ in range(60):
        if not env.agents:
            obs, _ = env.reset()
        for agent in env.agents:
            soldier = env.roster.by_callsign[agent]
            if soldier.effective_authority == 0:
                assert obs[agent]["action_mask"][ORDER_INDICES].sum() == 0
        obs, *_ = env.step(_random_legal(env, obs, rng))


def test_nomask_unmissioned_leader_can_order():
    """"Regardless of their own mission": TL1 holds no mission at t=0 but may
    order under nomask; under full it may not (nothing to derive from)."""
    env = make_env("squad_nomask")
    obs, _ = env.reset(seed=3)
    assert obs["TL1"]["action_mask"][ORDER_INDICES].sum() > 0
    env_full = make_env("squad")
    obs_full, _ = env_full.reset(seed=3)
    assert obs_full["TL1"]["action_mask"][ORDER_INDICES].sum() == 0


def test_nomask_orders_apply_and_cooldown_still_masks():
    env = make_env("squad_nomask")
    obs, _ = env.reset(seed=3)
    # SL1 orders slot 0 (TL1) to HOLD — a doctrine-INVALID derivation of SEIZE
    assert MissionType.HOLD not in DOCTRINE[MissionType.SEIZE]
    order = next(
        s.index
        for s in CATALOG
        if s.kind == "order" and s.order_slot == 0 and s.order_mission is MissionType.HOLD
    )
    assert obs["SL1"]["action_mask"][order]
    acts = {a: STAY for a in env.agents}
    acts["SL1"] = order
    obs, *_ = env.step(acts)
    # the order applied (mission set, WILCO on the net)
    assert env.roster.by_callsign["TL1"].mission.type is MissionType.HOLD
    assert any(m.kind is MessageKind.ACK for m in env.transcript.messages)
    # within the cooldown TL1 (slot 0) cannot be re-tasked, TL2 (slot 1) can
    slot0 = [s.index for s in CATALOG if s.kind == "order" and s.order_slot == 0]
    slot1 = [s.index for s in CATALOG if s.kind == "order" and s.order_slot == 1]
    assert obs["SL1"]["action_mask"][slot0].sum() == 0
    assert obs["SL1"]["action_mask"][slot1].sum() > 0


# ------------------------------------------------------------------ #
# arm (iii): flat — no ranks in effect
# ------------------------------------------------------------------ #


def test_flat_no_orders_for_anyone_ever():
    env = make_env("squad_flat")
    obs, _ = env.reset(seed=11)
    rng = np.random.default_rng(0)
    for _ in range(80):
        if not env.agents:
            obs, _ = env.reset()
        for agent in env.agents:
            assert obs[agent]["action_mask"][ORDER_INDICES].sum() == 0
        obs, *_ = env.step(_random_legal(env, obs, rng))


def test_flat_everyone_holds_the_opord_at_reset():
    env = make_env("squad_flat")
    env.reset(seed=7)
    spec = get_scenario("squad_flat")
    obj = env.world.objective_by_name(spec.root_objective)
    for s in env.roster.soldiers:
        assert s.mission is not None, f"{s.callsign} untasked at reset"
        assert s.mission.type is spec.root_mission
        assert s.mission.objective_id == obj.id
        assert s.mission.issuer_id == HQ_ID
        assert s.mission.step_assigned == 0
    opords = [m for m in env.transcript.messages if m.kind is MessageKind.OPORD]
    assert len(opords) == len(env.roster.soldiers)
    assert {m.recipient_id for m in opords} == {s.id for s in env.roster.soldiers}


def test_flat_team_observation_stays_root_only():
    """Flat RECON: the root keeps the team-adjudicated OPORD (#9); the other
    agents hold personal observation tasks, like subordinates in full."""
    env = make_env(replace(get_scenario("squad_recon"), name="recon_flat", ablation="flat"))
    env.reset(seed=7)
    root = env.roster.root()
    for s in env.roster.soldiers:
        assert s.mission.team_observation is (s is root)


def test_flat_reports_stay_and_command_reward_is_neutral():
    env = make_env("squad_flat")
    obs, _ = env.reset(seed=11)
    sitrep = next(s.index for s in CATALOG if s.kind == "sitrep")
    done = next(s.index for s in CATALOG if s.kind == "done")
    for a in env.agents:
        assert obs[a]["action_mask"][sitrep]
        assert obs[a]["action_mask"][done]  # SEIZE is completable
    rng = np.random.default_rng(1)
    for _ in range(60):
        if not env.agents:
            break
        obs, _r, _t, _tr, infos = env.step(_random_legal(env, obs, rng))
        for a in infos:
            assert infos[a]["components"]["command"] == 0.0
    # contrast: in the full arm the SL accrues command reward immediately
    # (coverage gap at t=1 — its subordinates are still untasked)
    env_full = make_env("squad")
    env_full.reset(seed=11)
    _, _, _, _, infos_full = env_full.step({a: STAY for a in env_full.agents})
    assert infos_full["SL1"]["components"]["command"] != 0.0
