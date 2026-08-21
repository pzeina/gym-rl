"""Observation-width bisect: the `core` profile and the arm built on it.

Four v1.10 runs collapsed and four converged. Three explanations were tested
and killed — `done_false` (squad_screen_v5 reproduced the collapse at -0.5),
`contact_redundant` (squad_v6 ran at -0.02 and collapsed anyway), and learning
rate (squad_screen_v7 at 1e-4 died rather than stalled). The space break is
what remains, by elimination. `core` turns that into a measurement.

A bisect is only worth running if the arm differs from its control in exactly
ONE way, so that is what these tests pin: same rewards, same geometry, same
OpFor, same step budget, same ablation arm — different input width.
"""

from dataclasses import replace

import numpy as np
import pytest

from cohort import make_env
from cohort.config import get_scenario
from cohort.env.observations import (
    CORE_PATCH_RADIUS,
    OBS_DIM,
    OBS_PROFILES,
    PATCH_RADIUS,
    obs_dim,
    patch_radius,
)


def test_core_is_the_pre_v110_width():
    """220 - 54 = 166: tempo (2) + cover (3) + sitrep-due (1) + patch (48) —
    plus the degraded-communications blocks (94 acoustic + 14 cohesion),
    appended to BOTH profiles so the bisect keeps its single variable."""
    assert obs_dim("full") == OBS_DIM == 220 + 94 + 14 + 23
    assert obs_dim("core") == 166 + 94 + 14 + 23
    widened_patch = (2 * PATCH_RADIUS + 1) ** 2 * 2 - (2 * CORE_PATCH_RADIUS + 1) ** 2 * 2
    assert widened_patch == 48
    assert obs_dim("full") - obs_dim("core") == 2 + 3 + 1 + widened_patch


def test_an_unknown_profile_is_refused_not_guessed():
    with pytest.raises(ValueError, match="Unknown observation profile"):
        obs_dim("v1.9")
    with pytest.raises(ValueError, match="Unknown observation profile"):
        replace(get_scenario("squad_screen"), observation_profile="narrow")


@pytest.mark.parametrize("profile", OBS_PROFILES)
def test_every_profile_writes_exactly_its_own_width(profile):
    """The layout assertion inside build_observation is the real guard; this
    proves it is exercised for both profiles and that the space agrees."""
    spec = replace(get_scenario("squad_screen"), observation_profile=profile)
    env = make_env(spec)
    obs, _ = env.reset(seed=3)
    width = obs_dim(profile)
    for a in env.agents:
        assert obs[a]["observation"].shape == (width,)
        assert np.all(np.isfinite(obs[a]["observation"]))
    assert env.observation_space(env.agents[0])["observation"].shape == (width,)
    assert patch_radius(profile) == (PATCH_RADIUS if profile == "full" else CORE_PATCH_RADIUS)


def test_the_bisect_arm_moves_one_variable_and_no_other():
    """squad_screen_core vs squad_screen: everything but the width is held."""
    control = get_scenario("squad_screen")
    arm = get_scenario("squad_screen_core")

    assert arm.observation_profile == "core"
    assert control.observation_profile == "full"

    held = {
        f
        for f in control.__dataclass_fields__
        if f not in ("name", "description", "observation_profile", "experiment_arm")
    }
    differing = {f for f in held if getattr(control, f) != getattr(arm, f)}
    assert not differing, f"the bisect arm also moved: {sorted(differing)}"


def test_the_shared_prefix_is_bit_identical_up_to_the_first_dropped_block():
    """The two profiles are the same vector until tempo, which `core` omits.

    If the prefix drifted, a difference in outcome could be the encoding
    rather than the width, and the bisect would prove nothing.
    """
    from cohort.env.observations import OFF_TEMPO

    full = make_env(get_scenario("squad_screen"))
    core = make_env(get_scenario("squad_screen_core"))
    obs_full, _ = full.reset(seed=11)
    obs_core, _ = core.reset(seed=11)
    assert set(obs_full) == set(obs_core)
    for a in obs_full:
        np.testing.assert_array_equal(
            obs_full[a]["observation"][:OFF_TEMPO],
            obs_core[a]["observation"][:OFF_TEMPO],
        )
        np.testing.assert_array_equal(obs_full[a]["action_mask"], obs_core[a]["action_mask"])


def test_a_core_run_trains_and_checkpoints_at_its_own_width(tmp_path):
    """The trainer must size the net off the env, not the module constant —
    otherwise a core run silently builds a 220-wide first layer."""
    import torch

    from cohort.training.train import PPOConfig, Trainer

    cfg = PPOConfig(n_envs=2, horizon=32)
    trainer = Trainer("squad_screen_core", cfg, tmp_path / "run", seed=0, tensorboard=False)
    core = obs_dim("core")
    assert core != OBS_DIM
    assert trainer.obs_dim == core
    assert next(trainer.net.parameters()).shape[1] == core, "first layer is core-wide"
    trainer.train(total_steps=256)
    ckpt = torch.load(trainer.save_checkpoint("ckpt_test.pt"), weights_only=True)
    assert ckpt["obs_dim"] == core, "the checkpoint must record the width it was built at"
