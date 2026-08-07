"""Reward weights on the CLI (v1.12).

Until this landed, a reward weight could only be changed by editing the tree.
That blocked the one run ROADMAP kept naming as missing — ``squad_v9``, which
separates ``d44ee8d`` from the ``done_false`` revert on the five confounded
arms — and it made every price experiment unrecorded by construction, since
``economics.json`` dumped ``RewardConfig()`` rather than what the run used.

The hazards these tests pin are all silent ones: a typo'd key, a boolean read
by truthiness, and an ``economics.json`` that disagrees with the policy. Each
would produce a run that trains fine and means something other than it says.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, fields

import pytest

from cohort.env.cohort_env import make_env
from cohort.env.rewards import RewardConfig


def test_an_override_changes_that_price_and_nothing_else():
    cfg = RewardConfig.from_overrides(["done_false=-2.0"])
    assert cfg.done_false == -2.0
    baseline = asdict(RewardConfig())
    changed = {k: v for k, v in asdict(cfg).items() if baseline[k] != v}
    assert changed == {"done_false": -2.0}


def test_several_overrides_apply_together():
    cfg = RewardConfig.from_overrides(["done_false=-2.0", "death=-3", "success_team=80"])
    assert (cfg.done_false, cfg.death, cfg.success_team) == (-2.0, -3.0, 80.0)


def test_no_overrides_is_exactly_the_default():
    assert asdict(RewardConfig.from_overrides([])) == asdict(RewardConfig())


def test_every_field_is_reachable():
    """A field that cannot be set from the CLI is a hole this test closes.

    The parser reads the dataclass, so this is really asserting that no field
    has a type the coercer does not handle — the failure mode being a new
    weight that silently cannot be swept.
    """
    for f in fields(RewardConfig):
        raw = "true" if f.type == "bool" else "1"
        assert RewardConfig.from_overrides([f"{f.name}={raw}"]) is not None


@pytest.mark.parametrize(
    ("typo", "meant"),
    [("done_flase", "done_false"),      # transposition
     ("sucess_team", "success_team"),   # dropped letter
     ("deth", "death")],
)
def test_unknown_key_is_rejected_and_suggests_the_near_miss(typo, meant):
    # a typo silently ignored would train the DEFAULT price under an
    # economics.json claiming otherwise — the exact confound the audit exists
    # to catch, reintroduced one layer lower
    with pytest.raises(ValueError, match="unknown reward key") as exc:
        RewardConfig.from_overrides([f"{typo}=-2.0"])
    hint = str(exc.value).split("Valid keys:")[0]  # not merely present in the full list
    assert meant in hint


def test_malformed_pairs_are_rejected():
    for bad in ["done_false", "=-2.0", "done_false:-2.0"]:
        with pytest.raises(ValueError):
            RewardConfig.from_overrides([bad])


def test_a_boolean_is_read_by_name_not_by_truthiness():
    """``bool("false")`` is ``True``; a naive coercer trains the opposite run."""
    assert RewardConfig.from_overrides(["fire_discipline=false"]).fire_discipline is False
    assert RewardConfig.from_overrides(["fire_discipline=0"]).fire_discipline is False
    assert RewardConfig.from_overrides(["fire_discipline=true"]).fire_discipline is True
    with pytest.raises(ValueError, match="expected a boolean"):
        RewardConfig.from_overrides(["fire_discipline=-2.0"])


def test_an_integer_field_stays_an_integer():
    cfg = RewardConfig.from_overrides(["sitrep_interval=40"])
    assert cfg.sitrep_interval == 40
    assert isinstance(cfg.sitrep_interval, int)
    with pytest.raises(ValueError, match="expected int"):
        RewardConfig.from_overrides(["sitrep_interval=40.5"])


def test_a_non_numeric_value_is_rejected():
    with pytest.raises(ValueError, match="expected float"):
        RewardConfig.from_overrides(["done_false=cheap"])


def test_the_env_actually_charges_the_overridden_price():
    """End of the chain: the parsed config has to reach the ledger."""
    cfg = RewardConfig.from_overrides(["time_penalty=-0.5"])
    env = make_env("fireteam", reward_config=cfg)
    assert env.rewards_cfg.time_penalty == -0.5
    env.reset(seed=0)
    _, rewards, *_ = env.step({a: 0 for a in env.agents})
    assert rewards  # every agent pays the (now large) time penalty each step
    assert min(rewards.values()) <= -0.5


@pytest.mark.slow
def test_a_run_records_the_prices_it_actually_used(tmp_path):
    """``economics.json`` must describe the run, not the tree.

    ``run_report.py --vs`` diffs this file to decide whether an A/B is
    single-variable (assurance #20). If overrides were left out of it, every
    CLI-driven experiment would read as a no-op change against its baseline —
    an audit that reports "nothing differs" about the one thing that does.
    """
    run = tmp_path / "runs" / "cli_smoke"  # train.py writes runs/<name> under cwd
    subprocess.run(
        [sys.executable, "-m", "cohort.training.train",
         "--scenario", "fireteam", "--total-steps", "600", "--n-envs", "2",
         "--horizon", "32", "--run-name", run.name, "--no-tb", "--no-eval",
         "--reward", "done_false=-2.0", "--reward", "death=-3.0"],
        check=True, capture_output=True, text=True, cwd=tmp_path,
    )
    econ = json.loads((run / "economics.json").read_text())
    assert econ["rewards"]["done_false"] == -2.0
    assert econ["rewards"]["death"] == -3.0
    assert econ["reward_overrides"] == ["done_false=-2.0", "death=-3.0"]
    # untouched weights still report their defaults
    assert econ["rewards"]["success_team"] == RewardConfig().success_team

    # ...and the checkpoint carries them too, so evaluate() can score the
    # policy under its own prices instead of the tree's
    import torch

    ckpt = torch.load(run / "ckpt_latest.pt", map_location="cpu", weights_only=True)
    assert ckpt["reward_config"]["done_false"] == -2.0
    assert RewardConfig(**ckpt["reward_config"]).death == -3.0


def test_a_pre_v112_checkpoint_still_evaluates_under_the_defaults():
    """The published fleet has no ``reward_config`` key and must not break.

    Those runs trained on the tree defaults, so the fallback is not a
    compromise — it is the correct reconstruction. This pins the fallback
    itself: ``evaluate`` leaves ``rewards`` as ``None`` for such a checkpoint
    and hands that to ``make_env``, which must produce the defaults.
    """
    env = make_env("fireteam", reward_config=None)
    assert asdict(env.rewards_cfg) == asdict(RewardConfig())


def test_a_round_trip_through_asdict_preserves_every_price():
    """How the checkpoint stores and evaluate restores; must be lossless."""
    cfg = RewardConfig.from_overrides(["done_false=-2.0", "fire_discipline=false"])
    assert asdict(RewardConfig(**asdict(cfg))) == asdict(cfg)
