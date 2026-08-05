"""End-to-end training smoke: PPO runs, learns nothing dumb, checkpoints load."""

import csv

import numpy as np
import torch

from cohort.env.actions import N_ACTIONS
from cohort.env.observations import OBS_DIM
from cohort.training.ppo import PolicyNet, PPOConfig, RolloutBuffer
from cohort.training.train import Trainer, load_policy


def test_policy_respects_mask():
    net = PolicyNet(OBS_DIM, N_ACTIONS, hidden=32)
    obs = torch.randn(16, OBS_DIM)
    mask = torch.zeros(16, N_ACTIONS, dtype=torch.int8)
    legal = [0, 5, 9]
    mask[:, legal] = 1
    for _ in range(5):
        action, logp, _ = net.act(obs, mask)
        assert all(int(a) in legal for a in action)
        assert torch.isfinite(logp).all()


def test_gae_handles_death_gaps():
    buf = RolloutBuffer(horizon=4, n_envs=1, n_agents=2, obs_dim=3, n_actions=2)
    # agent 0 lives all 4 steps; agent 1 dies at t=1 (done, then absent)
    buf.valid[:, 0, 0] = True
    buf.valid[:2, 0, 1] = True
    buf.rewards[:, 0, 0] = 1.0
    buf.rewards[1, 0, 1] = -1.0
    buf.dones[1, 0, 1] = 1.0
    buf.values[:] = 0.5
    next_values = np.array([[0.7, 0.0]], dtype=np.float32)
    next_valid = np.array([[True, False]])
    adv, ret = buf.compute_gae(next_values, next_valid, gamma=0.99, lam=0.95)
    assert np.isfinite(adv).all() and np.isfinite(ret).all()
    assert adv[2, 0, 1] == 0.0 and adv[3, 0, 1] == 0.0, "no advantage after death"
    # the dying step bootstraps nothing: delta = r - V = -1 - 0.5
    assert np.isclose(adv[1, 0, 1], -1.5)


def test_trainer_end_to_end(tmp_path):
    cfg = PPOConfig(n_envs=2, horizon=32)
    trainer = Trainer("fireteam", cfg, tmp_path / "run", seed=5, tensorboard=False)
    trainer.train(total_steps=256)

    with (tmp_path / "run" / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 2
    for row in rows:
        assert np.isfinite(float(row["policy_loss"]))
        assert np.isfinite(float(row["value_loss"]))
        assert float(row["entropy"]) > 0

    net, ckpt = load_policy(tmp_path / "run" / "ckpt_latest.pt")
    assert ckpt["scenario"] == "fireteam"
    obs = torch.zeros(1, OBS_DIM)
    mask = torch.ones(1, N_ACTIONS, dtype=torch.int8)
    action, *_ = net.act(obs, mask, greedy=True)
    assert 0 <= int(action) < N_ACTIONS


def test_evaluate_random_baseline():
    from cohort.training.evaluate import evaluate

    summary = evaluate(None, scenario="fireteam", episodes=2, seed=7)
    assert summary["episodes"] == 2
    assert 0.0 <= summary["success_rate"] <= 1.0
    assert np.isfinite(summary["mean_return"])


def test_eval_episodes_reproduce_standalone():
    """Episode k of a sampled evaluation must reproduce independently: its
    RNG streams may not depend on how many draws episodes 0..k-1 consumed."""
    from cohort.env.cohort_env import make_env
    from cohort.training.evaluate import _seeded_episode

    net = PolicyNet(OBS_DIM, N_ACTIONS, hidden=32)  # untrained → sampled actions
    env = make_env("fireteam")
    seq = [_seeded_episode(env, net, 300 + i) for i in range(3)]
    env_alone = make_env("fireteam")
    alone = _seeded_episode(env_alone, net, 302)
    assert (alone["outcome"], alone["length"]) == (seq[2]["outcome"], seq[2]["length"])
    assert env_alone.transcript.render() == env.transcript.render(), (
        "episode 2 standalone must be byte-identical to episode 2 in sequence"
    )
