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


def test_defend_run_logs_the_positional_gate_columns(tmp_path):
    """issue #11: a DEFEND retrain must show its fight disposition live.

    ``fireteam_defend_v7`` burned a 3M-step budget before anyone saw that the
    unit had walked off the position; these two columns make the collapse
    visible in metrics.csv while the run is still cheap to kill. They are
    written for DEFEND roots only — every other root pays nothing.
    """
    cfg = PPOConfig(n_envs=2, horizon=32)
    defend = Trainer("fireteam_defend", cfg, tmp_path / "d", seed=3, tensorboard=False)
    assert defend._track_disposition
    defend.train(total_steps=256)
    with (tmp_path / "d" / "metrics.csv").open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert {"cover_under_threat", "objective_dist_under_threat"} <= set(reader.fieldnames)
    # blank/NaN while the preparation period keeps the OpFor off the position:
    # "no firefight" must never be logged as "fought in the open on the OBJ"
    measured = [r for r in rows if r["cover_under_threat"] not in ("", "nan")]
    for row in measured:
        assert 0.0 <= float(row["cover_under_threat"]) <= 1.0
        assert float(row["objective_dist_under_threat"]) >= 0.0

    seize = Trainer("fireteam", cfg, tmp_path / "s", seed=3, tensorboard=False)
    assert not seize._track_disposition
    seize.train(total_steps=128)
    with (tmp_path / "s" / "metrics.csv").open() as f:
        assert all(r["cover_under_threat"] == "" for r in csv.DictReader(f))


def test_run_logs_the_clock_and_the_whole_net(tmp_path):
    """issue #18: the stall signature must be visible while a run is cheap to kill.

    ``tx_per_agent_step`` charges by design — SYNC PROPOSE / GO are voice and
    cost nothing — so the collapsed ``squad_screen_v4`` read 0.029 ("the whole
    radio goes quiet") while carrying 2.5x the messages of its own successful
    checkpoint. Counting every message alongside the charged ones is what makes
    that readable, and ``timeout_rate_rolling`` names the failure mode over the
    same window ``success_rate_rolling`` uses.
    """
    cfg = PPOConfig(n_envs=2, horizon=32)
    trainer = Trainer("fireteam", cfg, tmp_path / "run", seed=5, tensorboard=False)
    trainer.train(total_steps=256)
    with (tmp_path / "run" / "metrics.csv").open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert {"messages_per_agent_step", "timeout_rate_rolling"} <= set(reader.fieldnames)
    for row in rows:
        messages = float(row["messages_per_agent_step"])
        assert messages >= float(row["tx_per_agent_step"]), (
            "charged transmissions are a subset of everything said"
        )
        assert 0.0 <= float(row["timeout_rate_rolling"]) <= 1.0
    # the two rolling outcome shares are shares of the same window
    last = rows[-1]
    assert float(last["success_rate_rolling"]) + float(last["timeout_rate_rolling"]) <= 1.0


def test_rolling_best_gate_requires_full_window_turnover():
    """D4: ckpt_best may only be written once the rolling window contains ONLY
    post-start episodes (the deque fully turned over). Simulates the Trainer's
    bookkeeping for the documented failure — a fine-tune whose strong parent
    pins rolling at 1.0 over the first ~20 episodes (which the old >=20 gate
    saved at ~3-4k steps and then froze forever) before collapsing."""
    from collections import deque

    from cohort.training.train import best_save_gate

    window: deque[str] = deque(maxlen=100)
    episodes_seen = 0
    best = -1.0

    def rolling() -> float:
        return sum(o == "success" for o in window) / len(window)

    # phase 1 — the strong parent: 30 straight successes. The old gate saved
    # at episode 20 with rolling 1.0; the fixed gate must refuse every one.
    for _ in range(30):
        window.append("success")
        episodes_seen += 1
        assert not best_save_gate(episodes_seen, window.maxlen, rolling(), best)

    # phase 2 — the fine-tune collapses (the squad_v3d signature): rolling
    # decays, still no save allowed while the window has not turned over.
    for _ in range(69):
        window.append("timeout")
        episodes_seen += 1
        assert not best_save_gate(episodes_seen, window.maxlen, rolling(), best)

    # phase 3 — the 100th episode completes the turnover: saving opens, and
    # the first save reflects a full window of THIS run's training.
    window.append("timeout")
    episodes_seen += 1
    assert episodes_seen == window.maxlen
    assert best_save_gate(episodes_seen, window.maxlen, rolling(), best)
    best = rolling()
    assert best < 0.5, "the parent's early 1.0 was never allowed to pin the best"

    # phase 4 — recovery to a genuine peak beats the recorded best and saves.
    for _ in range(80):
        window.append("success")
        episodes_seen += 1
    assert best_save_gate(episodes_seen, window.maxlen, rolling(), best)


def test_rolling_best_gate_refuses_a_mute_commander(tmp_path):
    """v1.20: winning is not sufficient to be a run's ``ckpt_best``.

    Measured across six squad arms, ``ckpt_best`` sat at a closed-on-root-report
    of 0.00-0.01 while the FINAL policy of the same run reported normally
    (0.82-0.92): the completion report is learned LATE, so selecting on rolling
    success alone reliably picks the window before the commander starts
    reporting. Published numbers were never affected — the project quotes the
    FINAL policy — but ``ckpt_best`` is what ``cohort.play`` and every
    spot-check load by default.
    """
    from cohort.training.train import best_save_gate

    # Selection is lexicographic, not a veto. A mute run still records a best —
    # a veto leaves runs with NO ckpt_best, which fails baseline.py's
    # "every checkpoint loadable" (verified on a 120k-step smoke run).
    assert best_save_gate(100, 100, 1.0, -1.0, 0.0, False)
    assert best_save_gate(100, 100, 1.0, -1.0, None, False)

    # The inversion this exists to prevent: a mute 0.95 recorded early must not
    # lock out the reporting 0.90 that follows it. Reporting always supersedes.
    assert best_save_gate(100, 100, 0.90, 0.95, 0.82, best_was_reporting=False)

    # …and once the best is reporting, a mute window may never take it back,
    # however well it scores.
    assert not best_save_gate(100, 100, 1.0, 0.90, 0.0, best_was_reporting=True)
    assert not best_save_gate(100, 100, 1.0, 0.90, None, best_was_reporting=True)

    # among windows of the same kind, higher rolling success still wins
    assert best_save_gate(100, 100, 0.95, 0.90, 0.82, best_was_reporting=True)
    assert not best_save_gate(100, 100, 0.85, 0.90, 0.82, best_was_reporting=True)

    # and it composes with D4 rather than replacing it: a reporting commander
    # still may not save before the window has turned over
    assert not best_save_gate(99, 100, 1.0, -1.0, 0.9)


def test_best_save_reporting_side_is_floored_on_success():
    """refs #57 — a two-episode reporting rate may not outrank a winning policy.

    ``closed_on_root_report_rate``'s denominator is ENDEXes sent and the env
    sends ENDEX only on a WIN, so the reporting sample is conditioned on success
    and is thinnest exactly where success is worst. At 2% rolling success it is
    a couple of episodes and reads 0.500 — ``ROOT_REPORT_CLOSE_FLOOR`` exactly —
    on a coin toss.

    This is not hypothetical: ``patrol_brique_v19_rdb3_seed13`` was selected
    that way at iteration 25 of 2930 and, because ``best_was_reporting`` is
    absorbing, never replaced it. Its ``ckpt_best`` evaluates at 0.17 success
    against a final policy at 0.99.
    """
    from cohort.training.train import (
        BEST_SAVE_SUCCESS_FLOOR,
        best_save_gate,
        selects_on_reporting,
    )

    # THE REGRESSION, stated precisely. v19's actual selecting window (2%
    # success, closed-on-root at the floor) still SAVES — selection is not a
    # veto and this is the run's first checkpoint. What changes is that it no
    # longer takes the REPORTING side of the order…
    assert best_save_gate(100, 100, 0.02, -1.0, 0.500, best_was_reporting=False)
    assert not selects_on_reporting(0.02, 0.500)

    # …so it does not set the absorbing flag, and the later 100%-success window
    # supersedes it. Under the old rule the flag went True here and locked
    # ckpt_best at iteration 25 for the remaining 99.1% of the run: that is the
    # whole defect, and this pair of asserts is the difference.
    assert best_save_gate(100, 100, 1.0, 0.02, 0.0, best_was_reporting=False)
    assert not best_save_gate(100, 100, 1.0, 0.02, 0.0, best_was_reporting=True)

    # A coin-toss rate at the floor cannot outrank a winning policy, whatever
    # the reporting number says.
    assert not selects_on_reporting(0.02, 1.0)

    # Still lexicographic above the floor: a reporting 0.60 beats a mute 0.95.
    assert selects_on_reporting(0.60, 0.82)
    assert best_save_gate(100, 100, 0.60, 0.95, 0.82, best_was_reporting=False)

    # NOT a second success gate — a below-floor run still records a ckpt_best
    # through the success branch, so baseline.py's "every checkpoint loadable"
    # and publish_baseline's artifact both survive a run that never reports.
    assert best_save_gate(100, 100, 0.02, -1.0, None, best_was_reporting=False)
    assert best_save_gate(100, 100, 0.30, 0.02, 0.500, best_was_reporting=False)

    # The floor is on rolling success, and it is inclusive.
    assert selects_on_reporting(BEST_SAVE_SUCCESS_FLOOR, 0.82)
    assert not selects_on_reporting(BEST_SAVE_SUCCESS_FLOOR - 0.01, 0.82)


def test_trainer_records_the_flag_the_gate_actually_used():
    """refs #57 — the absorbing flag must record ``selects_on_reporting``.

    If the trainer recorded a bare ``is_reporting`` it would desynchronise from
    the ordering the gate just applied: a below-floor window would save on the
    success branch, then set ``best_was_reporting=True``, and every later
    reporting window would be compared as if the best were already reporting.
    """
    import inspect

    from cohort.training import train as train_mod

    src = inspect.getsource(train_mod.Trainer.train)
    assignment = [ln for ln in src.splitlines() if "self.best_was_reporting =" in ln]
    assert assignment, "trainer no longer records best_was_reporting"
    assert all("selects_on_reporting" in ln for ln in assignment), (
        "best_was_reporting must be recorded with selects_on_reporting, not a "
        "bare is_reporting — see refs #57"
    )


def test_trainer_counts_episodes_for_the_best_gate(tmp_path):
    """The Trainer wires the D4 gate: episodes_seen tracks completed episodes
    and no ckpt_best exists while the window has not fully turned over."""
    cfg = PPOConfig(n_envs=2, horizon=32)
    trainer = Trainer("fireteam", cfg, tmp_path / "run", seed=11, tensorboard=False)
    assert trainer.episodes_seen == 0
    trainer.train(total_steps=256)  # 256 steps: nowhere near 100 fireteam episodes
    assert trainer.episodes_seen == len(trainer.recent_outcomes)
    assert trainer.episodes_seen < (trainer.recent_outcomes.maxlen or 0)
    assert not (tmp_path / "run" / "ckpt_best.pt").exists(), (
        "ckpt_best must not be written before the rolling window turns over"
    )
    assert (tmp_path / "run" / "ckpt_latest.pt").exists()


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
