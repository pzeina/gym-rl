"""Train the cohort with masked PPO.

Usage:
    python -m cohort.training.train --scenario fireteam --total-steps 500000
    python -m cohort.training.train --scenario squad --n-envs 8 --device cpu

Outputs land in runs/<run-name>/: metrics.csv, training_curves.png,
checkpoints (latest/best), TensorBoard logs, and a post-training eval GIF.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from cohort.config import get_scenario
from cohort.env.actions import N_ACTIONS
from cohort.env.cohort_env import CohortEnv, make_env
from cohort.env.observations import OBS_DIM
from cohort.env.rewards import COMPONENTS
from cohort.training.ppo import PolicyNet, PPOConfig, RolloutBuffer, ppo_update

METRIC_FIELDS = [
    "iteration",
    "env_steps",
    "ep_return",
    "ep_length",
    "success_rate",
    "success_rate_rolling",
    "entropy",
    "policy_loss",
    "value_loss",
    "approx_kl",
    "sps",
    *[f"comp_{c}" for c in COMPONENTS],
]


class Trainer:
    """Vectorized rollout collection + PPO updates for CohortEnv."""

    def __init__(
        self,
        scenario: str,
        cfg: PPOConfig,
        run_dir: Path,
        seed: int = 1,
        *,
        tensorboard: bool = True,
        init_from: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.scenario = scenario

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.envs: list[CohortEnv] = [make_env(scenario) for _ in range(cfg.n_envs)]
        self.agent_ids = list(self.envs[0].possible_agents)
        self.slot = {a: i for i, a in enumerate(self.agent_ids)}
        self.n_agents = len(self.agent_ids)
        self.current_obs: list[dict] = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=seed + i * 1000)
            self.current_obs.append(obs)

        self.device = torch.device(cfg.device)
        self.net = PolicyNet(OBS_DIM, N_ACTIONS, cfg.hidden).to(self.device)
        if init_from is not None:
            ckpt = torch.load(init_from, map_location=self.device, weights_only=True)
            self.net.load_state_dict(ckpt["model"])
            print(f"initialized weights from {init_from} (scenario {ckpt.get('scenario')}, {ckpt.get('env_steps')} steps)")
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)

        self.env_steps = 0
        self.iteration = 0
        self.recent_outcomes: deque[str] = deque(maxlen=100)
        self.best_rolling_success = -1.0
        self._ep_return = [0.0] * cfg.n_envs
        self._ep_len = [0] * cfg.n_envs

        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(run_dir / "tb"))
        self.metrics_path = run_dir / "metrics.csv"
        if not self.metrics_path.exists():
            with self.metrics_path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=METRIC_FIELDS).writeheader()

    # ------------------------------------------------------------------ #

    def _forward_present(
        self, rows: list[tuple[int, str]], *, greedy: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = np.stack([self.current_obs[e][a]["observation"] for e, a in rows])
        mask = np.stack([self.current_obs[e][a]["action_mask"] for e, a in rows])
        t_obs = torch.as_tensor(obs, device=self.device)
        t_mask = torch.as_tensor(mask, device=self.device)
        action, logp, value = self.net.act(t_obs, t_mask, greedy=greedy)
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()

    @torch.no_grad()
    def _values_of(self, obs_list: list[dict]) -> np.ndarray:
        obs = torch.as_tensor(np.stack([o["observation"] for o in obs_list]), device=self.device)
        mask = torch.as_tensor(np.stack([o["action_mask"] for o in obs_list]), device=self.device)
        _, value = self.net.dist_value(obs, mask)
        return value.cpu().numpy()

    def collect(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Fill the buffer with cfg.horizon steps from every env."""
        cfg = self.cfg
        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        outcomes: list[str] = []
        comp_sums = dict.fromkeys(COMPONENTS, 0.0)
        agent_steps = 0

        for t in range(cfg.horizon):
            rows = [(e, a) for e, env in enumerate(self.envs) for a in env.agents]
            if rows:
                actions, logps, values = self._forward_present(rows)
            row_of = {key: i for i, key in enumerate(rows)}

            for e, env in enumerate(self.envs):
                present = list(env.agents)
                if not present:
                    continue
                act_dict = {a: int(actions[row_of[(e, a)]]) for a in present}
                obs_next, rewards, terms, truncs, infos = env.step(act_dict)

                trunc_rows = []
                for a in present:
                    i = row_of[(e, a)]
                    s = self.slot[a]
                    buffer.obs[t, e, s] = self.current_obs[e][a]["observation"]
                    buffer.masks[t, e, s] = self.current_obs[e][a]["action_mask"]
                    buffer.actions[t, e, s] = actions[i]
                    buffer.logprobs[t, e, s] = logps[i]
                    buffer.values[t, e, s] = values[i]
                    buffer.rewards[t, e, s] = rewards[a]
                    buffer.dones[t, e, s] = float(terms[a] or truncs[a])
                    buffer.valid[t, e, s] = True
                    if truncs[a] and not terms[a]:
                        trunc_rows.append((a, s))
                    for comp, val in infos[a]["components"].items():
                        comp_sums[comp] += val
                    self._ep_return[e] += rewards[a]
                agent_steps += len(present)
                self._ep_len[e] += 1

                # truncation: bootstrap the final state's value into the reward
                if trunc_rows:
                    vals = self._values_of([obs_next[a] for a, _ in trunc_rows])
                    for (_a, s), v in zip(trunc_rows, vals, strict=True):
                        buffer.rewards[t, e, s] += cfg.gamma * float(v)

                if env.agents:
                    self.current_obs[e] = obs_next
                else:  # episode over
                    ep_returns.append(self._ep_return[e] / self.n_agents)
                    ep_lengths.append(self._ep_len[e])
                    outcomes.append(env.outcome or "timeout")
                    self.recent_outcomes.append(outcomes[-1])
                    self._ep_return[e] = 0.0
                    self._ep_len[e] = 0
                    obs0, _ = env.reset()
                    self.current_obs[e] = obs0
            self.env_steps += cfg.n_envs

        # bootstrap values for streams still alive at the horizon boundary
        next_values = np.zeros((cfg.n_envs, self.n_agents), dtype=np.float32)
        next_valid = np.zeros((cfg.n_envs, self.n_agents), dtype=bool)
        rows = [(e, a) for e, env in enumerate(self.envs) for a in env.agents]
        if rows:
            vals = self._values_of([self.current_obs[e][a] for e, a in rows])
            for (e, a), v in zip(rows, vals, strict=True):
                next_values[e, self.slot[a]] = v
                next_valid[e, self.slot[a]] = True
        self._bootstrap = (next_values, next_valid)

        n_eps = max(1, len(ep_returns))
        stats = {
            "ep_return": float(np.mean(ep_returns)) if ep_returns else float("nan"),
            "ep_length": float(np.mean(ep_lengths)) if ep_lengths else float("nan"),
            "success_rate": sum(o == "success" for o in outcomes) / n_eps if outcomes else 0.0,
            "success_rate_rolling": (
                sum(o == "success" for o in self.recent_outcomes) / len(self.recent_outcomes)
                if self.recent_outcomes
                else 0.0
            ),
        }
        for comp in COMPONENTS:
            stats[f"comp_{comp}"] = comp_sums[comp] / max(1, agent_steps)
        return stats

    # ------------------------------------------------------------------ #

    def train(self, total_steps: int) -> None:
        """Main loop: collect → GAE → update → log, until total_steps."""
        cfg = self.cfg
        start = time.time()
        while self.env_steps < total_steps:
            self.iteration += 1
            if cfg.anneal_lr:
                frac = 1.0 - min(1.0, self.env_steps / total_steps)
                for group in self.optimizer.param_groups:
                    group["lr"] = cfg.lr * max(0.05, frac)

            buffer = RolloutBuffer(cfg.horizon, cfg.n_envs, self.n_agents, OBS_DIM, N_ACTIONS)
            t0 = time.time()
            stats = self.collect(buffer)
            next_values, next_valid = self._bootstrap
            advantages, returns = buffer.compute_gae(next_values, next_valid, cfg.gamma, cfg.gae_lambda)
            losses = ppo_update(self.net, self.optimizer, buffer, advantages, returns, cfg)
            sps = cfg.horizon * cfg.n_envs / max(1e-9, time.time() - t0)

            row = {
                "iteration": self.iteration,
                "env_steps": self.env_steps,
                "sps": round(sps),
                **{k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()},
                **{k: round(v, 6) for k, v in losses.items()},
            }
            with self.metrics_path.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore").writerow(row)
            if self.writer is not None:
                for key in ("ep_return", "success_rate_rolling", "entropy", "policy_loss", "value_loss"):
                    val = row.get(key, stats.get(key, losses.get(key)))
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        self.writer.add_scalar(key, float(val), self.env_steps)

            if self.iteration % 5 == 0 or self.env_steps >= total_steps:
                elapsed = time.time() - start
                print(
                    f"iter {self.iteration:>4} | steps {self.env_steps:>8,} | "
                    f"return {stats['ep_return']:>7.2f} | success {stats['success_rate_rolling']:.0%} | "
                    f"len {stats['ep_length']:>5.0f} | ent {losses['entropy']:.2f} | "
                    f"sps {sps:>5.0f} | {elapsed:>5.0f}s"
                )
            self.save_checkpoint("ckpt_latest.pt")
            if stats["success_rate_rolling"] > self.best_rolling_success and len(self.recent_outcomes) >= 20:
                self.best_rolling_success = stats["success_rate_rolling"]
                self.save_checkpoint("ckpt_best.pt")
        if self.writer is not None:
            self.writer.close()

    def save_checkpoint(self, name: str) -> Path:
        """Persist model weights + everything needed to reload them."""
        path = self.run_dir / name
        torch.save(
            {
                "model": self.net.state_dict(),
                "obs_dim": OBS_DIM,
                "n_actions": N_ACTIONS,
                "hidden": self.cfg.hidden,
                "scenario": self.scenario,
                "iteration": self.iteration,
                "env_steps": self.env_steps,
                "ppo_config": asdict(self.cfg),
            },
            path,
        )
        return path


def load_policy(checkpoint: str | Path, device: str = "cpu") -> tuple[PolicyNet, dict]:
    """Load a trained policy from a checkpoint file."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    net = PolicyNet(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden"]).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, ckpt


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the cohort with masked PPO.")
    parser.add_argument("--scenario", default="fireteam", help="scenario preset name")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--init-from", default=None, help="checkpoint to initialize weights from (curriculum)")
    parser.add_argument("--no-tb", action="store_true", help="disable TensorBoard logging")
    parser.add_argument("--no-eval", action="store_true", help="skip post-training eval + GIF")
    args = parser.parse_args()

    get_scenario(args.scenario)  # fail fast on typos
    run_name = args.run_name or f"{args.scenario}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_name
    cfg = PPOConfig(
        n_envs=args.n_envs,
        horizon=args.horizon,
        lr=args.lr,
        ent_coef=args.ent_coef,
        device=args.device,
    )
    print(f"training scenario={args.scenario} → {run_dir}")
    (run_dir / ".").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"scenario": args.scenario, "seed": args.seed, "total_steps": args.total_steps, **asdict(cfg)}, indent=2)
    )
    trainer = Trainer(
        args.scenario, cfg, run_dir, seed=args.seed, tensorboard=not args.no_tb, init_from=args.init_from
    )
    trainer.train(args.total_steps)

    from cohort.viz.plots import plot_training

    print(f"curves → {plot_training(run_dir)}")
    if not args.no_eval:
        from cohort.training.evaluate import evaluate

        ckpt = run_dir / ("ckpt_best.pt" if (run_dir / "ckpt_best.pt").exists() else "ckpt_latest.pt")
        evaluate(str(ckpt), episodes=20, gif_path=str(run_dir / "eval.gif"), transcript_path=str(run_dir / "eval_transcript.txt"))


if __name__ == "__main__":
    main()
