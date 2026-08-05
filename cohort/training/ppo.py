"""Parameter-shared masked PPO for CohortEnv.

One policy network is shared by every agent; rank, mission, and org context
live in the observation, so the network learns rank-conditional behavior
("what does an agent in this position do?") rather than per-agent habits.
Illegal actions are excluded at the distribution level via the action mask,
so rank admissibility holds during exploration, not just after convergence.

The rollout buffer is rectangular over (time, env, agent-slot) with a
validity mask: agents that died (or an env between episodes) simply
contribute no transitions. Truncated episodes bootstrap the final state's
value into the last reward, so timeouts are not mistaken for failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    """Hyperparameters (defaults tuned for the bundled scenarios)."""

    n_envs: int = 8
    horizon: int = 128
    lr: float = 3.0e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    hidden: int = 256
    device: str = "cpu"


def _layer(m: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(m.weight, std)
    nn.init.constant_(m.bias, bias)
    return m


class PolicyNet(nn.Module):
    """Shared actor-critic MLP with action masking."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.torso = nn.Sequential(
            _layer(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            _layer(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        self.pi = _layer(nn.Linear(hidden, n_actions), std=0.01)
        self.v = _layer(nn.Linear(hidden, 1), std=1.0)

    def dist_value(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """Masked action distribution and state value."""
        h = self.torso(obs)
        logits = self.pi(h)
        logits = logits.masked_fill(mask == 0, -1e9)
        return Categorical(logits=logits), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, mask: torch.Tensor, *, greedy: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) actions; returns (action, logprob, value)."""
        dist, value = self.dist_value(obs, mask)
        action = dist.probs.argmax(dim=-1) if greedy else dist.sample()
        return action, dist.log_prob(action), value


@dataclass
class RolloutBuffer:
    """Rectangular (T, V, A) buffers with a validity mask."""

    horizon: int
    n_envs: int
    n_agents: int
    obs_dim: int
    n_actions: int
    obs: np.ndarray = field(init=False)
    masks: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    logprobs: np.ndarray = field(init=False)
    values: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    valid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        t, v, a = self.horizon, self.n_envs, self.n_agents
        self.obs = np.zeros((t, v, a, self.obs_dim), dtype=np.float32)
        self.masks = np.zeros((t, v, a, self.n_actions), dtype=np.int8)
        self.actions = np.zeros((t, v, a), dtype=np.int64)
        self.logprobs = np.zeros((t, v, a), dtype=np.float32)
        self.values = np.zeros((t, v, a), dtype=np.float32)
        self.rewards = np.zeros((t, v, a), dtype=np.float32)
        self.dones = np.zeros((t, v, a), dtype=np.float32)
        self.valid = np.zeros((t, v, a), dtype=bool)

    def compute_gae(
        self, next_values: np.ndarray, next_valid: np.ndarray, gamma: float, lam: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE over each (env, agent) stream, skipping invalid gaps.

        ``next_values``/``next_valid``: value estimates for the state after
        the last stored step, for streams still alive there.
        """
        advantages = np.zeros_like(self.rewards)
        carry_adv = np.zeros((self.n_envs, self.n_agents), dtype=np.float32)
        carry_value = np.where(next_valid, next_values, 0.0).astype(np.float32)
        carry_nonterminal = next_valid.astype(np.float32)
        for t in range(self.horizon - 1, -1, -1):
            valid_t = self.valid[t]
            # streams that terminate at t bootstrap nothing
            nonterminal = np.where(self.dones[t] > 0, 0.0, carry_nonterminal)
            next_val = np.where(self.dones[t] > 0, 0.0, carry_value)
            delta = self.rewards[t] + gamma * next_val * nonterminal - self.values[t]
            adv = delta + gamma * lam * nonterminal * carry_adv
            advantages[t] = np.where(valid_t, adv, 0.0)
            carry_adv = np.where(valid_t, advantages[t], carry_adv)
            carry_value = np.where(valid_t, self.values[t], carry_value)
            carry_nonterminal = np.where(valid_t, 1.0, carry_nonterminal)
        returns = advantages + self.values
        return advantages, returns


def ppo_update(
    net: PolicyNet,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    advantages: np.ndarray,
    returns: np.ndarray,
    cfg: PPOConfig,
) -> dict[str, float]:
    """One PPO update over all valid transitions; returns loss metrics."""
    device = torch.device(cfg.device)
    idx = buffer.valid.reshape(-1)
    flat = lambda arr, extra=():  torch.as_tensor(  # noqa: E731
        arr.reshape(-1, *extra)[idx], device=device
    )
    b_obs = flat(buffer.obs, (buffer.obs_dim,))
    b_masks = flat(buffer.masks, (buffer.n_actions,))
    b_actions = flat(buffer.actions)
    b_logprobs = flat(buffer.logprobs)
    b_advantages = flat(advantages.astype(np.float32))
    b_returns = flat(returns.astype(np.float32))
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

    n = b_obs.shape[0]
    minibatch = max(64, n // cfg.num_minibatches)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    updates = 0
    for _ in range(cfg.update_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, minibatch):
            mb = perm[start : start + minibatch]
            if mb.shape[0] < 2:
                continue
            dist, value = net.dist_value(b_obs[mb], b_masks[mb])
            logprob = dist.log_prob(b_actions[mb])
            entropy = dist.entropy().mean()
            logratio = logprob - b_logprobs[mb]
            ratio = logratio.exp()

            adv = b_advantages[mb]
            pg1 = -adv * ratio
            pg2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
            policy_loss = torch.max(pg1, pg2).mean()
            value_loss = 0.5 * ((value - b_returns[mb]) ** 2).mean()
            loss = policy_loss - cfg.ent_coef * entropy + cfg.vf_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                metrics["approx_kl"] += ((ratio - 1) - logratio).mean().item()
            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.item()
            updates += 1
    return {k: v / max(1, updates) for k, v in metrics.items()}
