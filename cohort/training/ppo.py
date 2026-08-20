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

v1.11 — two changes to how the critic is fitted, both measured:

* **Return normalization** (``normalize_value``). Value targets here span
  ~-3 (a stalled episode) to ~+78 (a fast win paid to every agent), so the
  unnormalized ``value_loss`` reached 94-188 on the shipped fleet. Measured on
  a realistic bimodal batch, that put **95-99% of the total gradient norm in
  the value head**, and since ``max_grad_norm`` clips the *whole* gradient, the
  policy update was attenuated ~5x on exactly the iterations where something
  happened. Fitting the critic against standardized returns removes the
  attenuation entirely (clip scale 0.19 -> 1.00).
* **Split critic** (``separate_critic``). With one torso the value objective
  and the policy objective compete for the same features. Given a separate
  trunk they no longer do, and their gradients can be clipped independently —
  which is what makes the split mean something, since a shared global clip
  would re-couple them at the first value spike.

Both default OFF so every checkpoint published before v1.11 reconstructs its
exact original architecture (``load_policy`` reads the flags off the file).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

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
    #: v1.11: 0.99 -> 0.999. At 0.99 the effective planning horizon is
    #: 1/(1-gamma) = 100 steps against episodes of 300-600, so gamma^T ran to
    #: 0.0024 on platoon — the terminal reward was invisible to the optimizer
    #: while the per-step compliance rent was paid in full. Measured over all
    #: 69 runs, the three families whose DISCOUNTED stall out-earned their
    #: discounted win are exactly the three that collapsed to 0% success
    #: (8/8 correlation; see ROADMAP). ``RewardConfig.win_beats_stall`` is the
    #: invariant, and it is checked against this value.
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    #: Stop the update epochs early once the mean per-minibatch approximate KL
    #: exceeds this. Guards against the destructive updates that four times
    #: collapsed a converged policy mid-run (ROADMAP D4). None disables.
    #: NOTE: measured approx_kl across the shipped fleet is 0.0001-0.006, so
    #: this guard has never fired — the collapses were an economics failure,
    #: not a too-large-step failure. Kept as a cheap ceiling, not a fix.
    target_kl: float | None = 0.02
    #: D4 collapse stop (the passive attractor, ROADMAP 2026-08-19): end the
    #: run once rolling success has spent ``collapse_patience`` consecutive
    #: iterations at or below ``peak - collapse_margin``, where the peak is
    #: the highest full-window rolling success the run has recorded and the
    #: guard only arms once that peak reaches ``collapse_floor``. Unlike
    #: ``target_kl`` (which never fired — the collapses were an economics
    #: failure, not a step-size failure) this acts on the outcome itself:
    #: the platoon_hard cycle showed the attractor capturing 6/6 hierarchy
    #: runs from learned peaks of 75-93%, each spending its final third at
    #: 0% — pure compute spent entrenching a policy the run will not ship.
    #: ``ckpt_best`` already preserves the peak; stopping means ckpt_latest
    #: stops drifting ever further from it. 0 disables (and does so for the
    #: attractor-observation arms, where the collapse IS the experiment).
    #: Defaults calibrated by replaying every metrics.csv on disk
    #: (scripts/collapse_replay.py, 46 runs): every dip that ever recovered
    #: lasted <= 596 iterations (platoon_hard_flat seed 12, which went on to
    #: finish at 91% and publish), while every terminal capture sat below the
    #: line for 1,849-2,582 iterations to the end of its budget. Patience is
    #: set at 2x the longest observed recovery: all 7 captured runs still
    #: fire (saving 22-47% of their budgets), no recoverer does. NOTE the
    #: unit is iterations (horizon x n_envs steps each, ~1,024 at defaults) —
    #: re-run the replay before trusting these numbers at other batch shapes.
    collapse_patience: int = 1200
    collapse_margin: float = 0.5
    collapse_floor: float = 0.5
    #: D4 rescue (v1.21): roll a capture BACK instead of only ending it. The
    #: collapse stop above ends a captured run to save its budget; the rescue
    #: spends that budget on another attempt. Watching the same line
    #: (``peak - collapse_margin``, armed at ``collapse_floor``): once rolling
    #: success has sat below it for ``rescue_patience`` consecutive iterations,
    #: restore ckpt_best, rebuild the optimizer from scratch (Adam's moments
    #: carry the migration's direction — a restored policy with the old moments
    #: resumes the same walk), and multiply ``target_kl`` by
    #: ``rescue_kl_scale`` as a compounding brake. At most ``rescue_max``
    #: rescues; after that the collapse stop proceeds as before. 0 disables
    #: (the default — no existing run's semantics change).
    #: Calibration inherits the collapse replay: every recovered dip lasted
    #: <= 596 iterations, so at 700 the rescue never interrupts a policy that
    #: was coming back on its own, and it beats the stop (1200) to every
    #: capture. NOTE the four observed captures migrated at approx_kl
    #: 0.0001-0.006 — far under target_kl 0.02 — so the KL brake alone is
    #: known NOT to prevent capture; the restore is the active ingredient,
    #: and each rescue is a fresh draw against the attractor, not a cure.
    rescue_patience: int = 700
    rescue_max: int = 0
    rescue_kl_scale: float = 0.5
    hidden: int = 256
    #: fit the critic against standardized returns (see module docstring)
    normalize_value: bool = True
    #: give the critic its own torso, and clip its gradient independently
    separate_critic: bool = True
    device: str = "cpu"


#: PPOConfig fields that cannot change what a run LEARNS, only where it ends:
#: the D4 collapse guard truncates a run when the attractor captures it, and
#: up to that stop the trajectory is bit-identical to an unguarded run's.
#: They are therefore excluded from config.json — the record of the
#: experiment's trajectory-determining config, whose exact shape every
#: existing reader (and the campaign preflight's duplicate matcher) parses —
#: and the guard's one observable effect records itself in early_stop.json.
TRAJECTORY_NEUTRAL_FIELDS = ("collapse_patience", "collapse_margin", "collapse_floor")

#: The D4 rescue knobs (v1.21). Their neutrality is conditional, so they get
#: their own tuple rather than a seat in TRAJECTORY_NEUTRAL_FIELDS: with
#: ``rescue_max == 0`` the rescue can never fire and the fields are exactly as
#: neutral as the collapse stop's, but the moment it is enabled a rescue
#: rewrites the run's weights mid-flight — a different experiment, which
#: config.json (and the campaign duplicate matcher) must see. The shared rule
#: lives in ``trajectory_config``.
RESCUE_FIELDS = ("rescue_patience", "rescue_max", "rescue_kl_scale")


def trajectory_config(cfg: PPOConfig) -> dict:
    """The PPOConfig fields that determine what a run learns, as a dict.

    The single source of the config.json shape: ``train.py`` writes it and
    ``scripts/campaign_preflight.py`` predicts it, and the duplicate matcher
    compares the two byte-for-byte — so the inclusion rules must never fork.
    Always excludes TRAJECTORY_NEUTRAL_FIELDS; excludes RESCUE_FIELDS only
    while the rescue is disabled, which also keeps every pre-rescue run's
    committed config.json reproducible from its original arguments.
    """
    fields = asdict(cfg)
    for key in TRAJECTORY_NEUTRAL_FIELDS:
        del fields[key]
    if cfg.rescue_max == 0:
        for key in RESCUE_FIELDS:
            del fields[key]
    return fields


def _layer(m: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(m.weight, std)
    nn.init.constant_(m.bias, bias)
    return m


def _torso(obs_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        _layer(nn.Linear(obs_dim, hidden)),
        nn.Tanh(),
        _layer(nn.Linear(hidden, hidden)),
        nn.Tanh(),
    )


class ValueNorm(nn.Module):
    """Running mean/variance of returns, for standardizing value targets.

    The critic learns in units of standard deviations; everything outside the
    value loss (GAE, advantages, logging) works in reward units and goes
    through :meth:`denormalize`. Statistics live in buffers so they ride along
    in ``state_dict`` and a resumed run does not restart its critic's scale.

    With ``enabled=False`` this is the identity, which is how pre-v1.11
    checkpoints keep their exact original behavior.
    """

    def __init__(self, enabled: bool = True, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.enabled = enabled
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(()))
        self.register_buffer("running_var", torch.ones(()))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, returns: torch.Tensor) -> None:
        """Fold a batch of returns into the running statistics (Welford)."""
        if not self.enabled or returns.numel() == 0:
            return
        batch_mean = returns.mean()
        batch_var = returns.var(unbiased=False)
        batch_count = torch.tensor(float(returns.numel()), device=returns.device)
        delta = batch_mean - self.running_mean
        total = self.count + batch_count
        new_mean = self.running_mean + delta * batch_count / total
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.count.copy_(total)

    def _std(self) -> torch.Tensor:
        return torch.sqrt(self.running_var + self.epsilon)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reward units -> standardized units (what the critic head fits)."""
        if not self.enabled or float(self.count) == 0.0:
            return x
        return (x - self.running_mean) / self._std()

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Standardized units -> reward units (what GAE and logging want)."""
        if not self.enabled or float(self.count) == 0.0:
            return x
        return x * self._std() + self.running_mean


class PolicyNet(nn.Module):
    """Shared actor-critic MLP with action masking.

    ``separate_critic`` gives the value head its own torso; ``normalize_value``
    makes the head predict standardized returns. Both default to False so that
    ``PolicyNet(obs_dim, n_actions, hidden)`` reconstructs the pre-v1.11
    architecture exactly — the published fleet's checkpoints load unchanged.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 256,
        *,
        separate_critic: bool = False,
        normalize_value: bool = False,
    ) -> None:
        super().__init__()
        self.separate_critic = separate_critic
        self.torso = _torso(obs_dim, hidden)
        self.critic_torso = _torso(obs_dim, hidden) if separate_critic else None
        self.pi = _layer(nn.Linear(hidden, n_actions), std=0.01)
        self.v = _layer(nn.Linear(hidden, 1), std=1.0)
        self.value_norm = ValueNorm(enabled=normalize_value)

    # -- parameter groups: what the actor owns vs what the critic owns ----- #

    def actor_parameters(self) -> list[nn.Parameter]:
        """Parameters the policy loss may move."""
        return [*self.torso.parameters(), *self.pi.parameters()]

    def critic_parameters(self) -> list[nn.Parameter]:
        """Parameters the value loss may move (the shared torso, if shared)."""
        if self.critic_torso is None:
            return [*self.torso.parameters(), *self.v.parameters()]
        return [*self.critic_torso.parameters(), *self.v.parameters()]

    # -- forward ----------------------------------------------------------- #

    def dist_value_raw(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """Masked distribution and the RAW critic output (standardized units)."""
        h = self.torso(obs)
        logits = self.pi(h).masked_fill(mask == 0, -1e9)
        # a shared torso is computed once; a split one needs its own pass
        h_v = h if self.critic_torso is None else self.critic_torso(obs)
        return Categorical(logits=logits), self.v(h_v).squeeze(-1)

    def dist_value(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """Masked action distribution and state value IN REWARD UNITS."""
        dist, raw = self.dist_value_raw(obs, mask)
        return dist, self.value_norm.denormalize(raw)

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


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """1 - Var(returns - values) / Var(returns); 1.0 is a perfect critic.

    The one number that says whether the critic is doing its job. Negative
    means the value head is worse than predicting the batch mean — under which
    every advantage in the update is noise, and the run is not learning from
    what it thinks it is learning from.
    """
    var = float(np.var(returns))
    if var < 1e-9:
        return float("nan")
    return float(1.0 - np.var(returns - values) / var)


def ppo_update(
    net: PolicyNet,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    advantages: np.ndarray,
    returns: np.ndarray,
    cfg: PPOConfig,
) -> dict[str, float]:
    """One PPO update over all valid transitions; returns loss metrics.

    Diagnostics returned alongside the losses — ``grad_norm``, ``clipfrac``,
    ``explained_variance``, the value/return scales and how many epochs
    actually ran. Every collapse investigated on this project so far
    (issues #16-#19) was conducted without them.
    """
    device = torch.device(cfg.device)
    idx = buffer.valid.reshape(-1)
    flat = lambda arr, extra=():  torch.as_tensor(  # noqa: E731
        arr.reshape(-1, *extra)[idx], device=device
    )
    b_obs = flat(buffer.obs, (buffer.obs_dim,))
    b_masks = flat(buffer.masks, (buffer.n_actions,))
    b_actions = flat(buffer.actions)
    b_logprobs = flat(buffer.logprobs)
    b_values = flat(buffer.values)
    b_advantages = flat(advantages.astype(np.float32))
    b_returns = flat(returns.astype(np.float32))
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

    # Fold this batch's returns into the critic's scale BEFORE building its
    # targets, so the head always fits against the statistics it will be
    # denormalized with. (MAPPO's ValueNorm ordering.)
    net.value_norm.update(b_returns)
    b_targets = net.value_norm.normalize(b_returns)

    ev = explained_variance(buffer.values.reshape(-1)[idx], returns.reshape(-1)[idx])

    n = b_obs.shape[0]
    minibatch = max(64, n // cfg.num_minibatches)
    metrics = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
        "approx_kl": 0.0, "clipfrac": 0.0, "grad_norm": 0.0,
    }
    updates = 0
    epochs_used = 0
    kl_stop = False
    for _ in range(cfg.update_epochs):
        if kl_stop:
            break
        epochs_used += 1
        perm = torch.randperm(n, device=device)
        for start in range(0, n, minibatch):
            mb = perm[start : start + minibatch]
            if mb.shape[0] < 2:
                continue
            dist, value = net.dist_value_raw(b_obs[mb], b_masks[mb])
            logprob = dist.log_prob(b_actions[mb])
            entropy = dist.entropy().mean()
            logratio = logprob - b_logprobs[mb]
            ratio = logratio.exp()

            adv = b_advantages[mb]
            pg1 = -adv * ratio
            pg2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
            policy_loss = torch.max(pg1, pg2).mean()
            value_loss = 0.5 * ((value - b_targets[mb]) ** 2).mean()
            loss = policy_loss - cfg.ent_coef * entropy + cfg.vf_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            if net.critic_torso is None:
                grad_norm = float(nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm))
            else:
                # A split critic that shares a global gradient clip is still
                # coupled to the actor: one value spike scales BOTH down. Clip
                # the two groups independently so the split does what it says.
                g_pi = nn.utils.clip_grad_norm_(net.actor_parameters(), cfg.max_grad_norm)
                g_v = nn.utils.clip_grad_norm_(net.critic_parameters(), cfg.max_grad_norm)
                grad_norm = float(torch.sqrt(g_pi**2 + g_v**2))
            optimizer.step()

            with torch.no_grad():
                kl = ((ratio - 1) - logratio).mean().item()
                metrics["approx_kl"] += kl
                metrics["clipfrac"] += ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
            metrics["grad_norm"] += grad_norm
            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.item()
            updates += 1
            if cfg.target_kl is not None and kl > cfg.target_kl:
                kl_stop = True  # this update has moved the policy far enough
                break
    out = {k: v / max(1, updates) for k, v in metrics.items()}
    out["explained_variance"] = ev
    out["epochs_used"] = float(epochs_used)
    out["return_mean"] = float(b_returns.mean())
    out["return_std"] = float(b_returns.std())
    out["value_mean"] = float(b_values.mean())
    out["value_std"] = float(b_values.std())
    return out
