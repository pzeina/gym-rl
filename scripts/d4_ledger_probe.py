#!/usr/bin/env python
"""What does a policy actually earn, per reward component, per agent-step?

The repo's rule is diagnose-first: before proposing a reward change, show the
ledger. This rolls a checkpoint out and accumulates
``infos[agent]["components"]``, then prices the counterfactual — because the D4
attractor is an economic fact, not a mood. A captured policy runs every episode
to the clock and lives on the compliance+command trickle; if that trickle
exceeds the time price, idling is net-positive income and the stall is stable.

    scripts/d4_ledger_probe.py runs/<run>/ckpt_latest.pt [--time-penalty -0.03]

The counterfactual line is only meaningful for a CAPTURED policy: for a working
one the total is dominated by terminal reward and "idling" is not what is being
measured. It is printed for both so the two can be compared side by side.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cohort.env.cohort_env import make_env
from cohort.training.evaluate import _pick_actions
from cohort.training.train import load_policy

DEFAULT_TIME_PENALTY = -0.01


def component_ledger(
    checkpoint: str, scenario: str | None = None, episodes: int = 12, seed: int = 10_000
) -> tuple[dict[str, float], int, float]:
    """Per-agent-step mean of every reward component over ``episodes`` rollouts."""
    net, ckpt = load_policy(checkpoint)
    env = make_env(scenario or ckpt["scenario"])
    totals: dict[str, float] = {}
    agent_steps = 0
    lengths: list[int] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        rng = np.random.default_rng(777 + ep)
        steps = 0
        while env.agents:
            obs, _rew, _term, _trunc, infos = env.step(
                _pick_actions(env, obs, net, rng, greedy=False)
            )
            for info in infos.values():
                for name, value in (info.get("components") or {}).items():
                    totals[name] = totals.get(name, 0.0) + float(value)
                agent_steps += 1
            steps += 1
        lengths.append(steps)

    per_step = {k: v / max(1, agent_steps) for k, v in totals.items()}
    return per_step, agent_steps, float(np.mean(lengths))


def price_at(per_step: dict[str, float], time_penalty: float) -> tuple[float, float, float]:
    """Re-price a ledger at a different time penalty.

    Returns ``(total_now, non_time_income, total_at_new_price)``. The time
    component scales linearly with the price, every other component is held —
    which is the assumption the counterfactual rests on and the reason this run
    is an experiment rather than a conclusion.
    """
    total = sum(per_step.values())
    time = per_step.get("time", 0.0)
    non_time = total - time
    return total, non_time, non_time + (time_penalty / DEFAULT_TIME_PENALTY) * time


def report(checkpoint: str, episodes: int, time_penalty: float) -> None:
    per_step, agent_steps, mean_len = component_ledger(checkpoint, episodes=episodes)
    total, non_time, counterfactual = price_at(per_step, time_penalty)
    time = per_step.get("time", 0.0)

    print(f"\n== {Path(checkpoint).parent.name} ==  "
          f"agent-steps {agent_steps}  mean ep len {mean_len:.0f}")
    for name in sorted(per_step, key=lambda k: -abs(per_step[k])):
        print(f"    {name:<14}{per_step[name]:+.5f}")
    print(f"    {'-' * 26}")
    print(f"    {'TOTAL/step':<14}{total:+.5f}   "
          f"(non-time {non_time:+.5f}, time {time:+.5f})")
    print(f"    at time={time_penalty:<7}{counterfactual:+.5f}   "
          f"-> {'STILL PAYS' if counterfactual > 0 else 'BLEEDS'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--time-penalty", type=float, default=-0.03,
                        help="counterfactual time price to re-score at")
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        report(checkpoint, args.episodes, args.time_penalty)


if __name__ == "__main__":
    main()
