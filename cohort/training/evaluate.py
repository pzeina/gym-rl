"""Evaluate a trained policy: metrics, GIF replay, radio transcript.

Usage:
    python -m cohort.training.evaluate runs/<run>/ckpt_best.pt --episodes 20
    python -m cohort.training.evaluate --random --scenario fireteam   # baseline
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch

from cohort.env.cohort_env import CohortEnv, make_env


def _pick_actions(
    env: CohortEnv, obs: dict, net, rng: np.random.Generator, *, greedy: bool = False
) -> dict[str, int]:
    """Policy actions (sampled by default), or uniform-over-legal if net is None."""
    agents = list(env.agents)
    if not agents:
        return {}
    if net is None:
        actions = {}
        for a in agents:
            legal = np.flatnonzero(obs[a]["action_mask"])
            actions[a] = int(rng.choice(legal))
        return actions
    t_obs = torch.as_tensor(np.stack([obs[a]["observation"] for a in agents]))
    t_mask = torch.as_tensor(np.stack([obs[a]["action_mask"] for a in agents]))
    act, _, _ = net.act(t_obs, t_mask, greedy=greedy)
    return {a: int(act[i]) for i, a in enumerate(agents)}


def run_episode(
    env: CohortEnv,
    net,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    frames: list | None = None,
    frame_every: int = 2,
    *,
    greedy: bool = False,
) -> dict:
    """Roll one episode; returns summary stats. Optionally collects frames."""
    rng = rng or np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    steps = 0
    if frames is not None:
        frames.append(env.render())
    while env.agents:
        actions = _pick_actions(env, obs, net, rng, greedy=greedy)
        obs, rewards, _terms, _truncs, _infos = env.step(actions)
        total += sum(rewards.values())
        steps += 1
        if frames is not None and (steps % frame_every == 0 or not env.agents):
            frames.append(env.render())
    return {
        "outcome": env._episode_outcome or "timeout",
        "return": total / len(env.possible_agents),
        "length": steps,
        "survivors": sum(s.alive for s in env.roster.soldiers),
        "orders": sum(m.kind.value == "order" for m in env.transcript.messages),
        "reports": sum(m.kind.value in ("contact", "sitrep", "done") for m in env.transcript.messages),
    }


def evaluate(
    checkpoint: str | None,
    scenario: str | None = None,
    episodes: int = 20,
    seed: int = 123,
    gif_path: str | None = None,
    transcript_path: str | None = None,
    *,
    greedy: bool = False,
) -> dict:
    """Run evaluation episodes; optionally save a GIF + transcript of the last one."""
    net = None
    if checkpoint is not None:
        from cohort.training.train import load_policy

        net, ckpt = load_policy(checkpoint)
        scenario = scenario or ckpt["scenario"]
    if scenario is None:
        msg = "Need --scenario when evaluating the random baseline."
        raise ValueError(msg)

    env = make_env(scenario)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)  # reproducible sampled-action evaluations
    results = [run_episode(env, net, seed=seed + i, rng=rng, greedy=greedy) for i in range(episodes)]

    outcomes = Counter(r["outcome"] for r in results)
    summary = {
        "episodes": episodes,
        "success_rate": outcomes.get("success", 0) / episodes,
        "outcomes": dict(outcomes),
        "mean_return": float(np.mean([r["return"] for r in results])),
        "mean_length": float(np.mean([r["length"] for r in results])),
        "mean_survivors": float(np.mean([r["survivors"] for r in results])),
        "mean_orders": float(np.mean([r["orders"] for r in results])),
        "mean_reports": float(np.mean([r["reports"] for r in results])),
    }
    label = "random baseline" if net is None else checkpoint
    print(f"eval [{label}] on {scenario}:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if gif_path or transcript_path:
        env_r = make_env(scenario, render_mode="rgb_array")
        frames: list = []
        # replay a few seeds, keep the first success (or the last attempt)
        for i in range(5):
            frames.clear()
            stats = run_episode(env_r, net, seed=seed + 1000 + i, rng=rng, frames=frames, greedy=greedy)
            if stats["outcome"] == "success":
                break
        if gif_path:
            from cohort.viz.render import save_gif

            save_gif(frames, gif_path)
            print(f"gif → {gif_path} ({stats['outcome']})")
        if transcript_path:
            from pathlib import Path

            Path(transcript_path).write_text(env_r.transcript.render() + "\n")
            print(f"transcript → {transcript_path}")
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate a cohort policy.")
    parser.add_argument("checkpoint", nargs="?", default=None)
    parser.add_argument("--random", action="store_true", help="masked-random baseline instead of a checkpoint")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gif", default=None, help="path to save an episode GIF")
    parser.add_argument("--transcript", default=None, help="path to save the episode radio transcript")
    parser.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    args = parser.parse_args()
    checkpoint = None if args.random else args.checkpoint
    if checkpoint is None and not args.random:
        parser.error("Provide a checkpoint path or --random.")
    evaluate(
        checkpoint,
        scenario=args.scenario,
        episodes=args.episodes,
        seed=args.seed,
        gif_path=args.gif,
        transcript_path=args.transcript,
        greedy=args.greedy,
    )


if __name__ == "__main__":
    main()
