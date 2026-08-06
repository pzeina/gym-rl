"""Evaluate a trained policy: metrics, GIF replay, radio transcript.

Usage:
    python -m cohort.training.evaluate runs/<run>/ckpt_best.pt --episodes 20
    python -m cohort.training.evaluate --random --scenario fireteam   # baseline

By default every evaluation also computes the behavioral metrics suite
(``cohort.metrics``, ROADMAP B2) over the same episodes — printed as a table
and written to ``behavior.json`` next to the checkpoint. ``--no-behavior``
skips it. Any regression gates that apply to the run's root mission
(``cohort.metrics.regression_gates`` — currently the positional gate on
DEFEND roots, issue #11) are printed under the table and stored alongside it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

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
    recorder=None,
) -> dict:
    """Roll one episode; returns summary stats. Optionally collects frames.

    ``recorder`` (a ``cohort.metrics.TraceRecorder``) hooks the behavioral
    trace: reads are deterministic and consume no RNG, so the episode is
    bit-identical with or without it.
    """
    rng = rng or np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    steps = 0
    if recorder is not None:
        recorder.on_reset(env)
    if frames is not None:
        frames.append(env.render())
    while env.agents:
        actions = _pick_actions(env, obs, net, rng, greedy=greedy)
        if recorder is not None:
            recorder.before_step(env)
        obs, rewards, _terms, _truncs, _infos = env.step(actions)
        if recorder is not None:
            recorder.after_step(env)
        total += sum(rewards.values())
        steps += 1
        if frames is not None and (steps % frame_every == 0 or not env.agents):
            frames.append(env.render())
    return {
        "outcome": env.outcome or "timeout",
        "return": total / len(env.possible_agents),
        "length": steps,
        "survivors": sum(s.alive for s in env.roster.soldiers),
        "orders": sum(m.kind.value == "order" for m in env.transcript.messages),
        "reports": sum(m.kind.value in ("contact", "sitrep", "done") for m in env.transcript.messages),
        "done_reports": sum(m.kind.value == "done" for m in env.transcript.messages),
    }


def _seeded_episode(env: CohortEnv, net, ep_seed: int, **kw) -> dict:
    """One episode with self-contained seeding (numpy + torch from ``ep_seed``).

    Episode k of an evaluation reproduces standalone: its sampling streams do
    not depend on how many random draws earlier episodes consumed.
    """
    torch.manual_seed(ep_seed)
    return run_episode(env, net, seed=ep_seed, rng=np.random.default_rng(ep_seed), **kw)


def evaluate(
    checkpoint: str | None,
    scenario: str | None = None,
    episodes: int = 100,
    seed: int = 123,
    gif_path: str | None = None,
    transcript_path: str | None = None,
    *,
    greedy: bool = False,
    behavior: bool = True,
    behavior_path: str | None = None,
) -> dict:
    """Run evaluation episodes; optionally save a GIF + transcript of the last one.

    With ``behavior=True`` (default) the behavioral metrics suite (B2) is
    computed over the very same episodes, printed as a table, returned under
    ``summary["behavior"]``, and written as JSON to ``behavior_path``
    (default: ``behavior.json`` next to the checkpoint; the random baseline
    writes nothing unless a path is given).
    """
    net = None
    if checkpoint is not None:
        from cohort.training.train import load_policy

        net, ckpt = load_policy(checkpoint)
        scenario = scenario or ckpt["scenario"]
    if scenario is None:
        msg = "Need --scenario when evaluating the random baseline."
        raise ValueError(msg)

    if behavior:
        from cohort.metrics import TraceRecorder, episode_behavior

        recorders = [TraceRecorder() for _ in range(episodes)]
    else:
        recorders = [None] * episodes

    env = make_env(scenario)
    results = [
        _seeded_episode(env, net, seed + i, greedy=greedy, recorder=recorders[i])
        for i in range(episodes)
    ]

    outcomes = Counter(r["outcome"] for r in results)
    p = outcomes.get("success", 0) / episodes
    ci95 = 1.96 * (p * (1 - p) / episodes) ** 0.5
    summary = {
        "episodes": episodes,
        "success_rate": p,
        "success_ci95": f"{p:.2f} ± {ci95:.2f}",
        "outcomes": dict(outcomes),
        "mean_return": float(np.mean([r["return"] for r in results])),
        "mean_length": float(np.mean([r["length"] for r in results])),
        "mean_survivors": float(np.mean([r["survivors"] for r in results])),
        "mean_orders": float(np.mean([r["orders"] for r in results])),
        "mean_reports": float(np.mean([r["reports"] for r in results])),
        "mean_done_reports": float(np.mean([r["done_reports"] for r in results])),
    }
    label = "random baseline" if net is None else checkpoint
    print(f"eval [{label}] on {scenario}:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if behavior:
        from cohort.metrics import (
            aggregate_behavior,
            format_behavior_table,
            format_gate_report,
            regression_gates,
        )

        per_episode = [episode_behavior(r.trace) for r in recorders]
        agg = aggregate_behavior(per_episode)
        gates = regression_gates(agg)
        summary["behavior"] = agg
        summary["gates"] = gates
        print(format_behavior_table(agg))
        if gates:
            print(format_gate_report(gates))
        out = behavior_path
        if out is None and checkpoint is not None:
            out = str(Path(checkpoint).parent / "behavior.json")
        if out is not None:
            payload = {
                "checkpoint": checkpoint,
                "scenario": scenario,
                "episodes": episodes,
                "seed": seed,
                "greedy": greedy,
                "success_ci95": summary["success_ci95"],
                "metrics": agg,
                "gates": gates,
                "per_episode": per_episode,
            }
            Path(out).write_text(json.dumps(payload, indent=1) + "\n")
            print(f"behavior → {out}")

    if gif_path or transcript_path:
        env_r = make_env(scenario, render_mode="rgb_array")
        frames: list = []
        # replay a few seeds, keep the first success (or the last attempt)
        for i in range(5):
            frames.clear()
            stats = _seeded_episode(env_r, net, seed + 1000 + i, frames=frames, greedy=greedy)
            if stats["outcome"] == "success":
                break
        if gif_path:
            from cohort.viz.render import save_gif

            save_gif(frames, gif_path)
            print(f"gif → {gif_path} ({stats['outcome']})")
        if transcript_path:
            Path(transcript_path).write_text(env_r.transcript.render() + "\n")
            print(f"transcript → {transcript_path}")
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate a cohort policy.")
    parser.add_argument("checkpoint", nargs="?", default=None)
    parser.add_argument("--random", action="store_true", help="masked-random baseline instead of a checkpoint")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gif", default=None, help="path to save an episode GIF")
    parser.add_argument("--transcript", default=None, help="path to save the episode radio transcript")
    parser.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    parser.add_argument(
        "--behavior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compute the behavioral metrics suite (B2); writes behavior.json next to the checkpoint",
    )
    parser.add_argument("--behavior-out", default=None, help="override the behavior.json output path")
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
        behavior=args.behavior,
        behavior_path=args.behavior_out,
    )


if __name__ == "__main__":
    main()
