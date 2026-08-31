#!/usr/bin/env python
"""WHEN in an episode does the bunching price get charged? A converge-tax check.

    scripts/bunching_phase_probe.py                       # every baseline member
    scripts/bunching_phase_probe.py --unit-price -0.05 --episodes 20

**The claim this exists to refute.** ``bunching_price_calibration.py`` reports what
a member WOULD pay over a whole episode, and that per-episode total is what sized
rung 1 at -0.05. Two members then collapsed under it in a way a total cannot
explain: ``squad_v30`` and ``platoon_v15_seed12`` did not merely lose reward, they
INVERTED — their pre-price incumbents converge to fast episodes (length 214->100,
return 39.9->70.0) and the priced runs converge to the opposite (length ->max,
clock-out 1.000, return ->0), with element bunching 0.291->0.013. A policy that
disperses perfectly and never finishes is not paying a dispersion tax; it is
avoiding something that arrives with success.

So the hypothesis is that the price is not flat across an episode: it is charged
hardest in the FINAL APPROACH, because finishing requires the element to close on
one point. Where the terminal needs convergence, "cheap" and "successful" are in
direct conflict and stalling at distance strictly dominates. Where success is
achieved in place (hold, screen, observe), the same total can be restructured
without giving up the terminal.

**What would refute it.** A charge that is uniform across the episode, or heaviest
early (the spawn pile), for the two collapsed scenarios. Then the collapse is
about magnitude and the fix is a smaller rung — a completely different repair from
the one this predicts.

Read-only, and exact for the same reason the calibration is: a policy acts on
observations, so arming the price moves no trajectory of a trained checkpoint.
Quintiles are of each episode's REALISED length, so Q5 of a successful episode is
by construction the approach that scored it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT))

from cohort.env.cohort_env import make_env  # noqa: E402
from cohort.training.evaluate import _pick_actions  # noqa: E402
from scripts.fleet_status import find_run  # noqa: E402

QUINTILES = 5


def probe(checkpoint: str, unit_price: float, episodes: int, seed: int) -> dict:
    """Roll the policy, keeping the bunching charge per step rather than summing it.

    Same rollout as the calibration probe — same action selection, same double
    seeding — so the two are comparable line by line. The only difference is that
    the ledger is kept as a time series and collapsed at the end.
    """
    from cohort.training.train import load_policy

    net, ckpt = load_policy(checkpoint)
    scenario = ckpt["scenario"]
    # charge[q] and steps[q] accumulate over episodes of DIFFERENT lengths, which
    # is why each episode is bucketed by its own realised length before adding in.
    charge = np.zeros(QUINTILES)
    agent_steps = np.zeros(QUINTILES)
    won_charge = np.zeros(QUINTILES)
    won_steps = np.zeros(QUINTILES)
    wins = 0
    lengths: list[int] = []
    for i in range(episodes):
        env = make_env(scenario)
        torch.manual_seed(seed + i)
        obs, _ = env.reset(seed=seed + i)
        env.rewards_cfg = replace(env.rewards_cfg, bunching_penalty=unit_price)
        rng = np.random.default_rng(seed + i)
        per_step: list[tuple[float, int]] = []
        while env.agents:
            actions = _pick_actions(env, obs, net, rng, greedy=False)
            obs, _rewards, _terms, _truncs, infos = env.step(actions)
            step_charge = 0.0
            living = 0
            for info in infos.values():
                step_charge += (info.get("components") or {}).get("bunching", 0.0)
                living += 1
            per_step.append((step_charge, living))
        n = len(per_step)
        lengths.append(n)
        won = env.outcome == "success"
        wins += won
        for t, (c, living) in enumerate(per_step):
            # floor, clipped: the last step of the episode belongs to Q5, not to a
            # sixth bucket that would exist only for it.
            q = min(int(t * QUINTILES / n), QUINTILES - 1) if n else 0
            charge[q] += c
            agent_steps[q] += living
            if won:
                won_charge[q] += c
                won_steps[q] += living
    per_agent_step = np.divide(charge, agent_steps,
                               out=np.zeros(QUINTILES), where=agent_steps > 0)
    won_per_agent_step = np.divide(won_charge, won_steps,
                                   out=np.zeros(QUINTILES), where=won_steps > 0)
    return {
        "scenario": scenario,
        "success": wins / episodes,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "quintiles": per_agent_step.tolist(),
        "won_quintiles": won_per_agent_step.tolist(),
    }


def _ratio(q: list[float]) -> float:
    """Q5 over Q1. The converge-tax ratio: >1 means the price arrives with success."""
    return q[-1] / q[0] if q[0] else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit-price", type=float, default=-0.05)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--member", action="append", default=[])
    args = ap.parse_args()

    members = json.loads((RUNS / "BASELINE.json").read_text())["runs"]
    if args.member:
        members = {k: v for k, v in members.items() if k in args.member}

    print(f"bunching charge per agent-step by episode quintile, "
          f"unit price {args.unit_price}, N={args.episodes}")
    print("Q1..Q5 are fifths of each episode's realised length; "
          "Q5 of a win IS the approach that scored it.\n")
    head = " ".join(f"{'Q' + str(i + 1):>7s}" for i in range(QUINTILES))
    print(f"  {'scenario':17s} {'succ':>5s} {'len':>6s} {head} {'Q5/Q1':>7s}")
    for scenario, run in sorted(members.items()):
        d = find_run(run, RUNS)
        ckpt = d / "ckpt_latest.pt" if d else None
        if ckpt is None or not ckpt.exists():
            print(f"  {scenario:17s} (no ckpt_latest.pt)")
            continue
        r = probe(str(ckpt), args.unit_price, args.episodes, args.seed)
        q = r["won_quintiles"] if r["success"] else r["quintiles"]
        cells = " ".join(f"{v:7.4f}" for v in q)
        print(f"  {scenario:17s} {r['success']:5.2f} {r['mean_length']:6.0f} "
              f"{cells} {_ratio(q):7.2f}")
    print("\n  (rows with no win are scored over ALL episodes; every other row is "
          "scored over its wins only)")


if __name__ == "__main__":
    main()
