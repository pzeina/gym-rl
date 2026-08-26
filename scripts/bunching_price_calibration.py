#!/usr/bin/env python
"""What would the bunching price charge today's policies, before anyone retrains?

    scripts/bunching_price_calibration.py                 # every baseline member
    scripts/bunching_price_calibration.py --episodes 20 --unit-price -0.01

**Why this can be measured exactly.** A policy acts on observations, not on
rewards, so arming ``bunching_penalty`` does not change a single trajectory of an
already-trained checkpoint. The episodes here are bit-identical to the ones the
committed evaluations scored; only the ledger differs. So the charge each member
WOULD have paid is not an estimate — it is arithmetic over the rollouts that
already exist, at zero training cost.

**What it is for.** Picking the three rungs of the declared ladder from data
rather than from taste. A price is only meaningful against what else the episode
pays, so the read-out puts the bunching charge beside the time cost the same
episode accrued and beside the terminal reward it is trying to outweigh. The
failure this prevents is the one AREA FIRE walked into: launching 18 jobs at a
price nobody had checked could reach the members.

The charge is reported per agent per episode, because that is the unit a policy
optimises: an agent decides where to stand, and what it feels is its own ledger.
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


def calibrate(checkpoint: str, unit_price: float, episodes: int, seed: int) -> dict:
    """Roll the policy and accumulate the per-step reward breakdown.

    ``run_episode`` drops ``infos``, and the component split only exists there,
    so the loop is written out. It is otherwise the same rollout: same action
    selection, same seeding, same RNG stream.
    """
    from cohort.training.train import load_policy

    net, ckpt = load_policy(checkpoint)
    scenario = ckpt["scenario"]
    bunching = time_cost = 0.0
    agents = wins = 0
    for i in range(episodes):
        env = make_env(scenario)
        # Seed BOTH streams per episode, exactly as evaluate._seeded_episode does.
        # Action selection samples through torch; leaving its global RNG alone made
        # two invocations of this probe disagree, and the whole point here is that
        # arming the price cannot move a trajectory — an unreproducible rollout
        # cannot demonstrate that.
        torch.manual_seed(seed + i)
        obs, _ = env.reset(seed=seed + i)
        env.rewards_cfg = replace(env.rewards_cfg, bunching_penalty=unit_price)
        rng = np.random.default_rng(seed + i)
        while env.agents:
            actions = _pick_actions(env, obs, net, rng, greedy=False)
            obs, _rewards, _terms, _truncs, infos = env.step(actions)
            for info in infos.values():
                comp = info.get("components") or {}
                bunching += comp.get("bunching", 0.0)
                time_cost += comp.get("time", 0.0)
        wins += env.outcome == "success"
        agents += len(env.possible_agents)
    return {
        "scenario": scenario, "success": wins / episodes,
        "bunching_per_agent_ep": bunching / agents if agents else 0.0,
        "time_per_agent_ep": time_cost / agents if agents else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit-price", type=float, default=-0.01)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--member", action="append", default=[])
    args = ap.parse_args()

    members = json.loads((RUNS / "BASELINE.json").read_text())["runs"]
    if args.member:
        members = {k: v for k, v in members.items() if k in args.member}

    print(f"bunching price at {args.unit_price} per excess teammate per agent-step, "
          f"N={args.episodes}\n")
    print(f"  {'scenario':17s} {'succ':>5s} {'bunching/agent/ep':>18s} "
          f"{'time/agent/ep':>14s} {'ratio':>7s}")
    for scenario, run in sorted(members.items()):
        # via the resolver, never RUNS / run: a superseded member lives in
        # runs/archive/ and the evidence behind a published claim must stay
        # resolvable after the cycle files it away.
        d = find_run(run, RUNS)
        ckpt = d / "ckpt_latest.pt" if d else None
        if ckpt is None or not ckpt.exists():
            print(f"  {scenario:17s} (no ckpt_latest.pt)")
            continue
        r = calibrate(str(ckpt), args.unit_price, args.episodes, args.seed)
        ratio = (r["bunching_per_agent_ep"] / r["time_per_agent_ep"]
                 if r["time_per_agent_ep"] else float("nan"))
        print(f"  {scenario:17s} {r['success']:5.2f} {r['bunching_per_agent_ep']:18.2f} "
              f"{r['time_per_agent_ep']:14.2f} {ratio:7.2f}")


if __name__ == "__main__":
    main()
