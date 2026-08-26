#!/usr/bin/env python
"""Is AREA FIRE priced too cheaply, or is it simply not firing?

    scripts/burst_engagement_probe.py runs/defend_brique_v19/ckpt_latest.pt --episodes 20
    scripts/burst_engagement_probe.py <ckpt> --fraction 0.5 0.75 1.0

**Why this exists.** Dropping the two DEFEND incumbents into the armed world
without retraining barely moved them: ``defend_brique_v19`` lost 2 episodes of
100 and ``fireteam_defend_v25`` lost none, with nearest-teammate distance
unchanged at 0.21/0.23. Before reading that as "rung 1 is too cheap, climb the
ladder", it has to be separated from "the pile is never sprayed at all" — and
those two have opposite fixes. A higher ``burst_fraction`` multiplies a splash
that fires; it cannot multiply one that does not.

So this counts the mechanism rather than its consequences: how often an enemy
round lands on a soldier, how often that round sprays anyone, how much splash
damage the element eats per episode, and how many of its deaths the splash
actually caused. Deterministic reads over ordinary rollouts — the probe wraps
two methods and consumes no RNG of its own, so the episodes are the same
episodes.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cohort.env.cohort_env import make_env  # noqa: E402
from cohort.training.evaluate import run_episode  # noqa: E402


class Counter:
    """Wraps the two damage paths and tallies what the mechanic did."""

    def __init__(self, env):
        self.env = env
        self.hits_on_soldiers = 0      # enemy rounds that landed
        self.bursts = 0                # landed rounds that sprayed at least one neighbour
        self.sprayed = 0               # neighbour-hits delivered
        self.splash_damage = 0         # total health removed by splash
        self.splash_deaths = 0         # soldiers whose killing blow was splash
        self._in_burst = False
        self._real_burst = env._burst_on_soldiers
        self._real_damage = env._damage_soldier
        env._burst_on_soldiers = self._burst
        env._damage_soldier = self._damage

    def _burst(self, struck, damage, ledger, deaths):
        self.hits_on_soldiers += 1
        before = self.sprayed
        self._in_burst = True
        try:
            self._real_burst(struck, damage, ledger, deaths)
        finally:
            self._in_burst = False
        if self.sprayed > before:
            self.bursts += 1

    def _damage(self, soldier, damage, ledger, deaths):
        if self._in_burst:
            self.sprayed += 1
            self.splash_damage += min(damage, max(soldier.health, 0))
            alive_before = soldier.alive
        self._real_damage(soldier, damage, ledger, deaths)
        if self._in_burst and alive_before and not soldier.alive:
            self.splash_deaths += 1


def probe(checkpoint: str, fraction: float, episodes: int, seed: int) -> dict:
    from cohort.training.train import load_policy

    net, ckpt = load_policy(checkpoint)
    scenario = ckpt["scenario"]
    totals = {k: 0 for k in ("hits_on_soldiers", "bursts", "sprayed",
                             "splash_damage", "splash_deaths")}
    wins = deaths = 0
    for i in range(episodes):
        env = make_env(scenario)
        env.reset(seed=seed + i)
        env.combat = replace(env.combat, burst_fraction=fraction)
        counter = Counter(env)
        out = run_episode(env, net, seed=seed + i, rng=np.random.default_rng(seed + i))
        wins += out["outcome"] == "success"
        deaths += len(env.roster.soldiers) - out["survivors"]
        for k in totals:
            totals[k] += getattr(counter, k)
    n = episodes
    return {
        "scenario": scenario, "fraction": fraction, "episodes": n,
        "success": wins / n, "deaths_per_ep": deaths / n,
        "enemy_hits_per_ep": totals["hits_on_soldiers"] / n,
        "bursts_per_ep": totals["bursts"] / n,
        "burst_share_of_hits": (totals["bursts"] / totals["hits_on_soldiers"]
                                if totals["hits_on_soldiers"] else float("nan")),
        "neighbours_sprayed_per_ep": totals["sprayed"] / n,
        "splash_damage_per_ep": totals["splash_damage"] / n,
        "splash_deaths_per_ep": totals["splash_deaths"] / n,
        "share_of_deaths_from_splash": (totals["splash_deaths"] / deaths
                                        if deaths else float("nan")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--fraction", type=float, nargs="+", default=[0.5])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rows = [probe(args.checkpoint, f, args.episodes, args.seed) for f in args.fraction]
    print(f"{rows[0]['scenario']}  N={args.episodes}  ({Path(args.checkpoint).parent.name})")
    head = ("frac", "succ", "deaths", "enemy hits", "bursts", "burst%",
            "sprayed", "splash dmg", "splash kills", "% deaths")
    print("  " + " ".join(f"{h:>12s}" for h in head))
    for r in rows:
        print("  " + " ".join(f"{v:>12}" for v in (
            f"{r['fraction']:.2f}", f"{r['success']:.2f}", f"{r['deaths_per_ep']:.2f}",
            f"{r['enemy_hits_per_ep']:.1f}", f"{r['bursts_per_ep']:.1f}",
            f"{r['burst_share_of_hits']:.2f}", f"{r['neighbours_sprayed_per_ep']:.1f}",
            f"{r['splash_damage_per_ep']:.0f}", f"{r['splash_deaths_per_ep']:.2f}",
            f"{r['share_of_deaths_from_splash']:.2f}")))


if __name__ == "__main__":
    main()
