#!/usr/bin/env python
"""CONTACT report economics — how redundant is a redundant report?

fireteam_defend_v8 reports at precision 0.38 (N=100): 289 informative contacts
against 480 that carry nothing. That is not a training failure, it is the
priced optimum. A CONTACT is adjudicated in three tiers by
``CohortEnv._report_contact``:

    new intel     +contact_new       (+0.50)  some visible enemy is off the picture
    refresh        0.00                       some entry has aged >= refresh_age
    redundant     +contact_redundant (-0.02)  every entry is fresh

with ``transmission_cost`` (-0.01) charged on top of all three. So spamming
pays whenever ``p x 0.49 > (1-p) x 0.03``, i.e. down to a precision of **5.8%**.
At 0.38 the policy is nowhere near the pain threshold; the price has no teeth.

Before repricing, this measures the shape of the waste, because two very
different defects produce the same precision number:

* **duplicate storms** — redundant reports bunched at age ~0, the same enemy
  re-reported every few ticks. Pure noise; a flat penalty is the right answer.
* **near-miss refreshes** — redundant reports bunched just under
  ``contact_refresh_age``, where the intel genuinely was going stale and the
  tier cliff denied credit for it. A flat penalty would punish near-useful
  traffic, and the right answer is to make value decay with age instead of
  stepping at a threshold.

The histogram below separates them. "age" is the **oldest** picture entry among
the enemies the sender could see, since that is what the env's ``refreshes``
test looks at: a report is redundant exactly when that maximum is still under
``contact_refresh_age``.

Read-only; consumes no randomness and never feeds observations or rewards.

    scripts/contact_probe.py runs/<run>/ckpt_best.pt --episodes 30 --seed 500
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cohort.env.actions import CATALOG
from cohort.env.cohort_env import make_env
from cohort.training.evaluate import _pick_actions
from cohort.training.train import load_policy

CONTACT_IX = next(spec.index for spec in CATALOG if spec.kind == "contact")


def classify(env, soldier) -> tuple[str, int | None]:
    """(tier, oldest picture age) for a CONTACT this soldier is about to send.

    Mirrors ``CohortEnv._report_contact``. Returns ("none", None) when the
    soldier can see nothing, which the env treats as a no-op rather than a
    transmission.
    """
    visible = env._visible_enemies(soldier)
    if not visible:
        return "none", None
    if any(e.id not in env._known_enemies for e in visible):
        return "new", None
    ages = [env._step_count - env._known_enemies[e.id][2] for e in visible]
    oldest = max(ages)
    if oldest >= env.rewards_cfg.contact_refresh_age:
        return "refresh", oldest
    return "redundant", oldest


def probe(checkpoint, scenario, episodes, first_seed, greedy):
    net = None
    if checkpoint is not None:
        net, ckpt = load_policy(checkpoint)
        scenario = scenario or ckpt.get("scenario")
    if scenario is None:
        raise SystemExit("need --scenario when probing without a checkpoint")

    env = make_env(scenario)
    tiers: Counter[str] = Counter()
    ages: Counter[int] = Counter()
    per_ep_reports = []

    for k in range(episodes):
        ep_seed = first_seed + k
        torch.manual_seed(ep_seed)
        rng = np.random.default_rng(ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        by_cs = {s.callsign: s for s in env.roster.soldiers}
        ep_reports = 0

        while env.agents:
            actions = _pick_actions(env, obs, net, rng, greedy=greedy)
            for cs, act in actions.items():
                if act != CONTACT_IX:
                    continue
                soldier = by_cs.get(cs)
                if soldier is None or not soldier.alive:
                    continue
                tier, oldest = classify(env, soldier)
                tiers[tier] += 1
                if tier != "none":
                    ep_reports += 1
                if tier == "redundant":
                    ages[oldest] += 1
            obs, _, _, _, _ = env.step(actions)
        per_ep_reports.append(ep_reports)

    return env, tiers, ages, per_ep_reports


def report(env, tiers, ages, per_ep, episodes) -> None:
    cfg = env.rewards_cfg
    sent = tiers["new"] + tiers["refresh"] + tiers["redundant"]
    new_value = cfg.contact_new + cfg.transmission_cost
    redundant_cost = -(cfg.contact_redundant + cfg.transmission_cost)
    breakeven = redundant_cost / (new_value + redundant_cost) if new_value > 0 else float("nan")

    print(f"contact probe: {env.spec_cfg.name}  episodes={episodes}")
    print()
    print("PRICES")
    print(f"  contact_new                   {cfg.contact_new:+.3f}")
    print(f"  contact_redundant             {cfg.contact_redundant:+.3f}")
    print(f"  transmission_cost             {cfg.transmission_cost:+.3f}")
    print(f"  contact_refresh_age           {cfg.contact_refresh_age}")
    print(f"  informative report is worth   {new_value:+.3f}")
    print(f"  redundant report costs        {-redundant_cost:+.3f}")
    print(f"  => spamming pays down to precision {breakeven:.3f}")
    print()
    print("TIERS (reports actually transmitted)")
    for tier in ("new", "refresh", "redundant"):
        share = tiers[tier] / sent if sent else 0.0
        print(f"  {tier:<12} {tiers[tier]:>7}   {share:>6.3f}")
    print(f"  {'TOTAL':<12} {sent:>7}")
    informative = tiers["new"] + tiers["refresh"]
    print(f"  precision (new+refresh)     {informative / sent:.3f}" if sent else "  precision —")
    print(f"  reports / episode           {sum(per_ep) / episodes:.1f}")
    print(f"  suppressed (saw nothing)    {tiers['none']}")
    print()
    print(f"REDUNDANT REPORT AGE (0 = re-sent the same tick's intel; cliff at {cfg.contact_refresh_age})")
    total_r = sum(ages.values())
    if not total_r:
        print("  none")
        return
    width = max(1, cfg.contact_refresh_age)
    buckets = 10
    per = max(1, width // buckets)
    binned: Counter[str] = Counter()
    for age, n in ages.items():
        lo = (age // per) * per
        binned[f"{lo:>3}-{lo + per - 1:<3}"] += n
    for label in sorted(binned, key=lambda s: int(s.split("-")[0])):
        n = binned[label]
        bar = "#" * max(1, round(40 * n / total_r))
        print(f"  {label} {n:>6} {n / total_r:>6.3f} {bar}")
    mean_age = sum(a * n for a, n in ages.items()) / total_r
    half = cfg.contact_refresh_age / 2
    young = sum(n for a, n in ages.items() if a < half)
    print(f"  mean age {mean_age:.1f}   share below half the cliff: {young / total_r:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--scenario")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    env, tiers, ages, per_ep = probe(
        args.checkpoint, args.scenario, args.episodes, args.seed, args.greedy
    )
    report(env, tiers, ages, per_ep, args.episodes)


if __name__ == "__main__":
    main()
