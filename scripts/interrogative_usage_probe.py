#!/usr/bin/env python
"""REQUEST_STATUS usage probe — is the interrogative exchange used at all?

Check 1 of the pre-registered interrogative read (docs/interrogative-cycle.md
§"The read") asks for REQUEST_STATUS >= 1 per WON episode on the reporting
draws; ``scripts/interrogative_read.py`` consumes that number as
``--usage RUN=N``. Nothing committed carries it — ``evaluate()`` keeps no
per-kind message counts — so this probe rolls episodes and counts, under the
same rollout conventions as ``root_evidence_probe.py`` (same default seed and
episode count, same greedy flag, ``env.outcome`` decides "won" exactly as
``evaluate()`` does). It prints numbers and recommends nothing.

The bar as registered is unqualified — a REQUEST_STATUS from any sender
counts. The initial-root split is printed beside it because "the root asks"
and "a subordinate leader asks" are different findings; the split is colour,
not the bar.

    scripts/interrogative_usage_probe.py runs/fireteam_v24_seed12/ckpt_latest.pt \
        [more checkpoints ...] --episodes 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cohort.core.orders import MessageKind
from cohort.env.cohort_env import make_env
from cohort.training.evaluate import _pick_actions
from cohort.training.train import load_policy


def run_arm(checkpoint: str, episodes: int, first_seed: int, greedy: bool):
    net, ckpt = load_policy(checkpoint)
    scenario = ckpt.get("scenario")
    env = make_env(scenario)
    won = 0
    requests_total = 0
    requests_in_won = 0
    requests_by_initial_root = 0
    episodes_with_request = 0
    for k in range(episodes):
        ep_seed = first_seed + k
        torch.manual_seed(ep_seed)
        rng = np.random.default_rng(ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        msgs_before = len(env.transcript.messages)
        initial_root = env.roster.root()
        while env.agents:
            actions = _pick_actions(env, obs, net, rng, greedy=greedy)
            obs, _, _, _, _ = env.step(actions)
        reqs = [m for m in env.transcript.messages[msgs_before:]
                if m.kind is MessageKind.REQUEST_STATUS]
        requests_total += len(reqs)
        if reqs:
            episodes_with_request += 1
        if initial_root is not None:
            requests_by_initial_root += sum(
                1 for m in reqs if m.sender_id == initial_root.id)
        if env.outcome == "success":
            won += 1
            requests_in_won += len(reqs)
    label = f"{Path(checkpoint).parent.name} [{scenario}]"
    per_won = requests_in_won / won if won else None
    print(f"\n== {label} ==")
    print(f"  episodes {episodes}   won {won}   REQUEST_STATUS total "
          f"{requests_total} (in won eps {requests_in_won}, by initial root "
          f"{requests_by_initial_root}, episodes with >=1: "
          f"{episodes_with_request})")
    if per_won is None:
        print("  requests/won-ep — (no won episodes)")
    else:
        print(f"  requests/won-ep {per_won:.2f}   --usage "
              f"{Path(checkpoint).parent.name}={per_won:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()
    for c in args.checkpoints:
        run_arm(c, args.episodes, args.seed, args.greedy)


if __name__ == "__main__":
    main()
