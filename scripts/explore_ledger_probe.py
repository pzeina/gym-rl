"""Masked-random rollout: what does exploration actually experience, per price?

Usage: python explore_tally.py <repo_root> <scenario> <episodes> <seed>
Instruments RewardLedger.add (read-only wrap) and tallies (component, value)
counts; prints per-agent-step component means, the report component broken
down by price value, and transcript kind counts.
"""
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
scenario, episodes, seed0 = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

import numpy as np  # noqa: E402

from cohort.env import rewards as reward_mod  # noqa: E402
from cohort.env.cohort_env import make_env  # noqa: E402
from cohort.training.evaluate import _pick_actions  # noqa: E402

tally = Counter()      # (component, value) -> count
real_add = reward_mod.RewardLedger.add
def wrapped(self, agent, component, value):
    tally[(component, round(float(value), 4))] += 1
    return real_add(self, agent, component, value)
reward_mod.RewardLedger.add = wrapped

env = make_env(scenario)
agent_steps = 0
kinds = Counter()
for k in range(episodes):
    rng = np.random.default_rng(seed0 + k)
    obs, _ = env.reset(seed=seed0 + k)
    m0 = len(env.transcript.messages)
    while env.agents:
        agent_steps += len(env.agents)
        actions = _pick_actions(env, obs, None, rng)
        obs, _, _, _, _ = env.step(actions)
    for m in env.transcript.messages[m0:]:
        kinds[m.kind.value] += 1

print(f"tree={root}  scenario={scenario}  eps={episodes}  agent-steps={agent_steps}")
comp_sums = Counter()
for (c, v), n in tally.items():
    comp_sums[c] += v * n
print("component means/agent-step:",
      {c: round(s / agent_steps, 5) for c, s in sorted(comp_sums.items())})
print("report breakdown (value -> count):")
for (c, v), n in sorted(tally.items()):
    if c == "report":
        print(f"  {v:+.4f} x {n}   (/agent-step {v*n/agent_steps:+.5f})")
print("message kinds:", dict(kinds.most_common(12)))
