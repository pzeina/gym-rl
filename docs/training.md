# Training

## Stack

Self-contained masked PPO in PyTorch (`cohort/training/ppo.py`, ~250 lines). No RLlib —
the legacy repo died of framework version drift, so the trainer here depends only on
`torch` + `numpy` and treats the environment purely through the PettingZoo parallel API.

* **Parameter sharing**: one actor-critic MLP (2×256 tanh) for every agent. Rank,
  mission, and org context are inputs, so the network learns *rank-conditional* policy:
  the same weights command as a TL and rifle as an RFN.
* **Action masking**: illegal logits → −1e9 before sampling *and* during updates, so
  probability mass (and entropy) live only on admissible actions.
* **Rollout buffer**: rectangular `(time, env, agent-slot)` arrays with a validity mask.
  Agents that die mid-episode simply stop producing transitions (their final step carries
  `done=1`); GAE runs per-stream and skips gaps. Truncated episodes bootstrap
  `γ·V(s_final)` into the last reward so timeouts aren't treated as failures.
* **Vectorization**: N independent `CohortEnv`s stepped in-process; all present agents
  across all envs share one forward pass per tick.

## Commands

```bash
# defaults: 8 envs, horizon 128, lr 3e-4 annealed, CPU
python -m cohort.training.train --scenario fireteam --total-steps 1500000
python -m cohort.training.train --scenario squad --total-steps 3000000 --run-name squad_v1

# monitor
python -m cohort.viz.dashboard            # interactive dashboard (live charts + episode explorer)
tensorboard --logdir runs
python -m cohort.viz.plots runs/<run>     # regenerate curves PNG anytime, even mid-run
# training_curves.png is also regenerated automatically when the run finishes

# evaluate
python -m cohort.training.evaluate runs/<run>/ckpt_best.pt --episodes 50
python -m cohort.training.evaluate --random --scenario fireteam   # baseline

# play against/with it
python -m cohort.play --checkpoint runs/<run>/ckpt_best.pt
```

## What the metrics mean

`runs/<run>/metrics.csv` (plotted by `cohort/viz/plots.py`):

| Column | Meaning |
|---|---|
| `ep_return` | mean per-agent episode return (team average) |
| `success_rate_rolling` | rolling success over the last 100 episodes |
| `comp_compliance` | mean per-agent-step compliance component — are orders being executed? |
| `comp_report` | reporting component — contacts / sitreps / completion reports |
| `comp_command` | leader component — doctrine-preferred orders, coverage, churn |
| `comp_combat` | hits, kills, casualties |
| `entropy` | policy entropy over *legal* actions |

A healthy run: `comp_compliance` climbs first (subordinates learn to execute),
`comp_command` turns positive as leaders learn to task everyone (coverage bonus dominates
churn), `comp_report` rises as contact reporting becomes routine, then `success_rate`
follows once the pieces compose.

## Curriculum notes

* `fireteam` trains in ~10 min on a laptop CPU and is the fastest sanity check.
* `squad` adds a second command echelon (the SL orders TLs, TLs order riflemen); expect
  to need 2–4× more steps.
* `platoon` (16 agents, three echelons) is the stretch goal; consider initializing from
  a `squad` checkpoint via `--init-from` (same observation/action spaces — checkpoints
  are compatible across scenarios). When fine-tuning from a converged checkpoint, lower
  the learning rate (e.g. `--lr 1e-4`): the default 3e-4 with a fresh anneal schedule
  can destabilize an already-good policy.
* Reward weights live in `cohort/env/rewards.py::RewardConfig`; scenario definitions in
  `cohort/config.py`. Both are plain dataclasses — experiment freely.
