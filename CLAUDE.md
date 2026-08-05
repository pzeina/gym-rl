# cohort (gym-rl)

Chain-of-command multi-agent RL: NATO-ranked agents (RFN…CO, STANAG grades) learn to
obey, report, and issue doctrine-valid orders; all C2 traffic is NATO voice procedure.

## Commands

```bash
.venv/bin/python -m pytest tests/ -q                  # 66 tests, ~2s
.venv/bin/python -m ruff check cohort/ tests/
.venv/bin/python -m cohort.training.train --scenario fireteam --total-steps 1500000
.venv/bin/python -m cohort.training.evaluate runs/<run>/ckpt_best.pt --episodes 20
.venv/bin/python -m cohort.play --checkpoint runs/<run>/ckpt_best.pt
```

## Architecture (see docs/architecture.md)

- `cohort/core/` — pure domain logic, no RL deps: ranks/authority, missions/doctrine/
  compliance, radio messages, command-language parser, roster/succession, terrain/LOS.
- `cohort/env/` — PettingZoo ParallelEnv; agent ids are callsigns (TL1, RFN2…).
  Rank admissibility = hard action masks (`env/actions.py`); behavior = rewards
  (`env/rewards.py::RewardConfig`).
- `cohort/training/` — self-contained masked PPO (torch only, NO RLlib — the legacy
  RLlib stack is why the old repo died). Handles agent death via validity-masked GAE.
- Scenarios/org charts: `cohort/config.py`. Runs → `runs/<name>/`.
- `legacy/` — archived pre-rewrite implementation; never import from it; excluded
  from pytest via pyproject `norecursedirs`.

## Conventions

- Determinism: all env randomness through `env._rng` (seeded in `reset`); never use
  global `np.random` in env/core code.
- Every C2 event must appear on the transcript as a formatted message
  (`core/language.py` formatters); formatter/parser stay inverses (round-trip test).
- Obs layout changes require updating `OBS_DIM` math in `env/observations.py`
  (asserted at build time) — old checkpoints become incompatible.
