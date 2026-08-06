# cohort (gym-rl)

Chain-of-command multi-agent RL: NATO-ranked agents (RFN…CO, STANAG grades) learn to
obey, report, and issue doctrine-valid orders; all C2 traffic is NATO voice procedure.

## Commands

```bash
.venv/bin/python -m pytest tests/ -q                  # 340 tests, ~6s
.venv/bin/python -m ruff check cohort/ tests/
scripts/train.sh <run-name> --scenario fireteam --total-steps 1500000   # detached; see below
scripts/train_queue.sh <jobs-file>                    # a whole campaign, detached
.venv/bin/python scripts/train_status.py [run]        # cheap check-in
.venv/bin/python scripts/run_report.py <run> [--vs <baseline>]          # compact digest
.venv/bin/python -m cohort.training.evaluate runs/<run>/ckpt_best.pt --episodes 20
.venv/bin/python -m cohort.play --checkpoint runs/<run>/ckpt_best.pt
```

## Training workflow — token discipline (READ THIS FIRST)

Training is a shell process, not a reasoning task: a 3M-step run is ~20–40 min of
CPU and **zero model tokens**. Historically this repo's cost came from running or
babysitting training inside long, high-context sessions. The rules below are not
style preferences — they are the cost model.

**Launching.**
- Always `scripts/train.sh` / `scripts/train_queue.sh`. They `nohup` the run, so it
  survives `/clear`, session exit, and a closed terminal.
- Never run `python -m cohort.training.train` in the foreground of a turn, and never
  wrap a raw train command in a background Bash call.
- Delegate launches to the **train-ops** agent (haiku). Never use `general-purpose`
  for training ops — it is the single largest line item in this project's usage.

**Waiting.**
- Do not poll. Do not spawn an agent to watch a run. Either (a) tell the user the run
  is detached and *stop the session*, or (b) for a run under ~30 min,
  `scripts/train_wait.sh <run>` as a **background** Bash call — the harness wakes up
  once, on exit, instead of once per poll.
- A launch is a natural end of session. After confirming it, say so and recommend
  `/clear`; come back later with `/train-status`.

**Reading results.**
- `runs/*/metrics.csv`, `runs/*/tb/`, and `logs/*.log` are **deny-listed for Read** in
  `.claude/settings.json`. A 3000-row CSV in a 150k-token context is the exact failure
  this workflow exists to prevent. Use `scripts/run_report.py` — it collapses the run
  to ~30 lines (curve, decile deltas, reward-component drift, behavior suite).
- Bounded `grep`/`tail` over a log via Bash is fine for diagnosing a crash.
- Use the **run-digest** agent (haiku) to sweep several runs; it returns facts only.

**Division of labour.** Cheap models move data (launch, extract, summarise). The big
model does what only it can: reading a digest, judging whether an effect is real
against the CI, deciding the next experiment. Never invert this.

**Session hygiene.** One campaign per session. `/clear` when switching scenarios or
after a launch — context carried across unrelated runs is pure cost, since every run's
state lives in `runs/<name>/` and is re-readable in ~30 lines at any time.

Slash commands: `/train`, `/train-status`, `/train-report`.

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
