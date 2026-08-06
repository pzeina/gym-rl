---
description: Reconstruct session state in one step — handoff, git, tests, running trainings, next action
allowed-tools: Bash, Read, Glob, Grep
---

Reconstruct where this project stands and what to do next. Be brief: this runs
at the *start* of a session, so it must cost little and say much.

Gather (batch these — they are independent):
1. `sed -n '1,60p' ROADMAP.md` — the "⟳ Session handoff" block is the source of
   truth for state and priorities.
2. `git log --oneline -5` and `git status --short`, plus how far ahead of
   `main` the branch is: `git rev-list --count main..HEAD`.
3. `.venv/bin/python -m pytest tests/ -q 2>&1 | tail -2` and
   `.venv/bin/python -m ruff check cohort/ tests/ 2>&1 | tail -2`.
4. `.venv/bin/python scripts/train_status.py` — anything training right now.
5. Current spaces, since a breaking cycle invalidates every checkpoint:
   `.venv/bin/python -c "from cohort.env.observations import OBS_DIM; from cohort.env.actions import N_ACTIONS; print(f'Discrete({N_ACTIONS})/Box({OBS_DIM})')"`

Then report in **under 15 lines**:

```
branch <b> @ <sha> (<n> ahead of main, pushed|UNPUSHED) · <n> tests <green|N FAILING> · ruff <clean|N>
spaces Discrete(x)/Box(y)   [flag loudly if an open breaking cycle means the fleet is stale]
training: <none | run (pid, ~n% done)>
uncommitted: <files, or "clean">

STATE   <2-3 lines from the handoff block — what is actually true right now>
NEXT    1. <the top recommended action, with the exact command if there is one>
        2. <the second>
```

Rules:
- **Do not read** `metrics.csv`, `tb/`, or training logs. Do not open run
  directories. Everything needed is above.
- If tests fail or ruff complains, that is the headline — say it first and name
  the failing tests.
- Do not start work, do not launch training, do not offer to. This command
  reports; I decide.
- If the handoff block contradicts the repo (e.g. it names a commit that is not
  HEAD), say so explicitly rather than trusting the prose.
