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
.venv/bin/python scripts/update_boards.py             # re-render all three boards (automatic on landing)
.venv/bin/python scripts/baseline.py                  # is the shipping fleet one system?
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

**Boards & artifacts.** Three published boards visualise the project:
`runs/fleet_board.html` (the baseline, then every other run, CI, gates),
`runs/program_board.html` (the settled experiments and what they rest on) and
`runs/scenario_gallery.html` (one evaluated episode per scenario, as radio
traffic — the only one where the artifact IS the claim). **All three refresh
themselves when a run lands** — `scripts/train.sh` launches through `scripts/train_then_boards.sh`, which
re-runs `scripts/update_boards.py` after training exits, crash or not. Zero tokens,
no session involved.
- The one step a shell cannot do is *publishing* to claude.ai — that needs the
  Artifact tool, which only exists inside a session. So `update_boards.py` records a
  content digest of what was last published in `runs/.boards.json`, and
  `scripts/train_status.py` prints **PUBLISH PENDING** when they diverge. `/boards`
  closes that gap in one step and stamps `--mark-published`.
- The digest is over what the boards *say* (results, gates, overrides, run state),
  not the rendered bytes — a live run's percentage ticking must not mark the
  artifacts stale, or the signal is worthless.
- **Never hand-edit the HTML.** It is generated; fix `scripts/fleet_board.py`,
  `scripts/program_board.py` or `scripts/scenario_gallery.py`. Board numbers come from each run's committed
  `behavior_final.json` (final policy) / `behavior.json` (best checkpoint) — the
  board states which and at what N, because captioning N=20 rows as N=100 is exactly
  the overstatement `publish_audit.py` exists to catch.

## The baseline fleet (v1.19 onward) — READ BEFORE PUBLISHING ANYTHING

`runs/BASELINE.json` names **one run per doctrine scenario**, and that set is what
the project ships. It exists because the fleet it replaced was not one thing:
eight champions at seven commits, four reproducible only with a `--reward`
override that had since become the default, one published with a flag saying it
missed the bar. Every number was honest; the set was not a system.

```bash
.venv/bin/python scripts/baseline.py            # the gate — exit 0 or the reasons
.venv/bin/python scripts/baseline.py --seal     # stamp the cohort/ tree it holds
.venv/bin/python scripts/publish_baseline.py    # score every member at N=100 (detach it)
.venv/bin/python scripts/results_table.py --write   # regenerate the README table
.venv/bin/python scripts/archive_runs.py [--apply]  # file the superseded runs away
```

- **Provenance is the `cohort/` tree, never the commit sha.** Resolved from each
  run's recorded `git_commit`. A tooling commit between two launches is routine
  and says nothing about the runs; two members either side of an env change are
  not one system however adjacent their shas look.
- **A campaign freezes `cohort/`.** A queue launches each job when it reaches it
  and `train.py` imports the tree that exists at that moment, so a commit to
  `cohort/` mid-campaign trains the last members against a different environment.
  Tooling, tests, docs and boards stay free to move.
- **No `--reward` overrides in a baseline run.** What ships is what was trained.
  A scenario that needs an override to work is a finding about the defaults.
- **The README results table is generated** from the members' committed
  evaluations; `tests/test_results_table.py` fails when it drifts. Do not hand-edit
  it — every overstatement this repo has corrected was a hand-kept number.
- **Archiving is a move, never a delete.** `runs/archive/` keeps the evidence
  behind published claims resolvable; every reader goes through
  `fleet_status.find_run` / `run_report.run_dir`, and a test enforces that.

**Division of labour.** Cheap models move data (launch, extract, summarise). The big
model does what only it can: reading a digest, judging whether an effect is real
against the CI, deciding the next experiment. Never invert this.

**Session hygiene.** One campaign per session. `/clear` when switching scenarios or
after a launch — context carried across unrelated runs is pure cost, since every run's
state lives in `runs/<name>/` and is re-readable in ~30 lines at any time.

Slash commands: `/train`, `/train-status`, `/train-report`, `/boards`.

## Operating guide (established practice — follow unless the owner redirects)

- **Session start**: read the "⟳ Session handoff" block at the top of ROADMAP.md —
  it carries current state, verdicts, and the prioritized next steps. For what to
  do next rather than where things stand, `docs/next-cycles.md` is the plan: it
  says which changes force a fleet retrain and which can ride along.
- **Delegation**: substantial build/retrain campaigns run in ONE background
  general-purpose agent with a precise phased brief: one commit per phase,
  pytest+ruff green per commit, spaces frozen unless the cycle is explicitly
  breaking, NO git push from agents, honest-DoD protocol (one retrain + one
  diagnosed adjustment, then document the miss and stop). Agents idle out while
  their nohup trainings run — resume them via SendMessage with the active-polling
  reminder (`until ! ps -p $(cat PIDFILE) ...` in <9-min Bash calls). Training
  *launch/monitor* ops go to the cheap-model workflow below, never general-purpose.
- **Standing authority — do not ask for a "go"** (owner's instruction,
  2026-08-11: *"minimize such necessity of a 'go' command, I lose time with
  those, whereas you can proceed directly"*). These are pre-authorised on
  `multi-agent-dev`; do them and say what you did:
  - **Commit** finished work. Conditions, not formalities — full pytest + ruff
    green, one commit per coherent unit, a message that says what changed and
    why, and the repo's two trailers.
  - **Push** `multi-agent-dev`.
  - **Launch training** — experiment arms, confirmation seeds, retrains, whole
    campaigns. A run costs wall-clock and zero model tokens and publishes
    nothing by itself, so asking costs more than the run.
  - **Apply and publish** a result that BEATS its incumbent, via `/publish`.

  **The distinction that makes this safe: running an experiment is not deciding.**
  An arm that tests a hypothesis adds evidence and can be ignored; changing a
  default, a reward, a vocabulary or a scenario's semantics is the decision, and
  that stays the owner's — present options, a recommendation, and the measurement
  that would settle it.

  Still an explicit ask, every time: **merging to `main`**, **tagging**, anything
  **destructive** (`reset --hard`, history rewrites, force-anything, deleting a
  run directory), and **publishing a MISS over an incumbent** — misses ship with
  numbers and a diagnosis, but whether one supersedes a published claim is a
  judgement about the project's claims.

  Pre-authorised is not the same as silent: say what was committed, pushed or
  launched, in the reply.
- **Shipping** (main session only): review gate (full pytest + ruff + a functional
  spot-check) → push `multi-agent-dev` → merge `main` (fast-forward, re-test) →
  annotated monotonic tag `vX.Y.0` for milestone-sized ships (at v1.9.0).
- **Design decisions are the owner's**: reward structure, vocabulary, scenario
  semantics, breaking cycles — present options + a recommendation and ask; don't
  autopilot. Diagnose with the oracle (`env.oracle()`) BEFORE changing rewards.
- **Honesty**: misses ship with numbers and diagnosis in ROADMAP's progress log;
  regression-hazard tests (terminal dominance, churn/rotation, weapons-tight,
  Message text-only schema) each encode a real exploit — keep them green.
- **Assurance layer** (separate project, contract in ASSURANCE-SYNC.md): the Stop
  hook queues commits automatically and surfaces new GitHub issues once — handle
  via ONE dedicated fix agent, commits `refs #N`, NEVER close/comment issues,
  never touch ~/Documents/gym-rl-fork or delete ~/Documents/gym-rl-sync.

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
