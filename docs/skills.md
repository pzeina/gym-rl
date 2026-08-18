# The skills (slash commands)

Every command in `.claude/commands/`, what it does, and what it deliberately
refuses to do. The refusals are not politeness — each one encodes a cost or an
honesty rule this repo learned the hard way (the token-discipline section of
CLAUDE.md is the background).

The daily loop, as a map:

    /resume ──► work ──► /train ──► /clear ──► (hours pass) ──► /train-status
                  ▲                                                  │
                  │                                       landed?  /train-report
                  │                                                  │
             /diagnose ◄── behavior wrong, reward change tempting    │
                                                                     ▼
                                              beats incumbent?   /publish ──► /boards
    unattended:  /autocycle (days, redesigns, never pushes)
                 /night-watch (one night, measures & trains, pushes, designs nothing)

## /resume

**Effect**: reconstructs session state in one step — ROADMAP handoff block,
git position vs `main`, test/ruff status, live trainings, current
Discrete/Box spaces — and reports in under 15 lines with a recommended NEXT.
**Use at** the start of any session.
**Never**: reads `metrics.csv`/`tb/`/logs, starts work, launches anything.
It reports; the owner decides.

## /train

**Effect**: launches a training run (or, via a jobs file, a whole campaign)
fully detached through `scripts/train.sh` / `scripts/train_queue.sh`, by
delegating to the cheap **train-ops** agent. Confirms name, pid, log, ETA in
three lines and recommends `/clear` — the run survives it.
**Use when** anything needs training. A run costs wall-clock and zero model
tokens; launching is pre-authorised.
**Never**: trains in the foreground, polls, analyses, or uses an expensive
agent for what is a shell operation.

## /train-status

**Effect**: one cheap `scripts/train_status.py` call — live runs with
progress/ETA, recent landings, and whether the published boards have drifted
(PUBLISH PENDING → `/boards`).
**Use for** checking in on detached runs, any time, from any session.
**Never**: reads raw logs or `metrics.csv`, speculates about mid-run numbers,
or keeps watching.

## /train-report `<run> [vs baseline]`

**Effect**: the analysis step that is actually worth big-model tokens. Pulls
the ~30-line `scripts/run_report.py` digest (curve, decile deltas,
reward-component drift, behavior suite, A/B economics with a CONFOUNDED
verdict when the `cohort/` tree moved), then judges: learned or regressed,
which reward component moved and whether that matches the change under test,
whether behavior agrees with the success rate, whether the effect clears the
baseline's CI. Ends with exactly one next action — and launches a warranted
follow-up retrain itself rather than asking.
**Never**: pulls `metrics.csv`/`tb/`/logs into context; replaces something
published without saying so first.

## /diagnose `<run> [vs baseline]`

**Effect**: the mandatory step before any reward or scenario change. A cheap
**oracle-diagnose** agent gathers the `env.oracle()` fact-sheet (protocol
seeds 500+, comparable to every prior diagnosis); the session then reasons
about **mechanism, not score** — did it fight when it could, where did the
fight happen, what was it doing under threat, where did they die — and states
the mechanism as a falsifiable claim with the measurement that would refute
it.
**Never**: launches anything or proposes more than one change. "No mechanism
identified" is a valid result.

## /publish `<run>`

**Effect**: the publication ritual. A cheap **publish-ops** agent runs the
mechanics (loadability, N=100 on the checkpoint, `behavior.json`,
`probe.json`); the session judges beat/match/miss against the incumbent's CI
and the regression gates, drafts the README results-table row and the dated
ROADMAP progress-log entry, then applies and commits a BEAT (the honesty
rules are machine-enforced: the table is generated, `publish_audit.py`
catches overstatement).
**Never**: publishes a MISS over an incumbent without an explicit owner ask —
misses ship with numbers and a diagnosis, but superseding a published claim
is the owner's judgement. Stops cleanly when a checkpoint predates a breaking
cycle.

## /boards

**Effect**: closes the one gap a shell cannot — republishing the three HTML
boards to claude.ai (fleet 📡, program 🧭, gallery 📻) at their recorded
artifact URLs, then stamps `--mark-published`. The HTML itself refreshes
automatically whenever a run lands.
**Use when** `/train-status` prints PUBLISH PENDING.
**Never**: hand-edits the generated HTML — a wrong number is fixed in
`scripts/fleet_board.py` / `program_board.py` / `scenario_gallery.py`.

## /autocycle `[scope]`

**Effect**: the unattended *improvement* loop — pick ONE item (blast radius
of a live run first, then failed runs, then the roadmap), diagnose, fix or
redesign, retrain, verify, log, repeat. Carries its own standing authority
(2026-08-06): may redesign rewards/scenarios/vocabulary and run full
retrains, with a stated bias for clean redesigns over minimal patches.
**Never**: pushes, touches GitHub issues, forces over live runs, deletes run
directories, or asks a question — decisions are written down for the owner to
reverse.

## /night-watch `[focus]`

**Effect**: hands one night over (owner instruction 2026-08-18: "do not stay
doing nothing — TRAIN! and experiment"). Writes the dated
`docs/night-orders-<date>.md` contract (landing-gated queue, per-branch
decision rules with the single diagnosed adjustment named up front, idle-time
zero-token experiments), launches pending measurements detached, arms the
persistent watch (`scripts/night_watch_monitor.sh` — one wake event per
training that ends, any outcome, plus sentinel-marked detached jobs), then
self-paces through the landings: bookkeeping, gated reads, pre-authorised
follow-up launches, commit + push per unit. Morning: ROADMAP ledger, one push
notification, loop stopped.
**The deliberate inverse of /autocycle**: pushes freely, designs nothing.
**Never**: merges/tags `main`, does anything destructive, publishes a MISS,
makes design decisions (rewards, vocabulary, semantics, enforcement —
measured axes stay measured), reworks owner-decided claims, or asks a
question at night.

## Division of labour, in one line each

- Cheap agents move data: **train-ops** (launch), **run-digest** (sweep
  digests), **oracle-diagnose** (fact-sheets), **publish-ops** (N=100
  mechanics).
- The session spends big-model tokens only where judgement lives: reading a
  digest, calling an effect real against a CI, choosing the next experiment.
- Shell processes cost zero tokens: training, evaluation, probes, boards.
  Nothing in this repo babysits them; the harness wakes on landings.
