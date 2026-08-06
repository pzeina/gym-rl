---
description: Unattended improvement loop — diagnose, fix, retrain, verify, log; no user input
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

Run the unattended improvement cycle. Scope, if given: $ARGUMENTS

This runs while the owner is away. **Never ask a question — decide, act, and
write the decision down.** The owner reads the ROADMAP progress log afterwards
and reverses anything they dislike; a blocked loop that waited for input is a
worse outcome than a documented wrong call.

## Standing authority (granted 2026-08-06)

Permitted without asking: reward/scenario/vocabulary redesign, breaking obs
changes, full retrains, local commits.

Explicitly forbidden, no exceptions: `git push`; closing or commenting on
GitHub issues; touching `~/Documents/gym-rl-fork` or deleting
`~/Documents/gym-rl-sync`; `FORCE=1` over a run directory that holds results
worth keeping; deleting any `runs/<name>/`.

**Design bias — the owner stated it directly.** Prefer a clean, well-defined
redesign that needs a full retrain over a minimal patch that preserves a
legacy quirk. The project is early; correctness and clarity beat continuity.
When a fix and a redesign both work, take the redesign, and say so in the
commit.

## The cycle

One pass = one item. Do not batch unrelated changes into a commit.

**1. Orient.** `scripts/train_status.py`. Any run that is `RUNNING` is
untouchable — never `FORCE=1` over it, never kill it. Note its ETA.

**2. Pick ONE item**, highest first:

   a. **A live run's blast radius.** A defect that will damage or has damaged
      a run currently training. Fix before it lands.
   b. **A FAILED or artifact-less finished run.** Reproduce the failure
      directly (run the failing stage on the checkpoint) rather than reading
      the deny-listed log; fix the cause; recover the lost artifacts.
   c. **An open diagnosis with a named mechanism** from the ROADMAP progress
      log or a previous cycle. Implement it.
   d. **A finished run with no verdict.** Digest it, judge it, log it.
   e. **A measurement that does not exist yet** for a defect already
      observed. Write the probe before theorising.

   Nothing in a–e? Write the handoff and stop the loop. Do not invent work.

**3. Diagnose before touching rewards.** `CLAUDE.md`'s rule, and it has paid
   every time. Use `scripts/oracle_probe.py` / `scripts/done_probe.py`, or
   write a new read-only probe under `scripts/`. State the mechanism as a
   claim plus the measurement that would refute it. **"No mechanism
   identified" is a valid result** — log it and move to the next item rather
   than shipping a plausible story.

**4. Change it.** Small, complete, one concern. Every new metric must be
   recorded by every scenario or explicitly blank-and-tolerated — that
   asymmetry is what cost fireteam_v7 its artifacts.

**5. Gate.** `pytest -q` (all green) **and** `ruff check cohort/ tests/
   scripts/`. A behavioural change also needs a functional spot-check: run the
   thing and read the output. Never commit on red — fix or revert.

**6. Encode the hazard.** A defect that reached a run gets a test that fails
   without the fix. Regression-hazard tests are this repo's memory.

**7. Commit.** One commit per item, `refs #N` when an assurance issue matches.
   The message states the mechanism and the evidence, not just the change.

**8. Retrain when behaviour could have moved.** `scripts/train.sh <name>`,
   detached, next free version suffix. Never foreground, never a raw
   `python -m cohort.training.train`. A spaces break (`OBS_DIM`) orphans every
   older checkpoint — say so in the commit and treat the first post-break run
   as the new baseline.

**9. Log it.** Append to the ROADMAP progress log: what changed, why, the
   numbers, and the verdict — **including misses, with their numbers**. An
   honest miss is a result; a quiet one is a lie the next session inherits.

**10. Loop.** Re-orient and take the next item. While a retrain is in flight,
   work items that do not touch what it is training.

## Pacing

Training costs zero tokens; polling costs tokens. Schedule the next wake for
roughly the ETA of whatever is training, or ~20-30 min when nothing is. Never
poll in a tight loop, never spawn an agent to watch a run.

## Verdicts

- Compare against a **named baseline** and its CI. Non-overlapping intervals
  or it is not an effect.
- A behaviour suite that contradicts the success rate outranks the success
  rate — that is where the exploits live.
- A metric that improves because its denominator vanished has **not**
  improved. Say so explicitly.
- N=20 is a smoke test. Publish decisions need N=100.

## Ending

Stop the loop when a–e is empty, when a change needs a design call only the
owner can make (log the options and stop), or on repeated unexplained
failure. Always finish by writing the ⟳ Session handoff block at the top of
ROADMAP.md so the next session resumes cold.
