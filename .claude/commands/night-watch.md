---
description: Overnight AUTO mode — write the night orders, arm the watch, then self-pace through landings until morning
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Monitor, TaskList, TaskStop, ScheduleWakeup, PushNotification, Skill
---

The owner is retiring and hands the night over. Focus, if given: $ARGUMENTS

Run the night exactly as 2026-08-18's first watch did (docs/night-orders-2026-08-18.md
is the worked example). **Never ask a question at night — decide, act, write it
down.** A documented wrong call beats a loop blocked until morning.

This is NOT `/autocycle`: that loop redesigns and never pushes. The night
watch is the opposite trade — it pushes freely and designs nothing.

## Authority (standing, owner 2026-08-18: "do not stay doing nothing — TRAIN! and experiment")

Pre-authorised: launching training runs and campaigns (including confirm
seeds a scout protocol calls for), every zero-token measurement (probes,
evals, oracle, report digests), committing finished work (full pytest + ruff
green, one commit per unit, repo trailers), pushing `multi-agent-dev`.

Forbidden regardless of findings, write up for morning instead: merging or
tagging `main`; anything destructive (reset --hard, force-anything, deleting
run dirs); publishing a MISS over an incumbent; design decisions — rewards,
vocabulary, scenario semantics, masks/enforcement (measured axes stay
measured), rewording owner-decided claims. Honest-DoD everywhere: one
retrain + one diagnosed adjustment per miss, then document and stop that
thread. Token discipline holds at night: digests only, never raw logs,
`metrics.csv`, or checkpoints into context.

## Setup (once, at invocation)

1. **Orient**: ROADMAP "⟳ Session handoff", `scripts/train_status.py`,
   docs/next-cycles.md. What is running, what lands tonight, what is pending
   and zero-token?
2. **Write `docs/night-orders-<date>.md`**: tonight's landing-gated queue
   (every read gated on its landing; neutrality/confound gates first; arms
   before verdicts), the decision rules for each branch (separates → next
   pre-authorised step; wall → the ONE diagnosed adjustment in scope, named
   now; ceiling → write-up, knob back to the owner), the idle-time
   experiment list, and the authority block above. $ARGUMENTS override or
   focus the queue. Commit and push it — it is the contract the morning
   reader audits.
3. **Fill the idle time**: launch pending zero-token measurements detached
   (`nohup ... > logs/<name>.log`), ending each script with an
   `echo <NAME>-DONE` sentinel line.
4. **Arm the watch** (check TaskList first — never arm twice):
   `Monitor` with `persistent: true`, command
   `scripts/night_watch_monitor.sh logs/<sentinel1>.log logs/<sentinel2>.log ...`
   — it emits one line per training that ends (any outcome) and one per
   finished sentinel job.
5. **Enter the loop**: confirm the setup to the (absent) owner in one short
   message, then `ScheduleWakeup` with prompt
   `/loop self-paced overnight: follow docs/night-orders-<date>.md — read landings, launch follow-up training, run zero-token probes — until morning`
   and a 1500–1800s fallback delay (the Monitor is the primary wake signal).

## Each wakeup

`train_status.py` → act on whatever landed, per the night orders:
bookkeeping first (same-config draws declared in `seed_spread`, declared ⇒
tracked, artifacts committed, suite green, push), then the gated reads
(`run_report.py` digests, `baseline.py` for neutrality/reproduction gates,
`ablation_report.py` for arm reads), then any launch the decision rules call
for (`scripts/train.sh` / `train_queue.sh`, detached, next free version
suffix). Update the night-orders file's queue state if it changed shape.
Re-arm nothing; schedule the next fallback wakeup with the same /loop prompt
and go back to sleep.

## Morning (owner returns, or ~08:00, or the queue is exhausted)

1. ROADMAP handoff updated with the night's ledger: what landed, what was
   read, what was launched, every commit, every miss with its diagnosis.
2. Commit + push; note PUBLISH PENDING boards for the owner's `/boards`.
3. One `PushNotification` with the outcome the owner would act on.
4. `TaskStop` the monitor, `ScheduleWakeup` with `stop: true`.
