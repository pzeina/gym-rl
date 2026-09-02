# Night orders — 2026-09-03 (AUTO mode, owner retired ~00:30)

The owner handed over the night with the standing instruction: do not idle,
train and experiment. This file is the queue and the guardrails; the
self-paced loop reads it at every wakeup. **No questions at night — decide,
act, write it down.** A documented wrong call beats a thread blocked to dawn.

## Authority tonight

Pre-authorised: launching training runs and campaigns, every zero-token
measurement (probes, evals, oracle, digests), committing finished work (full
pytest + ruff green, one commit per unit, repo trailers), pushing
`multi-agent-dev`.

NOT tonight, regardless of findings — written up for morning instead:
merging or tagging `main`; anything destructive; publishing a MISS over an
incumbent; design decisions (rewards, vocabulary, scenario semantics,
masks/enforcement, rewording owner-decided claims). **And tonight
specifically: `cohort/` is FROZEN at tree `19a8da08` until the queue drains.**
Tooling, tests, docs, boards and `runs/` stay free.

## The situation, in one paragraph

`4e82807` removed five heard-mission scalars: OBS_DIM 351 → 346. **Every
v1.25 member is unloadable** (`mat1 and mat2 shapes cannot be multiplied
(16x346 and 351x256)`). `BASELINE.json` still names v1.25, and nothing in it
can be evaluated on this tree. v1.26 is the forced retrain: 21 jobs, queue
**pid 75021**, launched 00:20, `logs/queue_20260903_002004_74885.log`.

**The campaign will not drain tonight.** Job 1 is running at ~1300 sps, so a
3M job is ~40 min and 21 jobs is ~14h. Expect **roughly jobs 1–11 by 08:00**
— the five singles, the four `platoon` seeds, and the first `platoon_hard`.
Morning inherits a queue still running. That is expected, not a failure.

## What to read, and it is NOT success

The removal takes away exactly what a leader knew about who is already
tasked. So the hazard is the **order loop**, not the win rate:

- `retasks_per_episode` — a leader that cannot see a subordinate's current
  mission may re-task an agent already doing the job.
- `obedience_latency_mean`
- `closed_on_root_report_rate` (v1.25: `platoon_hard` 0.01, `platoon` 0.58)

Success moving is the least interesting outcome available here. The v1.25
artifacts are **committed JSON and still readable** even though their weights
are not loadable, so every comparison below is a file read, not an eval.

## The queue, gated on landings

Every landing, in this order:

1. **Bookkeeping first.** Same-config draws declared in `seed_spread`
   (`platoon_hard_control_v1_seed13` was the last one to catch this out);
   declared ⇒ tracked; artifacts committed; suite green; push.
2. **Digest.** `scripts/run_report.py <run> --vs <v1.25 counterpart>` — the
   order-loop triple above, then success.
3. **No launch.** The queue is serial and already saturating CPU; a second
   lane would slow it, not speed the night. **The one exception:** if pid
   75021 dies with jobs left, relaunch the remainder from the jobs file
   (pre-authorised, and say so in the ledger).

## Decision rules, named now

- **A job collapses (success ≈ 0)** → declare it, note it, carry on. The four
  seed searches exist precisely so one bad draw is not a verdict. v1.24's
  collapses were both seed 12 and it changed nothing about the fleet.
- **The order loop degrades materially against v1.25** → that IS the finding
  this cycle exists to produce. Write it up with numbers. Do **not** act: the
  observation contract is an owner-decided change, not a regression to fix.
- **The order loop is flat** → equally a finding, and the more surprising one
  — it would mean a designed information channel was carrying nothing.
- **`BASELINE.json` is not touched tonight.** It names an unloadable fleet,
  and the swap needs the full campaign plus an owner call on any MISS.

## Idle-time work: deliberately none launched

The honest position rather than invented busywork. Almost every zero-token
probe this repo owns reads a checkpoint, and **every v1.25 checkpoint is
unloadable on this tree** — the probes are dead until v1.26 members exist.
The one thing worth having accrues for free: `mean_second_nearest_teammate_dist`
survived into the frozen tree (verified), so every v1.26 run writes it at
exit. By morning the landed jobs give the **first fleet-wide reading of the
pair-vs-pile statistic** — the evidence base for the buddy-pair pricing
question the owner carried forward — at zero cost and without stealing CPU
from the campaign.

If `patrol_brique` seeds land before dawn (jobs 15–18, unlikely), the pending
oracle diagnosis of its vanished denominator becomes runnable on a loadable
checkpoint. Gated on that landing, not scheduled.

## Morning

ROADMAP handoff updated with the night's ledger: what landed, what was read,
what was launched, every commit, every miss with its diagnosis. Commit and
push. Boards will read PUBLISH PENDING for the owner's `/boards`. One
PushNotification with the outcome worth acting on.

## Queue state (updated as the night runs)

- 00:20 — campaign launched (not by this watch), 21 jobs, pid 75021.
- 00:30 — watch opened. Job 1 `fireteam_v16` at 18%. Nothing else running.
