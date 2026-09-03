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

- 01:08 — **job 1 `fireteam_v16` landed** (succ 0.95 at N=20). Declared in
  `seed_spread[fireteam]`, artifacts tracked, suite green, pushed `09b3814`.
  Queue pid 75021 alive and starting job 2; the momentary "live (0)" was the
  gap between jobs, not a death.
- 01:10 — **first order-loop read, and it is the predicted signature.**
  Against `fireteam_v15` (N=100, obs 351):

  | | v15 | v16 |
  |---|---|---|
  | success | 0.970 | 0.950 |
  | `closed_on_root_report_rate` | 0.722 | **0.000** |
  | `retasks_per_episode` | 2.100 | **3.150** |
  | `false_complete_rate` | 0.721 | 1.000 |
  | `obedience_latency_mean` | 1.682 | 1.745 |

  Success barely moves; the order loop does. Re-tasking +50% is exactly the
  mechanism the removal predicts — a leader that cannot see a subordinate's
  current mission re-tasks one already doing the job. **One run, and the read
  crosses N (20 vs 100), so it is a direction and not a verdict.** N=100 on
  `fireteam_v16` launched to remove that caveat for the headline scenario.
  Per the decision rules this is written up and NOT acted on: the observation
  contract is owner-decided, not a regression to fix.
- 01:10 — `mean_second_nearest_teammate_dist` reads 7.699 on the first v1.26
  run (v15 has None — it predates the metric). The pair-vs-pile evidence base
  the owner asked to accrue is accruing.

- 01:38 — **N=100 anchor on `fireteam_v16` done; the caveat is removed and the
  finding holds.** Both sides now N=100:

  | fireteam | v15 (obs 351) | v16 (obs 346) |
  |---|---|---|
  | success | 0.970 | 0.950 |
  | `closed_on_root_report_rate` | 0.722 | **0.000** |
  | `retasks_per_episode` | 2.100 | **3.270** |
  | `obedience_latency_mean` | 1.682 | 1.965 |
  | `false_complete_rate` | 0.721 | 0.875 |

  Re-tasking +56%, obedience latency +17%, and the root's DONE channel closed
  outright — while success moves 0.02. The prediction in the handoff was
  exactly right: the cost lands on the order loop, not the win rate.

  **Anchoring policy for the rest of the night** (CPU discipline): a
  training-exit N=20 read is enough to SEE a 0.722 -> 0.000 collapse, so
  subsequent scenarios are read at N=20 and anchored at N=100 only where the
  N=20 read is ambiguous. Nine blanket anchors would cost ~36 min of CPU taken
  from a queue that already will not drain by morning.
- 01:38 — job 2 `fireteam_defend_v27` at 57%, eta ~23m. Queue healthy.

- 02:05 — **job 2 `fireteam_defend_v27` landed, and it REFUTES the simple
  reading.** Declared in `seed_spread[fireteam_defend]`. Its order loop did not
  degrade — it improved:

  | fireteam_defend | v26 (N=100) | v27 (N=20) |
  |---|---|---|
  | success | 0.990 | 0.950 |
  | `closed_on_root_report_rate` | 0.970 | **1.000** |
  | `retasks_per_episode` | 0.160 | **0.000** |
  | `obedience_latency_mean` | 2.027 | 1.538 |
  | `human_death_rate` | 0.070 | 0.200 |

  So "the observation removal breaks the order loop" is already too coarse
  after two scenarios.

### A prediction, registered at 02:05 BEFORE the scenarios that test it land

The two results differ in how much their scenario USES the order loop.
`fireteam` re-tasks 2.10 times an episode and broke; `fireteam_defend` re-tasks
0.16 times and did not. A leader that rarely re-tasks cannot be hurt by losing
the telemetry about who is already tasked.

**Hypothesis: the cost of the removal scales with the scenario's baseline
re-tasking rate.** The v1.25 fleet's rates, which are committed and were not
chosen for this purpose:

| scenario | retasks/ep | status |
|---|---|---|
| `platoon` | 43.03 | predicted WORST |
| `platoon_hard` | 28.05 | predicted severe |
| `squad_recon` | 6.46 | predicted moderate |
| `squad` | 5.33 | predicted moderate |
| `squad_screen` | 4.18 | predicted moderate |
| `fireteam` | 2.10 | **observed: broke** (report 0.722 -> 0.000) |
| `defend_brique` | 1.46 | predicted mild |
| `patrol_brique` | 0.28 | predicted none |
| `fireteam_defend` | 0.16 | **observed: no degradation** |

**Predicted, in order of landing:** `squad` / `squad_recon` / `squad_screen`
degrade clearly; `defend_brique` mildly or not at all; the four `platoon` seeds
worst of all. **Falsified if** `defend_brique` (1.46) degrades as hard as
`fireteam` did, or if any `platoon` seed holds its report rate while `fireteam`
lost all of its. Registered now so the reading cannot be fitted afterwards.

- 02:30 — **the `fireteam` collapse is not a seed draw, and the record settles
  it at zero cost.** `fireteam` is single-seed in this campaign, so the obvious
  challenge to the headline was that 0.722 -> 0.000 is a bad draw rather than
  the observation removal. Its whole recorded lineage says otherwise:

  | run | seed | N | `closed_on_root_report_rate` | retasks/ep |
  |---|---|---|---|---|
  | `fireteam_v9` | 12 | 100 | 0.897 | 0.85 |
  | `fireteam_v10` | 12 | 20 | 1.000 | 1.50 |
  | `fireteam_v11` | 12 | 100 | 0.915 | 1.70 |
  | `fireteam_v12` | 12 | 100 | 0.915 | 1.70 |
  | `fireteam_v13_seed13` | 13 | 100 | 0.928 | 0.61 |
  | `fireteam_v15` | 12 | 100 | 0.722 | 2.10 |
  | **`fireteam_v16`** | 12 | 100 | **0.000** | **3.27** |

  Six prior runs, two seeds, several trees: the rate spans **0.722 to 1.000 and
  has never once approached zero**. Re-tasking spans 0.61-2.10 and v16 is 3.27,
  above every one of them. There is no bimodality here to hide behind — unlike
  `platoon`/`platoon_hard`/`patrol_brique`, whose reporting genuinely is
  bimodal. The v1.26 run is outside the entire historical range on both
  numbers, in the direction the mechanism predicts.

  This is the strongest cheap challenge available to the headline and it
  survives it. Still one scenario; the registered prediction is what the rest
  of the night tests.

- 02:57 — **the `fireteam_defend` non-result is real too, checked the same way.**
  Challenging only the finding that fits the story would be motivated
  reasoning, so its lineage got the identical treatment:

  | run | N | rootrep | retasks/ep | latency |
  |---|---|---|---|---|
  | `fireteam_defend_v20` | 100 | 1.000 | 0.050 | 6.219 |
  | `fireteam_defend_v21` | 20 | 1.000 | 0.300 | 3.341 |
  | `fireteam_defend_v22` | 100 | 1.000 | 0.120 | 3.245 |
  | `fireteam_defend_v23` | 100 | 1.000 | 0.120 | 3.245 |
  | `fireteam_defend_v24_seed13` | 100 | 0.990 | 0.120 | 4.127 |
  | `fireteam_defend_v26` | 100 | 0.970 | 0.160 | 2.027 |
  | **`fireteam_defend_v27`** | 20 | **1.000** | **0.000** | **1.538** |

  Reporting has never left 0.970-1.000 and v27 sits at the top of it.
  Re-tasking spans 0.05-0.52 and v27 is at/below the floor. Latency's history
  is 2.03-6.22 and v27 beats all of it. So this scenario is not merely
  "unbroken" — it is unchanged or better on every order-loop number, while
  `fireteam` left its historical range entirely on two of them.

  **Both sides of the contrast are now checked against the record**, which is
  what makes the registered prediction worth testing rather than a story fitted
  to two runs.

- 03:25 — **job 3 `squad_v34` landed and is INCONCLUSIVE, not a confirmation.**
  Declared in `seed_spread[squad]`. Read naively it looks like a confirmation:
  `closed_on_root_report_rate` 0.842 -> 0.000. It is not, for two reasons that
  the record supplies and the run does not.

  First, the comparator is `squad_v33_seed15` and `squad_v34` is **seed 12** —
  seed and observation move together, so the read is confounded. Second, and
  decisive: **`squad`'s reporting channel is bimodal and 0.000 is one of its
  normal modes, well before the removal.** `squad_v16` (seed 12, obs 351) reads
  0.000. So does `squad_v31_seed13`, and `squad_v10c` at seed 14. Re-tasking is
  equally uninformative: seed-12 history spans 1.29-11.11 and v34's 6.05 is
  mid-range.

  `squad` therefore cannot be read until its seed-matched partners land —
  `squad_v35_seed13`/`v36_seed14`/`v37_seed15` are jobs 19-21, after morning.

### Which scenarios can be read at all, decided from the record

Prompted by the above, every scenario's pre-removal reporting history:

| scenario | runs | min | max | zeros | verdict |
|---|---|---|---|---|---|
| `fireteam` | 6 | 0.722 | 1.000 | 0 | **stable — readable** |
| `fireteam_defend` | 6 | 0.970 | 1.000 | 0 | **stable — readable** |
| `squad_recon` | 6 | 0.900 | 1.000 | 0 | **stable — readable** |
| `squad_screen` | 6 | 0.710 | 0.980 | 0 | **stable — readable** |
| `defend_brique` | 5 | 0.980 | 1.000 | 0 | **stable — readable** |
| `squad` | 21 | 0.000 | 0.959 | 6 | bimodal — needs seed match |
| `platoon` | 10 | 0.000 | 0.940 | 3 | bimodal — needs seed match |
| `patrol_brique` | 24 | 0.000 | 0.949 | 14 | bimodal — needs seed match |
| `platoon_hard` | 5 | 0.000 | 0.011 | 5 | floor — cannot fall further |

**This narrows tonight's clean test set to five scenarios, three of which are
jobs 4, 5 and 6** (`squad_recon`, `squad_screen`, `defend_brique`) and land
before morning. The registered prediction is unchanged; what changes is which
NUMBER can test it where. `platoon` and `platoon_hard` — the two highest
re-tasking scenarios, and the prediction's strongest cases — must be read on
`retasks_per_episode` and `obedience_latency_mean`, which are continuous, and
NOT on a reporting rate that is bimodal in one and already on the floor in the
other. Recorded before jobs 4-6 land, so this is a scope statement and not a
goalpost moved after the fact.
