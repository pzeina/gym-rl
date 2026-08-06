# Behavioral metrics (B2)

Success rate says *whether* the cohort wins; the behavioral metrics suite
measures *how it behaves* while doing so — what "behaves like its rank" means,
made checkable. Implemented in `cohort/metrics.py`; motivation and status in
`ROADMAP.md` (item B2, and the issue-#9 finding that added the human-exposure
block).

## Running it

```bash
# every evaluation computes the suite by default over the same episodes:
.venv/bin/python -m cohort.training.evaluate runs/<run>/ckpt_best.pt --episodes 30 --seed 500
# -> printed table + runs/<run>/behavior.json     (--no-behavior to skip,
#    --behavior-out PATH to redirect)
```

`behavior.json` carries the run-level aggregate under `"metrics"` plus the
raw per-episode event counts under `"per_episode"`. The dashboard's Episode
sidebar shows the suite of the selected run (Behavior card), served from
`/api/behavior?run=<name>`.

Two cheap columns also land in every training run's `metrics.csv`, computed
per iteration over the episodes that finished in it: `human_death_rate`
(fraction whose human root died — the issue-#9 signal rolling success is
blind to) and `false_complete_rate` (rejected DONE / total DONE).

## How it is measured

A `TraceRecorder` rides along the evaluation episodes and records, per step:
every soldier's state (position, standing mission and its assignment step,
effective authority, living subordinates), its **per-step compliance score**
(recomputed exactly as the environment scores it), ground-truth visibility
(which enemies each soldier can see — the oracle side), every enemy's state,
and the step's radio messages. Recording reads the environment only — it
consumes no randomness, so a recorded episode is bit-identical to an
unrecorded one under the same seed (covered by test). The metric functions
are pure functions over this trace, each verified against constructed
mini-episodes with hand-known values (`tests/test_metrics.py`).

## Definitions

Throughout: an *order event* is a step `t0` at which an agent's standing
mission carries `step_assigned == t0` — this covers agent-issued orders, the
HQ OPORD at `t=0`, injected human orders, and re-issues (the environment
always restamps the mission). Metrics whose denominator never occurred are
`null`, never 0 — the counts are kept alongside so a `null` is always
explainable.

### Obedience latency

For each order event: the number of steps from `t0` to the first step (the
assignment step included) at which the recipient's per-step compliance score
for that mission is **> 0** (i.e. it is in position, or measurably moving
toward it). Reported as the mean over all order events of the run.

*Censoring*: if the mission is replaced or cleared, the agent dies, or the
episode ends before compliance, the event counts (`obedience_orders`) but
contributes no latency (`obedience_censored`). *Edge cases*: no orders ever
applied → `null`. OPORD latency is measured from `t=0`.

### Report precision / recall

Ground truth is the oracle-side visibility recorded per step.

* **Precision** = informative CONTACTs / all CONTACTs transmitted. A
  CONTACT's content is the sender's ground-truth visible enemy set at tick
  start (the state the policy acted on). It is *informative* when it carries
  an enemy absent from the replayed team picture (**new intel**) or one whose
  picture entry has aged ≥ `RewardConfig.contact_refresh_age` (**a
  legitimate refresh**); otherwise it is redundant noise. The picture replay
  mirrors the environment: entries expire after `KNOWLEDGE_TTL` steps and
  dead enemies drop off.
* **Recall** = enemy ids ever reported (the environment's own
  `reported_enemy_ids` bookkeeping) / enemy ids ever seen by any friendly
  (union with the reported set, so recall ≤ 1 even for sightings that only
  existed mid-tick).

*Edge cases*: no CONTACTs → precision `null`; no enemy ever seen → recall
`null`; a CONTACT whose content cannot be reconstructed (sender saw nothing
at either bracketing step — possible only in degenerate mid-tick races) is
excluded entirely.

### Doctrine-preference rate

Share of **agent-issued** orders whose ordered mission is the *preferred*
derivation — `DOCTRINE[own][0]` for the issuer's own mission at the moment
of transmission. Issuer missions are replayed in within-step message order
(an order counts as applied when the recipient's end-of-step mission matches
it with `step_assigned == t`), so a leader re-tasked earlier in the same
tick is judged under its new mission. HQ traffic (OPORD, injected orders) is
excluded: it is not an agent decision. Note that the action mask already
guarantees every issued order is doctrine-*allowed*; this measures how often
the policy picks the doctrine-*preferred* option among the legal ones.

*Edge cases*: no orders issued → `null`. An issuer observed mid-succession
(mission inherited the same tick it issued) is judged against its
pre-succession mission.

### Order volume and re-task economics (B5)

`orders_per_episode` is `orders_issued / episodes` (agent-issued ORDER
transmissions, as in the doctrine-preference denominator; HQ traffic
excluded).

The re-task counts are **not** reconstructed from the transcript: the
environment adjudicates every applied order that replaced a standing mission
(a *re-task* — fresh taskings of untasked subordinates and identical
reissues are not re-tasks) and the recorder copies its per-step event log
(`env.retask_events_last_step`) into the trace. Per run:

* **`retasks`** (+ `retasks_per_episode`) — all re-task events;
* **`retasks_priced`** / **`retasks_excepted`** — split by whether the
  rank-scaled re-task price was charged or waived under the
  tactical-picture carve-out (a CONTACT on the net since the standing
  order, a casualty in the issuer's element since, or the issuer's own
  mission changed since; a truthful DONE never appears here — it clears
  the mission, so the follow-up order is a fresh tasking);
* **`retask_rotations`** — re-tasks that changed the order's *anchor*
  (another objective, another supported unit…), as opposed to same-anchor
  mission-type changes;
* **`retasks_by_rank`** — `{issuer effective rank: {priced, excepted}}`,
  because the price scales with the issuer's authority: under the B5
  doctrine, priced re-tasks should approach zero and higher ranks should
  re-task strictly more rarely than lower ones.

### False-COMPLETE rate

Rejected DONE / total DONE transmitted. Every MISSION COMPLETE claim is
answered on the net (`DONE_CONFIRM` / `DONE_REJECT`, issue #4), so the rate
is exactly the share of claims the umpire judged false. *Edge case*: no DONE
→ `null`.

### Succession recovery time

A *leader death* is the death of an agent with living direct subordinates.
The *orphaned set* is those subordinates minus the successor (the agent
announcing `I AM ASSUMING COMMAND` for the fallen leader). The event
recovers at the first step at which every still-living orphan holds a
mission assigned **at or after** the death step — i.e. the new command has
re-tasked everyone the death orphaned; recovery time is that step minus the
death step. Dead orphans drop out of the requirement; an event whose orphan
set is empty (the only subordinate became the successor) recovers at 0.
Reported as the mean over recovered events; events not recovered by episode
end are censored (`succession_unrecovered`).

*Edge cases*: no leader deaths → `null`. Deaths of agents without living
subordinates are not events (there is no command to devolve).

### Subordinate coverage time

Over all steps ≥ 1: the share of *(living leader, step)* pairs — leaders
with effective authority > 0, a standing mission, and at least one living
direct subordinate, mirroring the environment's own coverage scoring — in
which **every** living direct subordinate holds a mission that step.
*Edge case*: no such pairs (e.g. every leader dead from step 1) → `null`.

### Human exposure (issue #9)

Measured over the steps the human root is alive; all `null` when the
scenario has no human.

* **`human_mean_enemy_dist`** — mean distance to the nearest living enemy
  (steps without a living enemy contribute nothing; `null` if none ever).
* **`human_mean_objective_dist`** — mean distance to the root objective
  (`null` without one).
* **`human_ring_entries`** — outside→inside transitions of the objective
  observation ring (radius `IN_POSITION_RADIUS[RECON]` = 7.0 around the
  root objective, where #9 measured the root's exposure); spawning inside
  counts as the first entry. Defense scenarios spawn the human on the
  objective, so their baseline is ≈ 1 by construction — the metric is
  comparable *within* a scenario, across checkpoints.
* **`human_death_rate`** — fraction of episodes in which the human dies.

These exist because #9 showed rolling success is blind to a policy
re-learning to walk the commander into the ring: checkpoint selection for
human-preservation claims must read these numbers, not the success curve.

### Fight disposition and the positional gate (issue #11)

*Where* the cohort fights, once the enemy is actually on it. The population
is the *(living soldier, step)* pairs **under threat** — a living enemy
within `threat_radius` (the scenario's own `CombatParams.weapon_range`,
recorded per trace). Conditioning on threat is the whole point: averaged
over an episode, an approach march and a prepared defense look alike.

* **`cover_occupancy_under_threat`** — share of threatened pairs spent on
  cover terrain (forest).
* **`mean_distance_from_objective_under_threat`** — mean distance from the
  root objective over those pairs (`null` without a root objective).
* **`threat_pairs`** — the denominator, so a `null` is always explainable.

`regression_gates(agg)` turns these into pass/fail verdicts **for DEFEND
roots only** — holding ground is the one root mission for which "fought here
rather than there" is correctness rather than style; an assault is *supposed*
to leave its start point. The bounds are `cover ≥ 0.40` and
`dist ≤ 5.0` cells, placed in the empty band between the two groups on the
record: `_v5` 24/30 at 0.793 / 2.90 and `defend_brique_v1` 27/30 at
0.956 / 1.99 on one side, `_v6` 14/30 at 0.496 / 3.46 and `_v7` 12/30 at
0.060 / 9.09 on the other. A gate whose metric was never measured reports
`passed: null` — unmeasured is not a pass.

This exists because success rate did not catch it, and neither did the
suspected cause: `fireteam_defend_v7` halved the root-death rate the ROADMAP
had blamed (26/30 → 14/30) and fired on essentially every threatened step
(p(fire | threatened) 0.005 → 1.000), yet success went 14/30 → 12/30, because
it had walked 9 cells off the position it was ordered to hold. Three million
steps bought that lesson. DEFEND runs therefore also log
`cover_under_threat` / `objective_dist_under_threat` per iteration in
`metrics.csv`, so the collapse is visible while the run is still cheap to
kill; both are blank on iterations with no firefight (never `0`, which would
read as "fought in the open on the objective") and on every non-DEFEND root,
which pays nothing for the scan.

### Aggregation

Event-level metrics pool events across the run's episodes (one latency mean
over all orders, rates over total counts); the human-exposure means average
per-episode values so long and short episodes weigh equally. Fight
disposition pools *pairs*, so a long firefight weighs more than a brief
brush — which is the intent, the question being where the fighting happened.

## Baseline (published checkpoints, N=30, seeds 500–529)

The reference baseline the assurance layer and future runs compare against —
sampled policy, the assurance-protocol seeds. Committed as
`runs/<run>/behavior.json` per run. `—` = `null` (the denominator never
occurred; e.g. DEFEND holders cannot claim MISSION COMPLETE, so defense runs
have no false-COMPLETE rate, and runs without leader deaths have no
succession events). Refreshed for the B5 campaign: the fireteam / squad /
patrol columns are the B5-retrained checkpoints (`fireteam_v5b`,
`squad_v4b`, `patrol_brique_v2b` — trained under re-task pricing + tenure);
the other five checkpoints predate B5 but were re-swept so the re-task rows
exist for all eight (their episodes reproduce bit-identically — only reward
arithmetic changed — and every previously published metric is unchanged;
the pre-B5 fireteam/squad/patrol columns live in git history).

| metric | fireteam_v5b | squad_v4b | squad_recon_v4b | squad_screen_v2 | platoon_v2 | fireteam_defend_v5 | patrol_brique_v2b | defend_brique_v1 |
|---|---|---|---|---|---|---|---|---|
| success (N=30) | 0.77 | 0.83 | 0.93 | 1.00 | 1.00 | 0.80 | 0.97 | 0.90 |
| obedience latency (steps) | 1.8 | 2.8 | 2.6 | 2.2 | 3.5 | 0.2 | 1.8 | 0.0 |
| orders issued / ep | 7.4 | 17.5 | 86.2 | 50.4 | 150.7 | 24.1 | 6.0 | 23.4 |
| re-tasks / ep | 4.2 | 9.6 | 79.7 | 44.4 | 136.0 | 20.9 | 0.1 | 20.4 |
| priced re-tasks / ep | 2.0 | 5.0 | 27.0 | 15.2 | 40.2 | 20.4 | 0.0 | 14.8 |
| anchor rotations (30 eps) | 70 | 210 | 1710 | 1072 | 3489 | 557 | 1 | 549 |
| report precision | 0.45 | 0.17 | 0.43 | 0.12 | 0.14 | 1.00 | 1.00 | 0.12 |
| report recall | 0.77 | 0.96 | 0.77 | 0.97 | 0.99 | 0.03 | 0.06 | 0.90 |
| doctrine preference | 0.35 | 0.55 | 0.36 | 0.42 | 0.46 | 0.31 | 0.51 | 0.21 |
| false-COMPLETE rate | 0.82 | 0.72 | 0.84 | 0.84 | 0.66 | — | 0.51 | — |
| succession recovery (steps) | 6.6 | 6.5 | 10.0 | 2.5 | — | 3.3 | 11.0 | — |
| succession events | 10 | 29 | 7 | 4 | 1 | 3 | 3 | 0 |
| coverage time | 0.82 | 0.67 | 0.97 | 0.97 | 0.77 | 0.94 | 0.35 | 0.97 |
| human: dist to enemy | 21.9 | 19.9 | 24.6 | 23.5 | 22.6 | 13.0 | 13.5 | 11.6 |
| human: dist to objective | 21.9 | 24.4 | 22.3 | 20.5 | 33.9 | 3.4 | 23.4 | 2.4 |
| human: ring entries / ep | 0.07 | 3.03 | 0.60 | 0.20 | 0.07 | 1.10 | 0.27 | 1.10 |
| human: death rate | 0.27 | 0.27 | 0.07 | 0.07 | 0.00 | 0.10 | 0.03 | 0.00 |

Event volumes behind the rates (pooled over the 30 episodes): orders applied
251 / 555 / 2605 / 1539 / 4551 / 751 / 211 / 731; CONTACTs 100 / 758 / 167 /
477 / 1182 / 3 / 4 / 900; DONE claims 179 / 428 / 219 / 185 / 175 / 0 / 367 /
0 (column order as in the table).

What the baseline says (updated for B5, 2026-08-06):

* **Order volume divides the runs into two eras.** The B5-trained columns
  re-task 0.1–9.6 times per episode (patrol: one anchor rotation in 30
  episodes); the pre-B5 columns re-task 20–136 times per episode with
  hundreds-to-thousands of anchor rotations — the exact churn the B4
  probe diagnosed, now measurable per run. The per-rank split
  (`retasks_by_rank` in `behavior.json`) shows pre-B5 leaf TLs re-task
  almost exclusively under what would be the contact/casualty carve-out,
  while the mid-rank SL/PL stations carry most of the whim-priced churn.
* **Obedience is fast everywhere** — 0–3.5 steps from order to measurable
  compliance; ~0 on defenses because holders are typically already in
  position when (re)tasked.
* **False-COMPLETE is the weakest behavior**: where DONE is admissible,
  51–84% of claims are rejected as premature — completion *reporting*
  works (#3), completion *judgment* does not yet.
* **Human exposure separates checkpoints as intended (#9)**: recon/screen
  hold at 2/30 human deaths; the B5 assault retrains sit at 8/30
  (fireteam improved from 11/30; squad regressed from 7/30 and its ring
  entries rose to 3.0/ep — the SL leads from the front under the new
  economics; checkpoint selection for preservation claims must keep
  reading these rows, not the success curve).
* Doctrine preference is a *preference* rate among doctrine-legal orders
  (the mask makes illegal ones impossible); 0.2–0.55 is what trained
  policies actually choose, not a compliance failure.
* `defend_brique_v1`'s precision 0.12 over 900 CONTACTs is the A4 dedup
  residual under siege: the band stays visible for hundreds of steps and
  gets re-reported far inside the refresh age. `patrol_brique_v2b`'s
  recall 0.06 is the other extreme — the B5 silent rush barely touches
  the net (4 CONTACTs in 30 episodes) because it barely touches the band.
