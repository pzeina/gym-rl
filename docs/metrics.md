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

### Aggregation

Event-level metrics pool events across the run's episodes (one latency mean
over all orders, rates over total counts); the human-exposure means average
per-episode values so long and short episodes weigh equally.

## Baseline (published checkpoints, N=30, seeds 500–529)

The reference baseline the assurance layer and future runs compare against —
sampled policy, the assurance-protocol seeds. Committed as
`runs/<run>/behavior.json` per run. `—` = `null` (the denominator never
occurred; e.g. DEFEND holders cannot claim MISSION COMPLETE, so defense runs
have no false-COMPLETE rate, and runs without leader deaths have no
succession events).

| metric | fireteam_v4d | squad_v3e | squad_recon_v4b | squad_screen_v2 | platoon_v2 | fireteam_defend_v5 | patrol_brique_v1 | defend_brique_v1 |
|---|---|---|---|---|---|---|---|---|
| success (N=30) | 0.80 | 0.87 | 0.93 | 1.00 | 1.00 | 0.80 | 1.00 | 0.90 |
| obedience latency (steps) | 2.7 | 2.4 | 2.6 | 2.2 | 3.5 | 0.2 | 2.2 | 0.0 |
| report precision | 0.43 | 0.23 | 0.43 | 0.12 | 0.14 | 1.00 | 0.68 | 0.12 |
| report recall | 0.59 | 0.95 | 0.77 | 0.97 | 0.99 | 0.03 | 0.52 | 0.90 |
| doctrine preference | 0.45 | 0.52 | 0.36 | 0.42 | 0.46 | 0.31 | 0.52 | 0.21 |
| false-COMPLETE rate | 0.76 | 0.81 | 0.84 | 0.84 | 0.66 | — | 0.53 | — |
| succession recovery (steps) | 3.6 | 1.9 | 10.0 | 2.5 | — | 3.3 | 2.0 | — |
| succession events | 12 | 12 | 7 | 4 | 1 | 3 | 3 | 0 |
| coverage time | 0.83 | 0.96 | 0.97 | 0.97 | 0.77 | 0.94 | 0.92 | 0.97 |
| human: dist to enemy | 20.3 | 21.1 | 24.6 | 23.5 | 22.6 | 13.0 | 15.5 | 11.6 |
| human: dist to objective | 20.1 | 24.1 | 22.3 | 20.5 | 33.9 | 3.4 | 24.2 | 2.4 |
| human: ring entries / ep | 0.57 | 0.10 | 0.60 | 0.20 | 0.07 | 1.10 | 0.17 | 1.10 |
| human: death rate | 0.37 | 0.23 | 0.07 | 0.07 | 0.00 | 0.10 | 0.07 | 0.00 |

Event volumes behind the rates (pooled over the 30 episodes): orders applied
754 / 2008 / 2605 / 1539 / 4551 / 751 / 1905 / 731; CONTACTs 70 / 480 / 167 /
477 / 1182 / 3 / 31 / 900; DONE claims 100 / 263 / 219 / 185 / 175 / 0 / 45 /
0 (column order as in the table). The platoon's single leader-death event is
the one censored recovery (unrecovered at episode end).

What the baseline says (first honest read, 2026-08-06):

* **Obedience is fast everywhere** — 0–3.5 steps from order to measurable
  compliance; ~0 on defenses because holders are typically already in
  position when (re)tasked.
* **False-COMPLETE is the weakest behavior**: where DONE is admissible,
  53–84% of claims are rejected as premature — completion *reporting*
  works (#3), completion *judgment* does not yet. The defenses show `—`
  by construction (DEFEND is a continuous posture, DONE is inadmissible).
* **Human exposure separates checkpoints as intended (#9)**: the retrained
  recon/screen sit at 2/30 human deaths (exactly the numbers that selected
  them), while the assault fireteam — trained before the #9 economics —
  loses its human in 11/30 episodes. Defense ring entries ≈ 1 and small
  objective distances are by construction (the human spawns on the
  objective).
* Doctrine preference is a *preference* rate among doctrine-legal orders
  (the mask makes illegal ones impossible); 0.2–0.5 is what trained
  policies actually choose, not a compliance failure.
* `defend_brique_v1`'s precision 0.12 over 900 CONTACTs is the A4 dedup
  residual under siege: the band stays visible for hundreds of steps and
  gets re-reported far inside the refresh age.
