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

*Staged orders are not order events* (issue #15). While an A5-2 order is
pending — `AT MY COMMAND` before its `EXECUTE`, `AT T PLUS n` before its tick
— the recipient is obeying by **not** executing, and the environment scores it
as `HOLD` at the staging spot, i.e. exactly where it already stands. Its
compliance is therefore positive from the tick the order lands, and counting
that tick as an order event made every staged order resolve at latency **0**:
an identical un-staged ADVANCE whose recipient never moved was censored, while
the staged one read "obeyed instantly". Release restamps `step_assigned`, so
the order books its real event at the release tick regardless — the staged
tick was a second, free, zero-latency copy of the same order, and the mean
fell toward 0 in proportion to how much a policy staged. Pending ticks are
skipped, and staging is measured separately below.

### A5-2 staging (issue #15)

A *staging span* runs from the tick a pending order lands to the tick it
becomes effective (release restamps `step_assigned` to that tick, which is how
a span is recognised as released).

* `orders_staged` — spans opened.
* `staged_released` — spans that reached their `EXECUTE` / due tick.
* `staged_abandoned` — spans that did not: re-tasked or killed while staged,
  or still staged when the episode ended. This is the one that reads as a
  fault — an order transmitted, a recipient held in place, and no execution
  ever ordered. An outside tap measured 61 of one checkpoint's 130 staged
  orders in that state.
* `staging_gap_mean` — mean steps from landing to release, over released
  spans only; `null` when nothing was released.

*Edge case*: a staged order superseded **at its release tick** by a fresh
immediate order of the same task is indistinguishable from a release in the
trace (same mission, same restamped `since`). It is a coincidence of one tick
and one task, and it can only under-count abandonment.

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

#### Reading it: containment and the ordered-task mix (issue #14)

The preference rate alone is **not** a command-quality score. `preferred` is
`allowed[0]`, so every order that is not the head of the issuer's derivation
list scores the same, whether it is a legal alternative or a breach. A5 put
`ADVANCE` into `DOCTRINE[DEFEND]`, `[SEIZE]` and `[RECON]` as a maneuver leg,
and `DOCTRINE[DEFEND][0]` is `DEFEND` — so a policy that orders ADVANCE
wholesale reads ~0.00 preference while violating doctrine zero times. Across
this repo's own corpora the rate tracks ADVANCE adoption almost exactly: the
pre-ADVANCE defends sit at 0.31 / 0.21, and every corpus that adopted ADVANCE
(69–99% of its orders) sits at 0.17 down to 0.0016.

Two companions make that readable, both derived from the same tier split
(`preferred` / `allowed` / `violating` / `underivable` — the last being an
issuer with no mission to derive from, which the mask forbids in play):

* **`doctrine_allowed_rate`** ("doctrine containment") — `(preferred +
  allowed) / derivable`. This is the one that answers *is the mask leaking*;
  it should be 1.000 under `full`, and it is the arm that moves under B3
  `nomask`. A low preference rate with containment at 1.000 is catalog
  adoption, not indiscipline.
* **`orders_by_task`** — per ordered mission, the same four counts. The
  digest prints it as `TASK share/preference`, e.g. `ADVANCE 0.96/0.00`:
  96% of orders were ADVANCE and none of them was the preferred derivation.

Quote the preference rate against the mix, never on its own — the two
pre-ADVANCE corpora are not comparable to the post-A5 ones on this number.

#### The mix needs its own denominator: order availability (issue #16)

`orders_by_task` shares are **availability-confounded**. The order mask does
not offer the tasks in equal numbers, so a share answers "how often was this
ordered", which is two findings in one face: *the policy declined this task*
and *this task was barely on the menu*.

The inequality is structural, not incidental. SUPPORT is unit-targeted
(`ORDER_S{i}_SUPPORT_U{j}`), so it needs a **second** living subordinate slot
and contributes one entry per supported slot; OBSERVE is objective-targeted
and contributes one entry per objective on the map. SCREEN cannot derive
SUPPORT at all, so a SCREEN-rooted scenario offers it zero entries — for any
policy, forever.

`order_availability` is the matched control. For every order the policy
actually issued, it records the share of the **issuer's own admissible order
vocabulary** that belonged to each task at that exact state — read off the
mask recorded in the previous step (`order_opts`, i.e. the observation the
issuer acted on), never re-derived. Pooled over the run and divided by
`orders_matched`, it is precisely the expected task mix of a masked-random
policy making the same set of order decisions.

* **`order_selection_lift(agg)`** = `share / availability` per task. **1.00 is
  the masked-random floor**: no preference. `> 1` is a task chosen above what
  was on the menu, `< 1` a task whose opportunities were declined. `None` for
  a task never offered — no opportunity, so no selection to measure.
* The digest prints `TASK share/availability (xLIFT)`, ranked by
  *availability* rather than by orders issued, because the reading that
  matters is the task the mask offered and the policy did not take.
* **`orders_matched`** below `orders_issued` means orders were seen that the
  issuer's own mask never offered — injected or replayed, not selected, and
  so excluded from the control.

Entries are counted, not tasks: a uniform-over-legal policy picks an entry,
which is what makes the floor exact. A5-3 stance orders (`FORMATION X`) carry
no mission and are excluded from both sides.

**Why this is load-bearing.** The confound does not point one way, so the
uncorrected reading is wrong in *opposite directions* by scenario family.
Measured on masked-random (12–20 episodes, seeds 500+):

| scenario | OBSERVE avail | SUPPORT avail | raw ratio a uncorrected reading would report |
|---|---|---|---|
| `squad` | 0.23 | 0.08 | OBSERVE 2.9x — **entirely** the mask |
| `squad_screen` | 0.42 | 0.00 | infinite — there is no policy in it |
| `fireteam_defend` | 0.11 | 0.21 | SUPPORT 1.9x — **against** the trained direction |

So `fireteam_defend_v8`'s widely-quoted `OBSERVE 0.10 / SUPPORT 0.01` is not
the OBSERVE preference it looks like: corrected, it is **OBSERVE x0.92**
(the floor — no preference at all) against **SUPPORT x0.04** (96% of the
SUPPORT opportunities it held, declined). The finding is SUPPORT avoidance,
and the raw ratio understated it by more than 2x while misnaming its cause.

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

#### Who issued them, and did it pay: `orders_by_rank` / `order_pay_by_rank` (refs #52)

`orders_by_task` is team-wide, so a commander that orders from the objective
and one that orders from 40 units behind it produce the same row.
`orders_by_rank` (`{issuer effective rank: {preferred, allowed, violating,
underivable}}`, keyed on the issuer's EFFECTIVE rank so a promoted TL holding
the squad counts as the SL it is acting as) and `order_pay_by_rank`
(`{issuer effective rank: {fresh, churn, retask, pay}}`, read from the
environment's own order-payment ledger — see B5's fresh/churn/retask split
below) answer that: how much a rank commanded, and by `pay`, whether it paid.

**Both are raw per-run totals, and a raw total is not comparable across an
arm that also changes episode length** (refs #53). On the mute-vs-reporting
root comparison these two were built for, the mute root *survives* where its
reporting counterpart dies — so its episodes time out instead of ending
early, and it racks up ~3.4x the root's raw order volume while commanding at
the SAME rate. `rank_alive_steps` — `{effective rank: (soldier, step) pairs
alive}`, counted the same way `done_admissible` counts opportunity (every
recorded state but the last) — is the denominator that removes it:
`sum(orders_by_rank[rank].values()) / rank_alive_steps[rank]` is orders per
step alive, comparable across arms whose episodes run different lengths; the
raw sum on its own is not. `format_behavior_table` prints the rate beside the
raw count and its denominator for exactly this reason (`SL 12/40a (0.30/step,
9 pref)`). Episode count alone is not a sufficient normaliser here — episode
*length* is itself part of the treatment on this comparison.

### False-COMPLETE rate

Rejected DONE / total DONE transmitted. Every MISSION COMPLETE claim is
answered on the net (`DONE_CONFIRM` / `DONE_REJECT`, issue #4), so the rate
is exactly the share of claims the umpire judged false — and, for the same
reason, **realised acceptance is exactly `1 - false_complete_rate`** and is
deliberately not carried under a second name. *Edge case*: no DONE → `null`.

`false_complete_rate_root` is the same ratio over the claims made by whoever
held the root at the moment of the claim (issue #23). It is not derivable from
the pooled rate: a fireteam's riflemen can carry the pooled number on their
own, and the root's channel is the one that closes an operation.

### COMPLETE claims per claiming episode (issue #23)

DONE transmitted / episodes in which anything was claimed —
`done_claims_per_claiming_episode`, with `done_claims_per_claiming_episode_root`
over the root's own claims. **1.00 is a policy filing a report**; a large number
is a policy spamming a channel, and the two can share a rejection rate exactly.
That is the point: `false_complete_rate` is a ratio, so 13-claims-13-accepted
and 128-claims-27-accepted are the same shape of object and opposite
behaviours. Measured (`ckpt_best`, N=20, seed 123): `fireteam_v8` files 14.40
claims per claiming episode in 20 of 20 episodes; `squad_screen_v5` files 3.00
in 14 of 20 — a 5× separation on policies whose pooled rejection ratios (0.84,
0.69) read as the same failure.

The denominator is *claiming* episodes, not episodes: silence in nine episodes
must not read as restraint in the tenth. *Edge case*: nobody claimed → `null`.

### COMPLETE claim rate (issue #13)

DONE transmitted / `done_admissible`, where `done_admissible` counts the
agent-steps at which MISSION COMPLETE was admissible — read off
`cohort.env.actions.is_done_admissible`, the same predicate the action mask
admits on, so the metric and the mask cannot drift. `done_admissible_root` is
the root's share of it.

This is the denominator `done_reports` never had. Zero DONE reports is the
same number for two opposite findings, and only the denominator separates
them:

* `done_admissible == 0` — **absence**: the channel was shut and no price was
  ever consulted. This was true of every DEFEND-rooted run before `cc07199`,
  and the silence read as a taught behaviour for a whole generation.
* `done_admissible >> 0`, `done_reports == 0` — **suppression**: the act was
  offered and declined, which is a statement about `done_false`, not about
  reachability. Measured on `squad_v6` (`scripts/done_probe.py`, 10 episodes
  from seed 500): 11,528 admissible agent-steps, 0 claims, against an oracle
  regime taking 57 confirmed completions on the same seeds.

*Edge case*: no admissible step → `null`, never `0.0` — an undefined rate and
a declined opportunity are the distinction the metric exists to preserve.

### Closing an operation: timing vs volume (issue #35)

COMMAND ends **every** operation with ENDEX since v1.19; what the root's own
report buys is closing the window *early*, plus `root_done_bonus` (+3.0). Four
numbers describe that route, and the first one alone is a trap.

Before v1.19 this whole block only existed on the defend family, because a
completable root got no ENDEX and every rate below has ENDEXes-sent for a
denominator. It now reads on all eight scenarios, and that is the point: the
behaviour `successes_announced` used to carry — did the root report, or did HQ
have to close for it — moved here when the announcement itself became
guaranteed.

* **`closed_on_root_report_rate`** = `endex_on_root_report` / `endex_sent` —
  of the operations COMMAND closed, the share closed by the root's own report
  rather than by the grace window expiring. *Edge case*: no ENDEX → `null`,
  which since v1.19 means only "no successful operation" — every win gets an
  ENDEX, so the denominator is the win count. (Before v1.19 a completable root
  closed with MISSION COMPLETE and no ENDEX at all, and reading that as "never
  reported" is the denominator confusion `false_complete_rate` fell into on
  `fireteam_defend_v12`.)
* **`root_sitreps_per_episode`** — SITREPs transmitted by whoever held the
  root at that step (read per step, because succession moves the root).
* **`closes_per_root_sitrep`** = `endex_on_root_report` / `root_sitreps` —
  closes bought per report emitted. **High means the reports were timed; low
  means the close was bought with volume.**
* **`closed_on_cadence_report_rate`** = `endex_on_cadence_report` /
  `endex_sent` — same denominator as the first rate, but the numerator counts
  only closes made by a report that was itself **cadence-compliant**: at least
  `sitrep_interval` steps after that soldier's previous SITREP, i.e. exactly
  the report the environment pays `sitrep_fresh` for instead of `sitrep_spam`.
  This is the cell that answers *is the policy timing anything at all*,
  because it excludes reports bought as lottery tickets on the bonus.
  *Edge case*: no ENDEX → `null`. A close made by a confirmed MISSION COMPLETE
  instead of a SITREP counts in the denominator and not the numerator — the
  question is about the SITREP channel, and a claim-route close did not use it.

`root_sitreps_off_cadence` and `root_sitrep_off_cadence_share` carry the
density beside the rate, scored against the `sitrep_interval` the episode was
actually played under (`ScenarioSpec.sitrep_cadence` where the reporting
doctrine is on, `RewardConfig.sitrep_interval` otherwise; the recorder writes
it into the trace, with the clock origin, so freshness is recomputed exactly
as `CohortEnv._apply_action` prices it). Freshness is tracked per *soldier*,
over every SITREP it sent — the environment's clock is per soldier, not per
role, so a successor's first report is fresh however loud its predecessor was.

**Why the first rate needed the other three** (issue #35, measured N=100 seed
123 at the v1.17 cut, `ckpt_latest`): the two policies trained with the root's
claim masked shut close on their own report essentially always, and read as a
large improvement in reporting discipline. They are not.

| cell | closed on root's report | closes / root SITREP | closed on a cadence report | root SITREPs / ep (off-cadence) |
|---|---|---|---|---|
| `defend_brique_v13` (control) | 0.79 | 0.130 | **0.38** | 6.07 (69%) |
| `defend_brique_v14` | **1.00** | 0.033 | **0.00** | 30.30 (97%) |
| `fireteam_defend_v18` (control) | 0.53 | 0.063 | **0.28** | 8.22 (73%) |
| `fireteam_defend_v19` | **1.00** | 0.032 | **0.08** | 30.60 (96%) |

One report every ~3.2 steps against an interval of 25 makes a close near
certain, so the headline rate saturates at 1.00 while the share of closes that
a cadence-compliant report accounts for *falls* — 0.38 → 0.00 and 0.28 → 0.08.
The same behavioural change reads as an improvement on one number and a
regression on the other, and only the pair says which it was.

### Did the win get announced at all? (issues #31, #38)

`successes_announced` / `successes_announced_rate` — of the operations that
**succeeded**, how many said so on the net: COMMAND's ENDEX **or** the root's
own confirmed MISSION COMPLETE, deliberately either/or. *Edge case*: no success
→ `null`.

**Since v1.19 this is complete by construction and that is deliberate.** It used
to be gated on the root's mission, which made it a protocol act on a defence and
an agent behaviour everywhere else — 391/391 on the defend family, 91–98% on the
squad ones, 49/80 on `fireteam_v8`, and 0/100 on `platoon_v5` and 0/99 on
`patrol_brique_v5`, two scenarios that succeed on essentially every episode and
never once say so. A metric pinned at 1.00 is worth less as a *measurement* and
far more as a *guarantee*: `scripts/baseline.py` fails any fleet where a win went
unannounced, which is now a broken protocol rather than a shy policy. The
behaviour it used to carry is `closed_on_root_report_rate`, above.

**A zero on it had three causes, and the integer could not say which** (issue
#38, from the assurance layer, which reads the same zeros off the radio). Kept
because it is how the pre-v1.19 corpora must be read, and because it is the
diagnosis to reach for if the guarantee ever regresses. The
announcement line therefore renders the root's own claim channel beside it
(`format_root_claim_shape`), which is the #13 argument above one level up:

* **channel shut** — `done_admissible_root == 0`. No price was ever consulted;
  the v1.17 defend family, by mask and by design.
* **declined** — admissible and never used. `patrol_brique_v5`/final: 0 of 99
  announced, **0 claims in 7772 admissible agent-steps**.
* **refused** — claimed and rejected by the umpire. `platoon_v5`/final: 0 of 100
  announced, **5 claims in 5 episodes, 5 refusals**.

The first two are properties of the mask and of the policy; the third is
upstream of the announcement entirely, and a change to *who may announce* would
move the first two and not the third.

**Quote it at both checkpoints.** It is the least stable published column
measured here: `squad_v8` announces 0/97 at `ckpt_best` and 91/98 at
`ckpt_latest` (success 97 vs 98, p = 1.00; announcement p = 8.0e-48), and three
more runs swing 74–97 points across the same pair. The `|best − final| ≤ 5pt`
result from `publish_audit.py --validate` is measured on **success** and says
nothing here; `--validate` prints this axis under its own table to keep the two
apart.

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

### Clock expiry and traffic composition (issue #18)

The stall signature, measured from the transcript and the clock alone — no
reward, no success flag, no ground truth.

* **`timeout_rate`** — share of evaluated episodes that ended by running the
  step ceiling out. The environment scores `timeout` exactly when
  `max_steps` is reached with neither success nor defeat, so this *is*
  "pinned at `max_steps`"; the trace records `max_steps` so the mean length
  can be read next to its own ceiling.
* **`messages_per_episode`** — every message on the transcript, learned or
  automatic. The suite counted each channel it had a question about and
  never the total, so composition was not computable: "the net went quiet"
  and "the net changed hands" read the same.
* **`command_traffic_share`** — orders and EXECUTE releases over all
  messages. The OPORD is excluded: it is HQ's, once, and no policy chose it.
* **`voice_traffic_share`** — SYNC PROPOSE / GO over all messages. Voice is
  free by design (A5-4): uncharged and never net-arbitrated, so it is the
  one learned transmission a policy with nothing to say can emit for
  nothing.

`regression_gates(agg)` gates **`timeout_rate ≤ 0.5` for every root
mission** — running the clock out is a failure mode no scenario wants, and
it is not the same finding as a low success rate: it says *how* the episodes
were lost, and a cohort that rides out the clock is a different repair from
one that gets killed on the way in.

The bound is the middle of an empty band on the record (10 episodes per
checkpoint, seeds 500–509, every checkpoint that loads under the v1.10
spaces): healthy checkpoints run 0.0–0.2 (worst: `fireteam_defend_v9/best`
and `_v8/latest`, 2/10) and all three stalled ones sit at exactly 1.0
(`squad_recon_v6`, `squad_screen_v4`, `squad_screen_v5`, at `ckpt_latest`).

**The composition is reported and deliberately not gated.** Measured across
the same fleet it separates nothing: the healthy `fireteam_defend_v10/best`
(8/10 success) carries a command share of **0.026**, *below* the collapsed
`squad_recon_v6/latest` (0/10) at **0.022**, and `fireteam_v7/latest` scores
8/10 while issuing 1.5 orders per episode. Command share is scenario idiom —
a fireteam holding ground does not talk like a platoon assaulting. It reads
as a *within-scenario* contrast instead, and there it is stark:
`squad_screen_v4` carries 537 messages/episode at 15.5% command from
`ckpt_best` and 1326 at 0.6% from `ckpt_latest`. The net gets louder and
emptier at once.

Training runs log the same two facts per iteration —
`timeout_rate_rolling` (the window `success_rate_rolling` uses) and
`messages_per_agent_step` — so a stall is visible while the run is still
cheap to kill. Read `messages_per_agent_step` *next to* `tx_per_agent_step`:
tx charges by design, so through the `squad_screen_v4` flood it read 0.029
and the run was written up as "the whole radio goes quiet" when the net had
got 2.5× louder.

### The success axis: defeat-shaped collapse (issue #21)

Issue #21 pre-registered and **confirmed** a premise behind the survivor-scaled
defend terminal: "no defend scenario ever collapsed" is true when "collapse"
means the D4 stall (`timeout_rate` above, ≈ 30/30). It also found the premise
is shape-specific. The defend family's worst measured runs do not stall —
they are wiped, well before `max_steps`:

| corpus (`ckpt_best`) | success | defeat | timeout |
|---|---|---|---|
| `fireteam_defend_v6` | 14/30 | 12 | 4 |
| `fireteam_defend_v6b` | 1/30 | 27 | 2 |
| `fireteam_defend_v7` | 12/30 | 11 | 7 |
| `squad_screen_v7` (not defend; same shape) | 6/30 | 24 | 0 |

None of the four is within an order of magnitude of the stall signature
(≥ 28/30 timeout on record), so `timeout_rate` reads all four as healthy on
the clock. The repo's own composite gate happened to catch every one of them
anyway, but on `human_death_rate` — a wiped team's commander usually dies
with it — which is right about these four runs for a reason other than the
one it names, and has no axis of its own for "the team lost."

`regression_gates(agg)` closes that gap with a floor on `success_rate`,
gated **only once `timeout_rate` has already cleared its own ceiling** —
i.e. only once the run is known not to be stall-shaped. That ordering keeps
the two axes mutually exclusive in a report: a collapsed run reads as
**STALLED** (`timeout_rate` fails) or **WIPED** (`success_rate` fails), never
both, because the two shapes want opposite fixes. The bound, `0.5`, sits in
the empty band between the highest documented defeat-shaped corpus
(`fireteam_defend_v6`, 0.467) and the lowest healthy record on file
(`fireteam_defend_v11`, 0.74) — the same style of placement as
`TIMEOUT_RATE_CEILING`, and, not by design, the same value.

This gives the success axis independently of commander death — issue #21's
own closing question ("can a defeat-shaped collapse leave the commander
alive?") is not answered by this change and is not answerable from the
record on file; it remains open.

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
  policies actually choose, not a compliance failure. **Post-A5 the number
  is not comparable to those** — it collapses to 0.00–0.17 wherever the
  policy adopts ADVANCE, with containment intact; read it with the
  ordered-task mix (issue #14).
* `defend_brique_v1`'s precision 0.12 over 900 CONTACTs is the A4 dedup
  residual under siege: the band stays visible for hundreds of steps and
  gets re-reported far inside the refresh age. `patrol_brique_v2b`'s
  recall 0.06 is the other extreme — the B5 silent rush barely touches
  the net (4 CONTACTs in 30 episodes) because it barely touches the band.
