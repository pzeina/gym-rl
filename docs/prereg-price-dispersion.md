# Pre-registration — the price-dispersion cycle

**Written 2026-08-26, before job 1.** Nothing in this document may be edited
once the first run of the cycle has launched; `tests/test_prereg_dispersion.py`
pins every threshold in it against `docs/prereg-price-dispersion.json`, and
`scripts/prereg_dispersion.py` is the only thing that scores it.

The mechanism is still the owner's to pick (`docs/next-cycles.md` → "The design
decision this cycle needs from the owner"). **The bar does not depend on which
one is picked** — it is written over `stacked_rate`, nearest-teammate distance
and success, all of which exist today. Registering it before the choice is
deliberate: a bar written after the mechanism is chosen is a bar written by
someone who already knows which way it would like the answer to go.

---

## The question

Both DEFEND-root members hold their objective as one pile and win every episode:

| | stacked | nearest teammate | spatially sound | success |
|---|---|---|---|---|
| `defend_brique_v19` | 0.976 | **0.205** | 0.024 | 100/100 |
| `fireteam_defend_v25` | 0.960 | **0.225** | 0.040 | 100/100 |

`STACK_RADIUS` is 1.5, so 0.21 is not "as spread as a small objective allows" —
a team on adjacent distinct cells reads ~1.0. **The piling is chosen, not
geometric**: that hypothesis was tested and refuted (`docs/next-cycles.md`).

`stacked_rate` became a MARKER on 2026-08-26, so the behaviour now ships and is
visible rather than blocked. This cycle asks whether making bunching *cost*
something changes it. **Does it disperse the element, does the cohort still win,
and if the number moves, did anything actually spread out?**

## What is read, and from where

The **FINAL policy at N=100** — `runs/<member>/behavior_final_n100.json` — for
every member of the post-cycle fleet. Never `ckpt_best` (a peak; that is what
`publish_audit.py` is for) and never N=20 (the incumbents are frozen at 100, and
comparing across N widens a candidate's CI until it overlaps whatever it is up
against). `scripts/prereg_dispersion.py` returns `None` rather than a reading for
either, and the suite pins that.

The incumbents are the nine v1.23 members, snapshotted into
`docs/prereg-price-dispersion.json` **before job 1**, with each member's
`checkpoint_sha256`. A test re-derives that snapshot from the committed
evaluations on every run of the suite, so the baseline cannot drift underneath
the comparison — the failure mode behind the retracted seed-carry claim
(assurance #60).

## The three conditions, per DEFEND member

| condition | bound | why this one |
|---|---|---|
| **clears the bar** | `stacked_rate < 0.70` | the marker bound the cycle is trying to clear |
| **moved at all** | `stacked_rate` falls by ≥ 0.10 | separates "the mechanism did nothing" from "it did something small" |
| **actually spread** | `mean_nearest_teammate_dist` rises by ≥ 0.50 cells | the denominator guard, below |
| **still winning** | one-sided Fisher `p > 0.05` vs the frozen incumbent | non-inferiority, directional |

### Why success is a Fisher test and not "inside the incumbent's CI"

The prose bar in `docs/next-cycles.md` said *success inside the incumbents' CI
(1.00 ± 0.00)*. That CI has zero width, so it refuses a single lost episode out
of a hundred. The exact test says one lost episode across two arms of 100 is a
coin flip:

| arm, vs a 100/100 incumbent | 99 | 98 | 97 | 96 | 95 | 94 |
|---|---|---|---|---|---|---|
| one-sided Fisher *p* | 0.5000 | 0.2487 | 0.1231 | **0.0606** | **0.0297** | 0.0145 |

So the bar tolerates **96/100** and convicts at **95/100**. That number is
arithmetic over margins that existed before any run launched, not a threshold
chosen once the arm landed.

### The denominator guard — the condition that is not in the prose bar

`stacked_rate` is the share of agent-steps with **≥ 2 living teammates** within
1.5. Any mechanism that prices bunching through casualties — AREA FIRE, shape
(3) and the recommended one — lowers that rate by killing teammates, with no
agent having learned anything. The rate improves because its denominator
vanished.

This project has already made exactly that mistake once and caught it: human
death rate "fell" under jamming, 0.450 → 0.050, and the finding was that the
human had stopped being brought forward at all. `human_in_action_rate` exists
because of it.

So a fall in `stacked_rate` **without** a matching rise in
`mean_nearest_teammate_dist` is not a win, and the rule names it
**DENOMINATOR** rather than letting it be written up as one.

## The fleet guard

The mechanism changes the environment for all nine scenarios, not just the
DEFEND pair. The other seven are read on success only, by the same one-sided
test, **Holm-corrected as one family at α = 0.05**.

Correcting matters in the direction that is easy to forget: this family is a
*guard*, so a false alarm does not invent an effect — it wrongly convicts the
cycle of having broken a scenario it did not break. Seven uncorrected tests at
α = 0.05 raise one better than 30% of the time. Concretely, a member landing
95/100 against a perfect incumbent reads *p* = 0.0297, which rejects on its own
and is retained across the family of seven. Both readings are pinned in the
suite.

## The ladder, and what may be called a ceiling

CEILING is a claim about the **maps**, not about a price. It may only be
declared at the top of the declared ladder:

> `burst_fraction` ∈ **0.5 → 0.75 → 1.0** (the rider's recommended opening
> value first). For a reward-term mechanism — shapes (1) or (2) — the owner
> declares the equivalent three-rung ladder before job 1 and it is recorded
> here.

`--top-of-ladder` is what lets `scripts/prereg_dispersion.py` say CEILING; below
the top, the same readings say NO EFFECT AT THIS PRICE or PARTIAL, both of which
mean *climb*.

## The verdicts

Checked in this order, so a read-out names the first thing that failed:

| verdict | what produced it |
|---|---|
| **INCOMPLETE** | a DEFEND member has no N=100 final evaluation. Not a result. |
| **NO EFFECT AT THIS PRICE** | `stacked_rate` fell < 0.10 in at least one member, below the top of the ladder |
| **CEILING** | the same, at the top of the ladder — **DEFEND cannot be held dispersed on these maps, and the marker is documentation rather than a defect** |
| **DENOMINATOR** | bunching fell in both, but the element is no further apart. Teammates died; nobody spread. |
| **PARTIAL** | real dispersion, still short of `stacked_rate < 0.70`. Climb the ladder. |
| **WALKS** | dispersion is real and clears the bar, but a scenario lost episodes it was winning. **A cohort that disperses and loses is a worse cohort.** |
| **SEPARATES** | all three conditions in both DEFEND members, and the fleet guard holds |

Every one of these is reachable — the suite drives each branch with synthetic
readings. A decision rule nobody has shown can reach its own verdicts is not a
pre-registration; `scripts/design_power.py` was written after a six-run campaign
turned out able to reject on 1 of its 64 possible outcomes.

## What is deliberately NOT scored

`closed_on_root_report_rate` and `human_in_action_rate`. Both are bimodal across
seeds — the record has `patrol_brique` at 0.43 over 14 runs, and three scenarios
landing at 0.750–1.000 or exactly 0.000 with nothing in between — so scoring the
cycle on them charges the mechanism for a seed draw. They are measured, printed
on every surface, and read separately.

Related: the cycle re-rolls every bimodal draw, because **a seed search does not
survive a tree change** (`patrol_brique` carried 2 of 4 seeds on the last
transition, one flipping 0.949 → 0.000). `patrol_brique`, `platoon` and
`platoon_hard` each need their declared searches re-run at 4 seeds — ~18 jobs,
not 11. Reading the reporting labels across that transition also answers the
open §12.146 seed-carry question for free, and that read is not part of this bar.

## How it is run

```bash
scripts/prereg_dispersion.py --freeze          # done 2026-08-26, committed
scripts/prereg_dispersion.py --setting burst_fraction=0.5 \
    --run fireteam_defend=<run> --run defend_brique=<run>
scripts/prereg_dispersion.py --manifest --setting burst_fraction=1.0 --top-of-ladder
```

Exit 0 on SEPARATES, 1 on everything else.

---

# Amendment 1 — 2026-08-26, before job 1

**The bar above is unchanged.** Thresholds, verdicts, incumbents and the fleet
guard all stand exactly as registered. What is withdrawn is the AREA FIRE
**mechanism and its ladder**, and one clause is **added** to stop a mechanism
that cannot reach the members from producing a false CEILING.

## What happened

The owner picked AREA FIRE (shape 3) on my recommendation. It was armed at
`burst_fraction = 0.5` — rung 1 — and, before launching 18 jobs, the two DEFEND
incumbents were dropped into the armed world **without retraining** to check
that the mechanism had teeth:

| N=100, final policy | success | stacked | nearest |
|---|---|---|---|
| `defend_brique_v19` | 1.000 → 0.980 | 0.976 → 0.965 | 0.205 → 0.207 |
| `fireteam_defend_v25` | 1.000 → 1.000 | 0.960 → 0.960 | 0.225 → 0.227 |

Trajectories moved, so the mechanic is not a structural no-op — but the pile
paid essentially nothing. **`scripts/burst_engagement_probe.py`** was written to
separate "priced too cheaply" from "never fires", because those have opposite
fixes: a higher fraction multiplies a splash that fires and cannot multiply one
that does not.

| N=20, `burst_fraction` 0.5 | enemy hits/ep | bursts/ep | splash dmg/ep | deaths/ep |
|---|---|---|---|---|
| `defend_brique_v19` | **0.6** | 0.5 | 22 | 0.20 |
| `fireteam_defend_v25` | **0.2** | 0.1 | 8 | 0.00 |
| `platoon_hard_v7_seed13` | **16.4** | 6.8 | 217 | 6.10 (from 4.55) |

## The finding

**The mechanism works, on the scenarios that are not the problem.** The two
members that pile up are the two that are essentially never shot at —
`defend_brique` takes 0.9 enemy hits per episode with the mechanic off,
`fireteam_defend` 0.5, against `platoon_hard`'s 17.6. Their cover occupancy
under threat is 0.996 and 0.999. **The pile is safe because it is in cover and
outguns what reaches it, not because bunching is unpriced** — so a price coupled
to incoming fire has no channel to charge it.

Climbing the ladder makes this worse rather than better. At `burst_fraction`
0.75 and 1.0 `defend_brique`'s enemy hits *fall* to 0.2, because AREA FIRE is
symmetric: heavier splash clears the attacking enemies faster, so the piled
cohort is shot at less. **The ladder does not increase the pressure on the pile;
past rung 1 it reduces it.**

## What this changes

1. **AREA FIRE's ladder (0.5 → 0.75 → 1.0) is withdrawn.** The mechanic is back
   to `burst_fraction = 0.0`, shipped OFF, with the measurement recorded at the
   field and pinned by `tests/test_burst_fire.py`. Re-arming is a decision.
2. **New clause — a CEILING requires a mechanism that can reach the members.**
   CEILING's registered meaning is *"DEFEND cannot be held dispersed on these
   maps"*. Had the ladder been run, it would have returned NO EFFECT at every
   rung and then CEILING at the top — and that sentence would have been
   published as a finding about the maps when the true cause was a price that
   never arrived. So: **before a CEILING may be written down, the cycle must show
   that the chosen mechanism actually charged the DEFEND pair** — for a
   casualty coupling, that the splash fired; for a reward term, that the priced
   quantity was non-zero on those episodes. A mechanism that cannot bill the
   member cannot license a claim about the member.
3. **The mechanism is open again**, and it is the owner's. Shape (1) per-step
   dispersion price and shape (2) threshold price both charge the behaviour
   directly rather than through the enemy, so both reach a cohort that is never
   shot at. Shape (2) remains the spec's own fallback.

## What is NOT retracted

AREA FIRE is not refuted as a mechanic — only as the lever for *this* bar. It
bites hard exactly where fire is heavy (`platoon_hard`: +34% deaths, 6.8 bursts
per episode), which is where the rider's original evidence came from: the flat
platoon arms win by piling at stacked 0.84–0.94. Whether it prices *that* pile
is untested — the archived flat arms are OBS_DIM 220 against a tree at 351 and
cannot be loaded, so answering it needs a retrain and it is not claimed here.

---

# Amendment 2 — 2026-08-26, before job 1: the mechanism and its ladder

**Owner-decided, replacing AREA FIRE: a THRESHOLD price (shape 2) at N = 1.**
`RewardConfig.bunching_penalty` charges each living agent, each step, for every
teammate inside `bunching_radius` **beyond the first**.

## Why N = 1, and why this is not a free parameter

`stacked_rate` — the marker the whole cycle is about — is defined as *the share
of agent-steps with **≥ 2** living teammates within `STACK_RADIUS`*. One free
teammate therefore makes **the first charged step exactly the first stacked
step**: the priced quantity and the measured quantity are the same quantity.
This is the same discipline that tied `CombatParams.burst_radius` to
`STACK_RADIUS`, and `tests/test_bunching_price.py` pins both halves.

It also answers the objection the spec raised against shape (1): a buddy pair
pays **nothing**, so the cohesion this project measures and wants is untaxed.
The pile is what pays.

## The ladder, sized from measurement rather than taste

Arming the price cannot move a trajectory of an already-trained policy — a
policy acts on observations, not on rewards — so what each member *would* pay
today is arithmetic over rollouts that already exist, at zero training cost.
`scripts/bunching_price_calibration.py`, N=20, per agent per episode, at a unit
price of −0.01 (the charge is exactly linear in the price):

| scenario | stacked | charge @ −0.01 | **@ −0.05 (rung 1)** | vs `success_team` = 60 |
|---|---|---|---|---|
| `fireteam_defend` | 0.960 | −1.95 | **−9.75** | 16% |
| `platoon` | 0.567 | −1.84 | −9.20 | 15% |
| `defend_brique` | 0.976 | −1.58 | **−7.90** | 13% |
| `platoon_hard` | 0.370 | −0.96 | −4.80 | 8% |
| `squad_recon` | 0.572 | −0.70 | −3.50 | 6% |
| `squad` | 0.291 | −0.34 | −1.70 | 3% |
| `squad_screen` | 0.234 | −0.20 | −1.00 | 2% |
| `patrol_brique` | 0.227 | −0.14 | −0.70 | 1% |
| `fireteam` | 0.167 | −0.11 | −0.55 | 1% |

**Declared ladder: `bunching_penalty` = −0.05 → −0.10 → −0.20.** Rung 3 puts the
worst piler at 65% of its team terminal, which is deliberately near the point
where holding stops being worth it — that is what WALKS is for, and it is the
strongest price this registration will ask for.

**Amendment 1's new clause is satisfied up front.** A CEILING requires showing
the mechanism actually charged the DEFEND pair; at rung 1 it bills them 16% and
13% of their team terminal, measured above, before any job runs. This is exactly
what AREA FIRE could not show.

## Declared risks, so they are not discovered as surprises

- **`platoon` and `squad_recon` are exposed too** (15% and 6% at rung 1) — they
  sit mid-band at 0.567 and 0.572 stacked. They are in the **fleet guard**, not
  the primary pair, so a success loss there reads WALKS. That is the correct
  reading and it is registered here rather than argued afterwards.
- **Over-dispersal is a real hazard and is NOT in the bar.** An element could
  answer the price by scattering past the point where teammates can see each
  other. That would show as `no_close_teammate` rising while
  `spatially_sound_rate` fails to improve — `spatially_sound` is `close AND seen
  AND NOT crowded`, so genuine dispersion raises it and scattering does not.
  Both are measured and will be reported beside the verdict. They are **not**
  promoted to gates: that is a decision, and the owner has made this project's
  gating calls.
- **The term is global, not per-scenario**, for the same reason AREA FIRE would
  have been: a price that exists in `fireteam` but not in `squad` is not one
  environment, and the fleet ships as one system.
