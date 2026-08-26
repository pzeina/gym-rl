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
