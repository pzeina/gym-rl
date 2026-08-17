# Next cycles — what to do after baseline v1.19

Written 2026-08-11, immediately after the v1.19 baseline landed. `ROADMAP.md`'s
⟳ handoff block is the *state*; this is the *plan*. It is deliberately opinionated
about order and about cost, because the single most expensive mistake available
here is starting a breaking cycle without knowing it is one.

**The rule that shapes everything below.** Any change under `cohort/` that moves
action masks, observations or rewards **invalidates the eight-member baseline**,
because `runs/BASELINE.json` is sealed to a `cohort/` tree (`5f848fb6`) and
`scripts/baseline.py` fails the fleet the moment a member is not on it. A
retrain of the whole fleet is ~5 hours of wall-clock on an uncontended 10-core
box, three lanes in parallel, and **zero model tokens**. So the question is never
"can we afford this change" — it is "what else should ride in the same cycle".

---

## Addendum (2026-08-17) — standing riders for the next breaking cycle

The v1.20 succession cycle below has shipped, and v1.21 sealed the fleet on one
tree with its seed searches declared. What follows it is not yet chosen, but
whatever it is, these ride along:

> **LANDED (2026-08-18): the measurement campaign** —
> `scripts/campaigns/measurement_ablation_and_spread.jobs`, 13 runs on the
> sealed tree, N=100 both checkpoints, no member changed. The ablation at
> 3 seeds/arm: success full = nomask (p = 1.000), **flat beats full
> 296/300 vs 282/300 (p = 0.004)** and is the most robust arm (defeats/100:
> 0.3 vs full 1.7, nomask 5.0) — the outcome AND robustness halves of B3 are
> inverted on this tree; interpretability (doctrine-valid 1.00 vs 0.58) and
> commander survival (root death 0.13 vs 0.20/0.21) are what the hierarchy
> still buys. The spread: success deltas ≤ 0.03 on all six one-seed members,
> but `platoon_v9_seed13` is a MUTE ROOT and `squad_v29_seed14` makes squad
> 2-of-3 reporting — the bimodal set for the rider below is now
> patrol_brique, squad, platoon (squad_screen wobbling at 0.71). README
> ablation wording and any manifest `seed_spread` block are owner decisions,
> presented in ROADMAP's handoff.

- **§12.146 within-scenario seed-carry.** The v1.21 seed-carry claim was
  retracted (assurance #60): every cited pair compared a checkpoint with a
  bit-identical reproduction of itself, so the question is open — on the 20
  readable cross-tree pairs where a transition DID move the policy, the
  reporting label carries 10, chance either way. Do not manufacture a
  `cohort/` change to answer it: that is a design decision, and it confounds
  the answer with itself. Instead, when the next breaking cycle retrains the
  fleet, retrain the declared search seeds beside it — `patrol_brique`
  12/14/18/19 and `squad` 12/13, six runs riding a campaign already burning
  twelve — and read the reporting labels across that transition.
  `baseline.py`'s reproduction digests now disclose bit-identical pairs, so an
  immobile transition cannot silently re-produce the #60 mistake. If the
  labels carry, a seed search amortizes across cycles; if they do not, the
  manifest rule becomes: re-search bimodal scenarios every breaking cycle.

## v1.20 — the succession cycle (BREAKING, forced retrain) — SHIPPED

Four `cohort/` patches are written, tested and deliberately unapplied. They want
one window because each one alone costs a full fleet retrain.

### 1. `_fill_vacancy` does not re-point the superior's chart  ⟵ the reason for the cycle

`cohort/core/units.py::_fill_vacancy` sets `successor.leader_id =
vacated.leader_id` and never adds `successor.id` to that leader's
`subordinate_ids`. Its own recursive branch does exactly that; the top-level
call does not. The promoted branch therefore drops off the commander's chart:
unorderable (`env/actions.py` masks on `living_subordinates`), unobserved
(`env/observations.py`), absent from the trace's `subs`, and **never devolved to
when the commander falls in turn**.

    squad chart, structural sweep    4,080 / 5,040 death orders orphan a branch
                                     1,928 / 5,040 reach root() == None
    fireteam chart                   0 / 24 — exempt, its successors report to HQ
    realised, 660 episodes, v1.19    44 broken charts, 1 with no commander

The patch is one statement. It was validated **in memory** (injected as a pytest
plugin, nothing written under `cohort/`): the suite stays green, the sweep goes
to 0/5040, and the succession announcement then emits through the existing
formatter with no new vocabulary. Test is written and skipped:
`tests/test_succession.py::test_a_promoted_leader_is_on_the_chart_of_the_superior_it_reports_to`.

**Why it is breaking**: it restores an order edge, so masks change, so every
rollout changes. Expect real behavioural movement on the deep charts
(`platoon`, `squad`), none on `fireteam`.

**Owner decision inside this one**: `MessageKind.TAKING_COMMAND` currently
carries two acts (root appointment, and backfill) told apart only by prose. A
separate `ASSUMING_POSITION` kind would be cleaner to read with no parsing, but
it changes the net's vocabulary and how every committed trace reads. Recommended
as a question, not taken.

### 2. `parse_succession()` — the formatter's inverse (refs #40)

Additive, in `cohort/core/language.py`. Returns
`Succession(successor, replaced, assumes_command)`, with the two regexes lifted
verbatim from `probe._TAKING_RE` / `_FILLING_RE`, so the probe, the metrics, the
gallery and the assurance layer's net-only reconstruction stop keeping four
private copies of the same pattern. Test written and skipped in
`tests/test_language.py`.

### 3. `eval_commit` in the evaluation artifact (refs #39)

`cohort/training/evaluate.py` should stamp the commit it ran at, beside #28's
`checkpoint_sha256`. Today `publish_audit.era_gap` has to *derive* an artifact's
era from the commit that committed the file, which is an upper bound. Lift
`_git_commit()` out of `train.py:506` into `cohort/training/provenance.py` and
call it from both. Test written and skipped in `tests/test_publish_audit_era.py`.

### 4. The close-route denominator (found 2026-08-11)

v1.19 gives every scenario an ENDEX, so `closed_on_cadence_report_rate` and
`closes_per_root_sitrep` now read 0.000 and 11.0 on completable roots — their
numerator is SITREP-only while their denominator became "all operations".
Record the close ROUTE (`"sitrep"` / `"claim"`) beside `_root_close_step`, write
it into the trace, and scope both rates to SITREP-route closes so they read
`null` on a claim close. `closed_on_root_report_rate` is unaffected.

### Doing the cycle

1. Apply 1–4, each its own commit, suite green per commit.
2. **Re-measure the neutrality claim**: 1 and 4 are *not* rollout-neutral; say so
   rather than inheriting v1.19's measurement.
3. Retrain the fleet: `campaigns/baseline_v1_19_lane{A,B,C}.jobs` with the run
   names bumped, same seeds, same steps, no overrides.
4. `scripts/publish_when_ready.sh` detached, then `--seal --version v1.20`,
   `results_table.py --write`, `/boards`.
5. Read the deep charts against v1.19 at both checkpoints. A `platoon` or
   `squad` move is the expected signal; a `fireteam` move is a surprise worth
   chasing.

---

## The squad regression — diagnose before repricing

`squad_v10` 0.92 ± 0.05 and `squad_v10b` 0.88 ± 0.06 against the previous era's
0.98 and 0.97. Two seeds agree (p = 0.48); pooled p = 0.0031. Real, and the
weakest member of the fleet.

**What is known**: success against false-claim rate is r = −0.952 across the five
squad-family runs that claim at all, and the whole net got chattier (101 and 167
messages/episode against 77 and 83; root SITREPs 0.00 → 1.64 and 5.26).

**What is refuted**: claims crowding orders off the single-frequency net. The
weaker runs issue *more* orders (17.4, 17.6 vs 13.4) and carry more traffic.
Nothing is being starved.

**What is not known**: direction of causation. A policy that has not finished the
mission claims falsely more often for that reason alone.

**The experiment, in order:**

1. `scripts/done_probe.py squad_v10b` in all three regimes. Golden steps
   separate *pricing* (claiming was reachable and declined / over-taken) from
   *reachability* (the mask never allowed it). This is CLAUDE.md's rule and it
   has paid every time.
2. Only then, one arm at `done_false=-2.0` against `squad_v10b` as a named
   single-variable A/B on a frozen tree — `squad_v9` ran exactly that price,
   claimed **zero** times and scored 0.97.
3. If pricing is not the mechanism, the next suspect is the transmission budget
   itself: every transmission is an agent-step not spent moving, firing or
   taking cover, and this repo has fixed order-spam and stall-farming before.

**Read `squad_v12` on the claim ORDINAL, at both checkpoints** (refs #46, added
2026-08-11). Steps 1 and 2 above have since run — the probe said pricing, and
`squad_v11` at `done_false=-2.0` confirmed the mechanism and was rejected for
buying muteness. `squad_v12` now tests `root_done_bonus_first_claim_only`, which
is a rule about the FIRST claim versus later ones, and its pre-registered EV was
computed from a *pooled* precision that in fact splits 0.543 / 0.314 on
`squad_v10`'s final policy and inverts to 0.474 / 0.547 on its best.
`run_report.py` prints the split and the burn on every behavior block and files
them for `--vs`. The corrected arithmetic is in ROADMAP's 2026-08-11 (#46)
entry; the thing to check first when v12 lands is the honest half, because on
`squad_v10b`'s rates the flag prices the FIRST truthful report at **−1.713**.

**Do not** change `done_false` as part of v1.20. It would confound the succession
fix with a reward change across the same retrain — precisely the confound class
`run_report --vs` was extended to catch this morning.

---

## The ablation, at the strength it deserves

The v1.19 replication **inverted** B3's outcome half (success 0.92 / 0.98 / 1.00,
defeats 7.0 / 1.0 / 0.0) on **one seed per arm**, while the interpretability half
held (doctrine-valid 1.000 vs 0.592). The README now claims the second and not
the first.

Read it together with the squad regression: the full arm *is* the run that got
weaker. These are one observation, not two, and a v1.20 fleet may resolve both
at once.

**Then do it properly**: 3 seeds per arm, 9 runs, control arm = the squad
baseline member so the trio stays single-variable. `scripts/ablation_report.py`
already prints the axes and states its own strength; it will need the seed
aggregation the 2026-08-06 campaign had.

The original's own stated follow-up is still open too: **the platoon-depth
rerun**. B3 was measured on a squad, and its honest verdict said raw sample
efficiency against flat "does not hold at this scale". Depth is the whole
argument for a chain of command; nobody has measured it at three echelons.

---

## Measurement gaps worth closing

- **Every headline is one seed.** The fleet is eight scenarios × one seed. A
  second seed per scenario (8 runs, one lane, ~3h) would put a seed-spread
  bound on every published number, which is currently assumed rather than known.
  `squad_v10` vs `squad_v10b` is the only pair that has one — and it mattered.
- **Net-only reconstruction fidelity.** The assurance layer's #42 measured an
  18-point gap between its net-derived root pointer and ours. Our metrics read
  env ground truth, so the gap is a property of *their* reconstruction — but
  "the net alone explains the behaviour" is this project's founding promise, so
  their reconstruction failing IS our defect. `parse_succession()` (v1.20 item 2)
  is the first half; a `cohort/probe.py` round-trip test over committed corpora
  is the second.
- **The transparency probe has not run at v1.19.** B4's rule engine predicts
  every living agent's next-15-step destination from the transcript alone. It
  was measured at Box(137). It should be re-measured on the current fleet and,
  after v1.20, on a fleet whose succession traffic is complete.
- **`squad_screen`'s 24% root-death rate** is the second-highest in the fleet
  behind `squad`, on a scenario whose entire doctrine is *observe without
  engaging*. Weapons-tight is mask-enforced, so this is not a discipline
  failure — it is exposure. Worth an oracle pass.

---

## Repository hygiene, left deliberately undone

The same `.gitignore` depth bug that hid the fleet's final checkpoints (#44) also
swept in two classes of file nobody wants tracked, and they are still tracked:

    79   runs/**/tb/  tensorboard event files   (~66 MB)
    44   runs/**/.job.json                      host pids and absolute log paths

They were left alone on purpose. Untracking them shrinks a *checkout* and never
the pack — the bytes are in history either way — and `runs/` is not a tree to do
reversible-looking surgery on for tidiness. The new depth-independent rules shed
both for every FUTURE archived run, which is where the ongoing cost actually is.

If a clone size ever becomes a real constraint, the honest fix is a history
rewrite, and that is an owner decision with a force-push in it, not a cleanup.

## Scale and product

- **A company echelon.** `CO`/`XO` exist in the rank table with authority 6 and
  5, and no scenario uses them. The deepest chart trained is `platoon` at three
  echelons and 16 agents. A company scenario is the natural next depth, and it
  is the setting where the succession fix above matters most.
- **`cohort.play` is the deployment surface** and is exercised only by hand. A
  scripted end-to-end test — type an order, assert it parses, validates against
  rank authority, and lands as a mission — would make the human-in-the-loop path
  a tested interface rather than a demo.
- **Packaging** (`D2` in the backlog) becomes worth doing once v1.20 settles:
  `pip install cohort-marl` with `cohort-train` / `cohort-dashboard` entry
  points, and the baseline manifest as the thing a new user gets.

---

## Two habits that paid today, and should continue

**Run the thing and read the output.** Three defects in code written hours
earlier were caught that way and none by re-reading source: the publisher waiting
in manifest order across parallel lanes, the publish guard's `>` that should have
been `>=` (it clobbered a run's artifacts on its first smoke test), and the
gallery's standfirst promising a succession its episodes do not contain. The
baseline gate's own `str`-vs-`Path` bug reported "checkpoint does not load" for
all eight members and would have read as a spaces break.

**A gate that fails for the wrong reason teaches people to ignore it.** Two were
fixed today on that principle: the README-table check now stands down while a
campaign is in flight, and provenance is checked on the `cohort/` tree rather
than the commit sha — a tooling commit between two launches says nothing about
the runs.
