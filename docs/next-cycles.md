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
> 296/300 vs 282/300 (p = 0.004)** and holds the fewest defeats (1/300 vs
> full 5/300, p = 0.22 — power only against nomask 15/300, p = 0.0004) — the
> outcome AND robustness halves of B3 are inverted on this tree;
> interpretability (doctrine-valid 1.00 vs 0.58) is what the hierarchy still
> buys. (Commander survival — root death 0.13 vs 0.20/0.21 — was promoted
> here too and is retracted per assurance #64: the separation is seed 14
> alone, the #63 re-executed policy, and flips to p = 0.56 at ckpt_best.) The spread: success deltas ≤ 0.03 on all six one-seed members,
> but `platoon_v9_seed13` is a MUTE ROOT and `squad_v29_seed14` fails the
> root-report gate on both checkpoints (0.052 / 0.000 — and is a bit-identical
> re-derivation of archived `squad_v10c`, assurance #63), making squad 2-of-3
> reporting on the sealed tree, 2-of-4 expected counting archived
> `squad_v20_seed15` — the bimodal set for the rider below is now
> patrol_brique, squad, platoon (squad_screen wobbling at 0.71). Both owner
> decisions are closed (2026-08-18): the README ablation wording is applied,
> and the manifest now carries a `seed_spread` block — every other
> same-config draw the record holds, deduped by model-tensor digest, audited
> for completeness, rendered beside the declared search as counts with
> cross-tree draws annotated. Campaign queues now pre-flight each job's
> config against the record (`scripts/campaign_preflight.py`) so a queued
> config the archive already answered is refused before it burns 3M steps.

- **RIDER (2026-08-21, owner-decided): the dispersion pair on branch
  `dispersion-mechanic`** — merge into the acoustics breaking cycle BEFORE its
  fleet retrain launches; two commits, pytest+ruff green (`3d9adf7`,
  `358ee13`). (1) The bunching gate: `stacked_rate <= 0.70`, unconditional —
  `fireteam_defend_v23` wins DEFEND at 1.00 success while stacked on 0.940 of
  its living-agent-steps (nn 0.39 cells), the exploit the gate encodes; the
  fleet's healthy band is 0.18–0.57 (2026-08-21 provenance sweep,
  `scripts/spatial_probe_provenance.py`). (2) AREA FIRE: a hit sprays
  `int(damage * burst_fraction)` onto every other living unit of the struck
  side within `burst_radius` (1.5 = the metric's own radius, tied by test).
  Shipped OFF (`burst_fraction=0.0`, byte-identical scenarios, no RNG draws,
  no obs change); which scenarios turn it on — recommended opening value 0.5 —
  is the cycle's decision, made per scenario in `config.py`. It lands on a
  branch and not `multi-agent-dev` because the matched voice campaign froze
  `cohort/` while its queue feeds. DISCLOSE with that campaign's analysis: its
  job 1 (`squad_ctrl_v1_seed12`, tree `0f37e6a`) and jobs 2+ (`1f6d2cd`)
  differ by one `cohort/metrics.py` measurement-only commit — env dynamics
  byte-identical, but the tree hashes are two, and the record should say why.
  EVIDENCE for the mechanic (2026-08-21 platoon-depth spatial sweep, N=20/arm
  at provenance): the flat arms win by PILING — stacked 0.840/0.944/0.890
  across the three seeds (nn 0.45–0.77 cells, 16 agents in a blob, success
  1.00), versus 0.30–0.57 for every hierarchy arm (masked and nomask); 3-vs-6
  complete separation, exact p ≈ 0.012. All three flat arms FAIL the new
  stacked gate; all six hierarchy arms pass. So B3's "flat beats full on
  success" rests on a tactic AREA FIRE is built to price — rerunning the
  depth ablation with the mechanic on is the measurement that would settle
  whether the flat advantage survives dispersion being real.

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

## The platoon-depth ablation cycle (chosen by the owner, 2026-08-18) — LANDED 2026-08-18

**Landed as planned, verdict in the README beside the squad table.** The
neutrality re-derivation held (`platoon_v10_seed12` ≡ `platoon_v8`,
bit-for-bit, both checkpoints), all eight runs converged, N=100 on both
checkpoints, draws declared, BASELINE OK. Of the three honest outcomes named
below, the first: flat wins again — every outcome axis saturates (299–300/300
per arm, zero defeats in 900 episodes), so the squad-depth cost is
unmeasurable here and the hierarchy's value case narrows to interpretability
at every measured depth. The follow-up question this cycle leaves is what
scenario or budget gives the deepest chart outcome power at all.

The README now ends its ablation section with the open question: does the
hierarchy's measured ~5-point cost against a flat team shrink, vanish or
invert at three echelons? Depth is the whole argument for a chain of command;
the deepest chart trained is `platoon` (PL + PSG + 2 squads, 16 agents), and
B3 has only ever been measured on a squad. This cycle measures it.

### The `cohort/` change, and what it does NOT break

Two registrations in `cohort/config.py`, mechanical mirrors of the squad arms
(`SCENARIOS["squad_nomask"]` / `["squad_flat"]` at line ~450): `platoon_nomask`
and `platoon_flat` via `replace(SCENARIOS["platoon"], name=..., ablation=...)`.
Identical geometry, OpFor, rewards, spaces and step budget — only the
`ablation` field differs, exactly as decided for squad. Any deviation from the
exact mirror is a design decision and goes back to the owner.

**This is NOT a fleet-breaking cycle, and the plan must not pretend it is.**
The addition is purely additive: no existing scenario's masks, observations,
rewards or RNG stream moves, so the sealed v1.21 members stay on their
recorded tree and `baseline.py` keeps passing. Two consequences, both from
the #60/#63 lessons:

1. **The additive claim is measured, not assumed.** The campaign's first job
   re-derives the platoon full arm at seed 12 on the new tree (`FORCE=1` past
   the pre-flight, which will rightly call it a duplicate). If its tensors
   digest identical to `platoon_v8`, the transition is proven neutral and the
   existing `platoon_v8` / `platoon_v9_seed13` stand as the full arm's seeds
   12/13, disclosed as the same draws. If it does NOT reproduce, stop the
   campaign: the additive claim was wrong, the cycle is breaking after all,
   and that is a finding to diagnose before anything else trains.
2. **The §12.146 rider does NOT ride here and stays open.** The rider needs a
   transition that moves the weights; an additive registration moves nothing,
   and manufacturing a `cohort/` change to move them is the confound the
   addendum already forbids. The rider waits for the next genuinely
   weight-moving change (e.g. `ASSUMING_POSITION`, if the owner ever decides
   it).

### The campaign (~7 runs, but platoon-priced)

    platoon_v10_seed12      --scenario platoon         3M  seed 12   (neutrality re-derivation, FORCE)
    platoon_v11_seed14      --scenario platoon         3M  seed 14   (full arm, third seed)
    platoon_nomask_v1_seed12 / _v2_seed13 / _v3_seed14 --scenario platoon_nomask  3M
    platoon_flat_v1_seed12  / _v2_seed13 / _v3_seed14  --scenario platoon_flat    3M

  - Seeds 12/13/14, matching the squad ablation, shipped defaults, no overrides.
  - **Wall-clock is the real cost**: a 3M-step platoon run is ~2.5 h (16 agents),
    so eight runs are ~20 h sequential — run two lanes (`train_queue.sh` per
    lane) and it is an overnight campaign, still zero tokens.
  - After landing: `publish_baseline.py` at N=100 on all arms, then
    `scripts/ablation_report.py` in its nine-run mode (full: v8-or-v10, v9_seed13,
    v11_seed14).
  - **Bookkeeping the audit will enforce**: `platoon_v11_seed14` (and the
    re-derivation) are same-config platoon draws, so they join
    `BASELINE.json.seed_spread` — the completeness gate fails until they do.
    The nomask/flat arms are their own scenarios and stay out of it.

### How to read it

The squad result to beat: flat 296/300 vs full 282/300 (p = 0.004), flat
with the fewest defeats (a lead with power only against nomask), the
hierarchy's remaining value case doctrine-valid traffic alone (commander
survival retracted per assurance #64 — one re-executed seed, one checkpoint). The depth hypothesis is that a 16-agent flat team, where every agent
holds the full OPORD and no order traffic exists, pays a coordination cost the
9-agent squad did not — if so, the success gap narrows or inverts and the
defeats row moves first. Read robustness and interpretability before success
(the original's own instruction), read every DONE/completion cell per seed
(platoon's reporting channel is measured bimodal — `platoon_v9_seed13` is a
mute root), and pool Fisher across seeds exactly as the squad read did. Three
honest outcomes: flat wins again (the hierarchy's value case narrows to
interpretability at every measured depth), flat degrades (depth is where
structure pays, measured at last), or a split — and any of the three goes in
the README next to the squad table, at the same strength.

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
- **The transparency probe has not run at v1.19.** — CLOSED 2026-08-18: the
  sweep ran on all eight v1.21 members at both checkpoints and is read in
  docs/transparency.md ("The v1.21 fleet sweep"). Majority still unbeaten on
  destination (16/16 cells); maneuver finals within 3–9 points; three final
  posture cells beat majority for the first time; `patrol_brique_v41_seed14`
  is the readability miss to read before any pricing/probe-model work.
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
