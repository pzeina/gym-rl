# Roadmap

## ⟳ Session handoff — resume here (2026-08-15, **v1.20b is TRAINED, AUDITED and NOT SHIPPING; v1.21 is the open window**)

**★ READ THE LAST ENTRY FIRST — the blocking question is ANSWERED and v1.21's
shape is decided.** The incumbent price is a seed lottery too: `rdb=3.0` fails
`closed_on_root_report_rate` at seed 13, **0.000 at both checkpoints, N=100**,
with success 0.99 and not one root claim in 100 episodes. So
`patrol_brique_v6`'s 0.808 is a draw, both prices flip on seed, and **v1.21 is a
GATE cycle, not a tuning cycle** — a per-run pass/fail bar over a quantity
bimodal in the seed scores the draw, not the policy. The pre-registered stopping
rule fired: **no test is named over these arms.** The 12-seed campaign keeps
running re-purposed as a distribution estimate for the gate redesign; the
decision does not wait on it. Everything below this line predates that answer —
read the v1.21 framing through it.

**★★ AND THE SAME CONFLATION EXISTS ONE LEVEL DOWN, in `best_save_gate` (refs
assurance #57).** `patrol_brique_v19_rdb3_seed13`'s `ckpt_best` was written at
**iteration 25 of 2930 — 25,600 steps of 3,000,320 — on a window at 2% rolling
success**, and was the run's only save. Not a bug: the reporting window is
success-conditioned (`recent_root_closed` is appended only on episodes that sent
an ENDEX, which `cohort_env` sends only on success), has no turnover
requirement of its own, and sets an absorbing flag — so a handful of wins reads
0.500, clears the floor, and locks the checkpoint against the 99%-success policy
that follows. Replayed off `metrics.csv` and verified against the iteration
stamped in `ckpt_best.pt`, 91 of 91 runs agreeing:
`scripts/checkpoint_selection.py <run>`. **Fleet-wide it is one run wide** — of
38 runs carrying the reporting axis, only `v19` selects differently under a 0.5
success floor. The `cohort/` fix is the owner's call AND blocked by the live
campaign; the digest line that would have caught it on the day has landed.

**State**: `multi-agent-dev`; tag still v1.18.0. **925 tests green, 0 skipped**,
ruff clean, spaces **Discrete(228)/Box(220)** frozen. **`runs/BASELINE.json`
still names the v1.19 members and `scripts/baseline.py` prints BASELINE OK on
them** — that is the shipping fleet and it is unchanged.

**⛔ v1.20b IS BLOCKED BY ITS OWN GATE — do not publish it.** All eight members
trained, share one `cohort/` tree (`5fa24bad`), carry no `--reward` overrides,
and were scored at N=100 on both checkpoints. Seven match their v1.19
incumbents on success (every CI overlaps; give-backs 0.2–4.6 pts, bar is 10).
**`patrol_brique_v7` fails `closed_on_root_report_rate` at BOTH checkpoints —
0.000, zero root claims in 100 episodes — against an incumbent that reports at
0.808.** The gate is new in v1.20, so v1.20 is stopped by the thing v1.20 added.
No substitute exists and this was checked, not assumed: `patrol_brique_v8_rdb3`
reports and is on the fleet's tree but carries a `--reward` override (forbidden
in a member); `patrol_brique_v11_seed14` reports but is on a different tree; and
a single-member retrain now lands on a third tree, because `8a7645c`/`847acec`
moved `cohort/` after the campaign. **Any fix is a full eight-member retrain.**

**⚑ WHAT v1.21 HAS TO DECIDE — two levers produce the same silence, and v1.20
moved both at once.** The mute commander is the same regime everywhere it
appears: ADVANCE-dominated orders with no SEIZE, the root far from its
objective, never dying, never claiming. What differs is which lever put it
there.

    patrol_brique, current tree, seed 12, N=100 — THE PRICE
      root_done_bonus 3.0   closed-on-root 0.867  (136 root claims)
      root_done_bonus 2.0   closed-on-root 0.000  (0)
      root_done_bonus 1.0   closed-on-root 0.000  (0)
      and at 1.0 across seeds: 12, 13, 15 all mute; only seed 14 reports

    squad, rdb=1.0 fixed, N=100 — THE CHART BLOCK
      with #42's block      2 of 4 seeds mute (14, 15 at 0.000)
      block removed          4 of 4 report (0.825–0.866, a 0.041-wide band)

  `root_done_bonus` was cut 3.0 → 1.0 *because* 3.0 regressed squad. At 1.0 it
  takes patrol_brique's commander mute. **The right price is scenario-dependent
  and that is the finding, not a tuning detail.** The reading that fits both:
  the root claims iff it occupies, and it occupies only while completing is
  worth more than what it gets from not completing — ordering and surviving.
  The chart block raises the alternative (more subordinates to order); cutting
  the bonus lowers the completion side. Both shrink the same margin, and the
  eight scenarios sit at different distances from the cliff.

  **The falsifiable prediction v1.21 should test first, because it separates the
  levers in one campaign**: patrol_brique with the chart block REMOVED at
  rdb=1.0 should report, the way squad does. If it stays mute, the block is not
  implicated in that scenario. (`patrol_brique_v13_pre42_seed12` and
  `..._v14_pre42_seed13` are training this now.)

  **⚠ ANSWERED AT FIVE SEEDS, AND THE ANSWER IS NO — the block does not control
  this channel (`patrol_brique` 2/5 vs 1/4, Fisher p = 1.000; pooled with squad
  over 17 runs, p = 0.347). Seed 14 reports WITH the block and is mute without
  it, the exact mirror of seed 12. See the RETRACTION entry at the bottom, and
  read the rest of this handoff's v1.21 framing through it: the reporting
  channel is seed-determined, not lever-determined, and v1.21 is not a tuning
  cycle. The paragraph below is the superseded intermediate reading, kept
  because the entries reference it.**

  **One of the three routes that leaves open is already narrowed: a DECLARED
  seed-selection policy would have to be declared PER SCENARIO**, because a seed
  that reports on squad says nothing about `patrol_brique` — agreement is at
  chance over eight matched pairs (`scripts/reporting_channel.py --pairs squad
  patrol_brique`, and the last progress-log entries).

  **⚠ READ THE RUNNING `rdb3_seeds` CAMPAIGN DESCRIPTIVELY.** Its six runs can
  produce a significant result on 1 of 64 outcomes and only if
  `patrol_brique_v23` comes back mute; the paired reading cannot reject at five
  pairs at all. The first mute `rdb=3.0` cell settles "3.0 splits too" with no
  test, and a Fisher p from this design is not a null result
  (`scripts/design_power.py`, and the last progress-log entry). An extension to
  eight seeds is queued behind it.

  **⚠ (superseded) it split — see the 2026-08-15 entries at the bottom.**
  Seed 12 flips (0.000 → **0.825**, gate FAIL → pass, a clean paired flip) and
  seed 13 does not (0.000 → 0.000). So the block is implicated in
  `patrol_brique` but removing it is neither necessary (`v5` reported with the
  block present) nor sufficient (`v14` is mute without it). **`v13` and `v14`
  differ only in the seed, and one reports at 0.825 while the other is
  absolutely mute — so seed alone flips this channel, and no single-seed arm can
  decide anything about it.** Plan v1.21's campaign at ≥3 seeds per cell or it
  will produce another unreadable result. Two associations died here too: the
  ADVANCE/SEIZE order mix does not track muteness (the mute arm orders SEIZE
  0.98), and neither does occupancy magnitude (the mute arm occupies **twenty
  times** the vocal one). The occupancy account is retired as a mechanism and
  survives only as a within-policy, per-episode regularity.

**⚑ THE FIRST v1.20 FLEET WAS VOID AND HAS BEEN RELAUNCHED — read this before
reading anything else about v1.20.** `#42`'s `_fill_vacancy` change added a
guarded `parent.subordinate_ids.append(successor.id)` and thereby made the
pre-existing `successor.subordinate_ids.append(promoted.id)` redundant — but
that older line had **no `not in` guard, so both fired** and the backfilled
agent was linked into its new leader twice. It triggers on the commonest
succession in the game: kill SL1 in `squad` and TL1's chart reads
**`['TL2', 'RFN1', 'RFN1']`**. `living_subordinates` is what
`observations.py` writes into the four subordinate slots and what
`actions.py` indexes with `order_slot`, so the promoted root observed a
phantom subordinate and carried **two distinct ORDER indices addressing one
agent** — inside the very fix meant to make the promoted branch observable and
orderable. Found by the `#49` investigation, fixed in `da24b42`.

`scripts/campaigns/v1_20_fleet.jobs` was **stopped after four members** and is
superseded by **`scripts/campaigns/v1_20b_fleet.jobs`**, now training all eight
against the corrected tree at the same budgets and seed 12. **`fireteam_v10`,
`fireteam_defend_v21`, `squad_v16`, `squad_recon_v9` and the killed partial
`squad_screen_v12` trained against the defective tree and MAY NOT BECOME
BASELINE MEMBERS.** They are kept, not deleted — they are the only corpus
measuring what the double-link cost, and `squad_v16`'s finding below rests on
one of them.

**`runs/BASELINE.json` still names the v1.19 members and `baseline.py` will
FAIL against the new tree until the new campaign lands and is re-sealed — that
is expected, not a regression.** The v1.19 members keep their place until their
replacements beat them; publishing a MISS over an incumbent is still an ask.

**Which scenarios the defect could actually reach — read the landing fleet
through this.** The removed line only ran inside the *recursive* `_fill_vacancy`,
which needs the successor to have had a team of its own. So chart depth decides
it, and three of eight scenarios were structurally immune:

    depth 1, no non-root commanders — IMMUNE
      fireteam, fireteam_defend, defend_brique
    depth 2, TL1/TL2 below the root — exposed
      squad, squad_recon, squad_screen, patrol_brique
    depth 3, six non-root commanders — most exposed
      platoon

Confirmed rather than argued: `fireteam_v11` reproduces the void `fireteam_v10`
to three decimals (`beh_success` 1.000, `final_success` 0.900, rolling 0.954),
which is the positive control for the fix being a no-op where the chart is flat.
So on the depth-1 members, movement against their first-campaign counterparts is
noise; on the depth-2/3 members it is the defect. **`platoon_v7` is where it
should show most.** Note this does NOT make the first campaign's depth-1 runs
usable — provenance is the `cohort/` tree and a baseline must be one system —
but it does mean their retrain is a formality rather than a rescue.

**⚠ THE OPEN QUESTION THE NEW FLEET IS THE MEASUREMENT FOR: `squad_v16` trained
to a mute commander.** (`squad` is depth 2, so the defect *was* live in it.) 0 root claims in **1,865 admissible root steps**,
`closed_on_root_report_rate` **0.000**, gate **FAIL** at both checkpoints —
while success was excellent (1.00 ± 0.00 at FINAL, 0 root deaths). Its matched
arm `squad_v15_bonus1` — same scenario, same seed 12, same budget, and the
digest's own economics block confirms *prices identical* — filed **196 root
claims at 0.866** on the v1.19 tree. **The double-link cannot be the whole
story**: `is_root_opord_claim` and `is_done_admissible` never read
subordinates, which is exactly consistent with 1,865 admissible steps. So if
`squad_v17` reports normally the defect explains it; **if `squad_v17` is mute
too, something else is silencing the commander and `root_done_bonus=1.0` is
back under question.** Do not read the new squad member as a routine landing.

**⇒ THAT QUESTION IS NOW ANSWERED (2026-08-14, refs #52) — it was the chart
block, and `root_done_bonus=1.0` is exonerated.** `squad_v17` (seed 12) reports
normally at **0.959**, so the fleet member is not mute; but seeds 14 and 15 on
the same tree are (`squad_v10c` 0.000, `squad_v20_seed15` 0.000). A single-
variable branch with `56ada9a`'s 8-line chart block removed and everything else
held flips **both** of them back — 0.000 → **0.825** and 0.000 → **0.857** at
N=100, and 0.052 → 0.835 / 0.000 → 0.811 at `ckpt_best` — while success stays a
null (p = 0.08, 0.68). The mechanism is positional: with the block the root sits
at **2.2× the distance** from its objective (41 vs 19.5) and never takes the
ground it would report. Seed 16 fails on both trees, so this is not a universal
explanation of squad failure. **Across four seeds tree A sits in a 0.041-wide
band (0.825–0.866) while tree C is bimodal — 0.937–0.959 or exactly zero** —
and on the reporting seeds the two trees are near-identical in distance and
casualties, so removing the block has **no measured cost**; what it removes is
the availability of the mute regime. Which way to go is still an owner's
decision — #42 fixed a real structural defect, and where tree C reports it
reports *better* — written up with three options and the correction in the two
2026-08-14 progress entries at the bottom. Nothing was applied.


**When the campaign lands**: `publish_baseline.py` at N=100 → compare each
replacement against its incumbent → `baseline.py --seal` → `results_table.py
--write` → `/boards`. Watch two things in particular: whether the #42 chart fix
moves succession-heavy scenarios (platoon most of all), and whether
`closed_on_root_report_rate` clears its new 0.5 floor everywhere — the gate is
new, so its first fleet is also its first real test. **Read that floor at BOTH
checkpoints** (`(assurance, #48)`): on the v1.19 fleet two of eight members are
mute at `ckpt_best` — `patrol_brique_v6` 0.000 and `platoon_v6` 0.021, both
reporting normally at FINAL — and the publish path reads gates from
`behavior_final.json` only, so nothing would have said so. Whether a mute `best`
should be *refused* is an open owner decision, and the landing fleet is the
measurement it is waiting on.

**The first fleet's job 1 (`fireteam_v10`) took the digest with it — fixed,
tooling only.** (The run itself is void per the ⚑ above.) It was PUBLISHABLE
(final 95%, best-final gap 5 pts,
ckpt_best 1.00 ± 0.00 at N=20), but `run_report.py` raised `ClaimOrdinalError`
on its own artifact: ep19 carries 4 root claims, 2 rejected and **2
successions**, with `endex_on_root_report` still 1. That is the limit
`test_confirmed_claim_is_last` already took on 2026-08-12 off
`squad_v14d_nobonus` — `done_reports_root` counts *root-sender*, the operation
closes on a root-*OPORD* claim, and a promoted successor may truthfully
complete its own personal mission while the operation correctly runs on. The
guard test had the exclusion; the digest had not, so it crashed on a corpus its
own invariant considers sound. `root_claim_ordinal` now skips succession
episodes, still raises for a non-succession violation, and **prints the
exclusion count** so a split is never read as covering episodes it was not
derived from. Expect this to matter more as the campaign reaches the
succession-heavy scenarios — #42 is precisely a change that drives successions
up. Re-pinned on `squad_v10`, where the counts move and **the #46 finding
strengthens**: the first/later gap goes +0.230 → +0.277 on the final policy and
−0.073 → −0.223 at `ckpt_best`, so the inversion is wider once the episodes the
proxy could not attribute stop diluting both ordinals. The pre-registered
pooled 0.433 is untouched and is now asserted off the corpus totals rather than
off the split.

**⚑ READ THIS FIRST — the squad pricing arc reversed twice today, and it now
has an answer and a recommendation.** Four findings, in the order they must be
read:

1. **The claiming is not EV-driven.** The zero-price probe (v13/v13b, N=100)
   answered its pre-registered question NO: claim volume did not survive its EV
   going to zero, it went **4.5× and 6.5×** the shipped rate on both seeds,
   with strictly −EV non-root claims going 1.47 → 8.67 and 9.84 per episode.
   **The pricing axis is closed as an explanation of the spam**; the remedy
   belongs in masking or the claim API.
2. **Claim volume was never the regression's signature at all.** `squad_v8` and
   `squad_v10` have *identical* claim rates (0.0060) and differ by six points of
   success. Every fix in the v11/v12/v12b arc was aimed at the wrong quantity.
3. **The regression's actual cause is one price, and it replicates:
   `root_done_bonus=3.0`.** `squad_v14_nobonus`/`v14b` set it to 0 alone,
   single-variable, both seeds landing on **exactly 98/100** — pooled 196/200
   against the shipped 180/200 (**p = 0.0011**) and **p = 1.000** against the
   previous era. First result in the arc that is single-variable, replicated
   and significant at once.

4. **And the value to set it to is `1.0`. The two candidates tie on success;
   the reporting decides.** Four seeds each, FINAL policy at N=100:

       shipped rdb=3.0   180/200 = 0.900              mute 0/2
       rdb=0             388/400 = 0.970   p=0.00074  mute 2/4
       rdb=1.0           389/400 = 0.9725  p=0.00030  mute 0/4
       previous era      195/200 = 0.975

   Both candidates are a null against the previous era (p = 0.801 and
   **p = 1.000**) and **indistinguishable from each other on success
   (p = 1.000)**. The separator is the completion report: `rdb=0` leaves two of
   four commanders **absolutely mute** — 0 root claims in 11,973 and 10,112
   admissible steps, a regime and not a low rate — while all four `rdb=1.0`
   seeds claim, in a tight band (closed-on-root **0.866 / 0.866 / 0.825 /
   0.857**). Report recall is uniformly better than shipped (0.874–0.957 vs
   0.795–0.848) and pooled false-COMPLETE improves (0.581 vs 0.693).

   **Stated honestly, because two earlier claims here did not survive four
   seeds**: 0-of-4 versus 2-of-4 is Fisher **p = 0.43** — the mute difference
   is *not* significant on seed counts alone. What carries it is that the zeros
   are absolute, that 1.0 lands in a 0.83–0.87 band every time, and above all
   the **paired flip on seed 14, which files zero root claims at `rdb=0` and
   0.825 at `rdb=1.0`**. Also corrected: false-COMPLETE is better pooled but
   *mixed per seed* (0.459/0.531/0.655/0.623 against 0.560/0.805), and
   commander-survival-within-successes is a **null**, not an improvement —
   0.185 vs 0.206, p = 0.57, once all four seeds are pooled against both
   shipped seeds rather than two against `squad_v10` alone.

**⇒ ACCEPTED AND APPLIED (2026-08-12): `RewardConfig.root_done_bonus = 1.0`,
bundled with `_fill_vacancy` into the v1.20 window.** Both arms reached four
seeds and the confirmation held: 0/4 mute at 1.0, including the paired flip on
seed 14 (0 root claims at `rdb=0`, 0.825 at 1.0). Full numbers — and three
corrections to earlier two-seed claims — in the six 2026-08-12 progress entries
at the bottom.

**⚠ A gate hole found on the way — ACCEPTED AND CLOSED in v1.20**:
`successes_announced_rate` reads **1.00 for a commander that never claims** —
it counts the ENDEX, not who claimed it, and passed a mute policy three times
(v11, v14b, v14c). `closed_on_root_report_rate` is now a regression gate at a
floor of 0.5.

**What shipped (v1.19 — still the published fleet until v1.20 lands).**
`runs/BASELINE.json` names one run per doctrine scenario and
`scripts/baseline.py` passes on it: **one `cohort/` tree (5f848fb6), no
`--reward` overrides, every headline the FINAL policy at N=100, every gate
green, every give-back under the bar, every checkpoint loadable, every win
announced.**

    fireteam         fireteam_v9          0.97 ± 0.03     97/97 announced
    fireteam_defend  fireteam_defend_v20  0.98 ± 0.03     98/98
    squad            squad_v10            0.92 ± 0.05     92/92
    squad_recon      squad_recon_v8       0.99 ± 0.02     99/99
    squad_screen     squad_screen_v11     0.98 ± 0.03     98/98
    patrol_brique    patrol_brique_v6     0.99 ± 0.02     99/99
    defend_brique    defend_brique_v15    1.00 ± 0.00   100/100
    platoon          platoon_v6           1.00 ± 0.00   100/100

`platoon` and `patrol_brique` used to win in complete silence (0/100 and 0/99);
`fireteam` was the one champion published with a flag saying it missed the bar,
at 0.80 with a fifth of its episodes timing out. 96 superseded runs are in
`runs/archive/` — moved, never deleted, and every reader resolves through
`fleet_status.find_run` / `run_report.run_dir`. Every member's FINAL checkpoint
is committed and hashes to the `checkpoint_sha256` its evaluation recorded (8/8
verified from `HEAD` blobs), so the published numbers are reproducible from a
clone — they were not until #44, because the archive move had inverted a
`.gitignore` glob.

**⚑ THE NEXT ITEM, and it is not a tuning question.** `cohort/core/units.py::
_fill_vacancy` sets `successor.leader_id = vacated.leader_id` and never adds the
successor to that leader's `subordinate_ids`. Its own recursive branch does; the
top-level call does not. The promoted branch therefore drops off the commander's
chart — unorderable, unobserved, absent from the trace, and **never devolved to
when the commander falls in turn**. On the squad chart **4,080 of 5,040** death
orderings orphan a branch and **1,928** reach a state with nobody in command;
fireteam is exempt (0/24). Realised on this fleet, 660 episodes: 44 with a broken
chart, 1 with no commander. The one-statement patch is in the `(assurance, #42)`
entry below and was **validated in memory without being applied** — suite stays
green, the sweep goes to 0/5040. It stays out because restoring an order edge
moves action masks and therefore every rollout: **applying it invalidates this
baseline and requires the fleet to be retrained.** That is the v1.20 cycle.

**Also open, in order:**
1. **The squad regression — SOLVED, awaiting the owner's call on the default.**
   Cause: `root_done_bonus=3.0`. Fix: **1.0**, replicated on both seeds on
   success (194/200, p = 1.000 vs the previous era) *and* on the completion
   report (closed-on-root 0.866/0.866), improving every other axis over the
   shipped default too. `rdb=0` also fixes success (196/200) but leaves the
   reporting a coin flip and is not the recommendation. The claim-spam account
   of the regression is dead on two counts: volume is action-mass, not
   economics, and `squad_v8`/`squad_v10` share a claim rate (0.0060) while
   differing by six points of success. **Applying it is a reward default —
   owner's call, and it retrains the fleet; see the ⇒ recommendation above.**
2. **Three deferred `cohort/` patches** from the assurance cycle, each with its
   exact diff and a skipped test: #39 (`eval_commit` in the artifact), #40
   (`parse_succession`, the formatter's inverse), #42 (above). All three want
   the same v1.20 window.
3. **A denominator v1.19 widened**: `closed_on_cadence_report_rate` and
   `closes_per_root_sitrep` read 0.000 and 11.0 on completable roots because
   every scenario now sends an ENDEX while the numerator stayed SITREP-only.
   Patch written out; `closed_on_root_report_rate` is unaffected and is what the
   README quotes.
4. **The B3 ablation inverts on outcome** at v1.19 (success 0.92 / 0.98 / 1.00,
   defeats 7.0 / 1.0 / 0.0) while the interpretability half holds. Read it with
   item 1: the full arm IS the run that regressed. The README claims the
   interpretability result and no longer claims the outcome one.

**The plan, not just the state**: [`docs/next-cycles.md`](docs/next-cycles.md)
carries the v1.20 cycle (four written-and-unapplied `cohort/` patches that want
one window), the squad-regression experiment in order, the ablation at three
seeds, the measurement gaps, and the two habits that paid today.

**How to work here**: `CLAUDE.md` first — especially its new **"The baseline
fleet (v1.19 onward)"** section, which carries the four rules that are easy to
break by accident (provenance is the tree not the sha; a campaign freezes
`cohort/`; no overrides in a member; archiving is a move). Commits are
pre-authorised; **pushing is not**. Quote every between-run delta at both
checkpoints or not at all.

---

## ⟳ Session handoff — 2026-08-11 earlier, v1.18.0 shipped (SUPERSEDED)

**State**: `main` and `multi-agent-dev` both at HEAD and in sync with origin;
**`v1.18.0` tagged and pushed** (first tag since v1.9.0). 687 tests green, ruff
clean, spaces **Discrete(228)/Box(220)** frozen. Nothing training. Boards
published and current. Read the tag annotation first — it states what the
release does *not* claim.

**⚑ THE NEXT ITEM, and it needs a decision before code.** The fleet re-published
at N=100 exposed that **the announcement guarantee covers two scenarios of
nine**. Measured, final policy, wins announced on the net:

  defend (ENDEX, a protocol act)   391/391 — complete by construction
  squad_screen / squad_recon / squad   91–98%
  fireteam_v8                       49/80
  **platoon_v5  0/100   ·   patrol_brique_v5  0/99**

`platoon` and `patrol_brique` succeed on essentially every episode and **never
once say so**. Same shape as `fireteam_defend_v16`'s 0/99 before ENDEX. Where the
announcement is a protocol act it is complete; where it is an agent behaviour it
ranges from 98% to nothing and does not track how well the scenario is solved.

**Options, none taken:**
  (a) Extend COMMAND's close announcement to completable roots — ENDEX, or a
      confirm-shaped act, on SEIZE/RECON/CLEAR too. Mirrors v1.16 exactly and is
      the only option that yields a guarantee. Touches every scenario's
      transcript; masking-only, so spaces stay frozen; needs a fleet retrain to
      publish honestly.
  (b) Leave it and say so in the README, as it now does.
  (c) Attack root-claim reliability instead — the option four price experiments
      say does not work.
I recommend (a). It is the same argument v1.14–v1.17 established, and it
reproduces across the fleet with no new experiment.

**Also open**: `fireteam_v8` does **not** clear the publishing bar (12.0-point
give-back, 80% ± 8 at N=100 against a superseded N=20 90% ± 13) and is published
with the flag. It is the one scenario whose champion is unfit; a retrain is the
obvious move and nobody has taken it.

**Autocycle findings, 2026-08-11** (three items, all logged below): `squad_v7`'s
lost artifacts recovered and its crash found unreproducible; the publish gate
**validated** — give-back predicts *signed* overstatement at r = 0.564, p = 0.015,
via new `scripts/publish_audit.py --validate`; and the README's `—` in the
announced column found to be hiding a zero. **Two of the three items were
corrections to claims I had made ahead of a measurement that was already
available.** That is the pattern to watch in this file.

**How to work here**: `CLAUDE.md` (Operating guide + Training workflow) first;
assurance contract in `ASSURANCE-SYNC.md`. Commits are pre-authorised; **pushing
is not**. Quote every between-run delta at both checkpoints or not at all.

---

## ⟳ Session handoff — 2026-08-10, the DONE-channel trilogy (SUPERSEDED)

**Everything in the 2026-08-08 block below is superseded.** It is kept because
its D4 and option-4 reasoning is still the best account of those decisions, but
its state, its numbers and its next-steps list are three cycles stale.

**State**: `multi-agent-dev`, ~37 commits ahead of `origin/multi-agent-dev`,
**nothing pushed**; tag still v1.9.0; **618+ tests green, ruff clean**; spaces
**Discrete(228)/Box(220)** frozen throughout — the whole fleet still loads.
Boards are **PUBLISH PENDING** and the README has deliberately gained no defend
row since v15: the family has been superseded twice in a day and publishing
mid-flight would have shipped a policy we already knew we were replacing.

**⚑ THE ONE THING TO UNDERSTAND: three cycles, one problem, and the answer was
not a price.**

- **v1.14** (`eccf816`) made DEFEND success *conservation of the position to a
  stated hour* — occupation required continuously from H, no retake, early
  release when the band is neutralised, horizon `int(0.5·max_steps)`. Owner's
  call, and it was right: on `defend_brique` it changed what is *learned*
  (`v9` loses the position in 12/100 episodes at a median of H+7; `v11` 0/100).
  On `fireteam_defend` it only re-scored — `v15` already held its ground.
  It also made DEFEND `COMPLETABLE`, which reopened the root's MISSION COMPLETE
  **and silently switched ENDEX off**, because `continuous_root` was the only
  gate on it. That was a side effect, not a decision.
- **v1.15** (`727ef60`) tried to price the reopened channel honest: pay
  `root_done_bonus` only on the episode's first claim. **It bought silence.**
  Root claims 321 → 0, and P(DONE | a true claim is available) fell
  **0.401 → 0.000083**, 40/40 episodes declining an available true claim. The
  arithmetic was wrong by 7×: a confirmed claim *ends the episode*, so a probe
  burns the bonus and the first claim really costs `done_false − bonus × P(close
  by claim)` = **−3.50**, a tariff on speaking at all. **Reverted** in v1.16.
- **v1.16** (**landed 2026-08-10** — `14d8b02`, `8dbb299`, `f5f3b97`; full
  numbers at the end of the progress log) reverts that flag and **decouples
  ENDEX from completability**. The reason is structural, and it is the most useful thing
  the assurance layer has contributed: **ENDEX is a protocol act** — COMMAND
  emits it, unpriceable, which is why it announced **103/103** successes across
  four pre-v1.14 corpora — whereas **a root claim is an agent behaviour**,
  optional and learnable in either direction. Identical prices bought 0.71-false
  spam on one scenario and total silence on the other. v1.14 changed the
  channel's *type*, from a guarantee to whatever the policy happened to learn.
  **No price restores a guarantee, only an average.** Keep both: the claim is a
  report, the ENDEX is the fact.

**Arms and controls.** v1.16 = `defend_brique_v13` / `fireteam_defend_v18`
against **`v11` / `v16`** — NOT v12/v17, which carry the reverted economics and
would make it a two-variable comparison. Watch `successes_announced` (bar:
103/103) and whether ENDEX *suppresses* honest claiming, which would be swapping
channels rather than holding both.

**⇒ BOTH QUESTIONS ARE ANSWERED.** `successes_announced` is **391/391** at
N=100, seed 123, both checkpoints on both arms (the v1.13 bar, reconstructed at
the same N and seed, is 348/348; v1.14 scored 94/391 and v1.15 0/391). ENDEX did
not suppress claiming and provably could not have: **the arms' weights are
bit-identical to their controls** (`max|Δ| = 0.000e+00`, all 15 tensors, both
checkpoints, both scenarios), because ENDEX is emitted on the terminating step
after the last action is chosen, so it never enters an observation. The open
thread is unchanged and is a *pricing* question, not an announcement one:
`defend_brique` still claims at 0.71 false and `fireteam_defend` still does not
claim at all (P(DONE | a true claim available) = 1e-6, 39/39 episodes
declining). The two problems are now separable — the announcement no longer
depends on solving either.

**What the day also established, and is worth not relearning:**
- **Quote every between-run delta at both checkpoints or not at all.** Three
  separate published claims turned on an unstated reference checkpoint
  (refs #24, #25, #26).
- The `defend_brique` priced regression survives equal footing on the FINAL
  policy (v6 0.950 vs ENDEX 0.890, p = 0.0153 over two seed sets) and **reverses
  at best/best** (0.865 vs 0.905). Across all four cells the arms are the same
  (p = 0.61). Evidence: `runs/defend_brique_v6/equal_footing_n100.json` and
  `seed_robustness_n100.json`.
- Evaluations now record `checkpoint_sha256` (refs #28), and `config.briefing()`
  publishes `defend_horizon` (refs #30). ~~The OPORD hold-until clause is
  **deliberately deferred** — it is not rollout-neutral.~~ **Corrected
  2026-08-11**: it *is* rollout-neutral, and shipped in `9dd4edf` — measured,
  not asserted, at mechanism and outcome level (see the last progress-log
  entry). The deferral rested on an assertion nobody had run.
- Boards regenerate themselves when a run lands (`scripts/train_then_boards.sh`);
  only publishing needs a session (`/boards`).

**Next, in order**: read v1.16's report → ~~decide the OPORD hold-until clause
(#30)~~ **done 2026-08-11, `9dd4edf`** → re-publish the defend family off FINAL
numbers at both checkpoints (the numbers are unchanged by v1.18 — verified
field-by-field, so the re-publish is a table edit, not a re-score) →
the fleet-wide staleness left by the v1.15 flag (other scenarios' claim numbers
were measured under a rule that no longer exists; not retrained, owner's call).

---

## ⟳ Session handoff — 2026-08-08, v1.12 A/B resolved (SUPERSEDED, see above)

**State**: `multi-agent-dev`, **82 commits ahead of `origin/main`**; latest tag
v1.9.0; **538 tests green, ruff clean**; **nothing training**. Spaces
**Discrete(228)/Box(220)** — unchanged by the v1.12 reward work *and* by the
`defend_brique` spec repair, so the whole fleet stays loadable. One caveat on
that repair: `defend_brique` now draws an H-hour, which consumes RNG, so its
seeds are a new era — pre-`450b392` brique runs (`_v1`…`_v5`) are not
episode-comparable with `_v6`/`_v7`, though every checkpoint still loads.

**The remote is now the standard shape — resolved, on the owner's instruction.**
Three earlier handoffs (mine included) described a repo with no named remote:
every branch carried `branch.<name>.remote` as a bare URL, which git accepts for
push/pull, so pushing always worked — but `git remote -v` was empty, there were
no remote-tracking refs, ahead/behind in `git status` was blank, and `gh` said
"no git remotes found". The owner asked for the conventional setup, so:
`git remote add origin https://github.com/pzeina/gym-rl.git`, `git fetch origin`,
`git branch -u origin/<name>` for `main`, `multi-agent-dev`, `single-agent-dev`.
All three now track. `gh` authenticates as `pzeina` and resolves `pzeina/gym-rl`
from this directory. All 11 local tags (through v1.9.0) were already on the
remote; nothing but the 75 commits is unpushed. `origin/multi-agent-dev` and
`origin/main` both sit at `d8fa125`. The assurance fork at
`~/Documents/gym-rl-fork` is untouched and keeps its own `origin` + `local`
remotes — the ASSURANCE-SYNC.md contract is unaffected.

**⚑ THE REWARD DECISION IS TAKEN — option 4, by the owner, 2026-08-07.** Not
the recommended option 1 (scope the payout by scenario) but the more principled
one: *make the defend terminal proportional to survivors*. Implemented in
`f39b5a9` and, as of 2026-08-08, **trained and confirmed on both defend
scenarios** — see the progress log. (This paragraph described the economics
before any result existed; the result is now in.) On DEFEND/DENY
roots only, the terminal is multiplied by
`(1 - scale) + scale x surviving_weight / starting_weight`, rank-weighted, at
`defend_survivor_scale = 0.35`. It is not forfeiture again because the
multiplier is identical for every agent, **fallen included** — a death is a
shared loss, not a private one, so the D4 asymmetry cannot re-form. The
constant is fixed by the dominance invariant, not by taste: the multiplier can
only scale the terminal down, so `win_beats_stall` must clear 2x at the FLOOR
(hold, and be ground down doing it), and `fireteam_defend` at 3.42 undiminished
puts the ceiling at `1 - 2.0/3.42 = 0.415`. **The owner authorised the fallback
in advance**: if the A/B goes against it, revert to option 1 — which needs no
code change, only `--reward defend_survivor_scale=0`.

**⇒ THE A/B IS RUN AND OPTION 4 IS CONFIRMED — keep `defend_survivor_scale`
at 0.35.** See the 2026-08-08 progress-log entry for the numbers. Headline:
`fireteam_defend` `v11`→`v12` root deaths 0.35→0.15 (p=0.001) and success
0.74→0.86 (p=0.034); `defend_brique` needed its scenario repaired first
(`450b392` — it declared a DEFEND root with prepared positions and never gave
the fire team time to occupy them), after which `v6`(0.35) beats `v7`(flat)
0.97 vs 0.89 success (p=0.027) with both arms publishable and all gates
passing. The pre-authorised option-1 fallback is **not** indicated.

**⇒ THE DONE-CHANNEL THREAD IS CLOSED — not by pricing it, by deleting the
act.** Two earlier versions of this block sent the next session after
`done_false`, in opposite directions; both were pricing something a DEFEND
root should never have been offered. **Ignore both.** v1.13 (`16cb2a6`,
owner's decision): MISSION COMPLETE is masked shut on a continuous posture,
the root reports the situation, and COMMAND transmits **ENDEX**. The
`fireteam_defend_v13/v14` + `defend_brique_v8` campaign was killed mid-flight
once the point landed; `runs/fireteam_defend_v13/` is a partial (2.17M/3.5M)
left on disk, not a result.

**⇒ THE NEXT THING IS TO READ `endex_v1_13`.** `fireteam_defend_v15` and
`defend_brique_v9` train the defend family against the loop it now has to close
(every checkpoint on disk learned under the old rule). The number to watch is
**`closed_on_root_report_rate`**, against v12's own policy re-scored under the
new rule: **0.19 at `ckpt_best`, 0.47 at `ckpt_latest`** (N=100, seed 123,
`runs/fireteam_defend_v12/endex_rescore.json`). The bare **0.22** this block
and the boards used to quote named no checkpoint and matches neither (refs #24,
corrected 2026-08-09). Both arms have since landed — 1.00 at final on each,
success/root-death still to be settled at N=100 by `/publish`. Success/root-death
were expected to hold at the v1.12 levels (fireteam 0.86/0.15, brique
0.97/0.05). Nothing else here is blocked.

**Reward weights are on the CLI now** (`b8ed7f1`): `--reward KEY=VALUE`,
repeatable, typed off the dataclass. This was the mechanical blocker ROADMAP
kept naming — `squad_v9` (separating `d44ee8d` from the `done_false` revert on
the five confounded arms) is now one flag, `--reward done_false=-2.0`. Three
silent failure modes went with it: `economics.json` recorded `RewardConfig()`
rather than the prices in use (so every override-driven A/B would have read as
a no-op to the assurance-#20 confound audit), checkpoints carried no prices (so
`evaluate` scored every policy under tree defaults), and a typo'd key would
have trained the default under an `economics.json` claiming otherwise. `train.py`
also now prints a pre-flight warning when the requested prices put the
discounted win/stall ratio under 2x — the v1.11 collapse economics is one
keystroke away (`--reward success_team=10` scores 1.37x on fireteam).

**D4 IS SOLVED.** The collapse that has haunted this repo since v1.0 was one
shared policy free-riding on a terminal its casualties could not collect: the
payout read `for s in roster.living`, so a soldier who died at step 50 of an
episode that succeeded at step 200 got none of the 60 points. Per agent, hanging
back cuts P(die) 0.129→0.022 (+6.4) while team success goes 1.00→0.00 (−52.3) —
but ONE shared policy updates EVERY agent at once, and a per-agent advantage only
sees the first number. `d44ee8d` keeps casualties in the episode (STAY-only,
accruing nothing) and pays them the team terminal. Clean A/B, both seeds:
`squad_screen_v9`/`v10` **0.00 ± 0.00** → `fallen_v1`/`v2` **1.00 ± 0.00**,
non-overlapping, `done_false` held fixed. Observation width is exonerated by
direct evidence — the fallen arms run the same 220-input space that collapsed.

**…AND IT REGRESSES DEFEND-TYPE SCENARIOS.** Second clean pair,
`defend_brique_v3` → `v4` at N=100: success **0.88 ± 0.06 → 0.91 ± 0.06**
(indistinguishable), but commander death **0.24 → 0.61**, cover 0.513 → 0.416,
and fight distance **2.87 → 6.09** — the final policy **FAILS**
`mean_distance_from_objective_under_threat` (bar ≤ 5.0), an encoded regression
gate that `v3` passes. Oracle friendly deaths/ep 0.60 → 1.85. `fireteam_defend_v11`
agrees directionally (0.87 → 0.74). **Mechanism**: where a decisive objective
exists, engaging ends the episode sooner, so cover and survival rise as
instruments (`squad_screen` 165 → 53 steps, deaths halved, cover ×15). In a
defend scenario there is no fast win — the mission is to still be there later —
so removing forfeiture makes bodies cheap with nothing to buy.

**v1.11 fleet, final-policy numbers, all under identical current code**:
`squad_v8` **1.00 ± 0.00** · `squad_recon_v7` **1.00 ± 0.00** · `platoon_v5`
**1.00 ± 0.00** · `patrol_brique_v5` **1.00 ± 0.00** · `fireteam_v8` 0.90 ± 0.13 ·
`defend_brique_v4` 0.91 ± 0.06 (N=100) · `fireteam_defend_v11` 0.74 ± 0.09 (N=100)
· `squad_screen_fallen_v1`/`v2` **1.00 ± 0.00**.

**⚠ Attribution: only two pairs are single-variable.** `run_report.py <run> --vs
<baseline>` now prints this automatically from `economics.json` (assurance #20,
`80166d9`). `squad_screen` ×2 and `defend_brique` are **CLEAN**; `squad`,
`squad_recon`, `platoon`, `fireteam_defend`, `patrol_brique` are **CONFOUNDED**
by `done_false` −2.0 → −0.5; `fireteam_v7`→`v8` is uncheckable. So the collapse
being gone on those five is *consistent with* the fix generalizing, **not
established** — the `done_false` revert is a live alternative explanation, and
`rewards.py` records that −2.0 killed terminal income in report-centric
scenarios. My campaign file caused this: it pinned budgets/seeds/lr and let
economics drift with the tree.

**Open residuals, none closed by the fix**: `platoon_v5` has gone **mute** —
MISSION COMPLETE claims 35 (`v4`) → 3 → **0** at the final, forfeiting
`root_done_bonus` every episode; obedience latency 3.9 → 21.0, staged-order
release 60% → 4%. Suspect is exploration not price (entropy 1.957 → 1.380,
`done_false` −0.5 makes claiming cheap at break-even p ≈ 0.14); **refutable** —
if exploration, `done_reports` recovers with higher `ent_coef` at unchanged
`done_false`. `fireteam_v8` false-DONE **0.908** at the final and report recall
0.75 → 0.34, the one arm whose reporting gets worse. Staging abandon is
fleet-wide (18/20, 18/23, 68/68 on platoon).

**Next, in order**:
1. **Make the reward call** (last progress-log entry). Everything downstream waits
   on it, because re-publishing a fleet that fails a regression gate on two
   scenarios would repeat the mistake the publish audit just corrected.
2. **Disentangle the five confounded arms** — one run, `squad_v9` at `done_false`
   −2.0 with the fix. Needs `done_false` on the CLI first (only `PPOConfig` is
   exposed); small change.
3. **Then re-publish** at N=100 off FINAL numbers (`/publish`), correcting README
   and the v1.9 table, which are superseded twice over. `scripts/publish_audit.py`
   is the gate: 11 of 18 older published runs fail it.
4. **Land the single-legal-action sampling fix** — an agent with one legal action
   should take it without drawing. Held all session on purpose: it shifts the RNG
   stream (42 of 55 metrics move on the *same* checkpoint across `d44ee8d`), so
   landing it earlier would have desynchronized the A/Bs above.
5. **Transparency probe** still trails the OPORD-only baseline (best squad gap
   −0.090); `docs/transparency.md` §A5. Untouched by v1.10/v1.11.
6. **`docs/vision.md`** — directional vision, designed and decided, after the
   fleet ships. `vision_arc 180°` / `fire_arc 90°` / 4-dir facing / all-round at 2
   cells. Binding constraint: `PolicyNet` is a **memoryless MLP**, so an explicit
   remembered-contact block is mandatory and its stale-track invariant is a
   first-class exploit hazard. `squad_short_vision` is the registered V0 probe.
7. **A3 self-play**, buildings + pathfinding (v1.4 deferral).

**How to work here**: read `CLAUDE.md` (Operating guide + Training workflow) first;
the assurance contract is `ASSURANCE-SYNC.md` (Stop hook active: commits auto-queue;
new GitHub issues → ONE dedicated fix agent, commits `refs #N`, never close issues).

---

Tracks planned work and its advancement. Update statuses in place; append outcomes to
the [progress log](#progress-log) with a date and, where relevant, a run name or commit.

**Status legend**: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked
(reason in item) · `[-]` dropped (reason in log). Every item states its **DoD**
(definition of done) so "done" is checkable, not vibes.

Baseline (v1.1.0, 2026-08-05): working env with hard rank admissibility, NATO
nomenclature/voice procedure/APP-6 symbology, masked PPO that trains (fireteam ~95%,
squad 80–95% eval success), interactive dashboard, 71 tests, fresh-clone verified.

---

## Milestone v1.2 — Proven depth

*The three-echelon thesis demonstrated; results trustworthy.*

- `[x]` **A1. Train `platoon` by curriculum** — flagship experiment. *(done: run
  `platoon_v1`, 89% ± 6 at N=100 — see progress log)*
  `python -m cohort.training.train --scenario platoon --total-steps 8000000
  --init-from runs/squad_v1/ckpt_best.pt --lr 1e-4 --run-name platoon_v1`.
  **DoD**: ≥70% success over 100 eval episodes; transcript shows PL→SL→TL order
  cascade; curves + GIF committed under `runs/platoon_v1/`. If it fails to train,
  the failure analysis (dashboard episode traces) is the deliverable instead —
  suspects: subordinate-slot pressure, coverage dilution across depth, credit
  assignment across three echelons.
- `[x]` **A2. Train the untrained scenarios** — `fireteam_defend`, `squad_recon`.
  **DoD**: committed runs with ≥70% success each; RECON transcript shows the squad
  observing without engaging (stealth compliance visibly working).
  *(done: fireteam_defend_v4 **91% ± 6** — zero defeats, deaths all at the
  objective once terrain doctrine landed; squad_recon_v2 **89% ± 6**. The
  original stealth clause was doctrinally misplaced: PROTERRE RECONNAÎTRE may
  engage — pure stealth becomes the SCREEN mission in the v1.4 cycle (ex-A7).)*
- `[x]` **B1. Honest evaluation standard** — bump `evaluate` default to 100 episodes;
  report a 95% CI next to success rates; regenerate the README numbers.
  **DoD**: README results carry N=100 and a CI; eval variance across reruns < 5 pts.
- `[x]` **D1. CI** — GitHub Actions: `pytest` + `ruff` on push/PR (CPU torch).
  **DoD**: badge in README; a PR that deletes `cohort/env/` fails CI (the exact
  failure mode that shipped broken clones pre-v1.1.0).

## Milestone v1.3 — Commandable

*A human commands the cohort comfortably, in the browser, on a clean net.*

- `[x]` **C1. Dashboard commander mode** — merge `play.py` into the Episode view:
  a live-simulation mode (server keeps an env stepping; WebSocket or polling), an
  order input box wired to `inject_order`, pause-on-contact option.
  **DoD**: from the browser, issue `TL1, seize obj bravo`, watch the WILCO land and
  the maneuver happen; permission errors surface in the UI; works with any checkpoint.
  *(done: Command tab + /api/live/{start,step,order,state}; DoD flow verified
  end-to-end at the API level incl. WILCO and the rank-violation error; in-browser
  click-through pending the next Chrome session — endpoints are exactly what the
  tab consumes.)*
- `[x]` **A4. Comms discipline** — small per-transmission cost, a "net busy" step
  (one transmission per net per tick, priority-arbitrated), and dedup credit so the
  first accurate CONTACT wins.
  **DoD**: ≤1 SITREP per agent per 25 steps and no duplicate CONTACT storms in eval
  transcripts, without success-rate regression (>–3 pts vs. baseline runs).
  *(done with one honest DoD miss: mechanics + traffic goals landed on both
  retrained scenarios (SITREPs ≤1 ✓, fireteam storm erased ✓, squad dedup residual
  documented); squad success 84% ± 7 (bound 81 ✓, zero regression); fireteam
  83% ± 7 (bound 89 ✗ by 6) after its retrain + diagnosed adjustment both hit
  D4 collapses — see the progress log.)*

## Milestone v1.4 — PROTERRE alignment (breaking cycle)

*Missions grounded in the doctrinal source (`docs/manuel-proterre.pdf`, MICAT);
owner-approved scope, 2026-08-05. One coordinated space-breaking cycle: every
scenario retrains, old runs kept for provenance, results republished.*

- `[x]` **P1. Full MICAT mission set** — English names, PROTERRE semantics:
  RECON (may engage), SCREEN (ÉCLAIRER: no engagement, break contact), OBSERVE
  (SURVEILLER: detect & alert), SUPPORT (APPUYER: unit-targeted fire support),
  COVER (COUVRIR: flank guard), DEFEND (TENIR), DENY (INTERDIRE, section+),
  SEIZE, CLEAR, RALLY, HOLD. Per-echelon mission admissibility from the manual's
  tableau récapitulatif (groupe/section/compagnie menus) enforced via masks;
  doctrine derivation tables rebuilt per echelon.
  **DoD**: doctrine documented in `docs/missions.md` with manual page refs;
  round-trip language tests; per-echelon admissibility tests.
- `[x]` **P2. SUPPORT mechanics** ("pas un pas sans appui") — SUPPORT missions
  anchor on a friendly element; a supporting unit in position (LOS, range)
  grants the supported unit covered movement (attacker accuracy debuff) and
  focus-fire bonus on shared targets.
  **DoD**: oracle shows supported bounds taking measurably fewer hits; a
  scenario's transcript reads `TL2, THIS IS SL1: SUPPORT TL1. OUT.`
- `[x]` **P3. Human agents** — root commander human by default (knob),
  observable to teammates, death penalty at mission-failure scale (~ -25,
  episode continues; succession exercises). Rank must satisfy the
  humans-outrank-non-humans invariant (validated at org build).
- `[x]` **P4. Rank-weighted casualties** — death/teammate-death penalties scale
  with the fallen agent's effective authority.
- `[x]` **P5. Maps ×1.5 in place** — all scenario maps and step budgets grow
  ~1.5×; objective layouts rescale. *(+ new `squad_screen` scenario)*
- `[x]` **P6. Full retrain + republication** — all scenarios retrained under
  P1–P5 (with the KL guard), N=100 evals, README/dashboard artifacts refreshed.
  **DoD**: every scenario ≥ its v1.2 success number − 5 pts, SCREEN scenario
  ≥80% with oracle-verified <0.01 shots/agent-step by SCREEN-holders.
  *(done with two honest DoD misses — squad 84% ± 7 (v1.2: 97) and defend
  73% ± 9 (v1.2: 91) — after each spent its one retrain + one diagnosed
  adjustment; see the progress log. SCREEN: 93% ± 5, unprovoked fire 0.0025
  shots/agent-step < 0.01, but total incl. riposte-under-detection 0.016.)*

Deferred to v2.0: BRIQUE asymmetric OpFor (the manual's armed-bands threat
model, p. 9) — *done 2026-08-06, see P7 below* — and buildings + pathfinding
terrain (still deferred).

## Milestone v2.0 — Adversarial & scientific

*Real tactics under a learning adversary; the design justified by measurement.*

- `[x]` **P7. BRIQUE asymmetric OpFor** (the v1.4 deferral) — the manual's
  armed-bands threat (p. 9): a flat leaderless band under a band-level intent
  machine (LURK / AMBUSH with hold-fire discipline / HARASS hit-and-run /
  RAID-and-linger / SCATTER only under 30% strength), casualty-maximizing
  targeting (human commander > wounded > isolated), and hidden traps/mines
  (route/approach placement, CASUALTY-style "HIT A DEVICE" broadcasts,
  oracle-visible from step 0, provably absent from blue observations).
  Environment-side only: spaces frozen at Discrete(157)/Box(137), v1.4/v1.5
  checkpoints load unchanged. Asymmetric terminal semantics for DEFEND vs. a
  band: success = band destroyed OR scattered with contact fully broken,
  objective held. **DoD (≥70% each at N=100): met** — `patrol_brique_v1`
  **99% ± 2**, `defend_brique_v1` **87% ± 7**; oracle-verified behavior
  shifts in the progress log.
- `[ ]` **A3. Self-play OpFor** — a second cohort (own org chart, own OPORD) replaces
  the scripted enemy; alternating or league-style training.
  **DoD**: red-vs-blue episodes render in the dashboard with both transcripts; blue
  policy trained vs. self-play beats the scripted-garrison-trained policy head-to-head.
- `[x]` **B2. Behavioral metrics suite** — measure what "behaves like its rank" means:
  obedience latency (order → first compliant action), report precision/recall
  (contacts reported vs. enemies actually seen), doctrine-preference rate,
  false-COMPLETE rate, succession recovery time, subordinate coverage time.
  **DoD**: emitted per eval run (JSON + dashboard panel); tracked in training metrics.
  *(done 2026-08-06: `cohort/metrics.py` + evaluate `--behavior` (default on,
  `behavior.json` + printed table) + dashboard Behavior card + training
  columns `human_death_rate`/`false_complete_rate`; human-exposure block
  added per the #9 finding. Definitions + the published-checkpoint baseline
  (N=30, seeds 500–529): `docs/metrics.md`. See the progress log.)*
- `[x]` **B3. Hierarchy ablation** — same parameter count, three arms: (i) full
  hierarchy + masked doctrine, (ii) hierarchy without doctrine masks, (iii) flat team
  with free comms and no ranks.
  **DoD**: sample-efficiency and final-success comparison across ≥3 seeds per arm,
  written up in `docs/ablation.md`. This is the publishable claim if it holds.
  *(done 2026-08-06 — 9 runs from scratch on squad, 2.5M steps each. The claim
  holds for final performance vs flat (0.92/0.91 vs 0.85 at N=100; flat wipes
  2.2×) and for interpretability (100% vs 40% doctrine-valid net traffic;
  completion reporting survives only under masks), and for efficiency only
  WITHIN hierarchy (masks: 436 ± 30k to sustained-80% vs 583 ± 182k without) —
  the flat all-tasked team is fastest to 80% (310k) on this 7-agent scenario.
  Full write-up + curves: `docs/ablation.md`; see the progress log.)*
- `[x]` **B4. Transparency probe** — can a reader predict behavior from the net alone?
  **DoD**: a scripted probe (show transcript-so-far, predict each agent's next
  destination/posture; measure accuracy) — a proxy for the founding promise that
  the command language explains the behavior.
  *(done 2026-08-06, and the promise is honestly **half-kept**: `cohort/probe.py`
  — a deterministic net-following rule engine scored per (step × living agent)
  over all 8 published checkpoints (N=30, seeds 500–529, K=15) against
  majority + random baselines. Posture beats random everywhere (+0.07…+0.37);
  destination LOSES to the OPORD-only majority baseline on every checkpoint
  (−0.16…−0.43) — leaf-level order traffic churns objectives faster than
  execution and does not bind behavior. Where anchors are stable the net does
  explain behavior (defenses 0.55–0.60; the defend-BRIQUE TL at 0.99), and
  FIRING predictability tracks CONTACT discipline exactly (0.57 vs 0.02 recall
  on the two defenses). Full method, tables, failure modes, and candidate
  fixes: `docs/transparency.md`; see the progress log.)*
- `[x]` **B5. Binding orders by economics** — implement the B4 fix candidates:
  orders must BIND (an order should be the best predictor of its recipient's
  near-term behavior, because changing it is expensive). Rank-scaled re-task
  pricing (`order_retask_cost_base × (1 + rank_scale × authority)`; half price
  for same-anchor type changes; waived when the tactical picture changed:
  CONTACT on the net / element casualty / new superior intent / the
  subordinate's truthful DONE) + standing-order tenure (compliance credit
  ×(1 + 0.5·min(held,40)/40); `success_team` 45 → 60 keeps terminal dominance)
  + the campaign's one diagnosed adjustment (`coverage_gap` −0.02 → −0.1: the
  first retrains showed pricing suppressing *initial* tasking — an order never
  issued cannot bind). Retrained squad/fireteam from scratch + patrol-BRIQUE
  fine-tune; every re-task logged by the env and reported per rank in B2.
  **DoD**: (a) N=100 within 5 pts of published — **met** (fireteam 78/83,
  squad 82/84, patrol 99/99); (b) probe destination accuracy beats the
  OPORD-majority baseline — **missed on all three** after one retrain + one
  diagnosed adjustment each, documented honestly: the churn mechanism is dead
  (squad re-tasks 58.8 → 9.6/ep, patrol rotations 1364 → 1/30 eps; fireteam
  accuracy 0.31 → 0.54, gap −0.163 → −0.065) but the residual error is
  vocabulary (formation-keeping/untasked drift/route doglegs have no radio
  form → A5), and the majority baseline rises with the fix. See the progress
  log and `docs/transparency.md` §B5.
- `[x]` **A5. Richer order vocabulary** — *(owner scope 2026-08-06: NO
  FOLLOW-ME order — rejected; ATTACH/DETACH out of scope. In: control
  measures + ADVANCE, order timing, formations, trinôme sync; one breaking
  cycle, all scenarios retrained.)* Implemented as A5-1..A5-5: named
  waypoints/phase lines with `ADVANCE TO WP/PL <X>`; `AT T PLUS n` /
  `AT MY COMMAND` + EXECUTE staging; `FORMATION COLUMN|LINE|WEDGE` element
  stances (manual pp. 14-15), reward-shaped, never forced; SYNC_PROPOSE/GO
  voice bounds with covered-movement debuff. BREAKING: Discrete 157 → 228,
  Box 137 → 166. **DoD met**: round-trips for every new form, doctrine +
  masks extended, and every retrained scenario exercises the new orders
  (platoon: 38.6 ADVANCE + 48.7 FORMATION orders/ep, stance-governed 73%
  of steps, 101 sync bounds/ep). The A5-5 stretch target — probe beating
  the OPORD-majority baseline on ≥2 of the three B5 scenarios — was
  **missed on all three** (documented honestly in
  `docs/transparency.md` §A5); squad/patrol gaps still improved to the
  best ever measured. See the progress log.

## Milestone v1.11 — Directional vision (breaking cycle)

Full design, reasoning, and decisions of record: **`docs/vision.md`**. Owner's
sequencing call (2026-08-06): **v1.10 ships fully first** — all 8 scenarios
retrained and published — and only then does this cycle open. The fleet retrain
is therefore paid twice on purpose, buying an uncontaminated v1.10 verdict on
`human_death` → 0.0, the prep period, and the false-COMPLETE fix.

The case for the feature is **not** realism. It is that 360°/10-cell vision in
open terrain makes every agent's picture nearly common knowledge, which is a
candidate cause of the standing transparency-probe failure: a CONTACT report is
informationally close to a no-op when the recipient already sees the contact.
Arcs manufacture information asymmetry structurally — two soldiers *on the same
cell facing different ways* see different worlds.

Binding constraint: `PolicyNet` (`training/ppo.py:55`) is a memoryless MLP, so
every phase below must keep the observation near-Markov. Recurrence is
deliberately deferred (`docs/vision.md` §2c).

- `[~]` **V0. Information-asymmetry probe** — gate for the whole cycle, runs
  *before* the v1.10 publish since it is independent of it. Scenario
  `squad_short_vision` (registered): the squad with vision halved (10 → 5,
  forest 6 → 3, ratio preserved) and nothing else touched. **DoD**: trained
  under v1.10 spaces, `cohort.probe` gap reported against the OPORD-only
  baseline *of the v1.10 `squad` control run* (not the v1.9 published number),
  written up in `docs/transparency.md`. Informative, not blocking — isotropic
  reduction is a lower bound on the arc effect (§6).
- `[ ]` **V1. Foliage attenuates LOS** — optical depth accumulated along the
  existing Bresenham walk; `foliage_density = 0.5` reproduces today's
  single-cell forest penalty to within 0.07 cells (`exp(-0.5) = 0.6065` vs the
  current 6/10), so only rays *through* woods change. No new state, **no space
  break** — old checkpoints still load. **DoD**: monotonicity test (effective
  range decreases in forest cells traversed), walls still hard-block, endpoints
  still never block; `forest_vision_range` kept in `briefing()` as a derived
  value. **Owner call outstanding**: `prep_in_position` pays for cover
  occupancy, but under attenuation "in cover" and "has fields of fire" become
  competing — a reward that pays only for occupancy trains a policy that goes
  blind in deep woods.
- `[ ]` **V2. Facing and vision arc** — BREAKING: Box 220 → 244 (facing one-hot
  4, `in_fire_arc` per enemy slot 4, remembered contacts 4×4), Discrete 228 →
  232 (`FACE_*`). Decided semantics: `vision_arc 180°`, `fire_arc 90°`,
  `all_round_awareness_range 2.0`, 4-dir facing. Movement sets facing; `FACE_*`
  consumes the step — so an individual **cannot advance while covering a
  flank**, which is what makes distributed sectors necessary. Ships with **V4
  (OpFor symmetry)** in the same cycle, never after. **DoD**: the five
  regression-hazard tests of `docs/vision.md` §4 green — above all the
  **stale-track invariant** (a remembered contact stores the last-seen position
  and never updates while unobserved; anything else is omniscience wearing a
  memory costume); fleet retrained and republished; `sector_coverage`,
  `flank_exposure_rate`, `detect_latency`, `facing_changes_per_step` in the
  behavior suite. Use `facing`/`sector` vocabulary, **never** `rotation` — that
  already means patrol-anchor rotation here (the v1.8 economics result).
- `[ ]` **V3. Sector-of-fire orders** — `ORDER_S{i}_COVER_SECTOR_{N|S|E|W}`,
  +16 actions. The C2 payoff: all-round defense becomes a *commanded* act on the
  transcript rather than an emergent accident, and the most direct test of the
  hypothesis above. Follows V2 so that "arcs work" and "commanding arcs helps"
  stay separable findings. Vocabulary expansion — owner's call before build.
- `[ ]` **V5. Assurance contract amendment** — `briefing()` publishes scalar
  `vision_range` / `forest_vision_range` (`config.py:496-497`) so the external
  layer can define "under threat". Directional vision silently turns that into
  an *overestimate* and the threat envelope quietly loosens with no error to
  show for it — the exact failure `briefing()` was built to prevent, recurring
  one level up. **DoD**: arc parameters published alongside the ranges;
  amendment recorded in `ASSURANCE-SYNC.md`. Blocks the V2 publish.

## Backlog (unscheduled)

- `[ ]` **C2. Voice input** — speech-to-order through the existing parser (grammar is
  small; feasible offline with whisper-class models).
- `[ ]` **D2. PyPI packaging** — `pip install cohort-marl`; entry-point scripts
  (`cohort-train`, `cohort-dashboard`).
- `[ ]` **D3. Config sweeps** — seed sweeps + reward-weight sensitivity via a simple
  runner (no framework; a shell loop + run-name suffixes suffices).
- `[ ]` **A6. Medic/auxiliary roles** — the legacy project's auxiliary-role idea:
  a MEDIC tag with a stabilize action; casualty play on the net (MEDEVAC request).
- `[~]` **D4. PPO stability guard** — four converged-policy collapses observed
  (squad continuation @3e-4; platoon_v1 transient dip; squad_recon_v1 and _v2
  terminal collapses @1e-4). Target-KL early stop implemented (PPOConfig.target_kl,
  default 0.02). **DoD outstanding**: a platoon-curriculum rerun with no
  rolling-success dip below 70% (the recon reruns were confounded by A7).
  **v1.4 update — now the project's dominant open problem**: four MORE
  collapse/oscillation events in the P6 campaign (fireteam_v4 late dip
  91→57%; squad_recon_v3 terminal @0.4M; squad_v3 terminal @0.7M;
  squad_v3b terminal @1.5M despite ent 0.02 + lr 2e-4; fireteam_defend_v5
  oscillated 0–79% for 2M steps). KL stayed < 0.01 through every collapse —
  the target-KL guard does not catch them. New evidence: collapse onsets
  coincide with human-commander death bursts (−25 × n_agents in one step;
  comp_combat spikes ≈ −0.05/agent-step at the squad collapse onset) —
  suspect the correlated catastrophic-penalty shocks destabilize the value
  function. Candidate fixes for the D4 rerun: value-loss clipping, reward
  normalization, larger batches, or spreading the human-death penalty over
  several steps.
  **A4 update — four MORE events, now in fine-tunes too**: squad_v3d
  (lr 1e-4 fine-tune from a converged parent, fresh Adam state) decayed
  90→0% inside 0.1M with value_loss 10–20 from the first iterations;
  squad_v3e (gentler: lr 5e-5, ent 0.003) held 0.81–0.90 for 0.6M then
  collapsed terminally, with the death-shock signature clearly visible
  beforehand (comp_combat −0.02 → −0.06/agent-step building over 300k
  steps pre-onset — the strongest evidence yet for the shock hypothesis);
  fireteam_v4d dipped 0.93→0 at ~0.6M and self-recovered to 0.87;
  fireteam_v4e (5e-5/0.003) collapsed terminally at ~0.7M. Also learned:
  the rolling-best checkpoint tracker is **degenerate for fine-tunes** —
  the strong parent pins rolling at ~1.0 over the first window, so
  ckpt_best freezes at ~3–4k steps; the A4 deliverables had to be selected
  by N=100 eval instead (fireteam: final checkpoint; squad: a genuine
  0.94-rolling peak at 51k). A D4 fix should gate best-saving on a full
  window or an eval probe.
  **2026-08-06 update — the rolling-best degeneracy is FIXED** (commit
  2f42441): ckpt_best is only written once the 100-episode outcome window
  has fully turned over (`best_save_gate`, episodes_seen >= maxlen).
  Verified in the wild on the first post-fix fine-tunes: defend_brique_v1's
  metrics show the parent pinning rolling at 1.0 in the very first window
  (the exact failure), yet ckpt_best was saved at its genuine 2.1M peak
  (87% ± 7 at N=100). The underlying collapse/oscillation problem remains
  open (patrol_brique_v1 oscillated 0.06–0.94 for ~1.5M steps before
  converging, value-loss spiking to ~30 at each dip — consistent with the
  death-shock hypothesis, and BRIQUE bands *deliberately target the human
  commander*, making these fine-tunes a worst case); the D4 rerun DoD
  still stands.
  **fireteam_defend diagnosis update (2026-08-06)**: with the defend fire
  economics repaired (defense-of-the-position carve-out, commit 9519326)
  and the TL fire pathology verifiably eliminated in fireteam_defend_v7
  (TL p(fire | threatened) 0.005 → 1.000), the scenario STILL oscillates
  0.31–0.55 rolling for 3M steps with value_loss 15–95 and human-death
  bursts 0–0.86 — the shock instability is now the leading suspect for
  the defend miss, no longer confounded by the reward hole.
- `[x]` **A7. Stealth-recon economics** — negative result from A2: under strict
  weapons-tight (no combat pay on RECON), the squad_recon policy *abandons the
  task* — subordinates park on OVERWATCH at 8–9 cells farming posture compliance
  without ever triggering the ≤7-cell observation that ends the episode, because
  γ-discounted terminal success loses to safe indefinite shaping (the stall
  exploit through a mission-posture side door; the undiscounted dominance bound
  doesn't cover it). Candidate designs: pay observation *progress* (per novel
  team-observe step, telescoping, completion-bounded), count subordinate
  OVERWATCH-with-LOS toward team observation, or a posture-compliance budget per
  mission. **DoD**: a squad_recon run ≥80% at N=100 where RECON-holders fire
  <0.01/agent-step (oracle-verified), no stalling.
  *(resolved by the v1.4 redesign: the stealth task moved to SCREEN
  (ÉCLAIRER) with `RewardConfig.observe_progress` — the telescoping
  observation-progress payment proposed here. squad_screen_v1b: 93% ± 5 at
  N=100, no stalling; unprovoked fire 0.0025/agent-step < 0.01; total incl.
  riposte-while-detected 0.016 — ÉCLAIRER doctrinally sanctions riposte
  (manual p. 32), so the strict bar is met for unprovoked fire only. RECON
  itself now *may engage* per doctrine, so the old DoD no longer applies
  to squad_recon.)*

---

## Working agreements

- Every completed item gets a progress-log entry (date, run/commit, one-line outcome).
- Trained results are only claimed with N≥100 eval episodes once B1 lands.
- Reward changes must keep `test_terminal_dominates_stalling` and the churn regression
  tests green — both exploits were found the expensive way.
- New order forms must keep the formatter/parser round-trip property.

## Progress log

- **2026-08-05** — v1.0.0: ground-up rewrite (env, masked PPO, language, succession,
  dashboard-less). Fireteam + squad trained. Two reward exploits found and fixed
  (order-spam farming; stall farming — terminal-dominance invariant).
- **2026-08-05** — interactive dashboard shipped (training charts + episode explorer).
- **2026-08-05** — v1.1.0: NATO nomenclature/voice procedure/APP-6 symbology;
  `.gitignore` bug fixed (cohort/env was never committed — fresh-clone verified);
  merged to `main`.
- **2026-08-05** — B1 done: eval default N=100 with 95% CI. Honest numbers:
  fireteam_v2 90% ± 6, squad_v1 89% ± 6 (earlier 95%/80% were N=20 noise).
- **2026-08-05** — D1 done: GitHub Actions CI (pytest + ruff, CPU torch).
- **2026-08-05** — A1 started: `platoon_v1`, 6M steps, curriculum from
  squad_v1/ckpt_best at lr 1e-4.
- **2026-08-05** — A1 **done**: platoon_v1 hit **89% ± 6** (N=100), 11.8/16 mean
  survivors; full HQ→PL→SL→TL→RFN cascade within ~16 steps of the OPORD.
  Curriculum transfer was instant (>90% rolling inside 200k steps). Observed a
  transient mid-training collapse (94%→52% around 1.8M steps, self-recovered as
  LR annealed) — second instability of this kind, motivating **D4** below.
- **2026-08-05** — dashboard Training-tab fix: delegated run-list clicks
  (auto-refresh was destroying per-item handlers), hidden-tab render guard.
- **2026-08-05** — assurance-review fixes (external review, issues #3–#8):
  net hygiene (CASUALTY from HQ, succession text fully in `language.py`,
  per-episode reproducible eval seeding, public `env.outcome`, `auto_ack`
  knob, order-mask cooldown — refs #8); MISSION COMPLETE verdicts answered
  on the net (`DONE_CONFIRM`/`DONE_REJECT` — refs #4).
- **2026-08-05** — **#3 fixed: the cohort now reports MISSION COMPLETE.**
  Completion-report grace window (`ScenarioSpec.grace_window=12`): success
  locks in at T0 but the episode stays open for the root's report, which ends
  it (+`root_done_bonus`); the root's OPORD claim is judged against the *team*
  success condition. Retrained: `fireteam_v3` **86% ± 7** (N=100; 67/86
  successes end with the root's COMPLETE + HQ confirmation, ~5.5 DONE/ep) and
  `squad_v2` **97% ± 3** (N=100; 44/97 end with the report, ~5.6 DONE/ep).
  Old checkpoints keep their numbers under the new env (fireteam_v2 91% ± 6,
  squad_v1 96% ± 4, platoon_v1 93% ± 5 — all ≥ their README claims), with 0
  DONE reports, as measured by the assurance layer. Note: `squad_v2`'s
  rolling-best checkpoint was saved before a mid-run collapse (D4, observed
  again in both runs) and predated the reporting behavior — `ckpt_best` was
  re-pointed to the final checkpoint after N=100 evaluation of both
  (best-at-save 91% ± 6 / 8 report-endings vs final 97% ± 3 / 44).
- **2026-08-05** — oracle observer + payloads forbidden (owner decision): the net
  carries voice-procedure text only; `env.oracle()` exposes ground truth incl.
  OpFor internals and behavior observables, provably invisible to the cohort.
- **2026-08-05** — A2 campaign (six runs, four design findings):
  1. *Fire discipline* — oracle diagnosis showed combat pay dominating mission
     compliance (RECON elements out-shot OVERWATCH 149:82; defenders died 32:5
     away from the objective chasing kills). Fix: `RewardConfig.fire_discipline` —
     weapons-tight RECON, position-anchored combat pay for static postures.
  2. *D4 implemented* — target-KL early stop after recon collapses #3 and #4.
  3. *Terrain doctrine* — a defense needs defensible ground: `objective_cover`
     + `assault_spawn_min_dist=14` took fireteam_defend from a ~55-60% plateau
     (three runs: curriculum 46%, scratch 59% ± 10, scratch+discipline 58% ± 10)
     to ~90% rolling (`fireteam_defend_v4`). Likewise `observation_concealment`
     guarantees concealed OPs for recon.
  4. *Negative result (→ A7)* — strict weapons-tight induced task abandonment on
     squad_recon (v3/v4 degraded 100%→0% by posture-farming outside the trigger
     radius). Published recon result: `squad_recon_v2` **89% ± 6** (N=100), whose
     policy still fires on RECON (~0.05 shots/agent-step) — success DoD met,
     stealth DoD deferred to A7.
- **2026-08-05** — A2 **done**, v1.2 complete: fireteam_defend_v4 **91% ± 6**
  (N=100, zero defeats, 3.8/4 survivors; oracle: all deaths at the objective) via
  terrain doctrine; squad_recon_v2 **89% ± 6**. Old A7 reframed by doctrine.
- **2026-08-05** — `docs/manuel-proterre.pdf` adopted as the doctrinal source
  (MICAT missions, BRIQUE threat model, group/section/company mission menus,
  reaction drills). Owner decisions: full MICAT set, SUPPORT as unit-targeted
  mission, English names with PROTERRE semantics, one breaking v1.4 cycle
  (missions+support+humans+rank-weighted deaths+maps ×1.5); BRIQUE OpFor and
  buildings deferred to v2.0.
- **2026-08-05** — **P1 done** (commit fd5c355): full MICAT set — RECON/
  SCREEN/OBSERVE/SUPPORT/COVER/DEFEND/DENY/SEIZE/CLEAR/RALLY/HOLD (OVERWATCH
  removed), per-echelon admissibility (DENY: authority ≥ 2, mask +
  inject_order), doctrine rebuilt, language round-trips for every mission
  incl. unit-targeted `SUPPORT <callsign>`; `observe_progress` (+0.3,
  telescoping to the 10-step counter) closes the A7 stall economics;
  success_team 25→45 / success_speed 10→15, dominance re-proven at the
  600-step cap. BREAKING: Discrete(97)→157, Box(131)→135. `docs/missions.md`
  with manual page refs (pp. 8, 29–38).
- **2026-08-05** — **P2 done** (commit 6e4cb29): SUPPORT mechanics — covered
  movement (attacker accuracy ×0.7 inside the in-position supporter's 8-cell
  umbrella) + focus fire (follow-up shooters ×1.15, cap 0.95) via a
  `modifier` argument on `resolve_fire`; oracle `supporting`/`supported`
  tags; fixed-seed RNG-parity tests (29 vs 47 hits over 60 volleys).
- **2026-08-05** — **P3 done** (commit f6cda3d): human agents —
  `Soldier.human`, `root_human=True` on every preset,
  humans-outrank-all-non-humans validated at build, +2 obs fields
  (Box→137), `human_death=-25` paid by every present agent, gold-ring
  visuals in dashboard/renderer.
- **2026-08-05** — **P4 done** (commit ef7cf8f): rank-weighted casualties —
  death & teammate_death × (1 + 0.25 × the fallen agent's *effective*
  authority); an RFN costs ×1.0, a PL ×2.0.
- **2026-08-05** — **P5 done** (commit 36a1885): maps, coordinates, and step
  budgets ×1.5 across every preset (24→36, 28→42, 36→54; caps up to 600);
  early-warning distance 14→21; new `squad_screen` scenario (ÉCLAIRER, 3
  enemies, concealed OPs, 375 steps).
- **2026-08-05/06** — **P6 retrain campaign** (v1.4 spaces, fresh nets, KL
  guard on; all evals N=100 sampled):
  1. *fireteam_v4* (2.5M, seed 1) — **92% ± 5** (v1.2: 86 — DoD ✓). Late
     D4-style dip 91→57% rolling after 2.3M; ckpt_best saved at the peak.
  2. *fireteam_defend_v5* (3.5M, seed 12) — **73% ± 9** (v1.2: 91 —
     **DoD ✗ by 13 pts**, documented). First attempt abandoned the
     objective: the oracle showed enemies parked ON the objective at full
     health while defenders farmed location-free SUPPORT/HOLD posture
     compliance 25 cells away (flight beat fighting under human/rank death
     economics). Diagnosed adjustment: `RewardConfig.objective_lost` (−0.05
     per living agent per step while a living enemy stands on a DEFEND/DENY
     root objective; pure penalty, farm bound unchanged) → relaunched from
     scratch, rolling peaked 79% but oscillated 0–79% for the rest of the
     run (D4). Oracle: deaths back at the position (14/22 within 6 cells,
     all within 10; none in flight). Retrain + adjustment both spent.
  3. *squad_v3* (3M, seed 3) — collapsed 90→0% at 0.7M (collapse onset
     coincides with human-death penalty bursts); ckpt_best **85% ± 7**.
     Diagnosed rerun *squad_v3b* (ent 0.02, lr 2e-4): survived its first
     dip (84→45→recovered), collapsed terminally at 1.5M; ckpt_best
     **84% ± 7** with much richer completion reporting (15.3 vs 0.6
     DONE/ep). **DoD ✗** (v1.2: 97): published squad_v3b, both runs kept.
  4. *squad_recon_v3* (3M, seed 13) — collapsed 88→0% at 0.4M (fourth
     recon collapse); ckpt_best **88% ± 6** (v1.2: 89 — DoD ✓).
  5. *squad_screen_v1* (seed 17) — converged ≥93% within 200k; stopped at
     1.15M (converged 700k, collapse risk) → exploration-anneal
     continuation *squad_screen_v1b* (1M, ent 0.003, lr 1e-4, init-from):
     **93% ± 5** (DoD ≥80 ✓). Oracle fire discipline over 30 eps:
     unprovoked-from-concealment 0.0025 shots/agent-step (< 0.01 ✓);
     total 0.016, 84% of shots are riposte while already detected
     (ÉCLAIRER sanctions riposte, manual p. 32) — strict total bar ✗.
  6. *platoon_v2* (curriculum from squad_v3b/ckpt_best at lr 1e-4, seed 7)
     — **91% ± 6** (N=100, zero defeats, 10.9/16 mean survivors; v1.2: 93 —
     DoD ✓). Curriculum transfer across the space break was instant (>80%
     rolling within 25k, 93% by 600k); the transcript cascades
     HQ→PL→PSG/SL→TL→RFN within ~5 steps, with SUPPORT taskings at every
     echelon (`PSG1, THIS IS PL1: SUPPORT SL1. OUT.`). **Budget deviation**:
     the planned 7M was stopped at ~0.7M — wall-clock infeasible on this
     machine (~45 env-steps/s for the 16-agent 54×54 env ⇒ 7M ≈ 39 h) and
     the policy had converged; rolling was already dipping (93→43%) when
     the run was killed, so the stop also dodged a collapse in progress.
  Deviations from the plan: defend restarted once with the diagnosed
  objective_lost fix (its budget re-run in full); squad_recon/squad/screen
  stopped early after terminal collapse or long-converged plateaus
  (rolling-best checkpoints unaffected); squad rerun squad_v3b added;
  platoon launched from squad_v3b's best snapshot while squad_v3b was
  still training. SUPPORT verified in live transcripts
  (`TL1, THIS IS SL1: SUPPORT TL2. OUT.` in the squad eval); dashboard
  traces regenerated per scenario with humans/missions/SUPPORT rendering.
- **2026-08-06** — v1.4.0 tagged and merged to main (P1–P6). C1 done: dashboard
  Command tab — live sessions with any checkpoint, orders typed on the net
  (HQ or commander callsigns), pause-on-CONTACT; DoD flow verified via the
  live API (WILCO + rank-violation rejection).
- **2026-08-06** — **A4 mechanics** (commit b9ffe3e): single-frequency net —
  one learned transmission per tick, deterministic priority arbitration
  (CONTACT > DONE > orders > SITREP, ties by agent order), losers dropped
  with a NET BUSY outcome (no cost, no effect; flagged in infos + oracle);
  `transmission_cost` −0.01 on every emitted learned transmission
  (auto-traffic free); CONTACT dedup — first accurate report pays
  `contact_new`, a refresh of intel aged ≥ `contact_refresh_age` (20) earns
  exactly 0, all-fresh re-reports draw `contact_redundant`. Busy-ness stays
  global under `comm_model="range"` (one frequency). Spaces frozen
  (157/137, asserted) — v1.4 checkpoints load unchanged. 178 tests.
- **2026-08-06** — **A4 retrains + republication** (fireteam ≥89 bound ✗ 83,
  squad ≥81 bound ✓ 84; traffic numbers over 20 sampled eval episodes,
  before → after):
  1. *fireteam*: `fireteam_v4d` (1.5M @ lr 1e-4 from fireteam_v4) dipped
     0.93→0 rolling at ~0.6M, self-recovered to 0.87; final checkpoint
     **83% ± 7** (N=100) — bound missed by 6. Diagnosed rerun
     `fireteam_v4e` (lr 5e-5, ent 0.003 — the squad_screen_v1b recipe)
     collapsed terminally at ~0.7M (watchdog-stopped; kept for the D4
     record). Retrain + adjustment spent → v4d's final checkpoint
     published (`ckpt_best` re-pointed to it; the rolling-best tracker is
     degenerate for fine-tunes, see D4). Traffic: SITREPs/agent/25 steps
     0.50 → **0.98** (≤1 ✓); CONTACTs 22.6/ep → **2.5/ep**, duplicate
     rate 0.92 → 0.51 (no storm left); transmissions/agent-step
     0.178 → **0.088**.
  2. *squad*: `squad_v3d` (2M @ lr 1e-4 from squad_v3b) decayed 100→0%
     inside 0.1M (D4; stopped early, kept). Diagnosed rerun `squad_v3e`
     (lr 5e-5, ent 0.003) held 0.81–0.90 for 0.6M, then collapsed
     (watchdog-stopped); its genuine 0.94-rolling peak at 51k steps
     published: **84% ± 7** (N=100) — bound met, zero regression vs the
     pre-discipline parent. Traffic: SITREPs/agent/25 steps 2.92 →
     **0.74** (≤1 ✓); transmissions/agent-step 0.256 → **0.134**;
     duplicate-CONTACT rate 0.85 → 0.83 (**A4 residual**: only 51k
     discipline steps — the dedup economics need the ~1M+ steps the
     fireteam run got to erase re-reporting; volume still capped at one
     transmission per tick by arbitration).
  Deviations: fireteam_v4e stopped at 0.75M/1.5M and squad_v3d at
  0.28M/2M, squad_v3e at 0.7M/2M (terminal D4 collapses; rolling-best
  degeneracy meant nothing further was recoverable from the budget).
  Eval-transcript spot check: fireteam episode carries 1 CONTACT +
  22 SITREPs over ~200 steps; squad cascade unchanged
  (`TL1, THIS IS SL1: SUPPORT TL2. OUT.`), one transmission per tick.
- **2026-08-06** — **D4 rolling-best fix** (commit 2f42441): ckpt_best gated
  on full turnover of the 100-episode outcome window (`best_save_gate`) —
  kills the freeze-at-3k-steps fine-tune degeneracy documented in the A4
  campaign. Bookkeeping-simulation test + an end-to-end assert that short
  runs write no ckpt_best.
- **2026-08-06** — **BRIQUE OpFor mechanics** (commit 56fecb6): the manual's
  armed-band threat (p. 9) as `opfor_mode="brique"` — flat band, band-level
  intent machine (LURK/AMBUSH/HARASS/RAID/SCATTER; hold-fire ambush
  discipline with a compromised-ambush spring; scatter only under 30%
  strength), casualty-maximizing targeting (human > wounded > isolated),
  hidden traps (40 dmg, first friendly only, revealed on trigger,
  `ALL STATIONS: <cs> HIT A DEVICE AT GRID xxyy` broadcast, new
  MessageKind.TRAP), BRIQUE DEFEND terminal semantics (band destroyed OR
  scattered+contact broken, objective held). Oracle: band intent, posts,
  per-member behavior, traps — all enemy-side non-observables; obs
  bit-identical with and without traps (tested). Spaces frozen 157/137;
  v1.5 checkpoints verified loading. 198 → 201 tests with the scenarios.
- **2026-08-06** — **P7 done: BRIQUE scenarios + training** (evals N=100
  sampled, 95% CI; both 3M fine-tunes @ lr 1e-4 under the D4 fix):
  1. *patrol_brique_v1* (from squad_v3e) — **99% ± 2**, 6.4/7 survivors.
     Oracle before → after (30 fixed seeds, parent vs. trained): total
     casualties 3.2 → 0.8/ep; ambushes sprung 29/30 → 15/30 (the patrol
     refuses the kill zone); in-ambush-window casualties 1.2 → 0.8; trap
     casualties 0.63 → 0.0/ep (the mined corridor avoided outright);
     SUPPORT taskings during movement 1.8 → 3.9/ep (bounding starts
     supported at t=11 of the eval transcript). Training oscillated
     0.06–0.94 rolling for ~1.5M steps (D4, worst case by design: the band
     hunts the human commander) before converging 0.96–1.0; ckpt_best at
     a genuine 1.66M peak.
  2. *defend_brique_v1* (from fireteam_defend_v5, whose baseline vs. the
     band was 73%) — **87% ± 7** (final ckpt 88% ± 6; zero defeats
     either way), 3.9/4 survivors. Oracle: casualties 0.43 → 0.13/ep;
     band scattered-or-destroyed in 29/30 episodes (4.1/5 members killed);
     trap casualties 0 → 0 — no sorties into the mined approaches. The
     D4 fix verified in the wild here (parent pinned rolling 1.0 in the
     first window; ckpt_best still saved at the genuine 2.1M peak).
  One design adjustment mid-cycle (before the published patrol run): the
  initial ambush held fire even while being destroyed at standoff (blue
  spots forest-posted members at range 6 > ambush_range 5) — 4/30 baseline
  episodes ever sprung. Added the compromised-ambush spring (any member
  hit → volley); baseline re-measured at 29/30 sprung, 86.7% success.
  Deferred: buildings + pathfinding terrain (the other half of the v1.4
  deferral); dashboard frontend renders revealed traps only (hidden traps
  are trace data, not drawn).
- **2026-08-06** — **#9 fixed: root RECON/SCREEN team-adjudicated — the
  commander commands from cover** (commit 995acba + retrains; external
  assurance finding: root-personal completion adjudication made the human
  root die 9/30 assurance episodes at recon vs 1–4/30 elsewhere, against
  the P3/P4 economics). `Mission.team_observation` on the HQ-issued
  OPORD: completion (is_complete AND the DONE-truthfulness check, which
  #3 had already team-judged) reads the squad's aggregated observation
  counter (`TEAM_OBSERVE_STEPS` = 10, the success condition, env-mirrored
  into the mission), and the root's in-position/compliance credit follows
  the team — subordinates keep personal `observe_steps` (their DONE is
  their own task). Spaces frozen 157/137; 202 → 208 tests. Retrains
  (fine-tunes, 30-ep assurance protocol seeds 500–529 + N=100 sampled):
  1. *squad_screen_v2* (1.5M from squad_screen_v1b @ lr 1e-4, no
     collapse; published ckpt_best @110k) — **92% ± 5** (bound ≥90 ✓);
     human deaths 4→**2**/30, root-COMPLETE endings 26→**27**/30.
  2. *squad_recon_v4* (1.5M from squad_recon_v3 @ lr 1e-4) — terminal
     D4 collapse at ~0.5M (recon's fifth); rolling-best @308k measured
     89% ± 6 but kept parent exposure (deaths 7/30) — kept for the
     record, not published as the fix. Diagnosed adjustment
     *squad_recon_v4b* (ent 0.02 — survived all 1.5M — plus periodic
     snapshots selected by the measured exposure metric): published
     ckpt_best = the 183k snapshot — **85% ± 7** (bound ≥85 ✓); human
     deaths 9(→8 local harness)→**2**/30, root-COMPLETE endings
     20→**22**/30; oracle probe: root ring entries 16→13/30, mean
     min-dist to the objective 7.2→8.8. **Finding**: snapshots later in
     v4b re-learn exposure (deaths 7–9/30 from 350k on) — with
     adjudication no longer *requiring* exposure, RECON's may-engage
     combat pay still *attracts* the root forward; rolling success is
     blind to this, so P3-style metrics must drive checkpoint selection
     for human-preservation claims (B2 candidate metric).
- **2026-08-06** — **B3 done: hierarchy ablation** (228 → 242 tests; commits
  34da151 + the campaign). Arms as env knobs (`ScenarioSpec.ablation`:
  full | nomask | flat; default untouched, spaces frozen 157/137, masking-only).
  Campaign: squad, 3 arms × 3 seeds (3/5/7), 2.5M steps each from scratch,
  identical PPO defaults, all evals on gated `ckpt_best`. Headline
  (mean ± sd across seeds): N=100 success full **0.92 ± 0.01**, nomask
  **0.91 ± 0.03**, flat **0.85 ± 0.06** (defeats 5.0/4.7/**11.0** per 100);
  sustained-80% at 436 ± 30k / 583 ± 182k / **310 ± 80k** — flat is fastest
  to threshold (tasking arrives free at reset) but finishes worst and least
  stable; doctrine masks halve the within-hierarchy efficiency spread.
  Interpretability probe (30 eps/run): doctrine-valid orders 100% (full,
  by construction) vs 33–48% (nomask, incl. 109 orders from unmissioned
  leaders); DONE claims 128 vs ~0 per 30 eps — completion reporting only
  survives under masks. **D4 data point**: 6/6 hierarchy seeds hit a deep
  self-recovered collapse at 0.99–1.37M vs 1/3 flat seeds — collapse
  concentrates in the order-capable arms under identical death economics.
  Honest verdict in `docs/ablation.md`: robustness + interpretability
  claim holds; raw sample efficiency vs flat does not at this scale
  (platoon-depth rerun is the follow-up).
- **2026-08-06** — **B4 done: transparency probe** (262 → 267 tests;
  commits b0edb48 + 08baa23 + the campaign). `cohort/probe.py`: given ONLY
  the transcript-so-far plus static briefing (objectives, spawn, org chart
  — no positions, no oracle), a deterministic rule engine predicts every
  living agent's next-15-step destination (per-objective / LEADER / HOLD)
  and posture (STATIC / MOVING / FIRING): standing orders from ORDER/OPORD
  text, DONE+confirmation clears, succession broadcasts replayed through
  the roster's devolution rules, SITREP/TRAP grids + assumed 1 cell/step
  route progress as the only position model, CONTACT grids as the only
  enemy picture. Scored per (step × agent) on all 8 published checkpoints
  (B2 protocol; ~239k pairs total) vs majority + random baselines; CLI
  `python -m cohort.probe`, results in `runs/<run>/probe.json`. **The
  honest verdict**: posture beats random everywhere (best +0.37); but
  destination loses to the majority baseline — which IS the OPORD
  objective — on all 8 checkpoints (squad 0.25 vs 0.52, platoon 0.14 vs
  0.38 and below random, patrol-BRIQUE 0.24 vs 0.46): **doctrine-valid
  order traffic churns objectives every cooldown window and does not bind
  behavior at K=15** — B3's interpretability claim holds for form, not
  semantics. Counter-evidence that proves the mechanism: stable-anchor
  defenses probe at 0.55–0.60 (defend-BRIQUE's TL 0.99), and FIRING
  predictability tracks CONTACT discipline exactly (defenses: FIRING
  recall 0.57 vs 0.02 at report recall 0.90 vs 0.03). One documented
  calibration pass (closure-based destination truth replacing a
  nearest-anchor definition that measured formation geometry). Write-up
  with failure modes (untasked drift, riposte, no radio form for
  formation-keeping, commander self-preservation off-net) and candidate
  order-economics fixes: `docs/transparency.md`.
- **2026-08-06** — **B5 done (DoD honestly split): binding-order economics +
  retrains** (267 → 282 tests; commits f1a95a5 + the campaign). Mechanics:
  rank-scaled re-task pricing with the tactical-picture carve-out (contact /
  element casualty / superior intent / truthful DONE — the exact exception
  set mirrors the order-cooldown lifts), standing-order tenure on positive
  compliance (H=40, factor 0.5; `max_step_farm` updated, `success_team`
  45 → 60 re-proves dominance at the 600-step cap), identical-reissue stays
  a churn no-op that never restamps tenure; spaces frozen 157/137, masks and
  cooldown untouched; env logs every re-task (priced/excepted + reason,
  rotation vs type change) → B2 rows `orders/ep`, `re-tasks/ep`,
  `retasks_by_rank`. Campaign (all evals N=100 sampled + B2/probe at N=30
  seeds 500–529): first retrains squad_v4 (3M, seed 3, from scratch;
  81% ± 8) and fireteam_v5 (2.5M, seed 1; 82% ± 8) killed the churn
  (re-tasks 58.8 → 0.3 and 21.2 → 2.0/ep) but exposed **tasking
  suppression** — squad coverage time 0.96 → 0.61, a TL2 untasked for 100+
  steps: an order never issued cannot bind. One diagnosed adjustment
  (`coverage_gap` −0.02 → −0.1) → published `fireteam_v5b` **78% ± 8**
  (bound 78 ✓, D4 dip 6% at 1.65M self-recovered), `squad_v4b` **82% ± 8**
  (bound 79 ✓, coverage 0.67, re-tasks 9.6/ep with TL re-orders 91 priced /
  99 excepted), `patrol_brique_v2b` **99% ± 2** (from squad_v4's ckpt; the
  pre-adjustment `patrol_brique_v2` 97% ± 3 kept) — the patrol converged to
  a **silent rush** (60-step episodes, 6.7/7 survivors, 6 orders/ep, ONE
  anchor rotation in 30 episodes). Probe before → after (gap vs majority):
  fireteam −0.163 → **−0.065** (accuracy 0.314 → 0.544, truth-ALPHA
  predicted at 0.87), squad −0.273 → **−0.156** (best-arm −0.098), patrol
  −0.216 → **−0.281**. Probe DoD (beat majority) missed on all three with
  both budgets spent — stopped per protocol and documented in
  `docs/transparency.md` §B5: the majority baseline rises with the fix
  (fireteam OPORD-class truth share 0.477 → 0.609) and the remaining error
  is vocabulary, not churn — truth LEADER accuracy 0.000 everywhere (no
  radio form for formation-keeping), route doglegs close on un-named
  objectives (squad/patrol truth CHARLIE ~0.31–0.37) — pointing directly at
  A5. Predictor and ground truth untouched (the B4 measuring stick stands).
  metrics.md baseline refreshed for all 8 published checkpoints (the five
  non-B5 checkpoints re-swept bit-identically for the new re-task rows).
- **2026-08-06** — **A5 mechanics** (282 → 337 tests; commits 8ae8223 /
  c1d55ae / def4afe / 955ffaa — one breaking cycle, owner scope: no
  FOLLOW-ME, no ATTACH/DETACH):
  1. *A5-1 control measures + ADVANCE* — named WAYPOINTS (GOLD/SILVER/
     COPPER/IRON, standable, objective-like slots in obs) and PHASE LINES
     (AMBER/COBALT/CRIMSON segments; dynamic nearest-point anchor, crossing
     by side flip) on every preset — they name the terrain B4/B5 showed
     routes dogleg through. `MissionType.ADVANCE` (appended; earlier
     one-hot indices stable) with full round-trips
     (`TL1, ADVANCE TO WP GOLD`), doctrine (derivable from RECON/SEIZE/
     DEFEND/DENY; derives ADVANCE/SUPPORT/OBSERVE), rendering
     (dashboard + matplotlib), and probe truth/predictor support
     (CM_REGION 4.0).
  2. *A5-2 timing* — `AT T PLUS n` / `AT MY COMMAND` qualifiers on any
     order (parser+formatter+inject), pending orders stage the recipient
     (compliance = HOLD at the staging spot, DONE masked, tenure restamps
     at release), EXECUTE_SIGNAL broadcasts release all of an issuer's
     staged orders at once (the COMMANDEMENT DU BOND); learned AMC
     variants for ADVANCE orders; pending state observable; probe honors
     timing (staging → target after EXECUTE/T).
  3. *A5-3 formations* — COLUMN/LINE/WEDGE element stances ordered to
     LEADERS (`FORMATION COLUMN`; pp. 14-15 — WEDGE stands in for the
     colonne double per owner scope), persistent, dying with the leader;
     geometry in the leader's heading frame reward-shaped
     (`formation_bonus` 0.03, watermark-gated on the leader's best anchor
     distance so it telescopes — terminal dominance untouched), never
     masked/forced; stance one-hot in obs; probe predicts untasked
     stanced members from their leader (first-ever LEADER-class
     predictions).
  4. *A5-4 trinôme sync* — SYNC_PROPOSE/SYNC_GO by VOICE (voice_range 6,
     never net-arbitrated, no airtime; `Message.voice` flag), peers =
     same element or adjacent trinôme at propose time, 8-step windows;
     synchronized movers closing NEW ground under a COVERING group-mate
     earn `bound_bonus` 0.05 (order-keyed watermark: re-bounding old
     ground pays zero) and the P2 covered-movement debuff vs attackers.
  BREAKING: Discrete(157) → 228, Discrete obs Box(137) → 166; all
  pre-A5 checkpoints incompatible by construction.
- **2026-08-06/07** — **A5-5 retrain campaign + re-probe + republication**
  (all evals N=100 sampled, 95% CI; bounds = previous published − 5;
  KL guard + D4 best-save gate on; probe N=30 seeds 500–529):
  1. *fireteam_v6* (2.5M scratch, seed 1) — **84% ± 7** (bound 73 ✓,
     prev 78). D4 dips at ~1.0M and ~1.55M, both self-recovered;
     ckpt_best at a genuine 0.95-rolling peak (1.16M).
  2. *squad_v5* (3M scratch, seed 3) — **93% ± 5** (bound 77 ✓, prev
     82 — the strongest squad since v1.2). No terminal collapse; peak
     0.98.
  3. *fireteam_defend_v6* (3.5M scratch, seed 12) — **51% ± 10**
     (bound 68 **✗ by 17**, documented). Never stabilized above 0.54
     rolling; oracle: deaths ON the objective (mean 5.2 cells), the
     four-attacker attrition fight is simply lost. Diagnosed adjustment
     *fireteam_defend_v6b* (ent 0.02, full 3.5M rerun) peaked 0.19 —
     retrain + adjustment both spent; the miss stands. ckpt_latest
     measured too (25% ± 8); ckpt_best published.
  4. *squad_recon_v5* (3M scratch, seed 13; stopped 0.71M at its
     signature terminal D4 collapse — flat 0.0 for 50k steps, sixth in
     the scenario's history) — 77% ± 8 (bound 80 ✗ by 3). Diagnosed
     adjustment *squad_recon_v5b* (ent 0.02 — the recipe that survives
     recon) survived all 3M: **94% ± 5** (bound 80 ✓, prev 85) —
     published.
  5. *squad_screen_v3* (2M scratch, seed 17) — **98% ± 3** (bound 87 ✓,
     prev 92).
  6. *patrol_brique_v3* (fine-tune from squad_v5 @1e-4; converged
     0.96–1.0 from the first window, stopped at 0.81M/3M long-converged
     per the P6 precedent) — **95% ± 4** (bound 94 ✓).
  7. *defend_brique_v2* (fine-tune from fireteam_defend_v6 @1e-4;
     stopped 2.05M/3M in a converged 0.81–0.93 band, ckpt_best at a
     genuine 0.98 peak) — **85% ± 7** (bound 82 ✓): the 51% assault
     parent transferred cleanly to the asymmetric-band defense.
  8. *platoon_v3* (curriculum from squad_v5 @1e-4, seed 7; instant
     transfer across the space break — ≥0.9 rolling within 50k; stopped
     0.40M/3M long-converged) — **98% ± 3** (bound 86 ✓, prev 91 — best
     platoon ever, and its net now carries 39 ADVANCE + 49 FORMATION
     orders/ep with 101 sync bounds).
  **Vocabulary adoption** (N=100 eval traffic): ADVANCE 4–39/ep, timed
  orders 2.6–17/ep with EXECUTE releases, FORMATION 13–49/ep with
  stances governing 54–76% of agent-steps wherever orderable, sync
  bounds 6–101/ep. **Probe headline (honest)**: the DoD stretch target —
  destination beating the OPORD-majority baseline on ≥2 of the three B5
  scenarios — **missed on all three** (fireteam −0.196, squad −0.090,
  patrol −0.172 gaps vs majority) under the extended measuring stick
  (control-measure truth classes; disclosed in transparency.md §A5).
  Squad and patrol gaps are the best their scenarios have ever measured;
  the fireteam re-learned churn *through* the new vocabulary (7.1 priced
  re-tasks/ep — pricing is paid, not avoided). Deviations: recon_v5
  stopped early at a terminal collapse; patrol/platoon stopped
  long-converged (P6 precedent); defend spent retrain + adjustment and
  missed; B2 behavior recorded via each run's `behavior.json` at the
  N=100 eval (seed 123) rather than a separate N=30/seed-500 sweep —
  the probe runs carry the seeds-500–529 protocol.
- **2026-08-06** — **fireteam_defend regression diagnosed; fix landed; retrain
  missed — documented, v6 stays published.** Oracle campaign (30 seeded
  episodes, seeds 500–529) over fireteam_defend_v6 (published 51%), _v6b, and
  defend_brique_v2 — the winning defense on the same spaces and economics:
  * **Mechanism demonstrated**: the v6 human TL, under threat with FIRE
    legal, fired on **0.5%** of its opportunities (1/207) vs 0.97 for its own
    RFNs, 0.90 for succeeded leaders, and 0.995 for the brique TL. It
    wandered ~13 cells off ALPHA, died in 26/30 episodes at ~step 28
    (eval human_death_rate 0.77; v5-era: 0.10), absorbed 88 hp/ep of enemy
    fire while contributing nothing — then the −25 × 4 human-death shock
    landed and the remaining three fought the four-attacker assault 3-vs-4
    and lost the attrition: 312 dmg dealt vs the 400 needed, 30% outright
    wipes, comp_combat −72/ep.
  * **Candidates refuted by measurement**: (a) tenure-static posture — the
    WINNING brique defense is MORE static under threat (0.96 vs 0.85), LESS
    in cover (0.22 vs 0.52), and repositions less (0.03 vs 0.30
    cover-to-cover moves/ep); (d) re-task pricing — the winner pays the same
    (−2.3 vs −2.7/ep at 5.9 vs 5.8 re-tasks/ep); (b) terminal economics — no
    ordering flip (win ≫ stall ≫ defeat under success_team 45 and 60 alike),
    and the winner trained under identical B5 economics; (c) catalog
    dilution — real but concentrated on the TL (its order vocabulary grew
    ~130 entries vs 2 for an RFN); v6b (ent 0.02) shows the diffuse-fire
    extreme (0.449 team-wide under threat).
  * **Fix** (commit 9519326): defense-of-the-position carve-out in fire
    discipline — position-anchored fire (OBSERVE/SUPPORT/COVER/DEFEND/DENY/
    HOLD) also pays in full when the TARGET stands inside the anchor's
    engagement envelope (IN_POSITION_RADIUS + weapon_range): fire against an
    enemy assaulting the position is the mission wherever the melee pushed
    the defender. Off-envelope kill-chasing still pays zero — the v1.2 sally
    exploit stays closed (its regression test untouched; 338 → 340 tests).
  * **Retrain missed** — fireteam_defend_v7 (3M fine-tune from
    defend_brique_v2 @ lr 1e-4, seed 12) oscillated 0.31–0.55 rolling for
    the whole run: ckpt_best **35% ± 9**, ckpt_latest 31% ± 9 at N=100
    (target ≥ 68). **fireteam_defend_v6 (51% ± 10) stays published**; v7
    kept on disk for the record.
  * **Oracle post-mortem — the mechanism is verifiably gone, the residual is
    different**: the v7 TL fires at **1.000** under threat, TL deaths
    26/30 → 14/30, team fire 0.997, combat shock halved (−31/ep) — but the
    brique parent's dispersed open-ground disposition transferred with the
    curriculum: cover occupancy under threat collapsed 0.52 → **0.05**, the
    fight happens 9.7 cells off the objective (v6: 4.3) with ADVANCE
    missions holding 48% of threatened agent-steps, and damage output stayed
    309/400. The assault defense needs fire AND the prepared position
    (v1.2's terrain-doctrine lesson); no policy has held both since the A5
    space break. Budgets spent per protocol — the scenario remains the
    honest open miss, with the fire-gradient hole now closed for the next
    campaign and the D4 shock instability the remaining suspect (value_loss
    15–95 throughout v7, human-death bursts 0–0.86).
- **2026-08-06** — **B2 done: behavioral metrics suite** (208 → 228 tests).
  `cohort/metrics.py`: a TraceRecorder rides along eval episodes (reads
  only, consumes no RNG — recorded episodes bit-identical, tested) and
  pure metric functions verified on hand-constructed mini-episodes.
  Emitted per eval run (`evaluate --behavior`, default on: printed table +
  `runs/<run>/behavior.json`), shown in the dashboard Episode sidebar
  (Behavior card, `/api/behavior`), and tracked per training iteration
  (`human_death_rate`, `false_complete_rate` columns). Human-exposure
  metrics (mean dist to nearest enemy, objective-ring entries, death
  rate) added per the #9 finding so checkpoint selection can include
  preservation. Baseline over all 8 published checkpoints (N=30, seeds
  500–529) committed in `docs/metrics.md` + per-run `behavior.json` —
  cross-validates #9 exactly (recon/screen human deaths 2/30 both).
  First honest read: obedience is fast everywhere (0–3.5 steps);
  **false-COMPLETE is the weakest behavior** (53–84% of DONE claims
  rejected as premature wherever DONE is admissible); the pre-#9
  assault fireteam loses its human in 11/30 episodes vs 2/30 for the
  #9 retrains — the exposure gap the suite exists to make visible.
- **2026-08-06** — **human-death shock removed** (`RewardConfig.human_death`
  −25.0 → **0.0**, owner's call; mechanism and knob retained, tested). The term
  paid every present agent −25 in the single step a human died — a correlated
  −100 hit on a fireteam — and every D4 collapse onset measured to date
  coincides with a human-death burst (`fireteam_defend_v7`: value_loss 15–95
  throughout, death bursts 0–0.86). It is the standing suspect for the value
  function destabilisation, and the rank-weighted `teammate_death` already
  expresses a preservation preference in kind. **The trade, stated honestly**:
  a fireteam human TL (authority 1) now costs surviving teammates
  `teammate_death × 1.25` = −0.25 each instead of −25 — a 100× reduction (60×
  for a platoon PL) — so the commander is priced roughly like any casualty and
  `human_death_rate` may rise. Preservation is now **measured, not priced**:
  the #9 exposure block and `human_death_rate` stay instrumented in
  `behavior.json`, the training columns, and the dashboard, so any regression
  is visible in the next campaign's digest. Published checkpoints are
  unaffected (rewards are not part of the observation or action spaces); the
  next retrain measures the effect. 340 → 341 tests.
- **2026-08-06** — **v1.10 breaking cycle opened: observation space Box(166) →
  Box(220)** (Discrete(228) unchanged). Owner's call to spend a space break, so
  everything needing one rode in the same break rather than forcing a second:
  * **tempo block (+2)** — `episode_progress` (step/max_steps, every scenario)
    and `time_to_contact`, the countdown to the announced H-hour of the defend
    preparation period. There was previously **no absolute episode-time feature
    at all**: agents saw only relative times (steps since order, sync window),
    so an "approximately known" enemy arrival was literally unknowable.
  * **nearest-cover vector (+3)** — present/dx/dy to the nearest cover cell
    within `COVER_SEARCH_RADIUS` (8), encoded like objectives and control
    measures. `World.nearest_cover` is pure and tie-broken by (distance, y, x),
    so it consumes no RNG and is scan-order free (determinism convention).
  * **terrain patch 5×5 → 7×7 (+48)** — `PATCH_RADIUS` 2 → 3. At radius 2 an
    agent 5 cells off the objective could not perceive the `objective_cover`
    ring (chebyshev 2 around the objective) it is meant to occupy: the defend
    scenario was paying for ground the policy was partly blind to.
  * **SITREP due-ness gets its own slot (+1)** — it previously overloaded the
    comms "known enemy present" flag purely to keep OBS_DIM frozen (a
    compromise documented at the time); the flag now means what it says.
  * **derived block offsets** (`OFF_SELF`…`OFF_PATCH` + named field offsets)
    exported from `env/observations.py`. Tests indexed the layout with magic
    numbers, so this break broke seven of them for no signal; offsets are now
    computed from the block constants and a future layout change surfaces as
    the `OBS_DIM` assertion instead.
  **Cost, stated plainly**: all eight published checkpoints are unloadable and
  the whole fleet needs retraining. The v1.9 numbers stay published as the
  standing baseline until that campaign runs. 341 → 348 tests.
- **2026-08-06** — **defend preparation period** (`ScenarioSpec.assault_h_hour`,
  owner's design call). `fireteam_defend` now draws its H-hour per episode from
  **(55, 75)** and runs 450 steps (was 375, so the prep is bought without
  shortening the fight). Before H the OpFor is spawned, alive, oracle-visible
  and spottable, but does not move, fire, or advance. The OPORD announces the
  band's **midpoint** on the net — `DEFEND OBJ ALPHA. EXPECT ASSAULT AT H PLUS
  65.` — and that nominal H drives `time_to_contact`, so the arrival is
  approximately, not exactly, known: a defense that waits for the announced
  tick is late half the time, and the habit the scenario pays for is *being
  set early*.
  **Why this and not more reward weight**: the fire team spawns at (17,17) with
  ALPHA at (18,18) — it starts ON the objective, so its problem was never
  reaching the ground. v7 *left* it (cover occupancy 0.05, the fight 9.7 cells
  out, ADVANCE missions holding 48% of threatened agent-steps, inherited from
  the `defend_brique_v2` parent). ~21 steps of warning was just enough to walk
  out and meet the assault in the open. A contact-free phase makes occupying
  the position the only thing worth doing, and makes leaving expensive —
  whoever walks out has to walk back before H.
  Deliberately a **timer, not a new C2 obligation** (the TL owes no positioning
  orders during prep): one variable at a time, so a miss stays diagnosable.
  The draw is guarded, so scenarios without a preparation period consume no
  randomness and reproduce their old seeds exactly. 348 → 357 tests.
- **2026-08-06** — **preparation-period occupancy pay** (`RewardConfig.
  prep_in_position` = 0.05/step, owner's design call — option B2). While the
  assault is still forming up, an agent standing **in cover** within
  `IN_POSITION_RADIUS` of the root objective earns it. The prep phase grants
  the *time* to occupy a prepared position; this grants the *motive*. Without
  it the contact-free phase is a null period a policy can idle through and
  still meet the assault in the open — the v7 failure exactly.
  **Cover is required, not proximity**: bare ground at the objective is not a
  prepared position (the v1.2 terrain lesson). **Not farmable**: it stops
  paying at H, so its lifetime ceiling is 0.05 × 75 = **3.75 per agent**
  against `success_team` 60 — the terminal-dominance regression test now
  carries `prep_cap` explicitly alongside `observe_cap`.
  *Risk on the record*: this is the second variable in the defend cycle
  (prep period + occupancy pay). The B5 precedent says compound changes make
  misses undiagnosable — accepted deliberately here because a timer with no
  motive was judged likelier to teach nothing than to teach the wrong thing.
  If the retrain misses, the oracle should separate them: cover occupancy
  under threat is the prep-period metric, off-objective fight distance the
  occupancy-pay metric. 357 → 360 tests.
- **2026-08-06** — **false-COMPLETE priced and rate-limited** (owner's call).
  B2 measured **53–84% of DONE claims rejected as premature** wherever DONE is
  admissible. The diagnosis before changing anything: a penalty already existed
  (`done_false` −0.5 against `done_true` +1.0), and under it over-claiming was
  **rational, not a training failure** — claiming pays whenever
  `p × done_true > (1−p) × |done_false|`, i.e. **p > 0.33**. A 53% rejection
  rate is p≈0.47, comfortably profitable. Two levers, both applied:
  * **price** — `done_false` −0.5 → **−2.0**, moving the break-even to
    **p > 0.67**. Deliberately moderate, not −9: the B5 precedent is that
    over-pricing a speech act suppresses the *honest* one too, and a cohort
    that stops transmitting DONE never closes the grace window or earns
    `root_done_bonus`. **A mute cohort is a worse failure than an
    over-claiming one.**
  * **structure** — `ScenarioSpec.done_cooldown` = **8**, masking DONE for 8
    steps after a DONE_REJECT. A rejected claim never cleared the mission and
    DONE was admissible on *every* step, so a premature claimant could re-roll
    each tick until one landed. Mirrors `order_cooldown`, the mechanism that
    made orders bind in B5: price the act, rate-limit the retry. Only the
    *retry* is limited — an honest first claim is never delayed.
  New regression-hazard test file (`tests/test_false_complete.py`) encodes the
  muteness hazard explicitly: the honest claim must stay reachable, the first
  claim must never be rate-limited, and the break-even is asserted directly.
  360 → 367 tests.
- **2026-08-06** — **positional regression gate for DEFEND roots** (refs #11,
  external measurement). The assurance layer re-measured the defend family
  from the outside and demolished the handoff's lead clue: `_v7` **halved**
  the root-death rate the ROADMAP had blamed (26/30 → 14/30) and fired on
  essentially every threatened step (p(fire | threatened) 0.005 → 1.000), yet
  success went **14/30 → 12/30**. Human mortality is not the binding
  constraint — `defend_brique_v2` carries the same 14/30 root-death rate and
  still wins 25/30. What separates the record is *where the unit fights*:
  every defend policy that ever cleared its bound fought ≤ 2.9 cells from OBJ
  with cover ≥ 0.79 (`_v5` 24/30 at 0.793/2.90, `brique_v1` 27/30 at
  0.956/1.99); the two that missed sat at 0.496/3.46 and **0.060/9.09**.
  Shipped as a gate rather than a reward change (rewards are the owner's
  call, and the v1.10 prep period is already an untested bet on exactly this
  mechanism):
  * `cohort.metrics` now scores **fight disposition** over the *(living
    soldier, step)* pairs **under threat** — a living enemy within the
    scenario's weapon range. Conditioning on threat is the point: averaged
    over an episode an approach march and a prepared defense look alike.
  * `regression_gates(agg)` fails a DEFEND retrain below **cover 0.40** or
    above **5.0 cells** from the objective. Bounds sit in the empty band
    between the two groups above, so the gate separates every checkpoint on
    record. DEFEND only — an assault is *supposed* to leave its start point.
    An unmeasured gate reports `passed: null`; unmeasured is not a pass.
  * DEFEND runs log `cover_under_threat` / `objective_dist_under_threat` per
    iteration in `metrics.csv` (blank, never `0`, when nothing was
    threatened), so the collapse is visible while the run is still cheap to
    kill. `_v7` spent a 3M-step budget before anyone saw it. No other root
    mission pays for the scan.
  Sanity check on the shipped instrument: the masked-random baseline on
  `fireteam_defend` scores 0.216 / 6.20 and fails both gates. **Not done**:
  no retrain — the fleet is unloadable under the open v1.10 space break, and
  the gate is meant to judge the next defend run, not to be tuned against the
  old ones. 367 → 375 tests.
- **2026-08-06** — **static briefing + SITREP posture** (refs #10, external
  request). The assurance layer's fight-disposition instrument (the companion
  to #11) needed two things this repo was not publishing, and was
  compensating with a hand-maintained coordinate table — silently
  era-sensitive, since `fireteam_defend` moved OBJ ALPHA from (12,12) to
  (18,18), so re-tapping a `_v4`-era checkpoint against today's table gives
  wrong numbers with no error to show for it.
  * **`cohort.config.briefing(scenario)` / `env.briefing()`** — the static
    operations overlay as a JSON-ready dict: objective coordinates by name,
    waypoint/phase-line geometry, map size, spawn, root tasking, the
    doctrinal terrain guarantees (`objective_cover`,
    `observation_concealment`) and the engagement envelope (weapon/vision
    ranges, so an outside monitor can define "under threat" the way
    `metrics.py` does). Pure function of the `ScenarioSpec` — identical
    across episodes, valid *before* `reset()`, which is what makes it header
    material rather than a leak. Read from the scenario a checkpoint names,
    it cannot go stale.
  * **No terrain layer, deliberately.** The grid is regenerated at every
    `reset()` from the episode seed, so no static cover map exists to
    publish; `terrain_static: false` states that in the payload rather than
    leaving a consumer to infer it from an absent key.
  * **SITREP posture clause** — `..., AMMO 24, IN COVER. OVER.` /
    `IN THE OPEN`. Self-reported, exactly like grid/health/ammo: what the
    soldier *says* about its ground, not a readout of the ground. Per-step
    cover stays ground truth in `env.oracle()` and enters the observable
    stream by no other route, while the strongest known correlate of defend
    performance becomes measurable from the transcript alone.
    `language.parse_sitrep` ships with it (inverse of the formatter over the
    fields it formats), so no monitor hand-rolls a regex — the #10 failure
    mode in miniature. A regression test asserts the self-report is
    *truthful*: what a station claims must equal `world.cover_at`.
  Note on the request's wording: it asked for the change in `cohort/tap.py`,
  which lives only on the assurance layer's own `assurance-integration`
  branch and is theirs to edit. This side supplies the data; the header
  writer stays with them. 375 → 391 tests.
- **2026-08-06** — **dashboard: usable again, and organised by doctrine.** Three
  fixes, one of them a regression this session caused:
  * **blank episodes fixed.** Every checkpoint is Box(166); v1.10 made the env
    Box(220), so loading one died inside a forward pass with a torch
    `RuntimeError` the handler did not catch — the request died and the UI
    showed nothing. `checkpoint_meta()` now refuses incompatible checkpoints up
    front with a readable reason, the handler catches everything, and failures
    render in the sidebar.
  * **picker: task → echelon → version** (was one ~100-entry list), all derived
    from `ScenarioSpec` via `scenario_facets()`, with a test asserting
    (task, echelon) stays unique. The threat qualifies the task — `Defend` vs
    `Defend · irregular` — since both are DEFEND at fireteam level.
  * **legacy checkpoints replay without retraining** (`scripts/legacy_trace.py`).
    A shim was never an option: BOTH the observation layout and the action
    indices moved between eras, so a padded observation would silently mean the
    wrong thing. Instead the run is replayed at its OWN release tag in a
    throwaway worktree and written out as plain JSON — a trace is data, not a
    model, so it survives every future space break. `ERA_REF`: 137 → v1.8.0,
    166 → v1.9.0, 220 → in-process. Traces are gitignored (deterministic from
    tag+scenario+seed, ~0.5 MB each).
  * **chain-of-command panel** under the episode: the org chart as it stands at
    the current step (succession re-parents it live), each station's standing
    order and the step it was issued, health, and click-through to the
    inspector. 393 → 395 tests.
- **2026-08-06** — **the OPORD's assault estimate reaches a monitor** (refs #12,
  external request). This session's own preparation period (`75cc51a`) added
  the first forward-looking clause the net has ever carried — `EXPECT ASSAULT
  AT H PLUS 65` — and then dropped it at the boundary: said once in the OPORD,
  recoverable nowhere. It is the substrate for a class of property the
  assurance layer could not previously express (time-bounded readiness: "by H
  minus k, is every station set?"), with the deadline named on the radio
  rather than supplied by hand.
  * **`language.parse_opord(text)`** — inverse of `format_opord` over exactly
    the fields it formats: `{recipient, mission, objective,
    announced_assault_step}`, `None` for traffic that is not an OPORD. Same
    remedy as `parse_sitrep` for #10: a monitor reads the clause back through
    the language module instead of hand-rolling a regex that goes stale when
    the wording moves. `parse_order` still returns the identical tasking with
    or without the clause — the round trip covers the clause now, not just
    the task statement.
  * **`briefing()["announced_assault_step"]`** — the same number as header
    material, so a corpus that predates the clause (or a listener that never
    heard the single un-repeated broadcast under `comm_model="range"`) still
    gives a monitor the deadline. It is a pure function of the scenario, so
    it stays header material rather than per-episode state.
  * **One definition of "what HQ announces"**: `config.announced_assault_step`
    (the band's midpoint). The radio wording, the `time_to_contact`
    observation and the briefing all read it, so the three cannot drift —
    the #10 failure mode again, one level up.
  * **The actual arrival stays hidden, on purpose.** Only the NOMINAL hour is
    published; the per-episode draw goes to `env.oracle()` as
    `actual_assault_step` (next to the announcement, so a consumer can score
    one against the other without re-reading the transcript). The arrival
    band is deliberately *not* in the briefing either: the spread between
    announcement and arrival is precisely what an outside monitor should
    characterise from behaviour. Tests assert no observable payload and no
    message ever names the drawn step — the transcript's only `H PLUS n` is
    the nominal 65, before and after H.
  Same boundary note as #10: the request asked for the parse in
  `cohort/tap.py`, which exists only on the assurance layer's
  `assurance-integration` branch and is theirs to edit. This side supplies
  the data and the parser; the payload writer stays with them. 395 → 402
  tests, ruff clean.

- **2026-08-06** — **Unattended cycle 1: a completed run must never lose its
  artifacts.** `fireteam_v7` trained all 2,500,000 steps (44 min CPU) and kept
  only its checkpoints — no curves, no eval, no transcript, no gif, no
  `behavior.json`. Cause: the positional regression gate (`91e5d05`, refs #11)
  added `cover_under_threat` / `objective_dist_under_threat`, which **only
  DEFEND roots record**; every other scenario writes them blank, and
  `plot_training` called `float('')` on row 1. `train.py` plots *before* it
  evaluates, so one un-parseable cell discarded the whole post-training phase —
  and would have taken every fireteam/squad/platoon run launched after 17:01
  with it. `squad_v6` was live at the time and was saved only because that
  import is lazy.
  Fixed at two independent layers, since either alone still loses artifacts:
  `plots.py` degrades blank/junk cells to NaN and drops all-NaN series from
  their panel (a metric a scenario never records now *gaps* the curve); and
  `train.py` attempts each post-training artifact independently, collecting
  failures and still exiting non-zero so `train_status.py` keeps calling a
  damaged run FAILED. `tests/test_plot_robustness.py` pins both, including the
  exact v7 shape and an "empty metrics.csv still raises" case so tolerance for
  blanks never becomes tolerance for a run that recorded nothing. v7's curves
  are recovered. 402 → 408 tests. (`56849ac`)

- **2026-08-06** — **Unattended cycle 2: the DEFEND root could never report its
  OPORD complete.** Diagnosis first, per the standing rule. `fireteam_defend_v8`
  emitted **0 DONE reports in 100 episodes**, which read as a policy taught
  silence by `done_false` −2.0 — the exact over-pricing failure `rewards.py`
  predicts in its own comment. **That reading was wrong**, and a new read-only
  probe (`scripts/done_probe.py`) proved it: the action mask gated MISSION
  COMPLETE on `mission.type in COMPLETABLE`, while `_report_done`'s root branch
  gated on `mission.type is spec.root_mission`. On a DEFEND- or DENY-rooted
  scenario those cannot both hold, so the root's claim was **hard-masked on
  every step of every episode** and the root branch was unreachable code.
  `root_done_bonus` (3.0) was unearnable, and `grace_window` could only ever
  expire by timeout. Measured: 0 admissible root claims, 0 truthful-and-
  admissible agent-steps, 30 episodes. **The v1.10 false-COMPLETE pricing was
  therefore never tested by this scenario** — it remains unvalidated, not
  vindicated. v6's 90-claims/90-rejections were subordinates claiming ADVANCE
  complete without reaching the waypoint, a different failure.
  Fixed at the source of the divergence rather than by patching one side:
  `is_root_opord_claim()` is now the single predicate both `compute_mask` and
  `_report_done` consult. After the change (done_probe, 10 eps, seed 500, v8
  checkpoint): admissible root steps 0 → 1424; golden steps 0 → 120 (= 10 × the
  12-step grace window, exactly); episodes with an opportunity 0/10 → 10/10;
  oracle-regime accept rate 1.000; naive-regime 0.096, so random claiming stays
  −EV while a claim timed at T0 pays +4.0 against −2.0 (break-even p > 0.33).
  No observation change — `OBS_DIM` and existing checkpoints are unaffected.
  408 → 416 tests. (`cc07199`)
  **Also found, not yet fixed**: the oracle probe shows ADVANCE at 0.545 of
  threatened agent-steps in a DEFEND while the team sits 2.73 from the
  objective at 95.5% cover — the TL issues ~4.9 ADVANCE orders/ep to control
  measures the fireteam never travels to, so those missions never complete
  (8885 ADVANCE agent-steps, **0** completions). And report precision has
  collapsed to 0.38 at N=100 (v6: 0.79) while recall rose to 0.86: true
  positives 127 → 289, but false reports 34 → **480**. `contact_redundant`
  −0.02 against `contact_new` +0.5 means spamming stays +EV until precision
  falls below ~3.8%, so the pricing has no precision defence at all. Both are
  open items for the next cycles.

- **2026-08-06** — `fireteam_defend_v9` launched (3.5M steps, seed 12, lr 3e-4 —
  **identical to v8 so the root-claim fix is the only variable**). v8
  (0.87 ± 0.07, N=100) is the baseline and the *only* in-space defend run:
  v6 and v7 are both pre-`14b83ca` 166-dim checkpoints and **cannot be loaded
  into the 220-dim env at all**, so no oracle-level comparison against them is
  possible — a constraint the v1.10 break imposed on every defend comparison
  from here. Watch: `done_reports/ep` off zero with a non-zero accept rate
  (the fix working), success holding at v8's level (no regression), and
  `false_complete_rate` becoming *defined* — it is currently undefined for
  want of a denominator, which is not the same as improved.

- **2026-08-06** — **Unattended cycle 4: `squad_v6` is a MISS — 0.83 ± 0.07 (N=100)
  against the published 0.93 ± 0.05.** Logged with its numbers per the honest-DoD
  rule. The run first read as a triumph — `train_status` shows "succ 97%" and the
  N=20 behavior suite said **0.95 ± 0.10** — and both are artefacts:
  * `ckpt_best.pt` captures the best *rolling window*, not the final policy. The
    curve is `▂▆▇▁▁█▇▅▆▅`: it reached 98% rolling, **collapsed twice**, and ended
    the final decile at **65%**. The eval scores the peak; the run ended far
    below it. Any run whose curve is non-monotonic needs both numbers quoted.
  * N=20 (±0.10) to N=100 (±0.07) moved the point estimate 0.95 → 0.83. N=20
    remains a smoke test, never a verdict.
  Not attributable: `squad_v5` is a pre-`14b83ca` 166-dim checkpoint, so this is
  the same orphaned-baseline problem as the defend line — v6 is the first
  in-space squad run and the comparison spans the whole v1.10 change set.
  Two candidate mechanisms, neither yet tested: **(a)** the unsolved D4
  peak-then-collapse; **(b)** `human_death` −25 → 0, with the eval-time human
  death rate at **0.45** (v5 trained at 0.207 → 0.395). Preservation was moved
  from priced to measured, and the measurement says the humans now die roughly
  twice as often. Also down: orders/ep 17.99 → 8.15, re-tasks 12.10 → 3.25,
  doctrine preference 0.694 → 0.460 — the squad commands far less than v5 did.
  **Fleet-wide confirmation of the contact-spam finding**: precision 0.16 at
  n=2208 reports (v5: 0.12 at n=2594). The spam is not a defend quirk; every
  scenario measured sits far below the 5.8% break-even the old price implied,
  which is exactly why `contact_redundant` was repriced this session. `squad_v6`
  and `platoon_v4` bracket the change — v6 trained at −0.02, v4 at −0.25 — so
  the next squad run is the clean read.
  **New open item**: `done_reports` is **0** in both squad_v5 and squad_v6, even
  though squad's root mission is SEIZE and therefore always was completable. The
  mask was never the obstacle here, so this is a second, *different* silence from
  the DEFEND one fixed in cycle 2 — opening the channel is necessary but may not
  be sufficient. Worth diagnosing once v9 reports.

- **2026-08-06** — **Unattended cycle 5: `fireteam_defend_v9` — the fix works, the
  policy traded it badly.** Clean A/B: identical config and seed to v8, the
  root-claim mask the *only* variable. Both at N=100:

  | | v8 | v9 |
  |---|---|---|
  | success | 0.87 ± 0.07 | **0.88 ± 0.06** |
  | DONE reports / rejected | 0 / 0 | **188 / 126** |
  | false_complete_rate | *undefined* | **0.670** |
  | contact reports | 769 | **69** |
  | report recall | 0.855 | **0.098** |
  | report precision | 0.376 | 0.464 |
  | obedience latency | 1.26 | **11.24** |
  | doctrine preference | 0.172 | **0.001** |

  **The channel is alive and the metric got its denominator back** — exactly the
  thing to watch. `false_complete_rate` is 0.670, i.e. an accept rate of 0.33,
  sitting almost exactly on the break-even the pricing implies (p > 0.33). The
  policy learned to claim at the point of economic indifference.
  **Success did not move**: the intervals overlap almost completely. So the fix
  bought no success — it was still right (it made unreachable code reachable and
  dead reward earnable), but it is not a win on its own.
  **What it cost**: the team stopped reporting contacts (769 → 69 reports, recall
  0.855 → 0.098 — the picture is now essentially blind), obedience latency went
  1.26 → 11.24 with 361 censored orders, and doctrine preference collapsed to
  **0.001 at n=689** — not one doctrine-preferred order in 689.
  **Mechanism, falsifiable**: comms actions are mutually exclusive per agent-step,
  so opening an always-admissible break-even action displaces a profitable one.
  v8 earned roughly 289 informative contacts × 0.49 ≈ 142 from the contact
  channel; v9 earns ~32 × 0.49 ≈ 16 there, plus 62 confirmed DONE × 4.0 = 248
  less 126 rejected × 2.0 = 252, i.e. **≈ −4 net** from a channel it reorganised
  its whole comms behaviour around. It abandoned a profitable channel for a
  break-even one. *Refuted if* v10 (contact spam repriced) leaves recall at ~0.1:
  that would mean the displacement is not about relative comms value.
  Corroborating: `done_probe` shows subordinate golden steps 0 → 825, so the subs
  now actually complete their ADVANCE missions — the TL shifted to ADVANCE-heavy
  ordering (4.88 → 6.6/ep), which explains the doctrine-preference number.
  ~~which is *not* doctrine-valid under a DEFEND root~~ — **corrected (issue
  #14, 2026-08-06): ADVANCE under DEFEND is doctrine-*allowed*.** `DOCTRINE
  [DEFEND]` contains `ADVANCE` (A5 added it as a maneuver leg), so
  `derivation_quality` returns 0.5, not −0.5, and zero doctrine violations are
  committed. What collapsed is the *preference tier* only: `preferred` is
  `allowed[0]` = DEFEND, so ADVANCE adoption reads as 0.001 by construction.
  "Not doctrine-valid" and "not doctrine-preferred" imply very different
  defects and the mask is not leaking. **The ADVANCE-under-DEFEND item is a
  question about which leg the policy prefers, not about containment.**
  **Cycle 1 validated in production**: v9 is the first run to finish under the
  artifact guard and produced every artifact — curves, eval, transcript, gif,
  behavior.json.
  `fireteam_defend_v10` launched: v9 + `contact_redundant` −0.25, one variable,
  v9 as baseline.

- **2026-08-06** — **Issue #13 (assurance): `done_reports = 0` in squad had a
  denominator problem, and the fix is a metric, not a price.** Handled by the
  dedicated fix agent; commit `dac323a`, `refs #13`, issue deliberately left open
  for the assurance layer to verify and close.
  #13's diagnosis is confirmed by `done_probe` on `squad_v6` (10 eps, seed 500):
  the channel is **wide open** — squad's root mission is SEIZE, which was always
  in `COMPLETABLE`, so `cc07199` never applied here — and the policy declines
  **every one of 3327 truthful opportunities** (root 96, subordinate 3231); the
  oracle regime accepts at 1.000. So squad's silence is **suppression, not
  absence** — a genuinely different failure from the DEFEND mask bug, and the
  pipeline could not tell them apart because `done_reports = 0` had no
  denominator. It has one now: `done_admissible`, `done_admissible_root` and
  `done_claim_rate` in `behavior.json`, with **no opportunity → `null`, never
  `0.0`** — that distinction is the entire point. `is_done_admissible()` is
  lifted out of `compute_mask` as a single shared predicate, the same pattern as
  `is_root_opord_claim` and for the same reason. 421 → 425 tests.
  **Design decision deferred to the owner — I am not making this one.** The
  agent recommended `done_false` −2.0 → −0.6. Checked against v9's data I would
  *not* apply it blindly, because the same price produces opposite behaviour in
  two scenarios:
  * **squad**: 0 claims on 3327 truthful opportunities — suppressed;
  * **fireteam_defend v9**: 188 claims at an accept rate of 0.33 — active, and
    sitting exactly on break-even.
  A blanket −0.6 would revive squad while pushing fireteam_defend's root, whose
  break-even would fall from 0.33 to **0.13**, into claim-spam — and v9 already
  logs 126 rejected claims. The price is not uniformly wrong; the scenarios
  differ in whether subordinates hold missions that can actually complete.
  **The hidden second tax**, found by the agent in `_report_done` and not in
  #13's economics: a truthful claim sets `soldier.mission = None`, so an honest
  *subordinate* forfeits its ongoing compliance pay (up to ~0.09/step at full
  tenure — worth far more than `done_true` +1.0 over a 450-step episode) and
  forces a re-task its leader pays for. The root is unaffected: its truthful
  claim ends the episode. So the honest act is priced very differently by role,
  and the nominal break-even understates the subordinate's.
  **Options for the owner** — (1) reprice `done_false` toward −0.6, cheap but
  mis-targeted per the above; (2) accept the silence, which re-kills
  `root_done_bonus` and the grace window that `cc07199` just resurrected — not
  recommended; (3) price the two acts by role, or stop the honest claim
  forfeiting compliance pay until re-tasked, which targets the actual asymmetry.
  **My recommendation is (3)**, on the reading that an honest report silently
  costing a soldier its income is a defect rather than a calibration, and that
  fixing it lets the nominal price mean what it says. But "what a subordinate is
  owed between completing and being re-tasked" is a doctrine question, so it
  waits. Either way the effect is now measurable from `behavior.json` alone.

- **2026-08-06** — **Unattended cycle 6: the fleet's numbers are peaks, not
  results — and `fireteam_v7` is the proof.** Its N=100 eval reads **0.95 ± 0.04**
  against a published fireteam baseline of 84 ± 7. It is not a 0.95 policy. Its
  curve is `▁▃▆▅▅▂▅▃█▂`: best rolling **94%**, final decile **26%**. `ckpt_best.pt`
  is written on the best rolling *window*, so evaluating it measures the spike.
  Best-vs-final across everything measured this session:

  | run | best | final | gap |
  |---|---|---|---|
  | squad_v5 (pre-v1.10) | 98% | 93% | 5 |
  | fireteam_defend_v8 | 97% | 87% | 10 |
  | fireteam_defend_v9 | 96% | 79% | 17 |
  | squad_v6 | 98% | 65% | **33** |
  | fireteam_v7 | 94% | 26% | **68** |

  **Claim, falsifiable**: v1.10 destabilised training. The cleanest contrast is
  squad at fixed scenario/seed/lr — v5 gives back 5 points, v6 gives back 33.
  *Refuted by* any post-v1.10 run that converges with a gap under ~10 on a
  scenario where its pre-v1.10 counterpart also did. `fireteam_defend_v10` and
  `platoon_v4` are both in flight and will be the next two data points.
  This is the **D4 collapse the roadmap still lists as unsolved**, and it now
  has a number. It also means the published fleet table is measuring peaks of
  runs that may never have converged — the comparison that has been driving
  every verdict in this project is weaker than it looks.
  **Tooling fix so no future session repeats it** (`scripts/run_report.py`): a
  `stability` line now prints the best-final gap on every digest and classifies
  it `converged` / `UNSTABLE` / `COLLAPSED`, with the threshold set at 15 points
  from the pre-v1.10 baseline's 5. `tests/test_run_report_stability.py` pins each
  band against the real shapes above. 425 → 431 tests.
  This trap caught two consecutive sessions — squad_v6 at N=20 and nearly v9 —
  which is why it is now a printed verdict rather than two numbers a reader is
  trusted to subtract.

- **2026-08-06** — **Unattended cycle 7: `fireteam_defend_v10` — the contact
  reprice works, and it isolates what does not.** One variable off v9
  (`contact_redundant` −0.02 → −0.25). All three at N=100:

  | | v8 | v9 | v10 |
  |---|---|---|---|
  | success | 0.87 ± 0.07 | 0.88 ± 0.06 | 0.89 ± 0.06 |
  | report precision | 0.376 | 0.464 | **0.602** |
  | report recall | 0.855 | 0.098 | **0.732** |
  | contact reports | 769 | 69 | 347 |
  | obedience latency | 1.26 | 11.24 | 13.06 |
  | doctrine preference | 0.172 | 0.0015 | 0.0016 |
  | human death rate | 0.08 | 0.18 | 0.07 |

  Stability: `best-final gap 10 pts [converged]` — the first converged run since
  v8, and a data point *against* the blanket "v1.10 destabilised training" claim
  from cycle 6, which now needs narrowing rather than accepting.
  **The reprice is the best comms result the defend line has had**: precision
  0.602 at recall 0.732, versus v6's 0.79 precision bought at recall 0.42. Human
  death also recovered (0.18 → 0.07).
  **Correction to cycle 5's mechanism.** I predicted recall would recover if
  contact economics changed, and it did (0.098 → 0.732) — but not for the reason
  I gave. I framed it as a break-even DONE act displacing a *profitable* contact
  act; on that story, making redundant contacts costlier should have made the
  channel less attractive still. The better reading: at −0.02 the channel carried
  almost no gradient — informative and redundant reports were worth nearly the
  same, so the policy could not learn *which* report to send and abandoned the
  whole act. At −0.25 the channel is **learnable** (send informative, skip
  duplicates) and the policy re-engaged. The fix was making the signal shaped,
  not making it dearer.
  **What the reprice did NOT fix, now cleanly attributable**: obedience latency
  (1.26 → 11.24 → 13.06) and doctrine preference (0.172 → 0.0015 → 0.0016) both
  arrived with the root-claim change in v9 and are untouched by contact pricing.
  They are the next target — but see issue #14 below: the doctrine half of that
  sentence does not survive the longer record. The preference number tracks
  ADVANCE adoption, which v10 pushed to 6.31 of 6.35 orders/ep (99%), so 0.0016
  is arithmetic, not a second regression. Obedience latency remains real.
  **Success has not moved across v8/v9/v10** — three changes, all overlapping
  intervals. Everything gained this cycle is behavioural quality, not outcome.

- **2026-08-06** — **A run must hold ONE snapshot of the code.** v10 trained
  3.5M steps and produced **no evaluation**: `post-training artifact FAILED:
  evaluate (ImportError: cannot import name 'is_done_admissible' from
  'cohort.env.actions')`. It started at `a6a4335`, so its in-memory
  `cohort.env.actions` predates `dac323a`; the *newer* `cohort.metrics` it then
  read off disk at the end could not import the name. Editing the tree during a
  run is normal here — losing a finished run to it is not. `train.py` now
  imports `evaluate` and `plot_training` **before** `trainer.train()`, pinned by
  a test that reads `main()`'s source and asserts both imports precede the
  training call. Cycle 1's artifact guard is validated a second time and in a
  way I did not design it for: it caught this, named it precisely in the log,
  and still produced the curves. `platoon_v4` (launched 19:42, also pre-`dac323a`)
  **will hit the same ImportError** — its eval is recoverable by hand afterwards,
  and its curves and checkpoints are safe. 431 → 432 tests.

- **2026-08-06** — **Issue #14 (assurance): the doctrine-preference collapse was
  the metric, not the policy.** Handled by the dedicated fix agent; commit
  `refs #14`, issue left open for the assurance layer to verify and close.
  #14's claim reproduces against this repo's own `behavior.json` corpus without
  re-running anything. `preferred` is `allowed[0]`, and A5 put `ADVANCE` into
  `DOCTRINE[DEFEND]` — so preference tracks ADVANCE *adoption*, not command
  quality. ADVANCE share vs. preference, from `runs/*/behavior.json`:
  `v5` (pre-ADVANCE) — / **0.306**; `defend_brique_v1` (pre-ADVANCE) — /
  **0.213**; `v6` 0.81 / 0.011; `v7` 0.75 / 0.003; `v8` 0.69 / 0.172;
  `v9` 0.96 / 0.0015; `v10` **0.99** / 0.0016. Every corpus that adopted
  ADVANCE sits at 0.00–0.17; both that predate it sit at 0.21–0.31, and no
  post-A5 defend corpus has ever come near them. (The relation is a ceiling,
  not an identity: under a DEFEND-holding issuer an ADVANCE order can never be
  preferred, so preference is capped at `1 − ADVANCE share` and the rest of the
  gap is which non-ADVANCE leg gets ordered — v8's 0.172 at share 0.69 against
  v7's 0.003 at 0.75 is that residual.)
  So v9's 0.001 is the A5 norm — `v6`, `v7` and
  `defend_brique_v2` were already there two epochs *before* the root-claim fix
  existed. The quantity that was actually odd is **v8's 0.172**.
  **Two corrections to the v9 verdict, made in place above.** ADVANCE under
  DEFEND is doctrine-**allowed**, not "not doctrine-valid": `derivation_quality`
  returns 0.5, and zero doctrine violations are committed in any defend corpus.
  And the v10 entry's "doctrine preference is the next target" does not survive
  the longer record.
  **The fix is a metric, not a price** — same shape as #13, and deliberately so:
  #14 offered two remedies (regrade `derivation_quality`, or condition the
  report on the ordered task). Regrading is a reward change — `order_preferred`
  / `order_allowed` are paid from `derivation_quality` — and one A/B cannot
  price it, so it is left for the owner. What shipped is the decomposition:
  every agent-issued order is now booked into a tier (`preferred` / `allowed` /
  `violating` / `underivable`), giving **`doctrine_allowed_rate`** ("doctrine
  containment", the number that answers *is the mask leaking* — 1.000 under
  `full`, and the arm that moves under B3 `nomask`) and **`orders_by_task`**,
  printed as `TASK share/preference` (`ADVANCE 0.96/0.00`). `doctrine_
  preference_rate` is unchanged to the digit, so the pinned corpora stay
  comparable. 432 → 435 tests.

- **2026-08-06** — **Correction to cycles 5 and 7, owned here rather than buried
  in the #14 entry.** I twice attributed the doctrine-preference collapse to the
  root-claim fix ("both arrived with v9", "the ADVANCE-under-DEFEND doctrine gap
  is the leading suspect for the doctrine half"). **That is refuted by the longer
  record**: `fireteam_defend_v6` sat at 0.011 and `v7` at 0.003, two epochs
  before `is_root_opord_claim` existed. I compared v9 against v8 alone and
  treated a single-run difference as a trend; v8's 0.172 was the outlier in the
  series, not v9's 0.0015. I also stated ADVANCE under DEFEND was "not
  doctrine-valid" — it is doctrine-**allowed** (`allowed_derivations(DEFEND)` =
  DEFEND, SUPPORT, OBSERVE, HOLD, ADVANCE; ADVANCE is simply not `[0]`), and no
  defend corpus commits a single doctrine violation.
  **What survives, now properly isolated.** Obedience latency across the whole
  defend line at N=100: v6 **1.19**, v7 4.67, v8 **1.26**, v9 **11.24**, v10
  **13.06**. That regression *is* new with v9, is ~9× the v6/v8 level and ~2.7×
  v7's elevated value, and the contact reprice did not touch it. **Obedience
  latency alone is the next target**, and it is genuinely attributable to
  opening the root's MISSION COMPLETE — the one behavioural cost of `cc07199`
  that stands up to the longer series.
  **Method note, for the next session**: two of this session's three wrong calls
  came from A/B-ing against the single most recent run instead of the series.
  A "collapse" needs at least three prior points before it is a trend.

- **2026-08-06** — **Unattended cycle 8: the obedience "regression" is mostly a
  task-mix artefact — third correction, and the measurement was built to test my
  own claim.** Cycle 7 named obedience latency "the next target… genuinely
  attributable to opening the root's MISSION COMPLETE". Split by ordered task
  (new `obedience_by_task`, N=40 probe on both checkpoints):

  | | v8 | v10 |
  |---|---|---|
  | ADVANCE | 1.01 (n=255) | **16.21** (n=286) |
  | DEFEND | 0.68 (n=83) | **1.00** (n=40) |
  | pooled | 1.26 | 13.06 |

  **DEFEND — the mission the cohort exists to hold — is flat.** The pooled rise
  is ADVANCE's, amplified by ADVANCE's share of orders going 0.69 → 0.99. So
  "the cohort stopped obeying" is wrong. What *is* real and unexplained: ADVANCE
  latency rose **16× within the task**, which a mix shift alone cannot produce.
  **Leading hypothesis, untested**: `is_pending` — an AT-MY-COMMAND order is
  staged until the issuer's EXECUTE, and latency is measured from
  `step_assigned`, so AMC staging time is currently counted as disobedience.
  *Testable without a retrain* by splitting latency at the EXECUTE release
  rather than at assignment. That is the next measurement, not a reward change.
  `format_obedience_by_task` prints it in every digest; the pooled mean is
  unchanged to the digit so pinned corpora stay comparable. 435 → 440 tests.

- **2026-08-06** — **Retraction: "v1.10 destabilised training" (cycle 6) is NOT
  supported.** The claim rested on squad_v6 (gap 33) and fireteam_v7 (gap 68).
  Three post-v1.10 runs have since converged — `fireteam_defend_v8` (10),
  `fireteam_defend_v10` (10), `platoon_v4` (**7**) — against three that did not
  (`v9` 17, `squad_v6` 33, `fireteam_v7` 68). Both groups sit entirely after the
  break, so v1.10 is not the discriminator. lr is not obviously it either
  (platoon ran 1e-4, but pre-v1.10 squad_v5 converged at 3e-4). **No mechanism
  identified** — recorded as such rather than replaced with a second guess. The
  `stability` line stays valuable regardless of cause: it is what caught the
  peak-vs-result confusion, which was the real defect.

- **2026-08-06** — **`platoon_v4` — 0.93 ± 0.05 (N=100), converged (gap 7).**
  Against a published platoon of 98 ± 3 this is a slight regression with
  overlapping intervals. Eval recovered by hand after the predicted
  `is_done_admissible` ImportError; curves and checkpoints were never at risk.
  Notably the healthiest command behaviour in the fleet: obedience latency 2.13,
  doctrine preference **0.550**, containment 1.000. It carries the same 0.82
  ADVANCE share as the defend line but scores 0.61 preference *within* ADVANCE
  rather than 0.00 — because preference depends on the issuer's own mission, not
  on the task ordered. That is issue #14's point demonstrated across scenarios.
  Also the first run trained end-to-end under `contact_redundant −0.25`.
  **Process hazard found the hard way**: `evaluate` writes `behavior.json` into
  the run directory by default, so a low-N diagnostic silently destroys a
  canonical N=100 record — it overwrote v8's and v10's with N=40 here, and both
  were re-run and restored. Use `--behavior-out` for probes. Worth a guard.
  `squad_v7` launched: squad + the contact reprice (the root-claim fix is inert
  on squad, whose SEIZE root was always completable), testing whether the
  defend line's precision win reproduces where squad_v6 measured 0.16.

- **2026-08-06** — **Owner-reported doctrine defect: OBSERVE ordered where
  SUPPORT belongs. The reward was paying for exactly the observed behaviour.**
  Diagnosed before changing anything. SUPPORT (APPUYER — unit-targeted fire
  support, *"pas un pas sans appui"*) shared OBSERVE's compliance branch
  verbatim: full pay only when in position **and stationary**. But SUPPORT's
  anchor is the supported *soldier*. Measured on the squad map, six steps of a
  bounding element:

  | supporter behaviour | per step | total |
  |---|---|---|
  | follows the bound (correct doctrine) | **0.10** | 0.60 |
  | stands still, element walks away | **0.60** | 3.60 |
  | OBSERVE, static objective | 0.60 | 3.60 |

  **A 6× premium for not supporting**, and OBSERVE paid the same for watching a
  point that cannot outrun you. The order shares follow the incentive exactly:
  OBSERVE beats SUPPORT **0.098 vs 0.010** in `fireteam_defend_v8` and **0.057
  vs 0.016** in `platoon_v4`. The policy was behaving correctly against a wrong
  reward — the same shape as the contact-spam finding, and the third time this
  session that a "policy problem" has turned out to be a pricing problem.
  **Second defect, found while measuring**: `IN_POSITION_RADIUS[SUPPORT]` was
  **10.0** while `CombatConfig.support_umbrella` is **8.0**, so a supporter drew
  full posture pay from 9-10 cells while `_covered_by_support` protected nobody
  — the reward describing support the environment never delivered. That is
  literally the owner's "staying remote… no support".
  Fixed structurally rather than by tuning: SUPPORT gets its own compliance
  branch keyed on a new `ComplianceContext.anchor_moved` (true only for anchors
  that are themselves soldiers — SUPPORT's supported unit, RALLY's leader), so
  **movement is excused exactly when the element moved**; drifting while the
  element holds is still 0.1, so it cannot pay for wandering. And the station is
  now `min(table radius, combat.support_umbrella)`, read per scenario, so pay
  and mechanism can never decouple again. OBSERVE is untouched — its anchor
  cannot move, so settling is correct. 440 → 447 tests; four fail without the
  fix. (`d780a3e`)
  **Not yet retrained.** `squad_v7` (contact reprice vs squad_v6) and
  `squad_recon_v6` both launched *before* this landed, so neither tests it —
  they stay valid for what they do test. Next: `squad_v8` = `squad_v7` + this,
  one variable, watching the SUPPORT/OBSERVE order share invert and
  `obedience_by_task` for SUPPORT. `squad_recon` is the scenario where the
  owner's specific complaint — no support to the reconning element — should
  show most, since RECON derives SUPPORT second.

- **2026-08-06** — **Cycle 8's `is_pending` hypothesis was not just untested, it
  had the sign backwards — and it was the metric, not the policy (refs #15).**
  Cycle 8 called AMC staging "the leading hypothesis" for ADVANCE latency 1.01
  → 16.21: staging time counted *as disobedience*, inflating latency. An
  outside tap refuted it from the net — v8 staged **0.878** of its ADVANCE
  orders and held them **44.4** steps against v10's 0.369 / 20.7, so the
  checkpoint that stages more and longer measured **16× lower** latency.
  Incidence and duration both ran backwards.
  **They ran backwards because staging *deflates* the metric.** The environment
  scores a pending order as HOLD at the staging spot — where the recipient
  already stands (`extra["staging"] = recipient.pos`) — so a staged agent's
  compliance is positive from the tick the order lands. `_obedience` booked
  that tick as an order event and resolved it at latency **0**, while an
  identical *un-staged* ADVANCE whose recipient never moved was censored. And
  since release restamps `step_assigned`, the real event was booked again at
  the release tick: every staged order donated a free zero to its task's mean.
  Measured on the checkpoints themselves (greedy, seeds 500-511, 12 eps —
  read-only, nothing written under `runs/`):

  | | v8 | v10 |
  |---|---|---|
  | ADVANCE latency, staged ticks counted | **0.00** (n=19) | — (n=1, censored) |
  | ADVANCE latency, staged ticks skipped | *no events* | unchanged |
  | staged / released / abandoned | 19 / **0** / **19** | 0 / 0 / 0 |

  **Every ADVANCE "obedience" event v8 has at these seeds is a staged order
  that is never released.** v8's 1.01 was not a fast policy; it was staging
  measured as obedience. v10, which stages nothing here, is bit-identical
  before and after — the correction touches exactly the staging policy.
  Fixed by skipping pending ticks in `_obedience` and recording pendingness in
  the trace (`TraceRecorder` now scores a staged mission as HOLD, as the
  environment pays it — it was scoring it as the ordered task). What the fix
  removes is not deleted but *named*: new `_staging` reports `orders_staged`,
  `staged_released`, `staged_abandoned`, `staging_gap_mean` in `behavior.json`
  and every digest — abandonment being the fault worth seeing (the tap found
  61 of v8's 130 staged orders never released; greedy here says 19 of 19).
  No reward changed: same shape as #13/#14, a measurement rather than a knob.
  447 → 451 tests; each part fails without its fix. **Pinned corpora move**:
  any behavior number from a staging checkpoint is now different, and more
  honest. The within-task ADVANCE rise remains unexplained — the correction
  *widens* the v8/v10 gap rather than closing it.

- **2026-08-06** — **The SUPPORT/OBSERVE inversion is real, but "OBSERVE is
  ordered where SUPPORT belongs" is not: no policy on record prefers OBSERVE
  (refs #16).** Issue #16 confirmed the inversion across 38 of 43 corpora and
  raised the confound: part of it is *availability*, not incentive — SUPPORT is
  unit-targeted and needs a second living subordinate slot, OBSERVE is
  objective-targeted and almost always admissible, and masked-random corpora
  already show the inversion with no reward pressure at all. It asked that the
  *excess over the masked-random floor* be what the d780a3e fix is judged on.
  Diagnosed on the real env before writing anything, per the standing rule.
  **The confound is real, larger than stated, and it does not point one way.**
  Masked-random on this repo's own scenarios (seeds 500+), share of the
  admissible order menu:

  | scenario | OBSERVE | SUPPORT | what an uncorrected reading would report |
  |---|---|---|---|
  | `squad` | 0.23 | 0.08 | OBSERVE 2.9× — **entirely** the mask |
  | `squad_screen` | 0.42 | **0.00** | ∞ — SCREEN cannot derive SUPPORT at all |
  | `fireteam_defend` | 0.11 | **0.22** | SUPPORT 1.9× — **against** the trained direction |

  So the correction *shrinks* the effect in the squad family and **doubles** it
  in the defend family. Measured on the checkpoints, read-only, nothing written
  under `runs/` (share / availability / lift, lift 1.00 = the floor):

  | corpus | OBSERVE | SUPPORT | verdict |
  |---|---|---|---|
  | `fireteam_defend_v8` (30 ep) | 0.102 / 0.112 / **x0.92** | 0.010 / 0.219 / **x0.04** | SUPPORT declined 21× harder; OBSERVE **at the floor** |
  | `platoon_v4` (20 ep) | 0.058 / 0.250 / **x0.23** | 0.017 / 0.069 / **x0.24** | lifts **identical** — the whole 3.4× raw gap is the mask |
  | `fireteam_defend_v10` (30 ep) | 0.000 / 0.111 / x0.00 | 0.000 / 0.221 / x0.00 | neither ordered once; ADVANCE **x2.25** |

  **The shares reproduce the pinned numbers exactly** (v8 0.102/0.010 against
  the tap's 0.102/0.010; platoon_v4 0.058/0.017 against 0.057/0.016), so this
  is the same measurement with a denominator, not a different one.
  **Correcting the cycle-9 entry above and `d780a3e`'s message**: both read
  `OBSERVE 0.098 vs 0.010` and `0.057 vs 0.016` as OBSERVE being *ordered where
  SUPPORT belongs*. Half of that is unsupported. OBSERVE's lift is ≤ 0.92 in
  every corpus measured — at or below what picking uniformly among its own
  legal orders would produce. On `platoon_v4` the two are declined at literally
  the same rate. The one real, task-specific effect is **SUPPORT avoidance in
  the defend family** (v8 at x0.04: it declined 96% of the SUPPORT it held),
  which is exactly what d780a3e's reward diagnosis predicts — the 6× premium
  for *not* supporting. **The fix direction stands; its stated mechanism was
  half wrong.** What both corpora actually show underneath is ADVANCE
  monopolization (x1.62 / x1.66 / x2.25), with everything else declined.
  **Issue #16's own inference does not survive either**: it read the four
  defend-family "exceptions" that ordered SUPPORT above OBSERVE
  (`defend_brique_v1` 0.164/0.397, `fireteam_defend_v5` 0.176/0.360) as
  evidence that SUPPORT-heaviness bought the best defence on record. Those
  ratios are 0.41 and 0.49 against a defend menu whose own OBSERVE:SUPPORT
  ratio is **0.5** — they are sitting on the floor, showing no preference in
  either direction. (The menu ratio is 2:1 SUPPORT structurally — 6 unit-slot
  pairs against 1 objective per slot — and is unchanged by ADVANCE's post-A5
  arrival, which rescales both, so it applies to those pre-A5 corpora too.)
  Shipped as a measurement, not a knob — the fourth in a row (#13/#14/#15/#16)
  where the metric was at fault: `order_options()` reads the admissible order
  menu off the mask itself, `TraceRecorder` records it per agent-step
  (`order_opts`), and `_doctrine` accumulates the **matched control** — for
  every order actually issued, the share of that issuer's own legal order
  vocabulary belonging to each task. `order_availability` / `orders_matched` /
  `order_selection_lift()` land in `behavior.json`, the behavior table and
  `run_report.py`. Pinned by a masked-random calibration test: no reward
  pressure must measure at lift 1.00 while its raw mix stays inverted. No
  reward, space or scenario changed; `OBS_DIM` 220 / `N_ACTIONS` 228 untouched.
  451 → 453 → 463 tests; each half fails without its half of the fix.
  **Retrain judgement, going forward**: the number to move is
  `order_selection_lift["SUPPORT"]` from 0.04 toward 1.0, *not* the raw share —
  and any SUPPORT/OBSERVE claim quoted without its availability is not a claim.
- **2026-08-06** — **#17: the sighting-knowledge lattice is already non-constant;
  the truth stream was reporting corpses.** #17 is not a defect report — it
  pre-registers, before v1.11 lands, that vision arcs will make the sighting
  lattice non-constant "where today there is none", on the stated grounds that
  the system is near-common-knowledge on both channels: comms because the net is
  global, sightings because "vision is isotropic and long". **The comms half is
  right and the sightings half is inverted.** Measured from `env.oracle()` on
  the shipped checkpoints (seeds 500+, sampled actions, 30k+ (step, living
  enemy) checks): conditioned on a sighting existing at all, it is a *minority*
  sighting on **44.8%** of checks (`fireteam_defend_v10`), **93.6%** (`squad_v6`)
  and **100.0%** (`platoon_v4` — no enemy is ever seen by all 13 stations). The
  team picture is *absent* for 93–97% of living enemies, so "does HQ know there
  is an enemy at grid X" is answered **no** almost always: where the lattice is
  constant it is constant at ¬K, not at K. Vision is 10 cells on maps of 36–54,
  and 11.7–26.4% of in-range pairs are already denied by LOS.
  This also corrects `docs/vision.md` §1, which made the same claim first: "two
  soldiers three cells apart see very nearly the same world" is true (Jaccard
  0.56–0.86) but the cohort rarely stands that way — only 6.0% of platoon
  station pairs are ≤3 cells apart, and a CONTACT report is **novel** to 65.5%
  of listeners at squad scale and 83.5% at platoon scale, against 13.3% at
  `fireteam_defend`. Arcs therefore add most where the picture is already shared
  and least where it is already private; the §6 probe (run on `squad`) samples
  the middle of that range. Baseline tabulated in **§6.1** so V1 cannot be
  fitted afterwards. The §0 design decisions are untouched — this narrows where
  the payoff is expected, and that call remains the owner's.
  **Fixed** the one implementable ask (per-agent sighting sets in the truth
  stream), and found the surface it would have been built on was broken: the
  oracle computed `seen_by` for the dead from their last position, so **8,901 of
  9,647** enemy sighting entries over eight `squad` episodes named corpses, and
  transposing `seen_by` into per-agent sighting sets — the only way a consumer
  could get one — disagreed with the environment's own `_visible_enemies` on
  **36–47%** of agent-steps. The fifth assurance issue in a row (#13/#14/#15/#16)
  where the instrument, not the policy, was at fault. `oracle()` now publishes
  `soldiers[].sees` (living enemies visible this step, nearest first, computed by
  one shared function so truth stream and simulation cannot drift), and the dead
  neither see nor are seen. Verified 0/43,745 agent-steps of disagreement and 0
  corpse entries across three families. Truth only: the reported team picture
  stays off the stream deliberately — deriving it from CONTACT traffic is the
  external observer's job. No reward, space or scenario changed; `OBS_DIM` 220 /
  `N_ACTIONS` 228 untouched. 466 → 468 tests; each half of the fix fails
  independently without it.
- **2026-08-07** — **`done_false` is EXONERATED: the revert changed nothing, and
  the failure is not the DONE channel.** The v1.10 fleet retrain produced two
  total collapses — `squad_recon_v6` and `squad_screen_v4` both ended at **0%**
  with terminal reward exactly **0.0000**, episodes pinned at `max_steps` (375)
  and `tx/agent-step` at 0.058/0.029 — while their predecessors `squad_recon_v5b`
  and `squad_screen_v3` converged (best-final gaps 12 and 5). The diagnosis
  offered was `done_false` −0.5 → −2.0: final-decile false-DONE fell to ~0 in
  four scenarios, RECON/SCREEN completion is team-adjudicated through
  `_team_observe_steps`, and that counter is in no observation slot, so p > 0.67
  asks for a confidence the agent cannot form. Committed as `ac1fb19` and tested
  properly, one variable, `squad_screen_v5` vs `squad_screen_v4`.
  **It is wrong.** At `done_false` −0.5 the run collapsed identically: 0% final,
  terminal **0.0000**, ep_length **375.0**, `tx/agent-step` **0.026** (v4: 0.029),
  false-DONE **0.005** (v4: 0.005), entropy 1.809 → **1.025**. The price was
  never what silenced the claim — the claim rate is unchanged at 4× the price.
  **⚑ SCOPED 2026-08-11 (assurance, #46): that null belongs to the COLLAPSED
  regime and is not a falsification of the price.** Both figures above are the
  FINAL policies and both arms are D4-collapsed (0% success, `ep_length` 375 =
  `max_steps`), so the manipulation had no power to detect an incentive
  response — a collapsed policy emits the cheapest no-op because it must emit
  something. At the same two runs' `ckpt_best`, which are **not** collapsed
  (both 1.00 ± 0.00 at N=20), the committed corpora read **0** DONE claims at
  −2.0 against **55** at −0.5. Same direction as `squad_v10` → `squad_v11`.
  Read the 2026-08-11 (#46) entry at the end of this log before quoting the
  sentence above.
  What the test *did* buy is a much better-posed question, because it proves the
  failure is **not DONE-specific**. `tx/agent-step` falls 0.123 → 0.026 (4.7×)
  against `squad_screen_v3`, and that counts *every* channel: `comp_report` goes
  −0.0016 (v3, actively paying transmission costs to report) → +0.0002 (v4/v5,
  no traffic to pay for), and orders/episode at `ckpt_best` is **3.75** (v5)
  against 67.70 (v4). The whole radio goes quiet, and the cohort parks: final
  decile draws compliance 0.0641 + command 0.0042 − time 0.0100 ≈ 0.058
  /agent-step × 375 ≈ 22, against an observed `ep_return` of **21.80**. A
  stall-farm at 22 beats nothing, and terminal — worth ~59 in `v3` — has become
  unreachable rather than unattractive.
  **Also falsified**: the correlation "the runs that kept claiming DONE are the
  runs that succeeded" (`fireteam_defend_v10` 0.553 → 89%, `platoon_v4` 0.382 →
  93%) reads the causation backwards. They claim because they finish.
  **Suspects remaining**, in order: (1) `contact_redundant` −0.02 → **−0.25**,
  the only other transmission tax in the cycle, whose isolated effect is known
  only at squad scale (`squad_v6`→`v7`: terminal −33%, sub-lethal); (2) the
  v1.10 **space change itself** — `OBS_DIM` 166 → 220, `N_ACTIONS` 228 — with
  `ent_coef` left at 0.01. Note (2) is *not* answered by raising the entropy
  bonus alone: `squad_recon_v6` ran at `ent_coef` 0.02 and collapsed anyway.
  `done_false` is left at −0.5 pending the owner's call — its v1.10 raise is now
  known to buy nothing (false-DONE ~0.005 at either price), but the revert's
  stated reasoning in `ac1fb19` did not survive contact with the test, and that
  commit message should be read with this entry.

- **2026-08-07** — **Correction: the collapsed runs did not go quiet, they went
  free. Issue #18 confirmed on the clock, falsified on the chatter.** The
  assurance layer re-measured the two v1.10 collapses from the outside and
  proposed a net-only stall detector: command traffic down 13×, voice-sync up
  5×, episodes pinned at `max_steps`. Reproduced exactly here at 30 episodes,
  seeds 500–529 — `squad_screen_v4` 30/30 success from `ckpt_best` and 30/30
  timeout from `ckpt_latest`; `squad_recon_v6` 29/30 and 30/30; orders/episode
  66.4 → 6.0 with sync bounds 80.8 → 436.9. Its **correction of us is right**
  too: `squad_recon_v6`'s *succeeding* checkpoint emits **0 DONE reports in 30
  episodes**, so completion silence is present in the winner and cannot be the
  collapse.
  **Two things in its diagnosis do not survive measurement, and one of ours
  does not either.**
  1. *"The policy is farming SYNC_PROPOSE/SYNC_GO."* It is not. Sync is
     **voice** (A5-4): uncharged, never net-arbitrated. Re-scoring
     `squad_screen_v4/ckpt_latest` with `bound_bonus=0` changes its compliance
     income by **3.65 over 10 episodes** — 0.0001/agent-step, 0.2% of the
     0.0624 it actually draws — against **12,388 sync messages**. The traffic
     earns nothing. It is where a policy's action mass drifts when every other
     transmission costs −0.01 and the mission income is a posture drip: a free
     sink, not a farm. The stall arithmetic stands unchanged (compliance
     0.0624 + command 0.0042 − time 0.0100 ≈ 0.056/agent-step ≈ 21.2 measured
     `ep_return`), and `max_step_farm()` (0.09) still dominates it — the
     terminal-dominance invariant never broke, terminal simply became
     *unreachable*.
  2. *"A traffic-composition check would have flagged both runs."* It would
     also have flagged healthy ones. Across every checkpoint that loads under
     v1.10 (10 eps, seeds 500–509): healthy `fireteam_defend_v10/best` 8/10 at
     a **0.026** command share, *below* collapsed `squad_recon_v6/latest` 0/10
     at **0.022**; `fireteam_v7/latest` 8/10 at 1.5 orders/episode. Command
     share is scenario idiom, not a threshold. **The clock separates the record
     completely**: 0.0–0.2 healthy against exactly **1.0** for all three stalls
     (`squad_recon_v6`, `squad_screen_v4`, `squad_screen_v5` at `ckpt_latest`).
  3. *Ours*: the entry above says "the whole radio goes quiet". **It does not.**
     `squad_screen_v4/ckpt_latest` carries **1326 messages/episode** against its
     own `ckpt_best`'s 537 — 2.5× *louder*. We read that off `tx/agent-step`,
     which counts **charged** transmissions only, so the free channel was
     invisible to the one volume number in the digest. The conclusions that
     entry drew about `done_false` are unaffected (they rest on claim rates and
     terminal income); the sentence about the radio is wrong and is corrected
     here.
  **Shipped** (`5626977`, `1f0326a`): `timeout_rate` in the behavior suite and
  in `regression_gates` **for every root mission** at ≤ 0.5 (the middle of the
  empty band); `max_steps` in the trace so a length reads against its own
  ceiling; `messages_per_episode` / `command_traffic_share` /
  `voice_traffic_share` as reported diagnosis, never gated, with the
  false-positive measurement pinned in a test; `timeout_rate_rolling` and
  `messages_per_agent_step` per training iteration, so the stall is visible
  ~3M steps before anyone evaluates a checkpoint; the digest now names the
  checkpoint `behavior.json` was measured on (on `squad_screen_v4` it printed
  "success 1.00 ± 0.00" three lines under a curve ending at 0%). 475 tests.
  **Open questions for the owner** (both design, neither taken):
  (a) **Should voice stay free?** A4 charges every *learned* transmission
  −0.01; A5-4 exempts SYNC by fiat, and that exemption is what makes a stalled
  policy's radio load *rise*. Charging it would make the airtime rule uniform;
  leaving it free keeps the trinôme bound cheap to coordinate. Either is
  defensible — it is a vocabulary/economics call, and no measurement here says
  the exemption *caused* anything.
  (b) **Should the training loop act on the stall gate**, e.g. abort a run whose
  `timeout_rate_rolling` sits at 1.0 for a decile? That changes campaign
  semantics (a killed run has no `ckpt_latest` to diagnose), so only the
  measurement half was built.
  **Owner's answers (2026-08-07)**: (a) **charge it** — shipped as `cf3f5fe`;
  (b) **log only**, and `done_false` **stays at −0.5**.
- **2026-08-07** — **Learning rate is eliminated too: `squad_screen_v7` failed a
  third way.** The lr hypothesis was the best of the three and it is also wrong.
  Sorted by best-final gap, every v1.10 run at 1e-4 had converged (`patrol_
  brique_v4` 3, `platoon_v4` 7, `defend_brique_v3` 8) and every collapse was at
  3e-4 (33/39/68/97/100/100), while the *pre*-v1.10 `squad_screen_v3` ran 3e-4
  at `OBS_DIM` 166 and converged with a gap of 5. So: `squad_screen` again,
  seed 17, 2M steps, **one variable — `lr` 1e-4** — against `squad_screen_v5`.
  **It never learned at all**, and not in the stall's shape: best rolling 29%,
  final **3%**, `ep_return` **−0.278** (negative), `human_death_rate` **0.983**,
  entropy 1.668 → **0.453**, episodes *short* at 73 steps. The new clock gate
  correctly **passed** (`timeout_rate` 0.033 — this is not a stall, it is a
  cohort that dies), which is the first evidence the #18 gate is specific and
  not a catch-all. Two failure modes, one scenario: at 3e-4 it learns a 100%
  policy and then pours 93% of its traffic into the free SYNC channel until the
  clock expires; at 1e-4 it goes deterministic early and everyone dies.
  **Checked before concluding**: `git diff ac1fb19 2d14510 -- cohort/training/
  train.py` is purely additive accounting (`message_total += len(env.last_
  messages)` plus two derived stats) — the rollout, loss and optimizer are
  untouched — so the instrumentation did not cause this and the result is
  attributable to `lr`.
  **`OBS_DIM` 166 → 220 is now the last suspect standing**, by elimination
  rather than by evidence, which is a weaker position than it sounds: three
  hypotheses have been tested and killed (`done_false`, `contact_redundant` via
  the `squad_v6` control, `lr`), and the space change is what remains, not what
  has been shown. The v1.10 fleet stands at **4 converged** (`fireteam_defend_
  v10` 0.89±0.06 with both gates green, `platoon_v4` 0.93±0.05, `patrol_brique_
  v4` 97% final, `defend_brique_v3` 90% final) and **4 collapsed** (`squad`,
  `fireteam`, `squad_recon`, `squad_screen`). Nothing republishes until the
  second half is understood; `fireteam_defend_v10`'s publish is held with them.
- **2026-08-07** — **The width bisect returns a result: `squad_screen_core_v1`
  (166, A5 vocabulary and the voice channel both present, only the space
  narrowed) converges where every same-scenario 220-width run collapsed.**
  Issue #19 (assurance layer) first established *why* the record alone could
  not settle this: the only pre-A5 corpora we hold (`squad_screen_v1/v1b/v2`,
  `OBS_DIM` 166) predate the trinôme voice channel entirely — voice share reads
  exactly **0.000** in all three, by their measurement — so a 166-vs-220
  contrast drawn from history is really a v1.8-vs-v1.10 contrast wearing a
  width label. It offered one number to hold onto regardless:
  `squad_screen_v2`'s pre-break profile, **30/30 success, 111-step episodes,
  command share 0.790** — the last surviving description of this scenario
  working, since that checkpoint no longer loads under the current spaces. It
  also dated the channel itself: voice share goes 0.000 → 0.41–0.57 exactly at
  **A5 (v1.9), two versions before the collapses**, which makes an un-taxed
  voice channel *necessary-at-most* for the stall, not sufficient — so
  `cf3f5fe` (charging SYNC airtime) was never expected to fix this alone.

  `squad_screen_core_v1` (registered `ba688c2`, trained since) is the
  controlled instrument the issue asked for: same voice channel, same A5
  vocabulary, `OBS_DIM` frozen at 166. Read with `run_report.py`, one variable
  against the four 220-width runs that collapsed: **final-decile success
  (rolling) 0.965, best-final gap 4 pts, `[converged]`** — the same band as
  the four v1.10 runs that held (gaps 3/4/7/8), not the four that didn't (gap
  100/100/26, final 0%/0%/3%). The clock gate that separated the record
  completely in the #18 entry separates it again: `ran clock out` final
  decile **0.027** (behavior suite: **0.10**), `timeout_rate` gate **PASS** —
  against exactly 1.0 for every collapsed `ckpt_latest`.

  **One caveat, and it is the one the issue flagged in advance rather than one
  found after the fact.** Command share (our protocol: 20 eps, `ckpt_best`,
  greedy=False) reads **0.318** for `squad_screen_core_v1` — inside the
  collapsed group's own command-share range from the issue's table (`v4`
  0.311, `v5` 0.099, `v7` 0.392; their protocol: 30 eps, seeds 500–529, a
  different corpus than ours), not the pre-break reference's 0.790. That is
  consistent with the #18 correction that command share is scenario idiom and
  separates nothing on its own — the training curve and the clock gate carry
  this result, not composition, and the two protocols are not directly
  comparable besides.

  **Width is no longer the last suspect by elimination — it now has a
  single-variable bisect in its favor.** One run, one seed (17); not yet a
  replication, and the `squad_screen_core` profile still differs from the
  true v1.9 166-vector in the one documented way (`ba688c2`: it omits the
  SITREP-due slot rather than reproducing the overload that packed it into
  the "known enemy present" flag) — the issue independently checked its
  corpora for SITREP-cadence traffic in the screen family and found none,
  which is the condition under which `ba688c2` called the omission exact for
  this scenario specifically. Next: replicate on a second seed before
  treating width as confirmed, then decide what a width-caused collapse
  implies for the rest of the v1.10 fleet (`squad`, `fireteam`, `squad_recon`
  — none yet re-run at 166). refs #19
- **2026-08-07** — **Correction to the entry above: the bisect arm shows
  "did not collapse", not "reached the pre-break profile" — and the entry
  quoted the one protocol that flatters it.** The previous entry's headline
  numbers (final-decile rolling success 0.965, `[converged]`) come from the
  *training curve*; the clock number it cites next to them (0.10) comes from
  the *behavior suite*. On the behavior suite — the protocol issue #19 asked
  to be compared on, `ckpt_best`, greedy=False — `squad_screen_core_v1`
  reads **success 0.85 ± 0.16**, and that number appears nowhere in the entry.
  For scale, collapsed `squad_screen_v4` reads **1.00 ± 0.00** on the same
  protocol, because its `ckpt_best` predates its own collapse. Comparing a
  converged run's training tail against a collapsed run's best checkpoint is
  not one series.

  Issue #19 named an explicit bar: roughly `squad_screen_v2`'s pre-break
  profile — **30/30 success, 2/30 root deaths, 111-step episodes, command
  share 0.790**. The arm misses it on every dimension except non-collapse:
  **root death 0.822** (final decile, and *rising* through training: 0.205 →
  0.822), **episodes 165 steps**, **command share 0.318**, **false DONE 0.944**
  (behavior suite). Its reward is dominated by the terminal component
  (**0.3453** final decile, ~100x every other component; the collapsed runs
  sit at 0.000 and -0.0042) with **41.2 retasks against 47.15 orders per
  episode** — terminal dominance and churn are two of the named
  regression-hazard signatures in CLAUDE.md, and this run wears both.

  What survives the correction, and it is not nothing: the collapse itself
  did not happen at 166. `squad_screen_core_v1` goes 0.742 → 0.965 where
  `squad_screen_v4` goes 0.658 → **0.000** and `v7` 0.115 → 0.028 — the
  latest policy is alive rather than dead, which is the width-relevant
  signal and is real. What does *not* survive is "leading suspect by
  evidence": the exploit-shaped metrics are family-wide (root death 0.983 on
  `v7`, false DONE 1.000 on `v7`, churn ratios high across all three), so
  they neither indict nor exonerate width — but they do mean the arm
  converged to something well short of what the scenario looked like when it
  worked, and a bisect whose treated arm lands on a terminal-reward exploit
  cannot yet distinguish "width caused the collapse" from "the narrower
  vector made the exploit easier to reach".

  Standing: width remains the last suspect, now with one run showing
  non-collapse at 166 and no run showing recovery of the pre-break profile.
  Before a second seed is worth spending, the arm needs diagnosing against
  the oracle (`env.oracle()`, this repo's diagnose-first rule) on why root
  death rises with success and why false DONE sits at 0.944 — a 0.85 ± 0.16
  built on those is not a result to replicate yet. Owner's call; no reward
  change proposed here. refs #19
- **2026-08-07** — **D4 diagnosed: the dominance test was in the wrong units,
  and the collapse is a free-ride the shared policy takes collectively.** Three
  defects found and fixed, one diagnosis retracted, one new mechanism measured.
  Commits `60cb6c3`, `da5bdb1`, `d44ee8d`. 491 tests green, ruff clean.

  **1. The dominance invariant never applied the discount.**
  `test_terminal_dominates_stalling` asserts `success_team > max_step_farm() *
  max_steps` — 60 > 54 on platoon — and has passed since v1.0. PPO maximizes
  the DISCOUNTED return, and at γ 0.99 the planning horizon is 1/(1-γ) = 100
  steps against episodes of 300-600. Discounted, platoon's terminal was worth
  **4.52 against a stall's 8.98**: not marginal, inverted. squad sat at exactly
  1.00, which is what its oscillation between 22% and 99% success was.
  Scoring every run's OBSERVED final-decile reward stream this way separates
  the record **8 times out of 8** — collapsed runs at farm 0.044-0.068/step
  with terminal 0.0000, converged runs at ≤0.03 with terminal 0.12-0.39.
  The mechanism is `compliance()`'s flat in-position credit: the only
  non-telescoping term in the shaping, a per-step RENT whose lifetime total is
  proportional to `max_steps`, where `_progress` is potential-based and
  `observe_progress`/`prep_in_position` are explicit budgets. Raising
  `success_team` 25 → 45 → 60 fought it three times and could not have won —
  the shortfall is in the exponent. **Fixed**: γ 0.99 → 0.999, position rents
  halved behind names (`POSTURE_HOLD`/`POSITION_HOLD`/`POSITION_DRIFT`).
  Worst-case discounted win/stall **0.50 → 2.56**; observed farm in the
  collapsed state **0.068 → 0.012/step**. The invariant is now stated in the
  units that decide it, at a 2x bar, and a companion test asserts the OLD
  economics FAIL it — a regression test that also passes on the broken
  configuration proves nothing.

  **2. The value head owned 95-99% of the gradient budget.** Value targets of
  scale 60, unnormalized, drove `value_loss` to 94-188 across the fleet; since
  `max_grad_norm` clips the WHOLE gradient, the POLICY update was attenuated
  ~5x on exactly the iterations where something happened. Measured live in the
  bisect: control `grad_norm` **46.1** (clipped ~90x) against treatment
  **0.3-1.6** (clip inert). **Fixed**: the critic fits standardized returns and
  has its own torso and its own gradient clip. Checkpoints stay loadable — the
  architecture flags default to the pre-v1.11 shape when absent.

  **3. Nothing was instrumented.** `grad_norm`, `clipfrac`,
  `explained_variance`, the value/return scales, `epochs_used`, `lr` and
  `n_episodes` are now logged, and the rest of `PPOConfig` — including γ — is
  on the CLI, so a campaign can sweep it without editing the tree mid-run.
  Issues #16-#19 were all diagnosed without any of it.

  **RETRACTION: the discount inversion is NOT the trigger.** The bisect
  (`squad_screen_ctl_gamma099_v1` control; `squad_screen_v9`/`v10` treatment,
  seeds 17 and 23) collapsed on **all three arms** — control at 395k, treatment
  at 118k and 151k — with no recovery in 350k+ further steps. The 8/8
  correlation was real but it describes where collapsed runs LAND, not why they
  leave. The economics fix removed the reward for staying; it did not stop the
  policy arriving.

  **What the instrumentation bought instead: three hypotheses killed in one
  run.** Through the collapse (12k steps, 147k → 160k) entropy is **flat at
  1.70**, `approx_kl` flat at **0.005**, `grad_norm` **0.3-1.9**, `clipfrac`
  flat. So: not an entropy collapse, not a destructive update, not a numerical
  blow-up — the three standing suspects. A fourth, mine, died too: raw
  advantage std in the collapsed basin is **0.82**, amplified 1.2x by
  normalization. There is a healthy gradient there; it points nowhere.

  **What the oracle found.** The collapsed squad stands **19.96 cells from the
  root objective against 10.39** before, takes **13.9 threatened steps/ep
  against 24.9**, and loses **0.20 friendly/ep against 1.12**. A survival
  policy: hang back, sit in cover, never trigger the team observation.
  `terminal` goes to exactly 0.0000 and never returns.

  **The mechanism.** The terminal was paid `for s in roster.living`, so a
  soldier who died at step 50 of an episode that succeeded at step 200 received
  none of the 60 points. Per agent on a 9-agent squad: hanging back cuts P(die)
  0.129 → 0.022, worth **+6.4** — but ONE shared policy updates EVERY agent at
  once, so team success goes 1.00 → 0.00, worth **−52.3**. A per-agent
  advantage only ever sees the first number. Parameter sharing converts an
  individually-rational free-ride into a simultaneous collective defection,
  which explains the abruptness, the non-recovery, and the flat entropy.
  **Fixed** (`d44ee8d`): a casualty is no longer terminated out of the episode.
  It stays, STAY-only and accruing nothing, and is paid the team terminal with
  everyone else — the policy takes no spurious gradient from the fallen (one
  legal action, zero entropy, ratio 1) while the critic gets the correct value
  target for "dead, outcome still pending". **Honest caveat: platoon has 16
  agents, the most dilution, and converged at 92%. Free-riding alone does not
  predict that.** Not yet validated — the A/B against these three arms is the
  next run.

  **Publishing standard changed.** `ckpt_latest` is now evaluated alongside
  `ckpt_best` (`behavior_final.json`), `run_report` prints both with the FINAL
  as the headline, and `scripts/publish_audit.py` applies the gate: **11 of 18
  published runs fail it, mean give-back 25.9 points**, six carrying a headline
  ≥10 points above where their run finished (`squad_recon_v6` published 91±6
  off a run that ended at **0.00**). README and the handoff block are corrected.
- **2026-08-07** — **the import snapshot is now closed, not one level deep.**
  Commit `3eceaa9`. Autocycle item, found by measurement rather than by another
  lost run.

  Yesterday's guard hoists post-training entry points before training starts so
  a run holds one consistent snapshot of the code for its whole life. It was
  stated at **depth one**, against a hardcoded two-entry list — the same shape
  of mistake it was fixing. Measured in a clean interpreter: after train.main's
  entire eager block runs, **`cohort.core.oracle` is not in `sys.modules`**. The
  path is `cohort.metrics` → `cohort.env.cohort_env` → `cohort.core.oracle`,
  that last edge a function-scope import inside `CohortEnv.oracle()`. Both
  intermediate modules are hoisted; nothing walks past them.

  **Cost so far: zero, and that is the point.** Nothing on the artifact path
  calls `env.oracle()` — only `scripts/oracle_probe.py` does — so this hole is
  latent where its four predecessors were not (`fireteam_defend_v10` on
  `is_done_admissible`; `squad_v7`, `squad_recon_v6`, `platoon_v4` on
  `order_options`, 3M steps each, no evaluation produced). It would not have
  stayed latent: the diagnose-first rule keeps pulling the oracle toward the
  evaluation path, and an oracle-backed behavior metric in `evaluate()` would
  have resumed killing runs at the very end of their step budget.

  **Fixed as a redesign, not a patch** (owner's stated bias). The invariant is
  no longer "the entry points are hoisted" but *nothing reachable from the
  snapshot may be read fresh off disk later* — a closure, with roots **derived**
  from whatever `train.main` imports before `trainer.train` rather than listed.
  A hardcoded list is precisely what needed editing twice. `test_import_snapshot`
  walks the deferred-import graph transitively and reports every open edge at
  once; against the pre-fix tree it fails with all three
  (`cohort.core`, `cohort.core.language`, `cohort.core.oracle`), against this
  one it is clean. A fourth test guards the walk itself, because a traversal
  that silently stopped recursing would decay into the weaker check it replaced.
  493 tests green, ruff clean.

  **Not done, deliberately**: the artifacts `squad_v7` / `squad_recon_v6` /
  `platoon_v4` lost are *not* being recovered. Their `behavior.json` was
  re-generated by hand on 2026-08-07 00:14, but `eval.gif`, `eval_transcript.txt`
  and `behavior_final.json` are still missing, and re-running evaluation now
  would score those checkpoints under **post-`d44ee8d` physics** — the fallen
  are paid the team terminal and casualties stay in the episode. That is not the
  environment they trained in, so the number would be neither the run's result
  nor a current one. They are superseded by the v1.11 retrain regardless.
- **2026-08-07** — **the v1.11 bisect baselines close at 0.00, and seeded
  evaluation no longer crosses the `d44ee8d` boundary.** Commit `d4f3be8`.

  All three arms finished 2M steps. Their own processes predate `da5bdb1`, so
  none wrote `behavior_final.json`; recovered by evaluating `ckpt_latest` under
  the current tree. **Final policy, N=20: `ctl_gamma099_v1` 0.00 ± 0.00,
  `v9` 0.00 ± 0.00, `v10` 0.00 ± 0.00, clock-out 1.00 on all three** — against
  `ckpt_best` figures of 1.00, a **100-point best–final gap per arm**, the
  largest in the record and a clean illustration of why the publish gate
  changed. The free-ride fix now has an unambiguous floor to beat.

  **The optimizer fixes completed the disengagement rather than softening it.**
  `v9` and `v10` (γ0.999, separate critic, value normalization) record **zero
  threatened agent-steps across 20 episodes** — the squad never comes within
  threat range of anything. The control (γ0.99, no value fix) still musters 39
  and a 0.308 cover occupancy. Whatever the value-head fix bought, it was not
  engagement.

  **Constraint on every cross-boundary comparison from here.** Re-evaluating
  `ctl`'s `ckpt_best` under the current tree — same seed 123, same 20 episodes —
  does **not** reproduce its own `behavior.json`: **42 of 55 numeric metrics
  move**, success 1.00 → 0.95, mean episode length 164 → 175. Diagnosed to
  `d44ee8d`: the fallen now stay in the episode, the policy is queried for them,
  and each masked sample consumes a draw that shifts the RNG stream for every
  agent after it. The physics are unchanged — dead soldiers were already inert —
  but **a pre-`d44ee8d` artifact and a post-`d44ee8d` artifact are not the same
  measurement**, which is why the three baselines were re-measured rather than
  compared as found. Implied follow-up, deliberately **not** taken while the
  treatment arms are in flight: an agent with exactly one legal action should
  take it without drawing, which is both cheaper and stream-stable. Landing that
  now would desynchronize the A/B it is meant to clean up.
- **2026-08-07** — **#19's open question answered: the width arm buys its success
  with a no-cover firefight, and `d44ee8d` just removed the price.** Commit
  `b5ab8cc`. Oracle diagnosis, as the #19 entry required before spending a
  second seed.

  **`squad_screen_core_v1` FINAL policy: success 1.00 ± 0.00 (N=20)**, against
  `ckpt_best`'s 0.85 ± 0.16 — a 4-point best–final gap the *right* way. It also
  predated `da5bdb1` and had no `behavior_final.json`; recovered here. The
  width-bisect treated arm is genuinely converged, which #19 could not yet say.

  **What it costs** (oracle, 20 eps, seeds 500–519, `ckpt_latest`):
  fire rate [human] **0.830** (team 0.676, leader 0.549, rifleman 0.624) ·
  cover occupancy [human] **0.000** (team 0.016) · friendly deaths at OBJ
  **0.00**/ep · friendly deaths in the open **1.30**/ep · human death rate
  **0.700** (0.900 on the behavior suite's seed block) · success **1.000**.

  The commander is *not* dying to reach the objective — **nobody** dies at the
  objective. The squad has learned a stand-up firefight in the open with
  essentially no cover, and the commander is its most aggressive shooter: highest
  fire rate on the field, cover occupancy exactly zero. Root death rises with
  success because success is bought that way. #19 guessed exposure-to-complete;
  the oracle says trade-bodies-for-terminal.

  **The economics, and why this is urgent.** Nothing in `RewardConfig` pays for
  cover under threat — `bound_bonus` pays a covering shooter for someone *else's*
  bound, `prep_in_position` is defend-only and bounded by H. Being in cover while
  being shot at is worth **zero**. `death` is **−1.0** against `success_team`
  **60.0**. The only disincentive to dying ever commensurate with the objective
  was **forfeiture** — a casualty received no terminal, and 60 forfeited is 60×
  the explicit price of the death. **`d44ee8d` removes exactly that.** The whole
  remaining price of a soldier's life is now −1.0 plus −0.2 per surviving
  teammate against a +60 objective: a tax under 2%.

  **Prediction, recorded before the treatment arms land** (both ~42%, 99–100%
  rolling): `squad_screen_fallen_v1/v2` should **succeed** where the baselines
  closed at 0.00, *and* show friendly deaths/ep **no lower than 1.30** and cover
  occupancy **no better than 0.016**. If it holds, the free-ride fix is right
  about the collapse and has traded it for a body-count policy. The follow-up is
  then a **reward call for the owner**: price cover under threat, or raise
  `death` now that forfeiture is gone. Not taken unilaterally, and not while the
  arms measuring `d44ee8d` are still in flight.
- **2026-08-07** — **D4 is solved.** Commits `9933a3a` (runs), `d44ee8d` (the
  fix). The A/B against `squad_screen_v9`/`v10` — identical config, identical
  seeds, `d44ee8d` the only difference — and it is not close.

  | seed | baseline (final N=20) | + `d44ee8d` (final N=20) | gate |
  |---|---|---|---|
  | 17 | `v9` **0.00 ± 0.00**, clock-out 1.00 | `fallen_v1` **1.00 ± 0.00** | 99-pt gap → **0-pt gap** |
  | 23 | `v10` **0.00 ± 0.00**, clock-out 1.00 | `fallen_v2` **1.00 ± 0.00** | 99-pt gap → **1-pt gap** |

  Non-overlapping CIs on both seeds, both arms **PUBLISHABLE**. The treatment
  arms never collapsed at all — they cleared 118k, 151k and 395k (the three
  baseline collapse points) without a dip and closed at 100% and 99% rolling.
  The collapse that has haunted this repo since v1.0 was one shared policy
  free-riding on a terminal its casualties could not collect.

  **This also kills the width suspect outright**, which #19 could only eliminate
  by exhaustion. The fallen arms run the **same 220-input observation** that
  `v9`/`v10` collapsed on. Width was never the cause; the v1.10 space break is
  exonerated by direct evidence.

  **The prediction logged one entry above was WRONG, on every element.** I
  predicted a body-count policy — deaths/ep ≥ 1.30, cover ≤ 0.016 — reasoning
  that `d44ee8d` removes forfeiture and leaves a life priced at −1.0 against a
  +60 terminal. Oracle, 20 eps, seeds 500–519, `ckpt_latest`:

  | | `core_v1` (pre-fix) | `fallen_v1` | `fallen_v2` |
  |---|---|---|---|
  | cover occupancy [team] | 0.016 | **0.260** | **0.245** |
  | cover occupancy [human] | 0.000 | **0.227** | **0.211** |
  | friendly deaths open/ep | 1.30 | **0.65** | **0.60** |
  | human death rate | 0.700 | **0.050** | **0.100** |
  | threatened steps/ep | 21.8 | **36.9** | 23.5 |
  | fire rate [human] | 0.830 | 0.200 | 0.408 |
  | fire rate [rifleman] | 0.624 | 0.593 | 0.709 |

  Deaths halved, cover up 15×, commander death 0.70 → 0.05, and **more**
  engagement, not less. The error was treating cover and survival as goods that
  must be priced. They are **instrumental**: removing the incentive to hang back
  let the policy engage, and once engaged the fastest route to +60 dominates —
  episode length falls **165 → 53**. A short fight is a survivable fight, so
  cover and survival rose in service of a terminal now reachable by everyone,
  including the dead. The role structure corrected itself unprompted: the
  commander stopped being the lead shooter (0.830 → 0.200) and started using
  cover (0.000 → 0.227) while the riflemen shoot. Doctrine falling out of
  economics.

  **The reward call escalated one entry above is therefore WITHDRAWN.** Cover
  needs no price and `death` needs no raise. *Caveat*: `core_v1` is
  `squad_screen_core` (166 inputs) against the fallen arms' `squad_screen` (220)
  — a cross-scenario reference, not a controlled contrast, used because every
  pre-fix `squad_screen` arm ended at zero with ~no threatened steps to measure.
  The effect sizes are far too large for that to explain them.

  **Residuals this does NOT fix** — the standing exploit signatures, all still
  open: false-DONE **0.279/0.288** final-decile (0.500–0.600 on the behavior
  suite), retask/order churn **0.69** and **0.51**, and `fallen_v2` reporting a
  contact recall of **0.00**. None contradicts the success rate; none is closed.

- **2026-08-07** — **v1.11 fleet retrain launched** (`scripts/campaigns/v1_11_fleet.jobs`,
  7 jobs, ~19.5M steps, detached, `logs/queue_20260807_123439.log`). Every
  published number in the repo predates `d44ee8d` and is superseded. Budgets,
  seeds and lr are each scenario's own last run, unchanged on purpose, so the
  only difference is the environment and a miss stays attributable; PPO defaults
  now carry the validated recipe (γ 0.999, `normalize_value`, `separate_critic`),
  so no arm passes them. `squad_screen` is deliberately absent — `fallen_v1/v2`
  already are its v1.11 result. **Watch, in order**: (1) the collapse — four of
  these scenarios collapsed pre-fix; (2) **`platoon`**, 16 agents and the most
  dilution, which converged at 92% *without* the fix and remains the one arm the
  diagnosis does not explain; (3) `fireteam_defend`, where v1.10's prep period
  and `prep_in_position` get measured for the first time; (4) false-DONE and
  churn on every arm.

  **Held back deliberately**: the single-legal-action sampling fix (an agent with
  one legal action should take it without drawing). It is a strict improvement,
  but it shifts the RNG stream, and keeping the tree exactly as validated means
  this campaign reproduces the result that justified it. Land it **after** the
  fleet is retrained and published, not during.
- **2026-08-07** — **v1.11 fleet, arm 1/7: `fireteam_v8` — the fix generalizes.**
  `fireteam` is one of the four scenarios that collapsed pre-fix, and the
  contrast against its own last run is the point:

  | | `fireteam_v7` (pre-fix) | `fireteam_v8` (post-fix) |
  |---|---|---|
  | rolling best → final | 94% → **26%** (68-pt gap, **COLLAPSED**) | 96% → **84%** (12-pt gap, converged) |
  | measured `ckpt_best` | 0.95 ± 0.04 (N=100) | 0.75 ± 0.19 (N=20) |
  | measured FINAL | — | **0.90 ± 0.13** (N=20) |
  | terminal (final decile) | 0.0712 | **0.6892** |
  | ep length | 276 | **121** |
  | human death rate | 0.216 | **0.000** |

  `v7` is the publish-audit pathology in one line: **0.95 ± 0.04 at N=100 off a
  policy whose run ended at 26%**. `v8` does not collapse, and its terminal
  income is ~10× — the same signature the `squad_screen` arms showed. Two of two
  collapse-prone scenarios now hold.

  **Not published, and the gate is worth watching.** `run_report` calls `v8`
  NOT PUBLISHABLE on a **12-point rolling** best–final gap, while its *measured*
  final (0.90 ± 0.13) sits **above** its measured best (0.75 ± 0.19) — the CIs
  overlap, so the honest statement is that the two are indistinguishable at N=20
  and the rolling gap is the only evidence of instability. The gate is doing its
  job as designed (`stability()` compares the rolling curve), but a run whose
  measured final beats its measured best is not the failure mode the audit was
  built to catch. **Deliberately not changed on one run's evidence**: that guard
  was installed hours ago after 21 runs were mis-published, and loosening it
  against a single N=20 pair is exactly the mistake it exists to prevent. If
  several v1.11 arms show the same shape, that is the evidence to act on.

  **Residuals, consistent with the `squad_screen` arms and worse here**:
  false-DONE **0.830** final-decile (0.837 behavior suite) — `v7`'s 0.078 is not
  a real improvement, a collapsed policy claims nothing. Doctrine preference
  **0.115**, with the top three order tasks (OBSERVE 0.42, CLEAR 0.23, SUPPORT
  0.13) each at preference **0.00**. A5-2 staging: 23 staged, 5 released, **18
  abandoned**. None of these is touched by the free-ride fix and all are open.
- **2026-08-07** — **v1.11 fleet, arm 2/7: `squad_v8` — three of three.**

  | | `squad_v7` (pre-fix) | `squad_v8` (post-fix) |
  |---|---|---|
  | rolling best → final | 99% → **60%** (39-pt gap, **COLLAPSED**) | 100% → **96%** (4-pt gap, converged) |
  | publish gate | NOT PUBLISHABLE | **PUBLISHABLE** |
  | measured FINAL | — (artifacts died on the import bug) | **1.00 ± 0.00** |
  | terminal (final decile) | 0.0758 | **0.8664** |
  | ep length | 336 → 406 | 170 → **87** |

  Every collapse-prone scenario tested so far now holds: `squad_screen` (both
  seeds), `fireteam`, `squad`. The signature repeats exactly — terminal income up
  ~10×, episode length halved, no collapse.

  **CORRECTION to the two entries above.** I logged "a contact recall of **0.00**"
  as an open residual for `squad_screen_fallen_v2`, and `squad_v8` first appeared
  to repeat it. It is an artifact of *which checkpoint the suite scored*, not a
  behaviour. Contacts reported / recall / precision, `ckpt_best` → `ckpt_latest`:

  | run | `ckpt_best` | FINAL (`ckpt_latest`) |
  |---|---|---|
  | `squad_v8` | 0 reports, recall 0.00 | 54 reports, recall **0.87**, prec **0.91** |
  | `squad_screen_fallen_v2` | 0 reports, recall 0.00 | 34 reports, recall **0.90**, prec **0.94** |
  | `squad_screen_fallen_v1` | 27 reports, recall 0.50 | 54 reports, recall **0.93**, prec **0.98** |
  | `fireteam_v8` | 45 reports, recall 0.75 | 16 reports, recall **0.34**, prec 0.81 |

  So the residual as I stated it is **withdrawn**: the final policies report well,
  and on the three squad-family arms they report *far* better than the checkpoint
  that scored the best rolling window. That is a third independent argument for
  the FINAL being the headline — `ckpt_best` here captures a policy that wins
  without talking, and the run goes on to learn the reporting. `fireteam_v8`
  moves the other way (0.75 → 0.34) and is the one arm where reporting is a real
  open question.

  **Still open**: false-DONE **0.568** final-decile (0.750 on the suite), doctrine
  preference 0.391 with COVER at 0.00 preference on 22% of orders, and A5-2
  staging at 20 staged / 2 released / **18 abandoned** — the same abandon ratio as
  `fireteam_v8`, on a different scenario, which makes it a fleet-wide pattern
  rather than an arm's quirk.
- **2026-08-07** — **v1.11 fleet, arm 3/7: `squad_recon_v7` — four of four, with a
  caveat I introduced myself.**

  | | `squad_recon_v6` (pre-fix) | `squad_recon_v7` (post-fix) |
  |---|---|---|
  | rolling best → final | 97% → **0%** (97-pt gap, **COLLAPSED**) | 100% → **98%** (2-pt gap, converged) |
  | publish gate | NOT PUBLISHABLE | **PUBLISHABLE** |
  | measured FINAL | — | **1.00 ± 0.00** (`ckpt_best` 0.90 ± 0.13) |
  | terminal (final decile) | — | **1.2919** |
  | ep length | 220 → 374 | 98 → **59** |

  `v6` is the worst entry in the publish audit — **published at 91 ± 6 off a run
  whose rolling success ended at 0.00**. The same scenario, same seed, now ends
  at 98% and measures 1.00.

  **The caveat: this arm is confounded and the campaign file is why.** It claims
  budgets/seeds/lr are each scenario's own last run "unchanged on purpose", and
  that is true — but `ent_coef` and `gamma` are *not* in that list, and `v6` ran
  at **ent 0.02 / γ 0.99** against `v7`'s **ent 0.01 / γ 0.999**. So three things
  differ here, not one. Checked the other arms: `fireteam_v7`→`v8`, `squad_v7`→`v8`
  and `v9`/`v10`→`fallen` are all matched at ent 0.01, so **this is the only
  confounded arm.**

  It survives anyway, because the bisect already exonerated both confounders
  separately: `squad_screen_v9`/`v10` ran at **ent 0.01, γ 0.999** and collapsed,
  and `ctl_gamma099_v1` ran at **ent 0.01, γ 0.99** and collapsed. Neither value
  prevents the collapse on its own. The contrast is weaker than the other three
  arms and should be quoted as such.

  **A fleet-level pattern, now on four arms — `ckpt_best` is systematically the
  worse-behaved policy**, not merely the luckier one:

  | run | `ckpt_best` recall / false-DONE | FINAL recall / false-DONE |
  |---|---|---|
  | `squad_recon_v7` | **0.01** / **0.956** | **0.95** / **0.343** |
  | `squad_v8` | 0.00 / 0.750 | 0.87 / — |
  | `squad_screen_fallen_v2` | 0.00 / 0.500 | 0.90 / — |
  | `squad_screen_fallen_v1` | 0.50 / 0.600 | 0.93 / — |

  The best *rolling-window* checkpoint is a mute policy that wins by claiming
  completion falsely; the policy the run actually ends with talks, and claims
  honestly. That is a fourth independent argument for the FINAL being the
  headline, and it means the pre-audit fleet was not just over-quoted — it was
  quoting the wrong *behaviour*.

  **`platoon_v5` is now training** — 16 agents, the most dilution, and the one
  arm that converged at 92% *without* the fix. It is the test the diagnosis
  cannot currently explain.
- **2026-08-07** — **v1.11 fleet, arm 4/7: `platoon_v5` — five of five, the open
  question answered, and a muteness regression.**

  This was the arm the free-ride diagnosis could not explain: 16 agents, the most
  dilution, and it converged at 92% *without* the fix. The answer is that it was
  leaving a great deal on the table.

  | | `platoon_v4` (pre-fix) | `platoon_v5` (post-fix) |
  |---|---|---|
  | rolling best → final | 99% → 92% (7-pt gap) | 100% → **99%** (1-pt gap) |
  | measured | 0.93 ± 0.05 (N=100, `ckpt_best`) | **1.00 ± 0.00** FINAL |
  | terminal (final decile) | 0.2089 | **0.7060** |
  | ep length | 501 → 270 | 583 → **108** |
  | human death rate | 0.233 | **0.000** |
  | **retasks / ep** | **65.09** | **0.05** |
  | orders / ep | 75.75 | 6.70 |
  | messages / ep | 664 | 70 |

  **The churn hazard is gone.** 65.09 retasks per episode against 75.75 orders is
  the rotation-churn signature this repo keeps a regression test for; `v5` runs at
  **0.05**. Command traffic falls 10× and the platoon still wins more often. So
  the fix helps even the arm that never collapsed — `platoon` was not exempt from
  the free-ride, it was paying for it in churn instead of in collapse.

  **The regression, and it is the failure mode this file already named as the
  worse one.** MISSION COMPLETE claims, 20 episodes: `v4` **35** claims (80%
  false) → `v5` `ckpt_best` **3** claims (all three rejected) → `v5` **FINAL: 0
  claims**. The platoon has gone **mute**. My first read of the training line
  called this "false DONE 100%"; that rate is **n=3**, and the real finding is the
  denominator, not the ratio. A policy that never claims forfeits `root_done_bonus`
  (+3) on every episode and ends each one at T0 + `grace_window` instead of on its
  own report — it wins the mission and never says so, which for a project whose
  premise is that every C2 event reaches the transcript is a worse outcome than
  claiming badly.

  Consistent with it: obedience latency **3.9 → 21.0** steps (ADVANCE 21.1,
  OBSERVE 35.7), and staged orders released **1900/3179 (60%) → 4/93 (4%)**.

  **Mechanism NOT established** — logged as the next diagnosis, not as a story.
  `done_false` is −0.5 (the −2.0 experiment was reverted precisely because it
  bought silence, not precision), so the claim is cheap: break-even is p ≈ 0.14
  against `root_done_bonus`. The suspect is exploration, not price — final-decile
  entropy is **1.380** against `v4`'s **1.957**, and a policy this economical
  (6.7 orders/ep) may simply never sample DONE often enough to learn when it is
  true. **Refutable**: if it is exploration, `done_reports` should recover with a
  higher `ent_coef` at unchanged `done_false`. Probe before touching the price.

  **Fleet-wide DONE picture** (claims / rejected / false-rate), `ckpt_best` →
  FINAL: `squad_v8` 4/3/0.750 → 65/31/**0.477** · `squad_recon_v7` 113/108/0.956 →
  35/12/**0.343** · `fallen_v1` 10/6/0.600 → 36/7/**0.194** · `fallen_v2`
  2/1/0.500 → 35/10/**0.286** · `fireteam_v8` 288/241/0.837 → 87/79/**0.908**.
  Four of five improve substantially at the FINAL checkpoint. `fireteam_v8` is the
  one that gets *worse*, and with `platoon_v5`'s silence those two are the fleet's
  open reporting defects.
- **2026-08-07** — **v1.11 fleet, arm 5/7: `fireteam_defend_v11` — and a CORRECTION
  to the four entries above.**

  **The arm.** FINAL policy at **N=100: 0.74 ± 0.09**, against
  `fireteam_defend_v10`'s FINAL **0.87 ± 0.07** — both measured under identical
  current code, so it is a fair policy comparison. CIs overlap at 0.80–0.83, so
  by this file's own bar (*non-overlapping intervals or it is not an effect*)
  **it is not an effect** — but it is the only arm in the fleet that did not
  improve, 13 points below its pre-fix predecessor, fighting further out (3.80 vs
  2.52 cells) with less cover (0.776 vs 0.897). v1.10's prep period does work:
  in-cover-at-OBJ during prep **0.786**, and both v6's and v7's documented misses
  are gone (fire rate 0.750 where v6 would not fire; 4.06 cells where v7 fought
  9.7 out).

  **A hypothesis raised and refuted in the same pass.** `fireteam_defend` is the
  one scenario where survival *is* the mission — you hold ground to H+N, with no
  fast win to substitute — so removing forfeiture should make trading bodies
  cheaper with nothing to trade it for. Test: `v11`'s defeat (cohort-wiped) rate
  should exceed both predecessors'. It does not — **`v11` 0.150 sits between `v9`
  0.200 and `v10` 0.050**. Dropped.

  **⚠ THE CORRECTION.** Diffing `economics.json` across every fleet pair turns up
  exactly one non-commit difference, and it is in almost all of them:
  **`done_false` −2.0 → −0.5**, the v1.10 setting against the reverted one.

  | pair | status |
  |---|---|
  | `squad_screen_v9` → `fallen_v1` | **CLEAN** |
  | `squad_screen_v10` → `fallen_v2` | **CLEAN** |
  | `squad_v7` → `v8` · `squad_recon_v6` → `v7` · `platoon_v4` → `v5` · `fireteam_defend_v10` → `v11` | **CONFOUNDED** (`done_false`) |
  | `fireteam_v7` → `v8` | uncheckable — `v7` predates `economics.json` |

  So: **the D4 verdict stands and is untouched.** The `squad_screen` A/B is clean
  on both seeds — `v9`/`v10` and the `fallen` arms all ran at `done_false` −0.5,
  and at *fixed* `done_false` the fix is the difference between 0.00 and 1.00.

  **But "five of five, the fix generalizes" is DOWNGRADED to "consistent with
  generalization, not established."** Each of those four pairs differs by two
  variables, not one, and the second is not innocent: `rewards.py`'s own note says
  −2.0 bought *silence* and cost the report-centric scenarios their terminal
  income entirely (`squad_recon_v6`, `squad_screen_v4` both ended at 0%). The
  revert to −0.5 is therefore a live alternative explanation for why
  `squad`/`squad_recon`/`fireteam` stopped collapsing — and, symmetrically, for
  why `fireteam_defend` got *worse*: it is not report-centric, and it scored
  0.87–0.89 at −2.0.

  **This is my error and it is in the campaign file.** `v1_11_fleet.jobs` claims
  "the ONLY difference from the superseded fleet is the environment, so a miss
  stays attributable". True as written and wrong in effect — the environment
  carried *two* changes, and I pinned budgets, seeds and lr while leaving the
  economics to drift with the tree.

  **Next, and it needs a code change so it waits for the campaign to drain**:
  `done_false` is not on the CLI (only `PPOConfig` was exposed). Add it, then run
  one arm — `squad_v9` at `done_false` −2.0 with the fix — to separate the two.
  One run decides it.
- **2026-08-07** — **v1.11 fleet, arm 6/7: `patrol_brique_v5`, and a second CLEAN
  pair found in the queue.**

  **The arm.** 100% best → **99% final**, 1-pt gap, **PUBLISHABLE**, FINAL policy
  **1.00 ± 0.00**. Its predecessor `patrol_brique_v4` was already converged at 98%
  final, so there is little here to attribute — and the pair is **CONFOUNDED**
  (`done_false` −2.0 → −0.5) like the others. `v4` was built at `b8b3763`
  (08-06 22:24), an hour before the revert at `ac1fb19`. Worth recording anyway:
  ep length 250 → 80, human death 0.716 → 0.129, churn 16.6 retasks against 22.2
  orders.

  **The find.** `defend_brique_v3` was built at **`ac1fb19` — the revert commit
  itself** — so it ran at `done_false` −0.5 and **PRE-`d44ee8d`**. Its
  `economics.json` diffs CLEAN against `defend_brique_v4`, now training. That
  makes **`defend_brique_v3` → `v4` a second single-variable A/B for the free-ride
  fix**, on a scenario unrelated to `squad_screen`, arriving by luck of timing
  rather than design.

  It is the arm that decides the generalization question the correction above left
  open. `v3` finished at **91%**; if `v4` clears it with `done_false` held fixed,
  the fix generalizes on evidence rather than on consistency. Full pair status
  after the economics audit:

  | pair | status |
  |---|---|
  | `squad_screen_v9`/`v10` → `fallen_v1`/`v2` | **CLEAN** (the D4 verdict) |
  | `defend_brique_v3` → `v4` | **CLEAN** ← decides generalization |
  | `squad`, `squad_recon`, `platoon`, `fireteam_defend`, `patrol_brique` | CONFOUNDED |
  | `fireteam_v7` → `v8` | uncheckable |
- **2026-08-07** — **assurance #20: the confound audit is now mechanical, not
  manual.** Commit `80166d9` (dedicated fix agent). 497 tests green, ruff clean.

  The issue is about the audit two entries above. The external review diffed
  `squad_v6` → `squad_v8` and found **two** differing reward keys
  (`done_false` *and* `contact_redundant` −0.02 → −0.25); this file's table said
  one. Both are right for the pair each chose: `squad_v7` is the correct baseline
  here — it is the scenario's last run, and the campaign inherited *its* budget,
  seed and lr — but the review's real point lands anyway. **Two careful readings
  of the same artifacts produced two different answers, because the check was a
  JSON diff done by eye.** `train.py` writes `economics.json` precisely so that
  "two runs a reward commit apart are [not] indistinguishable after the fact",
  and nothing was reading it.

  Now `run_report.py <run> --vs <baseline>` diffs the `rewards` and `spec`
  sections automatically and prints `CLEAN` / `single-variable A/B` /
  `CONFOUNDED — N keys differ` with every differing key named; a missing
  `economics.json` reads as `uncheckable` instead of crashing. Verified against
  the live artifacts: `squad_v8 --vs squad_v6` → **CONFOUNDED, 2 keys**;
  `squad_v8 --vs squad_v7` → **single-variable A/B**; `fallen_v1 --vs
  squad_screen_v9` → **CLEAN**. It reproduces the review's finding and this
  file's claim, and the D4 pair still audits clean independently.

  Every `--vs` from here carries its own attribution verdict, so no future entry
  can quietly assert a single-variable A/B that the artifacts do not support.
- **2026-08-07** — **v1.11 fleet COMPLETE, arm 7/7 `defend_brique_v4`: the second
  clean pair says where the fix does not reach.** Commit `2957ae8`.

  **Final-policy numbers, all under identical current code**: `fireteam_v8`
  0.90 ± 0.13 · `squad_v8` **1.00 ± 0.00** · `squad_recon_v7` **1.00 ± 0.00** ·
  `platoon_v5` **1.00 ± 0.00** · `patrol_brique_v5` **1.00 ± 0.00** ·
  `fireteam_defend_v11` 0.74 ± 0.09 (N=100) · `defend_brique_v4` 0.91 ± 0.06
  (N=100). The collapse is gone everywhere it used to happen.

  **The clean pair, at N=100** (`defend_brique_v3` → `v4`, `done_false` held,
  `d44ee8d` the only difference):

  | | `v3` (pre-fix) | `v4` (post-fix) |
  |---|---|---|
  | success | 0.88 ± 0.06 | 0.91 ± 0.06 — **indistinguishable** |
  | commander death rate | 0.24 | **0.61** |
  | cover occupancy under threat | 0.513 | 0.416 |
  | dist from OBJ under threat | 2.87 | **6.09 — GATE FAIL** (bar ≤ 5.0) |
  | friendly deaths/ep (oracle) | 0.60 | **1.85** |

  Success is flat; the behaviour is worse. The commander dies 2.5× as often, and
  a **DEFEND** mission is fought twice as far from the objective it exists to
  hold — far enough that the FINAL policy **fails
  `mean_distance_from_objective_under_threat`**, one of this repo's encoded
  regression gates. `v3`'s final passes it at 2.87. Per the standing rule, a
  behaviour suite that contradicts the success rate outranks the success rate.

  **The prediction I withdrew was not wrong — it was unscoped.** Where a decisive
  objective exists (screen, recon, patrol, assault), engaging ends the episode
  sooner, so cover and survival *rise* as instruments of a terminal now reachable
  by everyone: `squad_screen` went 165 → 53 steps with deaths halved and cover up
  15×. In a defend scenario there is no fast win — the mission is to still be
  there later — so removing forfeiture makes bodies cheap with nothing to buy.
  Both defend scenarios agree: `fireteam_defend_v11` (confounded, same direction)
  is 0.87 → 0.74 with dist 2.52 → 3.80 and cover 0.897 → 0.776.

  **VERDICT: `d44ee8d` fixes the collapse — cleanly, two seeds, 0.00 → 1.00 — and
  regresses defend-type scenarios.** Both halves are now on clean single-variable
  evidence.

  **⚑ OWNER'S CALL — reward structure, not taken here.** Options, with the
  measurement each is aimed at:
  1. **Scope the payout by scenario** — pay the fallen the team terminal only
     where a decisive objective exists; keep forfeiture in defend-type scenarios.
     *Fits the mechanism*, which is structural (fast win available or not), and
     costs nothing: no defend scenario ever collapsed. **Recommended.**
  2. **Price cover under threat** (currently worth exactly zero). Targets the
     measured defect directly and is scenario-agnostic — but `squad_screen`
     reached cover 0.245 *without* it, so it may distort where nothing is broken.
  3. **Raise `death`** from −1.0, now commensurate with a +60 terminal that no
     longer forfeits. Blunt; hits every scenario including the five that are fine.
  4. **Make the defend terminal proportional to survivors** — the mission is to
     hold, so pay for holding *with a force*. Most faithful to doctrine, largest
     redesign.

  Held pending that call, and **not** attempted: `done_false` is not on the CLI,
  so the one run that separates `d44ee8d` from the `done_false` revert on the five
  confounded arms (`squad_v9` at −2.0 with the fix) still needs a small CLI
  change first.

- **2026-08-07 — v1.12: the owner took option 4, and the CLI blocker is gone.**
  Two commits, both untrained; spaces unchanged at Discrete(228)/Box(220).

  **`b8ed7f1` — reward weights on the CLI.** `--reward KEY=VALUE`, repeatable,
  typed off the dataclass so it cannot go stale when a weight is added.
  `squad_v9` is now `--reward done_false=-2.0` and needs no tree edit — which
  also removes the hazard that killed `fireteam_defend_v10`. Three *silent*
  failure modes closed rather than documented: `economics.json` dumped
  `RewardConfig()` instead of the prices in use, so every override-driven A/B
  would have read as a no-op change to the assurance-#20 confound audit — the
  audit reporting "nothing differs" about the one thing that does; checkpoints
  carried no prices, so `evaluate` scored every policy under tree defaults and
  `mean_return` invited comparisons it could not support (pre-v1.12 checkpoints
  have no such key and correctly fall back to the defaults they trained under);
  and an unknown key raised nothing, so a typo trained the default price under
  an `economics.json` claiming otherwise. Booleans are parsed by name because
  `bool("false")` is `True` — a coerce-by-constructor reads
  `fire_discipline=false` as ON and trains the opposite experiment. Plus a
  pre-flight dominance warning, since the v1.11 collapse economics is now one
  keystroke away: `--reward success_team=10` scores 1.37x on fireteam. A
  warning, not an error — ablating below the bar is legitimate here; doing it
  by accident and reading the wreckage as a finding about something else is not.

  **`f39b5a9` — option 4, the survivor-scaled defend terminal.** On DEFEND/DENY
  roots only, the payout is multiplied by
  `(1 - scale) + scale x surviving_weight / starting_weight`.

  *Why it is not forfeiture again*, which is the whole design question:
  forfeiture caused D4 because the gain from hanging back is visible to a
  per-agent advantage and the collective cost is not. This multiplier is
  **identical for every agent, fallen included**, so a death is a shared loss.
  The residual private gain is the ~1/n of the multiplier your own body
  accounts for — on a fireteam at most `60/4 x 0.35 ≈ 5.3` against the same
  −52.3, an order of magnitude short of the D4 arithmetic rather than
  comparable to it.

  *Why 0.35 and not more.* The invariant, not taste. The multiplier can only
  scale the terminal down, so `win_beats_stall` has to clear 2x at the **floor**
  — a force that holds and is ground down doing it, which is the case a defend
  scenario is actually about. `fireteam_defend` scores 3.42 undiminished, so the
  ceiling is `1 - 2.0/3.42 = 0.415`; 0.35 leaves 2.22. Confirmed by mutation:
  0.45 and 0.5 both fail `test_defend_terminal_scaling_preserves_dominance`,
  which names the ceiling in its message. Note this made option 4 *cheaper* than
  it looked — "largest redesign" overstated it; the terminal was already a
  single payout site, and scoping plus one multiplier was the whole change.

  *One trap worth recording.* Bodies are rank-weighted like casualties already
  are, but by **intrinsic** rank, not `effective_authority`. Succession promotes
  a survivor into the dead leader's slot, so an effective-authority sum over the
  living RISES after the commander falls — a cohort could raise its own terminal
  by getting its leader killed. Measured, that mutation yields 0.9028 against
  the correct 0.8971, and the test catches it. On a fireteam the commander is
  worth 60 → 52.0 against a rifleman's 60 → 54.4; commander death (0.24 → 0.61)
  was the measured half of the regression.

  **Not judged.** This is economics, not a result. The A/B is `defend_brique_v5`
  and `fireteam_defend_v12` against `v4`/`v11`, single-variable — and the gate
  to watch is `mean_distance_from_objective_under_threat` (bar ≤ 5.0), which v4
  fails at 6.09 and v3 passes at 2.87. **The owner pre-authorised the fallback**:
  if it goes against option 4, revert to option 1, which needs no code change —
  `--reward defend_survivor_scale=0` restores v1.11 behaviour exactly.

  Also still open and now unblocked, independent of the above: `squad_v9`.

- **2026-08-07 — assurance #21: premise CONFIRMED, and the instrument that
  confirmed it just got a second axis.** Not a bug fix — no reward code
  touched, no reward weight or semantic changed, v1.12's survivor-scaled
  defend terminal is untouched.

  Issue #21 pre-registered a falsifier for the owner's-call line "costs
  nothing: no defend scenario ever collapsed" (option 1's justification for
  scoping the payout by scenario) — read as the D4 **stall** (≈0.00 success,
  clock runs out). **Confirmed**: ten defend corpora over five generations
  run 2–7/30 timeout everywhere on record; the stall signature reads 28–30/30
  in this repo's history. No defend scenario has ever stalled.

  The scoping note, which is why the issue exists: the defend family's worst
  measured runs do not collapse by stalling — they collapse by getting
  **wiped**, and that shape was invisible to the instrument tuned to the
  other one. Four corpora on record: `fireteam_defend_v6b` 1/30 success at
  2/30 timeout (27/30 defeat); `fireteam_defend_v6` 14/30 at 4/30 timeout;
  `fireteam_defend_v7` 12/30 at 7/30 timeout; `squad_screen_v7` (not defend,
  same shape) 6/30 at 0/30 timeout. `timeout_rate` reads all four as healthy
  on the clock. The repo's own composite gate (`human_death_rate` gated on
  `timeout_rate ≤ 0.5`) happened to catch all four anyway — but on the death
  axis, because a wiped team's commander usually dies with it. Right about
  every run on record, for a reason other than the one it names, and with no
  axis of its own for "the team lost."

  **Added**: `SUCCESS_RATE_FLOOR = 0.5` (`cohort/metrics.py`), a floor on
  `success_rate` in `regression_gates`, gated only once `timeout_rate` has
  already cleared `TIMEOUT_RATE_CEILING` — i.e. only once the run is known
  *not* to be stall-shaped. The bound sits in the empty band between the
  highest documented defeat-shaped corpus (`fireteam_defend_v6`, 0.467) and
  the lowest healthy record on file (`fireteam_defend_v11`, 0.74); it does
  not fire on `fireteam_v8` (0.90) or the fleet's 1.00 runs either. Ordering
  the check on `timeout_rate` first keeps the two axes mutually exclusive in
  a report by construction: a collapsed run reads as **STALLED**
  (`timeout_rate` fails) or **WIPED** (`success_rate` fails), never both,
  which matters because the two shapes want opposite fixes. `run_report.py`
  needed no change — it already prints every gate in `behavior.json`
  generically by name.

  Tests: 530 → 534 (`tests/test_metrics.py`), fixtured on the exact corpora
  above plus the healthy fleet, and three pre-existing positional-gate tests
  updated for the new gate now appearing alongside `timeout_rate`.
  Mutation-checked by hand: zeroing `SUCCESS_RATE_FLOOR` and removing the
  `timeout_rate`-first ordering each broke a distinct new test, then both
  were reverted. `pytest` green, `ruff` clean.

  **Not answered, and not this change's job**: the issue's closing question —
  can a defeat-shaped collapse leave the commander alive? — has no case on
  record either way. The new axis reads `success_rate` directly and does not
  consult `human_death_rate`, so going forward the two are independently
  measured and a defeat-shaped collapse with a surviving commander *would*
  now show up as a `success_rate`-only failure instead of nothing — but that
  is a statement about what the instrument can now see, not a claim that such
  a case exists. It doesn't, yet, in anything we've trained.

- **2026-08-08 — assurance #22: the digest could not settle the A/B it was
  filed about.** Issue #22 is a pre-registration, not a defect report: it
  commits, before `defend_brique_v5` / `fireteam_defend_v12` exist, to how the
  option-4 A/B will be judged. Every prediction in it is stated **at the final
  policy** — primary root deaths ≤ 12/30 (falsifier ≥ 20/30), positional gate
  passes, cover ≥ 0.40, success ≥ 20/30 — and its method note is explicit
  about why: §12.76 found the `v3`→`v4` regression **invisible at
  `ckpt_best`**, appearing only at final.

  `run_report.py` could not answer any of them. It printed the full behavior
  suite plus every gate for `ckpt_best`, and for the FINAL policy **one
  success number**. So (a) the pre-registered PRIMARY axis was absent at either
  checkpoint — `human_death_rate` is aggregated on every behavior run and
  printed by `evaluate`'s own table, but this digest, the only artifact a
  verdict is written against, dropped it; and (b) a gate the headline policy
  **fails** was hidden behind `ckpt_best`'s PASS. `defend_brique_v4` is that
  case on file: three PASSes at `ckpt_best`, and
  `mean_distance_from_objective_under_threat` **6.09 against a bound of 5.0**
  at `ckpt_latest` — the exact regression the whole v1.12 reward decision was
  taken over, absent from the digest of the run that produced it.

  **Changed**: the behavior block is one function called twice — `ckpt_best`
  under `beh_`, the final policy under `final_` — so both get the same rows and
  the same gates, and both sets of metrics enter the `--vs` delta. Added
  `human_death_rate` ("root death rate") to the printed rows. The four
  command-quality composites (task mix, availability lift, per-task obedience,
  staging) stay on the `ckpt_best` block alone: they diagnose how the cohort
  commands rather than whether the run cleared its bars, and the digest's whole
  point is to stay short. Digest length ~30 → ~50 lines; `docs/training.md`
  updated to say so. **No reward code, weight, or semantic touched** —
  `defend_survivor_scale` is exactly as `f39b5a9` left it.

  Also from the issue's verified findings: its §1 notes that **no scenario
  roots on `DENY`**, so that half of `root_mission in (DEFEND, DENY)` is dead
  code today and nothing exercised it. Pinned with one test covering both
  scaling sites (`RewardConfig.terminal_scale_floor` and
  `CohortEnv._defend_terminal_scale`) — narrowing the branch to DEFEND-only
  would otherwise pass every test and surface as an unexplained price shift on
  the first DENY scenario anyone writes.

  Tests 534 → 537. Mutation-checked by hand: reverting `run_report.py` broke
  both new digest tests, and dropping `MissionType.DENY` from the four
  branch sites broke the new DENY test. `pytest` green (537), `ruff` clean.

  **What this does not do**: it does not judge the A/B. With the digest fixed,
  `defend_brique_v5` vs `v4` reads `final_human_death_rate` **0.61 → 0.40** on
  the pre-registered primary — but `v5`'s final block is N=20 against `v4`'s
  N=100, so the two sides are not measured at equal power and the call belongs
  to whoever runs `evaluate` on `ckpt_latest` at matched episodes. Not run
  here, and no training launched.

- **2026-08-08** — **v1.12 A/B resolved: option 4 confirmed on BOTH defend
  scenarios — but only after `defend_brique` was repaired.** The result at
  matched power, final policy, N=100, seed 123 (the fireteam pair needed
  re-evaluating: the treatment arms were sitting at N=20 against N=100
  baselines, so the first read of them was underpowered and partly wrong).

  `fireteam_defend`, `v11` (flat) → `v12` (0.35): root deaths **0.35 → 0.15**
  (p = 0.001), success **0.74 → 0.86** (p = 0.034), defeats **13 → 0**, all
  four final-policy gates pass. Unambiguous.

  `defend_brique`, `v4` → `v5`: primary landed at **14.4/30 root deaths**,
  inside the 13–19 band #22 declared **partial** in advance. Both of its
  positional predictions were **falsified** (cover 0.273 against a predicted
  ≥ 0.40; distance 6.10 against a bound of 5.0, unmoved). Success *fell*
  0.91 → 0.80 (p = 0.027). We went looking for why.

  **The scenario could not be defended.** `defend_brique` declared a `DEFEND`
  root and `objective_cover=True` — it built defensible ground — and then set
  `assault_h_hour=None`, so the band ran from step 0 and the fire team never
  had a moment to occupy it. Not a hard defense: a meeting engagement on
  defensible ground. `_v4` fails the positional gate at its final policy too
  (6.09), so **the miss predates the survivor-scaled terminal and the reward
  A/B on this scenario was scoring a broken instrument.** Fixed in `450b392`
  with v1.10's arithmetic: `assault_h_hour=(35, 55)` and `max_steps` 375 →
  420, buying the preparation rather than taking it out of the 375-step fight.
  Narrower and earlier than `fireteam_defend`'s (55, 75) — a band infiltrating
  from the far edges gives less warning than a formed assault. Everything that
  makes brique the hard one is untouched (5 enemies to 4, mines, the
  probe/harass/raid intent machine, and now *less* warning than the easier
  defend scenario gets). **Not a difficulty giveaway, measured**: masked-random
  is 0/40 success before and after.

  The repair was the dominant variable. `v5` → `v6` is single-variable at
  constant reward (both 0.35): success **0.80 → 0.97** (p = 0.0002), root
  deaths **0.48 → 0.05** (p < 0.0001), defeats 11 → 0, cover 0.273 → 0.799,
  distance 6.10 → **3.72 (PASS)**. First brique run ever to clear the publish
  gate (best-final gap 6 pts).

  **The A/B re-asked with the scenario held constant** (`v6` 0.35 vs `v7`
  flat, `economics.json` confirms single-variable): success **0.89 → 0.97**
  (p = 0.027), timeouts **0.110 → 0.030**, episode length 126 → 109, cover
  identical (0.802 / 0.799), root deaths **0.070 → 0.050 (n.s., p = 0.55)**.
  Both arms publishable, both pass all four gates. Option 4 is kept at 0.35.

  Note the mechanism honestly: on the repaired brique the multiplier is **no
  longer buying casualty reduction** — the preparation period already drove
  deaths to a 5–7% floor where the multiplier has no room to act. What it buys
  there is *decisiveness* (timeouts 11% → 3%, 17 steps shorter). The
  casualty-preservation effect is real and large, but it is `fireteam_defend`
  that demonstrates it. Costs on `v6`: obedience latency 3.14 → 6.58 and
  distance-under-threat 2.10 → 3.72 (both still inside the gate).

  **Biggest open defect, unrelated to either change**: false DONE sits at
  **0.80** at the final policy in both brique arms and **1.00** in both
  fireteam arms. It is saturated across the whole defend family regardless of
  reward or spec, and it is now the largest thing wrong with these scenarios.
  `--reward done_false=-2.0` is the ready lever and has never been trained.

- **2026-08-09** — **the DONE channel is priced, not unlearnable — and the
  lever this file named for it is backwards.** Diagnosis first, per the
  operating guide; no reward changed, nothing shipped on the strength of it.

  `false_complete_rate` is `done_rejected / done_reports`, and yesterday's
  entry read it as one saturated pathology across the defend family. It is
  **two opposite pathologies wearing one number**. `fireteam_defend_v12`:
  **1 DONE report in 100 episodes**, rejected, so the rate reads 1.00 off a
  denominator of one — 86 successes, **zero** accepted reports, against
  16,152 root-admissible agent-steps. The channel is not noisy, it is
  **dead**, and the mask is wide open (161 admissible steps per 163-step
  episode), so this is a declined act, not a reachability bug — exactly the
  reading `metrics._done_opportunity`'s docstring was written to enable.
  `defend_brique_v6`: 4.4 claims/ep, 80% rejected, but **0.89 accepted per
  episode** — essentially every success does end with a report; the policy
  simply spams until one lands, inside the existing `done_cooldown=8`.

  **One margin explains both.** A claim pays `done_true + root_done_bonus` =
  +4.0 inside the grace window and `done_false` = −0.5 outside, so a blind
  claim breaks even at P = 0.5/4.5 = **0.111**. Grace is 12 steps of a
  **114.6**-step mean `fireteam_defend` success → P = **0.105**, EV
  **−0.029** → *never claim is optimal play*. On `defend_brique` it is 12 of
  **99.1** (v6) and **89.5** (v7) → P = **0.121** / **0.134**, EV **+0.045** /
  **+0.103** → *claiming pays*. The two scenarios straddle break-even by a
  hair, and each policy does the arithmetic correctly. Reinforced by
  observability: plain-DEFEND success is `not living_enemies` — global
  knowledge the root cannot see (the obs carries `known_enemies`, i.e. what
  was *spotted*, which reads 0 both before first contact and after the last
  kill) — while the BRIQUE branch's "scattered with contact broken" is
  locally observable. The lineage agrees: `fireteam_defend` has never
  established this channel (v5 0.00, v6 0.90/0 accepted, v8 0.00, v9 1.88/0.62
  — the peak — v10 1.22/0.40, v11 0.02/0, v12 0.01/0), while the SEIZE-rooted
  `fireteam` reports 3–14 claims/ep throughout.

  **So `--reward done_false=-2.0` is the wrong lever** — it moves break-even
  to **0.333**, above *both* scenarios' blind P. It would kill brique's
  working channel and leave fireteam at zero: it suppresses claiming, it
  cannot teach correct claiming. This file named it as "the ready lever" and
  yesterday's session recommended it; both were wrong, and the note in the
  handoff is corrected rather than quietly dropped.

  Testing rather than asserting (`campaigns/done_channel_v1_13.jobs`, single
  variable against `v12` / `v6`, `defend_survivor_scale` held at the confirmed
  0.35): `fireteam_defend_v13` at `done_false=-0.1` (break-even 0.024, well
  under 0.105) **predicts the channel opens**; `fireteam_defend_v14` at −2.0
  **predicts no help**; `defend_brique_v8` at −2.0 **predicts it kills a
  channel that works**. Falsified if v13 does not raise accepted DONE/ep, or
  if v14/v8 move the way this file used to assume.

  **What is NOT decided here**: if the economics test confirms, the deeper
  question is still the owner's — repricing buys claiming but not
  *discrimination*, and on a plain-DEFEND root the completion condition is
  genuinely not observable. Options are (a) reprice only, (b) align DEFEND
  success with what a defender can perceive, as the BRIQUE branch already
  does, (c) make completion observable in the obs — which is a breaking
  cycle, OBS_DIM and the whole checkpoint fleet. Not autopiloted.

- **2026-08-09** — **v1.13: COMMAND ends a defense, not the section holding
  the ground.** Owner's decision, and it retires the false-DONE thread by
  dissolving it rather than pricing it.

  The point: a DEFEND is not a task with an end state its holder may declare.
  It is held until relieved or re-tasked, so the order that ends it comes
  DOWN the chain. `cohort/core/missions.py` already said so —`COMPLETABLE`
  excludes DEFEND/DENY because they "end when a new order arrives" — and the
  env carved around its own doctrine table: `is_root_opord_claim` (v1.4) let
  a DEFEND root declare the *operation* complete. That was added to repair a
  real bug (mask and adjudicator disagreed, the root could never claim,
  `root_done_bonus` was dead reward), and it opened the wrong door.

  **The same measurement, misread twice, now settled.** `fireteam_defend_v12`
  filed ONE claim in 100 episodes against 16,152 admissible agent-steps.
  2026-08-08 called it the family's largest defect; the 2026-08-09 entry
  above called it a dead channel priced shut. Both were looking at a policy
  correctly declining an act it should never have been offered. The economics
  in that entry are still right as arithmetic — and the half-run
  `fireteam_defend_v13` confirmed them, opening the channel to **89% false
  DONE** at `done_false=-0.1` before it was killed — but they were the price
  of the wrong act. Repricing bought spam; it was never going to buy
  discrimination, because there was nothing legitimate to discriminate.

  **Now**: `is_root_opord_claim` requires the root mission to be in
  `COMPLETABLE`, so MISSION COMPLETE is masked shut on a continuous posture.
  The root reports the situation and COMMAND transmits **ENDEX** (new
  `MessageKind` + formatter, no parser — same shape as every other auto-kind).
  Same grace window, same `root_done_bonus`, opposite direction on the net: a
  SITREP once the end state holds closes the operation early and pays, and
  COMMAND closes it either way. Masking-only on the action space —
  **Discrete(228)/Box(220) unchanged, the whole fleet still loads**.
  Spot-checked on v12's own checkpoint: 3/3 success, 0 DONE, and
  `[t=105] TL1, THIS IS HQ: ENDEX. OUT.` on the transcript.

  **New signal, because the old one goes vacuous.** `false_complete_rate` is
  structurally 0 on a defend scenario now. Replaced by
  **`closed_on_root_report_rate`**: of the operations COMMAND closed, how many
  the root's own report closed early. `None` (not 0) when no ENDEX was sent,
  so a SEIZE root does not read as "never reported" — that is exactly the
  denominator confusion that made v12's single claim read as a 1.00 failure
  rate, and it is now impossible to repeat in the other direction. v12's
  checkpoint measures **0.22** under the new rule. *(Corrected 2026-08-09,
  refs #24: that figure named no checkpoint and no N, and re-scoring v12's own
  policy at N=100 reproduces neither — 0.19 at `ckpt_best`, 0.47 at
  `ckpt_latest`. See the entry at the foot of this log.)*

  `test_root_opord_claim.py` is rewritten, not deleted: it pins the reversal
  *and* the hazard that outlived it (mask and adjudicator must never drift
  apart). Both halves mutation-checked by hand. Tests 538 → 540, ruff clean.

  **Training**: `campaigns/endex_v1_13.jobs` — `fireteam_defend_v15` and
  `defend_brique_v9`, `defend_survivor_scale` held at 0.35, same seeds and
  budgets as the arms they replace, so the variable is the close rule. Watch
  `closed_on_root_report_rate` against 0.22, with success/root-death expected
  to hold at the v1.12 levels (fireteam 0.86/0.15, brique 0.97/0.05).

  **Left open**: `defend_brique`'s success is "band destroyed, or scattered
  with contact broken, objective held" — a *neutralize the band* end state,
  which is genuinely completable. It is rooted DEFEND because it is also a
  hold. ENDEX covers it correctly either way, but the taxonomy is worth a
  look: that scenario may want a different root mission rather than a
  different close rule.

- **2026-08-09** — **assurance #24: the boards were reading a family norm as a
  finding about one run, and quoting a baseline nothing on disk backed.** Two
  of the three open threads on the program board named a level, not a
  movement, and both levels turn out to be scenario-typical. Confirmed against
  this repo's *own* committed evaluations, not only the reviewer's series:

  | family | earlier generations on disk | the run the thread named |
  |---|---|---|
  | `fireteam` false-COMPLETE | `v4d` 0.76, `v5b` 0.82, `v6` 0.87, `v5` 0.88 | `v8` **0.84 best → 0.91 final** |
  | `platoon` false-COMPLETE | `v2` 0.66, `v4` 0.80; `v3` filed **0 claims** in 100 eps and still succeeded | `v5` 3 claims → **0** |

  So "claims completion at nearly every opportunity" and "has gone mute" both
  described the family. Worse, the first was also wrong on its own terms: 0.91
  is the share of claims the net *rejects*; `done_claim_rate` says `v8` takes
  the act at **0.028** of the agent-steps where the mask offers it. The
  surviving finding in each case is the within-run delta — contact recall
  0.75 → 0.34 on `v8`, obedience latency 3.9 → 17.0 on `platoon` — and the
  threads now lead with it.

  **The structural fix, so this is not just a re-write**: `_family()` in
  `scripts/program_board.py` renders, beside any thread that leads with a
  level, that metric's spread across the scenario's other generations, read
  off disk like everything else on the page. It widens by itself as runs land.
  `tests/test_program_board.py` pins it (7 tests), including that a prefix
  never reaches into a neighbouring scenario (`fireteam_v` must not eat
  `fireteam_defend_v*`).

  **The 0.22 baseline is corrected and now has a source.** It was published on
  two boards and in this log with no checkpoint and no N, for a metric that
  moves 2.5× between the two checkpoints of that one run. Re-scored under the
  ENDEX rule, `fireteam_defend_v12` closes **0.19** (`ckpt_best`, n=81 ENDEX)
  and **0.47** (`ckpt_latest`, n=86), N=100 seed 123 —
  `runs/fireteam_defend_v12/endex_rescore.json`, committed beside the run, and
  the board prints both rows with their N. Direction is unchanged: `v15`/`v9`
  score 1.00, so the retrain still clears the bar at either checkpoint. The
  reviewer's independent replay (0.115/0.593, seeds 500–529 × 30) agrees on
  the shape and not the level — different instrument, and neither of us
  reproduces 0.22.

  **One more they were right about.** "Every arm passes all four behavior
  gates" is true and reads as a clean bill: no gate bounds commander survival.
  The option-4 card now carries `human_death_rate` at final as its own panel —
  fireteam_defend 0.35 → 0.15, defend_brique 0.07 → 0.05 — captioned *no gate
  covers this*. Improvement, not a clean bill.

  Not changed: no reward default, no space, no scenario semantics. Boards
  re-rendered and flagged PUBLISH PENDING; publishing is a session action.

- **2026-08-09** — **assurance #23: the blind-claim premise is refuted, the
  campaign it pre-registers is already dead, and half its discriminator is now
  in the suite.** The issue pre-registers `fireteam_defend_v13/v14` and
  `defend_brique_v8` on the `done_false` pricing question. That campaign was
  killed by v1.13 and is **not** being restarted; what is actionable is the
  premise check and the instrument, and both are handled here.

  **The premise correction, and it is a miss on this side.** `8c839ef` priced a
  MISSION COMPLETE against **blind P** — the chance a claim filed at a random
  admissible step happens to be truthful — and derived break-evens of 0.111 at
  `done_false=-0.5` and 0.333 at −2.0. Measured against every corpus with a
  live channel, realised acceptance runs **2–10× blind P** (their measurement,
  70 pinned corpora, `results/done_channel.json`): `fireteam_defend_v10`
  **13 claims, 13 accepted** against blind P 0.100; `v9` 0.654 vs 0.099;
  `squad_screen_v5` 0.500 vs 0.097; `defend_brique_v6`/latest 0.211 vs 0.144.
  A policy that can *time* the act is not the claimant that arithmetic
  describes, so the break-evens do not describe the choice any policy on record
  faced — and the conclusion "−2.0 is backwards, −0.1 is the ready lever"
  **cannot be settled from that model in either direction**. That entry stated
  it as settled; it was not. Their endogeneity note compounds it and is
  accepted too: P is estimated from episode lengths, and a truthful root claim
  *ends* an episode, so the estimate is shortened exactly where the channel is
  alive. Any future pricing arithmetic must take P from a fixed reference
  policy, not from the arm under evaluation.

  **Why the campaign is not restarting.** v1.13 (`16cb2a6`, owner's decision)
  dissolved the question rather than pricing it: MISSION COMPLETE is masked
  shut on a continuous posture, the root reports and COMMAND transmits ENDEX.
  `false_complete_rate` is structurally 0/None on defend scenarios now and was
  replaced by `closed_on_root_report_rate`; the replacement campaign
  `endex_v1_13` landed at **1.00 on both arms** against v12's re-scored
  0.19/0.47. So the pre-registration is unadjudicable rather than refuted:
  `v14` and `defend_brique_v8` will never run, and `v13` exists only as a
  2.17M/3.5M partial. Recorded, not scored.

  **One line of it does bear on a live owner option.** Their point 2 is that
  `fireteam_defend_v10`'s 13-for-13 is an existence proof that the plain-DEFEND
  completion condition was *already inferable from what the root can see*. If
  option (c) — make completion observable in the obs, breaking `OBS_DIM` and
  the whole checkpoint fleet — ever comes back to the table, it needs a
  justification other than observability. Owner's call; flagged, not taken.

  **Their discriminator, judged on the merits and adopted in part**
  (`cohort/metrics.py`, +5 tests). Added: `done_claims_per_claiming_episode`
  and its root-only twin, with `done_reports_root` / `done_rejected_root` /
  `false_complete_rate_root` and the claiming-episode counts. **Not** added:
  "realised acceptance" as a named metric — every DONE is adjudicated on the
  step it is transmitted (DONE_CONFIRM or DONE_REJECT, never neither), so
  accepted ≡ reports − rejected and realised acceptance ≡
  `1 - false_complete_rate` at each level. A second name for a number the
  suite already carries is noise. What was genuinely missing is **volume
  against the episodes that carried it**, and the root/subordinate split: the
  root's channel is the one that closes an operation, and `done_admissible_root`
  has had no numerator since refs #13. Measured today, `ckpt_best`, N=20,
  seed 123:

  | run | claims / claiming ep | root's | root rejected | episodes claiming |
  |---|---|---|---|---|
  | `fireteam_v8` | **14.40** | 11.15 | 0.94 | 20/20 |
  | `squad_screen_v5` | **3.00** | 2.77 | 0.67 | 14/20 |

  Two policies whose pooled rejection ratios are 0.84 and 0.69 — near enough to
  read as the same failure — behaving five times apart on the axis that says
  whether a channel carries reports or spam. That is the issue's point,
  reproduced on this repo's own instrument. Their method ask is satisfied by
  the behavior table, which now prints concentration and the root split beside
  `false_complete_rate`. **Caveat**: metrics are computed at evaluation time,
  so the committed `behavior.json` fleet does not carry these keys — they
  appear from the next evaluation onward, and any cross-fleet reading of them
  needs a re-score.

  Not changed: no reward default, no space, no scenario semantics, and the
  pricing decision stays closed. Tests 563 → 568, ruff clean.

- **2026-08-09** — **`fireteam_defend_v15` publishes: the close rule works, and
  success held.** N=100, seed 123, final policy **0.84 ± 0.07** against
  `fireteam_defend_v12`'s **0.86 ± 0.07**. The intervals overlap almost
  entirely — this is a hold, not an improvement, which is what the v1.13 entry
  predicted and required. Peak `ckpt_best` 0.83 ± 0.07. First row of the new
  v1.13 README table, which publishes FINAL numbers.

  **The variable under test moved and nothing else had to.**
  `closed_on_root_report_rate` **0.99**, against **0.47** for v12's own
  `ckpt_latest` re-scored under the same rule
  (`runs/fireteam_defend_v12/endex_rescore.json`, refs #24). Same scenario,
  same rule, same N — the close rule is the only difference. All four behavior
  gates pass on both checkpoints (timeout 0.16, success 0.84, cover 0.904,
  distance 2.39).

  **One gain, one cost, both stated.** `human_death_rate` 0.15 → **0.08**,
  roughly halved, on an axis no gate covers. Against it, contact reporting —
  read against **both** of v12's checkpoints, the way the success column already
  is, because the sign of half of this depends on which one is the reference
  (amended refs #25; as first written the line quoted the `v12`/final row alone,
  unlabelled, and called both axes degraded):

  | reference → `v15` final | precision | recall |
  |---|---|---|
  | `v12` final (N=100) | 0.562 → **0.465** (−0.097) | 0.786 → **0.699** (−0.086) |
  | `v12` best (N=20) | 0.702 → **0.465** (−0.237) | 0.628 → **0.699** (**+0.071**) |

  **Precision degrades against either reference; recall improves against one of
  them.** And `v12`'s own two checkpoints move further than either delta —
  precision 0.702 → 0.562 (−0.140), recall 0.628 → 0.786 (+0.158) — so the
  reference moves more than the effect being read off it. The caveat cuts back:
  `v12`/best is N=20 where everything else here is N=100, so it is the noisier
  number — which is the point rather than a reason to drop it, since an unstable
  reference cannot anchor a 0.09 delta. The open question survives on precision
  only, and no mechanism is identified for that either.

  Precision is the half with nothing to qualify. It degrades under **all four**
  pairings of the two runs' checkpoints — final/final −0.097, `v12`/final →
  `v15`/best −0.089, best/best −0.229, `v12`/best → `v15`/final −0.237 — and
  `v15`'s own two checkpoints sit 0.473/0.465, a −0.008 spread. On this scenario
  it is the *reference* that is unstable and the arm under test that is not.

  **But no mechanism is claimed from it, in either direction** (amended
  refs #26). The reading this entry first invited — that the closing SITREP
  crowds out contact reporting — is unsupported. So is the counter-example that
  was offered to retire it (refs #25: `defend_brique_v6` old rule →
  `defend_brique_v9` ENDEX, precision 0.437 → **0.590**, recall
  0.834 → **0.841**, final policy, N=100, seed 123). Those figures are correct,
  but `v9`'s own two checkpoints are 0.439 → 0.590, so that entire +0.152 is
  within-run movement and the best/best pairing reads −0.008 (with `v6`/best
  that run's N=20 exit evaluation, the same caveat as `v12`/best above). Which
  `v9` checkpoint you read decides the sign — and −0.008 is flat, not the matching
  decline crowding-out would predict, so `defend_brique` neither refutes the
  reading nor restores it. **Contact precision is not stable enough across
  checkpoints in this family to support a between-run mechanism claim in either
  direction.** Net-side contact *volume* is not an alternative route to one: it
  disagrees across seeds on this scenario (refs #26 entry below). `v9` was cited
  here as measured evidence rather than a published result; it has since
  published in `b4a0d6d` as a priced regression on success and is in the v1.13
  README table.

  **Two caveats the tooling rounds away.** The stability give-back is **9.82
  points** against a bar of `< 10` — `publish_audit.py` prints `10` at
  `:>6.0f`, so this is a borderline pass and is published as one. And
  `false_complete_rate` reads 1.00 over **n=2** DONE reports in 100 episodes:
  root claims are masked shut under v1.13, so those are subordinate claims and
  the ratio is not a rate of anything. Transparency probe unchanged and still
  trailing (destination 0.471, −0.498 vs majority; RFN1 0.012).

  Supersedes `v11`/`v12` as the published `fireteam_defend` policy. Both stay on
  disk: `v12` is the option-4 evidence this file cites *and* the ENDEX baseline,
  and its committed evaluations are untouched. `defend_brique_v9` is still at its
  N=20 exit evaluation — publishing it completes the pair. *(Written at commit
  time. `v9` was re-evaluated at N=100 on both checkpoints three minutes later —
  those are the numbers quoted above — but it is still not published: the
  confirmation seed `defend_brique_v10` is training.)*

- **2026-08-09** — **assurance #25: a published claim and an outside
  verification of it, anchored to provably identical weights — and the v15
  contact-reporting line corrected.** Two results, one of them a first for this
  repo and one of them a miss on this side.

  **The provenance first.** `runs/fireteam_defend_v15/ckpt_best.pt` hashes to
  `770aaa59e72a9570ac28ae048c935864e59031bf5a1181a2be68f75d1539b621`, which is
  exactly the `checkpoint_sha256` the assurance layer's `fireteam_defend_v15_best`
  corpus carries in its tap header. **Every claim published here about that
  checkpoint and every independent measurement of it are known to be about the
  same weights**, rather than argued to be — the first time on this project. The
  best previously available was behavioural agreement plus an argument that the
  weights were probably the same, and that argument was explicitly refused as an
  upgrade for `squad_v6`, whose tap predated their digest and whose RNG stream
  had since shifted. On these weights the two pipelines agree: `ckpt_best`
  success 83/100 here against 25/30 there (Fisher p = 1.00), final policy 84/100
  against 24/30 (p = 0.59), and `closed_on_root_report_rate` 0.99 here against
  24/24 there. Nothing to change — recorded because provenance of this strength
  is new, and it is the standard every later publication should be held to.

  **The correction, and the entry above is amended in place.** `2052856` logged
  contact reporting as an open question reading "precision 0.562 → 0.465, recall
  0.786 → 0.699" without stating that both v12 numbers came from
  `behavior_final.json`. Read against `v12`/`ckpt_best` instead, recall
  **improves** (0.628 → 0.699), and `v12`'s own two checkpoints move further on
  both axes (−0.140 precision, +0.158 recall) than the v12→v15 delta being
  claimed. So the sign of half that open question was a function of an unstated
  and unstable reference. All six figures re-read here from the committed
  `runs/fireteam_defend_v12/behavior{,_final}.json` and
  `runs/fireteam_defend_v15/behavior_final.json`; the entry now carries both v12
  rows in a table, the way its success column already did. **Precision degrading
  survives the correction; recall degrading does not.**

  **And a second scenario was offered as killing the mechanism the line implied
  — that half of this entry did not hold** (amended refs #26; the correction is
  the last entry in this log). The v15 entry wrote "the closing SITREP became
  near-universal while contact reporting got worse", which reads as
  crowding-out. This entry answered it with `defend_brique`, where the same rule
  change moves contact reporting the *other* way: `v6` (old rule) → `v9` (ENDEX)
  is precision 0.437 → **0.590**, recall 0.834 → **0.841**, both final policy at
  N=100, seed 123, from those runs' own `behavior_final.json`. Those four figures
  are right and re-verified. Reading "one scenario down, one up" off them was
  not: **the pairing is final/final and `v9`'s own two checkpoints are
  0.439 → 0.590**, so the whole +0.152 is within-run movement and best/best is
  −0.008. A correction about an unstated reference checkpoint reached for a
  counter-example with the same defect, in the treatment arm this time. The
  crowding-out sentence stays gone — it was never supported — but nothing
  replaces it, and the position is now that no between-run mechanism claim is
  available here in either direction. `defend_brique_v9` was cited as measured
  evidence only — its confirmation seed `defend_brique_v10` was still training,
  and it was not in the README table at the time (it published later, in
  `b4a0d6d`).

  Not changed, deliberately: no code, no reward default, no space, no scenario
  semantics; the README v1.13 table never carried the contact figures and is
  untouched; `scripts/program_board.py` does not repeat the claim either — its
  `THREADS` list quotes `report_recall` only for `fireteam_v8`'s within-run
  movement and the ENDEX card quotes only `closed_on_root_report_rate` — so the
  boards were not re-rendered. The issue's §4 caution on `human_death_rate`
  (0.15 → 0.08 is 15/100 vs 8/100, Fisher p = 0.18, and their n=30 reading moves
  the same way at p = 0.51) is accepted as read: the entry already calls it a
  gain on an axis no gate covers rather than a proven effect, so the wording
  stands and the caution is recorded here instead. Tests 569 pass, ruff clean.

- **2026-08-09** — **ENDEX costs `defend_brique` real success, and the
  comfortable explanation is wrong.** Published as a priced regression, both
  seeds, N=100 seed 123 final policy: `defend_brique_v9` (seed 12) **0.91 ±
  0.06**, `defend_brique_v10` (seed 13) **0.88 ± 0.06**, pooled **179/200 =
  0.895** (Wilson 0.845–0.930) against `defend_brique_v6`'s **0.97 ± 0.03**
  under the old close rule. Fisher **p = 0.024**, intervals non-overlapping,
  and below the `prev − 5` bound of 92. Both seeds pass 4/4 behavior gates and
  clear the stability bar (give-back 6.72 and 8.13 against 10).

  **v10 was run to settle whether v9's 0.91 was the seed.** It was not: the two
  ENDEX seeds agree with each other (p = 0.65) and the pair sits exactly on
  `defend_brique_v7`, the flat-terminal old-rule arm, at **p = 1.00**. So the
  survivor-scaled terminal's measured gain on this scenario (v7 0.89 → v6 0.97,
  the v1.12 option-4 result, p = 0.049) **does not survive the close rule**.
  Option 4 stands on `fireteam_defend`, where v15 held it; its `defend_brique`
  evidence no longer stands as measured. No reward default changed — that call
  is the owner's.

  **The hypothesis this entry was expected to confirm is refuted.** The obvious
  reading was that the old rule let the root bank a win early — 442 MISSION
  COMPLETE claims per 100 episodes in v6 against 1 and 5 in v9/v10 — and that
  ENDEX merely stopped the banking. Two facts kill it. First, from the episode
  records: every failure in every arm is a **timeout at the 420-step cap**,
  never a defeat, and the arms are *indistinguishable* on episodes closing by
  step 150 (v6 89/100, ENDEX 178/200, **p = 1.00**). The whole gap accrues
  after step ~200, where v6 gains 7 successes and the ENDEX pair gains 1.
  Second, and decisively, from `cohort_env.py`: `_success_step` is set once and
  never cleared, and `success` fires on `success_locked and (root_reported or
  grace_window or step >= max_steps or cohort_wiped)`. **`max_steps` is one of
  the disjuncts** — so once the end state is met the episode is recorded a
  success whatever the close rule does. The close rule can change when an
  episode *ends* and who gets `root_done_bonus`; it cannot change whether one
  succeeds.

  **So the timed-out episodes never met the end state at all**, and the finding
  is a genuine capability difference concentrated in the slowest, hardest
  episodes: the ENDEX-trained policy reaches "band neutralised, objective held"
  in 7 fewer of them. The mechanism is credit assignment during training, not
  adjudication at evaluation — `root_done_bonus` moved from a claim the root
  could spam to a SITREP that closes early, and the policy learned from a
  different signal. Not diagnosed further: *why* that signal produces a weaker
  late game. That is the open question, and it is the honest one.

  **Left to the owner, unchanged by this**: v1.13 already flagged that
  `defend_brique`'s success condition — band destroyed, or scattered with
  contact broken, objective held — is genuinely *completable*, so the scenario
  may want a different root mission rather than a different close rule. This
  result is the argument for looking at that now rather than later.

- **2026-08-09** — **assurance #26: the counter-example that retired
  crowding-out was checkpoint-selected too, so the position is now that no
  between-run mechanism claim is available here in either direction.** One entry
  up, refs #25 corrected the v15 contact-reporting line for quoting an unstated
  reference checkpoint — and then retired the crowding-out reading it had
  invited by citing a second scenario moving the other way: `defend_brique_v6`
  (old rule) → `v9` (ENDEX), precision 0.437 → 0.590, final policy, N=100, seed
  123. Those figures are correct; all eight below were re-read here at full
  precision from the committed `behavior.json` / `behavior_final.json` of the
  four runs. Offering them as a clean counter-example was the mistake.

  | `defend_brique` · `report_precision` | `v6` (old rule) | `v9` (ENDEX) |
  |---|---|---|
  | `ckpt_best` | 0.4476 *(N=20, n=143)* | 0.4391 *(N=100, n=681)* |
  | `ckpt_latest` | 0.4373 *(N=100, n=718)* | 0.5896 *(N=100, n=519)* |
  | within-run best → final | **−0.010** | **+0.151** |

  **Which `v9` checkpoint you read decides the sign.** The two pairings that end
  at `v9`/final are large and positive — final/final **+0.152** (the published
  claim), `v6`/best → `v9`/final +0.142. The two that end at `v9`/best are flat
  — `v6`/final → `v9`/best **+0.002**, best/best **−0.008**. The published +0.152
  is almost exactly `v9`'s own +0.151 best→final movement, i.e. the between-run
  "improvement" is one run's internal drift, and the reference arm barely moves
  at all (−0.010). "One scenario down, one up" is a final/final statement and
  cannot carry the weight of retiring a mechanism.

  It does not *restore* crowding-out either — −0.008 is flat, not the matching
  decline that reading predicts. **The honest position, and the one this repo
  now holds: contact precision is not stable enough across checkpoints in this
  family to support a between-run mechanism claim in either direction.** That is
  a stronger statement than either reading and costs nothing to make. The v15
  entry and the #25 entry are both amended in place to say it.

  **What survives untouched.** `fireteam_defend` precision degrades under
  *every* pairing of the two runs' four checkpoints: **−0.097** (final/final),
  **−0.089** (`v12`/final → `v15`/best), **−0.229** (best/best), **−0.237**
  (`v12`/best → `v15`/final). `v15`'s own checkpoints sit 0.4734/0.4653, a
  −0.008 spread, so on that scenario the arm under test is the stable one and
  the reference is not. That is a fact about `fireteam_defend`, it stays stated
  as one, and it needs no checkpoint caveat — what it does not do, on its own,
  is identify a mechanism.

  **Where our reading differs from the issue's.** Its −0.011, +0.153 and −0.009
  are differences of 3dp-rounded figures; at full precision they are **−0.010,
  +0.152 and −0.008**. Sign, magnitude and the whole argument are unaffected —
  recorded because this entry re-derives rather than copies. And its caveat that
  `v6`/best is "N=20 against N=100 elsewhere" understates the situation:
  `fireteam_defend_v12`/best is N=20 as well (47 reports), and that is the row
  the surviving −0.229 rests on. Both N=20 rows are the older runs' exit
  evaluations; both newer runs were re-evaluated at N=100 on both checkpoints.

  **Contact volume is unusable and is written down as such**, so nobody reaches
  for it later thinking it is clean: on `fireteam_defend` it disagrees across
  seeds. The assurance layer's seeds 500–529 read 5.50 → 3.07 reports/episode
  (v12 → v15, final) where our seed 123 reads **4.34 → 4.90** — opposite
  directions. The pipelines do agree on `defend_brique` (ours 7.18 → 5.19 final
  and 7.15 → 6.81 best, both down), but agreement on one scenario does not
  rescue disagreement on the other.

  **The lesson, once, plainly.** Two corrections in a row have turned on an
  unstated reference checkpoint. Any between-run behavioural delta in this repo
  should be quoted at both checkpoints or not quoted at all.

  Not changed, deliberately: no code, no reward default, no space, no scenario
  semantics. The README v1.13 table carries no contact figures and is untouched.
  `scripts/program_board.py` states no version of this claim — re-checked rather
  than assumed: its `THREADS` quote `report_recall` only for `fireteam_v8`'s
  *within-run* movement, the ENDEX card and verdict quote only
  `closed_on_root_report_rate` and success, and no campaign verdict mentions
  contact reporting at all — so the boards were not re-rendered. The
  `defend_brique` priced-regression entry above (`b4a0d6d`) inherited no version
  of it either. Tests 577 pass, ruff clean.

- **2026-08-09** — **An evaluation now records the sha256 of the weights it
  scored** (`checkpoint_sha256` in `behavior.json` / `behavior_final.json`;
  refs #28). The gap the issue names is structural and ours: 90 published runs
  commit `ckpt_best.pt`, **zero** commit `ckpt_latest.pt` (gitignored
  fleet-wide) — while since the publish audit the README's headline column is
  the *final* policy. The anchored checkpoint was never the quoted one, so an
  outsider reproducing a headline was reproducing it against weights they
  cannot obtain, and the weights they can obtain produce the secondary column.

  The issue's framing is the point and is worth repeating: the digest costs
  nothing to write and converts "our numbers agree with yours" into "our
  numbers are of the same object". It has already paid off three times in
  three days on the `ckpt_best` side — the digest the assurance layer quotes
  for `defend_brique_v9`, `9bfbafa6…2673`, is byte-for-byte the one this
  backfill computed from our own tree.

  **Be plain about what this is not.** The tensors themselves remain
  uncommitted for `ckpt_latest`. A digest lets a re-measurement *prove* it
  scored the same object; it does not let anyone *obtain* that object. This is
  the cheap 95%, not the whole fix — committing the final weights (~1.2 MB a
  run) stays the strictly better answer and stays undone.

  Backfilled, without re-running a single evaluation, onto exactly the runs the
  v1.13 README table quotes — the numbers in those files are published and not
  one of them moved (each diff is +1 line, −0). Every digest verified against
  `shasum -a 256` of the file on disk:
  `fireteam_defend_v15` best `770aaa59…b621` / final `64ca988e…1667`;
  `defend_brique_v9` best `9bfbafa6…2673` / final `ad81f9ac…7fb3`;
  `defend_brique_v10` best `683e0dba…acab` / final `9dabb0a5…fbcc`.
  `runs/fireteam_defend_v12/endex_rescore.json` gets the same treatment
  (best `d0ffe49b…b293`, final `464d1221…cfbc`) because it already names two
  checkpoints explicitly and its own note flagged this exact hole — "reproducible
  only where the run still lives on disk" was the gap, stated a fortnight early.

  Hashing is best-effort by construction: it streams the file, runs once per
  evaluation, and any `OSError` drops the field rather than the numbers — a
  vanished checkpoint must never cost an evaluation its results. Tests 588 pass
  (11 new, `tests/test_checkpoint_provenance.py`), ruff clean on the files
  touched. Not changed: no published figure, no README row, no board.

- **2026-08-09** — **the ordered hour is on the header** (refs #30). `config.briefing()`
  now publishes `defend_horizon`: the step a DEFEND/DENY root is ordered to hold to,
  or `None` for an indefinite posture. v1.14 made "occupied at every step from H
  until the ordered hour" the definition of DEFEND success **and** opened the root's
  MISSION COMPLETE bit precisely when `defend_horizon is not None`, so both the
  criterion and the claim-admissibility gate turned on a value no outside observer
  could see — a monitor watching a root transmit MISSION COMPLETE could not classify
  the claim. A criterion only the environment can evaluate is *measured*; the same
  criterion with its deadline on the header is *auditable*. Here that cost one
  dictionary key. Same argument, and the same shape, as `announced_assault_step`
  (#12): a pure function of the `ScenarioSpec`, identical across episodes, valid
  before `reset()`, and read-only with respect to the rollout — which is why it was
  safe to land while `fireteam_defend_v16` / `defend_brique_v11` were training.

  **Deliberately not done:** no hold-until clause in `format_opord`. Changing the
  transmitted text is *not* rollout-neutral, and landing it mid-campaign would have
  broken the one property `campaigns/horizon_v1_14.jobs` was built to guarantee —
  that the close criterion is the only variable. The issue asks for that to be
  decided on purpose rather than by accident; it is the owner's call once the
  campaign clears. `cohort/core/language.py` is untouched. What was fixed is the
  wording of `is_done_admissible`'s comment, which said "the horizon is stated in
  the OPORD" and read as a claim about the transmitted text when it is true only of
  the spec the root holds — comment only, no behaviour and no text change.
  608 tests pass, ruff clean. No published number, README row or board touched.

- **2026-08-09** — **v1.14: DEFEND success is conservation of the position, to a
  stated hour** (owner's decision; `eccf816`, `6babcd3`, `0e1a452`). A defense
  succeeds by still being on the ground when the ordered hour comes, not by
  killing everyone. `ScenarioSpec.defend_horizon` states the hour —
  `int(0.5 * max_steps)`, so 225 on `fireteam_defend` and 210 on
  `defend_brique`. From H:

  ```
  occupied(t)  a living friendly within root_obj.radius + 1 of the objective
  FAIL         permanently, at the first t >= H with occupied(t) false
  SUCCESS      at the first t >= H with the threat out of the fight
               (_band_neutralized — early release) or t >= the horizon
  ```

  Three clauses are decisions rather than derivations. **Occupation, not
  safety**: the criterion is the `manned` half of `_objective_held` and never
  the `clear` half, because an enemy assaulting into contact on the position is
  the mission arriving, not the mission failing — scoring the strict
  conjunction costs 29 of 100 episodes on the committed checkpoints, and 26 of
  the 40 first breaks it counts are exactly "the assault got here". **No
  retake**: `_success_step` has always latched success, so conservation has to
  latch the other way (`_defend_lost_step`), checked before the success test in
  the same step. **A fixed step count, not H + D**: `PolicyNet` is a memoryless
  MLP whose only clock is the `step / max_steps` tempo feature, so an
  H-relative deadline would be unperceivable, not merely hard. Casualties stay
  priced by `defend_survivor_scale` and are never gated — a gate rebuilds the
  forfeiture asymmetry that caused D4. Termination is deliberately unchanged: a
  failed defense runs the clock out, so the retrain has one variable, not two.

  `COMPLETABLE` is **refined, not reversed**. v1.13's finding was that a posture
  with *no stated end* has no end state its holder may declare. A defense
  ordered to a horizon does have one, so `missions.is_completable(mission,
  defend_horizon=…)` reopens MISSION COMPLETE to the root — and only the root:
  the hour is in the OPORD, and a subordinate tasked DEFEND by its leader still
  holds an indefinite posture. On a horizon scenario the closure route swaps
  accordingly, from SITREP + ENDEX to an adjudicated MISSION COMPLETE.

  **Acceptance (Phase 2) reproduced an offline reference exactly**, at N=100
  seed 123 on `ckpt_latest`: `defend_brique_v9` 91 → **88**, `v10` 88 → **99**,
  `v6` 97 → **99**. The control matters more than the match. The change has two
  halves — the criterion, which is read-only with respect to the rollout (it
  moves when an episode *ends*, never which action is sampled), and the mask
  bit, which renormalises the policy's masked softmax and so perturbs the
  trajectory. Both arms were measured: with the mask pinned at pre-v1.14 the
  rollouts are bit-identical to the reference replay and give the table above;
  shipped, they give 88 / 99 / 98 — exactly one episode in 300 flips
  (`defend_brique_v6`, seed 142). The perturbation is measured, not assumed.

  **The retrain says the two scenarios are different stories.** N=100, both
  checkpoints, every regression gate PASS on all four:

  | scenario | policy | old criterion | new criterion |
  |---|---|---|---|
  | fireteam_defend | v15 best / final | 0.83 ± .07 / 0.84 ± .07 | 0.97 ± .03 / **1.00 ± .00** |
  | fireteam_defend | **v16** best / final | — | 0.94 ± .05 / **0.99 ± .02** |
  | defend_brique | v9 best / final | 0.90 ± .06 / 0.91 ± .06 | 0.99 ± .02 / **0.88 ± .06** |
  | defend_brique | **v11** best / final | — | 0.98 ± .03 / **1.00 ± .00** |

  On **`fireteam_defend` the criterion changed only the scoring**: read down the
  new-criterion column and the retrain bought nothing — v16 final 0.99 ± .02
  against v15 final 1.00 ± .00, indistinguishable and if anything slightly
  worse. v15 already never left the ground (0/100 occupation failures), so
  there was nothing for the new clause to teach it.

  On **`defend_brique` the criterion changed what is learned**, and the two
  claims separate cleanly: v9's final policy scores 0.88 ± .06 under the new
  criterion — *lower* than its committed 0.91, because it loses the position in
  12 of 100 episodes at a median of **H+7**, stepping off the ground within
  seven steps of the band arriving. v11 loses it in **0** of 100 and scores
  1.00 ± .00, CI-separated from v9. The behavioural signature agrees: cover
  under threat 0.783 → 0.981, distance from the objective under threat 3.17 →
  2.23 cells, human death 0.05 → 0.03, mean episode 141 → 99 steps. Both runs
  converged (best-final gap 1 pt and 0 pts) and both are `[PUBLISHABLE]` by the
  run-report gate.

  Worth stating because the owner pre-empted it: **the horizon is not the
  route.** Early release carries 78–87 of every 100 wins; the ordered hour
  carries 13–22. It is a backstop that fires when the fight is still on at
  half-time, exactly as designed — but the defend family is now at
  0.99–1.00 and has no gradient left in it.

  **The miss, undiagnosed at the level of cause, and NOT fixed.** Reopening
  MISSION COMPLETE retired the ENDEX route (`endex_sent` 0, so
  `closed_on_root_report_rate` is None by construction — its denominator is the
  ENDEX count). Measuring the replacement directly, as wins closed on the
  root's own report: `defend_brique_v11` closes **94/100**, but files **321**
  root claims to do it and has **227** rejected — a root false-complete rate of
  **0.71**. `fireteam_defend_v16` closes **0/99**: it files **zero** claims
  against 13,787 admissible root agent-steps, an open channel declined
  outright. Under v1.13's SITREP route both scenarios closed at 0.99. So the
  same prices produced spam on one scenario and silence on the other, and never
  the single truthful report the channel is for.

  The spam side is arithmetic, not mystery: a true claim pays `done_true` 1.0 +
  `root_done_bonus` 3.0, a false one costs `done_false` −0.5, and
  `done_cooldown` 8 rate-limits retries without changing their sign. v11's
  ledger is 94 × (+4.0) − 227 × (0.5) = **+262 per 100 episodes** — rolling the
  dice is strongly positive-EV, which is the same shape as the pre-v1.10
  re-roll exploit the cooldown was built against. The silence side has **no
  measured cause**: v16's false-DONE rate falls 0.596 → 0.050 across training
  while v11's *rises* 0.528 → 0.655, two opposite local optima on identical
  prices, and nothing measured here distinguishes them. Stated as unexplained
  rather than explained.

  Not repaired, deliberately: the lever is `root_done_bonus` / `done_false` /
  `done_cooldown`, and reward structure is the owner's call. Three options, in
  the order I would try them — (i) make the bonus conditional on the claim
  being the *first* of the episode, which kills the EV of the retry without
  touching the honest path; (ii) scale `done_false` with the number of prior
  rejected claims in the episode; (iii) leave the price and accept that the
  DONE channel reports at 0.71 precision on this family. One retrain and one
  diagnosed adjustment is the standing budget; the retrain is spent and the
  adjustment is a design decision, so it stops here.

- **2026-08-10** — **v1.15: the spam is dead, and the channel died with it.**
  `root_done_bonus` is now paid only on the episode's **first** root claim, and
  the first claim spends it **whether or not it is accepted** (`RewardConfig
  .root_done_bonus_first_claim_only`, default true, settable from `--reward`).
  This is option (i) from the v1.14 entry above, taken by the owner. The A/B is
  `defend_brique_v12` vs `defend_brique_v11` and `fireteam_defend_v17` vs
  `fireteam_defend_v16` — same scenarios, budgets, seed 12 and
  `defend_survivor_scale=0.35`, one flag apart, confirmed **single-variable** by
  `run_report.py --vs` (`economics: single-variable A/B`, the one key being the
  new flag). N=100, seed 123, both checkpoints, all 16 regression gates PASS.

  **The arithmetic checked out before the build and again after.** A true claim
  pays 4.0 (`done_true` 1.0 + `root_done_bonus` 3.0), a false one −0.5, so
  probing breaks even at p > 0.111 and v11's realised acceptance of 94/321 =
  0.293 earned +262.5 per 100 episodes. With the slot spent a further claim is
  worth 1.0 / −0.5: break-even moves to p > 0.333 and probing at 0.293 turns
  negative, −0.0707 a claim with `transmission_cost` included.

  **`defend_brique`: the exploit is gone, completely.** `ckpt_latest`, N=100:

  | | v11 (control) | v12 (v1.15) |
  |---|---|---|
  | success | 1.00 ± .00 | 1.00 ± .00 |
  | occupation failures | 0/100 | 0/100 |
  | root claims filed | **321** | **0** |
  | ...rejected | 227 | 0 |
  | root false-complete rate | **0.71** | — (no claims) |
  | wins closed on the root's report | **94/100** | **0/100** |
  | human death rate | 0.03 | 0.06 |
  | cover under threat | 0.981 | 0.923 |
  | dist from OBJ under threat | 2.23 | 2.19 |
  | stability (best−final) | 0 pts, converged | 1 pt, converged |

  **This is not a pass.** The brief's own bar was that a policy which simply
  stops claiming is a failure, because it trades spam for silence and loses the
  channel v1.14 reopened — and that is exactly what happened. Re-measured at an
  independent seed (N=200, seed 7) to rule out a seed artifact: v11 files 664
  claims in 190 of 200 episodes, v12 files **0** across **21,832** admissible
  root agent-steps.

  **Diagnosed, and measured rather than inferred.** Rolling 40 episodes and
  reading the root's action distribution at every step where its MISSION
  COMPLETE is admissible *and would be adjudicated true*:

  | | v11 | v12 |
  |---|---|---|
  | P(DONE) when a TRUE claim is available | **0.401** (max 0.999) | **0.000083** (max 0.008) |
  | P(DONE) at any admissible step | 0.039 | 0.000028 |
  | episodes declining an available true claim | 2/40 | **40/40** |

  Four orders of magnitude. The channel is not unexercised, it is **dead in the
  policy**. v12 now rides out the full 12-step grace window every episode (480
  true-claim-available steps in 40 episodes = exactly `grace_window` per
  episode, against v11's 2.4).

  **Why, as far as the measurements support.** The rule did what it was designed
  to do — it made repeat probing negative-EV — but v11's *behaviour* is what the
  gradient started from, and that behaviour scores 0.99 × 1.0 − 2.39 × 0.5 =
  **−0.21 per episode** under the new rule. So the gradient points down from the
  incumbent policy, all the way to zero, while the genuinely optimal policy
  (one well-timed claim, +4.0, since a true claim is available for 12
  consecutive steps once success holds) sits on the far side of a region the
  policy must explore *through* — and each exploratory claim now costs the
  episode's bonus as well as `done_false`. Probing was the exploration
  mechanism; removing its profit removed the route to the honest act. This
  paragraph is the one inference in this entry; everything above it is measured.

  It is the `done_false=-2.0` lesson (see `rewards.py`) in a second instrument:
  precision that is really muteness. The observation is not the blocker here —
  `episode_progress` carries the clock and the horizon sits at 0.5 of
  `max_steps` — which is what makes this different from the RECON/SCREEN case
  where p was structurally unobservable.

  **`fireteam_defend`: a no-op, provably.** v17's weights are **bit-identical**
  to v16's at both checkpoints (`max|Δ| = 0.000e+00` over every tensor; only the
  embedded `reward_config` dict differs, which is why the file sha256 differs).
  v16 files zero root claims, so no reward value ever changed, so the gradient
  stream was byte-identical and seed 12 reproduced the run exactly. The control
  arm is as clean as a control arm can be, and it is also an unplanned
  end-to-end determinism result for the trainer. `defend_brique_v12`'s
  `ckpt_best` is likewise bit-identical to v11's — the runs had not yet diverged
  when best was saved, so the only real contrast in this cycle is
  `defend_brique` at `ckpt_latest`.

  **The adjustment was deliberately NOT spent.** Honest-DoD allows one retrain
  plus one diagnosed adjustment. The retrain is spent; the diagnosis points at
  the *balance* of `root_done_bonus` against `done_true` (the bonus is 75% of a
  claim's value, and the first-claim rule makes all of it ride on a single
  decision), and reward structure is the owner's call, not an agent's. Guessing
  a new balance is the autopilot the operating guide forbids. Options, in the
  order I would try them: (i) shift value out of the one-shot bonus into
  `done_true` so the honest claim keeps paying when the bonus is gone — e.g.
  `root_done_bonus` 3.0 → 1.5 with `done_true` 1.0 → 2.0, which leaves an
  accepted first claim at 3.5 and a later accepted one at 2.0 instead of 1.0;
  (ii) keep the rule and restore exploration another way (entropy floor on the
  DONE action, or an initial grace period of N updates before the rule engages);
  (iii) accept silence on this family and revert to v1.13's SITREP/ENDEX route,
  which closed at 0.99 on both scenarios and was never farmable.

  **Stale numbers, flagged not fixed.** `root_done_bonus` pays every completable
  root, so this flag is fleet-wide by design — `fireteam`, `squad`,
  `squad_recon`, `squad_screen`, `platoon` and the rest were all measured under
  the OLD rule and their published claim/false-complete numbers are now stale.
  They were **not** retrained; that is a separate owner decision. Note also that
  `evaluate` rebuilds an older checkpoint's config with
  `RewardConfig(**ckpt["reward_config"])`, and a pre-v1.15 dict has no key for
  this flag, so re-scoring an old policy today applies the NEW rule at its
  default — harmless for the behaviour suite (claims filed and rejected are
  policy, not price) and wrong for any reward comparison against that run's own
  training curve.

  Commits: mechanism + 10 tests (`727ef60`), campaign (`50f5505`), this entry.
  618 tests pass, ruff clean. No README row and no artifact — publication is the
  owner's `/publish`.

- **2026-08-10** — **v1.16: ENDEX restored, and the arms came back bit-identical.**
  Two owner decisions, both after measurement. (1) `RewardConfig
  .root_done_bonus_first_claim_only` defaults to **False** again — the mechanism
  and its tests stay, but it is no longer what the fleet trains under. (2)
  **ENDEX is decoupled from completability**: COMMAND transmits it whenever it
  closes a continuous-posture operation, whether or not the root also claimed.
  Arms `defend_brique_v13` / `fireteam_defend_v18`; controls `v11` / `v16`,
  which share the reverted economics and differ only in ENDEX.

  **Why the first-claim rule was mispriced.** A confirmed root claim ENDS the
  episode, so at most one claim per episode is ever confirmed and it is
  necessarily the last. A probe therefore does not roll cheap dice; it spends
  the bonus on a claim that cannot collect it. The real price of the first claim
  is `done_false − root_done_bonus × P(the episode later closes by root claim)`,
  and on `defend_brique_v11` at N=100 that P was **1.000** across all 63 probed
  episodes: **−3.50**, not the −0.5 the design assumed — a 7× tariff on opening
  the channel at all, charged before the claim's truth is used. Rebalancing
  `done_true` against the bonus was considered and rejected on arithmetic: at
  `done_true` 2.0 the later-claim break-even falls to p = 0.20 against measured
  later-claim acceptance of 0.279, so probing returns at +0.198/claim. One knob,
  both failure modes, opposite directions.

  **Why no reward lever could have restored ENDEX.** ENDEX is a **protocol
  act** — COMMAND emits it, not optional, not learned, not trainable away. A
  root claim is an **agent behaviour** — optional, priced, learnable in either
  direction, as identical prices producing 0.71-false spam on one scenario and
  total silence on the other already proved. v1.14 changed the channel's
  *type*, from a guarantee of the protocol to a property of whatever the policy
  learned. No price restores a guarantee. The gate was
  `not is_completable(root_mission, defend_horizon=…)`, the same predicate that
  chooses the closing *route*; giving DEFEND a horizon switched the
  announcement off as a side effect. The two are now separate predicates
  (`root_may_declare_the_end`, `command_closes_the_operation`), and a SEIZE root
  still closes its own operation with no ENDEX.

  **Result 1 — `successes_announced`: 391 of 391, complete.** New metric
  (`close_announced` per episode): of the operations that SUCCEEDED, how many
  said so on the net at all — COMMAND's ENDEX or the root's own confirmed claim.
  It exists because no published number could see this failure:
  `closed_on_root_report_rate` has ENDEXes-sent for a denominator, so an
  operation that closes in silence never enters it. All figures N=100, seed 123,
  both checkpoints; the era rows are reconstructed from each run's committed
  `per_episode` block, so every cell is the same N and the same seed.

  | era | arms | successes announced |
  |---|---|---|
  | v1.13 (ENDEX) | `defend_brique_v9`, `fireteam_defend_v15` | **348/348** = 1.000 |
  | v1.14 (horizon) | `defend_brique_v11`, `fireteam_defend_v16` | 94/391 = 0.240 |
  | v1.15 (first claim) | `defend_brique_v12`, `fireteam_defend_v17` | **0/391** = 0.000 |
  | **v1.16 (this)** | `defend_brique_v13`, `fireteam_defend_v18` | **391/391** = 1.000 |

  Three of the four v1.14 checkpoints announced **zero** successes; all four
  v1.15 ones did. Per arm, v1.16: `defend_brique_v13` 98/98 and 100/100,
  `fireteam_defend_v18` 94/94 and 99/99 — every win announced, every one by
  ENDEX.

  **Result 2 — the arms are BIT-IDENTICAL to their controls.** `max|Δ| =
  0.000e+00` over all 15 tensors, at both checkpoints, on both scenarios
  (v13 vs v11, v18 vs v16); only the embedded `reward_config` differs, by the
  one key now stated explicitly. Two things follow, and both are measured
  rather than argued:

  * the reverted economics reproduce the v1.14 runs exactly, as they must —
    v11/v16 predate the flag, so `False` is their own rule;
  * **the ENDEX change is rollout-neutral in fact**, contrary to the campaign
    brief's own caution that "it puts a message on the net, and traffic feeds
    the observation". It does not, because of *when*: ENDEX is emitted inside
    the terminal branch, on the step the episode ends, after the last action was
    chosen. It never enters an observation anyone acts on and never moves a
    reward, so the gradient stream was byte-identical and seed 12 reproduced
    both runs tensor for tensor. This is a fact about this message's timing, not
    a general licence — a message emitted mid-episode would not have this
    property.

  So **ENDEX did not suppress honest claiming**: it could not have, the policy
  is the same policy. That answers the cycle's second question as strongly as it
  can be answered — but not in the direction the question hoped for.

  **Result 3 — honest claiming did not return either, and this is the miss.**
  The revert restores the pre-v1.15 status quo *exactly*, spam included.

  | | `defend_brique_v13` best / final | `fireteam_defend_v18` best / final |
  |---|---|---|
  | success | 0.98 ± .03 / **1.00 ± .00** | 0.94 ± .05 / **0.99 ± .02** |
  | successes announced | 98/98 / 100/100 | 94/94 / 99/99 |
  | root claims filed | 5 / **321** | 1 / **0** |
  | ...accepted / rejected | 0 / 5 · 94 / 227 | 0 / 1 · — |
  | root false-COMPLETE rate | 1.00 / **0.71** | 1.00 / — (no claims) |
  | closed on the root's report | 0.00 / 0.94 | 0.00 / 0.00 |
  | occupation failures | 2/100 / **0/100** | 6/100 / 1/100 |
  | human death rate | 0.06 / 0.03 | 0.05 / 0.10 |
  | cover under threat | 0.763 / 0.981 | 0.992 / 0.989 |
  | dist from OBJ under threat | 2.34 / 2.23 | 2.96 / 2.60 |
  | stability (best−final) | 0 pts, converged | 1 pt, converged |
  | regression gates | 4/4 PASS both | 4/4 PASS both |

  `defend_brique` files 321 claims at 0.71 false — the v1.14 spam, back
  unchanged, because it is the v1.14 policy. `fireteam_defend` files none.
  Re-measured directly on the policy's own distribution, 40 episodes from seed
  500, at every step where the root's MISSION COMPLETE is admissible **and
  would be adjudicated true**:

  | | `defend_brique_v13` final | `fireteam_defend_v18` final |
  |---|---|---|
  | P(DONE) when a TRUE claim is available | 0.302 (max 0.998, n=134) | **0.000001** (max 0.000, n=468) |
  | episodes declining an available true claim | 7/39 | **39/39** |
  | claims taken at a true-claim step | 32, all 32 confirmed | 0 |

  Against v1.15's `defend_brique_v12` (0.000083, 40/40 declining) the channel is
  alive again on `defend_brique` and dead on `fireteam_defend` — which is where
  v1.14 left it. **Both channels are now held simultaneously on
  `defend_brique`**: 94 confirmed root claims AND 100 ENDEXes in 100 episodes,
  the report and the fact, which is the arrangement this cycle argued for. On
  `fireteam_defend` only the protocol act is present — and that is precisely why
  it had to be a protocol act.

  **Measured vs inferred.** Measured: every number above, the bit-identity, the
  era table, the probe distributions. Inferred: nothing load-bearing. The one
  claim that is an argument rather than a measurement is *why* ENDEX being
  terminal-step makes it rollout-neutral — and even that is confirmed
  downstream by the tensor comparison, which is what a rollout change would have
  broken.

  **Instrument bug found and fixed** (`scripts/done_probe.py`). It keyed
  "who was root at this step" by `_step_count` as read *before* `env.step()`,
  while `_say` stamps messages with the *incremented* counter — so every claim
  made on an episode's last step was attributed to nobody. Those are exactly the
  confirmed ones, since a confirmed root claim ends the episode. On
  `defend_brique_v13`/`ckpt_latest`, 40 episodes from seed 500: **55 root claims
  and 0 confirmed** before the fix, **87 and 32** after. Any past reading of this
  script's "…by the root" row undercounts by the accepted claims.

  **Stale numbers, corrected and re-flagged.** The v1.15 entry above flagged
  `fireteam`, `squad`, `squad_recon`, `squad_screen`, `platoon` and the rest as
  stale because the then-new default re-scored them under a price they never
  trained under. **That staleness is undone by this revert** — the default is
  their own era's rule again, and `test_checkpoints_from_either_era_reconstruct
  _as_the_rule_they_trained_under` pins both directions (a v1.15 dict carries the
  key explicitly and keeps its rule). What is newly stale is narrower and
  cosmetic: on horizon-DEFEND runs measured before today, `endex_sent` was 0 and
  `closed_on_root_report_rate` was `None`; both would move if re-scored. No
  behaviour number would, because the change is rollout-neutral — so re-scoring
  buys a metric that did not exist, not a corrected result. Not retrained; not
  published.

  **Honest-DoD.** One retrain, spent, on both arms. The diagnosed adjustment was
  **not** spent: the miss is that `fireteam_defend`'s DONE channel is still dead
  (P = 1e-6, 39/39 declining) and `defend_brique`'s is still 0.71 false, and the
  lever for both is `root_done_bonus` / `done_true` / `done_cooldown` — reward
  structure, the owner's call. What this cycle *does* establish is that the two
  problems are separable: the announcement no longer depends on solving either.

  Commits: revert (`14d8b02`), ENDEX + metric + tests (`8dbb299`), this entry
  and the probe fix. 623 tests pass, ruff clean. Spaces frozen at
  `Discrete(228)`/`Box(220)`, verified across all 12 scenarios; all 186
  checkpoints on disk load. No README row and no artifact — publication is the
  owner's `/publish`.

- **2026-08-10** — **The defend family publishes on v1.16: 391/391 operations
  announced.** First defend rows since v15. `fireteam_defend_v18` **0.99 ± 0.02**
  final / 0.94 ± 0.05 peak; `defend_brique_v13` **1.00 ± 0.00** / 0.98 ± 0.03.
  All N=100 seed 123, 4/4 gates on every cell, stability give-back 0.7 and 0.4
  against a bar of 10 — the most stable arms this repo has published.

  **The headline is the announcement, not the success rate.** `successes_announced`
  is complete on both checkpoints of both scenarios. Across eras at the same N and
  seed: v1.13 **348/348** → v1.14 94/391 → v1.15 **0/391** → v1.16 **391/391**.

  **Nothing here is new capability, and the README says so.** `v18`'s weights are
  bit-identical to `v16`'s and `v13`'s to `v11`'s — `max|Δ| = 0.000e+00` over every
  tensor at both checkpoints. ENDEX is emitted in the terminal branch after the last
  action is chosen, so it enters no observation and moves no reward. **My own
  campaign brief asserted this change was "NOT rollout-neutral" and sequenced two
  cycles around that; it is measured false.** The caution was reasonable before
  measuring and wrong after, and it cost a retrain that reproduced its control
  exactly — which is also what proved it.

  **Cross-era comparison is retired, not hidden.** v1.14 redefined DEFEND success as
  occupation maintained continuously from H-hour, so `v18`/`v13` cannot be set
  against `v15`/`v9`/`v10`; those rows are gone from the table rather than left to
  invite the comparison. `defend_brique`'s priced regression against the old close
  rule keeps its equal-footing grid in `runs/defend_brique_v6/`.

  **Published with the defect on the page.** Claim honesty is unsolved:
  `defend_brique` files 321 root claims at 0.71 false, `fireteam_defend` files none
  against 13,787 admissible root steps. Identical prices bought spam on one scenario
  and silence on the other; pricing it bought total silence at a measured −3.50 on
  the first claim. What v1.16 bought is that this no longer costs the record — the
  claim is a report, the ENDEX is the fact, and only the first is unreliable.

- **2026-08-10** — **What is early close worth? Nothing we can measure — so
  `done_true` must not be repriced as if it were the announcement** (refs #33).
  The assurance layer's reframing is the right one and is adopted: before v1.16
  the root's claim WAS the announcement, so pricing the claim priced that; after
  v1.16 ENDEX announces unconditionally, so the claim's residual value is only
  **early close** plus the root's own assessment. Repricing it as an announcement
  would pay twice for something now free.

  **The measurement, at our N.** `defend_brique_v13`/`ckpt_latest`, N=100, seed
  123, from the committed `per_episode` block — all 100 episodes succeeded and
  all 100 were announced; 94 carry a confirmed root claim, 6 close on ENDEX
  alone.

  | group | n | median length | mean |
  |---|---|---|---|
  | claim + ENDEX | 94 | 82.5 | 99.5 |
  | ENDEX only | 6 | 80.5 | 83.7 |

  **Mann-Whitney two-sided p = 0.9942** (U = 283). Early close "saves" **−2.0**
  steps at the median and **−15.8** at the mean: the episodes carrying a
  confirmed claim are if anything *longer*. Nothing is detectable at N=100.

  **Why that split cannot be read causally, in either direction.** The two groups
  are not randomised — they differ in when T0 was reached, not only in how the
  operation closed. Conditional on T0 the arithmetic is exact and one-sided: an
  ENDEX-only episode ends at `T0 + grace_window`, a confirmed claim ends at the
  claim step, which lies in `[T0, T0 + grace_window]`. So **early close can
  advance the close by at most `grace_window` = 12 steps** — ≤15% of a median
  82-step episode — and never by more, whatever it is priced at. The observed
  −2.0/−15.8 is T0 selection swamping a real but small effect, not evidence
  against it. And in reward terms the ceiling is lower still: the terminal speed
  bonus is keyed on `_success_step`, not on the close step
  (`cohort_env.py`), so closing early buys **no** speed payout — only
  `root_done_bonus` and the per-step costs of the ≤12 steps avoided.

  **Verdict**: under v1.16 the claim has **no measurable operational value**, and
  a structural ceiling far below what the pre-v1.16 price was buying. Its
  residual value is informational — the root's own assessment — and on this
  scenario that assessment is wrong **71%** of the time (321 root claims, 94
  confirmed, 227 rejected = 0.707, re-derived from the same file). **Caveat,
  stated rather than buried**: the ENDEX-only cell is n=6, so power is low. What
  is established is that no early-close benefit is *detectable* at our N, not
  that none exists.

  **No reward default was changed.** #33's §4 is right that chasing the 0.71
  false rate with a price is the wrong move before the type question is settled,
  and every price this fleet has tried moved claim *volume* without moving claim
  *informedness*. The lever is what the root can observe, not what the claim
  costs — and that is the owner's decision, not this entry's.

  **Their two confirmations, on the record.** (1) Bit-identity reproduced
  independently from the weights: `max|Δ| = 0.000e+00` over all 15 tensors, `v13`
  vs `v11` and `v18` vs `v16`, at both checkpoints on both arms, with all four
  published `checkpoint_sha256` matching the bytes on disk. Their replays of
  `v11`/`v16` under v1.16 give event and truth bodies byte-identical to the
  retrains, so training against the restored channel changes nothing at all —
  ENDEX is emitted after the last action is chosen and was never in the
  optimisation problem. (2) Our `done_probe.py` fix confirmed net-only, no env
  and no ground truth: **87 claims / 32 confirmed** at our exact cut
  (`defend_brique_v13`/latest, 40 episodes from seed 500), and their
  `root_rejects` of **55 = 87 − 32** pins the bug's scope from outside to exactly
  the confirmed claims and nothing else.

  **Built: their negative control, as a regression-hazard test.** *The confirmed
  root claim is always the LAST claim of its episode* — 0 violations across their
  86 corpora, and the property our probe's keying violated. Now asserted in three
  places (`tests/test_confirmed_claim_is_last.py`, 22 tests): env-level, that a
  confirmed root claim ends the episode in the same step and nothing root-claims
  after it, on a SEIZE root, a horizon DEFEND, a BRIQUE DEFEND and a RECON;
  data-level, that `done_reports_root − done_rejected_root ∈ {0,1}` for every
  episode of every committed evaluation carrying the split (**0 violations over
  1800 episodes in 18 files**); and inline in `scripts/done_probe.py`, which now
  raises instead of counting an episode it cannot attribute soundly.

  **One honest correction to the suggestion.** The ratio form alone would *not*
  have caught our keying bug: dropping an episode's last step removes the claim
  and its confirmation together, so 55 − 55 = 0 confirmed still satisfies "at
  most one". What catches that class is a **coverage** guard — every adjudicated
  message's step must be attributable to some root — so `_audit_root_claims`
  carries both, and reverting the keying to its pre-fix form now aborts the probe
  on the first episode instead of reporting 55/0. The invariant still earns its
  place: it catches the other half of the class, mis-attribution that invents a
  claim after the close, and any future terminal-branch change that lets an
  episode run on past a confirmed claim.

  Test suite 645 pass, ruff clean. No reward default, README row or artifact
  touched; the issue is left open for the assurance layer's re-measurement.

- **2026-08-10** — **The v1.11 confound is closed: the D4 fix carries it, the
  price did not.** `squad_v9` is `squad_v8` with one flag — `done_false` −0.5 →
  **−2.0**, same seed 3, same 3M budget, the D4 fix present in both — which is
  the single-variable run this file has been asking for since v1.11. N=100,
  seed 123, both checkpoints:

  | | success | human death | timeout | DONE reports | false-DONE |
  |---|---|---|---|---|---|
  | `squad_v8` best / final | 0.97 ± 0.03 / **0.98 ± 0.03** | 0.15 / 0.23 | 0.00 / 0.00 | 41 / 266 | 0.63 / 0.44 |
  | `squad_v9` best / final | 0.94 ± 0.05 / **0.97 ± 0.03** | 0.19 / 0.18 | 0.01 / 0.00 | 1 / **0** | 0.00 / — |

  **The survival cell is printed because it is a null, and the null is the
  result** (refs #34). `human_death_rate` moves nowhere: p = 0.58 best/best,
  p = 0.49 final/final, **p = 1.00 pooled** (38/200 vs 37/200). It is here beside
  `timeout_rate` because each covers the other's blind spot — a policy that never
  fights loses no commanders — and because an earlier `done_false` change was
  once *associated* with root deaths moving 4/30 → 12/30 on `squad_screen` while
  success held. This is the manipulation that reattributes that to the D4
  collapse rather than to the price: with D4 fixed, the price does nothing to
  survival in either direction. A reader should not have to infer that from an
  omitted column.

  **Provenance, so a later reader does not try**: `squad_v8`'s row was measured
  to a scratch path to avoid overwriting its committed N=20 evaluation, so the
  baseline arm of this A/B is **not reproducible from the repository** as it
  stands. `ckpt_latest` is gitignored for both runs, so every `final` figure here
  traces to a file rather than a commit; `ckpt_best` is committed and its
  `checkpoint_sha256` is recorded (refs #28).

  best/best **p = 0.50**, final/final **p = 1.00**. No difference on success at
  either checkpoint. So the collapse being gone across `squad`, `squad_recon`,
  `platoon`, `fireteam_defend` and `patrol_brique` is **established** as the D4
  fix generalising, not an artefact of the `done_false` revert that rode along in
  `d44ee8d`. The v1.11 attribution caveat comes off the list.

  **A second finding, not asked for, that lands on the DONE-channel argument.**
  `done_false` −2.0 collapses the claim channel outright: **266 reports → 0**.
  That is the same silence v1.15's first-claim rule produced on `defend_brique`,
  reached by a different lever, on a different scenario, with a SEIZE root. Three
  independent price experiments now agree — `done_false` −2.0 → silence,
  first-claim-only → silence, `done_false` −0.5 → spam at 0.44–0.71 false. **Every
  price tried moves claim volume without moving claim informedness**, which is the
  assurance layer's ρ = −0.702 result (§12.84) reproduced from this side. Note also
  that `squad_v8` loses its entire DONE channel in `v9` with no cost to success:
  the channel contributes nothing to task performance here either.

  `squad_v8`'s committed evaluations are untouched; its N=100 figures above were
  measured to a scratch path.


- **2026-08-10** — **v1.17: the root's MISSION COMPLETE is masked shut on
  DEFEND/DENY roots, and the ENDEX keeps firing.** Owner's decision, closing the
  DONE-channel trilogy. DEFEND/DENY stop being `COMPLETABLE`; the horizon no
  longer opens the root's DONE bit at any step, true claim or false.
  `HORIZON_COMPLETABLE` and the `defend_horizon` parameter are **deleted** rather
  than left as a knob that does nothing, and the parameter's pass-throughs go
  with them (`is_root_opord_claim`, `is_done_admissible`, `compute_mask`).
  `ScenarioSpec.defend_horizon` is untouched and still adjudicates DEFEND
  success — it is an adjudication clause only now, and `config.briefing()`
  (#30) says so. Commit `09913d0`; 661 tests, ruff clean; spaces frozen
  Discrete(228)/Box(220), all 193 committed checkpoints load.

  **Why (the owner's reasoning, and the record of it).**
  - *The claim buys nothing operationally.* Early close is bounded at
    `grace_window` = 12 steps **by construction**, and the terminal speed bonus
    keys on `_success_step` rather than on the close step, so closing early pays
    **no speed bonus at all**. At N=100 the difference between claim+ENDEX and
    ENDEX-only episodes is **p = 0.9942** (`defend_brique_v13`/final, refs #33
    §3).
  - *Its informational value is negative in practice.* `defend_brique` files
    **321 root claims at 0.71 false**.
  - *Pricing cannot fix it.* Three independent experiments, three scenarios, two
    root types: `done_false` −2.0 → silence (`squad_v9`, 266 → 0);
    first-claim-only → silence (`defend_brique_v12`, 321 → 0); `done_false` −0.5
    → spam at 0.44–0.71 false. Every price moves claim **volume** without moving
    claim **informedness**.
  - *The announcement is now free and guaranteed* — **391/391** under v1.16 — so
    masking the claim costs no observability.

  **What made this possible, and must not be undone.** v1.16 split
  `root_may_declare_the_end` (which act closes the window) from
  `command_closes_the_operation` (who announces it). v1.13 masked the claim shut
  and **lost the announcement with it, because they were one predicate**. The two
  names stay separate in `CohortEnv._step` even now that they evaluate alike, and
  the comment there says why.

  **⚑ FLAGGED FOR THE OWNER — the predicted dead-reward consequence does NOT
  occur, and the reason matters.** The brief for this cycle expected
  `root_done_bonus` to become unreachable on defend roots (the v1.4 dead-reward
  condition), and asked for one of three options to be recommended. **Measured,
  not inferred: it stays reachable, and option (ii) arrives with the revert
  rather than as a separate decision.** `root_may_declare_the_end` is *defined by
  the same* `is_completable`, so reverting it also reactivates v1.13's SITREP
  early-close route — a root SITREP at or after T0 closes the grace window, sets
  `_root_close_callsign`, and collects the bonus. In the re-score below the bonus
  is collected on **58 / 79 / 28 / 52** episodes per 100 across the four cells,
  where under v1.16 it was collected on **0 / 94 / 0 / 0**.
  `tests/test_first_claim_bonus.py::test_a_defense_still_pays_its_endex_close_at_either_horizon`
  now runs on the shipped horizon scenario as well as the indefinite one and
  asserts the root's terminal gap equals `root_done_bonus` exactly.

  So the live options are not the three in the brief. They are: **(a) accept
  option (ii) as implemented** — the claim is masked, the SITREP closes, the
  bonus survives; or **(b)** re-add a horizon-aware special case *to the close
  route only*, which would make the bonus genuinely dead on defend and reinstate
  the knob this cycle removed. **Recommendation: (a).** Three reasons. It is the
  coherent v1.13 object rather than a new hybrid — "the root reports the
  situation and COMMAND transmits ENDEX" was always the replacement loop, and
  `test_command_transmits_endex_and_the_root_sitrep_closes_the_window` has
  guarded it since v1.13 as the thing that keeps the bonus from dying twice.
  It prices the report the root *can* honestly make (a SITREP is adjudicated
  against nothing and cannot be false in the way a claim can), which is exactly
  the "move informedness, not volume" property three price experiments failed to
  buy. And (b) reintroduces a second horizon-conditional predicate — the shape
  that produced the v1.14 side effect in the first place. **Nothing was decided
  here beyond following the brief's stated preference for the plain revert; if
  the owner wants (b), it is a one-line special case and a re-run.**

  **Measured before retraining — this is a bigger rollout perturbation than the
  ~1-in-300 the mask-only class predicts.** Both incumbent checkpoints re-scored
  under the new mask at N=100 seed 123, to scratch paths outside the repo; no
  committed `behavior*.json` overwritten; `checkpoint_sha256` verified equal.

  | cell | success (v1.16 → v1.17) | announced | root claims | admissible-root steps | episodes bit-identical | mean length Δ |
  |---|---|---|---|---|---|---|
  | `defend_brique_v13` best | 0.98 → **0.98** | 98/98 → **98/98** | 5 → **0** | 10943 → **0** | 50/100 | −3.51 |
  | `defend_brique_v13` final | 1.00 → **1.00** | 100/100 → **100/100** | 321 → **0** | 8037 → **0** | 6/100 | +1.70 |
  | `fireteam_defend_v18` best | 0.94 → **0.94** | 94/94 → **94/94** | 1 → **0** | 15301 → **0** | 74/100 | −1.63 |
  | `fireteam_defend_v18` final | 0.99 → **0.99** | 99/99 → **99/99** | 0 → **0** | 13787 → **0** | 51/100 | −3.57 |

  Read it in three parts. **Nothing that matters moved**: success is identical in
  all four cells, **every one of the 400 episodes kept its outcome**, and
  `successes_announced` is **391/391 → 391/391** — the v1.16 bar preserved
  exactly, which is the one thing this change had to not break. **The claim
  channel closed by construction**: `done_admissible_root` goes to 0 in all four
  cells, so the silence is attributable to the mask and not to a policy that
  declined — the distinction `done_ok` exists to make (#13), and the reason
  `false_complete_rate_root` correctly reports `None` rather than 0.00. **But
  trajectories moved a lot**: 26 to 94 episodes in 100 change length, against the
  ~1 in 300 the v1.14 agent measured for a pure mask renormalisation. The
  dominant driver is not the mask — it is the SITREP close reactivating.
  `closed_on_root_report_rate` goes 0.00 → 0.30/0.53/0.59 on the three cells that
  had no route to an early close, so SITREPs the policy was *already
  transmitting* now terminate the episode. On `defend_brique_v13`/final it goes
  the other way, 0.94 → 0.79, because a SITREP close is less reliable than the
  confirmed claim it replaces.

  **What that means for the retrain** (`campaigns/mask_defend_claim_v1_17.jobs`,
  `defend_brique_v14` / `fireteam_defend_v19`, controls `v13` / `v18`, same
  budgets, seed and overrides): the arms differ from their controls in the reward
  *flow* as well as the action set — `root_done_bonus` is newly collectible on
  scenarios where the incumbent collected it 0 times in 100 episodes. That is the
  change, not a confound, but any success delta must be read against it rather
  than attributed to the mask alone.


- **2026-08-10** — **The survival cell is now the instrument's job, not the
  author's** (refs #34, standing half). The one-off was done first: the
  `squad_v9` A/B table above prints `human_death_rate` and `timeout_rate` at
  matched N=100 with the nulls and the provenance limits (`c93b400`). That fixes
  one table and nothing else, so the pair now comes out of
  **`scripts/run_report.py --vs`** — this repo's A/B instrument — instead of
  depending on whoever writes the next entry remembering.

  **Built.** A new `== A/B: <run> vs <baseline> ==` block, printed before the
  raw delta dump, carrying success · root death rate · ran-the-clock-out at
  **both** checkpoints, with **each side's N in the header** and an explicit
  verdict on it: `[matched]`, `[MISMATCHED N — 20 vs 100; the deltas below are
  NOT an effect size]`, or `[N UNKNOWN on one side]`. A metric one arm never
  measured prints an em dash and `[not measured on <run>]` rather than a zero, a
  dropped row or a traceback — an unmeasured axis is not a passed one. Episode
  counts are kept **out** of the delta dump, where `100.000 → 20.000` reads as
  an axis that moved. `tests/test_run_report_comparison.py` (9 tests) drives the
  real `--vs` CLI path: the pair present at both checkpoints, matched N stated,
  mismatched N labelled, unknown N not silently called matched, each of the
  three metrics droppable, and a run with no behavior suite at all survivable.
  670 tests, ruff clean. No gate, reward default or price touched.

  It pays for itself on the first real invocation. `run_report.py squad_v9 --vs
  squad_v8` — the exact comparison this issue is about — now leads with
  **`MISMATCHED N — 20 vs 100`** on both checkpoints, because the `squad_v8`
  comparator committed in the repository is the old N=20 artifact. The A/B a
  reader can rebuild from the repo was 5x mismatched while looking exactly like
  a matched one once both sides were printed to three decimals.

  **Their verification of the single-variable claim is stronger than ours.**
  `squad_v8`/`ckpt_best` tapped against a worktree pinned at `792b16a` and
  against head gives **byte-identical observable *and* truth bodies** — not
  economics-equal, identical — plus branch-by-branch inspection that every
  v1.14–v1.16 addition is gated off on a SEIZE root (`_defend_terminal_scale()`
  returns 1.0, `command_closes_the_operation` false so ENDEX never fires,
  `_horizon_defense()` None, `root_done_bonus_first_claim_only` default False).
  Their stated limit, which we adopt: byte-identity is measured on **inference**,
  so the training-side reward rests on the inspection alone.

  **Their gate observation, reported against their own interest, and what we
  found when we checked it.** `squad_v9`/best sits at **7/30 root deaths at seed
  500**, outside the **1–4/30** band named in #9, and fails that composite gate
  of theirs. Not significantly worse than `v8`'s 4/30 (p = 0.51), and its own
  `latest` is 2/30. Recorded as an observation. The question it raises is
  whether *our* gate set should carry that bound at all — `squad` runs pass 2/2
  gates (`timeout_rate`, `success_rate`) and none of them bounds survival.
  **Checked, and the answer is no.**
  - It was never a bound here. `regression_gates` has never gated
    `human_death_rate`, and `scripts/program_board.py` says so out loud (refs
    #24: "no gate covers commander survival"), printing the number on the page
    instead — which is the same remedy #34 asks for, generalised.
  - As a ceiling it fails half the fleet. Over the **104 committed
    `behavior*.json` cells** carrying `human_death_rate`, **50 sit above 4/30**
    (0.133) — DEFEND 11, SEIZE 9, SCREEN 7, RECON 3, plus 20 cells predating
    `root_mission`. Among them are healthy, published, gate-passing runs:
    `defend_brique_v5`/final 0.48, `fireteam_defend_v11`/final 0.35. In the
    `squad` family specifically, **only `squad_v8`/best (0.05 at N=20) is inside
    the band**; every other squad cell on record is ≥ 0.133, and `squad_v9`/best
    at 0.19 sits near that family's median (range 0.05–0.45).
  - At N=30 it cannot function as a threshold. Clopper-Pearson on 4/30 is
    **[0.038, 0.307]** — up to 9.2/30, an interval that *covers* the 9/30
    outlier the band was drawn to contrast with; 9/30 vs 4/30 alone is p = 0.21.
    #9's finding survives because it pooled its four contrast runs (9/30 vs
    12/120, **p = 0.015**), and against that pooled band their 7/30 reads
    **p = 0.065**. The band is a description of four v1.4-era N=30 runs, not a
    bound.
  - So: **no gate added.** If one is ever wanted the honest form is a
    per-scenario-family band read against that family's own distribution — the
    #24 pattern — not a fleet-wide constant. Printing the cell in every A/B is
    the cheaper half of the same idea and is what shipped.

  **Their refutation of their own model, disclosed unprompted.** Their
  blind-claim margin model predicted `squad_v9`'s silence correctly
  out-of-sample (EV −1.16 / −1.14), and then fails on `squad_v8`/best *outside
  its own confidence band* — blind_p 0.1354 against a 0.1111 break-even, margin
  0.0243 past their pre-registered 0.02, predicting a claim where the root files
  none in 30 episodes. They retire their "makes no confident error" claim.

  **The sharpening that matters most, and it is theirs.** Under `done_false`
  −2.0 the root break-even is **0.333**, and `squad_v8`/latest's root
  demonstrably runs at **0.765** realised precision (34 claims, 26 taken; a
  claim ends the episode, so the root forfeits no compliance pay). A root as
  informed as v8's would therefore be **profitable under v9's price by 2.3×**.
  The silence is *not* the priced-rational response. That upgrades "every price
  moves claim volume without moving claim informedness" into something sharper:
  **the volume moves even where the economics say it should not.** Which is the
  strongest argument yet that pricing was never the lever on this channel — the
  premise v1.17 acted on by masking the claim rather than repricing it again.

  Issue left open for the assurance layer's re-measurement; nothing closed,
  commented or labelled.

- **2026-08-10** — **v1.17 retrained and scored: the claim is gone, the
  announcement is complete, and nothing paid for it.** Campaign
  `campaigns/mask_defend_claim_v1_17.jobs` — `defend_brique_v14` (3.0M, seed 12)
  and `fireteam_defend_v19` (3.5M, seed 12), same overrides as their controls
  `v13` / `v18`, claim open on the controls and masked on the arms. One variable.
  N=100, seed 123, **both checkpoints quoted, per the standing rule**.

  | cell | success (control → arm) | p | announced | root claims | claim-admissible steps |
  |---|---|---|---|---|---|
  | `defend_brique` best | 0.98 ± 0.03 → **0.97 ± 0.03** | 0.65 | 98/98 → **97/97** | 5 → **0** | 10943 → **0** |
  | `defend_brique` final | 1.00 ± 0.00 → **1.00 ± 0.00** | 1.00 | 100/100 → **100/100** | 321 → **0** | 8037 → **0** |
  | `fireteam_defend` best | 0.94 ± 0.05 → **0.96 ± 0.04** | 0.52 | 94/94 → **96/96** | 1 → **0** | 15301 → **0** |
  | `fireteam_defend` final | 0.99 ± 0.02 → **0.98 ± 0.03** | 0.56 | 99/99 → **98/98** | 0 → **0** | 13787 → **0** |

  **The headline: `successes_announced` is 391/391.** Both eras happen to total
  391 successes across the four cells, and both announce every one of them, so
  the bar is met at the bar's own number. The v1.16 split predicate held — the
  ENDEX is a protocol act and masking the claim did not touch it, which is
  precisely what v1.13 could not do.

  **Root claims are 0 by construction, not by preference.**
  `done_admissible_root` is **0 in all four cells** (from 8k–15k admissible
  agent-steps), so the silence is attributable to the mask and not to a policy
  declining an open channel — the `done_ok` distinction (#13) that made v1.4's
  dead channel invisible for a training generation. `false_complete_rate_root`
  correctly reports **None** (no denominator) rather than 0.00.

  **No success cost at any checkpoint**: two-proportion p = 0.65 / 1.00 / 0.52 /
  0.56, all four indistinguishable. All 16 regression gates PASS. Both arms
  `[converged]` / `[PUBLISHABLE]` with a 1-point best–final gap.

  **No reward path silently vanished — measured directly, not inferred.** Reading
  the terminal ledger on 40 episodes per cell, the closing root's terminal
  component exceeds a teammate's by **exactly +3.000 = `root_done_bonus`**, on
  every early close, min = max = mean. Early closes: `v14` **40/40**, `v19`
  **39/40** (`v13` 32/40 and `v18` 23/40 under the same v1.17 rules). Across the
  N=100 evaluations the bonus is collected on **96 / 100 / 96 / 98** episodes
  against **0 / 94 / 0 / 0** on the controls as they were scored under v1.16.
  The arms learned to close with the SITREP essentially always:
  `closed_on_root_report_rate` **0.99 / 1.00 / 1.00 / 1.00**.

  **Occupation failures** (`_defend_lost_step` latched — the direct count, which
  no aggregate metric carries): `defend_brique` **2 → 3** (best), **0 → 0**
  (final); `fireteam_defend` **6 → 4** (best), **1 → 2** (final). Noise at N=100.

  **Honest counts against the arms**, none of them gates, all of them stated:
  - `fireteam_defend_v19` turns some clock-outs into wipes: outcomes go
    `{success 94, timeout 6}` → `{success 96, defeat 3, timeout 1}` at best and
    `{99, timeout 1}` → `{98, defeat 2}` at final. The arm's episodes are **26 and
    14 steps shorter** (the SITREP close firing), so an episode that would have
    idled out now ends one way or the other. Root death 0.05 → 0.11 at best,
    unchanged 0.10 at final.
  - `defend_brique_v14`/best is the weakest cell: root death 0.06 → 0.13, cover
    under threat 0.76 → 0.54, coverage time 0.81 → 0.72, succession unrecovered
    2 → 11. **Its own final checkpoint reverses all of it** (0.03 → 0.01, 0.98 →
    0.996, and 3 → 1), so this reads as best-checkpoint selection noise — the
    best checkpoint is picked on rolling success and can be an early snapshot —
    rather than as a learned regression. Quoted because the standing rule is both
    checkpoints or neither.
  - Obedience latency rises on three cells (1.53 → 2.05, 0.36 → 3.75, 2.80 →
    6.22) and falls on one (10.44 → 8.42). Doctrine containment stays 1.000 and
    `orders_violating` / `orders_underivable` stay 0 everywhere, so this is order
    *mix* moving (`v14`/final issues 3.06 orders/ep against `v13`'s different
    ADVANCE/OBSERVE split), not a compliance regression. Not diagnosed further —
    flagged as the one number a follow-up should explain.
  - Report precision/recall improve on three of four cells and drop slightly on
    `fireteam_defend`/best.

  **Honest-DoD**: one retrain, no adjustment needed — the arms met the bar on the
  first pass, so the diagnosed-adjustment budget is unspent. **Not published**: no
  README row, no artifact, boards left PUBLISH PENDING. What is still open is the
  owner's call flagged in the previous entry — accept the SITREP early-close
  route that arrived with the revert (recommended), or make `root_done_bonus`
  genuinely dead on defend with a horizon-aware special case on the close route.
  The retrain above is the *former*; if the owner picks the latter, `v14`/`v19`
  are the wrong arms and it is a one-line change plus a re-run.

- **2026-08-11** — **#35: the announcement bar held on policies *trained* under
  the mask, and the close route it left behind is bought with volume, not
  timing.** The assurance layer verified v1.17 at its own protocol against our
  controls at the same cut: gates clean on all four new corpora (110,220 replay
  checks / 0 violations, 1,377 orders / 0 doctrine violations, no adapter
  change), and all four published `checkpoint_sha256` matched before tapping —
  **finals included**, so a final figure now traces to a digest even with
  `ckpt_latest` gitignored. Their cells:

  | cell | success | root deaths | root claims | announced | `closed_on_root_report_rate` | root SITREPs/ep |
  |---|---|---|---|---|---|---|
  | `defend_brique_v13`@v1.17 *(control)* | 100/100 | 3/100 | 0 | **100/100** | 0.79 | 6.1 |
  | `defend_brique_v14` / best | 97/100 | 13/100 | 0 | **97/97** | 0.99 | 29.9 |
  | `defend_brique_v14` / final | 100/100 | 1/100 | 0 | **100/100** | 1.00 | 30.3 |
  | `fireteam_defend_v18`@v1.17 *(control)* | 30/30 | 0/30 | 0 | **30/30** | 0.50 | 8.8 |
  | `fireteam_defend_v19` / final | 28/30 | 3/30 | 0 | **28/28** | 1.00 | 32.8 |

  **The bar held in the case that actually tests it — the part worth keeping.**
  `v13`/`v18` were trained with the claim open and only *replayed* into the
  mask, so they never had the opportunity to unlearn announcing. `v14`/`v19`
  are the first policies **trained** with the claim bit dead (3.0M and 3.5M
  steps), and every success is still announced. ENDEX is COMMAND's act, so
  there was no gradient against it to follow: a protocol act cannot be trained
  away, an agent behaviour can. That is the strongest available form of the
  v1.16 argument, and it is now tested rather than assumed.

  **The finding to act on: the reactivated close route is bought with volume,
  not timing.** `closed_on_root_report_rate` goes 0.79 → 1.00 and 0.50 → 1.00,
  which read alone says the root learned to *time* its report to the closing
  moment. It didn't — root SITREPs per episode went **6.1 → 30.3 and 8.8 →
  32.8**, one every ~3.2 steps against a `sitrep_interval` of 25. At that
  density the root is essentially certain to have reported at or after the
  success step, so the close is its by default. The rate saturates, and the
  metric — theirs and ours — could not separate the two readings.

  **So we built the denominator** (`cohort/metrics.py`, `docs/metrics.md`,
  tests): `closes_per_root_sitrep` (closes bought per report emitted — high is
  timed, low is bought), `closed_on_cadence_report_rate` (of the operations
  COMMAND closed, the share closed by a report that was itself at least
  `sitrep_interval` steps after the sender's last one — exactly the report the
  environment pays `sitrep_fresh` for rather than `sitrep_spam`), and
  `root_sitreps_per_episode` with its off-cadence count and share, so density
  reads beside the rate instead of being inferable from it. Freshness is
  recomputed with the environment's own rule and tracked per soldier; the
  recorder now writes `sitrep_interval` and the clock origin into the trace,
  the way it already writes `contact_refresh_age` and `max_steps`. They said
  they would not add it on their side until there was a shape we would use;
  this is the shape. Re-scored at N=100, seed 123, at HEAD (`ckpt_latest`):

  | cell | closed on root's report | closes / root SITREP | closed on a cadence report | root SITREPs/ep (off-cadence) |
  |---|---|---|---|---|
  | `defend_brique_v13` *(control)* | 0.79 | 0.130 | **0.38** | 6.07 (69%) |
  | `defend_brique_v14` | **1.00** | 0.033 | **0.00** | 30.30 (97%) |
  | `fireteam_defend_v18` *(control)* | 0.53 | 0.063 | **0.28** | 8.22 (73%) |
  | `fireteam_defend_v19` | **1.00** | 0.032 | **0.08** | 30.60 (96%) |

  (`v14`/best: 0.99, 0.032, 0.05, 29.86 at 96%. Our control cells reproduce
  their protocol closely — 0.79 vs 0.79 and 6.07 vs 6.1 on `defend_brique`,
  0.53 vs 0.50 and 8.22 vs 8.8 on `fireteam_defend` at their N=30.)

  **The new cells return a sharper verdict than the issue asked for.** The arms
  do not merely fail to improve their timing: they close on a cadence-compliant
  report **less often than their controls did** — 0.38 → 0.00 and 0.28 → 0.08 —
  while the published rate goes to 1.00. One behavioural change, an improvement
  on one number and a regression on the other, and only the pair says which.

  **Their arithmetic, re-derived from `RewardConfig` and our own counts —
  agreement in shape, disagreement in the number.** Their account: ~0.53 of
  `sitrep_spam` buys a +3.0 `root_done_bonus`, **5.7:1**, converged on
  independently by two scenarios at different seeds; explicitly filed as the
  account to beat rather than the established one, since profitability is not
  proof of motive. Three corrections, none of which overturns the finding:
  - **On their own terms it is 5.1:1, not 5.7:1.** They derived off-cadence
    reports as volume minus *cadence slots* (episode length / 25 ≈ 3.9–4.9 per
    episode). The reports actually priced fresh are far fewer — **1.00/ep on
    `v14`, 1.10 on `v19`** — because a report only resets the clock when it
    lands and these cluster. Measured off-cadence: 29.3 and 29.5/ep, so the
    spam bill is 0.586 and 0.590, not 0.53 and 0.56.
  - **It omits airtime.** Every SITREP is also charged `transmission_cost`
    −0.01 into the same `report` component, fresh or spam
    (`CohortEnv._charge_transmission`). At the true −0.03 marginal price the
    channel costs **0.839 and 0.841 per episode**, and the gross ratio is
    **3.6:1**.
  - **The decision-relevant ratio is marginal, not gross, and on one scenario
    it is a loss.** Against its own control `v14` spends **0.788/ep more** on
    reports to move the bonus from 0.79×3.0 to 1.00×3.0, i.e. **+0.630** —
    **0.80:1**. `v19` spends 0.749 more for +1.424 — 1.90:1. So the trade pays
    on `fireteam_defend` and **does not pay on `defend_brique`**, which is the
    scenario whose independent convergence was part of the argument. Both
    figures are floors on the cost: a step spent transmitting is a step not
    earning compliance credit, and closing early shortens the episode (96.5 vs
    100.2 steps, 123.5 vs 134.3) and forgoes the rest of it, while the speed
    bonus keys on `_success_step` and is untouched. Caveat ours: two
    independently trained policies differ in more than one channel, so this is
    a decomposition of two ledger components, not a controlled experiment.
  - **What that means for the price question.** It weakens "this is exactly
    what the reward specifies" — on `defend_brique` the reward as written is
    marginally *against* the extra 24 reports/ep by about −0.16/ep, and the run
    converged there anyway. That is the same shape as the claim channel's three
    price experiments: **the volume moves where the economics say it should
    not.** Which is an argument against a fourth price experiment, not for one.
    Recommendation to the owner: **do not reprice `sitrep_spam`** (the
    assurance layer does not recommend it either); if anything is ever done
    here, the honest lever is structural — what the close route *is* — not its
    tariff. Owner's call; nothing was changed.

  **A qualification of our own framing, which we owe.** We said masking the
  claim "costs no observability". That is true of the **announcement** and
  false of **traffic composition**. v1.17 removed a root channel filing **321
  claims per 100 episodes at 0.71 false** and replaced it with a root channel
  emitting **~30 SITREPs per episode of which ~97% are off cadence** (their
  ~87% by the slot method; ours by the priced one). Both are noise on the C2
  net, and the replacement is roughly **five times the volume** — messages per
  episode 120 → 145 on `defend_brique` and 78 → 221 on `fireteam_defend` — with
  no gate on either side watching it. Removing the claim remains well justified
  (three price experiments, no informedness at any price) and the announcement
  is strictly better, at 391/391. But the sentence needed the qualifier, and
  now the metric exists to state it with.

  **Root deaths, reported without the word we withdrew.** `defend_brique_v14`
  /best is **13/100 against the control's 3/100** (p = 0.0165) while its own
  final is **1/100** (p = 0.62 vs control) — a 13× within-run difference at
  p = 0.0013. Per our own correction on #34 this is **not** a gate failure: it
  is a measured between-checkpoint difference, with **final the headline and
  the better policy on that axis**. `fireteam_defend_v19` moves 0/30 → 3/30 at
  final (p = 0.24, not significant at N=30).

  **Left alone, deliberately**: no reward repriced, no gate added, no README
  row, no artifacts published, `scripts/update_boards.py` not run, no training
  launched, nothing closed or commented on the issue. The new keys are computed
  from the trace, so every future evaluation carries them, but they are
  **absent from every already-committed `behavior*.json`** — the four cells
  above were re-scored to a scratch path rather than over the published
  artifacts, and the fleet was not re-scored.

- **2026-08-11** — **v1.18: the OPORD says the hour it will be judged by, and
  the deferral that guarded it was measured wrong.** Commit `9dd4edf` (refs #30)
  adds the horizon clause and rewords its neighbour:

  ```
  before  TL1, THIS IS HQ: OPORD — DEFEND OBJ ALPHA. EXPECT ASSAULT AT H PLUS 45. OUT.
  after   TL1, THIS IS HQ: OPORD — DEFEND OBJ ALPHA. EXPECT ASSAULT AT STEP 45. HOLD UNTIL STEP 210. OUT.
  ```

  (`fireteam_defend`: STEP 65 / STEP 225. Those two scenarios are the *only*
  ones with either an `assault_h_hour` or a `defend_horizon`, so they are the
  only transcripts that change — checked against `SCENARIOS`, not assumed from
  the two we had in hand.)

  **Both clauses are now absolute step references.** The neighbour had to move
  with the new one: `announced_assault_step` was always the absolute step — the
  band's midpoint — but was spoken `AT H PLUS 65`, which reads as *65 steps
  after H-hour*. It said one thing and meant another, which was survivable
  while it was the only time-bearing clause on the line and stopped being
  survivable the moment a second sat beside it. The moods stay different on
  purpose: `EXPECT` is an estimate drawn from a band, `HOLD UNTIL` is tasking
  and does not borrow the hedge.

  The clause is gated on `missions.HOLDS_GROUND` (DEFEND/DENY) — now one named
  predicate read by both `format_opord` and `CohortEnv._horizon_defense`
  instead of an inline tuple in the env plus a caller's discipline in the
  language. **HQ says exactly what is adjudicated**: a SEIZE root handed a
  horizon gets no clause, because nothing would score it. `parse_opord`
  round-trips both clauses (new key `defend_horizon`, the briefing's key for
  the same number) and still *reads* `AT H PLUS n`, because every
  `runs/*/eval_transcript.txt` committed before today says it that way and a
  monitor pointed at that corpus must not silently lose the announcement.
  Nothing emits it.

  **The handoff said this clause "is not rollout-neutral". That was an
  assertion, and it was wrong.** It is the third neutrality assertion in two
  days (after ENDEX and the ~1-in-300 mask estimate) and the third to be
  measured other than as stated — so this one was measured before it was
  believed, at two levels, with the code isolated from concurrent work by
  running `git archive` of the parent and the child commit into scratch trees
  outside the repo:

  - **Mechanism.** Same seed, same fixed action sequence, before-code vs
    after-code, 6 cells (both defend scenarios × seeds 123/7/41, 120 steps):
    every observation vector, every action mask, every per-agent reward, the
    message count, the drawn H-hour, and the sha of the whole transcript *after*
    the OPORD line are identical. The one difference in the episode is the OPORD
    string itself. Message length feeds nothing — no airtime, no arbitration, no
    RNG.
  - **Outcome.** `defend_brique_v14` and `fireteam_defend_v19` re-scored at
    N=100 seed 123 on **both** checkpoints, to scratch paths, and compared field
    by field against the committed `behavior.json` / `behavior_final.json`: all
    four payloads **identical in every field** — 73 aggregate metrics, the CI
    string, all 4 gates, and all 100 per-episode records per cell. A control run
    at the parent commit is also identical, so "identical" is a property of the
    change and not of a re-scorer that reproduces nothing. Headline, unmoved:
    best 0.97 ± 0.03 / final 1.00 ± 0.00 (`v14`), best 0.96 ± 0.04 / final
    0.98 ± 0.03 (`v19`).

  **So no retrain. The published defend policies stay valid and only the
  transcript gains a clause.** Which is the honest scope of the change: this is
  transcript completeness, not capability. `PolicyNet` is a memoryless MLP whose
  only clock is `step / max_steps`, so it cannot use a spoken deadline; the
  beneficiary is the human commander and any monitor reading the transcript
  alone. The audit-side half of #30 — a monitor holding the *header* — was
  already delivered by the `defend_horizon` briefing key.

  687 tests green (670 + 7 new; a further 10 landed concurrently from the
  metrics side), ruff clean, spaces unchanged at `Discrete(228)/Box(220)`, and
  the fleet-load map is byte-identical before and after: 44 of 98 committed
  `ckpt_best.pt` load into current-era spaces, the other 54 being the same
  pre-existing 131/137/166-dim obs eras as before this commit.

- **2026-08-11 (autocycle)** — **`squad_v7`'s lost artifacts recovered; the
  original failure is not reproducible, and the recovery contradicts the gate
  that flagged it.** `train_status` classifies `squad_v7` FAILED — a v1.11-era
  run that reached 100% of its steps and then lost `behavior_final.json`,
  `eval.gif` and `eval_transcript.txt` to a post-training artifact crash.

  **Mechanism: not identified, and that is the result.** Re-running the failing
  stage on the same checkpoint under current code succeeds cleanly — gif,
  transcript and behavior file all produced. The cause lived in code that has
  since moved, and inventing a story for it would be worse than recording that
  it is gone. Artifacts regenerated instead; both checkpoints now carry
  `checkpoint_sha256` (refs #28), which the originals never had.

  **A near-miss of my own, logged because it was one keystroke from a quiet
  downgrade.** The recovery initially overwrote `behavior.json`, which was
  *present, not lost*, replacing a committed **N=100, 0.92 ± 0.05** evaluation
  with a fresh **N=20, 0.85 ± 0.16**. Fewer episodes, triple the interval, and
  no announcement that a published number had moved — exactly the failure mode
  `publish_audit.py` exists to catch, arriving through the repair rather than
  the publication. Caught by diffing before committing, restored with
  `git checkout --`, and the recovered final-policy eval was then matched to
  N=100 so both checkpoints agree. **Recovery must touch only what is missing.**

  **The finding the recovery exposed.** With `behavior_final.json` in place,
  `squad_v7` audits at peak 0.99, final-decile **0.596**, give-back **39.4
  points** — the worst in the fleet by a wide margin. Its FINAL POLICY scores
  **0.91 ± 0.06** at N=100, against its peak checkpoint's **0.92 ± 0.05**. A
  one-point difference behind a thirty-nine-point gate reading. So for this run
  the give-back statistic — computed from the rolling training curve — does not
  describe the divergence it is meant to stand in for. Compare `fireteam_v8`,
  where a *smaller* gap of 12.0 sat over a genuine final-policy drop to 0.80.
  The gate is not thereby wrong; it is a curve statistic being read as a
  checkpoint statement, and the two came apart here. **Next item: measure
  give-back against measured best-vs-final divergence across every run that now
  has both, and say whether the gate predicts what it is used to predict.**

- **2026-08-11 (autocycle)** — **The publish gate is vindicated, and my own
  previous entry set it up to be refuted.** `scripts/publish_audit.py --validate`
  now asks whether give-back predicts what it is used to predict, over every run
  carrying both checkpoints at N=100.

  **It does.** Give-back vs signed (best − final): **Pearson r = 0.564,
  p = 0.015**, n = 18 distinct policies. Higher give-back ⇒ `ckpt_best`
  overstates the final policy more, which is exactly the gate's claim.

  **The entry above got this wrong by picking the wrong statistic.** It read the
  gate against **|best − final|**, where the correlation is *negative*
  (r = −0.40, p = 0.097) and squad_v7 looks like a refutation: give-back 39.4
  over a 1-point difference. But the gate does not claim the checkpoints
  *differ*, it claims the published one is too **HIGH** — a signed quantity.
  Absolute divergence is dominated by runs near the ceiling, where neither
  checkpoint can move far. Same 18 runs, same gate, opposite verdict, decided
  entirely by whether the sign is kept.

  **Two facts worth having beside it.** `ckpt_best` overstates in only **4 of 18**
  runs; the fleet mean is **−1.5pt**, so the peak checkpoint usually *understates*
  the policy the run ended with. And |best − final| never exceeds **5 points**
  across the fleet, so the practical exposure the FINAL-policy standard removes
  is real but small — the standard is right for being honest, not for being large.

  **A bug in the validator, found because its output made a claim.** It first
  deduplicated by hashing the checkpoint FILE, which silently failed: a
  checkpoint embeds its `reward_config`, so the v1.15 revert and the v1.16 ENDEX
  restoration each produced arms whose tensors match to 0.000e+00 and whose files
  do not. It reported 21 "distinct policies" including three duplicates. Now
  hashes the weights and names each drop. The correlation survives either way
  (r = 0.571 → 0.564), but "distinct" had to be true.

- **2026-08-11 (autocycle)** — **The README printed `—` where a zero was
  sitting.** The v1.17 table gave the non-defend rows a dash in the `announced`
  column, with a note asserting those roots announce by their own MISSION
  COMPLETE. The figure existed all along: `successes_announced` counts ENDEX
  **or** a confirmed root claim, deliberately either/or, and it was in the
  artifacts committed the same hour. I wrote the dash without opening them.

  Measured at N=100, final policy, successes announced on the net:

  | run | announced | | run | announced |
  |---|---|---|---|---|
  | `squad_screen_fallen_v2` | 98/100 | | `squad_v8` | 91/98 |
  | `squad_recon_v7` | 94/98 | | `fireteam_v8` | 49/80 |
  | `squad_screen_fallen_v1` | 96/100 | | **`platoon_v5`** | **0/100** |
  | | | | **`patrol_brique_v5`** | **0/99** |

  **`platoon_v5` and `patrol_brique_v5` succeed on essentially every episode and
  never once say so.** Same shape as `fireteam_defend_v16`'s 0/99 before ENDEX
  was restored — and these are scenarios nobody was worried about, publishing
  100% and 99%.

  **The v1.14–v1.17 argument reproduces across the fleet with no new experiment.**
  Where the announcement is a *protocol act* it is complete by construction
  (defend, 391/391). Where it is an *agent behaviour* it ranges from 98% to
  nothing, uncorrelated with how well the scenario is otherwise solved. The
  table now carries the numbers instead of the dash.

  **No metric change was needed** — `close_announced` was already right. The
  defect was mine, in the publication, and it is the second time this cycle that
  a claim went out ahead of the measurement that was already available.

- **2026-08-11 (autocycle, reopened)** — **`fireteam_v7` recovered, and it
  overturns the previous entry's caveat.** The monitor surfaced `fireteam_v7`
  as FAILED — the canonical artifact-loss case, the one `CLAUDE.md`'s own
  autocycle text names. It was still missing `behavior_final.json`, `eval.gif`
  and `eval_transcript.txt`. Recovered, touching only what was absent
  (`--no-behavior` on the transcript pass so the surviving **N=100, 0.95 ± 0.04**
  `behavior.json` could not be overwritten — the mistake logged three entries
  above, not repeated). `economics.json` is **not** reconstructible: it records
  the prices actually in use at training time and inventing one would be a
  fabricated provenance record, so it stays missing.

  **The recovered number is the largest overstatement in the fleet.**
  `fireteam_v7`: best **0.95 ± 0.04**, final **0.78 ± 0.08**, signed
  **+17pt**, give-back **67.8**. Nothing else on record comes close.

  **Which means the sample that validated the gate was biased by the very defect
  this cycle was fixing.** The entry above concluded "|best − final| never
  exceeds 5 points across the fleet, so what the FINAL-policy standard removes is
  real but small". That was measured over 18 runs — and it excluded `fireteam_v7`
  **precisely because its final-policy evaluation had been lost to the crash**.
  The run with the worst overstatement was invisible to the audit of
  overstatement. With it restored, n = 19 and the relationship sharpens from
  **r = 0.564, p = 0.015** to **r = 0.889, p < 0.001**.

  So the correction runs the other way from the one before it: the gate is not
  merely valid, it is strongly predictive, and the FINAL-policy standard removes
  an exposure that reaches **17 points** on a real published run. Retract
  "real but small".

  **The general hazard, stated once:** a missing artifact is not neutral in an
  audit built from artifacts. It removes the run from the statistic, and the runs
  whose artifacts crash are not a random sample of runs.

- **2026-08-11 (assurance, #37)** — **The cadence metric now says what it is
  measured against.** `closed_on_cadence_report_rate` (#35) counts the closes
  made by a report the cadence would have produced anyway, and "would have"
  means at least `sitrep_interval` steps after the sender's last report. That
  value is a `RewardConfig` field: not in the words, and — until now — not in
  `briefing()` either, so the external monitor was scoring every cadence number
  against an assumed 25 and stamping `sitrep_interval_source: "assumed"` on it.

  **The assumption was load-bearing, not cosmetic.** Their sweep over the
  threshold, on the v1.17 cells: at any assumed interval ≥ 15 the retrains lose
  to their controls on the cadence rate (the inversion #35 predicted), and below
  ~12 the `fireteam_defend_v18` → `v19` pair *reverses* — 0.500 → 0.929 at an
  assumed 5, against 0.300 → 0.036 at the real 25. A finding whose direction
  depends on an unpublished constant is a finding the monitor cannot state.

  `briefing()` now publishes `sitrep_interval` = `spec.sitrep_cadence or
  RewardConfig().sitrep_interval` — the same resolution the environment performs
  at step time and the recorder writes into every trace, in one place so the
  three cannot drift. Pure spec function, rollout-neutral, `defend_horizon`'s
  treatment from #30 exactly.

  **Not in the OPORD, deliberately.** `9dd4edf` put the horizon on the net
  because HQ *orders* an hour; a reward threshold is not something HQ says, and
  an OPORD clause reading out a price would be the mistake that commit avoided
  by gating its own clause on `HOLDS_GROUND`. The overlay is the right home for
  the standard a number is computed against.

  One limit, stated in the docstring: this is the scenario as shipped, so a run
  trained with `--reward sitrep_interval=N` is scored against N. The per-episode
  trace already records the value actually in force, and for a given run that
  one is authoritative.

- **2026-08-11 (assurance, #38)** — **Two zeros, two silences — and a column
  published at one checkpoint that swings 93 points between them.**

  **1. The decomposition.** `ed19418` put `successes_announced` on every row and
  grouped the two zeros: "`platoon_v5` 0/100 and `patrol_brique_v5` 0/99, the
  same shape as `fireteam_defend_v16`'s 0/99". They are not the same shape. Read
  off the committed artifacts at our own cut (N=100, seed 123, `ckpt_latest`):

  | run | successes | announced | root claims | refused | admissible steps |
  |---|---|---|---|---|---|
  | `patrol_brique_v5` | 99 | 0 | **0** | 0 | 7772 |
  | `platoon_v5` | 100 | 0 | **5** | **5** | 10211 |
  | `fireteam_defend_v19` | 98 | 98 | 0 | 0 | **0** (masked) |

  `patrol_brique_v5`'s root is **offered the act 7772 times and declines it**.
  `platoon_v5`'s root **claims in five episodes and is refused in all five**. On
  the radio: a silent policy and a rejected one. The fix they want differs —
  extending COMMAND's close to completable roots (the option logged in `e27863b`)
  changes who announces and leaves five refusals untouched — and a single integer
  cannot express the difference. This is exactly #13's argument about zero DONE
  reports, one level up, so `format_root_claim_shape` now renders the root's
  channel beside the announcement in every behavior table: "root never claimed,
  7772 admissible steps" / "root claimed 5, all refused" / "channel shut".

  **2. We broke our own both-checkpoints rule on the column we had just added.**
  All day we have enforced "quote a between-run delta at both checkpoints or not
  at all" (refs #24–#26), and then published the announcement at the FINAL policy
  only. It is the least stable column in the table. `squad_v8`, both checkpoints
  at one commit: **0/97 at `ckpt_best`, 91/98 at `ckpt_latest`** — success 97 vs
  98 (Fisher p = 1.00), announcement 0.00 vs 0.93 (**p = 8.0e-48**). Not one run:
  `squad_screen_fallen_v2` 1/99 → 98/100, `_v1` 8/98 → 96/100, `squad_recon_v7`
  21/94 → 94/98, `fireteam_v8` 67/82 → **49/80** the other way. The table now
  prints `final · best` on every row.

  **3. The ≤5-point figure must not travel to it.** That bound is measured on
  `success_rate` by `publish_audit.py --validate` (and was already retracted once
  when `fireteam_v7` came back at +17pt). On the announcement axis the same
  policies swing up to **97 points**. `--validate` now prints the announcement
  axis under its own table, so the scope is stated by the tool rather than
  assumed by the reader.

  **Independently verified, and it reproduces exactly**: 0/99 · 0 claims,
  0/100 · 5 claims · 5 refusals, and 0/97 → 91/98, all read from committed
  `behavior*.json` with no re-scoring. Their 91/98, computed net-only on their
  side, matches ours computed from the artifacts — a cross-check on both
  pipelines. One number of theirs we cannot confirm and do not need to: their
  `patrol_brique_v2b` 27/29 is a `ckpt_best` figure on a checkpoint carrying
  input dim 137, which cannot load at head at any N.

- **2026-08-11 (assurance, #36)** — **The squad row reads as a regression and is
  the recovery — the README had no `_family`.** `177ba5b` published
  `squad_v8`'s root-death rate as **0.23, the highest in the fleet, and no gate
  covers it**. Both halves are true. Against its own lineage the same number is
  the bottom of a falling series. From the committed artifacts, N=100, seed 123:

  | run | best | final |
  |---|---|---|
  | `squad_v6` | 0.450 [0.350, 0.553] | — (no final evaluation committed) |
  | `squad_v7` | 0.350 | 0.350 [0.257, 0.452] |
  | **`squad_v8`** | 0.150 [0.086, 0.235] | **0.230** [0.152, 0.325] |
  | `squad_v9` | 0.190 | 0.180 (`done_false` arm, not a published champion) |

  Fisher exact, two-sided: `v8`/final vs `v6`/best **p = 0.0016**, `v8`/best vs
  `v7`/best **p = 0.0017**. Against it: `v8`/final vs `v7`/final **p = 0.086**,
  not significant; `v8` vs `squad_v4` — the only squad trained with
  `human_death` −25 in force — **p = 0.807**, a wash, and permanently closed
  because `v4` carries input dim 137.

  **Where we differ from the filing.** They quote `squad_v6`/final at 48/100;
  we hold no `behavior_final.json` for `squad_v6` at all, so our v6 cell is
  `ckpt_best` at 45/100 and their v8-vs-v6 p = 0.00036 is ours at p = 0.0016.
  Same verdict, different cell — and their "14/14 checkpoint_sha256 verified,
  finals included" cannot cover a v6 final we never wrote. We also decline the
  headline "the lowest rate squad has ever recorded": `squad_v8`/best is 0.15,
  `squad_v9`/final 0.18, and `squad_v5` read 0.23 at best in the `Box(166)`
  space. The direction survives all of it; the superlative does not. Their
  `v8` vs `v7` p = 0.086 and `v8` vs `v4` p = 0.807 reproduce exactly, as does
  the `squad_v4` CP95 [0.123, 0.459].

  **And the v7 → v8 pair is not single-variable.** `v8` is the first squad run
  carrying `d44ee8d` (the fallen share in the win) *and* it moved `done_false`
  −2.0 → −0.5. `run_report --vs` reported that pair as a single-variable A/B
  because it diffs `economics.json`, and the fallen fix is a code change no
  price diff can see. Worth remembering the next time a pair is called clean.

  **Mechanism, not just prose:** `scripts/publish_audit.py --series <metric>
  [--scenario <name>]` prints one metric across every generation of each
  scenario at BOTH checkpoints, from committed artifacts only — the `_family`
  `program_board.py` grew after #24 and the README never had. A missing
  `behavior_final.json` prints as `—`; it never lets a `ckpt_best` number stand
  in for a final one.

  **Their correction, offered unprompted.** Their N=30 protocol reads lower than
  our N=100 on bit-identical weights in 10 of 14 cells (sign test p = 0.0117).
  Decomposed with pre-registered arms: not their detector (100/100 per-episode
  agreement at our protocol), not env drift (byte-identical bodies four versions
  later) — the N=30 protocol itself, since a policy at 0.230 resamples anywhere
  in 3/30–11/30. Two of their figures are withdrawn, including "the fallen fix
  achieves on squad what the price never did", which was never significant even
  at N=30 (p = 0.334). Their standard is now N ≥ 100 for any root-death number
  entering a comparison — ours since #34. Both figures are quoted in the README
  as theirs, and the ROADMAP's citations of their root-death numbers should be
  read at N=100 from here.

  **One of our own sentences goes with them.** The `squad_v9` entry above says
  "in the `squad` family specifically, only `squad_v8`/best (0.05 at N=20) is
  inside the band". At N=100 that cell is **0.150**, outside the 1–4/30 band,
  so **no squad cell on record is inside it** — which strengthens that entry's
  conclusion (the band is a description of four v1.4-era N=30 runs, not a bound;
  no gate added) while removing its one counter-example. Dated entries stand as
  written; the correction is here.

- **2026-08-11** — **v1.19: HQ closes every operation, and the fleet becomes one
  system.** The v1.18 handoff's flagged item and the owner's standing goal met in
  one cycle: the announcement gap is closed by making the announcement a protocol
  act everywhere, and the fleet is retrained once, from one commit, on the shipped
  defaults, so that "the baseline" names a thing rather than a habit.

  **The gap, restated in one table.** Measured at N=100 on the final policy of
  every published champion, successes announced on the net:

      defend (ENDEX, a protocol act)        391/391
      squad / squad_recon / squad_screen    91-98%
      fireteam_v8                            49/80
      platoon_v5  0/100   ·   patrol_brique_v5  0/99

  `platoon` and `patrol_brique` succeed on essentially every episode and never
  once say so. The fix is option (a) of the three the handoff listed:
  `command_closes_the_operation` stops being a predicate on the root mission and
  becomes True. The claim is still the root's REPORT and still pays
  `root_done_bonus`; the ENDEX is HQ's FACT. What used to hide inside
  `successes_announced` moves to `closed_on_root_report_rate`, which has
  ENDEXes-sent for a denominator and therefore did not exist outside the defend
  family before today.

  **Rollout neutrality, measured rather than asserted** — the same claim was made
  in the opposite direction once and measured false, so this one was checked
  before it was relied on. Same checkpoints, same seed (4242), 30 episodes, two
  git worktrees (`91f0438` vs `628fecf`):

      patrol_brique_v5   8 of 8 top-level metrics identical
      platoon_v5         8 of 8 top-level metrics identical

  identical including `mean_return` and `mean_length` to full float precision —
  a behavioural difference could not survive that. Exactly four behaviour-suite
  rows moved, and all four are the ENDEX itself: successes announced 0.00 → 1.00
  on both, `closed on root's report` gaining a denominator (— → 0.00), and
  `messages / ep` 28 → 29 and 113 → 114, i.e. **plus exactly one message**. So
  v1.19 is a scoring-and-transcript change; numbers stay comparable across it.

  **The fleet was never one system, and now says so out loud.** Eight champions
  sat at seven commits (`fireteam_v8` 9933a3a, `squad_v9` 48716cc, `squad_recon_v7`
  4395c12, `squad_screen_fallen_v2` a0649de, `fireteam_defend_v19`/`defend_brique_v14`
  e91b753, `patrol_brique_v5` 0e3cf43, `platoon_v5` 6571b70), and four of them
  only reproduced with `--reward defend_survivor_scale=0.35` — a setting that has
  since become the default, so the override was describing the tree of its day.
  `runs/BASELINE.json` names the members; `scripts/baseline.py` is the gate over
  it (coverage, provenance, purity, evidence, gates, stability, loadability, and
  the announcement guarantee), exit 0 only if all hold.

  **A real defect in the attribution tooling, found by the assurance layer's #36
  and confirmed here.** `run_report --vs` printed "CLEAN — share every reward/spec
  value" for `squad_v7` → `squad_v8` and that was read as "single-variable A/B".
  The pair is 35 commits apart, **17 of them touching `cohort/`**, including
  `d44ee8d` — the fallen-share-the-win fix. A code change never touches
  `economics.json`'s prices, so the confound auditor was blind to half its own
  class. It now reads `git_commit`, lists the intervening `cohort/` commits, and
  prints one verdict over both axes; an unchecked axis reads UNCHECKABLE and never
  as agreement. The test that had encoded the wrong belief
  (`test_squad_v7_to_v8_is_a_single_variable_ab`) now pins the corrected reading.

- **2026-08-11 (assurance, #39)** — **The audit differenced two artifacts written
  five days and 36 `cohort/` commits apart, and "real but small" was retracted on
  the strength of it.** `--validate` builds `signed = best − final` out of
  `behavior.json` and `behavior_final.json`. Nothing ever asked whether those two
  files were written in the same environment. For three runs they were not, and
  git says so without needing to re-measure anything:

  | run | `behavior.json` | `behavior_final.json` | `cohort/` commits between |
  |---|---|---|---|
  | `fireteam_v7` | `703a6ac` (08-06) | `f18462d` (08-11) | **36** (21 in `env`/`core`/`config`) |
  | `fireteam_defend_v10` | `2bada50` (08-06) | `2957ae8` (08-07) | 16 (incl. `d44ee8d`) |
  | `squad_v7` | `351eaca` (08-08) | `b1a6c0e` (08-11) | 14 |

  Among those 36 are `d44ee8d` (the fallen share in the win), `ac1fb19` (DONE
  repriced), `eccf816` (DEFEND success redefined) and `09913d0`/`8dbb299` (who may
  claim, and who announces). Any of them can move a measured score. The three
  mixed-era pairs are the fleet's **first, second and fifth** largest give-backs,
  so the gate's validation rested on precisely the rows it should not have used.

  **What the correction costs and what survives.** Over the 16 pairs measured at
  one commit: **r = 0.749, p = 0.0008** — the gate still predicts overstatement,
  and it is not the r = 0.889 the entry above published. `ckpt_best` overstates in
  2/16 rather than 5/19, mean **−1.9pt**, and the largest same-commit
  overstatement in the whole fleet is **+3pt** (`defend_brique_v10`), not +17pt.

  **So the retraction two entries above is itself retracted.** That entry said
  `fireteam_v7` at +17pt refuted "|best − final| never exceeds 5 points across the
  fleet, so what the FINAL-policy standard removes is real but small". On every
  one of the 16 same-commit pairs |best − final| ≤ 5. The bound was never broken
  by a checkpoint; it was broken by the one number that was not a checkpoint
  comparison. Dated entries stand as written; the correction is here, and the
  README's ≤5-point sentence now carries the same-commit qualifier.

  **Where we differ from the filing.** Their tap re-scored `fireteam_v7`/best at
  head as 0.87 (so +9pt at one commit rather than +17pt). We did not reproduce
  that number — six baseline trainings were saturating the box and a 100-episode
  re-score would have contended with them — and the fix does not rest on it: git
  provenance settles that the pair is mixed-era without re-measuring either arm.
  Everything else in the filing reproduces exactly: `--validate` reads n = 19,
  r = 0.889, p = 3.7e-07 before the change; `fireteam_v7`'s `behavior.json`
  carries no `checkpoint_sha256` (it predates #28) while `behavior_final.json`
  carries `4920ae93…`; `ckpt_best.pt` entered at `351eaca` and has not been
  modified since. Their announcement-axis figures were already conceded at #38.

  **Mechanism.** `publish_audit.evaluation_era()` dates each artifact by the
  commit that committed it — an upper bound on when it was written, which is
  enough — and `era_gap()` counts what moved under `cohort/` between the two.
  `--validate` prints the era per row, names and excludes the mixed-era pairs,
  takes its headline over same-commit pairs, and prints the all-pairs figure
  underneath labelled CONFOUNDED. Commits touching only `scripts/`, `tests/` or
  docs cannot move a number and are not counted; a pair git cannot date reads
  `unknown` and is excluded, because "we could not tell" and "there is no
  difference" are opposite findings. This is `run_report.code_diff`'s rule
  (#36) turned on the audit itself.

  **The durable half is deferred, and here is the patch.** Git provenance dates an
  artifact only from outside, and only while it stays committed and unmoved; the
  artifact should date itself, next to the `checkpoint_sha256` that #28 put there.
  That is a change to the writer, `cohort/training/evaluate.py`, and `cohort/` is
  frozen today — `train.py` imports the tree that exists when a job starts, and
  the baseline retrain campaign has six runs in flight with more queued, so an
  edit now would train the later members against a different environment than the
  earlier ones. The patch, to apply after the campaign lands:

      # cohort/training/evaluate.py, in the behavior-artifact block (~line 227)
      sha = _file_sha256(checkpoint) if checkpoint is not None else None
      if sha is not None:
          payload["checkpoint_sha256"] = sha
    + # When, not just what: a score is only comparable to another score taken
    + # against the same tree (refs #39). train.py already records HEAD per RUN;
    + # an evaluation can be re-run at any later commit, so it needs its own.
    + commit = _git_commit()
    + if commit is not None:
    +     payload["eval_commit"] = commit

  with `_git_commit()` lifted verbatim from `train.py:506` into a module both can
  import (`cohort/training/provenance.py`) and re-exported there, so `train.py`'s
  `economics.json:git_commit` keeps its meaning. `publish_audit.evaluation_era()`
  then prefers `payload["eval_commit"]` and keeps the git fallback for the 22
  artifacts already on disk, none of which will ever carry the field.
  `tests/test_publish_audit_era.py::test_an_evaluation_records_the_tree_it_was_measured_against`
  is written and skipped with that reason; unskip it when the patch lands.

- **2026-08-11 (assurance, #40)** — **One kind, two acts: `taking_command` is
  both "the command passes to me" and "I am filling the slot you just left", and
  only the prose says which.** The filing's premise holds; its proposed fix does
  not, and the instance that actually shipped was in our own board code.

  **What the net emits** (`cohort/env/cohort_env.py:704-710`): for every
  `(successor, replaced)` pair `Roster.succeed` returns, the text is
  `format_taking_command` when `replaced` is dead and `format_assuming_position`
  when it is alive. Same `MessageKind.TAKING_COMMAND`, same two callsigns, same
  sender field. A consumer following the root has to decide on each one whether
  the root pointer moves, and the only thing that tells it is the wording.

  | act | text | pointer moves |
  |---|---|---|
  | root appointment | `ALL STATIONS, THIS IS RFN1: TL1 IS DOWN. I AM ASSUMING COMMAND. OUT.` | yes |
  | backfill | `ALL STATIONS, THIS IS RFN2: ASSUMING RFN1'S POSITION. OUT.` | no |

  **What reproduces, and what does not.** `runs/squad_v8/behavior_final.json`
  (N=100) carries `successes` 98 and `successes_announced` **91** — exactly the
  "91 of 98" and the 91/98 = **0.929** the filing quotes, and its 73/98 = 0.745
  is internally consistent. So the arithmetic is theirs and it checks out. But
  the metric it is attached to is not ours: **`closed_on_root_report_rate` in
  this repo never follows a net-derived root pointer.** `metrics._endex_close`
  reads `trace["root_close_step"]` (set by the environment at
  `cohort_env.py:936`/`1301`) and `metrics._root_sitreps` recomputes
  `roots = {rec["cs"] for rec in step["soldiers"] if rec.get("root")}` **per
  step**, from the trace's own ground-truth `root` flag — so succession moves the
  root for free and no single-pointer rule is ever applied. On `squad_v8` the
  rate is `None` on both artifacts anyway (`endex_sent` 0 — no denominator).
  There is also no `runs/squad_v8_v119/` on disk, and `succession_records_
  ambiguous` is not a metric this repo computes. **The 0.745 is a property of
  their net-only reconstruction, not of any number we publish** — which is the
  correct reading of it, because reconstructing command state from traffic alone
  is a promise this project makes (`docs/transparency.md`, `cohort/probe.py`),
  and a promise that costs an external reader 18 points is a real defect even
  when our own scoreboard is unaffected.

  **The instance that shipped is `scripts/scenario_gallery.py`.** Its `ACTS`
  table classifies each transcript line by regex, and the rule for the casualty
  colour was `IS DOWN|ASSUMING COMMAND`. `ASSUMING RFN1'S POSITION` contains
  neither, so the backfill fell through the whole table to the ORDER default and
  was rendered as an order — on a page whose own standfirst promises to show
  "that a rifleman took over a dead leader's fire team". `ALL STATIONS: RFN2 HIT
  A DEVICE …` fell through the same way. Neither had reached the published HTML
  yet (the current gallery's eight episodes contain no leader death), but the
  page regenerates from whatever the baseline members' transcripts hold, so it
  was one casualty away. Fixed, plus a legend that now names both halves, plus
  `test_every_act_the_net_can_carry_is_colored_by_what_it_is`, which drives every
  `format_*` in `core/language.py` through `_classify` and fails on any it does
  not name — so the next message kind cannot quietly become an order. Against the
  pre-fix table it fails naming exactly `format_assuming_position` and
  `format_trap`.

  **Where we differ from the filing: not a payload.** #40 asks for
  `role: "command" | "position"` or `assumes_command: true|false` **in the
  message payload**. `Message` has no payload field and will not get one:
  structured payloads on the net are forbidden by owner decision, the transcript
  is the single source of truth for what was said, and
  `tests/test_orders_flow.py::test_radio_messages_are_text_only` pins the schema
  so they cannot return. The filing also rules out "a regex on the text" — but on
  a text-only net a matcher is not a workaround, it is the interface. What is
  actually wrong is that the matcher is written *four times*
  (`probe._TAKING_RE`/`_FILLING_RE`, `metrics._succession`'s inline marker,
  `scenario_gallery.ACTS`, and every external monitor) instead of shipping once
  as the formatter's inverse — which is this repo's stated contract for every
  other act on the net (CLAUDE.md: "formatter/parser stay inverses").

  **The durable half is deferred, and here is the patch.** It is a change to
  `cohort/core/language.py`, and `cohort/` is frozen: five baseline trainings
  were live when this was written, `train.py` imports the tree that exists when a
  job starts, and a commit under `cohort/core/` would additionally date every
  best/final pair in the fleet as mixed-era under `publish_audit.era_gap` (#39).
  Purely additive, no behavioural change, nothing on the net moves:

      # cohort/core/language.py, beside format_taking_command (~line 394)
    + @dataclass(frozen=True)
    + class Succession:
    +     """One succession move, as the net reports it (refs #40)."""
    +
    +     successor: str
    +     replaced: str
    +     #: True: the root appointment, command passed to `successor`.
    +     #: False: `successor` backfilled the slot `replaced` vacated moving up.
    +     assumes_command: bool
    +
    + _TAKING_COMMAND_RE = re.compile(
    +     r"THIS IS (?P<successor>[A-Za-z]{2,3}\d+): "
    +     r"(?P<replaced>[A-Za-z]{2,3}\d+) IS DOWN\. I AM ASSUMING COMMAND")
    + _ASSUMING_POSITION_RE = re.compile(
    +     r"THIS IS (?P<successor>[A-Za-z]{2,3}\d+): "
    +     r"ASSUMING (?P<replaced>[A-Za-z]{2,3}\d+)'S POSITION")
    +
    + def parse_succession(text: str) -> Succession | None:
    +     """Inverse of the two succession formatters: which act, and by whom."""
    +     if (m := _TAKING_COMMAND_RE.search(text)) is not None:
    +         return Succession(m["successor"], m["replaced"], assumes_command=True)
    +     if (m := _ASSUMING_POSITION_RE.search(text)) is not None:
    +         return Succession(m["successor"], m["replaced"], assumes_command=False)
    +     return None

  The two regexes are `probe._TAKING_RE`/`_FILLING_RE` lifted verbatim and named;
  landing it means `probe._step` calls `parse_succession` and branches on
  `assumes_command`, `metrics._succession` drops its `f"{cs} IS DOWN. I AM
  ASSUMING COMMAND"` marker for it, and `scenario_gallery.ACTS` can be derived
  from it rather than transcribed.
  `tests/test_language.py::test_a_succession_message_says_which_act_it_performs`
  is written and skipped with that reason; unskip it when the patch lands.

  **One thing that is the owner's call, not ours.** The cleanest answer to #40 is
  a *separate `MessageKind`* — `ASSUMING_POSITION` beside `TAKING_COMMAND` — so
  the act is readable from the kind with no parsing at all, which is what a
  monitor keys on. That is a change to the net's vocabulary: every consumer
  switching on `taking_command` learns a new kind, `cohort/viz/dashboard.html`'s
  `KIND_COLORS`/`KIND_GROUPS` change, and every committed trace and transcript
  reads differently across the boundary. Recommendation: ship `parse_succession`
  after the campaign (additive, closes the four-matchers hole immediately) and
  put the kind split to the owner as a v-cycle vocabulary question.

- **2026-08-11 — v1.19 changed a denominator, and the new fleet is where it
  showed.** Found by reading the first three baseline members' behaviour suites
  rather than their headlines. `closed_on_cadence_report_rate` reads **0.000 on
  every completable root** — `fireteam_v9`, `squad_v10`, `squad_recon_v8` alike —
  and `closes_per_root_sitrep` reads **11.0** on `squad_recon_v8`, off 0.09 root
  SITREPs per episode.

  Neither is a policy result. `cohort/metrics.py::_root_sitreps` says it in its
  own docstring: *"An operation closed by a confirmed MISSION COMPLETE rather
  than by a SITREP counts in that rate's denominator and not in its numerator …
  On the v1.17 defend family the claim route is masked shut, so every close
  there is a SITREP."* That last sentence was the load-bearing assumption:
  `endex_sent > 0` used to imply a defence, and on a defence every close *is* a
  SITREP. v1.19 gives every scenario an ENDEX, so the denominator quietly went
  from "defend operations" to "all operations" while the numerator stayed
  SITREP-only. A completable root closes with its claim, so it can only ever
  score 0.

  Read naively, `squad_recon_v8` reports a policy that times nothing. It has no
  SITREP channel in use at all — which is a different statement, and the honest
  one. Same shape as the `false_complete_rate` denominator confusion on
  `fireteam_defend_v12`, and as v1.14 announcing 0 of 57: a predicate that was
  true of the corpus it was written against, and stopped being true.

  **Deferred — `cohort/` is frozen while the fleet trains.** The patch, to apply
  once the campaign lands:
  1. `cohort_env.py` records the close ROUTE beside `_root_close_step` — `"sitrep"`
     where the SITREP branch sets it, `"claim"` where `_report_done` does — and
     the recorder writes it into the trace as `root_close_route`.
  2. `metrics._root_sitreps` takes SITREP-route closes as the denominator of
     `closed_on_cadence_report_rate` and of `closes_per_root_sitrep`, so both read
     `null` where the operation closed on a claim, exactly as they read `null`
     today where no ENDEX was sent.
  3. `docs/metrics.md`'s close-rate block gains the route distinction; the
     paragraph rewritten this morning already says the block now reads on all
     eight scenarios, and this is the half of that which is not yet true.

  `closed_on_root_report_rate` is unaffected and stays the cross-scenario number
  worth reading — it asks whether the root closed the window at all, by either
  route: `fireteam_v9` 0.90, `squad_v10` 0.84, `squad_recon_v8` 1.00. That axis
  is what the README quotes, and it is correct as it stands.

- **2026-08-11 — B3 replicated on the v1.19 tree, and it REVERSES on outcome.**
  One seed per arm (12), 3M steps each, shipped defaults, same `cohort/` tree;
  the control arm is `squad_v10`, which is also a baseline member, so the trio is
  single-variable by construction (`ScenarioSpec.ablation` is the only field that
  differs — checked by dataclass diff, not assumed). N=100, final policy:

  | | full `squad_v10` | nomask `squad_nomask_v1` | flat `squad_flat_v1` |
  |---|---|---|---|
  | success | 0.92 ± 0.05 | 0.98 ± 0.03 | **1.00 ± 0.00** |
  | defeats / 100 | **7.0** | 1.0 | 0.0 |
  | root death | **0.30** | 0.12 | 0.17 |
  | doctrine-valid | 1.000 | 0.592 | — no orders |
  | DONE reports | 325 (182 rejected) | 280 (115) | 231 (47) |

  full vs flat: success p = 0.007, defeats p = 0.014, root death p = 0.045.
  full vs nomask: success p = 0.101 (not a difference), root death p = 0.003.

  **The 2026-08-06 result was the other way round on exactly these cells** —
  success full 0.92 / nomask 0.91 / flat 0.85, defeats 5.0 / 4.7 / **11.0**, with
  the flat arm wiping 2.2x as often. The outcome-robustness half of the published
  claim does not reproduce on this tree; on this seed it inverts.

  The interpretability half **does**: doctrine-valid 1.000 against 0.592 is the
  same ordering as 100% against the original's 0.395 ± 0.079, and it is the row
  the original's own three seeds agree on seed by seed (full's worst 0.387 above
  nomask's best 0.208), so one seed is entitled to settle it.

  The *completion-reporting* half is dead either way, and not because of the
  ablation: the original's nomask arm claimed 0.3 DONE per 30 episodes and this
  one claims 84. That is a code-era difference, not a hierarchy one. "Completion
  reporting only survives under masks" should not be repeated.

  **Provisional, and the check is already running.** `squad_v10` is the weakest
  squad run on record (0.92 against 0.98 and 0.97) and it is the control arm, so
  the reversal could be one bad draw of the control rather than a property of the
  tree. `squad_v10b` (seed 13) lands ~13:45 and bounds it: if it comes in at 0.98
  the reversal is a control artifact; if it comes in at 0.92 the outcome claim is
  genuinely gone on this build. **No README change until that lands** — the
  ablation section keeps the 2026-08-06 numbers, which are three seeds and remain
  the stronger evidence for what they measured.

- **2026-08-11 — the squad regression is real, is not the seed, and comes with a
  behavioural signature.** `squad_v10b` (seed 13) was launched to test whether
  `squad_v10`'s 0.92 was a bad draw. It came in **lower**: 0.88 ± 0.06.

      squad_v8    0.98 ± 0.03      squad_v10   0.92 ± 0.05     (v1.19 tree)
      squad_v9    0.97 ± 0.03      squad_v10b  0.88 ± 0.06     (v1.19 tree)

  The two v1.19 seeds agree with each other (p = 0.48). Pooled 180/200 against
  the previous era's 195/200: **p = 0.0031**. The squad scenario is genuinely
  ~7.5 points weaker on this tree.

  **What moved with it**, across the whole squad family including the two
  ablation arms trained on the same tree:

      run              success   false-claim rate   messages/ep   root SITREPs/ep
      squad_flat_v1     1.00          0.203            17.2            0.13
      squad_v8          0.98          0.444            77.3            0.00
      squad_nomask_v1   0.98          0.411            44.9            0.73
      squad_v10         0.92          0.560           101.2            1.64
      squad_v10b        0.88          0.805           166.8            5.26

  Success against false-claim rate is r = -0.952 over those five. But the
  first-guess mechanism — claims crowding orders off a single-frequency net —
  is **wrong**, and the table is what refutes it: the weaker runs issue MORE
  orders (17.4 and 17.6 against 13.4) and carry MORE traffic overall. Nothing is
  being starved. What separates the eras is that the whole net got chattier:
  101 and 167 messages per episode against 77 and 83, with root SITREPs going
  from 0.00 to 1.64 and 5.26.

  **Direction of causation is NOT established** and the confound is obvious: a
  policy that is worse at the mission claims falsely more often *because* it has
  not finished, so `false_complete_rate` rises without doing any work. What can
  be said is that on this tree the squad policy converged to a chattier
  equilibrium, and that both the false claims and the lost success ride with it.
  Every transmission is an agent-step not spent moving, firing or taking cover —
  the same shape as the order-spam and stall-farming exploits this repo has
  fixed twice before — but that is a hypothesis with a mechanism, not a result.

  **Not fixed today, and deliberately.** The lever people reach for is
  `done_false` (-0.5 → -2.0, which is exactly what `squad_v9` ran with, and it
  claimed **zero** times and scored 0.97). Three reasons to leave it alone:
  CLAUDE.md's own rule is to diagnose with the oracle before touching rewards;
  the correlation above cannot tell a price problem from a policy problem; and
  changing a reward default now would put the eight members on two different
  trees and destroy the provenance the whole baseline rests on.

  **The discriminating experiment**, for the next cycle: `scripts/done_probe.py`
  on `squad_v10b` in all three regimes — golden steps say whether truthful
  claiming was reachable and declined (a pricing problem) or unreachable (a
  masking one), and the observe regime gives the unperturbed opportunity count.
  Then one arm at `done_false=-2.0` against `squad_v10b` as a named single-
  variable A/B on a frozen tree.

  **What ships meanwhile**: `squad_v10` as the squad member, at 0.92 ± 0.05, with
  this entry as its caveat. It clears every gate — N=100 final policy, all gates
  green, give-back 7.0 under the bar of 10, 92/92 announced — and it is the
  weakest member of the fleet by a clear margin.

- **2026-08-11 (assurance, #42)** — **Command does not devolve to a branch whose
  leader was itself a successor: the silence #42 heard on the net is an org-chart
  hole, not a reporting gap.** The filing's worked example reproduces line for
  line on this tree — including the silence — and its ground truth does not.

  **The defect, in one statement.** `Roster._fill_vacancy`
  (`cohort/core/units.py:251-288`) sets `successor.leader_id = vacated.leader_id`
  and never adds `successor.id` to that leader's `subordinate_ids`. Its own
  *recursive* branch does exactly this (`successor.subordinate_ids.append(
  promoted.id)`, line 287) — the top-level call does not. So the first succession
  into a **mid-chart** slot leaves the new leader pointing at a superior that does
  not list it, and the link is one-way from then on. Minimal repro, pure core, no
  policy involved:

      roster = squad()                      # SL1 → TL1(RFN1,RFN2), TL2(RFN3,RFN4)
      tl2.alive = False; roster.succeed(tl2)
      RFN3.leader_id      == SL1.id         # RFN3 reports to the squad leader
      SL1.subordinate_ids == [1, 4]         # …which still lists TL2's corpse
      SL1.living_subordinates(roster)       # ['TL1'] — RFN3 and RFN4 are gone

  RFN3 and RFN4 are now off the squad leader's chart. `env/actions.py` masks
  orders on `living_subordinates`, so they cannot be tasked; `env/observations.py`
  builds the leader's subordinate slots from the same list, so they are not
  perceived; `metrics.py:321` records the same list as the trace's `subs`, so
  `succession_recovery`'s orphan set omits them and scores a recovery that did not
  happen. And when SL1 falls in turn, `_pick_successor` cannot see that branch, so
  command does not devolve to it.

  **Why the net goes quiet, which is what they measured.** The announcement is
  emitted per succession *event* (`cohort_env.py:704-710`). No event, no message.
  Replaying the filing's own death order on our roster — SL1 69, TL1 78, TL2 82,
  RFN1 95, RFN2 103 — reproduces their transcript exactly:

      step  69  TL1: SL1 IS DOWN. I AM ASSUMING COMMAND   /  RFN1: ASSUMING TL1'S POSITION
      step  78  RFN1: TL1 IS DOWN. I AM ASSUMING COMMAND  /  RFN2: ASSUMING RFN1'S POSITION
      step  82  RFN3: TL2 IS DOWN. I AM ASSUMING COMMAND
      step  95  RFN2: RFN1 IS DOWN. I AM ASSUMING COMMAND
      step 103  (silence)                         root=None   living=['RFN3','RFN4']

  **Where their ground truth is wrong, and it matters.** #42 says "truth promotes
  RFN3 — the top-ranked survivor". This repo's truth is `Roster.root()` = the
  senior living agent with `leader_id is None`, and at step 103 that is **`None`**.
  Two agents alive, nobody in command. So the cost is not that a monitor lags the
  truth; it is that there is no root, and every root-keyed mechanism in the
  environment goes dead with it: `is_root_opord_claim` requires `soldier is
  roster.root()` so the operation's MISSION COMPLETE channel shuts,
  `_root_sitrep_step` (`cohort_env.py:1127`) stops being recorded so the v1.19
  COMMAND close is unreachable, and the ENDEX branch (`cohort_env.py:966`) is
  guarded by `if root is not None` and emits nothing. A rootless episode can only
  end by timeout or the grace window. Their measurement is right, their diagnosis
  is one level too shallow, and their proposed fix would paper over it — announcing
  a root change would leave the branch untaskable and unobserved, and here there is
  no new root to announce.

  **Measured, twice.** Structurally, over *every* order in which a team can die
  (pure `Roster`, no policy, no seeds):

      squad     4080/5040 orderings orphan a branch (81.0%)
                1928/5040 reach a living team with no root (38.3%)
      fireteam     0/24    /    0/24        — exempt: a fireteam successor's new
                                              leader is HQ, which keeps no list

  Realised on the shipped fleet, sampled actions, seeds 500+, 660 episodes over
  the seven readable baseline members (`squad_screen_v11` was still training and
  was not touched):

      fireteam        0/100    fireteam_defend  0/100    defend_brique  0/100
      patrol_brique   1/100    squad            6/100    squad_recon   18/100
      platoon        19/60
      → 44/660 episodes end with a broken chart; 1/660 (squad_recon) loses the
        root entirely, for 11 of 62,090 (step × living-team) checks = 0.018%

  The org kind predicts it exactly: every `fireteam`-org scenario is clean, every
  `squad`/`platoon`-org scenario is not. **We do not reproduce their rates** —
  0.343% of checks and 18.5% of episodes against our 0.018% and 0.15%. Their
  corpora are older, weaker policies that lose leaders far more often (their
  `squad_v2` corpus carries 62 successions in 30 episodes; `squad_v10` carries 38
  in 100). The mechanism is identical; the frequency is a property of the policy
  being watched, not of the environment, and should be quoted as theirs.

  Also measured and **zero**: a root that moves *silently to another living agent*.
  0 in 5,040 orderings structurally, 0 in 62,090 checks on the fleet. In this repo
  the root is either announced or lost — which is why the announced chain never
  names a live non-root, and why "hold a dead commander" is the correct
  description of the failure.

  **The patch, and why it is not applied today.** One statement, in
  `cohort/core/units.py::_fill_vacancy`, immediately after
  `successor.leader_id = vacated.leader_id`:

      + # The superior inherits the successor in the slot it just filled. Without
      + # this the promoted agent is unreachable from above: unorderable (masks
      + # read living_subordinates), unobserved, and — when the superior falls —
      + # not devolved to, which is how an operation ends up with no root (#42).
      + if successor.leader_id is not None:
      +     parent = self.by_id[successor.leader_id]
      +     if successor.id not in parent.subordinate_ids:
      +         parent.subordinate_ids.append(successor.id)

  With it, the filing's own example produces the message they asked for and
  restores the root, using the existing formatter, with no new vocabulary and no
  new emission rule:

      step 103  RFN3: RFN2 IS DOWN. I AM ASSUMING COMMAND  /  RFN4: ASSUMING RFN3'S POSITION
                root=RFN3

  Evidence it is safe: applied in memory as a pytest plugin (nothing on disk under
  `cohort/` changed), the full suite is **794 passed, 4 skipped** — identical to
  head — with the new test unskipped and green, and the structural sweep goes to
  0/5040 on both counts.

  `cohort/` is **frozen**: `squad_screen_v11`, the last baseline member, was 64%
  through training when this was written, and `train.py` imports the tree that
  exists when a job starts — landing this now would train one member of the
  shipping fleet against a different environment than the other seven and destroy
  the baseline's provenance. It is also not a patch but a **v-cycle** change: it
  restores an order edge, so action masks and observations move and every rollout
  changes. `tests/test_succession.py::
  test_a_promoted_leader_is_on_the_chart_of_the_superior_it_reports_to` is written
  and skipped with that reason; unskip it when the patch lands. README's succession
  paragraph now carries the scope.

  **Coverage note, on us.** Six succession tests existed and all six killed the
  **root** — where `vacated.leader_id is None` and there is no superior to
  re-point. Not one killed a mid-chart leader. The defect has been reachable since
  the roster was written and the suite could not see it, which is why the new test
  kills TL2 and then sweeps every death order rather than asserting one more
  transcript.

  **What we would not do.** #42 offers "emit `TAKING_COMMAND` whenever the root
  changes" and, as a cheaper alternative, "have `probe.py` report when the
  announced chain and the rank-derived root disagree". The first treats the symptom
  and buys an external monitor an announcement that the environment itself cannot
  act on; the second would report a disagreement that, measured here, does not
  exist — the announced chain and the truth never disagree about a *living* root.
  Fixing the chart closes both, and pays for itself in the env: two agents that
  were unorderable for the rest of the episode become orderable again.

- **2026-08-11 (assurance, #44)** — **The sealed fleet published eight numbers and
  withheld the eight sets of weights that produce them.** Filed against the seal,
  true as filed, and the mechanism is one character in `.gitignore`.

  **Verified on disk before changing anything**, because the layer has been wrong
  three times today:

      tracked ckpt_latest.pt   runs/<member>/     0 of 8      ← the shipping fleet
                               runs/archive/<n>/  96 of 96    ← superseded, nobody cites
      tracked ckpt_best.pt     runs/             13 of 13
                               runs/archive/     96 of 96

  A gitignore `*` does not cross `/`. `runs/*/ckpt_latest.pt` matched
  `runs/<name>/…` and stopped at `runs/archive/<name>/…`, so `f80097d` filing 96
  runs one level deeper inverted the rule without touching it. #44 spotted one
  consequence; there were **three**, all the same bug: the move also swept in **79
  `tb/` event files** (`runs/*/tb/`) and **44 host-specific `.job.json`**
  (`runs/*/.job.json`), ~99 MB of weights and 66 MB of tensorboard nobody asked
  for. The rules are now `runs/**/…`, which means the same thing at every depth,
  so the next archive move is a no-op here.

  **Why it is the wrong way round, specifically.** The headline is the FINAL
  policy — `behavior_final.json`, scored from `ckpt_latest.pt`. What was committed
  is `ckpt_best.pt`, a best-rolling-*window* snapshot, which this repo's own audit
  says is not the policy the headline describes; best and final have disagreed on
  this fleet by 30/30 success vs 30/30 timeout on one run. So a fresh clone could
  read all eight figures and re-derive none of them.

  **The bytes are the right bytes** — checked before committing them, since a
  committed checkpoint that is not the one scored would be worse than the absence
  reported. All 11 live runs carrying a `behavior_final.json` hash to exactly the
  digest #28 recorded:

      fireteam_v9 675bce50  fireteam_defend_v20 b7221b3a  squad_v10 baa049ad
      squad_recon_v8 56ebf10a  squad_screen_v11 2afb2549  patrol_brique_v6 ba9d2bb0
      defend_brique_v15 33b60d62  platoon_v6 63355bf1  squad_v10b 5c375de3
      squad_nomask_v1 d85b6388  squad_flat_v1 95c29900        — 11/11 MATCH, 0 mismatches

  13 `ckpt_latest.pt` committed (~15 MB): the 8 members plus the 5 runs
  `BASELINE.json` cites in `referenced_history`, on the same argument — an
  ablation arm's number is published too. The two Box(137) `squad_abl_*_s3`
  originals have no `behavior_final.json` and their weights do not load under the
  current spaces; committed for uniformity, claimed for nothing.

  **The 96 archived stay tracked.** Untracking them shrinks a checkout and never
  the pack — the bytes are in history either way — and `runs/` is not a tree to
  do reversible-looking surgery on for tidiness. The rule now sheds
  `ckpt_latest.pt` for *future* archived runs, which is where the cost is.

  **Turned into a gate, because the failure was silent.** Nothing broke when the
  weights went missing: `behavior_final.json` was present and complete, every gate
  was green, and the one artifact needed to reproduce the figure was absent.
  `scripts/baseline.py` gains **committed** to its list of what a baseline *is* —
  both checkpoints in the repository — and it answers `[]` rather than failing
  wherever git cannot say (a tarball, the audit's own `tmp_path` fixtures), since
  a gate that fires for an unrelated reason teaches people to ignore it.

  **A second defect found in the same place.** `_loadable` — the check whose
  docstring says "a baseline whose weights no longer load is a historical
  artifact" — was loading `ckpt_best.pt` **only**, the one checkpoint the
  `evidence` rule four lines above it explicitly says is *not* the headline. It
  now checks both. All 8 members' `ckpt_latest.pt` load at obs=220; the fleet
  passes the stricter gate unchanged.

  Both new gates were watched to fail and recover (un-stage one checkpoint →
  `BASELINE NOT READY — platoon_v6: ckpt_latest.pt is not committed`; re-stage →
  `BASELINE OK`), on the principle that a gate nobody has seen fail is a gate
  nobody knows works. Suite **807 passed, 3 skipped** (was 795/3): +8 parametrized
  per-member reproducibility checks, +1 pinning both directions of the gitignore
  rule, +3 on the gate. `scripts/baseline.py` prints BASELINE OK. **No `cohort/`
  file was touched — the v1.19 seal at `5f848fb6` is intact and no retrain is
  implied.**

  **Where we disagree with the filing.** #44 says "no urgency, nothing blocks us"
  and offers to keep tapping weights from the working tree with a provenance
  caveat. The caveat was the finding. A number whose weights are not in the
  repository is a claim, not a result, and that is exactly the distinction this
  repo keeps trying to hold — so it is fixed and gated rather than noted.

- **2026-08-11 (assurance, #45)** — **The seal gated the headline and published the
  peak ungated: a member re-scored at N=5 passed `baseline.py` with byte-identical
  output.** True as filed on the mechanism, wrong on one detail, and the fix is two
  conditions in `scripts/`.

  **Reproduced on the real tree before changing anything.** Restore the N=5 blob
  into `runs/platoon_v6/behavior.json`, run the gate, restore the repair:

      control   (repaired, N=100)  exit 0, BASELINE OK
      treatment (restored,  N=5)   exit 0, BASELINE OK
      diff(control, treatment)     EMPTY

  **Where the filing is wrong.** It names `9819696` (the seal commit) as where the
  N=5 blob was committed. It was not — `9819696` carries N=100. The corrupt window
  is exactly one commit, `a321329`, repaired the next commit in `bcdbfab`:

      38808fe N=100 · 9819696 N=100 · a321329 N=5 · bcdbfab N=100 · e6b600c N=100

  That the window was one commit rather than three does not weaken the finding —
  it sharpens it. The corruption arrived **after** the seal and the seal did not
  notice, which is the whole point.

  **Why nothing saw it.** `runs/BASELINE.json` was byte-identical throughout:
  `cohort_tree 5f848fb6` and every `checkpoint_sha256` correct, because the
  environment and the weights had not moved. What moved was a number *derived*
  from them, and the derived side was undigested. Confirmed by reading:
  `_run_facts` gates `episodes >= 100` on `behavior_final.json` alone;
  `publish_audit.audit_run` opens `behavior.json` but only requires it to exist,
  taking `gap` from `metrics.csv`; `results_table.py` sources the README's **peak**
  column from it. Published, and gated by nothing.

  **The one catcher was host-dependent.**
  `test_the_readme_table_matches_the_runs_on_disk` skips whenever any member is
  `RUNNING` — live state, not tree state. Had the fleet still been in flight the
  N=5 peak would have sealed with nothing in the tree able to detect it, and the
  skip message points at `baseline.py`, which was blind to this field.

  **Both fixes taken, because they cover different failures.** The evidence rule
  now holds the peak evaluation to N >= 100 like the headline, and `--seal` stamps
  a sha256 of every published evaluation into the manifest (16 digests, 8 members ×
  2 files). The first catches a bad number as it is written; the second catches a
  *re-scored* one, at full N and plausible, which the evidence bar cannot see —
  drift in a sealed member is now detectable from the tree alone, by anyone, with
  no live campaign to compare against.

  **A third defect in the same place, which "when present, require N>=100" would
  have left open.** `audit_run` returns `None` when `behavior.json` is missing and
  `_run_facts` applies the give-back gate only `if a:` — so *deleting* the file did
  not merely skip the new check, it silently switched off the stability gate too
  and printed `—` in the give-back column. Two gates standing down for one absence,
  saying nothing. Absence is now named.

  Watched to fail and recover on the real fleet, on the principle that a gate
  nobody has seen fail is a gate nobody knows works — the same manipulation now
  gives `platoon_v6 … FAIL`, `peak evaluated at N=5, needs 100` and `behavior.json
  changed since the seal (ef13e69daca0 -> fe43ead3ec5e)`, and `BASELINE OK` on the
  restored tree. An unstamped manifest stays silent rather than accusing, and
  `publish_baseline.py` now ends by telling the operator to re-seal, since
  re-scoring is a normal thing to do and is precisely what invalidates the stamp.

  Suite **815 passed, 3 skipped** (was 807/3), ruff clean, `scripts/baseline.py`
  prints BASELINE OK. **No `cohort/` file was touched — the v1.19 seal at
  `5f848fb6` is intact and no retrain is implied.**

  **Where we disagree with the filing.** #45 offers the two fixes as alternatives
  ("either; the second is the durable one"). They are not alternatives: the N gate
  cannot see a silent re-score at full N, and the digest cannot see a bad number
  written before the seal. Its closing table — for each checkpoint exactly one of
  {weights committed, evaluation gated} holds — is now false in the good direction:
  after #44 and #45 both hold for both checkpoints.

- **2026-08-11 — the DONE probe answers the squad regression: PRICING, not
  reachability.** `scripts/done_probe.py runs/squad_v10b/ckpt_latest.pt
  --episodes 40 --seed 700`, observe regime (the unperturbed one):

      golden steps [root]              134        eps with >=1 golden   40/40
      claims transmitted by the root   120        confirmed              33
      root accept rate               0.275        naive-regime accept  0.064

  **Reachability is not the problem.** Every one of 40 episodes offers the root
  at least one golden step — a step where MISSION COMPLETE is admissible by the
  mask *and* would be adjudicated truthful. The channel is open and the policy
  uses it 120 times.

  **The price is.** With the shipped defaults (`root_done_bonus` +3.0,
  `done_false` −0.5) the break-even accept rate is **0.143**, and the root is
  running at **0.275** — nearly double. Every claim is worth **+0.463** in
  expectation, so claiming at low precision is not a failure of the policy, it
  is the policy correctly reading its own incentives.

      done_false = -0.5   break-even 0.143   EV/claim +0.463   (shipped)
      done_false = -2.0   break-even 0.400   EV/claim -0.625   (squad_v9's price)

  At −2.0 the break-even rises above the observed accept rate and the behaviour
  goes −EV. That is consistent with what `squad_v9` actually did: trained at
  −2.0, it transmitted **zero** DONE claims in 100 episodes and scored 0.97.

**⚑ The EV arithmetic in this entry is wrong and is corrected in the
  `2026-08-11 (assurance, #46)` entry at the end of this log: it prices a claim on
  `root_done_bonus` alone and drops `done_true` (+1.0). Break-even is 1/9 = 0.111
  and 1/3 = 0.333, exactly as `rewards.py` states them in its own comments — not
  the 0.143 and 0.400 quoted here. Every DIRECTION in this entry survives the
  correction (0.275 is above 0.111 and below 0.333), which is why the verdict
  stands; the margins were overstated.**

  **What this does NOT establish** is that the claiming *causes* the lost
  success. It establishes that the claiming is rationally priced in, which is a
  mechanism where before there was only r = −0.952. The causal step is the A/B.

  **`squad_v11` launched**: `--scenario squad --total-steps 3000000 --seed 12
  --reward done_false=-2.0`. Single-variable against `squad_v10` by construction
  — same tree (`5f848fb6`), same seed, same steps, one price. If it lands at
  0.97-0.98 with the claim volume collapsed, the price is the mechanism and the
  DEFAULT should move as part of v1.20. If it lands at 0.92 with claims gone,
  the claiming was a symptom and the regression is elsewhere.

- **2026-08-11 — the price A/B: mechanism CONFIRMED, fix REJECTED.** `squad_v11`
  (`done_false=-2.0`, single-variable against `squad_v10` — same tree, seed,
  steps, one price), N=100 final policy:

      axis                          squad_v10 (-0.5)   squad_v11 (-2.0)
      success, final                     0.92 ± 0.05        0.96 ± 0.04
      best -> final give-back                  -7 pt              +2 pt
      root claims (rejected)                178 (101)               0 (0)
      messages / episode                        101.2               77.5
      orders / episode                          17.38               7.85
      dist from OBJ under threat                11.21               9.53
      root death                                 0.30               0.20
      ROOT CLOSED ITS OWN OPERATION              0.84               0.00

  **Pre-registered before the run landed** (previous entry): recovery to
  0.97-0.98 with claims collapsed, AND a smaller give-back. The second was met
  cleanly and with its sign reversed — the drift *is* the give-back, and removing
  the incentive removed both. Every corroborating axis moved as predicted,
  including the cohort standing 1.7 cells closer to the objective under threat.
  **The mechanism is established**: the price drives the claiming, the chatter
  and the decay.

  **The first prediction only half held, and the half that failed is the
  interesting one.** Success 0.92 → 0.96 is **not significant** (Fisher
  p = 0.373; pooled against both −0.5 seeds, 180/200 vs 96/100, p = 0.076).
  Called as suggestive and underpowered, not as a win.

  **−2.0 is rejected on behaviour, not on the p-value.** It takes root-closed
  from 0.84 to **0.00**: the root never reports at all. That is `rewards.py`'s
  own warning — *over-pricing a speech act suppresses the HONEST one too* —
  reproduced exactly. Both runs announce 100% of wins, but only because v1.19's
  ENDEX guarantees it; the agent behaviour underneath is gone. Trading the thing
  v1.19 was built to measure for an unestablished 4 points is a bad deal.

  **`squad_v12` launched**: `root_done_bonus_first_claim_only=true` at the
  shipped `done_false=-0.5`. At `squad_v10`'s own claim precision (77/178 =
  0.433) the flag leaves the FIRST truthful claim at EV **+1.014** and drops the
  second and later to **−0.284** — spam stops paying, the honest report does not.
  Tested on the defend family and reverted; never tested on squad, which is
  where the spam lives.

  **⚑ The `+1.014 / −0.284` figures here are wrong — same dropped `done_true`
  term; see the `(assurance, #46)` entry. Corrected: at the POOLED precision the
  flag leaves a later claim at **+0.150**, still positive, so "spam stops paying"
  did not follow from the number it was derived from. It follows only when later
  claims are priced at the LATER-claim precision (0.314), where the flag gives
  **−0.029** — marginally negative, not comfortably so. Pricing an ordinal rule
  at a pooled rate was the error; `run_report.py` now prints the split.**
 Baseline for the read: `squad_v10`. The result wanted is
  `squad_v11`'s stability with `squad_v10`'s 84% reporting kept.

- **2026-08-11 (assurance, #46)** — **A pooled precision cannot price an ordinal
  rule, `squad_v12`'s pre-registration was written on one, and the EV under it
  had dropped `done_true`.** Three claims arrived while `squad_v12` was running.
  All three check out against the committed corpora — and the arithmetic
  underneath two of our own entries did not.

  **1. The control arm reproduces, net-only.** Their `squad_v10`/latest tap
  reads 178 root DONE claims, 101.22 messages/episode, 17.38 orders/episode.
  Those are `runs/squad_v10/behavior_final.json`'s own numbers to the digit
  (N=100, seed 123). Three for three, from the radio alone.

  **2. The 2026-08-07 `squad_screen` null belongs to the collapsed regime.**
  Confirmed here, on our corpora rather than theirs. `economics.json` says the
  pair is exactly single-variable: the only difference in the entire reward
  dict is `done_false` −2.0 (`v4`) vs −0.5 (`v5`), specs identical, same seed
  17, same 2M steps, same hyperparameters. Both FINAL policies are D4-collapsed
  and their final-decile false-DONE rates are 0.005 either way — that is the
  "unchanged at 4× the price" the entry recorded. At `ckpt_best`, where both
  arms are 1.00 ± 0.00 (N=20, seed 123):

      squad_screen_v4 / best   done_false −2.0    0 DONE claims
      squad_screen_v5 / best   done_false −0.5   55 DONE claims, 44 rejected

  Same direction as `squad_v10` → `squad_v11` (178 → 0); their own tap at N=30
  from seed 500 reads 0 and 35. **A null from a degenerate manipulation is not a
  falsification** — a collapsed policy emits the cheapest no-op because it must
  emit something, so a manipulation whose two arms are both degenerate has no
  power to detect an incentive response. The 2026-08-07 entry now carries that
  scope inline. Caveat kept and theirs too: the two bests are otherwise very
  different policies (orders/episode 67.70 vs 3.75), so this is one
  checkpoint-matched pair, suggestive rather than settled.

  **3. The ordinal split reproduces exactly — off our own published artifacts.**
  It needs no re-scoring and no `cohort/` change. `done_reports_root −
  done_rejected_root` is 0 or 1 per episode because the confirmed root claim is
  the LAST one (`tests/test_confirmed_claim_is_last.py`), so the claim count
  says which ordinal collected the acceptance. Every corpus we have committed
  carries the fields:

      corpus                 first claim      later claim    pooled   burn P
      squad_v10 / FINAL     50/92 = 0.543    27/86 = 0.314    0.433   27/42 = 0.643
      squad_v10 / best      45/95 = 0.474    41/75 = 0.547    0.506   41/50 = 0.820
      squad_v10b / FINAL    13/91 = 0.143   56/216 = 0.259    0.225   56/78 = 0.718

  `burn P` is P(the episode still closes by a root claim | the first claim was
  rejected) — what a spent opening probe forfeits under the first-claim rule,
  the quantity `rewards.py` measured at 1.000 on `defend_brique_v11` and
  reverted the rule over. The pool describes neither ordinal; the split INVERTS
  between one run's two checkpoints, and inverts again on the second seed. Any
  single number here is a confident wrong finding, which is why the digest now
  prints all of it.

  **4. OURS, and it changes a verdict: the EV arithmetic in the two entries
  above dropped `done_true`.** `rewards.py` states its own break-evens — 1/9 =
  **0.111** with the bonus on the table, 1/3 = **0.333** once the slot is spent.
  The DONE-probe entry quotes 0.143 and 0.400; the price-A/B entry quotes first
  **+1.014** and later **−0.284**. Every one of those reproduces exactly by
  pricing the claim on `root_done_bonus` alone. An accepted claim also pays
  `done_true` +1.0. Corrected, at the same pooled 0.433, airtime included:

      claim under the flag         as written     corrected
      first (bonus on the table)       +1.014        +1.437
      second and later                 −0.284        +0.139

  In the probe entry the slip is harmless — 0.275 clears 0.111 as easily as it
  clears 0.143, and `squad_v11` then confirmed that reading empirically. In the
  `squad_v12` pre-registration it is not: **at the pooled rate the flag does not
  make later claims −EV at all.** "Spam stops paying" does not follow from the
  number it was derived from.

  **5. What `squad_v12` is actually pre-registered on.** EV per root claim at
  each corpus's own realised rates, `done_true` and `transmission_cost`
  included, the first claim carrying its burn:

      corpus                first: shipped → flag    later: shipped → flag
      squad_v10 / FINAL        +1.936 → +1.055          +0.903 → −0.039
      squad_v10 / best         +1.622 → +0.327          +1.950 → +0.310
      squad_v10b / FINAL       +0.133 → −1.713          +0.657 → −0.121

  The flag's premise — *spam stops paying, the honest report does not* — holds
  on **one of three corpora and only just**. On `squad_v10`/best it fails on the
  spam half (a later claim is still worth +0.310). On `squad_v10b` it fails on
  the honest half: the first truthful report goes to **−1.713**, which is
  muteness — the failure that reverted this flag at v1.16 and that got
  `done_false=−2.0` rejected yesterday on behaviour rather than on a p-value.
  That is the falsifiable risk in v12, now quantified instead of asserted. The
  rates are realised under the CURRENT pricing and a retrained policy's will
  move; what they fix is the direction the incentive points from where v12
  starts. The run continues, and is read at BOTH checkpoints on the split,
  against `squad_v10` at both of its own.

  **Where their §3 is wrong, and it matters for what v12 tests.** They read
  0.314 against "the 0.333 root break-even at the shipped −0.5" and conclude the
  later claims are "already −EV before your change". 0.333 is the break-even
  *with the slot spent* — under the flag. At the shipped
  `root_done_bonus_first_claim_only=false` every accepted root claim earns the
  bonus (`cohort_env.py::_report_done`), so the break-even is 0.111 and a later
  claim is worth **+0.903**. The 86 later claims per 100 episodes are not an
  anomaly the flag cannot reach; they are precisely what the incentive asks for,
  and the flag is a +0.903 → −0.039 price change on them. Their conclusion —
  that later-claim spam might not be EV-driven, the way §12.61's voice-sync was
  action-mass — stands as a hypothesis. The premise it was argued from does not.

  **Shipped.** `run_report.py` prints the split and the burn on every behavior
  block and files `root_claims` / `first_claim_precision` /
  `later_claim_precision` into the summary, so `--vs` carries them as deltas:
  reading v12 against v10 on a pooled precision is now something the instrument
  has to be worked around to do. `tests/test_run_report_claim_ordinal.py` pins
  the derivation, both corrected break-evens, and `squad_v10`'s split at both
  checkpoints. **No `cohort/` change** — the quantity was always derivable from
  what every corpus already records — so the baseline seal (`5f848fb6`) and the
  eight published numbers are untouched. 815 → 822 tests.

- **2026-08-11 — the ordinal flag A/B, second seed: DOES NOT REPLICATE. Fix
  rejected; the spam looks less like EV every round.** `squad_v12b` (seed 13,
  `root_done_bonus_first_claim_only=true`, single-variable against `squad_v10b`
  — economics verdict: one price, same code) landed and was scored at N=100,
  seed 123, both checkpoints on the split. Side by side with seed 12, FINAL
  policy:

      axis                       seed 12: v10 → v12    seed 13: v10b → v12b
      success                    0.92 → 0.96           0.88 → 0.86
      give-back (best → final)    −7 pt → +2 pt         −5 pt → −10 pt
      root claims                178 → 132             307 → 328
      first claim precision      0.543 → 0.696         0.143 → 0.053
      later claim precision      0.314 → 0.375         0.259 → 0.184
      timeout rate               0.01 → 0.01           0.02 → 0.14
      root death                 0.30 → 0.17           0.25 → 0.00
      messages / episode         101 → 86              167 → 138

  **The seeds disagree on every axis the pre-registration cares about.** Seed 12
  is the wanted result to the letter — `squad_v11`'s stability with the
  reporting kept. Seed 13 answers the same flag with MORE claims (+21), the
  first-claim slot burned on a 0.053-precision probe, a doubled give-back and a
  +12 pt timeout mode. Pooled success 182/200 vs 180/200, Fisher p = 0.86 — a
  clean null — and the regression against the previous era stands under the
  flag: 182/200 vs 195/200, p = 0.0086.

  **The #46 quantification called the failing seed in advance.** It put the
  flag's premise at "holds on one of three corpora and only just", with
  `squad_v10b` the corpus where the honest first claim goes to −1.713 — and that
  is the seed that failed. But not by muteness, the failure the number
  predicted: the realised response is slot-burning and more volume under worse
  prices. On seed 13 every claim ordinal became less valuable and claiming
  ROSE. An EV-driven claimer claims less when claims pay less; this is the
  second consecutive result (after the −2.0 collapse's asymmetry) pointing at
  their §3 hypothesis — that later-claim volume is action-mass, not economics,
  the way §12.61's voice-sync was.

  **Rejected.** Both ends of the pricing axis are now closed by evidence:
  `done_false=−2.0` buys stability by muting the honest report (v11);
  `first_claim_only` fails to replicate and adds a timeout mode on exactly the
  seed where its own EV analysis predicted trouble (v12b). A default does not
  move on one seed in two. `runs/BASELINE.json` keeps `squad_v10`; the squad
  regression stays open with the discriminating question sharpened: is the
  claiming EV-driven at all? A zero-price probe — claim economics zeroed on an
  otherwise-shipped config — would answer it: volume that survives its own EV
  going to zero is action-mass, and the fix then belongs in masking or the
  claim API, not in prices.

- **2026-08-11 (assurance, #47)** — **The rejection stands; the record now says
  what it cost.** `squad_v12b` was rejected one entry up on success,
  replication and the timeout mode. Their finding, reproduced here from the
  committed corpora before writing this entry: on the axis nobody was weighing
  it is the best squad policy on record — **0/100 root deaths at BOTH
  checkpoints**, against `squad_v10b`'s 20/100 and 25/100 (Fisher p = 6.6e-07
  and 1.1e-08 as computed here; they quote 1e-06 and 4e-07 — smaller here, same
  verdict). The squad family's history is 17–30/100 across three economics;
  this is the only arm ever measured inside the 1–4/30 bound the
  commander-preservation question was framed with, and the rejection table
  carried the row (root death 0.25 → 0.00) while the rationale weighed success,
  the pooled null and the timeout mode only. Measured, not gated — the exact
  shape the assurance layer exists to point at.

  **Part of the zero is an artifact the raw rate cannot survive.** `squad_v12b`
  takes zero defeats, converting them into timeouts (2 → 14 at the FINAL
  policy), and in its control every defeat IS a root death (5/5 and 10/10), so
  some of the improvement is the policy declining the fight — the exact
  conversion `timeout_rate` was added to flag. But it is the smaller part:
  **within successful episodes alone, 14/93 → 0/96 (p = 2.9e-05) and 14/88 →
  0/86 (p = 7e-05)** — episodes that achieved the mission either way, where no
  clock-riding can produce the number. Their caveat stays attached to their
  finding: the seed-12 flag arm reads 19/100 and 17/100 against 22 and 30,
  n.s. at `best` — the same non-replication that rejected the flag, pointing
  the other way. The property belongs to the *seed-13 policy*, not to
  `first_claim_only`; nothing is un-rejected and no default moves.

  **Shipped: the axis is printed by the instrument now, not remembered.**
  `run_report.py::root_death_in_success` derives deaths-within-successes from
  the per-episode `outcome` / `human_died` fields every behavior corpus already
  records, prints it with its denominators on every behavior block, files it so
  `--vs` carries it as a delta, and adds it as the FOURTH cell of the #34
  comparison triple — success, raw deaths and the clock leave exactly this
  loophole open between them. Their first suggestion therefore costs nothing
  extra: the zero-price probe (`squad_v13_zeroprice`, in flight as this is
  written) gets read on this axis at both checkpoints by the same digest as
  every other run — and whether commander survival moves when claim volume does
  is the cheapest available test of whether the two are connected at all. Runs
  evaluated before per-episode outcomes read as an em dash, never a zero.
  `tests/test_run_report_root_death.py` pins the derivation, the
  gaming-immunity in miniature, and all four corpus counts at both checkpoints.
  **No `cohort/` change** — the quantity was always derivable from what every
  corpus already records — so the baseline seal (`5f848fb6`) is untouched.
  822 → 827 tests.

  **Left as the owner's call, on the record with its measurement:** their
  second suggestion — root-death-within-successes as a third term of the
  publish/stability bar. It is derivable from every committed corpus, it
  separated these arms at p < 1e-4, and unlike the raw rate it cannot be bought
  by declining the engagement. Adding it changes what is allowed to ship, which
  is a decision about the project's claims, not an experiment.

- **2026-08-12 — the zero-price probe answers its pre-registered question NO,
  and closes the squad regression on the way past. Both seeds agree. Nothing
  ships.** `squad_v13_zeroprice` (seed 12) and `squad_v13b_zeroprice` (seed 13)
  ran `done_true=0 done_false=0 root_done_bonus=0` on the otherwise-shipped
  config, and were scored at N=100 on both checkpoints. The arms and every
  comparator below are on **one `cohort/` tree, `5f848fb6`** — the frozen v1.19
  environment — so the whole pricing axis is readable side by side.

  **1. The pre-registered question: is the claiming EV-driven at all? No.**
  The entry above set the test — "volume that survives its own EV going to zero
  is action-mass, and the fix then belongs in masking or the claim API, not in
  prices." Volume did not survive. It *multiplied*, on both seeds:

      arm            done_false   claim rate / admissible step   x shipped
      squad_v10        -0.5              0.0060                    1.00
      squad_v10b       -0.5              0.0064                    1.07
      squad_v11        -2.0              0.0000                    0.00
      squad_v12b       -0.5 first-only   0.0208                    3.47
      squad_v13         0.0              0.0268                    4.47
      squad_v13b        0.0              0.0387                    6.45

  The cleanest cell is the **non-root** DONE claim. At zero price it earns
  literally nothing and still pays `transmission_cost` −0.01, so it is strictly
  −EV every time it is emitted; it goes from **1.47/ep on `squad_v10` to 8.67
  and 9.84** on the arms. Third result in a row where claims rose as their EV
  fell, and this time the EV is exactly zero and both seeds agree. **The
  pricing axis is closed.** `done_false=−2.0` buys silence (v11), the ordinal
  flag does not replicate (v12b), and zero price buys a claim-rate explosion —
  the volume is action-mass, as §12.61's voice-sync was, and the remedy belongs
  in masking or the claim API.

  **2. Unasked-for, and the bigger result: the squad regression closes.**
  Pooled FINAL-policy success at N=100, Fisher exact (the method reproduces the
  published p = 0.0031 exactly):

      previous era (v8+v9)          195/200 = 0.975
      shipped v1.19 (v10+v10b)      180/200 = 0.900     <- the regression
      ordinal flag (v12+v12b)       182/200 = 0.910     p = 0.86 vs shipped
      zero price (v13+v13b)         192/200 = 0.960     p = 0.0295 vs shipped

  Against the previous era the zero-price arm is **p = 0.57 — a clean null**.
  The 7.5-point regression that has been open since 2026-08-11 is not
  detectable under it. The two seeds agree (98/100 vs 94/100, p = 0.28), which
  neither v12 nor any earlier fix managed.

  **3. Which demolishes the premise the whole arc was built on.** The arm with
  the *highest* claim volume and the *worst* false-COMPLETE rate ever measured
  on squad is also the *best* on success; the arms with zero claims (v9, v11)
  score 0.97 and 0.96; the shipped middle scores 0.90. Across a claim rate of
  0.0000 → 0.0387 success runs 0.86–0.98 with no monotone relation. The
  tell was in the corpora the whole time and nobody had put the two columns
  adjacent: **`squad_v8` and `squad_v10` have identical claim rates (0.0060,
  0.0060) and differ by six points of success.** Claim volume was never the
  regression's signature. Every arm in this arc was aimed at the wrong quantity.

  **4. What this does NOT license.** Nothing ships and no default moves:
  - The recovery is **CONFOUNDED by construction** — three prices moved at
    once, and `run_report`'s economics audit says so. Which one carries it is
    unknown.
  - The price of the recovery is the report itself: **false-COMPLETE 0.844 /
    0.897** (`squad_v10`: 0.560), root claims 309 and 488 per 100 episodes with
    72–84% refused, first-claim precision 0.061 and 0.000. Unlike v11 this does
    *not* mute the net — contact `report_recall` holds at 0.88/0.86 against
    `squad_v10`'s 0.795, precision 0.93/0.95, doctrine containment 1.000, and
    every win is still announced (98/98 and 94/94 by ENDEX) — but a C2 project
    cannot ship a COMPLETE claim that is wrong nine times in ten.
  - **`runs/BASELINE.json` could not take it anyway.** It needs a `--reward`
    override to exist, and CLAUDE.md's rule is that what ships is what was
    trained: *"a scenario that needs an override to work is a finding about the
    defaults."* That is exactly what this is. `squad_v10` stays the member;
    `baseline.py` re-verified green after the re-scoring (seal `5f848fb6`
    untouched — the arms are not members, so their artifacts are not digested).

  **Read on the #47 axis, as that entry asked.** Root death *within successes*,
  FINAL at N=100: `squad_v13` 13/98 = 0.133 and `squad_v13b` 19/94 = 0.202
  against `squad_v10`'s 23/92 = 0.250 — pooled 32/192 vs 23/92, **p = 0.109**,
  n.s. Commander survival did not move when claim volume moved 4.5–6.5×, which
  is the cheapest evidence yet that the two are unconnected. `squad_v12b`'s
  0/100 remains a property of that policy, not of any price.

  **Launched, not decided** (`scripts/campaigns/squad_v14_nobonus.jobs`):
  `squad_v14_nobonus` / `squad_v14b_nobonus`, seeds 12 and 13,
  `root_done_bonus=0` **alone** — single-variable against `squad_v10`/`v10b` on
  the same tree. It isolates the largest of the three prices and the one #46's
  EV analysis centred on: at the shipped settings an accepted root claim pays
  `done_true` 1.0 + 3.0 = 4.0 against a rejected −0.5, break-even p = 0.111 and
  a later claim worth +0.903; removing the bonus alone moves break-even to
  0.333 without touching the penalty. If the recovery survives, the squad
  regression has a named cause and it is a **default**, which is the owner's
  call and wants the v1.20 window that `_fill_vacancy` (⚑) already forces a
  fleet retrain for. If it does not, the recovery is in `done_true`/
  `done_false` and needs its own split.

- **2026-08-12 — the squad regression has a cause, it is one price, and it
  replicates: `root_done_bonus=3.0`. The behaviour that price was buying does
  NOT replicate, so no default moves.** `squad_v14_nobonus` (seed 12) and
  `squad_v14b_nobonus` (seed 13) ran `root_done_bonus=0` **alone** —
  single-variable against `squad_v10`/`v10b`, same tree (`5f848fb6`), same 3M
  steps — and were scored at N=100 on both checkpoints.

  **Success: the strongest and cleanest result of the whole arc.**

      previous era (v8+v9)        195/200 = 0.975
      shipped v1.19 (v10+v10b)    180/200 = 0.900    <- the regression
      zero price (v13+v13b)       192/200 = 0.960    p = 0.0295 vs shipped
      no bonus  (v14+v14b)        196/200 = 0.980    p = 0.0011 vs shipped

  Against the previous era, **p = 1.000** — not merely a null, the same number.
  Both seeds landed on **exactly 98/100**. After v11 (one seed), v12/v12b (split
  on every axis) and v13/v13b (confounded by three prices), this is the first
  thing in the arc that is single-variable, replicated and significant at once.
  `root_done_bonus=3.0` is what costs the squad scenario its 7.5 points.

  **And it retires the zero-price result as the explanation.** v13's recovery
  was real but confounded; the bonus carries it. The claim-spam explosion in
  v13 was `done_false=0` removing the penalty, not the bonus — because removing
  the bonus alone moves claim volume the *other* way (rate 0.0050 on seed 12
  against `squad_v10`'s 0.0060, and 0.0000 on seed 13).

  **What does not replicate is everything else.** Side by side, FINAL at N=100:

      axis                        seed 12 (v14)    seed 13 (v14b)   squad_v10
      success                        0.98             0.98            0.92
      claim rate / admissible        0.0050           0.0000          0.0060
      closed on root's report        0.908            0.000           0.837
      false-COMPLETE                 0.508            n/a (1 claim)   0.560
      first-claim precision          0.619            --              0.543
      report recall                  0.908            0.927           0.795
      root death in success          0.367            0.224           0.250
      root claims / 100 eps          157              0               178

  Seed 12 is the wanted policy outright — the best completion reporting on
  record, closing 91% of its wins on the commander's own report against the
  shipped 84%, with the best first-claim precision the squad family has
  produced. **Seed 13 is `squad_v11` again**: the root files *zero* claims in
  100 episodes and closes 0.000 of its wins on its own report. The
  `successes_announced` gate reads 1.00 on both, because ENDEX is still sent —
  the gate counts announcement, not *who claimed it*, and this is the second
  time a mute commander has passed it. Commander survival split the same way
  and in the opposite direction (0.367 vs 0.224, seed 12 the bad one), so it
  is not a property of the change either.

  **The only quantity that replicates is success itself.** Removing the bonus
  reliably restores the mission and reliably does nothing predictable to the
  reporting — which is exactly the shape that must not be shipped on two seeds.
  `squad_v10` stays the member; `runs/BASELINE.json` untouched; the seal
  (`5f848fb6`) untouched; these arms need a `--reward` override to exist and so
  are findings about the defaults, not candidates for the fleet.

  **Launched, not decided** (`scripts/campaigns/squad_v15_bonus_axis.jobs`, 4
  jobs): `squad_v15_bonus1`/`v15b_bonus1` put `root_done_bonus=1.0` on seeds 12
  and 13 — an accepted root claim then pays `done_true` 1.0 + 1.0 = 2.0 against
  a rejected −0.5, break-even p = 0.20, keeping the claim worth filing while
  removing the +0.903 later-claim farming that 3.0 funds; and
  `squad_v14c/v14d_nobonus` add seeds 14 and 15 at `rdb=0` to turn "one seed
  each way" into a rate for the mute commander. **The decision this is
  evidence for — lowering `root_done_bonus` in `RewardConfig` — is the owner's,
  and it invalidates the baseline: it is a reward default, so the fleet
  retrains. That is the v1.20 window `_fill_vacancy` (⚑) already forces.**

  **One thing worth fixing regardless of how the price lands**, because it is a
  gate hole and not a tuning question: `successes_announced_rate` is 1.00 for a
  commander that never claims. `closed_on_root_report_rate` already measures the
  real thing (0.908 / 0.000 / 0.837 above) and is already committed on every
  corpus. Whether it becomes a gate term is a decision about what may ship —
  owner's call, noted here with its measurement, alongside the identical
  open question from #47 on root-death-within-successes.

- **2026-08-12 — `root_done_bonus=1.0` restores the squad scenario AND keeps
  the honest report, on both seeds. The arc has a recommendation; the decision
  is the owner's.** `squad_v15_bonus1` (seed 12) and `squad_v15b_bonus1`
  (seed 13), single-variable, same tree (`5f848fb6`), N=100 both checkpoints.
  At 1.0 an accepted root claim pays `done_true` 1.0 + 1.0 = 2.0 against a
  rejected −0.5 — break-even p = 0.20, the claim still worth filing, without
  the +0.903 later-claim farming that 3.0 funds.

  **The whole axis, FINAL policy at N=100:**

      arm                success   closed-on-root-report   false-COMPLETE   recall   rootDeathInSucc
      rdb=3.0 SHIPPED    180/200     0.837 / 0.784          0.560 / 0.805   .795/.848   0.250 / --
      rdb=0              196/200     0.908 / 0.000          0.508 /  n/a    .908/.927   0.367 / 0.224
      rdb=1.0            194/200     0.866 / 0.866          0.459 / 0.531   .896/.957   0.165 / 0.134

  Success p = **0.0073** against the shipped 180/200 and **p = 1.000** against
  the previous era's 195/200. But the number that settles it is the second
  column: **0.866 and 0.866**, identical across seeds, against `rdb=0`'s coin
  flip between the best reporting on record and a commander that never claims.
  Both v15 seeds converged (best-final gap 3 pts) and both pass the publish gate.

  **Every axis the project cares about improves over the shipped default, on
  both seeds, from one price change.** Success +7 points; completion reporting
  up (0.866 vs 0.837/0.784) and *stable*; false-COMPLETE down on both seeds
  (0.459/0.531 vs 0.560/0.805); report recall up (0.896/0.957 vs 0.795/0.848);
  and commander survival within successes — the #47 axis — **0.165/0.134
  against `squad_v10`'s 0.250**, which no arm in this arc had improved without
  declining the fight. `rdb=0` beats it on raw success by two episodes
  (196 vs 194, n.s.) and loses the only thing that mattered.

  **The recommendation, and it is a recommendation and not an action:** set
  `RewardConfig.root_done_bonus = 1.0`. That is a reward default — it
  invalidates `runs/BASELINE.json`, so the fleet retrains, and it is the
  owner's call under this repo's standing rule that experiments are
  pre-authorised and defaults are not. It wants the **v1.20 window**, which
  `_fill_vacancy` (⚑) already forces a full retrain for; the two changes cost
  one campaign together and two separately. Nothing has been applied:
  `squad_v10` is still the member, the seal is untouched, and these arms exist
  only under `--reward`.

  **Still running** (`squad_v14c/v14d_nobonus`, seeds 14 and 15 at `rdb=0`):
  they no longer choose anything — 1.0 is the candidate — but they turn "one
  seed each way" into a rate for the mute commander, which is what the record
  needs to say *why* 0 was not taken.

- **2026-08-12 — `rdb=0` at four seeds: the mute commander is 2 of 4, and the
  two prices are tied on success. The reporting is the whole difference.**
  `squad_v14c_nobonus` (seed 14) and `squad_v14d_nobonus` (seed 15) landed and
  were scored at N=100. The `rdb=0` arm now reads, FINAL policy:

      run                  seed  success  closed-on-root  root claims  false-COMPLETE
      squad_v14_nobonus      12   0.98        0.908           157          0.508
      squad_v14b_nobonus     13   0.98        0.000             0          --      MUTE
      squad_v14c_nobonus     14   0.93        0.000             0          0.789   MUTE
      squad_v14d_nobonus     15   0.99        0.919           139          0.552

  **Two of four commanders never file a completion claim** — 0 root claims in
  11,973 and 10,112 admissible steps respectively. It is a coin flip, not a bad
  seed, and seed 14 pays for it in outcome as well: 0.93 with **7 timeouts**,
  the weakest arm since the regression itself. Success spread across the four
  seeds is 0.93–0.99.

  **Pooled, the two candidate prices are indistinguishable on success:**

      rdb=3.0 SHIPPED   180/200 = 0.900
      rdb=0             388/400 = 0.970   p = 0.0007 vs shipped, p = 0.801 vs prev era
      rdb=1.0           194/200 = 0.970   p = 0.0073 vs shipped, p = 1.000 vs prev era
      prev era (v8+v9)  195/200 = 0.975

  Identical to three decimals. **The entire difference between them is whether
  the commander still reports the mission complete** — 2-of-4 mute at 0, 0-of-2
  at 1.0 with an identical 0.866 both times. For a project whose claim is
  doctrine-valid C2 traffic, that is not a tiebreak, it is the decision, and it
  points at **1.0**. The recommendation is unchanged and now rests on why the
  alternative was refused rather than on success alone.

  **Launched** (`scripts/campaigns/squad_v15_bonus1_seeds.jobs`):
  `squad_v15c_bonus1` / `squad_v15d_bonus1`, seeds 14 and 15 at `rdb=1.0` —
  matching `rdb=0`'s seed set exactly. Applying this changes a reward default
  and retrains the whole fleet; the recommended value should not rest on half
  the evidence of the value it is recommended over. If either of seeds 14/15
  goes mute at 1.0, the recommendation weakens to "better than 0, still not
  safe" and the honest move is a masking/claim-API fix instead of a price.

  **Unrelated, and worth knowing before it wastes someone's morning:** the
  training-end monitor reported `squad_nomask_v1` — an August-11 ablation run —
  as having just ended, because its stale `.job.json` records pid 69953 and
  today's campaign drew 70830, so a bare liveness check on the old pid flipped.
  The run directory is untouched and git-clean. `scripts/train.sh` already
  documents this exact hazard and `train_status.py --is-running` guards it by
  checking the pid actually carries `--run-name`; the monitor does not.

- **2026-08-12 — `squad_v14d_nobonus` turned the #33 negative control red, and
  the invariant was right: the *proxy* conflates two predicates, and succession
  is where they split.** Two episodes carried `done_reports_root -
  done_rejected_root` of 2 and 3 — "a second confirmed root claim", which
  `tests/test_confirmed_claim_is_last.py` calls structurally impossible and
  therefore "a broken measurement, not a strange policy". It is neither.

  **The two quantities are not the same quantity.** The invariant is about the
  root's **OPORD** claim: `cohort_env._report_done` closes the operation only
  when `is_root_opord_claim` holds (`cohort_env.py:1247`). But
  `metrics._done_traffic` counts `done_reports_root` as *any* DONE whose sender
  held the root at that step — deliberately, and its own comment says so
  (`metrics.py:801`, "the claim that matters is the one made BY whoever held
  the root"). Those agree while one soldier is root for the whole episode and
  **diverge the moment a successor is promoted**: the promoted commander still
  carries its personal SEIZE/ADVANCE mission, may truthfully complete *that* —
  confirmed by `is_complete`, counted here — while the operation correctly runs
  on to its real close.

  Both failing episodes are succession episodes with a dead commander (3
  confirmed over 2 successions; 2 over 1), which is what pointed at the cause,
  and the **env-level form of the test — which drives a real episode and reads
  the actual close — passes untouched.** So: no `cohort/` defect, no broken
  measurement, and the v14d numbers stand.

  **Fixed in the test, scoped rather than loosened.** The data-level assertion
  now runs on succession-free episodes, where the proxy is exact, and asserts
  that those remain at least half the corpus — so a regression that starts
  orphaning roots (the `_fill_vacancy` defect, ⚑) cannot hide inside the
  exemption instead of tripping it. 827 tests green again. **No `cohort/`
  change**, seal `5f848fb6` untouched.

  **Small honesty note on numbers quoted above and in the two prior entries:**
  every `root claims` / `false_complete_rate_root` figure counts sender-is-root,
  so in succession episodes it includes a commander's personal completions. The
  effect is 2 episodes in 400 here and changes nothing, but the corpus wants a
  root-*mission* claim counter before that column is ever quoted finely.

- **2026-08-12 — the arc closes at four seeds per arm. `root_done_bonus=1.0`
  is the recommendation, on narrower grounds than two seeds suggested.**
  `squad_v15c_bonus1` (seed 14) and `squad_v15d_bonus1` (seed 15) landed,
  matching `rdb=0`'s seed set exactly. FINAL policy, N=100 each:

      run                  seed  success  closed-on-root  false-COMPLETE  recall
      -- rdb=0 --
      squad_v14_nobonus      12   0.98        0.908           0.508        0.908
      squad_v14b_nobonus     13   0.98        0.000            --          0.927   MUTE
      squad_v14c_nobonus     14   0.93        0.000           0.789        0.889   MUTE
      squad_v14d_nobonus     15   0.99        0.919           0.552        0.921
      -- rdb=1.0 --
      squad_v15_bonus1       12   0.97        0.866           0.459        0.896
      squad_v15b_bonus1      13   0.97        0.866           0.531        0.957
      squad_v15c_bonus1      14   0.97        0.825           0.655        0.874
      squad_v15d_bonus1      15   0.98        0.857           0.623        0.929

      shipped rdb=3.0   180/200 = 0.900               mute 0/2
      rdb=0             388/400 = 0.970    p=0.00074  mute 2/4
      rdb=1.0           389/400 = 0.9725   p=0.00030  mute 0/4
      previous era      195/200 = 0.975

  **The two candidates are indistinguishable on success (p = 1.000)** and both
  null against the previous era. The whole case for 1.0 is the completion
  report, and the strongest single piece of it is **paired**: seed 14 files
  *zero* root claims at `rdb=0` and 0.825 at `rdb=1.0` — same seed, same tree,
  one price apart. Alongside it, the two `rdb=0` failures are absolute zeros
  over ~11k admissible steps each (a regime, not a low rate), and all four
  `rdb=1.0` seeds land in a 0.825–0.866 band.

  **Three claims from the two-seed entries above are corrected here, because
  four seeds did not support them:**
  1. **0-of-4 vs 2-of-4 mute is Fisher p = 0.43** — not significant on seed
     counts. The evidence is the paired flip and the absoluteness of the zeros,
     not the contingency table. Said plainly rather than quietly dropped.
  2. **false-COMPLETE is mixed per seed**, not uniformly better than shipped:
     0.459 / 0.531 / 0.655 / 0.623 against `squad_v10`'s 0.560 and `v10b`'s
     0.805. Pooled it does improve (0.581 vs 0.693).
  3. **Commander survival within successes is a null**, not an improvement:
     0.185 vs 0.206, **p = 0.57**, pooling all four seeds against both shipped
     seeds. The earlier "0.165/0.134 vs 0.250" compared two seeds against
     `squad_v10` alone and read as a gain that is not there.

  Report recall is the one secondary axis that holds uniformly: 0.874–0.957
  across all four, against shipped 0.795–0.848.

  **The recommendation stands and its grounds are now exactly this**: at
  `rdb=1.0` the squad scenario recovers the previous era's success (p = 1.000,
  four seeds) and the commander keeps reporting the mission complete on every
  seed tried, including the one that goes silent without it. `rdb=0` buys the
  same success and loses the report half the time. Nothing else separates them.

  **One pattern worth a look before the retrain**, visible on three of the four
  1.0 seeds and both late `rdb=0` ones: `ckpt_best` is often near-mute
  (closed-on-root 0.01, 0.00) while the FINAL policy claims normally — the
  claiming behaviour arrives late in training. The project publishes the FINAL
  policy, so every number above is unaffected, but a `best`-selected checkpoint
  would ship a mute commander, and `ckpt_best` is selected on success alone.

  **⚑ RESCOPED 2026-08-12 (assurance, #48) — this is not a cost of `rdb=1.0`.**
  The shipped price does the same thing: `squad_v10b` (rdb=**3.0**, seed 13, one
  of the two arms this whole comparison is anchored on) files **0 root claims in
  100 episodes at `ckpt_best` against 307 at FINAL** — closed-on-root **0.000 vs
  0.784** on its own committed corpora. Mute-at-`best` occurs under 3.0, 0 and
  1.0 alike, so as written this paragraph charged the challenger on an axis the
  incumbent is not graded on. The like-for-like reading is unchanged: the two
  prices are indistinguishable at `best` and separated only at FINAL, which is
  where the project publishes. What survives — and it is a **checkpoint-selection**
  property, not a pricing one — is the other clause: a `ckpt_best` chosen on
  success alone can silently discard the completion report. See the
  `(assurance, #48)` entry at the end of this log for the full ten-arm table,
  what v1.20's lexicographic `best_save_gate` closes, and the one gap it does not.

- **2026-08-12 — v1.20 OPENED: four `cohort/` changes landed together, the
  fleet is retraining.** The owner accepted all three recommendations, so the
  breaking window the last four entries were waiting for is open. Everything
  that needed it went in at once, because each change alone invalidates
  `runs/BASELINE.json` and a campaign freezes `cohort/`:

  1. **`RewardConfig.root_done_bonus` 3.0 → 1.0.** The named cause of the squad
     regression, on the evidence in the entries above. The `rewards.py` block
     now carries the four-seed table, the paired seed-14 flip, and why 1.0
     rather than 0 — an accepted root claim still pays `done_true` 1.0 + 1.0 =
     2.0 against a rejected −0.5, so the first claim breaks even at p = 0.20 and
     the channel stays open, while the +0.903 later-claim farming that 3.0
     funded is gone. The v1.16 note that "one knob controls both failure modes
     and cannot be set to avoid both" is annotated, not deleted: it is right
     about `done_true` and was wrongly generalised to the whole pricing axis.
  2. **`_fill_vacancy` links the successor into its new superior's chart
     (#42).** The one-statement patch that has been written and unapplied since
     2026-08-11. **It moves action masks** — the promoted branch becomes
     orderable, observable and devolvable-to — so it changes every rollout,
     which is exactly why it needed this window. The exhaustive sweep goes from
     **1,928 of 5,040** death orderings reaching a state with nobody in command
     to **0**, and `test_a_promoted_leader_is_on_the_chart_of_the_superior_it_
     reports_to` is unskipped and green.
  3. **`ckpt_best` is no longer selected on success alone.** `best_save_gate`
     is now lexicographic: a reporting window always supersedes a mute best
     whatever the success numbers say, a mute window may never take a reporting
     best back, and among windows of the same kind higher rolling success wins.
     `CohortEnv.root_close_step` is public so training can watch it, and
     `root_report_close_rolling` is a new `metrics.csv` column.

     **It is deliberately NOT a veto, and that was found by testing rather than
     reasoning.** The first implementation refused every mute save outright; a
     120k-step smoke run then produced **no `ckpt_best` at all**, which fails
     `baseline.py`'s "every checkpoint loadable" and makes `publish_baseline`
     report a missing artifact. A veto also creates its own inversion: a mute
     0.95 recorded early would lock out the reporting 0.90 that follows it.
     Training prefers; the *gate* refuses.
  4. **`closed_on_root_report_rate` is a regression gate**, floor 0.5, and
     unconditional — muteness is a third axis, not a collapse shape, so a run
     that wins everything and never reports must be able to fail on it alone.
     0.5 sits in a band nothing has ever occupied: every non-mute corpus on
     record is ≥ 0.784, every mute one ≤ 0.01. It exists because
     `successes_announced_rate` counts the ENDEX and not who claimed it, and so
     read **1.00 for three mute policies in one day** (`squad_v11`,
     `squad_v14b`, `squad_v14c` — zero root claims across 100 episodes each,
     passing every gate on the board at 0.93–0.98 success).

  **Also landed, because they were held for the same window and each changes
  the tree hash:** #40 (`parse_succession`, the succession formatters' inverse,
  so the four hand-written matchers for "did command actually pass" can collapse
  to one) and #39 (`eval_commit` in every evaluation artifact, via a new
  `cohort/training/provenance.py` that `train.py` and `evaluate.py` now share;
  `publish_audit.evaluation_era` prefers it and keeps the git fallback for the
  artifacts already on disk). Both tests unskipped. **The suite has no skips
  left: 827 → 833 passed, 0 skipped**, ruff clean.

  **Campaign launched**: `scripts/campaigns/v1_20_fleet.jobs` — all eight
  scenarios, same step budgets and same seed as their v1.19 counterparts, so
  the only thing that differs from the shipping fleet is the tree. The v1.19
  members stay in `runs/BASELINE.json` until their replacements beat them;
  publishing a MISS over an incumbent is still an ask.

- **2026-08-12 (assurance, #48)** — **Mute-at-`ckpt_best` was charged to the
  challenger; the incumbent does it too, and the digest never printed the axis.**
  The note in the four-seed entry above scoped near-mute `ckpt_best` as a thing
  to watch about `root_done_bonus=1.0`. It is not: the SHIPPED price does it on
  its own committed corpora. Every cell below was re-read here from
  `behavior.json` / `behavior_final.json` (all N=100, seed 123, greedy=False),
  not copied from the report:

      arm                  seed  rdb   root claims       closed-on-root
                                       best -> final     best -> final
      squad_v10             12   3.0    170 ->  178      0.869 -> 0.837
      squad_v10b            13   3.0      0 ->  307      0.000 -> 0.784
      squad_v14_nobonus     12   0        43 ->  157     0.125 -> 0.908
      squad_v14b_nobonus    13   0         2 ->    0     0.011 -> 0.000
      squad_v14c_nobonus    14   0         0 ->    0     0.000 -> 0.000
      squad_v14d_nobonus    15   0       131 ->  139     0.758 -> 0.919
      squad_v15_bonus1      12   1.0     255 ->  196     0.726 -> 0.866
      squad_v15b_bonus1     13   1.0       0 ->  169     0.000 -> 0.866
      squad_v15c_bonus1     14   1.0       3 ->  273     0.010 -> 0.825
      squad_v15d_bonus1     15   1.0       0 ->  175     0.000 -> 0.857

  Below the 0.5 floor at `ckpt_best`: **1 of 2 at rdb=3.0, 3 of 4 at rdb=0, 3 of
  4 at rdb=1.0** — the same failure at all three prices, and 1/2 against 3/4 is
  no separation at all. So the arc's verdict is untouched (it was decided at
  FINAL, where the two prices do separate) and the paragraph that scoped this to
  the new price is rescoped in place rather than deleted.

  **What did and did not reproduce, stated because the report is the evidence.**
  The root-claim counts reproduce **exactly**, all ten arms, both checkpoints.
  Two figures do not. (a) The report gives `squad_v10b` closed-on-root **0.043 →
  0.818**; the committed corpora say **0.000 → 0.784** — same verdict, and 0.784
  is the number `metrics.py` already cites as the weakest non-mute corpus on
  record. 0.043 is also not derivable from a corpus with zero root claims: on a
  completable root `endex_on_root_report` is set only by a truthful root-*mission*
  COMPLETE, so the rate cannot be positive where the claim count is 0 — unless
  the two counters are split by the #46 root-sender/root-mission distinction,
  which is the one mechanism that could produce it and is not what the committed
  aggregation does. (b) The aside that `squad_v8` filed "1 root claim at `best`
  while succeeding 97 times" reads **0 claims** at 0.97 success in
  `runs/squad_v8/behavior.json` — the point is if anything stronger.
  **A caution for anyone reading the claim column as the gate**: on the defend
  family they are different quantities. `fireteam_defend_v20` and
  `defend_brique_v15` file **0** root DONE claims at both checkpoints and read
  closed-on-root 1.000 and 0.990, because a continuous-posture root closes the
  window with a SITREP and has MISSION COMPLETE masked shut.

  **The actionable clause, against the current tree — v1.20 closes most of it.**
  The report was measured against v1.19 (`cohort_tree = ef52c421`), where
  `ckpt_best` really was selected on rolling success alone. On `HEAD` it is not:
  `best_save_gate` is **lexicographic and thresholded**, not a tie-break — a
  window is "reporting" iff `root_report_close_rolling >= ROOT_REPORT_CLOSE_FLOOR`
  (0.5), a reporting window supersedes a mute best whatever the success numbers
  say, and once the best is reporting a mute window may never take it back. That
  is the report's option 1, and it was in the v1.20 bundle before the report
  arrived. Two things it deliberately does not do: it is a **preference, not a
  veto** (a run that never reports still writes a mute `ckpt_best` — a veto left
  a 120k smoke run with none at all, which fails `baseline.py`'s "every
  checkpoint loadable"), and it selects on the **rolling training window**, which
  is a stochastic on-policy estimate and not the N=100 evaluation, so a `best`
  chosen as reporting can still evaluate under the floor.

  **The residual gap is exactly the report's option 2, and it is unimplemented.**
  `publish_baseline.py` scores BOTH checkpoints and `evaluate.py` now writes the
  `closed_on_root_report_rate` gate into both artifacts — but nothing on the
  publish path reads the `best` one. `baseline.py::_run_facts` takes
  `gates_failed` from `behavior_final.json` only (`behavior.json` is checked for
  its episode count and nothing else); `results_table.py`'s gate cell is the
  final artifact's; `fleet_status.py` sets `head = final or best`, so on any run
  with a final evaluation the best artifact's gates are never surfaced. A mute
  `ckpt_best` will therefore record its own failed gate and still pass the fleet
  audit.

  **Owner's call, with the cost measured rather than asserted.** On the v1.19
  fleet as published, **two of eight members are mute at `ckpt_best`** —
  `patrol_brique_v6` 0.000 (1 root claim / 100 eps) and `platoon_v6` 0.021 (10) —
  while both report normally at FINAL (0.808 and 0.930). A publish gate that
  refused a mute `best` would refuse those two today. **Recommendation: decide it
  when the v1.20 campaign lands, not now.** That fleet is the first ever trained
  under lexicographic selection, so it supplies the one datum the decision is
  missing — whether preferring a reporting window at training time already lifts
  `ckpt_best` over the floor fleet-wide. If it does, the refusal costs nothing
  and is pure insurance; if it does not, the choice is a real one between
  shipping mute peaks and blocking members, and it should be made with those
  numbers in hand. Either way it changes what is allowed to ship, so nothing was
  applied here.

  **Shipped: the digest prints the axis now, at both checkpoints.** The reason
  this scoping error was available to make is that `run_report.py` — "the ONLY
  thing the big model reads" — printed `report_precision`, `report_recall` and
  `false_complete_rate` but never `closed_on_root_report_rate`, the quantity the
  v1.20 default was chosen on and the one it is now gated on. Same fix as
  `human_death_rate` got for #22: one row in `_BEHAVIOR_ROWS`, so it prints under
  both the `ckpt_best` and FINAL blocks and `--vs` carries it as a delta at both.
  `tests/test_run_report_root_report.py` pins it against these corpora, including
  the best/final split that started this. Tooling only, **no `cohort/` change** —
  the v1.20 campaign's tree is untouched.

- **2026-08-12 (assurance, #49)** — **No, there is no silent reattachment path —
  and the residual is almost certainly the monitor's own missing rule.** #49 asks
  whether a soldier can acquire a new commander with no radio act, and what
  happens to a vacated branch when no eligible successor exists. Both halves are
  answerable by reading code, and now by test:
  `tests/test_succession_silence.py`, write-up `docs/succession-on-the-net.md`.
  **Documentation and tests only — no `cohort/` change**, the v1.20 campaign's
  tree is untouched.

  **The structural answer.** `leader_id` — the only representation of "who
  commands me" — is assigned in exactly **two statements in the whole package**,
  both inside `Roster._fill_vacancy`, both between `_pick_successor` returning a
  successor and `events.append(...)`, i.e. on the branch `CohortEnv.step` turns
  into a `TAKING_COMMAND` broadcast. That is asserted with an AST sweep rather
  than by enumerating paths, so a third write site fails the suite. Exhaustively:
  over all 5,040 death orderings plus every same-step pair and triple, the parent
  map rebuilt from the announcements alone equals the parent map in state, and a
  chart that did not move produces no traffic.

  **The no-successor case.** `_pick_successor` returns `None` only when the
  vacated leader has no living *direct* subordinate, so the branch is empty by
  construction and `_fill_vacancy` returns without touching the roster. Nothing
  is re-homed onto the grandparent — living descendants under an already-dead
  direct subordinate stay where they are. Kill SL1, TL1 and TL2 in one tick and
  four of seven soldiers stand under a dead squad leader with `root()` None.

  **The two divergences, both reachable in one tick of `squad`, neither silent.**
  Measured against `cohort.probe.NetPredictor`, this repo's own transcript-only
  reconstruction. (i) *A real orphan the net hides*: the casualty loop devolves a
  tick's deaths one at a time against alive-flags that already count all of them,
  so SL1+TL1 together leaves RFN1 under the dead SL1 — while a replay, which must
  consume messages in sequence, has not yet heard TL1's CASUALTY when it replays
  TL2's succession, sweeps TL1 up to TL2, and reports the chain intact. (ii) *A
  false orphan the net invents*: `_assume` re-points a vacated slot's downward
  edges but never files the successor under its new superior, which is exactly
  the link #42 added to state — so TL2+SL1 together leaves state **whole** (RFN3
  swept up to TL1) while the net leaves RFN3 hanging off the dead SL1.

  **(ii) is the shape of an orphaned-branch residual**, it needs no new radio
  act, and the fix is one rule on the monitor's side: *a successor joins the
  subordinate list of the slot it assumed.* "Takes the vacated slot" has to be
  read in both directions. Whether the genuinely headless branch of (i) should
  get a line on the net is a vocabulary/semantics decision and stays the
  owner's; nothing here changes what is transmitted.

  **#42's own footprint, since it is the code under suspicion.** It introduced no
  silent transition — the link it adds is the upward half of a move the broadcast
  already describes. It did change *who* succeeds in same-step cascades (17 of
  the 252 ordered pairs/triples differ from the pre-#42 tree, including one where
  a twice-promoted rifleman takes the squad ahead of an intact team leader, since
  `_pick_successor` breaks the authority tie on `-id` and an acting-TL ties a
  real TL), and it halved the damage without closing it (same-step batches
  leaving a headless branch 58 → 30, leaving no root 6 → 2). README's #42 scope
  box was still describing the pre-fix world; corrected with the new counts.

  **⚑ A defect #42 introduced, found on the way — one line, `cohort/` frozen.**
  `_fill_vacancy` links the backfilled agent into its new leader's
  `subordinate_ids` **twice**: once at #42's `parent.subordinate_ids.append`,
  once at the pre-existing `successor.subordinate_ids.append(promoted.id)` that
  #42 made redundant (the `not in` guard runs first, so it does not help). The
  commonest succession in the game triggers it — SL1 falls, TL1 takes the squad,
  RFN1 backfills, TL1's chart reads `[TL2, RFN1, RFN1]`. `living_subordinates` is
  what `env/observations.py` writes into the four subordinate slots and what
  `env/actions.py` indexes with `order_slot`, so from the moment it takes command
  the new **root** spends an observation slot on a duplicate and carries two
  distinct ORDER action indices addressing the same agent. Hit in 4 of 50
  `patrol_brique` episodes under random play; pre-#42 the same cascade produced
  `[TL2, RFN1]`. Pinned by a strict `xfail` so the marker must come out with the
  fix. **Not fixed here**: it is a `cohort/` change and the v1.20 fleet is
  mid-campaign, so it belongs to the next breaking window — and it is a live
  candidate explanation for post-#42 behaviour changes on succession-heavy
  scenarios, since it corrupts the root's observation and order-slot mapping
  exactly when the root is a promoted agent.

- **2026-08-14 (refs #52)** — **#42's chart block is what silences the squad
  commander, and the mechanism is positional: with the block, the root stops
  going to the objective.** Two paired seeds, single-variable, N=100 at *both*
  checkpoints, plus a control seed that fails on both trees.

  **The design.** Branch `exp/pre42-chart-link` (`cd23d44`, never merged) is
  current HEAD minus `56ada9a`'s 8-line block in `_fill_vacancy` — tree state
  **A**. Everything else is held: `root_done_bonus=1.0`, both new gates,
  lexicographic `ckpt_best` selection, all measurement code. Tree **C** is HEAD.
  Ordering matters and the first design of this probe had it wrong: `56ada9a`
  introduced the block, `da24b42` only corrected the double-link inside it, so
  reverting `da24b42` alone lands in the middle state B, where `squad_v16` is
  *also* mute. B is not a control; A is.

  **The result — FINAL policy, N=100, seed-paired.**

        seed 14   squad_v23_pre42_seed14 (A)   vs  squad_v10c (C)
          closed-on-root        0.825   vs  0.000      eps w/ root claim  97 vs 0
          success               0.970   vs  0.900      Fisher p = 0.0818, CIs overlap
          mean obj distance     19.56   vs  41.68
          commander death       0.240   vs  0.000      successions  36 vs 10

        seed 15   squad_v22_pre42_seed15 (A)   vs  squad_v20_seed15 (C)
          closed-on-root        0.857   vs  0.000      eps w/ root claim  97 vs 2
          success               0.980   vs  0.960      Fisher p = 0.6827, CIs overlap
          mean obj distance     19.54   vs  40.52
          commander death       0.220   vs  0.030      successions  33 vs 9

  `ckpt_best` says the same thing and is not a weaker version of it: 0.835 vs
  0.052 on seed 14, 0.811 vs 0.000 on seed 15. Tree C fails
  `closed_on_root_report_rate` at both checkpoints on both seeds; tree A fails
  no gate anywhere.

  **Success is a null; the reporting channel is the entire effect.** Neither
  seed separates on success (p = 0.08 and 0.68, CIs overlapping), which is the
  point — on tree C the squad still takes the objective at 90–96%, it just takes
  it without the commander, who sits at **2.2× the distance** and therefore has
  no truthful DONE to file.

  **The branch is the historical tree, confirmed rather than asserted.**
  `squad_v22`'s 0.857 and `squad_v23`'s 0.825 reproduce `squad_v15d_bonus1`
  (0.8571) and `squad_v15c_bonus1` (0.8247) — the same seeds on the tree the
  v1.19-era runs actually ran on — to three decimals. That is the positive
  control for A being state A.

  **The control that keeps this honest: seed 16 fails on BOTH trees.**
  `squad_v24_pre42_seed16` is 0/100 with `timeout_rate` 1.000, exactly like
  `squad_v21_seed16` on HEAD. The block does not explain every squad failure and
  seed 16 is a bad draw, not a tree effect.

  **#52's new rows earned their place immediately.** Probing the mute policy
  (`squad_v20_seed15`, tree C, 40 eps) with the claim-split added in `e19589b`:
  silent-episode occupancy **0.002**, inside the assurance layer's own *absent*
  band (≤ 0.004). So this root is the **never-arrives** mechanism, not the
  present-but-silent one their 8 arms populate. The two mechanisms are now
  distinguishable on the digest instead of pooled.

  **What this does NOT settle.** Seeds 12 and 13 report normally on tree C
  (`squad_v17` 0.959, `squad_v18` 0.889), so muteness is a regime the block makes
  *reachable* on some seeds, not a universal consequence of it. Whether tree A is
  uniformly reporting is being measured now by `squad_v25_pre42_seed12` and
  `squad_v26_pre42_seed13`.

  **⇒ OWNER'S DECISION, not taken here.** #42 fixed a structural defect — 4,080
  of 5,040 squad death orderings orphan a branch, 1,928 reach `root() is None` —
  and this says removing it buys the commander's voice back at **0.22–0.24
  commander deaths per episode against 0.00–0.03**, with successions 3–4× up and
  12–15 of them unrecovered. That is trading a structural defect for a
  behavioural one, in both directions. Three options, none applied:
  (1) keep #42, accept a mute squad commander on some seeds, and revisit whether
  `closed_on_root_report_rate`'s 0.5 floor should block a member that wins
  silently; (2) revert the chart block and take the casualties; (3) keep #42 and
  address *why* a fully-charted root stops advancing — the reward question, which
  wants an oracle diagnosis before any price moves. The measurement that would
  most cheaply separate (1) from (3) is whether the seed-12/13 A-arms also show
  the commander forward at ~19.5 with the same casualty cost, i.e. whether the
  distance/survival trade is intrinsic to the block or specific to the seeds that
  go mute.

- **2026-08-14 (refs #52) — the seed-12/13 arms landed and CORRECT the cost
  claim in the entry above: there is no measured casualty trade. The block does
  not move the commander's mean behaviour, it makes a second regime reachable.**
  Four seeds per tree, FINAL policy, N=100 each, matched (`squad_v18` re-scored
  from N=20 to N=100 so every cell is like-for-like).

        seed   closed-on-root        mean obj distance      commander death
               A (no block)  C        A        C            A       C
         12       0.866    0.959    19.28    19.97        0.180   0.120
         13       0.866    0.937    19.89    20.57        0.150   0.270
         14       0.825    0.000    19.56    41.68        0.240   0.000
         15       0.857    0.000    19.54    40.52        0.220   0.030
         16      collapse on both trees — 0/100 and 0/20, timeout 1.000

  **Tree A's band is 0.825–0.866 across four seeds — 0.041 wide. Tree C is
  bimodal: 0.937–0.959 or exactly zero.** Success is a null on every pair
  (Fisher p = 1.00, 0.72, 0.08, 0.68).

  **The correction.** The entry above reported the cost of removing the block as
  "0.22–0.24 commander deaths against 0.00–0.03". Those numbers are right but the
  attribution was wrong: 0.00–0.03 is what the *mute* seeds pay, not what tree C
  pays. Where tree C lands in the same forward regime it pays **0.120 and
  0.270** — overlapping tree A's 0.150–0.240. The casualties belong to the
  regime, not to the tree: a commander that walks onto the objective and reports
  it also dies there. **Removing the block has no measured cost on this
  evidence.** What it removes is the availability of the alternative — a
  commander that stays at 2.2× the distance, survives, says nothing, and still
  wins 90–96% of the time because the squad takes the ground without it.

  **Stated with its weakness, as the rdb arc taught.** 0-of-4 mute versus
  2-of-4 is Fisher **p = 0.43** on seed counts alone — not significant. What
  carries it is what carried the rdb=1.0 decision: the zeros are *absolute* (0
  and 2 root claims in ~11k and ~10k admissible steps, not a low rate), the two
  paired flips are exact and hold at both checkpoints, and the A band is tight
  where the C band is not.

  **One thing that runs the other way and should not be buried**: where tree C
  reports, it reports *better* — 0.959 and 0.937 against A's 0.866 and 0.866.
  The block helps on the seeds where it does not hurt. So option (3) — keep #42
  and find why a fully-charted root sometimes stops advancing — is not the
  consolation prize; it is the option that keeps both properties, and the oracle
  diagnosis it needs has not been run.

- **2026-08-15 (refs #52) — the two mute commanders have OPPOSITE order
  economics, so no single reward lever explains them.** `order_pay_by_rank`
  (`847acec`) splits every adjudicated mission order into fresh (paid) / churn
  (charged) / re-task (order channel pays nothing), per issuer rank. Root (SL)
  only, 30 episodes at seed 500 — the diagnosis protocol, not N=100:

        policy          fresh  churn  retask   order-channel total   per episode
        C mute  s14       528     66       7          +19.8             +0.66
        A rep   s14        61     17      99           +1.4             +0.05
        C mute  s15        60     67     190           −3.7             −0.12
        A rep   s15        60     15      51           +6.1             +0.20

  **Seed 14's mute root is farming the channel.** 528 fresh taskings — 17.6 an
  episode, none of them doctrine-preferred — earning **+0.66/episode against a
  `root_done_bonus` of 1.0 paid once**. Ordering from 40 units behind the
  objective returns two thirds of the completion bonus it forgoes by never
  arriving, every episode, at no risk: that root is never in contact and never
  dies.

  **Seed 15's mute root is doing the opposite — it PAYS to stay silent**, −3.7
  over the same block, mostly 190 re-tasks and 67 churned reissues. Whatever
  holds it off the objective, order income is not it.

  **⇒ This argues against a pricing fix, and that is the useful part.** A cap or
  tariff on the order channel would address seed 14 and leave seed 15 exactly
  where it is — and the pricing axis has already been closed once on this
  scenario (the zero-price probe, 2026-08-12). What the two mute policies share
  is not economics but geometry: occupancy 0.000 and 0.004, zero SEIZE orders,
  ~40 from the objective. The mechanism established on 2026-08-14 — *the root
  claims iff the root occupies* — still holds and is still the only thing both
  seeds obey.

  **A defect in the record, found by the invariant test.** A churned reissue is
  charged `order_churn` and then returns **without `_say`**, so it never reaches
  the transcript: every transcript-derived order count in this repo undercounts
  a reissuing commander by exactly the orders it is being charged for (6 of 198
  on the pinned squad episode; 66 and 67 for the two mute roots above). Whether
  a charged no-op should appear on the net is a vocabulary decision and stays
  the owner's — but until it does, `orders_issued` is not a measure of what a
  commander did, and `tests/test_metrics.py` now says so where someone will read
  it.

- **2026-08-15 (assurance, #53) — the raw count WAS inflated by survival, the
  metric is fixed, and on the corrected denominator the separation is real and
  tighter than either party's numbers: 2.04× on both seeds.** Three denominators,
  same four policies, 30 eps at seed 500, root only:

        seed   raw per-episode   per root-alive-step   per root SITREP
         14         3.34×              2.04×                0.81×
         15         2.25×              2.04×                0.92×

  **#53's critique is correct and the fix is in** (`1487461`): the mute root
  survives (0.00–0.03 commander deaths against tree A's 0.22–0.24) and its
  episodes run longer, so a raw per-episode count credits it for time rather
  than for commanding. `orders_by_rank` and `order_pay_by_rank` now carry
  `rank_alive_steps` beside them, and the printed table shows the rate.

  **But their conclusion — "the mute root does not command more, it commands for
  longer" — does not survive the internal denominator, and the reason is their
  own instrument.** They normalised by root SITREPs, the only clock the net
  exposes, and said so explicitly. On this repo's own artifacts the mute roots
  emit **2.51× and 2.23× the SITREPs per alive-step**, at an off-cadence share of
  **0.75 and 0.68** against 0.32 and 0.44. That denominator is a behaviour, not a
  clock: divide the true 2.04× order rate by it and you get exactly the 0.81×
  and 0.92× they report. The inversion is the SITREP rate, not the order rate.

  **So the order-rate hypothesis is alive on its third reversal, and the
  bookkeeping is worth stating plainly**: `retasks_by_rank` measured the wrong
  channel (opposite signs on the two mute seeds); raw `orders_by_rank` measured
  the wrong denominator (3.34× vs 2.25×, inconsistent); root-alive-steps gives
  **2.04× on both seeds** — the first version of this quantity that agrees with
  itself across the pair. A separator that lands on the same figure twice is
  worth more than one that lands on two.

  **What it still does not do is identify a cause.** Commanding twice as fast
  per unit of its own life is a description of the mute regime, not a mechanism
  for it, and the economics beneath it remain opposite (+19.8 against −3.7 on
  the order channel). The falsifiable claim from 2026-08-14 is still the only
  one both seeds obey: *the root claims iff the root occupies*.

- **2026-08-15 — v1.20b is measured and NOT SHIPPING: seven members match their
  incumbents and `patrol_brique_v7` fails the gate v1.20 itself introduced.**
  Owner's decision, taken on these numbers: hold v1.19, open v1.21. FINAL policy,
  N=100 against N=100, every candidate on one tree (`5fa24bad`), no overrides:

        scenario          candidate             incumbent          success        verdict
        fireteam          fireteam_v11          fireteam_v9        94 vs 97       match
        fireteam_defend   fireteam_defend_v22   fireteam_defend_v20 99 vs 98      match
        squad             squad_v17             squad_v10          97 vs 92       match
        squad_recon       squad_recon_v10       squad_recon_v8    100 vs 99       match
        squad_screen      squad_screen_v13      squad_screen_v11  100 vs 98       match
        patrol_brique     patrol_brique_v7      patrol_brique_v6   96 vs 99       BLOCKED
        defend_brique     defend_brique_v16     defend_brique_v15 100 vs 100      match
        platoon           platoon_v7            platoon_v6        100 vs 100      match

  Every CI overlaps — no success regression anywhere — and give-backs run
  0.2–4.6 points against a 10-point bar. **The blocker is the reporting channel,
  not the mission**: `patrol_brique_v7` files **0 root claims in 100 episodes**,
  `closed_on_root_report_rate` 0.000 at both checkpoints, where
  `patrol_brique_v6` reports at 0.808. Its digest is the mute regime exactly as
  the squad investigation described it — ADVANCE 0.96 / SEIZE 0.01, root death
  rate 0.000, false-DONE 1.000, and a commander that never stands on the ground
  it would report.

  **No substitute member exists, and that was checked rather than assumed.**
  `patrol_brique_v8_rdb3` reports (0.867) and sits on the fleet's own tree, but
  it was trained with `--reward root_done_bonus=3.0` and a member may not carry
  an override — what ships is what was trained. `patrol_brique_v11_seed14`
  reports but resolves to a different tree (`c0f85409`). And a patrol_brique-only
  retrain now lands on a *third* tree, since `8a7645c` and `847acec` moved
  `cohort/` after the campaign closed. **The fleet is all-or-nothing from here.**

  **⇒ The finding underneath, at N=100: `root_done_bonus` has no
  scenario-independent value.**

        patrol_brique, current tree, seed 12      rdb=3.0   0.867   (136 claims)
                                                  rdb=2.0   0.000   (0)
                                                  rdb=1.0   0.000   (0)
        at rdb=1.0, seeds 12 / 13 / 15                      0.000, 0.000, 0.000
        (seed 14 reports, 0.850, N=20 — the only one)

  The cliff is between 3.0 and 2.0 and it is not a seed lottery: three of four
  seeds are mute at 1.0. **3.0 is the price that was removed in v1.20 because it
  regressed squad** — the arc that produced `rdb=1.0` was four squad seeds deep
  and entirely squad. It never asked what the price does elsewhere, and in
  patrol_brique it is the difference between a commander that reports and one
  that never speaks.

  **Two levers, one silence.** Squad's muteness moves with the chart block (2 of
  4 seeds mute with it, 4 of 4 reporting without, band 0.041 wide);
  patrol_brique's moves with the price on a fixed tree. The reading that covers
  both is the mechanism already established: the root claims iff it occupies,
  and it occupies only while completing is worth more than not completing.
  The block raises what the root gets from *not* completing (more subordinates
  to order); cutting the bonus lowers what it gets from completing. Both shrink
  the same margin from opposite sides, and the scenarios sit at different
  distances from the cliff — which is why one fleet-wide number for either lever
  has now failed twice.

  **v1.21's first campaign should be the one that separates them**: patrol_brique
  with the chart block removed at rdb=1.0. If it reports, the block is
  implicated in both scenarios and the price cut is survivable; if it stays
  mute, the price is the whole story there and the two scenarios need different
  treatment — which is itself the answer to whether `RewardConfig` can stay
  scenario-independent.

- **2026-08-15 (assurance, #54) — CORRECTION, twice over: "the root claims iff
  it occupies" was too strong in BOTH directions, and "the price is the whole
  story in patrol_brique" was a sufficiency claim I had no licence for.** #54
  produces `patrol_brique_v5` as a counterexample and it holds. Investigated on
  this repo's own artifacts; oracle probe at the standard `--episodes 30 --seed
  500`, run against each checkpoint:

        checkpoint      claim eps  silent eps   dist    OCCUPANCY   occ|claimed  occ|silent
        v5 best             0          30      17.87     0.0110         —          0.0110
        v5 latest           0          30      18.51     0.0208         —          0.0208
        v6 best             0          30      30.72     0.0000         —          0.0000
        v6 latest          23           7      22.78     0.0299       0.0408       0.0000

  **What #54 gets right.** `patrol_brique_v5` files 0 root claims at
  `root_done_bonus = 3.0`, the price at which v6 files 103. A sufficiency claim
  dies to one counterexample and this is one: **3.0 does not buy reporting**, and
  the entry of 2026-08-15 above overstated the rdb sweep, which only ever
  licensed *lowering* the price causing silence. Corrected.

  **And it breaks the mechanism's sufficiency too — on the quantity #54 chose not
  to quote.** They argue from distance and say occupancy tracks the tree. On
  occupancy their headline reading does not hold: the *vocal* v6-latest has the
  highest occupancy of the four cells (0.0299) and the mute v6-best the lowest
  (0.0000), so it is not true that the mute root occupies more. But v5 occupies
  **0.011–0.021 and never claims, across 3M steps and both checkpoints** — it
  reaches the objective and declines. That is their present-but-silent band, on
  the better statistic, and it kills the *if* direction of my claim.

  **What survives is the necessary direction, and it survives well.** Every
  zero-occupancy cell is mute — v6-best, squad's `v10c`/`v20_seed15` (0.000–0.004),
  and v6-latest's own silent episodes. The sharpest cell in the whole
  investigation is **within one checkpoint**: v6-latest's 23 claiming episodes
  occupy at **0.0408** and its 7 silent episodes at **exactly 0.0000** — same
  policy, same seed, same tree, same price, split by whether the root claimed.
  Nothing between-run can be that well matched.

        RESTATED, and this is the version to attack next:
          necessary      no occupancy -> no truthful claim        (well supported)
          NOT sufficient occupancy > 0 does not produce a claim   (v5 refutes it)

  **A confound in #54's pair that its own table does not flag.** v5 and v6 are
  matched on `root_done_bonus` and `done_false`, but v5 ran at **lr 1e-4** and v6
  at **3e-4** — a 3× difference — on **seeds 3 and 12**. The economics are
  matched; the optimisation is not. That does not rescue the sufficiency claim
  (a counterexample is a counterexample) but it does mean the pair cannot be
  read as "the same setup flipping", and the third factor #54 invites us to look
  for has two unflagged candidates sitting in `config.json`.

  **Two provenance notes, both checked.** (i) The `cohort/` trees genuinely
  differ (`0293107` vs `5f848fb`), and all 19 intervening commits touching
  `cohort/` were walked: every substantive change is `DEFEND`/`DENY`-scoped or a
  metrics/CLI addition, and `is_completable` / `is_root_opord_claim` /
  `is_done_admissible` reduce to the identical test for a SEIZE root either side.
  Reward mechanics are ruled out. (ii) **#54's derived rates for v5 are not
  comparable to ours**: v5 was scored by a pre-ENDEX-generalization evaluator —
  `endex_sent = 0` on a 99%-successful run is the fingerprint — so its
  `closed_on_root_report_rate` is `None` here, not 0.020. The raw claim counts
  (0 / 1 / 103) are unaffected and match exactly; the counting logic is unchanged
  across the span. This is the `eval_commit` gap that `56ada9a` closed one day
  after both runs were scored.

  No code change: the masking and reward logic for a SEIZE-rooted
  `patrol_brique` is provably unchanged across v5→v6, and stamping `eval_commit`
  retroactively onto published artifacts would be fabrication, not repair.

- **2026-08-15 — the patrol_brique block-removal arms SPLIT, and they kill two
  more associations I had drawn. Pre-registered outcome three of three.** Both
  arms trained on the branch with `56ada9a`'s chart block removed, `rdb=1.0`
  default, no overrides, 3M steps, N=100 on both checkpoints:

        seed   A (block removed)          C (fleet tree)             closed-on-root
         12    patrol_brique_v13_pre42    patrol_brique_v7           0.825  vs  0.000
         13    patrol_brique_v14_pre42    patrol_brique_v10_seed13   0.000  vs  0.000

  **Seed 12 is a clean paired flip** — 154 root claims in 90 of 100 episodes
  against zero, success 0.97 vs 0.96, and the gate goes from FAIL to pass. The
  block IS implicated in `patrol_brique`. **Seed 13 does not move**: mute on both
  trees, 0 claims either side. So removing the block is neither necessary
  (v5 reported with it present) nor sufficient (v14 is mute without it). As
  pre-registered: a split decides nothing on its own, and two arms is two arms.

  **Association 1, dead: ADVANCE-dominance ↔ muteness.** The squad mute policies
  were ADVANCE 0.69–0.93 with SEIZE ≈ 0, and I read the order mix as part of the
  regime. Here it reverses cleanly. `v14` (mute) orders **SEIZE 0.98 / ADVANCE
  0.01**; `v13` (vocal) orders **ADVANCE 0.88 / SEIZE 0.09**. A commander can
  order nothing but SEIZE for a whole run and never claim one.

  **Association 2, dead: occupancy magnitude ↔ claiming.** Oracle probe, 30 eps,
  seed 500:

        patrol_brique_v13_pre42_seed12   27/30 claiming episodes   occupancy 0.003
        patrol_brique_v14_pre42_seed13    0/30 claiming episodes   occupancy 0.059

  **The mute policy occupies twenty times more than the vocal one** — and 0.059
  is the highest root occupancy measured anywhere in this investigation. #54
  argued exactly this shape from `patrol_brique_v5` at 0.011–0.021; this is the
  same finding at triple the effect, from a manipulation rather than an
  observation. Occupancy does not order policies by whether they report.

  **What is left of the mechanism, stated at its true strength.** Within a single
  policy the per-episode split still holds everywhere it has been measured —
  claiming episodes occupy at least as much as silent ones, `v13` 0.003 vs 0.000,
  `v6-latest` 0.041 vs 0.000, and the squad arms likewise. Between policies it
  predicts nothing. And the surviving cross-policy statement — *a root that never
  enters the objective radius never files a truthful claim* — is close to
  definitional for a SEIZE mission, not an explanation. **The occupancy account
  is retired as a mechanism; it survives only as a within-policy regularity.**

  **The one thing these arms establish positively, and it is worth more than
  what they retire.** `v13` and `v14` differ in **the seed and nothing else** —
  same tree, same `lr`, same price, same budget, same scenario — and one reports
  at 0.825 while the other is absolutely mute. **Seed alone flips the reporting
  channel in `patrol_brique`.** That makes the third factor #54 asked us to find
  at least partly an optimisation-path property, not an economic or structural
  one, and it independently supports the lr/seed pair its own v5-vs-v6
  comparison was confounded by. It also means **every single-seed claim about
  this channel in this scenario is worthless**, including several above, and that
  v1.21 cannot be decided by one arm per cell.

- **2026-08-15 — RETRACTION: the chart block does not control the reporting
  channel. After 17 runs the effect is not distinguishable from zero, and the
  "clean paired flip" I reported is cancelled by its mirror image.** The
  decisive cell finished at five seeds, all on one tree (`742b28a6`), `rdb=1.0`,
  no overrides, N=100 on both checkpoints:

        seed   block REMOVED                    block PRESENT
         12    v13_pre42   0.825  REPORTS       patrol_brique_v7           0.000  mute
         13    v14_pre42   0.000  mute          patrol_brique_v10_seed13   0.000  mute
         14    v15_pre42   0.000  mute          patrol_brique_v11_seed14   0.878  REPORTS
         15    v16_pre42   0.000  mute          patrol_brique_v12_seed15   0.000  mute
         16    v17_pre42   0.794  REPORTS       —

                          patrol_brique   2/5 vs 1/4   Fisher p = 1.000
                          squad           4/4 vs 2/4   Fisher p = 0.429
                          POOLED          6/9 vs 3/8   Fisher p = 0.347

  **Seed 14 is the anti-flip.** It reports *with* the block (0.878) and is mute
  *without* it (0.000) — the exact mirror of seed 12, which I reported earlier
  today as evidence the block was implicated. Two paired flips in opposite
  directions are no evidence at all. Success is unaffected everywhere (0.95–1.00
  in all nine patrol_brique runs), so this was never about the mission.

  **What this retracts.** The 2026-08-14 entry called the squad result "the first
  version of this quantity that agrees with itself"; the 2026-08-15 entry called
  seed 12 "a clean paired flip … the block IS implicated in `patrol_brique`".
  Both were built on two- and four-seed patterns, and squad's own 4/4 vs 2/4 was
  **p = 0.429 — never significant, as that entry itself said before leaning on it
  anyway.** Pooled over both scenarios and 17 runs the block's effect is
  **p = 0.347**. It is not established, and the honest description of every
  block-vs-no-block table in this log is *a pattern in small n*.

  **What actually survives, and it is a bigger finding than the one it
  replaces.** The reporting channel is **seed-determined under every
  configuration tested**: 3 of 9 patrol_brique runs report and 6 of 8 squad runs
  do, spread across both trees and three prices, with success pinned near 1.00
  throughout. Whether a commander ever learns to file a truthful completion is
  currently a property of the optimisation path, not of the reward, the chart, or
  the scenario. **`closed_on_root_report_rate`'s 0.5 floor is therefore rejecting
  runs on a coin-flip**, which is exactly what happened to `patrol_brique_v7` and
  what blocked the v1.20b fleet.

  **⇒ v1.21 is not a tuning cycle, and should not be planned as one.** Three
  routes, none taken here because each changes what the project claims:
  (1) treat it as exploration / credit assignment — the DONE claim is a rare,
  precisely-timed act whose reward arrives once, and nothing in training shapes
  the approach to it; (2) accept the variance and **declare** a selection policy
  — train k seeds per scenario and ship one that reports, which is seed-shopping
  and must be stated in the README or it is exactly the overstatement
  `publish_audit.py` exists to catch; (3) change the gate, on the argument that a
  silent commander that wins is a real policy and the floor encodes a preference
  that was never measured to be achievable. **The measurement that would decide
  between (1) and (2) is whether the reporting/mute split is stable under a
  restart at fixed seed** — if it is, it is a basin the optimiser falls into and
  (1) is tractable; if a rerun of the same seed lands differently, only (2) and
  (3) remain.

- **2026-08-15 — route (2) would have to be declared PER SCENARIO: a reporting
  seed does not transfer (refs assurance #55).** The route above says "train k
  seeds and ship one that reports", and it only pays for itself if a seed is
  *globally* good. It is not. The squad and `patrol_brique` cells already share
  seeds at fixed arms, so this needed no new run —
  `scripts/reporting_channel.py` labels every run on disk from
  ``done_claim_episodes_root`` at BOTH published checkpoints, re-derives each
  arm from `config.json`/`economics.json`, and pairs the two scenarios at a
  fixed arm:

        arm                     seed   squad       patrol_brique
        rdb 1, chart absent      12    REPORTING   REPORTING
        rdb 1, chart absent      13    REPORTING   mute
        rdb 1, chart absent      14    REPORTING   mute
        rdb 1, chart absent      15    REPORTING   mute
        rdb 1, chart present     12    REPORTING   mute
        rdb 1, chart present     13    REPORTING   mute
        rdb 1, chart present     15    mute        mute
        rdb 2, chart present     12    REPORTING   mute

        rdb=1.0 only   agreement 2 of 7, 1.71 expected   McNemar p = 0.0625
        every arm      agreement 2 of 8, 1.75 expected   McNemar p = 0.0312

  **Agreement is at chance.** A seed carries no propensity to report that
  survives a change of scenario, so a declared seed list is a per-scenario
  artifact and route (2) costs a fleet per scenario rather than a fleet. #55
  reported the `rdb=1.0` reading and this reproduces it exactly, run for run; the
  eighth pair is the `rdb=2.0` cell (`squad_v19_rdb2` vs `patrol_brique_v9_rdb2`,
  same seed, same tree), which #55's scope excluded and which is the cleanest
  pair on the table. **The direction is not the claim** — six unanimous
  discordant pairs is p = 0.0312, but seeds 12–15 are four seeds and this is 8
  pairs, not 80. It is enough to say a global seed list is unsupported; it is not
  enough to rank the scenarios.

  **Ten runs are dropped rather than resolved, and that is the other half of the
  method.** `squad_v10c` claims in 18 of 100 episodes at `ckpt_best` and 0 of 100
  at `ckpt_latest`; no label describes that policy, and choosing the checkpoint
  that suits the argument is how a 2-of-4 becomes a 4-of-4.

  **The label is a claim count, never `closed_on_root_report_rate`, and the
  script prints where the two disagree.** On the corpus as it stands that check
  fires on 20 checkpoints — the whole defend family, which reads 0.97–1.00 on
  **zero** root claims because a continuous-posture root closes the window with a
  SITREP. #55 measured the same hazard in the other direction: a root SITREP
  landing on the ENDEX step enters the numerator, so `patrol_brique` arms with no
  claims at all read 0.020–0.104 and a 0.05 rate-cut calls them reporting. Both
  directions are pinned in `tests/test_reporting_channel.py`. This does not touch
  the v1.20 gate, whose 0.5 floor is far above that artifact — the gate's problem
  is the one the retraction named, that the quantity is bimodal by seed.

- **2026-08-15 — the `rdb3_seeds` campaign can reject on 1 of its 64 outcomes,
  and it is the outcome its own evidence argues against (refs assurance #56).
  Read the result descriptively; do NOT report a Fisher p from it as a null.**
  The six-run design asks the right question and the disjunction is the right
  decision rule, but the two readings it could be given are bounded before any
  run lands. `scripts/design_power.py` enumerates every outcome the design can
  produce and scores both:

        design: 5 seeds, 6 pending cells, 64 possible outcomes, alpha = 0.05
          unpaired (Fisher)   ceiling p = 0.0476   rejects on 1 of 64 outcomes
          paired  (McNemar)   ceiling p = 0.1250   CANNOT REJECT

  The ceiling is the *smallest p the design can attain*, best case, over every
  outcome including perfect separation. So the paired reading cannot reject at
  five pairs at all (0.0625 unconditionally; 0.125 here, because seed 14 already
  reports at `rdb=1.0` and a concordant pair is not evidence), and the unpaired
  reading rejects on exactly one outcome: `rdb=3.0` reporting at all five seeds
  **and** `patrol_brique_v23` coming back mute. Seed 16's only `patrol_brique`
  observation reports (`v17_pre42`, 0.794), so the branch that would vindicate
  the gate is the one the design cannot supply.

  **The other branch needs no test and that is the point.** "3.0 splits across
  seeds too" is settled descriptively by the first mute `rdb=3.0` cell — one cell,
  no inference. The jobs file's header names a "paired 5-vs-5 Fisher test"; that
  phrase is what this entry corrects, because a p = 0.17 out of a design that
  could never have gone below 0.17 is not a null result and will be read as one.
  The header was left alone rather than rewritten: the queue reads that file line
  by line while it runs.

  **Sizing, for the next one.** Eight seeds is where both readings become
  capable — `scripts/design_power.py --size 5 6 7 8` gives 0.0476/0.1250 at five,
  0.0210/0.0625 at seven, 0.0070/0.0312 at eight, for 6/10/12 new runs. The
  extension to eight (`scripts/campaigns/patrol_brique_incumbent_seeds_ext.jobs`,
  six more runs) is chained behind the running queue. The Fisher column is not
  monotone — six seeds is worse than five — because the assumed comparison count
  rounds 1 → 2 there; that is an artifact of the assumption, not of the design.

  **A rank test is not the cheap way out.** `closed_on_root_report_rate` reads a
  floor of 0.020–0.104 on a wholly mute arm, so its low mode is instrument
  artifact and a Mann-Whitney over it would be ranking noise. The quantity is
  binary or it is nothing — which is why the label is a claim count (see the
  entry above).

  **What this does not say.** Nothing here asks for the campaign to be stopped.
  Five new `rdb=3.0` seeds on one tree is an asset whatever the read-out can
  certify, and the descriptive branch is exactly the one the disjunction was
  built to settle.

- **2026-08-15 — ANSWERED: the incumbent price is a seed lottery too. `rdb=3.0`
  fails the gate at seed 13, 0.000 at BOTH checkpoints, N=100.** The question
  that blocked v1.20b was whether `patrol_brique_v6`'s 0.808 is a property of
  the incumbent config or a draw. It is a draw.

        patrol_brique, rdb=3.0 (the incumbent price), N=100, current tree
          seed 12  v18_rdb3_seed12   final 0.867 (92/100 claim-eps)  REPORTS
          seed 12  v8_rdb3           final 0.867 (92/100)            REPORTS
          seed 13  v19_rdb3_seed13   final 0.000 ( 0/100)            MUTE
                                     best  0.000 (40/100 claim-eps)  gate FAIL

  Seed 13's final policy succeeds at 0.99 and never once claims in 100 episodes.
  This is the same gate, at the same N, that failed `patrol_brique_v7` and
  stopped the fleet — so the price that was supposed to be the fix reproduces
  the failure it was meant to explain.

  **The headline does not rest on a labelling choice, and that is deliberate.**
  `v19` is `SPLIT` under the both-checkpoint claim-share label (best claims in
  40 episodes while closing none of them — #55's rate/claim divergence, in the
  one cell where it decides something) and `mute` under a final-only label. It
  does not matter: `closed_on_root_report_rate` reads **0.000 at both
  checkpoints**, so under the project's own gate the incumbent price does not
  report at seed 13 on any reading.

  **Corroboration that was already on disk.** `patrol_brique_v5` (seed 3,
  `rdb=3.0`) is mute at both checkpoints. An older tree, so it is support and
  not proof — but the counterexample did not need a campaign to find.

  **And the incumbent itself is SPLIT.** `patrol_brique_v6`'s published 0.808 is
  its FINAL policy; its `ckpt_best` is mute at 0.000 (1/100 claim-episodes). The
  incumbent the gate protects is half mute by the gate's own measure.

  **WHAT THIS SETTLES — v1.21 IS A GATE CYCLE, NOT A TUNING CYCLE.** Both prices
  flip on seed (`rdb=1.0` reports at 1 of 4 seeds; `rdb=3.0` at 2 of 3 measured
  cells, and mute at one). A per-run pass/fail gate over a quantity that is
  bimodal in the SEED scores the draw, not the policy — and it is what is
  currently holding a fleet whose seven other members match their incumbents.
  Whether `closed_on_root_report_rate` stops being a per-run gate, becomes a
  fleet-level or median-over-k criterion, or keeps its role with a declared
  per-scenario seed policy, is a decision about the project's claims and is the
  owner's.

  **THE PRE-REGISTERED STOPPING RULE FIRES: no test is named.** `8518f33` fixed
  the read-out before any cell was scored — "NO TEST AT ALL if any `rdb=3.0`
  cell comes back mute, because '3.0 splits too' is settled descriptively by the
  first mute cell and a Fisher p from this design would not be a null result."
  That cell arrived second. No McNemar, no Fisher, no p-value over these arms.

  **The campaign keeps running, re-purposed and re-scoped.** Eighteen runs
  remain (~15h, zero tokens). They are no longer a hypothesis test — they are a
  DISTRIBUTION ESTIMATE: how often does each price report, across twelve seeds?
  A gate redesign cannot pick a threshold or a k without that, so the runs are
  worth more to v1.21 than they were to the test they were launched for. The
  v1.21 decision no longer waits on them.

  **AMENDMENT to the labelling, made after seeing the corpus and recorded as
  such.** `8518f33` pre-registered labels from both checkpoints. That is
  circular: `train.py::best_save_gate` selects `ckpt_best` **on the reporting
  channel** (v1.20 — a reporting window lexicographically supersedes a mute one
  whatever the success numbers say), so a both-checkpoint label partly measures
  the selection rule rather than the price. Final-only is primary from here;
  both-checkpoint stays as a reported sensitivity. This changes no conclusion
  above — stated plainly because amending a pre-registration after seeing data
  is exactly the move that needs to be visible.

  Also corrected: `v19`'s `ckpt_best` scoring 0.17 success against its final
  policy's 0.99 is NOT a checkpoint-selection bug. It is `best_save_gate`
  working as designed — reporting beats success, lexicographically. The side
  effect is real (a `ckpt_best` that fails 4 episodes in 5) and belongs to the
  same v1.21 gate question, but it is a consequence of a decision, not a defect.

- **2026-08-15 — that `ckpt_best` was decided at iteration 25 of 2930, on a
  window at 2% success, and it was the run's ONLY save (refs assurance #57).**
  The entry above is right that this is the lexicographic rule working as
  designed, and stops one step short of the mechanism. `best_save_gate` reads
  the *rolling training window*, not either evaluation, so neither of #57's two
  columns can settle anything about it. `scripts/checkpoint_selection.py`
  replays that window off each run's own `metrics.csv` — calling the shipped
  `best_save_gate`, never a copy — and checks the answer against the `iteration`
  stamped inside `ckpt_best.pt`. **91 of 91 replayable runs agree**, so what
  follows is a reconstruction and not an inference:

        patrol_brique_v19_rdb3_seed13, 2930 iterations, 31,233 episodes
          ckpt_best written  iteration   25 /    25,600 steps  (0.9% of the run)
            that window      success 0.020   closed-on-root 0.500   1 save, ever
          last window        success 1.000   closed-on-root 0.000
          with a success floor before the reporting comparison, ANY of 0.25/0.50/
          0.75/0.90:         iteration  550 /   563,200 steps  success 1.000

  So the shipped `ckpt_best` is a **25,600-step policy out of 3,000,320**,
  written at the first iteration the D4 turnover check allowed and never
  replaced across the remaining 2,905 iterations.

  **The mechanism is a denominator, not the emitted/admitted split #57 names.**
  The gate's reporting input is `root_report_close_rolling`, i.e.
  `env.root_close_step is not None`, and `root_close_step` is set only by a
  *truthful* close — so the input is already truth-conditioned and reading
  admitted claims would not change it. The 92 rejected claims at `ckpt_best` are
  real, visible on the net, and selected nothing. Three lines do the selecting:

  1. `recent_root_closed` is appended once per episode **that sent an ENDEX**,
     and `cohort_env` transmits ENDEX only in the success branch. The reporting
     rate is therefore conditioned on winning, and its sample is thinnest
     exactly where success is worst. `v19`'s window read **0.500 — the floor,
     exactly** — which is what a handful of wins produces.
  2. The two deques are **misaligned and only one has a turnover requirement**.
     D4's check is `episodes_seen >= window` on `recent_outcomes`;
     `recent_root_closed` has none, so the first eligible save can compare a
     100-episode success window against a reporting window holding a few.
  3. `best_was_reporting` is **absorbing**. Once set, no mute window may take the
     best back at any success level — so one thin early window is final.

  **Fleet-wide, this is one run wide, and that is the useful part.** Of 91
  replayable runs (38 carry the reporting axis; 53 predate it; 69 more predate
  the `n_episodes` column and cannot be replayed at all, which the reader reports
  as NOT REPLAYABLE rather than as "never saved"), **exactly one selects a
  different checkpoint under a 0.5 success floor** — `v19`, recovering +0.980
  rolling success. Every other run's `ckpt_best` sits where a floor would leave
  it. `patrol_brique_v18_rdb3_seed12`, the same price at the neighbouring seed,
  chose iteration 876 at 30% of the run on a 100%-success window.

  **Why nobody saw it.** No digest printed where `ckpt_best` came from.
  `run_report.py`'s `stability` compares the best rolling window to the last one
  — both ~1.0 here — so it called `v19` converged and PUBLISHABLE, correctly, of
  a different object. The digest now prints the selecting iteration, its share of
  the run, and that window's success and closed-on-root, and flags a selection
  more than 10 points below the run's final window.

  **No test is named over these arms**, per `8518f33`'s stopping rule: every
  number above is a replay of runs on disk, one run against itself.

  **What is NOT done, and why.** The `cohort/` change — a success floor under
  the reporting comparison, or a minimum denominator under
  `root_report_close_rolling`, or dropping the absorbing flag — is a training
  default and the owner's decision, and it is **blocked** regardless: a campaign
  is live and `train.py` imports the tree as it exists at each job's launch. It
  belongs in the same window as the v1.21 gate decision, since both are about the
  same conflation at two levels. `cohort/metrics.py`'s
  `closed_on_root_report_rate` docstring is on the same blocked list (it should
  note the non-zero floor on a completable root, from #55).
