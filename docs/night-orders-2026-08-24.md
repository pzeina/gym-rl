# Night orders — 2026-08-24 (AUTO mode, owner asleep)

The owner retired at ~00:30 handing over the watch, no focus given. This file
is the queue and the guardrails; the self-paced loop reads it at every wakeup.

## Authority tonight

Pre-authorised (standing, owner 2026-08-18 "do not stay doing nothing — TRAIN!
and experiment"): launch training runs and campaigns including confirm seeds,
every zero-token measurement (probes, evals, oracle, report digests), commit
finished work (full pytest + ruff green, one commit per unit, repo trailers),
push `multi-agent-dev`.

NOT tonight, regardless of findings: merge or tag `main`; anything destructive;
publishing a MISS over an incumbent; **design decisions** — reward structure,
vocabulary, scenario semantics, masks/enforcement, rewording owner-decided
claims. In particular: **whether `squad_range_control` carries a
`time_penalty` override is the owner's open decision and is NOT settled
tonight.** Every run below is an arm. Nothing edits `config.py`.

Honest-DoD: one retrain + one diagnosed adjustment per miss, then document and
stop that thread. Digests only — never raw logs, `metrics.csv`, or checkpoints
into context.

## The thread tonight

`squad_range_control` seed 14 captures at the default time price (0.00 success,
every episode to the clock). `time_penalty=-0.03` removes the capture at all
four seeds. The casualty worry it raised is refuted (pooled 34/400 vs 46/400,
p = 0.19). What survives is **churn**: priced retasks rise at 4/4 seeds when
the price is on, and two of four breach the pre-registered clause-4 health bar
of ≤ 2.0 at N=100.

Two things are already known and must not be re-run:
- **Churn is not dose-responsive in the time price.** `-0.02` is strictly worse
  than `-0.03`: identical churn at seed 13 (3.12 vs 2.97, p = 0.88), 6.6× worse
  at seed 14 (3.68 vs 0.56, p < 1e-4), and at seed 14 the `-0.02` policy stops
  reporting entirely (sees 3.70 enemies/episode, reports 0.00).
- **Churn is economically rational at `-0.03`.** Retasks are already priced at
  `order_retask_cost_base = -0.5`. At seed 13 the policy spends 2.72 extra
  retasks (−1.36) to save 61.4 steps (+1.84): net **+0.48**. Break-even sits at
  a retask cost of **0.677**, so the shipped −0.5 makes buying time with orders
  correctly-priced income.

Tonight tests the mechanism that arithmetic predicts.

## The queue

Gate every read on its landing. Bookkeeping at every landing first (same-config
draws declared in `seed_spread`, declared ⇒ tracked, artifacts committed, suite
green, push), then the gated read, then any launch the rules call for.

1. **`squad_range_control_retaskcost_v1_seed14` lands (~00:37)** and
   **`..._seed13` (~00:47)** — both at `time_penalty=-0.03` plus
   `order_retask_cost_base=-1.0`, i.e. the retask cost moved past the 0.677
   break-even. Per landing: `run_report.py` digest, then N=100 on the final
   policy at the arm's fixed protocol (`--episodes 100 --seed 123`, sampling,
   written to `behavior_final_n100.json`, never over the committed N=20).
   Read against `squad_range_control_timecost_v1_seed13/14`.

   **Pre-registered decision rules** (fixed now, before the runs land):

   - **SEPARATES** — seed 13's `retasks_priced_per_episode` < 2.0 at N=100 AND
     seed 14 does not recapture (success ≥ 0.50). The retask cost is the knob
     that governs churn and the capture escape survives it. Next pre-authorised
     step, launched immediately: **seeds 12 and 15 at the same two overrides**,
     giving the matched four-seed row the price decision needs.
   - **WALL** — seed 14 recaptures (success < 0.50). The −0.5 retask cost is
     load-bearing for escaping D4: making orders expensive re-freezes the
     policy. The ONE diagnosed adjustment in scope, **named now so it cannot be
     invented later**: `order_retask_cost_base=-0.75` at seed 14 only — between
     the 0.677 break-even and −1.0. If that recaptures too, `-0.03` alone
     stands, document the miss and stop this thread.
   - **CEILING** — seed 13's churn stays ≥ 2.0 despite the retask cost
     doubling, and seed 14 converges. Then churn responds to neither knob and
     is a basin property rather than an economic one. No second adjustment
     overnight: write it up, the knob goes back to the owner.
   - **SPLIT** (seed 13 separates, seed 14 walls, or the reverse) — treat as
     WALL for seed 14 and SEPARATES for seed 13: launch the −0.75 seed-14 arm
     and the seeds 12/15 pair, and say plainly in the morning ledger that the
     configuration is seed-dependent.

2. **If the seeds 12/15 pair is launched, it lands ~02:00 and ~03:10** —
   digest, N=100 each, then the four-seed churn row for the morning. Draw no
   verdict on whether the price ships; that is the owner's.

## Idle-time measurements (zero tokens, detached, sentinel-terminated)

- `logs/oracle_squad_screen.log` — `scripts/oracle_probe.py` on the sealed
  `squad_screen_v14`. Closes the standing gap in docs/next-cycles.md §332:
  a 24% root-death rate on a scenario whose entire doctrine is *observe
  without engaging*. Weapons-tight is mask-enforced, so this is exposure, not
  indiscipline — the oracle says which. **Read-only: no reward change follows
  tonight.** Ends `ORACLE-SQUAD-SCREEN-DONE`.
- `logs/d4_ledger_new_arms.log` — `scripts/d4_ledger_probe.py` over the four
  new price arms (both `timecost02`, both `retaskcost`). Extends the idle-income
  ledger to the new price points and checks the −0.02 arms against the capture
  line the −0.03 arms cleared. Ends `D4-LEDGER-NEW-ARMS-DONE`.

## Morning

ROADMAP handoff updated with the night's ledger — what landed, what was read,
what was launched, every commit, every miss with its diagnosis. Commit and
push. Note PUBLISH PENDING boards for the owner's `/boards`. One
PushNotification with the outcome the owner would act on.

## Queue state (updated as the night runs)

- 00:30 — orders written; both `retaskcost` arms in flight (83% / 92%); idle
  probes launching; monitor arming.
- 00:32 — **idle probe 1 is a MISS, diagnosed.** The `squad_screen` oracle pass
  cannot run: `squad_screen_v14` is a 220-dim checkpoint and the current tree
  is 351-dim. `baseline.py` confirms it is not that member alone — **all nine
  sealed members fail to load under the current spaces**, the documented and
  explicitly authorized acoustics spaces break (ROADMAP 15:31, OBS_DIM
  220 → 328, since moved to 351). Pre-existing, not caused tonight, and not
  fixable tonight: retraining the sealed fleet is the owner's call.
  The adjustment in scope, taken: **`squad_screen_v18_seed12` launched** on the
  current tree, shipped defaults, no overrides, 2.5M steps — the same
  configuration as the sealed member. It unblocks the oracle pass and gives the
  owner the first post-break data point on a fleet scenario, which the
  fleet-retrain decision will need. It is one scenario, not a fleet retrain.
- 00:32 — **idle probe 2 landed.** Ledger over the four new price arms, 12
  episodes each: `timecost02` seed13 +0.349/agent-step (ep len 154), seed14
  +0.360 (177); `timecost` seed13 +0.721 (101), seed14 +0.564 (102). All four
  are 85–175× above the +0.0041 capture line, so no new arm is anywhere near
  idling being profitable. The two `-0.03` arms earn roughly double the two
  `-0.02` arms on shorter episodes — the same ordering the N=100 behaviour
  suite gave, now from the reward stream instead.
- 00:32 — note for the morning: `logs/*.log` is hook-denied to `tail` and to
  single-path `grep` in this session, so detached probe output must be written
  to a readable path (JSON under `runs/`) rather than only to a log.
- 00:52 — **Queue item 1 resolves: SEPARATES.** Both `retaskcost` arms landed
  and both N=100 evals are down. Criterion was seed 13 churn < 2.0 AND seed 14
  not recapturing; both hold. Seeds 12 and 15 launched at the same two
  overrides per the rule.

  | N=100 final | success | retasksP | ep len | closed-on-root | hdr |
  |---|---|---|---|---|---|
  | s13 default | 0.86 | 0.25 | 159.2 | 0.000 | 0.380 |
  | s13 −0.03 | 0.93 | 2.97 | 97.8 | 0.000 | 0.050 |
  | **s13 −0.03/−1.0** | **0.97** | **0.14** | **73.5** | **0.938** | 0.170 |
  | s14 default | 0.00 | 0.16 | 450.0 | — | 0.020 |
  | s14 −0.03 | 0.96 | 0.56 | 102.3 | 0.000 | 0.030 |
  | s14 −0.03/−1.0 | 0.94 | 1.95 | 111.0 | 0.000 | 0.020 |

  **The two seeds move in opposite directions, both significantly.** Seed 13's
  churn collapses 2.97 → 0.14 (p < 1e-4, 21×) on shorter episodes and higher
  success; seed 14's *rises* 0.56 → 1.95 (p < 1e-4) on slightly longer
  episodes. So the retask cost does not govern churn either — it moves it, hard,
  in a seed-dependent direction. The economic story that motivated this arm
  (churn as rational time-trading priced against break-even 0.677) predicted a
  uniform fall and did not get one. Seeds 12 and 15 will say which behaviour is
  typical; the four-seed row is the deliverable, not a verdict.

  **The result the owner will actually want**: seed 13 reads
  `closed_on_root_report_rate` **0.938** against 0.000 for every other member of
  this arm (p ≈ 4e-50) — the shipped `>= 0.5` gate has never once passed in
  `squad_range_control` and here it passes emphatically, with report recall
  0.529 → 0.883. **It is one seed**: seed 14 at the identical configuration
  reads 0.000. Not a property of the configuration until 12/15 say so. Its cost
  is visible too — human death 0.050 → 0.170 (p = 0.012) and stacked 0.224 →
  0.525, still inside the 0.70 gate.
- 00:52 — **bookkeeping blocked, deliberately.** `squad_screen_v18_seed12` is a
  same-config draw, so `test_the_shipped_record_holds_no_draw_outside_the_declared_blocks`
  fails until `BASELINE.json` declares it — but declaring it now fails the
  companion gate, which requires a tracked identity-bearing checkpoint, and a
  mid-flight `ckpt_latest.pt` is not the run's identity. So the declaration and
  every artifact commit wait for that run to land (~01:30) and go in together
  with the suite green. The `retaskcost` results are on disk and lose nothing by
  waiting. My earlier reformat of `BASELINE.json` (indent churn across 165
  lines) was reverted rather than committed.
- 01:22 — **what the seed-13 root close actually is** (per-episode counters,
  N=100, zero tokens). It is not a scoring artifact: seed 13 at −0.03/−1.0 is
  the only policy in the arm where the **root files a DONE claim at all**
  (`done_reports_root` 1.77/episode against 0.00 for the other three arms), and
  the mission closes on that claim in 91% of episodes (`endex_on_root_report`
  0.91 vs 0.00 everywhere else). `endex_sent` is ~0.95 in all four arms, so the
  missions were always ending — what changed is *who* ends them.
  Two honest qualifications: the root files eagerly, and 0.86 of its 1.77 claims
  are rejected (~51% precision on root claims); and routine root chatter falls
  at the same time (`root_sitreps` 5.43 → 0.33), so the root talks less overall
  while finally saying the one thing the gate asks for. `succession_events`
  0.29 is consistent with the raised human-death rate — sometimes it is a
  successor closing, not the original root.
- 01:42 — **the standing `squad_screen` oracle gap is CLOSED** (next-cycles
  §332), on `squad_screen_v18_seed12` at 30 episodes, seeds 500–529. The
  hypothesis in next-cycles was right: **it is exposure, not indiscipline.**
  - **94.1% of deaths happen out of cover**, and cover occupancy under threat is
    0.137 for the team, 0.068 for the human. Friendly deaths in the open are
    1.00/episode against 0.13 at the objective.
  - Deaths by mission-at-death: OBSERVE 0.382 (92% of them in the open), SCREEN
    0.324 (100% in the open), COVER 0.176, HOLD 0.088. The agents are dying
    while doing exactly the doctrinally correct thing, standing up.
  - Human death rate 0.200 here against the sealed member's 0.24 — the exposure
    survived the spaces break, so it is a property of the scenario rather than
    of one checkpoint.
  - Fire rate under threat is 0.660 team-wide on a scenario whose doctrine is
    observe-without-engaging, but weapons-tight is mask-enforced, so this is
    permitted return fire, not a discipline failure. Outcomes are 0.967 success
    / 0.033 defeat: the scenario is not degenerate, it is won expensively.
  - **Read-only, as the orders said.** Cover occupancy is priced by the reward
    config, so acting on this is a design decision and belongs to the owner. The
    fact sheet is at `runs/squad_screen_v18_seed12/oracle_night.txt`, raw
    counters beside it as `oracle_night.json`.
- 02:08 — **a tempting story about the root close, tested and rejected.** The
  obvious reading of seed 13 is "fewer retasks frees the root to report". The
  record does not support it. Across the six N=100 arms there is no monotone
  relation between priced retasks and `closed_on_root_report_rate`: s14 at −0.03
  has the second-lowest churn in the whole set (0.56) and closes on root 0.000,
  while s15 at −0.03 has the highest (3.71) and closes 0.340. Low churn is
  neither sufficient nor necessary. Whatever produced the root close at seed 13,
  it is not the churn collapse that came with it.
  What the root close *is*, is stable within its own run rather than a
  last-checkpoint fluke: 0.500 at `ckpt_best` (N=20), 0.947 at the final policy
  (N=20), 0.938 at N=100 — rising and consistent across two checkpoints and two
  sample sizes.
- 02:26 — **Queue item 2 resolves, and it reframes the seed-13 finding.** Seeds
  12 and 15 landed (97%/98%, gaps 3 and 2 pts) and their N=100 evals are down.
  The four-seed row at `time_penalty=-0.03` vs `-0.03` + `order_retask_cost_base=-1.0`:

  | seed | success | retasksP | ep len | closed-on-root | hdr | report recall | stacked |
  |---|---|---|---|---|---|---|---|
  | 12 −0.03 | 0.97 | 1.47 | 96.4 | 0.000 | 0.020 | 0.477 | 0.246 |
  | **12 −1.0** | 0.97 | **0.05** | 77.1 | **0.907** | 0.220 | 0.889 | 0.503 |
  | 13 −0.03 | 0.93 | 2.97 | 97.8 | 0.000 | 0.050 | 0.529 | 0.224 |
  | **13 −1.0** | 0.97 | **0.14** | 73.5 | **0.938** | 0.170 | 0.883 | 0.525 |
  | 14 −0.03 | 0.96 | 0.56 | 102.3 | 0.000 | 0.030 | 0.654 | 0.244 |
  | 14 −1.0 | 0.94 | **1.95** | 111.0 | **0.000** | 0.020 | 0.708 | 0.206 |
  | 15 −0.03 | 0.97 | 3.71 | 93.5 | 0.340 | 0.240 | 0.470 | 0.204 |
  | **15 −1.0** | 0.99 | **0.11** | 68.5 | **0.919** | 0.110 | 0.938 | 0.559 |

  **Seed 13 was not special — seed 14 is.** At three of four seeds the doubled
  retask cost collapses churn by 10–30× (1.47→0.05, 2.97→0.14, 3.71→0.11, each
  p < 1e-4), shortens episodes, raises report recall from ~0.49 to ~0.90, and
  **passes `closed_on_root_report_rate`** at 0.907/0.938/0.919 — the shipped
  `>= 0.5` gate that has never once passed in this scenario. Pooled, root closes
  go 34/400 → 277/400 (p ≈ 8e-76). Seed 14 alone moves the other way on churn
  (0.56 → 1.95, p < 1e-4) and stays at 0.000 root closes.
  My 02:08 note stands but needs narrowing: low churn alone still does not
  produce the root close (s14 at −0.03 has churn 0.56 and closes 0.000). What is
  true is that *within* the −1.0 configuration the churn collapse and the root
  close co-occur at 4/4 seeds, seed 14 resisting both.
  **The honest cost, which is heterogeneous and must not be averaged away**:
  human death is pooled 34/400 → 52/400, p = 0.052 — borderline — but the seeds
  disagree in direction. It rises at 12 (0.020 → 0.220, p < 1e-4) and 13 (0.050
  → 0.170, p = 0.012), *falls* at 15 (0.240 → 0.110, p = 0.025) and is flat at
  14. Stacked rises to 0.50–0.56 at the three closing seeds, still inside the
  0.70 gate.
- 02:26 — **launched beyond the written queue, deliberately**:
  `squad_range_control_retaskcost_v1_seed16` and `_seed17` at the same two
  overrides. The queue stopped at four seeds; the one question the row leaves
  open is whether seed 14 is an exception or whether 3-of-4 is the rate, and
  that needs only the −1.0 arm at fresh seeds — the −0.03 control is already
  0/100, 0/100, 0/100, 34/100 and its effect size is not in doubt. Two runs
  rather than four, for that reason. Confirm seeds are pre-authorised.
  **No verdict is drawn**: whether `squad_range_control` carries either override
  is the owner's decision, and this is now a two-knob decision, not one.
- 02:55 — **what separates the closing seeds, from data already on disk.** The
  eight N=100 arms split cleanly in two, and seed 14 at −1.0 sits on the wrong
  side of the split:

  | arm | stacked | closed-on-root | ep len |
  |---|---|---|---|
  | s15 −0.03 | 0.204 | 0.340 | 93.5 |
  | **s14 −1.0** | **0.206** | **0.000** | **111.0** |
  | s13 −0.03 | 0.224 | 0.000 | 97.8 |
  | s14 −0.03 | 0.244 | 0.000 | 102.3 |
  | s12 −0.03 | 0.246 | 0.000 | 96.4 |
  | s12 −1.0 | 0.503 | 0.907 | 77.1 |
  | s13 −1.0 | 0.525 | 0.938 | 73.5 |
  | s15 −1.0 | 0.559 | 0.919 | 68.5 |

  Every arm above stacked 0.50 closes on root; every arm below 0.25 does not
  (except s15 −0.03 at 0.340). The three closing arms are also the three
  shortest. Spearman over the eight arms: stacked vs closed-on-root rho = +0.660
  (leave-one-out [+0.473, +0.867]), episode length vs closed-on-root
  rho = −0.913 (leave-one-out [−0.927, −0.867]). Neither range straddles zero,
  so by `jackknife_rho`'s own criterion neither relation is carried by a single
  arm. **Seed 14 did not fail to close so much as it never made the transition**
  — the −1.0 cost moved three seeds into a bunched-and-fast regime and left seed
  14 in the old one.
  This is association across eight arms, not a mechanism, and it is the kind of
  thing seeds 16/17 can break. It is also the tradeoff to put in front of the
  owner: `stacked_rate` is a shipped gate with a 0.70 bound, and the regime that
  finally closes on root sits at 0.50–0.56 — inside the gate, but at roughly
  double the arm's historical bunching.
  (Correction for the record: I first read `jackknife_rho`'s return as
  (rho, standard error) and took its second value as a negative SE, i.e. as a
  tooling bug. It returns (min, max) over leave-one-out. No defect; my misread.)
- 03:25 — **pre-registered, before seeds 16/17 land** (they are at 82%, ~15 min
  out; nothing below is written with knowledge of their N=100 numbers). The
  02:55 regime split says the −1.0 configuration moves a seed into a
  bunched-and-fast regime, and that closing on root is a property of that regime
  rather than of the seed. If that is right, then for each of seeds 16 and 17:
  - a seed with `closed_on_root_report_rate` ≥ 0.5 will ALSO show
    `stacked_rate` ≥ 0.45 and `episode_length_mean` ≤ 85;
  - a seed that stays at ≈ 0.000 will show `stacked_rate` ≤ 0.30 and
    `episode_length_mean` ≥ 95, the way seed 14 does.
  A seed that closes on root while bunching like seed 14 (stacked ~0.2, long
  episodes), or one that bunches and stays mute, **breaks the split** and the
  02:55 note should be struck rather than reworded.
  Rate prediction: if 3-of-4 is the true rate, the likeliest outcome is one of
  the two closing; two closing or none are both unremarkable at n=2. **No
  outcome here settles whether the override ships — that stays the owner's.**
- 03:27 — **a concrete look at the eager-root caveat**, from
  `runs/squad_range_control_retaskcost_v1_seed13/eval_transcript.txt` (the run's
  own single-episode eval, not the N=100 protocol — one episode, illustrative
  only). It happens to show the failure mode rather than the headline:

  ```
  [t= 68] ALL STATIONS: SL1 IS DOWN. OUT.
  [t= 68] ALL STATIONS, THIS IS TL1: SL1 IS DOWN. I AM ASSUMING COMMAND. OUT.
  [t= 77] HQ, THIS IS TL1: SEIZE OBJ ALPHA — COMPLETE. OVER.
  [t= 77] TL1, THIS IS HQ: NEGATIVE, CONTINUE MISSION. OUT.
  [t= 91] HQ, THIS IS TL1: SEIZE OBJ ALPHA — COMPLETE. OVER.
  [t= 91] TL1, THIS IS HQ: NEGATIVE, CONTINUE MISSION. OUT.
  [t= 98] TL1, THIS IS HQ: ENDEX. OUT.
  ```

  The root here is a *successor* — SL1 dies at t=68 and TL1 assumes command,
  which is what `succession_events` 0.29 and the raised human-death rate look
  like on the net. It then files COMPLETE twice, is refused both times, and the
  mission ends on ENDEX after a SITREP rather than on its claim. That is the
  0.86-rejected-of-1.77-filed number as radio traffic: **the new behaviour is the
  root speaking up, and its precision is the open question, not its silence.**
  Worth the owner seeing next to the 0.938 headline.
- 03:55 — **the pre-registered test BREAKS the split, and both mechanism notes
  are STRUCK.** Seeds 16 and 17 landed (94%/96%, gaps 5 and 4 pts) with N=100.

  | seed | success | closed-on-root | stacked | ep len | retasksP | hdr | repR |
  |---|---|---|---|---|---|---|---|
  | 12 | 0.97 | 0.907 | 0.503 | 77.1 | 0.05 | 0.220 | 0.889 |
  | 13 | 0.97 | 0.938 | 0.525 | 73.5 | 0.14 | 0.170 | 0.883 |
  | 14 | 0.94 | 0.000 | 0.206 | 111.0 | 1.95 | 0.020 | 0.708 |
  | 15 | 0.99 | 0.919 | 0.559 | 68.5 | 0.11 | 0.110 | 0.938 |
  | 16 | 0.95 | 0.000 | 0.215 | 97.5 | 0.11 | 0.040 | 0.546 |
  | **17** | 0.94 | **0.745** | **0.211** | **90.6** | 1.26 | 0.140 | 0.665 |

  **Seed 17 closes on root at 0.745 while bunching like the mute seeds**
  (stacked 0.211 against seed 14's 0.206) and with a 90.6-step episode. The
  03:25 pre-registration named this exact case as fatal — "a seed that closes on
  root while bunching like seed 14 breaks the split and the 02:55 note should be
  struck rather than reworded" — so **the 02:55 regime-split note is struck.**
  Closing on root does not require the bunched-and-fast regime; the ρ = +0.660
  over eight arms was association that a ninth arm falsified.

  **The 02:26 co-occurrence claim is struck too**, by the other new seed: seed
  16 has priced retasks 0.11 — as collapsed as any closer — and closes 0.000. So
  churn and the root close do not travel together inside the −1.0 configuration
  either. I asserted 4/4 co-occurrence at 02:26 on four seeds; two more seeds
  broke it in both directions at once.

  **What survives is the effect itself, with no mechanism attached.** At
  `order_retask_cost_base=-1.0` the root closes the mission at **four of six
  seeds** (0.907 / 0.938 / 0.919 / 0.745) and stays mute at two (0.000, 0.000),
  against the −0.03 arm's 0.000 / 0.000 / 0.000 / 0.340. Which seeds close is
  currently unexplained by anything measured: not bunching, not churn, not
  episode length, not success.
- 03:55 — **launched**: `squad_range_control_timecost_v1_seed16` and `_seed17`
  at `time_penalty=-0.03` only. Seeds 16 and 17 were run at −1.0 without matched
  controls, so "4 of 6 close" currently leans on a 4-seed control arm. These two
  make the six-seed row matched and land ~05:05. Nothing about them can settle
  whether the override ships; that stays the owner's.
- 04:20 — **a third candidate mechanism, tested and killed before it was
  written up as a finding.** Scanning nine behavioural variables across the six
  −1.0 seeds for anything separating the four closers {12,13,15,17} from the two
  mute ones {14,16}, three separate without overlap besides the definitional
  `done_reports_root`: `succession_events` (17–34 per 100 episodes for closers
  against 0–2), `doctrine_preference_rate` and `obedience_latency_mean`.
  **The arithmetic of that scan first**: with six points split 4/2, a variable
  separates perfectly by chance with probability 2/C(6,2) = 0.133, so across
  nine variables ~1.2 separations are expected from noise alone. Three is not
  much of a signal. `succession_events` was promoted above the others only
  because it had a mechanism and an independent witness — the 03:27 transcript,
  where SL1 dies, TL1 assumes command, and it is *the successor* that files
  COMPLETE. The story: the root close is really a successor effect.
  **It is not.** Tested within-run on the closing seeds' own 400 episodes, where
  cross-seed variation cannot confound it:

  | seed | succ+close | succ+no close | no succ+close | no succ+no close | p |
  |---|---|---|---|---|---|
  | 12 | 24 | 4 | 64 | 8 | 0.735 |
  | 13 | 17 | 4 | 74 | 5 | 0.089 |
  | 15 | 11 | 2 | 80 | 7 | 0.331 |
  | 17 | 1 | 13 | 69 | 17 | **0.0000** |
  | pooled | 53 | 23 | 287 | 37 | **0.0001** |

  **287 of the 340 closes happen with no succession at all** — the original root
  closes the mission the overwhelming majority of the time — and the pooled
  association runs *backwards*: a root that survives closes 89% of the time, one
  that was replaced only 70% (p = 0.0001). Seed 17 is starkest: when its root
  dies it closes once in fourteen. The transcript was a real episode of a real
  behaviour and still not the typical case; one episode was never evidence of a
  rate, which is the same error `cf4d6cd` corrected earlier this week.
  So the cross-seed succession correlation is a confound, and **the effect
  remains unexplained by anything measured.** That is now three mechanisms
  proposed and three refuted (bunching regime, churn co-occurrence, succession).
  The honest state for the morning is an effect with a strong measurement and no
  story — which is a finding, not a gap in the work.
- 04:50 — **the bifurcation is total, and it is early.** Closed-on-root at the
  best checkpoint (N=20) / final policy (N=20) / final policy (N=100), across
  the six −1.0 seeds:

  | seed | ckpt_best | final N=20 | final N=100 |
  |---|---|---|---|
  | 12 | 0.632 | 0.895 | 0.907 |
  | 13 | 0.500 | 0.947 | 0.938 |
  | 15 | 0.526 | 0.900 | 0.919 |
  | 17 | 0.842 | 0.722 | 0.745 |
  | 14 | **0.000** | **0.000** | **0.000** |
  | 16 | **0.000** | **0.000** | **0.000** |

  The four closers are already above 0.50 at their best checkpoint — 19% to 70%
  of the way through the run — and the two mute seeds read *exactly* 0.000 at
  every checkpoint and every sample size. **No seed partially closes.** The
  outcome is binary rather than a continuum, it is settled well before
  convergence, and no run crosses between the modes. That makes the unexplained
  seed variation sharper, not vaguer: it is not noise around a mean, it is a
  fork taken early and never revisited.
- 05:20 — **the matched six-seed row is complete.** Controls at seeds 16 and 17
  landed (94%/98%) with N=100.

  | seed | arm | success | closed-on-root | retasksP | stacked | ep len | hdr | repR |
  |---|---|---|---|---|---|---|---|---|
  | 12 | −0.03 | 0.97 | 0.000 | 1.47 | 0.246 | 96.4 | 0.020 | 0.477 |
  | 12 | −1.0 | 0.97 | **0.907** | 0.05 | 0.503 | 77.1 | 0.220 | 0.889 |
  | 13 | −0.03 | 0.93 | 0.000 | 2.97 | 0.224 | 97.8 | 0.050 | 0.529 |
  | 13 | −1.0 | 0.97 | **0.938** | 0.14 | 0.525 | 73.5 | 0.170 | 0.883 |
  | 14 | −0.03 | 0.96 | 0.000 | 0.56 | 0.244 | 102.3 | 0.030 | 0.654 |
  | 14 | −1.0 | 0.94 | 0.000 | 1.95 | 0.206 | 111.0 | 0.020 | 0.708 |
  | 15 | −0.03 | 0.97 | 0.340 | 3.71 | 0.204 | 93.5 | 0.240 | 0.470 |
  | 15 | −1.0 | 0.99 | **0.919** | 0.11 | 0.559 | 68.5 | 0.110 | 0.938 |
  | 16 | −0.03 | 0.93 | 0.000 | 2.66 | 0.223 | 111.3 | 0.040 | 0.239 |
  | 16 | −1.0 | 0.95 | 0.000 | 0.11 | 0.215 | 97.5 | 0.040 | 0.546 |
  | 17 | −0.03 | 0.95 | 0.000 | 1.00 | 0.204 | 105.8 | 0.030 | 0.561 |
  | 17 | −1.0 | 0.94 | **0.745** | 1.26 | 0.211 | 90.6 | 0.140 | 0.665 |

  **Closed-on-root pooled: 34/600 → 351/600, p = 6.8e-95.** Four of six seeds
  close; the control arm closes at one seed out of six and only at 0.340.
  Success is unchanged everywhere (0.93–0.99 both arms). Report recall improves
  at five of six.
  **The cost is now significant, where at four seeds it was borderline**: human
  death pooled 41/600 → 70/600, **p = 0.0051** (it was p = 0.052 on four seeds —
  the fuller data sharpened it rather than softening it). It is still
  heterogeneous: up at 12, 13, 17; *down* at 15 (0.240 → 0.110); flat at 14, 16.
  Seed 16 is also the cleanest single refutation of the churn story on the
  record — its retask cost collapsed churn 2.66 → 0.11, a 24× fall, and bought
  no root close at all.
- 05:20 — **launched** `squad_range_control_retaskcost_v1_seed18` and `_seed19`
  at −1.0. "Four of six" is the number the owner's decision rests on and six
  draws is a thin estimate of a rate; these two make it eight and land ~06:30,
  before morning. No controls: the −0.03 arm is now 34/600 and its behaviour is
  not in question.
- 06:35 — **the rate is 4 of 8, not 4 of 6.** Seeds 18 and 19 landed (97%/97%,
  gaps 5 and 5) and **both are mute** — closed-on-root 0.000 at `ckpt_best` and
  0.000 at N=100, with the mute profile throughout (retasksP 0.73 / 0.39,
  stacked 0.236 / 0.232, ep len 99.1 / 111.0).

  | seed | success | closed-on-root | retasksP | stacked | ep len | hdr | repR |
  |---|---|---|---|---|---|---|---|
  | 12 | 0.97 | 0.907 | 0.05 | 0.503 | 77.1 | 0.220 | 0.889 |
  | 13 | 0.97 | 0.938 | 0.14 | 0.525 | 73.5 | 0.170 | 0.883 |
  | 14 | 0.94 | 0.000 | 1.95 | 0.206 | 111.0 | 0.020 | 0.708 |
  | 15 | 0.99 | 0.919 | 0.11 | 0.559 | 68.5 | 0.110 | 0.938 |
  | 16 | 0.95 | 0.000 | 0.11 | 0.215 | 97.5 | 0.040 | 0.546 |
  | 17 | 0.94 | 0.745 | 1.26 | 0.211 | 90.6 | 0.140 | 0.665 |
  | 18 | 0.94 | 0.000 | 0.73 | 0.236 | 99.1 | 0.050 | 0.563 |
  | 19 | 0.92 | 0.000 | 0.39 | 0.232 | 111.0 | 0.080 | 0.684 |

  **Four of eight. The knob is a coin flip, not a two-thirds bet**, and that is
  exactly why the two extra seeds were worth an hour: six draws read 67% and
  eight read 50%. The 05:37 ROADMAP handoff was written on the six-seed number
  and has been corrected.
  The causal claim is unaffected and stays on the **matched six seeds** — those
  are the ones with controls: closed-on-root 34/600 → 351/600, p = 6.8e-95;
  human death 41/600 → 70/600, p = 0.0051. Seeds 18 and 19 have no matched
  control, so they estimate the *rate*, not the effect, and are not pooled into
  the causal test. Success is 0.92–0.99 across all eight.
- 06:35 — **queue exhausted; the watch closes.** Every thread the night opened
  is resolved or explicitly documented as unresolved. Nothing further is
  launched: more seeds would refine 4-of-8 but cannot answer the question that
  matters, which is whether a knob that works half the time and costs a
  significant rise in human death should ship — and that is the owner's.
