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
