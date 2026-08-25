# Night orders — 2026-08-26 (AUTO mode, owner asleep)

Written at 01:10. The contract the morning reader audits: every read below is
gated on its landing, every branch has its decision rule named **before** the
number arrives, and the queue state at the bottom is updated as the night runs.

## Authority tonight

Pre-authorised: launching training (arms, confirm seeds, campaigns), every
zero-token measurement (probes, evals, oracle, digests), committing finished
work, pushing `multi-agent-dev`.

**Forbidden regardless of what is found** — write up for morning instead:
merging or tagging `main`; anything destructive; **publishing a MISS over an
incumbent**; design decisions (rewards, vocabulary, scenario semantics, masks,
gate thresholds, rewording owner-decided claims).

**`cohort/` IS FROZEN** until the campaign's last job lands. A commit to the
tree mid-campaign trains the remaining members against a different environment.
Tooling, tests, docs, `runs/` and boards stay free.

**The one that bites tonight**: `platoon_v13` fails
`closed_on_root_report_rate` (0.000) where its incumbent `platoon_v8` passes
(0.930). **So the v1.23 fleet MUST NOT be sealed or published tonight** — that
would be publishing a MISS over an incumbent. Score it, diagnose it, write it
up; the decision is the owner's.

## State at 01:10

Campaign `v1_23_fleet.jobs` launched 2026-08-25 17:26, **6 of 11 landed**:

| landed | final succ | closed-on-root | stacked | verdict |
|---|---|---|---|---|
| fireteam_v14 | 0.95 | 0.895 | 0.166 | clean |
| fireteam_defend_v25 | 1.00 | 1.000 | 0.927 | **bunching FAIL (first-ever measurement)** |
| squad_recon_v13 | 1.00 | 0.750 | 0.576 | clean |
| squad_screen_v19 | 1.00 | 0.950 | 0.255 | clean; **bit-identical to `squad_screen_v18_seed12`** |
| defend_brique_v19 | 1.00 | 1.000 | 0.931 | **bunching FAIL (first-ever measurement)** |
| platoon_v13 | 0.95 | **0.000** | 0.367 | **MISS vs incumbent 0.930** |

The two bunching FAILs are **not regressions**: every incumbent reads
`stacked_rate = —` because the metric postdates them. Unmeasured is not passed.

In flight: `platoon_hard_v6_seed12` (~01:00 ETA 1h54m) and
`platoon_v14_seed13` (~2h30m, the platoon reporting-mode spare launched 00:59).
Then campaign jobs 8–11: `patrol_brique_v42_seed12`, `v43_seed18`,
`v44_seed19`, `v45_seed14`, ~45 min each. **Campaign ends ~05:30–06:00.**

**The suite is RED and expected to stay red until the campaign ends**, on
exactly one assertion:
`test_the_shipped_record_holds_no_draw_outside_the_declared_blocks` — each
landing member is a same-config draw the manifest does not yet declare, and a
draw cannot be declared until its artifacts are committed. It clears in the
bookkeeping pass at campaign end. ruff is green. Any OTHER failure appearing
tonight is real and stops the thread that caused it.

## The queue, gated on landings

### G1 — `platoon_v14_seed13` lands (~03:30). The night's sharpest read.

`run_report.py platoon_v14_seed13`, read `closed_on_root_report_rate` on the
FINAL policy.

- **≥ 0.5 (it reports)** → `platoon`'s reporting channel is **bimodal**, exactly
  as `patrol_brique` (0.43 over 14 runs) and `squad` (mute at 2 of 4 seeds).
  `platoon_v13` was an unlucky draw. **Action**: declare a `seed_search` for
  `platoon` in `BASELINE.json` naming both draws, and record seed 13 as the
  v1.23 candidate member. Do NOT swap the published member — that is the
  owner's call in the morning.
- **0.000 (also mute)** → two of two mute on the new tree, against an incumbent
  at 0.930 on the old one. **Action**: launch ONE confirm seed,
  `platoon_v15_seed14`, to turn 2 draws into 3 — this is the honest-DoD single
  adjustment for this thread, and it is a confirm seed, not a redesign. Then
  **stop this thread** and write it up: "the tree transition may have cost
  `platoon` its reporting channel" is a finding for the owner, not a knob to
  turn at night.
- **Between 0 and 0.5** → record it and treat as mute for the branch above; the
  documented distribution has nothing between the modes, so a middling value is
  itself worth flagging.

### G2 — `platoon_hard_v6_seed12` lands (~03:00)

`run_report.py platoon_hard_v6_seed12`. Two things only:

1. Did it converge (best-final gap < 10 pts)? If it collapsed, that is a D4
   event — record it, do not re-launch tonight.
2. Its v1.22 `provenance:cohort_tree` waiver exists only because it trained one
   commit ahead of the fleet tree. Retrained here, **that waiver has nothing
   left to waive** — note it for the morning's seal. Its
   `gate:closed_on_root_report_rate` waiver is a separate diagnosed finding and
   **stays**; do not touch it.

### G3 — each `patrol_brique` seed lands (~04:00, 04:45, 05:30, 06:15)

Read `closed_on_root_report_rate` per seed as it lands. Two questions, both
pre-registered:

- **the member**: seeds run 12, 18, 19, 14 in that order precisely so the search
  is self-truncating. If 12 reports, the other three are a free distribution
  estimate rather than a search. Record which seeds report; the member choice is
  the morning's.
- **within-scenario seed-carry, asked for the first time.** The v1.21 attempt
  was void — every run reproduced bit-for-bit, so "does the reporting seed carry
  across a tree change" was unreachable by construction (assurance #60). The
  220 → 351 dimension change makes these genuine draws. Seeds 18, 19 and 14 have
  all reported at the current default on an older tree. **If the same seeds
  report again, the seed carries; if the reporting set is re-rolled, every seed
  search is a fresh 0.43.** Either answer is worth having; record it, decide
  nothing.

### G4 — campaign complete (~06:00). Bookkeeping, then the neutrality gate.

In this order, and the order matters:

1. Declare every new draw in `BASELINE.json` `seed_spread` (and `seed_search`
   for `platoon`/`patrol_brique` if G1/G3 call for it). **Declared ⇒ tracked**:
   commit the artifacts in the same commit, or the identity gate rejects them.
2. `pytest -q` must go **green** — that is the proof the bookkeeping is complete.
   ruff. Commit, push.
3. `baseline.py` — the neutrality/reproduction gate. Every v1.23 member should
   now LOAD (the whole point of the campaign: 220 → 351). Record how many of the
   nine load, and any reproduction the dedupe reports.
4. **Do not `--seal`. Do not touch `BASELINE.json`'s `runs` block.** The fleet
   contains a MISS; superseding a published member is the owner's decision.

## Idle-time work (zero-token, detached, sentinel-terminated)

**N=100 evaluations of the landed members.** N=20 is a smoke test and every
number in the table above is N=20; publish decisions need N=100, and the morning
needs publish-grade numbers to decide the `platoon` question on. Written to
`behavior_final_n100.json`, **never** over `behavior_final.json` — overwriting a
committed N=20 artifact with a different N is a hazard this session already hit
once (2026-08-25, caught and restored).

Runs sequentially in one nohup'd job so it does not fight the trainings for CPU,
and appends members as they land.

## Wake discipline

Monitor (`scripts/night_watch_monitor.sh`) is the primary signal: one line per
training that ends, any outcome, plus one per finished sentinel job. Fallback
`ScheduleWakeup` at 1500–1800 s. Each wakeup: `train_status.py` → act on what
landed per the gates above → update the queue state below → schedule the next
fallback and sleep. Never poll in a tight loop, never read a raw log or
`metrics.csv` into context.

## Queue state (updated as the night runs)

- 01:10 — orders written. 6/11 campaign jobs landed; `platoon_hard_v6_seed12`
  and `platoon_v14_seed13` in flight. N=100 sweep launched for the 6 landed
  members. Monitor armed.
