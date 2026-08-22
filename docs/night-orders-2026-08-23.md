# Night orders — 2026-08-22 → 23 (AUTO mode, owner asleep)

The owner retired ~23:59 handing the night over, no focus given. The matched
voice-only campaign is **complete** (18 jobs, committed). What is live instead
is the **squad-depth D4 thread** opened this evening on
`squad_range_control`: seed 14 captures under range-limited radio, and three
diagnostic arms have now been run against it. `squad_range_control_timecost_v1_seed14`
is in flight and lands ~00:33; it is the night's first gate.

`cohort/` is NOT frozen tonight — no campaign queue is feeding — but nothing
tonight has any reason to touch it, and nothing may (see Authority).

## Authority tonight

Pre-authorised (standing, owner 2026-08-18 "do not stay doing nothing —
TRAIN! and experiment"): launching training runs and confirm seeds a scout
protocol calls for; every zero-token measurement (probes, evals, oracle,
digests); committing finished work (full pytest + ruff green, one commit per
unit, repo trailers); pushing `multi-agent-dev`.

Forbidden tonight regardless of findings — write up for morning instead:
merging or tagging `main`; anything destructive; publishing a MISS over an
incumbent; **design decisions** — rewards, vocabulary, scenario semantics,
masks/enforcement; rewording owner-decided claims.

The distinction that governs the whole night: **launching a `--reward` arm is
an experiment and is allowed; changing a default or a `ScenarioSpec` is the
decision and is not.** No `reward_overrides` edit, no `config.py` commit.

One more, specific to tonight: **no six-arm comm-regime verdict.** The N=100
sweep below MEASURES the arms; concluding what the comm regime does to
reporting is a claim about the project's results and belongs to the owner.

## Provenance correction, made before anything else (done 00:1x)

Commit `19c56f7` states "all 18 campaign jobs share cohort/ tree d28592f."
**That is wrong**, and `docs/next-cycles.md` had already flagged it as a
required disclosure. The true record, resolved from each run's
`economics.json`:

- `squad_ctrl_v1_seed12` (job 1) — commit `0f37e6a`, cohort tree **`54a1305`**
- the other 17 jobs + the `seed15` reseed — cohort tree **`d28592f`**

The two trees differ by `cohort/metrics.py` alone (+71/-11): `STACK_RADIUS`,
the bunching half of the spatial-consistency axis, and the trace field that
records it. Measurement only — no env dynamics, no rewards, no masks, no obs.
Verified rather than argued, to the same standard used for the
dispersion-mechanic merge: 400-step rollouts over squad / squad_range_control /
squad_voice_liaison at seeds 14 and 15 hash **identical** observations, action
masks, rewards, terminations and final world state on `54a1305` and on HEAD
(itself already verified equal to `d28592f`).

So the campaign's runs remain mutually comparable and the operative conclusion
of `19c56f7` stands — but the run set is **two trees, not one**, and the record
now says so. Carry this into the ROADMAP handoff at morning.

## The queue tonight (landing-gated)

### Gate 1 — `squad_range_control_timecost_v1_seed14` (~00:33)

`--reward time_penalty=-0.03` at seed 14, the seed that captures. Baseline:
`squad_range_control_v1_seed14` (same seed, same tree, default prices).

Diagnose-first is on file (`scripts/d4_ledger_probe.py`, commit `85de524`):
the captured policy earns compliance +0.00961, command +0.00417, combat
+0.00021, report +0.00014, terminal 0.00000 against time −0.01000 =
**+0.00413/agent-step**. Idling is net-positive income. At −0.03 that becomes
**−0.0159/step**, while the healthy `v1_seed15` policy loses ~2.5% of its
+0.586/step and stays overwhelmingly positive.

**Bar, pre-registered before the run landed** (four clauses):
1. final rolling success ≥ 0.50
2. best-final gap < 10 pts
3. ran-clock-out < 0.50
4. `retasks_priced_per_episode` ≤ 2.0 at N=100

Clause 4 exists because `squad_range_control_rescue_seed14` cleared 1–3 and
still came back sick: 7.26 priced retasks/episode against 0.29 (`v1_s12`) and
1.79 (`v1_s15`), and 0.830 success at N=100 versus 0.940 (p = 0.015).
Surviving is not the same as being healthy.

**Decision rules:**
- **All four PASS** → the price beat the attractor at the seed that captures.
  Next pre-authorised step is the **neutrality gate**, launched immediately:
  `squad_range_control_timecost_v1_seed12` and `..._seed15`, both seeds that
  are healthy at the default price. What they answer: does tripling the price
  damage a policy that never needed it? Queue both, land them, digest, N=100.
- **1–3 PASS, 4 FAILS** → survives but churns like the rescue. Record as a
  PARTIAL. Still launch the two neutrality seeds (the outcome claim needs
  them), and flag the churn to the owner in the morning as the open knob.
- **FAILS (captures anyway)** → the ONE diagnosed adjustment in scope, named
  now so it cannot be invented later: `--reward time_penalty=-0.05` at seed 14
  (ledger: idle income → −0.0359/step). If that captures too → **ceiling**:
  write it up, hand the knob back to the owner, stop the thread. No third arm.
  Honest-DoD: one retrain + one diagnosed adjustment, then document and stop.

### Gate 2 — the neutrality seeds, only if Gate 1 opened them (~02:00, ~03:30)

Per landing: bookkeeping commit, `run_report.py` digest, N=100 on the final
policy. Read against `squad_range_control_v1_seed12` / `_seed15` at the same
seed — the comparison is price-on vs price-off at a fixed seed, nothing else.

## Idle-time measurements (launched at watch start, detached, sentinels)

- `logs/d4_ledger_six_arms.log` — `scripts/d4_ledger_probe.py` over the
  **seed-14 final policy of all six comm arms**. Question never asked before:
  does any other comm regime's converged policy sit near the capture line, or
  is net-positive idle income unique to the arm that actually captured? Ends
  `D4-LEDGER-SIX-ARMS-DONE`.
- `logs/n100_seed14_slice.log` — N=100 behavior evals on the same six final
  policies, written to `behavior_final_n100.json` beside the committed N=20
  files (never over them — the boards must keep quoting the N they mean).
  Gives morning a matched, publication-grade cross-regime row. Ends
  `N100-SEED14-SLICE-DONE`. **Measurement only** — see Authority.

## Per landing, in order

1. **Bookkeeping**: `git add runs/<run>` (whole dir), commit with the run's
   one-line status, push. Never add a dir still RUNNING.
2. **Digest**: `scripts/run_report.py <run> --vs <named baseline>`.
3. **Gate**: score against the pre-registered clauses above, verbatim. A
   clause is passed or failed, not interpreted.
4. **Launch** whatever the decision rule calls for, detached, next free suffix.
5. Update this file's queue state if it changed shape.

A landing that CRASHED: note it, no relaunch, write it up for morning.

## Morning

ROADMAP handoff with the night's ledger (landings, reads, launches, commits,
every miss with its diagnosis, and the provenance correction above); commit and
push; note PUBLISH PENDING boards for the owner's `/boards`; one push
notification with the outcome the owner would act on.

---

# Ledger (written as the night runs)

- **00:04 — `logs/d4_ledger_six_arms.log` lands** (idle job 1 of 2). The
  question it was launched to answer — is net-positive idle income unique to
  the arm that captured, or do other comm regimes sit near the line? — comes
  back unambiguous. Seed-14 final policy, per agent-step:

  | arm | terminal | non-time | TOTAL | at −0.03 | mean ep len |
  |---|---|---|---|---|---|
  | ctrl | +0.806 | +0.841 | **+0.832** | +0.814 | 81 |
  | global-acoustic | +0.492 | +0.518 | **+0.508** | +0.489 | 132 |
  | **range-control** | **0.000** | **+0.0127** | **+0.0032** | **−0.0158** | **450** |
  | no-acoustic ablation | +0.634 | +0.657 | **+0.649** | +0.632 | 90 |
  | voice-direct | +0.705 | +0.736 | **+0.727** | +0.708 | 92 |
  | voice-liaison | +0.770 | +0.797 | **+0.787** | +0.769 | 94 |

  The five converged arms earn +0.51 to +0.83/step, all dominated by terminal;
  the capture earns **+0.0032 with terminal exactly zero**. They are 150–250×
  above the line it sits on. Net-positive idle income is not a property of the
  comm regimes in general — it is what the captured policy alone is left with.

  The mechanism is sharper than "the trickle is too generous." The captured
  policy earns **less** compliance than the healthy ones (+0.0086 against
  +0.018 to +0.025) — it is not farming the trickle, it is merely staying
  barely above water once terminal is unreachable. Its whole margin is
  **+0.0032/agent-step**, so any time price above ~0.013 flips it; −0.03 is
  well past that, and costs a healthy arm ~2% of its income.

  Recorded as measurement. The comm-regime verdict is not drawn here.

- **00:07 — `logs/n100_seed14_slice.log` lands** (idle job 2 of 2), all six
  evals ok. The matched seed-14 slice at N=100, final policy, one seed and one
  tree across six comm regimes:

  | | ctrl | glob-ac | range-ctl | no-ac abl | v-direct | v-liaison |
  |---|---|---|---|---|---|---|
  | success | 0.980 | 0.960 | **0.000** | 0.960 | 0.950 | 0.980 |
  | closed on root report | 0.398 | 0.885 | — | 0.000 | 0.000 | 0.000 |
  | report recall | 0.815 | 0.804 | 0.033 | 0.059 | 0.010 | 0.192 |
  | report precision | 0.859 | 0.936 | 1.000 | 0.700 | 1.000 | 0.685 |
  | messages/ep | 97.9 | 95.9 | 65.8 | 71.9 | 28.5 | 38.6 |
  | root sitreps/ep | 4.17 | 3.54 | 5.01 | 0.000 | 0.000 | 0.000 |
  | timeout | 0.000 | 0.020 | **1.000** | 0.010 | 0.010 | 0.000 |
  | stacked | 0.220 | 0.441 | **0.829** | 0.280 | 0.389 | 0.273 |

  95% CIs on success: ctrl [0.953, 1.000], glob-ac [0.922, 0.998], no-ac abl
  [0.922, 0.998], v-direct [0.907, 0.993], v-liaison [0.953, 1.000],
  range-ctl [0.000, 0.000].

  **Recorded, not interpreted.** The reporting columns separate the arms by
  far more than the success column does, and root-close and sitrep rate split
  the six into two groups — but naming what the comm regime *does* is the
  verdict this watch is forbidden to draw. It is the first matched N=100 row
  the project has across all six regimes at a fixed seed, and it is now on
  file for the owner.

  One thing inside tonight's own thread, and so fair to state: the captured
  policy's **stacked rate is 0.829, over the shipped bunching gate of 0.70**,
  against 0.22–0.44 for the five converged arms. The D4 capture at squad depth
  is also a bunching failure — the squad piles into a blob and sits in it for
  all 450 steps. The gate merged this evening on `dispersion-mechanic` would
  fail this policy on spatial grounds alone, independently of its 0.000
  success. Whether that is coincidence or mechanism is a question for the
  owner, not a claim for tonight.
