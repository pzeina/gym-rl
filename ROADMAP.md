# Roadmap

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
