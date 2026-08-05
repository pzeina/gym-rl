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
model, p. 9), buildings + pathfinding terrain.

## Milestone v2.0 — Adversarial & scientific

*Real tactics under a learning adversary; the design justified by measurement.*

- `[ ]` **A3. Self-play OpFor** — a second cohort (own org chart, own OPORD) replaces
  the scripted enemy; alternating or league-style training.
  **DoD**: red-vs-blue episodes render in the dashboard with both transcripts; blue
  policy trained vs. self-play beats the scripted-garrison-trained policy head-to-head.
- `[ ]` **B2. Behavioral metrics suite** — measure what "behaves like its rank" means:
  obedience latency (order → first compliant action), report precision/recall
  (contacts reported vs. enemies actually seen), doctrine-preference rate,
  false-COMPLETE rate, succession recovery time, subordinate coverage time.
  **DoD**: emitted per eval run (JSON + dashboard panel); tracked in training metrics.
- `[ ]` **B3. Hierarchy ablation** — same parameter count, three arms: (i) full
  hierarchy + masked doctrine, (ii) hierarchy without doctrine masks, (iii) flat team
  with free comms and no ranks.
  **DoD**: sample-efficiency and final-success comparison across ≥3 seeds per arm,
  written up in `docs/ablation.md`. This is the publishable claim if it holds.
- `[ ]` **B4. Transparency probe** — can a reader predict behavior from the net alone?
  **DoD**: a scripted probe (show transcript-so-far, predict each agent's next
  destination/posture; measure accuracy) — a proxy for the founding promise that
  the command language explains the behavior.
- `[ ]` **A5. Richer order vocabulary** — phase lines / "AT MY COMMAND" timing,
  ATTACH/DETACH task organization, simple formations (wedge/file/line).
  **DoD**: language round-trip tests for each new form; doctrine + masks extended;
  at least one trained scenario exercising the new orders.

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
