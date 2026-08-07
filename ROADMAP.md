# Roadmap

## ⟳ Session handoff — resume here (2026-08-07, autocycle)

**State**: `multi-agent-dev` at `ccba9ae`, **64 commits ahead of `main`**; latest
tag v1.9.0; **493 tests green, ruff clean**. **No git remote is configured**
(`git remote -v` is empty) — one must be added before anything can be pushed.
Spaces **Discrete(228)/Box(220)**.

**⚙ A 7-job fleet campaign is running right now** —
`scripts/campaigns/v1_11_fleet.jobs`, ~19.5M steps, ~5–6h from 12:34,
`logs/queue_20260807_123439.log`. Check with `scripts/train_status.py`. **Do not
edit `cohort/` until it drains** unless you mean to; the import snapshot protects
already-imported modules, but the campaign is deliberately running the exact tree
that produced the result below, so that it reproduces it.

**🎉 D4 IS SOLVED.** The collapse that has haunted this repo since v1.0 was one
shared policy free-riding on a terminal its casualties could not collect: the
payout read `for s in roster.living`, so a soldier who died at step 50 of an
episode that succeeded at step 200 got none of the 60 points. Per agent, hanging
back cuts P(die) 0.129→0.022 (+6.4) while team success goes 1.00→0.00 (−52.3) —
but ONE shared policy updates EVERY agent at once, and a per-agent advantage only
ever sees the first number. `d44ee8d` keeps casualties in the episode (STAY-only,
accruing nothing) and pays them the team terminal.

**The A/B, identical config and seeds, `d44ee8d` the only difference**:

| seed | baseline, final N=20 | + fix, final N=20 |
|---|---|---|
| 17 | `squad_screen_v9` **0.00 ± 0.00** | `squad_screen_fallen_v1` **1.00 ± 0.00** |
| 23 | `squad_screen_v10` **0.00 ± 0.00** | `squad_screen_fallen_v2` **1.00 ± 0.00** |

Non-overlapping CIs on both seeds; both treatment arms publishable (0- and 1-pt
best–final gap) and neither ever collapsed — they cleared all three baseline
collapse points (118k, 151k, 395k) without a dip. **Observation width is
exonerated by direct evidence**, not just elimination: the fallen arms run the
same 220-input space `v9`/`v10` collapsed on. Two earlier suspects were refuted
by measurement: the discount inversion (`60cb6c3` — real, but all three bisect
arms collapsed anyway) and entropy/KL/grad-norm blow-up (all flat through it).

**The behaviour is better, not just the score** (oracle, seeds 500–519): cover
occupancy 0.016 → **0.245–0.260**, friendly deaths/ep 1.30 → **0.60–0.65**,
commander death 0.700 → **0.050–0.100**, with *more* engagement (threatened
steps/ep 21.8 → 36.9). Episode length 165 → 53: a short fight is a survivable
fight. The commander stopped being the lead shooter (fire rate 0.830 → 0.200) and
started using cover (0.000 → 0.227) while the riflemen shoot — doctrine falling
out of economics. **A prediction of the opposite was recorded and refuted; the
reward call it raised (price cover / raise `death`) is WITHDRAWN.**

**Still open — the residuals the fix does not touch**: false-DONE **0.279/0.288**
final-decile (0.500–0.600 on the behavior suite), retask/order churn **0.69** and
**0.51**, and `fallen_v2` reporting a contact recall of **0.00**. None
contradicts the success rate; none is closed. Also unexplained: **`platoon` has
16 agents, the most dilution, and converged at 92% *without* the fix** — free-
riding alone never predicted that, and `platoon_v5` in the campaign is the test.

**Every published number in the repo predates `d44ee8d` and is superseded.**
Separately, `scripts/publish_audit.py` gates on the FINAL policy and 11 of 18
published runs fail it (mean give-back 25.9 points). `ckpt_best` is a best-rolling-
*window* figure; both numbers are now measured by default (`behavior.json` +
`behavior_final.json`).

**Next, in order**:
1. **When the campaign drains, judge it** — `scripts/run_report.py <run>` per arm.
   Watch (a) the collapse, on the four scenarios that collapsed pre-fix; (b)
   **`platoon_v5`**, the arm the diagnosis cannot explain; (c) `fireteam_defend_v11`,
   where v1.10's prep period + `prep_in_position` are measured for the first time;
   (d) false-DONE and churn everywhere.
2. **Re-publish the fleet** off the FINAL-policy numbers at N=100 (`/publish`),
   and correct README + the v1.9 table, which are superseded twice over.
3. **Then** land the single-legal-action sampling fix — an agent with one legal
   action should take it without drawing. Held back on purpose: it shifts the RNG
   stream (42 of 55 metrics move on the *same* checkpoint across `d44ee8d`), so
   landing it mid-campaign would desynchronize the result that justified the
   campaign.
4. **Transparency probe** still trails the OPORD-only baseline (best squad gap
   −0.090); residuals in `docs/transparency.md` §A5. **Untouched by v1.10/v1.11.**
5. **`docs/vision.md`** is designed and decided — v1.11 as originally scoped
   (directional vision) comes after the fleet ships. Arc semantics: `vision_arc
   180°` / `fire_arc 90°` / 4-dir facing / all-round awareness at 2 cells. Binding
   constraint: `PolicyNet` is a **memoryless MLP**, so an explicit
   remembered-contact block is mandatory and its stale-track invariant is a
   first-class exploit hazard. `squad_short_vision` is registered as the V0 probe.
6. **A3 self-play**, buildings + pathfinding (v1.4 deferral).

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
