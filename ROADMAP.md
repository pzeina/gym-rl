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
- `[ ]` **A2. Train the untrained scenarios** — `fireteam_defend`, `squad_recon`.
  **DoD**: committed runs with ≥70% success each; RECON transcript shows the squad
  observing without engaging (stealth compliance visibly working).
- `[x]` **B1. Honest evaluation standard** — bump `evaluate` default to 100 episodes;
  report a 95% CI next to success rates; regenerate the README numbers.
  **DoD**: README results carry N=100 and a CI; eval variance across reruns < 5 pts.
- `[x]` **D1. CI** — GitHub Actions: `pytest` + `ruff` on push/PR (CPU torch).
  **DoD**: badge in README; a PR that deletes `cohort/env/` fails CI (the exact
  failure mode that shipped broken clones pre-v1.1.0).

## Milestone v1.3 — Commandable

*A human commands the cohort comfortably, in the browser, on a clean net.*

- `[ ]` **C1. Dashboard commander mode** — merge `play.py` into the Episode view:
  a live-simulation mode (server keeps an env stepping; WebSocket or polling), an
  order input box wired to `inject_order`, pause-on-contact option.
  **DoD**: from the browser, issue `TL1, seize obj bravo`, watch the WILCO land and
  the maneuver happen; permission errors surface in the UI; works with any checkpoint.
- `[ ]` **A4. Comms discipline** — small per-transmission cost, a "net busy" step
  (one transmission per net per tick, queued), and dedup credit so the first
  accurate CONTACT wins.
  **DoD**: ≤1 SITREP per agent per 25 steps and no duplicate CONTACT storms in eval
  transcripts, without success-rate regression (>–3 pts vs. baseline runs).

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
- `[ ]` **D4. PPO stability guard** — twice observed a converged policy collapse
  mid-fine-tuning (squad continuation @3e-4; platoon_v1 transient dip @1e-4).
  Add a target-KL early stop per update (and optionally entropy-coef annealing).
  **DoD**: rerun the platoon curriculum; no rolling-success dip below 70%.

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
