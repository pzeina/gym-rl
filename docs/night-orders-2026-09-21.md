# Night orders — 2026-09-21 (symmetric-enemy self-play cycle)

Owner's focus, verbatim intent: **a new cycle on a separate branch dedicated to
improving the enemy agents via the same learning the friendly team gets —
symmetric warfare (not guerrilla) — toward a maximally capable enemy. Train and
evaluate.** The cycle is owner-commissioned; the design choices below are the
night's faithful-minimal reading of it, recorded here for the morning audit,
not settled doctrine.

State at watch start: nothing training (box idle), boards PUBLISH PENDING
(left for the owner's `/boards`), branch `multi-agent-dev` clean at `83fa8be`,
v1.28 campaign runs landed in `runs/` (untracked — bookkeeping is NOT tonight's
focus; left as found). Suite last known green at 1318+.

## The design, as the night reads the commission

The enemy today is scripted (`enemy_decide` garrison/assault; the brique band
is guerrilla and explicitly OUT of scope tonight). Symmetric learning means the
red side becomes a second cohort: same ranks, same masks, same radio, same
rewards — and learns.

1. **Branch `enemy-symmetric-selfplay`** off `multi-agent-dev`. Nothing on
   `multi-agent-dev` moves tonight except these orders and the morning ledger.
2. **New scenario `fireteam_symmetric`**: two mirrored fireteams (blue TL1,
   RFN2-4; red RTL1, RRFN2-4), mirrored spawns, ONE central objective, both
   sides ordered to SEIZE it. Outcome is per-side: win / loss / draw
   (eliminate the other side or hold the objective; draw at horizon).
   Existing scenarios and the scripted OpFor are UNTOUCHED.
3. **Cross-team perception via Enemy-proxy adapter**: each side's observation
   builder sees the opposing soldiers as `Enemy`-shaped records through the
   existing `visible_enemies` path. OBS_DIM (361) and N_ACTIONS (240) DO NOT
   MOVE; current-tree checkpoints stay loadable. Each side has its own net —
   cross-team radio never delivers.
4. **Shared-policy self-play**: the one parameter-shared PPO policy controls
   BOTH sides. Symmetry is then by construction, and "the maximum capable
   enemy" IS the policy — every improvement against itself improves the enemy.
   A `--opponent-checkpoint` flag freezes one side for evaluation and
   best-response probes.
5. **Collapse-rescue must not misfire**: in self-play the blue-side success
   rate hovers near 0.5 by design; the rescue machinery keys on rolling
   success and would fire forever. Self-play runs disable it (or key it on a
   symmetric proxy); the flag and reasoning go in the run config.
6. **Evaluation = exploitability ladder** (the capability claim): fixed-budget
   best-response blue trained against a FROZEN red at three rungs —
   random-init, mid-training, final. Best-response win-rate should fall
   monotonically along the ladder; final-rung suppression is the "maximally
   capable enemy" number. Plus one evaluated episode digest (radio traffic)
   for the gallery-style read.

## Tonight's queue (every read gated on its landing)

| # | Gate | Action |
|---|------|--------|
| 1 | — | Commit + push these orders (`multi-agent-dev`). |
| 2 | — | ONE background build agent, phased brief below, on the new branch. |
| 3 | build P1-P2 committed, suite green | Agent launches `fireteam_symmetric` self-play, 2 seeds, ~3M steps, detached via `scripts/train.sh`. |
| 4 | self-play landing | `run_report.py` digest; sanity: outcome balance near 50/50, combat actually joined (casualty rates > 0), no degenerate mutual-avoidance equilibrium. |
| 5 | digest sane | Exploitability ladder: 3 best-response runs (fixed budget ~1M) vs frozen red rungs; then `ablation`-style read of the ladder. |
| 6 | ladder landed | Write the verdict in the ledger; commit artifacts on the branch; push branch. |

## Decision rules

- **Separates** (ladder monotone, final rung suppresses best-response win-rate
  vs random-rung by a clear margin): the cycle's claim holds — write it up,
  push the branch, leave merge/next-steps for the owner.
- **Wall** (self-play degenerates: mutual avoidance, 100% draws, or one-side
  collapse): the ONE pre-named adjustment is a draw-penalty / engagement
  shaping ride-along **taken from the existing RewardConfig knobs only** — no
  new reward components at night. One retrain, then document and stop.
- **Ceiling** (learns but ladder flat): write it up; opponent-pool/league
  training is the named next knob and it is the owner's call.
- Build agent misses honest-DoD (P1/P2 not green by ~04:00): stop the thread,
  document exactly where it stalled, leave the branch pushed with whatever is
  green.

## Idle-time list

- None launched: the box is kept free for the self-play runs; bookkeeping of
  the landed v1.28 campaign is the pending zero-token work but it belongs to
  the interrogative read on `multi-agent-dev`, not to this branch or night.

## Authority (standing, owner 2026-08-18)

Pre-authorised: launching training, every zero-token measurement, committing
finished work (pytest+ruff green, repo trailers), pushing `multi-agent-dev`
and the cycle branch. Forbidden tonight regardless of findings: merge/tag
`main`; anything destructive; publishing a MISS over an incumbent; design
decisions beyond the commissioned scope above (no reward-component additions,
no vocabulary changes, no touching existing scenarios' semantics). Honest-DoD:
one retrain + one diagnosed adjustment per miss, then document and stop.
