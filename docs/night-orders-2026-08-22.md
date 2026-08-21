# Night orders — 2026-08-21 → 22 (AUTO mode, owner asleep)

The owner retired ~00:10 handing the night over, no focus given. The matched
voice-only campaign (`campaigns/voice_only_matched_2026-08-21.jobs`, 18 jobs,
**`cohort/` FROZEN while its queue feeds**) is the night's training — it owns
the box and the tree. This watch therefore **launches no training of its own**
(a parallel run would contend the box and slow all 13 remaining jobs) and
does bookkeeping + zero-token measurement per landing. The campaign's
*verdicts* (matched-arm comparisons) are NOT tonight's to draw — that
analysis belongs to the campaign's own session; the night records facts.

## Authority tonight

Pre-authorised (standing): zero-token measurement (probes, digests,
`run_report`), committing finished work (pytest + ruff green per unit, repo
trailers), pushing `multi-agent-dev`. NOT tonight, regardless of findings:
merge/tag `main`; anything destructive; publishing a MISS; design decisions
(rewards, vocabulary, scenario semantics, masks — AREA FIRE stays OFF and
the stacked axis stays measured); rewording owner-decided claims; and — this
night's specifics — **no commits under `cohort/`** (the campaign freeze),
**no matched-arm verdicts**, **no new training**. The
`dispersion-mechanic` merge stays a DAY item: 13 jobs × ~1.4 h cannot drain
before morning, so the merge condition cannot arrive tonight.

## The queue tonight (landing-gated; ~1.4 h per job)

Jobs land in file order: `squad_voice_liaison_v1_seed12` (~01:30, the
first-ever liaison arm at the matched layout), then the seed-13 round —
`squad_ctrl_v1_seed13` (~03:00), `squad_global_acoustic_control_v1_seed13`
(~04:30), `squad_range_control_v1_seed13` (~06:00),
`squad_voice_no_acoustic_ablation_v2_seed13` (~07:30) — then seed-13 voice
arms into the morning.

Per landing, in order:
1. **Bookkeeping**: `git add runs/<run>` (whole dir, convention per
   `platoon_hard_v5_seed12`), commit with the run's one-line status, push.
   Never add a dir still RUNNING.
2. **Digest**: `scripts/run_report.py <run>` — record success/gates facts in
   the ledger below. Facts only; no cross-arm verdicts.
3. **Spatial read**: `scripts/cohesion_probe.py <run> --episodes 20` (current
   tree — these runs ARE the current tree) — record
   no_close/unseen/stacked/sound/nn per checkpoint in the ledger. This is
   the night's own measurement thread: the matched arms differ in comm
   model, and the spatial axis has never been read across comm regimes.
4. A landing that CRASHED: note it in the ledger, leave the queue to move
   on, no relaunch (the campaign owner sequences its own jobs) — write it
   up for morning.

## Idle-time measurements (launched at watch start, detached, sentinels)

- `logs/spatial_platoon_hard_probe.log` — provenance spatial probe over the
  platoon_hard family not yet read: `platoon_hard_flat_v4_seed12`,
  `platoon_hard_flat_v5_seed13`, `platoon_hard_flat_v6_seed14`,
  `platoon_hard_rdb3_v1_seed12`. Question: does the flat-piling result
  (0.84–0.94 at standard platoon) reproduce at hard difficulty, where 14
  enemies punish clustering more?
- `logs/spatial_squad_depth_probe.log` — provenance spatial probe over the
  squad-depth ablation record: `squad_abl_full_s3`, `squad_abl_flat_s3`,
  `squad_flat_v1`, `squad_nomask_v1`. Question: does flat-piling appear at
  squad depth (7 agents), or is it a platoon-scale (16-agent) phenomenon?
  (Runs may predate `economics.json`; the probe reports missing provenance
  per row and moves on.)

Both feed the `dispersion-mechanic` rider's evidence base; results go into
the ledger and, if they change the rider's claim materially, into its
EVIDENCE paragraph (docs are freeze-exempt).

## Decision rules

- Landings behave (success ≥ ~0.9, gates green) → bookkeeping + reads only.
- A landed arm shows stacked > 0.70 (fails the pending gate) → record it;
  it is evidence for the rider, not a verdict — no action.
- Idle probes surface a result contradicting the rider's claim (e.g. flat
  does NOT pile at hard difficulty) → update the rider's EVIDENCE paragraph
  to carry the contradiction too; never silently keep a one-sided claim.
- Anything wanting a design change, a relaunch, or a merge → morning
  write-up, no action.

## Morning (owner returns, or ~08:00)

ROADMAP handoff ledger (landings, reads, commits, misses), commit + push,
note PUBLISH PENDING boards for `/boards`, one PushNotification, stop the
monitor and the loop.

## Ledger (updated through the night)

- 00:42 — watch armed. Backlog bookkeeping: 9 landed-but-untracked run dirs
  committed (jobs 1–5 of the campaign + the two flat confirm seeds, whose
  landing evals turn out intact — the handoff's crash warning did not
  materialize; artifacts complete). Idle probes launched.
- 00:50 — the monitor died on arming: `/bin/bash` is 3.2, `declare -A` is
  bash-4. Fixed portably (`4422c16`), smoke-tested, re-armed. A watch that
  looks armed and is not is the worst night failure; the fix is the commit.
- 01:00 — both idle probes landed. platoon_hard: flat piles at hard too
  (0.843/0.886/0.720 vs hierarchy 0.354/0.452; all three fail the pending
  gate, v6 by a hair at 0.720). squad depth: `squad_flat_v1` 0.333 — no
  pile at 7 agents; the behavior is platoon-scale, not intrinsic to flat
  control. `squad_abl_*_s3`: no provenance recorded (pre-economics), noted.
  Rider EVIDENCE updated with the scope paragraph. Probe tables preserved
  at the job tmp; note for future watches: probe stdout in `logs/` is
  deny-listed even for grep — write measurement tables elsewhere, keep only
  the `-DONE` sentinel in `logs/`.
