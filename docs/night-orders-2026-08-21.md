# Night orders — 2026-08-21 (second watch)

Owner retired after choosing the KL-guard experiment and seeing it convert
both captured seeds (2/2, one rollback each, no re-migration; ROADMAP
handoff `ca5ca85`). The ship question — price as platoon_hard semantics,
rescue default vs opt-in — is the OWNER'S and nothing tonight touches it.
Tonight sharpens the evidence the morning decision will rest on.

## Authority (standing, owner 2026-08-18: "do not stay doing nothing — TRAIN! and experiment")

Pre-authorised: launching training runs and campaigns (incl. confirm
seeds), every zero-token measurement (probes, evals, oracle, digests),
committing finished work (full pytest + ruff green, one commit per unit,
repo trailers), pushing `multi-agent-dev`.

Forbidden regardless of findings — write up for morning instead: merging
or tagging `main`; anything destructive; publishing a MISS over an
incumbent; design decisions (rewards, vocabulary, scenario semantics,
masks/enforcement, rewording owner-decided claims); ANY further `cohort/`
change (the rescue is committed default-off; enabling it anywhere is the
owner's call). Honest-DoD: one retrain + one diagnosed adjustment per
miss, then document and stop that thread. Digests only — never raw logs,
metrics.csv, or checkpoints into context.

## Launched at watch start (~00:25)

1. `platoon_hard_rescue_timecost_v1_seed16` — seed spread of the candidate
   recipe (`--reward time_penalty=-0.03 --rescue-max 3`, 3M).
2. `platoon_hard_rescue_timecost_v1_seed17` — same, seed 17.

Declared now: 16/17 are same-config draws of the seed-14/15 conversion
arms — seed spread, not new experiments.

Idle zero-token jobs (detached, sentinel `<NAME>-DONE`):
- `oracle_final_rescue14` — oracle probe of seed-14 conversion arm's FINAL
  policy (ckpt_latest, 30 eps, seed 600) — the headline checkpoint for a
  publishable run; ckpt_best was probed at watch start (0.90).
- `oracle_final_rescue15` — same for seed 15 (ckpt_best probe: 0.833).

## Landing-gated queue

**When seed 16 or 17 lands (~03:05), per arm:**
- `run_report.py <arm>` + `rescues.json`. Same three clauses as the whole
  cycle: final rolling ≥ 0.5, best-final gap < 10, ran-clock-out < 0.5.
- PASS → oracle-probe ckpt_best (30 eps, seed 500), add to the table
  (4/4 → …/6). No further action.
- CAPTURE despite rescues → THE finding of the night: read `rescues.json`
  (did it fire? re-migrate after every restore? exhaust rescue_max?).
  Document the failure mode; do NOT launch more seeds on this thread
  (honest-DoD); the morning table says "4/6" and why.
- Non-capture miss (converged but under a clause) → quote both numbers,
  no follow-up arm; morning judges.

**When the first of 16/17 lands, also launch (machine slot frees):**
3. `platoon_hard_flat_rescue_v1_seed14` — the generalization test:
   `--scenario platoon_hard_flat --seed 14 --total-steps 3000000
   --rescue-max 3`, NO reward override. Its twin
   `platoon_hard_flat_v3_seed14` captured at 0% on defaults; if the
   rescue alone converts it, the rescue generalizes beyond the timecost
   arm — direct evidence for the owner's default question. If it fails,
   the rescue is so-far arm-specific; document. Bar: same three clauses.
   Read when it lands (~06:00); if it lands after morning, it is the
   morning's first read.

**Sentinel jobs finishing:** fold the final-policy oracle numbers into the
ledger; no action branches on them (they inform the morning table only).

## Deliberately NOT done tonight

- **Archive sweep (`archive_runs.py --apply`) — DEFERRED AGAIN, decided
  00:20**: the dry run wants to move the rescue conversion arms (landed
  30 min ago) and the full timecost evidence set — the live evidence for
  the open ship decision, which tonight's probes read and seeds 16/17
  mirror. Moving an open thread's evidence mid-cycle serves nothing
  (same call as 2026-08-20). Morning may apply it after the owner
  decides.
- Boards will go PUBLISH PENDING as landings refresh them — morning
  `/boards`.

## Ledger (append as the night runs)

- 00:05 both conversion arms read: 2/2 pass all clauses, one rescue each,
  oracle 0.90 / 0.833. ROADMAP updated, committed `ca5ca85`, pushed.
- 00:20 archive dry run reviewed — deferred (above). Night orders
  written, committed, pushed.
- 01:21 final-policy oracle (fresh seeds 600+): rescue14 ckpt_latest
  success 0.933 (above its ckpt_best 0.90), rescue15 0.833 (equal). Both
  conversion arms hold at the headline checkpoint. Seed 16 observed
  mid-dip (rolling 0% at 45%) — rescue expected to fire; live test of
  the mechanism. Seed 17 healthy (54%).
- 01:52 both arms slowed ~6x (ETAs 7-9h): owner user-space load (Android
  emulator 72% CPU, Hearthstone 61%) plus a pre-watch Python (pid 96620,
  running since 13:09) — none of it mine to kill; trainings will simply
  take longer. Seed 16 SELF-recovered from its 0% dip without a rescue
  (no rescues.json; rolling 60% at 49%) — consistent with the guard's
  recovered-dip calibration; the 700-iter patience correctly spared it.
  Landings now projected after morning; flat_rescue arm stays gated on a
  freed slot per orders. Heartbeat stretched to 60 min.
- 03:05 seed 17 LANDED, PASSES (final 88 / gap 8 / clock-out 0.119,
  publishable) with the rescue armed but NEVER fired — self-recovered
  from its dip; the patience spared it correctly. Oracle probe detached.
  flat_rescue_v1_seed14 launched in the freed slot (generalization test).
  Seed 16 landing imminently; its rescue fired at 76% (restored from
  iter 828) and rolling was still 0% at 91% — re-migration watch.
- 03:20 seed 16 LANDED, FAILS all clauses (final 0.000 / gap 94 /
  clock-out 1.000). The precise failure mode: rescue 1 fired at 76% and
  WORKED — the restored policy re-climbed to a 94% peak at 84% (ckpt_best
  post-dates the rollback) — then a third capture took it in the final
  decile and only ~470 iters remained, under the 700 rescue-2 needs. Died
  mid-draw, 2 rescues unused. Not "rescue fails": "this seed re-captures
  faster than the budget re-draws". Honest-DoD: no more seeds; the
  morning table reads: rescue-armed 5/6 converged+publishable (12, 13,
  14, 15, 17 — of which 14/15 needed a rollback, 13* pre-rescue), seed 16
  captured at budget end. flat_rescue generalization arm running.
