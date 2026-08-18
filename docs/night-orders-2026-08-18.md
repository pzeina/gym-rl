# Night orders — 2026-08-18 → 19 (AUTO mode, owner asleep)

The owner's instruction on retiring (2026-08-18 ~23:45): continue autonomously
through the night — do not idle; train and experiment. This file is the queue
and the guardrails; the session's self-paced loop reads it at every wakeup.

## Authority tonight

Pre-authorised (standing + tonight's instruction): launch training runs and
campaigns, run any measurement (probes, evals, oracle — all zero-token, CPU
only), commit finished work (pytest + ruff green, per-unit messages,
trailers), push `multi-agent-dev`.

NOT tonight, regardless of findings: merge/tag `main`; anything destructive;
publishing a MISS over an incumbent; design decisions — reward structure,
vocabulary, scenario semantics, **enforcing** cohesion (it stays measured
only), rewording owner-decided claims. Findings that want any of these are
written up for the morning instead.

## The queue

Gate every read on its landing; bookkeeping at every landing (same-config
draws → `seed_spread`; declared ⇒ tracked; artifacts committed; suite green).

1. **`platoon_v12_seed12` lands (~00:50)** — the neutrality gate for the whole
   scout: `baseline.py`'s reproduction table must show it ≡ `platoon_v8`
   bit-for-bit. If it does NOT: the platoon_hard reads are confounded — stop
   the confirm path, diagnose, write it up, and leave the campaign's remaining
   jobs to run (data is data) but draw no cross-tree conclusions.
2. **`platoon_hard_v1_seed12` lands (~00:50)** — `run_report` digest only.
   Mid-run it showed rolling success 0% at 54%: a wall is possible. Do NOT
   verdict the scenario off the full arm alone — the arms are the measurement.
3. **`platoon_hard_nomask_v1_seed12` (~03:50), `platoon_hard_flat_v1_seed12`
   (~06:50)** land in sequence — digest each.
4. **When neutrality + all three hard arms are down**: N=100 on both
   checkpoints (`publish_baseline.py`, detached), cohesion probe
   (`scripts/cohesion_probe.py`) over the hard trio, then the scout read —
   `ablation_report.py` on the three arms (1 seed; a scout, not a claim).
5. **Scout verdict, per the owner's scout-then-confirm protocol:**
   - **Separates** (success / defeats / root death / cohesion, scenario not
     degenerate) → launch the confirm: seeds 13/14 for all three arms, 6 runs
     in two lanes (`train_queue.sh`).
   - **Wall** (every arm ≈ 0 success) → ONE diagnosed adjustment is in scope:
     recalibrate `platoon_hard` (n_enemies 14 → 11, one commit, the three
     scout arms relaunched at seed 12). If the recalibration also walls,
     document the miss and stop — difficulty design goes back to the owner.
   - **Ceiling again** (every arm ≈ 1.0) → no second adjustment overnight;
     write it up, the difficulty knob goes back to the owner.
6. **Idle-time experiments while lanes train (zero tokens):**
   - Transparency probe over the six platoon-depth ablation arms (nomask ×3,
     flat ×3, both checkpoints, protocol seeds 500+, K=15) — does a flat
     cohort's net read better or worse than the hierarchy's? New cells for
     docs/transparency.md. (Launched 2026-08-18 ~23:55, detached.)
   - If still idle: the pending `squad_screen` oracle pass (read-only
     diagnosis of the 24% root-death exposure; no reward changes).
7. **Morning**: ROADMAP handoff updated with the night's ledger; one
   PushNotification with the outcome; boards will show PUBLISH PENDING for
   the owner's `/boards`.

## Wake discipline

A persistent monitor watches for training endings and crashes; wakeups
otherwise fall back every ~25 min. Each wakeup: `train_status.py`, act on
whatever landed per the queue above, commit/push, back to sleep. Token
discipline holds at night: digests only, never raw logs or metrics.csv.
