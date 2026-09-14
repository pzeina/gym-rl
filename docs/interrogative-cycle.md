# The interrogative cycle — SPEC (owner chose route (i), 2026-09-14; not yet built)

One new exchange: a leader may ASK its element for status instead of staking
a claim to find out. `REQUEST STATUS` is the non-penalized interrogative the
2026-09-14 diagnosis chain ends at; this document is the binding spec for
the build, written before job 1, with the read pre-registered at the bottom.

## Why this shape — the three measurements that force it

1. **Roots close only on their own eyes.** `root_evidence_probe`: own-sight
   0.91–1.00 at confirmed claims on every reporting draw ever probed, and
   the walk is where the deaths are (fireteam 15–25%, platoon up to 0.90).
2. **Information alone is not consumed.** The v1.27 DONE-heard flags went
   unused at every price tested (fresh-sub-DONE at confirmed claims: 0.00
   to 0.26, never above the pre-cycle 0.29). A passive channel that lights
   up when a subordinate happens to volunteer is not a channel the root
   learns to read.
3. **Price moves formation but not the mode.** `root_done_false=-0.1`
   formed the reporting mode on platoon (2/2) and patrol (1/2) — economics
   IS the binding constraint — but what formed was affordable own-sight
   probing with false-complete 0.73–0.94: the death tax became a spam tax.

The shape these force: the evidence must arrive because the ROOT ACTED
(active perception, so credit assignment runs ask → answer → claim), it
must not stake a claim (no −0.5 on the question), and it must not be an
oracle (the answer carries what the element knows, never the success
condition itself).

## A. The exchange

New `MessageKind.REQUEST_STATUS` (agent-issued) + `MessageKind.STATUS_REPLY`
(auto), formatters and parsers as inverses, round-trip pinned.

- **One broadcast action** `REQUEST_STATUS` (appended; the EXECUTE_SIGNAL
  precedent, not four slot-addressed variants):
  `"ALL STATIONS, THIS IS <cs>: REPORT STATUS. OVER."`
- **Every living DIRECT subordinate that hears it auto-answers**, in slot
  order, subject to audibility both ways:
  `"<ldr>, THIS IS <cs>: <mission phrase> — IN PROGRESS. OVER."` /
  `"… — COMPLETE. OVER."` / `"<ldr>, THIS IS <cs>: AWAITING ORDERS. OVER."`
  Completion is the subordinate's OWN mission state via the same
  `is_complete(mission, ctx)` predicate a DONE adjudication uses — the
  subordinate legitimately knows its own state, and this is exactly the
  information a truthful DONE would have carried. It is NOT the root's
  success condition: the root must still integrate "element complete" into
  "operation complete", which is the association this cycle exists to make
  learnable.
- **Auto-answer, not learned answer.** Precedent: WILCO/ACK, DONE_CONFIRM,
  READBACK's CORRECT are all auto — mechanical acknowledgment of a direct
  procedural demand. The diagnosis showed learning bottlenecks compound;
  the one learnable step this cycle adds is the root's (ask, read, claim),
  not a second one on the answerer's side.

## B. Who, when, at what price

- **Mask**: any agent with ≥1 living direct subordinate (TL, SL, PL, CO —
  it generalizes down the chain; a rifleman has no element to ask).
- **Cooldown**: `request_status_cooldown = 8` steps (the done_cooldown
  figure) — a second request inside the window is masked, not priced.
- **Price**: `transmission_cost` only (−0.01, the A4 every-transmission-
  pays rule) and net arbitration at SITREP priority for both the request
  and the replies. NO verdict, NO penalty — that is the entire point. The
  spam guard is the cooldown plus the transmission cost, not a stake.
- **Degraded comms apply unchanged**: the request and each reply ride the
  comm model (range, garble, jamming). Asking into a jammed net and
  hearing nothing back is the outage made legible — free coherence with
  the jamming cycle, no special casing.

## C. Observation

Appended per-direct-subordinate-slot block, mirroring the DONE-heard shape
exactly: **[status COMPLETE heard recently]**, 1.0 for `DONE_HEARD_WINDOW`
(10) steps after a STATUS_REPLY saying COMPLETE from that slot LANDED on
the asker. `MAX_SUB_SLOTS = 4` slots → **OBS_DIM +4** (357 → 361 expected;
the OBS_DIM math asserts the exact figure). Zero-filled for agents with no
element. IN PROGRESS / AWAITING ORDERS answers set nothing — the flag is
closing evidence, not a presence ping.

- The v1.27 DONE-heard flags STAY. They are unused today, but this cycle's
  bet is precisely that a root taught by its own asking to attend to
  element-completion state will start reading the passive channel too;
  removing them now would destroy the measurement.
- N_ACTIONS +1 (239 → 240), appended, existing indices pinned as always.

## D. What does NOT ride

- **No reward change anywhere.** `done_false` stays −0.5, `root_done_false`
  stays default-None. If the interrogative works, the −0.5 stake becomes
  learnable-around at full price; if a later arm wants to combine the two
  knobs, that is a new registered experiment.
- **The price-dispersion cycle** — its mechanism is still the owner's open
  choice. Not folded in.
- **Per-sensor degradation** (epistream item 3) — evaluation-time only, no
  obs change, does not need this break; rides whenever the owner says.
- **jammed observability** — unchanged, per 2026-08-24.

## The break, stated

OBS_DIM 357 → +4, N_ACTIONS 239 → 240: **every v1.27-tree checkpoint is
orphaned**, including the entire 24-run campaign and the six arms. The
first post-break fleet is the v1.28 candidate. This is the second break in
five days; the spec exists so it is also the LAST one this diagnosis chain
needs — (i) was chosen because information, price, and drag have each been
measured to their end.

## The read, pre-registered before job 1

On the retrained fleet (same 24-job shape: 4 seeds for fireteam, squad,
patrol_brique, platoon, platoon_hard; singles for the rest), FINAL policy,
N=100 for any publish-grade claim:

1. **The exchange is used**: REQUEST_STATUS ≥ 1 per won episode on the
   reporting draws, or the cycle is an honest NO EFFECT regardless of
   anything else that moved.
2. **The close becomes evidence-based**: `root_evidence_probe` (extended to
   count fresh status-COMPLETE answers in hand) shows fresh element
   evidence at ≥ 0.50 of confirmed root claims on at least one reporting
   draw per SEIZE scenario — against the 0.00–0.26 measured everywhere to
   date. Own-sight at confirmed claims falling below 0.90 corroborates;
   own-sight pinned at 1.00 with the flags dark refutes.
3. **The original repair bar, unchanged from the read-back cycle**:
   reporting floor ≥ 0.5 with root death ≤ 0.05 at N=100 on at least one
   fireteam draw — and now also testable on platoon/patrol if reporting
   forms there.
4. **Formation at full price**: patrol_brique and platoon each form
   sustained reporting in ≥ 1 of 4 seeds at default prices (vs 0-of-8
   sustained on the v1.27 tree).
5. **Precision guard**: root false-complete ≤ 0.35 on any draw claimed as
   a repair (the spam mode is a failure named SPAM, not a win).
6. **Fleet success guard**: all scenarios non-inferior to their v1.27-read
   candidates (one-sided Fisher, Holm as one family) — same machinery as
   `scripts/readback_read.py`, which this read extends.

Scored by a `scripts/interrogative_read.py` with these thresholds pinned by
test before the campaign launches; the probe extension lands with the
build, not after the results exist.

## Build order (when the owner says build)

1. Vocabulary: kinds, formatters, parsers, round-trips.
2. The exchange: broadcast handling, auto-replies in slot order, audibility
   both ways, cooldown mask; transcript pins.
3. Obs block +4, OBS_DIM math, zero-fill pins.
4. Probe extension + `interrogative_read.py` + threshold pins.
5. Smoke (fireteam short run + scripted big-scenario episode showing the
   full ask→answer→flag chain on the transcript).
6. The 24-job campaign, `cohort/` frozen, ROADMAP entry, stop.

## Addendum (owner, at build authorization) — three riders

**R1. The voice arm, finally, at the shipping layout.** The closing ask of
the last round, restated because the v1.27 cycle moved the goalposts
before meeting it: every voice-arm checkpoint on disk is obs_dim 351, all
24 v1.27 jobs were radio scenarios, and until a voice arm exists at the
shipping spaces `patrol_brique_voice` (granted) is unrecordable and the
degraded-comms side of the epistemic account is frozen at the old pin.
Sequencing correction, owner's intent over letter: training it at 357/239
hours before this cycle's 361/240 break would orphan it the same week —
so the voice arm joins THIS campaign at the post-break layout, EARLY in
the queue (it is the highest-value landing), 2 seeds of the voice-only
training scenario whose checkpoints `patrol_brique_voice` was built to
accept (steps mirroring its prior run config). After it lands: record the
fixture episode zero-shot and the account unfreezes.

**R2. The sender/leader-side states reach the seam.** `perception(cs)`
gains four records it currently hides while the observation vector shows
them: `say_again_pending` (the sharp case — the host sets it by matching
whose transmission was garbled to the requester, ground truth the sender
cannot derive, so a monitor either reads it at the seam or wrongly
under-attributes what the agent was shown), the per-subordinate
read-back-CORRECT-heard window, the per-subordinate DONE-confirmed-heard
window, and — same principle applied forward — this cycle's new
status-COMPLETE-heard window. Same shape as garble: coarse,
listener-attributed, no ground truth, mutating the return never touches
env state. One definition, not two drifting ones.

**R3. A run whose episodes actually hold the new traffic.** A field that
never varies validates nothing downstream. Garble is deterministic
geometry, so the deliverable is a POINTER: a named run + seed under
`comm_model="range"` whose recorded episodes demonstrably contain garble
pings, SAY_AGAIN, and at least one READBACK round-trip. The build ships
the verifier (a read-only probe counting all three in rolled episodes of
a checkpoint) and a range-scenario training job rides the campaign as the
candidate; the post-campaign read runs the verifier and registers the
pointer — or, if no member geometry produces the annulus, registers a
small fixture scenario in the `patrol_brique_voice` mold. The riders are
deliverables beside the read; none of them gates or is gated by the
pre-registered verdicts above.
