# The read-back cycle — APPROVED BY THE OWNER 2026-09-09 ("go with option (a), pair it with the read-back cycle")

One breaking cycle, three pieces that belong together, and the measurement
that motivated it. Written before the build; the build follows this spec and
documents any forced deviation in its commits.

**What the owner approved.** The 2026-09-09 root-evidence probe confirmed the
mechanism behind fireteam's root-death trade: reporting roots close ONLY from
their own sight of the objective (own-sight 1.00, median 1.41 cells at claim,
both reporting draws) and die there (death distance 2.6-3.1 cells vs episode
mean ~17); a fresh subordinate DONE is in hand for just 15-29% of confirmed
claims. The owner chose option (a) — make subordinate DONE traffic sufficient
closing evidence so a rearward root can claim on reports — paired with the
read-back cycle (epistream HOST_REQUESTS item 4, the response doc's §4), which
§5 of that doc already binds to the garble state ("a garble state and a SAY
AGAIN are natural companions").

**The principle the whole cycle obeys** (from the 4e82807 removal): no
telemetry of other agents' state. `mission_heard` died because it taught a
listener another soldier's mission without any transmission. Every new
observation slot below encodes something the observer HEARD ON THE NET — a
received report, a received request, a received-but-unintelligible ping —
never ground truth about another agent.

---

## A. Garble state (epistream item 5, the radio analog of the voice cue)

Under `comm_model="range"` only: a transmission whose listener is beyond
`comm_range` but within `GARBLE_RADIUS_FACTOR * comm_range`
(`GARBLE_RADIUS_FACTOR = 1.5`, a new comm constant) produces a **garble
ping** for that listener — non-semantic, no content, no sender identity, no
bearing (a garbled radio signal carries even less than a voice cue). The
message does NOT enter the transcript for that listener (it did not land);
the ping is listener-private state, like `_agent_cues`.

- `global` never garbles (everything lands). `voice_only` already has the
  voice cue for exactly this state — no duplicate. `jammed` outages stay
  **unobservable** (owner decision 2026-08-24, not relitigated): a jammed
  transmission produces NO ping.
- Deterministic geometry, no RNG.
- Records kept per listener with a TTL (mirror the cue TTL discipline);
  exposed in `env.perception(cs)` as `garble` records (epistream reads the
  seam; §5 promised this).
- **Obs**: appended garble block, 2 slots — [ping pending (0/1), freshness
  (ttl remaining / TTL)]. Appended after the existing acoustic/liaison
  blocks per the degraded-comms precedent (interior blocks never grow).

## B. SAY AGAIN (agent-issued)

New `MessageKind.SAY_AGAIN` + appended action `SAY_AGAIN`.

- **Mask**: available iff a fresh garble ping is pending OR a fresh
  unintelligible `voice` cue is held (the two spellings of "someone
  transmitted and I could not make it out").
- **Effect**: transcript line addressed to the unknown station —
  `"STATION CALLING, THIS IS <cs>: SAY AGAIN. OVER."` Every station whose
  transmission within the garble window was garbled to the requester AND
  that hears this request gets a **say-again-pending** flag (1 obs slot,
  sender side, TTL'd). No automatic re-transmission — the env never echoes;
  the sender's policy chooses to re-send or close distance. That is the
  learnable loop: ping → SAY AGAIN → sender sees it → sender acts.
- Formatter + parser, round-trip pinned.

## C. Read-back with negative ack (epistream item 4, agent-issued form)

New `MessageKind.READBACK` + appended action `READBACK`, answered
automatically by the issuer like DONE is: `READBACK_CORRECT` /
`READBACK_WRONG` message kinds (auto).

- **Mask**: available iff the agent holds a live mission with a superior to
  read it back to (HQ answers for the root's OPORD).
- **Content**: the mission the agent ACTUALLY holds, spoken with the
  existing mission phrase formatters —
  `"<ldr>, THIS IS <cs>: I READ BACK — <mission phrase>. OVER."`
- **Adjudication (auto, mirror of DONE_CONFIRM/REJECT)**: matches the order
  the superior last issued to that station → `READBACK_CORRECT`
  (`"<cs>, THIS IS <ldr>: CORRECT. OUT."`); else → `READBACK_WRONG`
  (`"<cs>, THIS IS <ldr>: NEGATIVE, I SAY AGAIN — <order phrase>. OUT."`).
  The WRONG branch restates the actual order — voice procedure's mandated
  correction repeat. This is the answer to an agent-initiated check, not an
  unprompted env echo, so it does not violate the agent-issued principle.
- **Obs**: issuer-side appended block, per subordinate slot: [read-back
  CORRECT heard recently] (MAX_SUB_SLOTS slots, window mirroring the
  10-step recent-contact-report flag).
- Formatters + parsers for all three kinds, round-trip pinned.

## D. Closing evidence — option (a), the piece the probe demanded

Appended per-subordinate-slot block: **[DONE confirmed heard recently]** —
1.0 for `DONE_HEARD_WINDOW` steps after this leader has RECEIVED that
subordinate's DONE **and answered DONE_CONFIRM** (a confirm the leader itself
spoke is knowledge it certainly has; rejected DONEs carry no closing
evidence). MAX_SUB_SLOTS = 4 slots. Window: mirror the recent-contact-report
10-step flag unless the build finds a reason not to (document it if so).

- **No adjudication change.** The root's OPORD claim is already judged
  against ground truth (`_check_success`) wherever the root stands; a
  rearward claim was always confirmable. What was missing is the root's
  ability to KNOW — this block is that knowledge, sourced strictly from
  received traffic.
- **No reward change.** The probe's finding is an information deficit, not a
  price deficit. Diagnose-first says: give the channel, retrain, re-run
  `scripts/root_evidence_probe.py`, and only then decide whether a price is
  needed. Shaping read-back or rearward claiming now would price a behavior
  before measuring whether information alone moves it.

## Spaces (the break, stated)

- OBS_DIM grows by 2 (garble) + 1 (say-again-pending) + 4 (read-back
  correct per slot) + 4 (DONE heard per slot) = **+11**, all appended
  blocks, zero-filled where structurally unavailable (e.g. garble under
  `global`). Exact figure asserted by the OBS_DIM math as always.
- N_ACTIONS grows by 2 (SAY_AGAIN, READBACK), **appended** so the existing
  indices never move (pinned by test_degraded_regression.py).
- **Every existing checkpoint is orphaned.** The first post-break fleet is
  the new baseline (v1.27 candidate); nothing can be published against
  v1.26 members on this tree once it lands. Same shape as the 4e82807
  break, said out loud this time before the build.

## What does NOT ride

- The price-dispersion cycle: its mechanism ((1) per-step vs (2) threshold)
  is still the owner's open choice. It does not sneak in here.
- `jammed` observability: stays unobservable, per the 2026-08-24 decision.
- Automatic ACK semantics: ACK stays auto; order uptake is not relitigated.

## The read, pre-registered before job 1

On the retrained fireteam 4-seed search, for every draw that clears the 0.5
reporting floor (`closed_on_root_report_rate >= 0.5` at the final policy):

1. `scripts/root_evidence_probe.py`, 50 eps: **fresh subordinate DONE in
   hand at confirmed claims must RISE above the 0.29/0.15 measured on
   v17/v19**, or the channel was built and not used — an honest NO EFFECT.
2. **Root death at or below the mute draws' band** (v1.26 evidence: 0.00 at
   N=20; the bar is root death <= 0.05 at N>=100 for a claimed repair)
   while reporting holds — that is the trade actually repaired.
3. A draw may still walk forward and die (policies choose); 2 failing with
   1 succeeding is PARTIAL: the information is used but does not substitute
   for presence — a finding, logged, not spun.

Fleet guard: the other eight scenarios ship success within CI of their
v1.26 members (Holm-corrected as one family, per the dispersion prereg's
precedent) — this cycle must not buy fireteam's root a bodyguard by breaking
someone else.

## Retrain plan

Full fleet on the frozen post-build tree, ~24 jobs via `train_queue.sh`:
4-seed searches (seeds 12-15) for the five known-bimodal reporting
scenarios — fireteam (bimodal as of 2026-09-09), squad, patrol_brique,
platoon, platoon_hard — and single seeds for fireteam_defend, squad_recon,
squad_screen, defend_brique. Steps per scenario mirror each v1.26 member's
`total_steps`. A campaign freezes `cohort/`: no cohort/ commit between the
first and last job launch; tooling/tests/docs stay free.
