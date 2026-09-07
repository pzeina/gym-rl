# Host response to epistream HOST_REQUESTS (2026-09-07)

*gym-rl's answer to `epistream-rl/HOST_REQUESTS.md`, item by item. Written the
same day the requests were forwarded; items marked GRANTED landed on
`multi-agent-dev` in the four commits noted at the bottom. The short version:
the accessor and items 1, 2, 6 and 8 are granted; item 3 was half granted
before it was asked (your fork is stale — see below) and the other half is
granted now; item 5 is half already-true; items 4, 5-radio and 7-as-training
are the owner's design decisions, queued with recommendations, not silently
dropped.*

*One thing to do on your side before anything else: **re-sync
`~/Documents/gym-rl-fork`**. `parse_sitrep` landed in `6b75cce` (your own
issue #10) and `parse_acoustic_contact` in Phase A — a third of item 3 was
written against code that had already moved.*

---

## The accessor — GRANTED

`env.perception(callsign)` is the public, per-observer product you asked for:
acoustic cues, visual contacts (new, below), friendly sightings coarsened to
the observer's frame, and a self record — copies, never live references, no
private attributes needed. `env.packet_events(since_step=0)` is an
episode-durable copy of the packet event log, so a missed per-step drain no
longer loses a step irrecoverably; your adapter can stop raising on that.
`_agent_cues` and `_packet_log` remain as they were — the seam is additive.

## 1. Hostile contact as a per-observer product — GRANTED (record-only)

`cohort/core/perception.py`: `VisualContact` — eight-way bearing sector,
distance band, count band, formed step; never the true cell, the enemy ids,
or the source's side beyond "hostile was seen". Bounded memory with exactly
the cue discipline (refresh per (sector, band) cell, TTL 6, freshest 4),
maintained per living observer each step, surfaced under
`perception(cs)["visual_contacts"]`.

Two deliberate deviations from the letter of the request:

- **No confidence field.** The host does not model sight confidence, and a
  number invented to fill the slot would be fabrication. The distance band
  is the coarseness; if a confidence model is ever designed, it will be
  designed, not improvised at the seam.
- **Epistemic status is stated, not blurred.** Acoustic cues enter the
  observation vector — they are memory the agent itself acts on. The visual
  contact record does not: agents already see enemies live (their private
  enemy slots — the premise of your request 1 was true only at the seam),
  and this record is the host coarsening and retaining what the agent
  sensed, for you. A test pins the boundary: with the maintenance call
  disabled, observations and rewards are bit-identical. If you attribute
  this record as agent knowledge, attribute it as "what the agent was shown",
  not "what the agent remembers" — its memory is its policy state.

## 2. Coarse spatial content on a friendly sighting — GRANTED

`perception(cs)["friendly"]`: per related station, `seen_now`, bearing
sector, range band, age — in the observer's own frame, not cells. Under
`voice_only` it reads the §3.7 decayed last-seen state (so age is real and
`inferred_sector` finally has a record to stand on); on a radio net
positional telemetry is live in the observations, so the coarsening is of
live positions and age is 0 — which is itself the honest statement of what a
radio net gives an observer.

## 3. Parsed structure for the report kinds — GRANTED (and half pre-existed)

Already in the tree before the request: `parse_sitrep` (commit `6b75cce`,
refs #10), `parse_acoustic_contact` (Phase A), plus `parse_order`,
`parse_opord`, `parse_receipt`, `parse_dispatch`, `parse_gesture`,
`parse_succession`. Landed now: `parse_contact`, `parse_done`,
`parse_done_confirm`, `parse_done_reject`, `parse_casualty`, `parse_trap`,
`parse_support_end`, and `parse_mission_phrase` (the spoken-tasking inverse
the DONE family shares). Every parser is its formatter's inverse over
exactly the fields the formatter takes, returns None on any other kind, and
round-trips under test across all 12 missions.

Two notes. There is no "help request" kind in the vocabulary — the closest
existing speech acts are SUPPORT orders and `SUPPORT_END`; if your plugin
has a slot for one, it maps to nothing here yet. And the structure ships as
parsers, never as payload on the `Message` object: radio messages are
text-only by owner decision, guarded by
`test_orders_flow.py::test_radio_messages_are_text_only`, because a
structured side-channel on the message is a real exploit surface. The
generator's structure is also recorded in the transcript's audit metadata
(`payload` on contact/acoustic/sitrep entries) — but the parsers are the
supported seam.

## 4. Legible receipt (read-back, SAY AGAIN, negative ack) — OWNER DECISION, QUEUED

You are right that this is the highest-leverage change, and it is exactly
the kind this host cannot ride along: new vocabulary changes the traffic
every agent hears, which changes the environment, which obsoletes every
trained checkpoint. It goes to the owner with a recommendation to take it in
the next breaking cycle (`docs/next-cycles.md`), in the agent-issued form —
read-back discipline as a *learnable* behavior fits this project's thesis
better than an automatic env echo, and the pattern already exists in
miniature in the liaison receipts (`format_receipt` / `format_negative` /
`format_undeliverable`). Until then, second-order belief has one legitimate
in-world source: the WILCO/ACK traffic that already exists for orders.

## 5. Partial reception — HALF ALREADY TRUE, half queued

On the voice plane this exists today: a transmission beyond intelligibility
but within `VOICE_DETECT_RADIUS` (4.0) already produces a non-semantic
`voice` cue — precisely "someone transmitted and I cannot make it out",
listener-attributed, no content. Your adapter can read it from
`perception(cs)["cues"]` now; the state you said collapses into silence has
been distinct since Phase A. What does not exist is the radio analog: under
`comm_model="range"`, out-of-earshot is total silence. Making garble
agent-perceivable needs an observation slot (OBS_DIM change, fleet retrain),
so it is queued with item 4 for the same breaking cycle — a garble state and
a SAY AGAIN are natural companions.

## 6. Self-knowledge of own condition — the explicit statement, plus a record

Self-condition was never absent — it is the first thing in every observation
vector (the own-state block); the perception *records* skip self because the
sense they model is directed outward. The assumption is now stated where you
read: `perception(cs)["self"]` returns the agent's own condition (alive,
position, health, ammo) and the accessor's docstring says in so many words
that self-knowledge is ASSUMED, not sensed. If your belief abstraction wants
a sixth-axis-style entry for it, model it as "known by assumption, at age 0,
always".

## 7. Per-agent, per-sensor degradation — OWNER DECISION, with a cheap framing

The negative-control argument is correct and we want you to have it. The
framing that makes it nearly free: an **evaluation-time intervention**
(degrade one station's hearing / optics / radio at `evaluate` time, default
off, forbidden in baseline runs) rather than a training-time configuration.
That yields falsifiable per-sensor attribution without touching the
baseline, any published number, or what any policy learned. Presented to the
owner in that form; if accepted it lands as an evaluation-harness feature,
not scenario semantics.

## 8. Scenarios that move the dimensions — GRANTED

`patrol_brique_voice`: the brique ambush patrol under the voice mode (voice
only, tactical acoustics, liaison). An ambush band and mines put weapon
(16-cell) and trap (12-cell) sounds at range and intermittently, so cues
span the distance bands and age in memory between harassments, and the
patrol walks into real contact. Pinned by test: a scripted patrol at seed
100 yields bands {0, 1}, ages 0–6, four sound kinds, and sightings. The
geometry matches `squad`, so the `squad_voice_liaison` member's checkpoints
transfer — you can record fixture episodes today, zero-shot, no retrain.
Excused from the baseline as a fixture scenario, priced with the voice
family's economics for the family's structural reason.

---

## The boundary, reciprocally

Nothing above changes what an agent can do: OBS_DIM is untouched, no reward
moved, no action was added, and the one genuinely behavioral request (4) was
routed to the owner instead of smuggled in. Your "not asking for" list is
honored — no oracle exposure, no sound-plane truth, no logits, no host-side
epistemic modelling: the host coarsens and retains what its agents sense;
what that *means* stays your job.

**Landed as:** `e71140d` (parsers), `9224212` (perception seam),
`b6298b0` (fixture scenario), plus this document.
