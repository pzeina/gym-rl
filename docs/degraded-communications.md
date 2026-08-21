# Voice-only degraded communications

> **STATUS: design and implementation brief; not implemented.** Nothing in
> `cohort/` currently provides the semantics specified here. The existing
> `comm_model="range"` remains a range-limited **radio** model and must not be
> relabelled as this mode.

## 0. Decision and scope

Add a third communications model:

```text
ScenarioSpec.comm_model = "voice_only"
```

In this model the friendly force has no individual radios once the episode
starts. Orders, reports, acknowledgements and coordination signals can be
spoken only at very short range. Information never propagates implicitly.
When distance separates the chain of command, a soldier must close the gap or
a commander must detach an **agent of liaison** carrying a fixed message.

This is a command-and-control degradation, not a generic difficulty switch.
Terrain, weapons, sight, OpFor, mission-completion predicates and rank
authority remain unchanged unless this document names an interaction
explicitly.

### Decisions of record

| Question | Decision |
|---|---|
| Mode name | `voice_only`; do not overload `range` |
| First-cut speaking radius | `voice_range=2.0` cells, Euclidean |
| Radio after step 0 | None, including no privileged HQ station |
| General voice audience | Every living friendly in range may hear; command effects still apply only to the addressed recipient |
| Terrain acoustics | Distance only in V1; no invented wall/forest attenuation |
| Simultaneous speech | No global `NET BUSY`; spatially separate conversations may occur in the same tick |
| Initial OPORD | Pre-mission briefing/initial condition, not an in-episode radio transmission |
| Remote HQ during play | Absent; post-reset `issuer="HQ"` injection is unavailable |
| Message carrier | A temporary `LiaisonTask`, not a new `MissionType`/MICAT task |
| Message representation | Immutable canonical voice-procedure text plus routing metadata; no live world state in the payload |
| Existing modes | `global` and `range` must remain behaviorally identical when the new mode is not selected |
| Enemy hearing | Out of V1 scope; low voice range represents discretion without adding an OpFor audio sensor |

### Explicit non-goals

- Do not add prediction, imagination or conceptualisation of another agent's
  future actions.
- Do not expose a subordinate's chosen action, policy logits, latent state or
  intended route.
- Do not add recurrence or a learned belief model as part of this change.
- Do not redesign vision, friendly sensing, formations, combat or the MICAT
  derivation table.
- Do not make voice automatically detectable by OpFor in V1. That is a
  separate sensor-and-stealth experiment and would confound the C2 result.

The later "imagination" project can therefore compare against a clean
voice-only baseline. Physical message memory and an assigned destination are
allowed here; inferred teammate behaviour is not.

---

## 1. Why the current `range` model is not enough

The relevant implementation is currently split across:

- `cohort/config.py::ScenarioSpec` (`comm_model`, `comm_range`,
  `voice_range`, `auto_ack`, `sitrep_cadence`);
- `cohort/env/cohort_env.py::_arbitrate_net`, `_audible_to`,
  `_report_contact`, `_report_done`, `_issue_order`, `_assign_mission`,
  `_sync_propose`, `_sync_go`, `_say` and `inject_order`;
- `cohort/env/actions.py::compute_mask`;
- `cohort/env/observations.py` (leader, subordinate and communications
  blocks);
- `cohort/core/orders.py::Message` and `Transcript`;
- `cohort/env/rewards.py::RewardConfig`;
- `cohort/metrics.py`, `cohort/probe.py` and the episode visualisations;
- `tests/test_comms_range.py`, `tests/test_comms_discipline.py`,
  `tests/test_voice_is_charged.py`, `tests/test_orders_flow.py` and
  `tests/test_sitrep_cadence.py`.

Current behavior:

| Mechanic | `global` | `range` today | Required `voice_only` |
|---|---|---|---|
| Medium | Perfect radio | Range-limited radio | Low voice only |
| Radius | Whole map | `comm_range=12` by default | `voice_range=2` initially |
| HQ | Globally heard and hears the root | Same privileged exception | No in-episode remote station |
| Frequency | One global net | Still one global net | No global net |
| Contention | One learned transmission/tick globally | Same | No global arbitration |
| CONTACT picture | Shared team picture | Per-listener inside radio range | Per-listener, one spoken hop at a time |
| Remote ORDER | Always lands | Is transmitted but not received | Is not spoken; it needs proximity or a courier |
| SITREP / DONE | Umpire adjudicates regardless of range | Same bypass remains | Cannot bypass distance |
| Auto traffic | Global protocol traffic | Largely unchanged by range | Local speech or external event, never global knowledge |
| Existing voice | `SYNC_PROPOSE` / `SYNC_GO` only | Same | Every C2 utterance uses voice |

The existing `range` model is a useful control: it tests radio audibility. It
does not test loss of radio, physical command presence or message carrying.

### Current-state hazards the implementation must not copy

1. `_audible_to()` treats HQ as high power. That exception is forbidden in
   `voice_only`.
2. `_report_done()` and several automatic messages currently receive umpire
   treatment. They must not become covert long-range communications.
3. `_last_net_contact_step` is global and waives re-task pricing across the
   force. Under voice only, a tactical-picture change is local to the issuer.
4. `build_observation()` exposes exact direct-leader and direct-subordinate
   positions and missions independently of the communications model. Do not
   expand that omniscience in this cycle. Record it as a known abstraction;
   changing it belongs with the later inference work.
5. `voice_peers()` is a trinome-specific eligibility function. General
   speech needs a separate range predicate; nearby speech is audible even
   when the listener is not a valid bounding peer.
6. Some comments still describe voice as free even though issue #18 charges
   `SYNC_PROPOSE` and `SYNC_GO`. The code and
   `tests/test_voice_is_charged.py` are authoritative: every learned speech
   act must remain charged.

---

## 2. Doctrinal basis and the boundary against invention

The project reference is `docs/manuel-proterre.pdf`; page numbers below are
the manual's printed page numbers.

- **p. 14, movement by bounds:** bounds are commanded "a la voix ou aux
  gestes", and spacing in column must preserve voice/gesture command.
- **p. 20, group command:** during action the group commander uses voice,
  gestures, imitation, and visual or sound signals; the same section requires
  the commander to keep the superior informed so that new orders or support
  can be received.
- **p. 24, orders during action:** changed circumstances produce an order in
  action containing only changed clauses; corrections and impulses are
  delivered by voice or preferably gesture, briefly and precisely.
- **p. 61, section OPORD:** `QUINTO - COMMANDEMENT / TRANSMISSIONS` explicitly
  plans command locations, transmission instructions and liaisons.
- **p. 63, reports:** the section commander reports regularly using "Je
  suis / Je vois / Je fais / Je demande" and reports again at mission end.
- **p. 72, SOUTENIR:** the link with the supported element must be reliable,
  preferably visual; where necessary an **agent de liaison** may be detached
  to the supported element.
- **p. 80, command/liaison aide-memoire:** the order states the radio regime
  as silence, discretion or free, and even verbal orders are to be written.
- **p. 125, combat in built-up areas:** radio links are described as
  precarious; compartmentalisation leads to command in depth and strong
  decentralisation.

These references support a low-voice command regime, fewer but still
necessary reports, written/fixed orders for carriage, and a detached liaison
agent. They do **not** support inventing a twelfth/thirteenth tactical mission
inside the MICAT catalogue.

Therefore:

- keep `MissionType` and `DOCTRINE` unchanged;
- model liaison as a temporary command/transmission duty, parallel to the
  standing tactical mission (as formation already is a stance rather than a
  mission);
- while detached, suspend tactical compliance and score liaison compliance;
- restore the suspended mission after the liaison cycle, unless a new valid
  order received during delivery supersedes it.

The soldier is functionally assigned exclusively to carrying the message,
but the project does not falsely label that duty as a MICAT mission.

---

## 3. Operational semantics

### 3.1 Audibility

Define one general predicate, distinct from `voice_peers()`:

```text
voice_audible(sender, listener) :=
    sender and listener are alive
    and euclidean_distance(sender.pos, listener.pos) <= voice_range
```

V1 deliberately has no LOS or terrain modifier. At two cells the abstraction
is close speech, not shouting. A later acoustics feature may test walls,
forest, alerting OpFor and a loud/quiet choice, but none belongs in this
baseline.

Every utterance is appended to the external transcript, tagged as voice. The
environment must separately record the callsigns that actually heard it in
the evaluation trace/oracle metadata. `heard_by` is audit metadata, not
message content and never enters an agent observation wholesale.

All friendly listeners in range may update their own knowledge from a spoken
CONTACT. Only the addressed recipient may receive an order, confirm a report
or change command state. Overhearing an order is information, not authority.

### 3.2 No radio channel and no global arbitration

`voice_only` has no shared frequency, so `_arbitrate_net()` must not drop a
speech action because somebody elsewhere is speaking. Multiple utterances
may occur in the same tick. Keep the one-action-per-agent rule: an agent may
move, fire, speak, prepare/hand off a message, or issue an order, not several
of them at once.

Do not introduce local voice collisions in V1. They would add a second
contention experiment before the basic propagation mechanics are measured.
Volume is still priced by `transmission_cost`; the name may remain for
checkpoint/config compatibility, but its documentation must call it a cost
per learned communication act, not radio airtime only.

### 3.3 Initial orders and the absent HQ

The step-0 OPORD remains the root's initial mission. In this mode it represents
a pre-departure briefing, before the force loses radio contact. It may remain
as the first transcript line for compatibility, but trace metadata must mark
its medium as `briefing`, not `voice` or `radio`, and it must not update every
agent's local picture.

After reset:

- HQ has no position and cannot be within voice range;
- `inject_order(..., issuer="HQ")` and `inject_execute("HQ")` must fail with a
  clear mode-specific error;
- a human can command as the embodied root callsign, subject to the same
  proximity rule as an agent;
- scenario success/failure remains external adjudication, but an absent HQ
  must not speak `ENDEX` into every agent's world.

For V1 voice-only scenario presets, set `root_done_bonus=0` and make root-to-HQ
CONTACT/SITREP/DONE opportunities structurally unavailable. Metrics must
report those channels as unavailable (`null` plus zero opportunity), not call
the root mute. Adding a physical command post, extraction point or return-to-
HQ objective is a separate scenario design.

### 3.4 Message-kind rules

| Message/event | Voice-only rule |
|---|---|
| OPORD | Pre-mission briefing to the root at reset |
| ORDER / FORMATION | Applies only when the addressed subordinate hears it directly or receives the exact order from a liaison carrier |
| ACK / WILCO | Local automatic reply after an order actually lands; never generated for an undelivered order |
| CONTACT | Spoken to a nearby superior; nearby friendlies may overhear and update their local pictures |
| SITREP | Spoken to the direct superior at delivery time; no range bypass |
| DONE | Spoken/delivered to the direct superior; only then is it adjudicated and answered |
| DONE_CONFIRM / DONE_REJECT | Local reply by the superior who received the claim |
| EXECUTE | Spoken locally; releases only this issuer's pending recipients that hear this occurrence |
| SYNC_PROPOSE / SYNC_GO | Existing voice mechanics, using the mode's short radius and the existing doctrinal peer restriction |
| SUPPORT_END | Local report or carried packet; mission state may clear when its supported unit dies, but the leader does not learn globally |
| CASUALTY / TRAP | External ground-truth event plus local alerts from actual witnesses; never an HQ all-stations broadcast |
| TAKING_COMMAND | Roster succession still occurs structurally; its announcement is local and must not grant distant knowledge |
| ENDEX | External episode outcome only; no audible all-stations line from absent HQ |

Automatic protocol may remain zero-cost because it is not policy-chosen, but
"automatic" never means "globally heard". Every automatic utterance still
passes the voice audience rule.

### 3.5 Knowledge is local and store-and-forward

In `voice_only`, every agent has a local enemy picture. It is updated only by:

1. that agent's own sighting, stored at the observed position and time; or
2. a CONTACT utterance the agent actually heard.

The entry is a stale-able memory. It must never track an enemy's live position
after the sighting/message. Repeating or carrying the report preserves the
captured coordinates and source time. This is physical message memory, not
the future teammate-imagination feature.

CONTACT becomes legal when the agent holds unexpired contact information and
can reach its intended superior by voice. It is no longer restricted to an
enemy visible on that exact tick. This is necessary for a scout to observe,
withdraw and report, and for reports to move one echelon at a time.

Novelty and re-task exceptions are listener-local:

- `contact_new` is earned when the intended superior receives information it
  did not have;
- an aging refresh is evaluated against that superior's picture;
- a leader's free re-task exception uses the time its **own** picture changed,
  never `_last_net_contact_step` for the whole force.

---

## 4. Liaison duty and physical message packets

### 4.1 Packet invariant

A `MessagePacket` is a single, immutable piece of information:

```text
packet id
origin callsign and command position
intended recipient command position (plus callsign at creation)
message kind
canonical formatted text
creation step and source-observation step, where applicable
expiry rule
acknowledgement required
status: held / dispatched / delivered / returning / lost / expired / cancelled
```

The canonical text is authoritative. Routing/status metadata is simulation
bookkeeping; it must not smuggle enemy ground truth, future positions or
structured oracle facts into the message. Parsers must be able to reconstruct
every delivered order/report from the text exactly as they do for direct
speech.

One issuer may hold at most one prepared outgoing packet and one soldier may
carry at most one packet. This makes the bottleneck visible and bounds both
the observation and action space. A packet has one owner at a time; dispatch
moves it, never copies it.

### 4.2 Preparing a message without doubling the order catalogue

Do not duplicate the full order catalogue into `WRITE_*` actions. In
`voice_only`, reuse the existing communication choice:

- if the addressed recipient is in voice range, the action speaks and applies
  normally;
- if the recipient is out of range and the sender's outbox is empty, the
  action spends the tick preparing the canonical packet but emits no
  transcript message, charges no communication cost and has no remote effect;
- while an outbox is occupied, mask further packet-producing actions except a
  dedicated `CANCEL_MESSAGE` action; cancellation is an internal command
  decision and must be counted as churn, not as speech.

For CONTACT, the packet captures the selected held-intel coordinates and
source time. For SITREP it captures the sender's state at preparation; delay
must therefore be visible as age, not silently refreshed. For DONE it captures
the claimant, task and claim time; the claim is heard and adjudicated only on
delivery. For ORDER it captures the complete directive and timing clause.

An agent may carry its own packet by moving to the recipient and selecting
`DELIVER_MESSAGE`; detaching another soldier is optional, not mandatory.

### 4.3 Dispatching an agent of liaison

Add four bounded actions, one per existing direct-subordinate slot:

```text
DISPATCH_LIAISON_S0 ... DISPATCH_LIAISON_S3
```

The action is legal only when:

- the issuer has a prepared packet;
- the selected direct subordinate is alive, within voice range, is not the
  packet's intended recipient and carries no packet;
- neither issuer nor carrier is already in an incompatible liaison handoff.

Dispatch is itself a local order and consumes the issuer's tick. The carrier
receives a `LiaisonTask`, the packet changes owner, and the carrier's standing
tactical mission is suspended. Do not insert `LIAISON` into `MissionType` or
the MICAT one-hot.

The carrier may STAY, MOVE, FIRE in immediate self-defence, or
`DELIVER_MESSAGE`. Other command/report/sync actions are masked until the
outbound packet is delivered or the liaison is cancelled. Personal sightings
may still enter the carrier's memory, but do not replace the entrusted packet.

### 4.4 Finding the recipient without omniscience

The liaison destination is the recipient's last known position at dispatch,
not a live tracking beacon. The liaison observation contains:

- delta to that fixed rendezvous/last-known anchor;
- whether the intended recipient is currently perceived nearby;
- if perceived, delta to the actually perceived recipient;
- packet age, kind and outbound/return state.

Do not expose a live remote target delta. At the anchor, the carrier may need
to search or wait. This is the intended cost of lost communications. The
initial version may use the existing friendly-visibility abstraction for
local recognition; it must not use the oracle.

Address packets to a **command position**, not only a mortal soldier id. If
succession fills that position before delivery, the current holder may receive
the packet. If the position is vacant, delivery fails and the carrier begins
the return leg with an undeliverable notice. This keeps succession meaningful
without clairvoyantly redirecting the carrier en route.

### 4.5 Delivery and receipt

`DELIVER_MESSAGE` is legal only when the current packet recipient is within
voice range. Delivery creates a normal formatted voice message and applies the
same validation as direct speech at that tick.

- An ORDER starts tenure/cooldown and changes the recipient's mission only at
  delivery. The recipient's local WILCO becomes a receipt carried back to the
  origin; the order already stands even if the courier dies on return.
- CONTACT/SITREP/DONE are credited and update state only at delivery.
- An invalid or obsolete order is rejected locally and returns a negative
  receipt; do not silently reinterpret it.
- A delivered factual report may be stale. Staleness is a measured property,
  not a reason to replace its coordinates with current ground truth.

For an ORDER, the liaison cycle completes only when the carrier returns the
WILCO/negative receipt to the origin. For a report, outbound delivery completes
the assigned liaison duty in V1; a generic report receipt may be added later
if evidence shows the open loop matters.

On completion, restore the suspended tactical mission unless the carrier
received a newer valid order at the destination. A packet carried by a dead
soldier is lost. No automatic copy survives and no distant agent learns that
it was lost.

---

## 5. Spaces and policy inputs

The final liaison-capable mode is a deliberate space break.

### New actions

- `DELIVER_MESSAGE`
- `CANCEL_MESSAGE`
- `DISPATCH_LIAISON_S0 ... S3`

Keep all existing actions and their order stable; append the new actions so
old indices do not move even though old checkpoints still become incompatible
with the larger actor head.

### New observation block

Add an explicit, bounded liaison/message block rather than overloading an
unrelated field:

- mode is voice-only;
- outbox present and packet-kind one-hot;
- carrying packet and packet-kind one-hot;
- packet age/TTL remaining;
- liaison outbound vs return;
- fixed-anchor `dx`, `dy`;
- intended recipient locally perceived, with perceived `dx`, `dy`;
- delivery currently possible;
- receipt positive/negative when returning.

The local known-enemy summary remains per-agent. If full carried-contact slots
are needed for correct store-and-forward behavior, add bounded stale records
with `(present, dx, dy, age)` and document the extra width. Never expose a
fresh coordinate for an unseen enemy.

The policy remains the shared feed-forward MLP initially. Explicit packet and
destination state makes the liaison process Markov enough without importing
the future imagination/recurrent-policy project.

Because this repository expects checkpoints to transfer across scenarios,
append the new fields/actions for every communications model and mask them
outside `voice_only`. That forces one honest fleet-breaking cycle instead of
creating incompatible scenario-specific network shapes.

---

## 6. Reward semantics

The reward must price delivered information, not button presses.

### Keep

- `transmission_cost` on every emitted learned utterance, including dispatch
  and delivery speech;
- existing truth/falsehood checks;
- existing doctrine, order-quality, re-task and coverage economics;
- shared terminal rewards and all combat economics.

### Change in `voice_only`

1. **No reward on packet preparation.** Nothing was communicated.
2. **Order reward at receipt.** `order_preferred`, `order_allowed`, objective
   match, churn and re-task effects occur when the addressed subordinate
   actually receives the order, not when it is written or handed to a courier.
3. **Information value to the origin.** CONTACT/DONE truth value is credited
   to the packet origin on successful delivery, even if a courier speaks the
   final line.
4. **Liaison credit to the carrier.** Pay bounded progress and delivery credit
   to teach the long physical chain:
   - recommended first cut: `liaison_progress=0.03` for each new best cell of
     closure toward the current fixed anchor, watermarked per packet;
   - `liaison_delivery=0.5` on accepted outbound delivery;
   - `liaison_receipt_return=0.25` when an ORDER receipt reaches its origin.
5. **No proximity/blob reward.** Merely staying near a leader earns nothing.
   Cohesion must remain a tactical trade, not a universal optimum.
6. **No extra packet-loss penalty initially.** Existing death, teammate-death
   and lost opportunity already price loss; add a penalty only after an oracle
   diagnosis shows deliberate packet abandonment.
7. **Coverage understands liaison.** A detached subordinate counts as tasked,
   but its suspended tactical mission earns no tactical compliance while the
   liaison duty is active.
8. **Local change exception.** Re-task-cost waivers depend on the issuer's
   received picture/casualties, not on global transcript events.
9. **Root reporting unavailable.** Voice-only presets set
   `root_done_bonus=0`; do not punish a root for lacking a non-existent HQ
   channel.

The numeric liaison values are starting hypotheses, not doctrine. Before
training, update `max_step_farm()` and terminal-dominance tests. Progress must
be telescoping/watermarked so walking back and forth cannot farm it.

---

## 7. Implementation sequence for Claude

### Phase A - direct voice-only probe (no space break)

Purpose: validate the medium semantics and measure how severe the physical
coordination problem is before adding carriage machinery.

1. Validate `comm_model` against `global | range | voice_only`.
2. Add general voice audibility and route every communication kind through
   the rules in section 3.
3. Disable global arbitration and all HQ/automatic range bypasses in this
   mode.
4. Make CONTACT pictures and tactical-change clocks per listener/issuer.
5. Register `squad_voice_direct` as an exact `squad` mirror except
   `comm_model="voice_only"`, `voice_range=2.0`, and root-report economics.
6. Add trace/metrics for voice reach and direct delivery opportunities.
7. Train one scout seed. Diagnose with the oracle before changing rewards.

This phase is an experiment, not the finished mode: it deliberately has no
way to send a separate messenger.

### Phase B - liaison-capable mode (breaking)

1. Add packet and `LiaisonTask` domain state with formatter/parser round trips.
2. Append actions and the explicit observation block.
3. Implement prepare, self-carry, dispatch, delivery, order receipt return,
   cancellation, expiry, loss and succession-address resolution.
4. Implement bounded rewards and terminal-dominance checks.
5. Add `squad_voice_liaison`, identical to `squad_voice_direct` except liaison
   actions are enabled. Keep a same-space `squad_global_control` on the new
   layout.
6. Update play, dashboard, renderer, trace, oracle and briefing so packet
   status, medium and actual hearers are auditable.
7. Retrain the matched controls and voice arms from scratch.

### Phase C - depth and hostile-terrain evaluation

Only after squad mechanics are sound:

1. mirror the arms on `platoon` to test multi-echelon propagation;
2. mirror the liaison arm on `patrol_brique` to test the intended hostile,
   compartmentalised setting;
3. do not add enemy hearing, acoustics or imagination in the same campaign.

### Honest adjustment rule

Per the repository's normal discipline: one full training per arm, one
oracle-diagnosed adjustment if a mechanism is clearly broken, then document
the miss and stop. Do not reward-tune until a silent policy happens to use
couriers.

---

## 8. Tests and invariants

### Regression safety

- `global` seeded episodes remain bit-identical.
- `range` seeded episodes remain bit-identical.
- action indices before the appended actions do not move.
- message formatter/parser inverses and text-only schema stay valid.
- all randomness still flows through `env._rng`.

### Voice-only mechanics

- nobody beyond `voice_range` receives any direct or automatic utterance;
- no HQ exception exists after reset;
- two distant speakers may both be heard by their local audiences in one tick;
- an out-of-range ORDER changes no mission and produces no WILCO;
- an out-of-range report changes no superior picture and receives no report
  reward;
- `EXECUTE` releases only pending recipients that heard it;
- direct speech is overheard by every local friendly but only the addressee
  receives command effects;
- root-to-HQ actions are masked/unavailable, not emitted into the void;
- auto CASUALTY/TRAP/succession/ENDEX traffic never creates global knowledge.

### Packet and liaison hazards

- preparing a packet emits no message, costs no communication charge and has
  no remote effect;
- packet text and captured coordinates never change in transit;
- a courier receives exactly one packet and dispatch removes the issuer copy;
- delivery is impossible outside voice range;
- order tenure/cooldown/reward starts at delivery, not preparation/dispatch;
- CONTACT reward uses the intended superior's pre-delivery picture;
- a courier's progress reward cannot be re-earned over walked ground;
- courier death loses the packet with no invisible backup;
- delivery to a succeeded command position reaches the current holder, not a
  dead id;
- an undeliverable position returns a negative receipt rather than silently
  retargeting;
- suspended tactical compliance does not pay during liaison duty;
- completing/cancelling liaison restores exactly the permitted prior mission;
- a stale carried CONTACT never follows the enemy through the map;
- liaison fields contain no subordinate-action or future-intent information.

### Documentation/provenance

- `briefing()` publishes `comm_model`, `voice_range`, HQ availability,
  liaison enabled/disabled and packet TTLs;
- run `config.json`/`economics.json` records every communication and reward
  parameter;
- dashboards and transcripts label `briefing`, `radio`, `voice` and external
  umpire events distinctly;
- metrics use `null` for structurally unavailable channels.

---

## 9. Evaluation design

### Hypotheses

1. Direct voice makes reports and re-tasking less frequent because delivery
   requires physical contact.
2. The cost grows with echelon depth: fireteam < squad < platoon.
3. Direct voice increases standing-order tenure and decentralised execution,
   but also increases stale orders and cross-echelon information latency.
4. Liaison restores part of the outcome/information gap without recreating a
   hidden global channel.
5. Courier use has a real opportunity cost in tactical manpower and exposure.

### Required new metrics

| Metric | Definition |
|---|---|
| `voice_utterances` | Learned speech acts emitted |
| `voice_hearers_mean` | Mean living friendly hearers per utterance, sender excluded |
| `direct_delivery_rate` | Addressed messages received directly / attempted or prepared |
| `command_pair_in_voice_rate` | Leader-direct subordinate pairs in voice range per step |
| `orders_prepared/dispatched/delivered/lost` | Packet funnel with counts, never a single rate without denominators |
| `order_delivery_latency` | Preparation to recipient receipt; censored on loss/end |
| `order_receipt_return_latency` | Delivery to WILCO reaching origin |
| `reports_prepared/delivered/expired` | Report packet funnel |
| `contact_to_leader_latency` | First sighting to direct superior receiving it |
| `contact_to_root_latency` | First sighting to root receiving the information, with censored count |
| `fresh_picture_coverage` | Share of living friendlies holding each fresh contact over time |
| `liaison_assignments/completions/losses` | Courier outcome counts |
| `liaison_agent_step_share` | Share of friendly agent-steps spent detached |
| `liaison_distance` | Outbound and return path lengths |
| `orders_stale_at_delivery` | Delivery after issuer/recipient situation changed, with definition recorded |

Retain all existing success, defeat, timeout, casualty, coverage, compliance,
order-tenure, report precision/recall, DONE and transparency metrics. Scope
their denominators to what was physically possible.

### Matched experiment

On the final common action/observation layout:

| Arm | Communications | Liaison |
|---|---|---|
| `squad_global_control` | Current global radio | Masked |
| `squad_range_control` | Current range radio (`comm_range=12`) | Masked |
| `squad_voice_direct` | Voice only (`voice_range=2`) | Masked |
| `squad_voice_liaison` | Voice only (`voice_range=2`) | Enabled |

- Same map, OpFor, org, PPO settings, total steps and reward defaults except
  the documented root/liaison terms.
- Start with one scout seed for mechanics, then seeds 12/13/14 for a claim.
- Evaluate both best and final checkpoints at N=100, seeds 500-599.
- Report 95% intervals and paired/exact tests using the repository's existing
  tooling; do not compare one seed's point estimates as a conclusion.
- Advance to platoon only after the squad packet funnel shows that delivery is
  reachable and not an exploit.

### Acceptance is not "match global radio"

This mode is meant to be degraded. A lower raw success rate can be the correct
result. The implementation is accepted when:

1. the structural tests prove no long-range information leak;
2. the direct arm measurably reduces reach and increases propagation latency;
3. the liaison arm actually delivers messages in evaluated episodes, or its
   failure is documented after the one diagnosed adjustment;
4. success, casualties, timeouts, message latency and manpower cost are all
   reported together;
5. no conclusion claims that liaison "helps" unless it improves a registered
   outcome or information metric with uncertainty reported;
6. all existing modes and doctrine invariants remain green.

---

## 10. Expected project meaning

The mode changes the learned problem in the intended direction:

- commanders must invest in physical presence or accept delegated execution;
- standing orders matter longer because correction is slow;
- a report is valuable only if it reaches somebody who needs it;
- casualties may partition the command graph without any omniscient alarm;
- a liaison soldier trades combat power and time for information flow;
- the transcript remains human-readable, but an external reader must now
  distinguish what was said from who could hear it.

That is a stronger test of this project's central claim than merely shrinking
a radio radius. It creates a real cost for command, reporting and common
knowledge while preserving the doctrinal mission catalogue and leaving the
future teammate-imagination hypothesis cleanly outside the experiment.
