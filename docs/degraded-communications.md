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
Speech, movement and weapons also create **sound events** that either side may
detect: the same voice that restores coordination can disclose a presence to
the enemy. When distance separates the chain of command, a soldier must close
the gap or a commander must detach an **agent of liaison** carrying a fixed
message.

This is a command-and-control and signature-management degradation, not a
generic difficulty switch. Terrain, weapons, sight, OpFor,
mission-completion predicates and rank authority remain unchanged unless this
document names an interaction explicitly. The required interactions are the
acoustic layer, the OpFor reaction to sound, and local visual cohesion.

### Decisions of record

| Question | Decision |
|---|---|
| Mode name | `voice_only`; do not overload `range` |
| First-cut speaking radius | `voice_range=2.0` cells for intelligible low voice |
| Radio after step 0 | None, including no privileged HQ station |
| General voice audience | Every living friendly in range and LOS may hear; command effects still apply only to the addressed recipient |
| Acoustic feature switch | Independent `sound_model="off" \| "tactical"`; the operational voice-only presets require `"tactical"`, while `"off"` exists only for regression and ablation |
| Acoustic signature | Successful movement, voice, coordination signals, weapon fire and traps create local sound events for both sides |
| Terrain acoustics | Movement source strength differs on OPEN and FOREST; walls/forest modify propagation using a deliberately small, published model |
| Voice risk | Friendly intelligibility and hostile detectability are different radii; detecting speech never grants its words or the speaker's exact cell |
| Deliberate signals | Existing `EXECUTE` and `SYNC_GO` are the initial pre-arranged sound signals; silent gesture variants use the same meanings and require visual contact |
| Simultaneous speech | No global `NET BUSY`; spatially separate conversations may occur in the same tick |
| Initial OPORD | Pre-mission briefing/initial condition, not an in-episode radio transmission |
| Remote HQ during play | Absent; post-reset `issuer="HQ"` injection is unavailable |
| Message carrier | A temporary `LiaisonTask`, not a new `MissionType`/MICAT task |
| Message representation | Immutable canonical voice-procedure text plus routing metadata; no live world state in the payload |
| Formation/cohesion | A continuous same-element visual-link graph is observed and scored; it is not a universal hard movement mask |
| Friendly position knowledge | Exact live leader/subordinate deltas are local-perception only in this mode; otherwise the policy gets stale last-known state plus age |
| Existing modes | `global` and `range` must remain behaviorally identical when the new mode is not selected |
| Enemy hearing | Essential to the mode; OpFor may investigate an uncertain acoustic cue but may fire only after visual detection |

### Explicit non-goals

- Do not add prediction, imagination or conceptualisation of another agent's
  future actions.
- Do not expose a subordinate's chosen action, policy logits, latent state or
  intended route.
- Do not add recurrence or a learned belief model as part of this change.
- Do not implement the directional/foveated vision design from `vision.md` in
  this cycle. Local friendly visibility reuses the shipped 360-degree LOS
  abstraction with a finite range.
- Do not add a new MICAT mission for sound, silence, formation or liaison.
- Do not add continuous decibels, frequency bands, weather, ambient noise,
  individual hearing acuity, equipment load or speech recognition.
- Do not let a sound cue identify a callsign, disclose message content, reveal
  an exact grid cell or make `FIRE` legal without a visible target.
- Do not add arbitrary whistle codes. Only meanings already present in the
  command model are eligible for the first signal catalogue.

The later "imagination" project can therefore compare against a clean
voice-only baseline. Physical message memory, last-perceived friendly state
and an assigned destination are allowed here; inferred teammate behaviour is
not.

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
- `cohort/core/world.py` (OPEN/FOREST/WALL, LOS and local terrain);
- `cohort/core/missions.py::in_formation` and
  `cohort/core/units.py::{voice_peers, enemy_decide, Enemy}`;
- `cohort/core/orders.py::Message` and `Transcript`;
- `cohort/env/rewards.py::RewardConfig`;
- `cohort/metrics.py`, `cohort/probe.py` and the episode visualisations;
- `tests/test_comms_range.py`, `tests/test_comms_discipline.py`,
  `tests/test_voice_is_charged.py`, `tests/test_orders_flow.py` and
  `tests/test_sitrep_cadence.py`, plus new focused acoustic/cohesion suites.

Current behavior:

| Mechanic | `global` | `range` today | Required `voice_only` |
|---|---|---|---|
| Medium | Perfect radio | Range-limited radio | Low voice only |
| Radius | Whole map | `comm_range=12` by default | `voice_range=2` initially |
| Sound footprint | None | None | Movement/voice/signals/fire/traps create detectable local cues |
| HQ | Globally heard and hears the root | Same privileged exception | No in-episode remote station |
| Frequency | One global net | Still one global net | No global net |
| Contention | One learned transmission/tick globally | Same | No global arbitration |
| CONTACT picture | Shared team picture | Per-listener inside radio range | Per-listener, one spoken hop at a time |
| Remote ORDER | Always lands | Is transmitted but not received | Is not spoken; it needs proximity or a courier |
| SITREP / DONE | Umpire adjudicates regardless of range | Same bypass remains | Cannot bypass distance |
| Auto traffic | Global protocol traffic | Largely unchanged by range | Local speech or external event, never global knowledge |
| Existing voice | `SYNC_PROPOSE` / `SYNC_GO` only | Same | Every C2 utterance uses voice |
| Friendly telemetry | Exact live leader/subordinate state | Same | Current local perception plus aging last-known state |
| Formation link | Soft geometry bonus only | Same | Continuous local visual-link graph plus soft rupture penalty |

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
   positions and missions independently of distance or LOS. Keeping that live
   telemetry would nullify the proposed visual-link problem. In `voice_only`,
   gate it behind local friendly perception and otherwise expose only bounded,
   aging last-known state. This is removal of oracle information, not the later
   teammate-inference feature.
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
- **p. 32, ECLAIRER:** the group moves discreetly, adapts formation to terrain,
  commands by gesture as often as possible, observes by sight and hearing, and
  aims to detect the enemy without itself being detected.
- **p. 35, APPUYER:** the supporting element keeps visual liaison throughout
  the mission and pre-arranges signals for shifting fire; observation is by
  both sight and hearing.
- **p. 43, section movement:** at night or in poor visibility the section
  closes distances and intervals to preserve links; the commander should be
  able to command by sight where possible, and formation adapts to situation,
  terrain and visibility.
- **p. 61, section OPORD:** `QUINTO - COMMANDEMENT / TRANSMISSIONS` explicitly
  plans command locations, transmission instructions and liaisons.
- **p. 63, reports:** the section commander reports regularly using "Je
  suis / Je vois / Je fais / Je demande" and reports again at mission end.
- **p. 72, SOUTENIR:** the link with the supported element must be reliable,
  preferably visual; where necessary an **agent de liaison** may be detached
  to the supported element.
- **p. 80, command/liaison aide-memoire:** the order states the radio regime
  as silence, discretion or free, and even verbal orders are to be written.
- **pp. 119-120, night combat:** darkness increases the importance of hearing,
  makes maintaining direction, links and command harder, requires silence,
  and makes noisy movement easier to locate by listening. The explicit example
  concerns vehicles; terrain-dependent infantry footstep strength below is a
  transparent simulation hypothesis, not a quoted doctrinal constant.
- **p. 125, combat in built-up areas:** radio links are described as
  precarious; compartmentalisation leads to command in depth and strong
  decentralisation.

These references support a low-voice command regime, silent gestures where
possible, sound as both information and disclosure, continuous local links,
fewer but still necessary reports, written/fixed orders for carriage, and a
detached liaison agent. They do **not** prescribe cell radii or acoustic
coefficients, and they do **not** support inventing a twelfth/thirteenth
tactical mission inside the MICAT catalogue.

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

Separate **understanding a message** from merely **detecting a sound**. Define
one general friendly-speech predicate, distinct from `voice_peers()`:

```text
voice_audible(sender, listener) :=
    sender and listener are alive
    and euclidean_distance(sender.pos, listener.pos) <= voice_range
    and world.line_of_sight(sender.pos, listener.pos)
```

At two cells the abstraction is low, intelligible speech. The mode deliberately
does not duplicate every message action at several volumes. A separate,
longer-range pre-arranged sound signal carries only a fixed command code; it
cannot carry an OPORD, grid reference or SITREP. This gives the policy a real
choice among silent gesture, low voice, a conspicuous simple signal, physical
movement and liaison without multiplying the complete order catalogue.

Every utterance is appended to the external transcript, tagged as voice. The
environment must separately record the callsigns that actually heard it in
the evaluation trace/oracle metadata. `heard_by` is audit metadata, not
message content and never enters an agent observation wholesale.

All friendly listeners in range may update their own knowledge from a spoken
CONTACT. Only the addressed recipient may receive an order, confirm a report
or change command state. Overhearing an order is information, not authority.
An enemy inside the larger acoustic-detection footprint gets a coarse sound
cue only; it does not become a friendly-language listener.

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
| ACOUSTIC_CONTACT | Coarse spoken/carried report of a heard presence or indication; preserves kind, bearing/distance bands and source time, never invents a grid reference |
| SITREP | Spoken to the direct superior at delivery time; no range bypass |
| DONE | Spoken/delivered to the direct superior; only then is it adjudicated and answered |
| DONE_CONFIRM / DONE_REJECT | Local reply by the superior who received the claim |
| EXECUTE | Pre-arranged sound signal; releases only this issuer's pending recipients that interpret this occurrence; gesture variant is silent |
| SYNC_PROPOSE / SYNC_GO | Proposal uses low voice; GO is a pre-arranged sound signal using the existing doctrinal peer restriction; gesture variant is silent |
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

### 3.6 Tactical acoustic environment

The acoustic layer is part of the final mode, not optional decoration. Keep it
independently configurable so the shipped modes remain reproducible and an
ablation can isolate its causal effect:

```text
ScenarioSpec.sound_model = "off" | "tactical"
```

`global` and `range` default to `off`. Operational `voice_only` presets set
`tactical`; a voice-only run with sound disabled must be labelled an ablation,
never the completed degraded mode.

#### 3.6.1 One event, two very different products

Create one immutable `SoundEvent` at its physical source:

```text
event id and step
source position and side (oracle only)
kind: movement | voice | signal | weapon_fire | trap
base detection radius
optional friendly semantic payload reference
actual listeners and received-strength bands (trace/oracle only)
```

The event produces two separate products:

1. eligible friendlies may understand the semantic voice/signal payload; and
2. either side may receive a non-semantic `AcousticCue`.

Never put the oracle source id, exact source position or `heard_by` list into
the cue. A friendly hearing an enemy movement and an enemy hearing a Blue
SITREP get the same class of uncertain sensor fact, not privileged prose.

| Trigger | Sound event | Semantic content |
|---|---|---|
| Successful MOVE by either side | `movement` at the traversed edge; use the noisier endpoint terrain | None |
| Emitted voice, including local automatic replies and liaison delivery | `voice` at the speaker | Only eligible friendly listeners receive the existing canonical message |
| `EXECUTE` or `SYNC_GO` in their acoustic form | `signal` at the emitter | The fixed, pre-briefed code only |
| FIRE by either side, hit or miss | `weapon_fire` at the shooter | None; it never identifies the shooter |
| Trap activation | `trap` at the trap cell | None; witnesses may separately report what they perceived |
| STAY, packet preparation or a visual gesture | No sound | None |

A blocked/illegal action emits nothing. Do not invent breathing, casualty
cries or equipment noise in the first version.

#### 3.6.2 Transparent first-cut propagation

Use a deterministic threshold model first. It is easier to test, audit and
learn than an uncalibrated probability curve. These are **simulation starting
hypotheses**, not doctrinal measurements:

| Source | Friendly intelligibility | Base acoustic-detection radius |
|---|---:|---:|
| Movement touching OPEN only | n/a | 2 cells |
| Movement entering or leaving FOREST | n/a | 3 cells |
| Low voice | `voice_range=2` | 4 cells |
| Pre-arranged sound signal | `signal_range=6` | 8 cells |
| Weapon fire | n/a | 16 cells |
| Trap activation | n/a | 12 cells |

For each source-listener ray:

- every non-endpoint FOREST cell multiplies the remaining detection radius by
  `forest_sound_factor=0.9`;
- one or more intervening WALL cells apply `wall_sound_factor=0.5` once;
- a wall always prevents understanding words or a visual gesture, even when a
  muffled sound cue is detected;
- the sound is detected when Euclidean distance is no greater than the final
  effective radius.

Reuse a Bresenham cell traversal compatible with `World.line_of_sight()`, but
keep acoustic loss in a separate function: sight and sound are not the same
sensor. Publish every coefficient in `config.json`; never hide them in the
OpFor controller.

Do not sum decibels in V1. Simultaneous movers create independent events, so a
large moving group already presents more cues and a wider footprint without
an untested logarithmic energy model. All ties and slot truncation are stable
by `(received strength, age, event id)`. If probabilistic hearing is added
later, all draws must use `env._rng` and it becomes a new registered ablation.

#### 3.6.3 Acoustic cues, memory and reports

An `AcousticCue` contains only:

```text
kind one-hot
side: friendly | hostile | unknown
eight-way bearing sector
distance band: near | medium | far
received-strength/confidence band
age and TTL remaining
```

Keep at most four freshest/strongest cues per agent and expire them after
`sound_memory_ttl=6` steps initially. A cue must not update the exact visual
enemy picture or make `FIRE` legal. It may update an investigation/belief
anchor built once from the bearing and distance band; that anchor then ages
and never follows the true source.

`side` is not an oracle label: mark a cue friendly only when the listener also
received a friendly semantic message or currently perceives the source;
otherwise use `unknown` unless a visible hostile event established the
association. The ground-truth side remains trace/oracle-only.

Add a distinct `ACOUSTIC_CONTACT` report grounded in the manual's requirement
to report an enemy presence **or indication of presence**. It carries the cue
kind, bearing/distance bands and source step, never a fabricated grid
reference. It obeys the same direct-voice, store-and-forward, staleness and
listener-local novelty rules as CONTACT. Formatter/parser round trips must
keep it distinct from a visually confirmed CONTACT.

#### 3.6.4 OpFor use of sound

Give each enemy its own `last_heard_blue` cue/estimated anchor; do not create a
band-wide acoustic omniscience channel. Decision priority is:

1. engage a currently visible Blue target;
2. pursue a recent visual sighting under the existing rule;
3. investigate a fresh acoustic anchor;
4. otherwise continue the existing garrison, assault or BRIQUE intent.

An acoustic cue authorizes movement/alerting only. The OpFor may not shoot at
an unheard identity or exact hidden cell. A BRIQUE member may become alert or
investigate, but one member hearing speech must not automatically expose Blue
to the whole band or trigger an oracle-perfect volley. Enemy movement and fire
create the same classes of sound for Blue listeners, preserving sensor
symmetry even though only Blue has the learned command language.

Blue actions resolve first under the current step loop. Their movement and
speech sounds may therefore inform the OpFor turn in that same step, but only
visual detection can produce fire. OpFor sounds enter the next Blue
observation. Record this ordering explicitly in the trace.

#### 3.6.5 Deliberate sound and visual signals

Do not invent a decorative catalogue. In the first implementation:

- `SYNC_GO` is the pre-arranged signal that opens the existing covered-bound
  window;
- `EXECUTE` is the pre-arranged signal that releases the issuer's existing
  `AT MY COMMAND` orders;
- append `GESTURE_SYNC_GO` and `GESTURE_EXECUTE` as silent alternatives with
  the same effects, `gesture_range=6`, LOS required, and the same eligible
  organizational audience.

The acoustic variants carry farther than low voice and create `signal`
events; the gestures create none. A receiver outside the semantic range may
still detect that a signal occurred without learning its code. Signals never
relay themselves: an intermediate soldier must spend a later action repeating
one. The supporting doctrine also names signals for shifting and lifting
fire, but the current simulator has no explicit fire-control state; add those
codes only together with a tested fire-control mechanic, not as inert actions.

### 3.7 Visual link, proximity and formation coherence

The shipped formation mechanic is insufficient for this mode. Today
`in_formation()` checks COLUMN/LINE/WEDGE geometry up to six cells from a
moving leader, and `formation_bonus` pays only when that leader closes new
ground. It does not require LOS, does not measure a connected device, does not
penalize a stationary rupture, and does not hide distant live friendly
telemetry.

Add one finite local-friendly predicate using the currently implemented
360-degree sight model:

```text
friendly_visible(a, b) :=
    a and b are alive
    and distance(a.pos, b.pos) <= visual_link_range  # initial value: 8
    and world.line_of_sight(a.pos, b.pos)
```

Do not import directional vision from `vision.md` here. When that separate
feature eventually lands, it may refine the predicate without changing the
cohesion contract.

For each organizational element, construct a graph containing its leader and
living direct subordinates. Two nodes share an edge when they are mutually
`friendly_visible`; sibling relay is allowed. The element has an intact
visual link when every non-detached member has a path to its leader. Applying
this recursively through team, squad and platoon elements produces a physical
command-link graph without requiring every soldier to see the root directly.

The doctrinal target is continuous link, so compute the state every tick with
no grace hidden from the metric. A soldier on an active `LiaisonTask` is
explicitly detached from the originating element and counted in separate
liaison-exposure metrics; otherwise courier duty would be impossible by
definition. Dead soldiers are removed, and succession immediately rebuilds
the relevant graphs.

Treat this as a tactical constraint and observation, not physics:

- do **not** hard-mask every MOVE that would break a visual edge; terrain and
  casualties could deadlock the whole element, and doctrine says formation is
  adapted rather than rigid;
- expose `visual_link_to_element`, `link_break_age`, current formation-station
  status and normalized formation error;
- apply a small per-agent-step `visual_link_broken` penalty to non-detached
  disconnected members, with no positive reward merely for standing in a
  blob;
- retain the watermarked formation-progress bonus, but report station keeping
  on every tick, including while halted;
- make gestures require a real current visual edge and low voice require its
  own shorter intelligibility predicate.

In `voice_only`, the existing leader/subordinate observation blocks must stop
being live trackers. For each relationship, expose a current-perception flag;
live `dx/dy` only when locally visible; otherwise expose the last perceived
`dx/dy` plus the last explicitly reported mission/formation state and age.
Unknown fields are zero with a false presence flag. Seeing a nearby teammate
refreshes position; a valid voice report may refresh semantic state, but no
remote movement refreshes either. This gives the policy the information
needed to keep formation while leaving prediction of what a
separated subordinate is doing to the later imagination project.

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
For ACOUSTIC_CONTACT it captures only the original cue's kind,
bearing/distance bands, confidence and source time; carriage must never
upgrade it into an exact CONTACT.

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
- CONTACT/ACOUSTIC_CONTACT/SITREP/DONE are credited and update state only at
  delivery.
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

- `REPORT_ACOUSTIC_CONTACT` (reports the selected freshest/strongest held cue)
- `GESTURE_SYNC_GO`
- `GESTURE_EXECUTE`
- `DELIVER_MESSAGE`
- `CANCEL_MESSAGE`
- `DISPATCH_LIAISON_S0 ... S3`

Keep all existing actions and their order stable; append the new actions so
old indices do not move even though old checkpoints still become incompatible
with the larger actor head.

### New observation blocks

Add explicit bounded blocks rather than overloading unrelated fields.

**Acoustic block:**

- `sound_model` active;
- up to four cues, each with kind and side one-hots, eight-way bearing,
  distance band, confidence, age/TTL;
- own last emitted sound kind and detection radius, so the policy can learn
  the consequence of its previous choice without seeing who heard it;
- a held-reportable-cue flag.

**Cohesion/local-friendly block:**

- element visual link intact and disconnected age;
- current formation-station flag and normalized formation error;
- current leader/subordinate perception flags plus bounded last-known ages;
- no current or intended teammate action.

**Liaison/message block:**

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
fresh coordinate for an unseen enemy, and never encode an acoustic cue in the
exact-enemy slots.

The policy remains the shared feed-forward MLP initially. Explicit packet and
destination state makes the liaison process Markov enough without importing
the future imagination/recurrent-policy project.

Because this repository expects checkpoints to transfer across scenarios,
append the new fields/actions for every communications model. Zero acoustic
and cohesion fields when structurally unavailable, and mask voice-only actions
outside that mode. That forces one honest fleet-breaking cycle instead of
creating incompatible scenario-specific network shapes. Retraining every
matched arm from scratch is mandatory.

---

## 6. Reward semantics

The reward must price delivered information, not button presses.

### Keep

- `transmission_cost` on every emitted learned utterance or acoustic signal,
  including dispatch and delivery speech; gestures have only their one-action
  opportunity cost;
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
   final line. A novel ACOUSTIC_CONTACT pays only on receipt by the intended
   superior and starts at `acoustic_contact_new=0.25`, half a visually
   confirmed CONTACT; repeating a fresh identical cue is redundant.
4. **Liaison credit to the carrier.** Pay bounded progress and delivery credit
   to teach the long physical chain:
   - recommended first cut: `liaison_progress=0.03` for each new best cell of
     closure toward the current fixed anchor, watermarked per packet;
   - `liaison_delivery=0.5` on accepted outbound delivery;
   - `liaison_receipt_return=0.25` when an ORDER receipt reaches its origin.
5. **Broken-link penalty, no proximity farm.** Start with
   `visual_link_broken=-0.01` per disconnected, non-detached agent-step,
   capped at `-0.03` per element-step. Merely staying near a leader earns
   nothing. Retain the watermarked formation-progress bonus; cohesion must
   remain a tactical trade, not a universal blob optimum.
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
10. **No blanket silence or movement-noise reward.** Do not pay STAY, quiet
    terrain or the absence of speech, and do not directly penalize every
    footstep. Noise matters through what either side detects and does; adding
    a second oracle noise penalty would double-count exposure and can teach
    paralysis.
11. **No reward for hearing alone.** An acoustic cue is sensor input. Reward
    only useful delivery, mission effects and outcomes, never the generation
    or private detection of an event.

The numeric liaison, acoustic-report and cohesion values are starting
hypotheses, not doctrine. Before training, update `max_step_farm()` and
terminal-dominance tests. Progress must be telescoping/watermarked so walking
back and forth cannot farm it.

---

## 7. Implementation sequence for Claude

### Phase A - deterministic acoustic substrate (mechanics before training)

1. Validate `sound_model` against `off | tactical`; keep `off` byte-identical
   for every existing seeded scenario.
2. Add `SoundEvent`, propagation and bounded `AcousticCue` memory, using only
   the published parameters in section 3.6.
3. Generate movement, voice, signal, weapon and trap events for both sides.
4. Add the OpFor's separate heard anchor and priority rule. Never overwrite
   `last_seen_player` with an acoustic estimate.
5. Add ACOUSTIC_CONTACT language, parsing, stale memory and local novelty.
6. Expose source truth, actual hearers, propagation loss and estimated anchors
   to trace/oracle only; update renderer/dashboard with distinct sound glyphs.
7. Finish the acoustic invariants and seeded regression tests before starting
   any PPO run.

### Phase B - direct voice, gestures and visual cohesion (space break)

Purpose: measure the core no-radio coordination problem before adding carriage
machinery, with the essential acoustic risk already active.

1. Validate `comm_model` against `global | range | voice_only`.
2. Add general low-voice intelligibility and route every communication kind
   through the rules in section 3.
3. Disable global arbitration and all HQ/automatic range bypasses in this
   mode.
4. Make CONTACT and ACOUSTIC_CONTACT pictures and tactical-change clocks
   per listener/issuer.
5. Append acoustic/cohesion fields and signal/report/gesture actions. Gate live
   friendly telemetry behind local perception in `voice_only`.
6. Implement the hierarchical visual-link graph, observations, metrics and
   bounded broken-link penalty; do not hard-mask movement.
7. Register `squad_voice_direct` as an exact `squad` mirror except
   `comm_model="voice_only"`, `sound_model="tactical"`, `voice_range=2.0`,
   and the documented root-report economics. Register a clearly named
   same-space `squad_voice_no_acoustic_ablation` with sound disabled.
8. Train one scout seed per registered direct arm. Diagnose acoustic exposure,
   visual-link reachability and actual signal use with the oracle before
   changing rewards.

This phase is an experiment, not the finished mode: it deliberately has no
way to send a separate messenger.

### Phase C - liaison-capable final squad mode

1. Add packet and `LiaisonTask` domain state with formatter/parser round trips,
   including coarse acoustic reports.
2. Append liaison actions and the explicit message block.
3. Implement prepare, self-carry, dispatch, delivery, order receipt return,
   cancellation, expiry, loss and succession-address resolution.
4. Implement bounded liaison rewards and terminal-dominance checks.
5. Add `squad_voice_liaison`, identical to the acoustic direct arm except
   liaison actions are enabled. Keep same-space radio controls.
6. Update play, dashboard, renderer, trace, oracle and briefing so packet
   status, medium, acoustic footprint, actual semantic hearers, sound
   detectors and visual-link state are auditable.
7. Retrain every matched control and voice arm from scratch.

### Phase D - depth and hostile-terrain evaluation

Only after squad mechanics are sound:

1. mirror the registered arms on `platoon` to test multi-echelon propagation;
2. mirror direct/liaison acoustic arms on `patrol_brique` to test the intended
   hostile, compartmentalised setting;
3. test OPEN/FOREST route differences without changing the coefficients after
   seeing evaluation outcomes;
4. keep directional vision, richer fire-control signals, probabilistic
   acoustics and teammate imagination outside this campaign.

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
- `sound_model="off"` consumes no new RNG and creates no behavioral state.
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

### Acoustic mechanics and information boundaries

- a successful OPEN-only move creates exactly one radius-2 movement event;
- a move touching FOREST creates exactly one radius-3 movement event; STAY,
  packet preparation, gesture and blocked/illegal movement create none;
- every FIRE creates a weapon event at the shooter whether it hits or misses,
  for Blue and OpFor alike;
- voice may be intelligible at two cells and merely detectable farther away;
- an intervening wall prevents semantics while applying the documented
  acoustic attenuation exactly once;
- forest-ray attenuation is deterministic and symmetric under reversing
  source/listener;
- two simultaneous movers create two independent events, never a silently
  summed super-event;
- a cue exposes kind/bearing/range/confidence/age only, never source id, exact
  cell, message text or `heard_by`;
- a heard but unseen target does not make FIRE legal for either side;
- OpFor investigates a frozen estimated anchor and does not track the hidden
  source after the event;
- one BRIQUE member hearing Blue does not update other members automatically;
- cue TTL expiration clears both observation and investigation eligibility;
- ACOUSTIC_CONTACT preserves coarse fields/source time through speech and
  carriage and can never become a visually confirmed CONTACT;
- the same seeded tactical-sound episode reproduces event ids, propagation,
  cue selection, OpFor anchors and actions exactly.

### Visual-link and formation mechanics

- `friendly_visible` requires finite range and LOS; a wall breaks the edge;
- sibling edges may connect an element to its leader through a path, but a
  different element cannot become an unplanned relay;
- a break is measured on its first tick and the element cap bounds the reward
  term;
- STAY cannot earn recurring cohesion reward; maintaining formation while
  halted is a metric, not a farm;
- an active liaison carrier is excluded only from its originating cohesion
  denominator and remains counted as detached/exposed;
- succession and casualties rebuild the graph without stale dead nodes;
- gestures affect only the intended eligible current visual audience and emit
  no sound;
- in `voice_only`, a non-visible moving leader/subordinate stops updating the
  observer's live deltas; last-known values age at their captured location;
- no cohesion or friendly-memory field contains a teammate's selected action,
  logits, intended route or future position;
- no movement action is masked solely because it would break formation.

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
  `sound_model`, acoustic radii/loss factors, signal/gesture ranges,
  visual-link range, liaison enabled/disabled and packet/cue TTLs;
- run `config.json`/`economics.json` records every communication and reward
  parameter;
- dashboards and transcripts label `briefing`, `radio`, `voice` and external
  umpire events distinctly, while the trace separately shows sound kind,
  semantic listeners and non-semantic detectors;
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
6. Tactical acoustics make speech and grouped movement predict later OpFor
   investigation/visual contact, while gestures and maintained visual links
   reduce the need to speak.
7. FOREST movement is locally noisier but longer forest paths attenuate sound;
   route choice therefore changes acoustic exposure without a direct reward
   for either terrain.
8. Visual-link continuity predicts lower separation and fresher local command
   state, but excessive compression may increase casualties or acoustic
   exposure; report both sides of that trade.

### Required new metrics

| Metric | Definition |
|---|---|
| `voice_utterances` | Learned speech acts emitted |
| `voice_hearers_mean` | Mean living friendly hearers per utterance, sender excluded |
| `voice_useful_delivery_rate` | Applied orders, accepted reports and required acknowledgements / voice utterances, with numerator classes published |
| `voice_when_gesture_possible` | Acoustic EXECUTE/GO uses for which the same semantic audience was reachable by gesture |
| `redundant_voice_rate` | Semantically redundant voice acts / voice utterances |
| `voice_detected_by_opfor` | Voice events detected by at least one enemy / voice events, with denominator |
| `sound_events_by_kind` | Movement/voice/signal/weapon/trap event counts for each side |
| `sound_detected_by_side_kind` | Events producing at least one opposing-side cue / events, by kind |
| `acoustic_cues_received` | Cue count per agent-step by side and kind |
| `acoustic_to_visual_latency` | First hostile cue to first visual confirmation, censored if none |
| `opfor_investigation_steps` | Enemy steps whose selected anchor came from sound |
| `acoustic_reports_attempted/delivered/redundant/expired` | Coarse report funnel with denominators |
| `sound_signal_uses/semantic_receipts/detections` | Acoustic signal outcome funnel |
| `gesture_uses/semantic_receipts` | Silent signal outcome funnel |
| `movement_sound_open/forest` | Successful moves and opposing detections by source-terrain class |
| `direct_delivery_rate` | Addressed messages received directly / attempted or prepared |
| `command_pair_in_voice_rate` | Leader-direct subordinate pairs in voice range per step |
| `element_visual_link_rate` | Intact element-steps / eligible element-steps |
| `disconnected_agent_step_share` | Non-detached disconnected agent-steps / eligible agent-steps |
| `visual_link_break_duration` | Distribution of contiguous link breaks, including first-tick breaks |
| `formation_station_rate` | Eligible member-steps satisfying current formation geometry |
| `formation_error_mean/max` | Distance from valid station band, scoped by formation |
| `friendly_state_age` | Age of leader/subordinate last-known state at decision time |
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

| Arm | Communications | Tactical sound | Liaison | Purpose |
|---|---|---|---|---|
| `squad_global_control` | Current global radio | Off | Masked | Frozen shipped reference |
| `squad_global_acoustic_control` | Global radio | On | Masked | Separates sound exposure from loss of radio |
| `squad_range_control` | Current range radio (`comm_range=12`) | On | Masked | Range-radio comparison under the same sound environment |
| `squad_voice_no_acoustic_ablation` | Voice only (`voice_range=2`) | Off | Masked | Isolates the causal contribution of enemy hearing; not an operational mode |
| `squad_voice_direct` | Voice only (`voice_range=2`) | On | Masked | Direct degraded mode |
| `squad_voice_liaison` | Voice only (`voice_range=2`) | On | Enabled | Final degraded mode |

- Same map, OpFor, org, PPO settings, total steps, observation/action layout
  and reward defaults except the documented root/liaison terms.
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
3. sound events cause uncertain, local OpFor investigation without granting
   semantic or exact-position knowledge, and the acoustic arm differs from
   its no-acoustic ablation on registered exposure/behavior metrics;
4. visual-link state is observable, continuously measured, non-farmable and
   actually maintained above a pre-registered target (start with 90% eligible
   element-steps; revise only before the first full campaign if unreachable);
5. communication is demonstrably frugal rather than absent: useful-delivery,
   redundancy and gesture-substitution denominators are reported, and
   necessary orders/reports still reach recipients;
6. the liaison arm actually delivers messages in evaluated episodes, or its
   failure is documented after the one diagnosed adjustment;
7. success, casualties, timeouts, message latency, acoustic exposure,
   visual-link continuity and manpower cost are all reported together;
8. no conclusion claims that liaison, silence or cohesion "helps" unless it
   improves a registered outcome or information metric with uncertainty
   reported;
9. all existing modes and doctrine invariants remain green.

---

## 10. Expected project meaning

The mode changes the learned problem in the intended direction:

- commanders must invest in physical presence or accept delegated execution;
- standing orders matter longer because correction is slow;
- a report is valuable only if it reaches somebody who needs it;
- speech is useful locally but creates an enemy-detectable acoustic footprint;
- movement, terrain and group dispersion affect whether the force is heard
  before it is seen;
- gestures and pre-arranged signals trade semantic richness, range and
  disclosure against each other;
- maintaining a visual chain preserves formation and command without granting
  live remote telemetry;
- casualties may partition the command graph without any omniscient alarm;
- a liaison soldier trades combat power and time for information flow;
- the transcript remains human-readable, but an external reader must now
  distinguish what was said from who could hear it.

That is a stronger test of this project's central claim than merely shrinking
a radio radius. It creates a real cost for movement, command, reporting and
common knowledge while preserving the doctrinal mission catalogue and leaving
the future teammate-imagination hypothesis cleanly outside the experiment.
