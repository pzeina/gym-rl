# The command language (NATO voice procedure)

The cohort speaks NATO radio voice procedure (ACP 125 prowords, four-digit GRID
references, phonetic objective names). The language is *closed-loop*: the formatter
(what agents say) and the parser (what humans may type) are inverses over the order
grammar, tested by round-trip tests.

## Message types

| Kind | Direction | Example |
|---|---|---|
| OPORD | HQ → senior agent | `TL1, THIS IS HQ: OPORD — SEIZE OBJ ALPHA. OUT.` |
| ORDER | leader → subordinate | `RFN1, THIS IS TL1: SEIZE OBJ ALPHA. OUT.` |
| ACK | subordinate → leader (auto) | `TL1, THIS IS RFN1: WILCO. OUT.` |
| CONTACT | agent → its leader | `TL1, THIS IS RFN2: CONTACT, GRID 1716, 2 x ENEMY. OVER.` |
| SITREP | agent → its leader | `TL1, THIS IS RFN1: SITREP, GRID 0912, HEALTH 66%, AMMO 24, IN COVER. OVER.` |
| DONE | agent → its leader | `TL1, THIS IS RFN1: SEIZE OBJ ALPHA — COMPLETE. OVER.` |
| DONE_CONFIRM | leader/HQ → claimant (auto) | `RFN1, THIS IS TL1: ROGER, SEIZE OBJ ALPHA CONFIRMED. OUT.` |
| DONE_REJECT | leader/HQ → claimant (auto) | `RFN1, THIS IS TL1: NEGATIVE, CONTINUE MISSION. OUT.` |
| SUPPORT_END | supporter → its leader (auto) | `SL1, THIS IS TL2: SUPPORT ENDED, TL1 IS DOWN. STANDING BY. OVER.` |
| EXECUTE | issuer → all stations | `ALL STATIONS, THIS IS SL1: EXECUTE. OUT.` — releases every AT-MY-COMMAND order this issuer has pending (A5-2) |
| SYNC_PROPOSE | agent → nearby peers (**voice**) | `RFN2 RFN3, THIS IS RFN1: PREPARE TO BOUND ON MY SIGNAL. OUT.` (A5-4) |
| SYNC_GO | proposer → its peers (**voice**) | `RFN1: GO! OUT.` (A5-4) |
| CASUALTY | HQ → all stations (auto) | `ALL STATIONS: TL1 IS DOWN. OUT.` |
| TRAP | HQ → all stations (auto) | `ALL STATIONS: RFN2 HIT A DEVICE AT GRID 1110. OUT.` |
| TAKING_COMMAND | broadcast (auto) | `ALL STATIONS, THIS IS RFN1: TL1 IS DOWN. I AM ASSUMING COMMAND. OUT.` — recursive fills further down the chain: `ALL STATIONS, THIS IS RFN2: ASSUMING RFN1'S POSITION. OUT.` |

CASUALTY reports are attributed to HQ (the net/umpire convention): the dead do not
transmit. TRAP broadcasts follow the same convention — a friendly triggered one of
the BRIQUE band's hidden devices (mine/booby trap, manual p. 9); the grid reference
is the only location intel the net ever carries about the trap layer. Succession announcements come in two shapes — the direct successor of the
casualty *assumes command*; agents filling the vacancies that promotion leaves further
down the chain *assume the position* of whoever moved up.

### Acknowledgements are protocol, not information

WILCO is emitted automatically by the environment in the same step as its order —
a voice-procedure convention, not a policy decision. An auto-ACK can never be absent,
late, or withheld, so it carries **zero information**; consumers doing traffic analysis
should treat ORDER+ACK as one event. To run an ACK-free net, disable it per scenario
with `ScenarioSpec(auto_ack=False)` — the order still lands and is applied; only the
WILCO line disappears.

## Order grammar (what you can type)

```
<CALLSIGN> [,:] <TASK-KEYWORD> [obj|objective] <ALPHA|BRAVO|CHARLIE|DELTA>  [<TIMING>]
<CALLSIGN> [,:] support|appuyer|cover [for] <CALLSIGN>     (unit-targeted SUPPORT)
<CALLSIGN> [,:] advance [to] [wp|pl] <CONTROL-NAME>        (control-measure ADVANCE, A5)
<CALLSIGN> [,:] formation column|colonne|line|ligne|wedge  (element stance, A5-3)
<CALLSIGN> [,:] rally|regroup [on me]
<CALLSIGN> [,:] hold [position]

TIMING       := at t plus <n> | at t+<n> | at my command          (A5-2)
CONTROL-NAME := gold|silver|copper|iron  (waypoints, "WP")
              | amber|cobalt|crimson     (phase lines, "PL")
```

Case-insensitive; a trailing `OUT.`/`OVER.` and an issuer prefix (`X, THIS IS Y:`) are
accepted and ignored. Callsigns are `<RANK><n>`: `SL1`, `TL2`, `RFN3`… A timing
qualifier may ride on any mission-carrying order: `AT T PLUS <n>` makes it effective
`n` ticks from now; `AT MY COMMAND` stages it until the issuer's EXECUTE (from the
console: `env.inject_execute(issuer)`). A pending recipient holds where the order
landed. FORMATION orders carry no mission — the recipient keeps its task; the order
sets how its *element* moves and requires a recipient that actually leads one.

### Tactical tasks and synonyms (MICAT set — see docs/missions.md)

| Task | Keywords |
|---|---|
| RECON | recon, reconnoiter, reconnoitre, reconnaitre, scout |
| SCREEN | screen, eclairer |
| OBSERVE | observe, surveiller, overwatch, watch — also plain `cover obj X` / `support obj X` (the retired OVERWATCH phrases) |
| SUPPORT | `support <callsign>`, `appuyer <callsign>`, `cover [for] <callsign>` — unit-targeted |
| COVER | flank (canonical `COVER FLANK OBJ X`), couvrir |
| DEFEND | defend, tenir, guard, retain — also `hold obj X` (holding a *place*) |
| DENY | deny, interdict, interdire — section level and above (authority ≥ 2) |
| SEIZE | seize, take, capture, assault, secure |
| CLEAR | clear, destroy, engage, attack, eliminate, neutralize, fix |
| RALLY | rally, regroup, assemble, return |
| HOLD | hold, halt, stop (no objective ⇒ hold in place) |
| ADVANCE | advance, proceed — needs a control measure: `advance to wp gold`, `advance pl amber` (A5) |
| *(stance)* | formation column/colonne, line/ligne, wedge — no mission payload (A5-3) |

### Validation

Orders injected by a human pass the same authority checks as agent-issued ones:

* as **HQ** you may order any living station;
* as a **callsign** (e.g. `--as SL1` in the console) you may only order your living
  *direct subordinates*, and only while you outrank them — otherwise `PermissionError`;
* per-echelon mission admissibility applies to everyone: `TL1, deny obj alpha` is a
  `PermissionError` even from HQ — INTERDIRE is a section mission (manual p. 8);
* a SUPPORT order must name a living station other than the recipient.

Agent-issued (learned) orders are additionally doctrine-constrained by the action mask;
a human at HQ deliberately is not — the human *is* higher headquarters.

## Reports are honest by incentive, not by construction

`MISSION COMPLETE` can be transmitted whenever the agent holds a completable task —
whether it is *true* is checked against the world (`core/missions.py::is_complete`):
truthful reports pay and clear the mission (the agent stands by for new orders); false
reports are penalized but still go out on the net, so the transcript reflects what was
actually said, including mistakes.

The verdict is command traffic, not a secret: the addressee (the claimant's leader, or
HQ for the senior agent) answers every completion report on the net — `ROGER, …
CONFIRMED. OUT.` when the claim is verified, `NEGATIVE, CONTINUE MISSION. OUT.` when it
is false. Command state therefore stays derivable from radio traffic alone through the
completion phase: a reader who sees no confirmation knows the mission still stands.

## Net discipline: one station transmits at a time

The net is a **single frequency**. Per tick, at most one *learned* transmission —
CONTACT, SITREP, MISSION COMPLETE, or an order — actually goes out; when several
stations key up together, a deterministic arbitration picks the transmission that
matters most:

```
CONTACT  >  MISSION COMPLETE  >  orders  >  SITREP      (ties: agent order)
```

The losers get a **NET BUSY**: their transmission is dropped that tick — it never
reaches the transcript, costs nothing, changes nothing. The dropped attempt is
visible to external observers (`infos[...]["net_busy"]`, oracle snapshot) but not to
the cohort: masks and observations are unchanged, a blocked station has simply lost
its tick. Auto-traffic — WILCO, DONE verdicts, CASUALTY, succession announcements —
is voice-procedure protocol, not a policy decision, and is never arbitrated (or
charged). Under `comm_model="range"` the arbitration stays **global**: range shapes
who *hears* a transmission, not who may make one — everyone shares the frequency.

Airtime itself is costed: every emitted learned transmission draws
`RewardConfig.transmission_cost` (default −0.01), so speaking is only worth it when
the message is. The EXECUTE broadcast (A5-2) is order-class command traffic and is
arbitrated and costed like an order.

### Voice traffic is not radio (A5-4)

`SYNC_PROPOSE` / `SYNC_GO` are **spoken** — the manual's bond par binôme is
commanded "à la voix ou aux gestes" (pp. 14-15). They carry `voice=True` on the
transcript, reach only ~`voice_range` (6 cells; peers are registered at propose
time), are never net-arbitrated, and cost no airtime. The frequency stays free
while a binôme synchronizes its bound.

### Dedup doctrine: the first accurate CONTACT wins

CONTACT credit is adjudicated by the umpire against the whole-team enemy picture
(under `comm_model="range"` too — the umpire hears everything even when a distant
station does not):

* the **first** report of an enemy the team did not know pays `contact_new` in full —
  and a report containing *any* unknown enemy counts as first;
* a re-report of known intel that has **aged** ≥ `contact_refresh_age` steps
  (default 20, half the 40-step knowledge TTL) earns exactly **0**: it is a
  legitimate refresh — it genuinely extends the picture's life — but carries no news;
* a report whose every enemy is already **fresh** on the picture is pure noise and
  draws the small `contact_redundant` penalty on top of the airtime cost.

Every accepted report, duplicate or not, still refreshes the picture's timestamps —
the doctrine prices the traffic, it never discards the information.

## Reporting doctrine (optional): mandatory SITREP cadence

By default SITREP timing is purely reward-shaped (`sitrep_fresh`/`sitrep_spam`) and the
*absence* of traffic carries no defined meaning. Setting `ScenarioSpec.sitrep_cadence=N`
turns reporting into doctrine: an agent **not in contact** (no visible enemies) owes a
SITREP every `N` steps —

* being overdue draws `RewardConfig.sitrep_overdue` per silent step (magnitude matches
  `sitrep_spam`), and the mandated cadence replaces `sitrep_interval` as the freshness
  gap, so a due report is never scored as spam;
* the SITREP clock starts at step 0 (the first report is owed within `N` steps);
* due-ness (`min(1, steps_since_last_sitrep / N)`) is surfaced to the agent in the
  comms-summary observation slot that is otherwise redundant (the "known enemy
  present" flag, fully implied by the known-count field next to it) — `OBS_DIM` is
  unchanged and old checkpoints are unaffected while the knob stays `None`;
* agents in contact are exempt: contact reporting governs there.

With a cadence set, silence acquires semantics: a station that is neither in contact
nor reporting is *failing* to report — which is what lets a commander (or an external
observer) distinguish "nothing to report" from "unable to report".

## SITREP posture: what a station says about its own ground (issue #10)

A SITREP carries the sender's **self-reported terrain posture** — `IN COVER` or
`IN THE OPEN` — next to grid, health and ammo:

```
TL1, THIS IS RFN1: SITREP, GRID 0912, HEALTH 66%, AMMO 24, IN COVER. OVER.
```

It is a *report*, not a sensor readout, and that is the point. Per-step cover is
ground truth and stays in `env.oracle()`; nothing about it enters the observable
stream by any other route. But a soldier obviously knows whether it is behind a
tree, and saying so on the net is exactly the kind of thing voice procedure is
for. The clause makes the strongest known correlate of defend performance
(`docs/metrics.md`, the fight-disposition gate of issue #11) measurable from the
transcript alone, without any observer needing ground state.

`language.parse_sitrep(text)` reads it back — `{grid, health, ammo, in_cover}`,
or `None` for traffic that is not a SITREP — so a monitor never hand-rolls a
regex over the transcript. Formatter and parser are inverses over exactly the
fields formatted (round-trip test).

## The OPORD's assault estimate (issue #12)

In a scenario with a preparation period the OPORD adds a forward-looking clause —
the only one in the language, and the only traffic that announces something that
has not happened yet:

```
TL1, THIS IS HQ: OPORD — DEFEND OBJ ALPHA. EXPECT ASSAULT AT H PLUS 65. OUT.
```

`language.parse_opord(text)` reads the line back — `{recipient, mission,
objective, announced_assault_step}`, or `None` for traffic that is not an OPORD
— so the deadline survives the trip out to a monitor instead of being said once
and dropped. The clause sits after the task statement, so `parse_order` still
returns the same tasking with or without it.

The announced step is the **nominal** hour: the midpoint of the scenario's
arrival band (`cohort.config.announced_assault_step`, the one definition the
radio wording, the `time_to_contact` observation and the briefing all read).
The assault actually arrives somewhere in the band, and that draw is never
spoken — it is ground truth in `env.oracle()["actual_assault_step"]`. So "was
the position set before the assault came?" stays an inference, not a lookup.

Under `comm_model="range"` this is a single un-repeated broadcast, so whether a
given subordinate knows when the assault is due is a real per-listener knowledge
question, with an operational consequence: a soldier that never heard it cannot
be expected to be set by H.

## The operations overlay: `env.briefing()` (issue #10)

`cohort.config.briefing(scenario)` — also `env.briefing()` — returns the static,
pre-mission overlay as a JSON-ready dict: objective coordinates by name,
waypoint and phase-line geometry, map size, spawn, the root tasking, the
doctrinal terrain guarantees (`objective_cover`, `observation_concealment`), the
engagement envelope (weapon/vision ranges), the announced assault step
(`announced_assault_step`, above — `None` where there is no preparation period)
and the hour a defense is ordered to hold to (`defend_horizon` — `None` for an
indefinite posture).

`defend_horizon` is there for a sharper reason than the geometry is (issue #30).
Since v1.14 it is both halves of what DEFEND means: success is "occupied at
every step from H until this hour", and the root's MISSION COMPLETE bit is
admitted only when the scenario states one. Unpublished, an outside monitor
watching a root transmit MISSION COMPLETE could not tell an admissible claim
from an inadmissible one, because the deciding input was invisible to it.
Published, the same traffic is auditable. Note it is **not** in the OPORD text:
the hour is in the order the root holds, not in the words on the net.

It is a pure function of the `ScenarioSpec`, so it is identical across every
episode and valid **before `reset()`** — header material for an episode stream,
leaking no per-episode state. An external monitor that reads it from the
scenario the checkpoint names cannot drift: `fireteam_defend` moved OBJ ALPHA
from (12,12) to (18,18), and a hand-maintained coordinate table gives wrong
numbers on a re-tap of a `_v4`-era checkpoint with no error to show for it.

There is deliberately **no terrain layer** in the briefing: the grid is
regenerated at every `reset()` from the episode seed, so no static cover map
exists to publish. `terrain_static: false` says so in the payload rather than
leaving it to be inferred from an absent key.
