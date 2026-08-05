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
| SITREP | agent → its leader | `TL1, THIS IS RFN1: SITREP, GRID 0912, HEALTH 66%, AMMO 24. OVER.` |
| DONE | agent → its leader | `TL1, THIS IS RFN1: SEIZE OBJ ALPHA — COMPLETE. OVER.` |
| DONE_CONFIRM | leader/HQ → claimant (auto) | `RFN1, THIS IS TL1: ROGER, SEIZE OBJ ALPHA CONFIRMED. OUT.` |
| DONE_REJECT | leader/HQ → claimant (auto) | `RFN1, THIS IS TL1: NEGATIVE, CONTINUE MISSION. OUT.` |
| SUPPORT_END | supporter → its leader (auto) | `SL1, THIS IS TL2: SUPPORT ENDED, TL1 IS DOWN. STANDING BY. OVER.` |
| CASUALTY | HQ → all stations (auto) | `ALL STATIONS: TL1 IS DOWN. OUT.` |
| TAKING_COMMAND | broadcast (auto) | `ALL STATIONS, THIS IS RFN1: TL1 IS DOWN. I AM ASSUMING COMMAND. OUT.` — recursive fills further down the chain: `ALL STATIONS, THIS IS RFN2: ASSUMING RFN1'S POSITION. OUT.` |

CASUALTY reports are attributed to HQ (the net/umpire convention): the dead do not
transmit. Succession announcements come in two shapes — the direct successor of the
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
<CALLSIGN> [,:] <TASK-KEYWORD> [obj|objective] <ALPHA|BRAVO|CHARLIE|DELTA>
<CALLSIGN> [,:] support|appuyer|cover [for] <CALLSIGN>     (unit-targeted SUPPORT)
<CALLSIGN> [,:] rally|regroup [on me]
<CALLSIGN> [,:] hold [position]
```

Case-insensitive; a trailing `OUT.`/`OVER.` and an issuer prefix (`X, THIS IS Y:`) are
accepted and ignored. Callsigns are `<RANK><n>`: `SL1`, `TL2`, `RFN3`…

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
the message is.

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
