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
<CALLSIGN> [,:] rally|regroup [on me]
<CALLSIGN> [,:] hold [position]
```

Case-insensitive; a trailing `OUT.`/`OVER.` and an issuer prefix (`X, THIS IS Y:`) are
accepted and ignored. Callsigns are `<RANK><n>`: `SL1`, `TL2`, `RFN3`…

### Tactical tasks and synonyms

| Task | Keywords |
|---|---|
| RECON | recon, reconnoiter, reconnoitre, scout, observe |
| SEIZE | seize, take, capture, assault, secure |
| DEFEND | defend, guard, retain — also `hold obj X` (holding a *place*) |
| OVERWATCH | overwatch, cover, support |
| CLEAR | clear, destroy, engage, attack, eliminate, neutralize, fix |
| RALLY | rally, regroup, assemble, return |
| HOLD | hold, halt, stop (no objective ⇒ hold in place) |

### Validation

Orders injected by a human pass the same authority checks as agent-issued ones:

* as **HQ** you may order any living station;
* as a **callsign** (e.g. `--as SL1` in the console) you may only order your living
  *direct subordinates*, and only while you outrank them — otherwise `PermissionError`.

Agent-issued (learned) orders are additionally doctrine-constrained by the action mask;
a human at HQ deliberately is not — the human *is* higher headquarters.

## Reports are honest by incentive, not by construction

`MISSION COMPLETE` can be transmitted whenever the agent holds a completable task —
whether it is *true* is checked against the world (`core/missions.py::is_complete`):
truthful reports pay and clear the mission (the agent stands by for new orders); false
reports are penalized but still go out on the net, so the transcript reflects what was
actually said, including mistakes.
