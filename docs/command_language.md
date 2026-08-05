# The command language

The cohort speaks terse radio voice-procedure. The language is *closed-loop*: the
formatter (what agents say) and the parser (what humans may type) are inverses over the
order grammar, tested by round-trip tests.

## Message types

| Kind | Direction | Example |
|---|---|---|
| OPORD | HQ → senior agent | `CAP1, THIS IS HQ: OPORD — SEIZE OBJ ALPHA. OUT.` |
| ORDER | leader → subordinate | `SLD1, THIS IS CAP1: SEIZE OBJ ALPHA. OUT.` |
| ACK | subordinate → leader (auto) | `CAP1, SLD1: WILCO.` |
| CONTACT | agent → its leader | `CAP1, THIS IS SLD2: CONTACT, 2 HOSTILES AT (17,16). OVER.` |
| SITREP | agent → its leader | `CAP1, THIS IS SLD1: SITREP, HP 66%, AMMO 24, POS (9,12). OVER.` |
| DONE | agent → its leader | `CAP1, THIS IS SLD1: SEIZE OBJ ALPHA — COMPLETE. OVER.` |
| CASUALTY | broadcast (auto) | `ALL STATIONS: CAP1 IS DOWN.` |
| TAKING_COMMAND | broadcast (auto) | `ALL STATIONS, THIS IS SLD1: CAP1 IS DOWN. I AM TAKING COMMAND.` |

## Order grammar (what you can type)

```
<CALLSIGN> [,:] <MISSION-KEYWORD> [obj|objective] <ALPHA|BRAVO|CHARLIE|DELTA>
<CALLSIGN> [,:] regroup|rally [on me]
<CALLSIGN> [,:] hold [position]
```

Case-insensitive; a trailing `OUT.`/`OVER.` and an issuer prefix (`X, THIS IS Y:`) are
accepted and ignored. Callsigns are `<RANK><n>`: `CDG1`, `CAP2`, `SLD3`…

### Mission keywords and synonyms

| Mission | Keywords |
|---|---|
| RECON | recon, reconnoiter, scout, observe |
| SEIZE | seize, take, capture, assault, secure |
| DEFEND | defend, guard — also `hold obj X` (holding a *place*) |
| OVERWATCH | overwatch, cover, support |
| ENGAGE | engage, attack, eliminate, neutralize, fix |
| REGROUP | regroup, rally, return |
| HOLD | hold, halt, stop (no objective ⇒ hold in place) |

### Validation

Orders injected by a human pass the same authority checks as agent-issued ones:

* as **HQ** you may order any living station;
* as a **callsign** (e.g. `--as CDG1` in the console) you may only order your living
  *direct subordinates*, and only while you outrank them — otherwise `PermissionError`.

Agent-issued (learned) orders are additionally doctrine-constrained by the action mask;
a human at HQ deliberately is not — the human *is* higher headquarters.

## Reports are honest by incentive, not by construction

`MISSION COMPLETE` can be transmitted whenever the agent holds a completable mission —
whether it is *true* is checked against the world (`core/missions.py::is_complete`):
truthful reports pay and clear the mission (the agent stands by for new orders); false
reports are penalized but still go out on the net, so the transcript reflects what was
actually said, including mistakes.
