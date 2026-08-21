# Architecture

## Design goal

Make a multi-agent cohort whose command behavior is **transparent and human-compatible**:
every decision that matters between agents travels as a radio message a human can read,
and the constraints that define each rank are *structural* (action masks), while the
judgment that defines a good soldier is *learned* (reward shaping).

## Layering

```
┌─────────────────────────────────────────────────────────┐
│ play.py / evaluate.py / train.py        (interfaces)    │
├─────────────────────────────────────────────────────────┤
│ CohortEnv (PettingZoo ParallelEnv)      (simulation)    │
│   actions.py  observations.py  rewards.py               │
├─────────────────────────────────────────────────────────┤
│ core domain                             (pure logic)    │
│   ranks  missions/doctrine  orders  language            │
│   units/roster/succession  world/LOS                    │
└─────────────────────────────────────────────────────────┘
```

The core domain has no environment or RL dependencies — ranks, doctrine, compliance,
succession, and the command language are all unit-testable pure logic. The environment
composes them and adds simulation (combat, OpFor, terrain) and the RL contract.

## The step loop

Each `env.step(actions)`:

0. **Timed-order release** (A5-2) — "AT T PLUS n" orders whose tick has come due
   become effective before anything else, so the tick's anchors and compliance are
   judged against the now-executing order (tenure restamps at release).
1. **Snapshot** previous positions and mission-anchor distances (for progress shaping).
2. **Net arbitration** — the radio is a single frequency: of this tick's *learned*
   transmission attempts (CONTACT / SITREP / MISSION COMPLETE / orders), at most **one**
   goes out. Deterministic priority: CONTACT > DONE > orders > SITREP, ties broken by
   agent order; the losers' transmissions are dropped this tick with a NET BUSY outcome
   (no cost, no effect — flagged per agent in `infos[...]["net_busy"]` and the oracle
   snapshot). Auto-traffic (WILCO, DONE verdicts, CASUALTY, succession) is protocol,
   not competition for airtime, and is never arbitrated. Contention is judged on
   tick-start legality — the same mask the policy acted under.
3. **Friendly actions** in agent order — each agent does exactly one thing per tick:
   move, fire, report, or issue one order. Every emitted learned transmission costs
   `RewardConfig.transmission_cost` (airtime is not free). Orders apply immediately:
   the recipient's mission changes, an ORDER + WILCO pair lands on the transcript.
   A move onto a hidden trap cell triggers it here (damage, reveal, net broadcast).
4. **OpFor actions** — scripted state machine (garrison/assault, chase, engage), or
   the BRIQUE band controller (see [The OpFor](#the-opfor) below): the band-level
   intent machine ticks once, then each member acts under the band intent.
5. **Casualties & succession** — deaths broadcast CASUALTY; the roster devolves command
   recursively and successors announce TAKING COMMAND.
6. **Knowledge decay** — the team enemy picture (fed *only* by CONTACT reports) expires
   stale entries and drops dead enemies.
7. **Compliance & rewards** — each agent's standing mission is scored against what it
   actually did (see `core/missions.py::compliance`; a pending A5-2 order scores as
   HOLD at its staging spot), leaders are scored on subordinate coverage, and the
   A5 maneuver terms pay: the formation bonus (members at their COLUMN/LINE/WEDGE
   station while their stanced leader closes NEW ground) and the trinôme bound bonus
   (a synchronized mover under a covering group-mate closing NEW ground) — both
   watermark-gated so they telescope with the advance and cannot be farmed. The
   ledger assembles per-component rewards.
8. **Terminal checks** — scenario success (root-mission-specific), defeat (cohort wiped),
   or timeout. PettingZoo semantics: dead agents return `terminated=True` once, then
   leave `env.agents`.

## Rank admissibility as masking

`env/actions.py` builds one flat catalog (228 actions at the A5 layout) shared by all
agents. Per step, `compute_mask` produces the legality vector:

* order actions require `effective_authority > 0`, a living subordinate in that slot,
  **and** the ordered mission to be doctrine-derivable from the issuer's own mission;
* per-echelon admissibility: a mission with a minimum hold authority (DENY → 2,
  section level) is masked for recipients below it; SUPPORT orders
  (`ORDER_S{i}_SUPPORT_U{j}`, unit-targeted) need living units in both slots;
* re-tasking a subordinate within `ScenarioSpec.order_cooldown` steps (default 8) of
  its last received order is masked — standing orders get time to be executed — unless
  the leader's own mission changed since, or a CONTACT report hit the net since
  (the tactical picture changed). Untasked subordinates are always orderable;
  `order_cooldown=0` disables the cooldown;
* ADVANCE orders (`ORDER_S{i}_ADVANCE_WP_GOLD` … + `_AMC` at-my-command variants)
  additionally require the named control measure to exist on THIS map; FORMATION
  stance orders require the recipient slot to lead an element and skip both the
  doctrine-derivation check and the cooldown (a stance is how the element moves,
  not what it does);
* EXECUTE_SIGNAL is legal only while ≥1 living subordinate holds an AT-MY-COMMAND
  order of this issuer still awaiting the signal; SYNC_PROPOSE needs ≥1 trinôme
  peer within `voice_range`; SYNC_GO needs the agent's own unexpired proposal;
* FIRE requires ammo and a visible enemy in weapon range;
* CONTACT requires a currently visible enemy;
* MISSION COMPLETE requires holding a mission with an end state
  (RECON/SCREEN/SEIZE/CLEAR/RALLY/ADVANCE) that is not pending (A5-2).

The policy applies the mask at the distribution level (logits of illegal actions →
−1e9), so illegal behavior is impossible even during exploration. The environment also
re-checks legality at application time (state may have shifted within the tick — e.g. the
target died to a teammate's shot) and treats a now-illegal action as STAY.

## Succession

`Roster.succeed(dead)` in `core/units.py`:

1. Successor = designated deputy if alive, else the senior living direct subordinate
   (ties → lowest id).
2. The successor assumes the vacated **position**: acting rank (mask expands), the dead
   leader's leader, subordinates, and standing mission (mission continuity).
3. The hole the successor left in its own team is filled by the same procedure,
   recursively; the promoted teammate becomes a direct subordinate of the successor.

Effective rank (`Soldier.effective_rank`) is what observation encoding, masking, and
authority checks use everywhere — a rifleman acting as squad leader *is* a squad leader
to the whole system, which is exactly the "soldiers can take the role of dead leaders"
requirement.

## Team knowledge and why reporting matters

Sightings are private: an agent's observation contains the enemies *it* can see. The
shared "known enemy" summary in every observation is fed exclusively by CONTACT reports.
So a subordinate that spots the garrison and reports it materially improves its leader's
(and everyone's) information state — the reward for a novel report is aligned with an
actual information channel, not a synthetic bonus.

## The comms model (audibility)

By default the net is a single, global, perfectly reliable channel
(`ScenarioSpec.comm_model="global"`): every message reaches every station. The
optional `comm_model="range"` (with `comm_range`, euclidean) makes audibility
per-listener:

* a message is heard only by stations within `comm_range` of the sender; the
  sender always hears itself. HQ is a high-power station: HQ traffic is always
  heard, and HQ always hears the root;
* a CONTACT report updates only the enemy pictures of stations in earshot — the
  team picture becomes *per-agent* (`_agent_known`), and each agent's "known
  enemy" observation summary is built from its own picture (observation layout
  unchanged);
* an ORDER to an out-of-earshot subordinate is transmitted (it lands on the
  transcript) but never received: no mission change, no WILCO, no command
  credit — a missing acknowledgement finally *means* something;
* other traffic (SITREP/DONE verdicts, CASUALTY, succession) is unchanged —
  reports up the chain are adjudicated by the umpire regardless of range.

Under `"global"` the behavior is byte-for-byte the shipped one; the knob is
purely additive. Net-busy arbitration (step 2 above) stays **global** under
`"range"` too: earshot shapes who *hears* a message, but every station shares
the one frequency, so simultaneous transmissions still contend.

> **Designed, not implemented:** [`degraded-communications.md`](degraded-communications.md)
> specifies a third, radio-less `comm_model="voice_only"`. It uses low voice
> range, makes movement/voice/signals/weapons produce uncertain acoustic cues
> for both sides, adds silent gestures and a continuous local visual-link
> graph, removes remote friendly telemetry plus the HQ/global-frequency
> exceptions, and permits physical store-and-forward delivery by an agent of
> liaison. The existing `"range"` model remains a radio model and is the
> control, not an approximation of that design.

## The OpFor

Two enemy families, both environment-side (blue's spaces never change):

**Scripted garrison/assault** (`opfor_mode="garrison" | "assault"`,
`core/units.py::enemy_decide`): hold near home or advance on a goal, engage
players on sight, chase briefly, return.

**BRIQUE armed band** (`opfor_mode="brique"`, `core/units.py::BriqueBand`) —
the PROTERRE manual's asymmetric threat (p. 9 "LA MENACE"): a *flat* band
(no hierarchy, no leader) of 5–20 fighters with light weapons. A band-level
**intent machine** drives per-member behavior states:

* `lurk` — hold in cover, avoid detection; posts the ambush when blue
  approaches within `lurk_trigger`;
* `ambush` — posted at a chokepoint on blue's *predicted route* (the
  spawn→objective line); **hold fire** until a blue unit is inside
  `ambush_range` — or until the ambush is compromised (a member hit) — then
  volley for `volley_steps` and dissolve into hit-and-run;
* `harass` — fire `harass_shots` from max range, then displace to a new
  cover cell ("engagement de moyens limités, très disparates");
* `raid` — move fast onto the objective/installation, linger `raid_linger`
  steps (sabotage), then withdraw ("raid à portée limitée visant à détruire
  des moyens de communication, des dépôts");
* `scatter` — break contact toward the map edges; irreversible, and entered
  only below `scatter_below` strength (default 30% — low self-preservation
  by design: the band fights nearly to the last).

Target selection is casualty-maximizing ("actions à fort impact
psychologique"): the human commander first, then wounded, then isolated blue
units, then the closest. All knobs live in `ScenarioSpec.band`
(`BriqueBandConfig`); all randomness flows through `env._rng`.

**Traps/mines** (`ScenarioSpec.n_traps`): hidden devices the band lays at
reset along blue's likely route (or, for a defense, on the position's
approaches — "y compris les mines et les pièges"). The first friendly
stepping on one takes its damage (default 40) through the normal casualty
pipeline (rank-weighted death, CASUALTY broadcast, succession); the device
is then spent and revealed, and the umpire broadcasts
`ALL STATIONS: RFN2 HIT A DEVICE AT GRID 1110. OUT.` Traps are oracle
ground truth from step 0 and **never appear in blue observations** — they
are the assurance layer's inference target.

**BRIQUE terminal semantics**: on DEFEND/DENY root missions against a band,
success = the band **destroyed**, or **scattered with contact fully broken**
(every living member ≥ `break_contact_dist` from every living friendly and
from the root objective — scatter is irreversible, so this is final), while
the objective is **held** (no living enemy on it, a living friendly manning
it). SEIZE-rooted BRIQUE scenarios keep the standard SEIZE check: the OPORD
is about the objective — a scattered band never blocks it, and destroying
the band does not seize anything. Defeat and timeout are unchanged.

The oracle exposes the whole band layer as enemy-side non-observables:
band intent, ambush posts, spring step, per-member behavior states, and
every trap with its armed/revealed state.

## Determinism

All randomness flows through one `np.random.Generator` seeded at `reset(seed=...)`:
map generation, spawns, combat rolls, OpFor jitter. Same seed + same actions ⇒ identical
episodes, transcripts included (covered by tests).

## The ground-truth oracle (external observers only)

`env.oracle()` (`core/oracle.py`) returns an omniscient per-step snapshot: every unit —
friendly and OpFor — with position, health, cover, visibility (who sees whom), the OpFor
AI's internal state (mode, goal, last sighting), and a defined vocabulary of *behavior
observables*: `attacking`, `advancing`, `retreating`, `covering` (protecting),
`holding`, `hidden`, `wounded`, `down`.

Two hard rules, both covered by tests:

1. **The cohort never sees it.** Oracle data enters no observation, reward, or mask;
   the observation layout is unchanged; calling the oracle consumes no randomness, so
   it cannot perturb a seeded episode.
2. **The net carries text only.** Radio messages are voice-procedure text — structured
   payloads are forbidden by design. Ground truth for external analysis lives here, in
   the oracle, not in the messages.

The intended consumer is an external assurance layer: treat the enemy side of the
snapshot as hidden ground truth and measure how well it can be inferred from the
friendly side alone (own units + radio traffic).

### The other side of that line: `env.briefing()`

`env.briefing()` / `cohort.config.briefing(scenario)` returns the **static operations
overlay** — objective and control-measure coordinates, map size, spawn, root tasking,
the doctrinal terrain guarantees, the engagement envelope, the assault step the
OPORD announces (`announced_assault_step`; the step it *actually* arrives at is drawn
per episode and stays in the oracle as `actual_assault_step`), and the hour a defense
is ordered to hold to (`defend_horizon` — since v1.14 both the DEFEND success
criterion and the gate on the root's MISSION COMPLETE bit, so publishing it is what
lets an outside monitor classify a completion claim as admissible), and the gap a
SITREP is priced fresh against (`sitrep_interval` — the standard
`closed_on_cadence_report_rate` is defined against, refs #37). It is the opposite of
the oracle in every respect that matters: a pure function of the `ScenarioSpec`,
identical across episodes, valid before `reset()`, and therefore legitimate for an
external monitor to consume — a real one reads it off the overlay before H-hour. There
is no terrain layer in it, because the grid is regenerated at every `reset()`
(`terrain_static: false` says so explicitly). Per-step cover stays in the oracle; the
radio-legitimate view of it is the sender's own SITREP posture clause.
