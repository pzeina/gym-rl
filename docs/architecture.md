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

1. **Snapshot** previous positions and mission-anchor distances (for progress shaping).
2. **Friendly actions** in agent order — each agent does exactly one thing per tick:
   move, fire, report, or issue one order. Orders apply immediately: the recipient's
   mission changes, an ORDER + WILCO pair lands on the transcript.
3. **OpFor actions** — scripted state machine (garrison/assault, chase, engage).
4. **Casualties & succession** — deaths broadcast CASUALTY; the roster devolves command
   recursively and successors announce TAKING COMMAND.
5. **Knowledge decay** — the team enemy picture (fed *only* by CONTACT reports) expires
   stale entries and drops dead enemies.
6. **Compliance & rewards** — each agent's standing mission is scored against what it
   actually did (see `core/missions.py::compliance`), leaders are scored on subordinate
   coverage, and the ledger assembles per-component rewards.
7. **Terminal checks** — scenario success (root-mission-specific), defeat (cohort wiped),
   or timeout. PettingZoo semantics: dead agents return `terminated=True` once, then
   leave `env.agents`.

## Rank admissibility as masking

`env/actions.py` builds one flat catalog (97 actions) shared by all agents. Per step,
`compute_mask` produces the legality vector:

* order actions require `effective_authority > 0`, a living subordinate in that slot,
  **and** the ordered mission to be doctrine-derivable from the issuer's own mission;
* re-tasking a subordinate within `ScenarioSpec.order_cooldown` steps (default 8) of
  its last received order is masked — standing orders get time to be executed — unless
  the leader's own mission changed since, or a CONTACT report hit the net since
  (the tactical picture changed). Untasked subordinates are always orderable;
  `order_cooldown=0` disables the cooldown;
* FIRE requires ammo and a visible enemy in weapon range;
* CONTACT requires a currently visible enemy;
* MISSION COMPLETE requires holding a mission with an end state (SEIZE/RECON/CLEAR/RALLY).

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
purely additive.

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
