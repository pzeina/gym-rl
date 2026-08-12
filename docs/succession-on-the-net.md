# Succession on the net: is there a silent reattachment path? (#49)

**Short answer: no.** A soldier's commander never changes without a radio act.
Every write to `Soldier.leader_id` in the package sits inside
`Roster._fill_vacancy`, on the branch that appends the succession event
`CohortEnv.step` turns into a `TAKING_COMMAND` broadcast. When no eligible
successor exists, `_fill_vacancy` returns having changed **nothing** — there is
no reattachment to a grandparent, silent or otherwise.

**But the net-only chart and the state chart do diverge**, in both directions,
and neither divergence is silence. One of the two is very likely what an
outside monitor's residual "orphaned branch" checks are measuring, and it is
fixable on the monitor's side with no new radio act.

Everything below is pinned by `tests/test_succession_silence.py`.

---

## 1. Where a commander can change at all

`leader_id` is the only representation of "who commands me". Across `cohort/`
it is assigned in exactly two statements, both in
`cohort/core/units.py::_fill_vacancy`:

| line | what it does | announced by |
| --- | --- | --- |
| `successor.leader_id = vacated.leader_id` | the successor takes the vacated slot's superior | the event appended two lines later |
| `self.by_id[i].leader_id = successor.id` | the vacated slot's living subordinates re-point to the successor | the same event |

Both are downstream of `_pick_successor` returning a successor and upstream of
`events.append((successor, vacated))`. `CohortEnv.step` emits one broadcast per
event, through the `core/language.py` formatters:

* `format_taking_command` — "X IS DOWN. I AM ASSUMING COMMAND" (the replaced
  agent is dead: command passes),
* `format_assuming_position` — "ASSUMING X'S POSITION" (the replaced agent
  moved up: this is the backfill of the slot it left).

So "is there ANY path" is answerable structurally rather than by enumeration, and
`test_a_commander_only_ever_changes_inside_the_announced_path` enforces it: a
third write site fails the suite.

Exhaustively, over all 5040 death orderings of a squad plus every same-step pair
and triple, the parent map rebuilt from the announcements alone equals the parent
map in state, and a chart that did not move produces no traffic.

## 2. The case #49 names: no eligible successor

`_pick_successor` returns `None` only when the vacated leader has **no living
direct subordinates**. The "vacated branch" is therefore empty by construction,
and `_fill_vacancy` returns immediately without touching the roster.

The interesting consequence is what does *not* happen one level down. If a
leader's living descendants sit under an already-dead direct subordinate, they
are **not** lifted onto the grandparent: `_pick_successor` looks at direct
subordinates only. Kill SL1, TL1 and TL2 in one tick and the squad devolves to
`RFN1` and `RFN3` — each in a team-leader slot whose superior, SL1, is dead —
and `roster.root()` is `None` while four of seven soldiers still stand. Every
move was broadcast; there is simply no rule that re-homes an orphaned limb.

## 3. Two ways the reconstruction diverges from state

Both are reachable in **one tick** of the shipping `squad` scenario, and both are
measured against `cohort.probe.NetPredictor` — this repo's own transcript-only
reconstruction, which applies exactly the rules #49 describes.

### 3a. A real orphan the net *hides* (state is worse than the net thinks)

`CohortEnv.step` collects every death of the tick into `player_deaths` and only
then devolves them, one at a time, against alive-flags that already count all of
them. SL1 and TL1 fall together:

```
ALL STATIONS: SL1 IS DOWN. OUT.
ALL STATIONS, THIS IS TL2: SL1 IS DOWN. I AM ASSUMING COMMAND. OUT.
ALL STATIONS, THIS IS RFN3: ASSUMING TL2'S POSITION. OUT.
ALL STATIONS: TL1 IS DOWN. OUT.
ALL STATIONS, THIS IS RFN1: TL1 IS DOWN. I AM ASSUMING COMMAND. OUT.
```

*State*: `succeed(SL1)` cannot see TL1 as a candidate, so TL2 takes the squad;
`succeed(TL1)` then promotes RFN1 into TL1's slot — whose superior is SL1, who
is already gone. RFN1 (with RFN2 under it) is a **genuinely headless branch**.

*Net*: a replay must consume the messages in order, and when it processes TL2's
succession the CASUALTY for TL1 has not been spoken yet — so TL1 is still a live
subordinate of SL1's slot, is swept up to TL2, and RFN1 inherits that a moment
later. The net reports the chain **intact**.

The reconstruction is optimistic here. This path cannot be the source of a
monitor's orphaned-branch residual; it is the reason such a residual would
*under*-count.

### 3b. A false orphan the net *invents* (the likely explanation of a residual)

`NetPredictor._assume` re-points a vacated slot's **downward** edges
(`self.subs[cs] = new_subs`, each of them re-parented to `cs`) but never files
the successor under its new superior — it sets `self.leader[cs] = leader` and
leaves `self.subs[leader]` alone. State has done that since #42
(`parent.subordinate_ids.append(successor.id)`).

TL2 then SL1 fall together:

```
ALL STATIONS: TL2 IS DOWN. OUT.
ALL STATIONS, THIS IS RFN3: TL2 IS DOWN. I AM ASSUMING COMMAND. OUT.
ALL STATIONS: SL1 IS DOWN. OUT.
ALL STATIONS, THIS IS TL1: SL1 IS DOWN. I AM ASSUMING COMMAND. OUT.
ALL STATIONS, THIS IS RFN1: ASSUMING TL1'S POSITION. OUT.
```

*State*: RFN3 takes TL2's slot, which makes it one of SL1's subordinates; when
TL1 then assumes SL1's command it sweeps RFN3 up with the rest of that slot's
team. **No orphan at all** — the chain is whole.

*Net*: the replay never learned that RFN3 belongs to SL1's slot, so TL1's
succession does not carry it, and RFN3 is left hanging off the dead SL1 — an
**orphaned branch that does not exist**.

This is the shape a residual would take, it needs no new radio act, and the fix
is one rule on the monitor's side: *a successor joins the subordinate list of
the slot it assumed.* "Takes the vacated slot" has to be read in both
directions, or a later succession into the superior's own slot will not sweep
the promoted agent up.

## 4. Did #42 change any of this?

#42 added `parent.subordinate_ids.append(successor.id)` so that a promoted agent
is reachable from above (orderable, observed, and devolved to when its superior
falls). It introduced **no silent transition** — the link it adds is the upward
half of a move the broadcast already describes, and it is the half a replay must
mirror (§3b). It did change two things worth knowing:

* **Who succeeds, in same-step cascades.** A promoted agent now sits in its new
  superior's candidate list, so it can win the vacancy above it. On the 7-agent
  squad, 17 of the 252 same-step death batches (every ordered pair and triple)
  produce a different succession from the pre-#42 tree — including one where a
  twice-promoted rifleman takes the squad ahead of an intact team leader,
  because `_pick_successor` breaks the `effective_authority` tie on `-id` and an
  acting-TL ties a real TL.
* **How often the chain survives.** Same-step batches that leave a living
  soldier under a dead commander fall from 58 to 30, and batches that leave no
  root at all from 6 to 2. #42 halved the problem; it did not close it, because
  the residue is the ordering effect of §3a, not the chart bug #42 fixed.

### A defect #42 introduced, found here

`_fill_vacancy` now links the backfilled agent into its new leader's
`subordinate_ids` **twice** — once at #42's general `parent.subordinate_ids`
append, once at the pre-existing `successor.subordinate_ids.append(promoted.id)`
that #42 made redundant. The commonest succession in the game triggers it: SL1
falls, TL1 takes the squad, RFN1 backfills, and TL1's chart reads `[TL2, RFN1,
RFN1]`.

`living_subordinates` is what `env/observations.py` writes into the four
subordinate slots and what `env/actions.py` indexes with `order_slot`, so from
the moment it takes command the new **root** spends an observation slot on a
duplicate and carries two distinct ORDER action indices addressing the same
agent. Reached in 4 of 50 `patrol_brique` episodes under random play.

The fix is one line in `cohort/core/units.py`, which is frozen for the v1.20
campaign; `test_a_backfilled_agent_is_linked_into_its_new_leader_exactly_once`
is a strict `xfail` so the marker has to come out with the fix.

## 5. What is not decided here

Whether the headless branch of §2/§3a should be *announced* — a one-line radio
act saying a branch is without a commander, or a rule that re-homes it onto the
nearest living superior — is a vocabulary and semantics change, and this repo
reserves those to the owner. Nothing in this note changes what goes out on the
net. The finding is that the missing piece is not a missing message.
