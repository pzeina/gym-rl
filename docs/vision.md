# Directional vision — design proposal

> **STATUS: designed and decided; not implemented.** Nothing in `cohort/`
> implements the vision mechanics yet. The owner's calls are recorded in
> **§0** and the body below is written to them.

## 0. Decisions of record (2026-08-06)

| Call | Decision |
|---|---|
| **Sequencing** | **Ship v1.10 fully first** — retrain and publish all 8 scenarios under v1.10, *then* vision as **v1.11**. Accepts a second fleet retrain in exchange for a published, uncontaminated v1.10 milestone. |
| **Gate** | **Run the information-asymmetry probe first** (§6). Registered as scenario `squad_short_vision`. Informative, not blocking: a null result weakens the §1 case without killing it. |
| **Arc semantics** | **Split arcs, 4-dir facing** — `vision_arc 180°`, `fire_arc 90°`, `all_round_awareness_range 2.0`. |

Consequences worth stating plainly, since the sequencing choice is the
expensive one: v1.11 will invalidate the v1.10 fleet a second time, so the
8-scenario retrain is paid **twice** on purpose. What it buys is a clean
v1.10 record — the prep-period and false-COMPLETE verdicts land on their own
numbers, with no vision confound to argue about later.

Everything gated behind the v1.10 publish is marked **[v1.11]** below. The
probe (§6) and this note are the only work that runs before it.

---

The ask: line of sight degraded by dense vegetation, a vision *angle* instead of
360° awareness, and a rotation cost reflecting that a rifleman cannot fire in
two directions at once.

Two questions have to be answered honestly before any of it gets built: **is
this relevant to what this project is actually about**, and **can a memoryless
MLP policy learn under it**. Both answers are qualified yes. The qualifications
are the design.

---

## 1. Relevance: the case is not realism

The weak case for this feature is fidelity — real soldiers have a field of view,
so the sim should too. That case is not good enough on its own. This project is
about chain of command, not tactical simulation, and every hour spent on optics
is an hour not spent on the transparency residuals, self-play, or buildings.

The strong case is different, and it runs through the project's own open failure:

**The transparency probe still trails the OPORD-only baseline** (best-ever squad
gap −0.090, residuals in `docs/transparency.md` §A5). Reading the C2 traffic does
not predict behavior better than reading the initial order does. The messages are
not carrying much beyond the OPORD.

One strong candidate explanation: **the environment does not manufacture enough
information asymmetry for talking to be worth anything.** Today every agent has
360° awareness out to 10 cells, and LOS is blocked only by `WALL`. Two soldiers
standing three cells apart in open terrain see very nearly the same world. Under
those conditions a CONTACT report is informationally close to a no-op — the
recipient almost certainly already sees what is being reported. The reward pays
for reporting, so agents report; but reporting is not *load-bearing*. A
decentralized policy loses almost nothing by ignoring the net.

> **Measured, after this was written — the paragraph above holds for one
> scenario family and fails for the other two.** The three-cells-apart claim is
> right (sighting-set Jaccard 0.56–0.86 for such pairs), but the cohort rarely
> stands that way: a CONTACT report is novel to 65.5% of listeners at squad
> scale and 83.5% at platoon scale, against 13.3% at `fireteam_defend`. The
> numbers and what they change are in **§6.1** (refs #17). This does not
> overturn the §0 decisions — it narrows where the payoff is expected.

Vision arcs break common knowledge structurally, and they do it in a way that
range reduction alone cannot: **two soldiers standing on the same cell facing
different ways see different worlds.** That makes a sighting genuinely private,
which makes the CONTACT → shared-picture channel the only route to an all-round
element picture — and makes a leader distributing sectors the only route to
covering 360° collectively.

That reframes the feature. It is not a realism side quest; it is a candidate
**precondition for the project's headline claim to be testable at all.** That is
a hypothesis, not a diagnosis — see §6 for the cheap experiment that tests it
before committing to the build.

### Second-order: mechanics that would start meaning what their docs say

| Mechanic | Today | With arcs |
|---|---|---|
| SUPPORT / overwatch | in position + LOS + radius; a supporter facing the wrong way is just as good | overwatch becomes **directional** — you cover an axis, which is what `d780a3e` ("SUPPORT is overwatch of a moving element") already asserts doctrinally but the sim does not enforce |
| COVER (flank guard) | static within radius 6 | a flank guard's entire job is *orientation outward*; currently that is a radius, and orientation is free |
| OBSERVE / SURVEILLER | radius 9 + LOS | an OP has a field of view and a blind rear |
| DEFEND / prepared position | cover is free: forest conceals you *and* you still see 10 cells out | cover becomes a **trade** — see §3.1 |

The last row lands directly on the open `fireteam_defend` problem. Two documented
misses: v6 held the ground but would not fire; v7 fires at 1.000 but fights 9.7
cells out with cover occupancy 0.05. Under foliage attenuation, "in cover" and
"has fields of fire" stop being independent, and the prepared position becomes a
real choice — the edge of the woods, not the depth of them. That is an
opportunity and a hazard, and it is why sequencing (§5) matters.

### The honest counter-argument

Cone vision may simply make every scenario harder to learn, drop success across
the fleet, and buy a muddier research story at the cost of a full retrain. That
risk is real. It is mitigated by generous first-cut parameters (§3.2), by
symmetry with OpFor (§3.4), and above all by running the cheap probe in §6 first.

---

## 2. Feasibility: the binding constraint is the policy net

`cohort/training/ppo.py:55` — `PolicyNet` is a feedforward MLP, two 256-wide Tanh
layers, actor and critic heads. **No recurrence.**

This is the fact that governs the design. Arc vision creates a POMDP whose
sufficient statistic is a belief over enemy positions, and a belief requires
memory. With a memoryless policy, an enemy that leaves the arc **vanishes from
the observation entirely**. The predictable failure mode is oscillation: turn →
see → turn away → forget → turn back. An agent with no memory cannot even
represent "there is someone behind me."

Three mitigations, in increasing cost:

**(a) Put the memory in the observation — mandatory, not optional.** A per-agent
decaying *remembered contact* track: for each enemy this agent has personally
seen recently, the position **where it was last seen** plus an age. Cheap,
realistic (you remember where he was), and it makes the observation nearly
Markov again. It also has a clean doctrinal reading: this is the soldier's own
mental picture, distinct from the *reported* team picture already in the comms
block. Without this, do not build the feature.

> **Exploit hazard, and it is a subtle one.** The remembered slot must store the
> **stale** position, never the enemy's live position. A "memory" that tracks a
> moving enemy through a wall is omniscience wearing a memory costume, and it
> would silently undo the entire feature. This belongs in the regression-hazard
> set alongside terminal dominance and weapons-tight (§4).

**(b) Hand the net the arc condition instead of making it derive one.** Keep
world-frame `dx, dy` everywhere (the dashboard, probe, and metrics all read
world-frame deltas — rotating the whole observation into the facing frame would
break every tool for a marginal gain), and *add* an explicit `in_fire_arc` flag
per enemy slot plus a facing one-hot. The sign condition becomes a lookup rather
than an arithmetic derivation across two Tanh layers.

**(c) Add a GRU to the policy — recommend deferring.** The correct solution to a
POMDP, but it means BPTT through the validity-masked GAE, sequence handling for
dead agents, and a checkpoint format change. `CLAUDE.md` is explicit that the
training stack is deliberately minimal ("NO RLlib — the legacy RLlib stack is why
the old repo died"). (a) + (b) should suffice at the recommended arc widths; if
they do not, that is a strong, publishable finding and *then* recurrence is
justified on evidence.

**Credit-assignment lengthening.** The causal chain grows from `see → fire` to
`turn → see → fire`. At 4-dir facing with a wide vision arc that is one extra
step, occasionally two. That is tolerable. At 8-dir facing with a narrow arc it
is three or four, which is not. This is the main argument for starting coarse and
generous.

---

## 3. Mechanics, as separately shippable increments

Deliberately phased so a miss stays diagnosable, per this repo's standing
discipline.

### 3.1 — V1: foliage attenuates line of sight

*No new state. No space break. Old checkpoints still load.*

Today `World.line_of_sight` is a Bresenham walk where only `WALL` blocks, and
`can_spot` shortens range only when the **target** stands in forest. Foliage
between observer and target is free.

Proposed: accumulate **optical depth** along the existing Bresenham walk. Each
`FOREST` cell traversed adds `foliage_density`; `WALL` remains infinite. Sighting
succeeds when `d <= vision_range * exp(-depth)`.

**This calibrates to a strict generalization of today's model.** Today
`vision_range=10`, `forest_vision_range=6` — a ratio of 0.6. Since
`exp(-0.5) = 0.6065`, setting `foliage_density = 0.5` reproduces the current
single-cell forest penalty almost exactly (6.07 cells vs today's 6.0). Nothing
about the existing forest behavior changes; what changes is only rays travelling
*through* woods, which are free today. `forest_vision_range` becomes a derived
quantity — keep publishing it in `briefing()` (assurance contract, §7) as a
derived value rather than deleting it.

Continuous attenuation is preferred over a hard "blocks after N forest cells"
cutoff: it is smoother for learning, it is one parameter instead of two, and it
subsumes the existing special case rather than sitting beside it.

**The reward consequence needs an owner decision.** v1.10 just added
`prep_in_position` (0.05/step for standing in cover at the objective) and the
nearest-cover observation vector. Under foliage attenuation, "stand in cover" and
"can see the threat" become competing objectives — and a reward that pays only
for cover occupancy will happily train a policy that goes blind in the deep
woods. Paying for *cover with LOS to the threat axis* is the doctrinally correct
fix, but it is a reward-structure change, which is the owner's call.

### 3.2 — V2: facing and vision arc

*Space break: observation and actions.*

`Soldier.heading` already exists (`units.py:82`, unit 4-dir) but is set only on
movement (`cohort_env.py:940`) and read only by formation geometry. Promote it to
a first-class **facing**.

**Two arcs, not one** (decided, §0). Peripheral awareness is wider than aimed
fire, and separating them buys the stated goal at a much lower learnability cost:

| Parameter | Decided | Rationale |
|---|---|---|
| `vision_arc_deg` | **180** | spotting/awareness. Generous — halves your world rather than quartering it, keeping the POMDP mild |
| `fire_arc_deg` | **90** | aimed fire. **This is the constraint that bites**: you cannot engage in two directions at once |
| `all_round_awareness_range` | **2.0** | you notice someone standing next to you regardless of facing. Prevents the absurd case and softens the POMDP further |

The user's item 3 — "impossibility to fire simultaneously at different
directions" — is delivered by `fire_arc_deg` alone. `vision_arc_deg` can stay
generous, or even be dropped from the first cut, without losing the tactical
point.

**Rotation coupling — recommended: movement sets facing, plus `FACE_*` actions
that cost the whole step.**

- Moving sets `heading = movement direction` (today's behavior, so advancing
  elements stay learnable and nothing regresses for free).
- Four new `FACE_NORTH/SOUTH/EAST/WEST` actions override facing and **consume the
  step** — a step spent turning is a step not spent moving, firing, or
  transmitting.

The consequence is the point of the whole design: **an individual cannot advance
while covering a flank.** That limitation is precisely what makes distributed
sectors and bounding overwatch necessary at the element level — the individual
constraint generates the doctrine, rather than the doctrine being priced in by
hand.

**Facing resolution — decided: 4-dir for the first cut.** It aligns with the
movement axes, the existing `heading` semantics, and `core/missions.in_formation`
geometry; rotation is at most two steps to reverse; and a 90° fire arc *exactly
tiles* at 4-dir, so four soldiers facing four ways cover 360° with no seam. 8-dir
is a natural later refinement, at the cost of a longer turn→see→fire chain.

**Observation budget: +24 → `OBS_DIM` 220 → 244**

| Block | Floats |
|---|---|
| facing one-hot | 4 |
| `in_fire_arc` flag, per enemy slot | 4 |
| remembered contacts — 4 slots × (present, dx, dy, age) | 16 |

**Action budget: +4 → `N_ACTIONS` 228 → 232** (`FACE_*`).

Layout work per `CLAUDE.md`: update the `OBS_DIM` arithmetic and the derived
`OFF_*` offsets in `env/observations.py` (the build-time assertion catches
mistakes), and extend `tests/test_observation_blocks.py`.

### 3.3 — V3: sector-of-fire orders *(later phase — the C2 payoff)*

`ORDER_S{i}_COVER_SECTOR_{N|S|E|W}` — 4 slots × 4 sectors = **+16 actions**.

This is where the research value actually lands: all-round defense becomes a
*commanded* act that appears on the transcript, rather than an emergent accident.
It is exactly the kind of signal the transparency probe should be able to read
off the net, and it is the most direct test of the §1 hypothesis.

It is also a vocabulary expansion — the owner's call — and it should follow V2
rather than ship with it, so that "arcs work" and "commanding arcs helps" stay
separable findings.

### 3.4 — V4: OpFor symmetry *(must ship inside V2, not after)*

`world.can_spot` is already called symmetrically — `(enemy, soldier)` at
`cohort_env.py:1634` and `(soldier, enemy)` at `1703`. That symmetry is a gift
and must be preserved.

`Enemy` needs a `heading` and `enemy_decide` a turn rule (face the movement
direction, or the last-seen-player bearing — keep it simple and scripted).

**If blue gets arcs and OpFor stays omniscient, every scenario gets flatly harder
for a reason that has nothing to do with C2, and every number moves for the wrong
cause.** Conversely, symmetric arcs make the scripted OpFor genuinely flankable,
which will *inflate* success rates. Either way the fleet's baseline moves; the
requirement is that it moves for a stated, symmetric reason.

---

## 4. Instrumentation and regression hazards

**Diagnose before changing rewards** (`CLAUDE.md`). `env.oracle()` needs to expose:
facing per agent; whether each agent's fire arc covers the nearest threat; the
element's **union arc coverage** as a fraction of 360°; detection latency; and
flank events (an enemy engaging from outside every arc).

New behavior-suite metrics: `sector_coverage`, `flank_exposure_rate`,
`detect_latency`, `facing_changes_per_step`.

**Every one of those needs a denominator, specified before it is first
measured.** Four consecutive assurance issues — #13, #14, #15, #16 — each turned
out to be the *metric* at fault rather than the policy, and #16 was exactly this
failure: an order mix reported as a raw share, with no correction for how often
each order was even admissible, which made a masking artefact read as a policy
preference for a whole generation. Three of the four metrics above are the same
shape and would fail the same way:

* `flank_exposure_rate` — conditioned on a flank being *available* to expose. An
  agent pinned in a corner cannot be flanked; one in the open almost always can.
* `facing_changes_per_step` — conditioned on facing changes being useful at that
  state, not on the raw step count.
* `sector_coverage` — the union arc must be scored against the arc the element
  *could* have covered given its living members, not against a flat 360°: a
  two-survivor element covering 180° with 90° arcs is at 1.00, not 0.5.

`detect_latency` is censored data, not a rate — an enemy never detected has no
latency, and dropping those episodes silently reports the mean of the *successes*
only. Report the censoring count beside it, the way `_obedience` was fixed to in
#15.

The shipped precedent to follow is `env/actions.order_options` +
`metrics.order_selection_lift` (`a5abdb4`): score against a masked-random floor
computed from the same mask that built the observation, so 1.00 means "indifferent
among what was legal" and the number carries its own null hypothesis.

**Naming.** Use `facing` / `sector` vocabulary throughout — **not** `rotation`,
which already means patrol-anchor rotation in this repo (the v1.8 economics
result, 1364 → 1). A collision there would make two unrelated metrics read alike.

Regression-hazard tests to add, in the tradition of terminal dominance /
churn / weapons-tight — each encodes a real exploit:

1. **Stale-track invariant** — a remembered contact stores the last-seen
   position and never updates while unobserved. *(The omniscience leak of §2a.)*
2. **Step exclusivity** — `FACE_*` consumes the step; no agent moves and turns in
   the same tick. *(This is what keeps "cannot fire two directions at once"
   true. Note that spinning to scan is legitimate soldiering, not an exploit —
   the economics are already right, because a turn trades against moving,
   firing, and reporting.)*
3. **Flank invariant** — an enemy outside the arc and beyond
   `all_round_awareness_range` is neither in `_visible_enemies` nor firable.
4. **OpFor symmetry** — a soldier can close on an enemy from behind undetected.
5. **Foliage monotonicity** — effective spotting range decreases monotonically in
   forest cells traversed; `WALL` still hard-blocks; endpoints still never block
   (existing invariant).

**Performance note, free win.** `_visible_enemies` is recomputed on every call —
`_make_view`, the mask path, `_report_contact`, `metrics.py:179`,
`viz/dashboard.py:80`. Arc and attenuation logic make each call dearer. Memoize
per `(soldier, step)` in the same commit; it should more than pay for the added
work.

---

## 5. Sequencing — this collides with the open v1.10 cycle

**The situation.** v1.10 is an open breaking cycle: spaces are already
Discrete(228)/Box(220), every published checkpoint is unloadable, and **the fleet
has not been retrained.** The standing published numbers are all v1.9.

That cuts both ways.

*In favor of doing vision now:* the expensive part of a space break is the fleet
retrain, and it has not been spent yet. Landing vision before that retrain costs
one observation-layout edit. Landing it after costs **a second full 8-scenario
retrain and publish cycle.**

*Against:* v1.10 already moves five things at once and none are validated. Adding
vision makes it eight unvalidated changes, and `fireteam_defend` — the entire
point of the v1.10 cycle — becomes undiagnosable. The ROADMAP's own words on that
run: *"Two variables moved at once here — that separation is how a miss stays
diagnosable."*

**Three sequencings were considered:**

**A — Fold vision into v1.10.** One space break total, cheapest in retrains.
Confounds the defend experiment with three more mechanics. Cheap and reckless
against this repo's stated discipline.

**B — v1.10 verdict on `fireteam_defend` only, then vision as v1.11, then one
fleet retrain.** Retrain only the one scenario whose *verdict* v1.10 was designed
to change, then break spaces again and retrain the fleet exactly once. Cheapest
path that keeps the defend miss diagnosable.

**C — Ship v1.10 fully (retrain and publish all 8), then vision as v1.11.
← CHOSEN.** The orthodox path: the whole fleet is retrained, measured, and
published under v1.10 before anything else breaks.

**What the choice costs and buys.** It pays for the 8-scenario retrain twice —
once under v1.10, once under v1.11 — where B would have paid roughly 1 + 8 runs
instead of 8 + 8. In exchange, v1.10 gets a complete published record on its own
numbers: the `human_death` → 0.0 call, the prep-period and occupancy-pay
experiment, and the false-COMPLETE fix all land with a full fleet behind them and
no vision confound anyone has to reason around later. Given that v1.10's five
changes are *themselves* the standing answer to the D4 collapse and the
fireteam_defend misses, a clean verdict on them is worth a retrain.

**Practical consequence.** The v1.10 fleet campaign is the near-term work and is
unrelated to this note; the only vision work that runs before it is the §6 probe,
which is independent of the fleet retrain and can train alongside it.

---

## 6. The information-asymmetry probe — runs first

*Registered and ready: scenario `squad_short_vision` (`cohort/config.py`).*

The §1 case rests on a hypothesis — *C2 traffic is not load-bearing because
observations are near-common-knowledge*. That hypothesis is testable for roughly
one training run, **without building any of §3**:

> Train the squad with vision halved (10 → 5 cells, forest 6 → 3, the 0.6 ratio
> preserved so cover economics do not shift) and re-run the transparency probe
> against the OPORD-only baseline. Compare the gap to `squad`'s best-ever −0.090.

```bash
scripts/train.sh squad_short_vision_v1 --scenario squad_short_vision --total-steps 3000000
.venv/bin/python -m cohort.probe runs/squad_short_vision_v1/ckpt_best.pt --episodes 30 --seed 500
```

**Reading the result.** If information asymmetry is what the C2 channel is
missing, the gap should narrow relative to the `squad` control trained under the
same v1.10 spaces — so the control must be the v1.10 `squad` run, not the v1.9
published number. If the gap does not move, arcs are unlikely to rescue it
either, and the feature reverts to a fidelity argument: still defensible, but
much weaker, and worth far less than a second fleet retrain.

**Caveat, stated precisely:** isotropic range reduction creates asymmetry only
between *separated* agents. Arcs additionally split the picture between
*co-located* agents, and only arcs create the sector-assignment coordination
problem of §3.3. The probe is therefore a **lower bound** on the effect, not a
full proxy — a null result weakens the case without killing it, and a positive
result is strong confirmation. Per §0 it is informative, not blocking.

One confound to control for: halving vision also makes the scenario *harder*, so
a drop in success rate is expected and is not itself evidence either way. The
number that matters is the probe **gap** against the OPORD-only baseline computed
on the same run, which is already how `docs/transparency.md` reports it.

### 6.1 Measured pre-arc baseline — the premise is only true for one family (refs #17)

Issue #17 pre-registers an expectation for V1: that the sighting-knowledge
lattice will become **non-constant under arcs**, "where today there is none".
Pre-registration is only worth anything against a baseline measured *before*
the change, so here is that baseline, taken on the shipped v1.10 checkpoints
from `env.oracle()` alone (truth stream, sampled actions, seeds 500+):

| | `fireteam_defend_v10` | `squad_v6` | `platoon_v4` |
|---|---|---|---|
| map / living stations | 36×36 / ~3.9 | 42×42 / ~5.8 | 54×54 / ~13.2 |
| a living enemy seen by **all** stations | 4.0% | 0.3% | **0.0%** |
| seen by **no** station | 92.7% | 94.9% | 93.2% |
| **split** (some see it, some do not) | 3.3% | 4.8% | 6.8% |
| split, *given ≥1 station sees it* | **44.8%** | **93.6%** | **100.0%** |
| mean pairwise sighting-set Jaccard | 0.680 | 0.206 | 0.080 |
| …for pairs ≤3 cells apart | 0.857 | 0.595 | 0.555 |
| share of station pairs that are ≤3 cells apart | 67.4% | 25.7% | **6.0%** |
| CONTACT content **novel** to a listener | 13.3% | **65.5%** | **83.5%** |
| in-range (soldier, enemy) pairs denied by LOS | 26.4% | 16.8% | 11.7% |

Three things follow, and they matter for how V1 is read.

1. **The lattice is already non-constant.** Whenever a sighting exists at all,
   it is a minority sighting 45–100% of the time. #17's baseline assumption —
   that today the answer to "does HQ know there is an enemy at grid X" is
   trivially yes — is inverted: 93–97% of living enemies are absent from the
   team picture entirely. Where the lattice *is* constant it is constant at
   ¬K, not at K.
2. **§1's premise is right locally and wrong globally.** "Two soldiers three
   cells apart see very nearly the same world" holds (Jaccard 0.56–0.86). The
   conclusion drawn from it — that a CONTACT report is close to a no-op — does
   not, because the cohort does not stand three cells apart: at platoon scale
   only 6.0% of station pairs are that close, and 83.5% of report/listener
   pairs learn something they could not see.
3. **Arcs bite where the picture is already shared.** The families ordered by
   co-location are exactly the families ordered by how much arcs can add:
   `fireteam_defend` (67% of pairs co-located, reports 87% redundant) has the
   most to gain, `platoon` (6%, reports 84% novel) the least. So the §6 probe
   run on `squad` measures the *middle* of that range, and V1's effect should
   be expected to be strongly family-dependent rather than fleet-wide.

Vision is not "isotropic and long" today in the sense the argument needs: it
is 10 cells on maps of 36–54, and 12–26% of in-range pairs are already denied
by walls. The information asymmetry the design wants to create largely exists;
what arcs would add is asymmetry *between co-located agents*, which §6's own
caveat already identifies as the part the probe cannot see.

---

## 7. Cross-project contract flag

`cohort.config.briefing()` publishes the engagement envelope — `vision_range` and
`forest_vision_range` (`config.py:496–497`) — so the assurance layer can define
"under threat" the way `metrics.py` does (refs #10).

**Directional vision silently invalidates that definition.** A published scalar
`vision_range` becomes an *overestimate* of what a soldier can actually see, so
the external instrument's threat envelope quietly loosens, with no error to show
for it — the exact failure mode `briefing()` was built to prevent, recurring one
level up.

If V2 ships, `briefing()` must publish the arc parameters alongside the ranges,
and the change belongs in `ASSURANCE-SYNC.md` as a contract amendment.
