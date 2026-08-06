# Transparency probe (B4)

The founding promise of this project is that the radio net *explains* the
cohort's behavior: every command decision is voice-procedure text on the
transcript, so a reader following the traffic should be able to say what
each agent is about to do. B3 measured the *form* of that promise (100% of
orders doctrine-valid under masks); this probe measures its *substance* —
can the traffic actually predict behavior? Implemented in
`cohort/probe.py`; status in `ROADMAP.md` (item B4).

**Headline, honestly stated up front:** it cannot, at the depth we hoped.
On every published checkpoint the net-following predictor beats uniform
random on posture (+0.07…+0.37) and (barely) on destination, but **loses to
the majority baseline everywhere** — and for destination the majority class
is simply the OPORD objective, so the finding reads: *everything the net
says after the OPORD line subtracts predictive value at a 15-step horizon.*
The mechanism is **order churn** — leaders re-task subordinates every few
steps with doctrine-valid orders whose named objectives rotate freely,
while actual movement is bound by the emergent team drift toward the OPORD
objective. Where mission anchors are stable (defenses), the net genuinely
explains behavior (destination 0.55–0.60, defenders' station 0.99 for the
defend-BRIQUE team leader); where traffic churns (assaults, three-echelon
platoon), it does not. The probe exists to make exactly this measurable.

## Running it

```bash
# B2 protocol: 30 episodes, assurance seeds 500-529, sampled policy
.venv/bin/python -m cohort.probe runs/<run>/ckpt_best.pt --episodes 30 --seed 500
# -> printed table + runs/<run>/probe.json     (--k to change the horizon)
```

## Method

### What the predictor is given

Only two things, both available to any outside reader:

* **The transcript-so-far** — literal radio text, parsed with the shipped
  `core/language.py` parser. Nothing that is not on the net enters.
* **Static briefing material** (`probe.make_briefing`) — the objective
  names/coordinates, the spawn area, and the org chart: what a briefing
  slide holds before step 0. **No positions, no oracle, no policy access.**

### The rule engine (`probe.NetPredictor`)

A deterministic state machine over the traffic, with no private state:

* **Standing orders** — the last ORDER/OPORD addressed to a station is its
  mission (recipient, task, objective parsed from the text). A DONE claim
  clears the mission only when the superior's CONFIRM answers it on the net
  (a rejected claim leaves the mission standing) — issue #4 made this
  derivable. `SUPPORT ENDED` notices clear the supporter's task.
* **Succession** — CASUALTY marks stations dead; `I AM ASSUMING COMMAND` /
  `ASSUMING X'S POSITION` broadcasts are replayed through the same
  devolution rules the roster uses (successor takes the vacated slot's
  leader, subordinates, and standing mission; recursive fills follow).
* **Position estimates** — the only position evidence is reported grids
  (SITREP, `HIT A DEVICE` broadcasts). Between reports the predictor
  assumes 1 cell/step of progress along the ordered route; arrival is
  assumed when the estimated travel time (Manhattan distance minus the
  mission's in-position radius) has elapsed.
* **Enemy picture** — CONTACT grids, fresh for `CONTACT_FRESH` = 10 steps.
* **Doctrine** (`core/missions.py`, public): SCREEN is weapons-tight and is
  never predicted firing; SUPPORT moves/halts/fires with its supported unit
  ("pas un pas sans appui") and inherits its destination; a root OPORD
  RECON/SCREEN is team-adjudicated (#9), so the commander is predicted to
  command from cover; RALLY anchors on the (living) leader.

Predictions per living agent, scored over the next **K = 15** steps:

* **destination** — one class per objective, `LEADER`, `HOLD`;
* **posture** — `STATIC` / `MOVING` / `FIRING` (firing predicted when a
  fresh CONTACT sits within `ENGAGE_RADIUS` = 12 of the agent's estimated
  position, or of its ordered station if it reaches it inside the window).

### Ground truth (recorded traces, oracle side)

* **Destination**: a *stationary* agent (never leaves `HOLD_REGION` = 3 of
  its position) belongs to the region it occupies — nearest objective if
  within `OBJ_REGION` = 9, else LEADER within `LEADER_REGION` = 6 of its
  leader, else HOLD. A *moving* agent belongs to the anchor it **closes on
  most** over the window (≥ `CLOSURE_MIN` = 2 cells) — an agent moving
  *with* its leader toward an objective closes on the objective, not the
  leader: formation-keeping is not a destination. Movers approaching
  nothing (dither, retreat) are classed by their endpoint region.
* **Posture**: FIRING if the agent fired on any window step; else MOVING if
  it changed cell on ≥ 1/3 of the window steps; else STATIC.

Scoring is per *(step × living agent)*; windows truncate at death or
episode end (empty windows are skipped). Protocol: N=30 episodes, seeds
500–529, sampled policy — the same episodes the B2 behavior baseline
measures.

### Baselines

* **majority** — the run's most frequent truth class, i.e. the strongest
  *constant* predictor in hindsight. For destination the majority class is
  the OPORD objective in every scenario, so this baseline is exactly a
  reader who stops after the first transcript line.
* **random** — uniform over the classes (destination: #objectives + 2;
  posture: 3).

The number that matters is the **gap** between the net-following predictor
and these two.

### Method notes

One calibration pass happened against `squad_v3e` and is part of the
record: the first ground-truth definition (nearest anchor by mean window
distance) classed 40% of pairs as LEADER because a squad in transit is a
moving cluster — it measured formation geometry, not destination. The
closure-based truth above replaced it, and the firing rule gained the
hot-station clause (destination accuracy 0.080 → 0.247; FIRING recall
0.05 → 0.35). The headline finding — leaf-level traffic underperforms the
OPORD-only baseline — predates the redesign and survived it unchanged. No
further tuning was done after the campaign ran.

## Results (published checkpoints, N=30, seeds 500–529, K=15)

Destination:

| checkpoint | pairs | accuracy | majority | random | **gap vs majority** | gap vs random |
|---|---|---|---|---|---|---|
| fireteam_v4d | 19022 | 0.314 | 0.477 | 0.250 | **−0.163** | +0.064 |
| squad_v3e | 27012 | 0.247 | 0.521 | 0.200 | **−0.273** | +0.047 |
| squad_recon_v4b | 32007 | 0.280 | 0.513 | 0.250 | **−0.233** | +0.030 |
| squad_screen_v2 | 21024 | 0.281 | 0.714 | 0.250 | **−0.433** | +0.031 |
| platoon_v2 | 90534 | 0.144 | 0.380 | 0.167 | **−0.236** | **−0.023** |
| fireteam_defend_v5 | 12153 | 0.551 | 0.711 | 0.333 | **−0.160** | +0.218 |
| patrol_brique_v1 | 23978 | 0.239 | 0.455 | 0.200 | **−0.216** | +0.039 |
| defend_brique_v1 | 13192 | 0.602 | 0.992 | 0.333 | **−0.389** | +0.269 |

Posture:

| checkpoint | accuracy | majority | random | gap vs majority | **gap vs random** |
|---|---|---|---|---|---|
| fireteam_v4d | 0.621 | 0.833 | 0.333 | −0.212 | **+0.288** |
| squad_v3e | 0.707 | 0.810 | 0.333 | −0.103 | **+0.374** |
| squad_recon_v4b | 0.556 | 0.803 | 0.333 | −0.247 | **+0.223** |
| squad_screen_v2 | 0.513 | 0.742 | 0.333 | −0.229 | **+0.179** |
| platoon_v2 | 0.399 | 0.555 | 0.333 | −0.156 | **+0.066** |
| fireteam_defend_v5 | 0.582 | 0.762 | 0.333 | −0.180 | **+0.248** |
| patrol_brique_v1 | 0.708 | 0.923 | 0.333 | −0.215 | **+0.374** |
| defend_brique_v1 | 0.622 | 0.690 | 0.333 | −0.068 | **+0.289** |

Per-class and per-callsign accuracy, full confusion matrices:
`runs/<run>/probe.json`.

### Confusion summaries (headline checkpoints)

`squad_v3e` destination (truth rows → dominant predictions): windows whose
truth is OBJ ALPHA (52% of pairs) are predicted ALPHA 5400 / CHARLIE 4981 /
BRAVO 3004 — the standing orders at those moments really did name CHARLIE
and BRAVO; truth HOLD (dithering far from anything, 15%) is predicted as
some objective 98% of the time (the net always claims a tasking); truth
LEADER (formation-keeping, 8%) is predicted LEADER exactly **0** times —
no order ever says "follow me".

Posture, the two defenses side by side — the same scenario geometry with
opposite CONTACT discipline:

| | truth FIRING predicted FIRING | truth FIRING predicted STATIC | B2 report recall |
|---|---|---|---|
| defend_brique_v1 | **2119 (57%)** | 1551 | 0.90 |
| fireteam_defend_v5 | **52 (2%)** | 2662 | 0.03 |

## What the probe shows (the honest discussion)

**1. The net explains *that* the cohort fights, not *where each agent
goes*.** Destination gaps vs the majority baseline are negative on all
eight checkpoints (−0.16 to −0.43); on `platoon_v2` the predictor even
loses to uniform random. Since the majority class is the OPORD objective
everywhere, the operational reading is blunt: a reader who takes the OPORD
and ignores all subsequent order traffic predicts movement *better* than
one who diligently follows every order. At K=15, the leaf-level order
stream is noise around the OPORD signal.

**2. The mechanism is order churn, and it is visible on any transcript.**
`squad_v3e`, seed 500, first minute of traffic:

```
[t=  0] SL1, THIS IS HQ: OPORD — SEIZE OBJ ALPHA. OUT.
[t=  1] TL2, THIS IS SL1: SEIZE OBJ BRAVO. OUT.
[t=  5] TL1, THIS IS SL1: CLEAR OBJ CHARLIE. OUT.
[t=  6] RFN2, THIS IS TL1: CLEAR OBJ ALPHA. OUT.
[t= 12] TL2, THIS IS SL1: CLEAR OBJ ALPHA. OUT.
[t= 16] TL1, THIS IS SL1: SEIZE OBJ CHARLIE. OUT.
[t= 21] TL2, THIS IS SL1: CLEAR OBJ CHARLIE. OUT.
```

Every line is doctrine-valid (the mask guarantees it) and each is obeyed
*briefly* (B2 obedience latency 2.4 steps), but stations are re-tasked to a
different objective every cooldown window, so no order binds behavior for
the 15 steps the probe asks about. The squad still wins because compliance
shaping plus combat and terminal rewards pull the *team* toward ALPHA
regardless. B3's interpretability claim ("a reader of the full arm's net
can reconstruct the plan") must therefore be scoped honestly: the reader
can reconstruct a *doctrine-legal decomposition*; they cannot reconstruct
*what will actually happen* from it at leaf level — churn is legal under
current economics (`order_churn` prices re-issuing the *same* order, not
rotating between different valid ones).

**3. Where anchors are stable, the promise holds.** The defenses are the
counter-example that proves the mechanism: `defend_brique_v1` destination
0.602 (its TL — the station whose DEFEND ALPHA order never changes — is
predictable at **0.992**), `fireteam_defend_v5` 0.551 with 0.758 on the
objective class. Orders that stand still explain behavior; the accuracy
deficit vs their (near-degenerate 0.99) majority baselines comes from
riflemen the net never re-anchors after skirmish displacements.

**4. Posture transparency = reporting discipline.** The probe beats random
on posture everywhere (+0.07 to +0.37), and its FIRING predictions are
exactly as good as the CONTACT traffic: the two defenses share geometry
and mission, yet firefights are predictable on `defend_brique_v1` (FIRING
recall 0.57, report recall 0.90) and invisible on `fireteam_defend_v5`
(0.02, report recall 0.03 — three CONTACTs in 30 episodes). A cohort that
fights without reporting is opaque to the net by construction. This
cross-validates the B2 finding from an independent direction.

**5. Systematic failure modes, named:**

* **Untasked agents are not idle.** The net says "standing by"; ground
  truth shows them drifting with their team (truth-HOLD ≈ dither of tasked
  agents; untasked ones follow objectives they were never given). HOLD
  class accuracy is 0.02–0.44.
* **Riposte and meeting engagements.** Fire without a fresh CONTACT on the
  net is invisible (the fireteam_defend row above; SCREEN riposte is
  doctrinally sanctioned yet weapons-tight doctrine forces a STATIC
  prediction).
* **Formation-keeping has no radio form.** Truth LEADER reaches 32% of
  pairs (fireteam assault); its accuracy is 0.000 on all eight runs —
  trained policies essentially never issue RALLY, and "moving with the
  unit" is never transmitted.
* **The commander's self-preservation is off-net.** Roots ordered SEIZE
  are predicted onto the objective; the human commanders actually hang
  back (platoon PL1 mean objective distance 33.9 — B2). Doctrine explains
  the #9 RECON/SCREEN roots (rule encoded, SL1 destination 0.375/0.227),
  but SEIZE roots' cover-keeping is learned economics the net never
  mentions.
* **Three echelons compound the churn.** `platoon_v2` is the worst row on
  both tasks (destination 0.144, below random; posture gap +0.066): with
  16 stations re-tasking each other, 38% of the platoon's windows are
  STATIC-waiting while its traffic keeps naming objectives; second-squad
  stations bottom out around 0.10/0.20.

**6. What would make the net explanatory (future work, not claimed):**
order-content economics — price *rotating* objectives, not just re-issuing
identical orders (extend `order_churn`); pay compliance on longer horizons
so orders must bind before they can be replaced; a `FOLLOW ME` /
formation order form (A5's phase lines and "AT MY COMMAND" timing would
also lengthen order lifetimes); SITREP-cadence doctrine
(`ScenarioSpec.sitrep_cadence`) to buy posture transparency at the price of
airtime. Re-running this probe is then the measurement of whether those
changes worked — that is what it is for.

## Artifacts

* `cohort/probe.py` — predictor, ground truth, scoring, CLI (pure and
  deterministic; tested on hand-built transcripts in `tests/test_probe.py`).
* `runs/<run>/probe.json` — per-checkpoint results (all eight published
  checkpoints committed).
* Trace fields `fired`/`leader` added to the B2 `TraceRecorder` records
  (recording is still read-only and RNG-free; bit-identical episodes).
