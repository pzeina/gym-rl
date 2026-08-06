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

**B5 update (2026-08-06):** the order-economics fixes proposed below were
implemented (rank-scaled re-task pricing + standing-order tenure) and the
three worst checkpoints retrained. The churn mechanism is **dead** — squad
re-tasking fell 58.8 → 9.6 per episode, patrol anchor rotations 1364 → 1
per 30 episodes — and destination accuracy rose sharply where orders bind
(fireteam 0.31 → 0.54). The majority baseline nevertheless stands unbeaten
on all three: the residual error is no longer churn but *vocabulary* —
formation-keeping, untasked drift, and route geometry have no radio form.
Full before/after tables and the honest verdict: [B5: binding
orders](#b5-binding-orders-by-economics-the-after-measurement) below.

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

## B5: binding orders by economics — the after-measurement

Item B5 implemented the candidate fixes from §6 and re-ran this probe as
the measurement of whether they worked. The changes (commit "B5: order
economics"): **re-task pricing** — replacing a subordinate's standing
mission costs the issuer `order_retask_cost_base × (1 + rank_scale ×
authority)` (TL −0.75, SL −1.0, PL −1.5; half price for a same-anchor
mission-type change), waived exactly when the tactical picture changed
since the standing order (a CONTACT on the net, a casualty in the issuer's
element, the issuer's own mission changed, the subordinate's truthful
DONE) — and **standing-order tenure** — positive compliance credit grows
with how long the current order has been held (×(1 + 0.5·min(held, 40)/40)),
so settled, executed orders out-earn churned ones. The three worst B4
checkpoints were retrained from scratch (squad, fireteam; 3M/2.5M steps)
or re-fine-tuned (patrol-BRIQUE, 3M from the new squad checkpoint). Each
scenario then received the campaign's one diagnosed adjustment: the first
retrains showed re-task pricing suppressing *initial* tasking too (squad
coverage time 0.96 → 0.61; a TL2 left untasked for 100+ steps) — an order
that is never issued cannot bind — so `coverage_gap` was raised −0.02 →
−0.1 and squad/fireteam retrained, patrol re-fine-tuned, under the final
economics. Published checkpoints: `fireteam_v5b`, `squad_v4b`,
`patrol_brique_v2b` (the pre-adjustment runs `fireteam_v5`, `squad_v4`,
`patrol_brique_v2` are kept alongside).

### The mechanism died

Order traffic over the B2 protocol (30 episodes, seeds 500–529),
before → after (published checkpoints):

| | orders/ep | re-tasks/ep | priced/ep | anchor rotations (30 eps) |
|---|---|---|---|---|
| fireteam | 24.2 → **7.4** | 21.2 → **4.2** | 18.1 → **2.0** | 397 → **70** |
| squad | 66.2 → **17.5** | 58.8 → **9.6** | 19.8 → **5.0** | 1404 → **210** |
| patrol_brique | 62.6 → **6.0** | 55.9 → **0.1** | 21.2 → **0.03** | 1364 → **1** |

Standing orders now stand: the seed-500 squad transcript issues six orders
in 215 steps (was: a re-task every cooldown window), and the B4 headline
transcript pattern — the same station rotated across three objectives in
21 steps — no longer occurs. N=100 success stayed within the campaign
bound (−5 pts of the published numbers): fireteam **78% ± 8** (was 83),
squad **82% ± 8** (was 84), patrol-BRIQUE **99% ± 2** (was 99).

### The DoD is still missed — honestly stated

The target was destination accuracy **beating** the majority baseline on
retrained checkpoints. Result: closer everywhere orders bind, but beaten
nowhere (N=30, seeds 500–529, K=15; the B4 rows for the same scenarios
are the before):

| checkpoint | pairs | accuracy | majority | **gap vs majority** | gap vs random |
|---|---|---|---|---|---|
| fireteam_v4d *(before)* | 19022 | 0.314 | 0.477 | −0.163 | +0.064 |
| fireteam_v5 *(retrain)* | 19559 | 0.456 | 0.520 | −0.064 | +0.206 |
| **fireteam_v5b** *(published)* | 13362 | **0.544** | 0.609 | **−0.065** | +0.294 |
| squad_v3e *(before)* | 27012 | 0.247 | 0.521 | −0.273 | +0.047 |
| squad_v4 *(retrain)* | 38744 | 0.269 | 0.367 | −0.098 | +0.069 |
| **squad_v4b** *(published)* | 31376 | **0.301** | 0.457 | **−0.156** | +0.101 |
| patrol_brique_v1 *(before)* | 23978 | 0.239 | 0.455 | −0.216 | +0.039 |
| patrol_brique_v2 *(retrain)* | 18200 | 0.131 | 0.503 | −0.372 | −0.069 |
| **patrol_brique_v2b** *(published)* | 19227 | **0.226** | 0.507 | **−0.281** | +0.026 |

**The fix worked partially.** What the economics fixed: the ordered class
is now genuinely predictable — fireteam truth-ALPHA windows are predicted
ALPHA at **0.87** (v5), and overall destination accuracy rose 0.31 → 0.54
(v5b); the gap vs majority halved on fireteam (−0.163 → −0.065) and squad
(−0.273 → −0.156, best-arm −0.098). What the economics cannot fix,
quantified from the after-run confusions:

1. **The majority baseline is a moving target.** Binding orders make
   behavior more direct, so the OPORD-objective truth share *rises with
   the fix* (fireteam 0.477 → 0.609) — the constant-prediction reader
   pockets most of the improvement the net paid for.
2. **Formation-keeping still has no radio form.** Truth LEADER is 13–31%
   of pairs on the three scenarios and its accuracy is 0.000 on every
   run, before and after — trailing your leader is never transmitted
   (the A5 `FOLLOW ME`/formation forms remain the missing vocabulary).
3. **Routes are not on the net.** The squad/patrol family walks a learned
   west-then-south axis to ALPHA; mid-transit windows close on CHARLIE
   (truth share 0.31–0.37, accuracy 0.000) while every standing order
   truthfully says ALPHA. Only phase-line/waypoint orders (A5) can put a
   dogleg on the net.
4. **Silence became optimal for the patrol.** Under re-task pricing plus
   the band's threat, `patrol_brique_v2b` converged to a *silent rush*:
   ~60-step episodes (v1: ~200+), 6.7/7 mean survivors, 99% ± 2 at
   N=100, 6 orders/ep, coverage time 0.35 — tactically the best patrol
   this project has produced, and the least radio-explained (gap −0.281;
   posture majority 0.986 MOVING). Command economics bought tactical
   excellence and paid for it in transparency: untasked sprinters are
   predicted HOLD by the net-reader, and there is nothing on the net to
   say otherwise.

Per the campaign protocol (one retrain + one diagnosed adjustment per
scenario, both spent everywhere), this is where the measurement stops.
The honest scope of the claim after B5: **the net now explains what was
ordered and that orders bind** (re-tasks are rare, priced, and mostly
carve-out-legitimate — the per-rank split in `behavior.json` shows leaf
TLs re-task almost only under the contact/casualty exceptions), but
**movement between order and arrival remains off-net** until the order
vocabulary can carry routes and formations (A5). The predictor and ground
truth were not modified in this campaign — the B4 measuring stick stands.

## A5: the vocabulary cycle — the after-after measurement

B5 ended with a diagnosis: the residual destination error was *vocabulary* —
formation-keeping, route geometry, and staged timing had no radio form. A5
added exactly that vocabulary (owner scope: **no FOLLOW-ME order**): control
measures (waypoints + phase lines) with the ADVANCE order, timing
qualifiers (`AT T PLUS n` / `AT MY COMMAND` + EXECUTE), element formations
(COLUMN/LINE/WEDGE stances), and voice-range trinôme sync
(SYNC_PROPOSE/GO). Every scenario retrained from scratch on the new spaces
(Discrete 157 → 228, Box 137 → 166) and this probe re-ran as the
measurement.

### The measuring stick grew with the vocabulary (disclosed)

The B4/B5 predictor and truth are unchanged EXCEPT that both now know the
control measures: the briefing carries them, ADVANCE traffic predicts the
named `WP <X>` / `PL <X>` class, and ground truth gained those classes
(`CM_REGION` = 4.0; closure competition includes segments). This is a
**stick change**: a phase line lying across the approach absorbs a large
share of transit windows (its perpendicular distance closes at travel
rate), so the old and new destination numbers are NOT directly comparable
— on scenarios with an approach-crossing line, `PL AMBER` truth alone is
14–41% of pairs. The majority baseline is computed on the same new stick,
so the **gap vs majority** is the comparison that carries meaning; the B5
rows quoted below are the old-stick record. Old checkpoints cannot be
re-probed: the space break makes them unloadable by construction.

### The vocabulary is used (adoption, N=100 eval episodes)

| run | success (N=100) | ADVANCE/ep | timed/ep | FORMATION/ep | stance share | sync GO/ep |
|---|---|---|---|---|---|---|
| fireteam_v6 | **84% ± 7** | 4.3 | 2.6 | n/a¹ | n/a¹ | 5.8 |
| squad_v5 | **93% ± 5** | 15.0 | 11.0 | 23.9 | 0.76 | 37.5 |
| fireteam_defend_v6 | 51% ± 10 | 7.6 | 5.4 | n/a¹ | n/a¹ | 33.5 |
| squad_recon_v5b | **94% ± 5** | 14.0 | 7.5 | 15.1 | 0.73 | 59.8 |
| squad_screen_v3 | **98% ± 3** | 0.0² | 0.0² | 13.1 | 0.76 | 29.2 |
| patrol_brique_v3 | **95% ± 4** | 11.7 | 8.9 | 14.8 | 0.54 | 14.1 |
| defend_brique_v2 | **85% ± 7** | 7.8 | 4.1 | n/a¹ | n/a¹ | 17.9 |
| platoon_v3 | **98% ± 3** | 38.6 | 16.9 | 48.7 | 0.73 | 101.4 |

¹ a fireteam has no subordinate LEADERS: FORMATION is structurally
inapplicable at that echelon. ² SCREEN doctrine derives no ADVANCE.

Trained policies genuinely speak the new language: the platoon issues ~39
ADVANCE orders and ~49 FORMATION stances per episode, formations govern
~3/4 of all agent-steps wherever they are orderable, staged AT-MY-COMMAND
advances are released by EXECUTE (~6/ep on squad/platoon), and trinôme
bounds run continuously. Success went **up** almost everywhere (squad
82 → 93, recon 85 → 94, screen 92 → 98, platoon 91 → 98, fireteam
78 → 84) — the maneuver vocabulary is not a tax; it pays.

### The probe verdict — honest: majority still unbeaten

Destination, new stick, N=30 seeds 500–529, K=15 (B5 rows: old stick,
for the record):

| checkpoint | pairs | accuracy | majority | **gap vs majority** | gap vs random |
|---|---|---|---|---|---|
| fireteam_v5b *(B5, old stick)* | 13362 | 0.544 | 0.609 | −0.065 | +0.294 |
| **fireteam_v6** | 13586 | 0.214 | 0.410 | **−0.196** | +0.048 |
| squad_v4b *(B5, old stick)* | 31376 | 0.301 | 0.457 | −0.156 | +0.101 |
| **squad_v5** | 51400 | 0.133 | 0.223 | **−0.090** | +0.008 |
| patrol_brique_v2b *(B5, old stick)* | 19227 | 0.226 | 0.507 | −0.281 | +0.026 |
| **patrol_brique_v3** | 30696 | 0.145 | 0.317 | **−0.172** | +0.020 |
| fireteam_defend_v6 | 7359 | 0.293 | 0.853 | −0.560 | +0.093 |
| squad_recon_v5b | 24048 | 0.203 | 0.423 | −0.220 | +0.037 |
| squad_screen_v3 | 22294 | 0.133 | 0.375 | −0.242 | −0.034 |
| defend_brique_v2 | 12527 | 0.080 | 0.668 | −0.588 | −0.120 |
| platoon_v3 | 112359 | 0.091 | 0.273 | −0.182 | −0.009 |

**The A5 DoD target — beat the majority baseline on ≥ 2 of the three B5
scenarios — is missed on all three.** What moved and what it means:

1. **The squad and patrol gaps improved** (−0.156 → −0.090; −0.281 →
   −0.172) *while their majority baselines collapsed* (0.457 → 0.223;
   0.507 → 0.317): under the richer stick the OPORD-only reader explains
   far less — truth is genuinely spread across control measures, LEADER
   stations, and staging — and the net-following reader keeps most of its
   ground. The squad gap is the smallest any squad checkpoint has ever
   measured.
2. **Truth LEADER is finally predictable in principle** — the A5-3 rule
   (untasked stanced members follow their leader) produces the first
   nonzero LEADER-class predictions (squad 0.055, recon 0.081 vs 0.000 on
   every previous run) — but most formation-keeping members ALSO hold a
   standing order, and orders bind first in the predictor, so the LEADER
   class remains mostly unclaimed. The vocabulary exists; the probe's
   order-primacy assumption is now the binding constraint.
3. **The fireteam regressed** (−0.065 → −0.196): its policy re-learned
   churn *through* the new vocabulary — riflemen rotate between
   `ADVANCE TO PL AMBER`, `SEIZE OBJ BRAVO`, and `SEIZE OBJ ALPHA` every
   few windows (7.1 priced re-tasks/ep vs v5b's 2.0; the B5 pricing is
   paid, not avoided — compliance + terminal profits outbid it). Binding
   orders by economics held only as long as the vocabulary was poor.
4. **Posture transparency worsened with tempo** (gap vs majority −0.25 to
   −0.34 on the maneuver scenarios): the new policies move nearly
   constantly (MOVING share 0.78–0.92), majority-posture is nearly
   degenerate, and CONTACT traffic still under-reports the firefights the
   probe would need (fireteam FIRING recall 0.001).

The scope of the honest claim after A5: **the net now carries the
maneuver** — routes are named and ordered (ADVANCE ~4–39/ep), formations
are on the air and adopted, staging and release are explicit — and the
cohort performs better under the richer language than it ever has. But a
constant OPORD-only reader still edges out the net-following reader at
K=15, because the policies exercise their re-task freedom (priced but
affordable) faster than the probe's standing-order model can follow.
The next candidate fix is no longer vocabulary: it is either steeper
rank-scaled pricing, or a probe that models *rotation cycles* rather than
single standing orders.

## Artifacts

* `cohort/probe.py` — predictor, ground truth, scoring, CLI (pure and
  deterministic; tested on hand-built transcripts in `tests/test_probe.py`).
* `runs/<run>/probe.json` — per-checkpoint results (all eight published
  checkpoints committed).
* Trace fields `fired`/`leader` added to the B2 `TraceRecorder` records
  (recording is still read-only and RNG-free; bit-identical episodes).
* A5: trace field `formation`, message kinds `execute`/`sync_propose`/
  `sync_go` (voice-flagged), and the vocabulary-usage rows in
  `behavior.json` (`advance_orders_per_episode`, `timed_orders_per_episode`,
  `formation_orders_per_episode`, `stance_share`, `sync_bounds_per_episode`).
