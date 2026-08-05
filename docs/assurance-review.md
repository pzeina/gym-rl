# cohort, reviewed from the outside

*An external monitoring layer was attached to this system (read-only: recorded
radio traffic + ground-truth scoring over 60 seeded episodes, 30 per shipped
checkpoint, seeds 500–529, sampled actions). This document collects everything
that review surfaced about **the system itself** — behavior, protocol, API,
engineering — independent of the monitoring layer. Corpus: 14,634 messages,
107 casualties, 74 succession events, 29/30 success rate for both
`fireteam_v2` and `squad_v1`. Base commit `b151eda` (v1.1.0).*

---

## What holds up under adversarial review — keep these

Verified externally, not just by the in-repo tests:

- **Doctrine masking is a real guarantee.** 2,229/2,229 observed orders
  (380 fireteam + 1,849 squad) were doctrine-derivable from the issuer's
  standing mission, through 74 real succession cascades. The claim "a trained
  leader is *guaranteed* to issue doctrine-valid orders" survives contact
  with the trained policies' actual behavior.
- **Command state is reconstructible from the net alone.** An observer that
  reads only radio traffic can track every agent's standing mission through
  order assignment and multi-level succession — validated at 39,459/39,459
  (step × living-agent) checks against ground truth. The transparency
  guarantee is not just human-readable text; it is *sufficient* text. (One
  exception: mission completion — F1/F2 below.)
- **Determinism is exact.** Same seed → byte-identical episodes including
  transcripts, verified by byte-comparison of independently re-run rollouts.
- **Format/parse inverses, self-contained checkpoints, a 71-test suite that
  runs in under 2 s** — all as advertised.

---

## Findings

Ranked by how much they matter. Each has evidence, why it matters, and a fix
direction. Line references are to `b151eda`. Filed as issues:
F1 → [#3](https://github.com/pzeina/gym-rl/issues/3),
F2 → [#4](https://github.com/pzeina/gym-rl/issues/4),
F3 → [#5](https://github.com/pzeina/gym-rl/issues/5),
F4–F9 → [#8](https://github.com/pzeina/gym-rl/issues/8),
FR-A → [#6](https://github.com/pzeina/gym-rl/issues/6),
FR-B → [#7](https://github.com/pzeina/gym-rl/issues/7).

### F1 — The trained policies never report MISSION COMPLETE  *(behavior, most important)*

**Evidence.** 0 DONE messages in 14,634 across all 60 episodes, both
checkpoints — while the environment's own state check declares success 29/30.
The `done_true` reward exists (`env/rewards.py`) but is never earned: it is a
dead reward path in both shipped policies.

**Why it matters.** The system's core promise is that a human commander can
follow the operation from radio traffic. That promise currently breaks at the
most important moment: the operation *ends* and nobody says so. HQ observes
success only out-of-band (the episode terminates). The README's own example
transcript shows `SEIZE OBJ ALPHA — COMPLETE` — the shipped checkpoints never
produce that line. Additionally, "honest MISSION COMPLETE reports" is listed
as a learned behavior; what was actually learned is *never claiming at all*,
which is vacuously honest.

**Fix direction.** Make completion reporting load-bearing rather than
decorative: gate the terminal success bonus (or part of it) on a root-mission
DONE having been transmitted, or add a curriculum stage that rewards it
specifically. Re-train and check the transcript ends with a completion
report.

### F2 — MISSION COMPLETE truthfulness is invisible on the net  *(protocol)*

**Evidence.** `env/cohort_env.py::_report_done`: the reporter's mission is
cleared only when the claim is verified true; the radio text is identical in
both branches. A false claimant silently keeps its mission; the leader is
never told whether the report was accepted.

**Why it matters.** This is the one place where command state stops being
derivable from traffic (the review had to model it as a two-way ambiguity).
It is also doctrinally odd: in real voice procedure the superior acknowledges
a completion report; here the net carries a claim whose effect is secret.

**Fix direction.** Close the loop on the net: leader (or HQ) responds with an
acceptance or rejection message; or make the mission-clear unconditional and
penalize the false claim separately. Either restores "the net tells the whole
command story" — and F1's fix makes this path actually exercised.

### F3 — Message.payload is never populated  *(API)*

**Evidence.** `core/orders.py::Message.payload` exists (default `{}`);
`cohort_env._say(kind, sender, recipient, text)` has no payload parameter,
so every message ships with an empty payload and all structure lives only in
the formatted text. Any consumer (dashboard, evaluation scripts, external
tools) must regex the NATO text back apart — the review had to, with
round-trip checks to survive format drift.

**Why it matters.** The structured content (order task + objective, contact
grid + count, sitrep health/ammo/grid, casualty callsign, succession
successor/replaced) is *known at every emission site* and then thrown away.
Text should be derived presentation, not the source of truth.

**Fix direction.** Fill `payload` at each `_say` call site; keep the text
form exactly as is. Purely additive, no behavioral change, removes a whole
class of parser bugs from every consumer at once.

### F4 — WILCO is automation, not communication  *(protocol/behavior)*

**Evidence.** 2,229/2,229 ACKs arrive in the same step as their order, by
construction (`_assign_mission` emits ORDER and ACK back-to-back).

**Why it matters.** An acknowledgment that cannot be absent, late, or
withheld carries zero information — it is protocol theatre that inflates the
net (orders + ACKs are ~32% of squad traffic). It also forecloses modeling
the situations ACKs exist for: comms failure, refusal, an overwhelmed
subordinate.

**Fix direction.** Either make acknowledging a (masked, cheap) policy action
so a missing WILCO becomes meaningful, or drop auto-ACKs from the transcript
as noise. Half-measures (auto but delayed) keep the worst of both.

### F5 — Order churn: re-tasking faster than any mission can progress  *(behavior)*

**Evidence.** Squad: 1,849 orders over 30 episodes (~62/episode for three
leaders; fireteam ~13). Median gap between consecutive orders to the *same*
subordinate: 7–8 steps; 10% of re-orders arrive within 1–2 steps of the
previous one. Redundant same-mission re-orders are ~0% — the churn is rapid
*task switching*, not repetition (e.g. a rifleman ordered CLEAR then
re-ordered OVERWATCH two steps later, never having moved meaningfully).

**Why it matters.** "Keeping every subordinate tasked, avoiding order churn"
is listed as a learned objective; the anti-churn part is at best partially
learned. Sub-mission-timescale re-tasking makes subordinate behavior hard to
attribute to any order (bad for the transparency story), floods the net, and
suggests the order-coverage reward outweighs the churn penalty.

**Fix direction.** Strengthen the churn penalty or add a short cooldown to
the order mask (cannot re-task the same subordinate within k steps unless a
CONTACT arrived). Worth a look at the reward-component plots when re-tuning.

### F6 — The dead transmit: CASUALTY is attributed to the casualty  *(provenance)*

**Evidence.** `cohort_env.py:291`: `self._say(MessageKind.CASUALTY, dead.id,
None, ...)` — "ALL STATIONS: TL1 IS DOWN" is *sent by TL1*, who is dead.

**Why it matters.** Any consumer that models who-said-what (provenance,
per-sender statistics, an eventual comms model where dead radios are silent)
gets a contradiction at every death. Doctrinally the report should come from
an observing teammate or the net/umpire.

**Fix direction.** Attribute to `HQ_ID` (net/umpire convention) — one-line
change — or, more ambitiously, to the nearest living teammate with line of
sight (which would also make death *observability* honest rather than
global and instantaneous).

### F7 — Succession text is built in two places, one bypassing language.py  *(hygiene)*

**Evidence.** The first successor uses `lang.format_taking_command`
(`I AM ASSUMING COMMAND`); recursive fills build a second shape inline in
`cohort_env.py` (~line 298): `ASSUMING {X}'S POSITION`. Consumers must know
both.

**Fix direction.** Move the second shape into `core/language.py` next to the
first; `cohort_env` should never assemble radio text inline.

### F8 — Sampled episodes are not independently reproducible  *(engineering)*

**Evidence.** `training/evaluate.py` seeds torch once and threads one
`np.random.Generator` through all episodes; episode *k*'s actions depend on
how many sampling calls episodes 0..k−1 consumed. Reproducing episode k alone
requires replaying the whole prefix. (Greedy mode is unaffected.)

**Fix direction.** Derive a fresh per-episode generator (and
`torch.Generator`) from `seed + i` inside the episode loop.

### F9 — Minor API/hygiene

- `_episode_outcome` is private but is the only complete outcome record;
  `evaluate.py` itself reaches into it. Expose a public property.
- The OPORD is emitted during `reset()` with `step=0`, before any action;
  fine, but worth one docstring line since consumers see traffic "before the
  episode starts".

---

## Missing capabilities (feature requests, not defects)

These are the two structural limits an external observer runs into. Both are
scenario knobs, not rewrites, and both are filed as feature requests in the
repo.

### FR-A — A comms model: per-listener audibility

The net is a single, global, perfectly reliable channel: every message
reaches everyone, including the enemy-contact picture feeding on CONTACT
reports addressed to someone else. Partial observability lives entirely in
*sightings*. Consequences: recipient fields are decorative (nothing changes
if a message is "for" TL1), silence proves nothing, comms discipline cannot
be learned, and distributed-operations phenomena (a cut-off fire team, relay
through terrain) cannot occur. A per-listener audibility model — range,
line-of-sight, or relay-based, as an optional `ScenarioSpec` knob — would
make the radio net as honest as the vision model already is.

### FR-B — A reporting doctrine: mandatory SITREP cadence

SITREP timing is purely reward-shaped (`sitrep_fresh`/`sitrep_spam`); no
protocol obligates a report. So the *absence* of traffic from an agent
carries no defined meaning, and a commander (human or automated) can never
distinguish "nothing to report" from "unable to report". A doctrine knob —
e.g. a SITREP is due every N steps when not in contact; overdue is a
violation — would give silence semantics. Combined with FR-A this is what
makes casualty *inference* (rather than the current global instant CASUALTY
broadcast) possible at all.

---

## Corpus reference

| | fireteam_v2 | squad_v1 |
|---|---|---|
| episodes / seeds | 30 / 500–529 | 30 / 500–529 |
| outcomes | 29 success, 1 defeat | 29 success, 1 defeat |
| messages | 3,120 | 11,514 |
| orders (+ACKs) | 380 (+380) | 1,849 (+1,849) |
| contacts / sitreps | 516 / 481 | 1,925 / 5,719 |
| casualties / successions | 27 / 12 | 80 / 62 |
| DONE reports | **0** | **0** |

Generated with `python -m cohort.tap runs/<run>/ckpt_best.pt --episodes 30
--seed 500 …` on branch `assurance-integration`.
