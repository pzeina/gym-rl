# Missions (MICAT / PROTERRE doctrine)

The mission set is the full MICAT catalog of the French Army PROTERRE manual
([`docs/manuel-proterre.pdf`](manuel-proterre.pdf)), adopted as the project's
doctrinal source (owner decision, 2026-08-05): **English names, PROTERRE
semantics**. Page references below are the manual's printed page numbers —
the group missions are defined on pp. 29–38 ("2.6 — Les missions du groupe"),
the per-echelon mission menus in the *tableau récapitulatif* on p. 8.

The enum order of `MissionType` (`cohort/core/missions.py`) is load-bearing:
it defines the observation one-hot layout and the action-catalog layout.

## The eleven tasks

| Mission | French (manual) | Targets | Completable | Semantics in the sim |
|---|---|---|---|---|
| RECON | RECONNAÎTRE (p. 30) | objective | yes | Get intel on the objective, **may engage** ("en engageant éventuellement le combat"). In position within radius 7 **with LOS**; completes after 5 cumulative observation steps. No fire penalty; full combat pay. |
| SCREEN | ÉCLAIRER (p. 32) | objective | yes | Intel **without engaging** ("rechercher du renseignement sans engager le combat"): same observation semantics as RECON, but firing costs compliance (−0.6) and earns nothing (weapons tight). |
| OBSERVE | SURVEILLER (p. 33) | objective | no | Continuous posture: static in an observation position (radius 9 + LOS), detect and alert. Compliance 0.6 static in position / 0.1 moving. Fire pays only from position. |
| SUPPORT | APPUYER (p. 35) | **friendly unit** | no | "Apporter une aide à une autre unité … fourniture de feux." The order names a friendly element, not an objective; the anchor *tracks the supported soldier*. In position: within 10 cells of it **with LOS to it** ("la liaison à vue avec l'élément appuyé", p. 35 note 1). Grants covered movement + focus fire (see below). Ends on re-tasking or on the supported unit's death (auto-clears with a net notice). |
| COVER | COUVRIR (p. 38) | objective | no | Flank guard: "s'opposer à une action éventuelle de l'ennemi pouvant menacer l'action principale amie." Static within radius 6 of the named objective (no LOS requirement), **free to fire from position**. Compliance 0.5 static / 0.1 moving. |
| DEFEND | TENIR (p. 36) | objective | no | "Occuper un point ou un espace de terrain." Radius 3.5; position-anchored fire discipline (combat pays only from the position). |
| DENY | INTERDIRE (p. 8, p. 37 n. 1) | objective | no | Section-level area denial — like DEFEND with radius 5, holdable only by **authority ≥ 2** (SL and above): the tableau récapitulatif lists INTERDIRE under SECTION/COMPAGNIE, never GROUPE. Enforced in the order mask and in `inject_order`. |
| SEIZE | — | objective | yes | Take possession of the objective and clear it (unchanged from v1.1). |
| CLEAR | — | objective | yes | Eliminate all hostiles at the objective (unchanged). |
| RALLY | — | leader | yes | Assemble on the direct leader; the anchor tracks the leader (unchanged). |
| HOLD | — | in place | no | Hold the position where the order was received (unchanged). |

OVERWATCH (v1.1) is **removed** — its roles are absorbed by OBSERVE (static
watch on an objective) and SUPPORT (fire support for a unit). For
backwards-friendly parsing, the old phrases `cover obj X` / `support obj X`
parse as OBSERVE at the objective.

## Derivation doctrine

A leader may only order subordinate missions derivable from its **own**
mission (preference-ordered; enforced by action masking):

| Own mission | May order subordinates to… |
|---|---|
| RECON | RECON, SUPPORT, OBSERVE, SCREEN |
| SCREEN | SCREEN, OBSERVE, HOLD |
| OBSERVE | OBSERVE, COVER, HOLD |
| SUPPORT | SUPPORT, OBSERVE, HOLD |
| COVER | COVER, OBSERVE, HOLD |
| DEFEND | DEFEND, SUPPORT, OBSERVE, HOLD |
| DENY | DEFEND, COVER, SUPPORT, OBSERVE |
| SEIZE | SEIZE, CLEAR, SUPPORT, OBSERVE |
| CLEAR | CLEAR, SUPPORT |
| RALLY | RALLY, HOLD |
| HOLD | HOLD, OBSERVE |

Two doctrinal notes:

* **DENY derives DEFEND, not DENY** — INTERDIRE is a section mission
  executed *through* group-level TENIR/COUVRIR ("le groupe sera le plus
  souvent chargé de cette mission [TENIR] dans le cadre d'une manœuvre
  défensive de la section (TENIR, INTERDIRE)", p. 37 note 1). No echelon in
  the sim can pass DENY down.
* **SUPPORT appears in every combat mission's derivations** — "pas un pas
  sans appui" is the PROTERRE maneuver principle: recon, defense, and
  assault all decompose into a supported element and a supporting element
  (cf. RECONNAÎTRE step 4 "APPUYER", p. 30).

## SUPPORT mechanics ("pas un pas sans appui")

The SUPPORT catalog entries are `ORDER_S{i}_SUPPORT_U{j}` (i ≠ j): the
subordinate in slot *i* supports the unit led by the subordinate in slot *j*
(or that soldier itself if it leads no one). On the net the order names the
supported callsign — `TL2, THIS IS SL1: SUPPORT TL1. OUT.`

While a supporter is **in SUPPORT position** (≤ 10 cells of the supported
soldier, LOS to it — "la liaison à vue avec l'élément appuyé", p. 35):

* **Covered movement** — an enemy firing at a member of the supported
  element (the supported soldier + its living direct subordinates) from
  inside the supporter's umbrella (`CombatParams.support_umbrella`, 8 cells)
  suffers an accuracy multiplier of ×0.7 (`support_cover_accuracy`).
* **Focus fire** — while a support relation is active, when ≥ 2 friendlies
  fire at the same enemy in the same step, each shot after the first gets
  hit probability ×1.15 (`focus_fire_bonus`); any final hit probability is
  capped at 0.95 (`max_hit`).

Both effects switch off the moment the supporter leaves its position (the
umbrella is computed from each step's snapshot), and are visible to
external observers via the oracle's `supporting` / `supported` tags —
never to the cohort itself.

## Observation progress (the anti-stall clause)

Root RECON/SCREEN campaigns pay `RewardConfig.observe_progress` (+0.3) for
each **novel** step of team observation toward the 10-step success counter.
The payout telescopes — bounded by the success threshold (at most 3.0 per
episode) — so observing the objective is worth approaching it, but parking
outside the trigger radius farms nothing. This closes the stall exploit
found in the v1.2 A2/A7 campaign, where strict weapons-tight economics made
the policy rationally abandon the observation task.

## Compliance summary

| Mission | In position | Approaching | Notes |
|---|---|---|---|
| RECON / SCREEN | 0.6 | progress | SCREEN: −0.6 if fired |
| OBSERVE / SUPPORT | 0.6 static / 0.1 moving | progress | |
| COVER | 0.5 static / 0.1 moving | progress | |
| DEFEND / DENY / SEIZE / RALLY | 0.5 | progress | |
| CLEAR | 0.8 if firing | progress | 0.0 while enemies visible and not firing |
| HOLD | 0.5 static / 0.1 moving | progress | |

Fire-discipline factors (`RewardConfig.fire_discipline`): SCREEN → 0
(weapons tight); OBSERVE / SUPPORT / COVER / DEFEND / DENY / HOLD → 1 from
position, else 0; RECON / SEIZE / CLEAR / RALLY / untasked → 1.
