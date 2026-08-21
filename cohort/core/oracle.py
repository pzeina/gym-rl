"""Ground-truth oracle observer: what an all-seeing referee knows, per step.

This module defines the vocabulary of *behavior observables* — semantic tags
(ATTACKING, RETREATING, COVERING, WOUNDED, HIDDEN...) describing what a unit
is doing — and computes them for **every** unit on the map, friendly and
OpFor alike, straight from simulation ground truth (including the enemy AI's
internal state: mode, goal, last sighting).

These observables are strictly **outside the simulation loop**:

* they never enter any agent observation (the observation layout is untouched),
* they never influence rewards, masks, or the OpFor AI,
* computing them consumes no randomness, so calling the oracle cannot perturb
  a seeded episode.

Their purpose is external analysis: an assurance layer can treat the enemy
side of this snapshot as *hidden* ground truth and evaluate how well it can
be inferred from the friendly side only (own units + radio traffic) — e.g.
"was that unseen enemy retreating or repositioning to attack?" — without the
cohort ever having access to the answer.

Vocabulary mapping from the design request: attacking → ATTACKING,
retreat → RETREATING, protect/cover → COVERING, wounded → WOUNDED,
hidden → HIDDEN; completed with ADVANCING, HOLDING, and DOWN so every unit
state has at least one tag.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from cohort.core.world import dist

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cohort.core.world import Coord

#: Health strictly below this fraction of full strength counts as WOUNDED.
WOUNDED_BELOW = 50


class Observable(str, Enum):
    """Behavior tags an omniscient referee can assign to any unit."""

    ATTACKING = "attacking"    # fired this step
    ADVANCING = "advancing"    # moved, closing on the nearest living opponent
    RETREATING = "retreating"  # moved, opening distance from the nearest living opponent
    COVERING = "covering"      # static, holding an opponent in line of sight and range
    HOLDING = "holding"        # static, no opponent covered, not firing
    HIDDEN = "hidden"          # concealed: in cover and seen by no living opponent
    WOUNDED = "wounded"        # health below WOUNDED_BELOW (and alive)
    SUPPORTING = "supporting"  # in SUPPORT position for its supported unit (P2)
    SUPPORTED = "supported"    # member of an element under an active support umbrella
    DOWN = "down"              # dead — always the only tag


def unit_observables(
    *,
    alive: bool,
    health: int,
    pos: Coord,
    prev_pos: Coord,
    fired: bool,
    in_cover: bool,
    seen_by_any_opponent: bool,
    opponents: Sequence[Coord],
    weapon_range: float,
    has_los: Callable[[Coord, Coord], bool],
    supporting: bool = False,
    supported: bool = False,
) -> list[Observable]:
    """Pure tag computation for one unit. Deterministic; consumes no RNG."""
    if not alive:
        return [Observable.DOWN]
    tags: list[Observable] = []
    if fired:
        tags.append(Observable.ATTACKING)
    if supporting:
        tags.append(Observable.SUPPORTING)
    if supported:
        tags.append(Observable.SUPPORTED)
    moved = tuple(pos) != tuple(prev_pos)
    if opponents:
        d_now = min(dist(pos, o) for o in opponents)
        d_prev = min(dist(prev_pos, o) for o in opponents)
        if moved and d_now < d_prev - 1e-9:
            tags.append(Observable.ADVANCING)
        elif moved and d_now > d_prev + 1e-9:
            tags.append(Observable.RETREATING)
    if not moved and not fired:
        covering = any(
            dist(pos, o) <= weapon_range and has_los(pos, o) for o in opponents
        )
        tags.append(Observable.COVERING if covering else Observable.HOLDING)
    if in_cover and not seen_by_any_opponent:
        tags.append(Observable.HIDDEN)
    if health < WOUNDED_BELOW:
        tags.append(Observable.WOUNDED)
    return tags


def _sighting_set(world, combat, pos: Coord, living_enemies) -> list[int]:
    """Living enemy ids visible from ``pos``, nearest first.

    Exactly the environment's own visibility rule
    (``CohortEnv._visible_enemies``): alive, inside the applicable range
    (shorter into forest), and with line of sight. Kept here as one function
    so the truth stream and the simulation cannot drift apart.
    """
    visible = [
        e
        for e in living_enemies
        if world.can_spot(pos, e.pos, combat.vision_range, combat.forest_vision_range)
    ]
    visible.sort(key=lambda e: (dist(pos, e.pos), e.id))
    return [e.id for e in visible]


def observe(env) -> dict:
    """Full ground-truth snapshot of the environment for external observers.

    Call after ``reset()`` or after each ``step()``. The ``enemies`` side —
    including the OpFor AI's internal state — is the "non-observable" ground
    truth an assurance layer may try to infer from the friendly side alone.

    **Sighting sets (issue #17).** Each soldier record carries ``sees`` — the
    living enemies that agent can see this step — so a consumer can score
    sighting-knowledge verdicts ("does this station know there is an enemy at
    grid X, and since when") against truth the same way command-state
    verdicts are scored against it. It is *truth*, not belief: what the agent
    can see, never what it has been told. The reported team picture stays off
    this stream deliberately — deriving it from CONTACT traffic is the
    external observer's job, and handing it over would score belief against
    itself.

    A dead unit sees nothing and is seen by nothing: ``sees`` and ``seen_by``
    are empty for it. Before this was enforced, ``seen_by`` was computed from
    a corpse's last position, so most sighting entries in the stream named
    dead units (measured on the shipped checkpoints: 8,901 of 9,647 enemy
    entries in ``squad``), and transposing ``seen_by`` into per-agent sighting
    sets disagreed with the environment's own visibility on 36-47% of
    agent-steps.
    """
    world = env.world
    combat = env.combat
    living_soldiers = [s for s in env.roster.soldiers if s.alive]
    living_enemies = [e for e in env.enemies if e.alive]
    los = world.line_of_sight

    # active SUPPORT relations (P2): supporters currently in position, and
    # the members of the elements they cover — recomputed from current
    # positions, consuming no randomness
    supporting_ids: set[int] = set()
    supported_ids: set[int] = set()
    for supporter, supported in env._active_supports():
        supporting_ids.add(supporter.id)
        supported_ids.update(env._supported_element(supported))

    soldiers = []
    for s in env.roster.soldiers:
        seen_by = (
            [
                e.id
                for e in living_enemies
                if world.can_spot(e.pos, s.pos, combat.vision_range, combat.forest_vision_range)
            ]
            if s.alive
            else []
        )
        sees = _sighting_set(world, combat, s.pos, living_enemies) if s.alive else []
        soldiers.append(
            {
                "cs": s.callsign,
                "rank": s.rank.name,
                "eff": s.effective_rank.name,
                "pos": list(s.pos),
                "prev_pos": list(s.prev_pos),
                "hp": s.health,
                "ammo": s.ammo,
                "alive": s.alive,
                "cover": bool(world.cover_at(s.pos)),
                "fired": s.fired_this_step,
                "mission": s.mission.type.name if s.mission else None,
                "formation": s.formation.name if s.formation is not None else None,
                # A5-4: inside an active trinôme sync window (post-GO)
                "synced": env._synchronized(s) is not None,
                # comms discipline (A4): True when this agent attempted a
                # learned transmission last step and lost net arbitration
                "net_busy": s.callsign in env._net_blocked,
                "seen_by": seen_by,
                # this agent's own sighting set (issue #17): the living
                # enemies it can see RIGHT NOW, nearest first — ground truth,
                # never belief. Identical to ``env._visible_enemies(s)``.
                "sees": sees,
                # tactical acoustics (§3.6): the coarse cue memory this agent
                # holds — exactly what its observation encodes, exposed here
                # so exposure/behavior analysis can audit it. Empty under
                # sound_model="off".
                "cues": [
                    {
                        "kind": c.kind,
                        "side": c.side,
                        "bearing": c.bearing,
                        "band": c.distance_band,
                        "strength": c.strength,
                        "age": c.age(env._step_count),
                    }
                    for c in env._agent_cues.get(s.callsign, [])
                ],
                "tags": [
                    t.value
                    for t in unit_observables(
                        alive=s.alive,
                        health=s.health,
                        pos=s.pos,
                        prev_pos=s.prev_pos,
                        fired=s.fired_this_step,
                        in_cover=world.cover_at(s.pos),
                        seen_by_any_opponent=bool(seen_by),
                        opponents=[e.pos for e in living_enemies],
                        weapon_range=combat.weapon_range,
                        has_los=los,
                        supporting=s.id in supporting_ids,
                        supported=s.id in supported_ids,
                    )
                ],
            }
        )

    enemies = []
    for e in env.enemies:
        seen_by = (
            [
                s.callsign
                for s in living_soldiers
                if world.can_spot(s.pos, e.pos, combat.vision_range, combat.forest_vision_range)
            ]
            if e.alive
            else []
        )
        enemies.append(
            {
                "id": e.id,
                "pos": list(e.pos),
                "prev_pos": list(e.prev_pos),
                "hp": e.health,
                "alive": e.alive,
                "cover": bool(world.cover_at(e.pos)),
                "fired": e.fired_this_step,
                # OpFor AI internals — ground truth the cohort never sees
                "mode": e.mode,
                "home": list(e.home),
                "goal": list(e.goal) if e.goal is not None else None,
                "last_seen_player": list(e.last_seen_player) if e.last_seen_player else None,
                "last_seen_step": e.last_seen_step,
                # tactical acoustics (§3.6.4): this member's own frozen
                # estimated anchor from the last Blue sound it detected —
                # NEVER written into last_seen_player, and per-member so one
                # hearing cannot become band-wide knowledge
                "heard_blue_anchor": (
                    list(e.heard_blue_anchor) if e.heard_blue_anchor is not None else None
                ),
                "heard_blue_step": e.heard_blue_step,
                # BRIQUE per-member behavior state ("posted", "volleying",
                # "sniping", "displacing", "raiding", "fleeing"...); None for
                # the scripted garrison/assault OpFor
                "behavior": e.behavior or None,
                "seen_by": seen_by,
                "tags": [
                    t.value
                    for t in unit_observables(
                        alive=e.alive,
                        health=e.health,
                        pos=e.pos,
                        prev_pos=e.prev_pos,
                        fired=e.fired_this_step,
                        in_cover=world.cover_at(e.pos),
                        seen_by_any_opponent=bool(seen_by),
                        opponents=[s.pos for s in living_soldiers],
                        weapon_range=combat.weapon_range,
                        has_los=los,
                    )
                ],
            }
        )

    # BRIQUE non-observables (enemy-side ground truth, ROADMAP v2.0): the
    # band's intent machine, its ambush posts, and every trap location —
    # exactly what an assurance layer should try to infer from the friendly
    # side (radio traffic incl. "HIT A DEVICE" broadcasts) alone.
    band = getattr(env, "band", None)
    band_rec = None
    if band is not None:
        band_rec = {
            "intent": band.intent,
            "sprung": band.sprung,
            "spring_step": band.spring_step,
            "objective": list(band.objective) if band.objective is not None else None,
            "posts": {str(mid): list(pos) for mid, pos in band.posts.items()},
            "strength": band.strength,
        }
    traps = [
        {
            "id": t.id,
            "pos": list(t.pos),
            "damage": t.damage,
            "armed": t.armed,
            "revealed": t.revealed,
        }
        for t in getattr(env, "traps", [])
    ]

    return {
        "step": env._step_count,
        "outcome": env.outcome if hasattr(env, "outcome") else env._episode_outcome,
        # the preparation period (issue #12), both None outside defend
        # scenarios with an ``assault_h_hour`` band:
        #   announced — the step HQ named on the net ("EXPECT ASSAULT AT STEP
        #     65"), repeated here only so a consumer can compare the two
        #     without re-reading the transcript; it is public, and also in
        #     env.briefing().
        #   actual — the step the assault really begins at, drawn per episode
        #     from the band. Ground truth the cohort is never told: it is the
        #     answer to "was the position set in time?", so it belongs on this
        #     side of the line and enters no observable payload.
        "announced_assault_step": env._h_hour_nominal,
        "actual_assault_step": env._h_hour,
        "soldiers": soldiers,
        "enemies": enemies,
        "band": band_rec,
        "traps": traps,
        # tactical acoustics (§3.6): the last step's sound events with full
        # source truth, semantic hearers and non-semantic detectors — oracle
        # material only; the cue an agent holds carries none of this
        "sound_events": [ev.to_record() for ev in getattr(env, "last_sound_events", [])],
        # liaison (§4): every packet's status and the active courier duties,
        # with the fixed anchors they walk toward — ground truth for audit,
        # never an agent input
        "packets": [p.to_record() for p in getattr(env, "packets", [])],
        "liaison": {
            env.roster.by_id[cid].callsign: {
                "packet": task.packet.id,
                "leg": task.leg,
                "anchor": list(task.current_anchor()),
                "dispatched": task.dispatched_step,
                "suspended_mission": (
                    task.suspended_mission.type.name if task.suspended_mission is not None else None
                ),
            }
            for cid, task in getattr(env, "_liaison", {}).items()
            if cid in env.roster.by_id
        },
        # audit metadata for the last step's messages (medium + actual
        # semantic hearers), parallel to env.last_messages
        "message_meta": list(getattr(env, "last_message_meta", [])),
    }
