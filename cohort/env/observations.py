"""Per-agent observation vectors.

Each agent sees: its own state (with *effective* rank, so a promoted acting
leader knows it now commands, and an is-human flag), its standing mission,
its direct leader (incl. whether the leader is human), its direct
subordinates, currently visible enemies, objectives, a comms summary, and a
local terrain patch. Enemy knowledge is deliberately split:

* ``enemy`` slots — what THIS agent can see right now (private).
* ``known enemy`` summary — the team picture, which only contains enemies
  someone has *reported* via CONTACT. Reporting is what turns a private
  sighting into shared knowledge, which is why reporting is worth doing,
  not just worth rewarding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cohort.core.acoustics import (
    CUE_SIDES,
    MAX_CUES,
    SOUND_KINDS,
    SOUND_MEMORY_TTL,
    WEAPON_DETECT_RADIUS,
)
from cohort.core.missions import Formation, MissionType
from cohort.core.ranks import Rank

if TYPE_CHECKING:
    from cohort.core.units import Enemy, Roster, Soldier
    from cohort.core.world import World

RANK_ORDER: tuple[Rank, ...] = (Rank.RFN, Rank.TL, Rank.SL, Rank.PSG, Rank.PL, Rank.XO, Rank.CO)
MISSION_ORDER: tuple[MissionType, ...] = tuple(MissionType)
FORMATION_ORDER: tuple[Formation, ...] = tuple(Formation)

N_SUB_SLOTS = 4
N_ENEMY_SLOTS = 4
N_OBJECTIVE_SLOTS = 4
#: control-measure slots (A5): one per catalog name — 4 waypoints (GOLD /
#: SILVER / COPPER / IRON) + 3 phase lines (AMBER / COBALT / CRIMSON); a
#: scenario using fewer leaves the remaining slots zeroed (like objectives)
N_WAYPOINT_SLOTS = 4
N_PHASE_LINE_SLOTS = 3
#: v1.10: 2 → 3 (7x7 local terrain). At radius 2 an agent standing 5 cells off
#: the objective could not perceive the ``objective_cover`` ring (chebyshev 2
#: around the objective) it is meant to occupy — a defender was being paid for
#: ground it was partly blind to.
PATCH_RADIUS = 3
#: radius of the nearest-cover search (v1.10) — beyond the patch, so an agent
#: can head for a prepared position it cannot yet see in detail
COVER_SEARCH_RADIUS = 8

#: mission block: one-hot over the 12 tasks (11 MICAT + ADVANCE) + has-mission
#: flag + 4 anchor fields (dx, dy, has-objective, age) + 2 pending fields
#: (A5-2: pending flag; time-to-effective — 1.0 while AT MY COMMAND, else
#: remaining/20 capped at 1) + 3 stance one-hot (A5-3: the governing element
#: formation — the agent's own stance if it leads one, else its direct
#: leader's; all zero when no stance applies)
_MISSION_BLOCK = len(MISSION_ORDER) + 1 + 4 + 2 + len(FORMATION_ORDER)

#: self block: x, y, health, ammo (4) + rank one-hot (7) + in-cover + is-human
_SELF_BLOCK = 4 + len(RANK_ORDER) + 1 + 1

#: leader block: present, dx, dy, mission index, leader-is-human
_LEADER_BLOCK = 5

#: sync block (A5-4): pending-bound flag + synchronized-window remaining
_SYNC_BLOCK = 2

#: tempo block (v1.10): episode progress (step / max_steps, every scenario) +
#: time-to-contact (the defend preparation period — 1.0 at reset, counting down
#: to the NOMINAL announced H-hour, 0.0 once it passes or when no H is set)
_TEMPO_BLOCK = 2

#: cover block (v1.10): present / dx / dy to the nearest cover cell within
#: COVER_SEARCH_RADIUS — same encoding as objectives and control measures. A
#: policy paid to occupy a prepared position has to be able to find one.
_COVER_BLOCK = 3

#: comms summary (5) + SITREP due-ness (v1.10: its own slot — it previously
#: overloaded the "known enemy present" flag to keep OBS_DIM unchanged, a
#: compromise the v1.10 space break pays off; the flag now means what it says)
_COMMS_BLOCK = 5 + 1

#: --- degraded communications (docs/degraded-communications.md §5) ---
#: Both blocks are APPENDED after the patch for EVERY comm model / profile,
#: zero-filled when structurally unavailable (sound off, radio modes), so one
#: honest fleet-breaking cycle replaces scenario-specific network shapes.
#:
#: acoustic block: [sound_model active, comm_model is voice_only] (2) +
#: MAX_CUES cues x (kind one-hot 5 + side one-hot 3 + eight-way bearing 8 +
#: distance band 3 + confidence 1 + ttl-remaining 1 = 21) + own last emitted
#: sound (kind one-hot 5 + radius/16 + age 1) + held-reportable-cue flag
#: = 2 + 84 + 7 + 1 = 94
_CUE_FIELDS = len(SOUND_KINDS) + len(CUE_SIDES) + 8 + 3 + 1 + 1
_ACOUSTIC_BLOCK = 2 + MAX_CUES * _CUE_FIELDS + (len(SOUND_KINDS) + 2) + 1

#: cohesion / local-friendly block: element visual link intact (1) +
#: disconnected age (1) + at formation station (1) + normalized formation
#: error (1) + leader currently perceived (1) + leader last-known age (1) +
#: per subordinate slot: perceived (1) + last-known age (1) = 6 + 2*N_SUB = 14
_COHESION_BLOCK = 6 + 2 * N_SUB_SLOTS

#: 13 self + 22 mission/stance + 2 sync + 2 tempo + 3 cover + 5 leader
#: + 5*N_SUB + 4*N_ENEMY + 3*N_OBJ + 3*N_WP + 3*N_PL (control measures:
#: present, dx, dy — for a phase line dx/dy point at its nearest segment
#: point) + 6 comms + patch (98, radius 3)
#: = 13 + 22 + 2 + 2 + 3 + 5 + 20 + 16 + 12 + 12 + 9 + 6 + 98 = 220
#: + 94 acoustic + 14 cohesion (degraded-communications cycle) = 328
#: Observation profiles.
#:
#: ``full`` is the shipped v1.10 vector. ``core`` drops exactly the four blocks
#: v1.10 added — tempo (2), nearest cover (3), the SITREP-due slot (1), and the
#: patch widened from 5x5 to 7x7 (+48) — reproducing the 166-wide vector the
#: fleet trained on through v1.9. 220 - 54 = 166.
#:
#: It exists to bisect the v1.10 space break. Three explanations for the four
#: collapsed v1.10 runs have been tested and killed (``done_false``,
#: ``contact_redundant``, learning rate), leaving the space change standing by
#: elimination rather than by evidence. ``core`` is how that becomes a
#: measurement: same code, same rewards, same scenario, one variable — the
#: width of the input.
#:
#: One honest difference from the real v1.9 vector: that one overloaded the
#: "known enemy present" comms flag to carry SITREP due-ness, to avoid changing
#: OBS_DIM. ``core`` simply omits the slot instead of reproducing the overload,
#: so the two differ ONLY when the SITREP doctrine is active
#: (``ScenarioSpec.sitrep_cadence``). It is off in ``squad_screen``, the
#: scenario this was built to bisect, so there the rebuild is exact.
OBS_PROFILES: tuple[str, ...] = ("full", "core")

#: the pre-v1.10 patch radius (5x5) — see PATCH_RADIUS for why it grew
CORE_PATCH_RADIUS = 2


def patch_radius(profile: str = "full") -> int:
    """Local terrain patch radius for an observation profile."""
    return PATCH_RADIUS if profile == "full" else CORE_PATCH_RADIUS


def obs_dim(profile: str = "full") -> int:
    """Width of the observation vector under ``profile``."""
    if profile not in OBS_PROFILES:
        msg = f"Unknown observation profile {profile!r} (expected one of {OBS_PROFILES})"
        raise ValueError(msg)
    wide = profile == "full"
    return (
        _SELF_BLOCK + _MISSION_BLOCK + _SYNC_BLOCK
        + (_TEMPO_BLOCK + _COVER_BLOCK if wide else 0)
        + _LEADER_BLOCK
        + 5 * N_SUB_SLOTS
        + 4 * N_ENEMY_SLOTS
        + 3 * N_OBJECTIVE_SLOTS
        + 3 * N_WAYPOINT_SLOTS
        + 3 * N_PHASE_LINE_SLOTS
        + (_COMMS_BLOCK if wide else _COMMS_BLOCK - 1)
        + (2 * patch_radius(profile) + 1) ** 2 * 2
        + _ACOUSTIC_BLOCK
        + _COHESION_BLOCK
    )


OBS_DIM = obs_dim("full")

# --- block offsets -------------------------------------------------------- #
# Derived, never hand-written: tests and tools index the layout through these
# so a future layout change breaks the OBS_DIM assertion (a real signal) and
# not a scatter of magic numbers across the suite (noise). Every block the
# writer below emits, in the order it emits them.
OFF_SELF = 0
OFF_MISSION = OFF_SELF + _SELF_BLOCK
OFF_SYNC = OFF_MISSION + _MISSION_BLOCK
OFF_TEMPO = OFF_SYNC + _SYNC_BLOCK
OFF_COVER = OFF_TEMPO + _TEMPO_BLOCK
OFF_LEADER = OFF_COVER + _COVER_BLOCK
OFF_SUBS = OFF_LEADER + _LEADER_BLOCK
OFF_ENEMIES = OFF_SUBS + 5 * N_SUB_SLOTS
OFF_OBJECTIVES = OFF_ENEMIES + 4 * N_ENEMY_SLOTS
OFF_WAYPOINTS = OFF_OBJECTIVES + 3 * N_OBJECTIVE_SLOTS
OFF_PHASE_LINES = OFF_WAYPOINTS + 3 * N_WAYPOINT_SLOTS
OFF_COMMS = OFF_PHASE_LINES + 3 * N_PHASE_LINE_SLOTS
OFF_PATCH = OFF_COMMS + _COMMS_BLOCK
OFF_ACOUSTIC = OFF_PATCH + (2 * PATCH_RADIUS + 1) ** 2 * 2
OFF_COHESION = OFF_ACOUSTIC + _ACOUSTIC_BLOCK

#: within-block field offsets referenced outside this module
SELF_COVER = OFF_SELF + 4 + len(RANK_ORDER)      # standing in cover
SELF_HUMAN = SELF_COVER + 1                      # is-human flag
TEMPO_PROGRESS = OFF_TEMPO                       # step / max_steps
TEMPO_TIME_TO_CONTACT = OFF_TEMPO + 1            # countdown to nominal H
COVER_PRESENT = OFF_COVER                        # nearest-cover present/dx/dy
LEADER_HUMAN = OFF_LEADER + 4                    # leader-is-human flag
COMMS_KNOWN_PRESENT = OFF_COMMS + 2              # a known enemy is on the picture
COMMS_SITREP_DUE = OFF_COMMS + 5                 # SITREP due-ness (v1.10 slot)
ACOUSTIC_SOUND_ON = OFF_ACOUSTIC                 # sound_model active
ACOUSTIC_VOICE_ONLY = OFF_ACOUSTIC + 1           # comm_model is voice_only
ACOUSTIC_CUES = OFF_ACOUSTIC + 2                 # first cue slot
ACOUSTIC_OWN = ACOUSTIC_CUES + MAX_CUES * _CUE_FIELDS   # own last sound
ACOUSTIC_REPORTABLE = ACOUSTIC_OWN + len(SOUND_KINDS) + 2
COHESION_LINK = OFF_COHESION                     # element visual link intact
COHESION_BREAK_AGE = OFF_COHESION + 1
COHESION_STATION = OFF_COHESION + 2
COHESION_FORM_ERR = OFF_COHESION + 3
COHESION_LEADER_SEEN = OFF_COHESION + 4
COHESION_LEADER_AGE = OFF_COHESION + 5
COHESION_SUBS = OFF_COHESION + 6                 # (seen, age) per slot


@dataclass
class AgentView:
    """Per-step, per-agent context the environment hands the obs builder."""

    visible_enemies: list[Enemy] = field(default_factory=list)  # sorted nearest-first
    known_enemies: list[tuple[float, float]] = field(default_factory=list)  # team picture
    step: int = 0
    #: SITREP due-ness in [0, 1] when the reporting doctrine
    #: (``ScenarioSpec.sitrep_cadence``) is active; None → doctrine off.
    #: v1.10: carried in its own slot (it used to overload the comms "known
    #: enemy present" flag to avoid changing OBS_DIM).
    sitrep_due: float | None = None
    #: fraction of the episode elapsed, in [0, 1] — step / max_steps
    episode_progress: float = 0.0
    #: defend preparation period (v1.10): 1.0 at reset falling linearly to 0.0
    #: at the NOMINAL announced H-hour, and 0.0 thereafter. Always 0.0 in
    #: scenarios with no ``assault_h_hour``. The actual arrival is jittered
    #: around the nominal H, so this is a warning, not a guarantee.
    time_to_contact: float = 0.0
    #: trinôme sync (A5-4): the agent is party to a live PREPARE-TO-BOUND
    #: proposal (as proposer or registered peer) awaiting its GO
    sync_pending: bool = False
    #: fraction of the synchronized window remaining after a GO, in [0, 1]
    sync_active: float = 0.0
    # --- degraded communications (§5) ---
    #: tactical acoustics active / comm model is voice_only
    sound_on: bool = False
    voice_only: bool = False
    #: the agent's bounded coarse cue memory (cohort.core.acoustics.AcousticCue),
    #: already pruned to the freshest/strongest MAX_CUES; empty when sound is off
    cues: list = field(default_factory=list)
    #: (kind, base radius, step) of the agent's own last emitted sound, or None
    own_sound: tuple | None = None
    #: a reportable (non-friendly, unexpired) cue or carried acoustic report
    has_reportable_cue: bool = False
    #: element visual link (§3.7): None when the agent is in no element graph
    #: (no leader and no subordinates, or the feature is off), else intact flag
    link_intact: bool | None = None
    link_break_age: int = 0
    #: current formation-station status under the governing stance (None: no
    #: stance governs) and the normalized formation error
    station: bool | None = None
    formation_error: float = 0.0
    #: local friendly perception (voice_only telemetry gating): per related
    #: soldier id -> (currently visible, last known pos, last known mission
    #: type or None, age of the last-known state). None → live telemetry.
    friendly_state: dict | None = None


def _mission_idx(mission_type: MissionType | None) -> float:
    if mission_type is None:
        return 0.0
    return (MISSION_ORDER.index(mission_type) + 1) / len(MISSION_ORDER)


def build_observation(
    soldier: Soldier,
    roster: Roster,
    world: World,
    view: AgentView,
    profile: str = "full",
) -> np.ndarray:
    """Assemble the flat observation vector for one agent.

    ``profile`` selects the layout: ``full`` (v1.10, 220) or ``core``
    (pre-v1.10, 166). See OBS_PROFILES.
    """
    wide = profile == "full"
    w, h = float(world.width), float(world.height)
    diag = float(np.hypot(w, h))
    x, y = float(soldier.pos[0]), float(soldier.pos[1])
    out = np.zeros(obs_dim(profile), dtype=np.float32)
    i = 0

    # --- self (13) ---
    out[i : i + 4] = (x / w, y / h, soldier.health / 100.0, soldier.ammo / 30.0)
    i += 4
    out[i + RANK_ORDER.index(soldier.effective_rank)] = 1.0
    i += len(RANK_ORDER)
    out[i] = 1.0 if world.cover_at(soldier.pos) else 0.0
    i += 1
    out[i] = 1.0 if soldier.human else 0.0
    i += 1

    # --- mission (19) ---
    m = soldier.mission
    if m is not None:
        out[i + MISSION_ORDER.index(m.type)] = 1.0
    i += len(MISSION_ORDER)
    out[i] = 1.0 if m is not None else 0.0
    i += 1
    if m is not None:
        anchor = m.anchor
        if m.type is MissionType.RALLY:
            leader = roster.leader_of(soldier)
            if leader is not None:
                anchor = leader.pos
        elif m.type is MissionType.SUPPORT:
            supported = roster.by_id.get(m.extra.get("supported_id"))
            if supported is not None and supported.alive:
                anchor = supported.pos
        elif m.type is MissionType.ADVANCE and m.extra.get("control") is not None:
            cm = world.control_by_name(m.extra["control"])
            if cm is not None:
                anchor = cm.nearest_point(soldier.pos) if hasattr(cm, "nearest_point") else cm.pos
        out[i] = (anchor[0] - x) / w
        out[i + 1] = (anchor[1] - y) / h
        out[i + 2] = 1.0 if m.objective_id is not None else 0.0
        out[i + 3] = min(1.0, (view.step - m.step_assigned) / 50.0)
    i += 4
    # pending state (A5-2): staged until "AT T PLUS n" comes due or the
    # issuer's EXECUTE releases an "AT MY COMMAND" order
    if m is not None:
        pending = m.awaiting_signal or (
            m.effective_at is not None and view.step < m.effective_at
        )
        if pending:
            out[i] = 1.0
            out[i + 1] = (
                1.0
                if m.awaiting_signal
                else min(1.0, (m.effective_at - view.step) / 20.0)
            )
    i += 2

    # --- governing element stance (A5-3, 3) ---
    leader = roster.leader_of(soldier)
    stance = soldier.formation
    if stance is None and leader is not None:
        stance = leader.formation  # the member is shaped under its leader's stance
    if stance is not None:
        out[i + FORMATION_ORDER.index(stance)] = 1.0
    i += len(FORMATION_ORDER)

    # --- trinôme sync (A5-4, 2) ---
    out[i] = 1.0 if view.sync_pending else 0.0
    out[i + 1] = view.sync_active
    i += 2

    # --- tempo (v1.10, 2): episode progress + time to the announced H-hour ---
    if wide:
        out[i] = view.episode_progress
        out[i + 1] = view.time_to_contact
        i += 2

        # --- nearest cover (v1.10, 3): present, dx, dy ---
        cover = world.nearest_cover(soldier.pos, COVER_SEARCH_RADIUS)
        if cover is not None:
            out[i] = 1.0
            out[i + 1] = (cover[0] - x) / w
            out[i + 2] = (cover[1] - y) / h
        i += 3

    # --- leader (5) ---
    # voice_only (§3.7): NOT a live tracker. ``view.friendly_state`` carries
    # what this agent can actually know — a live delta only while the leader
    # is locally visible, otherwise the last perceived delta and the last
    # reported mission, aging where they were captured. Radio modes keep the
    # shipped live telemetry (friendly_state is None).
    fs = view.friendly_state
    if leader is not None:
        out[i] = 1.0
        known = fs.get(leader.id) if fs is not None else None
        if fs is None:
            out[i + 1] = (leader.pos[0] - x) / w
            out[i + 2] = (leader.pos[1] - y) / h
            out[i + 3] = _mission_idx(leader.mission.type if leader.mission else None)
        elif known is not None:
            _seen, last_pos, last_mission, _age = known
            out[i + 1] = (last_pos[0] - x) / w
            out[i + 2] = (last_pos[1] - y) / h
            out[i + 3] = _mission_idx(last_mission)
        else:
            out[i] = 0.0  # relationship exists but nothing is known: false presence
        out[i + 4] = 1.0 if leader.human else 0.0
    i += 5

    # --- direct subordinates (5 each) ---
    subs = soldier.living_subordinates(roster)[:N_SUB_SLOTS]
    for k in range(N_SUB_SLOTS):
        if k < len(subs):
            s = subs[k]
            known = fs.get(s.id) if fs is not None else None
            if fs is None:
                out[i] = 1.0
                out[i + 1] = (s.pos[0] - x) / w
                out[i + 2] = (s.pos[1] - y) / h
                out[i + 3] = _mission_idx(s.mission.type if s.mission else None)
                out[i + 4] = 1.0 if view.step - s.last_contact_report_step <= 10 else 0.0
            elif known is not None:
                seen, last_pos, last_mission, age = known
                out[i] = 1.0
                out[i + 1] = (last_pos[0] - x) / w
                out[i + 2] = (last_pos[1] - y) / h
                out[i + 3] = _mission_idx(last_mission)
                # "reported recently" is something the observer heard, so it
                # is only asserted from a fresh (<= 10 step) perception/report
                out[i + 4] = 1.0 if (seen and view.step - s.last_contact_report_step <= 10) else 0.0
        i += 5

    # --- visible enemies (4 each) ---
    for k in range(N_ENEMY_SLOTS):
        if k < len(view.visible_enemies):
            e = view.visible_enemies[k]
            d = float(np.hypot(e.pos[0] - x, e.pos[1] - y))
            out[i] = 1.0
            out[i + 1] = (e.pos[0] - x) / w
            out[i + 2] = (e.pos[1] - y) / h
            out[i + 3] = d / diag
        i += 4

    # --- objectives (3 each) ---
    for k in range(N_OBJECTIVE_SLOTS):
        if k < len(world.objectives):
            obj = world.objectives[k]
            out[i] = 1.0
            out[i + 1] = (obj.pos[0] - x) / w
            out[i + 2] = (obj.pos[1] - y) / h
        i += 3

    # --- control measures (A5): waypoints then phase lines (3 each) ---
    for k in range(N_WAYPOINT_SLOTS):
        if k < len(world.waypoints):
            wp = world.waypoints[k]
            out[i] = 1.0
            out[i + 1] = (wp.pos[0] - x) / w
            out[i + 2] = (wp.pos[1] - y) / h
        i += 3
    for k in range(N_PHASE_LINE_SLOTS):
        if k < len(world.phase_lines):
            pl = world.phase_lines[k]
            near = pl.nearest_point(soldier.pos)
            out[i] = 1.0
            out[i + 1] = (near[0] - x) / w
            out[i + 2] = (near[1] - y) / h
        i += 3

    # --- comms summary (6) ---
    out[i] = 1.0 if view.step - soldier.last_order_step <= 1 else 0.0
    out[i + 1] = min(1.0, len(view.known_enemies) / 4.0)
    if view.known_enemies:
        nearest = min(view.known_enemies, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
        out[i + 2] = 1.0
        out[i + 3] = (nearest[0] - x) / w
        out[i + 4] = (nearest[1] - y) / h
    # v1.10: SITREP due-ness has its own slot — it no longer displaces the
    # "known enemy present" flag above (0.0 when the doctrine is off)
    if wide:
        out[i + 5] = view.sitrep_due if view.sitrep_due is not None else 0.0
        i += 6
    else:
        i += 5

    # --- terrain patch ---
    patch = world.local_patch(soldier.pos, patch_radius(profile)).reshape(-1)
    out[i : i + patch.shape[0]] = patch
    i += patch.shape[0]

    # --- acoustic block (degraded communications §5) ---
    # coarse cues only: kind / attributed side / bearing sector / distance
    # band / confidence / time left — never a source id, a cell, or text
    out[i] = 1.0 if view.sound_on else 0.0
    out[i + 1] = 1.0 if view.voice_only else 0.0
    i += 2
    for k in range(MAX_CUES):
        if k < len(view.cues):
            c = view.cues[k]
            out[i + SOUND_KINDS.index(c.kind)] = 1.0
            out[i + len(SOUND_KINDS) + CUE_SIDES.index(c.side)] = 1.0
            out[i + len(SOUND_KINDS) + len(CUE_SIDES) + (c.bearing % 8)] = 1.0
            base = i + len(SOUND_KINDS) + len(CUE_SIDES) + 8
            out[base + min(c.distance_band, 2)] = 1.0
            out[base + 3] = float(c.strength)
            out[base + 4] = max(0.0, c.ttl_remaining(view.step)) / SOUND_MEMORY_TTL
        i += _CUE_FIELDS
    # own last emitted sound: the consequence of the previous choice, without
    # ever seeing who heard it
    if view.own_sound is not None:
        kind, radius, at = view.own_sound
        out[i + SOUND_KINDS.index(kind)] = 1.0
        out[i + len(SOUND_KINDS)] = min(1.0, float(radius) / WEAPON_DETECT_RADIUS)
        out[i + len(SOUND_KINDS) + 1] = min(1.0, (view.step - at) / SOUND_MEMORY_TTL)
    i += len(SOUND_KINDS) + 2
    out[i] = 1.0 if view.has_reportable_cue else 0.0
    i += 1

    # --- cohesion / local-friendly block (§3.7) ---
    # no teammate's selected action, logits, route or future position here:
    # link state, station keeping, and how stale the last-known state is
    out[i] = 1.0 if view.link_intact else 0.0
    out[i + 1] = min(1.0, view.link_break_age / 20.0)
    out[i + 2] = 1.0 if view.station else 0.0
    out[i + 3] = float(view.formation_error)
    if leader is not None:
        if fs is None:
            out[i + 4] = 1.0  # live telemetry: always "perceived", age 0
        elif fs.get(leader.id) is not None:
            seen, _p, _m, age = fs[leader.id]
            out[i + 4] = 1.0 if seen else 0.0
            out[i + 5] = min(1.0, age / 20.0)
    i += 6
    for k in range(N_SUB_SLOTS):
        if k < len(subs):
            s = subs[k]
            if fs is None:
                out[i] = 1.0
            elif fs.get(s.id) is not None:
                seen, _p, _m, age = fs[s.id]
                out[i] = 1.0 if seen else 0.0
                out[i + 1] = min(1.0, age / 20.0)
        i += 2

    expected = obs_dim(profile)
    assert i == expected, f"obs layout mismatch: wrote {i}, expected {expected}"
    return np.clip(out, -1.0, 1.0)
