"""Scenario specifications: org charts, maps, OpFor, and the initial OPORD."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from cohort.core.missions import MissionType
from cohort.core.ranks import Rank
from cohort.core.units import BriqueBandConfig, CombatParams


@dataclass(frozen=True)
class OrgSlot:
    """One position in the org chart. ``leader`` indexes into the org list."""

    rank: Rank
    leader: int | None            # index of direct leader slot, None → reports to HQ
    deputy: bool = False          # designated successor of its leader


def _fireteam(leader_of: int | None, offset: int) -> list[OrgSlot]:
    """TL + 2 RFN."""
    return [
        OrgSlot(Rank.TL, leader_of),
        OrgSlot(Rank.RFN, offset),
        OrgSlot(Rank.RFN, offset),
    ]


def build_org(kind: str) -> list[OrgSlot]:
    """Return the org chart for a unit type.

    fireteam: TL + 3 RFN                                   (4 agents)
    squad:    SL + 2 fire teams (TL + 2 RFN each)          (7 agents)
    platoon:  PL + PSG (deputy) + 2 squads                 (16 agents)
    """
    if kind == "fireteam":
        return [
            OrgSlot(Rank.TL, None),
            OrgSlot(Rank.RFN, 0),
            OrgSlot(Rank.RFN, 0),
            OrgSlot(Rank.RFN, 0),
        ]
    if kind == "squad":
        org = [OrgSlot(Rank.SL, None)]
        org += _fireteam(0, 1)   # slots 1..3
        org += _fireteam(0, 4)   # slots 4..6
        return org
    if kind == "platoon":
        org = [
            OrgSlot(Rank.PL, None),
            OrgSlot(Rank.PSG, 0, deputy=True),
        ]
        # squad 1: slots 2..8
        org += [OrgSlot(Rank.SL, 0)]
        org += _fireteam(2, 3)
        org += _fireteam(2, 6)
        # squad 2: slots 9..15
        org += [OrgSlot(Rank.SL, 0)]
        org += _fireteam(9, 10)
        org += _fireteam(9, 13)
        return org
    msg = f"Unknown org kind: {kind!r} (expected fireteam | squad | platoon)"
    raise ValueError(msg)


@dataclass(frozen=True)
class ScenarioSpec:
    """Everything needed to instantiate an episode family."""

    name: str
    description: str
    org: str                                   # fireteam | squad | platoon
    map_size: tuple[int, int]                  # (width, height)
    objectives: tuple[tuple[str, tuple[int, int]], ...]
    spawn: tuple[int, int]                     # friendly spawn anchor
    n_enemies: int
    opfor_mode: str                            # garrison (hold objectives) | assault (advance
    #                                            on spawn objective) | brique (armed band —
    #                                            the manual's asymmetric threat, p. 9)
    root_mission: MissionType
    root_objective: str | None                 # objective name for the OPORD
    max_steps: int
    # --- control measures (A5): named terrain the vocabulary can reference ---
    waypoints: tuple[tuple[str, tuple[int, int]], ...] = ()
    #                               named points ("GOLD"/"SILVER"/"COPPER"/"IRON",
    #                               radius 2.5) an ADVANCE order can anchor on —
    #                               they name the terrain routes actually pass
    #                               through (the B4/B5 dogleg finding)
    phase_lines: tuple[tuple[str, tuple[int, int], tuple[int, int]], ...] = ()
    #                               named straight segments ("AMBER"/"COBALT"/
    #                               "CRIMSON"); ADVANCE TO PL X completes on
    #                               reaching/crossing the line
    forest_density: float = 1.0
    wall_density: float = 1.0
    combat: CombatParams = field(default_factory=CombatParams)
    experiment_arm: str | None = None
    #                             a named experimental arm of another scenario, for
    #                             labelling ONLY — it never changes behavior. The
    #                             `ablation` field cannot serve here: it drives
    #                             `env/actions.compute_mask`, so labelling through it
    #                             would silently alter the order vocabulary. Arms that
    #                             differ by a tuned parameter (e.g. `squad_short_vision`)
    #                             set this so the dashboard picker can tell them apart
    #                             from their control, which `tests/test_dashboard.py`
    #                             requires of every registered scenario.
    root_human: bool = True       # the root commander is a human embodied in the sim
    #                               (observable to teammates; its death costs the
    #                               rank-weighted teammate penalty, plus
    #                               RewardConfig.human_death for everyone when that knob
    #                               is enabled — it is 0.0 by default since v1.10; the
    #                               episode continues and succession exercises). The org
    #                               must satisfy the humans-outrank-all-non-humans
    #                               invariant, validated at roster build.
    # --- net protocol knobs (defaults preserve the shipped behavior) ---
    auto_ack: bool = True         # False → orders are not auto-acknowledged (no WILCO)
    order_cooldown: int = 8       # steps a leader cannot re-task the same subordinate
    #                               (masked); lifted early if the leader's own mission
    #                               changed or a CONTACT hit the net since. 0 → off.
    done_cooldown: int = 8        # steps an agent cannot re-claim MISSION COMPLETE
    #                               after a DONE_REJECT (masked). A rejected claim
    #                               does not clear the mission, and DONE was
    #                               admissible on EVERY step, so a premature claimant
    #                               could re-roll each tick until one landed — the
    #                               structural half of the 53-84% false-COMPLETE rate
    #                               B2 measured. Mirrors order_cooldown, the mechanism
    #                               that made orders bind in B5: price the act, and
    #                               rate-limit the retry. 0 → off (pre-v1.10 behavior).
    grace_window: int = 12        # steps the episode stays open after the root-mission
    #                               success condition is first met, giving the root time
    #                               to transmit MISSION COMPLETE; a truthful root DONE
    #                               ends the episode that step, otherwise it ends as
    #                               success at the window's end anyway. 0 → immediate
    #                               termination (pre-v1.2 behavior).
    comm_model: str = "global"    # "global" → every station hears every message (the
    #                               shipped behavior); "range" → a message is heard only
    #                               by stations within comm_range of the sender
    #                               (euclidean; the sender always hears itself; HQ is a
    #                               high-power station: HQ traffic is always heard and
    #                               HQ always hears the root).
    comm_range: float = 12.0      # audible radius under comm_model="range"
    voice_range: float = 6.0      # shouting distance (A5-4): trinôme sync proposals
    #                               register the peers within this radius at
    #                               propose time; voice traffic is not radio —
    #                               never net-arbitrated, never airtime-costed
    sitrep_cadence: int | None = None  # reporting doctrine: an agent not in contact
    #                               owes a SITREP every this-many steps; overdue draws
    #                               RewardConfig.sitrep_overdue per step and is surfaced
    #                               in the agent's observation. None (default) → no
    #                               doctrine, the shipped behavior.
    # --- defensive-scenario terrain doctrine ---
    objective_cover: bool = False # True → guarantee defensible ground: the cells ringing
    #                               the root objective (chebyshev distance 2) become
    #                               forest — a defense presumes prepared positions, and
    #                               random map generation may otherwise leave the
    #                               objective bare.
    assault_spawn_min_dist: float = 10.0  # minimum distance from the assaulted objective
    #                               at which "assault"-mode OpFor spawns; larger values
    #                               model the early warning a prepared defense earns.
    assault_h_hour: tuple[int, int] | None = None
    #                               preparation period (v1.10): inclusive (lo, hi) band
    #                               the actual H-hour is drawn from at reset. Before H
    #                               the OpFor is on the map but does not move, fire, or
    #                               advance — the defense gets time to occupy its
    #                               prepared positions, which is what a DEFEND mission
    #                               presumes. The OPORD announces the band's MIDPOINT as
    #                               the nominal H (and drives the time_to_contact
    #                               observation), so the arrival is approximately, not
    #                               exactly, known: a defense must be set early rather
    #                               than timed to the tick. None → no preparation
    #                               period, the shipped behavior.
    observation_concealment: bool = False  # True → guarantee concealed observation
    #                               positions: small forest patches on the ring at
    #                               observation distance (~6 cells) around the root
    #                               objective. Close reconnaissance of a garrisoned
    #                               objective over featureless ground is impossible in
    #                               reality too — recon presumes concealed OPs.
    # --- B3 hierarchy ablation (ROADMAP B3) — arms are env knobs ---
    ablation: str = "full"        # "full" → the shipped system (hierarchy + doctrine
    #                               masks). "nomask" → hierarchy WITHOUT doctrine masks:
    #                               a leader may issue any rank-admissible order
    #                               regardless of its own mission, even with none (the
    #                               doctrine-derivation constraint is removed from the
    #                               order mask; rank admissibility, per-echelon hold
    #                               authority and the order cooldown stay; rewards are
    #                               untouched — doctrine preference remains a soft
    #                               signal). "flat" → no ranks in effect: order actions
    #                               masked off for EVERYONE, every agent receives the
    #                               OPORD mission directly at reset (all-tasked), comms
    #                               limited to reports (CONTACT/SITREP/DONE), and the
    #                               leader coverage reward is neutralized (all-tasked by
    #                               construction, it would otherwise pay for free).
    #                               Spaces are frozen across arms: masking-only changes.
    # --- BRIQUE asymmetric OpFor (manual p. 9; used when opfor_mode="brique") ---
    band: BriqueBandConfig = field(default_factory=BriqueBandConfig)
    #                               intent machine tunables of the armed band
    n_traps: int = 0              # hidden devices (mines/booby traps) the band lays near
    #                               blue's likely route / the objective approaches at
    #                               reset. Each damages the first friendly stepping on
    #                               it (revealed once triggered). Oracle ground truth
    #                               from step 0; never in blue observations.

    #: which observation layout the scenario presents — "full" (v1.10, 220
    #: wide) or "core" (pre-v1.10, 166). A bisect knob, not a feature: the
    #: v1.10 space break is the last unfalsified explanation for the four
    #: collapsed v1.10 runs, and an arm that differs from its control ONLY in
    #: the width of the input is how that stops being a guess. See
    #: cohort.env.observations.OBS_PROFILES. Checkpoints are not portable
    #: across profiles — the network's first layer is a different shape.
    observation_profile: str = "full"

    def __post_init__(self) -> None:
        if self.ablation not in ("full", "nomask", "flat"):
            msg = f"Unknown ablation arm {self.ablation!r} (expected full | nomask | flat)"
            raise ValueError(msg)
        from cohort.env.observations import OBS_PROFILES

        if self.observation_profile not in OBS_PROFILES:
            msg = (
                f"Unknown observation profile {self.observation_profile!r} "
                f"(expected one of {OBS_PROFILES})"
            )
            raise ValueError(msg)
        from cohort.core.language import PHASE_LINE_NAMES, WAYPOINT_NAMES

        for name, _pos in self.waypoints:
            if name not in WAYPOINT_NAMES:
                msg = f"Unknown waypoint name {name!r} (expected one of {WAYPOINT_NAMES})"
                raise ValueError(msg)
        for name, _a, _b in self.phase_lines:
            if name not in PHASE_LINE_NAMES:
                msg = f"Unknown phase-line name {name!r} (expected one of {PHASE_LINE_NAMES})"
                raise ValueError(msg)


# v1.4 (P5): all maps, objective/spawn coordinates, and step budgets grew x1.5
# (coordinates rounded half-up) — more ground to cover makes screening lines,
# support umbrellas, and bounding movement meaningful.
SCENARIOS: dict[str, ScenarioSpec] = {
    "fireteam": ScenarioSpec(
        name="fireteam",
        description="A fire team (TL + 3 RFN) seizes OBJ ALPHA held by a small OpFor garrison.",
        org="fireteam",
        map_size=(36, 36),
        objectives=(("ALPHA", (27, 27)), ("BRAVO", (29, 6))),
        spawn=(5, 5),
        # A5 control measures: GOLD names the mid-route ground the assault
        # actually moves through; AMBER is the final-approach line to ALPHA.
        waypoints=(("GOLD", (16, 16)),),
        phase_lines=(("AMBER", (18, 28), (28, 18)),),
        n_enemies=3,
        opfor_mode="garrison",
        root_mission=MissionType.SEIZE,
        root_objective="ALPHA",
        max_steps=300,
    ),
    "fireteam_defend": ScenarioSpec(
        name="fireteam_defend",
        description="A fire team defends OBJ ALPHA against an OpFor assault.",
        org="fireteam",
        map_size=(36, 36),
        objectives=(("ALPHA", (18, 18)),),
        spawn=(17, 17),
        # control measures for the defense: GOLD is a southern outpost point,
        # AMBER the northern-approach trigger line
        waypoints=(("GOLD", (18, 26)),),
        phase_lines=(("AMBER", (8, 10), (28, 10)),),
        n_enemies=4,
        opfor_mode="assault",
        root_mission=MissionType.DEFEND,
        root_objective="ALPHA",
        # v1.10: 375 → 450, buying the preparation period below without
        # shortening the fight it precedes
        max_steps=450,
        # defensive doctrine: prepared positions + early warning (see ROADMAP —
        # three trainings on the bare spec all plateaued at a ~55-60% coin-flip
        # brawl; a defense without defensible ground isn't a defense)
        objective_cover=True,
        assault_spawn_min_dist=21.0,
        # preparation period (v1.10): the fire team spawns ON the objective, so
        # its problem was never reaching the ground — v7 left it, fighting 9.7
        # cells out with cover occupancy 0.05. A contact-free phase makes
        # occupying the position the only thing worth doing, and makes leaving
        # it expensive: whoever walks out has to walk back before H.
        assault_h_hour=(55, 75),
    ),
    "squad": ScenarioSpec(
        name="squad",
        description="A squad (SL + 2 fire teams) seizes OBJ ALPHA; OpFor garrisons ALPHA and BRAVO.",
        org="squad",
        map_size=(42, 42),
        objectives=(("ALPHA", (33, 33)), ("BRAVO", (35, 9)), ("CHARLIE", (9, 35))),
        spawn=(5, 5),
        # A5 control measures naming the B4/B5 dogleg: trained squads walk a
        # west-then-south axis (closing on CHARLIE mid-transit) before turning
        # east to ALPHA — GOLD names the dogleg corridor, SILVER the eastern
        # leg, AMBER the final-approach line to ALPHA.
        waypoints=(("GOLD", (10, 28)), ("SILVER", (22, 33))),
        phase_lines=(("AMBER", (26, 26), (26, 42)),),
        n_enemies=5,
        opfor_mode="garrison",
        root_mission=MissionType.SEIZE,
        root_objective="ALPHA",
        max_steps=450,
    ),
    "squad_recon": ScenarioSpec(
        name="squad_recon",
        description="A squad reconnoiters OBJ BRAVO (RECONNAÎTRE: may engage if needed).",
        org="squad",
        map_size=(42, 42),
        objectives=(("ALPHA", (33, 33)), ("BRAVO", (35, 9))),
        spawn=(5, 21),
        # GOLD: mid-route bound; AMBER: the line before BRAVO's observation ring
        waypoints=(("GOLD", (20, 15)),),
        phase_lines=(("AMBER", (27, 2), (27, 20)),),
        n_enemies=4,
        opfor_mode="garrison",
        root_mission=MissionType.RECON,
        root_objective="BRAVO",
        max_steps=375,
        # recon doctrine: concealed observation posts must exist (see ROADMAP —
        # under weapons-tight economics on featureless ground, the policy
        # rationally abandoned observation because it cost blood with no recourse)
        observation_concealment=True,
    ),
    "squad_screen": ScenarioSpec(
        name="squad_screen",
        description="A squad screens OBJ BRAVO (ÉCLAIRER: intel WITHOUT engaging — weapons tight).",
        org="squad",
        map_size=(42, 42),
        objectives=(("ALPHA", (33, 33)), ("BRAVO", (35, 9))),
        spawn=(5, 21),
        waypoints=(("GOLD", (20, 15)),),
        phase_lines=(("AMBER", (27, 2), (27, 20)),),
        n_enemies=3,
        opfor_mode="garrison",
        root_mission=MissionType.SCREEN,
        root_objective="BRAVO",
        max_steps=375,
        observation_concealment=True,
    ),
    "patrol_brique": ScenarioSpec(
        name="patrol_brique",
        description=(
            "A squad patrols across ambush country to seize OBJ ALPHA; a BRIQUE armed "
            "band (manual p. 9) is posted in ambush on the route, with mines laid."
        ),
        org="squad",
        # geometry matches the `squad` scenario (its checkpoints transfer): only
        # the OpFor changes — a flat band on the route instead of garrisons
        map_size=(42, 42),
        objectives=(("ALPHA", (33, 33)), ("BRAVO", (35, 9)), ("CHARLIE", (9, 35))),
        spawn=(5, 5),
        waypoints=(("GOLD", (10, 28)), ("SILVER", (22, 33))),
        phase_lines=(("AMBER", (26, 26), (26, 42)),),
        n_enemies=6,
        opfor_mode="brique",
        root_mission=MissionType.SEIZE,
        root_objective="ALPHA",
        max_steps=450,
        band=BriqueBandConfig(initial_intent="ambush"),
        n_traps=3,
        # exercises the manual's react-to-ambush (p. 18) + break-contact (p. 19)
        # drills and SUPPORT bounding ("pas un pas sans appui") on the blue side
    ),
    "defend_brique": ScenarioSpec(
        name="defend_brique",
        description=(
            "A fire team holds OBJ ALPHA against a BRIQUE band that probes, "
            "harasses and raids (manual p. 9), with mines on the approaches."
        ),
        org="fireteam",
        # geometry matches `fireteam_defend` (its checkpoints transfer)
        map_size=(36, 36),
        objectives=(("ALPHA", (18, 18)),),
        spawn=(17, 17),
        waypoints=(("GOLD", (18, 26)),),
        phase_lines=(("AMBER", (8, 10), (28, 10)),),
        n_enemies=5,
        opfor_mode="brique",
        root_mission=MissionType.DEFEND,
        root_objective="ALPHA",
        max_steps=375,
        objective_cover=True,
        assault_spawn_min_dist=21.0,  # the band infiltrates from the far edges
        band=BriqueBandConfig(initial_intent="harass", raid_period=60),
        n_traps=2,
        # BRIQUE terminal semantics: success = band destroyed OR scattered with
        # contact broken, while the objective is held (see docs/architecture.md)
    ),
    "platoon": ScenarioSpec(
        name="platoon",
        description="A platoon (PL, PSG, 2 squads — 16 agents) seizes OBJ ALPHA.",
        org="platoon",
        map_size=(54, 54),
        objectives=(("ALPHA", (44, 44)), ("BRAVO", (45, 11)), ("CHARLIE", (11, 45)), ("DELTA", (27, 27))),
        spawn=(6, 6),
        # GOLD/SILVER: successive bounds on the diagonal axis; COBALT: the
        # line of departure before DELTA; AMBER: the final line before ALPHA.
        waypoints=(("GOLD", (16, 16)), ("SILVER", (36, 36))),
        phase_lines=(("COBALT", (12, 32), (32, 12)), ("AMBER", (28, 48), (48, 28))),
        n_enemies=8,
        opfor_mode="garrison",
        root_mission=MissionType.SEIZE,
        root_objective="ALPHA",
        max_steps=600,
    ),
}


# B3 hierarchy-ablation arms of the `squad` scenario (ROADMAP B3): identical
# geometry, OpFor, rewards, and spaces — only `ablation` differs. Registered as
# first-class presets so training saves and evaluation reloads checkpoints
# under the correct arm (the checkpoint carries the scenario name).
SCENARIOS["squad_nomask"] = replace(
    SCENARIOS["squad"],
    name="squad_nomask",
    description=(
        "B3 ablation arm (ii): the squad scenario with the doctrine-derivation "
        "constraint removed from the order mask (rank admissibility and cooldown stay)."
    ),
    ablation="nomask",
)
SCENARIOS["squad_flat"] = replace(
    SCENARIOS["squad"],
    name="squad_flat",
    description=(
        "B3 ablation arm (iii): flat team — no orders at all; every agent receives "
        "the OPORD directly at reset; comms limited to reports."
    ),
    ablation="flat",
)

# Information-asymmetry probe (v1.11 gate, docs/vision.md §6): the squad
# scenario with eyes halved and nothing else touched — same geometry, OpFor,
# rewards, spaces, and ablation arm. It tests the hypothesis behind directional
# vision *before* the feature is built: if the transparency probe trails the
# OPORD-only baseline because every agent already sees what its neighbours see,
# then shrinking vision should narrow the gap. If the gap does not move, vision
# arcs are unlikely to rescue it either.
#
# The forest ratio is preserved deliberately (6/10 = 3/5): only the *scale* of
# what a soldier can see changes, not how forest attenuates it, so the result
# cannot be confounded by a shift in the cover economics.
#
# It is a lower bound on the effect, not a full proxy: isotropic reduction
# creates asymmetry only between SEPARATED agents, while arcs also split the
# picture between co-located ones. Read a null result accordingly.
SCENARIOS["squad_short_vision"] = replace(
    SCENARIOS["squad"],
    name="squad_short_vision",
    description=(
        "Information-asymmetry probe: the squad scenario with vision halved "
        "(10 → 5 cells, forest 6 → 3) and everything else identical."
    ),
    # derived from the squad's own combat model, not a fresh CombatParams(), so
    # the arm cannot silently drift from its control if squad is ever retuned
    combat=replace(SCENARIOS["squad"].combat, vision_range=5.0, forest_vision_range=3.0),
    experiment_arm="short vision",
)

# Observation-width bisect (2026-08-07): squad_screen with the pre-v1.10
# observation and nothing else touched — same geometry, OpFor, rewards,
# economics, ablation arm and step budget as `squad_screen`.
#
# Why it exists: four v1.10 runs collapsed (`squad`, `fireteam`, `squad_recon`,
# `squad_screen`) while four converged. Three explanations were tested and
# killed — `done_false` (squad_screen_v5 reproduced the collapse at -0.5),
# `contact_redundant` (squad_v6 ran at -0.02 and collapsed anyway), and
# learning rate (squad_screen_v7 at 1e-4 failed a third way, dying rather than
# stalling). The v1.10 space break is what remains, and it remains by
# ELIMINATION, not by evidence. This arm is the evidence: run it against
# `squad_screen` on identical code and the only difference is 220 vs 166
# inputs.
#
# Read a null result as exonerating the space, not as explaining the collapse:
# it would mean all four named suspects are dead and the cause is something
# nobody has proposed yet.
SCENARIOS["squad_screen_core"] = replace(
    SCENARIOS["squad_screen"],
    name="squad_screen_core",
    description=(
        "Observation-width bisect: the squad_screen scenario presented through "
        "the pre-v1.10 166-wide observation, everything else identical."
    ),
    observation_profile="core",
    experiment_arm="core observation",
)


def get_scenario(name: str) -> ScenarioSpec:
    """Look up a scenario preset by name."""
    if name not in SCENARIOS:
        known = ", ".join(sorted(SCENARIOS))
        msg = f"Unknown scenario {name!r}. Available: {known}"
        raise KeyError(msg)
    return SCENARIOS[name]


def announced_assault_step(scenario: str | ScenarioSpec) -> int | None:
    """The step at which HQ *announces* the assault, or None (refs issue #12).

    This is the "H PLUS <n>" the OPORD says on the net: the midpoint of the
    scenario's arrival band (``ScenarioSpec.assault_h_hour``), i.e. the nominal
    hour the cohort plans against. It is a pure function of the scenario, so
    it is announced identically in every episode and known before ``reset()``.

    The step the assault *actually* arrives at is drawn per episode from the
    band and is never said out loud — it stays in ``env.oracle()``
    (``actual_assault_step``). Keeping the announcement's definition here, in
    one place, is what stops the radio wording, the observation countdown and
    the published briefing from drifting apart.
    """
    spec = get_scenario(scenario) if isinstance(scenario, str) else scenario
    band = spec.assault_h_hour
    return None if band is None else (band[0] + band[1]) // 2


def briefing(scenario: str | ScenarioSpec) -> dict:
    """The operations overlay: static, pre-mission, JSON-ready (refs issue #10).

    Everything an observer legitimately holds *before* H-hour, by reading the
    overlay rather than the ground: where the objectives and control measures
    are, how big the map is, where the friendlies come from, what the OPORD
    will task, and the ranges the weapons and eyes work at. It is a pure
    function of the :class:`ScenarioSpec`, so it is identical across every
    episode of a scenario and available before ``reset()`` — which is what
    makes it header material for an episode stream.

    It exists because the alternative is worse: the external assurance layer
    was pinning objective coordinates in a hand-maintained table, which is
    silently era-sensitive — `fireteam_defend` moved OBJ ALPHA from (12,12)
    to (18,18), so re-tapping a `_v4`-era checkpoint against today's table
    produces wrong numbers with no error. Reading them from the scenario the
    checkpoint actually names cannot go stale.

    **No terrain layer, deliberately.** The grid is regenerated at every
    ``reset()`` from the episode seed (``World.generate``), so there is no
    static cover map to publish; ``terrain_static`` says so in the payload
    rather than leaving a consumer to infer it from an absent key. What *is*
    static is ``objective_cover`` — defensive scenarios guarantee the forest
    ring at chebyshev distance 2 around the root objective. Per-step cover is
    ground truth and belongs in ``env.oracle()``; the radio-legitimate view
    of it is the soldier's own SITREP posture (``language.format_sitrep``).
    """
    spec = get_scenario(scenario) if isinstance(scenario, str) else scenario
    return {
        "scenario": spec.name,
        "map_size": list(spec.map_size),
        "objectives": {name: list(pos) for name, pos in spec.objectives},
        "waypoints": {name: list(pos) for name, pos in spec.waypoints},
        "phase_lines": {name: [list(a), list(b)] for name, a, b in spec.phase_lines},
        # anchor tolerances: an objective/waypoint is "reached" inside this
        # radius (World builds both with the Objective/Waypoint default)
        "anchor_radius": 2.5,
        "spawn": list(spec.spawn),
        "org": spec.org,
        "root_mission": spec.root_mission.name,
        "root_objective": spec.root_objective,
        "max_steps": spec.max_steps,
        # the OPORD's forward-looking clause (issue #12): the step HQ names on
        # the net as when to expect the assault ("EXPECT ASSAULT AT H PLUS
        # 65"), or None for a scenario with no preparation period. Announced,
        # therefore header material — a monitor holds the deadline even for a
        # corpus that predates the clause, or a listener that never heard it.
        # The arrival band it is drawn from is deliberately NOT published: the
        # spread between the announcement and the actual arrival is what an
        # outside monitor is meant to characterise from behaviour, and the
        # actual arrival itself stays in env.oracle().
        "announced_assault_step": announced_assault_step(spec),
        # doctrinal terrain guarantees — static facts about the map family,
        # unlike the grid itself
        "objective_cover": spec.objective_cover,
        "observation_concealment": spec.observation_concealment,
        "terrain_static": False,
        # the engagement envelope, so an outside monitor can define "under
        # threat" the way the eval standard does (see metrics.py, issue #11)
        "weapon_range": spec.combat.weapon_range,
        "vision_range": spec.combat.vision_range,
        "forest_vision_range": spec.combat.forest_vision_range,
    }
