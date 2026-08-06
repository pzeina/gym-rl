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
    forest_density: float = 1.0
    wall_density: float = 1.0
    combat: CombatParams = field(default_factory=CombatParams)
    root_human: bool = True       # the root commander is a human embodied in the sim
    #                               (observable to teammates; its death costs
    #                               RewardConfig.human_death for everyone, the episode
    #                               continues and succession exercises). The org must
    #                               satisfy the humans-outrank-all-non-humans
    #                               invariant, validated at roster build.
    # --- net protocol knobs (defaults preserve the shipped behavior) ---
    auto_ack: bool = True         # False → orders are not auto-acknowledged (no WILCO)
    order_cooldown: int = 8       # steps a leader cannot re-task the same subordinate
    #                               (masked); lifted early if the leader's own mission
    #                               changed or a CONTACT hit the net since. 0 → off.
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

    def __post_init__(self) -> None:
        if self.ablation not in ("full", "nomask", "flat"):
            msg = f"Unknown ablation arm {self.ablation!r} (expected full | nomask | flat)"
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
        n_enemies=4,
        opfor_mode="assault",
        root_mission=MissionType.DEFEND,
        root_objective="ALPHA",
        max_steps=375,
        # defensive doctrine: prepared positions + early warning (see ROADMAP —
        # three trainings on the bare spec all plateaued at a ~55-60% coin-flip
        # brawl; a defense without defensible ground isn't a defense)
        objective_cover=True,
        assault_spawn_min_dist=21.0,
    ),
    "squad": ScenarioSpec(
        name="squad",
        description="A squad (SL + 2 fire teams) seizes OBJ ALPHA; OpFor garrisons ALPHA and BRAVO.",
        org="squad",
        map_size=(42, 42),
        objectives=(("ALPHA", (33, 33)), ("BRAVO", (35, 9)), ("CHARLIE", (9, 35))),
        spawn=(5, 5),
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


def get_scenario(name: str) -> ScenarioSpec:
    """Look up a scenario preset by name."""
    if name not in SCENARIOS:
        known = ", ".join(sorted(SCENARIOS))
        msg = f"Unknown scenario {name!r}. Available: {known}"
        raise KeyError(msg)
    return SCENARIOS[name]
