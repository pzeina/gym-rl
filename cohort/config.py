"""Scenario specifications: org charts, maps, OpFor, and the initial OPORD."""

from __future__ import annotations

from dataclasses import dataclass, field

from cohort.core.missions import MissionType
from cohort.core.ranks import Rank
from cohort.core.units import CombatParams


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
    opfor_mode: str                            # garrison (hold objectives) | assault (advance on spawn objective)
    root_mission: MissionType
    root_objective: str | None                 # objective name for the OPORD
    max_steps: int
    forest_density: float = 1.0
    wall_density: float = 1.0
    combat: CombatParams = field(default_factory=CombatParams)
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


SCENARIOS: dict[str, ScenarioSpec] = {
    "fireteam": ScenarioSpec(
        name="fireteam",
        description="A fire team (TL + 3 RFN) seizes OBJ ALPHA held by a small OpFor garrison.",
        org="fireteam",
        map_size=(24, 24),
        objectives=(("ALPHA", (18, 18)), ("BRAVO", (19, 4))),
        spawn=(3, 3),
        n_enemies=3,
        opfor_mode="garrison",
        root_mission=MissionType.SEIZE,
        root_objective="ALPHA",
        max_steps=200,
    ),
    "fireteam_defend": ScenarioSpec(
        name="fireteam_defend",
        description="A fire team defends OBJ ALPHA against an OpFor assault.",
        org="fireteam",
        map_size=(24, 24),
        objectives=(("ALPHA", (12, 12)),),
        spawn=(11, 11),
        n_enemies=4,
        opfor_mode="assault",
        root_mission=MissionType.DEFEND,
        root_objective="ALPHA",
        max_steps=250,
    ),
    "squad": ScenarioSpec(
        name="squad",
        description="A squad (SL + 2 fire teams) seizes OBJ ALPHA; OpFor garrisons ALPHA and BRAVO.",
        org="squad",
        map_size=(28, 28),
        objectives=(("ALPHA", (22, 22)), ("BRAVO", (23, 6)), ("CHARLIE", (6, 23))),
        spawn=(3, 3),
        n_enemies=5,
        opfor_mode="garrison",
        root_mission=MissionType.SEIZE,
        root_objective="ALPHA",
        max_steps=300,
    ),
    "squad_recon": ScenarioSpec(
        name="squad_recon",
        description="A squad reconnoiters OBJ BRAVO without being drawn into a fight.",
        org="squad",
        map_size=(28, 28),
        objectives=(("ALPHA", (22, 22)), ("BRAVO", (23, 6))),
        spawn=(3, 14),
        n_enemies=4,
        opfor_mode="garrison",
        root_mission=MissionType.RECON,
        root_objective="BRAVO",
        max_steps=250,
    ),
    "platoon": ScenarioSpec(
        name="platoon",
        description="A platoon (PL, PSG, 2 squads — 16 agents) seizes OBJ ALPHA.",
        org="platoon",
        map_size=(36, 36),
        objectives=(("ALPHA", (29, 29)), ("BRAVO", (30, 7)), ("CHARLIE", (7, 30)), ("DELTA", (18, 18))),
        spawn=(4, 4),
        n_enemies=8,
        opfor_mode="garrison",
        root_mission=MissionType.SEIZE,
        root_objective="ALPHA",
        max_steps=400,
    ),
}


def get_scenario(name: str) -> ScenarioSpec:
    """Look up a scenario preset by name."""
    if name not in SCENARIOS:
        known = ", ".join(sorted(SCENARIOS))
        msg = f"Unknown scenario {name!r}. Available: {known}"
        raise KeyError(msg)
    return SCENARIOS[name]
