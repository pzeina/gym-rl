"""Doctrine mapping for subordinate mission derivation.

This module centralizes the allowed subordinate missions given a superior's
current mission. The environment previously embedded this mapping directly
in `terrain_world.TerrainWorld` as `_MISSION_DERIVATION_DOCTRINE`. We keep the
mapping small and focused on the atomic `SoldierMission` enum actually used
by the simulation logic. If future, higher-echelon doctrinal constructs are
needed they can be layered here without touching environment or agent code.
"""
from __future__ import annotations

from mili_env.envs.classes.types_common import SoldierElementaryAct

# Public doctrine mapping (ordered tuples express preference order):
MISSION_DERIVATION_DOCTRINE: dict[SoldierElementaryAct, tuple[SoldierElementaryAct, ...]] = {
    SoldierElementaryAct.MOVE: (SoldierElementaryAct.MOVE, SoldierElementaryAct.POST),
    SoldierElementaryAct.POST: (SoldierElementaryAct.POST, SoldierElementaryAct.FIRE),
    SoldierElementaryAct.FIRE: (SoldierElementaryAct.FIRE, SoldierElementaryAct.POST),
}

# Central reward / shaping constant table (referenced by environment reward helpers).
REWARD_CONSTANTS: dict[str, float] = {
    # Terminal rewards
    "SUCCESS_BASE": 1000.0,
    "SUCCESS_ENERGY_FACTOR": 0.5,      # multiplied by total team energy
    "SUCCESS_HEALTH_FACTOR": 0.25,     # multiplied by total team health
    "FAILURE_BASE": -500.0,
    # Ongoing shaping
    "PROGRESS_SCALE": 10.0,            # distance improvement multiplier
    "ENERGY_SCALE": 5.0,               # normalized mean energy multiplier
    # C2 / Mission shaping (no explicit penalties; RL-based shaping instead)
    "ORDER_CHANGE_NEG_PENALTY": 0.5,   # negative per agent-normalized change ratio
    "FOLLOW_COMPLIANCE_SCALE": 0.5,    # normalized formation compliance scale
    "DERIVATION_DECENTRALIZED_SCALE": 0.5,  # per-good subordinate assignment
    "EVENT_REWARD_SCALE": 1.0,         # scale factor on mission-event rewards
    # Cooperation
    "COOP_PAIR_BONUS": 5.0,            # per close pair
    "COOP_DISTANCE_PENALTY": -1.0,     # per far pair beyond 2x comm range
}

# Mission-action intrinsic reward scores (small shaping values, per-agent, decentralized).
# Keys inside inner dict refer to abstracted action outcomes derived each step.
#   moved: agent displaced via FORWARD/BACKWARD
#   stationary: agent did not displace (IDLE or rotation only)
#   fired / not_fired: whether a fire action was executed
ACT_ACTION_SCORES: dict[SoldierElementaryAct, dict[str, float]] = {
    SoldierElementaryAct.MOVE: {"moved": 0.5, "stationary": -0.2},
    SoldierElementaryAct.POST: {"stationary": 0.4, "moved": -0.3},
    SoldierElementaryAct.FIRE: {"fired": 0.6, "not_fired": -0.2},
}

__all__ = [
    "ACT_ACTION_SCORES",
    "MISSION_DERIVATION_DOCTRINE",
    "REWARD_CONSTANTS",
    "derivation_quality",
    "mission_event_reward",
]

# ---------------- Mission derivation quality & event reward helpers ---------------- #

def derivation_quality(
    superior_mission: SoldierElementaryAct | None,
    proposed: SoldierElementaryAct,
    doctrine: dict[SoldierElementaryAct, tuple[SoldierElementaryAct, ...]] | None = None,
) -> float:
    """Return a small score for how well a proposed subordinate mission fits doctrine.

    Scoring:
    - If no superior mission is set, return 0.0 (no preference known).
    - If proposed not in allowed set -> -0.5
    - If proposed equals first allowed (preferred) -> +1.0
    - Else if allowed but not preferred -> +0.5
    """
    if superior_mission is None:
        return 0.0
    doc = doctrine or MISSION_DERIVATION_DOCTRINE
    allowed = doc.get(superior_mission, (superior_mission,))
    if proposed not in allowed:
        return -0.5
    if proposed == allowed[0]:
        return 1.0
    return 0.5


def mission_event_reward(  # noqa: PLR0911
    mission_order: str | None,
    events: dict[str, float] | dict[str, int],
) -> float:
    """Map high-level events to a mission-relevant reward contribution.

    Args:
        mission_order: The actual mission/order string (e.g., "PATROL_ZONE_A", "SECURE_BUILDING",
                      "ELIMINATE_TARGETS", "PROVIDE_OVERWATCH") given by superior or self-assigned
        events: Event counts per step with keys like enemy_killed, ally_killed, enemy_detected,
               ally_assist, zone_explored, zone_cleared, poi_reached

    Returns:
        Float reward contribution based on how well the events align with the mission objectives
    """
    if mission_order is None:
        # Neutral baseline: small weight on exploration, POIs, and detection
        return (
            0.1 * float(events.get("zone_explored", 0))
            + 0.2 * float(events.get("poi_reached", 0))
            + 0.1 * float(events.get("enemy_detected", 0))
        )

    e = {k: float(events.get(k, 0.0)) for k in (
        "enemy_killed", "ally_killed", "enemy_detected", "ally_assist", "zone_explored", "zone_cleared", "poi_reached"
    )}

    mission_lower = mission_order.lower()

    # Combat/Elimination missions
    if any(keyword in mission_lower for keyword in ["eliminate", "attack", "destroy", "engage", "fire"]):
        return (
            10.0 * e["enemy_killed"]        # Primary objective
            - 3.0 * e["ally_killed"]        # Heavy penalty for friendly fire
            + 2.0 * e["enemy_detected"]     # Good for target acquisition
            + 0.5 * e["ally_assist"]        # Coordination bonus
            + 0.1 * e["poi_reached"]        # Minor tactical position bonus
        )

    # Patrol/Reconnaissance missions
    if any(keyword in mission_lower for keyword in ["patrol", "recon", "scout", "search", "explore"]):
        return (
            3.0 * e["zone_explored"]        # Primary objective
            + 5.0 * e["enemy_detected"]     # Key intel gathering
            + 2.0 * e["zone_cleared"]       # Area security
            + 1.0 * e["poi_reached"]        # Checkpoint coverage
            + 0.5 * e["ally_assist"]        # Team coordination
            - 1.0 * e["ally_killed"]        # Avoid casualties while scouting
        )

    # Security/Defensive missions
    if any(keyword in mission_lower for keyword in ["secure", "defend", "hold", "guard", "overwatch", "post"]):
        return (
            2.0 * e["enemy_detected"]       # Early warning system
            + 3.0 * e["ally_assist"]        # Supporting teammates
            + 4.0 * e["zone_cleared"]       # Maintaining security
            + 2.0 * e["poi_reached"]        # Securing key positions
            - 2.0 * e["ally_killed"]        # Protecting allies is priority
            - 0.2 * e["zone_explored"]      # Discourage leaving post
        )

    # Support/Logistics missions
    if any(keyword in mission_lower for keyword in ["support", "assist", "resupply", "evacuate", "rescue"]):
        return (
            5.0 * e["ally_assist"]          # Primary objective
            + 1.0 * e["zone_explored"]      # Finding allies to help
            + 1.0 * e["poi_reached"]        # Reaching support positions
            - 3.0 * e["ally_killed"]        # Protecting allies critical
            + 0.5 * e["enemy_detected"]     # Situational awareness
        )

    # Movement/Maneuver missions
    if any(keyword in mission_lower for keyword in ["move", "advance", "retreat", "reposition", "maneuver"]):
        return (
            2.0 * e["zone_explored"]        # Covering ground
            + 8.0 * e["poi_reached"]        # Reaching objectives
            + 1.0 * e["enemy_detected"]     # Situational awareness
            + 0.5 * e["ally_assist"]        # Team coordination
            - 1.0 * e["ally_killed"]        # Safe movement
        )

    # Default for unrecognized missions - balanced approach
    return (
        1.0 * e["enemy_killed"]
        + 1.0 * e["zone_explored"]
        + 1.0 * e["enemy_detected"]
        + 1.0 * e["ally_assist"]
        + 1.0 * e["poi_reached"]
        - 1.0 * e["ally_killed"]
    )
