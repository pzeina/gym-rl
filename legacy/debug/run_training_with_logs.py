"""Run a short training loop while logging comprehensive C2 activities.

This script performs structured testing of command and control features including:
- Elementary act selection and execution
- Mission derivation by chief agents
- Order issuance and compliance
- Communication patterns
- Reward shaping components

Outputs detailed JSON logs for analysis of C2 behavior and learning.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from mili_env.envs.classes.c2_doctrine import MISSION_DERIVATION_DOCTRINE, derivation_quality
from mili_env.envs.classes.robot_base import Moves
from mili_env.envs.classes.types_common import AgentRole, SoldierElementaryAct
from mili_env.envs.terrain_world import TerrainWorldEnv

LOG_DIR = Path("debug/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Separate log files for different aspects
C2_LOG_FILE = LOG_DIR / "c2_activities.jsonl"
ELEMENTARY_ACTS_LOG = LOG_DIR / "elementary_acts.jsonl"
MISSION_DERIVATION_LOG = LOG_DIR / "mission_derivations.jsonl"
REWARD_BREAKDOWN_LOG = LOG_DIR / "reward_breakdown.jsonl"
COMMUNICATION_LOG = LOG_DIR / "communications.jsonl"

NUM_EPISODES = 3
STEPS_PER_EPISODE = 50

# Test scenarios for C2 behavior
C2_TEST_SCENARIOS = [
    "random_orders",      # Random order issuance by chiefs
    "elementary_acts",    # Focus on elementary act selection
    "communications",     # Heavy communication testing
]


def get_agent_hierarchy_info(env: TerrainWorldEnv) -> dict[int, dict[str, Any]]:
    """Get hierarchy information for all agents."""
    hierarchy = {}
    role_ranks = {
        AgentRole.CDU: 6, AgentRole.ADU: 6, AgentRole.CDS: 5,
        AgentRole.SOA: 5, AgentRole.CDG: 4, AgentRole.CAP: 3
    }

    for i, agent in enumerate(env.agents):
        hierarchy[i] = {
            "role": agent.role.value,
            "rank": role_ranks.get(agent.role, 0),
            "team_id": agent.team_id,
            "can_issue_orders": role_ranks.get(agent.role, 0) > 3
        }
    return hierarchy


def generate_c2_action(env: TerrainWorldEnv, scenario: str, step: int) -> dict[str, dict[str, Any]]:
    """Generate actions based on specific C2 test scenarios."""
    actions: dict[str, dict[str, Any]] = {}
    hierarchy = get_agent_hierarchy_info(env)

    for i in range(env.num_agents):
        agent_key = f"agent_{i}"
        agent_info = hierarchy[i]

        # Base action structure
        base_action = {
            "elementary_act": random.randrange(len(SoldierElementaryAct)),
            "combat": {
                "move": random.randrange(len(Moves)),
                "fire_enemy": env.num_agents  # No firing by default
            },
            "comm": {
                "types": [0] * env.MAX_COMMS_PER_STEP,
                "targets": [env.num_agents] * env.MAX_COMMS_PER_STEP,
            },
            "c2": {
                "give_order": 0,
                "orders": [env.MISSION_NO_CHANGE_SENTINEL] * env.num_agents
            },
        }

        # Scenario-specific modifications
        if scenario == "random_orders" and agent_info["can_issue_orders"]:
            if random.random() < 0.3:  # 30% chance to issue orders
                base_action["c2"]["give_order"] = 1
                # Issue random elementary acts to subordinates
                for j in range(env.num_agents):
                    if i != j and hierarchy[j]["rank"] < agent_info["rank"]:
                        base_action["c2"]["orders"][j] = random.randrange(len(SoldierElementaryAct))

        elif scenario == "elementary_acts":
            # Cycle through elementary acts systematically
            base_action["elementary_act"] = (step + i) % len(SoldierElementaryAct)

        elif scenario == "communications":
            # Increase communication frequency
            if random.random() < 0.5:  # 50% chance to communicate
                comm_types = [random.randint(0, 2) for _ in range(env.MAX_COMMS_PER_STEP)]
                base_action["comm"]["types"] = comm_types

        actions[agent_key] = base_action

    return actions


def log_c2_activities(env: TerrainWorldEnv, step: int, episode: int, actions: dict) -> None:
    """Log detailed C2 activities for analysis."""
    hierarchy = get_agent_hierarchy_info(env)

    with C2_LOG_FILE.open("a") as f:
        for i, agent in enumerate(env.agents):
            agent_key = f"agent_{i}"
            action = actions.get(agent_key, {})

            c2_data = {
                "episode": episode,
                "step": step,
                "agent_id": i,
                "role": agent.role.value,
                "rank": hierarchy[i]["rank"],
                "give_order_flag": action.get("c2", {}).get("give_order", 0),
                "orders_issued": [],
                "current_elementary_act": getattr(agent, "current_elementary_act", None),
                "last_derived_missions": getattr(agent, "last_derived_missions", {}),
                "free_order_credit": getattr(agent, "free_order_credit", False),
                "last_mission_change_step": getattr(agent, "last_mission_change_step", -1),
            }

            # Parse orders issued
            if action.get("c2", {}).get("give_order", 0):
                orders_vector = action.get("c2", {}).get("orders", [])
                for j, order_val in enumerate(orders_vector):
                    if order_val != env.MISSION_NO_CHANGE_SENTINEL and 0 <= order_val < len(SoldierElementaryAct):
                        c2_data["orders_issued"].append({
                            "target_agent": j,
                            "mission": SoldierElementaryAct(order_val).name,
                            "is_subordinate": hierarchy[j]["rank"] < hierarchy[i]["rank"]
                        })

            # Convert elementary act to string for JSON serialization
            if c2_data["current_elementary_act"] is not None:
                c2_data["current_elementary_act"] = c2_data["current_elementary_act"].name

            # Convert last derived missions dict
            derived_missions_str = {}
            for tgt_id, mission in c2_data["last_derived_missions"].items():
                if hasattr(mission, "name"):
                    derived_missions_str[str(tgt_id)] = mission.name
            c2_data["last_derived_missions"] = derived_missions_str

            f.write(json.dumps(c2_data) + "\n")


def log_elementary_acts(env: TerrainWorldEnv, step: int, episode: int, actions: dict) -> None:
    """Log elementary act selections and their alignment with actual actions."""
    with ELEMENTARY_ACTS_LOG.open("a") as f:
        for i, agent in enumerate(env.agents):
            agent_key = f"agent_{i}"
            action = actions.get(agent_key, {})

            elementary_act_data = {
                "episode": episode,
                "step": step,
                "agent_id": i,
                "selected_elementary_act": None,
                "actual_move": action.get("combat", {}).get("move", 0),
                "fire_action": action.get("combat", {}).get("fire_enemy", env.num_agents),
                "moved": action.get("combat", {}).get("move", 0) in [Moves.FORWARD.value, Moves.BACKWARD.value],
                "rotated": action.get("combat", {}).get("move", 0) in [Moves.ROTATE_LEFT.value, Moves.ROTATE_RIGHT.value],
                "stationary": action.get("combat", {}).get("move", 0) == Moves.IDLE.value,
                "fired": action.get("combat", {}).get("fire_enemy", env.num_agents) < env.num_agents,
                "energy": agent.get_energy(),
                "health": agent.get_health(),
            }

            # Get selected elementary act
            selected_act = action.get("elementary_act", 0)
            if 0 <= selected_act < len(SoldierElementaryAct):
                elementary_act_data["selected_elementary_act"] = SoldierElementaryAct(selected_act).name

            # Calculate alignment score
            alignment_score = 0.0
            if elementary_act_data["selected_elementary_act"] == "MOVE":
                alignment_score = 0.1 if elementary_act_data["moved"] else -0.05
            elif elementary_act_data["selected_elementary_act"] == "POST":
                alignment_score = 0.1 if elementary_act_data["stationary"] or elementary_act_data["rotated"] else -0.05
            elif elementary_act_data["selected_elementary_act"] == "FIRE":
                alignment_score = 0.15 if elementary_act_data["fired"] else -0.1

            elementary_act_data["alignment_score"] = alignment_score

            f.write(json.dumps(elementary_act_data) + "\n")


def log_mission_derivations(env: TerrainWorldEnv, step: int, episode: int) -> None:
    """Log mission derivation quality by chief agents."""
    hierarchy = get_agent_hierarchy_info(env)

    with MISSION_DERIVATION_LOG.open("a") as f:
        for i, agent in enumerate(env.agents):
            if not hierarchy[i]["can_issue_orders"]:
                continue

            derivation_data = {
                "episode": episode,
                "step": step,
                "chief_agent_id": i,
                "chief_role": agent.role.value,
                "chief_mission": None,
                "subordinate_assignments": [],
                "derivation_quality_scores": [],
            }

            # Get chief's current mission
            chief_mission = getattr(agent, "current_elementary_act", None)
            if chief_mission:
                derivation_data["chief_mission"] = chief_mission.name

            # Analyze subordinate assignments
            last_derived = getattr(agent, "last_derived_missions", {})
            for tgt_id, assigned_mission in last_derived.items():
                if isinstance(tgt_id, (int, str)) and hasattr(assigned_mission, "name"):
                    tgt_idx = int(tgt_id)
                    if 0 <= tgt_idx < len(env.agents):
                        subordinate_data = {
                            "subordinate_id": tgt_idx,
                            "subordinate_role": env.agents[tgt_idx].role.value,
                            "assigned_mission": assigned_mission.name,
                            "is_valid_subordinate": hierarchy[tgt_idx]["rank"] < hierarchy[i]["rank"]
                        }

                        quality = derivation_quality(chief_mission, assigned_mission, MISSION_DERIVATION_DOCTRINE)
                        subordinate_data["quality_score"] = quality
                        subordinate_data["quality_score"] = quality

                        derivation_data["subordinate_assignments"].append(subordinate_data)
                        derivation_data["derivation_quality_scores"].append(quality)

            # Calculate average quality
            if derivation_data["derivation_quality_scores"]:
                derivation_data["avg_quality"] = np.mean(derivation_data["derivation_quality_scores"])
            else:
                derivation_data["avg_quality"] = 0.0

            f.write(json.dumps(derivation_data) + "\n")


def log_reward_breakdown(env: TerrainWorldEnv, step: int, episode: int,
                        team_reward: float, per_agent_rewards: dict) -> None:
    """Log detailed reward breakdown components."""
    with REWARD_BREAKDOWN_LOG.open("a") as f:
        # Get reward components
        order_change_count = getattr(env, "_order_change_count", 0)

        reward_data = {
            "episode": episode,
            "step": step,
            "team_reward": team_reward,
            "per_agent_rewards": {k: float(v) for k, v in per_agent_rewards.items()},
            "order_changes_this_step": order_change_count,
            "total_agents": env.num_agents,
            "alive_agents": sum(1 for a in env.agents if a.state.is_alive()),
            "agents_at_target": sum(1 for a in env.agents if a.state.is_at_target()),
            "total_energy": sum(a.get_energy() for a in env.agents),
            "total_health": sum(a.get_health() for a in env.agents),
            "help_requests": getattr(env, "_help_requests", 0),
        }

        # Calculate cooperation metrics
        cooperation_pairs = 0
        total_pairs = 0
        for i in range(env.num_agents):
            for j in range(i + 1, env.num_agents):
                total_pairs += 1
                agent_i_pos = np.array(env.agents[i].get_position())
                agent_j_pos = np.array(env.agents[j].get_position())
                distance = np.linalg.norm(agent_i_pos - agent_j_pos)
                if distance <= env.agents[i].communication_range:
                    cooperation_pairs += 1

        reward_data["cooperation_ratio"] = cooperation_pairs / max(1, total_pairs)

        f.write(json.dumps(reward_data) + "\n")


def log_communications(env: TerrainWorldEnv, step: int, episode: int) -> None:
    """Log communication activities and message patterns."""
    with COMMUNICATION_LOG.open("a") as f:
        total_messages = 0
        message_types = {}

        for i, agent in enumerate(env.agents):
            outbox = getattr(agent, "message_outbox", [])
            history = getattr(agent, "communication_history", [])

            comm_data = {
                "episode": episode,
                "step": step,
                "agent_id": i,
                "role": agent.role.value,
                "outbox_size": len(outbox),
                "total_messages_sent": len(history),
                "last_communication_time": getattr(agent, "last_communication_time", 0.0),
                "known_agents_count": len(getattr(agent, "known_agents", {})),
                "message_types_sent": {},
            }

            # Count message types in history
            for msg in history:
                msg_type = getattr(msg, "message_type", None)
                if msg_type and hasattr(msg_type, "value"):
                    type_name = msg_type.value
                    comm_data["message_types_sent"][type_name] = comm_data["message_types_sent"].get(type_name, 0) + 1
                    message_types[type_name] = message_types.get(type_name, 0) + 1

            total_messages += len(history)
            f.write(json.dumps(comm_data) + "\n")

        # Log summary
        summary_data = {
            "episode": episode,
            "step": step,
            "summary": True,
            "total_messages": total_messages,
            "message_types_distribution": message_types,
            "avg_messages_per_agent": total_messages / max(1, env.num_agents),
        }
        f.write(json.dumps(summary_data) + "\n")


def clear_log_files() -> None:
    """Clear all log files at start of run."""
    for log_file in [C2_LOG_FILE, ELEMENTARY_ACTS_LOG, MISSION_DERIVATION_LOG,
                     REWARD_BREAKDOWN_LOG, COMMUNICATION_LOG]:
        if log_file.exists():
            log_file.unlink()


def main() -> None:
    """Run comprehensive C2 testing with detailed logging."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    # Clear previous logs
    clear_log_files()

    # Test with multiple agents to enable hierarchical C2
    env = TerrainWorldEnv(num_agents=6, target_zone_size=15, num_enemies=2)

    logger.info("Starting comprehensive C2 training with logging...")
    logger.info("Agents: %d, Episodes: %d, Steps per episode: %d",
                env.num_agents, NUM_EPISODES, STEPS_PER_EPISODE)

    for ep in range(NUM_EPISODES):
        scenario = C2_TEST_SCENARIOS[ep % len(C2_TEST_SCENARIOS)]
        logger.info("Episode %d: Testing scenario '%s'", ep, scenario)

        env.reset(seed=42 + ep)

        for step in range(STEPS_PER_EPISODE):
            # Generate scenario-specific actions
            actions = generate_c2_action(env, scenario, step)

            # Execute step
            _obs, rewards, terminated, truncated, info = env.step(actions)

            # Extract reward information
            team_reward = float(rewards) if isinstance(rewards, (int, float)) else 0.0
            per_agent_rewards = {}
            for agent_key, agent_info in info.items():
                if isinstance(agent_info, dict) and "per_agent_reward" in agent_info:
                    per_agent_rewards[agent_key] = agent_info["per_agent_reward"]

            # Log all aspects
            log_c2_activities(env, step, ep, actions)
            log_elementary_acts(env, step, ep, actions)
            log_mission_derivations(env, step, ep)
            log_reward_breakdown(env, step, ep, team_reward, per_agent_rewards)
            log_communications(env, step, ep)

            if terminated or truncated:
                logger.info("Episode %d terminated at step %d", ep, step)
                break

        logger.info("Episode %d completed", ep)

    # Log summary
    logger.info("C2 training complete. Logs written to:")
    logger.info("  C2 Activities: %s", C2_LOG_FILE)
    logger.info("  Elementary Acts: %s", ELEMENTARY_ACTS_LOG)
    logger.info("  Mission Derivations: %s", MISSION_DERIVATION_LOG)
    logger.info("  Reward Breakdown: %s", REWARD_BREAKDOWN_LOG)
    logger.info("  Communications: %s", COMMUNICATION_LOG)


if __name__ == "__main__":  # pragma: no cover
    main()
