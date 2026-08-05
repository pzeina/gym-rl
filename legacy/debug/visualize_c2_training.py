"""Visualization script for C2 training with comprehensive debugging displays.

This script provides visual debugging capabilities for command and control training including:
- Agent positions and movements on the map
- Elementary act selections and action execution
- Communication messages and ranges
- Mission derivations and order flows
- Reward components visualization
- Real-time C2 hierarchy display

Usage:
    python debug/visualize_c2_training.py [--scenario SCENARIO] [--agents N] [--speed SPEED]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D

from mili_env.envs.classes.robot_base import Moves
from mili_env.envs.classes.types_common import AgentRole, SoldierElementaryAct
from mili_env.envs.terrain_world import TerrainWorldEnv

# Visualization configuration
AGENT_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#FFA500", "#800080", "#008080", "#80FF00", "#808080", "#FFC0CB"
]

ROLE_MARKERS = {
    AgentRole.CDU: "s",  # Square - Company Commander
    AgentRole.ADU: "s",  # Square - Deputy Company Commander
    AgentRole.CDS: "^",  # Triangle - Section Commander
    AgentRole.SOA: "^",  # Triangle - Deputy Platoon Sergeant
    AgentRole.CDG: "o",  # Circle - Squad Leader
    AgentRole.CAP: ".",  # Point - Team Leader
}

ELEMENTARY_ACT_COLORS = {
    SoldierElementaryAct.MOVE: "#00FF00",  # Green
    SoldierElementaryAct.POST: "#0000FF",  # Blue
    SoldierElementaryAct.FIRE: "#FF0000",  # Red
}


class C2Visualizer:
    """Real-time visualizer for C2 training scenarios."""

    def __init__(self, env: TerrainWorldEnv, save_frames: bool = False):
        self.env = env
        self.save_frames = save_frames
        self.frame_count = 0

        # Setup matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle("C2 Training Visualization", fontsize=16)

        # Main map plot
        self.map_ax = self.axes[0, 0]
        self.map_ax.set_title("Agent Positions and Actions")
        self.map_ax.set_xlim(0, env.width)
        self.map_ax.set_ylim(0, env.height)
        self.map_ax.set_aspect("equal")
        self.map_ax.grid(True, alpha=0.3)

        # Communications plot
        self.comm_ax = self.axes[0, 1]
        self.comm_ax.set_title("Communications Network")
        self.comm_ax.set_xlim(0, env.width)
        self.comm_ax.set_ylim(0, env.height)
        self.comm_ax.set_aspect("equal")
        self.comm_ax.grid(True, alpha=0.3)

        # Hierarchy and orders plot
        self.hierarchy_ax = self.axes[1, 0]
        self.hierarchy_ax.set_title("Command Hierarchy & Orders")
        self.hierarchy_ax.set_xlim(0, 10)
        self.hierarchy_ax.set_ylim(0, 10)

        # Metrics plot
        self.metrics_ax = self.axes[1, 1]
        self.metrics_ax.set_title("Real-time Metrics")

        # Initialize data storage for metrics
        self.metrics_history = {
            "step": [],
            "team_reward": [],
            "energy": [],
            "communications": [],
            "order_changes": []
        }

        plt.tight_layout()

    def update_visualization(self, step: int, actions: dict, rewards: float, info: dict) -> None:
        """Update all visualization components with current state."""
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()

        self._draw_map(step, actions)
        self._draw_communications()
        self._draw_hierarchy(actions)
        self._update_metrics(step, rewards, info)

        # Update titles
        self.map_ax.set_title(f"Agent Positions and Actions (Step {step})")
        self.comm_ax.set_title("Communications Network")
        self.hierarchy_ax.set_title("Command Hierarchy & Orders")
        self.metrics_ax.set_title("Real-time Metrics")

        plt.tight_layout()

        if self.save_frames:
            self.fig.savefig(f"debug/frames/frame_{self.frame_count:04d}.png", dpi=100)
            self.frame_count += 1

    def _draw_map(self, step: int, actions: dict) -> None:
        """Draw the main map with agent positions, actions, and mission zones."""
        self.map_ax.set_xlim(0, self.env.width)
        self.map_ax.set_ylim(0, self.env.height)
        self.map_ax.set_aspect("equal")
        self.map_ax.grid(True, alpha=0.3)

        # Draw mission zones
        for i in range(self.env.num_agents):
            bbox = self.env.get_mission_bbox(i)
            if bbox:
                (min_x, min_y), (max_x, max_y) = bbox
                rect = patches.Rectangle(
                    (min_x, min_y), max_x - min_x, max_y - min_y,
                    linewidth=1, edgecolor=AGENT_COLORS[i % len(AGENT_COLORS)],
                    facecolor="none", alpha=0.3, linestyle="--"
                )
                self.map_ax.add_patch(rect)

            # Draw POIs
            pois = self.env.get_mission_pois(i)
            for poi_x, poi_y in pois:
                self.map_ax.plot(poi_x, poi_y, marker="*", markersize=8,
                               color=AGENT_COLORS[i % len(AGENT_COLORS)], alpha=0.7)

        # Draw agents
        for i, agent in enumerate(self.env.agents):
            x, y = agent.get_position()
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            marker = ROLE_MARKERS.get(agent.role, "o")

            # Agent position
            self.map_ax.plot(x, y, marker=marker, markersize=12, color=color,
                           markeredgecolor="black", markeredgewidth=1)

            # Agent ID label
            self.map_ax.annotate(f"{i}", (x, y), xytext=(5, 5),
                               textcoords="offset points", fontsize=8, fontweight="bold")

            # Elementary act indicator
            current_act = getattr(agent, "current_elementary_act", None)
            if current_act:
                act_color = ELEMENTARY_ACT_COLORS.get(current_act, "#888888")
                circle = patches.Circle((x, y), 0.5, facecolor=act_color, alpha=0.6)
                self.map_ax.add_patch(circle)

            # Action arrow (movement direction)
            agent_key = f"agent_{i}"
            if agent_key in actions:
                action = actions[agent_key]
                move_action = action.get("combat", {}).get("move", 0)
                if move_action in [Moves.FORWARD.value, Moves.BACKWARD.value]:
                    direction = agent.get_direction()
                    if move_action == Moves.BACKWARD.value:
                        direction += np.pi

                    dx = 1.5 * np.cos(direction)
                    dy = 1.5 * np.sin(direction)
                    self.map_ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.2,
                                    fc=color, ec=color, alpha=0.8)

        # Add legend
        legend_elements = []
        for act, color in ELEMENTARY_ACT_COLORS.items():
            legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                            markerfacecolor=color, markersize=8, label=act.name))
        self.map_ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    def _draw_communications(self) -> None:
        """Draw communication network and message flows."""
        self.comm_ax.set_xlim(0, self.env.width)
        self.comm_ax.set_ylim(0, self.env.height)
        self.comm_ax.set_aspect("equal")
        self.comm_ax.grid(True, alpha=0.3)

        # Draw communication ranges
        for i, agent in enumerate(self.env.agents):
            x, y = agent.get_position()
            color = AGENT_COLORS[i % len(AGENT_COLORS)]

            # Agent position
            self.comm_ax.plot(x, y, marker="o", markersize=8, color=color)
            self.comm_ax.annotate(f"{i}", (x, y), xytext=(5, 5),
                                textcoords="offset points", fontsize=8)

            # Communication range circle
            comm_circle = patches.Circle((x, y), agent.communication_range,
                                       facecolor="none", edgecolor=color, alpha=0.3)
            self.comm_ax.add_patch(comm_circle)

            # Draw outbox messages
            outbox = getattr(agent, "message_outbox", [])
            if outbox:
                for j, msg in enumerate(outbox):
                    # Draw message icon
                    msg_x = x + 0.5 + j * 0.3
                    msg_y = y + 0.5
                    self.comm_ax.plot(msg_x, msg_y, marker="s", markersize=4,
                                    color="red", alpha=0.8)

        # Draw communication links between agents in range
        for i, agent_i in enumerate(self.env.agents):
            x_i, y_i = agent_i.get_position()
            for j, agent_j in enumerate(self.env.agents):
                if i >= j:
                    continue
                x_j, y_j = agent_j.get_position()
                distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)

                if distance <= agent_i.communication_range:
                    self.comm_ax.plot([x_i, x_j], [y_i, y_j], "g--", alpha=0.5, linewidth=1)

    def _draw_hierarchy(self, actions: dict) -> None:
        """Draw command hierarchy and order flows."""
        self.hierarchy_ax.set_xlim(0, 10)
        self.hierarchy_ax.set_ylim(0, 10)
        # Don't remove the axes, just hide the ticks and labels for cleaner display
        self.hierarchy_ax.set_xticks([])
        self.hierarchy_ax.set_yticks([])
        self.hierarchy_ax.spines["top"].set_visible(False)
        self.hierarchy_ax.spines["right"].set_visible(False)
        self.hierarchy_ax.spines["bottom"].set_visible(False)
        self.hierarchy_ax.spines["left"].set_visible(False)

        # Create hierarchy tree
        hierarchy_info = self._get_hierarchy_info()

        # Position agents in hierarchy
        y_positions = {}
        role_order = [AgentRole.CDU, AgentRole.ADU, AgentRole.CDS, AgentRole.SOA,
                     AgentRole.CDG, AgentRole.CAP]

        for i, role in enumerate(role_order):
            agents_with_role = [aid for aid, info in hierarchy_info.items()
                              if info["role"] == role]
            if agents_with_role:
                y_pos = 8 - i * 1.3
                for j, agent_id in enumerate(agents_with_role):
                    x_pos = 1 + j * 2
                    y_positions[agent_id] = (x_pos, y_pos)

                    # Draw agent node
                    color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
                    self.hierarchy_ax.scatter(x_pos, y_pos, s=200, c=color,
                                            marker=ROLE_MARKERS.get(role, "o"))
                    self.hierarchy_ax.annotate(f"{agent_id}\n{role.value.upper()}",
                                             (x_pos, y_pos), ha="center", va="center",
                                             fontsize=8, fontweight="bold")

                    # Show orders issued
                    agent_key = f"agent_{agent_id}"
                    if agent_key in actions:
                        action = actions[agent_key]
                        if action.get("c2", {}).get("give_order", 0):
                            # Highlight order-giving agents
                            circle = patches.Circle((x_pos, y_pos), 0.3,
                                                  facecolor="red", alpha=0.3)
                            self.hierarchy_ax.add_patch(circle)

        # Draw command lines
        for agent_id, pos in y_positions.items():
            info = hierarchy_info[agent_id]
            for other_id, other_pos in y_positions.items():
                other_info = hierarchy_info[other_id]
                if info["rank"] > other_info["rank"]:  # Can command
                    self.hierarchy_ax.plot([pos[0], other_pos[0]], [pos[1], other_pos[1]],
                                         "k-", alpha=0.2, linewidth=1)

    def _update_metrics(self, step: int, rewards: float, info: dict) -> None:
        """Update real-time metrics plot."""
        # Update metrics history
        self.metrics_history["step"].append(step)
        self.metrics_history["team_reward"].append(rewards)

        total_energy = sum(agent.get_energy() for agent in self.env.agents)
        self.metrics_history["energy"].append(total_energy)

        total_comms = sum(len(getattr(agent, "message_outbox", [])) for agent in self.env.agents)
        self.metrics_history["communications"].append(total_comms)

        order_changes = getattr(self.env, "_order_change_count", 0)
        self.metrics_history["order_changes"].append(order_changes)

        # Plot metrics
        if len(self.metrics_history["step"]) > 1:
            steps = self.metrics_history["step"][-50:]  # Last 50 steps

            # Team reward
            self.metrics_ax.plot(steps, self.metrics_history["team_reward"][-len(steps):],
                               "b-", label="Team Reward", linewidth=2)

            # Energy (normalized)
            energy_norm = [e / (self.env.num_agents * 100) for e in self.metrics_history["energy"][-len(steps):]]
            self.metrics_ax.plot(steps, energy_norm, "g-", label="Energy (norm)", linewidth=2)

            # Communications
            self.metrics_ax.plot(steps, self.metrics_history["communications"][-len(steps):],
                               "r-", label="Messages", linewidth=2)

            self.metrics_ax.legend(fontsize=8)
            self.metrics_ax.grid(True, alpha=0.3)

    def _get_hierarchy_info(self) -> dict:
        """Get hierarchy information for all agents."""
        hierarchy = {}
        role_ranks = {
            AgentRole.CDU: 6, AgentRole.ADU: 6, AgentRole.CDS: 5,
            AgentRole.SOA: 5, AgentRole.CDG: 4, AgentRole.CAP: 3
        }

        for i, agent in enumerate(self.env.agents):
            hierarchy[i] = {
                "role": agent.role,
                "rank": role_ranks.get(agent.role, 0),
                "team_id": agent.team_id,
            }
        return hierarchy


def generate_c2_action(env: TerrainWorldEnv, scenario: str, step: int) -> dict[str, dict[str, Any]]:
    """Generate C2 actions for visualization (same as training script)."""
    actions: dict[str, dict[str, Any]] = {}

    # Get hierarchy info
    hierarchy = {}
    role_ranks = {
        AgentRole.CDU: 6, AgentRole.ADU: 6, AgentRole.CDS: 5,
        AgentRole.SOA: 5, AgentRole.CDG: 4, AgentRole.CAP: 3
    }

    for i, agent in enumerate(env.agents):
        hierarchy[i] = {
            "role": agent.role.value,
            "rank": role_ranks.get(agent.role, 0),
            "can_issue_orders": role_ranks.get(agent.role, 0) > 3
        }

    for i in range(env.num_agents):
        agent_key = f"agent_{i}"
        agent_info = hierarchy[i]

        # Base action structure
        base_action = {
            "elementary_act": random.randrange(len(SoldierElementaryAct)),
            "combat": {
                "move": random.randrange(len(Moves)),
                "fire_enemy": env.num_agents
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
            if random.random() < 0.3:
                base_action["c2"]["give_order"] = 1
                for j in range(env.num_agents):
                    if i != j and hierarchy[j]["rank"] < agent_info["rank"]:
                        base_action["c2"]["orders"][j] = random.randrange(len(SoldierElementaryAct))

        elif scenario == "elementary_acts":
            base_action["elementary_act"] = (step + i) % len(SoldierElementaryAct)

        elif scenario == "communications":
            if random.random() < 0.5:
                comm_types = [random.randint(0, 2) for _ in range(env.MAX_COMMS_PER_STEP)]
                base_action["comm"]["types"] = comm_types

        actions[agent_key] = base_action

    return actions


def main() -> None:
    """Run C2 visualization with real-time display."""
    parser = argparse.ArgumentParser(description="Visualize C2 training scenarios")
    parser.add_argument("--scenario", choices=["random_orders", "elementary_acts", "communications"],
                       default="random_orders", help="C2 scenario to visualize")
    parser.add_argument("--agents", type=int, default=6, help="Number of agents")
    parser.add_argument("--speed", type=float, default=1.0, help="Animation speed multiplier")
    parser.add_argument("--save-frames", action="store_true", help="Save frames as images")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to visualize")

    args = parser.parse_args()

    # Create frames directory if saving
    if args.save_frames:
        Path("debug/frames").mkdir(parents=True, exist_ok=True)

    # Setup environment
    env = TerrainWorldEnv(num_agents=args.agents, target_zone_size=15, num_enemies=2)
    env.reset(seed=42)

    # Create visualizer
    visualizer = C2Visualizer(env, save_frames=args.save_frames)

    print(f"Starting C2 visualization: {args.scenario}")
    print(f"Agents: {args.agents}, Steps: {args.steps}")
    print("Close the plot window to stop visualization")

    # Interactive mode
    plt.ion()
    plt.show()

    try:
        for step in range(args.steps):
            # Generate actions
            actions = generate_c2_action(env, args.scenario, step)

            # Execute step
            _obs, rewards, terminated, truncated, info = env.step(actions)

            # Update visualization
            visualizer.update_visualization(step, actions, rewards, info)

            # Refresh display
            plt.pause(0.1 / args.speed)

            if terminated or truncated:
                print(f"Episode terminated at step {step}")
                break

            # Handle window events
            if not plt.get_fignums():  # Window was closed
                break

    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")

    finally:
        plt.ioff()
        if args.save_frames:
            print(f"Frames saved to debug/frames/ (total: {visualizer.frame_count})")
        print("Visualization complete")


if __name__ == "__main__":
    main()
