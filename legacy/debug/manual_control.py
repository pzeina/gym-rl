"""Manual control classes for interactive agent parameter manipulation.

This module provides enhanced versions of the base agent and environment classes
with additional methods for manually controlling agent parameters during visualization
or debugging sessions. These classes inherit from the base classes without modifying
the original environment code.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from mili_env.envs.classes.robot_base import RobotBase
from mili_env.envs.classes.types_common import AgentRole, MessageType, SoldierElementaryAct
from mili_env.envs.terrain_world import TerrainWorldEnv


class ManualControlAgent(RobotBase):
    """Enhanced agent with manual control capabilities."""
    
    def __init__(self, position, attributes, game_map, constraints, agent_id, role, team_id):
        super().__init__(
            position=position,
            attributes=attributes, 
            game_map=game_map,
            constraints=constraints,
            agent_id=agent_id,
            role=role,
            team_id=team_id
        )
        self._manual_mode = False
        self._manual_actions_queue = []
        
    def enable_manual_mode(self) -> None:
        """Enable manual control mode for this agent."""
        self._manual_mode = True
        
    def disable_manual_mode(self) -> None:
        """Disable manual control mode for this agent."""
        self._manual_mode = False
        self._manual_actions_queue.clear()
        
    def is_manual_mode(self) -> bool:
        """Check if agent is in manual control mode."""
        return self._manual_mode
        
    # Role control methods
    def set_role_manually(self, new_role: AgentRole) -> None:
        """Manually change the agent's role."""
        old_role = self.role
        self.role = new_role
        print(f"Agent {self.agent_id}: Role changed from {old_role.value} to {new_role.value}")
        
    def cycle_role(self) -> None:
        """Cycle through available roles."""
        roles = list(AgentRole)
        current_idx = roles.index(self.role)
        next_idx = (current_idx + 1) % len(roles)
        self.set_role_manually(roles[next_idx])
        
    # Elementary act control methods
    def set_elementary_act_manually(self, act: SoldierElementaryAct) -> None:
        """Manually set the agent's current elementary act."""
        old_act = getattr(self, 'current_elementary_act', None)
        self.current_elementary_act = act
        print(f"Agent {self.agent_id}: Elementary act changed from {old_act} to {act.name}")
        
    def cycle_elementary_act(self) -> None:
        """Cycle through available elementary acts."""
        acts = list(SoldierElementaryAct)
        current_act = getattr(self, 'current_elementary_act', None)
        if current_act is None:
            next_act = acts[0]
        else:
            current_idx = acts.index(current_act)
            next_idx = (current_idx + 1) % len(acts)
            next_act = acts[next_idx]
        self.set_elementary_act_manually(next_act)
        
    # Mission control methods
    def set_mission_manually(self, mission: SoldierElementaryAct | None) -> None:
        """Manually set the agent's current mission."""
        old_mission = getattr(self, 'current_mission', None)
        self.current_mission = mission
        mission_name = mission.name if mission else "None"
        old_name = old_mission.name if old_mission else "None"
        print(f"Agent {self.agent_id}: Mission changed from {old_name} to {mission_name}")
        
    def clear_mission(self) -> None:
        """Clear the agent's current mission."""
        self.set_mission_manually(None)
        
    # Position and movement control
    def teleport_to(self, x: float, y: float) -> None:
        """Teleport agent to specific coordinates."""
        old_pos = self.get_position()
        self.set_position((x, y))
        new_pos = self.get_position()
        print(f"Agent {self.agent_id}: Teleported from {old_pos} to {new_pos}")
        
    def set_direction_manually(self, angle: float) -> None:
        """Manually set the agent's facing direction."""
        old_angle = self.get_direction()
        self.state.angle = angle % (2 * np.pi)
        print(f"Agent {self.agent_id}: Direction changed from {old_angle:.2f} to {angle:.2f} radians")
        
    def face_direction(self, direction: str) -> None:
        """Face a cardinal direction."""
        directions = {
            'north': np.pi / 2,
            'east': 0,
            'south': 3 * np.pi / 2,
            'west': np.pi,
            'northeast': np.pi / 4,
            'southeast': 7 * np.pi / 4,
            'southwest': 5 * np.pi / 4,
            'northwest': 3 * np.pi / 4
        }
        if direction.lower() in directions:
            self.set_direction_manually(directions[direction.lower()])
        else:
            print(f"Unknown direction: {direction}")
            
    # Resource control methods
    def set_health_manually(self, health: float) -> None:
        """Manually set the agent's health."""
        old_health = self.get_health()
        self.set_health(health)
        print(f"Agent {self.agent_id}: Health changed from {old_health:.1f} to {health:.1f}")
        
    def set_energy_manually(self, energy: float) -> None:
        """Manually set the agent's energy."""
        old_energy = self.get_energy()
        self.set_energy(energy)
        print(f"Agent {self.agent_id}: Energy changed from {old_energy:.1f} to {energy:.1f}")
        
    def set_ammunition_manually(self, ammo: float) -> None:
        """Manually set the agent's ammunition."""
        old_ammo = self.get_ammunition()
        self.set_ammunition(ammo)
        print(f"Agent {self.agent_id}: Ammunition changed from {old_ammo:.1f} to {ammo:.1f}")
        
    def restore_resources(self) -> None:
        """Restore all resources to maximum."""
        self.set_health_manually(self.max_health)
        self.set_energy_manually(self.max_energy)
        self.set_ammunition_manually(self.max_ammunition)
        print(f"Agent {self.agent_id}: All resources restored to maximum")
        
    # Communication control methods
    def send_manual_message(self, msg_type: MessageType, target_id: int | None = None, 
                           content: dict | None = None) -> None:
        """Manually send a message."""
        if content is None:
            content = self._get_default_message_content(msg_type)
            
        from mili_env.envs.classes.types_common import CommunicationMessage
        message = CommunicationMessage(
            sender_id=self.agent_id,
            receiver_id=target_id,
            message_type=msg_type,
            timestamp=0.0,  # Will be updated when sent
            content=content
        )
        
        self.send_message(message)
        target_str = f"agent {target_id}" if target_id is not None else "all agents"
        print(f"Agent {self.agent_id}: Sent {msg_type.value} message to {target_str}")
        
    def send_status_update(self) -> None:
        """Send a status update message."""
        self.send_manual_message(MessageType.STATUS_UPDATE)
        
    def send_help_request(self, help_type: str = "general") -> None:
        """Send a help request message."""
        content = {
            "help_type": help_type,
            "position": self.get_position(),
            "health": self.get_health(),
            "energy": self.get_energy(),
            "urgency": "high" if self.get_health() < 30 else "normal"
        }
        self.send_manual_message(MessageType.HELP_REQUEST, content=content)
        
    def report_enemy_manual(self, enemy_pos: tuple[float, float]) -> None:
        """Manually report an enemy sighting."""
        content = {
            "agent_id": -1,  # Unknown enemy
            "position": enemy_pos,
            "spotted_at": 0.0,
            "threat_level": "unknown"
        }
        self.send_manual_message(MessageType.ENEMY_SPOTTED, content=content)
        
    def clear_message_outbox(self) -> None:
        """Clear all pending messages."""
        count = len(self.message_outbox)
        self.message_outbox.clear()
        print(f"Agent {self.agent_id}: Cleared {count} messages from outbox")
        
    def _get_default_message_content(self, msg_type: MessageType) -> dict:
        """Get default content for different message types."""
        if msg_type == MessageType.STATUS_UPDATE:
            return {
                "position": self.get_position(),
                "health": self.get_health(),
                "energy": self.get_energy(),
                "ammunition": self.get_ammunition(),
                "role": self.role.value,
                "team": self.team_id,
            }
        elif msg_type == MessageType.HELP_REQUEST:
            return {
                "help_type": "general",
                "position": self.get_position(),
                "health": self.get_health(),
                "energy": self.get_energy(),
                "urgency": "normal"
            }
        else:
            return {}
            
    # Order issuance methods
    def issue_manual_order(self, target_agent_id: int, mission: SoldierElementaryAct) -> None:
        """Manually issue an order to another agent (if authorized)."""
        # This would need access to other agents and the C2 context
        # Store the order for the environment to process
        if not hasattr(self, '_manual_orders'):
            self._manual_orders = {}
        self._manual_orders[target_agent_id] = mission
        print(f"Agent {self.agent_id}: Queued order for agent {target_agent_id}: {mission.name}")
        
    def get_manual_orders(self) -> dict[int, SoldierElementaryAct]:
        """Get manually issued orders."""
        return getattr(self, '_manual_orders', {})
        
    def clear_manual_orders(self) -> None:
        """Clear all manually issued orders."""
        if hasattr(self, '_manual_orders'):
            self._manual_orders.clear()
            print(f"Agent {self.agent_id}: Cleared all manual orders")
            
    # Information display methods
    def get_status_summary(self) -> dict[str, Any]:
        """Get a comprehensive status summary."""
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "position": self.get_position(),
            "direction": self.get_direction(),
            "health": self.get_health(),
            "energy": self.get_energy(),
            "ammunition": self.get_ammunition(),
            "elementary_act": getattr(self, 'current_elementary_act', None),
            "mission": getattr(self, 'current_mission', None),
            "manual_mode": self._manual_mode,
            "pending_messages": len(self.message_outbox),
            "known_agents": len(self.known_agents),
        }
        
    def print_status(self) -> None:
        """Print a formatted status summary."""
        status = self.get_status_summary()
        print(f"\n=== Agent {self.agent_id} Status ===")
        print(f"Role: {status['role']}")
        print(f"Position: ({status['position'][0]:.1f}, {status['position'][1]:.1f})")
        print(f"Direction: {status['direction']:.2f} radians")
        print(f"Health: {status['health']:.1f}")
        print(f"Energy: {status['energy']:.1f}")
        print(f"Ammunition: {status['ammunition']:.1f}")
        
        act = status['elementary_act']
        print(f"Elementary Act: {act.name if act else 'None'}")
        
        mission = status['mission']
        print(f"Mission: {mission.name if mission else 'None'}")
        
        print(f"Manual Mode: {status['manual_mode']}")
        print(f"Pending Messages: {status['pending_messages']}")
        print(f"Known Agents: {status['known_agents']}")


class ManualControlEnvironment(TerrainWorldEnv):
    """Enhanced environment with manual control capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._manual_control_agents = {}
        
    def create_agents(self, *, grouped: bool = False) -> None:
        """Create agents using manual control versions."""
        # Call parent method to set up basic structure
        super().create_agents(grouped=grouped)
        
        # Clear any existing manual control agents
        self._manual_control_agents = {}
        
        # Replace agents with manual control versions
        manual_agents = []
        for i, agent in enumerate(self.agents):
            try:
                # Reconstruct constraints from agent attributes
                from mili_env.envs.classes.robot_base import RobotConstraints
                constraints = RobotConstraints(
                    vision_range=getattr(agent, 'vision_range', 100.0),
                    communication_range=getattr(agent, 'communication_range', 30.0),
                    max_speed_forward=getattr(agent, 'max_speed_forward', 1.0),
                    max_speed_backward=getattr(agent, 'max_speed_backward', -0.2),
                    max_angular_speed=getattr(agent, 'max_angular_speed', 0.785),
                    max_health=getattr(agent, 'max_health', 100.0),
                    max_energy=getattr(agent, 'max_energy', 100.0),
                    max_ammunition=getattr(agent, 'max_ammunition', 100.0),
                )
                
                # Reconstruct position from agent state
                from mili_env.envs.classes.robot_base import RobotPosition
                position = RobotPosition(
                    x=agent.state.x,
                    y=agent.state.y,
                    angle=agent.state.angle
                )
                
                # Create manual control agent with reconstructed parameters
                manual_agent = ManualControlAgent(
                    position=position,
                    attributes=agent.state.attributes,
                    game_map=agent.game_map,
                    constraints=constraints,
                    agent_id=agent.agent_id,
                    role=agent.role,
                    team_id=agent.team_id
                )
                
                # Copy any additional attributes that might exist
                if hasattr(agent, 'current_elementary_act'):
                    manual_agent.current_elementary_act = agent.current_elementary_act
                if hasattr(agent, 'current_mission'):
                    manual_agent.current_mission = agent.current_mission
                    
                manual_agents.append(manual_agent)
                self._manual_control_agents[i] = manual_agent
                
            except Exception as e:
                print(f"Warning: Could not create manual control agent {i}: {e}")
                # Fall back to original agent
                manual_agents.append(agent)
                
        self.agents = manual_agents
        
    def get_manual_agent(self, agent_id: int) -> ManualControlAgent | None:
        """Get a manual control agent by ID."""
        return self._manual_control_agents.get(agent_id)
        
    def enable_manual_mode_all(self) -> None:
        """Enable manual mode for all agents."""
        for agent in self.agents:
            if isinstance(agent, ManualControlAgent):
                agent.enable_manual_mode()
        print("Manual mode enabled for all agents")
        
    def disable_manual_mode_all(self) -> None:
        """Disable manual mode for all agents."""
        for agent in self.agents:
            if isinstance(agent, ManualControlAgent):
                agent.disable_manual_mode()
        print("Manual mode disabled for all agents")
        
    def print_all_agent_status(self) -> None:
        """Print status for all agents."""
        for agent in self.agents:
            if isinstance(agent, ManualControlAgent):
                agent.print_status()
                
    def teleport_agent(self, agent_id: int, x: float, y: float) -> None:
        """Teleport a specific agent."""
        agent = self.get_manual_agent(agent_id)
        if agent:
            agent.teleport_to(x, y)
        else:
            print(f"Agent {agent_id} not found")
            
    def set_all_resources_max(self) -> None:
        """Restore all agents' resources to maximum."""
        for agent in self.agents:
            if isinstance(agent, ManualControlAgent):
                agent.restore_resources()
                
    def get_environment_summary(self) -> dict[str, Any]:
        """Get a summary of the environment state."""
        return {
            "step": getattr(self, 'current_step', 0),
            "num_agents": self.num_agents,
            "num_enemies": self.num_enemies,
            "agents_alive": sum(1 for a in self.agents if a.state.is_alive()),
            "total_energy": sum(a.get_energy() for a in self.agents),
            "total_health": sum(a.get_health() for a in self.agents),
            "agents_at_target": sum(1 for a in self.agents if a.state.is_at_target()),
            "manual_mode_agents": sum(1 for a in self.agents 
                                    if isinstance(a, ManualControlAgent) and a.is_manual_mode()),
        }
        
    def print_environment_summary(self) -> None:
        """Print environment summary."""
        summary = self.get_environment_summary()
        print(f"\n=== Environment Summary ===")
        print(f"Step: {summary['step']}")
        print(f"Agents: {summary['num_agents']} (Alive: {summary['agents_alive']})")
        print(f"Enemies: {summary['num_enemies']}")
        print(f"Total Energy: {summary['total_energy']:.1f}")
        print(f"Total Health: {summary['total_health']:.1f}")
        print(f"Agents at Target: {summary['agents_at_target']}")
        print(f"Manual Mode Agents: {summary['manual_mode_agents']}")


# Convenience functions for manual control
def create_manual_control_environment(num_agents: int = 6, **kwargs) -> ManualControlEnvironment:
    """Create a manual control environment with sensible defaults."""
    return ManualControlEnvironment(
        num_agents=num_agents,
        target_zone_size=15,
        num_enemies=2,
        **kwargs
    )


def setup_manual_control_session() -> tuple[ManualControlEnvironment, list[ManualControlAgent | None]]:
    """Set up a manual control session with environment and agents."""
    env = create_manual_control_environment()
    env.reset(seed=42)
    env.enable_manual_mode_all()
    
    agents = [env.get_manual_agent(i) for i in range(env.num_agents)]
    
    print("Manual control session started!")
    print("Available commands:")
    print("  env.print_environment_summary()")
    print("  agent.print_status()")
    print("  agent.set_role_manually(AgentRole.CDU)")
    print("  agent.cycle_elementary_act()")
    print("  agent.teleport_to(x, y)")
    print("  agent.restore_resources()")
    print("  agent.send_status_update()")
    
    return env, agents
    print("  agent.teleport_to(x, y)")
    print("  agent.restore_resources()")
    print("  agent.send_status_update()")
    
    return env, agents
