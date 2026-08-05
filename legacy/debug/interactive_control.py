"""Interactive control interface for manual agent manipulation.

This module provides a command-line interface for interactively controlling
agents during visualization or debugging sessions.
"""
from __future__ import annotations

import cmd
from typing import Any

from mili_env.envs.classes.types_common import AgentRole, MessageType, SoldierElementaryAct
from debug.manual_control import ManualControlAgent, ManualControlEnvironment


class InteractiveController(cmd.Cmd):
    """Interactive command-line controller for agents."""
    
    intro = """
    Interactive Agent Controller
    Type 'help' for available commands.
    Type 'help <command>' for detailed help on a specific command.
    """
    prompt = '(agent-control) '
    
    def __init__(self, env: ManualControlEnvironment):
        super().__init__()
        self.env = env
        self.current_agent_id = 0
        
    @property
    def current_agent(self) -> ManualControlAgent | None:
        """Get the currently selected agent."""
        return self.env.get_manual_agent(self.current_agent_id)
        
    # Agent selection commands
    def do_select(self, arg: str) -> None:
        """Select an agent by ID: select <agent_id>"""
        try:
            agent_id = int(arg)
            if 0 <= agent_id < self.env.num_agents:
                self.current_agent_id = agent_id
                print(f"Selected agent {agent_id}")
                if self.current_agent:
                    self.current_agent.print_status()
            else:
                print(f"Invalid agent ID. Valid range: 0-{self.env.num_agents-1}")
        except ValueError:
            print("Usage: select <agent_id>")
            
    def do_list(self, arg: str) -> None:
        """List all agents and their basic info."""
        print(f"\n{'ID':<3} {'Role':<6} {'Position':<12} {'Health':<7} {'Energy':<7}")
        print("-" * 50)
        for i, agent in enumerate(self.env.agents):
            if isinstance(agent, ManualControlAgent):
                pos = agent.get_position()
                marker = "*" if i == self.current_agent_id else " "
                print(f"{marker}{i:<2} {agent.role.value:<6} "
                      f"({pos[0]:.1f},{pos[1]:.1f}){'':<3} "
                      f"{agent.get_health():<7.1f} {agent.get_energy():<7.1f}")
                      
    # Role and mission commands
    def do_role(self, arg: str) -> None:
        """Change agent role: role <role_name>"""
        if not self.current_agent:
            print("No agent selected")
            return
            
        if not arg:
            print(f"Current role: {self.current_agent.role.value}")
            print("Available roles:", [r.value for r in AgentRole])
            return
            
        try:
            role = AgentRole(arg.upper())
            self.current_agent.set_role_manually(role)
        except ValueError:
            print(f"Invalid role: {arg}")
            print("Available roles:", [r.value for r in AgentRole])
            
    def do_cycle_role(self, arg: str) -> None:
        """Cycle through available roles."""
        if self.current_agent:
            self.current_agent.cycle_role()
        else:
            print("No agent selected")
            
    def do_elementary_act(self, arg: str) -> None:
        """Change elementary act: elementary_act <act_name>"""
        if not self.current_agent:
            print("No agent selected")
            return
            
        if not arg:
            current = getattr(self.current_agent, 'current_elementary_act', None)
            print(f"Current elementary act: {current.name if current else 'None'}")
            print("Available acts:", [a.name for a in SoldierElementaryAct])
            return
            
        try:
            act = SoldierElementaryAct[arg.upper()]
            self.current_agent.set_elementary_act_manually(act)
        except KeyError:
            print(f"Invalid elementary act: {arg}")
            print("Available acts:", [a.name for a in SoldierElementaryAct])
            
    def do_cycle_act(self, arg: str) -> None:
        """Cycle through elementary acts."""
        if self.current_agent:
            self.current_agent.cycle_elementary_act()
        else:
            print("No agent selected")
            
    # Position and movement commands
    def do_teleport(self, arg: str) -> None:
        """Teleport agent: teleport <x> <y>"""
        if not self.current_agent:
            print("No agent selected")
            return
            
        try:
            coords = arg.split()
            if len(coords) != 2:
                raise ValueError
            x, y = float(coords[0]), float(coords[1])
            self.current_agent.teleport_to(x, y)
        except ValueError:
            print("Usage: teleport <x> <y>")
            
    def do_face(self, arg: str) -> None:
        """Face a direction: face <north|east|south|west|northeast|etc>"""
        if not self.current_agent:
            print("No agent selected")
            return
            
        if not arg:
            print("Usage: face <direction>")
            print("Directions: north, east, south, west, northeast, southeast, southwest, northwest")
            return
            
        self.current_agent.face_direction(arg)
        
    # Resource commands
    def do_health(self, arg: str) -> None:
        """Set health: health <value>"""
        if not self.current_agent:
            print("No agent selected")
            return
            
        if not arg:
            print(f"Current health: {self.current_agent.get_health()}")
            return
            
        try:
            health = float(arg)
            self.current_agent.set_health_manually(health)
        except ValueError:
            print("Usage: health <value>")
            
    def do_energy(self, arg: str) -> None:
        """Set energy: energy <value>"""
        if not self.current_agent:
            print("No agent selected")
            return
            
        if not arg:
            print(f"Current energy: {self.current_agent.get_energy()}")
            return
            
        try:
            energy = float(arg)
            self.current_agent.set_energy_manually(energy)
        except ValueError:
            print("Usage: energy <value>")
            
    def do_ammo(self, arg: str) -> None:
        """Set ammunition: ammo <value>"""
        if not self.current_agent:
            print("No agent selected")
            return
            
        if not arg:
            print(f"Current ammunition: {self.current_agent.get_ammunition()}")
            return
            
        try:
            ammo = float(arg)
            self.current_agent.set_ammunition_manually(ammo)
        except ValueError:
            print("Usage: ammo <value>")
            
    def do_restore(self, arg: str) -> None:
        """Restore all resources to maximum."""
        if self.current_agent:
            self.current_agent.restore_resources()
        else:
            print("No agent selected")
            
    # Communication commands
    def do_status_msg(self, arg: str) -> None:
        """Send status update message."""
        if self.current_agent:
            self.current_agent.send_status_update()
        else:
            print("No agent selected")
            
    def do_help_msg(self, arg: str) -> None:
        """Send help request: help_msg [help_type]"""
        if not self.current_agent:
            print("No agent selected")
            return
            
        help_type = arg if arg else "general"
        self.current_agent.send_help_request(help_type)
        
    def do_enemy_report(self, arg: str) -> None:
        """Report enemy sighting: enemy_report <x> <y>"""
        if not self.current_agent:
            print("No agent selected")
            return
            
        try:
            coords = arg.split()
            if len(coords) != 2:
                raise ValueError
            x, y = float(coords[0]), float(coords[1])
            self.current_agent.report_enemy_manual((x, y))
        except ValueError:
            print("Usage: enemy_report <x> <y>")
            
    def do_clear_messages(self, arg: str) -> None:
        """Clear agent's message outbox."""
        if self.current_agent:
            self.current_agent.clear_message_outbox()
        else:
            print("No agent selected")
            
    # Information commands
    def do_status(self, arg: str) -> None:
        """Show agent status."""
        if self.current_agent:
            self.current_agent.print_status()
        else:
            print("No agent selected")
            
    def do_env_status(self, arg: str) -> None:
        """Show environment status."""
        self.env.print_environment_summary()
        
    # Mode commands
    def do_manual_mode(self, arg: str) -> None:
        """Toggle manual mode: manual_mode [on|off]"""
        if not self.current_agent:
            print("No agent selected")
            return
            
        if arg.lower() == "on":
            self.current_agent.enable_manual_mode()
        elif arg.lower() == "off":
            self.current_agent.disable_manual_mode()
        else:
            current = self.current_agent.is_manual_mode()
            print(f"Manual mode is {'ON' if current else 'OFF'}")
            print("Usage: manual_mode [on|off]")
            
    def do_manual_all(self, arg: str) -> None:
        """Enable/disable manual mode for all agents: manual_all [on|off]"""
        if arg.lower() == "on":
            self.env.enable_manual_mode_all()
        elif arg.lower() == "off":
            self.env.disable_manual_mode_all()
        else:
            print("Usage: manual_all [on|off]")
            
    # Exit command
    def do_exit(self, arg: str) -> bool:
        """Exit the interactive controller."""
        print("Goodbye!")
        return True
        
    def do_quit(self, arg: str) -> bool:
        """Exit the interactive controller."""
        return self.do_exit(arg)


def start_interactive_session(env: ManualControlEnvironment | None = None) -> None:
    """Start an interactive control session."""
    if env is None:
        from debug.manual_control import create_manual_control_environment
        env = create_manual_control_environment()
        env.reset(seed=42)
        
    print("Starting interactive control session...")
    controller = InteractiveController(env)
    controller.cmdloop()


if __name__ == "__main__":
    start_interactive_session()
