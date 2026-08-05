"""Enhanced visualizer with integrated manual control capabilities.

This module combines the C2 visualizer with manual control functionality,
allowing real-time parameter changes during visualization.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.widgets import Button, Slider

from debug.visualize_c2_training import C2Visualizer

if TYPE_CHECKING:
    from debug.manual_control import ManualControlEnvironment


import argparse
from pathlib import Path

from debug.manual_control import ManualControlAgent, create_manual_control_environment
from debug.visualize_c2_training import generate_c2_action

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



class EnhancedC2Visualizer(C2Visualizer):
    """Enhanced visualizer with manual control integration."""

    def __init__(self, env: ManualControlEnvironment, *, save_frames: bool = False) -> None:
        """Initialize with manual control environment and setup widgets."""
        super().__init__(env, save_frames)
        self.manual_env = env
        self.selected_agent_id = 0
        self.manual_mode_states = {}  # Track manual mode for each agent

        # Add control widgets
        self._setup_control_widgets()

    def _setup_control_widgets(self) -> None:
        """Set up interactive control widgets."""
        # Add more space for controls at the bottom
        self.fig.subplots_adjust(bottom=0.25)

        # Control panel area
        control_ax = self.fig.add_axes((0.1, 0.02, 0.8, 0.2))
        control_ax.set_xlim(0, 10)
        control_ax.set_ylim(0, 4)
        control_ax.axis("off")

        # Add title and instructions
        control_ax.text(5, 3.7, "Manual Control Panel", ha="center", va="top", 
                       fontsize=12, fontweight="bold")
        control_ax.text(5, 3.4, "Select agent, then use controls to modify behavior", 
                       ha="center", va="top", fontsize=10)

        # Agent selection buttons
        self.agent_buttons = []
        for i in range(min(8, self.manual_env.num_agents)):  # Max 8 buttons
            button_ax = self.fig.add_axes((0.1 + i * 0.08, 0.16, 0.06, 0.04))
            button = Button(button_ax, f"A{i}")
            button.on_clicked(lambda _event, agent_id=i: self._select_agent(agent_id))
            self.agent_buttons.append(button)

        # Control buttons with better labels and explanations
        button_configs = [
            ("Cycle Act", 0.1, self._cycle_act, "Cycle through: MOVE → POST → FIRE"),
            ("Restore HP", 0.2, self._restore_resources, "Restore health, energy, ammo to 100%"),
            ("Manual OFF", 0.3, self._toggle_manual, "Toggle manual control mode"),
            ("Send Status", 0.4, self._send_status, "Send status update message"),
            ("Print Info", 0.5, self._print_agent_info, "Print agent details to console"),
        ]

        self.control_buttons = []
        for label, x_pos, callback, tooltip in button_configs:
            button_ax = self.fig.add_axes((float(x_pos), 0.11, 0.08, 0.04))
            button = Button(button_ax, label)
            button.on_clicked(callback)
            self.control_buttons.append((button, tooltip))

        # Add explanatory text for controls
        control_ax.text(0.1, 2.8, "Elementary Acts:", fontweight="bold", fontsize=9)
        control_ax.text(0.1, 2.6, "• MOVE: Agent focuses on movement/navigation", fontsize=8)
        control_ax.text(0.1, 2.4, "• POST: Agent takes defensive position", fontsize=8)
        control_ax.text(0.1, 2.2, "• FIRE: Agent focuses on combat/shooting", fontsize=8)

        control_ax.text(0.1, 1.9, "Manual Mode:", fontweight="bold", fontsize=9)
        control_ax.text(0.1, 1.7, "• ON: Agent uses manual parameters", fontsize=8)
        control_ax.text(0.1, 1.5, "• OFF: Agent uses AI decisions", fontsize=8)

        # Selected agent info display
        self.info_ax = self.fig.add_axes((0.65, 0.16, 0.3, 0.08))
        self.info_ax.set_xlim(0, 1)
        self.info_ax.set_ylim(0, 1)
        self.info_ax.axis("off")

        # Health slider
        slider_ax = self.fig.add_axes((0.1, 0.06, 0.25, 0.03))
        self.health_slider = Slider(slider_ax, "Health", 0, 100, valinit=100)
        self.health_slider.on_changed(self._update_health)

        # Energy slider
        slider_ax2 = self.fig.add_axes((0.4, 0.06, 0.25, 0.03))
        self.energy_slider = Slider(slider_ax2, "Energy", 0, 100, valinit=100)
        self.energy_slider.on_changed(self._update_energy)

        # Status display for feedback
        self.status_ax = self.fig.add_axes((0.7, 0.06, 0.25, 0.03))
        self.status_ax.set_xlim(0, 1)
        self.status_ax.set_ylim(0, 1)
        self.status_ax.axis("off")
        self.status_text = self.status_ax.text(0.5, 0.5, "", ha="center", va="center", 
                                              fontsize=9, color="green")

    def _select_agent(self, agent_id: int) -> None:
        """Select an agent for manual control."""
        if 0 <= agent_id < self.manual_env.num_agents:
            self.selected_agent_id = agent_id
            agent = self.manual_env.get_manual_agent(agent_id)
            if agent:
                # Update sliders to current values
                self.health_slider.reset()
                self.health_slider.set_val(agent.get_health())
                self.energy_slider.reset()
                self.energy_slider.set_val(agent.get_energy())
                
                # Update manual mode button label
                is_manual = agent.is_manual_mode()
                self.manual_mode_states[agent_id] = is_manual
                self.control_buttons[2][0].label.set_text("Manual ON" if is_manual else "Manual OFF")
                self.control_buttons[2][0].color = "lightgreen" if is_manual else "lightcoral"
                
                # Update agent info display
                self._update_agent_info_display(agent)
                
                # Show feedback
                self._show_status_message(f"Selected Agent {agent_id}")

    def _update_agent_info_display(self, agent: ManualControlAgent) -> None:
        """Update the agent info display area."""
        self.info_ax.clear()
        self.info_ax.set_xlim(0, 1)
        self.info_ax.set_ylim(0, 1)
        self.info_ax.axis("off")
        
        # Display current agent info
        info_text = f"Agent {agent.agent_id} ({agent.role.value})\n"
        current_act = getattr(agent, 'current_elementary_act', None)
        if current_act:
            info_text += f"Act: {current_act.name}\n"
        else:
            info_text += "Act: None\n"
        
        info_text += f"Health: {agent.get_health():.0f}\n"
        info_text += f"Energy: {agent.get_energy():.0f}"
        
        self.info_ax.text(0.05, 0.95, info_text, ha="left", va="top", fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    def _cycle_act(self, event) -> None:
        """Cycle the selected agent's elementary act."""
        agent = self.manual_env.get_manual_agent(self.selected_agent_id)
        if agent:
            old_act = getattr(agent, 'current_elementary_act', None)
            agent.cycle_elementary_act()
            new_act = getattr(agent, 'current_elementary_act', None)
            
            # Update info display
            self._update_agent_info_display(agent)
            
            # Show feedback
            old_name = old_act.name if old_act else "None"
            new_name = new_act.name if new_act else "None"
            self._show_status_message(f"Act: {old_name} → {new_name}")

    def _restore_resources(self, event) -> None:
        """Restore selected agent's resources."""
        agent = self.manual_env.get_manual_agent(self.selected_agent_id)
        if agent:
            agent.restore_resources()
            
            # Update sliders to reflect new values
            self.health_slider.set_val(agent.get_health())
            self.energy_slider.set_val(agent.get_energy())
            
            # Update info display
            self._update_agent_info_display(agent)
            
            # Show feedback
            self._show_status_message("Resources restored to 100%")

    def _toggle_manual(self, event) -> None:
        """Toggle manual mode for selected agent."""
        agent = self.manual_env.get_manual_agent(self.selected_agent_id)
        if agent:
            if agent.is_manual_mode():
                agent.disable_manual_mode()
                self.control_buttons[2][0].label.set_text("Manual OFF")
                self.control_buttons[2][0].color = "lightcoral"
                self._show_status_message("Manual mode OFF")
            else:
                agent.enable_manual_mode()
                self.control_buttons[2][0].label.set_text("Manual ON")
                self.control_buttons[2][0].color = "lightgreen"
                self._show_status_message("Manual mode ON")
            
            self.manual_mode_states[self.selected_agent_id] = agent.is_manual_mode()

    def _send_status(self, event) -> None:
        """Send status message from selected agent."""
        agent = self.manual_env.get_manual_agent(self.selected_agent_id)
        if agent:
            # Count messages before
            msg_count_before = len(getattr(agent, 'message_outbox', []))
            
            agent.send_status_update()
            
            # Count messages after
            msg_count_after = len(getattr(agent, 'message_outbox', []))
            
            # Show feedback
            if msg_count_after > msg_count_before:
                self._show_status_message("Status message sent")
            else:
                self._show_status_message("Message failed to send")

    def _print_agent_info(self, event) -> None:
        """Print detailed agent information to console."""
        agent = self.manual_env.get_manual_agent(self.selected_agent_id)
        if agent:
            agent.print_status()
            self._show_status_message("Info printed to console")

    def _show_status_message(self, message: str, duration: int = 3000) -> None:
        """Show a temporary status message."""
        self.status_text.set_text(message)
        self.status_text.set_color("green")
        
        # Schedule message clearing (simple implementation)
        import matplotlib.pyplot as plt
        plt.draw()

    def _update_health(self, val: float) -> None:
        """Update selected agent's health."""
        agent = self.manual_env.get_manual_agent(self.selected_agent_id)
        if agent:
            agent.set_health_manually(val)
            self._update_agent_info_display(agent)
            self._show_status_message(f"Health set to {val:.0f}")

    def _update_energy(self, val: float) -> None:
        """Update selected agent's energy."""
        agent = self.manual_env.get_manual_agent(self.selected_agent_id)
        if agent:
            agent.set_energy_manually(val)
            self._update_agent_info_display(agent)
            self._show_status_message(f"Energy set to {val:.0f}")

def main() -> None:
    """Run enhanced visualizer with manual control."""
    parser = argparse.ArgumentParser(description="Enhanced C2 visualization with manual control")
    parser.add_argument("--scenario", choices=["random_orders", "elementary_acts", "communications"],
                       default="random_orders", help="C2 scenario to visualize")
    parser.add_argument("--agents", type=int, default=6, help="Number of agents")
    parser.add_argument("--speed", type=float, default=1.0, help="Animation speed multiplier")
    parser.add_argument("--save-frames", action="store_true", help="Save frames as images")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps to visualize")

    args = parser.parse_args()

    # Create frames directory if saving
    if args.save_frames:
        Path("debug/frames").mkdir(parents=True, exist_ok=True)

    # Setup manual control environment
    env = create_manual_control_environment(num_agents=args.agents)
    env.reset(seed=42)

    # Create enhanced visualizer
    visualizer = EnhancedC2Visualizer(env, save_frames=args.save_frames)

    logger.info("Starting enhanced C2 visualization: %s", args.scenario)
    logger.info("Agents: %d, Steps: %d", args.agents, args.steps)
    logger.info("\nManual Controls Explained:")
    logger.info("=========================")
    logger.info("AGENT BUTTONS (A0-A7): Click to select which agent to control")
    logger.info("CYCLE ACT: Changes agent's behavior focus (MOVE/POST/FIRE)")
    logger.info("  - MOVE: Agent prioritizes movement and navigation")
    logger.info("  - POST: Agent takes defensive positions, holds ground")
    logger.info("  - FIRE: Agent focuses on combat and engaging enemies")
    logger.info("RESTORE HP: Instantly restore agent's health, energy, and ammo to 100%%")
    logger.info("MANUAL ON/OFF: Toggle between AI control and manual parameter control")
    logger.info("SEND STATUS: Agent broadcasts its current status to other agents")
    logger.info("PRINT INFO: Display detailed agent information in console")
    logger.info("HEALTH/ENERGY SLIDERS: Directly set agent's resource levels")
    logger.info("\nSelected agent is highlighted with yellow circle")
    logger.info("Status messages appear in bottom right")
    logger.info("Close window to stop visualization")

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
                logger.info("Episode terminated at step %d", step)
                break

            # Handle window events
            if not plt.get_fignums():  # Window was closed
                break

    except KeyboardInterrupt:
        logger.exception("\nVisualization interrupted by user")

    finally:
        plt.ioff()
        if args.save_frames:
            logger.info("Frames saved to debug/frames/ (total: %d)", visualizer.frame_count)
        logger.info("Visualization complete")


if __name__ == "__main__":
    main()
