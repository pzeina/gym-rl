from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyglet

import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from mili_env.envs.classes.robot_base import Actions, RobotAttributes, RobotBase, RobotConstraints, RobotPosition
from mili_env.envs.classes.terrain import GameMap, Terrain
from mili_env.envs.timing_utils import timing_log, timing_start, timing_stop

# No global passive checker monkeypatching or warning filtering here. The
# environment implements a multi-agent interface and returns dicts for
# observations/rewards/dones; RLlib wraps this appropriately. Keep the
# implementation focused and compliant with gymnasium's API where possible.

PI_OVER_4: float = np.pi / 4
PI_OVER_8: float = np.pi / 8
MIN_AGENT_DISTANCE: int = 3  # Minimum distance between agents at spawn
MIN_AGENTS_FOR_COOPERATION: int = 2  # Minimum agents needed for cooperation bonus
COMMUNICATION_UPDATE_INTERVAL: int = 10  # Steps between automatic status broadcasts

# Energy costs for different actions
ACTION_ENERGY_COSTS = {
    Actions.IDLE.value: 0.1,
    Actions.FORWARD.value: 1.0,
    Actions.BACKWARD.value: 1.2,
    Actions.ROTATE_LEFT.value: 0.5,
    Actions.ROTATE_RIGHT.value: 0.5,
}


class TerrainWorldEnv(gym.Env):
    """Custom environment for the terrain world with multi-agent support."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}  # noqa: RUF012

    def __init__(
        self,
        render_mode: str | None = None,
        target_zone_size: int = 20,
        terrain_filename: str = "plain_terrain.csv",
        num_agents: int = 1
    ) -> None:
        """Initialize the environment."""
        self.max_window_size: int = 512
        self.panel_width: int = 128  # Width of the right panel
        self.target_zone_size: int = target_zone_size  # Size of the target zone
        self.num_agents: int = num_agents  # Number of agents
        self.current_step: int = 0  # Step counter for communication timing

        # Random number generator for reproducible behavior when seed is set
        # via `reset(seed=...)`.
        self.np_random = np.random.default_rng()

        # Load the terrain map
        self.game_map = GameMap.load_from_csv(Path(__file__).parent / "data" / terrain_filename)
        self.width: int = self.game_map.width
        self.height: int = self.game_map.height

        # Calculate the window size to ensure square pixels
        self.cell_size = min(self.max_window_size // self.width, self.max_window_size // self.height)
        self.window_width = self.cell_size * (self.width + self.panel_width)
        self.window_height = self.cell_size * self.height

        # Initialize multi-agent components
        self.agents: list[RobotBase] = []
        self.agent_colors: list[tuple[int, int, int]] = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 128, 255),  # Sky Blue
            (128, 255, 0),  # Lime Green
            (128, 128, 128), # Gray
            (255, 192, 203), # Pink
        ]
        self.create_agents(grouped=True)

        # Observations are dictionaries with each agent's local information.
        # Each agent observes its own position, target, distance, direction, and energy
        self.observation_space = Dict({
            f"agent_{i}": Dict({
                "position": Box(0, max(self.width, self.height) - 1, shape=(2,), dtype=np.float64),
                "target_position": Box(0, max(self.width, self.height) - 1, shape=(2,), dtype=np.int64),
                "distance": Box(0.0, np.sqrt(self.width**2 + self.height**2), shape=(1,), dtype=np.float64),
                "direction": Box(0.0, 2 * np.pi, shape=(1,), dtype=np.float64),
                "target_direction": Box(0.0, 2 * np.pi, shape=(1,), dtype=np.float64),
                "energy": Box(0.0, 100.0, shape=(1,), dtype=np.float64),  # Using max energy constant
                "agent_id": Discrete(self.num_agents),
            }) for i in range(self.num_agents)
        })

        # Action space for multiple agents - each agent can take one discrete action
        self.action_space = Dict({
            f"agent_{i}": Discrete(len(Actions)) for i in range(self.num_agents)
        })

        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.batch = None
        self.fps = self.metadata["render_fps"]
        self.zoom_factor = 1.0

    def _find_group_center(self, target_zones: list) -> np.ndarray | None:
        """Find a suitable center location for grouped agent spawning."""
        # Calculate grid dimensions for compact square formation
        grid_size = int(np.ceil(np.sqrt(self.num_agents)))
        group_radius = max(2, grid_size // 2 + 1)  # Buffer based on grid size

        # Ensure we have enough space for the group
        min_dimension = 2 * group_radius + 1
        if self.width < min_dimension or self.height < min_dimension:
            return None  # Not enough space for grouped spawning

        max_attempts = 100  # Prevent infinite loops

        for _attempt in range(max_attempts):
            # Try to find a center that's not too close to the edges
            margin = group_radius
            center_x = self.np_random.integers(margin, self.width - margin)
            center_y = self.np_random.integers(margin, self.height - margin)
            group_center = np.array([center_x, center_y])

            valid_center = True

            # Check if group center area conflicts with any target zone
            for _, other_target_zone in target_zones:
                # Define the group area bounds
                group_min = group_center - group_radius
                group_max = group_center + group_radius

                # Check for overlap with target zone
                target_min = other_target_zone[0]
                target_max = other_target_zone[1]

                # Check if rectangles overlap
                if not (group_max[0] < target_min[0] or group_min[0] > target_max[0] or
                       group_max[1] < target_min[1] or group_min[1] > target_max[1]):
                    valid_center = False
                    break

            if valid_center:
                return group_center

        # If we couldn't find a valid center after max_attempts, return None
        return None

    def _get_grouped_agent_location(self, agent_id: int, group_center: np.ndarray) -> np.ndarray:
        """Get location for an agent in grid formation."""
        # Calculate grid dimensions for compact square formation
        grid_size = int(np.ceil(np.sqrt(self.num_agents)))

        # Calculate grid position for this agent
        grid_row = agent_id // grid_size
        grid_col = agent_id % grid_size

        # Calculate offset from center (with minimum spacing of 1 unit)
        # Center the grid around the group_center
        start_row = -(grid_size - 1) // 2
        start_col = -(grid_size - 1) // 2

        offset_x = start_col + grid_col
        offset_y = start_row + grid_row

        agent_location = group_center + np.array([offset_x, offset_y])

        # Ensure location is within bounds and valid
        return np.clip(agent_location, 0, [self.width - 1, self.height - 1])

    def _is_location_valid(
            self, agent_location: np.ndarray, target_zones: list, *, grouped: bool
        ) -> bool:
        """Check if an agent location is valid."""
        # Check if location conflicts with any target zone
        for _, other_target_zone in target_zones:
            if (
                (other_target_zone[0] <= agent_location).all()
                and (agent_location <= other_target_zone[1]).all()
            ):
                return False

        # Distance check logic depends on grouping mode
        min_distance = 1.0 if grouped else MIN_AGENT_DISTANCE
        for existing_agent in self.agents:
            existing_pos = existing_agent.get_position()
            distance = np.linalg.norm(agent_location - np.array(existing_pos))
            if distance < min_distance:
                return False

        return True

    def create_agents(self, *, grouped: bool = False) -> None: # noqa: C901, PLR0912
        """Create multiple robot agents."""
        self.agents = []
        target_zones = []

        # Create separate target zones for each agent
        for _agent_id in range(self.num_agents):
            # Define the target zone for this agent
            target_zone_center = self.np_random.integers(0, [self.width, self.height], size=2, dtype=int)
            target_zone = np.asarray([
                target_zone_center - self.target_zone_size // 2,
                target_zone_center + self.target_zone_size // 2,
            ])
            target_zone = np.clip(target_zone, 0, [self.width - 1, self.height - 1])
            target_zones.append((target_zone_center, target_zone))

        # Find group center if grouped spawning is enabled
        group_center = None
        use_grouped_spawning = False
        if grouped and self.num_agents > 1:
            group_center = self._find_group_center(target_zones)
            use_grouped_spawning = group_center is not None

        # Create agents with appropriate positioning
        for agent_id in range(self.num_agents):
            target_zone_center, target_zone = target_zones[agent_id]

            # Choose the agent's location with fallback mechanism
            max_attempts = 50  # Prevent infinite loops
            agent_location = None

            for _attempt in range(max_attempts):
                if use_grouped_spawning and group_center is not None:
                    candidate_location = self._get_grouped_agent_location(agent_id, group_center)
                else:
                    # Random location as fallback
                    candidate_location = self.np_random.integers(0, [self.width, self.height], size=2, dtype=int)

                if self._is_location_valid(candidate_location, target_zones, grouped=use_grouped_spawning):
                    agent_location = candidate_location
                    break

            # Final fallback: place agent at a guaranteed valid location if all attempts failed
            if agent_location is None:
                # Find the first valid location by systematic search
                for x in range(self.width):
                    for y in range(self.height):
                        candidate_location = np.array([x, y])
                        if self._is_location_valid(candidate_location, target_zones, grouped=False):
                            agent_location = candidate_location
                            break
                    if agent_location is not None:
                        break

                # If still no valid location found (very unlikely), use fallback position
                if agent_location is None:
                    agent_location = np.array([
                        min(agent_id, self.width - 1),
                        min(agent_id, self.height - 1)
                    ])

            # Create robot position and attributes
            initial_position = RobotPosition(
                x=agent_location[0],
                y=agent_location[1],
                angle=0,
                target_x=target_zone_center[0],
                target_y=target_zone_center[1],
                target_width=self.target_zone_size,
                target_height=self.target_zone_size,
            )
            initial_attributes = RobotAttributes(
                health=100.0,
                energy=100.0,
                ammunition=100.0,
                health_efficiency=1.0,
                energy_efficiency=1.0,
                speed_efficiency=5.0,
                ammunition_efficiency=1.0,
            )
            constraints = RobotConstraints(
                vision_range=100.0,
                communication_range=30.0,
                max_speed_forward=1.0,
                max_speed_backward=0.2,
                max_angular_speed=PI_OVER_8,
                max_health=100.0,
                max_energy=100.0,
                max_ammunition=100.0,
            )

            agent = RobotBase(
                position=initial_position,
                attributes=initial_attributes,
                game_map=self.game_map,
                constraints=constraints
            )
            self.agents.append(agent)

    def _get_obs(self) -> dict:
        """Get the decentralized observations for all agents."""
        observations = {}

        for agent_id, agent in enumerate(self.agents):
            direction: float = agent.get_direction()
            target_direction: float = agent.get_target_direction()
            target_x, target_y = agent.get_target()

            observations[f"agent_{agent_id}"] = {
                "position": np.asarray(agent.get_position(), dtype=np.float64),
                "target_position": np.array([target_x, target_y], dtype=np.int64),
                "distance": np.array(
                    [np.linalg.norm(np.asarray(agent.get_position()) - np.array([target_x, target_y]), ord=1)],
                    dtype=np.float64
                ),
                "direction": np.array([direction], dtype=np.float64),
                "target_direction": np.array([target_direction], dtype=np.float64),
                "energy": np.array([agent.get_energy()], dtype=np.float64),
                "agent_id": agent_id,
            }

        return observations

    def _normalize_observation(self, observation: dict) -> dict:
        """Normalize the observation values to be between 0 and 1."""
        normalized_obs = {}

        for agent_key, agent_obs in observation.items():
            normalized_agent_obs = {}
            for obs_key, value in agent_obs.items():
                if obs_key in ["position", "target_position"]:
                    # Normalize position and target position (remain 1-D or 2-D arrays)
                    normalized_agent_obs[obs_key] = np.asarray(value / np.array([self.width, self.height]))
                elif obs_key == "distance":
                    # Normalize distance -> ensure result is 1-D array
                    normalized_agent_obs[obs_key] = np.atleast_1d(
                        np.asarray(value / np.sqrt(self.width**2 + self.height**2))
                    )
                elif obs_key in ["direction", "target_direction"]:
                    # Normalize direction -> ensure result is 1-D array
                    normalized_agent_obs[obs_key] = np.atleast_1d(np.asarray(value / (2 * np.pi)))
                elif obs_key == "energy":
                    # Normalize energy -> ensure result is 1-D array (using max energy constant)
                    normalized_agent_obs[obs_key] = np.atleast_1d(np.asarray(value / 100.0))
                elif obs_key == "agent_id":
                    # Keep agent_id as is
                    normalized_agent_obs[obs_key] = value
                else:
                    msg = f"Unknown observation key: {obs_key}"
                    raise ValueError(msg)
            normalized_obs[agent_key] = normalized_agent_obs

        return normalized_obs

    def _get_info(self) -> dict:
        """Get the information of the environment for all agents."""
        info = {}

        for agent_id, agent in enumerate(self.agents):
            info[f"agent_{agent_id}"] = {
                "angle_to_target": agent.get_angle_to_target(),
                "distance": agent.get_distance_to_target(),
                "health": np.asarray(agent.get_health()),
                "energy": np.asarray(agent.get_energy()),
                "ammunition": np.asarray(agent.get_ammunition()),
            }

        return info

    def _get_reward(self, prev_info: dict) -> dict[str, np.floating[Any]]:
        """Get the centralized reward for all agents (same reward for cooperative behavior)."""
        current_info = self._get_info()

        # Calculate collective performance metrics
        total_distance_to_targets = 0.0
        agents_at_target = 0
        alive_agents = 0
        total_energy = 0.0
        total_health = 0.0

        # Calculate distance changes for progress tracking
        total_distance_improvement = 0.0

        for agent_id, agent in enumerate(self.agents):
            agent_key = f"agent_{agent_id}"
            current_distance = current_info[agent_key]["distance"]
            prev_distance = prev_info[agent_key]["distance"] if agent_key in prev_info else current_distance

            total_distance_to_targets += current_distance
            total_distance_improvement += (prev_distance - current_distance)

            if agent.state.is_at_target():
                agents_at_target += 1
            if agent.state.is_alive():
                alive_agents += 1

            total_energy += agent.get_energy()
            total_health += agent.get_health()

        # Centralized reward calculation
        base_reward = 0.0

        # Mission completion bonus (all agents reach targets)
        if agents_at_target == self.num_agents:
            base_reward = 1000.0 + 0.5 * total_energy + 0.25 * total_health

        # Mission failure penalty (any agent dies or runs out of energy)
        elif alive_agents < self.num_agents or any(not agent.state.has_energy() for agent in self.agents):
            base_reward = -500.0

        # Progress reward (cooperative movement towards targets)
        else:
            # Reward for collective progress towards targets
            progress_reward = total_distance_improvement * 10.0

            # Cooperation bonus: reward for staying close to each other
            cooperation_bonus = self._calculate_cooperation_bonus()

            # Energy efficiency bonus
            energy_bonus = total_energy / (self.num_agents * 100.0) * 5.0

            base_reward = progress_reward + cooperation_bonus + energy_bonus

        # Return the same centralized reward for all agents
        rewards = {}
        for agent_id in range(self.num_agents):
            rewards[f"agent_{agent_id}"] = np.float64(base_reward)

        return rewards

    def _calculate_cooperation_bonus(self) -> float:
        """Calculate bonus reward for agents staying together and cooperating."""
        if self.num_agents < MIN_AGENTS_FOR_COOPERATION:
            return 0.0

        total_bonus = 0.0
        cooperation_pairs = 0

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                agent_i_pos = np.array(self.agents[i].get_position())
                agent_j_pos = np.array(self.agents[j].get_position())
                distance = np.linalg.norm(agent_i_pos - agent_j_pos)

                # Reward for staying within communication range
                if distance <= self.agents[i].communication_range:
                    total_bonus += 5.0
                    cooperation_pairs += 1
                # Small penalty for being too far apart
                elif distance > 2 * self.agents[i].communication_range:
                    total_bonus -= 1.0

        # Normalize by number of possible pairs
        max_pairs = (self.num_agents * (self.num_agents - 1)) // 2
        return total_bonus / max_pairs if max_pairs > 0 else 0.0

    def _get_terminates(self) -> dict[str, np.bool_]:
        """Check if the environment terminates for each agent."""
        terminates = {}

        # Check if all agents reached their targets or any agent failed
        all_at_target = all(agent.state.is_at_target() for agent in self.agents)
        any_agent_dead = any(not agent.state.is_alive() for agent in self.agents)
        any_agent_no_energy = any(not agent.state.has_energy() for agent in self.agents)

        global_terminate = all_at_target or any_agent_dead or any_agent_no_energy

        for agent_id in range(self.num_agents):
            terminates[f"agent_{agent_id}"] = np.bool_(global_terminate)

        return terminates

    def _get_truncates(self) -> dict[str, np.bool_]:
        """Check if the environment truncates for each agent."""
        truncates = {}
        for agent_id in range(self.num_agents):
            truncates[f"agent_{agent_id}"] = np.bool_(False)  # noqa: FBT003
        return truncates

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple:  # noqa: D102, ARG002
        timing_start("reset")

        # Handle seeding manually if needed (skip super().reset() to avoid passive checker)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Reset step counter
        self.current_step = 0

        self.create_agents()

        timing_stop("reset")
        timing_log()
        return self._get_obs(), self._get_info()

    def _consume_energy(self, agent: RobotBase, action: int | np.ndarray, terrain: Terrain) -> None:
        """Consume energy for a specific agent based on action and terrain."""
        if isinstance(action, np.ndarray):
            for single_action in action:
                if single_action in ACTION_ENERGY_COSTS:
                    base_energy_cost = ACTION_ENERGY_COSTS[single_action]
                    energy_cost = terrain.get_movement_cost() * base_energy_cost
                    agent.state.consume_energy(energy_cost)
        elif action in ACTION_ENERGY_COSTS:
            base_energy_cost = ACTION_ENERGY_COSTS[action]
            energy_cost = terrain.get_movement_cost() * base_energy_cost
            agent.state.consume_energy(energy_cost)

    def _process_communications(self) -> None:
        """Process communications between agents within communication range."""
        current_time = float(self.current_step)

        # Clear any pending messages and process communications
        for sender in self.agents:
            outbox = getattr(sender, "communication_outbox", [])

            # Send messages to agents within communication range
            for message in outbox.copy():
                for _receiver_id, receiver in enumerate(self.agents):
                    if sender == receiver:
                        continue

                    # Check if agents are within communication range
                    sender_pos = np.array(sender.get_position())
                    receiver_pos = np.array(receiver.get_position())
                    distance = np.linalg.norm(sender_pos - receiver_pos)

                    if distance <= sender.communication_range:
                        # Deliver message to receiver
                        receiver.receive_message(message)

                # Remove sent message from outbox
                outbox.remove(message)

            # Process any received messages
            if hasattr(sender, "process_messages"):
                sender.process_messages(current_time)

            # Auto-broadcast status updates periodically
            if (hasattr(sender, "last_communication_time")
                and current_time - sender.last_communication_time > COMMUNICATION_UPDATE_INTERVAL):
                    sender.broadcast_status()
                    sender.last_communication_time = current_time

    def step(self, action: dict[str, int] | dict[str, np.ndarray]) -> tuple:  # noqa: D102
        timing_start("step")

        # Increment step counter
        self.current_step += 1

        # Store previous information for reward calculation
        prev_info = self._get_info()

        # Execute actions for all agents
        for agent_id, agent in enumerate(self.agents):
            agent_key = f"agent_{agent_id}"
            agent_action = action.get(agent_key, Actions.IDLE.value)

            position: tuple[float, float] = agent.get_position()
            terrain: Terrain = self.game_map.get_terrain(position[0], position[1])

            agent.move(agent_action)
            self._consume_energy(agent, agent_action, terrain)

        # Process communications between agents
        self._process_communications()

        # Get observations, per-agent rewards, termination flags and info
        observation = self._get_obs()
        per_agent_rewards = self._get_reward(prev_info)
        per_agent_terminated = self._get_terminates()
        per_agent_truncated = self._get_truncates()
        info = self._get_info()

        # Convert to scalar reward and global booleans for gymnasium compatibility
        # Scalar reward: use the centralized base_reward computed in _get_reward
        # (every agent currently gets the same value). We'll extract any agent's
        # reward (agent_0) as the scalar reward.
        try:
            scalar_reward = float(per_agent_rewards.get("agent_0", 0.0))
        except (AttributeError, TypeError, KeyError):
            scalar_reward = 0.0

        # Global termination/truncation: if any agent has terminated/truncated
        global_terminated = any(bool(v) for v in per_agent_terminated.values())
        global_truncated = any(bool(v) for v in per_agent_truncated.values())

        timing_stop("step")
        timing_log()
        return observation, scalar_reward, global_terminated, global_truncated, info

    # ----------------- Rendering -----------------
    def _draw_basic_map(self) -> None:
        """Draw the basic map with all agents."""
        if self.window is not None:
            self.window.clear()
        pix = int(self.cell_size * self.zoom_factor)

        # Draw terrain
        for i in range(self.game_map.height):
            for j in range(self.game_map.width):
                x = int(j * pix)
                y = int(i * pix)
                pyglet.graphics.draw(
                    4, pyglet.gl.GL_QUADS,
                    vertices=(
                        "v2i", (x, y, x + pix, y, x + pix, y + pix, x, y + pix)
                    )
                )

        # Draw all agents
        for agent_id, agent in enumerate(self.agents):
            if agent is not None:
                posx, posy = agent.get_position()
                rx = int(posx * pix + pix // 2)
                ry = int(posy * pix + pix // 2)
                r = int(pix * 0.4)
                n = 20
                verts = []
                for k in range(n):
                    theta = 2.0 * np.pi * k / n
                    verts += [int(rx + r * np.cos(theta)), int(ry + r * np.sin(theta))]

                # Use different colors for different agents
                color = self.agent_colors[agent_id % len(self.agent_colors)]
                pyglet.graphics.draw(n, pyglet.gl.GL_TRIANGLES,
                                   vertices=("v2i", tuple(verts)),
                                   color=("c3B", color * n))

    def _get_rgb_array(self) -> np.ndarray:
        """Get the current frame as an RGB array."""
        buffer = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        width = buffer.width
        height = buffer.height
        data = buffer.get_data("RGB", width * 3)
        arr = np.frombuffer(data, dtype=np.uint8)
        arr = arr.reshape((height, width, 3))
        return np.flipud(arr)

    def render(self, mode: str | None = None) -> np.ndarray | None:
        """Render the environment."""
        mode = mode or self.render_mode
        if mode is None:
            return None
        # self._draw_basic_map()
        if self.window is not None:
            self.window.switch_to()
            self.window.dispatch_events()
            self.window.flip()
        if mode == "rgb_array":
            return self._get_rgb_array()
        return None


    def close(self) -> None:
        """Close the Pyglet window."""
        if self.window is not None:
            self.window.close()
            self.window = None

    def render_vision_map(self, vision_map: np.ndarray, batch: pyglet.graphics.Batch, x: int, y: int) -> None:
        """Render the vision map using pyglet shapes."""
        cell_size = self.cell_size  # * self.zoom_factor
        for i, row in enumerate(vision_map):
            for j, terrain in enumerate(row):
                if terrain is not None:
                    color = terrain.get_color()
                    _ = pyglet.shapes.Rectangle(
                        x + j * cell_size, y + i * cell_size,
                        cell_size, cell_size, color=color, batch=batch
                    )

    def _draw_fps_slider(self, batch: pyglet.graphics.Batch, max_fps: int = 200) -> None:
        """Draw the FPS slider using pyglet."""
        _ = pyglet.text.Label(
            f"FPS: {self.fps}", font_size=14,
            x=self.window_width - self.panel_width + 10,
            y=self.window_height - 30, color=(255, 255, 255, 255),
            batch=batch
        )
        slider_x = self.window_width - self.panel_width + 10
        slider_y = self.window_height - 60
        slider_width = 200
        slider_height = 20
        pyglet.shapes.Rectangle(slider_x, slider_y, slider_width, slider_height, color=(255, 255, 255), batch=batch)
        handle_x = slider_x + int((self.fps / max_fps) * slider_width)
        pyglet.shapes.Rectangle(handle_x - 5, slider_y - 5, 10, slider_height + 10, color=(255, 0, 0), batch=batch)

    def _draw_debug_info(self, batch: pyglet.graphics.Batch, action: dict[str, int] | dict[str, np.ndarray]) -> None:
        """Draw debug information for all agents."""
        y_offset = 0
        for agent_id, agent in enumerate(self.agents):
            position = agent.get_position()
            terrain = self.game_map.get_terrain(position[0], position[1])
            target_x, target_y = agent.get_target()
            distance_to_target = np.linalg.norm(np.asarray(position) - np.array([target_x, target_y]), ord=1)

            agent_key = f"agent_{agent_id}"
            agent_action = action.get(agent_key, "N/A") if isinstance(action, dict) else "N/A"

            debug_info = [
                f"=== Agent {agent_id} ===",
                f"Action: {agent_action}",
                f"Position: ({position[0]:.2f}, {position[1]:.2f})",
                f"Angle: {agent.state.angle:.2f}",
                f"Target: ({target_x}, {target_y})",
                f"Current Tile: {terrain.__class__.__name__}",
                f"Distance to Target: {distance_to_target:.2f}",
                f"Health: {agent.get_health():.2f}",
                f"Energy: {agent.get_energy():.2f}",
                f"Ammunition: {agent.get_ammunition():.2f}",
                "",  # Separator
            ]

            for i, info in enumerate(debug_info):
                pyglet.text.Label(
                    info, font_size=12,
                    x=self.width * self.cell_size + 10,
                    y=self.window_height - (30 + y_offset + i * 20),
                    color=(255, 255, 255, 255), batch=batch
                )

            y_offset += len(debug_info) * 20
