from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from mili_env.envs.classes.robot_base import Actions, RobotAttributes, RobotBase, RobotConstraints, RobotPosition
from mili_env.envs.classes.terrain import GameMap, Terrain
from mili_env.envs.metrics import AgentEnvInteractionVisualization as Visualization

PI_OVER_4: float = np.pi / 4
PI_OVER_8: float = np.pi / 8


class TerrainWorldEnv(gym.Env):
    """Custom environment for the terrain world."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}  # noqa: RUF012

    def __init__(
        self,
        render_mode: str | None = None,
        target_zone_size: int = 20,
        terrain_filename: str = "plain_terrain.csv",
        *,
        visualization: bool = False,
    ) -> None:
        """Initialize the environment."""
        self.max_window_size: int = 512  # The maximum size of the PyGame window
        self.panel_width: int = 128  # Width of the right panel
        self.target_zone_size: int = target_zone_size  # Size of the target zone

        # Load the terrain map
        self.game_map = GameMap.load_from_csv(Path(__file__).parent / "data" / terrain_filename)
        self.width: int = self.game_map.width
        self.height: int = self.game_map.height

        # Calculate the window size to ensure square pixels
        self.cell_size = min(self.max_window_size // self.width, self.max_window_size // self.height)
        self.window_width = self.cell_size * (self.width + self.panel_width)
        self.window_height = self.cell_size * self.height

        self.create_robot()

        # Initialize visualization
        if visualization:
            self.visualization = Visualization(self.window_width, self.window_height, self.panel_width, np.bool_(False))  # noqa: FBT003
        else:
            self.visualization = None

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(0, max(self.width, self.height) - 1, shape=(2,), dtype=np.float64),
                "target_position": spaces.Box(0, max(self.width, self.height) - 1, shape=(2,), dtype=np.int64),
                "distance": spaces.Box(0.0, np.sqrt(self.width**2 + self.height**2), shape=(), dtype=np.float64),
                "direction": spaces.Box(0.0, 2 * np.pi, shape=(), dtype=np.float64),
                "target_direction": spaces.Box(0.0, 2 * np.pi, shape=(), dtype=np.float64),
                "energy": spaces.Box(0.0, self.robot.max_energy, shape=(), dtype=np.float64),
            }
        )

        # Populate the action space with the possible actions
        self.action_space = spaces.Discrete(Actions.count())

        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.fps = self.metadata["render_fps"]
        self.zoom_factor = 1.0

    def create_robot(self) -> None:
        """Create the robot object."""
        # Define the target zone
        self._target_zone_center = self.np_random.integers(0, [self.width, self.height], size=2, dtype=int)
        self._target_zone = np.asarray(
            [
                self._target_zone_center - self.target_zone_size // 2,
                self._target_zone_center + self.target_zone_size // 2,
            ]
        )
        self._target_zone = np.clip(self._target_zone, 0, [self.width - 1, self.height - 1])

        # Choose the agent's location uniformly at random and ensure it is not inside the target zone
        while True:
            agent_location = self.np_random.integers(0, [self.width, self.height], size=2, dtype=int)
            if not (self._target_zone[0] <= agent_location).all() or not (agent_location <= self._target_zone[1]).all():
                break

        # Create robot position and attributes using the random values
        initial_position = RobotPosition(
            x=agent_location[0],
            y=agent_location[1],
            angle=0,
            target_x=self._target_zone_center[0],
            target_y=self._target_zone_center[1],
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
        self.robot = RobotBase(
            position=initial_position, attributes=initial_attributes, game_map=self.game_map, constraints=constraints
        )

    def _get_obs(self) -> dict:
        """Get the observation of the environment."""
        direction: float = self.robot.get_direction()
        target_direction: float = self.robot.get_target_direction()
        return {
            "position": np.asarray(self.robot.get_position(), dtype=np.float64),
            "target_position": np.array(self.robot.get_target(), dtype=np.int64),
            "distance": np.asarray(
                np.linalg.norm(self.robot.get_position() - self._target_zone_center, ord=1),
                dtype=np.float64,
            ),
            "direction": np.asarray(direction, dtype=np.float64),
            "target_direction": np.asarray(target_direction, dtype=np.float64),
            "energy": np.asarray(self.robot.get_energy(), dtype=np.float64),
        }

    def _normalize_observation(self, observation: dict) -> dict:
        """Normalize the observation values to be between 0 and 1."""
        for key, value in observation.items():
            if key in ["position", "target_position"]:
                # Normalize position and target position
                observation[key] = np.asarray(value / np.array([self.width, self.height]))
            elif key == "distance":
                # Normalize distance
                observation[key] = np.asarray(value / np.sqrt(self.width**2 + self.height**2))
            elif key in ["direction", "target_direction"]:
                # Normalize direction
                observation[key] = np.asarray(value / (2 * np.pi))
            elif key == "energy":
                # Normalize energy
                observation[key] = np.asarray(value / self.robot.max_energy)
            else:
                msg = f"Unknown observation key: {key}"
                raise ValueError(msg)
        return observation

    def _get_info(self) -> dict:
        """Get the information of the environment."""
        return {
            "angle_to_target": self.robot.get_angle_to_target(),
            "distance": self.robot.get_distance_to_target(),
            "health": np.asarray(self.robot.get_health()),
            "energy": np.asarray(self.robot.get_energy()),
            "ammunition": np.asarray(self.robot.get_ammunition()),
            # "reward_dist": -np.linalg.norm(np.asarray(self.robot.get_position()) - self._target_zone_center, ord=1),
            # "reward_ctrl": -1
        }

    def _get_reward(
        self, distance_var: float, angle_var: float, energy_var: float, health_var: float
    ) -> np.floating[Any]:
        """Get the reward of the environment."""
        # float(-np.linalg.norm(np.asarray(self.robot.get_position()) - self._target_zone_center, ord=1))

        if self.robot.state.is_at_target():
            # Mission complete
            reward = TerrainWorldEnv.final_reward(
                0, self.robot.get_energy(), self.robot.get_health(), self.robot.get_ammunition()
            )
        elif not self.robot.state.is_alive() or not self.robot.state.has_energy():
            # Mission failed
            reward = -TerrainWorldEnv.final_reward(
                self.robot.get_distance_to_target(),
                self.robot.max_energy,
                self.robot.max_health,
                self.robot.max_ammunition,
            )
        else:
            # Default progress reward
            reward = np.float64(TerrainWorldEnv.step_reward(distance_var, angle_var, energy_var, health_var))
        # return self._normalize_reward(np.float64(reward))
        return np.float64(reward)

    @staticmethod
    def final_reward(distance: float, energy: float, health: float, ammunition: float) -> float:
        """Calculate the final reward based on distance and attributes."""
        # Calculate the final reward based on distance and attributes
        return distance + 0.5 * energy + 0.25 * health + 0.125 * ammunition

    @staticmethod
    def step_reward(distance_var: float, angle_var: float, energy_var: float, health_var: float) -> float:
        """Calculate the by-step reward based on distance/angle-to-target, energy, and health variations."""
        # Better to increase: energy, health
        # Better to decrease: distance/angle-to-target
        return -distance_var - (angle_var / np.pi) + energy_var + health_var

    def _normalize_reward(self, reward: np.floating[Any]) -> np.floating[Any]:
        """Normalize the reward values to be between 0 and 1."""
        # Normalize the reward
        max_reward = TerrainWorldEnv.final_reward(
            0, self.robot.max_energy, self.robot.max_health, self.robot.max_ammunition
        )
        normalized_reward = reward / max_reward
        return np.float64(normalized_reward)

    def _get_terminates(self) -> np.bool_:
        """Check if the environment terminates."""
        return np.bool_(
            self.robot.state.is_at_target() or not self.robot.state.is_alive() or not self.robot.state.has_energy()
        )

    def _get_truncates(self) -> np.bool_:
        """Check if the environment truncates."""
        return np.bool_(False)  # noqa: FBT003

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple:  # noqa: D102
        super().reset(seed=seed, options=options)

        self.create_robot()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame(action=Actions.IDLE.value)

        return observation, info

    # def _single_step(self, action: int) -> tuple:
    #     """Execute a single step in the environment."""
    #     # Check if the new location is traversable
    #     prev_position = self.robot.get_position()
    #     prev_info = self._get_info()
    #     terrain: Terrain = self.game_map.get_terrain(prev_position[0], prev_position[1])

    #     # Perform the action on the terrain and
    #     # Update position, vision rays, etc.
    #     self.robot.move(action)

    #     # Consume energy from
    #     if action in [Actions.FORWARD.value, Actions.BACKWARD.value]:
    #         # if terrain.get_properties().traversable:
    #         movement_cost = terrain.get_movement_cost()
    #         self.robot.state.consume_energy(movement_cost)
    #     elif action in [Actions.ROTATE_LEFT.value, Actions.ROTATE_RIGHT.value]:
    #         # Minimal energy consumption when rotating
    #         self.robot.state.consume_energy(0.15)
    #     elif action == Actions.IDLE.value:
    #         # Minimal energy consumption when staying with no movement
    #         self.robot.state.consume_energy(0.1)

    #     # Calculate the distance to the target
    #     # distance_to_target = np.linalg.norm(np.asarray(self.robot.get_position()) - self._target_zone_center, ord=1)

    #     # Check if the agent is within the target zone
    #     success = self.robot.state.is_at_target()
    #     terminated = success or not self.robot.state.is_alive() or not self.robot.state.has_energy()

    #     observation = self._get_obs()
    #     info = self._get_info()

    #     distance_var = info["distance"] - prev_info["distance"]
    #     energy_var = info["energy"] - prev_info["distance"]
    #     angle_var = info["angle_to_target"] - prev_info["angle_to_target"]

    #     if self.render_mode == "human":
    #         self._render_frame(action)

    #     # reward = self._get_reward()

    #     return observation, reward, terminated, False, info

    def _consume_energy(self, action: int | np.ndarray, terrain: Terrain) -> None:
        if isinstance(action, np.ndarray):
            for single_action in action:
                self._consume_energy(single_action, terrain)
        elif action in [Actions.FORWARD.value, Actions.BACKWARD.value]:
            movement_cost = terrain.get_movement_cost()
            self.robot.state.consume_energy(movement_cost)
        elif action in [Actions.ROTATE_LEFT.value, Actions.ROTATE_RIGHT.value]:
            self.robot.state.consume_energy(0.15)
        elif action == Actions.IDLE.value:
            self.robot.state.consume_energy(0.1)

    def step(self, action: int | np.ndarray) -> tuple:  # noqa: D102
        position: tuple[float, float] = self.robot.get_position()
        terrain: Terrain = self.game_map.get_terrain(position[0], position[1])

        # Store previous information
        prev_info = self._get_info()

        self.robot.move(action)
        self._consume_energy(action, terrain)

        # Get current info
        info = self._get_info()

        distance_var = info["distance"] - prev_info["distance"]
        angle_var = info["angle_to_target"] - prev_info["angle_to_target"]
        energy_var = info["energy"] - prev_info["distance"]
        health_var = info["health"] - prev_info["health"]

        # Get observation, rewar
        observation = self._get_obs()
        reward = self._get_reward(
            distance_var=distance_var, angle_var=angle_var, energy_var=energy_var, health_var=health_var
        )
        terminated = self._get_terminates()
        truncated = self._get_truncates()

        # Update visualization
        # distance_to_target = np.linalg.norm(np.asarray(self.robot.get_position()) - self._target_zone_center, ord=1)
        if self.visualization:
            self.visualization.update(reward, info["distance"], np.int64(action), self.visualization.random_flag)

        if self.render_mode == "human":
            self._render_frame(action)

        return observation, reward, terminated, truncated, info

    # def render(self) -> None:
    #     if self.render_mode == "rgb_array":
    #         return self._render_frame()
    #     return None

    def _render_frame(
        self, action: int | np.ndarray, *, show_vision_map: bool = False, draw_fps_glider: bool = False
    ) -> None:
        """Render the PyGame window."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height), pygame.NOFRAME)
            pygame.display.set_caption("Terrain World")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))

        # Render the game map
        # Removed zoom zoomed_cell_size: int = int(self.cell_size * self.zoom_factor)
        self.game_map.render(canvas, cell_size=self.cell_size)

        # Render the robot
        self.robot.render_robot(canvas, cell_size=self.cell_size)

        self.robot.render_status_bars(canvas, 10, 10, 200, 20)
        self.robot.render_vision_rays(canvas, self.cell_size, boundaries=(self.width, self.height))

        # Draw the target zone
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                self._target_zone[0] * self.cell_size * self.zoom_factor,
                (
                    self.target_zone_size * self.cell_size * self.zoom_factor,
                    self.target_zone_size * self.cell_size * self.zoom_factor,
                ),
            ),
            width=3,
        )

        # Draw the FPS slider
        if draw_fps_glider:
            self._draw_fps_slider(canvas)

        # Draw debug information
        self._draw_debug_info(canvas, action)

        if show_vision_map:
            # Get the vision map and render it in the right panel
            vision_map: np.ndarray = self.robot.get_vision_map()
            # Removed zoom: zoomed_coordinates: int = int(self.game_map.width * self.cell_size * self.zoom_factor)
            self.render_vision_map(vision_map, canvas, self.game_map.width * self.cell_size, 0)

        if self.render_mode == "human":
            if self.window is not None:
                self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            if self.clock is not None:
                self.clock.tick(self.fps)

    def render_vision_map(self, vision_map: np.ndarray, canvas: pygame.Surface, x: int, y: int) -> None:
        """Render the vision map on the screen."""
        cell_size = self.cell_size  # * self.zoom_factor
        for i, row in enumerate(vision_map):
            for j, terrain in enumerate(row):
                if terrain is not None:
                    color = terrain.get_color()
                    pygame.draw.rect(canvas, color, (x + j * cell_size, y + i * cell_size, cell_size, cell_size))

    def update_random_flag(self, *, random_flag: np.bool_) -> None:
        """Update the random flag in the visualization."""
        if self.visualization:
            self.visualization.random_flag = random_flag

    def _draw_fps_slider(self, canvas: pygame.Surface, max_fps: int = 200) -> None:
        """Draw the FPS slider on the screen."""
        # Draw the FPS slider
        font = pygame.font.Font(None, 24)
        label = font.render(f"FPS: {self.fps}", True, (255, 255, 255))  # noqa: FBT003
        canvas.blit(label, (self.window_width - self.panel_width + 10, 10))

        slider_rect = pygame.Rect(self.window_width - self.panel_width + 10, 40, 200, 20)
        pygame.draw.rect(canvas, (255, 255, 255), slider_rect, 2)

        # Draw the slider handle
        handle_x = slider_rect.x + int((self.fps / max_fps) * slider_rect.width)
        handle_rect = pygame.Rect(handle_x - 5, slider_rect.y - 5, 10, slider_rect.height + 10)
        pygame.draw.rect(canvas, (255, 0, 0), handle_rect)

        # Handle slider interaction
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0] and slider_rect.collidepoint(mouse_x, mouse_y):
            self.fps = int(((mouse_x - slider_rect.x) / slider_rect.width) * max_fps)
            self.fps = max(1, min(self.fps, max_fps))

    def _draw_debug_info(self, canvas: pygame.Surface, action: int | np.ndarray) -> None:
        font = pygame.font.Font(None, 24)
        position = self.robot.get_position()
        terrain = self.game_map.get_terrain(position[0], position[1])
        distance_to_target = np.linalg.norm(np.asarray(position) - self._target_zone_center, ord=1)

        debug_info = [
            f"Action: {action}",
            f"Position: ({position[0]:.2f}, {position[1]:.2f})",
            f"Angle: {self.robot.state.angle:.2f}",
            f"Target: ({self.robot.state.target_x}, {self.robot.state.target_y})",
            f"Current Tile: {terrain.__class__.__name__}",
            f"Distance to Target: {distance_to_target:.2f}",
            f"Health: {self.robot.get_health():.2f}",
            f"Energy: {self.robot.get_energy():.2f}",
            f"Ammunition: {self.robot.get_ammunition():.2f}",
        ]

        for i, info in enumerate(debug_info):
            label = font.render(info, True, (255, 255, 255))  # noqa: FBT003
            canvas.blit(label, (self.width * self.cell_size * 2 - 100, 80 + i * 30))

    def handle_keys(self) -> tuple:
        """Handle the PyGame key events."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            observation, reward, terminated, truncated, info = self.step(Actions.FORWARD.value)
        elif keys[pygame.K_DOWN]:
            observation, reward, terminated, truncated, info = self.step(Actions.BACKWARD.value)
        elif keys[pygame.K_LEFT]:
            observation, reward, terminated, truncated, info = self.step(Actions.ROTATE_LEFT.value)
        elif keys[pygame.K_RIGHT]:
            observation, reward, terminated, truncated, info = self.step(Actions.ROTATE_RIGHT.value)
        else:
            observation, reward, terminated, truncated, info = self.step(Actions.IDLE.value)
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the PyGame window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
