from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pyglet
from gymnasium import spaces

from mili_env.envs.classes.robot_base import Actions, RobotAttributes, RobotBase, RobotConstraints, RobotPosition
from mili_env.envs.classes.terrain import GameMap, Terrain
from mili_env.envs.timing_utils import timing_log, timing_start, timing_stop

PI_OVER_4: float = np.pi / 4
PI_OVER_8: float = np.pi / 8


class TerrainWorldEnv(gym.Env):
    """Custom environment for the terrain world."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}  # noqa: RUF012

    def __init__(
        self,
        render_mode: str | None = None,
        target_zone_size: int = 20,
        terrain_filename: str = "plain_terrain.csv"
    ) -> None:
        """Initialize the environment."""
        self.max_window_size: int = 512
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

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(0, max(self.width, self.height) - 1, shape=(2,), dtype=np.float64),
                "target_position": spaces.Box(0, max(self.width, self.height) - 1, shape=(2,), dtype=np.int64),
                # Scalars are represented as 1-D arrays of length 1 so downstream
                # libraries (stable-baselines3 / PyTorch) receive tensors with
                # an expected dimension (no 0-d tensors).
                "distance": spaces.Box(0.0, np.sqrt(self.width**2 + self.height**2), shape=(1,), dtype=np.float64),
                "direction": spaces.Box(0.0, 2 * np.pi, shape=(1,), dtype=np.float64),
                "target_direction": spaces.Box(0.0, 2 * np.pi, shape=(1,), dtype=np.float64),
                "energy": spaces.Box(0.0, self.robot.max_energy, shape=(1,), dtype=np.float64),
            }
        )

        # Populate the action space with the possible actions
        self.action_space = spaces.Discrete(len(Actions))

        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.batch = None
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
            "distance": np.array(
                [np.linalg.norm(self.robot.get_position() - self._target_zone_center, ord=1)], dtype=np.float64
            ),
            "direction": np.array([direction], dtype=np.float64),
            "target_direction": np.array([target_direction], dtype=np.float64),
            "energy": np.array([self.robot.get_energy()], dtype=np.float64),
        }

    def _normalize_observation(self, observation: dict) -> dict:
        """Normalize the observation values to be between 0 and 1."""
        for obs_key, value in observation.items():
            if obs_key in ["position", "target_position"]:
                # Normalize position and target position (remain 1-D or 2-D arrays)
                observation[obs_key] = np.asarray(value / np.array([self.width, self.height]))
            elif obs_key == "distance":
                # Normalize distance -> ensure result is 1-D array
                observation[obs_key] = np.atleast_1d(np.asarray(value / np.sqrt(self.width**2 + self.height**2)))
            elif obs_key in ["direction", "target_direction"]:
                # Normalize direction -> ensure result is 1-D array
                observation[obs_key] = np.atleast_1d(np.asarray(value / (2 * np.pi)))
            elif obs_key == "energy":
                # Normalize energy -> ensure result is 1-D array
                observation[obs_key] = np.atleast_1d(np.asarray(value / self.robot.max_energy))
            else:
                msg = f"Unknown observation key: {obs_key}"
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
    def step_reward(distance_var: float, angle_var: float, energy_var: float, health_var: float) -> float: # noqa: ARG004
        """Calculate the by-step reward based on distance/angle-to-target, energy, and health variations."""
        # Better to increase: energy, health
        # Better to decrease: distance/angle-to-target
        return -distance_var * 0.1 # - (angle_var / np.pi)

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
        timing_start("reset")
        super().reset(seed=seed, options=options)

        self.create_robot()

        timing_stop("reset")
        timing_log()
        return self._get_obs(), self._get_info()

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
        timing_start("step")
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

        timing_stop("step")
        timing_log()
        return observation, reward, terminated, truncated, info

    # ----------------- Rendering -----------------
    def _draw_basic_map(self) -> None:
        """Draw the basic map without the robot."""
        if self.window is not None:
            self.window.clear()
        pix = int(self.cell_size * self.zoom_factor)
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
        if self.robot is not None and hasattr(self.robot, "position"):
            posx, posy = self.robot.get_position()
            rx = int(posx * pix + pix // 2)
            ry = int(posy * pix + pix // 2)
            r = int(pix * 0.4)
            n = 20
            verts = []
            for k in range(n):
                theta = 2.0 * np.pi * k / n
                verts += [int(rx + r * np.cos(theta)), int(ry + r * np.sin(theta))]
            pyglet.graphics.draw(n, pyglet.gl.GL_TRIANGLES, vertices=("v2i", tuple(verts)))

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

    def _draw_debug_info(self, batch: pyglet.graphics.Batch, action: int | np.ndarray) -> None:
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
            pyglet.text.Label(
                info, font_size=14,
                x=self.width * self.cell_size * 2 - 100,
                y=self.window_height - (80 + i * 30),
                color=(255, 255, 255, 255), batch=batch
            )
