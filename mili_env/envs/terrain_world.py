import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from mili_env.envs.classes.terrain import Terrain, GameMap
from mili_env.envs.classes.robot_base import RobotPosition, RobotAttributes, RobotConstraints, RobotBase, Actions
from pathlib import Path
from mili_env.envs.visualization import AgentEnvInteractionVisualization as Visualization


class TerrainWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, target_zone_size: int = 20) -> None:
        self.max_window_size: int = 512  # The maximum size of the PyGame window
        self.panel_width: int = 256  # Width of the right panel
        self.target_zone_size: int = target_zone_size  # Size of the target zone

        # Load the terrain map
        self.game_map = GameMap.load_from_csv(Path(__file__).parent / "data" / "terrain.csv")
        self.width: int = self.game_map.width
        self.height: int = self.game_map.height

        # Calculate the window size to ensure square pixels
        self.cell_size = min(self.max_window_size // self.width, self.max_window_size // self.height)
        self.window_width = self.cell_size * self.width + self.panel_width
        self.window_height = self.cell_size * self.height

        self.create_robot()

        # Initialize visualization
        self.visualization = Visualization(self.window_width, self.window_height, self.panel_width)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(0, max(self.width, self.height) - 1, shape=(2,), dtype=float),
                "direction": spaces.Box(0, 6.29, shape=(1,), dtype=float),
                "target": spaces.Box(0, max(self.width, self.height) - 1, shape=(2,), dtype=int),
            }
        )

        # Populate the action space with the possible actions
        self.action_space = spaces.Discrete(Actions.count())

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.fps = self.metadata["render_fps"]
        self.zoom_factor = 1.0

    def create_robot(self) -> None:
        """Create the robot object."""

        # Define the target zone
        self._target_zone_center = self.np_random.integers(0, [self.width, self.height], size=2, dtype=int)
        self._target_zone = np.array([
            self._target_zone_center - self.target_zone_size // 2,
            self._target_zone_center + self.target_zone_size // 2
        ])
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
            target_height=self.target_zone_size
        )
        initial_attributes = RobotAttributes(
            health=100.0, 
            energy=100.0, 
            ammunition=100.0,
            health_efficiency=1.0,
            energy_efficiency=10.0,
            speed_efficiency=5.0,
            ammunition_efficiency=1.0
        )
        constraints = RobotConstraints(    
            vision_range = 30.0,
            communication_range = 30.0,
            max_speed_forward = 1.0,
            max_speed_backward = -0.2,
            max_angular_speed = np.pi / 4,
            max_health=100.0,
            max_energy=100.0,
            max_ammunition=100.0
        )
        self.robot = RobotBase(
            position=initial_position, 
            attributes=initial_attributes, 
            game_map=self.game_map, 
            constraints=constraints
        )


    def _get_obs(self) -> dict:
        return {
            "position": np.array(self.robot.get_position(), dtype=float),
            "direction": np.array(self.robot.get_direction(), dtype=float),
            "target": np.array(self.robot.get_target(), dtype=int)
        }

    def _get_info(self) -> dict:
        return {
            "distance": np.linalg.norm(
                np.array(self.robot.get_position()) - self._target_zone_center, ord=1
            ),
            "health": self.robot.get_health(),
            "energy": self.robot.get_energy(),
            "ammunition": self.robot.get_ammunition()
        }

    def reset(self, seed=None, options=None) -> dict:

        super().reset(seed=seed)

        self.create_robot()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action: Actions) -> tuple:

        # Check if the new location is traversable
        position = self.robot.get_position()
        terrain: Terrain = self.game_map.get_terrain(position[0], position[1])

        self.robot.move(action)

        if action in [Actions.FORWARD, Actions.BACKWARD]:
            # if terrain.get_properties().traversable:
            movement_cost = terrain.get_movement_cost()
            self.robot.state.consume_energy(movement_cost)
        elif action in [Actions.ROTATE_LEFT, Actions.ROTATE_RIGHT]:
            # Minimal energy consumption when rotating
            self.robot.state.consume_energy(0.15)
        elif action == Actions.IDLE:
            # Minimal energy consumption when staying with no movement
            self.robot.state.consume_energy(0.1)

        # Calculate the distance to the target
        distance_to_target = np.linalg.norm(
            np.array(self.robot.get_position()) - self._target_zone_center, ord=1
        )

        # Check if the agent is within the target zone
        success = self.robot.state.is_at_target()
        terminated = success or not self.robot.state.is_alive() or not self.robot.state.has_energy()

        # Reward based on the distance to the target
        reward = self.robot.get_energy() - distance_to_target
        if success:
            reward += self.robot.get_energy() + 0.25 * self.robot.get_ammunition() + 0.25 * self.robot.get_health()
        elif not self.robot.state.is_alive():
            reward -= 100
        elif not self.robot.state.has_energy():
            reward -= 100

        # Update visualization
        self.visualization.update(reward, distance_to_target, action.value)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> None:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height), pygame.NOFRAME)
            pygame.display.set_caption("Terrain World")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))

        # Render the game map
        self.game_map.render(canvas, cell_size=self.cell_size * self.zoom_factor)

        # Render the robot
        self.robot.render_robot(canvas, cell_size=self.cell_size * self.zoom_factor)

        self.robot.render_status_bars(canvas, 10, 10, 200, 20)
        self.robot.render_vision_rays(canvas, self.cell_size * self.zoom_factor)

        # Draw the target zone
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                self._target_zone[0] * self.cell_size * self.zoom_factor,
                (self.target_zone_size * self.cell_size * self.zoom_factor, self.target_zone_size * self.cell_size * self.zoom_factor),
            ),
            width=3
        )

        # Draw the FPS slider
        self._draw_fps_slider(canvas)

        # Draw debug information
        self._draw_debug_info(canvas)

        # Get the vision map and render it in the right panel
        vision_map: np.ndarray = self.robot.get_vision_map()
        self.render_vision_map(vision_map, canvas, self.game_map.width * self.cell_size * self.zoom_factor, 70)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.fps)

    def render_vision_map(
        self, 
        vision_map: np.ndarray,
        canvas: pygame.Surface, 
        x: int, 
        y: int
    ) -> None:
        """Render the vision map on the screen."""
        cell_size = self.cell_size * self.zoom_factor
        for i, row in enumerate(vision_map):
            for j, terrain in enumerate(row):
                if terrain is not None:
                    color = terrain.get_color()
                    pygame.draw.rect(canvas, color, (x + j * cell_size, y + i * cell_size, cell_size, cell_size))


    def _draw_fps_slider(self, canvas: pygame.Surface, max_fps: int = 200) -> None:
        # Draw the FPS slider
        font = pygame.font.Font(None, 24)
        label = font.render(f"FPS: {self.fps}", True, (255, 255, 255))
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

    def _draw_debug_info(self, canvas: pygame.Surface) -> None:
        font = pygame.font.Font(None, 24)
        position = self.robot.get_position()
        terrain = self.game_map.get_terrain(position[0], position[1])
        distance_to_target = np.linalg.norm(
            np.array(position) - self._target_zone_center, ord=1
        )

        debug_info = [
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
            label = font.render(info, True, (255, 255, 255))
            canvas.blit(label, (self.window_width - self.panel_width + 10, 200 + i * 30))

    def handle_keys(self) -> None:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.step(Actions.FORWARD)
        elif keys[pygame.K_DOWN]:
            self.step(Actions.BACKWARD)
        elif keys[pygame.K_LEFT]:
            self.step(Actions.ROTATE_LEFT)
        elif keys[pygame.K_RIGHT]:
            self.step(Actions.ROTATE_RIGHT)
        elif keys[pygame.K_z]:
            self.zoom_factor = 2.0 if self.zoom_factor == 1.0 else 1.0

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
