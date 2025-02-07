from __future__ import annotations

import random

import numpy as np
import pygame

from mili_env.envs.classes.robot_base import Actions
from mili_env.envs.terrain_world import TerrainWorldEnv


class DebugAgentEnv(TerrainWorldEnv):
    """Debug Agent environment for manual control and debugging."""
    def __init__(self, render_mode: str | None = None, target_zone_size: int = 20) -> None:
        """Initialize the Debug Agent environment."""
        super().__init__(render_mode, target_zone_size)
        self.zoom_factor = 1.0

    def step(self, action: Actions) -> tuple: # noqa: D102
        # Override step to handle manual control
        position = self.robot.get_position()
        terrain = self.game_map.get_terrain(position[0], position[1])

        self.robot.move(action)

        if action in [Actions.FORWARD.value, Actions.BACKWARD.value]:
            movement_cost = terrain.get_movement_cost()
            self.robot.state.consume_energy(movement_cost)
        elif action in [Actions.ROTATE_LEFT.value, Actions.ROTATE_RIGHT.value]:
            self.robot.state.consume_energy(0.15)
        elif action == Actions.IDLE.value:
            self.robot.state.consume_energy(0.1)

        distance_to_target = np.linalg.norm(np.array(self.robot.get_position()) - self._target_zone_center, ord=1)

        success = self.robot.state.is_at_target()
        terminated = success or not self.robot.state.is_alive() or not self.robot.state.has_energy()

        reward = -distance_to_target * 0.1
        if success:
            reward += 20
        elif not self.robot.state.is_alive() or not self.robot.state.has_energy():
            reward -= 10

        self.visualization.update(reward, distance_to_target, action.value)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self) -> None: # noqa: D102
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self) -> None:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Debug Agent")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))

        zoomed_cell_size = int(self.cell_size * self.zoom_factor)
        self.game_map.render(canvas, cell_size=zoomed_cell_size)

        self.robot.render_robot(canvas, cell_size=zoomed_cell_size)

        self.robot.render_status_bars(canvas, 10, 10, 200, 20)
        self.robot.render_vision_rays(canvas, zoomed_cell_size)

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

        self._draw_fps_slider(canvas)
        self._draw_debug_info(canvas)

        vision_map: np.ndarray = self.robot.get_vision_map()
        self.render_vision_map(vision_map, canvas, int(self.game_map.width * self.cell_size * self.zoom_factor), 70)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.fps)

    def _draw_debug_info(self, canvas: pygame.Surface) -> None:
        font = pygame.font.Font(None, 24)
        position = self.robot.get_position()
        terrain = self.game_map.get_terrain(position[0], position[1])
        distance_to_target = np.linalg.norm(np.array(position) - self._target_zone_center, ord=1)

        debug_info = [
            f"Position: {position}",
            f"Current Tile: {terrain.__class__.__name__}",
            f"Distance to Target: {distance_to_target:.2f}",
            f"Angle: {self.robot.state.angle:.2f}",
        ]

        for i, info in enumerate(debug_info):
            label = font.render(info, antialias=True, color=(255, 255, 255))
            canvas.blit(label, (self.window_width - self.panel_width + 10, 100 + i * 30))

    def handle_keys(self) -> None: # noqa: D102
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

    def run(self) -> None:
        """Run the environment in a loop."""
        pygame.init()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE:
                        self.step(random.choice(list(Actions)))

            self.handle_keys()
            self.render()

        pygame.quit()


if __name__ == "__main__":
    env = DebugAgentEnv(render_mode="human")
    env.run()
