from __future__ import annotations

import random

import pygame

from mili_env.envs.classes.robot_base import Actions
from mili_env.envs.terrain_world import TerrainWorldEnv


class DebugAgentEnv(TerrainWorldEnv):
    """Debug Agent environment for manual control and debugging."""

    def __init__(self, render_mode: str | None = None, target_zone_size: int = 20) -> None:
        """Initialize the Debug Agent environment."""
        super().__init__(render_mode, target_zone_size)
        self.zoom_factor = 1.0

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
                        self.step(random.choice(list(Actions)).value)

            self.handle_keys()
            self.render()

        pygame.quit()


if __name__ == "__main__":
    env = DebugAgentEnv(render_mode="human")
    env.run()
