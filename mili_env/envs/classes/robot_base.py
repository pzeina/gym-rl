from dataclasses import dataclass
import pygame

import numpy as np
import torch
from enum import Enum

from mili_env.envs.classes.terrain import GameMap, Terrain, EmptyTerrain, ObstacleTerrain


class Actions(Enum):
    """Enum class to define robot actions."""

    IDLE = 0
    FORWARD = 1
    BACKWARD = 2
    ROTATE_LEFT = 3
    ROTATE_RIGHT = 4

    @classmethod
    def count(cls):
        return len(cls.__members__)

@dataclass
class RobotAttributes:
    """Data class to store robot attributes."""

    health: float
    energy: float
    ammunition: float

    health_efficiency: float = 0.1
    energy_efficiency: float = 0.1
    speed_efficiency: float = 10.0
    ammunition_efficiency: float = 0.1

@dataclass
class RobotPosition:
    """Data class to store robot position."""

    x: float
    y: float
    angle: float
    target_x: int
    target_y: int
    target_width: int
    target_height: int

@dataclass
class RobotConstraints:
    """Data class to store robot constraint parameters."""

    vision_range: float = 10.0
    communication_range: float = 10.0
    max_speed_forward: float = 1.0
    max_speed_backward: float = -0.2
    max_angular_speed: float = np.pi / 4
        
    max_health: float = 100.0
    max_energy: float = 100.0
    max_ammunition: float = 100.0


class RobotState:
    """Class to manage the state of the robot."""

    def __init__(self, position: RobotPosition, attributes: RobotAttributes) -> None:
        """Initialize the robot's state with position, target, health, and energy."""
        self.x = position.x
        self.y = position.y
        self.angle = position.angle
        self.target_x = position.target_x
        self.target_y = position.target_y
        self.target_width = position.target_width
        self.target_height = position.target_height
        self.attributes = attributes

    def update_position(self, new_x: float, new_y: float, new_angle: float) -> None:
        """Update the robot's position."""
        self.x = new_x
        self.y = new_y
        self.angle = new_angle

    def update_target(self, new_target_x: int, new_target_y: int, new_target_width: int, new_target_height: int) -> None:
        """Update the robot's target position."""
        self.target_x = new_target_x
        self.target_y = new_target_y
        self.target_width = new_target_width
        self.target_height = new_target_height

    def get_energy(self) -> float:
        """Get the robot's energy."""
        return self.attributes.energy
    
    def get_health(self) -> float:
        """Get the robot's health."""
        return self.attributes.health
    
    def get_ammunition(self) -> float:
        """Get the robot's ammunition."""
        return self.attributes.ammunition

    def consume_energy(self, amount: float) -> None:
        """Reduce the robot's energy by a specified amount."""
        self.attributes.energy = max(0, self.attributes.energy - amount / self.attributes.energy_efficiency)

    def restore_energy(self, amount: float) -> None:
        """Restore the robot's energy by a specified amount."""
        self.attributes.energy = min(100, self.attributes.energy + amount)

    def take_damage(self, amount: float) -> None:
        """Reduce the robot's health by a specified amount."""
        self.attributes.health = max(0, self.attributes.health - amount / self.attributes.health_efficiency)

    def is_alive(self) -> bool:
        """Check if the robot is still alive."""
        return self.attributes.health > 0
    
    def has_energy(self) -> bool:
        """Check if the robot has energy remaining."""
        return self.attributes.energy > 0

    def is_at_target(self) -> bool:
        """Check if the robot has reached its target zone."""
        return (self.target_x - self.target_width // 2 <= self.x < self.target_x + self.target_width // 2 and
                self.target_y - self.target_height // 2 <= self.y < self.target_y + self.target_height)


    def consume_ammunition(self, amount: float) -> None:
        """Reduce the robot's ammunition by a specified amount."""
        self.attributes.ammunition = max(0, self.attributes.ammunition - amount / self.attributes.ammunition_efficiency)

    def restore_ammunition(self, amount: float) -> None:
        """Restore the robot's ammunition by a specified amount."""
        self.attributes.ammunition = min(100, self.attributes.ammunition + amount)

    def get_state_info(self) -> dict:
        """Retrieve the current state information of the robot."""
        return {
            "position": (self.x, self.y),
            "target": (self.target_x, self.target_y),
            "health": self.attributes.health,
            "energy": self.attributes.energy,
        }

    def encode(self) -> torch.Tensor:
        """Encode the robot stats properties into a tensor."""
        return torch.tensor(
            [
                self.x,
                self.y,
                self.angle,
                self.target_x,
                self.target_y,
                self.attributes.health,
                self.attributes.energy,
                self.attributes.ammunition,
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
    
    def get_size(self) -> tuple:
        """Get the size of the state tensor."""
        return self.encode().shape


class RobotBase:
    """Base class for robots."""

    def __init__(
        self,
        position: RobotPosition,
        attributes: RobotAttributes,
        game_map: GameMap,
        constraints: RobotConstraints,
    ) -> None:
        """Initialize the robot with its position and attributes."""
        self.state: RobotState = RobotState(position, attributes)
        self.vision_map: np.ndarray = np.full(
            (2 * int(constraints.vision_range) + 1, 2 * int(constraints.vision_range) + 1), EmptyTerrain(), dtype=Terrain
        )
        self.game_map: GameMap = game_map
        self.vision_range: float = constraints.vision_range
        self.communication_range: float = constraints.communication_range
        self.max_speed_forward: float = constraints.max_speed_forward
        self.max_speed_backward: float = constraints.max_speed_backward
        self.max_angular_speed: float = constraints.max_angular_speed
        self.max_health: float = constraints.max_health
        self.max_energy: float = constraints.max_energy
        self.max_ammunition: float = constraints.max_ammunition
        self.rays = self.compute_vision_rays()


    def move(self, action: Actions) -> float:
        """Move the robot based on the chosen action."""
        # Define actions: 0 = move forward, 1 = move backward, 2 = turn left, 3 = turn right
        dx, dy = 0, 0
        new_angle = self.state.angle
        if action == Actions.IDLE:
            dx, dy = 0, 0
        elif action == Actions.FORWARD:
            dx = self.max_speed_forward * np.cos(self.state.angle)
            dy = self.max_speed_forward * np.sin(self.state.angle)
        elif action == Actions.BACKWARD:
            dx = -self.max_speed_backward * np.cos(self.state.angle)
            dy = -self.max_speed_backward * np.sin(self.state.angle)
        elif action == Actions.ROTATE_LEFT:
            new_angle -= self.max_angular_speed
        elif action == Actions.ROTATE_RIGHT:
            new_angle += self.max_angular_speed
        else:
            error_message = "Invalid action"
            raise ValueError(error_message)
        
        # Keep the angle within the range [0, 2*pi]
        new_angle = (new_angle + 2 * np.pi) % (2 * np.pi)

        movement_speed = self.game_map.get_terrain(self.state.x, self.state.y).get_properties().movement_speed
        new_x = self.state.x + dx * movement_speed * self.state.attributes.speed_efficiency
        new_y = self.state.y + dy * movement_speed * self.state.attributes.speed_efficiency

        new_x = max(0, min(self.game_map.width - 1, new_x))
        new_y = max(0, min(self.game_map.height - 1, new_y))
        
        self.state.update_position(new_x, new_y, new_angle)

        self.rays = self.compute_vision_rays()
        self.update_vision_map()

        return np.sqrt((self.state.target_x - self.state.x) ** 2 + (self.state.target_y - self.state.y) ** 2)
    
    def get_health(self) -> float:
        """Get the robot's health."""
        return self.state.get_health()
    
    def get_energy(self) -> float:
        """Get the robot's energy."""
        return self.state.get_energy()
    
    def get_ammunition(self) -> float:
        """Get the robot's ammunition."""
        return self.state.get_ammunition()

    def get_vision_map(self) -> np.ndarray[Terrain]:
        """Get the robot's vision map."""
        return self.vision_map

    def get_position(self) -> tuple[float, float]:
        """Get the robot's current position."""
        return self.state.x, self.state.y

    def get_direction(self) -> float:
        """Get the robot's current direction."""
        return self.state.angle

    def get_target(self) -> tuple[int, int]:
        """Get the robot's target position."""
        return self.state.target_x, self.state.target_y

    def get_state(self) -> RobotState:
        """Get the robot's current state."""
        return self.state
    
    def get_game_map(self) -> GameMap:
        """Get the game map."""
        return self.game_map
    
    def get_game_map_size(self) -> tuple[int, int, int]:
        """Get the size of the game map."""
        terrain: Terrain = self.game_map.get_terrain(0, 0)
        return self.game_map.width, self.game_map.height, terrain.get_encode_size()

    def get_vision_rays(self) -> list:
        """Get the robot's current state."""
        return self.rays

    def compute_vision_rays(self) -> list:
        """Compute vision rays with terrain-dependent length and direction."""
        rays: list = []
        x, y, direction = self.state.x, self.state.y, self.state.angle
        # Increase the density of rays in the direction the robot is facing
        angles = np.concatenate([
            np.linspace(direction - np.pi / 4, direction + np.pi / 4, 32),  # Higher density in front
            np.linspace(direction + np.pi / 4, direction + 3 * np.pi / 4, 16),  # Medium density on sides
            np.linspace(direction - 3 * np.pi / 4, direction - np.pi / 4, 16)  # Medium density on sides
        ])

        for angle in angles:
            angle_diff = np.abs(
                (angle - direction + np.pi) % (2 * np.pi) - np.pi
            )  # Difference between ray angle and robot's direction
            direction_factor = 1 - angle_diff / np.pi  # Direction factor based on angle difference
            length: int = int(np.floor(self.vision_range * direction_factor))
            reduction: float = 0.0

            for step in range(1, length + 1):
                new_x = int(x + step * np.cos(angle))
                new_y = int(y + step * np.sin(angle))
                if 0 <= new_x < self.game_map.width and 0 <= new_y < self.game_map.height:
                    terrain_visibility = self.game_map.get_terrain(new_x, new_y).get_properties().visibility
                    if terrain_visibility < 1.0:
                        reduction += 1 - terrain_visibility
                        if reduction >= 1.8:
                            length = step
                            break
                else:
                    length = step
                    break
            rays.append((angle, length))
        return rays

    def communicate(self, other_robot: "RobotBase") -> bool:
        """Check if another robot is within communication range."""
        x, y = self.state.x, self.state.y
        other_robot_x, other_robot_y = other_robot.get_position()
        distance = np.sqrt((x - other_robot_x) ** 2 + (y - other_robot_y) ** 2)
        return distance <= self.communication_range

    def update_vision_map(self) -> None:
        """Update the vision map with the terrain in the robot's vision range (using rays)."""
        x, y = self.state.x, self.state.y
        vision_range = int(self.vision_range)
        vision_map_size = 2 * vision_range + 1

        # Initialize the vision map with None
        self.vision_map = np.full((vision_map_size, vision_map_size), EmptyTerrain(), dtype=Terrain)

        for ray in self.rays:
            angle, length = ray
            for step in range(1, length + 1):
                new_x = int(x + step * np.cos(angle))
                new_y = int(y + step * np.sin(angle))
                map_x = new_x - int(x) + vision_range
                map_y = new_y - int(y) + vision_range
                if 0 <= new_x < self.game_map.width and 0 <= new_y < self.game_map.height:
                    if 0 <= map_x < vision_map_size and 0 <= map_y < vision_map_size:
                        self.vision_map[map_y, map_x] = self.game_map.get_terrain(new_x, new_y)
                else:
                    self.vision_map[map_y, map_x] = ObstacleTerrain()
                    break

    def encode_state(self) -> torch.Tensor:
        """Encode the state properties into a tensor."""
        return self.state.encode()
    
    def get_state_size(self) -> tuple:
        """Get the size of the state tensor."""
        return self.state.get_size()


    def encode_vision_map(self) -> torch.Tensor:
        """Encode the vision map into a tensor."""
        encoded_map = np.array(
            [terrain.encode_properties() for row in self.vision_map for terrain in row if terrain is not None],
            dtype=np.float32
        )
        return torch.tensor(encoded_map, dtype=torch.float32).unsqueeze(0)
    
    def render_status_bars(
        self, screen: pygame.Surface, x: int, y: int, width: int, height: int
    ) -> None:
        """Render health, energy, and ammunition bars on the screen."""
        font = pygame.font.SysFont(None, 24)

        # Health bar
        health_ratio = self.state.attributes.health / self.max_health
        pygame.draw.rect(screen, (255, 0, 0), (x, y, width * health_ratio, height))
        pygame.draw.rect(screen, (255, 255, 255), (x, y, width, height), 2)
        health_label = font.render("Health", True, (255, 255, 255))
        screen.blit(health_label, (x, y))

        # Energy bar
        energy_ratio = self.state.attributes.energy / self.max_energy
        pygame.draw.rect(screen, (0, 255, 0), (x, y + height + 5, width * energy_ratio, height))
        pygame.draw.rect(screen, (255, 255, 255), (x, y + height + 5, width, height), 2)
        energy_label = font.render("Energy", True, (255, 255, 255))
        screen.blit(energy_label, (x, y + height + 5))

        # Ammunition bar
        ammunition_ratio = self.state.attributes.ammunition / self.max_ammunition
        pygame.draw.rect(screen, (0, 0, 255), (x, y + 2 * (height + 5), width * ammunition_ratio, height))
        pygame.draw.rect(screen, (255, 255, 255), (x, y + 2 * (height + 5), width, height), 2)
        ammunition_label = font.render("Ammunition", True, (255, 255, 255))
        screen.blit(ammunition_label, (x, y + 2 * (height + 5)))


    def render_vision_rays(self, screen: pygame.Surface, cell_size: int) -> None:
        """Render vision rays on the screen."""
        x = self.state.x * cell_size + cell_size // 2
        y = self.state.y * cell_size + cell_size // 2

        for angle, length in self.get_vision_rays():
            end_x = x + int(length * np.cos(angle))
            end_y = y + int(length * np.sin(angle))
            start_pos = (x, y)
            end_pos = (end_x, end_y)
            pygame.draw.line(screen, (255, 0, 0), start_pos, end_pos, 1)


    def render_robot(self, screen: pygame.Surface, cell_size: int) -> None:
        """Render the robot on the screen as an arrow-like object."""
        x = self.state.x * cell_size + cell_size // 2
        y = self.state.y * cell_size + cell_size // 2
        angle = self.state.angle

        # Define the points of the triangle (arrow)
        point1 = (x + cell_size * np.cos(angle), y + cell_size * np.sin(angle))
        point2 = (x + cell_size * np.cos(angle + 2 * np.pi / 3), y + cell_size * np.sin(angle + 2 * np.pi / 3))
        point3 = (x + 0.6 * cell_size * np.cos(angle + np.pi), y + 0.6 * cell_size * np.sin(angle + np.pi))
        point4 = (x + cell_size * np.cos(angle - 2 * np.pi / 3), y + cell_size * np.sin(angle - 2 * np.pi / 3))

        pygame.draw.polygon(screen, (255, 0, 0), [point1, point2, point3, point4])

    def render_target(self, screen: pygame.Surface, cell_size: int) -> None:
        """Render the target zone on the screen."""
        target_rect = pygame.Rect(
            self.state.target_x * cell_size,
            self.state.target_y * cell_size,
            self.state.target_width * cell_size,
            self.state.target_height * cell_size,
        )
        pygame.draw.rect(screen, (255, 255, 0), target_rect, 2)