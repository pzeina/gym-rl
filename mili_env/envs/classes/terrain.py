import argparse
import csv
import random
from abc import ABC, abstractmethod
from collections import deque  # For flood fill
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
import torch
from noise import pnoise2  # For Perlin noise
from scipy.ndimage import gaussian_filter  # For smoothing


class TerrainType(Enum):
    """Enum to categorize different terrain types."""

    FOREST = "forest"
    WATER = "water"
    ROAD = "road"
    UNKNOWN = "unknown"
    UNREACHABLE = "unreachable"

    def encode(self) -> str:
        """Return the string representation of the terrain type."""
        return self.value


class TroopType(Enum):
    """Enum to categorize different troops."""

    ALLY = 4
    ENNEMY = 3
    NEUTRAL = 2
    UNKNOWN = 1
    EMPTY = 0

    def encode(self) -> int:
        """Return the integer representation of the troop type."""
        return self.value


@dataclass
class TerrainProperties:
    """Properties that affect gameplay/ML training."""

    type_level: int = 0
    movement_speed: float = 0.0
    visibility: float = 0.0
    traversable: bool = True
    altitude: float = 0.0
    occupancy: TroopType = TroopType.EMPTY

    def encode(self) -> torch.Tensor:
        """Encode the terrain properties into a tensor."""
        occupancy = self.occupancy.encode()
        return torch.tensor(
            [occupancy, self.movement_speed, self.visibility],
            dtype=torch.float32,
        )


class Terrain(ABC):
    """Abstract base class for all terrain types."""

    terrain_type: TerrainType
    properties: TerrainProperties

    @abstractmethod
    def get_color(self) -> tuple[int, int, int]:
        """Return RGB color for rendering."""

    def get_properties(self) -> TerrainProperties:
        """Get the properties of the terrain."""
        return self.properties

    def get_terrain_type(self) -> TerrainType:
        """Return the type of terrain."""
        return self.terrain_type

    def get_type_level(self) -> int:
        """Return the level of the terrain type."""
        return self.properties.type_level

    def get_movement_speed(self) -> float:
        """Return the movement speed of the terrain."""
        return self.properties.movement_speed

    def get_movement_cost(self) -> float:
        """Return the movement cost of the terrain."""
        return 1.0 / self.properties.movement_speed

    def to_string(self) -> str:
        """Convert terrain to string representation."""
        return f"{self.terrain_type.value},{self.properties.type_level}"

    def encode_properties(self) -> torch.Tensor:
        """Encode the terrain properties into a tensor."""
        return self.properties.encode()

    def get_encode_size(self) -> int:
        """Return the size of the encoded terrain properties."""
        return len(self.encode_properties())

    @staticmethod
    def from_string(terrain_str: str) -> Optional["Terrain"]:
        """Create terrain from string representation."""
        try:
            terrain_type, level = terrain_str.split(",")
            level = int(level)

            if terrain_type == TerrainType.FOREST.value:
                return Forest(level)
            if terrain_type == TerrainType.WATER.value:
                return Water(level)
            if terrain_type == TerrainType.ROAD.value:
                return Road(level)
        except ValueError:
            error = f"Invalid terrain string: {terrain_str}"
            raise ValueError(error) from None


class EmptyTerrain(Terrain):
    """Empty terrain."""

    def __init__(self) -> None:
        """Initialize empty terrain."""
        self.terrain_type = TerrainType.UNKNOWN
        self.properties = TerrainProperties(
            type_level=0,
            occupancy=TroopType.EMPTY,
            movement_speed=0.0,
            visibility=0.0,
            traversable=True,
        )

    def get_properties(self) -> TerrainProperties:
        """Get the properties of the terrain."""
        return self.properties

    def get_color(self) -> tuple[int, int, int]:
        """Return RGB color for rendering."""
        return (255, 255, 255)


class ObstacleTerrain(Terrain):
    """Obstacle terrain."""

    def __init__(self) -> None:
        """Initialize obstacle terrain."""
        self.terrain_type = TerrainType.UNREACHABLE
        self.properties = TerrainProperties(
            type_level=0,
            occupancy=TroopType.EMPTY,
            movement_speed=0.0,
            visibility=0.0,
            traversable=False,
        )

    def get_properties(self) -> TerrainProperties:
        """Get the properties of the terrain."""
        return self.properties

    def get_color(self) -> tuple[int, int, int]:
        """Return RGB color for rendering."""
        return (255, 0, 0)


class Forest(Terrain):
    """Forest terrain with density levels 0-9."""

    MAX_FOREST_DENSITY: int = 9

    def __init__(self, density: int) -> None:
        """Initialize forest terrain with density level."""
        self.terrain_type = TerrainType.FOREST
        if not 0 <= density <= self.MAX_FOREST_DENSITY:
            msg = "Forest density must be between 0 and 9."
            raise ValueError(msg)
        self.properties = TerrainProperties(
            type_level=density,
            occupancy=TroopType.EMPTY,
            movement_speed=1.0 - (density * 0.1),
            visibility=1.0 - (density * 0.1),
            traversable=True,
        )

    def get_color(self) -> tuple[int, int, int]:
        """Return RGB color for rendering."""
        density = self.properties.type_level
        if density == 0:
            return (150, 200, 150)
        base_green = 200 - (density * 20)
        return (0, base_green, 0)


class Water(Terrain):
    """Water terrain with depth levels 0-4."""

    MAX_WATER_DEPTH: int = 4
    MAX_TRAVERSABLE_DEPTH: int = 3

    def __init__(self, depth: int) -> None:
        """Initialize water terrain with depth level."""
        if not 0 <= depth <= self.MAX_WATER_DEPTH:
            msg = "Water depth must be between 0 and 4."
            raise ValueError(msg)
        self.terrain_type = TerrainType.WATER
        self.properties = TerrainProperties(
            type_level=depth,
            occupancy=TroopType.EMPTY,
            movement_speed=1 - (depth * 0.2),
            visibility=1.0,
            traversable=depth < self.MAX_TRAVERSABLE_DEPTH,
        )

    def get_color(self) -> tuple[int, int, int]:
        """Return RGB color for rendering."""
        base_blue = 255 - (self.properties.type_level * 40)
        return (0, 0, base_blue)


class Road(Terrain):
    """Road terrain with quality levels 0-3."""

    MAX_ROAD_QUALITY: int = 3

    def __init__(self, quality: int) -> None:
        """Initialize road terrain with quality level."""
        if not 0 <= quality <= self.MAX_ROAD_QUALITY:
            msg = f"Road quality must be between 0 and {self.MAX_ROAD_QUALITY}."
            raise ValueError(msg)
        self.density = quality
        self.terrain_type = TerrainType.ROAD
        self.properties = TerrainProperties(
            type_level=quality,
            occupancy=TroopType.EMPTY,
            movement_speed=1.0 + (self.density * 0.2),
            visibility=1.0,
            traversable=True,
        )

    def get_properties(self) -> TerrainProperties:
        """Get the properties of the terrain."""
        return self.properties

    def get_color(self) -> tuple[int, int, int]:
        """Return RGB color for rendering."""
        base_gray = 100 + (self.density * 40)
        return (base_gray, base_gray, base_gray)


class GameMap:
    """Represents the game's terrain map."""

    ALTITUDE_CHANGE_THRESHOLD: float = 0.2

    def __init__(self, width: int = 100, height: int = 100, cell_size: int = 10) -> None:
        """Initialize the game map with dimensions."""
        self.width: int = width
        self.height: int = height
        self.cell_size: int = cell_size
        self.terrain: np.ndarray = np.empty((height, width), dtype=Terrain)
        self.altitude: np.ndarray = np.zeros((height, width))  # Added altitude map

    def generate_altitude_map(self, scale: float = 50.0, octaves: float = 6) -> None:
        """Generate altitude using Perlin noise."""
        for y in range(self.height):
            for x in range(self.width):
                # Generate basic Perlin noise
                nx = x / scale
                ny = y / scale
                altitude = pnoise2(nx, ny, octaves=octaves)
                self.altitude[y, x] = altitude

        # Normalize altitude to 0-1 range
        self.altitude = (self.altitude - self.altitude.min()) / (self.altitude.max() - self.altitude.min())

        # Apply slight smoothing
        self.altitude = gaussian_filter(self.altitude, sigma=1.0)

    def generate_forest_zones(self, num_zones: int = 10, min_zone_size: int = 20, max_zone_size: int = 50) -> None:
        """Generate forest zones with coherent density."""

        def flood_fill(x: int, y: int, density: int) -> None:
            queue = deque([(x, y)])
            filled = set()

            zone_size = random.randint(min_zone_size, max_zone_size)

            while queue and len(filled) < zone_size:
                x, y = queue.popleft()
                if (x, y) in filled:
                    continue

                tile = self.terrain[y, x]
                if (0 <= x < self.width and 0 <= y < self.height and tile is None) or (
                    isinstance(tile, Forest) and tile.get_type_level() == 0
                ):
                    self.terrain[y, x] = Forest(density)
                    filled.add((x, y))

                    # Add neighbors with similar altitude
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        new_x, new_y = x + dx, y + dy
                        if (
                            0 <= new_x < self.width
                            and 0 <= new_y < self.height
                            and abs(self.altitude[y, x] - self.altitude[new_y, new_x]) < self.ALTITUDE_CHANGE_THRESHOLD
                        ):
                            queue.append((new_x, new_y))

        # Generate forest zones
        for _ in range(num_zones):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            density = random.randint(3, 9)  # Create denser forests
            flood_fill(x, y, density)

    def generate_random_map(self) -> None:
        """Generate a natural-looking terrain map."""
        # 1. Generate base altitude
        self.generate_altitude_map()

        # 2. Initialize with plains (sparse forest)
        for y in range(self.height):
            for x in range(self.width):
                self.terrain[y, x] = Forest(0)

        # 4. Generate forest zones
        self.generate_forest_zones()

        # 5. Generate roads
        self.generate_roads()

    def generate_roads(self, num_roads: int = 3) -> None:
        """Generate roads avoiding water and extreme altitude changes."""
        deviation: float = 0.5
        for _ in range(num_roads):
            if random.random() < deviation:
                x, y = 0, random.randint(10, self.height - 10)
                dx, dy = 1, 0
            else:
                x, y = random.randint(10, self.width - 10), 0
                dx, dy = 0, 1

            while 0 <= x < self.width and 0 <= y < self.height:
                # Avoid water and check for reasonable altitude change
                current_altitude = self.altitude[y, x]
                next_x, next_y = x + dx, y + dy

                if 0 <= next_x < self.width and 0 <= next_y < self.height:
                    next_altitude = self.altitude[next_y, next_x]
                    terrain = self.terrain[y, x]

                    # Only place road if:
                    # 1. Not water
                    # 2. Altitude change is not too steep
                    if (
                        not isinstance(terrain, Water)
                        and abs(next_altitude - current_altitude) < self.ALTITUDE_CHANGE_THRESHOLD
                    ):
                        self.terrain[y, x] = Road(random.randint(2, 3))

                if random.random() < self.ALTITUDE_CHANGE_THRESHOLD:
                    if dx == 0:
                        x += random.choice([-1, 1])
                    else:
                        y += random.choice([-1, 1])
                x += dx
                y += dy

    def save_to_csv(self, path: Path) -> None:
        """Save the map including altitude to a CSV file."""
        with path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write dimensions
            writer.writerow([self.width, self.height])

            # Write combined terrain and altitude data
            for y in range(self.height):
                row = []
                for x in range(self.width):
                    terrain = self.terrain[y, x]
                    altitude = self.altitude[y, x]
                    # Combine terrain and altitude data in the same cell
                    cell_data = f"{terrain.to_string() if terrain else 'none'}|{altitude}"
                    row.append(cell_data)
                writer.writerow(row)

    @classmethod
    def load_from_csv(cls, path: Path) -> "GameMap":
        """Load a map with altitude from a CSV file."""
        with path.open("r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            # Read dimensions
            width, height = map(int, next(reader))
            game_map = cls(width, height)

            # Read terrain and altitude data
            for y, row in enumerate(reader):
                for x, cell_data in enumerate(row):
                    terrain_str, altitude_str = cell_data.split("|")
                    if terrain_str != "none":
                        terrain = Terrain.from_string(terrain_str)
                        if terrain:
                            game_map.terrain[y, x] = terrain
                    game_map.altitude[y, x] = float(altitude_str)

        return game_map

    def encode(self) -> torch.Tensor:
        """Generate a tensor with terrain properties for ML training."""
        zero_terrain: Terrain = self.terrain[0, 0]
        zero_property_tensor: torch.Tensor = zero_terrain.encode_properties()
        property_length: int = zero_property_tensor.size(0)

        properties_map: torch.Tensor = torch.zeros((self.height, self.width, property_length), dtype=torch.float32)
        for y in range(self.height):
            for x in range(self.width):
                terrain: Terrain = self.terrain[y, x]
                if terrain:
                    properties_map[y, x] = terrain.encode_properties()
        return properties_map

    def get_movement_speed_map(self) -> np.ndarray:
        """Generate a movement speed matrix for pathfinding."""
        speed_map = np.zeros((self.height, self.width), dtype=float)
        for y in range(self.height):
            for x in range(self.width):
                terrain: Terrain = self.terrain[y, x]
                if terrain:
                    speed_map[y, x] = terrain.get_properties().movement_speed
        return speed_map

    def get_visibility_map(self) -> np.ndarray:
        """Generate a visibility matrix for AI."""
        visibility_map = np.zeros((self.height, self.width), dtype=float)
        for y in range(self.height):
            for x in range(self.width):
                terrain: Terrain = self.terrain[y, x]
                if terrain:
                    visibility_map[y, x] = terrain.get_properties().visibility
        return visibility_map

    def get_terrain(self, x: float, y: float) -> Terrain:
        """Return the terrain object at the specified coordinates."""
        x_terrain: int = int(np.floor(x))
        y_terrain: int = int(np.floor(y))

        return self.terrain[y_terrain, x_terrain]

    def render(self, screen: pygame.Surface, cell_size: int) -> None:
        """Render the terrain map to a PyGame surface."""
        for y in range(self.height):
            for x in range(self.width):
                terrain: Terrain = self.terrain[y, x]
                if terrain:
                    pygame.draw.rect(
                        screen,
                        terrain.get_color(),
                        (x * cell_size, y * cell_size, cell_size, cell_size),
                    )


class GameRenderer:
    """Handles the game rendering and PyGame setup."""

    def __init__(self, game_map: GameMap) -> None:
        """Initialize the renderer with the game map."""
        pygame.init()
        self.game_map: GameMap = game_map
        self.screen_width: int = game_map.width * game_map.cell_size
        self.screen_height: int = game_map.height * game_map.cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Terrain Game")
        self.clock = pygame.time.Clock()

    def run(self) -> None:
        """Main game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:  # Save map when 'S' is pressed
                        self.game_map.save_to_csv(Path("map.csv"))
                    else:
                        pass

            self.screen.fill((0, 0, 0))
            self.game_map.render(self.screen, cell_size=10)
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main() -> None:
    """Example usage of the terrain system."""
    parser = argparse.ArgumentParser(description="Terrain system")
    parser.add_argument("--load", type=str, help="Load the map from the specified file")
    args = parser.parse_args()

    if args.load:
        # Load the map from the specified file
        loaded_map = GameMap.load_from_csv(args.load)
    else:
        # Create new random map
        game_map = GameMap(100, 100)
        game_map.generate_random_map()

        # Save the map
        maps_dir = Path(__file__).resolve().parent / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)

        map_path = maps_dir / "example_map.csv"
        game_map.save_to_csv(map_path)

        # Load the map
        loaded_map = GameMap.load_from_csv(map_path)

    # Visualize the loaded map
    renderer = GameRenderer(loaded_map)
    renderer.run()


if __name__ == "__main__":
    main()
