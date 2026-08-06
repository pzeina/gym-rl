"""Terrain grid, line of sight, and objectives.

Coordinates are (x, y) with x the column and y the row; the grid array is
indexed ``grid[y, x]``. Terrain cells: OPEN (free), FOREST (passable, gives
cover and shortens the range at which the occupant can be spotted), WALL
(blocks movement and line of sight).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

OPEN, FOREST, WALL = 0, 1, 2

Coord = tuple[int, int]


@dataclass(frozen=True)
class Objective:
    """A named map objective, addressed on the radio as 'OBJ <name>'."""

    id: int
    name: str
    pos: Coord
    radius: float = 2.5


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class Waypoint:
    """A named control-measure point, addressed on the radio as 'WP <name>'.

    Control measures (A5) name the terrain an operation maneuvers through —
    they are not objectives: nothing is seized or held AT a waypoint, but an
    ADVANCE order can anchor on one, which puts route geometry on the net.
    """

    id: int
    name: str
    pos: Coord
    radius: float = 2.5


@dataclass(frozen=True)
class PhaseLine:
    """A named control-measure line ('PL <name>'): a straight segment.

    An ADVANCE to a phase line completes on reaching/crossing it; the anchor
    of such a mission is the nearest point of the segment (dynamic — it
    follows the agent's own position along the line).
    """

    id: int
    name: str
    a: Coord
    b: Coord

    def nearest_point(self, pos: tuple[float, float]) -> tuple[float, float]:
        """Closest point of the segment to ``pos`` (projection, clamped)."""
        ax, ay = self.a
        bx, by = self.b
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            return (float(ax), float(ay))
        t = ((pos[0] - ax) * dx + (pos[1] - ay) * dy) / length_sq
        t = max(0.0, min(1.0, t))
        return (ax + t * dx, ay + t * dy)

    def distance_to(self, pos: tuple[float, float]) -> float:
        """Euclidean distance from ``pos`` to the segment."""
        return dist(pos, self.nearest_point(pos))

    def side(self, pos: tuple[float, float]) -> int:
        """Which side of the (infinite) line ``pos`` lies on: -1 / 0 / +1.

        Used to detect a crossing: the sign flips when an agent passes the
        line. Degenerate (zero-length) lines report 0 for every point.
        """
        ax, ay = self.a
        bx, by = self.b
        cross = (bx - ax) * (pos[1] - ay) - (by - ay) * (pos[0] - ax)
        if cross > 1e-9:
            return 1
        if cross < -1e-9:
            return -1
        return 0


class World:
    """Static terrain plus objectives and control measures; all sim queries."""

    def __init__(
        self,
        grid: np.ndarray,
        objectives: list[Objective],
        waypoints: list[Waypoint] | None = None,
        phase_lines: list[PhaseLine] | None = None,
    ) -> None:
        self.grid = grid
        self.height, self.width = grid.shape
        self.objectives = objectives
        self.waypoints = waypoints or []
        self.phase_lines = phase_lines or []

    # ---------------- terrain queries ---------------- #

    def in_bounds(self, pos: Coord) -> bool:
        """True if pos lies on the map."""
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, pos: Coord) -> bool:
        """True if an agent can stand on pos."""
        return self.in_bounds(pos) and self.grid[pos[1], pos[0]] != WALL

    def cover_at(self, pos: Coord) -> bool:
        """True if pos gives cover (forest)."""
        return self.in_bounds(pos) and self.grid[pos[1], pos[0]] == FOREST

    def line_of_sight(self, a: Coord, b: Coord) -> bool:
        """Bresenham LOS check; walls block, endpoints never block."""
        x0, y0 = a
        x1, y1 = b
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while (x, y) != (x1, y1):
            if (x, y) != a and self.grid[y, x] == WALL:
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True

    def can_spot(self, observer: Coord, target: Coord, vision_range: float, forest_range: float) -> bool:
        """True if observer detects target: range (shorter into forest) + LOS."""
        d = dist(observer, target)
        limit = forest_range if self.cover_at(target) else vision_range
        return d <= limit and self.line_of_sight(observer, target)

    def local_patch(self, pos: Coord, r: int) -> np.ndarray:
        """(2r+1, 2r+1, 2) float32 patch: [passable, cover]; off-map = impassable."""
        size = 2 * r + 1
        patch = np.zeros((size, size, 2), dtype=np.float32)
        for j in range(size):
            for i in range(size):
                p = (pos[0] + i - r, pos[1] + j - r)
                if self.passable(p):
                    patch[j, i, 0] = 1.0
                    if self.cover_at(p):
                        patch[j, i, 1] = 1.0
        return patch

    def objective_by_name(self, name: str) -> Objective | None:
        """Look up an objective by radio name (case-insensitive)."""
        for obj in self.objectives:
            if obj.name.upper() == name.upper():
                return obj
        return None

    def control_by_name(self, name: str) -> Waypoint | PhaseLine | None:
        """Look up a control measure (waypoint or phase line) by radio name."""
        for wp in self.waypoints:
            if wp.name.upper() == name.upper():
                return wp
        for pl in self.phase_lines:
            if pl.name.upper() == name.upper():
                return pl
        return None

    @property
    def control_names(self) -> set[str]:
        """Radio names of every control measure on this map."""
        return {w.name for w in self.waypoints} | {p.name for p in self.phase_lines}

    # ---------------- generation ---------------- #

    @staticmethod
    def _blob(grid: np.ndarray, rng: np.random.Generator, value: int, n_seeds: int, growth: int) -> None:
        h, w = grid.shape
        for _ in range(n_seeds):
            x, y = int(rng.integers(1, w - 1)), int(rng.integers(1, h - 1))
            for _ in range(growth):
                grid[y, x] = value
                x = int(np.clip(x + rng.integers(-1, 2), 1, w - 2))
                y = int(np.clip(y + rng.integers(-1, 2), 1, h - 2))

    @staticmethod
    def _connected(grid: np.ndarray, points: list[Coord]) -> bool:
        """Flood-fill: are all points mutually reachable over non-wall cells?"""
        if not points:
            return True
        h, w = grid.shape
        seen = np.zeros_like(grid, dtype=bool)
        stack = [points[0]]
        seen[points[0][1], points[0][0]] = True
        while stack:
            x, y = stack.pop()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < w and 0 <= ny < h and not seen[ny, nx] and grid[ny, nx] != WALL:
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        return all(seen[p[1], p[0]] for p in points)

    @classmethod
    def generate(
        cls,
        width: int,
        height: int,
        objective_specs: list[tuple[str, Coord]],
        rng: np.random.Generator,
        *,
        forest_density: float = 1.0,
        wall_density: float = 1.0,
        must_connect: list[Coord] | None = None,
        waypoint_specs: list[tuple[str, Coord]] | None = None,
        phase_line_specs: list[tuple[str, Coord, Coord]] | None = None,
    ) -> World:
        """Procedurally generate a map guaranteed to connect key points.

        Waypoints are kept standable and connected like objectives (an
        ADVANCE order must be executable); phase lines are pure geometry and
        constrain nothing.
        """
        waypoint_specs = waypoint_specs or []
        phase_line_specs = phase_line_specs or []
        key_points = (
            [pos for _, pos in objective_specs]
            + [pos for _, pos in waypoint_specs]
            + list(must_connect or [])
        )
        for _attempt in range(20):
            grid = np.zeros((height, width), dtype=np.int8)
            area = width * height
            cls._blob(grid, rng, FOREST, n_seeds=max(2, int(area * 0.006 * forest_density)), growth=24)
            cls._blob(grid, rng, WALL, n_seeds=max(1, int(area * 0.003 * wall_density)), growth=10)
            for _, pos in objective_specs:  # keep objectives standable
                grid[pos[1], pos[0]] = OPEN
            for _, pos in waypoint_specs:  # waypoints must be reachable too
                grid[pos[1], pos[0]] = OPEN
            if cls._connected(grid, key_points):
                break
        else:  # extremely unlikely: fall back to walls-free map
            grid[grid == WALL] = OPEN
        objectives = [
            Objective(id=i, name=name, pos=pos) for i, (name, pos) in enumerate(objective_specs)
        ]
        waypoints = [
            Waypoint(id=i, name=name, pos=pos) for i, (name, pos) in enumerate(waypoint_specs)
        ]
        phase_lines = [
            PhaseLine(id=i, name=name, a=a, b=b)
            for i, (name, a, b) in enumerate(phase_line_specs)
        ]
        return cls(grid, objectives, waypoints, phase_lines)
