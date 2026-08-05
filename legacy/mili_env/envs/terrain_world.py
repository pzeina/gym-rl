from __future__ import annotations

import contextlib
import inspect
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, ClassVar, TypedDict

import gymnasium as gym
import numpy as np
import pyglet
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from mili_env.envs.classes.c2_doctrine import (
    ACT_ACTION_SCORES,
    MISSION_DERIVATION_DOCTRINE,
    REWARD_CONSTANTS,
    derivation_quality,
    mission_event_reward,
)
from mili_env.envs.classes.c2_orders import C2OrdersMixin
from mili_env.envs.classes.robot_base import (
    AgentRole,
    Moves,
    RobotAttributes,
    RobotBase,
    RobotConstraints,
    RobotPosition,
    SoldierElementaryAct,
)

# Local terrain/map classes & timing utilities
from mili_env.envs.classes.terrain import GameMap
from mili_env.envs.timing_utils import timing_log, timing_start, timing_stop

# ---------------------------------------------------------------------------
# Module constants (reintroduced after refactor) used throughout the class.
# ---------------------------------------------------------------------------
PI_OVER_8: float = float(np.pi / 8.0)
MIN_AGENT_DISTANCE: float = 2.0  # Minimum spacing when not grouped
MIN_AGENTS_FOR_COOPERATION: int = 2  # Pairs start forming at 2+ agents
COMMUNICATION_UPDATE_INTERVAL: float = 10.0  # steps between auto status broadcasts


class TerrainWorldEnv(gym.Env):
    """Custom environment for the terrain world with multi-agent support."""

    custom_metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: str | None = None, target_zone_size: int = 20,
                 terrain_filename: str = "plain_terrain.csv", num_agents: int = 1,
                 num_enemies: int = 0, enable_visualization: bool = False) -> None:
        """Initialize environment (delegated for readability)."""
        self.render_mode = render_mode
        self.target_zone_size = target_zone_size
        self.num_agents = num_agents
        self.num_enemies = num_enemies
        self.enable_visualization = enable_visualization  # Flag to control vis computations
        # Mission POIs configuration (defaults; can be overridden via setter)
        self.pois_per_mission = 1
        self.poi_radius = 0
        self.current_step = 0
        self.max_window_size = 512
        self.panel_width = 128
        self.np_random = np.random.default_rng()
        self._init_map(terrain_filename)
        self._init_roles()
        self._init_agents_enemies()
        self._init_spaces()
        self._init_rewards()
        self._init_render()
        self._fired_this_step: set[int] = set()
        # Exploration and event counters
        self._explored: set[tuple[int, int]] = set()
        # Per-agent mission zones and POIs
        self._mission_bboxes: list[tuple[np.ndarray, np.ndarray]] = []  # [(min_xy, max_xy)] per agent
        self._mission_pois: list[list[tuple[int, int]]] = []  # list of POIs per agent
        self._agent_zone_explored: dict[int, set[tuple[int, int]]] = {}
        self._prev_ally_alive_count = self.num_agents
        self._prev_enemy_alive_count = self.num_enemies
        self._help_requests = 0

    # ---- helper initialization blocks ----
    def _init_map(self, terrain_filename: str) -> None:
        self.game_map = GameMap.load_from_csv(Path(__file__).parent / "data" / terrain_filename)
        self.width = self.game_map.width
        self.height = self.game_map.height
        self.cell_size = min(self.max_window_size // self.width, self.max_window_size // self.height)
        self.window_width = self.cell_size * (self.width + self.panel_width)
        self.window_height = self.cell_size * self.height

    def _init_roles(self) -> None:
        self._role_rank_map = {
            AgentRole.CAP: 1,
            AgentRole.CDG: 2,
            AgentRole.SOA: 3,
            AgentRole.CDS: 3,
            AgentRole.ADU: 4,
            AgentRole.CDU: 5,
        }
        self.ROLE_THRESHOLDS = {
            "TEAM_LEVEL": 5,
            "GROUP_LEVEL": 9,
            "PLATOON_LEVEL": 16,
            "COMPANY_LEVEL": 20,
        }

    def _init_agents_enemies(self) -> None:
        self.agents = []
        self.agent_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 255), (0, 128, 255), (128, 255, 0), (128, 128, 128), (255, 192, 203)
        ]
        self.create_agents(grouped=True)
        self.enemies: list[RobotBase] = []
        if self.num_enemies > 0:
            self._create_enemies()

    def _init_spaces(self) -> None:
        vr = 2 * int(self.agents[0].vision_range) + 1 if self.agents else 1
        enc_len = self.game_map.get_terrain(0, 0).get_encode_size()
        obs_spec = {}
        for i in range(self.num_agents):
            base = {
                "position": Box(0, max(self.width, self.height) - 1, shape=(2,), dtype=np.float64),
                "distance": Box(0.0, np.sqrt(self.width**2 + self.height**2), shape=(1,), dtype=np.float64),
                "direction": Box(0.0, 2 * np.pi, shape=(1,), dtype=np.float64),
                "energy": Box(0.0, 100.0, shape=(1,), dtype=np.float64),
                "health": Box(0.0, 100.0, shape=(1,), dtype=np.float64),
                "ammunition": Box(0.0, 1000.0, shape=(1,), dtype=np.float64),
                "elementary_act": Discrete(len(SoldierElementaryAct)),  # Current elementary act
                "agent_id": Discrete(self.num_agents),
                "ally_mask": Box(0, 1, shape=(self.num_agents,), dtype=np.int64),
                "vision_map": Box(0.0, 1.0, shape=(vr, vr, enc_len), dtype=np.float32),
            }
            # Mission orders from command hierarchy (separate from elementary acts)
            base["current_mission"] = Discrete(len(SoldierElementaryAct) + 1)  # Mission + sentinel
            if self.num_enemies > 0:
                base["enemy_mask"] = Box(0, 1, shape=(self.num_enemies,), dtype=np.int64)
            obs_spec[f"agent_{i}"] = Dict(base)
        self.observation_space = Dict(obs_spec)

        self.COMM_NONE, self.COMM_STATUS, self.COMM_HELP = 0, 1, 2
        self.MAX_COMMS_PER_STEP = 3
        comm_types_space = MultiDiscrete([3] * self.MAX_COMMS_PER_STEP)
        comm_targets_space = MultiDiscrete([self.num_agents + 1] * self.MAX_COMMS_PER_STEP)
        # C2 order issuance: missions/orders (strategic objectives)
        self.MISSION_NO_CHANGE_SENTINEL = len(SoldierElementaryAct)
        orders_space = MultiDiscrete([len(SoldierElementaryAct) + 1] * self.num_agents)
        act_spec = {}
        for i in range(self.num_agents):
            act_spec[f"agent_{i}"] = Dict({
                # Elementary act selection (tactical behavior choice)
                "elementary_act": Discrete(len(SoldierElementaryAct)),
                "combat": Dict({
                    "move": Discrete(len(Moves)),
                    "fire_enemy": Discrete(self.num_agents + 1),
                }),
                "comm": Dict({
                    "types": comm_types_space,
                    "targets": comm_targets_space,
                }),
                # C2: Mission orders (strategic objectives from hierarchy)
                "c2": Dict({
                    "give_order": Discrete(2),
                    "orders": orders_space,
                }),
            })
        self.action_space = Dict(act_spec)

    def _init_rewards(self) -> None:
        self.reward_registry = {}
        self.current_reward_name = "default"
        self.reward_mode = "team"
        self.reward_log_path = None
        try:
            self.register_reward_function("default", per_agent_fn=self._get_reward, team_fn=None)
        except KeyError:
            self.reward_registry["default"] = (None, None)

    def _init_render(self) -> None:
        self.window = None
        self.clock = None
        self.batch = None
        self.fps = self.custom_metadata["render_fps"]
        self.zoom_factor = 1.0
        self._recent_order_attempts = 0
        # Rank mapping note: defined earlier in _init_roles

    # Utility for advisors
    def _team_center_pos(self) -> tuple[float, float]:
        xs = [a.get_position()[0] for a in self.agents]
        ys = [a.get_position()[1] for a in self.agents]
        if not xs:
            return (0.0, 0.0)
        return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))

    def team_center_pos(self) -> tuple[float, float]:
        """Public accessor for the current team center position."""
        return self._team_center_pos()

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

    def _generate_target_zones(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate per-agent target zones (center, bbox)."""
        zones: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(self.num_agents):
            center = self.np_random.integers(0, [self.width, self.height], size=2, dtype=int)
            bbox = np.asarray([
                center - self.target_zone_size // 2,
                center + self.target_zone_size // 2,
            ])
            bbox = np.clip(bbox, 0, [self.width - 1, self.height - 1])
            zones.append((center, bbox))
        return zones

    def _generate_pois_for_bbox(
        self, center: np.ndarray, bbox_tuple: tuple[np.ndarray, np.ndarray], *, k: int = 1
    ) -> list[tuple[int, int]]:
        """Generate up to k points-of-interest inside bbox, always including center.

        Returns a list of (x, y) integer coordinates.
        """
        pois: list[tuple[int, int]] = [(int(center[0]), int(center[1]))]
        if k <= 1:
            return pois
        # Sample additional POIs uniformly inside bbox
        min_xy, max_xy = bbox_tuple
        for _ in range(k - 1):
            x = int(self.np_random.integers(int(min_xy[0]), int(max_xy[0]) + 1))
            y = int(self.np_random.integers(int(min_xy[1]), int(max_xy[1]) + 1))
            pois.append((x, y))
        return pois

    def _select_spawn_location(
        self,
        agent_index: int,
        target_zones: list[tuple[np.ndarray, np.ndarray]],
        *,
        grouped: bool,
        group_center: np.ndarray | None,
    ) -> np.ndarray:
        """Choose a valid spawn location with fallbacks."""
        attempts = 50
        for _ in range(attempts):
            if grouped and group_center is not None:
                candidate = self._get_grouped_agent_location(agent_index, group_center)
            else:
                candidate = self.np_random.integers(0, [self.width, self.height], size=2, dtype=int)
            if self._is_location_valid(candidate, target_zones, grouped=grouped):
                return candidate
        # Systematic search fallback
        for x in range(self.width):
            for y in range(self.height):
                candidate = np.array([x, y])
                if self._is_location_valid(candidate, target_zones, grouped=False):
                    return candidate
        # Last resort deterministic fallback
        return np.array([
            min(agent_index, self.width - 1),
            min(agent_index, self.height - 1),
        ])

    def _instantiate_agent(self, agent_id: int, location: np.ndarray) -> None:
        position = RobotPosition(x=float(location[0]), y=float(location[1]), angle=0)
        attributes = RobotAttributes(
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
        role = self._assign_role_for_index(agent_id)
        agent = RobotBase(
            position=position,
            attributes=attributes,
            game_map=self.game_map,
            constraints=constraints,
            agent_id=agent_id,
            role=role,
            team_id=0,
        )
        self.agents.append(agent)

    def create_agents(self, *, grouped: bool = False) -> None:
        """Create multiple robot agents with reduced complexity."""
        self.agents = []
        target_zones = self._generate_target_zones()
        # Store mission bboxes as (min,max) tuple and generate POIs per agent for mission definition
        self._mission_bboxes = [(bbox[0].copy(), bbox[1].copy()) for (_center, bbox) in target_zones]
        self._mission_pois = [
            self._generate_pois_for_bbox(center, (bbox[0], bbox[1]), k=self.pois_per_mission)
            for (center, bbox) in target_zones
        ]
        # Reset per-agent zone explored sets
        self._agent_zone_explored = {i: set() for i in range(self.num_agents)}
        group_center = self._find_group_center(target_zones) if (grouped and self.num_agents > 1) else None
        use_grouped = group_center is not None
        for agent_id in range(self.num_agents):
            spawn_loc = self._select_spawn_location(
                agent_id,
                target_zones,
                grouped=use_grouped,
                group_center=group_center,
            )
            self._instantiate_agent(agent_id, spawn_loc)

    def set_poi_config(self, *, pois_per_mission: int | None = None, poi_radius: int | None = None) -> None:
        """Configure number of POIs per mission and detection radius.

        pois_per_mission: minimum 1. poi_radius: minimum 0 (grid cells).
        Takes effect on next create_agents()/reset for POI generation; radius applies immediately.
        """
        if pois_per_mission is not None:
            self.pois_per_mission = int(max(1, pois_per_mission))
        if poi_radius is not None:
            self.poi_radius = int(max(0, poi_radius))

    # ---------- Enemy creation & hierarchy helpers ----------
    def _create_enemies(self) -> None:
        """Spawn simple enemy units clustered in a zone (team id 1)."""
        self.enemies = []
        if self.num_enemies <= 0:
            return
        # Choose enemy cluster center away from ally center (simple heuristic)
        ally_center = self._team_center_pos()
        for enemy_id in range(self.num_enemies):
            # Sample position; retry until sufficiently far from ally_center
            ex = self.np_random.integers(0, self.width)
            ey = self.np_random.integers(0, self.height)
            for _ in range(50):
                ex_candidate = self.np_random.integers(0, self.width)
                ey_candidate = self.np_random.integers(0, self.height)
                if (
                    (ex_candidate - ally_center[0]) ** 2
                    + (ey_candidate - ally_center[1]) ** 2
                    > (0.25 * (self.width**2 + self.height**2))
                ):
                    ex, ey = ex_candidate, ey_candidate
                    break
            pos = RobotPosition(x=float(ex), y=float(ey), angle=0.0)
            attrs = RobotAttributes(
                health=100.0,
                energy=100.0,
                ammunition=100.0,
                health_efficiency=1.0,
                energy_efficiency=1.0,
                speed_efficiency=5.0,
                ammunition_efficiency=1.0,
            )
            cons = RobotConstraints(
                vision_range=100.0,
                communication_range=30.0,
                max_speed_forward=1.0,
                max_speed_backward=0.2,
                max_angular_speed=PI_OVER_8,
                max_health=100.0,
                max_energy=100.0,
                max_ammunition=100.0,
            )
            enemy = RobotBase(
                position=pos,
                attributes=attrs,
                game_map=self.game_map,
                constraints=cons,
                agent_id=self.num_agents + enemy_id,  # unique id space if ever integrated
                role=AgentRole.CAP,
                team_id=1,
            )
            self.enemies.append(enemy)

    def _assign_role_for_index(self, idx: int) -> AgentRole:
        """Assign roles progressively as force size grows.

        Sequence (bottom-up accumulation):
        - First 4: one team (CAP leaders). We tag first as CAP, others CAP as well.
        - After 8 (two teams): add CDG.
        - After 16 (two groups): add SOA + CDS.
        - After 20+: add ADU then CDU.
        This is a heuristic mapping; refine as organizational detail expands.
        """
        # Threshold-based role allocation
        if self.num_agents >= self.ROLE_THRESHOLDS["COMPANY_LEVEL"]:
            mapping = (
                [AgentRole.CAP] * 4
                + [AgentRole.CAP] * 4
                + [AgentRole.CDG]
                + [AgentRole.CAP] * 4
                + [AgentRole.CAP] * 4
                + [AgentRole.CDG]
                + [AgentRole.SOA, AgentRole.CDS, AgentRole.ADU, AgentRole.CDU]
            )
        elif self.num_agents >= self.ROLE_THRESHOLDS["PLATOON_LEVEL"]:
            mapping = (
                [AgentRole.CAP] * 4
                + [AgentRole.CAP] * 4
                + [AgentRole.CDG]
                + [AgentRole.CAP] * 4
                + [AgentRole.CAP] * 4
                + [AgentRole.CDG]
                + [AgentRole.SOA, AgentRole.CDS]
            )
        elif self.num_agents >= self.ROLE_THRESHOLDS["GROUP_LEVEL"]:
            mapping = [AgentRole.CAP] * 8 + [AgentRole.CDG] + [AgentRole.CAP] * (self.num_agents - 9)
        elif self.num_agents >= self.ROLE_THRESHOLDS["TEAM_LEVEL"]:
            mapping = [AgentRole.CAP] * 4 + [AgentRole.CAP] * (self.num_agents - 4) + [AgentRole.CDG]
        else:
            mapping = [AgentRole.CAP]*self.num_agents
        if idx < len(mapping):
            return mapping[idx]
        return AgentRole.CAP

    def _get_highest_rank_agent(self) -> RobotBase | None:
        if not self.agents:
            return None
        ranked = sorted(self.agents, key=lambda a: self._role_rank_map.get(a.role, 0), reverse=True)
        return ranked[0]

    def _get_obs(self) -> dict:
        """Get the decentralized observations for all agents."""
        observations = {}
        for agent_id, agent in enumerate(self.agents):
            direction: float = agent.get_direction()

            # Build ally mask: same team = 1, else 0
            ally_mask = np.zeros((self.num_agents,), dtype=np.int64)
            agent_team = getattr(agent, "team_id", -1)
            for j, other in enumerate(self.agents):
                other_team = getattr(other, "team_id", -2)
                ally_mask[j] = 1 if other_team == agent_team else 0
            ally_mask[agent_id] = 1

            # Vision map encoding using underlying terrain properties seen by agent
            enc_len_local = self.game_map.get_terrain(0, 0).get_encode_size()
            vr_local = 2 * int(getattr(agent, "vision_range", 0)) + 1
            vision = np.zeros((vr_local, vr_local, enc_len_local), dtype=np.float32)
            enc = agent.encode_vision_map()
            if hasattr(enc, "squeeze"):
                with contextlib.suppress(Exception):
                    vision = enc.squeeze(0).numpy()

            # Current elementary act (tactical behavior)
            elementary_act_val = (
                agent.current_elementary_act.value
                if getattr(agent, "current_elementary_act", None) is not None
                else 0  # Default to first elementary act
            )

            # Current mission (strategic objective from command hierarchy)
            current_mission_val = (
                agent.current_mission.value
                if getattr(agent, "current_mission", None) is not None
                else len(SoldierElementaryAct)  # Sentinel for no mission
            )

            observations[f"agent_{agent_id}"] = {
                "position": np.asarray(agent.get_position(), dtype=np.float64),
                "distance": np.array([agent.get_distance_to_target()], dtype=np.float64),
                "direction": np.array([direction], dtype=np.float64),
                "energy": np.array([agent.get_energy()], dtype=np.float64),
                "health": np.array([agent.get_health()], dtype=np.float64),
                "ammunition": np.array([agent.get_ammunition()], dtype=np.float64),
                "elementary_act": elementary_act_val,
                "current_mission": current_mission_val,
                "agent_id": agent_id,
                "ally_mask": ally_mask,
                "vision_map": vision,
            }

            # Enemy awareness: binary mask of enemies currently within vision range
            if getattr(self, "num_enemies", 0) > 0 and getattr(self, "enemies", None):
                enemy_mask = np.zeros((self.num_enemies,), dtype=np.int64)
                try:
                    vrange = float(getattr(agent, "vision_range", 0.0))
                    ax, ay = agent.get_position()
                    for e_idx, enemy in enumerate(self.enemies):
                        enemy_x, enemy_y = enemy.get_position()
                        # Distance from agent to enemy; renamed variables for clarity (lint)
                        if np.linalg.norm(np.asarray([ax - enemy_x, ay - enemy_y])) <= vrange:
                            enemy_mask[e_idx] = 1
                except (AttributeError, ValueError) as exc:  # pragma: no cover - defensive
                    logging.getLogger(__name__).debug("enemy_mask build failed for agent %s: %s", agent_id, exc)
                observations[f"agent_{agent_id}"]["enemy_mask"] = enemy_mask

        return observations

    def _normalize_observation(self, observation: dict) -> dict:
        """Normalize observation values into bounded numeric ranges.

        Complexity kept low by delegating per-key handling to helper.
        """
        norm: dict[str, dict[str, Any]] = {}
        for agent_key, agent_obs in observation.items():
            agent_norm: dict[str, Any] = {}
            for obs_key, value in agent_obs.items():
                agent_norm[obs_key] = self._normalize_obs_value(obs_key, value)
            norm[agent_key] = agent_norm
        return norm

    def _normalize_obs_value(self, obs_key: str, value: np.ndarray | float) -> np.ndarray | float:
        """Normalize a single observation field.

        Uses a small dispatch map to keep branching/returns low.
        """
        handlers: dict[str, Callable[[np.ndarray | int | float], np.ndarray | int | float]] = {
            "position": lambda v: np.asarray(v / np.array([self.width, self.height])),
            "distance": lambda v: np.atleast_1d(np.asarray(v / np.sqrt(self.width**2 + self.height**2))),
            "direction": lambda v: np.atleast_1d(np.asarray(v / (2 * np.pi))),
            "energy": lambda v: np.atleast_1d(np.asarray(v / 100.0)),
            "health": lambda v: np.atleast_1d(np.asarray(v / 100.0)),
            "ammunition": lambda v: np.atleast_1d(np.asarray(v / 1000.0)),
            "mission": lambda v: int(v),
            "agent_id": lambda v: v,
            "ally_mask": lambda v: v,
            "vision_map": lambda v: v,
            "enemy_mask": lambda v: v,
            "last_orders": lambda v: v,
        }
        if obs_key not in handlers:
            msg = f"Unknown observation key: {obs_key}"
            raise ValueError(msg)
        return handlers[obs_key](value)

    def _get_info(self) -> dict:
        """Get the information of the environment for all agents."""
        info = {}

        for agent_id, agent in enumerate(self.agents):
            info[f"agent_{agent_id}"] = {
                "distance": agent.get_distance_to_target(),
                "health": np.asarray(agent.get_health()),
                "energy": np.asarray(agent.get_energy()),
                "ammunition": np.asarray(agent.get_ammunition()),
            }

        return info


    # ----------------- Reward function registry and helpers -----------------
    def register_reward_function(
            self, name: str, *, per_agent_fn: Callable[[dict], dict] | None = None,
            team_fn: Callable[[dict, dict], float] | None = None
        ) -> None:
        """Register a reward function pair under `name`.

        Parameters
        ----------
        name:
            Short identifier used to select the reward pair at runtime.
        per_agent_fn:
            Callable(prev_info) -> dict[str, float]. Should return a mapping
            from agent keys (e.g. "agent_0") to numeric rewards for each agent.
        team_fn:
            Callable(prev_info, per_agent_rewards) -> float. Computes a scalar
            team-level reward based on the previous info and the per-agent
            rewards.

        Either callable may be `None`. When a function is missing, the
        environment will fall back to sensible defaults (see `_compute_rewards`).
        """
        self.reward_registry[name] = (per_agent_fn, team_fn)

    def set_reward_function(self, name: str) -> None:
        """Select an existing registered reward function by name."""
        if name not in self.reward_registry:
            msg = f"Reward function '{name}' is not registered"
            raise KeyError(msg)
        self.current_reward_name = name

    def set_reward_mode(self, mode: str) -> None:
        """Set reward mode: 'team', 'per_agent', or 'both'."""
        if mode not in ("team", "per_agent", "both"):
            msg = "mode must be one of 'team', 'per_agent', or 'both'"
            raise ValueError(msg)
        self.reward_mode = mode

    def get_registered_reward_functions(self) -> list[str]:
        """Get a list of registered reward function names."""
        return list(self.reward_registry.keys())

    def _compute_rewards(self, prev_info: dict) -> tuple[dict, float]:
        """Compute per-agent and team rewards according to the currently selected functions.

        Returns (per_agent_rewards, team_reward_scalar)
        """
        per_agent_fn, team_fn = self.reward_registry.get(self.current_reward_name, (None, None))

        # Compute per-agent rewards (fallback to the default `_get_reward` which
        # returns the centralized scalar duplicated for each agent).
        per_agent_rewards = per_agent_fn(prev_info) if per_agent_fn is not None else self._get_reward(prev_info)

        # Compute team reward (fallback to mean of per-agent rewards)
        if team_fn is not None:
            try:
                team_reward = float(team_fn(prev_info, per_agent_rewards))
            except (TypeError, ValueError, KeyError):
                team_reward = float(next(iter(per_agent_rewards.values()), 0.0))
        else:
            # Default: mean of per-agent rewards
            try:
                vals = [float(v) for v in per_agent_rewards.values()]
                team_reward = float(sum(vals) / len(vals)) if vals else 0.0
            except (TypeError, ValueError, ZeroDivisionError):
                team_reward = float(next(iter(per_agent_rewards.values()), 0.0))

        return per_agent_rewards, team_reward

    def _archive_reward_run(
            self, *, log_path: str | None = None, seed: int | None = None, extra: dict | None = None
        ) -> None:
        """Append a short JSON line describing the reward selection and env config.

        If `log_path` is given it will be used, otherwise `self.reward_log_path` or
        default to `debug/reward_runs.jsonl`.
        """
        path = log_path or self.reward_log_path or (Path.cwd() / "debug" / "reward_runs.jsonl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        record = {
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "reward_name": self.current_reward_name,
            "reward_mode": self.reward_mode,
            "num_agents": self.num_agents,
            "target_zone_size": self.target_zone_size,
        }
        if seed is not None:
            record["seed"] = seed
        if extra:
            record.update(extra)

        logger = logging.getLogger(__name__)
        try:
            with Path(path).open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            # Non-fatal: archiving should not break environment behavior
            logger.exception("Failed to archive reward run")

    def _archive_reward_function_details(
            self, name: str, func: Callable | None = None, *, log_path: str | None = None
        ) -> None:
        """Archive reward function custom_metadata: name, docstring and best-effort source.

        Writes a JSON line to `debug/reward_functions.jsonl` (or `log_path` if given).
        """
        path = log_path or (Path.cwd() / "debug" / "reward_functions.jsonl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        record = {
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "reward_name": name,
        }

        if func is not None:
            try:
                doc = inspect.getdoc(func) or ""
                src = inspect.getsource(func)
            except (OSError, TypeError):
                doc = inspect.getdoc(func) or ""
                src = None
            record["docstring"] = doc
            if src:
                record["source"] = src

        try:
            with Path(path).open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            logging.getLogger(__name__).exception("Failed to archive reward function details")

    def archive_reward_function_details(self, name: str, *, log_path: str | None = None) -> None:
        """Public wrapper: archive docstring and source for a registered reward function.

        This finds the registered function by `name` and archives its custom_metadata. If the
        function isn't registered, it will still record the name.
        """
        per_agent_fn, team_fn = self.reward_registry.get(name, (None, None))
        # Archive per-agent function details first (if present)
        if per_agent_fn is not None:
            try:
                self._archive_reward_function_details(name, per_agent_fn, log_path=log_path)
            except OSError:
                logging.getLogger(__name__).exception("Failed archiving per-agent function details")
        # Archive team function details only if present and different
        if team_fn is not None and team_fn is not per_agent_fn:
            try:
                self._archive_reward_function_details(name + "::team", team_fn, log_path=log_path)
            except (OSError, TypeError, ValueError):
                logging.getLogger(__name__).exception("Failed archiving team function details")

    def _get_reward(self, prev_info: dict) -> dict[str, np.floating[Any]]:
        """Compute the default centralized reward and return per-agent mapping.

        This is intentionally concise: metrics collection -> base scalar reward -> shaping.
        """
        current_info = self._get_info()

        metrics = self._collect_reward_metrics(prev_info, current_info)
        base_reward = self._compute_base_reward(metrics)
        base_reward = self._apply_c2_shaping(base_reward)
        return self._distribute_mission_shaping(base_reward)

    # --- Reward helpers (factored out to reduce cognitive complexity) --- #
    class _RewardMetrics(TypedDict):
        total_distance_improvement: float
        agents_at_target: int
        alive_agents: int
        total_energy: float
        total_health: float

    def _collect_reward_metrics(self, prev_info: dict, current_info: dict) -> _RewardMetrics:
        total_distance_improvement = 0.0
        agents_at_target = 0
        alive_agents = 0
        total_energy = 0.0
        total_health = 0.0
        for agent_id, agent in enumerate(self.agents):
            key = f"agent_{agent_id}"
            cur_dist = current_info[key]["distance"]
            prev_dist = prev_info.get(key, {}).get("distance", cur_dist)
            total_distance_improvement += (prev_dist - cur_dist)
            if agent.state.is_at_target():
                agents_at_target += 1
            if agent.state.is_alive():
                alive_agents += 1
            total_energy += agent.get_energy()
            total_health += agent.get_health()
        metrics: TerrainWorldEnv._RewardMetrics = {
            "total_distance_improvement": float(total_distance_improvement),
            "agents_at_target": agents_at_target,
            "alive_agents": alive_agents,
            "total_energy": float(total_energy),
            "total_health": float(total_health),
        }
        return metrics

    def _events_fill_enemy_and_comm(self, events: dict[str, float]) -> None:
        ally_alive = sum(1 for a in self.agents if a.state.is_alive())
        delta_ally = self._prev_ally_alive_count - ally_alive
        if delta_ally > 0:
            events["ally_killed"] = float(delta_ally)
        self._prev_ally_alive_count = ally_alive

        events["ally_assist"] = float(self._help_requests)

        if self.num_enemies <= 0 or not getattr(self, "enemies", None):
            return
        enemy_alive = sum(1 for e in self.enemies if e.state.is_alive())
        delta_enemy = self._prev_enemy_alive_count - enemy_alive
        if delta_enemy > 0:
            events["enemy_killed"] = float(delta_enemy)
        self._prev_enemy_alive_count = enemy_alive
        for agent in self.agents:
            ax, ay = agent.get_position()
            vr = float(getattr(agent, "vision_range", 0.0))
            for enemy in self.enemies:
                ex, ey = enemy.get_position()
                if np.linalg.norm(np.asarray([ax - ex, ay - ey])) <= vr:
                    events["enemy_detected"] += 1.0
        if enemy_alive == 0:
            events["zone_cleared"] = 1.0

    def _collect_events_for_agent(self, agent_id: int) -> dict[str, float]:
        """Collect events for a specific agent; zone_explored is mission-zone specific."""
        events: dict[str, float] = {
            "enemy_killed": 0.0,
            "ally_killed": 0.0,
            "enemy_detected": 0.0,
            "ally_assist": 0.0,
            "zone_explored": 0.0,
            "zone_cleared": 0.0,
            "poi_reached": 0.0,
        }
        # Global enemy/ally/detection events
        self._events_fill_enemy_and_comm(events)
        # Mission-zone exploration for this agent
        if 0 <= agent_id < self.num_agents and self._mission_bboxes:
            agent = self.agents[agent_id]
            x0, y0 = agent.get_position()
            min_xy, max_xy = self._mission_bboxes[agent_id]
            newly_seen = 0
            explored = self._agent_zone_explored.setdefault(agent_id, set())
            for angle, length in agent.get_vision_rays():
                for step in range(1, int(length) + 1):
                    gx = int(x0 + step * np.cos(angle))
                    gy = int(y0 + step * np.sin(angle))
                    if gx < int(min_xy[0]) or gy < int(min_xy[1]) or gx > int(max_xy[0]) or gy > int(max_xy[1]):
                        continue
                    if 0 <= gx < self.width and 0 <= gy < self.height:
                        cell = (gx, gy)
                        if cell not in explored:
                            explored.add(cell)
                            newly_seen += 1
            events["zone_explored"] = float(newly_seen)
            # POI reached: if agent position is at any POI within a small tolerance
            poi_list = self._mission_pois[agent_id] if agent_id < len(self._mission_pois) else []
            tol = int(getattr(self, "poi_radius", 0))  # tolerance radius in grid cells
            px, py = round(x0), round(y0)
            for (tx, ty) in poi_list:
                if abs(px - int(tx)) <= tol and abs(py - int(ty)) <= tol:
                    events["poi_reached"] += 1.0
                    break
        return events

    # --- Public accessors for mission definitions ---
    def get_mission_pois(self, agent_id: int) -> list[tuple[int, int]]:
        """Return the list of mission points-of-interest for the given agent.

        The POIs are grid coordinates (x, y) inside the agent's mission zone.
        Returns an empty list if no POIs are defined for the agent.
        """
        if 0 <= agent_id < len(self._mission_pois):
            return list(self._mission_pois[agent_id])
        return []

    def get_mission_bbox(self, agent_id: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Return the mission zone bounding box for the given agent.

        Returns ((min_x, min_y), (max_x, max_y)) in grid coordinates, or None
        if the agent has no mission bbox defined.
        """
        if 0 <= agent_id < len(self._mission_bboxes):
            min_xy, max_xy = self._mission_bboxes[agent_id]
            return ((int(min_xy[0]), int(min_xy[1])), (int(max_xy[0]), int(max_xy[1])))
        return None

    def _compute_base_reward(self, m: _RewardMetrics) -> float:
        """Compute base reward from distance progress, survival, and mission events."""
        # Base progress and survival reward
        reward = (
            REWARD_CONSTANTS["PROGRESS_SCALE"] * m["total_distance_improvement"]
            + REWARD_CONSTANTS["ENERGY_SCALE"] * (m["total_energy"] / max(1, self.num_agents))
        )

        # Add mission-specific event rewards per agent (based on assigned missions, not elementary acts)
        for agent_id in range(self.num_agents):
            if agent_id < len(self.agents):
                agent = self.agents[agent_id]
                # Get the agent's current strategic mission (from command hierarchy)
                current_mission = getattr(agent, "current_mission", None)

                events = self._collect_events_for_agent(agent_id)
                mission_reward = mission_event_reward(current_mission, events)
                reward += REWARD_CONSTANTS["EVENT_REWARD_SCALE"] * mission_reward

        # Terminal conditions
        if m["agents_at_target"] == m["alive_agents"] and m["alive_agents"] > 0:
            reward += (
                REWARD_CONSTANTS["SUCCESS_BASE"]
                + REWARD_CONSTANTS["SUCCESS_ENERGY_FACTOR"] * m["total_energy"]
                + REWARD_CONSTANTS["SUCCESS_HEALTH_FACTOR"] * m["total_health"]
            )
        elif m["alive_agents"] == 0:
            reward += REWARD_CONSTANTS["FAILURE_BASE"]

        return reward

    def _apply_c2_shaping(self, reward: float) -> float:
        """RL-based shaping: penalize excessive mission changes; reward formation compliance."""
        with contextlib.suppress(Exception):
            change_count = float(getattr(self, "_order_change_count", 0))
            # Penalize if too many changes relative to agents count (normalized)
            change_ratio = change_count / max(1, self.num_agents)
            reward -= REWARD_CONSTANTS.get("ORDER_CHANGE_NEG_PENALTY", 0.0) * change_ratio
            # Formation/Compliance positive shaping (unchanged)
            follow_dist = 5.0
            compliant = 0
            for i, a_hi in enumerate(self.agents):
                rank_hi = self._role_rank(a_hi.role)
                pos_hi = np.array(a_hi.get_position(), dtype=float)
                for j, a_lo in enumerate(self.agents):
                    if i == j:
                        continue
                    rank_lo = self._role_rank(a_lo.role)
                    if rank_lo >= rank_hi:
                        continue
                    pos_lo = np.array(a_lo.get_position(), dtype=float)
                    if np.linalg.norm(pos_lo - pos_hi) <= follow_dist:
                        compliant += 1
            max_pairs = max(1, self.num_agents * (self.num_agents - 1))
            reward += REWARD_CONSTANTS.get("FOLLOW_COMPLIANCE_SCALE", 0.0) * (compliant / max_pairs)
        return reward

    def _elementary_act_action_alignment_reward(
            self,
            agent_idx: int,
            agent: RobotBase,
            move_action: int,
            fire_action: int
        ) -> float:
        """Reward alignment between chosen elementary act and actual actions taken.

        Elementary acts should guide action selection:
        - MOVE: favors forward/backward movement
        - POST: favors stationary position with rotation/fire capability
        - FIRE: favors fire actions when enemies in range
        - RECON: favors movement for exploration
        - etc.
        """
        elementary_act = getattr(agent, "current_elementary_act", None)
        if elementary_act is None:
            return 0.0

        alignment_reward = 0.0

        # Movement alignment
        moved = move_action in (Moves.FORWARD.value, Moves.BACKWARD.value)
        rotated = move_action in (Moves.ROTATE_LEFT.value, Moves.ROTATE_RIGHT.value)
        stationary = move_action == Moves.IDLE.value

        # Fire alignment
        fired = fire_action < self.num_agents and agent_idx in getattr(self, "_fired_this_step", set())

        def alignement_reward(
                aligned_value: float,
                misaligned_value: float,
                *,
                aligned: bool,
                misaligned: bool
            ) -> float:
            return (
                REWARD_CONSTANTS.get("ELEMENTARY_ACT_ALIGNMENT", aligned_value) if aligned else 0.0
                - REWARD_CONSTANTS.get("ELEMENTARY_ACT_MISALIGNMENT", misaligned_value) if misaligned else 0.0
            )

        # Elementary act specific alignment rewards
        if elementary_act == SoldierElementaryAct.MOVE:
            alignment_reward += alignement_reward(0.1, 0.05, aligned=moved, misaligned=stationary)

        elif elementary_act == SoldierElementaryAct.POST:
            alignment_reward += alignement_reward(0.1, 0.05, aligned=stationary or rotated, misaligned=moved)

        elif elementary_act == SoldierElementaryAct.FIRE:
            alignment_reward += alignement_reward(
                0.15,
                0.1,
                aligned=fired,
                misaligned=not fired and self._enemies_in_range(agent)
            )

        # Add other elementary acts as needed

        return alignment_reward

    def _enemies_in_range(self, agent: RobotBase) -> bool:
        """Check if there are enemies within the agent's vision/fire range."""
        if not getattr(self, "enemies", None):
            return False

        agent_pos = np.array(agent.get_position())
        vision_range = getattr(agent, "vision_range", 0.0)

        for enemy in self.enemies:
            if not enemy.state.is_alive():
                continue
            enemy_pos = np.array(enemy.get_position())
            if np.linalg.norm(agent_pos - enemy_pos) <= vision_range:
                return True
        return False

    def _intrinsic_action_shaping(self, agent_idx: int, agent: RobotBase) -> float:
        """Reward shaping based on elementary act execution and action alignment."""
        # Get the actions taken this step
        move_action = getattr(agent, "last_move_action", Moves.IDLE.value)
        fire_action = getattr(agent, "last_fire_action", self.num_agents)

        # Elementary act-action alignment reward
        alignment_reward = self._elementary_act_action_alignment_reward(agent_idx, agent, move_action, fire_action)

        # Legacy action scoring (for backward compatibility)
        legacy_reward = 0.0
        with contextlib.suppress(Exception):
            elementary_act = getattr(agent, "current_elementary_act", None)
            if elementary_act in ACT_ACTION_SCORES:
                scores = ACT_ACTION_SCORES[elementary_act]
                moved_flag = bool(getattr(agent, "moved_this_step", False))
                fired_flag = agent_idx in getattr(self, "_fired_this_step", set())
                if elementary_act == SoldierElementaryAct.MOVE:
                    legacy_reward = float(scores.get("moved" if moved_flag else "stationary", 0.0))
                elif elementary_act == SoldierElementaryAct.POST:
                    legacy_reward = float(scores.get("stationary" if not moved_flag else "moved", 0.0))
                elif elementary_act == SoldierElementaryAct.FIRE:
                    legacy_reward = float(scores.get("fired" if fired_flag else "not_fired", 0.0))

        return alignment_reward + legacy_reward

    def _derivation_decision_shaping(self, agent: RobotBase) -> float:
        """Reward good mission derivation decisions (separate from elementary acts)."""
        with contextlib.suppress(Exception):
            issuer_rank = self._role_rank(agent.role)
            last_derived = getattr(agent, "last_derived_missions", None)
            # Use current_mission (strategic) not elementary_act (tactical) for derivation
            current_mission = getattr(agent, "current_mission", None)
            if isinstance(last_derived, dict) and issuer_rank > 1 and current_mission is not None:
                qual = 0.0
                cnt = 0
                for tgt_id, proposed in last_derived.items():
                    tgt_i = int(tgt_id)
                    if 0 <= tgt_i < len(self.agents):
                        if self._role_rank(self.agents[tgt_i].role) >= issuer_rank:
                            continue
                        qual += float(derivation_quality(current_mission, proposed, MISSION_DERIVATION_DOCTRINE))
                        cnt += 1
                if cnt > 0:
                    return REWARD_CONSTANTS.get("DERIVATION_DECENTRALIZED_SCALE", 0.5) * (qual / cnt)
        return 0.0

    def _distribute_mission_shaping(self, base_reward: float) -> dict[str, np.floating[Any]]:
        rewards: dict[str, np.floating[Any]] = {}
        for agent_id, agent in enumerate(self.agents):
            shaped = base_reward
            shaped += self._intrinsic_action_shaping(agent_id, agent)
            shaped += self._derivation_decision_shaping(agent)
            rewards[f"agent_{agent_id}"] = np.float64(shaped)
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
                    total_bonus += REWARD_CONSTANTS["COOP_PAIR_BONUS"]
                    cooperation_pairs += 1
                # Small penalty for being too far apart
                elif distance > 2 * self.agents[i].communication_range:
                    total_bonus += REWARD_CONSTANTS["COOP_DISTANCE_PENALTY"]

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

        # Reset step counter & order change trackers
        self.current_step = 0
        self._order_change_count = 0

        self.create_agents()

        # Reset exploration memory and episode counters
        self._explored = set()
        self._prev_ally_alive_count = sum(1 for a in self.agents if a.state.is_alive())
        if getattr(self, "enemies", None):
            self._prev_enemy_alive_count = sum(1 for e in self.enemies if e.state.is_alive())
        else:
            self._prev_enemy_alive_count = 0
        self._help_requests = 0
        self._fired_this_step = set()

        # Archive run metadata (reward selection, seed, etc.) for reproducibility
        with contextlib.suppress(Exception):
            self._archive_reward_run(seed=seed)

        timing_stop("reset")
        timing_log()
        return self._get_obs(), self._get_info()


    def _process_communications(self) -> None:
        """Process communications between agents within communication range."""
        current_time = float(self.current_step)

        # Clear any pending messages and process communications
        for sender in self.agents:
            # Use the correct outbox attribute from RobotBase
            outbox = getattr(sender, "message_outbox", [])

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

    def _role_rank(self, role: AgentRole) -> int:
        """Relative rank for C2 compliance checks (higher can order lower)."""
        return {
            AgentRole.CDU: 4,
            AgentRole.ADU: 4,
            AgentRole.CDS: 3,
            AgentRole.SOA: 3,
            AgentRole.CDG: 2,
            AgentRole.CAP: 1,
        }.get(role, 0)

    def _execute_comm_action(self, agent_id: int, comm_type: int) -> None:
        """Execute a single communication action by placing a message in the outbox."""
        try:
            agent = self.agents[agent_id]
        except (IndexError, KeyError):
            return

        # Comm channels
        comm_none = getattr(self, "COMM_NONE", 0)
        comm_status = getattr(self, "COMM_STATUS", 1)
        comm_help = getattr(self, "COMM_HELP", 2)

        if comm_type == comm_none:
            return
        if comm_type == comm_status:
            agent.broadcast_status()
            return
        if comm_type == comm_help:
            agent.request_help("general")
            # Count help requests this step for ally_assist event
            self._help_requests = int(getattr(self, "_help_requests", 0)) + 1
            return
        # For now, other comm types are not supported; target is ignored
        return

    def _execute_comm_actions(
        self,
        agent_id: int,
        comm_types: list[int] | np.ndarray,
        comm_targets: list[int] | np.ndarray,
    ) -> None:
        """Execute multiple communication actions for an agent in one step.

        Any slot with comm_type == COMM_NONE is ignored.
        """
        if not isinstance(comm_types, (list, np.ndarray)) or not isinstance(
            comm_targets, (list, np.ndarray)
        ):
            return
        # Ensure equal lengths
        n = min(len(comm_types), len(comm_targets))
        for k in range(n):
            try:
                ctype = int(comm_types[k])
            except (TypeError, ValueError, IndexError) as exc:
                logging.getLogger(__name__).debug(
                    "Invalid comm slot for agent %s at %s: %s", agent_id, k, exc
                )
                continue
            # Skip NONE
            if ctype == self.COMM_NONE:
                continue
            self._execute_comm_action(agent_id, ctype)

    def _agent_fire_at(self, shooter_id: int, target_id: int) -> None:
        """Resolve a simple fire action: consume ammo/energy, apply damage if in range of vision.

        This is a lightweight placeholder for combat. If the target is within the shooter's
        vision range (line of sight approximated by Euclidean distance <= vision_range),
        apply a fixed damage and ammo/energy cost. No friendly-fire prevention here since
        enemies are not distinguished in this simplified environment.
        """
        try:
            shooter = self.agents[shooter_id]
            target = self.agents[target_id]
        except (IndexError, KeyError):
            return
        # Costs and damage
        ammo_cost = 5.0
        energy_cost = 1.0
        damage = 10.0
        # Check resources
        if shooter.get_ammunition() <= 0 or shooter.get_energy() <= 0:
            return
        # Range check: within vision_range distance
        sx, sy = shooter.get_position()
        tx, ty = target.get_position()
        dist = float(np.linalg.norm(np.asarray([sx - tx, sy - ty])))
        if dist > float(getattr(shooter, "vision_range", 0.0)):
            return
        # Apply effects and mark that the shooter fired this step
        shooter.state.consume_ammunition(ammo_cost)
        shooter.state.consume_energy(energy_cost)
        target.state.take_damage(damage)
        with contextlib.suppress(Exception):
            if not hasattr(self, "_fired_this_step"):
                self._fired_this_step = set()
            self._fired_this_step.add(int(shooter_id))

    def step(self, action: dict[str, dict[str, Any]]) -> tuple:  # noqa: C901, PLR0912, PLR0915
        """Execute a step in the environment for all agents.

        The `action` parameter is a dictionary mapping agent IDs (e.g., "agent_0") to their
        respective actions. Each action is itself a dictionary with keys for "elementary_act",
        "combat", "comm", and "c2" as per the defined action schema.
        Returns a tuple of (observations, rewards, terminations, truncations, info).
        """
        timing_start("step")

        # Increment step counter
        self.current_step += 1
        # Reset per-step firing record (mission shaping uses this)
        if hasattr(self, "_fired_this_step"):
            self._fired_this_step.clear()
        # Reset per-step help request counter
        self._help_requests = 0

        # Store previous information for reward calculation
        prev_info = self._get_info()

        # Reset C2 step metrics
        self._order_change_count = 0

        # Execute actions for all agents (combat + comm parsed first). C2 GiveOrder now decentralized.
        pending_orders: list[tuple[int, int, list[int]]] = []  # (agent_id, give_order_flag, orders_vector)
        for agent_id, agent in enumerate(self.agents):
            agent_key = f"agent_{agent_id}"
            agent_action = action.get(
                agent_key,
                {
                    "elementary_act": SoldierElementaryAct.MOVE.value,
                    "combat": {"move": Moves.IDLE.value, "fire_enemy": self.num_agents},
                    "c2": {
                        "give_order": 0,
                        "orders": [self.MISSION_NO_CHANGE_SENTINEL] * self.num_agents,
                    },
                },
            )
            # Defaults
            elementary_act_val = SoldierElementaryAct.MOVE.value
            comm_types = [self.COMM_NONE] * self.MAX_COMMS_PER_STEP
            comm_targets = [self.num_agents] * self.MAX_COMMS_PER_STEP  # sentinel (unused)
            move_val = Moves.IDLE.value
            fire_enemy = self.num_agents
            give_order_flag = 0
            orders_vector = [self.MISSION_NO_CHANGE_SENTINEL] * self.num_agents

            # Parse nested action schema
            if isinstance(agent_action, dict):
                # Elementary act selection (tactical behavior)
                ea = agent_action.get("elementary_act", SoldierElementaryAct.MOVE.value)
                if isinstance(ea, (int, np.integer)):
                    elementary_act_val = int(ea)

                combat = agent_action.get("combat")
                if isinstance(combat, dict):
                    mv = combat.get("move", Moves.IDLE.value)
                    fe = combat.get("fire_enemy", self.num_agents)
                    if isinstance(mv, (int, np.integer)):
                        move_val = int(mv)
                    if isinstance(fe, (int, np.integer)):
                        fire_enemy = int(fe)

                comm_field = agent_action.get("comm")
                if isinstance(comm_field, dict):
                    ct = comm_field.get("types", comm_types)
                    cg = comm_field.get("targets", comm_targets)
                    if isinstance(ct, (list, np.ndarray)):
                        comm_types = [int(x) for x in list(ct)]
                    if isinstance(cg, (list, np.ndarray)):
                        comm_targets = [int(x) for x in list(cg)]

                c2_field = agent_action.get("c2")
                if isinstance(c2_field, dict):
                    gv = c2_field.get("give_order", 0)
                    ov = c2_field.get("orders", orders_vector)
                    if isinstance(gv, (int, np.integer)):
                        give_order_flag = int(gv)
                    if isinstance(ov, (list, np.ndarray)) and len(ov) == self.num_agents:
                        try:
                            orders_vector = [int(x) for x in list(ov)]
                        except (TypeError, ValueError):  # pragma: no cover - defensive
                            orders_vector = [self.MISSION_NO_CHANGE_SENTINEL] * self.num_agents

            pending_orders.append((agent_id, give_order_flag, orders_vector))

            # Set elementary act on agent (tactical behavior)
            try:
                if 0 <= elementary_act_val < len(SoldierElementaryAct):
                    agent.current_elementary_act = SoldierElementaryAct(elementary_act_val)
            except (ValueError, AttributeError):
                agent.current_elementary_act = SoldierElementaryAct.MOVE

            # Store actions for reward shaping
            agent.last_move_action = move_val
            agent.last_fire_action = fire_enemy

            # Execute communication actions in parallel (place in outbox), then move
            with contextlib.suppress(Exception):
                self._execute_comm_actions(agent_id, comm_types, comm_targets)

            # Execute combat: move then (optional) fire
            # Mark movement flag for mission-action intrinsic shaping
            try:
                moved_flag = move_val in (Moves.FORWARD.value, Moves.BACKWARD.value)
                agent.moved_this_step = moved_flag
            except AttributeError:  # pragma: no cover - defensive
                pass
            agent.move(move_val)
            if isinstance(fire_enemy, (int, np.integer)) and 0 <= int(fire_enemy) < self.num_agents:
                with contextlib.suppress(Exception):
                    self._agent_fire_at(agent_id, int(fire_enemy))

        # Apply decentralized GiveOrder actions via RobotBase.issue_orders
        if pending_orders:
            c2_context = C2OrdersMixin.C2Context(
                agents=self.agents,
                doctrine=MISSION_DERIVATION_DOCTRINE,
                mission_no_change_sentinel=self.MISSION_NO_CHANGE_SENTINEL,
                current_step=self.current_step,
                stability_window=20,
            )
            changes = 0
            for issuer_id, give_flag, orders_vec in pending_orders:
                if not give_flag:
                    continue
                issuer = self.agents[issuer_id]
                changes += issuer.issue_orders(orders_vec, context=c2_context)
            self._order_change_count = changes

        # Process communications between agents (now includes mission assignments)
        self._process_communications()

        # Get observations, per-agent rewards, termination flags and info
        observation = self._get_obs()
        per_agent_rewards, team_reward = self._compute_rewards(prev_info)
        per_agent_terminated = self._get_terminates()
        per_agent_truncated = self._get_truncates()
        info = self._get_info()

        # Merge per-agent rewards into each agent's info entry so the top-level
        # `info` object remains a mapping of agent keys -> info dict. This
        # preserves the contract expected by tests and wrappers while still
        # providing per-agent reward values.
        for agent_key, rew in per_agent_rewards.items():
            if agent_key in info and isinstance(info[agent_key], dict):
                # Attach a single numeric reward under a consistent key
                try:
                    info[agent_key]["per_agent_reward"] = float(rew)
                except OSError:
                    info[agent_key]["per_agent_reward"] = rew
            else:
                # If the agent key is missing for any reason, create a lightweight
                # entry so downstream code can still find the reward.
                info[agent_key] = {"per_agent_reward": float(rew) if not isinstance(rew, dict) else rew}

        # Determine scalar reward to return depending on selected reward mode
        try:
            if self.reward_mode == "team":
                scalar_reward = float(team_reward)
            elif self.reward_mode == "per_agent":
                scalar_reward = float(per_agent_rewards.get("agent_0", 0.0))
            else:  # both
                scalar_reward = float(team_reward)
        except (AttributeError, TypeError, KeyError):
            scalar_reward = 0.0

        # Global termination/truncation: if any agent has terminated/truncated
        global_terminated = any(bool(v) for v in per_agent_terminated.values())
        global_truncated = any(bool(v) for v in per_agent_truncated.values())

        timing_stop("step")
        timing_log()
        return observation, scalar_reward, global_terminated, global_truncated, info

    # Doctrine mapping & subordinate mission derivation centralized in
    # `mili_env.envs.classes.c2_doctrine.MISSION_DERIVATION_DOCTRINE`.

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
        if not self.enable_visualization:
            return None
            
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

            agent_key = f"agent_{agent_id}"
            agent_action = action.get(agent_key, "N/A") if isinstance(action, dict) else "N/A"

            debug_info = [
                f"=== Agent {agent_id} ===",
                f"Action: {agent_action}",
                f"Position: ({position[0]:.2f}, {position[1]:.2f})",
                f"Angle: {agent.state.angle:.2f}",
                f"Current Tile: {terrain.__class__.__name__}",
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
