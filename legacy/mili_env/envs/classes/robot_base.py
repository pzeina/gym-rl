from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pyglet
import torch
from pyglet import shapes

from mili_env.envs.classes.c2_orders import C2OrdersMixin
from mili_env.envs.classes.terrain import EmptyTerrain, GameMap, ObstacleTerrain, Terrain
from mili_env.envs.classes.types_common import (
    AgentRole,
    CommunicationMessage,
    MessageType,
    SoldierElementaryAct,
)

PI_OVER_4 = np.pi / 4
EPSILON_ANGLE = 1e-3

# Communication system constants
CRITICAL_HEALTH_THRESHOLD = 30.0
HELP_HEALTH_THRESHOLD = 50.0
HELP_DISTANCE_MULTIPLIER = 2.0
DEFAULT_ORDER_MAX_AGE = 30.0

class Moves(Enum):
    """Enum class to define robot movement primitives."""

    IDLE = 0
    FORWARD = 1
    BACKWARD = 2
    ROTATE_LEFT = 3
    ROTATE_RIGHT = 4

    @classmethod
    def count(cls) -> int:
        """Get the number of movement primitives."""
        return len(cls.__members__)


# Energy cost mapping (centralized so tests calling agent.move directly still see energy consumption)
ACTION_ENERGY_COSTS: dict[int, float] = {
    Moves.IDLE.value: 0.0,
    Moves.FORWARD.value: 1.0,
    Moves.BACKWARD.value: 1.2,
    Moves.ROTATE_LEFT.value: 0.5,
    Moves.ROTATE_RIGHT.value: 0.5,
}


"""Robot base: movement, sensing & communications (C2 separated via mixin)."""


@dataclass
class AgentState:
    """Data class to represent the known state of an agent."""

    agent_id: int
    position: tuple[float, float]
    health: float
    energy: float
    ammunition: float
    last_seen: float
    role: AgentRole = AgentRole.CAP
    team_id: int = 0
    is_ally: bool = True


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


@dataclass
class RobotConstraints:
    """Data class to store robot constraint parameters."""

    vision_range: float = 100.0
    communication_range: float = 10.0
    max_speed_forward: float = 1.0
    max_speed_backward: float = -0.2
    max_angular_speed: float = PI_OVER_4

    max_health: float = 100.0
    max_energy: float = 100.0
    max_ammunition: float = 100.0


class RobotState:
    """Class to manage the state of the robot."""

    def __init__(self, position: RobotPosition, attributes: RobotAttributes) -> None:
        """Initialize the robot's state with position, orientation, and resources.

        Args:
            position: Robot position containing coordinates and angle
            attributes: Robot attributes containing health, energy, and ammunition values
        """
        self.x = position.x
        self.y = position.y
        self.angle = position.angle
        self.attributes = attributes

    def update_position(self, new_x: float, new_y: float, new_angle: float) -> None:
        """Update the robot's position."""
        self.x: float = new_x
        self.y: float = new_y
        self.angle: float = new_angle

    # Target-relative information removed; targets are now conveyed by missions/orders externally.

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
        """Target concept removed from state; always False (mission-driven externally)."""
        return False

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
                self.attributes.health,
                self.attributes.energy,
                self.attributes.ammunition,
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

    def get_size(self) -> tuple:
        """Get the size of the state tensor."""
        return self.encode().shape


class RobotBase(C2OrdersMixin):
    """Base class for robots."""

    CUT_RAY_THRESHOLD: float = 1.8

    def __init__( # noqa: PLR0913
        self,
        position: RobotPosition,
        attributes: RobotAttributes,
        game_map: GameMap,
        constraints: RobotConstraints,
        agent_id: int = 0,
        role: AgentRole = AgentRole.CAP,
        team_id: int = 0,
    ) -> None:
        """Initialize the robot with its position and attributes."""
        self.state: RobotState = RobotState(position, attributes)
        self.vision_map: np.ndarray = np.full(
            (2 * int(constraints.vision_range) + 1, 2 * int(constraints.vision_range) + 1),
            EmptyTerrain(),
            dtype=Terrain,
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

        # Communication system components
        self.agent_id: int = agent_id
        self.role: AgentRole = role
        self.known_agents: dict[int, AgentState] = {}  # Known states of other agents
        self.message_inbox: list[CommunicationMessage] = []  # Received messages
        self.message_outbox: list[CommunicationMessage] = []  # Messages to send
        self.communication_history: list[CommunicationMessage] = []  # Message history
        self.last_communication_time: float = 0.0
        # Team and mission-processing attributes
        self.team_id: int = team_id
        self.current_elementary_act: SoldierElementaryAct | None = None  # Self-chosen act
        self.last_derived_missions: dict[int, SoldierElementaryAct] = {}  # Acts decided for subordinates
        # C2 decentralization tracking
        self.free_order_credit: bool = False  # One-step waiver for stability penalty after receiving superior order
        self.last_mission_change_step: int = -10**9  # Large negative so first change never penalized

    # C2 order logic now provided by C2OrdersMixin

    def move(self, action: int | np.ndarray) -> float | np.ndarray:
        """Move the robot based on the chosen action."""
        if isinstance(action, np.ndarray):
            return np.array([self._move_single(Moves(a)) for a in action])
        # return self._move_single(Actions(action))
        return self._move_single_discrete(Moves(action))

    def _move_single(self, action: Moves) -> float:
        """Move the robot based on a single action."""
        dx, dy = 0, 0
        new_angle = self.state.angle
        if action == Moves.IDLE:
            dx, dy = 0, 0
        elif action == Moves.FORWARD:
            dx = self.max_speed_forward * np.cos(self.state.angle)
            dy = self.max_speed_forward * np.sin(self.state.angle)
        elif action == Moves.BACKWARD:
            dx = -self.max_speed_backward * np.cos(self.state.angle)
            dy = -self.max_speed_backward * np.sin(self.state.angle)
        elif action == Moves.ROTATE_LEFT:
            new_angle -= self.max_angular_speed
        elif action == Moves.ROTATE_RIGHT:
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
        # Apply energy consumption post-move (terrain-dependent)
        self._apply_energy_cost(action)

        # Target-relative distance no longer tracked in state
        return 0.0

    def _move_single_discrete(self, action: Moves) -> float:
        """Move the robot based on a single action."""
        dx, dy = 0, 0
        new_angle = self.state.angle
        if action == Moves.IDLE:
            dx, dy = 0, 0
        elif action in [Moves.FORWARD, Moves.BACKWARD]:
            if np.cos(self.state.angle) > EPSILON_ANGLE:
                dx = 1
            elif np.cos(self.state.angle) < -EPSILON_ANGLE:
                dx = -1
            if np.sin(self.state.angle) > EPSILON_ANGLE:
                dy = 1
            elif np.sin(self.state.angle) < -EPSILON_ANGLE:
                dy = -1
            if action == Moves.BACKWARD:
                dx, dy = -dx, -dy
        elif action == Moves.ROTATE_LEFT:
            new_angle -= self.max_angular_speed
        elif action == Moves.ROTATE_RIGHT:
            new_angle += self.max_angular_speed
        else:
            error_message = "Invalid action"
            raise ValueError(error_message)

        # Keep the angle within the range [0, 2*pi]
        new_angle = (new_angle + 2 * np.pi) % (2 * np.pi)

        new_x = self.state.x + dx
        new_y = self.state.y + dy

        new_x = max(0, min(self.game_map.width - 1, new_x))
        new_y = max(0, min(self.game_map.height - 1, new_y))

        self.state.update_position(new_x, new_y, new_angle)

        self.rays = self.compute_vision_rays()
        self.update_vision_map()

        # Apply energy consumption post-move (terrain-dependent)
        self._apply_energy_cost(action)

        # Target-relative distance no longer tracked in state
        return 0.0

    def _apply_energy_cost(self, action: Moves) -> None:
        """Consume energy for a movement action.

        This mirrors the environment-side consumption so tests invoking
        RobotBase.move() directly still observe energy decreasing for
        non-idle actions. Energy scaling uses terrain movement cost.
        """
        base_cost = ACTION_ENERGY_COSTS.get(action.value, 0.0)
        if base_cost <= 0.0:
            return
        try:  # pragma: no cover - defensive
            terrain = self.game_map.get_terrain(self.state.x, self.state.y)
            movement_modifier = getattr(terrain, "get_movement_cost", lambda: 1.0)()
        except (AttributeError, RuntimeError):  # pragma: no cover - defensive
            movement_modifier = 1.0
        energy_cost = movement_modifier * base_cost
        self.state.consume_energy(energy_cost)

    def set_position(self, position: tuple[float, float]) -> None:
        """Set the robot's position."""
        new_x, new_y = position
        new_x = max(0, min(self.game_map.width - 1, new_x))
        new_y = max(0, min(self.game_map.height - 1, new_y))
        self.state.update_position(new_x, new_y, self.state.angle)
        self.rays = self.compute_vision_rays()
        self.update_vision_map()

    def set_health(self, health: float) -> None:
        """Set the robot's health."""
        self.state.attributes.health = max(0, min(self.max_health, health))

    def get_health(self) -> float:
        """Get the robot's health."""
        return self.state.get_health()

    def set_energy(self, energy: float) -> None:
        """Set the robot's energy."""
        self.state.attributes.energy = max(0, min(self.max_energy, energy))

    def get_energy(self) -> float:
        """Get the robot's energy."""
        return self.state.get_energy()

    def set_ammunition(self, ammunition: float) -> None:
        """Set the robot's ammunition."""
        self.state.attributes.ammunition = max(0, min(self.max_ammunition, ammunition))

    def get_ammunition(self) -> float:
        """Get the robot's ammunition."""
        return self.state.get_ammunition()

    def get_vision_map(self) -> np.ndarray:
        """Get the robot's vision map."""
        return self.vision_map

    def get_position(self) -> tuple[float, float]:
        """Get the robot's current position."""
        return self.state.x, self.state.y

    def get_terrain_coordinates(self) -> tuple[int, int]:
        """Get the robot's current terrain coordinates."""
        return int(np.floor(self.state.x)), int(np.floor(self.state.y))

    def get_direction(self) -> float:
        """Get the robot's current direction in radians [0, 2pi]."""
        return self.state.angle

    def get_target_direction(self) -> float:
        """Get the robot's target direction in radians [0, 2pi]."""
        # Without target in state, direction to target is undefined; return current heading
        return self.state.angle

    def get_angle_to_target(self) -> float:
        """Get the robot's unsigned angle difference to the target in radians, between 0 and π."""
        # No target -> angle delta is zero by convention
        return 0.0

    def get_distance_to_target(self) -> float:
        """Get the robot's distance to the target."""
        # No target -> distance zero by convention
        return 0.0

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
        angles = np.concatenate(
            [
                np.linspace(direction - PI_OVER_4, direction + PI_OVER_4, 32),  # Higher density in front
                np.linspace(direction + PI_OVER_4, direction + 3 * PI_OVER_4, 16),  # Medium density on sides
                np.linspace(direction - 3 * PI_OVER_4, direction - PI_OVER_4, 16),  # Medium density on sides
            ]
        )

        for angle in angles:
            angle_diff = np.abs(
                (angle - direction + np.pi) % (2 * np.pi) - np.pi
            )  # Difference between ray angle and robot's direction
            direction_factor = 1 - angle_diff / np.pi  # Direction factor based on angle difference
            length: int = int(np.floor(self.vision_range * direction_factor))
            reduction: float = 0.0
            ray_length: int = 0

            for step in range(1, length + 1):
                last_x = int(x + (step - 1) * np.cos(angle))
                last_y = int(y + (step - 1) * np.sin(angle))
                new_x = int(x + step * np.cos(angle))
                new_y = int(y + step * np.sin(angle))

                if new_x == last_x and new_y == last_y:
                    continue
                inbound = 0 <= new_x < self.game_map.width and 0 <= new_y < self.game_map.height

                terrain_visibility = (
                    self.game_map.get_terrain(new_x, new_y).get_properties().visibility if inbound else 1.0
                )
                reduction += 1.0 - terrain_visibility

                if reduction >= self.CUT_RAY_THRESHOLD:
                    break

                ray_length = step

            rays.append((angle, ray_length))
        return rays

    def communicate(self, other_robot: RobotBase) -> bool:
        """Check if another robot is within communication range."""
        x, y = self.state.x, self.state.y
        other_robot_x, other_robot_y = other_robot.get_position()
        distance = np.sqrt((x - other_robot_x) ** 2 + (y - other_robot_y) ** 2)
        return distance <= self.communication_range

    # ==================== Communication System Methods ====================

    def send_message(self, message: CommunicationMessage) -> None:
        """Add a message to the outbox for transmission."""
        message.sender_id = self.agent_id
        message.timestamp = self.last_communication_time
        self.message_outbox.append(message)
        self.communication_history.append(message)

    def receive_message(self, message: CommunicationMessage) -> None:
        """Receive a message and add it to the inbox."""
        if message.receiver_id is None or message.receiver_id == self.agent_id:
            self.message_inbox.append(message)


    # Removed order processing; missions are handled locally per agent

    def report_ennemy_spotted(self, enemy_id: int, enemy_position: tuple[float, float]) -> None:
        """Report spotted enemy to team."""
        enemy_message = CommunicationMessage(
            sender_id=self.agent_id,
            receiver_id=None,  # Broadcast
            message_type=MessageType.ENEMY_SPOTTED,
            timestamp=self.last_communication_time,
            content={
                "agent_id": enemy_id,
                "position": enemy_position,
                "spotted_at": self.last_communication_time,
                "threat_level": "unknown"
            },
            priority=5  # High priority
        )
        self.send_message(enemy_message)

    def _can_give_orders(self, sender_id: int) -> bool:
        """Check if the sender has authority to give orders to this agent."""
        if sender_id in self.known_agents:
            sender_role = self.known_agents[sender_id].role

            # Command hierarchy
            role_hierarchy = {
                # Highest to lowest within French hierarchy
                AgentRole.CDU: 7,
                AgentRole.ADU: 7,  # Acts in place of CDU when needed
                AgentRole.CDS: 6,
                AgentRole.SOA: 6,  # Acts in place of CDS when needed
                AgentRole.CDG: 5,
                AgentRole.CAP: 4,
            }

            sender_rank = role_hierarchy.get(sender_role, 0)
            my_rank = role_hierarchy.get(self.role, 0)

            return sender_rank > my_rank
        return False

    # Removed conflicting order logic

    def broadcast_status(self) -> CommunicationMessage:
        """Create a status update message to broadcast to allies."""
        status_message = CommunicationMessage(
            sender_id=self.agent_id,
            receiver_id=None,  # Broadcast
            message_type=MessageType.STATUS_UPDATE,
            timestamp=self.last_communication_time,
            content={
                "position": self.get_position(),
                "health": self.get_health(),
                "energy": self.get_energy(),
                "ammunition": self.get_ammunition(),
                "role": self.role.value,
                "team": self.team_id,
            },
            priority=2
        )
        self.send_message(status_message)
        return status_message

    def report_ally_spotted(self, ally_id: int, ally_position: tuple[float, float]) -> CommunicationMessage:
        """Report spotted ally to team."""
        ally_message = CommunicationMessage(
            sender_id=self.agent_id,
            receiver_id=None,  # Broadcast
            message_type=MessageType.ALLY_SPOTTED,
            timestamp=self.last_communication_time,
            content={
                "agent_id": ally_id,
                "position": ally_position,
                "spotted_at": self.last_communication_time
            },
            priority=3
        )
        self.send_message(ally_message)
        return ally_message

    def report_enemy_spotted(self, enemy_id: int, enemy_position: tuple[float, float]) -> CommunicationMessage:
        """Report spotted enemy to team."""
        enemy_message = CommunicationMessage(
            sender_id=self.agent_id,
            receiver_id=None,  # Broadcast
            message_type=MessageType.ENEMY_SPOTTED,
            timestamp=self.last_communication_time,
            content={
                "agent_id": enemy_id,
                "position": enemy_position,
                "spotted_at": self.last_communication_time,
                "threat_level": "unknown"
            },
            priority=5  # High priority
        )
        self.send_message(enemy_message)
        return enemy_message

    # Removed give_*_order methods; orders are no longer communicated, missions are implicit

    def request_help(self, help_type: str = "general") -> CommunicationMessage:
        """Request help from nearby allies."""
        help_request = CommunicationMessage(
            sender_id=self.agent_id,
            receiver_id=None,  # Broadcast
            message_type=MessageType.HELP_REQUEST,
            timestamp=self.last_communication_time,
            content={
                "help_type": help_type,
                "position": self.get_position(),
                "health": self.get_health(),
                "energy": self.get_energy(),
                "urgency": "high" if self.get_health() < CRITICAL_HEALTH_THRESHOLD else "normal"
            },
            priority=5  # High priority
        )
        self.send_message(help_request)
        return help_request

    def update_known_agent(self, agent_state: AgentState) -> None:
        """Update the known state of another agent."""
        self.known_agents[agent_state.agent_id] = agent_state

    def get_known_agents(self) -> dict[int, AgentState]:
        """Get all known agent states."""
        return self.known_agents.copy()

    # Orders removed: missions are tracked via superior_mission/unit_mission

    def process_messages(self, current_time: float) -> None:
        """Process all messages in the inbox."""
        self.last_communication_time = current_time

        # Sort messages by priority (higher number = higher priority)
        self.message_inbox.sort(key=lambda msg: msg.priority, reverse=True)

        for message in self.message_inbox:
            self._handle_message(message)

        # Clear processed messages
        self.message_inbox.clear()

    def _handle_message(self, message: CommunicationMessage) -> None:
        """Handle a specific message based on its type."""
        if message.message_type == MessageType.STATUS_UPDATE:
            self._handle_status_update(message)
        elif message.message_type in [MessageType.ALLY_SPOTTED, MessageType.ENEMY_SPOTTED]:
            self._handle_spotted_report(message)
        elif message.message_type == MessageType.HELP_REQUEST:
            self._handle_help_request(message)
        elif message.message_type == MessageType.MISSION_ASSIGN:
            self._handle_mission_assign(message)
        # No ORDER_* messages are supported anymore

    def _handle_status_update(self, message: CommunicationMessage) -> None:
        """Handle status update messages."""
        agent_state = AgentState(
            agent_id=message.sender_id,
            position=message.content["position"],
            health=message.content["health"],
            energy=message.content["energy"],
            ammunition=message.content["ammunition"],
            last_seen=message.timestamp,
            role=AgentRole.from_str(message.content.get("role", "CAP")),
            team_id=int(message.content.get("team", 0)),
            is_ally=(int(message.content.get("team", 0)) == self.team_id),
        )
        self.update_known_agent(agent_state)

    def _handle_spotted_report(self, message: CommunicationMessage) -> None:
        """Handle ally/enemy spotted reports."""
        spotted_agent = AgentState(
            agent_id=message.content["agent_id"],
            position=message.content["position"],
            health=100.0,  # Unknown
            energy=100.0,  # Unknown
            ammunition=100.0,  # Unknown
            last_seen=message.content["spotted_at"],
            team_id=self.team_id if message.message_type == MessageType.ALLY_SPOTTED else -1,
            is_ally=(message.message_type == MessageType.ALLY_SPOTTED)
        )
        self.update_known_agent(spotted_agent)

    def _handle_help_request(self, message: CommunicationMessage) -> None:
        """Handle help requests from other agents."""
        if message.content.get("urgency") == "high":
            # Prioritize helping agents in critical condition
            requester_pos = message.content["position"]
            my_pos = self.get_position()
            distance = np.sqrt((my_pos[0] - requester_pos[0])**2 + (my_pos[1] - requester_pos[1])**2)

            # If close enough and have resources, consider helping
            if (distance <= self.communication_range * HELP_DISTANCE_MULTIPLIER
                and self.get_health() > HELP_HEALTH_THRESHOLD):
                # Could implement automatic help response logic here
                pass

    def _handle_mission_assign(self, message: CommunicationMessage) -> None:
        """Handle elementary act assignment from a superior."""
        mission_name = message.content.get("mission")
        if mission_name is None:
            return
        try:
            parsed = mission_name.upper()
            if parsed in SoldierElementaryAct.__members__:
                self.current_elementary_act = SoldierElementaryAct[parsed]
        except (KeyError, AttributeError):
            return

    # No order aging; missions are state, not messages

    def get_communication_stats(self) -> dict:
        """Get communication system statistics."""
        return {
            "messages_sent": len(self.communication_history),
            "active_orders": 0,
            "known_agents": len(self.known_agents),
            "role": self.role.value,
            "last_communication": self.last_communication_time
        }

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
            dtype=np.float32,
        )
        return torch.tensor(encoded_map, dtype=torch.float32).unsqueeze(0)

    def render_status_bars(self, batch: pyglet.graphics.Batch, x: int, y: int, width: int, height: int) -> None:
        """Render health, energy, and ammunition bars using pyglet."""
        # Health bar
        health_ratio = self.state.attributes.health / self.max_health
        shapes.Rectangle(x, y, width * health_ratio, height, color=(255, 0, 0), batch=batch)
        shapes.Rectangle(x, y, width, height, color=(255, 255, 255), batch=batch)
        pyglet.text.Label("Health", font_size=14, x=x, y=y + height // 2, color=(255, 255, 255, 255), batch=batch)

        # Energy bar
        energy_ratio = self.state.attributes.energy / self.max_energy
        shapes.Rectangle(x, y + height + 5, width * energy_ratio, height, color=(0, 255, 0), batch=batch)
        shapes.Rectangle(x, y + height + 5, width, height, color=(255, 255, 255), batch=batch)
        pyglet.text.Label(
            "Energy", font_size=14,
            x=x, y=y + height + 5 + height // 2,
            color=(255, 255, 255, 255), batch=batch
        )

        # Ammunition bar
        ammunition_ratio = self.state.attributes.ammunition / self.max_ammunition
        shapes.Rectangle(x, y + 2 * (height + 5), width * ammunition_ratio, height, color=(0, 0, 255), batch=batch)
        shapes.Rectangle(x, y + 2 * (height + 5), width, height, color=(255, 255, 255), batch=batch)
        pyglet.text.Label(
            "Ammunition", font_size=14, x=x,
            y=y + 2 * (height + 5) + height // 2, color=(255, 255, 255, 255),batch=batch
        )

    def render_vision_rays(self, batch: pyglet.graphics.Batch, cell_size: int, boundaries: tuple[int, int]) -> None:  # noqa: ARG002
        """Render vision rays using pyglet."""
        x = self.state.x * cell_size + cell_size // 2
        y = self.state.y * cell_size + cell_size // 2
        for angle, length in self.get_vision_rays():
            end_x = x + length * np.cos(angle) * cell_size
            end_y = y + length * np.sin(angle) * cell_size
            start_pos = (x, y)
            end_pos = (end_x, end_y)
            # clamp the end position to the boundaries
            end_pos = (
                max(0, min(end_pos[0], boundaries[0] * cell_size)),
                max(0, min(end_pos[1], boundaries[1] * cell_size)),
            )
            pyglet.graphics.draw(
                2, pyglet.gl.GL_LINES,
                vertices=("v2f", (start_pos[0], start_pos[1], end_pos[0], end_pos[1])),
                color=("c3B", (255, 0, 0, 255, 0, 0))
            )

    def render_robot(self, batch: pyglet.graphics.Batch, cell_size: int) -> None:  # noqa: ARG002
        """Render the robot as an arrow-like object using pyglet."""
        x = self.state.x * cell_size + cell_size // 2
        y = self.state.y * cell_size + cell_size // 2
        angle = self.state.angle
        # Define the points of the triangle (arrow)
        point1 = (x + cell_size * np.cos(angle), y + cell_size * np.sin(angle))
        point2 = (x + cell_size * np.cos(angle + 2 * np.pi / 3), y + cell_size * np.sin(angle + 2 * np.pi / 3))
        point3 = (x + 0.6 * cell_size * np.cos(angle + np.pi), y + 0.6 * cell_size * np.sin(angle + np.pi))
        point4 = (x + cell_size * np.cos(angle - 2 * np.pi / 3), y + cell_size * np.sin(angle - 2 * np.pi / 3))
        verts = [point1, point2, point3, point4]
        flat_verts = [coord for point in verts for coord in point]
        pyglet.graphics.draw(
            4, pyglet.gl.GL_TRIANGLES,
            vertices=("v2f", flat_verts),
            color=("c3B", (255, 0, 0) * 4)
        )

    def render_target(self, batch: pyglet.graphics.Batch, cell_size: int) -> None:  # noqa: ARG002
        """Render the target zone using pyglet."""
        # No-op: target zone not tracked at state level anymore
        return
