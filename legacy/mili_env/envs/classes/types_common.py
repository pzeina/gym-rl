"""Shared military C2 / communication types.

Isolated so that `robot_base` (physical behavior) and `c2_orders` (order
processing mixin) can both import without circular dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MessageType(Enum):
    """Communication message types (orders removed; missions handled locally)."""

    STATUS_UPDATE = "status_update"
    ALLY_SPOTTED = "ally_spotted"
    ENEMY_SPOTTED = "enemy_spotted"
    HELP_REQUEST = "help_request"
    MISSION_ASSIGN = "mission_assign"


class AgentRole(Enum):
    """Agent roles (French military hierarchy)."""

    CDU = "cdu"
    ADU = "adu"
    CDS = "cds"
    SOA = "soa"
    CDG = "cdg"
    CAP = "cap"

    @classmethod
    def from_str(cls, value: str) -> AgentRole:  # type: ignore[name-defined]
        """Return enum member from raw string; default to CAP if unknown."""
        try:
            return cls(value)
        except ValueError:
            return cls.CAP


ROLE_LABELS_EN: dict[AgentRole, str] = {
    AgentRole.CDU: "Company Commander",
    AgentRole.ADU: "Deputy Company Commander",
    AgentRole.CDS: "Section Commander",
    AgentRole.SOA: "Deputy Platoon Sergeant",
    AgentRole.CDG: "Squad Leader",
    AgentRole.CAP: "Team Leader",
}


class SoldierElementaryAct(Enum):
    """Atomic soldier-level elementary acts an agent can self-select."""

    MOVE = 0
    POST = 1
    FIRE = 2


class TacticalMission(Enum):
    """Simplified high-level tactical missions (translated & consolidated)."""

    RECON = 0  # Reconnaitre / Eclairer
    SEIZE = 1  # S'emparer de
    DEFEND = 2  # Defendre / Tenir
    MONITOR = 3  # Controler / Surveiller
    SUPPORT = 4  # Appuyer / Couvrir
    ENGAGE = 5  # Neutraliser / Interdire / Fixer
    CORDON = 6  # Boucler


# Role level grouping (approximate): Company(CDU, ADU) / Section(CDS, SOA) / Group(CDG) / Team(CAP)
ROLE_LEVEL: dict[AgentRole, str] = {
    AgentRole.CDU: "company",
    AgentRole.ADU: "company",
    AgentRole.CDS: "section",
    AgentRole.SOA: "section",
    AgentRole.CDG: "group",
    AgentRole.CAP: "team",
}

MISSION_LEVEL_APPLICABILITY: dict[TacticalMission, tuple[str, ...]] = {
    TacticalMission.RECON: ("company", "section", "group", "team"),
    TacticalMission.SEIZE: ("company", "section", "group", "team"),
    TacticalMission.DEFEND: ("company", "section"),
    TacticalMission.MONITOR: ("company", "section", "group", "team"),
    TacticalMission.SUPPORT: ("company", "section", "group", "team"),
    TacticalMission.ENGAGE: ("company", "section", "group", "team"),
    TacticalMission.CORDON: ("company", "section"),
}

TACTICAL_DERIVATION_DOCTRINE: dict[TacticalMission, tuple[TacticalMission, ...]] = {
    TacticalMission.RECON: (TacticalMission.RECON, TacticalMission.MONITOR),
    TacticalMission.SEIZE: (TacticalMission.SEIZE, TacticalMission.ENGAGE, TacticalMission.SUPPORT),
    TacticalMission.DEFEND: (TacticalMission.DEFEND, TacticalMission.SUPPORT, TacticalMission.MONITOR),
    TacticalMission.MONITOR: (TacticalMission.MONITOR, TacticalMission.RECON),
    TacticalMission.SUPPORT: (TacticalMission.SUPPORT, TacticalMission.ENGAGE),
    TacticalMission.ENGAGE: (TacticalMission.ENGAGE, TacticalMission.SUPPORT),
    TacticalMission.CORDON: (TacticalMission.CORDON, TacticalMission.DEFEND, TacticalMission.MONITOR),
}


@dataclass
class CommunicationMessage:
    """Data class for communication messages between agents."""

    sender_id: int
    receiver_id: int | None
    message_type: MessageType
    timestamp: float
    content: dict
    priority: int = 1

    def __post_init__(self) -> None:  # basic validation retained
        """Validate required content keys for each message type."""
        if self.message_type == MessageType.STATUS_UPDATE:
            required = {"position", "health", "energy", "ammunition", "team"}
            if not required.issubset(self.content):  # pragma: no cover - defensive
                msg = "STATUS_UPDATE missing keys"
                raise ValueError(msg)
        elif self.message_type in (MessageType.ALLY_SPOTTED, MessageType.ENEMY_SPOTTED):
            required = {"position", "agent_id"}
            if not required.issubset(self.content):  # pragma: no cover
                msg = "SPOTTED missing keys"
                raise ValueError(msg)
        elif self.message_type == MessageType.MISSION_ASSIGN:
            required = {"mission", "sender_role"}
            if not required.issubset(self.content):  # pragma: no cover
                msg = "MISSION_ASSIGN missing keys"
                raise ValueError(msg)

__all__ = [  # noqa: RUF022 - sorting considered acceptable domain grouping
    "AgentRole",
    "CommunicationMessage",
    "MessageType",
    "MISSION_LEVEL_APPLICABILITY",
    "ROLE_LABELS_EN",
    "ROLE_LEVEL",
    "SoldierElementaryAct",
    "TACTICAL_DERIVATION_DOCTRINE",
    "TacticalMission",
]
