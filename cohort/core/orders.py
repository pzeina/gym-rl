"""Radio traffic: orders, reports, and the episode transcript.

Every command-and-control event in the environment is materialized as a
:class:`Message` with a human-readable text form, appended to a
:class:`Transcript`. The transcript *is* the transparency guarantee: a human
can read the entire command flow of an episode as plain radio traffic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cohort.core.missions import MissionType

#: Sender/recipient id used for higher headquarters (the environment or a
#: human commander injecting orders).
HQ_ID = -1


class MessageKind(Enum):
    """Kinds of radio messages."""

    OPORD = "opord"                    # initial operations order from HQ
    ORDER = "order"                    # leader → subordinate order
    ACK = "ack"                        # subordinate acknowledges an order (auto)
    CONTACT = "contact"                # enemy sighting report (up the chain)
    SITREP = "sitrep"                  # status report (up the chain)
    DONE = "done"                      # mission-complete report (up the chain)
    CASUALTY = "casualty"              # agent down (auto broadcast)
    TAKING_COMMAND = "taking_command"  # succession announcement (auto broadcast)


@dataclass(frozen=True)
class Message:
    """One radio message, with its rendered text form."""

    step: int
    kind: MessageKind
    sender_id: int
    recipient_id: int | None  # None → broadcast to all stations
    text: str
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OrderDirective:
    """A parsed/validated order: recipient must adopt mission (at objective)."""

    issuer_id: int
    recipient_id: int
    mission: MissionType
    objective_id: int | None


class Transcript:
    """Append-only log of all radio traffic in an episode."""

    def __init__(self) -> None:
        self.messages: list[Message] = []

    def add(self, message: Message) -> None:
        """Append one message."""
        self.messages.append(message)

    def since(self, index: int) -> list[Message]:
        """Messages appended at or after ``index`` (for incremental display)."""
        return self.messages[index:]

    def render(self) -> str:
        """Full transcript as radio-log text."""
        return "\n".join(f"[t={m.step:>3}] {m.text}" for m in self.messages)

    def __len__(self) -> int:
        return len(self.messages)
