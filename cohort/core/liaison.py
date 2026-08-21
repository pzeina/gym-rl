"""Physical message packets and the agent of liaison
(docs/degraded-communications.md §4).

Pure domain state, no RL dependencies:

* :class:`MessagePacket` — one immutable piece of information: canonical
  voice-procedure text plus routing/status metadata. The TEXT is
  authoritative; the captured fields (``payload``) are what the text says,
  frozen at preparation so carriage can never refresh them with live world
  state. A packet has one owner at a time; dispatch moves it, never copies.
* :class:`LiaisonTask` — the temporary carrying duty (NOT a MissionType, not
  in the MICAT one-hot): the suspended tactical mission, the fixed anchor the
  carrier walks toward (a last-known position, never a live beacon), the leg
  (outbound / returning) and the receipt it carries home.

Command positions, not mortal ids: a packet is addressed to the position
its recipient held at creation. :func:`resolve_position` follows succession
so the current holder may receive it; a vacant position means delivery is
impossible and the carrier returns an undeliverable notice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: packet kinds, in the one-hot order the liaison observation block uses
PACKET_KINDS: tuple[str, ...] = ("order", "contact", "acoustic_contact", "sitrep", "done")

#: how many steps after preparation a packet expires, whoever holds it
#: (published through briefing()); delivery after expiry is impossible and
#: the duty ends. A starting hypothesis, not doctrine.
PACKET_TTL = 40

#: packet lifecycle states (§4.1)
STATUSES = ("held", "dispatched", "delivered", "returning", "lost", "expired", "cancelled")


@dataclass
class MessagePacket:
    """One immutable message and its routing bookkeeping."""

    id: int
    kind: str                       # one of PACKET_KINDS
    origin_id: int                  # soldier id of the origin (its position at creation)
    origin_cs: str
    recipient_id: int               # command position addressed (holder id at creation)
    recipient_cs: str
    text: str                       # canonical formatted voice-procedure line
    created_step: int
    source_step: int | None = None  # observation time carried inside (CONTACT/ACOUSTIC)
    ack_required: bool = False      # ORDER: the WILCO/negative receipt returns to origin
    payload: tuple = ()             # frozen captured fields (what the text says)
    status: str = "held"
    holder_id: int | None = None    # who physically holds it (one owner at a time)
    delivered_step: int | None = None
    receipt: bool | None = None     # ORDER: True WILCO / False rejected / None not yet

    def age(self, step: int) -> int:
        return step - self.created_step

    def ttl_remaining(self, step: int) -> int:
        return PACKET_TTL - self.age(step)

    def expired(self, step: int) -> bool:
        return self.age(step) > PACKET_TTL

    def to_record(self) -> dict:
        """Trace/oracle record."""
        return {
            "id": self.id,
            "kind": self.kind,
            "origin": self.origin_cs,
            "recipient": self.recipient_cs,
            "status": self.status,
            "holder": self.holder_id,
            "created": self.created_step,
            "delivered": self.delivered_step,
            "receipt": self.receipt,
            "text": self.text,
        }


@dataclass
class LiaisonTask:
    """The carrying duty a dispatched soldier holds until the cycle ends."""

    packet: MessagePacket
    carrier_id: int
    dispatched_step: int
    anchor: tuple[int, int]           # fixed last-known position of the recipient
    suspended_mission: object = None  # the carrier's tactical mission, restored after
    leg: str = "outbound"             # "outbound" | "returning"
    return_anchor: tuple[int, int] | None = None
    #: watermark of the best (smallest) cell distance to the current leg's
    #: anchor — progress pays only on NEW closure, so walking back and forth
    #: cannot farm it (§6.4)
    best_distance: float = field(default=float("inf"))
    outbound_path: int = 0            # cells walked on each leg (metrics)
    return_path: int = 0

    def current_anchor(self) -> tuple[int, int]:
        if self.leg == "returning" and self.return_anchor is not None:
            return self.return_anchor
        return self.anchor

    def current_target_id(self) -> int:
        """The command position the carrier is walking toward."""
        return self.packet.origin_id if self.leg == "returning" else self.packet.recipient_id


def resolve_position(position_id: int, roster, successions: dict[int, int]):
    """The living holder of a command position, or None if vacant.

    ``successions`` maps a replaced soldier id to the id of its successor
    (recorded by the environment at every succession event); the chain is
    followed until a living holder is found.
    """
    seen: set[int] = set()
    cur = position_id
    while cur is not None and cur not in seen:
        seen.add(cur)
        holder = roster.by_id.get(cur)
        if holder is not None and holder.alive:
            return holder
        cur = successions.get(cur)
    return None
