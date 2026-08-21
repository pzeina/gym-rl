"""Tactical acoustic environment (docs/degraded-communications.md §3.6).

One physical :class:`SoundEvent` at its source produces two very different
products: eligible friendlies may understand the semantic voice/signal payload
(that is the communication layer's business, not this module's), and either
side may receive a non-semantic :class:`AcousticCue` — kind, bearing sector,
distance band, confidence, age. The cue NEVER carries the oracle source id,
the exact source cell, message text, or the ``heard_by`` list.

Everything here is deterministic and consumes no randomness: propagation is a
threshold model over a canonicalized Bresenham ray (canonicalized so that
attenuation is symmetric under reversing source and listener), and all tie
breaks are stable by ``(received strength, age, event id)``. These are
published simulation starting hypotheses, not doctrinal constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cohort.core.world import FOREST, WALL, dist

if TYPE_CHECKING:
    from cohort.core.world import Coord, World

# --------------------------------------------------------------------- #
# published parameters (§3.6.2) — every one of these goes to briefing()
# and the run's config, never hidden in the OpFor controller
# --------------------------------------------------------------------- #

#: base acoustic-detection radii by source kind (cells)
MOVEMENT_OPEN_RADIUS = 2.0    # successful MOVE touching OPEN terrain only
MOVEMENT_FOREST_RADIUS = 3.0  # successful MOVE entering or leaving FOREST
VOICE_DETECT_RADIUS = 4.0     # low voice: detectable beyond intelligibility
SIGNAL_RANGE = 6.0            # pre-arranged sound signal: semantic range
SIGNAL_DETECT_RADIUS = 8.0    # ... and its (larger) detection footprint
WEAPON_DETECT_RADIUS = 16.0   # weapon fire, hit or miss
TRAP_DETECT_RADIUS = 12.0     # trap / device activation

#: per-ray attenuation (applied to the remaining detection radius)
FOREST_SOUND_FACTOR = 0.9     # per non-endpoint FOREST cell on the ray
WALL_SOUND_FACTOR = 0.5       # applied ONCE if >= 1 intervening WALL cell

#: cue memory: at most this many freshest/strongest cues per agent, expiring
#: after this many steps
MAX_CUES = 4
SOUND_MEMORY_TTL = 6

#: silent gesture variants of EXECUTE / SYNC_GO (§3.6.5): LOS required
GESTURE_RANGE = 6.0

#: hierarchical visual-link / local-friendly-perception radius (§3.7)
VISUAL_LINK_RANGE = 8.0

#: distance bands for cues (upper bounds; beyond the last is "far") and the
#: representative distance each band contributes to an estimated anchor
DISTANCE_BAND_EDGES = (4.0, 10.0)
BAND_ANCHOR_DISTANCE = (3.0, 7.0, 13.0)

#: cue kinds, in the one-hot order the observation uses
SOUND_KINDS = ("movement", "voice", "signal", "weapon_fire", "trap")
#: cue side attributions, in one-hot order (listener attribution, not oracle)
CUE_SIDES = ("friendly", "hostile", "unknown")


def published_parameters() -> dict:
    """The full acoustic model as JSON-ready facts (briefing / provenance)."""
    return {
        "movement_open_radius": MOVEMENT_OPEN_RADIUS,
        "movement_forest_radius": MOVEMENT_FOREST_RADIUS,
        "voice_detect_radius": VOICE_DETECT_RADIUS,
        "signal_range": SIGNAL_RANGE,
        "signal_detect_radius": SIGNAL_DETECT_RADIUS,
        "weapon_detect_radius": WEAPON_DETECT_RADIUS,
        "trap_detect_radius": TRAP_DETECT_RADIUS,
        "forest_sound_factor": FOREST_SOUND_FACTOR,
        "wall_sound_factor": WALL_SOUND_FACTOR,
        "max_cues": MAX_CUES,
        "sound_memory_ttl": SOUND_MEMORY_TTL,
        "gesture_range": GESTURE_RANGE,
        "visual_link_range": VISUAL_LINK_RANGE,
        "distance_band_edges": list(DISTANCE_BAND_EDGES),
        "band_anchor_distance": list(BAND_ANCHOR_DISTANCE),
    }


# --------------------------------------------------------------------- #
# events and cues
# --------------------------------------------------------------------- #


@dataclass
class SoundEvent:
    """One immutable physical sound at its source.

    ``pos``, ``side``, ``message_index`` and the listener lists are
    trace/oracle material only — they must never enter an agent observation
    or an :class:`AcousticCue`.
    """

    id: int
    step: int
    pos: Coord
    side: str                      # "friendly" | "hostile" (trap: triggering side)
    kind: str                      # one of SOUND_KINDS
    base_radius: float
    source: str | None = None          # oracle-only source tag (callsign / "E<id>")
    message_index: int | None = None   # transcript index of a semantic payload
    # audit metadata, filled at delivery (trace/oracle only):
    heard_by: list[str] = field(default_factory=list)              # semantic listeners
    detected_by_friendly: list[tuple[str, float]] = field(default_factory=list)
    detected_by_hostile: list[tuple[int, float]] = field(default_factory=list)

    def to_record(self) -> dict:
        """Trace/oracle record (full ground truth)."""
        return {
            "id": self.id,
            "step": self.step,
            "pos": list(self.pos),
            "side": self.side,
            "source": self.source,
            "kind": self.kind,
            "radius": self.base_radius,
            "heard_by": list(self.heard_by),
            "detected_by_friendly": [cs for cs, _ in self.detected_by_friendly],
            "detected_by_hostile": [eid for eid, _ in self.detected_by_hostile],
        }


@dataclass
class AcousticCue:
    """The coarse, non-semantic product of a detected sound.

    Contains ONLY what §3.6.3 allows: kind, attributed side, eight-way
    bearing sector, distance band, received-strength band, and (via
    ``event_step``) age. ``event_id`` is kept for the stable truncation
    order and never enters an observation.
    """

    kind: str            # one of SOUND_KINDS
    side: str            # "friendly" | "hostile" | "unknown" — attribution
    bearing: int         # 0..7 sector (E, SE, S, SW, W, NW, N, NE)
    distance_band: int   # 0 near, 1 medium, 2 far
    strength: float      # quantized received-strength band in (0, 1]
    event_step: int
    event_id: int

    def age(self, step: int) -> int:
        return step - self.event_step

    def ttl_remaining(self, step: int) -> int:
        return SOUND_MEMORY_TTL - self.age(step)

    def key(self) -> tuple:
        """Identity for report-novelty adjudication: what the report carries."""
        return (self.kind, self.bearing, self.distance_band, self.event_step)


# --------------------------------------------------------------------- #
# propagation
# --------------------------------------------------------------------- #


def _ray_cells(a: Coord, b: Coord) -> list[Coord]:
    """Non-endpoint cells of the Bresenham ray from ``a`` to ``b``.

    Same traversal family as ``World.line_of_sight`` — but kept separate:
    sight and sound are not the same sensor.
    """
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    cells: list[Coord] = []
    while (x, y) != (x1, y1):
        if (x, y) != a:
            cells.append((x, y))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


def effective_radius(world: World, src: Coord, dst: Coord, base: float) -> float:
    """Detection radius left after terrain loss along the src→dst ray.

    Every non-endpoint FOREST cell multiplies the remaining radius by
    ``FOREST_SOUND_FACTOR``; one or more intervening WALL cells apply
    ``WALL_SOUND_FACTOR`` exactly once. The ray is canonicalized (endpoints
    ordered) so the result is symmetric under reversing source and listener.
    """
    a, b = (src, dst) if (src[0], src[1]) <= (dst[0], dst[1]) else (dst, src)
    radius = base
    saw_wall = False
    for cell in _ray_cells(a, b):
        t = world.grid[cell[1], cell[0]]
        if t == FOREST:
            radius *= FOREST_SOUND_FACTOR
        elif t == WALL:
            saw_wall = True
    if saw_wall:
        radius *= WALL_SOUND_FACTOR
    return radius


def wall_between(world: World, src: Coord, dst: Coord) -> bool:
    """Is there >= 1 intervening WALL cell on the (canonicalized) ray?

    A wall always prevents understanding words or seeing a gesture, even
    when a muffled cue is still detected.
    """
    a, b = (src, dst) if (src[0], src[1]) <= (dst[0], dst[1]) else (dst, src)
    return any(world.grid[c[1], c[0]] == WALL for c in _ray_cells(a, b))


def received_strength(world: World, src: Coord, dst: Coord, base: float) -> float | None:
    """Quantized received-strength band, or None when not detected.

    Detected when euclidean distance <= the attenuated effective radius.
    The margin is quantized to three bands (1/3, 2/3, 1) so the cue carries
    a band, never a fine-grained rangefinder.
    """
    d = dist(src, dst)
    radius = effective_radius(world, src, dst, base)
    if radius <= 0.0 or d > radius:
        return None
    margin = max(0.0, min(1.0, 1.0 - d / radius))
    return (math.floor(min(margin, 0.999) * 3) + 1) / 3.0


def bearing_sector(listener: Coord, source: Coord) -> int:
    """Eight-way bearing sector from listener to source: 0=E, 1=SE ... 7=NE.

    Grid y grows south, so SE is (+x, +y). Coincident points read sector 0.
    """
    dx = source[0] - listener[0]
    dy = source[1] - listener[1]
    if dx == 0 and dy == 0:
        return 0
    angle = math.atan2(dy, dx)  # y-south grid: positive = southward
    return round(angle / (math.pi / 4)) % 8


def distance_band(d: float) -> int:
    """0 near, 1 medium, 2 far (edges in DISTANCE_BAND_EDGES)."""
    for i, edge in enumerate(DISTANCE_BAND_EDGES):
        if d <= edge:
            return i
    return len(DISTANCE_BAND_EDGES)


_SECTOR_UNIT = tuple(
    (math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)
)


def estimated_anchor(listener: Coord, bearing: int, band: int, world: World) -> Coord:
    """Frozen investigation anchor built ONCE from a cue's coarse fields.

    Deliberately reconstructed from the bearing sector center and the band's
    representative distance — never from the true source cell — and clamped
    on the map. It ages where it was built and never follows the source.
    """
    ux, uy = _SECTOR_UNIT[bearing % 8]
    r = BAND_ANCHOR_DISTANCE[min(band, len(BAND_ANCHOR_DISTANCE) - 1)]
    x = int(min(max(round(listener[0] + r * ux), 0), world.width - 1))
    y = int(min(max(round(listener[1] + r * uy), 0), world.height - 1))
    return (x, y)


def movement_radius(world: World, prev_pos: Coord, new_pos: Coord) -> float:
    """Base radius of a movement event: the noisier endpoint terrain wins."""
    touches_forest = world.cover_at(prev_pos) or world.cover_at(new_pos)
    return MOVEMENT_FOREST_RADIUS if touches_forest else MOVEMENT_OPEN_RADIUS


def build_cue(
    world: World,
    listener_pos: Coord,
    event: SoundEvent,
    side_attribution: str,
) -> AcousticCue | None:
    """The coarse cue ``listener`` receives from ``event``, or None.

    ``side_attribution`` is the LISTENER's attribution (see §3.6.3), decided
    by the caller from what the listener semantically received or currently
    perceives — never the oracle truth.
    """
    strength = received_strength(world, event.pos, listener_pos, event.base_radius)
    if strength is None:
        return None
    return AcousticCue(
        kind=event.kind,
        side=side_attribution,
        bearing=bearing_sector(listener_pos, event.pos),
        distance_band=distance_band(dist(listener_pos, event.pos)),
        strength=strength,
        event_step=event.step,
        event_id=event.id,
    )


def prune_cues(cues: list[AcousticCue], step: int) -> list[AcousticCue]:
    """Expire aged cues and keep the MAX_CUES freshest/strongest.

    Stable order by ``(received strength desc, age asc, event id asc)`` —
    the §3.6.2 tie rule.
    """
    fresh = [c for c in cues if c.age(step) <= SOUND_MEMORY_TTL]
    fresh.sort(key=lambda c: (-c.strength, c.age(step), c.event_id))
    return fresh[:MAX_CUES]
