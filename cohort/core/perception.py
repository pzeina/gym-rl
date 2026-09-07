"""Per-observer visual-contact records for the seam (epistream HOST_REQUESTS).

Sight's analog of the acoustic cue (:mod:`cohort.core.acoustics`): the coarse,
listener-attributed product of seeing hostiles, held per observer in a bounded
memory with the same discipline cues have — and, like a cue, it NEVER carries
the true cell, the enemy ids, or the observer's exact range.

Epistemic status, stated so it cannot be blurred: unlike acoustic cues, which
enter the observation vector and are therefore memory the agent itself acts
on, this record is HOST-SIDE TELEMETRY ONLY. Nothing here is read by
observations, rewards, action masks or the OpFor — the record coarsens what
the observer's live enemy slots already showed it, so an external monitor can
read what an observer perceived without touching the oracle. Removing this
module changes no agent's behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from cohort.core.acoustics import bearing_sector, distance_band
from cohort.core.world import dist

#: bounded memory, deliberately the acoustic-cue discipline (§3.6.2): at most
#: this many freshest contacts per observer, expiring after this many steps
MAX_VISUAL_CONTACTS = 4
VISUAL_CONTACT_TTL = 6

#: count bands (upper bounds; beyond the last is "several"): 1, 2-3, 4+.
#: A band, not a count — a glimpsed group does not yield an exact strength.
COUNT_BAND_EDGES = (1, 3)


def count_band(n: int) -> int:
    """0 lone, 1 pair/trio, 2 several."""
    for i, edge in enumerate(COUNT_BAND_EDGES):
        if n <= edge:
            return i
    return len(COUNT_BAND_EDGES)


@dataclass
class VisualContact:
    """The coarse product of one observer seeing hostiles in one sector.

    Carries ONLY the bounded-observer fields: eight-way bearing sector,
    distance band, count band, and the step it was formed (age follows).
    There is deliberately NO confidence field: the host exposes what it
    computes, and a sight confidence it does not model would be fabrication —
    the distance band is the coarseness.
    """

    bearing: int        # 0..7 sector (E, SE, S, SW, W, NW, N, NE)
    distance_band: int  # 0 near, 1 medium, 2 far (acoustic band edges)
    count_band: int     # 0 lone, 1 pair/trio, 2 several
    formed_step: int

    def age(self, step: int) -> int:
        return step - self.formed_step

    def key(self) -> tuple:
        """One remembered contact per (sector, band) cell."""
        return (self.bearing, self.distance_band)


def observe_contacts(
    observer_pos: tuple[int, int], enemy_positions: list[tuple[int, int]], step: int
) -> list[VisualContact]:
    """Coarsen one step's visible enemies into per-(sector, band) contacts."""
    groups: dict[tuple[int, int], int] = {}
    for pos in enemy_positions:
        cell = (bearing_sector(observer_pos, pos), distance_band(dist(observer_pos, pos)))
        groups[cell] = groups.get(cell, 0) + 1
    return [
        VisualContact(bearing=b, distance_band=d, count_band=count_band(n), formed_step=step)
        for (b, d), n in sorted(groups.items())
    ]


def merge_contacts(
    memory: list[VisualContact], fresh: list[VisualContact], step: int
) -> list[VisualContact]:
    """Refresh, expire, truncate — the cue-memory maintenance, for sight.

    A fresh contact replaces the remembered one in its (sector, band) cell;
    entries older than the TTL expire; the freshest MAX_VISUAL_CONTACTS
    survive, ties broken stably by (sector, band).
    """
    by_key = {c.key(): c for c in memory}
    for c in fresh:
        by_key[c.key()] = c
    kept = [c for c in by_key.values() if c.age(step) < VISUAL_CONTACT_TTL]
    kept.sort(key=lambda c: (-c.formed_step, c.key()))
    return kept[:MAX_VISUAL_CONTACTS]
