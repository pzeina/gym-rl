"""Local friendly perception and the hierarchical visual-link graph
(docs/degraded-communications.md §3.7).

Pure domain logic, no RL dependencies. Two predicates and one graph:

* :func:`friendly_visible` — finite range + terrain line of sight, using the
  shipped 360-degree sight abstraction (directional vision is a separate,
  later feature and must not be imported here);
* :func:`voice_audible` — the general low-voice intelligibility predicate,
  distinct from the trinôme ``voice_peers`` eligibility function: nearby
  speech is audible even when the listener is not a valid bounding peer;
* :func:`element_links` — for one organizational element (a leader and its
  living direct subordinates) the set of members with a path to the leader
  over mutual ``friendly_visible`` edges. Sibling relay is allowed; a
  different element can never be an unplanned relay because the graph is
  built per element.

Cohesion is a tactical constraint and an observation, never physics: nothing
here masks a move.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cohort.core.acoustics import VISUAL_LINK_RANGE, wall_between
from cohort.core.missions import FORMATION_DEPTH, in_formation
from cohort.core.world import dist

if TYPE_CHECKING:
    from cohort.core.units import Roster, Soldier
    from cohort.core.world import World


def friendly_visible(
    world: World, a: Soldier, b: Soldier, visual_range: float = VISUAL_LINK_RANGE
) -> bool:
    """Can ``a`` perceive teammate ``b`` (and vice versa — the relation is
    symmetric)? Both alive, within ``visual_range`` cells, terrain LOS."""
    if not (a.alive and b.alive):
        return False
    if dist(a.pos, b.pos) > visual_range:
        return False
    return bool(world.line_of_sight(a.pos, b.pos))


def voice_audible(world: World, sender: Soldier, listener: Soldier, voice_range: float) -> bool:
    """Low, intelligible speech: alive, within ``voice_range``, LOS (a wall
    always prevents understanding words even when a muffled cue gets through)."""
    if not (sender.alive and listener.alive):
        return False
    if dist(sender.pos, listener.pos) > voice_range:
        return False
    return bool(world.line_of_sight(sender.pos, listener.pos))


def signal_audible(world: World, sender: Soldier, listener: Soldier, signal_range: float) -> bool:
    """A pre-arranged sound signal carries its fixed code within
    ``signal_range`` unless a wall intervenes (the code is a sound, not a
    sight: forest does not block it, a wall does)."""
    if not (sender.alive and listener.alive):
        return False
    if dist(sender.pos, listener.pos) > signal_range:
        return False
    return not wall_between(world, sender.pos, listener.pos)


def element_links(
    world: World,
    leader: Soldier,
    roster: Roster,
    *,
    detached: frozenset[int] | set[int] = frozenset(),
    visual_range: float = VISUAL_LINK_RANGE,
) -> tuple[list[Soldier], set[int]]:
    """The element's non-detached members and the ids linked to its leader.

    Members are the leader's living direct subordinates minus ``detached``
    ids (an active liaison carrier is explicitly outside its originating
    element — otherwise courier duty is impossible by definition). Two nodes
    share an edge when mutually ``friendly_visible``; a member is linked when
    a path of such edges reaches the leader. Returns ``(members, linked_ids)``
    where ``linked_ids`` always contains the leader itself.
    """
    members = [s for s in leader.living_subordinates(roster) if s.id not in detached]
    nodes = [leader, *members]
    linked: set[int] = {leader.id}
    frontier = [leader]
    while frontier:
        cur = frontier.pop()
        for other in nodes:
            if other.id in linked:
                continue
            if friendly_visible(world, cur, other, visual_range):
                linked.add(other.id)
                frontier.append(other)
    return members, linked


def formation_station(leader: Soldier, member: Soldier) -> tuple[bool | None, float]:
    """Current formation-station status and a normalized formation error.

    ``(None, 0.0)`` when no stance governs the pair (no formation on the
    leader, or the leader has never moved and so has no heading).
    Otherwise ``(in_station, error)`` where ``error`` is 0.0 at station and
    otherwise the member's distance from the leader over twice the formation
    depth, clipped to 1.0 — a transparent proxy for "distance from the valid
    station band", published as such.
    """
    if leader.formation is None or leader.heading == (0, 0):
        return None, 0.0
    ok = in_formation(leader.formation, leader.pos, leader.heading, member.pos)
    if ok:
        return True, 0.0
    d = dist(leader.pos, member.pos)
    return False, min(1.0, max(1.0, d) / (2.0 * FORMATION_DEPTH))
