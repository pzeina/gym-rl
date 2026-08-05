"""Soldiers, the org chart (roster), OpFor, combat, and succession.

The roster holds the friendly org chart: every soldier has a direct leader
and direct subordinates. When a leader dies, command devolves: the designated
deputy (if alive) or the senior living direct subordinate assumes the fallen
leader's position — inheriting their effective rank, their subordinates, and
their standing mission — and the vacancy that promotion leaves in the lower
echelon is filled the same way, recursively. This is what lets a rifleman
end up commanding a squad after heavy casualties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cohort.core.ranks import AUTHORITY, Rank

if TYPE_CHECKING:
    import numpy as np

    from cohort.core.missions import Mission
    from cohort.core.world import Coord, World


@dataclass
class CombatParams:
    """Tunable weapon/vision model, shared by both sides unless overridden."""

    weapon_range: float = 8.0
    base_hit: float = 0.85          # hit probability at point blank
    falloff: float = 0.07           # hit probability lost per cell of distance
    min_hit: float = 0.15
    cover_multiplier: float = 0.5   # target in forest → hit probability halved
    damage: int = 34
    vision_range: float = 10.0
    forest_vision_range: float = 6.0  # spotting range against targets in forest


@dataclass
class Soldier:
    """One friendly agent."""

    id: int
    callsign: str
    rank: Rank
    pos: Coord
    health: int = 100
    ammo: int = 30
    alive: bool = True
    leader_id: int | None = None          # direct superior; None → reports to HQ
    subordinate_ids: list[int] = field(default_factory=list)
    deputy_id: int | None = None          # designated successor among subordinates
    acting_rank: Rank | None = None       # assumed position after succession
    mission: Mission | None = None
    # bookkeeping used by observations / rewards
    prev_pos: Coord = (0, 0)
    fired_this_step: bool = False
    last_sitrep_step: int = -10_000
    reported_enemy_ids: set[int] = field(default_factory=set)
    last_contact_report_step: int = -10_000
    last_order_step: int = -10_000        # when this agent last *received* an order
    last_issued: dict[int, tuple] = field(default_factory=dict)  # sub_id → (mission, obj)

    @property
    def effective_rank(self) -> Rank:
        """Rank of the position currently held (acting rank after succession)."""
        if self.acting_rank is not None and AUTHORITY[self.acting_rank] > AUTHORITY[self.rank]:
            return self.acting_rank
        return self.rank

    @property
    def effective_authority(self) -> int:
        """Authority of the position currently held."""
        return AUTHORITY[self.effective_rank]

    def living_subordinates(self, roster: Roster) -> list[Soldier]:
        """Direct subordinates still alive, in slot order."""
        return [roster.by_id[i] for i in self.subordinate_ids if roster.by_id[i].alive]


@dataclass
class Enemy:
    """One OpFor combatant (environment-controlled)."""

    id: int
    pos: Coord
    health: int = 100
    alive: bool = True
    mode: str = "garrison"          # "garrison" holds near home; "assault" advances
    home: Coord = (0, 0)
    goal: Coord | None = None       # assault objective
    last_seen_player: Coord | None = None
    last_seen_step: int = -10_000
    # bookkeeping for the ground-truth oracle (core/oracle.py) only —
    # never read by observations, rewards, masks, or the OpFor AI itself
    prev_pos: Coord = (0, 0)
    fired_this_step: bool = False


class Roster:
    """The friendly org chart plus id/callsign lookups."""

    def __init__(self, soldiers: list[Soldier]) -> None:
        self.soldiers = soldiers
        self.by_id: dict[int, Soldier] = {s.id: s for s in soldiers}
        self.by_callsign: dict[str, Soldier] = {s.callsign: s for s in soldiers}

    @property
    def living(self) -> list[Soldier]:
        """All living soldiers."""
        return [s for s in self.soldiers if s.alive]

    def root(self) -> Soldier | None:
        """The senior living commander (reports to HQ)."""
        candidates = [s for s in self.living if s.leader_id is None]
        if candidates:
            return max(candidates, key=lambda s: (s.effective_authority, -s.id))
        return None

    def leader_of(self, soldier: Soldier) -> Soldier | None:
        """Direct living leader, or None if soldier reports to HQ / leader dead."""
        if soldier.leader_id is None:
            return None
        leader = self.by_id[soldier.leader_id]
        return leader if leader.alive else None

    # ---------------- succession ---------------- #

    def succeed(self, dead: Soldier) -> list[tuple[Soldier, Soldier]]:
        """Devolve command after ``dead`` falls. Returns [(successor, replaced)].

        The deputy (if living) or senior living direct subordinate assumes the
        dead leader's position: rank (acting), leader, subordinates, and
        standing mission. The vacancy the successor leaves behind in its own
        team is filled recursively, so the chain of command never has holes
        while any subordinate is alive.
        """
        events: list[tuple[Soldier, Soldier]] = []
        self._fill_vacancy(dead, events)
        return events

    def _pick_successor(self, leader: Soldier) -> Soldier | None:
        subs = leader.living_subordinates(self)
        if not subs:
            return None
        if leader.deputy_id is not None:
            deputy = self.by_id[leader.deputy_id]
            if deputy.alive and deputy.id in leader.subordinate_ids:
                return deputy
        return max(subs, key=lambda s: (s.effective_authority, -s.id))

    def _fill_vacancy(self, vacated: Soldier, events: list[tuple[Soldier, Soldier]]) -> Soldier | None:
        successor = self._pick_successor(vacated)
        if successor is None:
            return None
        old_subordinates = [i for i in successor.subordinate_ids if self.by_id[i].alive]
        old_effective_rank = successor.effective_rank
        old_mission = successor.mission

        # Assume the vacated position.
        if AUTHORITY[vacated.effective_rank] > successor.effective_authority:
            successor.acting_rank = vacated.effective_rank
        successor.leader_id = vacated.leader_id
        successor.deputy_id = None
        new_subs = [i for i in vacated.subordinate_ids if i != successor.id and self.by_id[i].alive]
        successor.subordinate_ids = new_subs
        for i in new_subs:
            self.by_id[i].leader_id = successor.id
        if vacated.mission is not None:
            successor.mission = vacated.mission  # mission continuity
        vacated.subordinate_ids = []
        events.append((successor, vacated))

        # Fill the hole the successor left in its old team.
        if old_subordinates:
            placeholder = Soldier(
                id=successor.id,
                callsign=successor.callsign,
                rank=old_effective_rank,
                pos=successor.pos,
                leader_id=successor.id,  # promoted teammate will report to the successor
                subordinate_ids=old_subordinates,
                deputy_id=None,
                mission=old_mission,
            )
            promoted = self._fill_vacancy(placeholder, events)
            if promoted is not None:
                successor.subordinate_ids.append(promoted.id)
        return successor


def resolve_fire(
    shooter_pos: Coord,
    target_pos: Coord,
    target_in_cover: bool,
    distance: float,
    params: CombatParams,
    rng: np.random.Generator,
) -> tuple[bool, int]:
    """Resolve one shot: (hit?, damage). Caller applies damage and death."""
    p = max(params.min_hit, min(0.9, params.base_hit - params.falloff * distance))
    if target_in_cover:
        p *= params.cover_multiplier
    if rng.random() < p:
        return True, params.damage
    return False, 0


def enemy_decide(
    enemy: Enemy,
    visible_players: list[Soldier],
    world: World,
    step: int,
    params: CombatParams,
    rng: np.random.Generator,
) -> tuple[str, Coord | Soldier | None]:
    """Scripted OpFor policy. Returns ("fire", target) | ("move", pos) | ("stay", None).

    Garrison mode: hold near home, engage players on sight, chase briefly.
    Assault mode: advance on the goal, engage players on sight.
    """
    if visible_players:
        target = min(visible_players, key=lambda s: (abs(s.pos[0] - enemy.pos[0]) + abs(s.pos[1] - enemy.pos[1])))
        enemy.last_seen_player = target.pos
        enemy.last_seen_step = step
        d = ((target.pos[0] - enemy.pos[0]) ** 2 + (target.pos[1] - enemy.pos[1]) ** 2) ** 0.5
        if d <= params.weapon_range:
            return "fire", target
        return "move", _step_toward(enemy.pos, target.pos, world, rng)

    # No contact: chase recent sighting, else return home / advance on goal.
    if enemy.last_seen_player is not None and step - enemy.last_seen_step <= 8:
        return "move", _step_toward(enemy.pos, enemy.last_seen_player, world, rng)
    anchor = enemy.goal if (enemy.mode == "assault" and enemy.goal is not None) else enemy.home
    if enemy.pos != anchor:
        return "move", _step_toward(enemy.pos, anchor, world, rng)
    if rng.random() < 0.1:  # idle shuffle around the post
        return "move", _step_toward(enemy.pos, (enemy.home[0] + int(rng.integers(-2, 3)), enemy.home[1] + int(rng.integers(-2, 3))), world, rng)
    return "stay", None


def _step_toward(pos: Coord, goal: Coord, world: World, rng: np.random.Generator) -> Coord:
    """Greedy 4-neighbor step toward goal, avoiding walls; stays put if stuck."""
    options = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nxt = (pos[0] + dx, pos[1] + dy)
        if world.passable(nxt):
            d = (goal[0] - nxt[0]) ** 2 + (goal[1] - nxt[1]) ** 2
            options.append((d, rng.random(), nxt))
    if not options:
        return pos
    options.sort()
    return options[0][2]
