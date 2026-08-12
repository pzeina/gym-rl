"""Soldiers, the org chart (roster), OpFor, combat, and succession.

The roster holds the friendly org chart: every soldier has a direct leader
and direct subordinates. When a leader dies, command devolves: the designated
deputy (if alive) or the senior living direct subordinate assumes the fallen
leader's position — inheriting their effective rank, their subordinates, and
their standing mission — and the vacancy that promotion leaves in the lower
echelon is filled the same way, recursively. This is what lets a rifleman
end up commanding a squad after heavy casualties.

The OpFor side has two families:

* the scripted garrison/assault enemy (:func:`enemy_decide`), and
* the BRIQUE armed band (:class:`BriqueBand`) — the PROTERRE manual's threat
  model (p. 9): a flat band of 5-20 with light weapons, no hierarchy, driven
  by a band-level intent machine (LURK / AMBUSH / HARASS / RAID / SCATTER)
  with per-member behavior states, plus hidden traps/mines (:class:`Trap`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cohort.core.ranks import AUTHORITY, Rank
from cohort.core.world import dist

if TYPE_CHECKING:
    import numpy as np

    from cohort.core.missions import Formation, Mission
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
    # --- SUPPORT mechanics ("pas un pas sans appui", ROADMAP P2) ---
    support_cover_accuracy: float = 0.7  # attacker accuracy vs. a supported element,
    #                                      when the attacker is inside the supporter's
    #                                      umbrella (covered movement)
    support_umbrella: float = 8.0        # radius around an in-position supporter within
    #                                      which enemy fire on its supported element is
    #                                      degraded
    focus_fire_bonus: float = 1.15       # per-shot hit multiplier for the second and
    #                                      later friendly shooters at the same enemy in
    #                                      the same step (active support required)
    max_hit: float = 0.95                # hard cap on any final hit probability


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
    human: bool = False                   # a human commander embodied in the sim
    leader_id: int | None = None          # direct superior; None → reports to HQ
    subordinate_ids: list[int] = field(default_factory=list)
    deputy_id: int | None = None          # designated successor among subordinates
    acting_rank: Rank | None = None       # assumed position after succession
    mission: Mission | None = None
    #: element movement stance (A5-3): set on a LEADER by a FORMATION order;
    #: persists until changed; dies with the leader (succession does not
    #: transfer it — the new leader re-forms its element)
    formation: Formation | None = None
    heading: tuple[int, int] = (0, 0)     # last movement direction (unit 4-dir)
    # bookkeeping used by observations / rewards
    prev_pos: Coord = (0, 0)
    fired_this_step: bool = False
    last_sitrep_step: int = -10_000
    reported_enemy_ids: set[int] = field(default_factory=set)
    last_contact_report_step: int = -10_000
    last_order_step: int = -10_000        # when this agent last *received* an order
    last_done_reject_step: int = -10_000  # when a MISSION COMPLETE claim was last
    #                                       rejected — gates the DONE re-claim
    #                                       cooldown (ScenarioSpec.done_cooldown)
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
    mode: str = "garrison"          # "garrison" holds near home; "assault" advances;
    #                                 "brique" → member of a BriqueBand (armed band)
    home: Coord = (0, 0)
    goal: Coord | None = None       # assault objective
    last_seen_player: Coord | None = None
    last_seen_step: int = -10_000
    #: per-member behavior state, set by the band controller each decision
    #: ("posted", "volleying", "sniping", "displacing", "raiding",
    #: "sabotaging", "fleeing"...). Band AI internal state — exposed to the
    #: oracle as enemy-side ground truth, never to blue observations.
    behavior: str = ""
    # bookkeeping for the ground-truth oracle (core/oracle.py) only —
    # never read by observations, rewards, masks, or the OpFor AI itself
    prev_pos: Coord = (0, 0)
    fired_this_step: bool = False


def voice_peers(soldier: Soldier, roster: Roster, voice_range: float) -> list[Soldier]:
    """Peers within shouting distance for trinôme synchronization (A5-4).

    The manual's bond par binôme (pp. 14-15) is commanded "à la voix ou aux
    gestes" — no radio involved. A peer is a living soldier within
    ``voice_range`` cells that belongs to the soldier's own element (its
    direct leader, a sibling under the same leader, or a direct subordinate)
    or to an ADJACENT trinôme (a cousin: both direct leaders are siblings
    under the same superior). Deterministic; sorted by id.
    """
    peers: list[Soldier] = []
    my_leader = roster.by_id.get(soldier.leader_id) if soldier.leader_id is not None else None
    for other in roster.living:
        if other.id == soldier.id:
            continue
        if dist(soldier.pos, other.pos) > voice_range:
            continue
        same_element = (
            (soldier.leader_id is not None and other.leader_id == soldier.leader_id)
            or other.id == soldier.leader_id
            or other.leader_id == soldier.id
        )
        other_leader = roster.by_id.get(other.leader_id) if other.leader_id is not None else None
        adjacent = (
            my_leader is not None
            and other_leader is not None
            and my_leader.id != other_leader.id
            and my_leader.leader_id is not None
            and my_leader.leader_id == other_leader.leader_id
        )
        if same_element or adjacent:
            peers.append(other)
    peers.sort(key=lambda s: s.id)
    return peers


def validate_human_ranks(soldiers: list[Soldier]) -> None:
    """Enforce the humans-outrank-all-non-humans invariant at org build.

    Every human must sit strictly above every non-human by *intrinsic*
    authority — a human embedded below an AI commander would make the AI's
    hard rank guarantees meaningless (the mask cannot constrain a human).
    Raises ``ValueError`` on violation. No humans → vacuously valid.
    """
    humans = [s for s in soldiers if s.human]
    if not humans:
        return
    top_ai = max((AUTHORITY[s.rank] for s in soldiers if not s.human), default=-1)
    for h in humans:
        if AUTHORITY[h.rank] <= top_ai:
            offender = next(
                s for s in soldiers if not s.human and AUTHORITY[s.rank] >= AUTHORITY[h.rank]
            )
            msg = (
                f"Invalid org: human {h.callsign} ({h.rank.name}, authority "
                f"{AUTHORITY[h.rank]}) does not outrank non-human {offender.callsign} "
                f"({offender.rank.name}, authority {AUTHORITY[offender.rank]}) — "
                "humans must outrank all non-humans."
            )
            raise ValueError(msg)


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
        # The superior inherits the successor in the slot it just filled. Without
        # this the promoted agent is unreachable from above: unorderable (masks
        # read living_subordinates), unobserved, and — when the superior falls —
        # not devolved to, which is how an operation ends up with no root (#42).
        if successor.leader_id is not None:
            parent = self.by_id[successor.leader_id]
            if successor.id not in parent.subordinate_ids:
                parent.subordinate_ids.append(successor.id)
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
            # No append here, and none needed: `placeholder.leader_id` is
            # `successor.id`, so the recursive call's own #42 block already files
            # the promoted teammate under `successor`, guarded by `not in`. The
            # unguarded append this replaces double-linked the commonest
            # succession in the game — SL1 falls, TL1's chart reads
            # [TL2, RFN1, RFN1], and the promoted root then observes a phantom
            # subordinate and carries two ORDER slots addressing one agent.
            self._fill_vacancy(placeholder, events)
        return successor


def resolve_fire(
    shooter_pos: Coord,
    target_pos: Coord,
    target_in_cover: bool,
    distance: float,
    params: CombatParams,
    rng: np.random.Generator,
    *,
    modifier: float = 1.0,
) -> tuple[bool, int]:
    """Resolve one shot: (hit?, damage). Caller applies damage and death.

    ``modifier`` multiplies the hit probability after range and cover: the
    environment passes the SUPPORT effects through it (covered movement
    debuffs an attacker, focus fire buffs follow-up shooters). The final
    probability is capped at ``params.max_hit``.
    """
    p = max(params.min_hit, min(0.9, params.base_hit - params.falloff * distance))
    if target_in_cover:
        p *= params.cover_multiplier
    p = min(params.max_hit, p * modifier)
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


# ---------------------------------------------------------------------- #
# BRIQUE: the armed band (manuel PROTERRE, p. 9 "LA MENACE")
# ---------------------------------------------------------------------- #


@dataclass
class Trap:
    """A hidden device (mine / booby trap) laid by the band at reset.

    Harassment "par engagement de moyens limités ... y compris les mines et
    les pièges" (manual p. 9). Damages the FIRST friendly stepping on its
    cell, then is spent and revealed. Ground truth for the oracle from the
    start of the episode; NEVER present in any blue observation — the
    assurance layer's inference target.
    """

    id: int
    pos: Coord
    damage: int = 40
    armed: bool = True
    revealed: bool = False


@dataclass(frozen=True)
class BriqueBandConfig:
    """Tunables of the band-level intent machine (ScenarioSpec.band)."""

    initial_intent: str = "ambush"    # lurk | ambush | harass | raid
    lurk_trigger: float = 12.0        # blue this close to a lurking member → post the ambush
    ambush_range: float = 5.0         # hold fire until a blue unit is this close, then volley
    volley_steps: int = 6             # steps the sprung ambush volleys before going hit-and-run
    harass_shots: int = 2             # shots fired from max range before displacing
    displace_dist: float = 7.0        # displacement leg to a new cover cell (hit-and-run)
    standoff: float = 10.0            # loiter distance from the band objective while probing
    raid_linger: int = 10             # sabotage steps on the raided objective before withdrawing
    raid_period: int = 0              # 0 → never raid; else steps between raids (from HARASS)
    scatter_below: float = 0.3        # break contact only under this strength fraction —
    #                                   a HIGH threshold: low self-preservation by design
    break_contact_dist: float = 12.0  # scattered members this far from every living blue
    #                                   AND the root objective have broken contact for good


def select_band_target(
    shooter_pos: Coord,
    candidates: list[Soldier],
    all_blue: list[Soldier],
    *,
    isolation_radius: float = 6.0,
    wounded_below: int = 50,
) -> Soldier | None:
    """Casualty-maximizing target choice ("actions à fort impact psychologique").

    Preference order: the human commander first, then wounded units, then
    isolated units (no living teammate within ``isolation_radius``), then the
    closest; ids break remaining ties. Pure and deterministic — no RNG.
    """
    if not candidates:
        return None

    def key(s: Soldier) -> tuple:
        nearest_mate = min(
            (dist(s.pos, o.pos) for o in all_blue if o.alive and o.id != s.id),
            default=float("inf"),
        )
        return (
            0 if s.human else 1,
            0 if s.health < wounded_below else 1,
            0 if nearest_mate > isolation_radius else 1,
            dist(shooter_pos, s.pos),
            s.id,
        )

    return min(candidates, key=key)


def _nearest_edge(pos: Coord, world: World) -> Coord:
    """Closest map-edge cell (scatter destination)."""
    x, y = pos
    cands = [(1, y), (world.width - 2, y), (x, 1), (x, world.height - 2)]
    return min(cands, key=lambda c: dist(pos, c))


def _displacement_cell(
    pos: Coord, threat: Coord, leg: float, world: World, rng: np.random.Generator
) -> Coord:
    """New cell ~``leg`` away from ``pos``, biased away from ``threat``,
    preferring cover (forest). Deterministic through ``rng`` only."""
    base = math.atan2(pos[1] - threat[1], pos[0] - threat[0])
    fallback: Coord | None = None
    for _ in range(12):
        ang = base + rng.uniform(-0.9, 0.9)
        cand = (
            int(min(max(round(pos[0] + leg * math.cos(ang)), 1), world.width - 2)),
            int(min(max(round(pos[1] + leg * math.sin(ang)), 1), world.height - 2)),
        )
        if not world.passable(cand) or cand == pos:
            continue
        if world.cover_at(cand):
            return cand
        if fallback is None:
            fallback = cand
    return fallback if fallback is not None else pos


class BriqueBand:
    """A flat armed band (no hierarchy, no leader) with a band-level intent
    machine driving per-member behavior states.

    Intents (manual p. 9 modes of action):

    * ``lurk``    — hold in cover away from blue, avoid detection; posts the
      ambush when blue approaches within ``lurk_trigger``.
    * ``ambush``  — posted at a chokepoint on blue's predicted route; HOLD
      FIRE until a blue unit is within ``ambush_range``, then volley for
      ``volley_steps`` steps and dissolve into hit-and-run.
    * ``harass``  — fire ``harass_shots`` from max range, then displace to a
      new cover cell ("engagement de moyens limités").
    * ``raid``    — move fast onto the objective/installation, linger
      ``raid_linger`` steps (sabotage), then withdraw ("raid à portée
      limitée visant à détruire des moyens de communication, des dépôts").
    * ``scatter`` — break contact toward the map edges; irreversible. Entered
      only below ``scatter_below`` strength (low self-preservation → high
      casualty tolerance).

    Deterministic: every random draw goes through the ``rng`` handed in by
    the environment (``env._rng``).
    """

    def __init__(
        self,
        members: list[Enemy],
        cfg: BriqueBandConfig,
        *,
        objective: Coord | None,
        posts: dict[int, Coord],
    ) -> None:
        self.members = members
        self.cfg = cfg
        self.intent = cfg.initial_intent
        self.objective = objective      # raid target / operating focus (root objective)
        self.posts = posts              # member id → ambush/lurk post
        self.sprung = False             # ambush fired?
        self.spring_step: int | None = None
        self.last_raid_step = 0
        self._raid_arrived: int | None = None
        self._shots: dict[int, int] = {}          # member id → shots since last displace
        self._displace_to: dict[int, Coord] = {}  # member id → displacement destination

    @property
    def strength(self) -> float:
        """Living fraction of the band's original size."""
        return sum(1 for m in self.members if m.alive) / max(1, len(self.members))

    def update(self, step: int, blue_positions: list[Coord]) -> None:
        """Advance the band-level intent machine (call once per env step)."""
        cfg = self.cfg
        living = [m for m in self.members if m.alive]
        if not living:
            return
        if self.intent != "scatter" and self.strength < cfg.scatter_below:
            self.intent = "scatter"  # terminal: the band breaks contact for good
            return
        if self.intent == "lurk" and blue_positions:
            near = min(dist(m.pos, b) for m in living for b in blue_positions)
            if near <= cfg.lurk_trigger:
                self.intent = "ambush"
        if self.intent == "ambush" and not self.sprung:
            # spring when a blue unit walks into the kill zone — or when the
            # ambush is COMPROMISED (any member hit or down): a posted band
            # taking effective fire opens fire rather than dying in place
            compromised = any(not m.alive or m.health < 100 for m in self.members)
            if compromised or any(
                dist(m.pos, b) <= cfg.ambush_range for m in living for b in blue_positions
            ):
                self.sprung = True
                self.spring_step = step
        if (
            self.intent == "ambush"
            and self.sprung
            and self.spring_step is not None
            and step - self.spring_step >= cfg.volley_steps
        ):
            self.intent = "harass"          # the ambush dissolves into hit-and-run
            self.last_raid_step = step      # raid clock starts after the ambush phase
        if (
            self.intent == "harass"
            and cfg.raid_period > 0
            and self.objective is not None
            and step - self.last_raid_step >= cfg.raid_period
        ):
            self.intent = "raid"
            self._raid_arrived = None
        if self.intent == "raid":
            if self._raid_arrived is None:
                if self.objective is not None and any(
                    dist(m.pos, self.objective) <= 2.0 for m in living
                ):
                    self._raid_arrived = step
            elif step - self._raid_arrived >= cfg.raid_linger:
                self.intent = "harass"      # sabotage done: withdraw and resume probing
                self.last_raid_step = step
                for m in living:            # spent counters force an immediate displacement
                    self._shots[m.id] = cfg.harass_shots

    def member_decide(
        self,
        enemy: Enemy,
        visible_players: list[Soldier],
        all_blue: list[Soldier],
        world: World,
        step: int,
        params: CombatParams,
        rng: np.random.Generator,
    ) -> tuple[str, Coord | Soldier | None]:
        """Per-member policy under the band intent.

        Same contract as :func:`enemy_decide`:
        ("fire", target) | ("move", pos) | ("stay", None).
        """
        cfg = self.cfg
        if self.intent == "scatter":
            enemy.behavior = "fleeing"
            edge = _nearest_edge(enemy.pos, world)
            if dist(enemy.pos, edge) <= 1.0:
                return "stay", None
            return "move", _step_toward(enemy.pos, edge, world, rng)

        if self.intent == "lurk":
            post = self.posts.get(enemy.id, enemy.home)
            if dist(enemy.pos, post) <= 1.0:
                enemy.behavior = "hiding"
                return "stay", None
            enemy.behavior = "posting"
            return "move", _step_toward(enemy.pos, post, world, rng)

        if self.intent == "ambush":
            if not self.sprung:
                # ambush discipline: HOLD FIRE even with blue inside weapon
                # range — the volley waits for ambush_range
                post = self.posts.get(enemy.id, enemy.home)
                if dist(enemy.pos, post) <= 1.0:
                    enemy.behavior = "posted"
                    return "stay", None
                enemy.behavior = "posting"
                return "move", _step_toward(enemy.pos, post, world, rng)
            target = select_band_target(enemy.pos, visible_players, all_blue)
            if target is not None:
                enemy.last_seen_player = target.pos
                enemy.last_seen_step = step
                if dist(enemy.pos, target.pos) <= params.weapon_range:
                    enemy.behavior = "volleying"
                    return "fire", target
                enemy.behavior = "closing"
                return "move", _step_toward(enemy.pos, target.pos, world, rng)
            enemy.behavior = "posted"
            return "stay", None

        if self.intent == "raid":
            obj = self.objective if self.objective is not None else enemy.home
            if self._raid_arrived is not None and dist(enemy.pos, obj) <= 2.5:
                enemy.behavior = "sabotaging"   # lingering ON the installation
                return "stay", None
            target = select_band_target(enemy.pos, visible_players, all_blue)
            if target is not None and dist(enemy.pos, target.pos) <= 0.6 * params.weapon_range:
                enemy.behavior = "raiding"      # fight through point-blank threats only
                return "fire", target
            enemy.behavior = "raiding"
            return "move", _step_toward(enemy.pos, obj, world, rng)

        # harass (hit-and-run)
        dest = self._displace_to.get(enemy.id)
        if dest is not None:
            if dist(enemy.pos, dest) <= 1.0:
                self._displace_to.pop(enemy.id, None)
                self._shots[enemy.id] = 0       # new firing position: counter reset
            else:
                enemy.behavior = "displacing"
                return "move", _step_toward(enemy.pos, dest, world, rng)
        target = select_band_target(enemy.pos, visible_players, all_blue)
        if target is not None:
            enemy.last_seen_player = target.pos
            enemy.last_seen_step = step
            if self._shots.get(enemy.id, 0) < cfg.harass_shots:
                if dist(enemy.pos, target.pos) <= params.weapon_range:
                    self._shots[enemy.id] = self._shots.get(enemy.id, 0) + 1
                    enemy.behavior = "sniping"
                    return "fire", target
                enemy.behavior = "stalking"
                return "move", _step_toward(enemy.pos, target.pos, world, rng)
            dest = _displacement_cell(enemy.pos, target.pos, cfg.displace_dist, world, rng)
            self._displace_to[enemy.id] = dest
            enemy.behavior = "displacing"
            return "move", _step_toward(enemy.pos, dest, world, rng)
        if enemy.last_seen_player is not None and step - enemy.last_seen_step <= 12:
            enemy.behavior = "stalking"
            return "move", _step_toward(enemy.pos, enemy.last_seen_player, world, rng)
        anchor = self.objective if self.objective is not None else enemy.home
        if dist(enemy.pos, anchor) > cfg.standoff:
            enemy.behavior = "prowling"
            return "move", _step_toward(enemy.pos, anchor, world, rng)
        enemy.behavior = "hiding"
        return "stay", None
