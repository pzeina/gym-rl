"""CohortEnv — a PettingZoo ParallelEnv for a ranked military cohort.

Agents are identified by their radio callsigns (``TL1``, ``RFN2``...). Each
step every living agent picks one action from the shared catalog under its
rank/state legality mask. Orders and reports become radio messages on the
transcript; a CONTACT report is what shares an enemy sighting with the team.
When a leader falls, command devolves automatically (succession) and the
successor announces it on the net.

A human can speak the same language at any time via :meth:`inject_order`,
e.g. ``env.inject_order("TL1, SEIZE OBJ BRAVO")`` — exactly what agents say
to each other.
"""

from __future__ import annotations

import functools
import math
from typing import Any, ClassVar

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from cohort.config import ScenarioSpec, build_org, get_scenario
from cohort.core import language as lang
from cohort.core.missions import (
    IN_POSITION_RADIUS,
    LOS_REQUIRED,
    POSITION_ANCHORED_FIRE,
    RECON_OBSERVE_STEPS,
    WEAPONS_TIGHT,
    ComplianceContext,
    Mission,
    MissionType,
    compliance,
    derivation_quality,
    is_complete,
    min_hold_authority,
)
from cohort.core.orders import HQ_ID, Message, MessageKind, Transcript
from cohort.core.ranks import Rank
from cohort.core.units import Enemy, Roster, Soldier, enemy_decide, resolve_fire
from cohort.core.world import World, dist
from cohort.env.actions import CATALOG, N_ACTIONS, ActionSpec, compute_mask
from cohort.env.observations import OBS_DIM, AgentView, build_observation
from cohort.env.rewards import RewardConfig, RewardLedger

#: Steps after which an unrefreshed contact report goes stale.
KNOWLEDGE_TTL = 40


class CohortEnv(ParallelEnv):
    """Parallel multi-agent environment with a transparent chain of command."""

    metadata: ClassVar[dict] = {"name": "cohort_v1", "render_modes": ["ansi", "rgb_array"]}

    def __init__(
        self,
        scenario: str | ScenarioSpec = "fireteam",
        render_mode: str | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.spec_cfg = get_scenario(scenario) if isinstance(scenario, str) else scenario
        self.render_mode = render_mode
        self.rewards_cfg = reward_config or RewardConfig()
        self.combat = self.spec_cfg.combat

        org = build_org(self.spec_cfg.org)
        counters: dict[Rank, int] = {}
        self._org = org
        self._callsigns: list[str] = []
        for slot in org:
            counters[slot.rank] = counters.get(slot.rank, 0) + 1
            self._callsigns.append(f"{slot.rank.name}{counters[slot.rank]}")
        self.possible_agents: list[str] = list(self._callsigns)

        self._obs_space = spaces.Dict(
            {
                "observation": spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(N_ACTIONS,), dtype=np.int8),
            }
        )
        self._act_space = spaces.Discrete(N_ACTIONS)

        self._rng = np.random.default_rng()
        self.agents: list[str] = []
        self.world: World | None = None
        self.roster: Roster | None = None
        self.enemies: list[Enemy] = []
        self.transcript = Transcript()
        self.last_messages: list[Message] = []
        self._step_count = 0
        self._team_observe_steps = 0
        self._known_enemies: dict[int, tuple[float, float, int]] = {}  # id → (x, y, step)
        #: per-listener enemy pictures, used only when comm_model="range"
        self._agent_known: dict[str, dict[int, tuple[float, float, int]]] = {}
        self._illegal_actions = 0
        self._episode_outcome: str | None = None  # success | defeat | timeout
        self._last_net_contact_step: int | None = None  # last CONTACT on the net
        self._success_step: int | None = None  # T0: success condition first met
        self._root_done_step: int | None = None  # truthful root-mission DONE step
        self._root_done_callsign: str | None = None

    @property
    def outcome(self) -> str | None:
        """Episode outcome: ``"success"`` | ``"defeat"`` | ``"timeout"``, or
        ``None`` while the episode is still running."""
        return self._episode_outcome

    # ------------------------------------------------------------------ #
    # spaces
    # ------------------------------------------------------------------ #

    @functools.cache  # noqa: B019 - spaces are immutable
    def observation_space(self, agent: str) -> spaces.Dict:
        """Observation space (identical for all agents)."""
        del agent
        return self._obs_space

    @functools.cache  # noqa: B019
    def action_space(self, agent: str) -> spaces.Discrete:
        """Action space (identical for all agents; legality varies via mask)."""
        del agent
        return self._act_space

    # ------------------------------------------------------------------ #
    # reset
    # ------------------------------------------------------------------ #

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        """Start a new episode; returns (observations, infos).

        Note: the OPORD is emitted here with ``step=0``, before any action —
        transcript consumers see HQ traffic "before the episode starts".
        """
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.spec_cfg

        self.world = World.generate(
            cfg.map_size[0],
            cfg.map_size[1],
            [(name, pos) for name, pos in cfg.objectives],
            self._rng,
            forest_density=cfg.forest_density,
            wall_density=cfg.wall_density,
            must_connect=[cfg.spawn],
        )
        if cfg.objective_cover and cfg.root_objective:
            self._prepare_defensive_ground(cfg.root_objective)
        if cfg.observation_concealment and cfg.root_objective:
            self._prepare_observation_posts(cfg.root_objective)
        self._spawn_roster()
        self._spawn_enemies()
        if cfg.sitrep_cadence:
            # reporting doctrine: the SITREP clock starts at step 0 — the
            # first report is owed within sitrep_cadence steps
            for s in self.roster.soldiers:
                s.last_sitrep_step = 0

        self.agents = list(self.possible_agents)
        self.transcript = Transcript()
        self.last_messages = []
        self._step_count = 0
        self._team_observe_steps = 0
        self._known_enemies = {}
        self._agent_known = (
            {cs: {} for cs in self._callsigns} if cfg.comm_model == "range" else {}
        )
        self._illegal_actions = 0
        self._episode_outcome = None
        self._last_net_contact_step = None
        self._success_step = None
        self._root_done_step = None
        self._root_done_callsign = None

        # OPORD from HQ to the senior agent.
        root = self.roster.root()
        objective = self.world.objective_by_name(cfg.root_objective) if cfg.root_objective else None
        root.mission = Mission(
            type=cfg.root_mission,
            objective_id=objective.id if objective else None,
            anchor=objective.pos if objective else root.pos,
            issuer_id=HQ_ID,
            step_assigned=0,
        )
        root.last_order_step = 0
        self._say(
            MessageKind.OPORD,
            HQ_ID,
            root.id,
            lang.format_opord(root.callsign, cfg.root_mission, cfg.root_objective),
        )

        observations = self._all_observations()
        infos = {a: {"components": {}} for a in self.agents}
        return observations, infos

    def _prepare_defensive_ground(self, objective_name: str) -> None:
        """Ring the objective with forest: a defense presumes prepared positions.

        Deterministic (no RNG): every passable open cell at chebyshev distance 2
        from the objective center becomes forest, giving defenders concealment
        with the center kept clear. Random generation may otherwise leave the
        objective bare, reducing a defensive stand to an open-field brawl.
        """
        from cohort.core.world import FOREST, OPEN

        obj = self.world.objective_by_name(objective_name)
        if obj is None:
            return
        ox, oy = obj.pos
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if max(abs(dx), abs(dy)) != 2:
                    continue
                cell = (ox + dx, oy + dy)
                if self.world.in_bounds(cell) and self.world.grid[cell[1], cell[0]] == OPEN:
                    self.world.grid[cell[1], cell[0]] = FOREST

    def _prepare_observation_posts(self, objective_name: str) -> None:
        """Concealed OPs on the observation ring: recon presumes hidden positions.

        Deterministic: at each of the eight compass points ~6 cells from the
        objective center, a small forest patch (the cell + its 4-neighbors,
        where open). Inside forest, spotting range drops below the observation
        radius, so a stealthy approach to a garrisoned objective exists — over
        featureless ground it does not, and the policy learns to abandon the
        task instead (observed on squad_recon_v3).
        """
        from cohort.core.world import FOREST, OPEN

        obj = self.world.objective_by_name(objective_name)
        if obj is None:
            return
        ox, oy = obj.pos
        for k in range(8):
            angle = k * math.pi / 4
            cx = ox + round(6 * math.cos(angle))
            cy = oy + round(6 * math.sin(angle))
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                cell = (cx + dx, cy + dy)
                if self.world.in_bounds(cell) and self.world.grid[cell[1], cell[0]] == OPEN:
                    self.world.grid[cell[1], cell[0]] = FOREST

    def _spawn_roster(self) -> None:
        cells = self._bfs_free_cells(self.spec_cfg.spawn, len(self._org))
        soldiers: list[Soldier] = []
        for idx, slot in enumerate(self._org):
            soldiers.append(
                Soldier(
                    id=idx,
                    callsign=self._callsigns[idx],
                    rank=slot.rank,
                    pos=cells[idx],
                    leader_id=slot.leader,
                )
            )
        for idx, slot in enumerate(self._org):
            if slot.leader is not None:
                soldiers[slot.leader].subordinate_ids.append(idx)
                if slot.deputy:
                    soldiers[slot.leader].deputy_id = idx
        for s in soldiers:
            s.prev_pos = s.pos
        self.roster = Roster(soldiers)

    def _spawn_enemies(self) -> None:
        cfg = self.spec_cfg
        self.enemies = []
        if cfg.opfor_mode == "assault":
            target = self.world.objective_by_name(cfg.root_objective or cfg.objectives[0][0])
            for i in range(cfg.n_enemies):
                pos = self._random_edge_cell(
                    min_dist_from=target.pos, min_dist=cfg.assault_spawn_min_dist
                )
                self.enemies.append(
                    Enemy(id=i, pos=pos, home=pos, goal=target.pos, mode="assault", prev_pos=pos)
                )
            return
        # garrison: majority on the OPORD objective, remainder round-robin
        objectives = self.world.objectives
        root_obj = self.world.objective_by_name(cfg.root_objective) if cfg.root_objective else objectives[0]
        others = [o for o in objectives if o.id != root_obj.id] or [root_obj]
        n_root = max(1, math.ceil(cfg.n_enemies * 0.6)) if len(objectives) > 1 else cfg.n_enemies
        for i in range(cfg.n_enemies):
            obj = root_obj if i < n_root else others[(i - n_root) % len(others)]
            pos = self._random_cell_near(obj.pos, radius=2)
            self.enemies.append(Enemy(id=i, pos=pos, home=pos, mode="garrison", prev_pos=pos))

    def _bfs_free_cells(self, start: tuple[int, int], n: int) -> list[tuple[int, int]]:
        found: list[tuple[int, int]] = []
        seen = {start}
        queue = [start]
        while queue and len(found) < n:
            cell = queue.pop(0)
            if self.world.passable(cell):
                found.append(cell)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt not in seen and self.world.in_bounds(nxt):
                    seen.add(nxt)
                    queue.append(nxt)
        while len(found) < n:  # degenerate map fallback
            found.append(start)
        return found

    def _random_cell_near(self, center: tuple[int, int], radius: int) -> tuple[int, int]:
        for _ in range(50):
            pos = (
                int(center[0] + self._rng.integers(-radius, radius + 1)),
                int(center[1] + self._rng.integers(-radius, radius + 1)),
            )
            if self.world.passable(pos):
                return pos
        return center

    def _random_edge_cell(self, min_dist_from: tuple[int, int], min_dist: float) -> tuple[int, int]:
        w, h = self.world.width, self.world.height
        for _ in range(100):
            side = int(self._rng.integers(4))
            if side == 0:
                pos = (int(self._rng.integers(w)), 1)
            elif side == 1:
                pos = (int(self._rng.integers(w)), h - 2)
            elif side == 2:
                pos = (1, int(self._rng.integers(h)))
            else:
                pos = (w - 2, int(self._rng.integers(h)))
            if self.world.passable(pos) and dist(pos, min_dist_from) >= min_dist:
                return pos
        return (1, 1)

    # ------------------------------------------------------------------ #
    # step
    # ------------------------------------------------------------------ #

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict, dict[str, float], dict[str, bool], dict[str, bool], dict]:
        """Advance the simulation one tick."""
        self._step_count += 1
        step = self._step_count
        cfg = self.rewards_cfg
        ledger = RewardLedger()
        self.last_messages = []
        present = list(self.agents)

        # --- snapshot ---
        for s in self.roster.soldiers:
            s.prev_pos = s.pos
            s.fired_this_step = False
        for e in self.enemies:  # oracle bookkeeping only (core/oracle.py)
            e.prev_pos = e.pos
            e.fired_this_step = False
        prev_dist = {
            s.callsign: self._anchor_distance(s) for s in self.roster.living if s.mission is not None
        }

        # --- friendly actions ---
        enemy_kills: list[tuple[Soldier, Enemy]] = []
        for callsign in present:
            soldier = self.roster.by_callsign[callsign]
            if not soldier.alive or callsign not in actions:
                continue
            self._apply_action(soldier, int(actions[callsign]), ledger, enemy_kills)

        # --- OpFor actions ---
        player_deaths: list[Soldier] = []
        for enemy in [e for e in self.enemies if e.alive]:
            self._enemy_turn(enemy, ledger, player_deaths)

        # --- casualties and succession ---
        for dead in player_deaths:
            # net/umpire convention: the report comes from HQ, not the casualty
            self._say(MessageKind.CASUALTY, HQ_ID, None, lang.format_casualty(dead.callsign))
            for other in self.roster.living:
                ledger.add(other.callsign, "combat", cfg.teammate_death)
            for successor, replaced in self.roster.succeed(dead):
                text = (
                    lang.format_taking_command(successor.callsign, replaced.callsign)
                    if not replaced.alive
                    else lang.format_assuming_position(successor.callsign, replaced.callsign)
                )
                self._say(MessageKind.TAKING_COMMAND, successor.id, None, text)
        if player_deaths:
            self._end_orphaned_supports()

        # --- kill sharing ---
        for shooter, _enemy in enemy_kills:
            for other in self.roster.living:
                if other.id != shooter.id:
                    ledger.add(other.callsign, "combat", cfg.team_kill_share)

        # --- knowledge decay ---
        living_enemy_ids = {e.id for e in self.enemies if e.alive}

        def _fresh(picture: dict[int, tuple[float, float, int]]) -> dict:
            return {
                eid: entry
                for eid, entry in picture.items()
                if eid in living_enemy_ids and step - entry[2] <= KNOWLEDGE_TTL
            }

        self._known_enemies = _fresh(self._known_enemies)
        for callsign in self._agent_known:
            self._agent_known[callsign] = _fresh(self._agent_known[callsign])

        # --- mission progress + compliance + step costs ---
        root_obj = (
            self.world.objective_by_name(self.spec_cfg.root_objective)
            if self.spec_cfg.root_objective
            else None
        )
        views = self._compute_views()
        for callsign in present:
            soldier = self.roster.by_callsign[callsign]
            ledger.add(callsign, "time", cfg.time_penalty)
            if not soldier.alive:
                continue
            ctx = self._compliance_ctx(soldier, prev_dist.get(callsign), views[callsign])
            if soldier.mission is not None:
                if (
                    soldier.mission.type in (MissionType.RECON, MissionType.SCREEN)
                    and ctx.in_position
                ):
                    soldier.mission.observe_steps += 1
                ledger.add(callsign, "compliance", cfg.compliance_weight * compliance(soldier.mission.type, ctx))
            # reporting doctrine: out of contact and past the cadence → overdue
            cadence = self.spec_cfg.sitrep_cadence
            if (
                cadence
                and not views[callsign].visible_enemies
                and step - soldier.last_sitrep_step > cadence
            ):
                ledger.add(callsign, "report", cfg.sitrep_overdue)
            # leader coverage
            if soldier.effective_authority > 0 and soldier.mission is not None:
                subs = soldier.living_subordinates(self.roster)
                if subs:
                    all_tasked = all(sub.mission is not None for sub in subs)
                    ledger.add(callsign, "command", cfg.coverage_bonus if all_tasked else cfg.coverage_gap)

        # team observation progress for RECON / SCREEN campaigns. Each NOVEL
        # step toward the success counter pays observe_progress to the
        # observer (A2/A7 stall-exploit fix): the payout telescopes — it is
        # bounded by the success threshold and cannot be farmed indefinitely.
        if root_obj is not None and self.spec_cfg.root_mission in (
            MissionType.RECON,
            MissionType.SCREEN,
        ):
            for s in self.roster.living:
                if dist(s.pos, root_obj.pos) <= IN_POSITION_RADIUS[MissionType.RECON] and self.world.line_of_sight(
                    s.pos, root_obj.pos
                ):
                    if self._team_observe_steps < 2 * RECON_OBSERVE_STEPS:
                        ledger.add(s.callsign, "compliance", cfg.observe_progress)
                    self._team_observe_steps += 1
                    break

        # --- terminal conditions ---
        # Success does not terminate on the spot: when the root-mission
        # condition is first met (T0) a completion-report grace window opens,
        # giving the root time to transmit MISSION COMPLETE. A truthful root
        # DONE ends the episode that step (root_done_bonus); otherwise it ends
        # as success at T0 + grace_window anyway. Success is locked in at T0 —
        # the speed bonus is computed from T0, so policies that never report
        # keep their success rate and terminal reward.
        if self._check_success(root_obj) and self._success_step is None:
            self._success_step = step  # T0: the window opens
        success_locked = self._success_step is not None
        cohort_wiped = not any(s.alive for s in self.roster.soldiers)
        defeat = cohort_wiped and not success_locked
        root_reported = (
            success_locked
            and self._root_done_step is not None
            and self._root_done_step >= self._success_step
        )
        success = success_locked and (
            root_reported
            or step >= self._success_step + self.spec_cfg.grace_window
            or step >= self.spec_cfg.max_steps
            or cohort_wiped
        )
        truncated_all = step >= self.spec_cfg.max_steps and not success and not defeat
        if success:
            self._episode_outcome = "success"
            speed = cfg.success_speed * max(0.0, 1.0 - self._success_step / self.spec_cfg.max_steps)
            for s in self.roster.living:
                ledger.add(s.callsign, "terminal", cfg.success_team + speed)
            if root_reported and self._root_done_callsign is not None:
                ledger.add(self._root_done_callsign, "terminal", cfg.root_done_bonus)
        elif defeat:
            self._episode_outcome = "defeat"
            for callsign in present:
                ledger.add(callsign, "terminal", cfg.defeat)
        elif truncated_all:
            self._episode_outcome = "timeout"

        # --- assemble returns ---
        episode_over = success or defeat or truncated_all
        observations: dict[str, dict] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict] = {}
        for callsign in present:
            soldier = self.roster.by_callsign[callsign]
            observations[callsign] = self._observe(soldier, views[callsign])
            rewards[callsign] = ledger.total(callsign)
            terminations[callsign] = (not soldier.alive) or success or defeat
            truncations[callsign] = truncated_all
            infos[callsign] = {
                "components": ledger.breakdown(callsign),
                "outcome": self._episode_outcome if episode_over else None,
            }
        self.agents = [a for a in present if not (terminations[a] or truncations[a])]
        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------ #
    # action application
    # ------------------------------------------------------------------ #

    def _apply_action(
        self,
        soldier: Soldier,
        action: int,
        ledger: RewardLedger,
        enemy_kills: list[tuple[Soldier, Enemy]],
    ) -> None:
        cfg = self.rewards_cfg
        spec = CATALOG[action]
        mask = self._mask_for(soldier)
        if not mask[action]:
            self._illegal_actions += 1
            return  # illegal → treated as STAY

        if spec.kind == "move":
            soldier.pos = (soldier.pos[0] + spec.move[0], soldier.pos[1] + spec.move[1])
        elif spec.kind == "fire":
            self._resolve_player_fire(soldier, ledger, enemy_kills)
        elif spec.kind == "contact":
            self._report_contact(soldier, ledger)
        elif spec.kind == "sitrep":
            # under the reporting doctrine the mandated cadence *is* the
            # freshness interval, so a due report is never scored as spam
            interval = self.spec_cfg.sitrep_cadence or cfg.sitrep_interval
            fresh = self._step_count - soldier.last_sitrep_step >= interval
            ledger.add(soldier.callsign, "report", cfg.sitrep_fresh if fresh else cfg.sitrep_spam)
            soldier.last_sitrep_step = self._step_count
            self._say(
                MessageKind.SITREP,
                soldier.id,
                soldier.leader_id,
                lang.format_sitrep(
                    self._addressee(soldier), soldier.callsign, soldier.health, soldier.ammo, soldier.pos
                ),
            )
        elif spec.kind == "done":
            self._report_done(soldier, ledger)
        elif spec.kind == "order":
            self._issue_order(soldier, spec, ledger)

    def _resolve_player_fire(
        self, soldier: Soldier, ledger: RewardLedger, enemy_kills: list[tuple[Soldier, Enemy]]
    ) -> None:
        cfg = self.rewards_cfg
        target = self._nearest_visible_enemy(soldier, within_range=True)
        if target is None:
            return
        soldier.fired_this_step = True
        soldier.ammo -= 1
        d = dist(soldier.pos, target.pos)
        hit, damage = resolve_fire(
            soldier.pos, target.pos, self.world.cover_at(target.pos), d, self.combat, self._rng
        )
        if hit:
            target.health -= damage
            discipline = self._fire_discipline_factor(soldier)
            ledger.add(soldier.callsign, "combat", cfg.hit_enemy * discipline)
            if target.health <= 0:
                target.alive = False
                ledger.add(soldier.callsign, "combat", cfg.kill_enemy * discipline)
                enemy_kills.append((soldier, target))

    def _report_contact(self, soldier: Soldier, ledger: RewardLedger) -> None:
        cfg = self.rewards_cfg
        visible = self._visible_enemies(soldier)
        if not visible:
            return
        new_intel = any(e.id not in self._known_enemies for e in visible)
        for e in visible:
            self._known_enemies[e.id] = (float(e.pos[0]), float(e.pos[1]), self._step_count)
            soldier.reported_enemy_ids.add(e.id)
        if self.spec_cfg.comm_model == "range":
            # the report only reaches stations within earshot: their pictures
            # update; everyone else's stays stale (the sender hears itself)
            for listener in self.roster.living:
                if self._audible_to(listener, soldier.id):
                    picture = self._agent_known.setdefault(listener.callsign, {})
                    for e in visible:
                        picture[e.id] = (float(e.pos[0]), float(e.pos[1]), self._step_count)
        soldier.last_contact_report_step = self._step_count
        self._last_net_contact_step = self._step_count
        ledger.add(soldier.callsign, "report", cfg.contact_new if new_intel else cfg.contact_redundant)
        nearest = visible[0]
        self._say(
            MessageKind.CONTACT,
            soldier.id,
            soldier.leader_id,
            lang.format_contact(self._addressee(soldier), soldier.callsign, len(visible), nearest.pos),
        )

    def _report_done(self, soldier: Soldier, ledger: RewardLedger) -> None:
        cfg = self.rewards_cfg
        mission = soldier.mission
        if mission is None:
            return
        ctx = self._compliance_ctx(soldier, None, self._make_view(soldier))
        root_objective = (
            self.world.objective_by_name(self.spec_cfg.root_objective)
            if self.spec_cfg.root_objective
            else None
        )
        is_root_mission_claim = (
            soldier is self.roster.root()
            and mission.issuer_id == HQ_ID
            and mission.type is self.spec_cfg.root_mission
            and mission.objective_id == (root_objective.id if root_objective else None)
        )
        # The root's OPORD claim reports the *operation*: it is judged against
        # the team success condition (e.g. objective clear AND held by anyone),
        # not against the claimant's personal end state — a commander reports
        # the mission complete when the unit achieved it, wherever it stands.
        truthful = (
            self._check_success(root_objective)
            if is_root_mission_claim
            else is_complete(mission, ctx)
        )
        obj_name = (
            self.world.objectives[mission.objective_id].name if mission.objective_id is not None else None
        )
        self._say(
            MessageKind.DONE,
            soldier.id,
            soldier.leader_id,
            lang.format_done(self._addressee(soldier), soldier.callsign, mission.type, obj_name),
        )
        # the superior answers on the net: the verdict is command traffic, not
        # a secret side effect (a false claimant silently keeping its mission
        # was the one place command state stopped being derivable from traffic)
        leader = self.roster.leader_of(soldier)
        responder_id = leader.id if leader is not None else HQ_ID
        responder_cs = self._addressee(soldier)
        if truthful:
            self._say(
                MessageKind.DONE_CONFIRM,
                responder_id,
                soldier.id,
                lang.format_done_confirm(soldier.callsign, responder_cs, mission.type, obj_name),
            )
            if is_root_mission_claim:
                # truthful root-mission COMPLETE: closes the grace window
                self._root_done_step = self._step_count
                self._root_done_callsign = soldier.callsign
            ledger.add(soldier.callsign, "report", cfg.done_true)
            soldier.mission = None  # standing by for new orders
        else:
            self._say(
                MessageKind.DONE_REJECT,
                responder_id,
                soldier.id,
                lang.format_done_reject(soldier.callsign, responder_cs),
            )
            ledger.add(soldier.callsign, "report", cfg.done_false)

    def _issue_order(self, soldier: Soldier, spec: ActionSpec, ledger: RewardLedger) -> None:
        cfg = self.rewards_cfg
        subs = soldier.living_subordinates(self.roster)
        if spec.order_slot >= len(subs):
            return
        recipient = subs[spec.order_slot]
        objective = (
            self.world.objective_by_name(spec.order_objective) if spec.order_objective else None
        )
        obj_id = objective.id if objective else None
        # unit-targeted SUPPORT: the supported unit is the sibling in slot j
        supported_id: int | None = None
        if spec.order_mission is MissionType.SUPPORT:
            if spec.order_support_slot is None or spec.order_support_slot >= len(subs):
                return
            supported_id = subs[spec.order_support_slot].id

        # out of earshot (comm_model="range"): the transmission goes out but
        # nothing arrives — no mission change, no WILCO, no command credit
        if not self._audible_to(recipient, soldier.id):
            self._assign_mission(
                issuer_id=soldier.id,
                issuer_cs=soldier.callsign,
                recipient=recipient,
                mission_type=spec.order_mission,
                objective_id=obj_id,
                supported_id=supported_id,
            )
            return

        # churn: reissuing the standing order is radio noise, not command
        if (
            recipient.mission is not None
            and recipient.mission.type is spec.order_mission
            and recipient.mission.objective_id == obj_id
            and recipient.mission.extra.get("supported_id") == supported_id
        ):
            ledger.add(soldier.callsign, "command", cfg.order_churn)
            return

        # fresh tasking: subordinate untasked, or the issuer's own mission
        # changed after the subordinate was last ordered (propagation credit)
        fresh_tasking = recipient.mission is None or (
            soldier.mission is not None and soldier.mission.step_assigned > recipient.last_order_step
        )
        if fresh_tasking:
            quality = derivation_quality(soldier.mission.type if soldier.mission else None, spec.order_mission)
            if quality >= 1.0:
                ledger.add(soldier.callsign, "command", cfg.order_preferred)
            elif quality > 0.0:
                ledger.add(soldier.callsign, "command", cfg.order_allowed)
            if (
                obj_id is not None
                and soldier.mission is not None
                and soldier.mission.objective_id == obj_id
            ):
                ledger.add(soldier.callsign, "command", cfg.order_objective_match)
        elif self._step_count - recipient.last_order_step < cfg.order_stability_window:
            # premature re-tasking without new superior intent: churn (the
            # order still applies — commanding stays possible, just costly)
            ledger.add(soldier.callsign, "command", cfg.order_churn)

        self._assign_mission(
            issuer_id=soldier.id,
            issuer_cs=soldier.callsign,
            recipient=recipient,
            mission_type=spec.order_mission,
            objective_id=obj_id,
            supported_id=supported_id,
        )
        soldier.last_issued[recipient.id] = (spec.order_mission, obj_id, supported_id)

    def _assign_mission(
        self,
        issuer_id: int,
        issuer_cs: str,
        recipient: Soldier,
        mission_type: MissionType,
        objective_id: int | None,
        supported_id: int | None = None,
    ) -> bool:
        """Transmit an order; if the recipient hears it, apply it (+ WILCO).

        Under ``comm_model="global"`` every order is heard. Under ``"range"``
        an out-of-earshot recipient never receives the mission: the ORDER
        still lands on the transcript (it was transmitted), but nothing
        changes and no WILCO comes back — silence is the only clue.
        Returns True if the order was received and applied.
        """
        extra: dict = {}
        if mission_type is MissionType.SUPPORT and supported_id is not None:
            supported = self.roster.by_id[supported_id]
            anchor = supported.pos  # dynamic thereafter: tracks the supported soldier
            target = supported.callsign
            extra["supported_id"] = supported_id
        elif objective_id is not None:
            anchor = self.world.objectives[objective_id].pos
            target = self.world.objectives[objective_id].name
        elif mission_type is MissionType.RALLY:
            leader = self.roster.by_id.get(issuer_id)
            anchor = leader.pos if leader is not None else recipient.pos
            target = None
        else:  # HOLD (and any anchor-less mission): anchor where the order was received
            anchor = recipient.pos
            target = None
        self._say(
            MessageKind.ORDER,
            issuer_id,
            recipient.id,
            lang.format_order(issuer_cs, recipient.callsign, mission_type, target),
        )
        if not self._audible_to(recipient, issuer_id):
            return False
        recipient.mission = Mission(
            type=mission_type,
            objective_id=objective_id,
            anchor=anchor,
            issuer_id=issuer_id,
            step_assigned=self._step_count,
            extra=extra,
        )
        recipient.last_order_step = self._step_count
        if self.spec_cfg.auto_ack:
            self._say(
                MessageKind.ACK,
                recipient.id,
                issuer_id,
                lang.format_ack(issuer_cs, recipient.callsign),
            )
        return True

    # ------------------------------------------------------------------ #
    # OpFor
    # ------------------------------------------------------------------ #

    def _enemy_turn(self, enemy: Enemy, ledger: RewardLedger, player_deaths: list[Soldier]) -> None:
        cfg = self.rewards_cfg
        visible_players = [
            s
            for s in self.roster.living
            if self.world.can_spot(
                enemy.pos, s.pos, self.combat.vision_range, self.combat.forest_vision_range
            )
        ]
        act, arg = enemy_decide(enemy, visible_players, self.world, self._step_count, self.combat, self._rng)
        if act == "move" and arg is not None and self.world.passable(arg):
            enemy.pos = arg
        elif act == "fire":
            enemy.fired_this_step = True  # oracle bookkeeping only
            target: Soldier = arg
            d = dist(enemy.pos, target.pos)
            hit, damage = resolve_fire(
                enemy.pos, target.pos, self.world.cover_at(target.pos), d, self.combat, self._rng
            )
            if hit:
                target.health -= damage
                ledger.add(target.callsign, "combat", cfg.took_hit)
                if target.health <= 0 and target.alive:
                    target.alive = False
                    target.health = 0
                    ledger.add(target.callsign, "combat", cfg.death)
                    player_deaths.append(target)

    # ------------------------------------------------------------------ #
    # views, masks, observations, compliance
    # ------------------------------------------------------------------ #

    def _audible_to(self, listener: Soldier, sender_id: int) -> bool:
        """Can ``listener`` hear a transmission from ``sender_id``?

        ``comm_model="global"`` (default): always. ``"range"``: only within
        ``comm_range`` (euclidean). The sender always hears itself. HQ is a
        high-power station: its traffic is always heard, and it always hears
        the root (the root's up-channel reports are adjudicated regardless).
        """
        if self.spec_cfg.comm_model != "range":
            return True
        if sender_id == HQ_ID:
            return True
        sender = self.roster.by_id.get(sender_id)
        if sender is None or sender.id == listener.id:
            return True
        return dist(sender.pos, listener.pos) <= self.spec_cfg.comm_range

    def _visible_enemies(self, soldier: Soldier) -> list[Enemy]:
        visible = [
            e
            for e in self.enemies
            if e.alive
            and self.world.can_spot(
                soldier.pos, e.pos, self.combat.vision_range, self.combat.forest_vision_range
            )
        ]
        visible.sort(key=lambda e: (dist(soldier.pos, e.pos), e.id))
        return visible

    def _nearest_visible_enemy(self, soldier: Soldier, *, within_range: bool) -> Enemy | None:
        for e in self._visible_enemies(soldier):
            if not within_range or dist(soldier.pos, e.pos) <= self.combat.weapon_range:
                return e
        return None

    def _make_view(self, soldier: Soldier) -> AgentView:
        known = (
            self._agent_known.get(soldier.callsign, {})
            if self.spec_cfg.comm_model == "range"
            else self._known_enemies
        )
        cadence = self.spec_cfg.sitrep_cadence
        sitrep_due = (
            min(1.0, max(0.0, (self._step_count - soldier.last_sitrep_step) / cadence))
            if cadence
            else None
        )
        return AgentView(
            visible_enemies=self._visible_enemies(soldier),
            known_enemies=[(x, y) for (x, y, _t) in known.values()],
            step=self._step_count,
            sitrep_due=sitrep_due,
        )

    def _compute_views(self) -> dict[str, AgentView]:
        return {s.callsign: self._make_view(s) for s in self.roster.soldiers}

    def _mask_for(self, soldier: Soldier) -> np.ndarray:
        visible = self._visible_enemies(soldier)
        in_range = any(dist(soldier.pos, e.pos) <= self.combat.weapon_range for e in visible)
        return compute_mask(
            soldier,
            self.roster,
            self.world,
            in_range and soldier.ammo > 0,
            bool(visible),
            order_cooldown=self.spec_cfg.order_cooldown,
            step=self._step_count,
            net_contact_step=self._last_net_contact_step,
        )

    def _observe(self, soldier: Soldier, view: AgentView) -> dict[str, np.ndarray]:
        return {
            "observation": build_observation(soldier, self.roster, self.world, view),
            "action_mask": self._mask_for(soldier),
        }

    def _all_observations(self) -> dict[str, dict]:
        views = self._compute_views()
        return {
            callsign: self._observe(self.roster.by_callsign[callsign], views[callsign])
            for callsign in self.agents
        }

    def _mission_anchor(self, soldier: Soldier) -> tuple[float, float] | None:
        """Current anchor point of the soldier's mission (dynamic for
        RALLY — the leader — and SUPPORT — the supported soldier)."""
        mission = soldier.mission
        if mission is None:
            return None
        anchor = mission.anchor
        if mission.type is MissionType.RALLY:
            leader = self.roster.leader_of(soldier)
            if leader is not None:
                anchor = leader.pos
        elif mission.type is MissionType.SUPPORT:
            supported = self.roster.by_id.get(mission.extra.get("supported_id"))
            if supported is not None and supported.alive:
                anchor = supported.pos
        return anchor

    def _anchor_distance(self, soldier: Soldier) -> float:
        anchor = self._mission_anchor(soldier)
        if anchor is None:
            return 0.0
        return dist(soldier.pos, anchor)

    def _in_mission_position(self, soldier: Soldier, dist_now: float | None = None) -> bool:
        """Is the soldier at its mission station (radius + LOS where required)?

        SUPPORT stations relative to the supported soldier: within radius and
        holding line of sight to *it* (you cannot support what you cannot see).
        """
        mission = soldier.mission
        if mission is None:
            return False
        if dist_now is None:
            dist_now = self._anchor_distance(soldier)
        anchor = self._mission_anchor(soldier)
        in_position = dist_now <= IN_POSITION_RADIUS[mission.type]
        if mission.type in LOS_REQUIRED:
            in_position = in_position and self.world.line_of_sight(
                soldier.pos, (int(anchor[0]), int(anchor[1]))
            )
        return in_position

    def _fire_discipline_factor(self, soldier: Soldier) -> float:
        """Combat-reward multiplier enforcing fire discipline by mission.

        SCREEN is weapons tight: firing earns nothing (and compliance already
        penalizes it). Static postures (OBSERVE/SUPPORT/COVER/DEFEND/DENY/
        HOLD) pay for engagements fought FROM the mission position — chasing
        kills off the position earns nothing. RECON (which may engage, per
        PROTERRE), assault tasks, and untasked agents are free.
        """
        if not self.rewards_cfg.fire_discipline or soldier.mission is None:
            return 1.0
        mt = soldier.mission.type
        if mt in WEAPONS_TIGHT:
            return 0.0
        if mt in POSITION_ANCHORED_FIRE:
            return 1.0 if self._in_mission_position(soldier) else 0.0
        return 1.0

    # ------------------------------------------------------------------ #
    # SUPPORT relations
    # ------------------------------------------------------------------ #

    def _supported_element(self, supported: Soldier) -> set[int]:
        """The supported element: the supported soldier + its living direct
        subordinates (the unit it leads, or just itself if it leads none)."""
        ids = {supported.id}
        ids.update(s.id for s in supported.living_subordinates(self.roster))
        return ids

    def _active_supports(self) -> list[tuple[Soldier, Soldier]]:
        """(supporter, supported) pairs whose supporter is in SUPPORT position."""
        pairs: list[tuple[Soldier, Soldier]] = []
        for s in self.roster.living:
            m = s.mission
            if m is None or m.type is not MissionType.SUPPORT:
                continue
            supported = self.roster.by_id.get(m.extra.get("supported_id"))
            if supported is None or not supported.alive:
                continue
            if self._in_mission_position(s):
                pairs.append((s, supported))
        return pairs

    def _end_orphaned_supports(self) -> None:
        """Clear SUPPORT missions whose supported soldier died (with a notice).

        A SUPPORT mission ends on re-tasking or on the supported unit's
        death — succession does not transfer it: the supporter announces the
        end on the net and stands by for new orders.
        """
        for s in self.roster.living:
            m = s.mission
            if m is None or m.type is not MissionType.SUPPORT:
                continue
            supported = self.roster.by_id.get(m.extra.get("supported_id"))
            if supported is not None and supported.alive:
                continue
            supported_cs = supported.callsign if supported is not None else "STATION"
            self._say(
                MessageKind.SUPPORT_END,
                s.id,
                s.leader_id,
                lang.format_support_end(self._addressee(s), s.callsign, supported_cs),
            )
            s.mission = None  # standing by for new orders

    def _compliance_ctx(
        self, soldier: Soldier, dist_prev: float | None, view: AgentView
    ) -> ComplianceContext:
        mission = soldier.mission
        dist_now = self._anchor_distance(soldier)
        if dist_prev is None or (mission is not None and mission.step_assigned == self._step_count):
            dist_prev = dist_now

        in_position = False
        enemies_at_obj = 0
        if mission is not None:
            in_position = self._in_mission_position(soldier, dist_now=dist_now)
            if mission.objective_id is not None:
                obj = self.world.objectives[mission.objective_id]
                enemies_at_obj = sum(
                    1 for e in self.enemies if e.alive and dist(e.pos, obj.pos) <= obj.radius + 1.0
                )
        leader = self.roster.leader_of(soldier)
        return ComplianceContext(
            dist_prev=dist_prev,
            dist_now=dist_now,
            in_position=in_position,
            stationary=soldier.pos == soldier.prev_pos,
            fired=soldier.fired_this_step,
            visible_enemies=len(view.visible_enemies),
            enemies_at_objective=enemies_at_obj,
            dist_to_leader=dist(soldier.pos, leader.pos) if leader is not None else float("inf"),
        )

    def _check_success(self, root_obj: Any) -> bool:
        mission = self.spec_cfg.root_mission
        living_enemies = [e for e in self.enemies if e.alive]
        if mission in (MissionType.DEFEND, MissionType.DENY, MissionType.CLEAR):
            return not living_enemies
        if mission is MissionType.SEIZE:
            if root_obj is None:
                return not living_enemies
            clear = not any(dist(e.pos, root_obj.pos) <= root_obj.radius + 1.0 for e in living_enemies)
            occupied = any(dist(s.pos, root_obj.pos) <= root_obj.radius for s in self.roster.living)
            return clear and occupied
        if mission in (MissionType.RECON, MissionType.SCREEN):
            return self._team_observe_steps >= 2 * RECON_OBSERVE_STEPS
        return False

    # ------------------------------------------------------------------ #
    # radio + human interface
    # ------------------------------------------------------------------ #

    def _addressee(self, soldier: Soldier) -> str:
        leader = self.roster.leader_of(soldier)
        return leader.callsign if leader is not None else "HQ"

    def _say(self, kind: MessageKind, sender: int, recipient: int | None, text: str) -> None:
        msg = Message(step=self._step_count, kind=kind, sender_id=sender, recipient_id=recipient, text=text)
        self.transcript.add(msg)
        self.last_messages.append(msg)

    def inject_order(self, text: str, issuer: str = "HQ") -> Message:
        """Let a human speak on the net: parse and apply an order.

        ``issuer`` is "HQ" (may order anyone) or a callsign (must outrank the
        recipient and have them as a direct subordinate). Returns the ORDER
        message; raises ``OrderParseError`` / ``PermissionError`` otherwise.
        """
        parsed = lang.parse_order(text)
        recipient = self.roster.by_callsign.get(parsed.recipient_callsign)
        if recipient is None or not recipient.alive:
            msg = f"No living station {parsed.recipient_callsign!r} on the net."
            raise lang.OrderParseError(msg)

        if issuer.upper() == "HQ":
            issuer_id, issuer_cs = HQ_ID, "HQ"
        else:
            issuing = self.roster.by_callsign.get(issuer.upper())
            if issuing is None or not issuing.alive:
                msg = f"No living station {issuer!r} on the net."
                raise lang.OrderParseError(msg)
            if recipient.id not in issuing.subordinate_ids:
                msg = f"{issuing.callsign} cannot order {recipient.callsign}: not a direct subordinate."
                raise PermissionError(msg)
            if issuing.effective_authority <= recipient.effective_authority:
                msg = f"{issuing.callsign} does not outrank {recipient.callsign}."
                raise PermissionError(msg)
            issuer_id, issuer_cs = issuing.id, issuing.callsign
        # per-echelon admissibility (manual p. 8): e.g. DENY is a section
        # mission — a fire team or rifleman can never hold it
        if recipient.effective_authority < min_hold_authority(parsed.mission):
            msg = (
                f"{recipient.callsign} cannot hold {parsed.mission.name}: requires "
                f"authority >= {min_hold_authority(parsed.mission)} (section level and above)."
            )
            raise PermissionError(msg)
        supported_id: int | None = None
        if parsed.mission is MissionType.SUPPORT:
            supported = self.roster.by_callsign.get(parsed.target_callsign or "")
            if supported is None or not supported.alive:
                msg = f"No living station {parsed.target_callsign!r} to support."
                raise lang.OrderParseError(msg)
            if supported.id == recipient.id:
                msg = f"{recipient.callsign} cannot support itself."
                raise lang.OrderParseError(msg)
            supported_id = supported.id
        objective = (
            self.world.objective_by_name(parsed.objective_name) if parsed.objective_name else None
        )
        if parsed.objective_name and objective is None:
            msg = f"No objective named {parsed.objective_name!r} on this map."
            raise lang.OrderParseError(msg)
        self._assign_mission(
            issuer_id=issuer_id,
            issuer_cs=issuer_cs,
            recipient=recipient,
            mission_type=parsed.mission,
            objective_id=objective.id if objective else None,
            supported_id=supported_id,
        )
        return self.last_messages[-1] if self.last_messages else self.transcript.messages[-1]

    # ------------------------------------------------------------------ #
    # ground-truth oracle (external observers only)
    # ------------------------------------------------------------------ #

    def oracle(self) -> dict:
        """Ground-truth snapshot incl. OpFor internals — for external observers.

        Behavior observables (core/oracle.py) for every unit, friendly and
        enemy. Strictly outside the simulation loop: never feeds agent
        observations, rewards, or masks, and consumes no randomness. Call
        after reset() or after each step().
        """
        from cohort.core.oracle import observe

        return observe(self)

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #

    def render(self) -> str | np.ndarray | None:
        """ANSI grid or RGB frame, per ``render_mode``."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "rgb_array":
            from cohort.viz.render import render_frame  # local import: matplotlib is heavy

            return render_frame(self)
        return None

    def _render_ansi(self) -> str:
        from cohort.core import world as W  # noqa: N812

        chars = np.full((self.world.height, self.world.width), ".", dtype="<U1")
        chars[self.world.grid == W.FOREST] = "'"
        chars[self.world.grid == W.WALL] = "#"
        for obj in self.world.objectives:
            chars[obj.pos[1], obj.pos[0]] = obj.name[0]
        for e in self.enemies:
            if e.alive:
                chars[e.pos[1], e.pos[0]] = "X"
        for s in self.roster.soldiers:
            if s.alive:
                symbol = s.callsign[0].lower() if s.effective_rank is Rank.RFN else s.callsign[0]
                chars[s.pos[1], s.pos[0]] = symbol
        lines = ["".join(row) for row in chars]
        status = " | ".join(
            f"{s.callsign}({s.effective_rank.name},{s.health}hp,"
            f"{s.mission.type.name if s.mission else 'STANDBY'})"
            for s in self.roster.living
        )
        return "\n".join([*lines, f"t={self._step_count} {status}"])

    def close(self) -> None:
        """Nothing to release."""


def make_env(
    scenario: str | ScenarioSpec = "fireteam",
    render_mode: str | None = None,
    reward_config: RewardConfig | None = None,
) -> CohortEnv:
    """Factory used by training code and docs examples."""
    return CohortEnv(scenario=scenario, render_mode=render_mode, reward_config=reward_config)
