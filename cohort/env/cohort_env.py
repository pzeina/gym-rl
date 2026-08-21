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

from cohort.config import ScenarioSpec, announced_assault_step, build_org, get_scenario
from cohort.core import acoustics as snd
from cohort.core import cohesion
from cohort.core import language as lang
from cohort.core import liaison as lia
from cohort.core.missions import (
    HOLDS_GROUND,
    IN_POSITION_RADIUS,
    LOS_REQUIRED,
    POSITION_ANCHORED_FIRE,
    TEAM_OBSERVE_STEPS,
    WEAPONS_TIGHT,
    ComplianceContext,
    Mission,
    MissionType,
    compliance,
    derivation_quality,
    in_formation,
    is_completable,
    is_complete,
    is_pending,
    min_hold_authority,
)
from cohort.core.orders import HQ_ID, Message, MessageKind, Transcript
from cohort.core.ranks import AUTHORITY, Rank
from cohort.core.units import (
    BriqueBand,
    Enemy,
    Roster,
    Soldier,
    Trap,
    enemy_decide,
    resolve_fire,
    validate_human_ranks,
    voice_peers,
)
from cohort.core.world import World, dist
from cohort.env.actions import (
    CATALOG,
    N_ACTIONS,
    ActionSpec,
    compute_mask,
    is_root_opord_claim,
)
from cohort.env.observations import AgentView, build_observation, obs_dim
from cohort.env.rewards import RewardConfig, RewardLedger

#: Steps after which an unrefreshed contact report goes stale.
KNOWLEDGE_TTL = 40

#: A5-4 trinôme sync: how long a PREPARE-TO-BOUND proposal stays live
#: awaiting its GO, and how long the synchronized window lasts after it.
SYNC_PROPOSE_TTL = 20
SYNC_WINDOW = 8

#: Net arbitration priority for LEARNED transmissions (A4): lower wins.
#: CONTACT (perishable intel) > DONE (command state) > orders (EXECUTE is
#: command traffic of the same class) > SITREP (routine); ties break by agent
#: order. Auto-traffic is not listed — WILCO, verdicts, CASUALTY, and
#: succession are protocol, not competition for air.
_TX_PRIORITY: dict[str, int] = {
    "contact": 0, "acoustic_contact": 0, "done": 1, "order": 2, "execute": 2, "sitrep": 3,
}

#: §4.5 acceptance: which packet deliveries count as ACCEPTED (and so pay the
#: courier's liaison_delivery) — the content was new to the recipient
_ACCEPTED = ("applied", "novel", "refresh", "fresh", "confirmed")

#: how long a carried ACOUSTIC CONTACT report stays reportable onward
#: (store-and-forward, §3.6.3): measured from its SOURCE step, so relaying
#: never refreshes it. Published through briefing() with the acoustic model.
ACOUSTIC_REPORT_TTL = 20

#: The four static tasks priced by ``RewardConfig.exposed_under_threat`` —
#: the set the squad_screen death measurement named, not "everything static":
#: DEFEND/DENY posture is already governed by objective_lost and the
#: defensive-ground gate, and a mover's exposure is the cost of moving.
STATIC_EXPOSURE_MISSIONS = frozenset(
    {MissionType.OBSERVE, MissionType.SCREEN, MissionType.HOLD, MissionType.COVER}
)


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
        # No explicit config → the scenario's own economics (v1.21), not bare
        # defaults: a spec's reward_overrides are part of what the scenario IS.
        self.rewards_cfg = reward_config or RewardConfig.from_scenario(self.spec_cfg)
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
                "observation": spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(obs_dim(self.spec_cfg.observation_profile),),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(low=0, high=1, shape=(N_ACTIONS,), dtype=np.int8),
            }
        )
        self._act_space = spaces.Discrete(N_ACTIONS)

        self._rng = np.random.default_rng()
        self.agents: list[str] = []
        self.world: World | None = None
        self.roster: Roster | None = None
        self.enemies: list[Enemy] = []
        self.band: BriqueBand | None = None  # BRIQUE armed band (opfor_mode="brique")
        self.traps: list[Trap] = []          # hidden devices laid by the band at reset
        self.transcript = Transcript()
        self.last_messages: list[Message] = []
        self._step_count = 0
        self._team_observe_steps = 0
        self._known_enemies: dict[int, tuple[float, float, int]] = {}  # id → (x, y, step)
        #: per-listener enemy pictures, used only when comm_model="range"
        self._agent_known: dict[str, dict[int, tuple[float, float, int]]] = {}
        self._illegal_actions = 0
        self._episode_outcome: str | None = None  # success | defeat | timeout
        self._support_umbrellas: list[tuple[Soldier, set[int]]] = []  # per-step (P2)
        self._shots_at: dict[int, int] = {}  # enemy id → friendly shots this step
        self._last_net_contact_step: int | None = None  # last CONTACT on the net
        self._success_step: int | None = None  # T0: success condition first met
        #: v1.14: a horizon defense that lost the position, permanently. Success
        #: latches at T0; conservation of the position has to latch the other
        #: way, or a mission already failed could still be won by a retake.
        self._defend_lost_step: int | None = None
        #: the step the root closed the C2 loop on the operation, in time to
        #: earn ``root_done_bonus``. Two routes, by root mission (v1.13): a
        #: truthful MISSION COMPLETE where the root may declare one, or — on a
        #: continuous DEFEND/DENY posture, which nobody below COMMAND may
        #: declare over — the root's SITREP once the end state holds.
        self._root_close_step: int | None = None
        self._root_close_callsign: str | None = None
        #: has a root-OPORD MISSION COMPLETE been filed this episode at all
        #: (v1.15)? The first one spends ``root_done_bonus`` whether it is
        #: confirmed or rejected — see RewardConfig.root_done_bonus_first_claim_only.
        self._root_claim_filed: bool = False
        #: does the claim that closed the window still have the bonus to collect?
        #: True until a claim spends the slot without closing on it; the ENDEX
        #: route (which files no claim) is unaffected and keeps paying.
        self._root_close_earns_bonus: bool = True
        #: last step the root transmitted a SITREP (v1.13: the report COMMAND
        #: reads before it transmits ENDEX).
        self._root_sitrep_step: int | None = None
        #: the step COMMAND transmitted ENDEX, so it is transmitted once.
        self._endex_step: int | None = None
        #: defend preparation period (v1.10): the step the assault actually
        #: begins, and the nominal H the OPORD announced. Both None when the
        #: scenario has no ``assault_h_hour``.
        self._h_hour: int | None = None
        self._h_hour_nominal: int | None = None
        self._net_blocked: set[str] = set()  # NET BUSY losers this step (A4)
        self._tx_count = 0  # learned transmissions emitted this step (A4)
        #: soldier id → step of the last casualty in that soldier's element
        #: (its command subtree) — the B5 re-task pricing exception bookkeeping
        self._element_casualty_step: dict[int, int] = {}
        #: re-task events of the last step (B5): issuer/recipient, priced or
        #: excepted (and why), same-anchor or rotation — metrics bookkeeping
        self._retask_log: list[dict] = []
        #: order-payment events of the last step (refs #52): every agent-issued
        #: order that reached adjudication, tagged with which of the three
        #: outcomes it took — `fresh` (paid), `churn` (an identical reissue,
        #: charged), or `retask` (replaced a standing order without being a
        #: fresh tasking). Read-only bookkeeping; the sign of a commander's
        #: command income is not otherwise recoverable from the trace.
        self._order_pay_log: list[dict] = []
        #: formation shaping (A5-3): per-leader watermark of the best anchor
        #: distance reached under the current (mission, stance) — the bonus
        #: pays only on NEW closure, so it telescopes and cannot be farmed
        self._formation_watermark: dict[int, tuple] = {}
        #: trinôme sync (A5-4): proposer id -> (propose step, registered peer
        #: ids — those in voice range at propose time); GO consumes the entry
        self._sync_pending: dict[int, tuple[int, tuple[int, ...]]] = {}
        #: agent id -> (last synchronized step, group key) after a GO
        self._sync_until: dict[int, tuple[int, tuple]] = {}
        #: per-agent watermark of best own-anchor distance for the bound
        #: bonus — keyed by the standing order, NOT the window, so repeated
        #: propose/GO cycles can never re-earn already-covered ground
        self._bound_watermark: dict[int, tuple] = {}
        # --- tactical acoustics (§3.6) — ALL of it inert under sound_model
        # "off": no events, no cues, no anchors, no state, no RNG ---
        self._sound_seq = 0                      # event id counter (episode-scoped)
        self._step_sounds: list[snd.SoundEvent] = []   # this step's events
        self.last_sound_events: list[snd.SoundEvent] = []  # last completed step's
        self._pending_enemy_sounds: list[snd.SoundEvent] = []  # not yet heard by OpFor
        self._agent_cues: dict[str, list[snd.AcousticCue]] = {}  # bounded coarse memory
        self._own_sound: dict[str, tuple[str, float, int]] = {}  # cs → (kind, radius, step)
        #: audit metadata parallel to ``last_messages`` (medium + actual
        #: semantic hearers). Trace/oracle material — Message itself stays
        #: text-only by the repo's schema invariant.
        self.last_message_meta: list[dict] = []
        # --- voice-only degraded communications (§3) — inert outside the mode ---
        #: per-agent clock of the last change to ITS OWN enemy picture (new
        #: enemy id or an aged refresh): the listener-local replacement for
        #: the force-wide ``_last_net_contact_step`` re-task exception
        self._picture_changed_step: dict[str, int] = {}
        #: acoustic report memory per agent: report key -> step received
        #: (novelty is judged against the INTENDED superior's memory)
        self._acoustic_received: dict[str, dict[tuple, int]] = {}
        #: carried acoustic reports per agent (kind index, bearing, band,
        #: source step, strength) — store-and-forward onward, fields frozen
        self._held_acoustic: dict[str, list[tuple]] = {}
        #: voice_only friendly telemetry (§3.7): observer callsign -> related
        #: soldier id -> [last known pos, last known mission type, pos step,
        #: mission step]. Refreshed by local perception and heard reports only.
        self._friendly_state: dict[str, dict[int, list]] = {}
        #: visual-link state per agent: (intact or None, contiguous break age)
        self._link_state: dict[str, tuple[bool | None, int]] = {}
        #: formation station per agent: (at station or None, normalized error)
        self._station: dict[str, tuple[bool | None, float]] = {}
        #: metrics bookkeeping: (leader-direct-subordinate pairs, pairs in
        #: voice range) this step
        self._command_pairs: tuple[int, int] = (0, 0)
        #: metrics bookkeeping: enemy ids whose move this step investigated a
        #: heard anchor
        self._opfor_investigating: set[int] = set()
        # --- liaison / message packets (§4) — inert unless liaison_enabled ---
        self._packet_seq = 0
        self.packets: list[lia.MessagePacket] = []          # every packet this episode
        self._outbox: dict[int, lia.MessagePacket] = {}      # origin id -> held packet
        self._liaison: dict[int, lia.LiaisonTask] = {}      # carrier id -> duty
        #: replaced soldier id -> successor id, so a packet addressed to a
        #: command position reaches its current holder (§4.4)
        self._successions: dict[int, int] = {}
        #: this step's packet lifecycle events (trace/metrics bookkeeping)
        self._packet_log: list[dict] = []

    @property
    def _liaison_on(self) -> bool:
        return self._voice_only and bool(self.spec_cfg.liaison_enabled)

    @property
    def _voice_only(self) -> bool:
        return self.spec_cfg.comm_model == "voice_only"

    @property
    def _local_pictures(self) -> bool:
        """Per-listener enemy pictures (range radio and voice_only)."""
        return self.spec_cfg.comm_model in ("range", "voice_only")

    @property
    def outcome(self) -> str | None:
        """Episode outcome: ``"success"`` | ``"defeat"`` | ``"timeout"``, or
        ``None`` while the episode is still running."""
        return self._episode_outcome

    @property
    def root_close_step(self) -> int | None:
        """Step at which the root's own report closed the operation, or None.

        Set by a truthful root-mission MISSION COMPLETE (``_report_done``) or,
        on a continuous-posture root where the claim is masked shut, by the
        SITREP that reported the end state (the terminal check). None means the
        grace window simply expired — the operation ended because the world
        said so, not because anyone reported it.

        Public since v1.20 so training can watch it: ``ckpt_best`` is selected
        on rolling success, and a policy that wins without ever reporting must
        not be selected as a run's best work (``best_save_gate``).
        """
        return self._root_close_step

    @property
    def transmissions_last_step(self) -> int:
        """Learned transmissions actually emitted during the last ``step()``
        (CONTACT / SITREP / DONE / agent-issued orders; auto-traffic and
        NET BUSY-dropped attempts excluded). Training metrics bookkeeping."""
        return self._tx_count

    @property
    def retask_events_last_step(self) -> list[dict]:
        """Re-task events applied during the last ``step()`` (B5): every order
        that replaced a subordinate's standing mission, with issuer callsign /
        rank / authority, recipient, whether the anchor changed (a rotation),
        whether the price was waived (and the exception reason), and the cost
        charged. Read-only bookkeeping for metrics; fresh taskings of untasked
        subordinates and identical reissues are not re-tasks."""
        return list(self._retask_log)

    def _log_order_pay(self, issuer, outcome: str, tier: str | None, pay: float) -> None:
        """Record one adjudicated mission order (refs #52). Bookkeeping only —
        it never decides anything, and it is called AFTER the ledger entry it
        describes, so it cannot change what was charged. Stance orders are not
        logged: they carry no mission and never enter ``orders_by_task``, so
        including them would make the split incomparable with it."""
        self._order_pay_log.append(
            {
                "issuer": issuer.callsign,
                "rank": issuer.effective_rank.name,
                "outcome": outcome,
                "tier": tier,
                "pay": pay,
            }
        )

    @property
    def order_pay_events_last_step(self) -> list[dict]:
        """Order-payment events of the last ``step()`` (refs #52): issuer
        callsign / rank, and which outcome the order took — ``fresh`` (a
        tasking the issuer is paid for, at ``preferred``/``allowed``/nothing by
        derivation quality), ``churn`` (an identical reissue, charged
        ``order_churn``), or ``retask`` (replaced a standing order without
        qualifying as a fresh tasking).

        This exists because order VOLUME does not say whether commanding is
        profitable: only fresh taskings pay, while reissues are charged. A
        commander that orders constantly may be farming the channel or paying
        to stay in it, and the two are opposite diagnoses."""
        return list(self._order_pay_log)

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
            waypoint_specs=[(name, pos) for name, pos in cfg.waypoints],
            phase_line_specs=[(name, a, b) for name, a, b in cfg.phase_lines],
        )
        if cfg.objective_cover and cfg.root_objective:
            self._prepare_defensive_ground(cfg.root_objective)
        if cfg.observation_concealment and cfg.root_objective:
            self._prepare_observation_posts(cfg.root_objective)
        self._spawn_roster()
        self._spawn_enemies()
        self._spawn_traps()
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
            {cs: {} for cs in self._callsigns} if self._local_pictures else {}
        )
        self._illegal_actions = 0
        self._episode_outcome = None
        self._support_umbrellas = []
        self._shots_at = {}
        self._last_net_contact_step = None
        self._success_step = None
        self._defend_lost_step = None
        self._root_close_step = None
        self._root_close_callsign = None
        self._root_claim_filed = False
        self._root_close_earns_bonus = True
        self._root_sitrep_step = None
        self._endex_step = None
        self._h_hour = None
        self._h_hour_nominal = None
        self._draw_h_hour()
        self._net_blocked = set()
        self._tx_count = 0
        self._element_casualty_step = {}
        self._retask_log = []
        self._order_pay_log = []
        self._formation_watermark = {}
        self._sync_pending = {}
        self._sync_until = {}
        self._bound_watermark = {}
        self._sound_seq = 0
        self._step_sounds = []
        self.last_sound_events = []
        self._pending_enemy_sounds = []
        self._agent_cues = {cs: [] for cs in self._callsigns} if self._sound_on else {}
        self._own_sound = {}
        self.last_message_meta = []
        self._picture_changed_step = {}
        self._acoustic_received = {cs: {} for cs in self._callsigns}
        self._held_acoustic = {cs: [] for cs in self._callsigns}
        self._friendly_state = {}
        self._link_state = {}
        self._station = {}
        self._command_pairs = (0, 0)
        self._opfor_investigating = set()
        self._packet_seq = 0
        self.packets = []
        self._outbox = {}
        self._liaison = {}
        self._successions = {}
        self._packet_log = []

        # OPORD from HQ to the senior agent.
        root = self.roster.root()
        objective = self.world.objective_by_name(cfg.root_objective) if cfg.root_objective else None
        root.mission = Mission(
            type=cfg.root_mission,
            objective_id=objective.id if objective else None,
            anchor=objective.pos if objective else root.pos,
            issuer_id=HQ_ID,
            step_assigned=0,
            # the commander holds the OPERATION's observation task: completion
            # and in-position credit follow the squad's aggregated observation
            # (refs #9 — personal adjudication drove the human root into exposure)
            team_observation=cfg.root_mission in (MissionType.RECON, MissionType.SCREEN),
        )
        root.last_order_step = 0
        self._say(
            MessageKind.OPORD,
            HQ_ID,
            root.id,
            lang.format_opord(
                root.callsign,
                cfg.root_mission,
                cfg.root_objective,
                self._h_hour_nominal,
                # the ordered hour goes on the net (v1.18, refs #30): HQ says
                # what it adjudicates. ``format_opord`` speaks it only for a
                # HOLDS_GROUND mission, the same predicate as
                # :meth:`_horizon_defense`, so the two cannot drift.
                cfg.defend_horizon,
            ),
        )
        if cfg.ablation == "flat":
            # B3 flat arm: no ranks in effect — HQ tasks EVERY agent with the
            # OPORD mission directly (all-tasked at reset). Order actions are
            # masked off for everyone (env/actions.py), so this is the only
            # tasking that ever occurs. Non-root agents hold it as a personal
            # task (like any subordinate tasking in the full system); the
            # root keeps the team-adjudicated OPORD semantics above (#9).
            for s in self.roster.soldiers:
                if s is root:
                    continue
                s.mission = Mission(
                    type=cfg.root_mission,
                    objective_id=objective.id if objective else None,
                    anchor=objective.pos if objective else s.pos,
                    issuer_id=HQ_ID,
                    step_assigned=0,
                )
                s.last_order_step = 0
                self._say(
                    MessageKind.OPORD,
                    HQ_ID,
                    s.id,
                    lang.format_opord(s.callsign, cfg.root_mission, cfg.root_objective),
                )

        if self._voice_only:
            # the force departs together after the briefing: everyone knows
            # where everyone in its element stands, and the root's OPORD
            # (it was briefed, not overheard). Nothing else is known.
            self._init_friendly_state(root)
            self._refresh_friendly_perception()
        self._update_visual_links()
        observations = self._all_observations()
        infos = {a: {"components": {}, "net_busy": False} for a in self.agents}
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
        if self.spec_cfg.root_human:
            # the root commander (reports to HQ; senior on ties) is human
            root = max(
                (s for s in soldiers if s.leader_id is None),
                key=lambda s: (AUTHORITY[s.rank], -s.id),
            )
            root.human = True
        validate_human_ranks(soldiers)  # humans-outrank-all-non-humans, or raise
        for s in soldiers:
            s.prev_pos = s.pos
        self.roster = Roster(soldiers)

    def _spawn_enemies(self) -> None:
        cfg = self.spec_cfg
        self.enemies = []
        self.band = None
        if cfg.opfor_mode == "brique":
            self._spawn_band()
            return
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

    def _spawn_band(self) -> None:
        """BRIQUE armed band (manual p. 9): a flat, leaderless band whose
        members share a band-level intent machine (core/units.py::BriqueBand).

        AMBUSH/LURK bands post at a chokepoint on blue's predicted route —
        the straight line from the friendly spawn to the OPORD objective.
        HARASS/RAID bands infiltrate from the map edges at standoff distance
        (``assault_spawn_min_dist`` models the early warning a defense earns).
        """
        cfg = self.spec_cfg
        root_obj = self.world.objective_by_name(cfg.root_objective) if cfg.root_objective else None
        obj_pos = root_obj.pos if root_obj is not None else cfg.objectives[0][1]
        members: list[Enemy] = []
        posts: dict[int, tuple[int, int]] = {}
        if cfg.band.initial_intent in ("ambush", "lurk"):
            f = 0.45 + 0.25 * self._rng.random()  # chokepoint fraction along the route
            choke = (
                round(cfg.spawn[0] + f * (obj_pos[0] - cfg.spawn[0])),
                round(cfg.spawn[1] + f * (obj_pos[1] - cfg.spawn[1])),
            )
            for i in range(cfg.n_enemies):
                post = self._band_post_near(choke, radius=3, taken=set(posts.values()))
                members.append(
                    Enemy(id=i, pos=post, home=post, goal=obj_pos, mode="brique", prev_pos=post)
                )
                posts[i] = post
        else:  # harass / raid: infiltrate from the edges
            for i in range(cfg.n_enemies):
                pos = self._random_edge_cell(
                    min_dist_from=obj_pos, min_dist=cfg.assault_spawn_min_dist
                )
                members.append(
                    Enemy(id=i, pos=pos, home=pos, goal=obj_pos, mode="brique", prev_pos=pos)
                )
                posts[i] = pos
        self.enemies = members
        self.band = BriqueBand(members, cfg.band, objective=obj_pos, posts=posts)

    def _band_post_near(
        self, center: tuple[int, int], radius: int, taken: set[tuple[int, int]]
    ) -> tuple[int, int]:
        """An ambush/lurk post near ``center``: cover cells preferred, then any
        passable cell; deterministic through ``env._rng`` only."""
        cover: list[tuple[int, int]] = []
        open_cells: list[tuple[int, int]] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                cell = (center[0] + dx, center[1] + dy)
                if cell in taken or not self.world.passable(cell):
                    continue
                (cover if self.world.cover_at(cell) else open_cells).append(cell)
        pool = cover or open_cells
        if pool:
            return pool[int(self._rng.integers(len(pool)))]
        return self._random_cell_near(center, radius=radius + 2)

    def _spawn_traps(self) -> None:
        """The band mines blue's likely route / the objective approaches.

        Route scenarios (spawn far from the objective): trap cells sit along
        the spawn→objective line with jitter — where a patrol is likely to
        walk. Defense scenarios (blue spawns on the objective): traps ring
        the position's approaches instead, punishing undisciplined sorties.
        Hidden until triggered; oracle-visible from step 0.
        """
        cfg = self.spec_cfg
        self.traps = []
        if cfg.n_traps <= 0:
            return
        root_obj = self.world.objective_by_name(cfg.root_objective) if cfg.root_objective else None
        obj_pos = root_obj.pos if root_obj is not None else cfg.objectives[0][1]
        on_route = dist(cfg.spawn, obj_pos) >= 10.0
        taken: set[tuple[int, int]] = set()
        for i in range(cfg.n_traps):
            for _ in range(60):
                if on_route:
                    f = 0.3 + 0.55 * self._rng.random()
                    cand = (
                        round(cfg.spawn[0] + f * (obj_pos[0] - cfg.spawn[0]))
                        + int(self._rng.integers(-2, 3)),
                        round(cfg.spawn[1] + f * (obj_pos[1] - cfg.spawn[1]))
                        + int(self._rng.integers(-2, 3)),
                    )
                else:
                    ang = 2.0 * math.pi * self._rng.random()
                    r = 4.0 + 4.0 * self._rng.random()
                    cand = (
                        round(obj_pos[0] + r * math.cos(ang)),
                        round(obj_pos[1] + r * math.sin(ang)),
                    )
                if cand in taken or not self.world.passable(cand):
                    continue
                if dist(cand, cfg.spawn) < 6.0:
                    continue  # never mine the friendly spawn itself
                if any(dist(cand, o.pos) <= o.radius for o in self.world.objectives):
                    continue  # objectives themselves stay standable
                taken.add(cand)
                self.traps.append(Trap(id=i, pos=cand))
                break
            # degenerate map: fewer traps rather than an invalid placement

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
        self.last_message_meta = []
        self._retask_log = []
        self._order_pay_log = []
        self._step_sounds = []
        self._packet_log = []
        present = list(self.agents)

        # --- timed-order release (A5-2): "AT T PLUS n" comes due ---
        # Released BEFORE the snapshot so this tick's anchors/compliance are
        # judged against the now-effective order. Binding (tenure) starts at
        # execution, not receipt: step_assigned restamps to the release tick.
        for s in self.roster.living:
            m = s.mission
            if m is not None and m.effective_at is not None and step >= m.effective_at:
                m.effective_at = None
                m.step_assigned = step

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
        # SUPPORT effects for this tick, from the snapshot positions:
        # (supporter, supported-element ids) pairs whose supporter is in
        # position. Focus fire is a coordination effect — enabled while any
        # support relation is active. Per-target friendly shot counter for
        # the focus-fire follow-up bonus.
        self._support_umbrellas = [
            (supporter, self._supported_element(supported))
            for supporter, supported in self._active_supports()
        ]
        self._shots_at: dict[int, int] = {}

        # --- net arbitration (A4): one learned transmission per tick ---
        self._net_blocked = self._arbitrate_net(present, actions)
        self._tx_count = 0

        # --- friendly actions ---
        enemy_kills: list[tuple[Soldier, Enemy]] = []
        player_deaths: list[Soldier] = []  # traps can kill during the friendly phase
        for callsign in present:
            soldier = self.roster.by_callsign[callsign]
            if not soldier.alive or callsign not in actions:
                continue
            self._apply_action(soldier, int(actions[callsign]), ledger, enemy_kills, player_deaths)

        self._opfor_investigating = set()
        # --- OpFor hearing (§3.6.4 ordering): Blue actions resolved first, so
        # Blue movement/speech sounds may inform the OpFor turn this same
        # step; OpFor sounds enter the next Blue observation. Recorded in the
        # trace via each event's step and the delivery lists.
        self._deliver_sounds_to_enemies()

        # --- OpFor actions ---
        if self.band is not None:
            # band-level intent machine ticks once, on post-move blue positions
            self.band.update(step, [s.pos for s in self.roster.living])
        if not self._in_preparation():
            for enemy in [e for e in self.enemies if e.alive]:
                self._enemy_turn(enemy, ledger, player_deaths)

        # --- casualties and succession ---
        for dead in player_deaths:
            # net/umpire convention: the report comes from HQ, not the casualty
            self._say(MessageKind.CASUALTY, HQ_ID, None, lang.format_casualty(dead.callsign))
            # B5 re-task exception bookkeeping: a casualty is news to every
            # commander above the fallen agent — their element's picture
            # changed, so re-tasking within it is free until re-ordered
            # voice_only (§3.4, §6.8): the casualty is news only to a
            # commander who actually witnessed it — there is no HQ
            # all-stations CASUALTY line to tell a distant leader
            ancestor = self.roster.leader_of(dead)
            while ancestor is not None:
                if not self._voice_only or self._witnessed(ancestor, dead.pos):
                    self._element_casualty_step[ancestor.id] = step
                ancestor = self.roster.leader_of(ancestor)
            # rank-weighted: losing a leader costs more, by effective authority
            weight = 1.0 + cfg.rank_casualty_scale * dead.effective_authority
            for other in self.roster.living:
                ledger.add(other.callsign, "combat", cfg.teammate_death * weight)
            if dead.human:
                # losing the human commander approaches mission failure: every
                # present agent pays, on top of the normal death penalties; the
                # episode continues and succession exercises
                for callsign in present:
                    ledger.add(callsign, "combat", cfg.human_death)
            for successor, replaced in self.roster.succeed(dead):
                self._successions[replaced.id] = successor.id
                if self._voice_only:
                    self._succession_knowledge(successor, replaced)
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

        # --- Blue hearing: every one of this step's events (both phases)
        # becomes at most one coarse cue per living listener, entering the
        # observations assembled below — i.e. the NEXT Blue decision. ---
        self._deliver_sounds_to_blue()
        self.last_sound_events = list(self._step_sounds)

        # --- liaison duties (§4): loss, expiry, vacant positions, progress ---
        self._update_liaisons(ledger)

        # --- knowledge decay ---
        living_enemy_ids = {e.id for e in self.enemies if e.alive}

        def _fresh(picture: dict[int, tuple[float, float, int]]) -> dict:
            return {
                eid: entry
                for eid, entry in picture.items()
                if eid in living_enemy_ids and step - entry[2] <= KNOWLEDGE_TTL
            }

        self._known_enemies = _fresh(self._known_enemies)
        if self._voice_only:
            # §3.5: an agent's own sighting enters ITS picture, at the
            # observed position and time (a new id moves its picture clock)
            for s in self.roster.living:
                picture = self._agent_known.setdefault(s.callsign, {})
                for e in self._visible_enemies(s):
                    if e.id not in picture:
                        self._picture_changed_step[s.callsign] = step
                    picture[e.id] = (float(e.pos[0]), float(e.pos[1]), step)
            for cs in self._held_acoustic:
                self._held_acoustic[cs] = [
                    r for r in self._held_acoustic[cs] if step - r[3] <= ACOUSTIC_REPORT_TTL
                ]
            self._refresh_friendly_perception()
        for callsign in self._agent_known:
            self._agent_known[callsign] = _fresh(self._agent_known[callsign])
        # visual-link graph (§3.7): every tick, no grace hidden from the
        # metric; casualties and succession have already rebuilt the roster
        self._update_visual_links()

        # --- mission progress + compliance + step costs ---
        root_obj = (
            self.world.objective_by_name(self.spec_cfg.root_objective)
            if self.spec_cfg.root_objective
            else None
        )
        views = self._compute_views()
        for callsign in present:
            soldier = self.roster.by_callsign[callsign]
            if not soldier.alive:
                continue  # a casualty accrues nothing per step, not even the clock
            ledger.add(callsign, "time", cfg.time_penalty)
            ctx = self._compliance_ctx(soldier, prev_dist.get(callsign), views[callsign])
            if soldier.mission is not None:
                self._update_crossing(soldier)
                if (
                    soldier.mission.type in (MissionType.RECON, MissionType.SCREEN)
                    and not soldier.mission.team_observation
                    and ctx.in_position
                ):
                    # personal progress — a subordinate's own task; the root's
                    # OPORD counter is mirrored from the team counter below
                    soldier.mission.observe_steps += 1
                # a pending order (A5-2) is judged as HOLD at the staging
                # position until it becomes effective
                effective_type = (
                    MissionType.HOLD
                    if is_pending(soldier.mission, step)
                    else soldier.mission.type
                )
                score = compliance(effective_type, ctx)
                credit = cfg.compliance_weight * score
                if score > 0.0 and cfg.tenure_factor > 0.0:
                    # standing-order tenure (B5): positive compliance credit
                    # grows the longer the CURRENT order has been held, so
                    # settled, executed orders out-earn churned ones. Resets
                    # with step_assigned on re-tasking; identical reissues are
                    # no-ops and keep it. Negative scores are never amplified.
                    held = min(step - soldier.mission.step_assigned, cfg.tenure_horizon)
                    credit *= 1.0 + cfg.tenure_factor * max(0, held) / cfg.tenure_horizon
                ledger.add(callsign, "compliance", credit)
            # reporting doctrine: out of contact and past the cadence → overdue
            cadence = self.spec_cfg.sitrep_cadence
            if (
                cadence
                and not views[callsign].visible_enemies
                and step - soldier.last_sitrep_step > cadence
                # voice_only §3.3: a root with no HQ channel is not mute, the
                # channel is structurally absent — never punished for it
                and not (self._voice_only and self.roster.leader_of(soldier) is None)
            ):
                ledger.add(callsign, "report", cfg.sitrep_overdue)
            # leader coverage — neutralized in the flat ablation arm (B3):
            # with no ranks in effect and everyone OPORD-tasked at reset,
            # the bonus would pay for free (and the gap would punish agents
            # that finished truthful DONEs, with no way to re-task them)
            if (
                soldier.effective_authority > 0
                and soldier.mission is not None
                and self.spec_cfg.ablation != "flat"
            ):
                subs = soldier.living_subordinates(self.roster)
                if subs:
                    # §6.7: a detached courier counts as tasked (its tactical
                    # mission is suspended, not absent)
                    all_tasked = all(
                        sub.mission is not None or sub.id in self._liaison for sub in subs
                    )
                    ledger.add(callsign, "command", cfg.coverage_bonus if all_tasked else cfg.coverage_gap)

        # preparation-period occupancy (v1.10): before H, an agent in cover
        # within the objective's in-position radius is preparing the defense.
        # Bounded by H itself — it stops paying the moment the assault starts.
        if (
            cfg.prep_in_position != 0.0
            and root_obj is not None
            and self._in_preparation()
        ):
            radius = IN_POSITION_RADIUS[self.spec_cfg.root_mission]
            for s in self.roster.living:
                if dist(s.pos, root_obj.pos) <= radius and self.world.cover_at(s.pos):
                    ledger.add(s.callsign, "compliance", cfg.prep_in_position)

        # formation shaping (A5-3): members at their formation station while
        # their leader closes NEW ground toward its mission anchor earn the
        # formation bonus — watermark-gated, so it telescopes with the
        # advance and cannot be farmed by circling or pacing
        self._formation_shaping(ledger)

        # broken visual link (§3.7 / §6.5, voice_only): a small per-agent-step
        # penalty on disconnected non-detached members, capped per element;
        # no positive term exists for standing in a blob
        self._visual_link_penalty(ledger)

        # trinôme bound shaping (A5-4): synchronized movers under a covering
        # peer earn the bound bonus on NEW closure toward their own anchor
        self._bound_shaping(ledger)

        # objective-lost pressure for DEFEND / DENY campaigns: while a living
        # enemy stands on the root objective, every living agent bleeds —
        # a defense that ceded its ground is failing, hiding included
        if (
            cfg.objective_lost != 0.0
            and root_obj is not None
            and self.spec_cfg.root_mission in (MissionType.DEFEND, MissionType.DENY)
            and any(
                e.alive and dist(e.pos, root_obj.pos) <= root_obj.radius + 1.0
                for e in self.enemies
            )
        ):
            for s in self.roster.living:
                ledger.add(s.callsign, "compliance", cfg.objective_lost)

        # cover-exposure pressure (squad_screen diagnosis): a static-tasked
        # soldier out of cover with a living enemy in weapon range bleeds
        # exposed_under_threat per step — the four static tasks, because the
        # measured deaths split OBSERVE .44 / SCREEN .38 / COVER .19 (16/16
        # out of cover). Distance-only threat, the eval standard's definition.
        if cfg.exposed_under_threat != 0.0:
            for s in self.roster.living:
                if s.mission is None or s.mission.type not in STATIC_EXPOSURE_MISSIONS:
                    continue
                if self.world.cover_at(s.pos):
                    continue
                if any(
                    e.alive and dist(s.pos, e.pos) <= self.combat.weapon_range
                    for e in self.enemies
                ):
                    ledger.add(s.callsign, "compliance", cfg.exposed_under_threat)

        # team observation progress for RECON / SCREEN campaigns. Each NOVEL
        # step toward the success counter pays observe_progress to the
        # observer (A2/A7 stall-exploit fix): the payout telescopes — it is
        # bounded by the success threshold and cannot be farmed indefinitely.
        if root_obj is not None and self.spec_cfg.root_mission in (
            MissionType.RECON,
            MissionType.SCREEN,
        ):
            observer = self._team_observer(root_obj)
            if observer is not None:
                if self._team_observe_steps < TEAM_OBSERVE_STEPS:
                    ledger.add(observer.callsign, "compliance", cfg.observe_progress)
                self._team_observe_steps += 1
            # mirror into the root's OPORD counter (refs #9): the commander's
            # mission tracks the squad's aggregated observation, so every
            # completion path (is_complete included) is team-adjudicated
            for s in self.roster.living:
                if s.mission is not None and s.mission.team_observation:
                    s.mission.observe_steps = self._team_observe_steps

        # --- terminal conditions ---
        # Success does not terminate on the spot: when the root-mission
        # condition is first met (T0) a reporting grace window opens, giving
        # the root time to close the C2 loop. Closing it ends the episode that
        # step (root_done_bonus); otherwise it ends as success at
        # T0 + grace_window anyway. Success is locked in at T0 — the speed
        # bonus is computed from T0, so policies that never report keep their
        # success rate and terminal reward.
        #
        # How the loop is closed depends on the root mission (v1.13, owner's
        # decision). Where the root may declare an end state, it transmits
        # MISSION COMPLETE. Where it may not — DEFEND/DENY are held until a
        # new order arrives, so their holder is never the one who says they
        # are over — the root reports the situation and COMMAND transmits
        # ENDEX. Same window, same bonus, opposite direction on the net.
        # (v1.16 separates the ENDEX from that choice of route: see below.)
        #
        # v1.14: on a defense ordered to a horizon, conservation of the
        # position is adjudicated FIRST and latches permanently — a mission
        # already failed cannot be won later by walking back onto the ground.
        self._update_defend_hold(root_obj)
        if self._check_success(root_obj) and self._success_step is None:
            self._success_step = step  # T0: the window opens
        success_locked = self._success_step is not None
        cohort_wiped = not any(s.alive for s in self.roster.soldiers)
        defeat = cohort_wiped and not success_locked
        #
        # v1.16: WHO CLOSES THE WINDOW and WHO ANNOUNCES THE CLOSE are two
        # questions, and v1.14 answered both with one predicate (refs #31).
        # They are still two questions, and the two names below stay separate
        # even now that they evaluate alike: re-merging them is exactly how the
        # v1.13 mask change silently took the ENDEX with it.
        #
        # The window is closed by whichever act the OPORD leaves open to the
        # root. Since v1.17 a DEFEND/DENY root may never declare its operation
        # over — with or without a stated horizon — so its route is the v1.13
        # one: the root reports the situation and its SITREP closes the window.
        # That is deliberate and is what keeps ``root_done_bonus`` reachable on
        # a defense; masking the claim without it would make the bonus dead
        # reward, the v1.4 failure in v1.13 clothes.
        root_may_declare_the_end = is_completable(self.spec_cfg.root_mission)
        # The ANNOUNCEMENT is a different question with a different answer, and
        # since v1.19 the answer no longer depends on the mission at all:
        # COMMAND closes EVERY operation on the net. A confirmed claim and an
        # ENDEX are not redundant — the claim is the root's REPORT, the ENDEX is
        # the FACT — so both can go out on the same episode and on a completable
        # root they now do.
        #
        # Why it stopped being a predicate. Gating the announcement on the
        # mission made the guarantee cover two scenarios of nine. Measured at
        # N=100 on the final policy of every published champion:
        #
        #     defend (ENDEX, a protocol act)        391/391 successes announced
        #     squad / squad_recon / squad_screen    91-98%
        #     fireteam_v8                            49/80
        #     platoon_v5  0/100   ·   patrol_brique_v5  0/99
        #
        # `platoon` and `patrol_brique` succeed on essentially every episode and
        # never once say so. Where the announcement is a protocol act it is
        # complete; where it is left to be an agent behaviour it ranges from 98%
        # to nothing and does not track how well the scenario is solved. Two
        # policies that seize the objective equally well are not two different
        # standards of reporting — they are one standard and one silence.
        #
        # This does not make the root's report worthless, and that distinction
        # is the whole design: the ENDEX says the operation is over, the root's
        # own act says it reported before HQ had to ask. The second is still
        # priced (root_done_bonus), still closes the window early, and is now
        # measurable fleet-wide on one scale — `closed_on_root_report_rate` has
        # ENDEXes-sent for a denominator, so before v1.19 it simply did not
        # exist outside the defend family.
        command_closes_the_operation = True
        if (
            success_locked
            and not root_may_declare_the_end
            and self._root_close_step is None
            and self._root_sitrep_step is not None
            and self._root_sitrep_step >= self._success_step
        ):
            root = self.roster.root()
            if root is not None:
                self._root_close_step = self._root_sitrep_step
                self._root_close_callsign = root.callsign
        root_reported = (
            success_locked
            and self._root_close_step is not None
            and self._root_close_step >= self._success_step
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
            # COMMAND closes the operation on the net (v1.13 on a continuous
            # posture, decoupled from completability in v1.16, extended to every
            # root in v1.19). Transmitted whether or not the root reported in
            # time and whether or not it claimed the mission complete — the
            # order to stop is COMMAND's to give either way; what the root's
            # report buys is closing early, and root_done_bonus. Once per
            # episode, guarded by _endex_step.
            #
            # Emitted here, in the terminal branch, AFTER the step's actions
            # have been applied — which is why extending it is rollout-neutral:
            # no agent ever chooses an action from an observation containing it.
            # Asserted once and measured false in the other direction; it is
            # pinned by tests now rather than argued (test_endex_guarantee).
            if command_closes_the_operation and self._endex_step is None:
                root = self.roster.root()
                if root is not None:
                    self._say(
                        MessageKind.ENDEX, HQ_ID, root.id, lang.format_endex(root.callsign)
                    )
                    self._endex_step = step
            speed = cfg.success_speed * max(0.0, 1.0 - self._success_step / self.spec_cfg.max_steps)
            # v1.11: paid to everyone still IN the episode, which now includes
            # the fallen (see `terminations` below). It used to be
            # `roster.living`, so a soldier who died at step 50 of an episode
            # that succeeded at step 200 received none of this — and the
            # arithmetic of that, per agent on the squad, was:
            #   hanging back cuts P(die) 0.129 -> 0.022, worth +6.4 ... but
            #   ONE shared policy updates EVERY agent at once, so team success
            #   goes 1.00 -> 0.00, worth -52.3.
            # A per-agent advantage only ever sees the first number, which is
            # how an individually-rational free-ride became a simultaneous
            # collective defection. Measured with the oracle on the collapsed
            # squad_screen policy: 19.96 cells from the objective against
            # 10.39 before, 13.9 threatened steps/ep against 24.9, 0.20
            # friendly deaths/ep against 1.12 — hang back, sit in cover, never
            # trigger the observation, terminal exactly 0.0000 forever.
            # A soldier who falls taking the objective shares in its being
            # taken. Preservation is still priced, proportionately, by
            # `death` and `teammate_death`.
            #
            # v1.12 (owner's option 4): on a DEFEND/DENY root the whole payout
            # is scaled by the force that is still standing — holding is what
            # the mission IS, so it is paid for with a force or not fully paid
            # for. The multiplier is identical for every agent, fallen
            # included, which is the entire point: it restores the
            # preservation pressure that forfeiture used to supply without
            # restoring the per-agent asymmetry that made forfeiture cause D4.
            scale = self._defend_terminal_scale()
            for callsign in present:
                ledger.add(callsign, "terminal", (cfg.success_team + speed) * scale)
            # ``_root_close_earns_bonus`` is False (v1.15) when the root spent
            # its one claim on a probe that was rejected. The window still
            # closed, the operation still ends here, and the confirmed claim was
            # still paid ``done_true`` — but the bonus is gone, which is what
            # makes the first claim a judgement rather than a free roll.
            if root_reported and self._root_close_callsign is not None and (
                self._root_close_earns_bonus
            ):
                ledger.add(self._root_close_callsign, "terminal", cfg.root_done_bonus)
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
            # v1.11: a casualty is NOT removed from the episode. It stays, with
            # STAY as its only legal action (compute_mask already returns that
            # for a dead soldier) and no per-step reward, until the episode
            # actually ends — so it is still present to be paid the team
            # terminal above. Two things follow, both wanted: the policy gets
            # no spurious gradient from the fallen (a single legal action has
            # zero entropy and a ratio of 1), while the CRITIC gets the correct
            # value target for "dead, outcome still pending".
            terminations[callsign] = success or defeat
            truncations[callsign] = truncated_all
            infos[callsign] = {
                "components": ledger.breakdown(callsign),
                "outcome": self._episode_outcome if episode_over else None,
                # NET BUSY (A4): this agent's transmission lost arbitration
                # this tick and was dropped — externally measurable discipline
                "net_busy": callsign in self._net_blocked,
            }
        self.agents = [a for a in present if not (terminations[a] or truncations[a])]
        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------ #
    # action application
    # ------------------------------------------------------------------ #

    def _arbitrate_net(self, present: list[str], actions: dict[str, int]) -> set[str]:
        """Single-frequency net: at most ONE learned transmission per tick.

        Deterministic priority arbitration over this tick's transmission
        attempts — CONTACT > DONE > orders > SITREP, ties broken by agent
        order. Losers' transmissions are dropped this tick with a NET BUSY
        outcome: no message, no cost, no effect — surfaced per agent in
        ``infos[...]["net_busy"]`` and the oracle snapshot. Auto-traffic
        (WILCO, verdicts, CASUALTY, succession) is protocol, not competition
        for airtime, and is never arbitrated. Under ``comm_model="range"``
        earshot shapes who *hears* a message, but every station shares the
        one frequency, so busy-ness stays global. Contention is judged on
        tick-start legality (the same mask the policy acted under); masks
        and spaces are untouched — a blocked agent simply loses its tick.
        """
        if self._voice_only:
            return set()  # §3.2: no shared frequency, no global arbitration
        raw: list[tuple[int, int, Soldier]] = []
        for idx, callsign in enumerate(present):
            soldier = self.roster.by_callsign[callsign]
            if not soldier.alive or callsign not in actions:
                continue
            spec = CATALOG[int(actions[callsign])]
            prio = _TX_PRIORITY.get(spec.kind)
            if prio is not None:
                raw.append((prio, idx, soldier))
        if len(raw) <= 1:
            return set()
        # illegal attempts are STAY, not transmissions: they cannot contend
        contenders = [
            (prio, idx, s) for prio, idx, s in raw
            if self._mask_for(s)[int(actions[s.callsign])]
        ]
        if len(contenders) <= 1:
            return set()
        contenders.sort(key=lambda c: (c[0], c[1]))
        return {s.callsign for _, _, s in contenders[1:]}

    def _charge_transmission(self, soldier: Soldier, ledger: RewardLedger, component: str) -> None:
        """Airtime cost for one emitted learned transmission (A4)."""
        ledger.add(soldier.callsign, component, self.rewards_cfg.transmission_cost)
        self._tx_count += 1

    def _apply_action(
        self,
        soldier: Soldier,
        action: int,
        ledger: RewardLedger,
        enemy_kills: list[tuple[Soldier, Enemy]],
        player_deaths: list[Soldier],
    ) -> None:
        cfg = self.rewards_cfg
        spec = CATALOG[action]
        if spec.kind in _TX_PRIORITY and soldier.callsign in self._net_blocked:
            return  # NET BUSY: transmission dropped this tick — no cost, no effect
        mask = self._mask_for(soldier)
        if not mask[action]:
            self._illegal_actions += 1
            return  # illegal → treated as STAY

        if spec.kind == "move":
            prev = soldier.pos
            soldier.pos = (soldier.pos[0] + spec.move[0], soldier.pos[1] + spec.move[1])
            soldier.heading = spec.move  # formation geometry frame (A5-3)
            # §3.6.1: one movement event at the traversed edge, the noisier
            # endpoint terrain setting the radius (blocked moves emit nothing —
            # they returned above as illegal)
            self._emit_sound(
                "movement",
                soldier.pos,
                "friendly",
                snd.movement_radius(self.world, prev, soldier.pos),
                source_cs=soldier.callsign,
            )
            self._check_trap(soldier, ledger, player_deaths)
        elif spec.kind == "fire":
            self._resolve_player_fire(soldier, ledger, enemy_kills)
        elif spec.kind == "contact":
            self._report_contact(soldier, ledger)
        elif spec.kind == "acoustic_contact":
            self._report_acoustic_contact(soldier, ledger)
        elif spec.kind == "gesture_execute":
            self._execute_signal(soldier, ledger, gesture=True)
        elif spec.kind == "gesture_sync_go":
            self._sync_go(soldier, ledger, gesture=True)
        elif spec.kind == "deliver":
            self._deliver_message(soldier, ledger)
        elif spec.kind == "cancel":
            self._cancel_message(soldier, ledger)
        elif spec.kind == "dispatch":
            self._dispatch_liaison(soldier, spec.order_slot, ledger)
        elif spec.kind == "sitrep":
            # under the reporting doctrine the mandated cadence *is* the
            # freshness interval, so a due report is never scored as spam
            leader = self.roster.leader_of(soldier)
            interval = self.spec_cfg.sitrep_cadence or cfg.sitrep_interval
            fresh = self._step_count - soldier.last_sitrep_step >= interval
            if self._voice_only and (leader is None or not self._audible_to(leader, soldier.id)):
                if leader is not None and self._may_prepare(soldier):
                    # §4.2: the report captures the sender's state NOW; its
                    # delay is visible as age at delivery, never refreshed
                    self._prepare_packet(
                        soldier, "sitrep", leader,
                        lang.format_sitrep(
                            leader.callsign, soldier.callsign, soldier.health, soldier.ammo,
                            soldier.pos, in_cover=self.world.cover_at(soldier.pos),
                        ),
                        payload=(fresh, self._step_count),
                    )
                return  # mask guards: nobody is there to report to (defensive)
            ledger.add(soldier.callsign, "report", cfg.sitrep_fresh if fresh else cfg.sitrep_spam)
            self._charge_transmission(soldier, ledger, "report")
            soldier.last_sitrep_step = self._step_count
            if soldier is self.roster.root():
                # v1.13: on a continuous-posture root this is the report
                # COMMAND acts on. Recorded for everyone — the terminal block
                # decides whether it lands after the end state, and only a
                # DEFEND/DENY root closes the operation this way.
                self._root_sitrep_step = self._step_count
            self._say(
                MessageKind.SITREP,
                soldier.id,
                soldier.leader_id,
                lang.format_sitrep(
                    self._addressee(soldier),
                    soldier.callsign,
                    soldier.health,
                    soldier.ammo,
                    soldier.pos,
                    in_cover=self.world.cover_at(soldier.pos),
                ),
                useful=fresh,
                redundant=not fresh,
            )
        elif spec.kind == "done":
            self._report_done(soldier, ledger)
        elif spec.kind == "execute":
            self._execute_signal(soldier, ledger)
        elif spec.kind == "sync_propose":
            self._sync_propose(soldier, ledger)
        elif spec.kind == "sync_go":
            self._sync_go(soldier, ledger)
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
        # §3.6.1: every shot creates a weapon event at the shooter, hit or
        # miss; the event never identifies the shooter to a listener
        self._emit_sound(
            "weapon_fire",
            soldier.pos,
            "friendly",
            snd.WEAPON_DETECT_RADIUS,
            source_cs=soldier.callsign,
        )
        # focus fire ("pas un pas sans appui"): with an active support
        # relation, follow-up shooters at an already-engaged target hit harder
        modifier = 1.0
        if self._support_umbrellas and self._shots_at.get(target.id, 0) >= 1:
            modifier = self.combat.focus_fire_bonus
        self._shots_at[target.id] = self._shots_at.get(target.id, 0) + 1
        d = dist(soldier.pos, target.pos)
        hit, damage = resolve_fire(
            soldier.pos,
            target.pos,
            self.world.cover_at(target.pos),
            d,
            self.combat,
            self._rng,
            modifier=modifier,
        )
        if hit:
            target.health -= damage
            discipline = self._fire_discipline_factor(soldier, target)
            ledger.add(soldier.callsign, "combat", cfg.hit_enemy * discipline)
            if target.health <= 0:
                target.alive = False
                ledger.add(soldier.callsign, "combat", cfg.kill_enemy * discipline)
                enemy_kills.append((soldier, target))

    def _report_contact(self, soldier: Soldier, ledger: RewardLedger) -> None:
        cfg = self.rewards_cfg
        if self._voice_only:
            self._report_contact_voice(soldier, ledger)
            return
        visible = self._visible_enemies(soldier)
        if not visible:
            return
        # Dedup credit (A4), adjudicated by the umpire against the whole-team
        # picture (under comm_model="range" too — the umpire hears everything
        # even when a distant station does not): the FIRST accurate report of
        # an enemy takes contact_new; a re-report refreshing intel that has
        # aged >= contact_refresh_age is worth exactly 0 (it genuinely extends
        # the picture's life); a report whose every enemy is already fresh on
        # the picture is pure noise and draws contact_redundant.
        new_intel = any(e.id not in self._known_enemies for e in visible)
        refreshes = any(
            e.id in self._known_enemies
            and self._step_count - self._known_enemies[e.id][2] >= cfg.contact_refresh_age
            for e in visible
        )
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
        if new_intel:
            ledger.add(soldier.callsign, "report", cfg.contact_new)
        elif not refreshes:
            ledger.add(soldier.callsign, "report", cfg.contact_redundant)
        self._charge_transmission(soldier, ledger, "report")
        nearest = visible[0]
        self._say(
            MessageKind.CONTACT,
            soldier.id,
            soldier.leader_id,
            lang.format_contact(self._addressee(soldier), soldier.callsign, len(visible), nearest.pos),
            useful=new_intel or refreshes,
            redundant=not (new_intel or refreshes),
        )

    def _report_contact_voice(self, soldier: Soldier, ledger: RewardLedger) -> None:
        """CONTACT under voice_only (§3.5): spoken to the direct superior from
        HELD intel — the agent's own picture (own sightings at their observed
        position and time, plus reports it heard) — not only from an enemy
        visible this exact tick. Novelty is the intended superior's, every
        local listener updates its own picture, and the captured coordinates
        travel unchanged (a relayed report never tracks the enemy)."""
        leader = self.roster.leader_of(soldier)
        held = self._agent_known.get(soldier.callsign, {})
        if leader is None or not held:
            return  # mask guards: nothing to say (defensive)
        entries = tuple(sorted(held.items(), key=lambda kv: (-kv[1][2], kv[0])))  # freshest first
        _eid, (nx, ny, _t) = entries[0]
        text = lang.format_contact(leader.callsign, soldier.callsign, len(entries), (int(nx), int(ny)))
        if not self._audible_to(leader, soldier.id):
            if self._may_prepare(soldier):
                # §4.2: the packet captures the selected held-intel
                # coordinates and source time; carriage never refreshes them
                self._prepare_packet(
                    soldier, "contact", leader, text, payload=entries,
                    source_step=max(e[1][2] for e in entries),
                )
            return
        self._charge_transmission(soldier, ledger, "report")
        self._deliver_contact(soldier, leader, entries, soldier, text, ledger)

    def _deliver_contact(
        self, origin: Soldier, recipient: Soldier, entries: tuple, speaker: Soldier,
        text: str, ledger: RewardLedger,
    ) -> str:
        """Land a CONTACT (spoken by ``speaker`` — the origin itself or a
        courier) on ``recipient``: novelty is the INTENDED superior's, the
        credit is the origin's, every local listener updates its own picture
        with the captured coordinates and source time. Returns the outcome:
        novel | refresh | redundant."""
        cfg = self.rewards_cfg
        step = self._step_count
        superior_picture = self._agent_known.setdefault(recipient.callsign, {})
        new_intel = any(eid not in superior_picture for eid, _ in entries)
        refreshes = any(
            eid in superior_picture and step - superior_picture[eid][2] >= cfg.contact_refresh_age
            for eid, _ in entries
        )
        for eid, _entry in entries:
            origin.reported_enemy_ids.add(eid)
        for listener in self.roster.living:
            if listener is speaker or not self._audible_to(listener, speaker.id):
                continue
            picture = self._agent_known.setdefault(listener.callsign, {})
            changed = False
            for eid, entry in entries:
                have = picture.get(eid)
                if have is None or have[2] < entry[2]:
                    if have is None or step - have[2] >= cfg.contact_refresh_age:
                        changed = True
                    picture[eid] = entry
            if changed:
                self._picture_changed_step[listener.callsign] = step
        origin.last_contact_report_step = step
        if new_intel:
            ledger.add(origin.callsign, "report", cfg.contact_new)
        elif not refreshes:
            ledger.add(origin.callsign, "report", cfg.contact_redundant)
        outcome = "novel" if new_intel else ("refresh" if refreshes else "redundant")
        self._say(
            MessageKind.CONTACT, speaker.id, recipient.id, text,
            useful=outcome != "redundant", redundant=outcome == "redundant",
            relayed_by=speaker.callsign if speaker is not origin else None,
        )
        return outcome

    def _reportable_acoustic(self, soldier: Soldier) -> tuple | None:
        """The acoustic report this agent would make: its freshest/strongest
        held non-friendly cue, else its freshest carried report. Returns
        (kind index, bearing, band, source step, strength) or None."""
        step = self._step_count
        own = [
            c for c in self._agent_cues.get(soldier.callsign, [])
            if c.side != "friendly" and c.ttl_remaining(step) >= 0
        ]
        if own:
            c = own[0]  # already in the stable (strength, age, id) order
            return (snd.SOUND_KINDS.index(c.kind), c.bearing, c.distance_band, c.event_step, c.strength)
        carried = [r for r in self._held_acoustic.get(soldier.callsign, []) if step - r[3] <= ACOUSTIC_REPORT_TTL]
        if carried:
            return max(carried, key=lambda r: (r[3], r[4], -r[0], -r[1]))
        return None

    def _report_acoustic_contact(self, soldier: Soldier, ledger: RewardLedger) -> None:
        """ACOUSTIC CONTACT (§3.6.3): a coarse heard-presence report — cue
        kind, bearing/distance bands and source step, never a grid reference.
        Same direct-voice / store-and-forward / listener-local novelty rules
        as CONTACT; it never touches the exact enemy picture."""
        leader = self.roster.leader_of(soldier)
        report = self._reportable_acoustic(soldier)
        if leader is None or report is None:
            return  # mask guards (defensive)
        kind_idx, bearing, band, source_step, _strength = report
        text = lang.format_acoustic_contact(
            leader.callsign, soldier.callsign, kind_idx, bearing, band, source_step
        )
        if not self._audible_to(leader, soldier.id):
            if self._may_prepare(soldier):
                # §4.2: only the cue's coarse fields travel — carriage can
                # never upgrade it into an exact CONTACT
                self._prepare_packet(
                    soldier, "acoustic_contact", leader, text, payload=report,
                    source_step=source_step,
                )
            return
        self._charge_transmission(soldier, ledger, "report")
        self._deliver_acoustic(soldier, leader, report, soldier, text, ledger)

    def _deliver_acoustic(
        self, origin: Soldier, recipient: Soldier, report: tuple, speaker: Soldier,
        text: str, ledger: RewardLedger,
    ) -> str:
        """Land an ACOUSTIC CONTACT on ``recipient`` (see _deliver_contact).
        Returns novel | redundant."""
        cfg = self.rewards_cfg
        kind_idx, bearing, band, source_step, _strength = report
        key = (kind_idx, bearing, band, source_step)
        step = self._step_count
        superior_memory = self._acoustic_received.setdefault(recipient.callsign, {})
        novel = key not in superior_memory
        for listener in self.roster.living:
            if listener is speaker or not self._audible_to(listener, speaker.id):
                continue
            memory = self._acoustic_received.setdefault(listener.callsign, {})
            if key not in memory:
                memory[key] = step
                self._held_acoustic.setdefault(listener.callsign, []).append(report)
        ledger.add(origin.callsign, "report", cfg.acoustic_contact_new if novel else cfg.contact_redundant)
        self._say(
            MessageKind.ACOUSTIC_CONTACT, speaker.id, recipient.id, text,
            useful=novel, redundant=not novel,
            relayed_by=speaker.callsign if speaker is not origin else None,
        )
        return "novel" if novel else "redundant"

    def _report_done(self, soldier: Soldier, ledger: RewardLedger) -> None:
        mission = soldier.mission
        if mission is None:
            return
        if self._voice_only:
            leader = self.roster.leader_of(soldier)
            if leader is None:
                return
            if not self._audible_to(leader, soldier.id):
                if self._may_prepare(soldier):
                    # §4.2: the claim captures claimant, task and claim time;
                    # it is heard and adjudicated only on delivery
                    obj_name = (
                        self.world.objectives[mission.objective_id].name
                        if mission.objective_id is not None
                        else mission.extra.get("control")
                    )
                    self._prepare_packet(
                        soldier, "done", leader,
                        lang.format_done(leader.callsign, soldier.callsign, mission.type, obj_name),
                        payload=(mission.type, mission.objective_id, mission.extra.get("control"),
                                 mission.step_assigned, self._step_count),
                    )
                return  # §3.4: a claim nobody hears is not adjudicated (mask guards)
        self._charge_transmission(soldier, ledger, "report")
        self._adjudicate_done(soldier, soldier, ledger)

    def _adjudicate_done(self, soldier: Soldier, speaker: Soldier, ledger: RewardLedger) -> str:
        """Hear and adjudicate ``soldier``'s MISSION COMPLETE claim, spoken by
        ``speaker`` (the claimant or its courier). Returns confirmed | rejected."""
        cfg = self.rewards_cfg
        mission = soldier.mission
        ctx = self._compliance_ctx(soldier, None, self._make_view(soldier))
        root_objective = (
            self.world.objective_by_name(self.spec_cfg.root_objective)
            if self.spec_cfg.root_objective
            else None
        )
        # same predicate the action mask admits on — see is_root_opord_claim
        is_root_mission_claim = is_root_opord_claim(
            soldier,
            self.roster,
            self.spec_cfg.root_mission,
            self._root_objective_id(),
        )
        # The root's OPORD claim reports the *operation*: it is judged against
        # the team success condition (e.g. objective clear AND held by anyone),
        # not against the claimant's personal end state — a commander reports
        # the mission complete when the unit achieved it, wherever it stands.
        # (RECON/SCREEN OPORDs also mirror the team counter into the mission
        # itself — refs #9 — so the is_complete branch would return the same
        # verdict; the explicit branch covers SEIZE/DEFEND-style root claims.)
        truthful = (
            self._check_success(root_objective)
            if is_root_mission_claim
            else is_complete(mission, ctx)
        )
        # v1.15: root_done_bonus is on the table for the episode's FIRST root
        # claim, and the first claim SPENDS it either way — a rejected probe
        # burns the bonus for the rest of the episode. Read the verdict before
        # the claim is adjudicated, because it has to be the state as it was
        # when the root decided to transmit. Subordinate DONEs are untouched:
        # they never earned this bonus and are not what was being farmed.
        earns_bonus = not self._root_claim_filed or not cfg.root_done_bonus_first_claim_only
        if is_root_mission_claim:
            self._root_claim_filed = True
        obj_name = (
            self.world.objectives[mission.objective_id].name
            if mission.objective_id is not None
            else mission.extra.get("control")  # ADVANCE: the control-measure name
        )
        self._say(
            MessageKind.DONE,
            speaker.id,
            soldier.leader_id,
            lang.format_done(self._addressee(soldier), soldier.callsign, mission.type, obj_name),
            useful=truthful,
            relayed_by=speaker.callsign if speaker is not soldier else None,
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
                self._root_close_step = self._step_count
                self._root_close_callsign = soldier.callsign
                # ...and collects the bonus only if this was the first claim.
                # The close itself is never withheld: the operation is over when
                # the root says so truthfully, whatever it said before.
                self._root_close_earns_bonus = earns_bonus
            ledger.add(soldier.callsign, "report", cfg.done_true)
            soldier.mission = None  # standing by for new orders
            if self._voice_only:
                self._note_mission_heard(responder_id, soldier, None)
            return "confirmed"
        self._say(
            MessageKind.DONE_REJECT,
            responder_id,
            soldier.id,
            lang.format_done_reject(soldier.callsign, responder_cs),
        )
        ledger.add(soldier.callsign, "report", cfg.done_false)
        # a rejected claim cannot be re-rolled every tick (v1.10): the
        # superior said continue the mission, so continue it
        soldier.last_done_reject_step = self._step_count
        return "rejected"

    def _sync_propose(self, soldier: Soldier, ledger: RewardLedger) -> None:
        """Trinôme bound proposal (A5-4), by VOICE — no radio involved.

        Registers a pending bound with every peer currently within
        ``voice_range`` (same element or adjacent trinôme). A re-proposal
        overwrites the previous one. Voice is still never net-arbitrated —
        shouting to the soldier beside you does not contend for the net —
        but it now pays airtime like every other learned transmission
        (owner's call, issue #18): while it was free it was the only action
        a policy could emit at no cost, and squad_screen_v4/ckpt_latest
        poured 93% of its traffic into it, 1173 messages an episode, to run
        the clock out. A speech act nobody pays for is an action sink.

        Charged to ``report``, not ``command``: SYNC is speech between peers,
        not authority exercised over a subordinate, and the ``flat`` ablation
        arm — which has no chain of command and must show command reward of
        exactly 0.0 — can still say GO.
        """
        peers = voice_peers(soldier, self.roster, self.spec_cfg.voice_range)
        if not peers:
            return  # mask guards; defensive — nothing said, nothing charged
        self._sync_pending[soldier.id] = (self._step_count, tuple(p.id for p in peers))
        self._charge_transmission(soldier, ledger, "report")
        self._say(
            MessageKind.SYNC_PROPOSE,
            soldier.id,
            None,
            lang.format_sync_propose(soldier.callsign, [p.callsign for p in peers]),
            voice=True,
            useful=any(self._audible_to(p, soldier.id) for p in peers),
        )

    def _sync_go(self, soldier: Soldier, ledger: RewardLedger, *, gesture: bool = False) -> None:
        """The bound signal (A5-4): GO! Synchronizes the proposer and every
        registered peer still alive for the next ``SYNC_WINDOW`` steps.

        Charged like any learned transmission (see ``_sync_propose``). A GO
        that never lands — no live proposal, or one past its TTL — says
        nothing and so costs nothing.

        Degraded communications (§3.6.5): under voice_only the spoken GO is a
        pre-arranged SOUND SIGNAL — it reaches a registered peer only within
        ``SIGNAL_RANGE`` with no wall between, and makes a ``signal`` event;
        the gesture variant reaches only peers with a current visual edge
        within ``GESTURE_RANGE`` and makes no sound. Signals never relay.
        """
        pending = self._sync_pending.pop(soldier.id, None)
        if pending is None:
            return
        propose_step, peer_ids = pending
        if self._step_count - propose_step > SYNC_PROPOSE_TTL:
            return  # stale proposal: the moment has passed
        if not gesture:
            self._charge_transmission(soldier, ledger, "report")
        group = (soldier.id, self._step_count)
        until = self._step_count + SYNC_WINDOW
        self._sync_until[soldier.id] = (until, group)
        receivers: list[str] = []
        for pid in peer_ids:
            peer = self.roster.by_id.get(pid)
            if peer is None or not peer.alive:
                continue
            if gesture:
                if not self._gesture_visible(soldier, peer):
                    continue
            elif self._voice_only and not self._signal_reaches(soldier, peer):
                continue
            self._sync_until[pid] = (until, group)
            receivers.append(peer.callsign)
        if gesture:
            self._say(
                MessageKind.SYNC_GO, soldier.id, None,
                lang.format_gesture_sync_go(soldier.callsign), voice=True,
                medium="gesture", heard_by=receivers, useful=bool(receivers),
            )
        else:
            self._say(
                MessageKind.SYNC_GO, soldier.id, None,
                lang.format_sync_go(soldier.callsign), voice=True,
                medium="signal" if self._voice_only else None,
                heard_by=receivers if self._voice_only else None,
                useful=bool(receivers),
                gesture_possible=self._voice_only and bool(receivers) and all(
                    self._gesture_visible(soldier, self.roster.by_callsign[cs]) for cs in receivers
                ),
            )

    def _synchronized(self, soldier: Soldier) -> tuple | None:
        """The soldier's active sync-group key, or None outside a window.

        The window spans ``SYNC_WINDOW`` ticks starting at the GO tick
        itself (exclusive upper bound, so ``sync_active`` in the obs reaches
        0 exactly when the window closes)."""
        entry = self._sync_until.get(soldier.id)
        if entry is None or not soldier.alive:
            return None
        until, group = entry
        return group if self._step_count < until else None

    def _sync_cover_peers(self, soldier: Soldier, group: tuple) -> list[Soldier]:
        """Synchronized group-mates COVERING this soldier's bound: static
        this tick with line of sight to a visible threat, or overwatching
        the mover itself (LOS to it)."""
        covering: list[Soldier] = []
        for other in self.roster.living:
            if other.id == soldier.id or self._synchronized(other) != group:
                continue
            if other.pos != other.prev_pos:
                continue  # the cover element does not move during the bound
            watches_threat = any(
                dist(other.pos, e.pos) <= self.combat.weapon_range
                and self.world.line_of_sight(other.pos, e.pos)
                for e in self._visible_enemies(other)
            )
            overwatches_mover = self.world.line_of_sight(other.pos, soldier.pos)
            if watches_threat or overwatches_mover:
                covering.append(other)
        return covering

    def _covered_by_sync(self, target: Soldier) -> bool:
        """Covered bound (A5-4): the target is mid-bound (synchronized and
        moved this tick) with >= 1 group-mate covering it — the existing
        covered-movement accuracy debuff applies (B5/P2 machinery at the
        binôme scale)."""
        group = self._synchronized(target)
        if group is None or target.pos == target.prev_pos:
            return False
        return bool(self._sync_cover_peers(target, group))

    def _bound_shaping(self, ledger: RewardLedger) -> None:
        """Pay the A5-4 bound bonus: a synchronized mover that closes NEW
        ground toward its own mission anchor while >= 1 group-mate covers it.
        Watermark keyed by the standing order — repeated propose/GO cycles
        never re-earn covered ground, so this telescopes like A5-3."""
        bonus = self.rewards_cfg.bound_bonus
        if bonus == 0.0:
            return
        for s in self.roster.living:
            group = self._synchronized(s)
            if group is None or s.mission is None or s.pos == s.prev_pos:
                continue
            if is_pending(s.mission, self._step_count):
                continue
            key = (s.mission.step_assigned,)
            d = self._anchor_distance(s)
            stored = self._bound_watermark.get(s.id)
            if stored is None or stored[0] != key:
                self._bound_watermark[s.id] = (key, d)
                continue
            if d >= stored[1] - 1e-9:
                continue
            self._bound_watermark[s.id] = (key, d)
            if self._sync_cover_peers(s, group):
                ledger.add(s.callsign, "compliance", bonus)

    def _execute_signal(
        self, soldier: Soldier, ledger: RewardLedger, *, gesture: bool = False
    ) -> None:
        """EXECUTE (A5-2): release ALL of this issuer's pending AT-MY-COMMAND
        orders. One broadcast frees every staged recipient at once — the
        manual's COMMANDEMENT DU BOND ("PREPAREZ-VOUS ... EN AVANT !").
        Released orders start binding now: step_assigned restamps.

        Degraded communications (§3.6.5): under voice_only the spoken form is
        a pre-arranged sound signal and releases only the pending recipients
        it reaches (``SIGNAL_RANGE``, no wall); the gesture form releases only
        recipients with a current visual edge (``GESTURE_RANGE``, LOS) and
        makes no sound. A recipient it does not reach stays staged."""
        if not gesture:
            self._charge_transmission(soldier, ledger, "command")
        released: list[str] = []
        for s in self.roster.living:
            m = s.mission
            if m is None or not m.awaiting_signal or m.issuer_id != soldier.id:
                continue
            if gesture:
                if not self._gesture_visible(soldier, s):
                    continue
            elif self._voice_only and not self._signal_reaches(soldier, s):
                continue
            m.awaiting_signal = False
            m.step_assigned = self._step_count
            released.append(s.callsign)
        if gesture:
            self._say(
                MessageKind.EXECUTE, soldier.id, None,
                lang.format_gesture_execute(soldier.callsign),
                medium="gesture", heard_by=released, useful=bool(released),
            )
        else:
            self._say(
                MessageKind.EXECUTE, soldier.id, None, lang.format_execute(soldier.callsign),
                medium="signal" if self._voice_only else None,
                heard_by=released if self._voice_only else None,
                useful=bool(released),
                gesture_possible=self._voice_only and bool(released) and all(
                    self._gesture_visible(soldier, self.roster.by_callsign[cs]) for cs in released
                ),
            )

    def _issue_formation(self, soldier: Soldier, spec: ActionSpec, ledger: RewardLedger) -> None:
        """Element stance order (A5-3): set the recipient LEADER's formation.

        A stance, not a mission — the recipient keeps its task, nothing is
        re-tasked or priced; reissuing the standing stance is churn. The
        stance persists until changed and dies with the leader.
        """
        subs = soldier.living_subordinates(self.roster)
        if spec.order_slot >= len(subs):
            return
        recipient = subs[spec.order_slot]
        if not recipient.living_subordinates(self.roster):
            return  # not an element leader (mask guards; defensive)
        if self._voice_only and not self._audible_to(recipient, soldier.id):
            if self._may_prepare(soldier):
                self._prepare_packet(
                    soldier, "order", recipient,
                    lang.format_formation_order(soldier.callsign, recipient.callsign, spec.order_formation),
                    payload=("formation", spec.order_formation), ack_required=True,
                )
            return  # §3.4: not spoken — nobody there to hear it (mask guards)
        self._charge_transmission(soldier, ledger, "command")
        if not self._audible_to(recipient, soldier.id):
            self._say(
                MessageKind.ORDER,
                soldier.id,
                recipient.id,
                lang.format_formation_order(soldier.callsign, recipient.callsign, spec.order_formation),
            )
            return
        self._apply_formation(soldier, recipient, spec.order_formation, soldier, ledger)

    def _apply_formation(
        self, issuer: Soldier, recipient: Soldier, formation, speaker: Soldier, ledger: RewardLedger,
    ) -> str:
        """Land a stance order on its recipient (spoken by the issuer or a
        courier). Returns applied | churn."""
        cfg = self.rewards_cfg
        if recipient.formation is formation:
            ledger.add(issuer.callsign, "command", cfg.order_churn)
            return "churn"
        recipient.formation = formation
        self._say(
            MessageKind.ORDER,
            speaker.id,
            recipient.id,
            lang.format_formation_order(issuer.callsign, recipient.callsign, formation),
            useful=True,
            relayed_by=speaker.callsign if speaker is not issuer else None,
        )
        if self.spec_cfg.auto_ack:
            self._say(
                MessageKind.ACK,
                recipient.id,
                issuer.id,
                lang.format_ack(issuer.callsign, recipient.callsign),
                useful=True,
            )
        return "applied"

    def _issue_order(self, soldier: Soldier, spec: ActionSpec, ledger: RewardLedger) -> None:
        if spec.order_formation is not None:
            self._issue_formation(soldier, spec, ledger)
            return
        subs = soldier.living_subordinates(self.roster)
        if spec.order_slot >= len(subs):
            return
        recipient = subs[spec.order_slot]
        objective = (
            self.world.objective_by_name(spec.order_objective) if spec.order_objective else None
        )
        obj_id = objective.id if objective else None
        control_name = spec.order_control  # ADVANCE: control-measure name
        # unit-targeted SUPPORT: the supported unit is the sibling in slot j
        supported_id: int | None = None
        if spec.order_mission is MissionType.SUPPORT:
            if spec.order_support_slot is None or spec.order_support_slot >= len(subs):
                return
            supported_id = subs[spec.order_support_slot].id

        if self._voice_only and not self._audible_to(recipient, soldier.id):
            if self._may_prepare(soldier):
                # §4.2: the packet captures the complete directive and timing
                # clause; nothing is communicated, charged or rewarded now
                target = self._order_target_name(spec.order_mission, obj_id, supported_id, control_name)
                self._prepare_packet(
                    soldier, "order", recipient,
                    lang.format_order(
                        soldier.callsign, recipient.callsign, spec.order_mission, target,
                        at_my_command=spec.order_amc,
                    ),
                    payload=("mission", spec.order_mission, obj_id, supported_id, control_name,
                             bool(spec.order_amc)),
                    ack_required=True,
                )
            return  # §3.4: an order that cannot be spoken is not an order (mask guards)

        # airtime (A4): the ORDER goes out on the net either way below
        self._charge_transmission(soldier, ledger, "command")

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
                control_name=control_name,
                awaiting_signal=spec.order_amc,
            )
            return
        self._adjudicate_order(
            soldier, recipient, spec.order_mission, obj_id, supported_id, control_name,
            bool(spec.order_amc), soldier, ledger,
        )

    def _order_target_name(self, mission_type, obj_id, supported_id, control_name) -> str | None:
        """The spoken target of an order (mirrors ``_assign_mission``)."""
        if mission_type is MissionType.SUPPORT and supported_id is not None:
            return self.roster.by_id[supported_id].callsign
        if control_name is not None:
            return control_name
        if obj_id is not None:
            return self.world.objectives[obj_id].name
        return None

    def _adjudicate_order(
        self, soldier: Soldier, recipient: Soldier, mission_type: MissionType,
        obj_id: int | None, supported_id: int | None, control_name: str | None,
        amc: bool, speaker: Soldier, ledger: RewardLedger,
    ) -> str:
        """Price and apply an order that REACHED its recipient — spoken by the
        issuer, or by a courier at delivery (§4.5: order reward at receipt).
        Returns applied | churn."""
        cfg = self.rewards_cfg
        # churn: reissuing the standing order is radio noise, not command —
        # a no-op (the mission is NOT restamped, so tenure keeps accruing).
        # A timing-qualifier change (pending vs. live) is a different order.
        if (
            recipient.mission is not None
            and recipient.mission.type is mission_type
            and recipient.mission.objective_id == obj_id
            and recipient.mission.extra.get("supported_id") == supported_id
            and recipient.mission.extra.get("control") == control_name
            and bool(recipient.mission.awaiting_signal) == bool(amc)
        ):
            ledger.add(soldier.callsign, "command", cfg.order_churn)
            self._log_order_pay(soldier, "churn", None, cfg.order_churn)
            return "churn"

        standing = recipient.mission
        intent_changed = (
            soldier.mission is not None
            and soldier.mission.step_assigned > recipient.last_order_step
        )

        # Re-task pricing (B5): replacing a standing order is an act of
        # command with real weight — the issuer pays
        #   order_retask_cost_base x (1 + order_retask_rank_scale x authority),
        # half price when only the mission TYPE changes on the same anchor —
        # UNLESS the tactical picture changed since the standing order landed
        # (the doctrine's "major, critical changes" carve-out, free): a
        # CONTACT on the net since, a casualty in the issuer's element since,
        # or the issuer's own mission changed since. The fourth exception —
        # the subordinate's truthful DONE — is structural: the confirmed
        # claim cleared its mission, so the next order is a fresh tasking,
        # never a re-task. The steep price never punishes legitimate
        # intervention; it prices command by whim. Supersedes the old
        # stability-window churn for tasked subordinates.
        if standing is not None:
            excepted, reason = self._retask_exception(soldier, recipient, intent_changed)
            same_anchor = self._standing_anchor_key(standing) == self._order_anchor_key(
                mission_type, obj_id, supported_id, recipient, control_name
            )
            cost = 0.0
            if not excepted and cfg.order_retask_cost_base != 0.0:
                cost = cfg.order_retask_cost_base * (
                    1.0 + cfg.order_retask_rank_scale * soldier.effective_authority
                )
                if same_anchor:
                    cost *= 0.5
                ledger.add(soldier.callsign, "command", cost)
            self._retask_log.append(
                {
                    "issuer": soldier.callsign,
                    "rank": soldier.effective_rank.name,
                    "authority": soldier.effective_authority,
                    "recipient": recipient.callsign,
                    "same_anchor": same_anchor,
                    "excepted": excepted,
                    "reason": reason,
                    "cost": cost,
                }
            )

        # fresh tasking: subordinate untasked, or the issuer's own mission
        # changed after the subordinate was last ordered (propagation credit)
        fresh_tasking = standing is None or intent_changed
        if fresh_tasking:
            quality = derivation_quality(soldier.mission.type if soldier.mission else None, mission_type)
            # refs #52: `tier`/`pay` only accumulate what the ledger is charged
            # below; they decide nothing. A fresh tasking whose task derives
            # from nothing the issuer holds is logged as `unpaid` rather than
            # dropped, or the unpaid share of a commander's traffic would read
            # as if it had never been issued.
            tier, pay = "unpaid", 0.0
            if quality >= 1.0:
                ledger.add(soldier.callsign, "command", cfg.order_preferred)
                tier, pay = "preferred", cfg.order_preferred
            elif quality > 0.0:
                ledger.add(soldier.callsign, "command", cfg.order_allowed)
                tier, pay = "allowed", cfg.order_allowed
            if (
                obj_id is not None
                and soldier.mission is not None
                and soldier.mission.objective_id == obj_id
            ):
                ledger.add(soldier.callsign, "command", cfg.order_objective_match)
                pay += cfg.order_objective_match
            self._log_order_pay(soldier, "fresh", tier, pay)
        elif standing is not None:
            # replaced a standing order without qualifying as a fresh tasking:
            # the order channel pays nothing here, and the re-task price above
            # is charged separately (see `retasks_by_rank`)
            self._log_order_pay(soldier, "retask", None, 0.0)

        self._assign_mission(
            issuer_id=soldier.id,
            issuer_cs=soldier.callsign,
            recipient=recipient,
            mission_type=mission_type,
            objective_id=obj_id,
            supported_id=supported_id,
            control_name=control_name,
            awaiting_signal=amc,
            speaker_id=speaker.id,
        )
        soldier.last_issued[recipient.id] = (mission_type, obj_id, supported_id)
        return "applied"

    def _retask_exception(
        self, issuer: Soldier, recipient: Soldier, intent_changed: bool
    ) -> tuple[bool, str | None]:
        """Is a re-task of ``recipient`` free for ``issuer`` right now, and why?

        The exception set — the exact conditions under which the tactical
        picture counts as changed since the recipient's standing order landed
        (mirrors the order-cooldown lifts, plus element casualties):

        * ``"contact"`` — a CONTACT report hit the net after the standing
          order was received (strictly later, like the cooldown lift);
        * ``"casualty"`` — a soldier anywhere in the issuer's element (its
          command subtree) died at or after the standing order's step (deaths
          resolve after the order phase, so a same-step casualty is news);
        * ``"intent"`` — the issuer's own mission changed after the
          subordinate was last ordered (the cooldown's other lift; this path
          also earns the fresh-tasking propagation credit).

        The subordinate's truthful DONE needs no clause here: the confirmed
        claim cleared its mission, so a follow-up order is a fresh tasking and
        never reaches re-task pricing at all.
        """
        contact_clock = self._contact_clock(issuer)
        if contact_clock is not None and contact_clock > recipient.last_order_step:
            return True, "contact"
        casualty_step = self._element_casualty_step.get(issuer.id)
        if casualty_step is not None and casualty_step >= recipient.last_order_step:
            return True, "casualty"
        if intent_changed:
            return True, "intent"
        return False, None

    @staticmethod
    def _standing_anchor_key(mission: Mission) -> tuple:
        """Identity of a standing mission's anchor, for rotation detection.

        Two orders share an anchor when they name the same objective, the
        same control measure, support the same soldier, rally (on the
        leader), or hold the same ground — a mission-TYPE change on the same
        anchor is half-price re-tasking; an anchor change is a full-price
        rotation.
        """
        if mission.type is MissionType.SUPPORT:
            return ("support", mission.extra.get("supported_id"))
        if mission.extra.get("control") is not None:
            return ("cm", mission.extra["control"])
        if mission.objective_id is not None:
            return ("obj", mission.objective_id)
        if mission.type is MissionType.RALLY:
            return ("rally",)
        return ("pos", (int(mission.anchor[0]), int(mission.anchor[1])))

    @staticmethod
    def _order_anchor_key(
        mission_type: MissionType,
        objective_id: int | None,
        supported_id: int | None,
        recipient: Soldier,
        control_name: str | None = None,
    ) -> tuple:
        """Anchor identity a new order WOULD have (mirrors ``_assign_mission``)."""
        if mission_type is MissionType.SUPPORT:
            return ("support", supported_id)
        if control_name is not None:
            return ("cm", control_name)
        if objective_id is not None:
            return ("obj", objective_id)
        if mission_type is MissionType.RALLY:
            return ("rally",)
        return ("pos", (int(recipient.pos[0]), int(recipient.pos[1])))

    def _assign_mission(
        self,
        issuer_id: int,
        issuer_cs: str,
        recipient: Soldier,
        mission_type: MissionType,
        objective_id: int | None,
        supported_id: int | None = None,
        control_name: str | None = None,
        effective_at: int | None = None,
        awaiting_signal: bool = False,
        speaker_id: int | None = None,
    ) -> bool:
        """Transmit an order; if the recipient hears it, apply it (+ WILCO).

        ``speaker_id`` (liaison, §4.5) is the courier actually speaking the
        issuer's canonical line at delivery; audibility is the speaker's.

        Under ``comm_model="global"`` every order is heard. Under ``"range"``
        an out-of-earshot recipient never receives the mission: the ORDER
        still lands on the transcript (it was transmitted), but nothing
        changes and no WILCO comes back — silence is the only clue.
        Timing qualifiers (A5-2): ``effective_at`` = the tick an "AT T PLUS
        n" order comes due; ``awaiting_signal`` = "AT MY COMMAND". A pending
        order stages the recipient near its current position until released.
        Returns True if the order was received and applied.
        """
        extra: dict = {}
        if mission_type is MissionType.SUPPORT and supported_id is not None:
            supported = self.roster.by_id[supported_id]
            anchor = supported.pos  # dynamic thereafter: tracks the supported soldier
            target = supported.callsign
            extra["supported_id"] = supported_id
        elif control_name is not None:
            # ADVANCE to a control measure: waypoint → its point; phase line →
            # the nearest point of the segment (dynamic — recomputed as the
            # agent moves); the side at receipt detects a later crossing
            cm = self.world.control_by_name(control_name)
            if cm is None:
                return False  # masked/validated upstream; defensive
            if hasattr(cm, "nearest_point"):
                anchor = cm.nearest_point(recipient.pos)
                extra["side"] = cm.side(recipient.pos)
            else:
                anchor = cm.pos
            target = control_name
            extra["control"] = control_name
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
        if effective_at is not None or awaiting_signal:
            # staging: until the order is effective, compliance is HOLD here
            extra["staging"] = recipient.pos
        speaker = issuer_id if speaker_id is None else speaker_id
        lands = self._audible_to(recipient, speaker)
        self._say(
            MessageKind.ORDER,
            speaker,
            recipient.id,
            lang.format_order(
                issuer_cs,
                recipient.callsign,
                mission_type,
                target,
                delay=(effective_at - self._step_count) if effective_at is not None else None,
                at_my_command=awaiting_signal,
            ),
            useful=lands,
            relayed_by=(
                self.roster.by_id[speaker].callsign if speaker != issuer_id and speaker in self.roster.by_id
                else None
            ),
        )
        if not lands:
            return False
        # HQ re-issuing the OPORD observation task to the senior commander is
        # team-adjudicated, exactly like the reset OPORD (refs #9)
        root_objective = (
            self.world.objective_by_name(self.spec_cfg.root_objective)
            if self.spec_cfg.root_objective
            else None
        )
        team_observation = (
            issuer_id == HQ_ID
            and recipient is self.roster.root()
            and mission_type is self.spec_cfg.root_mission
            and mission_type in (MissionType.RECON, MissionType.SCREEN)
            and objective_id == (root_objective.id if root_objective else None)
        )
        recipient.mission = Mission(
            type=mission_type,
            objective_id=objective_id,
            anchor=anchor,
            issuer_id=issuer_id,
            step_assigned=self._step_count,
            team_observation=team_observation,
            effective_at=effective_at,
            awaiting_signal=awaiting_signal,
            extra=extra,
        )
        recipient.last_order_step = self._step_count
        if self._voice_only:
            self._note_mission_heard(speaker, recipient, mission_type)
        if self.spec_cfg.auto_ack:
            self._say(
                MessageKind.ACK,
                recipient.id,
                issuer_id,
                lang.format_ack(issuer_cs, recipient.callsign),
                useful=True,
            )
        return True

    # ------------------------------------------------------------------ #
    # OpFor
    # ------------------------------------------------------------------ #

    def _check_trap(self, soldier: Soldier, ledger: RewardLedger, player_deaths: list[Soldier]) -> None:
        """First friendly stepping on a live device takes its damage; the trap
        is spent and revealed, and the umpire broadcasts a CASUALTY-style net
        message. Enemy-side ground truth: never in any blue observation."""
        for trap in self.traps:
            if trap.armed and soldier.pos == trap.pos:
                trap.armed = False
                trap.revealed = True
                # §3.6.1: the detonation is a sound at the trap cell; its
                # side is the TRIGGERING side — the noise indicates Blue
                # activity, which is what an OpFor anchor may investigate
                self._emit_sound("trap", trap.pos, "friendly", snd.TRAP_DETECT_RADIUS)
                self._say(
                    MessageKind.TRAP, HQ_ID, None, lang.format_trap(soldier.callsign, trap.pos)
                )
                self._damage_soldier(soldier, trap.damage, ledger, player_deaths)
                return

    def _damage_soldier(
        self, target: Soldier, damage: int, ledger: RewardLedger, player_deaths: list[Soldier]
    ) -> None:
        """Apply damage to a friendly (enemy fire or a trap), with the
        rank-weighted death economics; deaths queue for casualty processing."""
        cfg = self.rewards_cfg
        target.health -= damage
        ledger.add(target.callsign, "combat", cfg.took_hit)
        if target.health <= 0 and target.alive:
            target.alive = False
            target.health = 0
            # rank-weighted death: dying as a leader costs more
            weight = 1.0 + cfg.rank_casualty_scale * target.effective_authority
            ledger.add(target.callsign, "combat", cfg.death * weight)
            player_deaths.append(target)

    def _enemy_turn(self, enemy: Enemy, ledger: RewardLedger, player_deaths: list[Soldier]) -> None:
        visible_players = [
            s
            for s in self.roster.living
            if self.world.can_spot(
                enemy.pos, s.pos, self.combat.vision_range, self.combat.forest_vision_range
            )
        ]
        # heard-anchor investigation window (§3.6.4): active only under
        # tactical sound — 0 keeps the shipped deciders on their exact
        # pre-acoustic control flow (and RNG stream)
        sound_ttl = snd.SOUND_MEMORY_TTL if self._sound_on else 0
        if self.band is not None and enemy.mode == "brique":
            act, arg = self.band.member_decide(
                enemy,
                visible_players,
                self.roster.living,
                self.world,
                self._step_count,
                self.combat,
                self._rng,
                sound_ttl=sound_ttl,
            )
        else:
            act, arg = enemy_decide(
                enemy, visible_players, self.world, self._step_count, self.combat, self._rng,
                sound_ttl=sound_ttl,
            )
        if act == "move" and arg is not None and self.world.passable(arg):
            prev = enemy.pos
            enemy.pos = arg
            if enemy.investigating_step == self._step_count:
                self._opfor_investigating.add(enemy.id)
            if prev != enemy.pos:
                # sensor symmetry (§3.6.4): enemy movement makes the same
                # class of sound Blue movement does
                self._emit_sound(
                    "movement",
                    enemy.pos,
                    "hostile",
                    snd.movement_radius(self.world, prev, enemy.pos),
                    source_cs=f"E{enemy.id}",
                )
        elif act == "fire":
            enemy.fired_this_step = True  # oracle bookkeeping only
            self._emit_sound(
                "weapon_fire", enemy.pos, "hostile", snd.WEAPON_DETECT_RADIUS,
                source_cs=f"E{enemy.id}",
            )
            target: Soldier = arg
            # covered movement: firing at a supported element from inside an
            # in-position supporter's umbrella degrades the attacker's
            # accuracy; a covered trinôme bound (A5-4) earns the same debuff
            # (effects do not stack — one covered-movement modifier applies)
            modifier = 1.0
            if self._covered_by_support(target, enemy.pos) or self._covered_by_sync(target):
                modifier = self.combat.support_cover_accuracy
            d = dist(enemy.pos, target.pos)
            hit, damage = resolve_fire(
                enemy.pos,
                target.pos,
                self.world.cover_at(target.pos),
                d,
                self.combat,
                self._rng,
                modifier=modifier,
            )
            if hit:
                self._damage_soldier(target, damage, ledger, player_deaths)

    # ------------------------------------------------------------------ #
    # views, masks, observations, compliance
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # tactical acoustics (§3.6) — inert when sound_model="off"
    # ------------------------------------------------------------------ #

    @property
    def _sound_on(self) -> bool:
        return self.spec_cfg.sound_model == "tactical"

    def _emit_sound(
        self,
        kind: str,
        pos: tuple[int, int],
        side: str,
        base_radius: float,
        *,
        source_cs: str | None = None,
        message_index: int | None = None,
    ) -> snd.SoundEvent | None:
        """Create one immutable SoundEvent at its physical source.

        Under ``sound_model="off"`` this is a no-op: no event, no state, no
        RNG. Two simultaneous movers create two independent events — nothing
        here ever sums or merges them.
        """
        if not self._sound_on:
            return None
        event = snd.SoundEvent(
            id=self._sound_seq,
            step=self._step_count,
            pos=(int(pos[0]), int(pos[1])),
            side=side,
            kind=kind,
            base_radius=base_radius,
            source=source_cs,
            message_index=message_index,
        )
        self._sound_seq += 1
        self._step_sounds.append(event)
        if side == "friendly":
            # only Blue-revealing sounds can anchor an OpFor investigation
            self._pending_enemy_sounds.append(event)
        if source_cs is not None:
            # the emitter's own last footprint (kind + radius): observable to
            # itself so the policy can learn the consequence of its previous
            # choice without ever seeing who heard it
            self._own_sound[source_cs] = (kind, base_radius, self._step_count)
        return event

    def _deliver_sounds_to_enemies(self) -> None:
        """Give each enemy its own frozen estimated anchor from the freshest/
        strongest Blue sound it detects (§3.6.4). Per-member: one BRIQUE
        member hearing Blue updates nobody else. Positions are the enemies'
        pre-turn positions — delivery precedes the OpFor turn."""
        if not self._sound_on or not self._pending_enemy_sounds:
            self._pending_enemy_sounds = []
            return
        events = self._pending_enemy_sounds
        self._pending_enemy_sounds = []
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            best: tuple[float, int, int, snd.SoundEvent] | None = None
            for ev in events:
                strength = snd.received_strength(
                    self.world, ev.pos, enemy.pos, ev.base_radius
                )
                if strength is None:
                    continue
                ev.detected_by_hostile.append((enemy.id, strength))
                key = (-strength, self._step_count - ev.step, ev.id, ev)
                if best is None or key[:3] < best[:3]:
                    best = key
            if best is not None:
                ev = best[3]
                bearing = snd.bearing_sector(enemy.pos, ev.pos)
                band = snd.distance_band(dist(enemy.pos, ev.pos))
                # the anchor is built ONCE from the coarse fields and then
                # frozen — it never follows the hidden source
                enemy.heard_blue_anchor = snd.estimated_anchor(
                    enemy.pos, bearing, band, self.world
                )
                enemy.heard_blue_step = self._step_count

    def _attribute_cue_side(self, listener: Soldier, event: snd.SoundEvent) -> str:
        """The LISTENER's side attribution for a cue — never the oracle label.

        Friendly only when the listener also received the friendly semantic
        message or currently perceives the source cell; hostile only when a
        visible hostile at the source cell establishes the association;
        otherwise unknown.
        """
        if event.side == "friendly":
            if listener.callsign in event.heard_by:
                return "friendly"
            if dist(listener.pos, event.pos) <= snd.VISUAL_LINK_RANGE and (
                self.world.line_of_sight(listener.pos, event.pos)
            ):
                return "friendly"
            return "unknown"
        if any(e.pos == event.pos for e in self._visible_enemies(listener)):
            return "hostile"
        return "unknown"

    def _deliver_sounds_to_blue(self) -> None:
        """Turn this step's events into bounded coarse cues per Blue listener,
        then expire/truncate each memory by the stable §3.6.2 order."""
        if not self._sound_on:
            return
        step = self._step_count
        for s in self.roster.soldiers:
            cues = self._agent_cues.setdefault(s.callsign, [])
            if not s.alive:
                cues.clear()
                continue
            for ev in self._step_sounds:
                if ev.source == s.callsign:
                    continue  # own footprint is its own observation slot
                cue = snd.build_cue(
                    self.world, s.pos, ev, self._attribute_cue_side(s, ev)
                )
                if cue is not None:
                    ev.detected_by_friendly.append((s.callsign, cue.strength))
                    cues.append(cue)
            self._agent_cues[s.callsign] = snd.prune_cues(cues, step)

    def _audible_to(self, listener: Soldier, sender_id: int) -> bool:
        """Can ``listener`` hear a transmission from ``sender_id``?

        ``comm_model="global"`` (default): always. ``"range"``: only within
        ``comm_range`` (euclidean). The sender always hears itself. HQ is a
        high-power station: its traffic is always heard, and it always hears
        the root (the root's up-channel reports are adjudicated regardless).
        """
        if self._voice_only:
            # §3.1 / §3.3: low voice only — no radio, no high-power HQ
            # station, no umpire bypass for anyone after the briefing
            if sender_id == HQ_ID:
                return False
            sender = self.roster.by_id.get(sender_id)
            if sender is None:
                return False
            if sender.id == listener.id:
                return True
            return cohesion.voice_audible(self.world, sender, listener, self.spec_cfg.voice_range)
        if self.spec_cfg.comm_model != "range":
            return True
        if sender_id == HQ_ID:
            return True
        sender = self.roster.by_id.get(sender_id)
        if sender is None or sender.id == listener.id:
            return True
        return dist(sender.pos, listener.pos) <= self.spec_cfg.comm_range

    # ------------------------------------------------------------------ #
    # voice-only degraded communications (§3) — helpers
    # ------------------------------------------------------------------ #

    def _signal_reaches(self, sender: Soldier, listener: Soldier) -> bool:
        """A pre-arranged sound signal's fixed code reaches ``listener``."""
        return cohesion.signal_audible(self.world, sender, listener, snd.SIGNAL_RANGE)

    def _gesture_visible(self, sender: Soldier, listener: Soldier) -> bool:
        """A silent gesture is seen: a real current visual edge within
        GESTURE_RANGE (LOS required; a wall always blocks)."""
        return cohesion.friendly_visible(self.world, sender, listener, snd.GESTURE_RANGE)

    def _witnessed(self, observer: Soldier, cell: tuple[int, int]) -> bool:
        """Could ``observer`` perceive what happened at ``cell`` (the local
        friendly-visibility radius with LOS)? Used for casualties, whose
        subject is no longer alive to satisfy ``friendly_visible``."""
        return (
            observer.alive
            and dist(observer.pos, cell) <= snd.VISUAL_LINK_RANGE
            and bool(self.world.line_of_sight(observer.pos, cell))
        )

    def _superior_reachable(self, soldier: Soldier) -> bool:
        """Can this agent's reports reach its direct superior right now?
        Radio: always (HQ included). voice_only: the superior must be an
        embodied station in low-voice range — a root has none."""
        if not self._voice_only:
            return True
        leader = self.roster.leader_of(soldier)
        return leader is not None and self._audible_to(leader, soldier.id)

    def _contact_clock(self, issuer: Soldier) -> int | None:
        """The tactical-picture clock a re-task exception / cooldown lift is
        judged against: the force-wide last CONTACT on a radio net, the
        issuer's OWN picture-change step under voice_only (§3.5)."""
        if self._voice_only:
            return self._picture_changed_step.get(issuer.callsign)
        return self._last_net_contact_step

    def _init_friendly_state(self, root: Soldier) -> None:
        """Reset-time friendly knowledge (voice_only): the force departs
        together after the briefing — positions of one's leader and direct
        subordinates are known, the root's OPORD is known to all (briefed,
        not overheard), and no other mission is known."""
        step = self._step_count
        for s in self.roster.soldiers:
            related = []
            leader = self.roster.leader_of(s)
            if leader is not None:
                related.append(leader)
            related.extend(s.living_subordinates(self.roster))
            self._friendly_state[s.callsign] = {
                o.id: [o.pos, (o.mission.type if (o is root and o.mission) else None), step, step]
                for o in related
            }

    def _related(self, s: Soldier) -> list[Soldier]:
        leader = self.roster.leader_of(s)
        return ([leader] if leader is not None else []) + s.living_subordinates(self.roster)

    def _refresh_friendly_perception(self) -> None:
        """Seeing a nearby teammate refreshes its last-known position (never
        its mission — that needs a heard report). No remote movement
        refreshes anything."""
        step = self._step_count
        for s in self.roster.living:
            known = self._friendly_state.setdefault(s.callsign, {})
            for o in self._related(s):
                if cohesion.friendly_visible(self.world, s, o):
                    rec = known.get(o.id)
                    if rec is None:
                        known[o.id] = [o.pos, None, step, step]
                    else:
                        rec[0] = o.pos
                        rec[2] = step

    def _note_mission_heard(self, speaker_id: int, subject: Soldier, mission_type) -> None:
        """A heard report refreshes the SEMANTIC half of friendly state: every
        living station that heard ``speaker_id`` (the speaker included) now
        knows ``subject`` holds ``mission_type`` (None: stood down)."""
        step = self._step_count
        for listener in self.roster.living:
            if listener.id != speaker_id and not self._audible_to(listener, speaker_id):
                continue
            known = self._friendly_state.setdefault(listener.callsign, {})
            rec = known.get(subject.id)
            if rec is None:
                if subject.id not in {o.id for o in self._related(listener)}:
                    continue
                known[subject.id] = [subject.pos, mission_type, step, step]
            else:
                rec[1] = mission_type
                rec[3] = step

    def _succession_knowledge(self, successor: Soldier, replaced: Soldier) -> None:
        """Succession is structural; what observers KNOW of it is local: a
        station that can perceive the successor (or itself) inherits its
        record of the replaced position for the successor."""
        for listener in self.roster.living:
            known = self._friendly_state.setdefault(listener.callsign, {})
            if replaced.id in known and (
                listener is successor or cohesion.friendly_visible(self.world, listener, successor)
            ):
                rec = list(known.pop(replaced.id))
                rec[0] = successor.pos
                known[successor.id] = rec

    def _friendly_view(self, soldier: Soldier) -> dict | None:
        """The friendly-state dict the observation builder consumes under
        voice_only: id -> (seen now, last pos, last mission, age). None on a
        radio net (live telemetry)."""
        if not self._voice_only:
            return None
        step = self._step_count
        known = self._friendly_state.get(soldier.callsign, {})
        out: dict[int, tuple] = {}
        for o in self._related(soldier):
            rec = known.get(o.id)
            if rec is None:
                continue
            seen = cohesion.friendly_visible(self.world, soldier, o)
            out[o.id] = (seen, rec[0], rec[1], step - rec[2])
        return out

    def _detached_ids(self) -> frozenset[int]:
        """Soldiers on an active liaison duty — explicitly outside their
        originating element's cohesion denominator (§3.7)."""
        return frozenset(self._liaison)

    # ------------------------------------------------------------------ #
    # liaison and message packets (§4) — inert unless liaison_enabled
    # ------------------------------------------------------------------ #

    def _may_prepare(self, soldier: Soldier) -> bool:
        """May this agent spend the tick preparing a packet? Liaison enabled,
        outbox empty, not itself on courier duty."""
        return self._liaison_on and soldier.id not in self._outbox and soldier.id not in self._liaison

    def _log_packet(self, event: str, packet: lia.MessagePacket, **extra: object) -> None:
        self._packet_log.append(
            {"event": event, "packet": packet.id, "kind": packet.kind, "origin": packet.origin_cs,
             "recipient": packet.recipient_cs, "created": packet.created_step,
             "step": self._step_count, **extra}
        )

    def _prepare_packet(
        self, soldier: Soldier, kind: str, recipient: Soldier, text: str, *,
        payload: tuple = (), source_step: int | None = None, ack_required: bool = False,
    ) -> lia.MessagePacket:
        """§4.2: spend the tick writing the canonical line. No transcript
        message, no communication charge, no remote effect, no reward."""
        packet = lia.MessagePacket(
            id=self._packet_seq, kind=kind, origin_id=soldier.id, origin_cs=soldier.callsign,
            recipient_id=recipient.id, recipient_cs=recipient.callsign, text=text,
            created_step=self._step_count, source_step=source_step, ack_required=ack_required,
            payload=tuple(payload), holder_id=soldier.id,
        )
        self._packet_seq += 1
        self.packets.append(packet)
        self._outbox[soldier.id] = packet
        self._log_packet("prepared", packet)
        return packet

    def _held_packet(self, soldier: Soldier) -> lia.MessagePacket | None:
        task = self._liaison.get(soldier.id)
        if task is not None:
            return task.packet
        return self._outbox.get(soldier.id)

    def _packet_target(self, soldier: Soldier) -> Soldier | None:
        """The living holder of the position the packet in hand is going to."""
        task = self._liaison.get(soldier.id)
        if task is not None:
            return lia.resolve_position(task.current_target_id(), self.roster, self._successions)
        packet = self._outbox.get(soldier.id)
        if packet is None:
            return None
        return lia.resolve_position(packet.recipient_id, self.roster, self._successions)

    def _can_deliver(self, soldier: Soldier) -> bool:
        """§4.5: DELIVER_MESSAGE is legal only with the current recipient in
        voice range (and the packet unexpired)."""
        packet = self._held_packet(soldier)
        if packet is None or packet.expired(self._step_count):
            return False
        target = self._packet_target(soldier)
        return target is not None and target is not soldier and self._audible_to(target, soldier.id)

    def _dispatch_slots(self, soldier: Soldier) -> frozenset[int]:
        """Direct-subordinate slots that may be detached as the agent of
        liaison for the prepared packet (§4.3)."""
        packet = self._outbox.get(soldier.id)
        if packet is None or soldier.id in self._liaison:
            return frozenset()
        slots = set()
        for k, sub in enumerate(soldier.living_subordinates(self.roster)[:4]):
            if sub.id == packet.recipient_id or sub.id in self._liaison or sub.id in self._outbox:
                continue
            if not self._audible_to(sub, soldier.id):
                continue
            slots.add(k)
        return frozenset(slots)

    def _last_known_pos(self, observer: Soldier, other_id: int) -> tuple[int, int] | None:
        rec = self._friendly_state.get(observer.callsign, {}).get(other_id)
        return tuple(rec[0]) if rec is not None else None

    def _dispatch_liaison(self, soldier: Soldier, slot: int, ledger: RewardLedger) -> None:
        """§4.3: hand the prepared packet to the subordinate in ``slot`` — a
        local spoken order that consumes the tick; the packet changes owner
        and the carrier's tactical mission is suspended."""
        if slot not in self._dispatch_slots(soldier):
            return  # mask guards (defensive)
        packet = self._outbox.pop(soldier.id)
        carrier = soldier.living_subordinates(self.roster)[slot]
        anchor = self._last_known_pos(soldier, packet.recipient_id)
        if anchor is None:
            rcpt = self.roster.by_id.get(packet.recipient_id)
            anchor = tuple(rcpt.pos) if rcpt is not None else tuple(soldier.pos)
        task = lia.LiaisonTask(
            packet=packet, carrier_id=carrier.id, dispatched_step=self._step_count,
            anchor=anchor, suspended_mission=carrier.mission,
        )
        carrier.mission = None
        packet.holder_id = carrier.id
        packet.status = "dispatched"
        self._liaison[carrier.id] = task
        self._charge_transmission(soldier, ledger, "command")
        self._say(
            MessageKind.DISPATCH, soldier.id, carrier.id,
            lang.format_dispatch(carrier.callsign, soldier.callsign, packet.recipient_cs, packet.kind),
            useful=True,
        )
        self._log_packet("dispatched", packet, carrier=carrier.callsign)

    def _cancel_message(self, soldier: Soldier, ledger: RewardLedger) -> None:
        """§4.2: cancel one's prepared packet, or abandon one's courier duty.
        An internal command decision, counted as churn — never as speech."""
        cfg = self.rewards_cfg
        packet = self._outbox.pop(soldier.id, None)
        task = self._liaison.get(soldier.id)
        if packet is None and task is None:
            return  # mask guards (defensive)
        if packet is None:
            packet = task.packet
            self._end_liaison(task, "cancelled")
        packet.status = "cancelled"
        packet.holder_id = None
        ledger.add(soldier.callsign, "command", cfg.order_churn)
        self._log_order_pay(soldier, "churn", None, cfg.order_churn)
        self._log_packet("cancelled", packet)

    def _end_liaison(self, task: lia.LiaisonTask, how: str) -> None:
        """Close a courier's duty: restore the suspended mission unless a
        newer valid order arrived meanwhile (§4.5), drop the task."""
        carrier = self.roster.by_id.get(task.carrier_id)
        self._liaison.pop(task.carrier_id, None)
        if carrier is not None and carrier.alive and carrier.mission is None:
            carrier.mission = task.suspended_mission
        self._log_packet(how, task.packet, carrier=carrier.callsign if carrier else None,
                         outbound_path=task.outbound_path, return_path=task.return_path)

    def _deliver_message(self, soldier: Soldier, ledger: RewardLedger) -> None:
        """§4.5: speak the carried line to the current holder of the addressed
        position; the content is validated and credited exactly as direct
        speech would be, to the ORIGIN; a dispatched courier earns the
        liaison credit when the content is accepted."""
        cfg = self.rewards_cfg
        if not self._can_deliver(soldier):
            return  # mask guards (defensive)
        task = self._liaison.get(soldier.id)
        packet = self._held_packet(soldier)
        target = self._packet_target(soldier)
        origin = lia.resolve_position(packet.origin_id, self.roster, self._successions)
        step = self._step_count
        if task is not None and task.leg == "returning":
            # the receipt reaches the origin position: the cycle completes
            positive = bool(packet.receipt)
            text = (
                lang.format_receipt(target.callsign, soldier.callsign, packet.recipient_cs, positive)
                if packet.status != "undeliverable"
                else lang.format_undeliverable(target.callsign, soldier.callsign, packet.recipient_cs)
            )
            self._charge_transmission(soldier, ledger, "report")
            self._say(MessageKind.RECEIPT, soldier.id, target.id, text, useful=True)
            ledger.add(soldier.callsign, "report", cfg.liaison_receipt_return)
            self._log_packet("returned", packet, carrier=soldier.callsign, positive=positive,
                             latency=step - (packet.delivered_step or step))
            self._end_liaison(task, "completed")
            return
        # outbound delivery: the spoken line is the canonical text, the
        # speaker is whoever carries it
        self._charge_transmission(soldier, ledger, "report" if packet.kind != "order" else "command")
        outcome = self._land_packet(packet, origin, target, soldier, ledger)
        packet.delivered_step = step
        packet.status = "delivered"
        stale = self._packet_stale(packet, origin, target)
        self._log_packet("delivered", packet, carrier=soldier.callsign, outcome=outcome,
                         latency=step - packet.created_step, stale=stale,
                         courier=task is not None)
        accepted = outcome in _ACCEPTED
        if task is None:
            # self-carried: the origin spoke its own line; outbox cleared
            self._outbox.pop(soldier.id, None)
            packet.holder_id = None
            return
        if accepted:
            ledger.add(soldier.callsign, "report", cfg.liaison_delivery)
        if packet.ack_required:
            # the recipient's local WILCO / NEGATIVE becomes a receipt carried
            # back to the origin position; the order already stands
            packet.receipt = outcome in ("applied", "churn")
            task.leg = "returning"
            task.return_anchor = self._last_known_pos(soldier, packet.origin_id) or (
                tuple(origin.pos) if origin is not None else tuple(soldier.pos)
            )
            task.best_distance = float("inf")
            packet.status = "returning"
            return
        packet.holder_id = None
        self._end_liaison(task, "completed")

    def _packet_stale(self, packet, origin, target) -> bool:
        """§9 orders_stale_at_delivery definition: the issuer's own mission or
        the recipient's standing order changed after the packet was written."""
        if packet.kind != "order":
            return False
        changed = []
        if origin is not None and origin.mission is not None:
            changed.append(origin.mission.step_assigned > packet.created_step)
        if target is not None and target.mission is not None:
            changed.append(target.mission.step_assigned > packet.created_step)
        return any(changed)

    def _land_packet(self, packet, origin, target, speaker: Soldier, ledger: RewardLedger) -> str:
        """Apply a delivered packet with the same validation as direct speech
        at this tick (§4.5). Returns the outcome class."""
        kind = packet.kind
        if kind == "order":
            return self._land_order(packet, origin, target, speaker, ledger)
        if origin is None:
            # a report whose origin position is vacant is still information:
            # credited to nobody, pictures still updated by the spoken line
            origin = speaker
        if kind == "contact":
            return self._deliver_contact(origin, target, packet.payload, speaker, packet.text, ledger)
        if kind == "acoustic_contact":
            return self._deliver_acoustic(origin, target, packet.payload, speaker, packet.text, ledger)
        if kind == "sitrep":
            fresh, prep_step = packet.payload
            cfg = self.rewards_cfg
            ledger.add(origin.callsign, "report", cfg.sitrep_fresh if fresh else cfg.sitrep_spam)
            origin.last_sitrep_step = max(origin.last_sitrep_step, prep_step)
            self._say(MessageKind.SITREP, speaker.id, target.id, packet.text, useful=fresh,
                      redundant=not fresh, relayed_by=speaker.callsign if speaker is not origin else None)
            return "fresh" if fresh else "spam"
        if kind == "done":
            mission_type, obj_id, control, assigned, _claim_step = packet.payload
            m = origin.mission
            if (
                m is None or m.type is not mission_type or m.objective_id != obj_id
                or m.extra.get("control") != control or m.step_assigned != assigned
                or self.roster.leader_of(origin) is not target
            ):
                # the claimed task is no longer the claimant's standing order,
                # or the recipient no longer commands it: rejected, aloud
                self._say(MessageKind.DONE, speaker.id, target.id, packet.text, useful=False,
                          relayed_by=speaker.callsign if speaker is not origin else None)
                self._say(MessageKind.DONE_REJECT, target.id, origin.id,
                          lang.format_done_reject(origin.callsign, target.callsign))
                return "obsolete"
            return self._adjudicate_done(origin, speaker, ledger)
        return "unknown"

    def _land_order(self, packet, origin, target, speaker: Soldier, ledger: RewardLedger) -> str:
        """Validate and apply a carried order at delivery: the origin position's
        current holder must still command the recipient, the recipient must
        still be able to hold the task. Invalid → spoken NEGATIVE, never a
        silent reinterpretation."""
        payload = packet.payload
        lawful = (
            origin is not None and origin.alive and origin is not target
            and target.leader_id == origin.id
            and origin.effective_authority > target.effective_authority
        )
        if lawful and payload and payload[0] == "mission":
            _tag, mission_type, obj_id, supported_id, control_name, amc = payload
            if target.effective_authority < min_hold_authority(mission_type):
                lawful = False
            if obj_id is not None and obj_id >= len(self.world.objectives):
                lawful = False
            if mission_type is MissionType.SUPPORT:
                supported = self.roster.by_id.get(supported_id)
                if supported is None or not supported.alive or supported.leader_id != origin.id:
                    lawful = False
            if control_name is not None and self.world.control_by_name(control_name) is None:
                lawful = False
        if (
            lawful and payload and payload[0] == "formation"
            and not target.living_subordinates(self.roster)
        ):
            lawful = False
        if not lawful:
            self._say(MessageKind.ORDER, speaker.id, target.id, packet.text, useful=False,
                      relayed_by=speaker.callsign)
            self._say(MessageKind.ACK, target.id, packet.origin_id,
                      lang.format_negative(packet.origin_cs, target.callsign), useful=False)
            return "rejected"
        if payload[0] == "formation":
            return self._apply_formation(origin, target, payload[1], speaker, ledger)
        _tag, mission_type, obj_id, supported_id, control_name, amc = payload
        return self._adjudicate_order(
            origin, target, mission_type, obj_id, supported_id, control_name, amc, speaker, ledger,
        )

    def _update_liaisons(self, ledger: RewardLedger) -> None:
        """Per-step duty bookkeeping (§4.4-4.5): a dead courier loses its
        packet (no backup, nobody told), an expired packet ends the duty, a
        vacant destination turns the courier around with an undeliverable
        notice, and NEW closure toward the fixed anchor pays progress."""
        cfg = self.rewards_cfg
        step = self._step_count
        for origin_id, packet in list(self._outbox.items()):
            holder = self.roster.by_id.get(origin_id)
            if holder is None or not holder.alive:
                packet.status = "lost"
                packet.holder_id = None
                del self._outbox[origin_id]
                self._log_packet("lost", packet)
            elif packet.expired(step):
                packet.status = "expired"
                packet.holder_id = None
                del self._outbox[origin_id]
                self._log_packet("expired", packet)
        for carrier_id, task in list(self._liaison.items()):
            carrier = self.roster.by_id.get(carrier_id)
            packet = task.packet
            if carrier is None or not carrier.alive:
                packet.status = "lost"
                packet.holder_id = None
                self._liaison.pop(carrier_id, None)
                self._log_packet("lost", packet, carrier=carrier.callsign if carrier else None)
                continue
            if packet.expired(step):
                packet.status = "expired"
                packet.holder_id = None
                self._end_liaison(task, "expired")
                continue
            target = lia.resolve_position(task.current_target_id(), self.roster, self._successions)
            if target is None:
                if task.leg == "outbound":
                    # §4.4: the addressed position is vacant — return with an
                    # undeliverable notice, never retarget clairvoyantly
                    packet.status = "undeliverable"
                    packet.receipt = False
                    task.leg = "returning"
                    task.return_anchor = self._last_known_pos(carrier, packet.origin_id) or task.anchor
                    task.best_distance = float("inf")
                    self._log_packet("undeliverable", packet, carrier=carrier.callsign)
                    continue
                packet.holder_id = None
                self._end_liaison(task, "orphaned")
                continue
            if carrier.pos != carrier.prev_pos:
                if task.leg == "outbound":
                    task.outbound_path += 1
                else:
                    task.return_path += 1
            d = dist(carrier.pos, task.current_anchor())
            if task.best_distance == float("inf"):
                task.best_distance = d
                continue
            if d < task.best_distance - 1e-9:
                cells = int(math.floor(task.best_distance) - math.floor(d))
                task.best_distance = d
                if cells > 0:
                    ledger.add(carrier.callsign, "report", cfg.liaison_progress * cells)

    def _liaison_view(self, soldier: Soldier) -> dict | None:
        """The liaison observation block's inputs (§4.4, §5): fixed anchor,
        local perception of the intended recipient, packet age, leg, receipt.
        None when the scenario cannot prepare packets."""
        if not self._liaison_on:
            return None
        step = self._step_count
        task = self._liaison.get(soldier.id)
        outbox = self._outbox.get(soldier.id)
        packet = task.packet if task is not None else outbox
        view: dict = {
            "outbox_kind": outbox.kind if outbox is not None else None,
            "carry_kind": task.packet.kind if task is not None else None,
            "ttl": max(0.0, packet.ttl_remaining(step)) / lia.PACKET_TTL if packet is not None else 0.0,
            "returning": task is not None and task.leg == "returning",
            "anchor": None, "recipient_pos": None,
            "can_deliver": self._can_deliver(soldier),
            "receipt": packet.receipt if (task is not None and task.leg == "returning") else None,
        }
        if packet is not None:
            if task is not None:
                view["anchor"] = task.current_anchor()
            else:
                view["anchor"] = self._last_known_pos(soldier, packet.recipient_id)
            target = self._packet_target(soldier)
            if target is not None and cohesion.friendly_visible(self.world, soldier, target):
                view["recipient_pos"] = tuple(target.pos)
        return view

    def _update_visual_links(self) -> None:
        """Rebuild every element's visual-link graph (§3.7) and the station
        status of every member, every tick. Link state is recorded for every
        comm model (a metric and an observation); only voice_only prices it."""
        step = self._step_count
        detached = self._detached_ids()
        new_link: dict[str, tuple[bool | None, int]] = {}
        new_station: dict[str, tuple[bool | None, float]] = {}
        pairs = in_voice = 0
        for leader in self.roster.living:
            members, linked = cohesion.element_links(
                self.world, leader, self.roster, detached=detached
            )
            if not members:
                continue
            intact = all(m.id in linked for m in members)
            for m in members:
                connected = m.id in linked
                prev = self._link_state.get(m.callsign, (None, 0))
                age = 0 if connected else prev[1] + 1
                new_link[m.callsign] = (connected, age)
                new_station[m.callsign] = cohesion.formation_station(leader, m)
                pairs += 1
                in_voice += cohesion.voice_audible(self.world, leader, m, self.spec_cfg.voice_range)
            prev = self._link_state.get(leader.callsign, (None, 0))
            new_link[leader.callsign] = (intact, 0 if intact else prev[1] + 1)
        self._link_state = new_link
        self._station = new_station
        self._command_pairs = (pairs, in_voice)
        del step

    def _visual_link_penalty(self, ledger: RewardLedger) -> None:
        """voice_only only: ``visual_link_broken`` per disconnected non-detached
        member-step, capped per element-step. A leader whose element is
        broken is not charged for its members; each member is charged once."""
        cfg = self.rewards_cfg
        if not self._voice_only or cfg.visual_link_broken == 0.0:
            return
        detached = self._detached_ids()
        for leader in self.roster.living:
            members = [m for m in leader.living_subordinates(self.roster) if m.id not in detached]
            broken = [m for m in members if self._link_state.get(m.callsign, (True, 0))[0] is False]
            if not broken:
                continue
            total = cfg.visual_link_broken * len(broken)
            if cfg.visual_link_broken_element_cap < 0.0:
                total = max(total, cfg.visual_link_broken_element_cap)
            share = total / len(broken)
            for m in broken:
                ledger.add(m.callsign, "compliance", share)

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
            if self._local_pictures
            else self._known_enemies
        )
        cadence = self.spec_cfg.sitrep_cadence
        sitrep_due = (
            min(1.0, max(0.0, (self._step_count - soldier.last_sitrep_step) / cadence))
            if cadence
            else None
        )
        # trinôme sync (A5-4): party to a live proposal / inside a GO window
        step = self._step_count
        sync_pending = any(
            step - propose_step <= SYNC_PROPOSE_TTL
            and (proposer_id == soldier.id or soldier.id in peer_ids)
            for proposer_id, (propose_step, peer_ids) in self._sync_pending.items()
        )
        entry = self._sync_until.get(soldier.id)
        sync_active = (
            max(0.0, (entry[0] - step) / SYNC_WINDOW) if entry is not None else 0.0
        )
        link_intact, link_age = self._link_state.get(soldier.callsign, (None, 0))
        station, form_err = self._station.get(soldier.callsign, (None, 0.0))
        return AgentView(
            visible_enemies=self._visible_enemies(soldier),
            known_enemies=[(x, y) for (x, y, _t) in known.values()],
            step=self._step_count,
            sitrep_due=sitrep_due,
            sync_pending=sync_pending,
            sync_active=sync_active,
            episode_progress=min(1.0, step / max(1, self.spec_cfg.max_steps)),
            time_to_contact=self._time_to_contact(),
            sound_on=self._sound_on,
            voice_only=self._voice_only,
            cues=list(self._agent_cues.get(soldier.callsign, [])),
            own_sound=self._own_sound.get(soldier.callsign),
            has_reportable_cue=self._sound_on and self._reportable_acoustic(soldier) is not None,
            link_intact=link_intact,
            link_break_age=link_age,
            station=station,
            formation_error=form_err,
            friendly_state=self._friendly_view(soldier),
            liaison=self._liaison_view(soldier),
        )

    def _draw_h_hour(self) -> None:
        """Draw the actual H-hour from the scenario's band (v1.10).

        The OPORD announces the band's MIDPOINT — the nominal H the cohort
        plans against — while the assault actually arrives anywhere in the
        band. A defense that waits for the announced tick is late half the
        time, so the trained habit has to be *set early*, not *timed exactly*.
        Consumes RNG only when the scenario has a preparation period, so seeds
        for every other scenario reproduce exactly as before.
        """
        band = self.spec_cfg.assault_h_hour
        if band is None:
            return
        lo, hi = band
        self._h_hour = int(self._rng.integers(lo, hi + 1))
        # what HQ announces is defined once, in cohort.config, so the radio
        # wording, the observation countdown and the published briefing
        # cannot drift apart (issue #12)
        self._h_hour_nominal = announced_assault_step(self.spec_cfg)

    def _in_preparation(self) -> bool:
        """True while the assault is still forming up (v1.10).

        The OpFor exists from step 0 — it is on the map, oracle-visible, and
        spottable by anyone who goes looking — but it does not move, fire, or
        advance until H. A defense is entitled to the time it was told it had.
        """
        return self._h_hour is not None and self._step_count < self._h_hour

    def _time_to_contact(self) -> float:
        """Countdown to the NOMINAL announced H-hour, 1.0 → 0.0 (v1.10).

        0.0 in scenarios with no preparation period, and once H has passed.
        The actual arrival is jittered around the nominal H (the OPORD is an
        estimate, not a timetable), so this warns without guaranteeing.
        """
        nominal = self._h_hour_nominal
        if nominal is None or nominal <= 0 or self._step_count >= nominal:
            return 0.0
        return (nominal - self._step_count) / nominal

    def _compute_views(self) -> dict[str, AgentView]:
        return {s.callsign: self._make_view(s) for s in self.roster.soldiers}

    def _mask_for(self, soldier: Soldier) -> np.ndarray:
        visible = self._visible_enemies(soldier)
        in_range = any(dist(soldier.pos, e.pos) <= self.combat.weapon_range for e in visible)
        pending_sync = self._sync_pending.get(soldier.id)
        has_pending_sync = (
            pending_sync is not None
            and self._step_count - pending_sync[0] <= SYNC_PROPOSE_TTL
        )
        return compute_mask(
            soldier,
            self.roster,
            self.world,
            in_range and soldier.ammo > 0,
            bool(visible),
            order_cooldown=self.spec_cfg.order_cooldown,
            done_cooldown=self.spec_cfg.done_cooldown,
            root_mission=self.spec_cfg.root_mission,
            root_objective_id=self._root_objective_id(),
            step=self._step_count,
            net_contact_step=self._contact_clock(soldier),
            ablation=self.spec_cfg.ablation,
            has_voice_peer=bool(
                voice_peers(soldier, self.roster, self.spec_cfg.voice_range)
            ),
            has_pending_sync=has_pending_sync,
            superior_reachable=self._superior_reachable(soldier),
            held_contact_intel=(
                self._voice_only and bool(self._agent_known.get(soldier.callsign))
            ),
            has_reportable_cue=(
                self._sound_on and self._reportable_acoustic(soldier) is not None
            ),
            gestures_enabled=self._voice_only,
            gesture_execute_audience=self._voice_only and any(
                sub.mission is not None
                and sub.mission.awaiting_signal
                and sub.mission.issuer_id == soldier.id
                and self._gesture_visible(soldier, sub)
                for sub in soldier.living_subordinates(self.roster)
            ),
            gesture_sync_audience=self._voice_only and has_pending_sync and any(
                (peer := self.roster.by_id.get(pid)) is not None
                and self._gesture_visible(soldier, peer)
                for pid in pending_sync[1]
            ),
            reachable_sub_ids=(
                frozenset(
                    sub.id
                    for sub in soldier.living_subordinates(self.roster)
                    if self._audible_to(sub, soldier.id)
                )
                if self._voice_only
                else None
            ),
            liaison=self._liaison_on,
            carrying=soldier.id in self._liaison,
            outbox_empty=soldier.id not in self._outbox,
            can_deliver=self._liaison_on and self._can_deliver(soldier),
            can_cancel=self._liaison_on and (soldier.id in self._outbox or soldier.id in self._liaison),
            dispatch_slots=self._dispatch_slots(soldier) if self._liaison_on else frozenset(),
        )

    def _observe(self, soldier: Soldier, view: AgentView) -> dict[str, np.ndarray]:
        return {
            "observation": build_observation(
                soldier, self.roster, self.world, view,
                profile=self.spec_cfg.observation_profile,
            ),
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
        RALLY — the leader —, SUPPORT — the supported soldier — and
        ADVANCE to a phase line — the segment's nearest point)."""
        mission = soldier.mission
        if mission is None:
            return None
        if is_pending(mission, self._step_count):
            # a pending order (A5-2) stages where it was received
            return mission.extra.get("staging", mission.anchor)
        anchor = mission.anchor
        if mission.type is MissionType.RALLY:
            leader = self.roster.leader_of(soldier)
            if leader is not None:
                anchor = leader.pos
        elif mission.type is MissionType.SUPPORT:
            supported = self.roster.by_id.get(mission.extra.get("supported_id"))
            if supported is not None and supported.alive:
                anchor = supported.pos
        elif mission.type is MissionType.ADVANCE and mission.extra.get("control") is not None:
            cm = self.world.control_by_name(mission.extra["control"])
            if cm is not None and hasattr(cm, "nearest_point"):
                anchor = cm.nearest_point(soldier.pos)
        return anchor

    def _anchor_moved(self, soldier: Soldier) -> bool:
        """Did this soldier's mission anchor move this step?

        Only anchors that are themselves soldiers can move — SUPPORT's
        supported unit and RALLY's leader. A fixed anchor (objective,
        waypoint, phase line) never does, so this reads False and those
        missions keep their existing semantics untouched.
        """
        mission = soldier.mission
        if mission is None or is_pending(mission, self._step_count):
            return False
        anchor_soldier = None
        if mission.type is MissionType.SUPPORT:
            anchor_soldier = self.roster.by_id.get(mission.extra.get("supported_id"))
        elif mission.type is MissionType.RALLY:
            anchor_soldier = self.roster.leader_of(soldier)
        if anchor_soldier is None or not anchor_soldier.alive:
            return False
        return anchor_soldier.pos != anchor_soldier.prev_pos

    def _anchor_distance(self, soldier: Soldier) -> float:
        anchor = self._mission_anchor(soldier)
        if anchor is None:
            return 0.0
        return dist(soldier.pos, anchor)

    def _formation_shaping(self, ledger: RewardLedger) -> None:
        """Pay the A5-3 formation bonus for this tick.

        For every living leader with a stance and an effective mission that
        MOVED this tick: if the leader's anchor distance sets a new minimum
        under the current (mission, stance) — genuine progress on the march —
        every element member standing at its formation station (geometry in
        the leader's heading frame, ``core.missions.in_formation``) earns
        ``RewardConfig.formation_bonus``. The watermark makes the total
        payout per (order, stance) bounded by the initial distance: no
        perpetual farm, so ``max_step_farm`` is unaffected.
        """
        bonus = self.rewards_cfg.formation_bonus
        if bonus == 0.0:
            return
        step = self._step_count
        for leader in self.roster.living:
            if leader.formation is None or leader.mission is None:
                continue
            if is_pending(leader.mission, step):
                continue
            key = (leader.mission.step_assigned, leader.formation)
            d = self._anchor_distance(leader)
            stored = self._formation_watermark.get(leader.id)
            if stored is None or stored[0] != key:
                self._formation_watermark[leader.id] = (key, d)
                continue
            if leader.pos == leader.prev_pos or d >= stored[1] - 1e-9:
                continue  # not marching, or no new closure
            self._formation_watermark[leader.id] = (key, d)
            for member in leader.living_subordinates(self.roster):
                if in_formation(leader.formation, leader.pos, leader.heading, member.pos):
                    ledger.add(member.callsign, "compliance", bonus)

    def _update_crossing(self, soldier: Soldier) -> None:
        """ADVANCE to a phase line: mark the mission crossed when the agent's
        side of the line flips relative to where the order was received."""
        mission = soldier.mission
        if (
            mission is None
            or mission.type is not MissionType.ADVANCE
            or mission.extra.get("control") is None
            or mission.extra.get("crossed")
            or is_pending(mission, self._step_count)  # staging: not advancing yet
        ):
            return
        cm = self.world.control_by_name(mission.extra["control"])
        if cm is None or not hasattr(cm, "side"):
            return  # waypoint: reach, not cross
        side_now = cm.side(soldier.pos)
        side_then = mission.extra.get("side", 0)
        if side_then == 0 and side_now != 0:
            mission.extra["side"] = side_now  # order received ON the line
        elif side_now != 0 and side_now != side_then:
            mission.extra["crossed"] = True

    def _team_observer(self, objective: Any) -> Soldier | None:
        """The first living soldier observing ``objective``: on the RECON
        observation ring (RECON and SCREEN share the radius) with LOS. The
        squad-aggregated observation predicate, shared by the team success
        counter and the root OPORD's in-position credit (refs #9)."""
        for s in self.roster.living:
            if dist(s.pos, objective.pos) <= IN_POSITION_RADIUS[
                MissionType.RECON
            ] and self.world.line_of_sight(s.pos, objective.pos):
                return s
        return None

    def _in_mission_position(self, soldier: Soldier, dist_now: float | None = None) -> bool:
        """Is the soldier at its mission station (radius + LOS where required)?

        SUPPORT stations relative to the supported soldier: within radius and
        holding line of sight to *it* (you cannot support what you cannot see).

        A root-held (OPORD) RECON / SCREEN is team-adjudicated (refs #9): the
        *operation* is in position when any living member observes the
        objective — the commander earns its posture credit from cover.
        """
        mission = soldier.mission
        if mission is None:
            return False
        if is_pending(mission, self._step_count):
            # staging (A5-2): in position = holding near where the order landed
            if dist_now is None:
                dist_now = self._anchor_distance(soldier)
            return dist_now <= IN_POSITION_RADIUS[MissionType.HOLD]
        if mission.team_observation and mission.objective_id is not None:
            return self._team_observer(self.world.objectives[mission.objective_id]) is not None
        if dist_now is None:
            dist_now = self._anchor_distance(soldier)
        anchor = self._mission_anchor(soldier)
        radius = IN_POSITION_RADIUS[mission.type]
        if mission.type is MissionType.SUPPORT:
            # "In support position" must MEAN the umbrella can do its work.
            # The table said 10.0 while combat.support_umbrella is 8.0, so a
            # supporter could sit at 9-10 cells drawing full posture pay while
            # covering nothing at all — the reward describing support the
            # environment never delivered. Bind the two: the station is the
            # umbrella, per scenario, so tuning one can never silently
            # decouple it from the other again.
            radius = min(radius, float(self.combat.support_umbrella))
        in_position = dist_now <= radius
        if mission.type in LOS_REQUIRED:
            in_position = in_position and self.world.line_of_sight(
                soldier.pos, (int(anchor[0]), int(anchor[1]))
            )
        return in_position

    def _fire_discipline_factor(self, soldier: Soldier, target: Enemy | None = None) -> float:
        """Combat-reward multiplier enforcing fire discipline by mission.

        SCREEN is weapons tight: firing earns nothing (and compliance already
        penalizes it). Static postures (OBSERVE/SUPPORT/COVER/DEFEND/DENY/
        HOLD) pay for engagements fought FROM the mission position — chasing
        kills off the position earns nothing. RECON (which may engage, per
        PROTERRE), assault tasks, and untasked agents are free.

        Defense-of-the-position carve-out (v1.9 defend diagnosis): a
        position-anchored shooter also earns full pay when its TARGET stands
        inside the position's engagement envelope (anchor distance <=
        IN_POSITION_RADIUS + weapon_range) — an enemy there is assaulting the
        position, and fire delivered against the assault is the mission,
        wherever the melee has pushed the defender. The oracle showed the v6
        human TL firing on 0.5% of its threatened opportunities (every other
        agent: 90-99%): off its 3.5-cell disc its fire earned nothing, so the
        policy never received a gradient toward fighting back. Hunting kills
        AWAY from the anchor still pays zero — the v1.2 sally exploit
        (defenders dying 32:5 chasing kills off the objective) stays closed.
        """
        if not self.rewards_cfg.fire_discipline or soldier.mission is None:
            return 1.0
        mt = soldier.mission.type
        if is_pending(soldier.mission, self._step_count):
            mt = MissionType.HOLD  # staging (A5-2): fire pays only from the staging spot
        if mt in WEAPONS_TIGHT:
            return 0.0
        if mt in POSITION_ANCHORED_FIRE:
            if self._in_mission_position(soldier):
                return 1.0
            if target is not None:
                anchor = self._mission_anchor(soldier)
                if anchor is not None and dist(
                    target.pos, (int(anchor[0]), int(anchor[1]))
                ) <= IN_POSITION_RADIUS[mt] + self.combat.weapon_range:
                    return 1.0
            return 0.0
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

    def _covered_by_support(self, target: Soldier, shooter_pos: tuple[int, int]) -> bool:
        """Covered movement: is ``target`` protected from a shot fired at it
        from ``shooter_pos``? True when the target belongs to a supported
        element and the shooter stands inside the umbrella of that element's
        in-position supporter (computed from this step's snapshot)."""
        return any(
            target.id in element and dist(shooter_pos, supporter.pos) <= self.combat.support_umbrella
            for supporter, element in self._support_umbrellas
        )

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
            if self._voice_only:
                self._note_mission_heard(s.id, s, None)

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
            anchor_moved=self._anchor_moved(soldier),
            fired=soldier.fired_this_step,
            visible_enemies=len(view.visible_enemies),
            enemies_at_objective=enemies_at_obj,
            dist_to_leader=dist(soldier.pos, leader.pos) if leader is not None else float("inf"),
        )

    def _defend_terminal_scale(self) -> float:
        """Terminal multiplier for this episode: 1.0 unless holding with losses.

        Non-defend roots are untouched — the five scenarios sitting at 1.00
        success under the flat terminal keep exactly the economics they
        converged on, which is why this is scoped rather than global.

        Bodies are weighted by rank the way casualties already are
        (``rank_casualty_scale``), so losing the commander costs more of the
        payout than losing a rifleman — the measured half of the v1.11 defend
        regression was commander death going 0.24 -> 0.61.

        The weights use INTRINSIC rank, not ``effective_authority``, and that
        is load-bearing: succession promotes a survivor into the dead leader's
        slot, so an effective-authority sum over the living RISES after the
        commander falls and would report a force that got stronger by losing
        its commander. Intrinsic rank fixes the denominator at the force that
        started the episode and lets the numerator only ever fall.
        """
        if self.spec_cfg.root_mission not in (MissionType.DEFEND, MissionType.DENY):
            return 1.0
        scale = self.rewards_cfg.rank_casualty_scale
        total = alive = 0.0
        for s in self.roster.soldiers:
            weight = 1.0 + scale * AUTHORITY[s.rank]
            total += weight
            if s.alive:
                alive += weight
        return self.rewards_cfg.survivor_multiplier(alive, total)

    def _band_neutralized(self, root_obj: Any) -> bool:
        """BRIQUE terminal semantics: the band is out of the fight.

        True when the band is destroyed (no living member), OR it has
        scattered AND fully broken contact: scatter is irreversible, so once
        every living member stands >= ``band.break_contact_dist`` from every
        living friendly AND from the root objective, the band never returns.
        """
        living = [e for e in self.enemies if e.alive]
        if not living:
            return True
        if self.band is None or self.band.intent != "scatter":
            return False
        pts = [s.pos for s in self.roster.living]
        if root_obj is not None:
            pts.append(root_obj.pos)
        d = self.spec_cfg.band.break_contact_dist
        return all(dist(e.pos, p) >= d for e in living for p in pts)

    def _objective_held(self, root_obj: Any) -> bool:
        """A DEFEND/DENY position is held: no living enemy stands on it and a
        living friendly mans it (consistent with the objective_lost economics)."""
        if root_obj is None:
            return True
        clear = not any(
            e.alive and dist(e.pos, root_obj.pos) <= root_obj.radius + 1.0 for e in self.enemies
        )
        manned = any(
            dist(s.pos, root_obj.pos) <= root_obj.radius + 1.0 for s in self.roster.living
        )
        return clear and manned

    def _position_occupied(self, root_obj: Any) -> bool:
        """Is the defended position OCCUPIED — a living friendly standing on it?

        The ``manned`` half of :meth:`_objective_held`, and deliberately only
        that half. Occupation is what a DEFEND conserves; the ``clear`` half —
        no hostile within the footprint — is a statement about safety, and an
        enemy assaulting into contact on the position is the mission arriving,
        not the mission failing. Measured on the committed defend checkpoints,
        scoring the strict conjunction instead costs 29 of 100 episodes, and 26
        of the 40 first breaks it counts are exactly "the assault got here".
        """
        if root_obj is None:
            return True
        return any(
            dist(s.pos, root_obj.pos) <= root_obj.radius + 1.0 for s in self.roster.living
        )

    def _horizon_defense(self) -> int | None:
        """The ordered horizon of this episode's defense, or None.

        None on every scenario that is not a DEFEND / DENY root ordered to a
        stated hour — those keep the pre-v1.14 criterion exactly.

        ``HOLDS_GROUND`` is the same predicate ``language.format_opord`` gates
        the spoken HOLD UNTIL clause on: HQ says exactly what is scored.
        """
        cfg = self.spec_cfg
        if cfg.root_mission not in HOLDS_GROUND:
            return None
        return cfg.defend_horizon

    def _update_defend_hold(self, root_obj: Any) -> None:
        """Latch the permanent loss of a horizon defense (v1.14).

        From H onward the position must be occupied every step. The first step
        it is not, the mission has failed and stays failed: there is no retake,
        by decision — a position handed over and walked back onto was not held.
        Before H nothing is adjudicated; the preparation period exists so the
        defense can occupy the ground it was ordered to hold.
        """
        horizon = self._horizon_defense()
        if horizon is None or self._defend_lost_step is not None:
            return
        if self._success_step is not None:
            # the operation is already won: the grace window is aftermath, and
            # ground given up in it did not fail anything. Without this guard
            # the latch keeps recording past the decision and reads as a
            # failure that never happened — measured on defend_brique_v9, 17
            # latched losses against 12 lost episodes, the 5 extra all after
            # an early release had already ended the mission. No verdict moves
            # either way (success is locked at T0); what moves is whether the
            # number means what its name says.
            return
        if self._step_count < (self._h_hour or 0):
            return
        if not self._position_occupied(root_obj):
            self._defend_lost_step = self._step_count

    def _root_objective_id(self) -> int | None:
        """Objective id named by the OPORD, or None when the scenario has none."""
        name = self.spec_cfg.root_objective
        if not name:
            return None
        obj = self.world.objective_by_name(name)
        return obj.id if obj is not None else None

    def _check_success(self, root_obj: Any) -> bool:
        mission = self.spec_cfg.root_mission
        living_enemies = [e for e in self.enemies if e.alive]
        horizon = self._horizon_defense()
        if horizon is not None:
            # v1.14, owner's decision: DEFEND success is conservation of the
            # position, with minimum casualties — not annihilation of the
            # enemy. Failure is permanent and is latched by
            # ``_update_defend_hold``; from H the operation is won at the first
            # step the threat is out of the fight (early release) or the
            # ordered horizon is reached, whichever comes first. Casualties are
            # priced by ``defend_survivor_scale`` on the terminal, never gated
            # here — a gate would rebuild the forfeiture asymmetry that caused
            # the D4 collapse.
            if self._defend_lost_step is not None:
                return False
            if self._step_count < (self._h_hour or 0):
                return False
            if not self._position_occupied(root_obj):
                return False
            return self._band_neutralized(root_obj) or self._step_count >= horizon
        if (
            self.spec_cfg.opfor_mode == "brique"
            and mission in (MissionType.DEFEND, MissionType.DENY)
        ):
            # Asymmetric defense: success = the band destroyed, OR the band
            # scattered with contact broken, while the objective is held.
            # (SEIZE-rooted BRIQUE scenarios keep the standard SEIZE check:
            # the OPORD is about the objective — a scattered band never
            # blocks it, but destroying the band does not seize anything.)
            return self._band_neutralized(root_obj) and self._objective_held(root_obj)
        if mission in (MissionType.DEFEND, MissionType.DENY, MissionType.CLEAR):
            return not living_enemies
        if mission is MissionType.SEIZE:
            if root_obj is None:
                return not living_enemies
            clear = not any(dist(e.pos, root_obj.pos) <= root_obj.radius + 1.0 for e in living_enemies)
            occupied = any(dist(s.pos, root_obj.pos) <= root_obj.radius for s in self.roster.living)
            return clear and occupied
        if mission in (MissionType.RECON, MissionType.SCREEN):
            return self._team_observe_steps >= TEAM_OBSERVE_STEPS
        return False

    # ------------------------------------------------------------------ #
    # radio + human interface
    # ------------------------------------------------------------------ #

    def _addressee(self, soldier: Soldier) -> str:
        leader = self.roster.leader_of(soldier)
        return leader.callsign if leader is not None else "HQ"

    def _semantic_audience(self, sender_id: int, kind: MessageKind, recipient: int | None) -> list[str]:
        """Living stations that actually receive a transmission's SEMANTIC
        content (sender excluded). Audit metadata for the trace/oracle —
        ``heard_by`` never enters an agent observation wholesale."""
        if kind is MessageKind.OPORD and self._voice_only:
            # the pre-departure briefing: delivered to its addressee and to
            # nobody else's picture (§3.3)
            rcpt = self.roster.by_id.get(recipient) if recipient is not None else None
            return [rcpt.callsign] if rcpt is not None else []
        sender_cs = None
        if sender_id != HQ_ID:
            sender = self.roster.by_id.get(sender_id)
            sender_cs = sender.callsign if sender is not None else None
        return [
            s.callsign
            for s in self.roster.living
            if s.callsign != sender_cs and self._audible_to(s, sender_id)
        ]

    def _message_medium(self, kind: MessageKind, sender: int, voice: bool) -> str:
        """Label the channel a message travelled on (§8 provenance).

        ``briefing`` — the pre-departure OPORD; ``external`` — umpire
        adjudication from an unembodied HQ (CASUALTY, TRAP, ENDEX, verdicts
        answered for HQ); ``voice`` — spoken; ``radio`` — transmitted.
        """
        if kind is MessageKind.OPORD:
            return "briefing"
        if sender == HQ_ID:
            return "external"
        if voice or self._voice_only:
            return "voice"
        return "radio"

    def _say(
        self, kind: MessageKind, sender: int, recipient: int | None, text: str,
        *, voice: bool = False, medium: str | None = None,
        heard_by: list[str] | None = None, **meta: object,
    ) -> None:
        """Put one message on the transcript (text only — the repo's schema
        invariant) and record its audit metadata beside it: ``medium``
        (briefing / radio / voice / signal / gesture / external), the
        stations that actually received the semantics, and any caller-stated
        outcome flags (``useful``, ``redundant``, ``gesture_possible``) the
        §9 metrics read. ``heard_by`` may be given by a caller that knows
        its exact semantic audience (signals, gestures); otherwise it is the
        audibility rule's audience."""
        msg = Message(
            step=self._step_count, kind=kind, sender_id=sender,
            recipient_id=recipient, text=text, voice=voice,
        )
        self.transcript.add(msg)
        self.last_messages.append(msg)
        if heard_by is None:
            heard_by = self._semantic_audience(sender, kind, recipient)
        self.last_message_meta.append(
            {
                "medium": medium or self._message_medium(kind, sender, voice),
                "heard_by": list(heard_by),
                **meta,
            }
        )
        # §3.6.1: an embodied speaker makes noise — every emitted utterance,
        # local automatic replies included, is a voice event at the speaker;
        # the pre-arranged EXECUTE / SYNC_GO forms are louder signal events.
        # Unembodied HQ (briefing / external umpire lines) and silent
        # gestures emit nothing.
        if sender != HQ_ID and self._sound_on and medium != "gesture":
            speaker = self.roster.by_id.get(sender)
            if speaker is not None and speaker.alive:
                signal = kind in (MessageKind.EXECUTE, MessageKind.SYNC_GO)
                event = self._emit_sound(
                    "signal" if signal else "voice",
                    speaker.pos,
                    "friendly",
                    snd.SIGNAL_DETECT_RADIUS if signal else snd.VOICE_DETECT_RADIUS,
                    source_cs=speaker.callsign,
                    message_index=len(self.transcript) - 1,
                )
                if event is not None:
                    event.heard_by = list(heard_by)

    def inject_order(self, text: str, issuer: str = "HQ") -> Message:
        """Let a human speak on the net: parse and apply an order.

        ``issuer`` is "HQ" (may order anyone) or a callsign (must outrank the
        recipient and have them as a direct subordinate). Returns the ORDER
        message; raises ``OrderParseError`` / ``PermissionError`` otherwise.
        """
        if issuer.upper() == "HQ" and self._voice_only:
            msg = (
                "comm_model='voice_only': there is no remote HQ station after the "
                "briefing — command as the embodied root callsign instead."
            )
            raise PermissionError(msg)
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
        # element stance order (A5-3): no mission payload — set and return
        if parsed.formation is not None:
            if not recipient.living_subordinates(self.roster):
                msg = (
                    f"{recipient.callsign} leads no element: FORMATION is an "
                    "element-level stance, ordered to a leader."
                )
                raise PermissionError(msg)
            recipient.formation = parsed.formation
            self._say(
                MessageKind.ORDER,
                issuer_id,
                recipient.id,
                lang.format_formation_order(issuer_cs, recipient.callsign, parsed.formation),
            )
            if self.spec_cfg.auto_ack:
                self._say(
                    MessageKind.ACK,
                    recipient.id,
                    issuer_id,
                    lang.format_ack(issuer_cs, recipient.callsign),
                )
            return self.last_messages[-1] if self.last_messages else self.transcript.messages[-1]
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
        if parsed.control_name and self.world.control_by_name(parsed.control_name) is None:
            msg = f"No control measure named {parsed.control_name!r} on this map."
            raise lang.OrderParseError(msg)
        self._assign_mission(
            issuer_id=issuer_id,
            issuer_cs=issuer_cs,
            recipient=recipient,
            mission_type=parsed.mission,
            objective_id=objective.id if objective else None,
            supported_id=supported_id,
            control_name=parsed.control_name,
            effective_at=(
                self._step_count + parsed.delay if parsed.delay is not None else None
            ),
            awaiting_signal=parsed.at_my_command,
        )
        return self.last_messages[-1] if self.last_messages else self.transcript.messages[-1]

    def inject_execute(self, issuer: str = "HQ") -> Message:
        """A human issuer broadcasts EXECUTE, releasing all its pending
        AT-MY-COMMAND orders (A5-2). ``issuer`` is "HQ" or a callsign."""
        if issuer.upper() == "HQ" and self._voice_only:
            msg = (
                "comm_model='voice_only': there is no remote HQ station after the "
                "briefing — signal EXECUTE as the embodied root callsign instead."
            )
            raise PermissionError(msg)
        if issuer.upper() == "HQ":
            issuer_id, issuer_cs = HQ_ID, "HQ"
        else:
            issuing = self.roster.by_callsign.get(issuer.upper())
            if issuing is None or not issuing.alive:
                msg = f"No living station {issuer!r} on the net."
                raise lang.OrderParseError(msg)
            issuer_id, issuer_cs = issuing.id, issuing.callsign
        issuing = self.roster.by_id.get(issuer_id)
        released: list[str] = []
        for s in self.roster.living:
            m = s.mission
            if m is None or not m.awaiting_signal or m.issuer_id != issuer_id:
                continue
            if self._voice_only and issuing is not None and not self._signal_reaches(issuing, s):
                continue
            m.awaiting_signal = False
            m.step_assigned = self._step_count
            released.append(s.callsign)
        self._say(
            MessageKind.EXECUTE, issuer_id, None, lang.format_execute(issuer_cs),
            medium="signal" if self._voice_only else None,
            heard_by=released if self._voice_only else None,
        )
        return self.last_messages[-1] if self.last_messages else self.transcript.messages[-1]

    # ------------------------------------------------------------------ #
    # static briefing + ground-truth oracle (external observers only)
    # ------------------------------------------------------------------ #

    def briefing(self) -> dict:
        """Static operations overlay for this env's scenario (issue #10).

        Objective/control-measure coordinates, map size, root tasking and the
        engagement envelope — pure function of the scenario, identical across
        episodes, valid before ``reset()``. Header material for an episode
        stream: it leaks no per-episode state, which is exactly why an
        external monitor may consume it. See ``cohort.config.briefing``.
        """
        from cohort.config import briefing

        return briefing(self.spec_cfg)

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
