"""Behavioral metrics suite: measure what "behaves like its rank" means (B2).

Success rate says *whether* the cohort wins; this module measures *how it
behaves* while doing so, per evaluation run:

* obedience latency      — order received → first compliant action
* A5-2 staging           — orders staged (AT MY COMMAND / AT T PLUS n), how
  many were released and after how long, and how many were **abandoned**:
  staged and then never released (refs issue #15: obedience latency cannot
  see a staged order, because its recipient complies by standing still —
  which is exactly why counting the staged tick made staging read as instant
  obedience and pulled the latency mean toward 0)
* report precision/recall — CONTACT reports vs. enemies actually seen
  (the oracle-side ground truth: per-step visibility)
* doctrine-preference rate — share of issued orders that were the preferred
  derivation of the issuer's own mission, reported next to the doctrine
  *containment* rate and the ordered-task mix (refs issue #14: preference is
  ``allowed[0]``, so since A5 put ADVANCE in the DEFEND / SEIZE / RECON
  tables, a policy that adopts ADVANCE wholesale scores ~0 preference with
  zero doctrine violations — without the split, catalog adoption is
  indistinguishable from disregarding doctrine)
* ordered-task availability — the same mix against what the order mask
  actually offered, per issued order (refs issue #16: the mask does not offer
  the tasks in equal numbers, so a raw share conflates "the policy declined
  this" with "this was barely on the menu" — and the confound runs in
  *opposite directions* by scenario family, so it flatters a policy in one
  and slanders it in another). ``share / availability`` is the selection
  lift, with 1.00 the masked-random floor
* false-COMPLETE rate    — MISSION COMPLETE claims rejected by the umpire,
  pooled and for the root's own claims (refs issue #23: the root's channel is
  the one that closes an operation, and a fireteam's riflemen can carry the
  pooled rate on their own)
* COMPLETE claims / claiming episode — claims transmitted over the episodes
  that carried any (refs issue #23: a rejection *ratio* reads 13-accepted-of-13
  and 27-accepted-of-128 as the same shape of object. 1.00 is a policy filing a
  report when it believes the end state holds; a large number is one spamming a
  channel, and the two can share a rejection rate exactly)
* COMPLETE claim rate    — claims transmitted over the agent-steps at which
  claiming was admissible (refs issue #13: zero DONE reports is either a shut
  channel or a declined opportunity, and only this denominator says which)
* succession recovery    — leader death → all orphaned subordinates re-tasked
* subordinate coverage   — share of steps every living subordinate is tasked
* human exposure         — the human root's distance to the nearest living
  enemy, its entries into the objective observation ring, and its death rate
  (refs issue #9: rolling success is blind to a policy re-learning to walk
  the commander into the ring, so checkpoint selection for preservation
  claims needs these numbers)
* fight disposition       — *where* the cohort fights once the enemy is on
  it: cover occupancy and distance from the root objective, measured only
  over the (soldier, step) pairs under threat, plus the pass/fail gate built
  on them (refs issue #11: an outside measurement showed the defend miss is
  positional, not mortal — halving the root-death rate bought no success,
  while every defend policy that ever cleared its bound fought <= 2.9 cells
  out with cover >= 0.79 and the one that missed fought 9.09 out at 0.06)
* clock expiry + traffic composition — the share of episodes that ran the
  step ceiling out, and what the net carried while they did: total messages
  per episode split into command traffic (orders, EXECUTE) and voice traffic
  (SYNC PROPOSE / GO, never net-arbitrated, and charged airtime since #18
  closed that hole — the split is still worth keeping because the two
  channels answer different questions, not because one is free). Refs issue
  #18: three collapsed checkpoints (`squad_recon_v6`, `squad_screen_v4`,
  `squad_screen_v5` at `ckpt_latest`) sat at the ceiling in 30/30 episodes
  while transmitting 2.5x *more* than their own successful `ckpt_best` —
  and the run digest's only volume number, `tx_per_agent_step`, counts
  charged transmissions only, so it read 0.029 ("the radio went quiet")
  through a message flood. Every channel had a counter; the net had no
  denominator
* the close route: timing vs volume (refs issue #35) — on a continuous-posture
  root the operation is closed by the root's SITREP, and
  `closed_on_root_report_rate` says how often that happened. It saturates:
  a root that transmits every third step is certain to have reported at or
  after the success step, so the close is its by default. `root_sitreps_per_
  episode`, `closes_per_root_sitrep` and `closed_on_cadence_report_rate` are
  the denominator and the conditioning that separate a policy which learned
  *when* to report from one that bought the close with volume
* success axis (refs issue #21) — a floor on `success_rate`, gated only once
  `timeout_rate` clears its own ceiling. Issue #21 pre-registered and
  CONFIRMED a premise ("no defend scenario ever collapsed" reads as the D4
  stall, and none do) while finding a scoping gap: the defend family's worst
  measured runs collapse DEFEAT-shaped instead of stall-shaped —
  `fireteam_defend_v6b` succeeds 1/30 at only 2/30 timeout — and the D4 stall
  detector above, tuned to >= 28/30 timeout, never trips on them. The repo's
  own composite gate caught all four documented cases anyway, but on
  `human_death_rate`, because a wiped team's commander usually dies with it —
  right for a reason other than the one it names. This axis reads
  `success_rate` directly so a wipe is measured, not inferred

The pipeline has two halves, split so the math is unit-testable:

1. :class:`TraceRecorder` — hooks into an evaluation episode (see
   ``cohort.training.evaluate.run_episode``) and accumulates a plain-dict
   **trace**: per-step soldier/enemy state, per-step ground-truth visibility,
   per-soldier compliance scores (recomputed exactly as the environment
   scores them), and the radio transcript. Reading the environment consumes
   no randomness, so a recorded episode is bit-identical to an unrecorded
   one under the same seed.
2. Pure functions over the trace — :func:`episode_behavior` reduces one
   trace to event counts/latency lists, :func:`aggregate_behavior` pools
   episodes into the run-level summary written to ``behavior.json``.

Every metric's precise definition (formula and edge cases: no orders, no
contacts, no humans, no successions) lives in ``docs/metrics.md``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from cohort.core import language as lang
from cohort.core.missions import (
    IN_POSITION_RADIUS,
    MissionType,
    allowed_derivations,
    compliance,
    is_pending,
)
from cohort.core.units import CombatParams
from cohort.env.actions import is_done_admissible, order_options
from cohort.env.rewards import RewardConfig

if TYPE_CHECKING:
    from cohort.env.cohort_env import CohortEnv

#: Radius of the objective observation ring used for human ring entries —
#: RECON/SCREEN share it; it is where issue #9 measured the root's exposure.
RING_RADIUS: float = IN_POSITION_RADIUS[MissionType.RECON]

#: Fallback threat radius for traces recorded before the scenario's own
#: weapon range was written into them. A (soldier, step) pair counts as
#: "under threat" when a living enemy stands within this distance — i.e. when
#: the OpFor can actually shoot that soldier, which is what makes the pair
#: informative about *where the unit chose to fight*.
THREAT_RADIUS: float = CombatParams().weapon_range

#: Positional regression gate for DEFEND roots (refs issue #11). Bounds are
#: set from the measured record, not from taste: `_v5` (24/30) and
#: `defend_brique_v1` (27/30) held cover 0.79/0.96 at 2.90/1.99 cells, while
#: `_v6` (14/30) and `_v7` (12/30) sat at 0.496/0.060 and 3.46/9.09. The
#: floor and ceiling are placed in the empty band between the two groups, so
#: the gate separates every checkpoint on record without hair-splitting.
DEFEND_COVER_FLOOR: float = 0.40
DEFEND_OBJECTIVE_DIST_CEILING: float = 5.0

#: Clock-expiry gate (refs issue #18), applied to every root mission: the
#: share of evaluated episodes that ended by running the step ceiling out.
#: Bound set from the measured record at 10 episodes/checkpoint, seeds
#: 500-509, over every checkpoint that loads under the v1.10 spaces — the
#: healthy band tops out at 0.2 (`fireteam_defend_v9/best`, `_v8/latest`,
#: both 2/10) and the three stalled checkpoints all sit at exactly 1.0
#: (`squad_recon_v6`, `squad_screen_v4`, `squad_screen_v5`, all at
#: `ckpt_latest`). 0.5 is the middle of an empty band, and it is also the
#: point past which a policy fails more often by the clock than it succeeds.
TIMEOUT_RATE_CEILING: float = 0.5

#: Success-axis gate (refs issue #21): a floor on ``success_rate``, checked
#: only once ``timeout_rate`` has already cleared ``TIMEOUT_RATE_CEILING`` —
#: i.e. once the run is known NOT to be stall-shaped. Issue #21's premise
#: check found the D4 stall detector above is tuned to one collapse shape
#: (near-zero success WITH the clock running out, >= 28/30 timeout on
#: record) and blind to another: the defend family's catastrophic runs are
#: DEFEAT-shaped instead — the team is wiped, not stalled — and never trip
#: it. Four documented corpora establish the shape and its ceiling:
#: `fireteam_defend_v6b` 1/30 success at 2/30 timeout (27/30 defeat),
#: `fireteam_defend_v7` 12/30 at 7/30 timeout, `squad_screen_v7` 6/30 at
#: 0/30 timeout, and `fireteam_defend_v6` 14/30 (0.467) at 4/30 timeout —
#: all defeat-shaped, none within an order of magnitude of the stall
#: signature. The healthy floor is `fireteam_defend_v11` at 0.74. 0.5 sits
#: in the empty band between them (0.467, 0.74): it is also the point past
#: which a cohort loses more episodes outright than it wins, so it separates
#: the record without hair-splitting, the same way TIMEOUT_RATE_CEILING does
#: on its own axis. Gating it only when timeout_rate already passes keeps
#: the two axes mutually exclusive in a report: a run reads as STALLED
#: (timeout_rate fails) or WIPED (success_rate fails), never both, which is
#: the point — the two shapes want opposite fixes.
SUCCESS_RATE_FLOOR: float = 0.5

#: Command-report gate (v1.20, owner's decision 2026-08-12): a floor on
#: ``closed_on_root_report_rate`` — the share of won episodes whose ENDEX came
#: from the commander's own truthful MISSION COMPLETE.
#:
#: It exists because ``successes_announced_rate`` does NOT measure this and was
#: read as though it did. That metric counts the ENDEX, not who claimed it, so
#: it reads a flat **1.00 for a commander that never transmits at all** — and
#: it did, three times in one day: ``squad_v11``, ``squad_v14b_nobonus`` and
#: ``squad_v14c_nobonus`` each filed ZERO root claims across 100 episodes
#: (0 in ~11k admissible agent-steps) and passed every gate on the board while
#: scoring 0.96-0.98. A chain-of-command project cannot call that a win: the
#: operation ended because the world said so, not because anyone reported it.
#:
#: 0.5 sits in a wide empty band. Every non-mute corpus on record is at or
#: above 0.784 (``squad_v10b``, the weakest); every mute one is at 0.000-0.01.
#: Nothing has ever been measured between 0.01 and 0.78, so the floor is not
#: separating close cases — it is refusing a regime. Deliberately NOT set near
#: the fleet's realised 0.81-1.00, because this must catch silence, not police
#: the difference between a good reporter and a very good one.
#:
#: Unlike SUCCESS_RATE_FLOOR this is not conditioned on ``timeout_rate``: it is
#: a third, independent axis rather than a collapse shape. A run can win
#: everything and still be mute — that is exactly the case it is here to fail —
#: so it must be able to fail alone, and on a run that also stalled it simply
#: reports a second true thing.
#:
#: ``None`` (no ENDEX was ever sent, so there is no denominator) stays ``None``
#: through ``_gate``: unmeasured is not passed.
ROOT_REPORT_CLOSE_FLOOR: float = 0.5

#: SITREP freshness interval for a trace recorded before the scenario's own
#: was written into it (refs issue #35). Read off ``RewardConfig`` rather than
#: restated, because it is the live price: a report at least this many steps
#: after the sender's last one is paid ``sitrep_fresh``, a closer one
#: ``sitrep_spam``. Scenarios that switch the reporting doctrine on override
#: it per-scenario (``ScenarioSpec.sitrep_cadence``) and the recorder writes
#: that value into the trace; the whole defend family runs on this default.
DEFAULT_SITREP_INTERVAL: int = RewardConfig().sitrep_interval

#: Freshness-clock origin for a trace that did not record one: far enough back
#: that the episode's first SITREP is on cadence, mirroring
#: ``Soldier.last_sitrep_step``'s own default. Under the reporting doctrine
#: the environment instead starts the clock at 0 — the first report is *owed*
#: within one interval — and a trace recorded under it carries that 0.
UNSET_SITREP_CLOCK: int = -10_000

#: Message kinds that carry command state — the learned acts of command an
#: agent pays airtime for. The OPORD is excluded: it is HQ's, once, and no
#: policy decided it.
COMMAND_KINDS: frozenset[str] = frozenset({"order", "execute"})

#: Message kinds spoken by VOICE (A5-4): never net-arbitrated — shouting to the
#: soldier beside you does not contend for the net — but charged airtime like
#: every other learned transmission since #18. Counted separately because voice
#: and radio answer different questions, NOT because voice is free: it was, and
#: `squad_screen_v4/ckpt_latest` poured 93% of its traffic into it to run the
#: clock out, which is what #18 closed. Do not reason from "the free channel"
#: — `tests/test_voice_is_charged.py` fails if this comment goes stale again.
VOICE_KINDS: frozenset[str] = frozenset({"sync_propose", "sync_go"})


def _dist(a: list | tuple, b: list | tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------------- #
# trace recording
# ---------------------------------------------------------------------- #


class TraceRecorder:
    """Accumulates the behavioral trace of one episode.

    Wire into ``run_episode`` via its ``recorder`` argument: ``on_reset``
    right after ``env.reset``, ``before_step`` right before each
    ``env.step`` (snapshots mission-anchor distances exactly like the
    environment does), ``after_step`` right after. All reads are
    deterministic and consume no RNG, so recording never perturbs a seeded
    episode (covered by test).
    """

    def __init__(self) -> None:
        self.trace: dict[str, Any] = {}
        self._prev_dist: dict[str, float] = {}

    # -- hooks ---------------------------------------------------------- #

    def on_reset(self, env: CohortEnv) -> None:
        """Capture episode statics + the state after ``reset()`` (t=0)."""
        cfg = env.spec_cfg
        obj = env.world.objective_by_name(cfg.root_objective) if cfg.root_objective else None
        human = next((s.callsign for s in env.roster.soldiers if s.human), None)
        self.trace = {
            "scenario": cfg.name,
            "outcome": None,
            "length": 0,
            "root_mission": cfg.root_mission.name,
            # the step ceiling this episode was played under: without it a
            # recorded length of 375 is just a number, and "pinned at
            # max_steps" — the stall signature of issue #18 — is not a
            # statement the trace can support
            "max_steps": int(cfg.max_steps),
            "root_objective": list(obj.pos) if obj is not None else None,
            "ring_radius": RING_RADIUS,
            # issue #11: the scenario's own weapon range defines "under
            # threat", so the fight-disposition metrics travel with the
            # combat model instead of a hard-coded constant.
            "threat_radius": float(env.combat.weapon_range),
            # cohesion (owner's axis, 2026-08-18, measured NOT enforced): the
            # radius that defines a "close" teammate is the combat model's own
            # support umbrella — the distance at which doctrine already says a
            # supporter counts — recorded so the trace states its definition.
            "support_umbrella": float(env.combat.support_umbrella),
            "contact_refresh_age": env.rewards_cfg.contact_refresh_age,
            # issue #35: the interval that prices a SITREP fresh rather than
            # spam, and where the freshness clock starts — the scenario's own
            # cadence when the reporting doctrine is on, the reward config's
            # otherwise. Recorded for the same reason the refresh age is:
            # without it, "the root reported off cadence" is not a statement
            # the trace can support.
            "sitrep_interval": int(cfg.sitrep_cadence or env.rewards_cfg.sitrep_interval),
            "sitrep_clock_start": min(
                (s.last_sitrep_step for s in env.roster.soldiers), default=UNSET_SITREP_CLOCK
            ),
            "knowledge_ttl": _knowledge_ttl(),
            "human": human,
            "reported": {},
            "steps": [self._step_record(env, initial=True)],
        }

    def before_step(self, env: CohortEnv) -> None:
        """Snapshot anchor distances (mirrors the env's own step snapshot)."""
        self._prev_dist = {
            s.callsign: env._anchor_distance(s) for s in env.roster.living if s.mission is not None
        }

    def after_step(self, env: CohortEnv) -> None:
        """Record the post-step state, this tick's messages, and compliance."""
        self.trace["steps"].append(self._step_record(env, initial=False))
        self.trace["length"] = len(self.trace["steps"]) - 1
        self.trace["outcome"] = env.outcome
        # v1.13: the step the root closed the C2 loop in time to earn
        # root_done_bonus — a truthful MISSION COMPLETE where one is
        # admissible, or the root's SITREP on a continuous posture, where it
        # is not. `None` means COMMAND closed the operation unprompted.
        self.trace["root_close_step"] = env._root_close_step
        if not env.agents:  # episode over: freeze the per-soldier report sets
            self.trace["reported"] = {
                s.callsign: sorted(s.reported_enemy_ids) for s in env.roster.soldiers
            }

    # -- recording ------------------------------------------------------ #

    def _step_record(self, env: CohortEnv, *, initial: bool) -> dict:
        soldiers = []
        # succession moves the root mid-episode, so who holds the OPORD is a
        # function of the step, not of the roster at reset
        root = env.roster.root()
        # cohesion (measured, not enforced): which living soldiers have a
        # living teammate close (within the support umbrella) and which are in
        # some living teammate's line of sight. Computed here rather than in
        # the reducer because LOS needs the terrain grid, which the trace does
        # not carry. Both relations are symmetric (euclidean distance;
        # Bresenham LOS blocks on walls both ways), so one pass over pairs
        # covers both directions. Pure grid/position lookups — no RNG.
        living = [s for s in env.roster.soldiers if s.alive]
        umbrella = float(env.combat.support_umbrella)
        has_close_mate: set[str] = set()
        seen_by_mate: set[str] = set()
        for i, a in enumerate(living):
            for b in living[i + 1:]:
                if _dist(a.pos, b.pos) <= umbrella:
                    has_close_mate.update((a.callsign, b.callsign))
                if env.world.line_of_sight(a.pos, b.pos):
                    seen_by_mate.update((a.callsign, b.callsign))
        for s in env.roster.soldiers:
            comp = None
            # A5-2: a pending order stages its recipient, and the environment
            # scores it as HOLD at the staging spot — not as the ordered task.
            # Scoring it as the ordered task here made a staged ADVANCE read
            # 0.5 ("in position") the tick it landed, which is the opposite of
            # what the recipient is doing: nothing, on purpose.
            pending = s.mission is not None and is_pending(s.mission, env._step_count)
            if not initial and s.alive and s.mission is not None:
                ctx = env._compliance_ctx(s, self._prev_dist.get(s.callsign), env._make_view(s))
                comp = compliance(MissionType.HOLD if pending else s.mission.type, ctx)
            leader = env.roster.leader_of(s)
            soldiers.append(
                {
                    "cs": s.callsign,
                    "alive": s.alive,
                    "pos": list(s.pos),
                    # effective, not nominal: an acting SL must count as an SL
                    # here, or a promoted commander's orders land in the rank
                    # bucket it no longer holds (`orders_by_rank`).
                    "rank": s.effective_rank.name,
                    "mission": s.mission.type.name if s.mission is not None else None,
                    "since": s.mission.step_assigned if s.mission is not None else None,
                    # A5-2: is the standing order staged (AT MY COMMAND not yet
                    # EXECUTEd, AT T PLUS n not yet due)? A staged recipient is
                    # obeying by NOT executing, so obedience is not measurable
                    # until release — see _obedience and _staging.
                    "pending": pending,
                    "auth": s.effective_authority,
                    "subs": [x.callsign for x in s.living_subordinates(env.roster)],
                    "leader": leader.callsign if leader is not None else None,
                    "comp": comp,
                    # issue #11: terrain posture, the strongest correlate of
                    # defend success found so far. Read-only grid lookup.
                    "cover": bool(env.world.cover_at(s.pos)),
                    # cohesion booleans: None when dead, so the reducer can
                    # tell "dead this step" and "pre-cohesion trace" (both
                    # skipped) apart from a measured False
                    "teammate_close": (s.callsign in has_close_mate) if s.alive else None,
                    "teammate_sees": (s.callsign in seen_by_mate) if s.alive else None,
                    "fired": bool(s.fired_this_step) if s.alive else False,
                    "sees": [e.id for e in env._visible_enemies(s)] if s.alive else [],
                    # A5-3: the element stance set ON this soldier (leaders only)
                    "formation": s.formation.name if s.formation is not None else None,
                    # was MISSION COMPLETE admissible to this agent at this
                    # state? Read off the mask's own predicate, so a run's
                    # DONE silence is attributable: no admissible step means
                    # the channel was shut, admissible steps with no claim
                    # means the policy declined it (refs #13).
                    "done_ok": is_done_admissible(
                        s,
                        env.roster,
                        root_mission=env.spec_cfg.root_mission,
                        root_objective_id=env._root_objective_id(),
                        step=env._step_count,
                        done_cooldown=env.spec_cfg.done_cooldown,
                    ),
                    # which orders the mask offered this agent at this state,
                    # by ordered task — the opportunity denominator of the
                    # ordered-task mix (refs #16). Recomputed from the same
                    # function that built the observation's own mask a moment
                    # ago, so it is the vocabulary the policy actually chose
                    # from, not a re-derivation of it.
                    "order_opts": order_options(env._mask_for(s)),
                    "root": s is root,
                }
            )
        messages = env.transcript.messages if initial else env.last_messages
        # audit metadata parallel to the messages (medium + actual semantic
        # hearers). At reset the transcript holds exactly the OPORD lines
        # ``_say`` just recorded meta for, so the zip holds there too; a
        # length mismatch (defensive) degrades to unlabeled messages.
        metas = list(getattr(env, "last_message_meta", []))
        if len(metas) != len(messages):
            metas = [None] * len(messages)
        return {
            "t": env._step_count,
            "soldiers": soldiers,
            "enemies": [
                {"id": e.id, "alive": e.alive, "pos": list(e.pos)} for e in env.enemies
            ],
            "messages": [
                _message_record(env, m, meta) for m, meta in zip(messages, metas, strict=False)
            ],
            # tactical acoustics (§3.6): this step's sound events with source
            # truth, semantic hearers and non-semantic detectors — the trace
            # is ground-truth material, like enemy positions
            "sounds": [] if initial else [
                ev.to_record() for ev in getattr(env, "last_sound_events", [])
            ],
            # B5 order economics: this tick's re-task events, straight from
            # the environment's own adjudication (issuer rank, priced or
            # excepted and why, anchor rotation or same-anchor type change)
            "retasks": [] if initial else env.retask_events_last_step,
            "order_pay": [] if initial else env.order_pay_events_last_step,
        }


def _knowledge_ttl() -> int:
    from cohort.env.cohort_env import KNOWLEDGE_TTL  # local: avoid a cycle at import time

    return KNOWLEDGE_TTL


def _message_record(env: CohortEnv, m, meta: dict | None = None) -> dict:
    def cs_of(agent_id: int | None) -> str:
        if agent_id is None:
            return "ALL"
        if agent_id == -1:
            return "HQ"
        soldier = env.roster.by_id.get(agent_id)
        return soldier.callsign if soldier is not None else f"#{agent_id}"

    mission = None
    if m.kind.value in ("order", "opord"):
        try:
            parsed_mission = lang.parse_order(m.text).mission
            # stance orders (A5-3, FORMATION X) carry no mission payload
            mission = parsed_mission.name if parsed_mission is not None else None
        except lang.OrderParseError:  # pragma: no cover - formats round-trip by invariant
            mission = None
    return {
        "kind": m.kind.value,
        "from": cs_of(m.sender_id),
        "to": cs_of(m.recipient_id),
        "mission": mission,
        "text": m.text,
        # §8: transcripts label briefing / radio / voice / signal / gesture /
        # external umpire events distinctly, and the trace separately records
        # who could actually hear the semantics
        "medium": meta.get("medium") if meta else None,
        "heard_by": meta.get("heard_by") if meta else None,
    }


# ---------------------------------------------------------------------- #
# per-episode metrics (pure functions over a trace)
# ---------------------------------------------------------------------- #


def _soldier_at(step: dict, cs: str) -> dict | None:
    for rec in step["soldiers"]:
        if rec["cs"] == cs:
            return rec
    return None


def _obedience(trace: dict) -> tuple[list[int], int, dict[str, dict]]:
    """(latencies, censored, by_task): order applied → first step with compliance > 0.

    An *order event* is a step where an agent's standing mission carries
    ``step_assigned == t`` (OPORD at t=0 included). The event resolves at the
    first step, from the assignment step on, where the agent's per-step
    compliance score for that mission is positive; it is censored (counted,
    no latency) if the mission is replaced or cleared, the agent dies, or
    the episode ends first.

    ``by_task`` splits both by the ordered mission, because the pooled mean
    conflates "the cohort became disobedient" with "the cohort was ordered to
    do slower things". The defend line ran 1.19 / 1.26 steps at v6 / v8 and
    11.24 / 13.06 at v9 / v10, over the same stretch in which the ADVANCE share
    of orders went 0.69 → 0.99 — and an ADVANCE to a distant control measure
    cannot resolve as fast as a DEFEND in place. Which of those two it is
    cannot be read off the pooled number, so it is no longer only pooled.

    **Staged orders are not order events** (refs issue #15). A pending A5-2
    order (AT MY COMMAND before EXECUTE, AT T PLUS n before its tick) is one
    the recipient is obeying *by not executing*, and the environment scores it
    as HOLD at the staging spot — where the recipient already stands, so its
    compliance is positive from the tick the order lands. Booking that as an
    order event made every staged order resolve at latency **0**: an identical
    un-staged ADVANCE whose recipient never moved was censored, while the
    staged one read "obeyed instantly" (measured, fireteam seed 3). Because
    release restamps ``step_assigned``, the order books its real event at the
    release tick anyway — so the staged tick was a second, free, zero-latency
    copy of the same order, and the metric fell toward 0 in proportion to how
    much a policy staged. That is the *opposite* sign to cycle 8's hypothesis
    that staging inflates latency, and it is why the outside tap found
    incidence and duration both running backwards: v8 staged 0.878 of its
    ADVANCE orders and measured 1.01 steps, v10 staged 0.369 and measured
    16.21. Staged ticks are therefore skipped here and counted by
    :func:`_staging` instead, which also names the orders that are never
    released at all (61 of v8's 130) — those now leave no obedience event,
    which is correct: an order that never became effective was never binding.
    """
    steps = trace["steps"]
    latencies: list[int] = []
    censored = 0
    by_task: dict[str, dict] = {}

    def book(task: str, latency: int | None) -> None:
        slot = by_task.setdefault(task, {"latencies": [], "censored": 0})
        if latency is None:
            slot["censored"] += 1
        else:
            slot["latencies"].append(latency)
    for i, step in enumerate(steps):
        t0 = step["t"]
        for rec in step["soldiers"]:
            if not (rec["alive"] and rec["mission"] is not None and rec["since"] == t0):
                continue
            if rec.get("pending"):
                continue  # staged: not binding yet, and release restamps
            resolved = False
            for later in steps[max(i, 1):]:
                later_rec = _soldier_at(later, rec["cs"])
                if (
                    later_rec is None
                    or not later_rec["alive"]
                    or later_rec["mission"] != rec["mission"]
                    or later_rec["since"] != t0
                ):
                    break  # re-tasked, cleared, or dead before complying
                if later_rec.get("pending"):
                    continue  # compliance while staged is HOLD, not the task
                if later_rec["comp"] is not None and later_rec["comp"] > 0.0:
                    latencies.append(later["t"] - t0)
                    book(rec["mission"], later["t"] - t0)
                    resolved = True
                    break
            if not resolved:
                censored += 1
                book(rec["mission"], None)
    return latencies, censored, by_task


def _staging(trace: dict) -> dict[str, Any]:
    """A5-2 staging: how many orders staged, how many released, how long held.

    The counterpart to :func:`_obedience` skipping staged ticks (refs issue
    #15). Obedience cannot see a staged order — the recipient is complying by
    standing still — so without this the whole AT MY COMMAND / AT T PLUS n
    channel would be invisible in the behavior suite, and the interesting
    failure it can hide would be invisible with it: an order that is *issued
    and then abandoned*, staged until the episode ends because its issuer
    never sends EXECUTE. An outside tap found 61 of 130 of one checkpoint's
    staged orders in that state.

    A *staging span* runs from the tick a pending order lands to the tick it
    becomes effective — release restamps ``step_assigned`` to that tick, so a
    span is released exactly when the same soldier's next record carries the
    same mission, not pending, with ``since == t``. Anything else closes the
    span **abandoned**: re-tasked while staged, killed while staged, or still
    staged at the end of the episode. (A staged order superseded at its
    release tick by a fresh immediate order of the same task is
    indistinguishable from a release in the trace; it is a coincidence of one
    tick and one task, and it can only under-count abandonment.)
    """
    spans: dict[str, tuple[str, int]] = {}  # callsign -> (mission, staged-at)
    staged = 0
    released = 0
    gaps: list[int] = []
    for step in trace["steps"]:
        t = step["t"]
        for rec in step["soldiers"]:
            cs = rec["cs"]
            if rec["alive"] and rec["mission"] is not None and rec.get("pending"):
                key = (rec["mission"], rec["since"])
                if spans.get(cs) != key:  # a different staged order: the old one lapsed
                    spans[cs] = key
                    staged += 1
                continue
            span = spans.pop(cs, None)
            if span is None:
                continue
            mission, since = span
            if rec["alive"] and rec["mission"] == mission and rec["since"] == t:
                released += 1
                gaps.append(t - since)
    return {"orders_staged": staged, "staged_released": released, "staging_gaps": gaps}


def _reports(trace: dict) -> dict[str, int]:
    """CONTACT precision/recall counts against ground-truth visibility.

    Precision counts *informative* reports: a CONTACT whose content (the
    sender's ground-truth visible enemies at tick start) includes an enemy
    absent from the replayed team picture (new intel) or one whose picture
    entry has aged >= ``contact_refresh_age`` (a legitimate refresh). The
    picture replay mirrors the environment: entries expire after
    ``knowledge_ttl`` steps and dead enemies drop off. Recall counts enemy
    ids ever reported over enemy ids ever seen (union with reported, so the
    ratio is always <= 1).
    """
    steps = trace["steps"]
    refresh_age = trace["contact_refresh_age"]
    ttl = trace["knowledge_ttl"]
    picture: dict[int, int] = {}  # enemy id -> step of last report
    contacts = 0
    informative = 0
    seen: set[int] = set()
    for i, step in enumerate(steps):
        t = step["t"]
        for rec in step["soldiers"]:
            seen.update(rec["sees"])
        for msg in step["messages"]:
            if msg["kind"] != "contact":
                continue
            sender = _soldier_at(steps[i - 1], msg["from"]) if i > 0 else None
            content = list(sender["sees"]) if sender and sender["sees"] else []
            if not content:  # fallback: post-step visibility
                here = _soldier_at(step, msg["from"])
                content = list(here["sees"]) if here else []
            if not content:
                continue  # degenerate: content unknown, excluded entirely
            contacts += 1
            new = any(eid not in picture for eid in content)
            refresh = not new and any(t - picture[eid] >= refresh_age for eid in content)
            if new or refresh:
                informative += 1
            for eid in content:
                picture[eid] = t
        living = {e["id"] for e in step["enemies"] if e["alive"]}
        picture = {
            eid: last for eid, last in picture.items() if eid in living and t - last <= ttl
        }
    reported = {eid for ids in trace.get("reported", {}).values() for eid in ids}
    return {
        "contacts": contacts,
        "contacts_informative": informative,
        "enemies_seen": len(seen | reported),
        "enemies_reported": len(reported),
    }


#: Doctrine tiers an agent-issued order can fall into, in order of quality.
#: ``underivable`` is the issuer-had-no-mission case, which the order mask
#: makes unreachable in play but which a replayed / injected trace can show.
DOCTRINE_TIERS: tuple[str, ...] = ("preferred", "allowed", "violating", "underivable")


def _doctrine(trace: dict) -> dict[str, Any]:
    """Agent-issued orders split by doctrine tier, overall and by ordered task.

    An issued order's quality is judged against the issuer's mission at the
    moment of transmission: missions are read from the previous step's
    records and updated in within-step message order as orders are applied
    (an order counts as applied when the recipient's end-of-step mission
    matches it with ``step_assigned == t``). HQ traffic (OPORD, injected
    orders) is not an agent decision and is excluded from the rate.

    The tier split exists because the preference rate alone cannot tell
    *adopting a legal alternative* from *disregarding doctrine* — both merely
    fail to be ``allowed[0]`` (refs issue #14). Since A5 added ADVANCE to the
    DEFEND / SEIZE / RECON derivation tables, a policy that orders ADVANCE
    almost exclusively scores ~0 preference while committing zero doctrine
    violations, so the rate reads as a collapse when it is catalog adoption.
    ``orders_by_task`` makes that adoption visible: preference conditioned on
    the ordered task, against the task's share of all orders.

    The task mix in turn needs its own denominator (refs issue #16). Order
    shares are *availability-confounded*: the order mask offers the tasks in
    wildly unequal numbers, so a task can be rare because the policy declines
    it or because it was barely on the menu. ``order_availability`` is the
    matched control — for every order the policy actually issued, the share of
    the issuer's admissible order vocabulary that belonged to each task at
    that exact state (from ``order_opts``, read off the mask at ``steps[i-1]``,
    which is the observation the issuer acted on). Summed here and divided by
    ``orders_matched`` at aggregation, it is precisely the expected task mix
    of a masked-random policy making the same set of order decisions — the
    floor a preference has to clear to be a preference at all.
    """
    steps = trace["steps"]
    issued = 0
    tiers = dict.fromkeys(DOCTRINE_TIERS, 0)
    by_task: dict[str, dict[str, int]] = {}
    by_rank: dict[str, dict[str, int]] = {}
    availability: dict[str, float] = {}
    matched = 0
    for i in range(1, len(steps)):
        step = steps[i]
        t = step["t"]
        prev = steps[i - 1]["soldiers"]
        cur: dict[str, str | None] = {rec["cs"]: rec["mission"] for rec in prev}
        # the rank the issuer held when it acted, i.e. on the observation it
        # acted on — same state `cur` and `offered` are read from
        rank_of: dict[str, str] = {rec["cs"]: rec["rank"] for rec in prev if "rank" in rec}
        offered: dict[str, dict[str, int]] = {
            rec["cs"]: rec.get("order_opts") or {} for rec in prev
        }
        for msg in step["messages"]:
            if msg["kind"] not in ("order", "opord") or msg["mission"] is None:
                continue
            if msg["kind"] == "order" and msg["from"] != "HQ":
                issued += 1
                opts = offered.get(msg["from"]) or {}
                # An order the issuer's own mask did not offer was injected or
                # replayed, not selected: it has no opportunity set to compare
                # against, so it stays out of the matched control. The gap
                # between orders_matched and orders_issued makes that visible.
                if msg["mission"] in opts:
                    matched += 1
                    total_opts = sum(opts.values())
                    for task, n in opts.items():
                        availability[task] = availability.get(task, 0.0) + n / total_opts
                own = cur.get(msg["from"])
                allowed = allowed_derivations(MissionType[own]) if own is not None else ()
                task = MissionType[msg["mission"]]
                if not allowed:
                    tier = "underivable"
                elif task is allowed[0]:
                    tier = "preferred"
                elif task in allowed:
                    tier = "allowed"
                else:
                    tier = "violating"
                tiers[tier] += 1
                bucket = by_task.setdefault(task.name, dict.fromkeys(DOCTRINE_TIERS, 0))
                bucket[tier] += 1
                issuer_rank = rank_of.get(msg["from"])
                if issuer_rank is not None:
                    rank_bucket = by_rank.setdefault(issuer_rank, dict.fromkeys(DOCTRINE_TIERS, 0))
                    rank_bucket[tier] += 1
            recipient = _soldier_at(step, msg["to"])
            applied = (
                recipient is not None
                and recipient["mission"] == msg["mission"]
                and recipient["since"] == t
            )
            if applied:
                cur[msg["to"]] = msg["mission"]
    return {
        "orders_issued": issued,
        "orders_preferred": tiers["preferred"],
        "orders_allowed": tiers["allowed"],
        "orders_violating": tiers["violating"],
        "orders_underivable": tiers["underivable"],
        "orders_by_task": by_task,
        # refs #52: `orders_by_task` is team-wide, so it cannot answer "how many
        # orders did the ROOT itself issue" — the question the mute-commander
        # diagnosis stalled on, since only re-tasks were rank-resolved.
        #
        # refs #53: this is a RAW count, and on a comparison that also changes
        # episode length it is not comparable across arms — a root that
        # survives (and times out instead of ending early) racks up more
        # orders at the SAME rate as one that dies early. Pair with
        # `rank_alive_steps` (below) before reading anything into a gap here.
        "orders_by_rank": by_rank,
        "order_availability": availability,
        "orders_matched": matched,
    }


def _rank_alive_steps(trace: dict) -> dict[str, dict[str, int]]:
    """(soldier, step) pairs alive under each effective rank tier (refs #53).

    The opportunity denominator for every by-rank order metric —
    `orders_by_rank` and `order_pay_by_rank` — both raw per-rank sums that a
    longer-lived issuer inflates without commanding any harder. Verified on
    the mute-vs-reporting root comparison this was built for: the mute root
    survives where its reporting counterpart dies, so its episodes time out
    instead of ending early and it racks up ~3.4x the root-order volume at
    the SAME rate (orders per step alive), which the raw count cannot show
    and the rate does. Episode count alone is not enough either — episode
    LENGTH is itself part of the treatment on exactly this comparison.

    Counted the same way :func:`_done_opportunity` counts admissible steps:
    over every recorded state but the last, since the terminal state is
    followed by no action and is not an opportunity anybody had. This mirrors
    the window `_doctrine` reads issuer rank from (`prev = steps[i-1]`, for
    `i` in ``range(1, len(steps))``), so an order's rank bucket and its
    alive-step denominator are always drawn from the same states.
    """
    alive: dict[str, int] = {}
    for step in trace["steps"][:-1]:
        for rec in step["soldiers"]:
            if rec["alive"] and rec.get("rank") is not None:
                alive[rec["rank"]] = alive.get(rec["rank"], 0) + 1
    return {"rank_alive_steps": alive}


def _retasks(trace: dict) -> dict[str, Any]:
    """Re-task economics counts (B5), read from the environment's own
    adjudication recorded per step: total re-tasks (orders replacing a
    standing mission — fresh taskings and identical reissues are not
    re-tasks), how many were priced vs. excepted (the tactical-picture
    carve-out: contact / element casualty / issuer intent change), how many
    rotated the anchor (vs. same-anchor mission-type changes), and the
    per-issuer-rank split of priced vs. excepted."""
    total = priced = excepted = rotations = 0
    by_rank: dict[str, dict[str, int]] = {}
    for step in trace["steps"]:
        for ev in step.get("retasks", []):
            total += 1
            bucket = by_rank.setdefault(ev["rank"], {"priced": 0, "excepted": 0})
            if ev["excepted"]:
                excepted += 1
                bucket["excepted"] += 1
            else:
                priced += 1
                bucket["priced"] += 1
            if not ev["same_anchor"]:
                rotations += 1
    return {
        "retasks": total,
        "retasks_priced": priced,
        "retasks_excepted": excepted,
        "retask_rotations": rotations,
        "retasks_by_rank": by_rank,
    }


def _order_pay(trace: dict) -> dict[str, Any]:
    """Command income, per issuer rank (refs #52).

    Order volume alone cannot say whether commanding is profitable: only fresh
    taskings are paid (``order_preferred`` / ``order_allowed``, and zero when
    the ordered task derives from nothing the issuer holds), while identical
    reissues are charged ``order_churn``. A commander that orders constantly is
    either farming the channel or paying to sit in it, and those are opposite
    diagnoses — the mute-commander investigation could not tell them apart,
    which is why this exists.

    ``pay`` sums the order-channel ledger entries only. It is NOT the issuer's
    net command income: the re-task price is charged separately (see
    ``retasks_by_rank``), so a rank can show positive ``pay`` here and still be
    losing on command overall.

    refs #53: ``fresh``/``churn``/``retask`` and ``pay`` are raw per-rank
    totals, same as ``orders_by_rank``, and the same artifact applies — a
    rank held by a longer-lived issuer accumulates more of all four without
    commanding any harder or farming the channel any more. Divide by
    ``rank_alive_steps`` before comparing this across runs whose episodes
    run different lengths.
    """
    by_rank: dict[str, dict[str, Any]] = {}
    for step in trace["steps"]:
        for ev in step.get("order_pay", []):
            bucket = by_rank.setdefault(
                ev["rank"], {"fresh": 0, "churn": 0, "retask": 0, "pay": 0.0}
            )
            bucket[ev["outcome"]] += 1
            bucket["pay"] += ev["pay"]
    return {"order_pay_by_rank": by_rank}


def _vocabulary(trace: dict) -> dict[str, Any]:
    """A5 vocabulary usage: what share of the traffic uses the new forms.

    Counts ADVANCE orders (control-measure targets), timing-qualified orders
    (AT T PLUS / AT MY COMMAND) with their EXECUTE releases, FORMATION stance
    orders, and trinôme sync traffic (proposals / GO bounds); plus the share
    of living-agent steps governed by a formation stance (own, or the direct
    leader's — the geometry actually being shaped).
    """
    advance = timed = formation = executes = proposals = bounds = 0
    stance_steps = 0
    agent_steps = 0
    for step in trace["steps"]:
        stanced = {
            rec["cs"] for rec in step["soldiers"] if rec.get("formation") is not None
        }
        for rec in step["soldiers"]:
            if not rec["alive"]:
                continue
            agent_steps += 1
            if rec["cs"] in stanced or rec.get("leader") in stanced:
                stance_steps += 1
        for msg in step["messages"]:
            kind = msg["kind"]
            if kind == "execute":
                executes += 1
            elif kind == "sync_propose":
                proposals += 1
            elif kind == "sync_go":
                bounds += 1
            elif kind in ("order", "opord"):
                try:
                    parsed = lang.parse_order(msg["text"])
                except lang.OrderParseError:  # pragma: no cover
                    continue
                if parsed.formation is not None:
                    formation += 1
                if parsed.mission is MissionType.ADVANCE:
                    advance += 1
                if parsed.delay is not None or parsed.at_my_command:
                    timed += 1
    return {
        "advance_orders": advance,
        "timed_orders": timed,
        "execute_signals": executes,
        "formation_orders": formation,
        "sync_proposals": proposals,
        "sync_bounds": bounds,
        "stance_steps": stance_steps,
        "stance_agent_steps": agent_steps,
    }


def _false_complete(trace: dict) -> dict[str, int]:
    """MISSION COMPLETE claims: how many, how many rejected, and how concentrated.

    ``false_complete_rate`` is a ratio, and a ratio cannot see the difference
    between a root that files one claim in each episode where the end state
    holds and a root that files eight an episode and is right an eighth of the
    time (refs #23, filed from the assurance layer against the fleet: 13 claims
    accepted 13 times in 13 episodes, and 128 claims accepted 27 times in 30,
    are the same shape of object and opposite behaviours). The denominator that
    separates them is **episodes in which anything was claimed** — one claim per
    claiming episode is a policy filing a report, eight is a policy spamming the
    net, and the rejection rate can be identical either way.

    The root's own claims are counted separately because the C2 question is
    about the root's channel: the root is the agent whose COMPLETE closes the
    operation, and ``done_admissible_root`` has counted its opportunities since
    refs #13 with no numerator to divide into. The pooled rate is not a stand-in
    — a fireteam's riflemen can carry the pooled rate on their own.

    Acceptance is deliberately NOT added as a metric of its own. Every DONE is
    adjudicated on the step it is transmitted (``cohort_env._done`` answers with
    DONE_CONFIRM or DONE_REJECT and never with neither), so accepted equals
    reports minus rejected exactly, and realised acceptance is
    ``1 - false_complete_rate`` at each level. A second name for a number the
    suite already carries would be noise, not evidence.
    """
    dones = rejected = root_dones = root_rejected = 0
    for step in trace["steps"]:
        # read per step: after a succession the root is a different callsign,
        # and the claim that matters is the one made BY whoever held the root
        roots = {rec["cs"] for rec in step["soldiers"] if rec.get("root")}
        for msg in step["messages"]:
            if msg["kind"] == "done":
                dones += 1
                root_dones += msg.get("from") in roots
            elif msg["kind"] == "done_reject":
                rejected += 1
                # the superior sends the rejection; the claimant receives it
                root_rejected += msg.get("to") in roots
    return {
        "done_reports": dones,
        "done_rejected": rejected,
        "done_reports_root": root_dones,
        "done_rejected_root": root_rejected,
        "done_claim_episode": int(dones > 0),
        "done_claim_episode_root": int(root_dones > 0),
    }


def _endex_close(trace: dict) -> dict[str, int]:
    """Did the root close the C2 loop, or did COMMAND close it unprompted?

    v1.13 replaces ``false_complete_rate`` as the completion signal on a
    continuous-posture root. There, MISSION COMPLETE is inadmissible — a
    DEFEND runs until a new order arrives — so the rate is structurally 0 and
    tells you nothing. What is worth knowing is whether the root reported the
    situation once the end state held: COMMAND transmits ENDEX either way, but
    only a timely SITREP closes the window early and earns ``root_done_bonus``.

    Counted only where an ENDEX was actually sent, so success-rate drift does
    not move the denominator. Zero ENDEX means the scenario has a completable
    root (SEIZE and friends keep reporting COMPLETE) or the operation never
    succeeded — neither is a reporting-quality statement, and both stay out.

    v1.16 had a horizon defense sending BOTH — the root's MISSION COMPLETE and
    COMMAND's ENDEX — so on that family ``root_close_step`` was set by the claim
    rather than by a SITREP. v1.17 masks the claim shut again, so every defend
    root is back to closing with its SITREP. The question the rate asks is
    unchanged either way (did the root's report close the window, or did the
    window simply expire); the act the report consists of is whichever one the
    OPORD leaves open to it. Note what that means for reading the rate across
    the change: on ``fireteam_defend_v18``/final it went 0.00 -> 0.53 without
    the policy moving at all, because the SITREPs it was already transmitting
    started closing the window.

    ``close_announced`` is the other half, and the one v1.14 lost: did anything
    at all go out on the net saying this operation is over — COMMAND's ENDEX or
    the root's own confirmed claim. It is deliberately either/or, because on a
    SEIZE root the claim IS the announcement and there is no ENDEX to want.
    """
    endex = 0
    prompted = 0
    for step in trace["steps"]:
        for msg in step["messages"]:
            if msg["kind"] == "endex":
                endex += 1
                # the root's report closed the window early: the episode ends
                # on the report rather than on the grace window expiring
                prompted += bool(trace.get("root_close_step") is not None)
    return {
        "endex_sent": endex,
        "endex_on_root_report": prompted,
        "close_announced": int(endex > 0 or trace.get("root_close_step") is not None),
    }


def _root_sitreps(trace: dict) -> dict[str, int]:
    """The root's SITREP volume, and the cadence the close was bought at (#35).

    ``closed_on_root_report_rate`` above has ENDEXes-sent for a denominator,
    and it **saturates**. Once the root transmits a SITREP every third step it
    is near-certain to have reported at or after the success step, so the
    close is its by default — the rate reads 1.00 for a policy that learned
    *when* to report and for one that simply never stops reporting. Measured
    on the first two policies trained with the root's MISSION COMPLETE masked
    shut: the rate went 0.79 -> 1.00 and 0.50 -> 1.00 while root SITREPs per
    episode went 6.1 -> 30.3 and 8.8 -> 32.8, against a ``sitrep_interval`` of
    25. One behavioural change, two readings, and the rate alone cannot say
    which of the two it is.

    These are the counts that separate them:

    * ``root_sitreps`` — every SITREP transmitted by whoever held the root at
      that step. Read per step, like the root's DONE claims, because
      succession moves the root mid-episode. Closes divided by this is
      ``closes_per_root_sitrep``: high means the reports were timed, low means
      the close was bought with volume.
    * ``root_sitreps_off_cadence`` — those transmitted sooner than
      ``sitrep_interval`` after the sender's previous SITREP, i.e. exactly the
      ones the environment prices ``sitrep_spam`` instead of ``sitrep_fresh``.
      Freshness is recomputed with the environment's own rule
      (``CohortEnv._apply_action``) over EVERY SITREP a soldier sent, root or
      not, because the environment's clock is per soldier and not per role.
    * ``endex_on_cadence_report`` — was the report that *actually closed the
      window* one the cadence would have produced anyway? This is the
      numerator of ``closed_on_cadence_report_rate``, and it is the cell that
      answers "is the policy timing anything at all", because it excludes the
      reports bought as lottery tickets on the ``root_done_bonus`` (+3.0 when
      this cell was built; **1.0 since v1.20**, which is exactly the change the
      lottery-ticket reading was meant to survive — the cell asks whether the
      SITREP channel is timing anything, at whatever the bonus happens to be).

    An operation closed by a confirmed MISSION COMPLETE rather than by a
    SITREP counts in that rate's denominator and not in its numerator, which
    is correct: the question is whether the SITREP channel is timing anything,
    and a claim-route close did not use it. On the v1.17 defend family the
    claim route is masked shut, so every close there is a SITREP.
    """
    interval = int(trace.get("sitrep_interval") or DEFAULT_SITREP_INTERVAL)
    clock_start = trace.get("sitrep_clock_start")
    start = UNSET_SITREP_CLOCK if clock_start is None else int(clock_start)
    close_step = trace.get("root_close_step")
    last_sitrep: dict[str, int] = {}
    sitreps = off_cadence = endex = 0
    closed_on_cadence = False
    for step in trace["steps"]:
        t = step["t"]
        roots = {rec["cs"] for rec in step["soldiers"] if rec.get("root")}
        for msg in step["messages"]:
            if msg["kind"] == "endex":
                endex += 1
                continue
            if msg["kind"] != "sitrep":
                continue
            sender = msg.get("from")
            fresh = t - last_sitrep.get(sender, start) >= interval
            last_sitrep[sender] = t
            if sender not in roots:
                continue
            sitreps += 1
            off_cadence += not fresh
            if close_step is not None and t == close_step:
                closed_on_cadence = fresh
    return {
        "root_sitreps": sitreps,
        "root_sitreps_off_cadence": off_cadence,
        # gated on an ENDEX for the same reason `endex_on_root_report` is: the
        # denominator is operations COMMAND closed, so the numerator cannot
        # count a close on an operation COMMAND never announced
        "endex_on_cadence_report": endex * int(closed_on_cadence),
        "sitrep_interval": interval,
    }


def _done_opportunity(trace: dict) -> dict[str, int]:
    """Agent-steps at which MISSION COMPLETE was admissible (all / the root's).

    The denominator ``done_reports`` never had (refs #13). Zero DONE reports
    is two opposite findings wearing the same face: an admissible-step count
    of zero means the channel was shut and no price was ever consulted (the
    ``is_root_opord_claim`` mask bug, which sat unseen for a generation
    because the pipeline could not see this); a large count with no claims
    means the policy was *offered* the act and declined it, which is a
    statement about the price of a false claim, not about reachability.

    Counted over every recorded state but the last — the terminal state is
    followed by no action, so it is not an opportunity anybody had.
    """
    admissible = 0
    root_admissible = 0
    for step in trace["steps"][:-1]:
        for rec in step["soldiers"]:
            if not rec.get("done_ok"):
                continue
            admissible += 1
            root_admissible += bool(rec.get("root"))
    return {"done_admissible": admissible, "done_admissible_root": root_admissible}


def _succession(trace: dict) -> tuple[int, list[int], int]:
    """(leader-death events, recovery times, unrecovered events).

    A leader death is the death of an agent with living direct subordinates.
    The orphaned set is those subordinates minus the successor (the agent
    announcing "I AM ASSUMING COMMAND" for the fallen leader). The event
    recovers at the first step where every still-living orphan holds a
    mission assigned at or after the death step; dead orphans drop out of
    the requirement. Events not recovered by episode end are censored.
    """
    steps = trace["steps"]
    events = 0
    recovery: list[int] = []
    unrecovered = 0
    for i in range(1, len(steps)):
        step = steps[i]
        t_death = step["t"]
        for rec in step["soldiers"]:
            prev = _soldier_at(steps[i - 1], rec["cs"])
            if rec["alive"] or prev is None or not prev["alive"]:
                continue  # not a death this step
            orphans = list(prev["subs"])
            if not orphans:
                continue  # no command to devolve
            marker = f"{rec['cs']} IS DOWN. I AM ASSUMING COMMAND"
            successor = next(
                (
                    m["from"]
                    for m in step["messages"]
                    if m["kind"] == "taking_command" and marker in m["text"]
                ),
                None,
            )
            orphans = [cs for cs in orphans if cs != successor]
            events += 1
            recovered = None
            for later in steps[i:]:
                ok = True
                for cs in orphans:
                    o = _soldier_at(later, cs)
                    if o is None or not o["alive"]:
                        continue  # dead orphans drop out of the requirement
                    if o["mission"] is None or o["since"] is None or o["since"] < t_death:
                        ok = False
                        break
                if ok:
                    recovered = later["t"] - t_death
                    break
            if recovered is None:
                unrecovered += 1
            else:
                recovery.append(recovered)
    return events, recovery, unrecovered


def _coverage(trace: dict) -> tuple[int, int]:
    """(leader-step pairs with >=1 living subordinate, of which all tasked).

    Mirrors the environment's coverage scoring: a pair counts when a living
    leader (effective authority > 0) holds a mission and has at least one
    living direct subordinate; it is covered when every one of them is
    tasked that step.
    """
    pairs = 0
    covered = 0
    for step in trace["steps"][1:]:
        tasked = {rec["cs"]: rec["mission"] is not None for rec in step["soldiers"]}
        for rec in step["soldiers"]:
            if not (rec["alive"] and rec["auth"] > 0 and rec["mission"] is not None and rec["subs"]):
                continue
            pairs += 1
            if all(tasked.get(cs, False) for cs in rec["subs"]):
                covered += 1
    return pairs, covered


def _human_exposure(trace: dict) -> dict[str, Any]:
    """Exposure of the human root: enemy proximity, ring entries, death.

    Measured over the steps the human is alive. ``ring_entries`` counts
    outside→inside transitions of the objective observation ring
    (``ring_radius`` around the root objective; spawning inside counts as
    the first entry). All values are None when the scenario has no human
    or, for the distance means, when the relevant reference never exists
    (no living enemy while the human lives; no root objective).
    """
    human = trace.get("human")
    if human is None:
        return {
            "human_died": None,
            "human_mean_enemy_dist": None,
            "human_mean_objective_dist": None,
            "human_ring_entries": None,
        }
    obj = trace.get("root_objective")
    ring = trace.get("ring_radius", RING_RADIUS)
    enemy_dists: list[float] = []
    obj_dists: list[float] = []
    entries = 0
    inside_prev = False
    died = False
    for step in trace["steps"]:
        rec = _soldier_at(step, human)
        if rec is None:
            continue
        if not rec["alive"]:
            died = True
            break
        living = [e["pos"] for e in step["enemies"] if e["alive"]]
        if living:
            enemy_dists.append(min(_dist(rec["pos"], p) for p in living))
        if obj is not None:
            d = _dist(rec["pos"], obj)
            obj_dists.append(d)
            inside = d <= ring
            if inside and not inside_prev:
                entries += 1
            inside_prev = inside
    return {
        "human_died": died,
        "human_mean_enemy_dist": (sum(enemy_dists) / len(enemy_dists)) if enemy_dists else None,
        "human_mean_objective_dist": (sum(obj_dists) / len(obj_dists)) if obj_dists else None,
        "human_ring_entries": entries if obj is not None else None,
    }


def _fight_disposition(trace: dict) -> dict[str, Any]:
    """Where the cohort fights when the enemy is on it (refs issue #11).

    Scores the (living soldier, step) pairs that are **under threat** — a
    living enemy within ``threat_radius`` (the scenario's weapon range), i.e.
    pairs where the soldier's ground actually costs or saves it something.
    Two numbers come out of that population:

    * cover occupancy — share of those pairs spent on cover terrain;
    * distance from the root objective — mean over those pairs.

    Conditioning on threat is what makes the pair informative: averaged over
    a whole episode, an approach march and a prepared defense look alike.
    Both are None when the episode never came under threat; the distance is
    also None when the scenario has no root objective.
    """
    obj = trace.get("root_objective")
    radius = trace.get("threat_radius", THREAT_RADIUS)
    pairs = 0
    in_cover = 0
    dist_sum = 0.0
    dist_n = 0
    for step in trace["steps"]:
        living = [e["pos"] for e in step["enemies"] if e["alive"]]
        if not living:
            continue
        for rec in step["soldiers"]:
            if not rec["alive"]:
                continue
            if min(_dist(rec["pos"], p) for p in living) > radius:
                continue
            pairs += 1
            in_cover += bool(rec.get("cover"))
            if obj is not None:
                dist_sum += _dist(rec["pos"], obj)
                dist_n += 1
    return {
        "threat_pairs": pairs,
        "threat_cover_pairs": in_cover,
        "threat_objective_dist_sum": dist_sum,
        "threat_objective_dist_pairs": dist_n,
    }


def _cohesion(trace: dict) -> dict[str, Any]:
    """Does anyone leave the cohort? (owner's axis, 2026-08-18 — MEASURED,
    never enforced: no mask and no reward reads these numbers.)

    The axis, verbatim: "no agent should be allowed to leave the cohort or go
    out of line of sight of all its teammates; there should always remain at
    least one close teammate to each teammate". Two counts over living-agent-
    steps, both recorded per step by the ``TraceRecorder`` (LOS needs the
    terrain grid, which the trace does not carry):

    * no close teammate — no living teammate within the combat model's
      support umbrella (``trace["support_umbrella"]``, 8.0 by default), the
      radius at which doctrine already counts a supporter as present;
    * unseen by any teammate — no living teammate holds terrain line of
      sight to the agent (walls block; the same ``World.line_of_sight``
      vision spotting runs on).

    A sole survivor counts as isolated on both — that is the finding, not an
    artifact. Steps where the soldier is dead, and whole traces recorded
    before these keys existed, contribute nothing to either count or to the
    denominator, so old committed ``behavior.json`` files simply read None.
    """
    agent_steps = 0
    no_close = 0
    unseen = 0
    for step in trace["steps"]:
        for rec in step["soldiers"]:
            close = rec.get("teammate_close")
            seen = rec.get("teammate_sees")
            if close is None or seen is None:
                continue  # dead this step, or a pre-cohesion trace
            agent_steps += 1
            no_close += not close
            unseen += not seen
    return {
        "cohesion_agent_steps": agent_steps,
        "no_close_teammate_steps": no_close,
        "unseen_by_any_teammate_steps": unseen,
    }


def _traffic(trace: dict) -> dict[str, Any]:
    """Everything said this episode, split by channel (refs issue #18).

    The suite counted the channels it had a question about — orders, sync
    bounds, CONTACTs, DONEs — but never the total, so *composition* was not
    computable from ``behavior.json``: "the net went quiet" and "the net
    changed hands" produced the same reading. They are not the same finding.
    A stalled `squad_screen_v4/ckpt_latest` carries 1326 messages/episode
    against its own successful `ckpt_best`'s 537, of which 0.6% are command
    traffic against 15.5% — the net is louder and emptier at once.

    Three counts, one denominator:

    * ``messages`` — every message on the transcript, learned or automatic;
    * ``messages_command`` — orders and EXECUTE releases (``COMMAND_KINDS``);
    * ``messages_voice`` — SYNC PROPOSE / GO (``VOICE_KINDS``), which are never
      net-arbitrated — shouting to the soldier beside you does not contend for
      the net — but which **do** pay airtime like every other learned
      transmission, and have since #18 closed exactly that hole (it was found
      when ``squad_screen_v4/ckpt_latest`` poured 93% of its traffic into the
      free channel, 1173 messages an episode, to run the clock out).

      This bullet said "cost no airtime … the one transmission a policy with
      nothing to say can emit for free" until 2026-08-13, years-of-the-project
      after it stopped being true, and that stale sentence was read as current
      fact off this docstring and used to diagnose ``patrol_brique_v7``. Two
      consequences worth keeping in view when reading a voice count. **Voice
      airtime is charged to the ``report`` component** (``cohort_env``
      ``_sync_propose``: SYNC is speech between peers, not authority, and the
      ``flat`` ablation arm must show command reward of exactly 0.0), so a
      negative ``report`` component does NOT by itself mean reporting was
      unfunded — a SYNC-heavy policy pays into that bucket while earning no
      ``contact_new``. And a high voice count is a policy *spending* on
      synchronisation, not helping itself to something free.

    ``ran_out_the_clock`` is the episode-level half of the signature: the
    environment scores ``timeout`` exactly when the step ceiling is reached
    with neither success nor defeat, so it *is* "pinned at ``max_steps``".
    """
    messages = command = voice = 0
    for step in trace["steps"]:
        for msg in step["messages"]:
            messages += 1
            command += msg["kind"] in COMMAND_KINDS
            voice += msg["kind"] in VOICE_KINDS
    return {
        "messages": messages,
        "messages_command": command,
        "messages_voice": voice,
        "ran_out_the_clock": trace.get("outcome") == "timeout",
        "max_steps": trace.get("max_steps"),
    }


def episode_behavior(trace: dict) -> dict[str, Any]:
    """Reduce one episode trace to its behavioral event counts and lists."""
    latencies, censored, obedience_by_task = _obedience(trace)
    events, recovery, unrecovered = _succession(trace)
    pairs, covered = _coverage(trace)
    return {
        "outcome": trace.get("outcome"),
        "length": trace.get("length"),
        # carried through so the aggregate knows which regression gates apply
        "root_mission": trace.get("root_mission"),
        "obedience_latencies": latencies,
        "obedience_censored": censored,
        "obedience_by_task": obedience_by_task,
        **_staging(trace),
        **_doctrine(trace),
        **_rank_alive_steps(trace),
        **_retasks(trace),
        **_order_pay(trace),
        **_reports(trace),
        **_false_complete(trace),
        **_done_opportunity(trace),
        **_endex_close(trace),
        **_root_sitreps(trace),
        "succession_events": events,
        "succession_recovery": recovery,
        "succession_unrecovered": unrecovered,
        "coverage_pairs": pairs,
        "coverage_covered": covered,
        **_vocabulary(trace),
        **_human_exposure(trace),
        **_fight_disposition(trace),
        **_cohesion(trace),
        **_traffic(trace),
    }


# ---------------------------------------------------------------------- #
# run-level aggregation
# ---------------------------------------------------------------------- #


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate_behavior(episodes: list[dict]) -> dict[str, Any]:
    """Pool per-episode behavior dicts into the run-level metric summary.

    Event-level metrics pool events across episodes (a latency mean over all
    orders of the run, rates over total counts); the human-exposure means
    average per-episode values so long and short episodes weigh equally.
    Metrics whose denominator never occurred are None, with the counts kept
    alongside so a None is always explainable.
    """
    def total(key: str) -> int:
        return sum(ep.get(key, 0) for ep in episodes)

    n_eps = len(episodes)
    retasks_by_rank: dict[str, dict[str, int]] = {}
    for ep in episodes:
        for rank, bucket in ep.get("retasks_by_rank", {}).items():
            dst = retasks_by_rank.setdefault(rank, {"priced": 0, "excepted": 0})
            dst["priced"] += bucket.get("priced", 0)
            dst["excepted"] += bucket.get("excepted", 0)

    orders_by_task: dict[str, dict[str, int]] = {}
    for ep in episodes:
        for task, bucket in ep.get("orders_by_task", {}).items():
            dst = orders_by_task.setdefault(task, dict.fromkeys(DOCTRINE_TIERS, 0))
            for tier in DOCTRINE_TIERS:
                dst[tier] += bucket.get(tier, 0)

    orders_by_rank: dict[str, dict[str, int]] = {}
    for ep in episodes:
        for rank, bucket in ep.get("orders_by_rank", {}).items():
            dst = orders_by_rank.setdefault(rank, dict.fromkeys(DOCTRINE_TIERS, 0))
            for tier in DOCTRINE_TIERS:
                dst[tier] += bucket.get(tier, 0)

    order_pay_by_rank: dict[str, dict[str, Any]] = {}
    for ep in episodes:
        for rank, bucket in ep.get("order_pay_by_rank", {}).items():
            dst = order_pay_by_rank.setdefault(
                rank, {"fresh": 0, "churn": 0, "retask": 0, "pay": 0.0}
            )
            for key in ("fresh", "churn", "retask", "pay"):
                dst[key] += bucket.get(key, 0)

    # refs #53: the opportunity denominator for `orders_by_rank` and
    # `order_pay_by_rank` above — both raw per-rank totals that a
    # longer-lived issuer inflates without commanding any harder. Carried
    # beside them rather than folded in, so `sum(bucket.values())` on either
    # dict stays exactly the count it always was.
    rank_alive_steps: dict[str, int] = {}
    for ep in episodes:
        for rank, n in ep.get("rank_alive_steps", {}).items():
            rank_alive_steps[rank] = rank_alive_steps.get(rank, 0) + n
    # refs #16: the matched availability control, pooled over the run. Stored
    # as a share (already divided by orders_matched) so behavior.json carries
    # something directly comparable to the task mix beside it.
    availability_sum: dict[str, float] = {}
    for ep in episodes:
        for task, value in (ep.get("order_availability") or {}).items():
            availability_sum[task] = availability_sum.get(task, 0.0) + value
    orders_matched = total("orders_matched")
    order_availability = (
        {task: value / orders_matched for task, value in sorted(availability_sum.items())}
        if orders_matched
        else {}
    )
    # judgeable orders: the issuer held a mission to derive from
    derivable = total("orders_preferred") + total("orders_allowed") + total("orders_violating")

    latencies = [v for ep in episodes for v in ep["obedience_latencies"]]
    # refs the v9/v10 obedience regression: pooled latency cannot separate
    # "the cohort stopped obeying" from "the cohort was ordered to do slower
    # things", so it is also reported per ordered task
    obedience_by_task: dict[str, dict] = {}
    for ep in episodes:
        for task, bucket in ep.get("obedience_by_task", {}).items():
            dst = obedience_by_task.setdefault(task, {"latencies": [], "censored": 0})
            dst["latencies"].extend(bucket["latencies"])
            dst["censored"] += bucket["censored"]
    obedience_task_summary = {
        task: {
            "latency_mean": _mean(b["latencies"]),
            "orders": len(b["latencies"]) + b["censored"],
            "censored": b["censored"],
        }
        for task, b in sorted(obedience_by_task.items())
    }
    recovery = [v for ep in episodes for v in ep["succession_recovery"]]
    humans = [ep for ep in episodes if ep["human_died"] is not None]
    roots = {ep.get("root_mission") for ep in episodes} - {None}
    # refs #35: the freshness interval the off-cadence count was scored
    # against, carried so the density reads against its own bound the way
    # episode length reads against max_steps. None on a mixed pool.
    intervals = {ep.get("sitrep_interval") for ep in episodes} - {None}
    successes = [ep for ep in episodes if ep["outcome"] == "success"]
    n_successes = len(successes)
    return {
        "episodes": n_eps,
        # the run's root mission when the pooled episodes agree on one; the
        # regression gates key off it (a mixed pool gates on nothing)
        "root_mission": roots.pop() if len(roots) == 1 else None,
        "success_rate": _ratio(n_successes, n_eps),
        "obedience_latency_mean": _mean(latencies),
        "obedience_orders": len(latencies) + total("obedience_censored"),
        "obedience_censored": total("obedience_censored"),
        "obedience_by_task": obedience_task_summary,
        # A5-2 staging (refs #15): the channel obedience latency deliberately
        # cannot see. `staged_abandoned` is the one that reads as a fault —
        # an order issued AT MY COMMAND whose EXECUTE never came.
        "orders_staged": total("orders_staged"),
        "staged_released": total("staged_released"),
        "staged_abandoned": total("orders_staged") - total("staged_released"),
        "staging_gap_mean": _mean([g for ep in episodes for g in ep.get("staging_gaps", [])]),
        "report_precision": _ratio(total("contacts_informative"), total("contacts")),
        "report_recall": _ratio(total("enemies_reported"), total("enemies_seen")),
        "contact_reports": total("contacts"),
        "doctrine_preference_rate": _ratio(total("orders_preferred"), total("orders_issued")),
        # refs #14: preference is `allowed[0]`, so a policy that adopts a
        # legal alternative leg (ADVANCE, added to the DEFEND/SEIZE/RECON
        # tables in A5) scores identically to one that disregards doctrine.
        # The containment rate separates them; orders_by_task shows whether a
        # low preference rate is catalog adoption or command quality.
        "doctrine_allowed_rate": _ratio(
            total("orders_preferred") + total("orders_allowed"), derivable
        ),
        "orders_allowed": total("orders_allowed"),
        "orders_violating": total("orders_violating"),
        "orders_underivable": total("orders_underivable"),
        "orders_by_task": orders_by_task,
        "orders_by_rank": orders_by_rank,
        "order_pay_by_rank": order_pay_by_rank,
        # refs #53: alive-steps per effective rank — the denominator that
        # turns the two raw dicts above into a rate (orders or pay per step
        # alive), comparable across arms whose episodes run different
        # lengths. `sum(orders_by_rank[rank].values()) / rank_alive_steps[rank]`
        # is the corrected reading; the raw sum on its own is not.
        "rank_alive_steps": rank_alive_steps,
        # refs #16: what the mask offered, for the same order decisions. A
        # task's share divided by its availability is the *selection lift* —
        # 1.00 is exactly the masked-random floor, and it is the only form of
        # the mix that can be compared across scenarios whose order menus
        # differ. `orders_matched` below the issued count means orders were
        # seen that the issuer's own mask did not offer (injected traces).
        "order_availability": order_availability,
        "orders_matched": orders_matched,
        "orders_issued": total("orders_issued"),
        "orders_per_episode": _ratio(total("orders_issued"), n_eps),
        "retasks": total("retasks"),
        "retasks_priced": total("retasks_priced"),
        "retasks_excepted": total("retasks_excepted"),
        "retask_rotations": total("retask_rotations"),
        "retasks_per_episode": _ratio(total("retasks"), n_eps),
        "retasks_priced_per_episode": _ratio(total("retasks_priced"), n_eps),
        "retasks_by_rank": retasks_by_rank,
        "false_complete_rate": _ratio(total("done_rejected"), total("done_reports")),
        "done_reports": total("done_reports"),
        "done_rejected": total("done_rejected"),
        # refs #23: the root's own channel, and the concentration the pooled
        # ratio cannot see. claims-per-claiming-episode near 1.0 is a policy
        # filing a report; a large one is a policy spamming a channel whose
        # rejection rate can look identical either way.
        "done_reports_root": total("done_reports_root"),
        "done_rejected_root": total("done_rejected_root"),
        "false_complete_rate_root": _ratio(
            total("done_rejected_root"), total("done_reports_root")
        ),
        "done_claim_episodes": total("done_claim_episode"),
        "done_claim_episodes_root": total("done_claim_episode_root"),
        "done_claims_per_claiming_episode": _ratio(
            total("done_reports"), total("done_claim_episode")
        ),
        "done_claims_per_claiming_episode_root": _ratio(
            total("done_reports_root"), total("done_claim_episode_root")
        ),
        # refs #13: the opportunity denominator. done_claim_rate is None only
        # when the channel was never open — which is itself the finding.
        "done_admissible": total("done_admissible"),
        "done_admissible_root": total("done_admissible_root"),
        "done_claim_rate": _ratio(total("done_reports"), total("done_admissible")),
        # v1.13: the completion signal on a continuous-posture root, where
        # false_complete_rate is structurally 0 and says nothing. None when no
        # ENDEX was sent — a completable root, or no successful operation.
        "endex_sent": total("endex_sent"),
        "closed_on_root_report_rate": _ratio(
            total("endex_on_root_report"), total("endex_sent")
        ),
        # refs #35: the denominator that rate never had. It reads 1.00 both
        # for a root that timed one report to the closing moment and for a
        # root that transmitted 30 of them; these three separate the two.
        # `closes_per_root_sitrep` is the closes bought per report emitted —
        # high is timed, low is bought with volume.
        # `closed_on_cadence_report_rate` asks the same question the rate
        # above asks, of the same denominator (operations COMMAND closed), but
        # counts only closes made by a report the cadence would have produced
        # anyway. The gap between the two rates is the volume-bought share.
        "root_sitreps": total("root_sitreps"),
        "root_sitreps_off_cadence": total("root_sitreps_off_cadence"),
        "root_sitreps_per_episode": _ratio(total("root_sitreps"), n_eps),
        "root_sitrep_off_cadence_share": _ratio(
            total("root_sitreps_off_cadence"), total("root_sitreps")
        ),
        "closes_per_root_sitrep": _ratio(total("endex_on_root_report"), total("root_sitreps")),
        "closed_on_cadence_report_rate": _ratio(
            total("endex_on_cadence_report"), total("endex_sent")
        ),
        "sitrep_interval": intervals.pop() if len(intervals) == 1 else None,
        # v1.16, the question #31 raised: of the operations that SUCCEEDED, how
        # many said so on the net at all? Separate from
        # closed_on_root_report_rate, whose denominator is ENDEXes sent and
        # which therefore cannot see an operation that closed in silence — the
        # exact blind spot that let v1.14 announce 0 of 57 successes on
        # fireteam_defend without any published number moving.
        "successes": n_successes,
        "successes_announced": sum(ep.get("close_announced", 0) for ep in successes),
        "successes_announced_rate": _ratio(
            sum(ep.get("close_announced", 0) for ep in successes), n_successes
        ),
        "succession_recovery_mean": _mean(recovery),
        "succession_events": total("succession_events"),
        "succession_unrecovered": total("succession_unrecovered"),
        "coverage_time": _ratio(total("coverage_covered"), total("coverage_pairs")),
        # A5 vocabulary usage
        "advance_orders_per_episode": _ratio(total("advance_orders"), n_eps),
        "timed_orders_per_episode": _ratio(total("timed_orders"), n_eps),
        "execute_signals_per_episode": _ratio(total("execute_signals"), n_eps),
        "formation_orders_per_episode": _ratio(total("formation_orders"), n_eps),
        "sync_proposals_per_episode": _ratio(total("sync_proposals"), n_eps),
        "sync_bounds_per_episode": _ratio(total("sync_bounds"), n_eps),
        "stance_share": _ratio(total("stance_steps"), total("stance_agent_steps")),
        "human_death_rate": _ratio(sum(ep["human_died"] for ep in humans), len(humans)),
        "human_mean_enemy_dist": _mean([ep["human_mean_enemy_dist"] for ep in humans]),
        "human_mean_objective_dist": _mean([ep["human_mean_objective_dist"] for ep in humans]),
        "human_ring_entries_mean": _mean([ep["human_ring_entries"] for ep in humans]),
        # fight disposition (issue #11): pooled over threatened (soldier,
        # step) pairs, so a long firefight weighs more than a brief brush —
        # which is the intent, the question being where the fighting happens
        "cover_occupancy_under_threat": _ratio(total("threat_cover_pairs"), total("threat_pairs")),
        "mean_distance_from_objective_under_threat": _ratio(
            total("threat_objective_dist_sum"), total("threat_objective_dist_pairs")
        ),
        "threat_pairs": total("threat_pairs"),
        # cohesion (owner's axis, 2026-08-18): pooled over living-agent-steps,
        # so a long episode weighs more — the question being how much of the
        # cohort's lived time is spent isolated. Measured, never enforced.
        "no_close_teammate_rate": _ratio(
            total("no_close_teammate_steps"), total("cohesion_agent_steps")
        ),
        "unseen_by_any_teammate_rate": _ratio(
            total("unseen_by_any_teammate_steps"), total("cohesion_agent_steps")
        ),
        "cohesion_agent_steps": total("cohesion_agent_steps"),
        # clock expiry + traffic composition (refs #18). The rate is the
        # gated number; the composition is diagnosis, deliberately not gated
        # — measured across the fleet, a healthy `fireteam_defend_v10` runs
        # at a command share of 0.026 and a stalled `squad_recon_v6` at
        # 0.022, so a fixed bound on composition separates nothing. It reads
        # as a *within-scenario* contrast (best vs latest), not a threshold.
        "timeout_rate": _ratio(sum(ep.get("ran_out_the_clock", False) for ep in episodes), n_eps),
        "episode_length_mean": _mean([ep.get("length") for ep in episodes]),
        "max_steps": next((ep["max_steps"] for ep in episodes if ep.get("max_steps")), None),
        "messages_per_episode": _ratio(total("messages"), n_eps),
        "command_traffic_share": _ratio(total("messages_command"), total("messages")),
        "voice_traffic_share": _ratio(total("messages_voice"), total("messages")),
    }


# ---------------------------------------------------------------------- #
# regression gates
# ---------------------------------------------------------------------- #


def regression_gates(agg: dict[str, Any]) -> list[dict[str, Any]]:
    """Pass/fail bounds a retrain must clear, given an aggregated summary.

    The positional gate (issue #11) applies to DEFEND roots only: holding
    ground is the one root mission for which "fought here rather than there"
    is a correctness property rather than a style. It exists because success
    rate alone did not catch it — ``fireteam_defend_v7`` halved the
    root-death rate the ROADMAP had blamed, was measured firing on every
    threatened step (p(fire | threatened) 0.005 -> 1.000), and still missed,
    because it had walked off the position it was ordered to hold. Three
    million steps bought that lesson; these two numbers are ~free.

    The clock-expiry gate (issue #18) applies to *every* root mission,
    because running the step ceiling out is a failure mode no scenario wants
    and the one signal that separated the collapsed checkpoints from the
    healthy ones across the whole fleet: 0.0-0.2 healthy against exactly 1.0
    for all three stalls. It is not redundant with the success rate — it
    says *how* the episodes were lost, and a policy that rides out the clock
    is a different repair from one that gets killed on the way in.

    The success-axis gate (issue #21) also applies to every root mission, and
    exists because the clock-expiry gate has a blind spot: it is tuned to the
    STALL shape (near-zero success WITH the clock running out) and the defend
    family's worst measured collapses are DEFEAT-shaped instead — the team is
    wiped well before `max_steps`, so `timeout_rate` reads low and the only
    reason the fleet's own composite gate (elsewhere: `human_death_rate`
    gated on `timeout_rate <= 0.5`) ever caught them was that a wiped team's
    commander usually dies with it — a proxy, not a measurement of success.
    This gate reads `success_rate` directly instead, but ONLY once
    `timeout_rate` has cleared its own ceiling: a run already failing on the
    clock is stall-shaped, not wipe-shaped, and should read as one verdict,
    not two. That makes the two axes mutually exclusive by construction — a
    collapsed run fails exactly one of them, so the report always says which
    shape it was.

    Gates whose metric is None (never measured this run) are reported with
    ``passed=None`` — unmeasured is not the same as passed.
    """
    timeout_rate = agg.get("timeout_rate")
    gates = [_gate("timeout_rate", timeout_rate, TIMEOUT_RATE_CEILING, "max")]
    # refs #21: only meaningful once we know the run isn't stall-shaped —
    # otherwise a stalled run would fail both axes and the report could not
    # say which collapse it was. `timeout_rate is None` (never measured)
    # leaves the shape unknown, so the gate is skipped rather than guessed.
    if timeout_rate is not None and timeout_rate <= TIMEOUT_RATE_CEILING:
        gates.append(_gate("success_rate", agg.get("success_rate"), SUCCESS_RATE_FLOOR, "min"))
    # refs v1.20: unconditional, because muteness is not a collapse shape — a
    # run can pass every other gate and still never have reported anything.
    gates.append(
        _gate(
            "closed_on_root_report_rate",
            agg.get("closed_on_root_report_rate"),
            ROOT_REPORT_CLOSE_FLOOR,
            "min",
        )
    )
    if agg.get("root_mission") != MissionType.DEFEND.name:
        return gates
    return [
        *gates,
        _gate(
            "cover_occupancy_under_threat",
            agg.get("cover_occupancy_under_threat"),
            DEFEND_COVER_FLOOR,
            "min",
        ),
        _gate(
            "mean_distance_from_objective_under_threat",
            agg.get("mean_distance_from_objective_under_threat"),
            DEFEND_OBJECTIVE_DIST_CEILING,
            "max",
        ),
    ]


def _gate(name: str, value: float | None, bound: float, direction: str) -> dict[str, Any]:
    if value is None:
        passed = None
    else:
        passed = value >= bound if direction == "min" else value <= bound
    return {"name": name, "value": value, "bound": bound, "direction": direction, "passed": passed}


def split_gates(gates: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Gate names that FAILED, and those that were never measured.

    ``_gate`` emits ``passed=None`` for a metric this run never produced, and
    the distinction is the whole point: unmeasured is not passed, and it is
    not failed either. Every reader that reduced the tri-state with a plain
    ``not g["passed"]`` silently filed the unmeasured ones under failures —
    ``squad_v21_seed16`` never completed an episode, so
    ``closed_on_root_report_rate`` had nothing to measure, and it was the run
    that surfaced this. Callers that need one bucket should say which.
    """
    failed = [g["name"] for g in gates if g.get("passed") is False]
    unmeasured = [g["name"] for g in gates if g.get("passed") is None]
    return failed, unmeasured


def format_gate_report(gates: list[dict[str, Any]]) -> str:
    """Human-readable gate verdicts; empty string when nothing gates."""
    if not gates:
        return ""
    lines = ["regression gates:"]
    for g in gates:
        mark = "—" if g["passed"] is None else ("PASS" if g["passed"] else "FAIL")
        value = "—" if g["value"] is None else f"{g['value']:.3f}"
        rel = ">=" if g["direction"] == "min" else "<="
        lines.append(f"  [{mark:^4}] {g['name']:<42} {value:>7}  ({rel} {g['bound']})")
    return "\n".join(lines)


#: (key, label, format) rows of the printed behavior table, in display order.
_TABLE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("obedience_latency_mean", "obedience latency (steps)", "{:.1f}"),
    ("orders_per_episode", "orders issued / ep", "{:.1f}"),
    ("retasks_per_episode", "re-tasks / ep", "{:.1f}"),
    ("retasks_priced_per_episode", "priced re-tasks / ep", "{:.2f}"),
    ("report_precision", "report precision", "{:.2f}"),
    ("report_recall", "report recall", "{:.2f}"),
    ("doctrine_preference_rate", "doctrine preference", "{:.2f}"),
    ("doctrine_allowed_rate", "doctrine containment", "{:.2f}"),
    ("false_complete_rate", "false-COMPLETE rate", "{:.2f}"),
    ("done_claims_per_claiming_episode", "COMPLETE claims / claiming ep", "{:.2f}"),
    ("done_claim_rate", "COMPLETE claim rate", "{:.4f}"),
    ("closed_on_root_report_rate", "closed on root's report", "{:.2f}"),
    ("closed_on_cadence_report_rate", "  ... on a cadence report", "{:.2f}"),
    ("root_sitreps_per_episode", "root SITREPs / ep", "{:.1f}"),
    ("closes_per_root_sitrep", "closes / root SITREP", "{:.3f}"),
    ("successes_announced_rate", "successes announced on the net", "{:.2f}"),
    ("succession_recovery_mean", "succession recovery (steps)", "{:.1f}"),
    ("coverage_time", "subordinate coverage time", "{:.2f}"),
    ("advance_orders_per_episode", "ADVANCE orders / ep", "{:.1f}"),
    ("timed_orders_per_episode", "timed orders / ep", "{:.2f}"),
    ("formation_orders_per_episode", "FORMATION orders / ep", "{:.2f}"),
    ("stance_share", "stance-governed step share", "{:.2f}"),
    ("sync_bounds_per_episode", "sync bounds (GO) / ep", "{:.2f}"),
    ("human_mean_enemy_dist", "human: mean dist to enemy", "{:.1f}"),
    ("human_ring_entries_mean", "human: ring entries / ep", "{:.2f}"),
    ("human_death_rate", "human: death rate", "{:.2f}"),
    ("cover_occupancy_under_threat", "fight: cover occupancy", "{:.3f}"),
    ("mean_distance_from_objective_under_threat", "fight: dist from OBJ", "{:.2f}"),
    ("no_close_teammate_rate", "cohesion: no close teammate", "{:.3f}"),
    ("unseen_by_any_teammate_rate", "cohesion: unseen by teammates", "{:.3f}"),
    ("timeout_rate", "ran the clock out", "{:.2f}"),
    ("messages_per_episode", "messages / ep", "{:.0f}"),
    ("command_traffic_share", "of which command", "{:.3f}"),
    ("voice_traffic_share", "of which voice (SYNC)", "{:.3f}"),
)


def format_obedience_by_task(agg: dict[str, Any], top: int = 3) -> str:
    """``TASK mean(n)`` obedience latency for the most-ordered tasks.

    The pooled mean cannot tell "the cohort stopped obeying" from "the cohort
    was ordered to do slower things" — the defend line went 1.26 steps at v8 to
    13.06 at v10 while the ADVANCE share of orders went 0.69 → 0.99, and an
    ADVANCE to a distant control measure resolves slower than a DEFEND in place
    no matter how obedient anyone is. Split per task, the two read differently:
    a rise concentrated in ADVANCE is a mix shift; a rise *within* DEFEND is a
    real obedience regression.
    """
    by_task = agg.get("obedience_by_task") or {}
    ranked = sorted(by_task.items(), key=lambda kv: -kv[1]["orders"])[:top]
    parts = []
    for task, b in ranked:
        mean = b["latency_mean"]
        shown = f"{mean:.1f}" if mean is not None else "—"
        parts.append(f"{task} {shown}({b['orders']})")
    return ", ".join(parts)


def format_order_task_mix(agg: dict[str, Any], top: int = 3) -> str:
    """``TASK share/preference`` for the most-ordered tasks (refs #14).

    Reads the preference rate conditioned on the ordered task against that
    task's share of all agent-issued orders, so a low overall preference can
    be attributed: ``ADVANCE .96/.00`` is a policy that adopted one legal
    alternative leg wholesale, not one issuing bad orders.
    """
    by_task = agg.get("orders_by_task") or {}
    issued = sum(sum(b.values()) for b in by_task.values())
    if not issued:
        return ""
    ranked = sorted(by_task.items(), key=lambda kv: -sum(kv[1].values()))[:top]
    return ", ".join(
        f"{task} {sum(b.values()) / issued:.2f}/{b.get('preferred', 0) / sum(b.values()):.2f}"
        for task, b in ranked
    )


def order_selection_lift(agg: dict[str, Any]) -> dict[str, float | None]:
    """``share / availability`` per ordered task — 1.00 is the mask's own floor.

    The availability-corrected form of the ordered-task mix (refs issue #16).
    A raw share answers "how often was this ordered", which conflates two
    different findings, because the order mask does not offer the tasks in
    equal numbers:

    * ``lift > 1`` — the policy chose the task *more* than picking uniformly
      among the legal orders it was actually holding would have;
    * ``lift == 1`` — indistinguishable from the mask; no preference at all;
    * ``lift < 1`` — the policy declined opportunities it had.

    The correction changes what the reading *says*, not just its size.
    `fireteam_defend` offers SUPPORT 0.219 of the menu against OBSERVE's
    0.112, so `fireteam_defend_v8`'s raw 0.102/0.010 — read for a generation
    as a strong OBSERVE preference — is OBSERVE **x0.92** (the floor: no
    preference at all) against SUPPORT **x0.04**. The finding is SUPPORT
    avoidance, and the raw ratio understated it while misnaming its cause. In
    `squad`, where OBSERVE is offered 2.9x more than SUPPORT, the same raw
    ratio overstates instead — the confound has no fixed sign.

    ``None`` for a task the mask never offered on this corpus: no opportunity,
    so no selection to measure (SUPPORT under a SCREEN root, which cannot
    derive it, is the standing example).
    """
    by_task = agg.get("orders_by_task") or {}
    availability = agg.get("order_availability") or {}
    issued = sum(sum(b.values()) for b in by_task.values())
    if not issued:
        return {}
    lift: dict[str, float | None] = {}
    for task in sorted(set(by_task) | set(availability)):
        offered = availability.get(task, 0.0)
        share = sum(by_task.get(task, {}).values()) / issued
        lift[task] = share / offered if offered else None
    return lift


def format_order_availability(agg: dict[str, Any], top: int = 3) -> str:
    """``TASK share/availability (xLIFT)`` for the tasks the mask offered most.

    The companion of :func:`format_order_task_mix`: same mix, read against
    what was on the menu (refs #16). Ranked by availability rather than by
    orders issued on purpose — the reading that matters is the task the mask
    offered and the policy did *not* take, and ranking by issued orders is
    exactly what hides it.
    """
    availability = agg.get("order_availability") or {}
    by_task = agg.get("orders_by_task") or {}
    issued = sum(sum(b.values()) for b in by_task.values())
    if not availability or not issued:
        return ""
    lift = order_selection_lift(agg)
    ranked = sorted(availability.items(), key=lambda kv: -kv[1])[:top]
    parts = []
    for task, offered in ranked:
        share = sum(by_task.get(task, {}).values()) / issued
        value = lift.get(task)
        shown = "—" if value is None else f"x{value:.2f}"
        parts.append(f"{task} {share:.2f}/{offered:.2f} ({shown})")
    return ", ".join(parts)


def format_staging(agg: dict[str, Any]) -> str:
    """``staged N, released N (gap X), abandoned N`` — the A5-2 staging channel.

    Empty when nothing was ever staged. ``abandoned`` is the number worth
    reading: those orders were transmitted, staged a recipient, and never
    released, so they bought a held agent and no execution (refs #15).
    """
    staged = agg.get("orders_staged") or 0
    if not staged:
        return ""
    gap = agg.get("staging_gap_mean")
    shown = f"{gap:.1f}" if gap is not None else "—"
    return (
        f"staged {staged}, released {agg.get('staged_released', 0)} "
        f"(mean gap {shown} steps), abandoned {agg.get('staged_abandoned', 0)}"
    )


def format_root_claim_shape(agg: dict[str, Any]) -> str:
    """What the root's own claim channel did, beside the announcement (#38).

    ``successes_announced`` is one integer, and a zero on it has at least three
    causes that want different fixes. Two published runs made the point: at
    N=100 on the final policy, `patrol_brique_v5` announces 0 of 99 with the
    root **never claiming**, and `platoon_v5` announces 0 of 100 with the root
    **claiming five times and refused every time**. On the radio one is a
    silent policy and the other a rejected one; grouped by the integer alone
    they read as the same result, and the README grouped them.

    The three shapes this renders, which are #13's argument about zero DONE
    reports carried over to the announcement:

    * **channel shut** — no admissible agent-step at all, so no price was ever
      consulted. The v1.17 defend family, by mask and by design.
    * **declined** — the act was admissible and never used. `squad_v8`/best
      declines it at 8812 admissible steps; a policy, not a mask.
    * **refused** — claimed and rejected by the umpire. Upstream of the
      announcement: extending COMMAND's close to completable roots changes who
      announces, and would leave this untouched.

    Denominator note, stated because the line sits next to a rate that has a
    different one: the claim counts pool over ALL episodes, while
    ``successes_announced`` counts over the successful ones.
    """
    claims = agg.get("done_reports_root")
    if claims is None:
        return ""
    admissible = agg.get("done_admissible_root") or 0
    refused = agg.get("done_rejected_root") or 0
    if not claims:
        return "root never claimed, channel shut" if not admissible else (
            f"root never claimed, {admissible} admissible step{'' if admissible == 1 else 's'}"
        )
    if refused == claims:
        return f"root claimed {claims}, all refused"
    return f"root claimed {claims}, {refused} refused"


def format_behavior_table(agg: dict[str, Any]) -> str:
    """Human-readable table of an aggregated behavior summary."""
    by_rank = ", ".join(
        f"{rank} {b['priced']}p/{b['excepted']}e"
        for rank, b in sorted(agg.get("retasks_by_rank", {}).items())
    )
    # refs #53: both lines below carry their alive-steps denominator beside
    # the raw count, and the rate it implies (per step alive) — the raw count
    # alone is not comparable across arms whose episodes run different
    # lengths (a longer-lived issuer racks up more of it at the SAME rate).
    rank_alive = agg.get("rank_alive_steps", {})
    orders_by_rank = ", ".join(
        f"{rank} {sum(b.values())}/{rank_alive.get(rank, 0)}a"
        f" ({_ratio(sum(b.values()), rank_alive.get(rank, 0)) or 0.0:.2f}/step,"
        f" {b.get('preferred', 0)} pref)"
        for rank, b in sorted(agg.get("orders_by_rank", {}).items())
    )
    order_pay = ", ".join(
        f"{rank} {b['fresh']}f/{b['churn']}c/{b['retask']}r/{rank_alive.get(rank, 0)}a"
        f" {b['pay']:+.1f} ({_ratio(b['pay'], rank_alive.get(rank, 0)) or 0.0:+.3f}/step)"
        for rank, b in sorted(agg.get("order_pay_by_rank", {}).items())
    )
    task_mix = format_order_task_mix(agg)
    notes = {
        "obedience_latency_mean": (
            f"n={agg['obedience_orders']}, censored {agg['obedience_censored']}"
            + (f", staged {agg['orders_staged']} excluded" if agg.get("orders_staged") else "")
        ),
        "timed_orders_per_episode": format_staging(agg),
        "retasks_per_episode": (
            f"priced {agg.get('retasks_priced', 0)}, excepted {agg.get('retasks_excepted', 0)}, "
            f"rotations {agg.get('retask_rotations', 0)}"
            + (f"; by rank: {by_rank}" if by_rank else "")
        ),
        "report_precision": f"n={agg['contact_reports']}",
        "doctrine_preference_rate": f"n={agg['orders_issued']}" + (f"; {task_mix}" if task_mix else ""),
        # refs #16: the same mix against what the mask offered, so a task
        # share is readable as a preference instead of as an opportunity count
        "orders_per_episode": (
            (f"share/avail: {avail}" if (avail := format_order_availability(agg)) else "")
            # refs #52: who issued them. The task mix is team-wide, so it cannot
            # tell a commander that orders from the rear from one that advances
            # with its element.
            + (f"; by rank: {orders_by_rank}" if orders_by_rank else "")
            # refs #52: and whether issuing them pays. fresh/churn/retask counts
            # with the order-channel total — order-channel only, the re-task
            # price is charged elsewhere and is not netted in here.
            + (f"; pay: {order_pay}" if order_pay else "")
        ),
        "doctrine_allowed_rate": (
            f"{agg.get('orders_allowed', 0)} allowed, {agg.get('orders_violating', 0)} violating"
        ),
        "false_complete_rate": (
            f"n={agg['done_reports']}"
            + (
                f"; root {agg['done_reports_root']} claims, "
                f"{agg['false_complete_rate_root']:.2f} rejected"
                if agg.get("false_complete_rate_root") is not None
                else "; none by the root"
            )
        ),
        # refs #23: volume against the episodes that carried it. 1.00 is one
        # claim per claiming episode; a large number is spam with the same
        # rejection rate.
        "done_claims_per_claiming_episode": (
            f"{agg['done_claim_episodes']} of {agg['episodes']} episodes claimed"
            + (
                f"; root {agg['done_claims_per_claiming_episode_root']:.2f} over "
                f"{agg['done_claim_episodes_root']}"
                if agg.get("done_claims_per_claiming_episode_root") is not None
                else ""
            )
        ),
        "closed_on_root_report_rate": f"n={agg['endex_sent']} ENDEX",
        # refs #35: the density beside the rate, not inferable from it
        "root_sitreps_per_episode": (
            f"{agg.get('root_sitreps', 0)} total"
            + (
                f", {agg['root_sitrep_off_cadence_share']:.0%} off cadence"
                if agg.get("root_sitrep_off_cadence_share") is not None
                else ""
            )
            + (f" (interval {agg['sitrep_interval']})" if agg.get("sitrep_interval") else "")
        ),
        "closes_per_root_sitrep": f"n={agg.get('root_sitreps', 0)} root SITREPs",
        # refs #38: the announcement and, beside it, what the root's own claim
        # channel did — a zero here is a silent policy, a declined one or a
        # refused one, and the integer alone cannot say which
        "successes_announced_rate": (
            f"{agg['successes_announced']} of {agg['successes']} wins, "
            f"{agg['endex_sent']} by ENDEX"
            + (f"; {shape}" if (shape := format_root_claim_shape(agg)) else "")
        ),
        "done_claim_rate": (
            f"{agg.get('done_admissible', 0)} admissible agent-steps "
            f"({agg.get('done_admissible_root', 0)} the root's)"
        ),
        "succession_recovery_mean": (
            f"n={agg['succession_events']}, unrecovered {agg['succession_unrecovered']}"
        ),
        "cover_occupancy_under_threat": f"n={agg.get('threat_pairs', 0)} threatened agent-steps",
        "no_close_teammate_rate": f"n={agg.get('cohesion_agent_steps', 0)} living agent-steps",
        # the length next to its own ceiling: "375" only means "pinned" once
        # the reader is told the cap is 375 (refs #18)
        "timeout_rate": (
            f"mean length {agg['episode_length_mean']:.0f}/{agg['max_steps']} steps"
            if agg.get("episode_length_mean") is not None and agg.get("max_steps")
            else ""
        ),
    }
    notes = {k: v for k, v in notes.items() if v}
    lines = [f"behavior over {agg['episodes']} episodes:"]
    for key, label, fmt in _TABLE_ROWS:
        value = agg.get(key)
        text = "—" if value is None else fmt.format(value)
        note = f"   ({notes[key]})" if key in notes else ""
        lines.append(f"  {label:<28} {text:>7}{note}")
    return "\n".join(lines)
