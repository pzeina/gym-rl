"""Reward weights and per-agent component bookkeeping.

The reward teaches each rank its trade:

* everyone     — execute the standing order (compliance shaping), stay alive
* subordinates — report contacts (only *new* intel pays), send timely
                 SITREPs, report MISSION COMPLETE truthfully
* leaders      — derive doctrine-preferred orders, keep every living
                 subordinate tasked (coverage), don't spam re-orders (churn),
                 and let standing orders STAND: re-tasking an already-tasked
                 subordinate is a burden on command, priced by the issuer's
                 rank, unless the tactical picture changed (B5)
* team         — win: shared terminal reward plus a speed bonus

Every scalar reward is the sum of named components; the components are
exposed per step in ``info`` so training curves can show *why* the cohort is
getting better (more compliance vs. more kills vs. better reporting).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardConfig:
    """All reward weights in one place."""

    time_penalty: float = -0.01
    compliance_weight: float = 0.1

    # Binding orders (B5) — standing-order tenure. Positive compliance credit
    # grows with how long the CURRENT order has been held:
    #   credit x (1 + tenure_factor x min(steps_held, tenure_horizon) / tenure_horizon)
    # so a settled, executed order out-earns a churned one. Tenure resets when
    # the mission is re-assigned (step_assigned restamps); an identical
    # reissue is a no-op and does NOT reset it. Negative compliance is never
    # amplified (the multiplier applies to positive scores only). 0 → off.
    tenure_factor: float = 0.5
    tenure_horizon: int = 40

    # Comms discipline (A4): every LEARNED transmission — CONTACT / SITREP /
    # MISSION COMPLETE / an order — costs airtime, charged at emission.
    # Auto-traffic (WILCO, DONE verdicts, CASUALTY, succession, SUPPORT_END)
    # is protocol, not a policy decision, and stays free. A transmission
    # dropped by net arbitration (NET BUSY) was never emitted: no cost.
    transmission_cost: float = -0.01

    contact_new: float = 0.5          # first report of an enemy the team didn't know
    contact_redundant: float = -0.02  # pure noise: every reported enemy fresh on the picture
    contact_refresh_age: int = 20     # a re-report is a legitimate picture REFRESH (worth
    #                                   exactly 0, not the noise penalty) once at least one
    #                                   reported enemy's picture entry is this many steps
    #                                   old; younger re-reports are duplicate storms.
    #                                   Must be < KNOWLEDGE_TTL for refreshes to exist.
    sitrep_fresh: float = 0.05        # sitrep after >= sitrep_interval quiet steps
    sitrep_spam: float = -0.02
    sitrep_interval: int = 25         # freshness gap; ScenarioSpec.sitrep_cadence
    #                                   overrides it when the reporting doctrine is on
    sitrep_overdue: float = -0.02     # per step out of contact past the mandated
    #                                   cadence without a SITREP (doctrine only)
    done_true: float = 1.0
    done_false: float = -0.5

    # Objective-lost pressure (v1.4 retrain diagnosis): on DEFEND/DENY root
    # missions, every living agent bleeds this per step while any living
    # enemy stands on the root objective. Without it the x1.5 maps + the
    # human/rank death economics made flight the equilibrium: the oracle
    # showed enemies parked ON the objective at full health while defenders
    # farmed location-free SUPPORT/HOLD posture compliance 25 cells away.
    # A defense that has ceded its objective is failing, and now feels it.
    # Pure penalty: adds nothing to the stall-farm bound.
    objective_lost: float = -0.05

    # Observation progress (A2/A7 lesson — the stall exploit): pay each NOVEL
    # step of team observation toward the root RECON/SCREEN success counter.
    # Telescoping: the counter pays only until the success threshold
    # (2 x RECON_OBSERVE_STEPS = 10 steps), so at most 3.0 per episode is
    # earnable — it rewards *finishing* the observation, and cannot be farmed
    # by parking outside the trigger radius.
    observe_progress: float = 0.3

    # Order quality bonuses pay only for *fresh* tasking: the subordinate is
    # untasked, or the leader's own mission changed after the subordinate was
    # last ordered (free propagation credit). Re-ordering inside the stability
    # window without such credit is churn — this is what keeps the radio net
    # readable instead of leaders spam-cycling orders for bonus farming.
    order_preferred: float = 0.15     # doctrine-preferred derivation
    order_allowed: float = 0.05       # doctrine-allowed, not preferred
    order_objective_match: float = 0.05  # subordinate tasked on leader's own objective
    order_churn: float = -0.1         # reissuing the identical standing order (a no-op)

    # Binding orders (B5) — re-task pricing. Re-tasking an already-tasked
    # subordinate (objective change OR mission-type change) is an act of
    # command with real weight, and the weight grows with rank — a high-rank
    # order is a broad intent subordinates adapt within, not a knob to twiddle:
    #   cost = order_retask_cost_base x (1 + order_retask_rank_scale x issuer authority)
    # (TL -0.75, SL -1.0, PL -1.5 at the defaults). A same-objective
    # mission-type-only change (e.g. SEIZE→CLEAR on the same objective) is
    # half price. The cost is WAIVED — legitimate intervention stays free —
    # exactly when the tactical picture changed since the standing order:
    #   * a CONTACT hit the net since the subordinate was last ordered,
    #   * a casualty occurred in the issuer's element (its command subtree)
    #     since, or
    #   * the issuer's own mission changed since (propagation of new intent;
    #     this also pays the fresh-tasking bonuses, as before), or
    #   * the subordinate truthfully reported DONE (structural: the confirmed
    #     claim cleared its mission, so the new order is a fresh tasking).
    # This supersedes the old stability-window churn penalty for tasked
    # subordinates (the identical-reissue no-op path keeps order_churn).
    # order_retask_cost_base = 0 → pricing off.
    order_retask_cost_base: float = -0.5
    order_retask_rank_scale: float = 0.5

    # Formation shaping (A5-3): a member standing at its formation station
    # (COLUMN/LINE/WEDGE geometry in the leader's heading frame) while its
    # stanced leader closes NEW ground toward its mission anchor earns this
    # per step. Watermark-gated on the leader's best-yet anchor distance, so
    # the total payout per (order, stance) is bounded by the initial distance
    # — it telescopes with the advance and is NOT a per-step farm, which is
    # why it does not enter max_step_farm(). Never masked/forced: geometry
    # is shaped, the manual's formations are doctrine, not physics (pp. 14-15).
    formation_bonus: float = 0.03

    # Trinôme bound (A5-4): a synchronized mover (inside the 8-step window a
    # SYNC_GO opens) closing NEW ground toward its own mission anchor while
    # >= 1 synchronized group-mate COVERS the bound (static with LOS to a
    # threat, or overwatching the mover) earns this per step. Watermarked on
    # the standing order like formation_bonus — repeated propose/GO cycles
    # cannot re-earn covered ground, so it telescopes and stays out of
    # max_step_farm(). The covered mover also gets the P2 covered-movement
    # accuracy debuff applied against attackers ("bond par binôme", manual
    # pp. 14-15).
    bound_bonus: float = 0.05

    coverage_bonus: float = 0.01      # all living subordinates tasked
    coverage_gap: float = -0.1        # some living subordinate left untasked, per step.
    #                                   Raised -0.02 → -0.1 in the B5 campaign (the one
    #                                   diagnosed adjustment): re-task pricing suppressed
    #                                   ordering so hard that INITIAL tasking collapsed
    #                                   too (squad coverage time 0.96 → 0.61; a TL2 left
    #                                   untasked for 100+ steps) — but an order that is
    #                                   never issued cannot bind. Initial tasking is free
    #                                   and pays the derivation bonus; silence must cost
    #                                   more than speaking once.

    # Fire discipline by mission (found via oracle diagnosis: combat rewards
    # were dominating mission compliance — recon elements out-shot the static
    # postures, and defenders sallied off the objective to chase kills and
    # died in the open 32:5). With the flag on, the shooter's hit/kill rewards
    # are scaled per core/missions.py: SCREEN → 0 (weapons tight); OBSERVE /
    # SUPPORT / COVER / DEFEND / DENY / HOLD → paid only when firing from the
    # mission position; RECON (may engage) / SEIZE / CLEAR / RALLY / untasked
    # → unchanged. Teammate kill-shares are NOT scaled (the shooter's
    # incentive is the lever).
    fire_discipline: bool = True
    hit_enemy: float = 0.2
    kill_enemy: float = 1.0
    team_kill_share: float = 0.2      # everyone else, per enemy killed
    took_hit: float = -0.1
    death: float = -1.0
    teammate_death: float = -0.2
    rank_casualty_scale: float = 0.25  # death & teammate_death scale with the FALLEN
    #                                    agent's effective authority:
    #                                    x (1 + scale x authority) — losing a PL
    #                                    (authority 4) costs twice a rifleman
    human_death: float = -25.0        # paid by EVERY present agent when a human dies
    #                                   (on top of the normal death penalties): losing
    #                                   the human commander is close to mission failure,
    #                                   but the episode continues — succession exercises

    # Terminal rewards must DOMINATE any achievable per-step shaping accrual:
    # with ~0.09/step of positive shaping over a 600-step episode (the v1.4
    # platoon cap; the B5 tenure multiplier raises the per-step ceiling from
    # 0.06 to 0.09 at full tenure) an agent can farm ~54 by never finishing —
    # mission success has to be worth strictly more, or the policy learns to
    # stall at 100% "in position" and 0% wins (observed in practice on the
    # squad scenario before this margin was set; raised 25 → 45 for the v1.4
    # x1.5 maps, 45 → 60 for the B5 tenure ceiling).
    success_team: float = 60.0
    success_speed: float = 15.0       # x fraction of steps remaining at success
    root_done_bonus: float = 3.0      # the root transmitted a truthful root-mission
    #                                   DONE inside the completion-report grace window
    #                                   (one-shot, paid with the terminal reward — not
    #                                   farmable per-step, so terminal dominance holds)
    defeat: float = -2.0              # whole cohort wiped out

    def max_step_farm(self) -> float:
        """Upper bound on per-step reward farmable by stalling (not winning).

        Best-case stall: perfect posture compliance (0.6 x weight) at FULL
        standing-order tenure (the B5 multiplier's ceiling, 1 + tenure_factor)
        plus the leader coverage bonus, minus the time penalty. Used by tests
        to prove terminal dominance for every scenario's episode cap.
        """
        return (
            self.compliance_weight * 0.6 * (1.0 + max(0.0, self.tenure_factor))
            + self.coverage_bonus
            + self.time_penalty
        )


#: Component names, fixed so metrics stay aligned across runs.
COMPONENTS: tuple[str, ...] = ("time", "compliance", "report", "command", "combat", "terminal")


@dataclass
class RewardLedger:
    """Per-step, per-agent named reward components."""

    components: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, agent: str, component: str, value: float) -> None:
        """Accumulate ``value`` into a named component for ``agent``."""
        if value == 0.0:
            return
        bucket = self.components.setdefault(agent, dict.fromkeys(COMPONENTS, 0.0))
        bucket[component] += value

    def total(self, agent: str) -> float:
        """Scalar reward for ``agent`` this step."""
        return sum(self.components.get(agent, {}).values())

    def breakdown(self, agent: str) -> dict[str, float]:
        """Named components for ``agent`` (zeros if untouched)."""
        return dict(self.components.get(agent, dict.fromkeys(COMPONENTS, 0.0)))
