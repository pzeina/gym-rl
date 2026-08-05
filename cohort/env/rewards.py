"""Reward weights and per-agent component bookkeeping.

The reward teaches each rank its trade:

* everyone     — execute the standing order (compliance shaping), stay alive
* subordinates — report contacts (only *new* intel pays), send timely
                 SITREPs, report MISSION COMPLETE truthfully
* leaders      — derive doctrine-preferred orders, keep every living
                 subordinate tasked (coverage), don't spam re-orders (churn)
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

    contact_new: float = 0.5          # first report of an enemy the team didn't know
    contact_redundant: float = -0.02
    sitrep_fresh: float = 0.05        # sitrep after >= sitrep_interval quiet steps
    sitrep_spam: float = -0.02
    sitrep_interval: int = 25         # freshness gap; ScenarioSpec.sitrep_cadence
    #                                   overrides it when the reporting doctrine is on
    sitrep_overdue: float = -0.02     # per step out of contact past the mandated
    #                                   cadence without a SITREP (doctrine only)
    done_true: float = 1.0
    done_false: float = -0.5

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
    order_churn: float = -0.1         # reissue / premature re-tasking
    order_stability_window: int = 10  # steps a standing order should stand
    coverage_bonus: float = 0.01      # all living subordinates tasked
    coverage_gap: float = -0.02       # some living subordinate left untasked

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
    human_death: float = -25.0        # paid by EVERY present agent when a human dies
    #                                   (on top of the normal death penalties): losing
    #                                   the human commander is close to mission failure,
    #                                   but the episode continues — succession exercises

    # Terminal rewards must DOMINATE any achievable per-step shaping accrual:
    # with ~0.06/step of positive shaping over a 600-step episode (the v1.4
    # platoon cap) an agent can farm ~36 by never finishing — mission success
    # has to be worth strictly more, or the policy learns to stall at 100%
    # "in position" and 0% wins (observed in practice on the squad scenario
    # before this margin was set; raised 25 → 45 for the v1.4 x1.5 maps).
    success_team: float = 45.0
    success_speed: float = 15.0       # x fraction of steps remaining at success
    root_done_bonus: float = 3.0      # the root transmitted a truthful root-mission
    #                                   DONE inside the completion-report grace window
    #                                   (one-shot, paid with the terminal reward — not
    #                                   farmable per-step, so terminal dominance holds)
    defeat: float = -2.0              # whole cohort wiped out

    def max_step_farm(self) -> float:
        """Upper bound on per-step reward farmable by stalling (not winning).

        Best-case stall: perfect posture compliance (0.6 x weight) plus the
        leader coverage bonus, minus the time penalty. Used by tests to prove
        terminal dominance for every scenario's episode cap.
        """
        return self.compliance_weight * 0.6 + self.coverage_bonus + self.time_penalty


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
