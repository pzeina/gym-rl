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
    sitrep_interval: int = 25
    done_true: float = 1.0
    done_false: float = -0.5

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

    hit_enemy: float = 0.2
    kill_enemy: float = 1.0
    team_kill_share: float = 0.2      # everyone else, per enemy killed
    took_hit: float = -0.1
    death: float = -1.0
    teammate_death: float = -0.2

    # Terminal rewards must DOMINATE any achievable per-step shaping accrual:
    # with ~0.05/step of positive shaping over a 300-step episode an agent can
    # farm ~13 by never finishing — mission success has to be worth strictly
    # more, or the policy learns to stall at 100% "in position" and 0% wins
    # (observed in practice on the squad scenario before this margin was set).
    success_team: float = 25.0
    success_speed: float = 10.0       # x fraction of steps remaining at success
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
