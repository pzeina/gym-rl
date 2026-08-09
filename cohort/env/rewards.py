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

from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace
from difflib import get_close_matches


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
    contact_redundant: float = -0.25  # v1.11: -0.02 → -0.25. Pure noise: every reported
    #                                   enemy already fresh on the picture. At -0.02 a
    #                                   redundant report cost -0.03 all-in against an
    #                                   informative one's +0.49, so spamming stayed
    #                                   profitable down to a precision of 5.8% — no
    #                                   defence at all, and fireteam_defend_v8 duly sat
    #                                   at 0.38 (N=100: 289 informative, 480 noise).
    #                                   Break-even is now p > 0.35.
    #                                   The B5 "over-pricing suppresses the honest act"
    #                                   hazard is much weaker here than it was for
    #                                   done_false: avoiding a duplicate needs no
    #                                   inference about hidden state, only the memory
    #                                   that you just sent it. scripts/contact_probe.py
    #                                   measured 91% of redundant reports at age 0-1 and
    #                                   NONE past half the cliff — duplicate storms, not
    #                                   near-miss refreshes — which is also why this is a
    #                                   flat price and not an age-decayed one.
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
    done_false: float = -0.5          # v1.10 took this -0.5 → -2.0 to move the
    #                                   claiming break-even from p > 0.33 to
    #                                   p > 0.67. REVERTED: the B5 hazard the
    #                                   v1.10 note called "deliberately moderate"
    #                                   is exactly what happened. Under -2.0 the
    #                                   final-decile false-DONE rate fell to ~0 in
    #                                   squad (0.010), squad_recon (0.000),
    #                                   squad_screen (0.005) and fireteam (0.078)
    #                                   — not precision, silence — and the two
    #                                   report-centric scenarios lost their
    #                                   terminal income entirely: squad_recon_v6
    #                                   and squad_screen_v4 both ended at 0%
    #                                   success with terminal 0.0000, episodes
    #                                   pinned at max_steps and tx/agent-step down
    #                                   to 0.058/0.029, riding out the clock on
    #                                   posture compliance. Their predecessors
    #                                   squad_recon_v5b and squad_screen_v3 both
    #                                   converged, claiming DONE at 0.767/0.830.
    #                                   The structural reason the price cannot be
    #                                   paid: RECON/SCREEN completion is
    #                                   team-adjudicated (_team_observe_steps vs
    #                                   TEAM_OBSERVE_STEPS) and that counter is in
    #                                   no observation slot, so p is not estimable
    #                                   from what the agent can see. Raise this
    #                                   again only once mission-completion
    #                                   progress is IN the observation. The spam
    #                                   half stays structural: ScenarioSpec
    #                                   .done_cooldown (8) is unchanged.

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

    # Preparation period (v1.10): while the assault is still forming up
    # (ScenarioSpec.assault_h_hour), an agent standing IN COVER within
    # IN_POSITION_RADIUS of the root objective earns this per step. The prep
    # phase grants the TIME to occupy a prepared position; this grants the
    # MOTIVE. Without it the contact-free phase is a null period a policy can
    # idle through and still meet the assault in the open — exactly the v7
    # failure (cover occupancy 0.05, the fight 9.7 cells off the objective).
    #
    # It cannot be farmed: it is paid only before H, so its lifetime ceiling is
    # prep_in_position x max(assault_h_hour) = 0.05 x 75 = 3.75 per agent,
    # bounded like observe_progress and accounted for in the terminal-dominance
    # test. Cover is required, not merely proximity — sitting on bare ground at
    # the objective is not a prepared position (the v1.2 terrain lesson).
    prep_in_position: float = 0.05

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
    # mission position, OR when the target itself stands inside the position's
    # engagement envelope (anchor distance <= IN_POSITION_RADIUS +
    # weapon_range) — fire against an enemy assaulting the position is the
    # mission wherever the melee pushed the defender (v1.9 defend diagnosis:
    # the human TL fired on 0.5% of threatened opportunities because off its
    # 3.5-cell disc its fire earned nothing); RECON (may engage) / SEIZE /
    # CLEAR / RALLY / untasked → unchanged. Teammate kill-shares are NOT
    # scaled (the shooter's incentive is the lever).
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
    human_death: float = 0.0          # paid by EVERY present agent when a human dies,
    #                                   on top of the normal death penalties. DISABLED
    #                                   by default since v1.10: at its former -25 this
    #                                   term delivered a correlated -25 x n_agents shock
    #                                   in a single step (-100 on a fireteam), and every
    #                                   D4 collapse onset measured to date coincides with
    #                                   a human-death burst (value_loss 15-95 through
    #                                   fireteam_defend_v7) — the suspected destabiliser
    #                                   of the value function, for a preservation
    #                                   preference the rank-weighted teammate_death
    #                                   already expresses in kind. The mechanism is kept
    #                                   (set it negative to restore the shock); human
    #                                   preservation is now measured, not priced, via the
    #                                   human_death_rate / exposure metrics.

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

    # Defend terminal, scaled by the force that held it (v1.12, owner's option
    # 4). On DEFEND/DENY root missions ONLY, the team terminal is multiplied by
    #   (1 - scale) + scale x surviving_weight / starting_weight
    # so a cohort that holds intact is paid in full and one that holds with
    # half its rank-weighted strength gone is not. 0 → off (flat terminal).
    #
    # WHY THIS AND NOT FORFEITURE. v1.11 paid the fallen nothing, which fixed
    # nothing and caused D4: the individual gain from hanging back (P(die)
    # 0.129 -> 0.022, +6.4) is visible to a per-agent advantage while the
    # collective cost (success 1.00 -> 0.00, -52.3) is not, so one shared
    # policy defected everywhere at once. `d44ee8d` paid everyone and fixed
    # that — and left defend scenarios with bodies that cost 1.0 apiece and
    # nothing to buy, so `defend_brique_v4` fought 6.09 cells off an objective
    # it was there to hold and failed the regression gate.
    #
    # A survivor-scaled terminal restores the preservation pressure WITHOUT
    # restoring the asymmetry, because it is paid to the fallen too: every
    # agent, living or dead, sees the same multiplier, so a death is a shared
    # loss rather than a private one. The residual private gain from hanging
    # back is the ~1/n of the multiplier your own body accounts for — on a
    # fireteam that is 60/4 x 0.35 ≈ 5.3 at most against the same -52.3, an
    # order of magnitude short of the D4 arithmetic instead of comparable to it.
    #
    # 0.35 IS SET BY THE DOMINANCE INVARIANT, NOT BY TASTE. The multiplier can
    # only reduce the terminal, so `win_beats_stall` must clear 2x at the FLOOR
    # (whole force lost). fireteam_defend sits at 3.42 undiminished, so the
    # largest admissible scale is 1 - 2.0/3.42 = 0.415; 0.35 leaves 2.22.
    # Raising this above ~0.41 re-creates the stall basin on fireteam_defend
    # and test_defend_terminal_scaling_preserves_dominance will say so.
    defend_survivor_scale: float = 0.35
    root_done_bonus: float = 3.0      # the root transmitted a truthful root-mission
    #                                   DONE inside the completion-report grace window
    #                                   (one-shot, paid with the terminal reward — not
    #                                   farmable per-step, so terminal dominance holds)

    # v1.15: the bonus is on the table for the episode's FIRST root claim only,
    # and **the first claim consumes it whether or not it is accepted** — a
    # rejected first claim burns the bonus for that episode. That second half is
    # the whole mechanism. If a rejection left the slot open, the exploit below
    # would survive untouched: spam until one lands, collect on "the first
    # ACCEPTED claim".
    #
    # WHAT IT PRICES. v1.14 reopened MISSION COMPLETE to horizon-DEFEND roots
    # and defend_brique_v11/ckpt_latest immediately filed 321 root claims in 100
    # episodes, 227 of them rejected — a false-complete rate of 0.71. It is not
    # irrationality, it is the arithmetic: an accepted claim pays done_true 1.0 +
    # root_done_bonus 3.0 = 4.0 against a rejected one's done_false -0.5, so
    # probing breaks even at p > 1/9 = 0.111 and pays +262 per 100 episodes at
    # v11's realised acceptance of 94/321 = 0.293. ScenarioSpec.done_cooldown (8)
    # rate-limits the retries but never changes their SIGN — the same shape as
    # the pre-v1.10 re-roll exploit the cooldown was built against.
    #
    # WITH THE SLOT SPENT a further claim is worth done_true on success and
    # done_false on rejection, so break-even moves to p > 0.333 and continued
    # probing at 0.293 turns NEGATIVE (-0.07 a claim, transmission_cost
    # included). The first claim keeps its old economics on purpose: this must
    # not become another price that buys silence (the done_false=-2.0 lesson
    # above — precision that is really muteness, and two report-centric
    # scenarios losing their terminal income entirely). A policy that simply
    # stops claiming has not passed; it has lost the channel v1.14 reopened.
    #
    # Fleet-wide, not defend-scoped: root_done_bonus pays every completable root,
    # and the probing predates the horizon work — fireteam_v8 filed 87 claims at
    # 0.908 false, defend_brique_v6 filed 442. The incentive is general.
    #
    # Per EPISODE, not per root agent: a successor that inherits the root after a
    # casualty inherits a spent slot too. The opportunity belongs to the
    # operation, not to whoever happens to be holding it.
    #
    # False → the pre-v1.15 rule exactly (every accepted root claim can pay).
    root_done_bonus_first_claim_only: bool = True

    defeat: float = -2.0              # whole cohort wiped out

    # ------------------------------------------------------------------ #
    # CLI overrides
    # ------------------------------------------------------------------ #

    @classmethod
    def from_overrides(cls, items: Sequence[str]) -> RewardConfig:
        """Build a config from ``KEY=VALUE`` strings, defaults for the rest.

        Reward weights were code-only defaults until v1.12, which meant an
        experiment about a PRICE could not be run without editing the tree —
        and editing the tree mid-campaign is how ``fireteam_defend_v10`` died.
        It also blocked the one run that would separate ``d44ee8d`` from the
        ``done_false`` revert on the five confounded arms.

        Every field is settable, typed off the dataclass, so this does not go
        stale when a weight is added. Unknown keys raise with the near misses
        listed — a silently-ignored typo would produce a run whose
        ``economics.json`` says one thing and whose policy learned another.
        """
        types = {f.name: f.type for f in fields(cls)}
        parsed: dict[str, float | int | bool] = {}
        for item in items:
            key, sep, raw = item.partition("=")
            key, raw = key.strip(), raw.strip()
            if not sep or not key:
                msg = f"--reward expects KEY=VALUE, got {item!r}"
                raise ValueError(msg)
            if key not in types:
                # close_matches catches the transposition (`done_flase`), which
                # a substring check does not — and a typo that reaches training
                # is a run whose economics.json disagrees with its policy
                near = get_close_matches(key, types, n=3, cutoff=0.6)
                near += [k for k in types if key in k or k in key if k not in near]
                hint = f" Did you mean: {', '.join(near)}?" if near else ""
                msg = f"unknown reward key {key!r}.{hint} Valid keys: {', '.join(sorted(types))}"
                raise ValueError(msg)
            parsed[key] = _coerce(key, raw, str(types[key]))
        return replace(cls(), **parsed)

    def max_step_farm(self) -> float:
        """Upper bound on per-step reward farmable by stalling (not winning).

        Best-case stall: perfect posture compliance (``POSTURE_HOLD`` x weight)
        at FULL standing-order tenure (the B5 multiplier's ceiling,
        1 + tenure_factor) plus the leader coverage bonus, minus the time
        penalty.

        This is the UNDISCOUNTED rate. On its own it is not a sufficient
        dominance test and must not be used as one again — see
        :meth:`win_beats_stall`.
        """
        from cohort.core.missions import POSTURE_HOLD

        return (
            self.compliance_weight * POSTURE_HOLD * (1.0 + max(0.0, self.tenure_factor))
            + self.coverage_bonus
            + self.time_penalty
        )

    # ------------------------------------------------------------------ #
    # Terminal dominance, as the optimizer sees it
    # ------------------------------------------------------------------ #
    #
    # The v1.0 invariant was ``success_team > max_step_farm() x max_steps``:
    # an UNDISCOUNTED comparison. PPO maximizes the DISCOUNTED return, and at
    # the shipped gamma of 0.99 the two disagree badly — 1/(1-gamma) is a
    # 100-step planning horizon against episodes of 300-600, so gamma^600 =
    # 0.0024 and platoon's terminal reward was worth 4.52 against a stall's
    # 8.98. The undiscounted test passed the whole time (60 > 54).
    #
    # 30% of all runs to date (21/69) ended >= 25 points below their own peak,
    # and scoring every run's observed reward stream this way separated the
    # collapsed from the converged 8 times out of 8. So the invariant is now
    # stated in the units that decide the outcome.

    def stall_value(self, gamma: float, max_steps: int) -> float:
        """Discounted value of farming shaping for a whole episode."""
        rate = self.max_step_farm()
        if gamma >= 1.0:
            return rate * max_steps
        return rate * (1.0 - gamma**max_steps) / (1.0 - gamma)

    def survivor_multiplier(self, surviving_weight: float, starting_weight: float) -> float:
        """Terminal multiplier for a defend force with this much strength left.

        Weights are the caller's business (the env uses rank-weighted bodies);
        this only has to be monotone in the ratio and equal to 1.0 when the
        force is intact. Clamped, so a caller that hands over a ratio above 1
        (succession inflating the numerator, say) cannot mint terminal reward.
        """
        if self.defend_survivor_scale <= 0.0 or starting_weight <= 0.0:
            return 1.0
        held = max(0.0, min(1.0, surviving_weight / starting_weight))
        return (1.0 - self.defend_survivor_scale) + self.defend_survivor_scale * held

    def terminal_scale_floor(self, root_mission=None) -> float:
        """Smallest multiplier the terminal can take on this root mission.

        The dominance invariant has to be checked against the WORST case a
        scenario can produce, not the nominal payout: a defend policy that
        holds but is wiped doing it collects ``success_team x this``, and if
        THAT is not worth more than stalling then stalling is the better play
        in exactly the situation the scenario is about.
        """
        from cohort.core.missions import MissionType

        if root_mission in (MissionType.DEFEND, MissionType.DENY):
            return self.survivor_multiplier(0.0, 1.0)
        return 1.0

    def win_value(
        self, gamma: float, max_steps: int, win_step: int, terminal_scale: float = 1.0
    ) -> float:
        """Discounted value of winning at ``win_step``, speed bonus included."""
        undiscounted = self.success_team + self.success_speed * max(
            0.0, 1.0 - win_step / max_steps
        )
        return undiscounted * terminal_scale * gamma**win_step

    def win_beats_stall(
        self,
        gamma: float,
        max_steps: int,
        win_fraction: float = 0.45,
        terminal_scale: float = 1.0,
    ) -> float:
        """Ratio of discounted win to discounted stall; > 1 means winning pays.

        ``win_fraction`` is when in the episode the win lands, as a fraction of
        the cap — 0.45 is roughly what the converged fleet actually achieves.
        The invariant tests require a margin of 2x, not a bare 1x: at 1.0 the
        two are equal and which basin a run falls into is left to noise, which
        is exactly what squad (ratio 1.00) and platoon (0.50) did.
        """
        stall = self.stall_value(gamma, max_steps)
        if stall <= 0.0:
            return float("inf")
        win = self.win_value(gamma, max_steps, int(max_steps * win_fraction), terminal_scale)
        return win / stall


#: Accepted spellings for a boolean reward flag on the CLI.
_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off"})


def _coerce(key: str, raw: str, annotation: str) -> float | int | bool:
    """Parse ``raw`` into the type ``key`` is annotated with.

    ``bool`` is handled by name, never by truthiness: ``bool("false")`` is
    ``True``, so a coerce-by-constructor would read ``--reward
    fire_discipline=false`` as ON and quietly train the opposite experiment.
    """
    if annotation == "bool":
        if raw.lower() in _TRUE:
            return True
        if raw.lower() in _FALSE:
            return False
        msg = f"reward {key}: expected a boolean, got {raw!r}"
        raise ValueError(msg)
    try:
        return int(raw) if annotation == "int" else float(raw)
    except ValueError as exc:
        msg = f"reward {key}: expected {annotation}, got {raw!r}"
        raise ValueError(msg) from exc


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
