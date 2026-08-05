from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from mili_env.envs.classes.types_common import (
    AgentRole,
    CommunicationMessage,
    MessageType,
    SoldierElementaryAct,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

"""C2 (Command & Control) order processing mixin for Robot agents.

This module isolates all military order / mission assignment logic from the
physical robot behavior (movement, sensing, comms). Import and mix into
`RobotBase` to keep responsibilities separated.

Public surface:
* C2OrdersMixin.issue_orders(orders_vector, context=...)
* C2OrdersMixin.C2Context dataclass (agents list, doctrine, sentinel, step)

Expected attributes on the concrete robot class (RobotBase):
* agent_id: int
* role: AgentRole
* team_id: int
* current_elementary_act: SoldierElementaryAct | None
* last_mission_change_step: int
* free_order_credit: bool
* message_outbox: list[CommunicationMessage]
* (Optionally) last_derived_missions: dict[int, SoldierElementaryAct]

The environment should build a C2Context each step and pass it along with the
orders vector for each agent that set ``give_order``.
"""


logger = logging.getLogger(__name__)


class C2OrdersMixin:
    """Mixin encapsulating decentralized C2 order logic.

    Separating this from `RobotBase` allows physical simulation and C2 authority
    evolution independently. All state mutations still happen on the concrete
    robot instance (self refers to RobotBase when mixed in).
    """

    # Stability window retained for telemetry/reference but no explicit penalties are applied.
    DEFAULT_STABILITY_WINDOW: ClassVar[int] = 20

    @staticmethod
    def _role_rank(role: AgentRole) -> int:
        return {
            AgentRole.CDU: 4,
            AgentRole.ADU: 4,
            AgentRole.CDS: 3,
            AgentRole.SOA: 3,
            AgentRole.CDG: 2,
            AgentRole.CAP: 1,
        }.get(role, 0)

    @dataclass
    class C2Context:
        """Immutable context for a single C2 processing step."""

        agents: Sequence  # sequence of RobotBase-like objects
        doctrine: dict
        mission_no_change_sentinel: int
        current_step: int
        stability_window: int = 20  # default stability window steps

    @dataclass
    class _StabilityCfg:
        current_step: int
        stability_window: int
        free_credit: bool

    # ---------------- Internal helpers ---------------- #
    def _doctrine_allowed(
        self, superior_mission: SoldierElementaryAct, doctrine: dict
    ) -> tuple[SoldierElementaryAct, ...]:
        allowed = doctrine.get(superior_mission)
        if not allowed:
            return (superior_mission,)
        return allowed

    def _order_change(self, prev: SoldierElementaryAct | None, new: SoldierElementaryAct) -> int:
        """Return 1 if mission actually changed, else 0.

        Explicit penalties are removed; we only count changes for RL reward shaping.
        """
        try:
            return int(prev != new)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0

    def _assign_self(self, mission: SoldierElementaryAct, cfg: _StabilityCfg) -> int:
        prev = getattr(self, "current_elementary_act", None)
        self.current_elementary_act = mission  # type: ignore[attr-defined]
        change = self._order_change(prev, mission)
        self.last_mission_change_step = cfg.current_step  # type: ignore[attr-defined]
        return change

    def _assign_subordinate(
        self,
        mission: SoldierElementaryAct,
        *,
        target_id: int,
        issuer_rank: int,
        context: C2Context,
        cfg: _StabilityCfg,
    ) -> int:
        subordinate = context.agents[target_id]
        if subordinate.team_id != self.team_id:  # type: ignore[attr-defined]
            return 0
        if self._role_rank(subordinate.role) >= issuer_rank:  # type: ignore[attr-defined]
            return 0
        if getattr(self, "current_elementary_act", None) is not None:
            allowed = self._doctrine_allowed(self.current_elementary_act, context.doctrine)  # type: ignore[attr-defined]
            if mission not in allowed:
                mission = allowed[0]
        prev_sub = getattr(subordinate, "current_elementary_act", None)  # type: ignore[attr-defined]
        subordinate.current_elementary_act = mission  # type: ignore[attr-defined]
        change = self._order_change(prev_sub, mission)
        subordinate.last_mission_change_step = cfg.current_step  # type: ignore[attr-defined]
        subordinate.free_order_credit = True  # type: ignore[attr-defined]
        # Track issuer's last derived mission for telemetry/rewards
        try:
            last_derived = getattr(self, "last_derived_missions", {})
            last_derived[target_id] = mission
            self.last_derived_missions = last_derived  # type: ignore[attr-defined]
        except (AttributeError, TypeError) as exc:  # pragma: no cover - defensive
            logger.debug("Failed to track last_derived_missions: %s", exc)
        # Emit mission assignment message (best-effort)
        try:
            self.message_outbox.append(  # type: ignore[attr-defined]
                CommunicationMessage(
                    sender_id=self.agent_id,  # type: ignore[attr-defined]
                    receiver_id=subordinate.agent_id,  # type: ignore[attr-defined]
                    message_type=MessageType.MISSION_ASSIGN,
                    timestamp=float(cfg.current_step),
                    content={
                        "mission": subordinate.current_elementary_act.name.lower(),  # type: ignore[attr-defined]
                        "sender_role": self.role.value,  # type: ignore[attr-defined]
                    },
                    priority=4,
                )
            )
        except (AttributeError, ValueError, TypeError):  # pragma: no cover - defensive
            logger.debug("Failed to append mission message", exc_info=True)
        return change

    # ---------------- Public API ---------------- #
    def issue_orders(self, orders_vector: list[int] | tuple[int, ...], *, context: C2Context) -> int:
        """Apply a GiveOrder vector using provided context. Returns change count.

        orders_vector: sequence indexed by agent_id with either mission index or sentinel.
        """
        issuer_rank = self._role_rank(self.role)  # type: ignore[attr-defined]
        if issuer_rank <= 0:
            return 0
        cfg = self._StabilityCfg(
            current_step=context.current_step,
            stability_window=context.stability_window,
            free_credit=bool(getattr(self, "free_order_credit", False)),
        )
        if cfg.free_credit:
            self.free_order_credit = False  # type: ignore[attr-defined]
        changes = 0
        missions = list(SoldierElementaryAct)
        for target_id, mission_val in enumerate(orders_vector):
            if mission_val == context.mission_no_change_sentinel:
                continue
            if not (0 <= target_id < len(context.agents)) or not (0 <= mission_val < len(missions)):
                continue
            mission = missions[mission_val]
            if target_id == self.agent_id:  # type: ignore[attr-defined]
                changes += self._assign_self(mission, cfg)
            else:
                changes += self._assign_subordinate(
                    mission, target_id=target_id, issuer_rank=issuer_rank, context=context, cfg=cfg
                )
        return changes
