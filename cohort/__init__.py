"""cohort — a transparent chain-of-command multi-agent RL environment.

A military cohort of NATO-ranked agents (CO > XO > PL > PSG > SL > TL > RFN)
learns rank-appropriate behavior: obey orders, report up the chain, derive
doctrine-valid orders for subordinates, and fight as a team. Every order and
report is a human-readable radio message in NATO voice procedure, so a human
commander can read the full command flow — and inject orders in the same
language.
"""

from cohort.config import SCENARIOS, ScenarioSpec, get_scenario
from cohort.env.cohort_env import CohortEnv, make_env

__version__ = "1.0.0"

__all__ = ["SCENARIOS", "CohortEnv", "ScenarioSpec", "get_scenario", "make_env"]
