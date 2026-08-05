"""Rank structure, command authority, and chain-of-command succession.

Ranks follow NATO nomenclature (STANAG 2116 grade codes) for a light-infantry
company slice; the lowest grade only executes, reports, and communicates — it
never commands:

    CO   Company Commander    OF-2 (CPT)
    XO   Executive Officer    OF-2 (deputy of CO)
    PL   Platoon Leader       OF-1 (LT)
    PSG  Platoon Sergeant     OR-7 (SFC, deputy of PL)
    SL   Squad Leader         OR-6 (SSG)
    TL   Fire Team Leader     OR-5 (SGT)
    RFN  Rifleman             OR-3 (PFC) — executes, reports, communicates

Authority is a strict total order; an agent may command another only when it
has strictly higher *effective* authority and the target is one of its direct
subordinates in the org chart. Effective authority can exceed the intrinsic
rank when an agent has assumed the position of a fallen leader (succession).
"""

from __future__ import annotations

from enum import Enum


class Rank(Enum):
    """NATO ranks, lowest to highest."""

    RFN = "rfn"
    TL = "tl"
    SL = "sl"
    PSG = "psg"
    PL = "pl"
    XO = "xo"
    CO = "co"

    @classmethod
    def from_str(cls, value: str) -> Rank:
        """Parse a rank from a case-insensitive string; unknown → RFN."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.RFN


#: Strict authority ladder. Deputies (XO, PSG) sit just below their principal
#: so succession is a total order with no ties.
AUTHORITY: dict[Rank, int] = {
    Rank.RFN: 0,
    Rank.TL: 1,
    Rank.SL: 2,
    Rank.PSG: 3,
    Rank.PL: 4,
    Rank.XO: 5,
    Rank.CO: 6,
}

RANK_LABELS: dict[Rank, str] = {
    Rank.RFN: "Rifleman",
    Rank.TL: "Fire Team Leader",
    Rank.SL: "Squad Leader",
    Rank.PSG: "Platoon Sergeant",
    Rank.PL: "Platoon Leader",
    Rank.XO: "Executive Officer",
    Rank.CO: "Company Commander",
}

#: STANAG 2116 grade codes.
NATO_GRADES: dict[Rank, str] = {
    Rank.RFN: "OR-3",
    Rank.TL: "OR-5",
    Rank.SL: "OR-6",
    Rank.PSG: "OR-7",
    Rank.PL: "OF-1",
    Rank.XO: "OF-2",
    Rank.CO: "OF-2",
}

#: APP-6 echelon indicator drawn above a unit frame for the element the rank
#: leads: team = ∅, squad = ●, platoon = ●●●, company = |. Riflemen are
#: individuals — no echelon mark.
ECHELON_MARKS: dict[Rank, str] = {
    Rank.RFN: "",
    Rank.TL: "∅",
    Rank.SL: "●",
    Rank.PSG: "●●●",
    Rank.PL: "●●●",
    Rank.XO: "|",
    Rank.CO: "|",
}

#: Ranks that hold command over subordinates (a decision rank owns an order
#: vocabulary; RFN does not).
COMMAND_RANKS: frozenset[Rank] = frozenset(
    {Rank.TL, Rank.SL, Rank.PSG, Rank.PL, Rank.XO, Rank.CO}
)

#: Deputy rank → the principal rank it succeeds first.
DEPUTY_OF: dict[Rank, Rank] = {
    Rank.XO: Rank.CO,
    Rank.PSG: Rank.PL,
}


def authority(rank: Rank) -> int:
    """Return the intrinsic authority level of a rank."""
    return AUTHORITY[rank]


def is_command_rank(rank: Rank) -> bool:
    """True if the rank is allowed to issue orders at all."""
    return rank in COMMAND_RANKS


def outranks(a: Rank, b: Rank) -> bool:
    """True if rank ``a`` sits strictly above rank ``b``."""
    return AUTHORITY[a] > AUTHORITY[b]
