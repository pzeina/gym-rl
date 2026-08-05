"""Rank hierarchy, command authority, and chain-of-command succession.

The hierarchy follows the French light-infantry structure the project was
built around, extended with a base rifleman rank (SLD) so that the lowest
echelon only executes, reports, and communicates — it never commands:

    CDU  Commandant d'Unité      company commander
    ADU  Adjoint d'Unité         company second-in-command (deputy of CDU)
    CDS  Chef de Section         platoon leader
    SOA  Sous-Officier Adjoint   platoon sergeant (deputy of CDS)
    CDG  Chef de Groupe          squad leader
    CAP  Chef d'Équipe           fire-team leader
    SLD  Soldat                  rifleman — executes only

Authority is a strict total order; an agent may command another only when it
has strictly higher *effective* authority and the target is one of its direct
subordinates in the org chart. Effective authority can exceed the intrinsic
rank when an agent has assumed the position of a fallen leader (succession).
"""

from __future__ import annotations

from enum import Enum


class Rank(Enum):
    """Military ranks, lowest to highest."""

    SLD = "sld"
    CAP = "cap"
    CDG = "cdg"
    SOA = "soa"
    CDS = "cds"
    ADU = "adu"
    CDU = "cdu"

    @classmethod
    def from_str(cls, value: str) -> Rank:
        """Parse a rank from a case-insensitive string; unknown → SLD."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.SLD


#: Strict authority ladder. Deputies (ADU, SOA) sit just below their principal
#: so succession is a total order with no ties.
AUTHORITY: dict[Rank, int] = {
    Rank.SLD: 0,
    Rank.CAP: 1,
    Rank.CDG: 2,
    Rank.SOA: 3,
    Rank.CDS: 4,
    Rank.ADU: 5,
    Rank.CDU: 6,
}

RANK_LABELS: dict[Rank, str] = {
    Rank.SLD: "Rifleman",
    Rank.CAP: "Fire-Team Leader",
    Rank.CDG: "Squad Leader",
    Rank.SOA: "Platoon Sergeant",
    Rank.CDS: "Platoon Leader",
    Rank.ADU: "Company XO",
    Rank.CDU: "Company Commander",
}

#: Ranks that hold command over subordinates (a decision rank owns an order
#: vocabulary; SLD does not).
COMMAND_RANKS: frozenset[Rank] = frozenset(
    {Rank.CAP, Rank.CDG, Rank.SOA, Rank.CDS, Rank.ADU, Rank.CDU}
)

#: Deputy rank → the principal rank it succeeds first.
DEPUTY_OF: dict[Rank, Rank] = {
    Rank.ADU: Rank.CDU,
    Rank.SOA: Rank.CDS,
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
