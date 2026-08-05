"""Rank hierarchy and authority invariants."""

from cohort.core.ranks import AUTHORITY, COMMAND_RANKS, DEPUTY_OF, Rank, authority, is_command_rank, outranks


def test_authority_is_strict_total_order():
    values = list(AUTHORITY.values())
    assert len(values) == len(set(values)), "authority levels must be unique (no ties)"
    assert set(AUTHORITY) == set(Rank)


def test_chain_of_command_ordering():
    assert outranks(Rank.CDU, Rank.ADU)
    assert outranks(Rank.ADU, Rank.CDS)
    assert outranks(Rank.CDS, Rank.SOA)
    assert outranks(Rank.SOA, Rank.CDG)
    assert outranks(Rank.CDG, Rank.CAP)
    assert outranks(Rank.CAP, Rank.SLD)
    assert not outranks(Rank.SLD, Rank.CAP)
    assert not outranks(Rank.CAP, Rank.CAP)


def test_sld_is_not_a_command_rank():
    assert not is_command_rank(Rank.SLD)
    assert authority(Rank.SLD) == 0
    for rank in Rank:
        if rank is not Rank.SLD:
            assert rank in COMMAND_RANKS


def test_deputies_succeed_their_principal():
    assert DEPUTY_OF[Rank.ADU] is Rank.CDU
    assert DEPUTY_OF[Rank.SOA] is Rank.CDS


def test_rank_from_str():
    assert Rank.from_str("cdg") is Rank.CDG
    assert Rank.from_str("CDS") is Rank.CDS
    assert Rank.from_str("nonsense") is Rank.SLD
