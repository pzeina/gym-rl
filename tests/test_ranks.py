"""Rank hierarchy and authority invariants."""

from cohort.core.ranks import AUTHORITY, COMMAND_RANKS, DEPUTY_OF, Rank, authority, is_command_rank, outranks


def test_authority_is_strict_total_order():
    values = list(AUTHORITY.values())
    assert len(values) == len(set(values)), "authority levels must be unique (no ties)"
    assert set(AUTHORITY) == set(Rank)


def test_chain_of_command_ordering():
    assert outranks(Rank.CO, Rank.XO)
    assert outranks(Rank.XO, Rank.PL)
    assert outranks(Rank.PL, Rank.PSG)
    assert outranks(Rank.PSG, Rank.SL)
    assert outranks(Rank.SL, Rank.TL)
    assert outranks(Rank.TL, Rank.RFN)
    assert not outranks(Rank.RFN, Rank.TL)
    assert not outranks(Rank.TL, Rank.TL)


def test_rifleman_is_not_a_command_rank():
    assert not is_command_rank(Rank.RFN)
    assert authority(Rank.RFN) == 0
    for rank in Rank:
        if rank is not Rank.RFN:
            assert rank in COMMAND_RANKS


def test_deputies_succeed_their_principal():
    assert DEPUTY_OF[Rank.XO] is Rank.CO
    assert DEPUTY_OF[Rank.PSG] is Rank.PL


def test_rank_from_str():
    assert Rank.from_str("sl") is Rank.SL
    assert Rank.from_str("PL") is Rank.PL
    assert Rank.from_str("nonsense") is Rank.RFN
