"""Reading rules the exposure probe must not break.

The probe exists to stop `human_death_rate` being quoted as a safety result for
the jammed arm when what actually fell was its denominator. Two of its own
readings could make the same class of mistake, so both are pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.jam_exposure_probe import (
    BLANK_UNDER_THIS_COMM_MODEL,
    avg,
    death_rate,
)


def test_an_unscored_episode_is_not_counted_as_a_survival():
    """`human_died is None` means not scored, not "lived".

    Dividing by every episode instead of by the scored ones deflates the death
    rate — the same denominator error the probe was written to catch, made by
    the probe itself.
    """
    eps = [{"human_died": True}, {"human_died": False}, {"human_died": None}]
    assert death_rate(eps) == 0.5, "denominator must be the SCORED episodes"


def test_a_field_no_episode_records_reads_as_absent_not_as_zero():
    """`avg` returns None for an unrecorded field, never 0.0.

    A mean of no observations is not zero, and printing 0.00 for a counter this
    comm model never touches states something false about the run.
    """
    assert avg([{"x": None}, {}], "x") is None
    assert avg([{"x": 2}, {"x": 4}], "x") == 3


def test_the_blank_counters_are_declared_rather_than_averaged():
    """Counters that `global`/`jammed` never exercise are named, not inferred.

    `orders_delivered` and `orders_lost` belong to the store-and-forward path;
    reporting "orders lost: 0.00" under a jammed net would read as a measured
    claim that the outage cost no orders, which is the opposite of true.
    """
    assert "orders_lost" in BLANK_UNDER_THIS_COMM_MODEL
    assert "orders_delivered" in BLANK_UNDER_THIS_COMM_MODEL
    # the exposure fields must NOT be in there — they are the actual evidence
    assert "human_ring_entries" not in BLANK_UNDER_THIS_COMM_MODEL
    assert "human_mean_objective_dist" not in BLANK_UNDER_THIS_COMM_MODEL
