"""Pins for the interrogative read's registered thresholds
(docs/interrogative-cycle.md §"The read, pre-registered before job 1").

The bars were written before job 1 of the v1.28 campaign; changing one after
the results exist is writing the bar to fit the data. This test is the
tamper seal, same as test_readback_read.py is for the read-back read.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import interrogative_read


def test_the_registered_thresholds_do_not_drift():
    assert interrogative_read.USAGE_BAR == 1.0
    assert interrogative_read.EVIDENCE_BAR == 0.50
    assert interrogative_read.REPORT_FLOOR == 0.5
    assert interrogative_read.ROOT_DEATH_BAR == 0.05
    assert interrogative_read.FORMATION_MIN_SEEDS == 1
    assert interrogative_read.FALSE_COMPLETE_BAR == 0.35
    assert interrogative_read.ALPHA == 0.05


def test_the_repair_bar_is_the_readback_reads_unchanged():
    """Check 3 is 'the original repair bar, unchanged from the read-back
    cycle' — the two scorers must never drift apart on it."""
    from scripts import readback_read

    assert interrogative_read.REPORT_FLOOR == readback_read.REPORT_FLOOR
    assert interrogative_read.ROOT_DEATH_BAR == readback_read.ROOT_DEATH_BAR


def test_the_formation_scenarios_and_seize_family_are_the_registered_ones():
    assert interrogative_read.FORMATION_SCENARIOS == ("patrol_brique", "platoon")
    assert interrogative_read.SEIZE_SCENARIOS == (
        "fireteam", "squad", "patrol_brique", "platoon", "platoon_hard",
    )


def test_the_fleet_guard_incumbents_are_the_v127_read_candidates():
    """Check 6 guards against the runs the READ-BACK verdict was scored on —
    named explicitly, never runs/BASELINE.json: the v1.27 seal is still the
    owner's open decision and must not move this read when it lands."""
    assert interrogative_read.V127_READ_CANDIDATES == {
        "fireteam": "fireteam_v21_seed13",
        "squad": "squad_v38_seed12",
        "patrol_brique": "patrol_brique_v54_seed12",
        "platoon": "platoon_v24_seed13",
        "platoon_hard": "platoon_hard_v17_seed13",
        "defend_brique": "defend_brique_v22",
        "fireteam_defend": "fireteam_defend_v28",
        "squad_recon": "squad_recon_v16",
        "squad_screen": "squad_screen_v22",
    }
