"""Pins for the read-back read's registered thresholds (docs/readback-cycle.md).

The bars were written before job 1 of the v1.27 campaign; changing one after
the results exist is writing the bar to fit the data. This test is the tamper
seal, same as test_prereg_dispersion.py is for the dispersion bar.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import readback_read


def test_the_registered_thresholds_do_not_drift():
    assert readback_read.REPORT_FLOOR == 0.5
    assert readback_read.ROOT_DEATH_BAR == 0.05
    assert readback_read.PROBE_BAR == 0.29
    assert readback_read.ALPHA == 0.05


def test_the_probe_bar_is_the_stricter_pre_cycle_measurement():
    """0.29 is v17's measured fresh-sub-DONE share, the higher of the 0.29/0.15
    pair — a candidate must beat the BEST pre-cycle draw, not the worst."""
    assert readback_read.PROBE_BAR == 0.29
