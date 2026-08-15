#!/usr/bin/env python
"""The two exact tests this repo argues from, written once.

Every seed-count claim in ROADMAP.md is a handful of runs against a handful of
runs, which is small-n territory: the normal approximations are wrong there and
the exact tests are cheap, so the repo uses the exact ones. There is no scipy in
this venv, so they are hand-rolled — and hand-rolled exact tests are exactly the
kind of code that agrees with the right answer on the symmetric cases everyone
checks and disagrees everywhere else, which is why both are pinned against
textbook values in ``tests/test_exact_tests.py``.

Which one to use is decided by the design, not by taste:

* **Fisher** compares two INDEPENDENT arms — k of n runs report here against j of
  m runs there. It is the right test when the arms do not share seeds, and the
  wrong one when they do, because it throws the pairing away.
* **McNemar** compares two MATCHED readings of the same units — the same seed
  under two treatments, or the same seed on two scenarios. Its evidence is the
  DISCORDANT pairs only: a pair that agrees carries no information about which
  way the treatment moves things, and adding concordant pairs to a McNemar does
  not make it more powerful.

The practical consequence, and the reason this module exists next to
``scripts/design_power.py``: **at five matched pairs the smallest p McNemar can
attain is 0.0625, over every outcome including perfect separation.** A five-pair
paired design cannot reject at 0.05 no matter what it measures. Reach for
``design_power`` before a campaign, not after.
"""

from __future__ import annotations

from math import comb


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact on the 2x2 ``[[a, b], [c, d]]``.

    Two-sided by the *point-probability* convention: sum the probability of
    every table at least as extreme as the observed one, where "as extreme"
    means "no more likely under the null". Summing one tail and doubling it is
    the classic way to write this wrong — it agrees on symmetric tables and
    quietly disagrees on the asymmetric ones that carry the findings.
    """
    n = a + b + c + d
    row1, col1 = a + b, a + c
    if not n or not row1 or not col1 or row1 == n or col1 == n:
        return 1.0

    def prob(x: int) -> float:
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)

    observed = prob(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return min(1.0, sum(p for x in range(lo, hi + 1)
                        if (p := prob(x)) <= observed * (1 + 1e-9)))


def mcnemar_two_sided(one_way: int, other_way: int) -> float:
    """Exact two-sided McNemar over the two counts of DISCORDANT pairs.

    ``one_way`` and ``other_way`` are the pairs that moved in each direction;
    concordant pairs are not arguments because they are not evidence. Under the
    null a discordant pair is a fair coin, so this is the two-sided exact
    binomial on ``min(one_way, other_way)`` successes in ``one_way +
    other_way`` tosses.

    No discordant pairs at all is ``1.0`` — a design that measured nothing has
    not shown that nothing is there.
    """
    n = one_way + other_way
    if n == 0:
        return 1.0
    k = min(one_way, other_way)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
