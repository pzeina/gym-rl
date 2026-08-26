#!/usr/bin/env python
"""The small-sample statistics this repo argues from, written once.

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

``spearman_rho`` is the third and it is not an exact test — it is here because
this repo wrote "a monotone spam correlation" into ROADMAP.md off an eyeballed
sorted list of nine runs, and the list read as monotone because it had been
sorted (refs assurance #59). A rank correlation at n = 9 is a weak instrument,
so it is never quoted alone: the caller is expected to print the leave-one-out
range beside it, and a rho whose sign flips when one run is dropped is not a
relation.
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


def fisher_one_sided_less(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact on ``[[a, b], [c, d]]``: is row 1's rate LOWER?

    ``P(X <= a)`` in the lower tail of the hypergeometric with the observed
    margins. Non-inferiority is directional — the question is whether the new
    arm LOST episodes against the incumbent, not whether the two differ — so
    the two-sided test is the wrong one here and halving its p-value is not
    the right one either on asymmetric tables.

    The arithmetic this repo cares about, worked once so the bar can be written
    before the runs: against a **100/100 incumbent at N=100**, conditioning on
    the margins puts every failure equally likely in either arm, so an arm at
    ``k`` successes reads ``C(100, 100 - k) / C(200, 100 - k)`` —

        k = 99 -> 0.5000   k = 97 -> 0.1231   k = 95 -> 0.0297
        k = 98 -> 0.2487   k = 96 -> 0.0606   k = 94 -> 0.0145

    which is why "inside the incumbent's CI" cannot be the bar when that CI is
    ``1.00 +/- 0.00``: it refuses a single lost episode out of a hundred, and
    the exact test says one lost episode is a coin flip.
    """
    n = a + b + c + d
    row1, col1 = a + b, a + c
    if not n or not row1 or not col1 or row1 == n or col1 == n:
        return 1.0

    lo = max(0, col1 - (n - row1))
    return min(1.0, sum(comb(row1, x) * comb(n - row1, col1 - x)
                        for x in range(lo, a + 1)) / comb(n, col1))


def holm_reject(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni over a family of one-sided tests: which are rejected.

    A fleet guard reads one test per member, and nine tests at alpha 0.05 raise
    one false alarm better than a third of the time. That matters here in the
    direction people forget: the family is a GUARD, so a false alarm does not
    invent an effect, it wrongly convicts a cycle of having broken a scenario
    it did not break. Holm controls that at family alpha while staying uniformly
    more powerful than Bonferroni.

    Step down the sorted p-values; the first that fails ``p <= alpha / (m - i)``
    stops the procedure and everything from there on is retained.
    """
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out, still_rejecting = {}, True
    for i, (name, p) in enumerate(ordered):
        still_rejecting = still_rejecting and p <= alpha / (m - i)
        out[name] = still_rejecting
    return out


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


def _midranks(values: list[float]) -> list[float]:
    """Ranks, with tied values sharing the average of the ranks they span.

    Ties are the whole reason this is written out rather than sorted twice: the
    series that prompted it carries two runs at a report rate of exactly 0.750
    with false rates 0.348 and 0.500, and breaking that tie by input order would
    let the row ordering decide the sign of the correlation.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        shared = (start + stop) / 2 + 1
        for i in order[start:stop + 1]:
            ranks[i] = shared
        start = stop + 1
    return ranks


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Tie-corrected Spearman rank correlation; ``nan`` when either side is constant."""
    n = len(xs)
    if n != len(ys):
        raise ValueError(f"spearman_rho needs paired inputs, got {n} and {len(ys)}")
    if n < 2:
        return float("nan")
    rx, ry = _midranks(list(xs)), _midranks(list(ys))
    mx, my = sum(rx) / n, sum(ry) / n
    covariance = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    spread = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return covariance / spread if spread else float("nan")


def jackknife_rho(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """The smallest and largest rho over dropping each observation in turn.

    The honest companion to a rank correlation at single-digit n. A relation
    whose range straddles zero is carried by one point, and calling it monotone
    is a statement about that point.
    """
    n = len(xs)
    if n < 3:
        return (float("nan"), float("nan"))
    values = [spearman_rho([x for j, x in enumerate(xs) if j != i],
                           [y for j, y in enumerate(ys) if j != i]) for i in range(n)]
    return (min(values), max(values))
