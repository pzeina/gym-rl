"""The ablation is read on the axes it actually separates on.

The 2026-08-06 B3 result separated `full` from `flat` by 7 points of success and
by **2.2x the wipe rate**, and separated `full` from `nomask` not at all on
success while separating them completely on interpretability — 100% doctrine-
valid traffic against 33-48%, and 128 completion reports against ~0. A
replication that reads the success column alone concludes the hierarchy does
nothing, and is reading the wrong column.

So what is pinned here is the arithmetic that makes the other columns readable:

* **defeats per 100 is derived, not stored.** An episode that neither succeeded
  nor ran the clock out is a cohort that was killed. Getting that subtraction
  wrong would silently move the robustness cell — the one the original claim
  rests on.
* **overlapping intervals are not a difference**, and the report has to say so
  rather than leaving a reader to compare two point estimates.
* **the original's per-seed cells are recomputed, not transcribed** (refs #41).
  `ablation_report.ORIGINAL` quotes the three-seed original one seed at a time;
  every one of those numbers is re-derived below from the nine committed
  `runs/squad_abl_*/behavior.json` corpora, so a drift in either the constant or
  the corpus breaks a test instead of quietly restating a claim.
"""

from __future__ import annotations

import json

import pytest

from scripts import ablation_report
from scripts.baseline import run_dir


def _arm(tmp_path, run: str, *, successes: int, timeout: float = 0.0, n: int = 100, **metrics):
    d = tmp_path / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "behavior_final.json").write_text(json.dumps({
        "episodes": n,
        "success_ci95": f"{successes / n:.2f} ± 0.05",
        "metrics": {"success_rate": successes / n, "successes": successes,
                    "timeout_rate": timeout, **metrics},
    }))


@pytest.fixture
def trio(tmp_path, monkeypatch):
    monkeypatch.setattr(ablation_report, "run_dir", lambda name: tmp_path / name)
    return tmp_path


def test_defeats_are_the_episodes_that_were_neither_won_nor_timed_out(trio):
    _arm(trio, "a", successes=85, timeout=0.04)

    facts = ablation_report._facts("a")

    assert facts["defeat_per_100"] == pytest.approx(11.0)


def test_a_perfect_arm_has_no_negative_defeats(trio):
    """Floating point on 1 - 1.0 - 0.0 must not print -0.0 wipes."""
    _arm(trio, "a", successes=100, timeout=0.0)

    assert ablation_report._facts("a")["defeat_per_100"] == pytest.approx(0.0)


def test_overlapping_intervals_are_reported_as_not_a_difference(trio, capsys):
    _arm(trio, "full_v1", successes=97)
    _arm(trio, "nomask_v1", successes=98)
    _arm(trio, "flat_v1", successes=96)

    ablation_report.report(["full_v1", "nomask_v1", "flat_v1"])
    out = capsys.readouterr().out

    assert out.count("intervals OVERLAP, not a difference") == 2
    assert "Fisher p = 1.000" in out


def test_a_real_separation_is_not_called_an_overlap(trio, capsys):
    _arm(trio, "full_v1", successes=97)
    _arm(trio, "nomask_v1", successes=96)
    _arm(trio, "flat_v1", successes=55)

    ablation_report.report(["full_v1", "nomask_v1", "flat_v1"])
    out = capsys.readouterr().out

    assert "— separated" in out


def test_the_report_leads_the_reader_to_the_axes_that_separate(trio, capsys):
    for run in ("full_v1", "nomask_v1", "flat_v1"):
        _arm(trio, run, successes=95, orders_per_episode=1.0,
             doctrine_allowed_rate=1.0, done_reports=10)

    ablation_report.report(["full_v1", "nomask_v1", "flat_v1"])
    out = capsys.readouterr().out

    assert "ROBUSTNESS" in out and "INTERPRET" in out
    assert "One seed per arm" in out, "the replication must state its own strength"


def test_an_unevaluated_arm_is_named_rather_than_silently_dropped(trio, capsys):
    _arm(trio, "full_v1", successes=97)

    ablation_report.report(["full_v1", "nomask_v1", "flat_v1"])
    out = capsys.readouterr().out

    assert "not evaluated yet: nomask_v1, flat_v1" in out


def _original_corpus(arm: str, seed: int) -> dict:
    """The committed N=30 behavior corpus for one arm of the 2026-08-06 original.

    Resolved through ``run_dir`` rather than by path, because archiving these is
    a move into ``runs/archive/`` and a name that resolved before must resolve
    after.
    """
    path = run_dir(f"squad_abl_{arm}_s{seed}") / "behavior.json"
    return json.loads(path.read_text())


@pytest.mark.parametrize("arm", ["full", "nomask", "flat"])
def test_the_originals_cells_are_recomputed_from_the_committed_corpora(arm):
    """Hand-kept numbers are how every overstatement here got in.

    So the trio the report quotes is checked against the nine corpora it was read
    off, cell by cell and seed by seed.
    """
    cells = ablation_report.ORIGINAL[arm]

    for i, seed in enumerate(ablation_report.ORIGINAL_SEEDS):
        corpus = _original_corpus(arm, seed)
        metrics = corpus["metrics"]
        assert corpus["episodes"] == ablation_report.ORIGINAL_EPISODES

        successes = round(metrics["success_rate"] * corpus["episodes"])
        assert cells["successes"][i] == successes, f"{arm} s{seed} successes"
        assert cells["done"][i] == metrics["done_reports"], f"{arm} s{seed} DONE"

        preferred = metrics["doctrine_preference_rate"]
        if cells["doctrine_preferred"] is None:
            assert preferred is None, "flat has no orders to judge"
        else:
            assert cells["doctrine_preferred"][i] == pytest.approx(preferred, abs=1e-4)


def test_the_flat_arms_completion_cell_is_measured_rather_than_unavailable():
    """DONE is a report, not an order, so the flat arm can transmit it — and did.

    Recording that cell as "n/a, no orders" (which is true of the *doctrine* cell
    beside it) would delete the arm's whole point: completion reporting is the
    only C2 channel the flat arm still has, and it is still not used. 0/2/1 is
    evidence; n/a is a missing measurement.
    """
    done = ablation_report.ORIGINAL["flat"]["done"]

    assert done == (0, 2, 1)
    assert any(_original_corpus("flat", s)["metrics"]["done_reports"]
               for s in ablation_report.ORIGINAL_SEEDS), "the flat arm did transmit DONE"

    # Everything the CLI shows a reader: the prose header plus the computed block.
    shown = f"{ablation_report.__doc__}\n{ablation_report.original_block()}"
    for row in shown.splitlines():
        if "DONE" in row:
            assert "n/a" not in row, f"the completion cell is measured, not n/a: {row!r}"


def test_the_completion_cell_is_shown_per_seed_because_its_seeds_disagree():
    """128.3 is 173 and 210 and a 2, and that 2 is the flat arm's own maximum.

    A reader given only the mean would take a one-seed replication landing near 2
    as a refutation. It is one draw from a cell the original never settled.
    """
    full = ablation_report.ORIGINAL["full"]["done"]
    flat = ablation_report.ORIGINAL["flat"]["done"]

    assert min(full) <= max(flat), "the full arm's worst seed is not above the flat arm"

    block = ablation_report.original_block()
    assert all(str(v) in block for v in full), "every seed is printed, not just the mean"
    assert "DOES NOT" in block


def test_doctrine_preferred_is_the_cell_one_seed_can_settle():
    """The contrast that makes the one-seed caveat specific instead of generic.

    Every full seed is above every nomask seed, so the ordering survives any
    single draw — which is why the report tells a one-seed reader to lead with
    this row and not with DONE.
    """
    full = ablation_report.ORIGINAL["full"]["doctrine_preferred"]
    nomask = ablation_report.ORIGINAL["nomask"]["doctrine_preferred"]

    assert min(full) > max(nomask)
    assert "SEPARATES SEED BY SEED" in ablation_report.original_block()


def test_the_report_hands_the_one_seed_reader_the_per_seed_block(trio, capsys):
    for run in ("full_v1", "nomask_v1", "flat_v1"):
        _arm(trio, run, successes=95, done_reports=3, successes_announced=0,
             successes_announced_rate=0.0)

    ablation_report.report(["full_v1", "nomask_v1", "flat_v1"])
    out = capsys.readouterr().out

    assert "the 2026-08-06 original, per seed" in out
    assert "173/210/2" in out, "the caveat names the spread, not just 'one seed per arm'"
    # refs #41: the axis a monitor on the radio alone can see
    assert "as a rate of wins" in out


@pytest.mark.parametrize(("table", "expected"), [
    # the squad root-death cell at ckpt_best: 15/100 against 35/100
    ((15, 85, 35, 65), 0.001748),
    # the same arm against squad_v6's 45/100
    ((15, 85, 45, 55), 5.547e-06),
    # a pair that does NOT separate, which is the harder case to get right
    ((97, 3, 91, 9), 0.133763),
])
def test_fisher_matches_known_two_by_twos(table, expected):
    """No scipy in this venv, so the exact test is hand-rolled and pinned.

    A two-sided Fisher summing the wrong tail is the classic way to write one:
    it agrees with the right answer on symmetric tables and quietly disagrees
    everywhere else. The three cells here are asymmetric and span six orders of
    magnitude, which a tail error cannot survive.
    """
    assert ablation_report._fisher(*table) == pytest.approx(expected, rel=1e-3)


def _seeded_arm(tmp_path, run: str, *, seed: int, successes: int, n: int = 100, **metrics):
    _arm(tmp_path, run, successes=successes, n=n, **metrics)
    (tmp_path / run / "config.json").write_text(json.dumps({"seed": seed}))


def _nine(tmp_path, *, full_n=100, done_full=(173, 210, 2)) -> list[str]:
    runs = []
    for arm, done_cells in (("full", done_full), ("nomask", (0, 0, 1)), ("flat", (0, 2, 1))):
        for seed, done in zip((12, 13, 14), done_cells, strict=True):
            run = f"{arm}_s{seed}"
            n = full_n if arm == "full" else 100
            _seeded_arm(tmp_path, run, seed=seed, successes=int(0.95 * n), n=n,
                        done_reports=done)
            runs.append(run)
    return runs


def test_the_seed_report_prints_every_seed_and_lets_the_mean_follow(trio, capsys):
    """A bimodal cell averaged into one number is a hiding place (refs the

    original's 173/210/2 DONE column). The per-seed mode must show the three
    draws, with the mean after them, never instead of them.
    """
    runs = _nine(trio)

    assert ablation_report.seed_report(runs) == 0
    out = capsys.readouterr().out

    assert "173 210 2  (128)" in out
    assert "seeds 12/13/14" in out


def test_the_seed_report_pools_success_across_seeds_for_the_exact_test(trio, capsys):
    runs = _nine(trio)

    ablation_report.seed_report(runs)
    out = capsys.readouterr().out

    assert "success pooled over seeds 285/300 vs 285/300" in out
    assert out.count("intervals OVERLAP, not a difference") == 2


def test_the_seed_report_states_its_n_and_refuses_to_flatten_a_mixed_one(trio, capsys):
    """Captioning N=20 rows as N=100 is the overstatement publish_audit exists

    to catch; when the nine corpora disagree on N the header must say so."""
    runs = _nine(trio, full_n=20)

    ablation_report.seed_report(runs)
    out = capsys.readouterr().out

    assert "N varies: 20/100" in out


def test_the_seed_report_names_an_unevaluated_run_rather_than_dropping_it(trio, capsys):
    runs = _nine(trio)
    (trio / "full_s13" / "behavior_final.json").unlink()

    assert ablation_report.seed_report(runs) == 1
    out = capsys.readouterr().out

    assert "not evaluated yet: full_s13" in out
