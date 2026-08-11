"""A level the whole family shows is a property of the family (refs #36).

`squad_v8`'s human-death rate of 0.23 is the highest in the published fleet and
no gate covers it — both true, and both printed. Against its own lineage the
same number is the lowest a squad champion has posted in the current
observation space: 0.45, then 0.35, then 0.23. Read without the series it
reads as a regression; read with it, it is the recovery.

`program_board.py` grew `_family` for exactly this after the board was caught
overstating (refs #24). Nothing gave the README the same view, so
`publish_audit.py --series` does: one metric, every generation, both
checkpoints, straight off the committed artifacts.
"""

import json

from scripts import publish_audit


def _run(root, name, scenario, *, best=None, final=None, episodes=100):
    """A run directory carrying whichever behavior artifacts it was given."""
    d = root / name
    d.mkdir()
    for value, filename in ((best, "behavior.json"), (final, "behavior_final.json")):
        if value is None:
            continue
        (d / filename).write_text(json.dumps({
            "scenario": scenario,
            "episodes": episodes,
            "metrics": {"human_death_rate": value},
        }))
    return d


def test_series_prints_every_generation_at_both_checkpoints(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(publish_audit, "RUNS", tmp_path)
    _run(tmp_path, "squad_v6", "squad", best=0.45)
    _run(tmp_path, "squad_v7", "squad", best=0.35, final=0.35)
    _run(tmp_path, "squad_v8", "squad", best=0.15, final=0.23)

    assert publish_audit.series("human_death_rate") == 0
    out = capsys.readouterr().out

    assert "0.450" in out and "0.350" in out and "0.230" in out
    # the row the README publishes is one of three, not a solitary level
    assert out.index("squad_v6") < out.index("squad_v7") < out.index("squad_v8")
    assert "N=100" in out


def test_a_missing_final_prints_as_absent_rather_than_borrowing_the_best(
    tmp_path, monkeypatch, capsys
):
    """The failure mode of the whole audit: `squad_v6` has no committed final
    evaluation, so quoting one is quoting `ckpt_best` under another name."""
    monkeypatch.setattr(publish_audit, "RUNS", tmp_path)
    _run(tmp_path, "squad_v6", "squad", best=0.45)

    publish_audit.series("human_death_rate")
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "squad_v6" in ln)

    assert "best" in line and "0.450" in line
    assert line.rstrip().endswith("—"), "a missing final must read as missing"


def test_series_keeps_the_families_apart(tmp_path, monkeypatch, capsys):
    """Grouped by the scenario the artifact names, not by the run's prefix:
    `squad_screen` is a different family from `squad` however the runs are
    named."""
    monkeypatch.setattr(publish_audit, "RUNS", tmp_path)
    _run(tmp_path, "squad_v8", "squad", best=0.15, final=0.23)
    _run(tmp_path, "squad_screen_fallen_v1", "squad_screen", best=0.03, final=0.07)

    assert publish_audit.series("human_death_rate", "squad") == 0
    out = capsys.readouterr().out
    assert "squad_v8" in out and "squad_screen_fallen_v1" not in out

    assert publish_audit.series("human_death_rate") == 0
    out = capsys.readouterr().out
    assert "  squad\n" in out and "  squad_screen\n" in out


def test_a_metric_no_artifact_carries_says_so_and_fails(tmp_path, monkeypatch, capsys):
    """An evaluation that predates a metric must not render as a run with no
    problem on it — the absence is reported, and the exit code is non-zero."""
    monkeypatch.setattr(publish_audit, "RUNS", tmp_path)
    _run(tmp_path, "squad_v8", "squad", best=0.15, final=0.23)

    assert publish_audit.series("successes_announced_rate") == 1
    assert "no committed evaluation carries" in capsys.readouterr().out
