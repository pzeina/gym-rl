"""An evaluation must name the weights it scored, not just their path.

Issue #28: every published run commits `ckpt_best.pt` and none commits
`ckpt_latest.pt` (it is gitignored fleet-wide), while the README's headline
column is the FINAL policy — so the quoted checkpoint is the one nobody
outside this machine can obtain. The digest does not fix that; it makes it
checkable, which is the cheap 95%: an independent re-measurement can prove
it scored the same object the headline was measured on.

The regression hazard these tests encode is a silent one. If the field
quietly disappears (or, worse, names `ckpt_best` in `behavior_final.json`),
nothing fails and nothing looks wrong — the numbers still publish, and the
anchor is gone.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from cohort.env.actions import N_ACTIONS
from cohort.env.observations import OBS_DIM
from cohort.training.evaluate import _file_sha256, evaluate
from cohort.training.ppo import PolicyNet


def _write_checkpoint(path, hidden: int = 16) -> None:
    """A loadable checkpoint of an untrained net — cheapest real weights."""
    net = PolicyNet(OBS_DIM, N_ACTIONS, hidden=hidden)
    torch.save(
        {
            "model": net.state_dict(),
            "obs_dim": OBS_DIM,
            "n_actions": N_ACTIONS,
            "hidden": hidden,
            "scenario": "fireteam",
        },
        path,
    )


def test_behavior_json_records_the_digest_of_the_weights_it_scored(tmp_path):
    """`checkpoint_sha256` matches `shasum -a 256` of the file loaded."""
    ckpt = tmp_path / "ckpt_best.pt"
    _write_checkpoint(ckpt)
    out = tmp_path / "behavior.json"

    evaluate(str(ckpt), episodes=1, seed=17, behavior=True, behavior_path=str(out))

    payload = json.loads(out.read_text())
    # computed independently of the streaming implementation under test
    expected = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert payload["checkpoint"] == str(ckpt)
    assert payload["checkpoint_sha256"] == expected
    assert len(payload["checkpoint_sha256"]) == 64


def test_each_behavior_file_carries_its_own_checkpoints_digest(tmp_path):
    """`behavior.json` anchors ckpt_best, `behavior_final.json` ckpt_latest.

    They sit in one directory and differ only in which weights they scored,
    so a digest copied from the wrong one would be worse than none at all.
    """
    best, latest = tmp_path / "ckpt_best.pt", tmp_path / "ckpt_latest.pt"
    _write_checkpoint(best, hidden=16)
    _write_checkpoint(latest, hidden=24)  # different weights → different file

    evaluate(str(best), episodes=1, seed=17, behavior_path=str(tmp_path / "behavior.json"))
    evaluate(
        str(latest), episodes=1, seed=17, behavior_path=str(tmp_path / "behavior_final.json")
    )

    b = json.loads((tmp_path / "behavior.json").read_text())
    f = json.loads((tmp_path / "behavior_final.json").read_text())
    assert b["checkpoint_sha256"] == hashlib.sha256(best.read_bytes()).hexdigest()
    assert f["checkpoint_sha256"] == hashlib.sha256(latest.read_bytes()).hexdigest()
    assert b["checkpoint_sha256"] != f["checkpoint_sha256"]


def test_unreadable_checkpoint_yields_no_digest_rather_than_an_error(tmp_path):
    """Hashing failures are silent: a missing file or a directory is None."""
    assert _file_sha256(tmp_path / "nope.pt") is None
    assert _file_sha256(tmp_path) is None


def test_a_vanished_checkpoint_does_not_cost_the_evaluation_its_numbers(
    tmp_path, monkeypatch
):
    """Provenance is best-effort; the scored episodes are not.

    Simulates the checkpoint disappearing between load and hash — the one
    window where the file is unreadable but the evaluation is already valid.
    """
    from cohort.training import train as train_mod

    ckpt = tmp_path / "ckpt_latest.pt"
    _write_checkpoint(ckpt)
    real_load = train_mod.load_policy

    def load_then_vanish(path, *a, **kw):
        loaded = real_load(path, *a, **kw)
        ckpt.unlink()
        return loaded

    monkeypatch.setattr(train_mod, "load_policy", load_then_vanish)
    out = tmp_path / "behavior.json"
    summary = evaluate(str(ckpt), episodes=1, seed=17, behavior_path=str(out))

    assert "behavior" in summary
    payload = json.loads(out.read_text())
    assert "checkpoint_sha256" not in payload
    assert payload["metrics"]["episodes"] == 1


def test_random_baseline_has_no_checkpoint_to_anchor(tmp_path):
    """No checkpoint, no digest — and no crash reaching for one."""
    out = tmp_path / "behavior.json"
    evaluate(None, scenario="fireteam", episodes=1, seed=17, behavior_path=str(out))
    payload = json.loads(out.read_text())
    assert payload["checkpoint"] is None
    assert "checkpoint_sha256" not in payload


@pytest.mark.parametrize(
    ("run", "file", "checkpoint"),
    [
        (run, file, ckpt)
        for run in ("fireteam_defend_v15", "defend_brique_v9", "defend_brique_v10")
        for file, ckpt in (
            ("behavior.json", "ckpt_best.pt"),
            ("behavior_final.json", "ckpt_latest.pt"),
        )
    ],
)
def test_published_runs_carry_a_digest_for_the_weights_they_quote(run, file, checkpoint):
    """The README's v1.13 table is anchored on both columns.

    Backfilled, not re-evaluated: the numbers in these files are published
    and unchanged; the digest only says which weights produced them. Skips
    where the run has been pruned from this working copy — `ckpt_latest.pt`
    is gitignored, so a fresh clone has the field but not the file to check
    it against, and that asymmetry is the whole point of issue #28.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "runs" / run
    payload_path = root / file
    if not payload_path.is_file():
        pytest.skip(f"{run}/{file} not present in this working copy")
    payload = json.loads(payload_path.read_text())
    assert payload["checkpoint"].endswith(checkpoint)
    assert len(payload["checkpoint_sha256"]) == 64
    if (root / checkpoint).is_file():
        assert payload["checkpoint_sha256"] == _file_sha256(root / checkpoint)
