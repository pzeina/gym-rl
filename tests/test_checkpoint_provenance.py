"""An evaluation must name the weights it scored, not just their path.

Issue #28: every published run commits `ckpt_best.pt` and none commits
`ckpt_latest.pt` (it was gitignored fleet-wide), while the README's headline
column is the FINAL policy — so the quoted checkpoint is the one nobody
outside this machine can obtain. The digest does not fix that; it makes it
checkable, which is the cheap 95%: an independent re-measurement can prove
it scored the same object the headline was measured on.

Issue #44 closed the other half for the live fleet: `runs/` now commits both
checkpoints and `runs/archive/` sheds the final one. So for a baseline member
the digest is no longer merely checkable in principle — the bytes it anchors
are in the repository, and the last test here checks them against it.

The regression hazard these tests encode is a silent one. If the field
quietly disappears (or, worse, names `ckpt_best` in `behavior_final.json`),
nothing fails and nothing looks wrong — the numbers still publish, and the
anchor is gone. Same for the weights: re-ignoring `ckpt_latest.pt` breaks
nothing visible here and quietly un-reproduces every headline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from cohort.env.actions import N_ACTIONS
from cohort.env.observations import OBS_DIM
from cohort.training.evaluate import _file_sha256, evaluate
from cohort.training.ppo import PolicyNet

ROOT = Path(__file__).resolve().parents[1]


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
    where the run has been pruned from this working copy — these three are
    superseded and live in `runs/archive/`, which under issue #44's rule is
    the half of the tree that sheds `ckpt_latest.pt`. The digest is asserted
    unconditionally; the bytes only when they are here to check.
    """
    from scripts.fleet_status import find_run

    # Through the resolver, never RUNS / run: archiving a superseded run must
    # not silently switch this invariant off. It did — 6 of these skipped the
    # moment 96 runs moved into runs/archive/, and a skip is not a pass.
    root = find_run(run, ROOT / "runs")
    if root is None:
        pytest.skip(f"{run} not present in this working copy")
    payload_path = root / file
    if not payload_path.is_file():
        pytest.skip(f"{run}/{file} not present in this working copy")
    payload = json.loads(payload_path.read_text())
    assert payload["checkpoint"].endswith(checkpoint)
    assert len(payload["checkpoint_sha256"]) == 64
    if (root / checkpoint).is_file():
        assert payload["checkpoint_sha256"] == _file_sha256(root / checkpoint)


def _git_tracked(*paths: Path) -> set[Path] | None:
    """What of ``paths`` is in this repository's index — None if git cannot say.

    A tarball export, or a CI that fetches without `.git`, has no index and so
    no opinion about what is committed. That is a different answer from "not
    committed", and conflating them would make this suite fail for a reason
    unrelated to what it checks.
    """
    import subprocess

    try:
        out = subprocess.run(["git", "ls-files", "--", *(str(p) for p in paths)],
                             cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return {ROOT / line for line in out.stdout.splitlines() if line}


def _baseline_members() -> list[tuple[str, str]]:
    # BASELINE.json is fleet state, not a run — addressed directly on purpose.
    manifest = json.loads((ROOT / "runs" / "BASELINE.json").read_text())
    return sorted(manifest.get("runs", {}).items())


@pytest.mark.parametrize(("scenario", "run"), _baseline_members())
def test_a_baseline_members_headline_weights_are_in_the_repository(scenario, run):
    """Issue #44: the number and the weights that produce it ship together.

    The sealed fleet's headline is the FINAL policy — `behavior_final.json`,
    scored from `ckpt_latest.pt`. Committing only `ckpt_best.pt` left a clone
    able to *read* all eight figures and re-derive none of them, because
    `ckpt_best.pt` is a best-rolling-WINDOW snapshot and by this repo's own
    audit not the policy the headline describes. Best and final have disagreed
    on this very fleet by 30/30 success vs 30/30 timeout on one run.

    So: the file is tracked, and its bytes are the ones the digest names. The
    second half is what makes the first worth anything — a committed
    `ckpt_latest.pt` that does not hash to `checkpoint_sha256` would be a
    reproducibility claim backed by the wrong weights, which is worse than the
    absence this issue reported.
    """
    from scripts.fleet_status import find_run

    root = find_run(run, ROOT / "runs")
    assert root is not None, f"{run} is a baseline member and is not in this tree"

    final = root / "behavior_final.json"
    assert final.is_file(), f"{run} has no FINAL-policy evaluation to anchor"
    payload = json.loads(final.read_text())
    assert payload["checkpoint"].endswith("ckpt_latest.pt")

    ckpt = root / "ckpt_latest.pt"
    tracked = _git_tracked(ckpt)
    if tracked is None:
        pytest.skip("no git index here — cannot tell committed from absent")
    assert ckpt in tracked, (
        f"{run}/ckpt_latest.pt is not committed: a reader can see this member's "
        f"headline in behavior_final.json and cannot re-derive it (issue #44)"
    )
    assert payload["checkpoint_sha256"] == _file_sha256(ckpt), (
        f"{run}/ckpt_latest.pt is committed but is not the object "
        f"behavior_final.json scored"
    )


def test_the_archive_is_the_half_of_the_tree_that_sheds_the_final_policy():
    """The rule that makes the fix survive the next archive move.

    `.gitignore`'s `runs/*/ckpt_latest.pt` was depth-dependent — a `*` does not
    cross `/` — so filing 96 runs into `runs/archive/` silently inverted it:
    every superseded run started carrying the final weights and not one
    shipping member did. The rule is now stated on the axis that actually
    means something (live vs superseded) rather than on directory depth, and
    this pins both directions of it.
    """
    import subprocess

    def ignored(path: str) -> bool:
        # --no-index: already-tracked paths are otherwise reported as unignored,
        # and the 96 the move swept in stay tracked by design.
        return subprocess.run(["git", "check-ignore", "--no-index", "-q", path],
                              cwd=ROOT, capture_output=True).returncode == 0

    if subprocess.run(["git", "rev-parse", "--git-dir"], cwd=ROOT,
                      capture_output=True).returncode != 0:
        pytest.skip("not a git work tree")

    for name in ("ckpt_latest.pt", "ckpt_best.pt"):
        assert not ignored(f"runs/a_live_run/{name}"), (
            f"a live run's {name} must be committable — it is what a reader "
            "reproduces the headline from"
        )
    assert ignored("runs/archive/a_superseded_run/ckpt_latest.pt")
    # Depth-independent for the bulky and the host-specific, at both levels.
    for prefix in ("runs/a_live_run", "runs/archive/a_superseded_run"):
        assert ignored(f"{prefix}/tb/events.out.tfevents.1")
        assert ignored(f"{prefix}/.job.json")
