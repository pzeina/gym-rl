"""The provenance probe restates the metric's definitions; they must not drift."""

from cohort.metrics import STACK_RADIUS
from scripts import spatial_probe_provenance as probe


def test_probe_stack_radius_matches_the_metric():
    # the child process replays inside an OLD tree and so cannot import the
    # current cohort.metrics — the radius is restated in the script, and this
    # is the tie that keeps the restatement honest
    assert probe.STACK_RADIUS == STACK_RADIUS


def test_worktree_cache_is_git_excluded():
    # the probe materializes whole historical trees; they must never be
    # commit-visible
    import subprocess
    res = subprocess.run(
        ["git", "check-ignore", "-q", str(probe.WORKTREE_CACHE / "provenance-abc1234")],
        cwd=probe.REPO,
    )
    assert res.returncode == 0
