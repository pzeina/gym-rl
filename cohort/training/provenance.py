"""Which tree a thing was made against.

``train.py`` records HEAD once per RUN, in ``economics.json:git_commit`` — the
code that produced the weights. That is not enough to date a *score*: an
evaluation can be re-run at any later commit, against weights trained long
before, and ``publish_audit`` differences two artifacts to decide whether a
best/final pair is comparable at all (refs #39). So an evaluation records its
own commit too, and both callers read it from here rather than each keeping a
copy of the same six lines.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def git_commit() -> str | None:
    """HEAD right now, or None outside a git worktree / when git is unavailable.

    Never raises: provenance is metadata, and a missing commit must degrade an
    artifact's dating rather than fail the run that was writing it.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
            cwd=_REPO_ROOT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None
