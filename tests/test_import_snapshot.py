"""A training process must hold ONE consistent snapshot of the code.

Editing the tree while a detached run trains is normal practice here (see
CLAUDE.md's training workflow); losing a finished run to it is not. A run that
starts at commit A and finishes after commit B landed mixes A's already-imported
modules with B's freshly-read ones, and the mismatch surfaces at the very end —
after the full step budget has been spent — as an ImportError in the
post-training artifact step.

``train.main`` guards against this by importing every post-training entry point
BEFORE the long run starts. That guard was incomplete on 2026-08-06: it hoisted
the entry points but not the modules those entry points import lazily, so
``cohort.metrics`` was still read off disk at the end of training. It imports
``order_options`` / ``is_done_admissible`` from ``cohort.env.actions`` at module
level, and three runs — squad_v7, squad_recon_v6, platoon_v4 — each completed
3M steps and then died importing a name their in-memory ``actions`` did not have.

The invariant is therefore not "the entry points are hoisted" but **"the snapshot
covers what the entry points reach"**, which is what this module tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "cohort"

#: entry points ``train.main`` calls after training, whose deferred imports must
#: be covered by the eager snapshot
_ENTRY_POINT_MODULES = ("training/evaluate.py", "viz/plots.py")


def _deferred_cohort_imports(path: Path) -> set[str]:
    """``cohort.*`` modules this file imports inside a function body.

    A function-scope import is not resolved until the call happens, which for a
    post-training entry point means *after* the run — the exact window in which
    the tree may have moved.
    """
    tree = ast.parse(path.read_text())
    deferred: set[str] = set()
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(func):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("cohort"):
                deferred.add(node.module)
            elif isinstance(node, ast.Import):
                deferred |= {a.name for a in node.names if a.name.startswith("cohort")}
    return deferred


def _imports_before_training_starts() -> set[str]:
    """``cohort.*`` modules ``train.py`` imports before it calls ``trainer.train``.

    Line-ordered on purpose: an import sitting *after* the training call is
    resolved at the same unsafe moment as a deferred one, so it does not count
    toward the snapshot however eager it looks.
    """
    path = _SRC / "training" / "train.py"
    tree = ast.parse(path.read_text())
    train_call_line = min(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "train"
    )
    snapshot: set[str] = set()
    for node in ast.walk(tree):
        # ast.walk yields the Module node too, which carries no position
        if getattr(node, "lineno", 0) >= train_call_line:
            continue
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("cohort"):
            snapshot.add(node.module)
        elif isinstance(node, ast.Import):
            snapshot |= {a.name for a in node.names if a.name.startswith("cohort")}
    return snapshot


def test_post_training_entry_points_defer_nothing_outside_the_snapshot():
    """Every cohort module an entry point imports lazily is hoisted in train.py.

    This is the test that would have caught the 2026-08-06 loss: `cohort.metrics`
    was reachable from `evaluate()` but absent from the snapshot.
    """
    snapshot = _imports_before_training_starts()
    for rel in _ENTRY_POINT_MODULES:
        for module in _deferred_cohort_imports(_SRC / rel):
            # a self-import back into train is already in memory by definition
            if module == "cohort.training.train":
                continue
            assert module in snapshot, (
                f"{rel} imports {module} lazily, but cohort/training/train.py does not "
                f"import it before training starts. A run that finishes after {module} "
                f"changes will lose its post-training artifacts. Hoist it into the eager "
                f"block in train.main()."
            )


def test_metrics_is_in_the_snapshot():
    """The specific module whose absence cost three 3M-step runs."""
    assert "cohort.metrics" in _imports_before_training_starts()


def test_snapshot_is_measured_against_the_training_call():
    """The helper must actually find the training call it orders imports against.

    Without this, a rename of `trainer.train(...)` would make `train_call_line`
    fall back to something meaningless and quietly pass everything above.
    """
    assert _imports_before_training_starts(), "no pre-training cohort imports found"
    assert "cohort.training.evaluate" in _imports_before_training_starts()
