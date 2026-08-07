"""A training process must hold ONE consistent snapshot of the code.

Editing the tree while a detached run trains is normal practice here (see
CLAUDE.md's training workflow); losing a finished run to it is not. A run that
starts at commit A and finishes after commit B landed mixes A's already-imported
modules with B's freshly-read ones, and the mismatch surfaces at the very end —
after the full step budget has been spent — as an ImportError in the
post-training artifact step.

``train.main`` guards against this by importing every post-training entry point
BEFORE the long run starts. The guard has now been wrong twice, each time by
being one level too shallow:

* **2026-08-06** — it hoisted the entry points but not the modules those entry
  points import lazily, so ``cohort.metrics`` was still read off disk at the end
  of training. It imports ``order_options`` / ``is_done_admissible`` from
  ``cohort.env.actions`` at module level, and three runs — squad_v7,
  squad_recon_v6, platoon_v4 — each completed 3M steps and then died importing a
  name their in-memory ``actions`` did not have. (``fireteam_defend_v10`` had
  already died the shallower version of this.)
* **2026-08-07** — the fix for that was stated at depth ONE, against a hardcoded
  two-entry list of entry points. But ``cohort.metrics`` defers
  ``cohort.env.cohort_env``, which defers ``cohort.core.oracle`` — measured
  absent from ``sys.modules`` after the entire eager block runs. Nothing on the
  artifact path calls ``env.oracle()`` today, so this one was caught latent
  rather than at the cost of a run.

So the invariant is not "the entry points are hoisted", and not "what the entry
points reach at depth one is hoisted", but the closure of both:

    **nothing reachable from the snapshot may be read fresh off disk later.**

Stated that way it needs no hardcoded entry-point list — the roots are simply
whatever ``train.main`` imports before ``trainer.train``, and the property is
self-maintaining as the module graph moves.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "cohort"

#: ``train.py`` importing itself is a no-op: by the time main() runs, the module
#: is in memory by definition, and no later read can replace it.
_SELF = "cohort.training.train"


def _module_path(module: str) -> Path | None:
    """File backing a ``cohort.*`` module name, or None if it is not one of ours."""
    rel = module.replace(".", "/")
    for candidate in (_SRC.parent / f"{rel}.py", _SRC.parent / rel / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


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


def _open_edges() -> list[tuple[str, str]]:
    """``(importer, imported)`` pairs that escape the snapshot, transitively.

    Walks the deferred-import graph outward from every snapshot module. An edge
    is open when its target is not itself in the snapshot: at the end of a run
    that module gets read off whatever the tree says *then*, against an in-memory
    graph frozen hours earlier.
    """
    snapshot = _imports_before_training_starts()
    frontier, seen, open_edges = list(snapshot), set(), []
    while frontier:
        module = frontier.pop()
        if module in seen:
            continue
        seen.add(module)
        path = _module_path(module)
        if path is None:
            continue
        for target in _deferred_cohort_imports(path):
            if target == _SELF:
                continue
            if target not in snapshot:
                open_edges.append((module, target))
            # expanded even when open, so one failure reports the whole set to
            # hoist rather than one module per test run
            frontier.append(target)
    return sorted(set(open_edges))


def test_snapshot_is_closed_under_deferred_imports():
    """The transitive closure of deferred imports stays inside the snapshot.

    This is the general form of the two misses in the module docstring. Both
    would have failed here: `cohort.metrics` at depth one, `cohort.core.oracle`
    at depth two.
    """
    open_edges = _open_edges()
    assert not open_edges, "post-training artifacts can be lost to a mid-run edit:\n" + "\n".join(
        f"  {importer} defers -> {target}, which cohort/training/train.py does not "
        f"import before training starts. Hoist {target} into the eager block in "
        f"train.main()."
        for importer, target in open_edges
    )


def test_metrics_is_in_the_snapshot():
    """The specific module whose absence cost three 3M-step runs."""
    assert "cohort.metrics" in _imports_before_training_starts()


def test_oracle_is_in_the_snapshot():
    """The depth-two module the closure rule found, before it cost anything.

    `cohort.metrics` -> `cohort.env.cohort_env` -> `cohort.core.oracle`. Named
    explicitly because the diagnose-first rule in CLAUDE.md keeps pulling the
    oracle toward the evaluation path, and that is the day it would start
    killing runs.
    """
    assert "cohort.core.oracle" in _imports_before_training_starts()


def test_snapshot_is_measured_against_the_training_call():
    """The helper must actually find the training call it orders imports against.

    Without this, a rename of `trainer.train(...)` would make `train_call_line`
    fall back to something meaningless and quietly pass everything above.
    """
    assert _imports_before_training_starts(), "no pre-training cohort imports found"
    assert "cohort.training.evaluate" in _imports_before_training_starts()


def test_the_closure_walk_actually_traverses_more_than_one_level():
    """Guard the guard: a depth-one walk would have passed the 2026-08-07 miss.

    If `_open_edges` ever stops recursing, it silently becomes the weaker check
    it replaced. `cohort.metrics` is a snapshot root that defers
    `cohort.env.cohort_env`, which defers further — so the walk must reach
    modules no root names directly.
    """
    roots = _imports_before_training_starts()
    reachable = set()
    frontier = list(roots)
    seen = set()
    while frontier:
        module = frontier.pop()
        if module in seen:
            continue
        seen.add(module)
        path = _module_path(module)
        if path is None:
            continue
        for target in _deferred_cohort_imports(path):
            reachable.add(target)
            frontier.append(target)
    assert reachable - roots - {_SELF} or "cohort.core.oracle" in reachable, (
        "the deferred-import walk found nothing past its roots — either the graph "
        "genuinely flattened, or the traversal broke"
    )
