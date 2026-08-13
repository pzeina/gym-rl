"""``cohort/play.py``'s docstring is documentation, not data (#51).

Every other module in ``cohort/`` treats its docstring as inert prose: a sweep
that rewords it changes what a reader sees and nothing else. ``play.py`` was the
exception. It recovered the console's help text by splitting its own
``__doc__``::

    HELP = __doc__.split("Console commands:")[1]

which made a prose edit into a behaviour change, in two ways the file did not
announce — the coupling sits thirty lines below the docstring it consumes:

1. **Rewording the lines under the marker silently rewrote what ``help``
   printed.** Nothing asserted the console's output, so a wording drift landed
   unnoticed.
2. **Rewording or dropping the marker line itself raised ``IndexError`` at
   import**, not at first use. ``play.py`` imports ``cohort.env.cohort_env`` and
   ``cohort.training.evaluate``, so the traceback surfaces late in a long chain
   and points at a docstring nobody was thinking about. The same line raised
   ``AttributeError`` under ``python -OO``, where ``__doc__`` is ``None``.

The fix inverted the arrow: ``HELP`` is the constant, the docstring is built
from it at import. Single-sourcing is kept — there is still exactly one copy of
the command list — so this file pins the *direction*, which is the part that can
silently come back.

The tests below exec the real ``cohort/play.py`` source with its docstring
rewritten the way a sweep would rewrite it. That is cheap: every module it
imports is already in ``sys.modules`` by the time the suite runs.
"""

from __future__ import annotations

import ast
import builtins
import pathlib
import sys

import pytest

import cohort.play as play

_SRC = pathlib.Path(play.__file__)
_SOURCE = _SRC.read_text()


def _with_docstring(new_doc: str) -> str:
    """The real source with its module docstring replaced by ``new_doc``.

    ``play.py`` opens with its docstring at character zero, so the literal is
    delimited by the first two ``\"\"\"`` in the file — no parsing needed, and
    nothing downstream of it moves.
    """
    assert _SOURCE.startswith('"""'), "play.py must open with its module docstring"
    end = _SOURCE.index('"""', 3) + 3
    return f'"""{new_doc}"""' + _SOURCE[end:]


def _exec(source: str, *, optimize: int = -1) -> dict:
    """Import ``source`` as a fresh namespace, the way CPython imports a module.

    ``__doc__`` is pre-seeded to ``None`` because module objects are created
    that way; compiling with ``optimize=2`` then leaves it ``None``, which is
    exactly what ``python -OO`` does.
    """
    namespace: dict = {
        "__name__": "cohort.play_under_test",  # never "__main__": main() must not run
        "__file__": str(_SRC),
        "__doc__": None,
    }
    exec(compile(source, str(_SRC), "exec", optimize=optimize), namespace)
    return namespace


#: A sweep that keeps the meaning and rewords the marker line — the edit that
#: used to raise IndexError at import.
_MARKER_REWORDED = """Interactive commander console.

    python -m cohort.play --scenario squad --as SL1

Console commands (or type `help` in-session):
"""

#: A sweep that leaves the marker alone and rewrites the prose under it — the
#: edit that used to change what the console printed, with nothing to catch it.
_ENTRIES_REWRITTEN = """Interactive commander console.

Console commands:
    m       draw the map
    q       leave the console
"""


def test_rewording_the_marker_line_does_not_break_the_import():
    """Consequence 2: the docstring may say anything; the module still imports."""
    namespace = _exec(_with_docstring(_MARKER_REWORDED))

    assert namespace["HELP"] == play.HELP, "the console help must not follow the prose"


def test_rewriting_the_docstring_entries_does_not_change_what_help_prints():
    """Consequence 1: prose is prose, even when it looks like the command list."""
    namespace = _exec(_with_docstring(_ENTRIES_REWRITTEN))

    assert namespace["HELP"] == play.HELP
    assert "leave the console" not in namespace["HELP"]


def test_the_console_help_survives_stripped_docstrings():
    """``python -OO`` removes every docstring; the console keeps its help."""
    namespace = _exec(_SOURCE, optimize=2)

    assert namespace["__doc__"] is None, "optimize=2 must actually strip the docstring"
    assert namespace["HELP"] == play.HELP


def test_the_docstring_still_carries_the_command_list_exactly_once():
    """Single-sourcing is the half of the old design worth keeping.

    The fix must not be "hand-copy the list into the docstring" — two copies
    drift, which is the failure this file exists to prevent, just slower.
    """
    assert play.HELP in play.__doc__, "the docstring must be built from HELP, not beside it"
    for line in play.HELP.strip().splitlines():
        assert play.__doc__.count(line) == 1, f"the command list is duplicated in the docstring: {line!r}"


def test_the_console_help_command_prints_the_help_constant(monkeypatch, capsys):
    """The end of the chain, driven rather than read.

    ``cohort.play`` is the deployment surface and is otherwise exercised only by
    hand, so nothing tied ``HELP`` to what a commander actually sees. This does.
    """
    typed = iter(["help", "q"])
    monkeypatch.setattr(sys, "argv", ["play", "--scenario", "fireteam", "--seed", "0"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(typed))

    play.main()

    assert play.HELP in capsys.readouterr().out


def test_the_docstring_is_no_longer_read_as_data():
    """The regression itself, named: nothing may be *parsed out of* ``__doc__``.

    Over the AST rather than the source lines, so that quoting the old
    expression in a comment — which the fix does, one line above ``HELP`` —
    is prose again and not a violation. Reading ``__doc__`` as a whole value
    stays legal: that is how the docstring is composed from ``HELP``. What is
    banned is calling a method on it, which is the only way to get a *part* of
    it back out.
    """
    offenders = [
        f"line {node.lineno}: __doc__.{node.attr}(...)"
        for node in ast.walk(ast.parse(_SOURCE))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "__doc__"
    ]

    assert not offenders, "play.py's docstring must not be parsed for its own console text (#51):\n  " + "\n  ".join(
        offenders
    )


def test_help_is_a_plain_literal():
    """No f-string, no ``.format``, no concatenation — a sweep must see the text it edits."""
    assignment = next(
        node
        for node in ast.parse(_SOURCE).body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "HELP" for t in node.targets)
    )
    assert isinstance(assignment.value, ast.Constant) and isinstance(assignment.value.value, str), (
        "HELP must be a plain string literal so the text and the edit site are the same place"
    )


@pytest.mark.parametrize("command", ["m", "net", "status", "help", "q"])
def test_every_documented_command_is_one_the_console_dispatches(command):
    """The list is only useful if it matches the console; nothing else asserts that.

    Deliberately one-directional: aliases (``quit``, ``map``, ``h``, ``?``) are
    intentionally undocumented, so requiring the reverse would pin a style
    choice rather than a fact.
    """
    documented = {line.split()[0] for line in play.HELP.strip().splitlines()}
    assert command in documented, f"the console handles {command!r} but HELP does not list it"
