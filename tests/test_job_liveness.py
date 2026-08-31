"""A recycled pid must not resurrect a finished run.

Every finished run keeps its ``.job.json``, and the liveness check was a bare
``kill(pid, 0)``. Pids are recycled, so once the OS handed a stale number to an
unrelated process, a run that ended days ago read as RUNNING and then "ended"
again when that stranger exited — two spurious "training ended" notifications in
one session. It is not only cosmetic: ``train.sh`` refuses to launch a run whose
recorded pid looks alive, and ``train_wait.sh`` (which the campaign queue blocks
on between jobs) would wait on the stranger instead of the trainer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts import train_status
from scripts.train_status import alive, state_of

ROOT = Path(__file__).resolve().parent.parent


def test_a_live_pid_that_is_not_our_trainer_reads_as_not_running():
    # this test process is certainly alive and certainly not training a run
    assert alive(os.getpid()) is True, "no run name given — pid liveness is all we can check"
    assert alive(os.getpid(), "defend_brique_v9") is False


def test_a_dead_pid_reads_as_not_running():
    assert alive(999_999, "defend_brique_v9") is False
    assert alive(-1, "defend_brique_v9") is False


def test_a_finished_run_holding_a_recycled_pid_does_not_read_as_running(tmp_path):
    """The exact shape of the false alarm: stale job file, pid now someone else's."""
    run = tmp_path / "squad_screen_fallen_v2"
    run.mkdir()
    job = {"run": run.name, "pid": os.getpid(), "total_steps": 100, "log": ""}
    rows = [{"env_steps": "100", "success_rate_rolling": "0.99"}]

    assert state_of(run, job, rows, 100) == "DONE"


# A real trainer's command line is long before it reaches its flags — the
# interpreter path alone is ~50 characters on a CI runner. Linux `ps` (procps)
# truncates to the terminal width, and a pipe reads as 80 columns, so the
# `--run-name <run>` that `alive` looks for fell off the end and every live run
# read as dead. The padding below puts these probes past that boundary, which is
# where the real command lines already were.
_PAST_EIGHTY_COLUMNS = "x" * 120


def _sleeper(run_name: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep(30)  # {_PAST_EIGHTY_COLUMNS}",
         "--run-name", run_name],
    )


def test_the_run_name_must_match_exactly_not_by_prefix():
    """`squad_v1` must not match a process training `squad_v10`."""
    proc = _sleeper("squad_v10")
    try:
        assert alive(proc.pid, "squad_v10") is True
        assert alive(proc.pid, "squad_v1") is False
    finally:
        proc.kill()
        proc.wait()


def test_a_long_command_line_is_read_whole_and_not_truncated_to_the_width():
    """The truncation bug in its own right, in the unsafe direction.

    Reporting a live trainer as dead is worse than the recycled pid this module
    was written for: `train.sh` would launch a second run over the first, and
    `train_wait.sh` — which the campaign queue blocks on between jobs — would
    return at once and let the queue stack the whole campaign on one machine.

    This only fires where `ps` truncates (Linux). macOS does not truncate a
    piped stream, which is exactly why the bug lived here undetected and showed
    up only in CI, so the companion test below pins the fix on every platform.
    """
    proc = _sleeper("squad_v10")
    try:
        assert alive(proc.pid, "squad_v10") is True
    finally:
        proc.kill()
        proc.wait()


def test_liveness_asks_ps_for_the_unabridged_command_line():
    """Pins the fix where the behaviour above cannot be observed.

    Asserting on the argv rather than the result is deliberate: on macOS no
    input makes `ps` truncate, so a behavioural test cannot tell the fix from
    its absence, and dropping `-ww` would go green on this machine and red on
    every Linux host. The flag is load-bearing, so it is checked directly.
    """
    seen = {}
    real = subprocess.run

    def spy(argv, *a, **kw):
        if argv and argv[0] == "ps":
            seen["argv"] = argv
        return real(argv, *a, **kw)

    train_status.subprocess.run = spy
    try:
        alive(os.getpid(), "squad_v10")
    finally:
        train_status.subprocess.run = real

    assert seen.get("argv"), "alive() no longer consults ps"
    assert "-ww" in seen["argv"], (
        f"ps must be asked for the unabridged command line, got {seen['argv']} — "
        "without -ww, Linux truncates at 80 columns and a live run reads as dead"
    )


def test_the_shell_scripts_ask_train_status_rather_than_kill_dash_zero():
    """Both guards had their own copy of the bug; neither may grow one back.

    Comment lines are stripped first — the fix is *explained* in a comment that
    says "kill -0", and a check that cannot tell a warning from the mistake it
    warns about would fail on its own documentation.
    """
    for name in ("train.sh", "train_wait.sh"):
        text = (ROOT / "scripts" / name).read_text()
        assert "--is-running" in text, f"{name} no longer asks train_status for liveness"
        code = "\n".join(
            ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
        )
        assert "kill -0" not in code, f"{name} went back to a bare kill -0"


def test_is_running_exits_nonzero_for_a_run_that_is_not_training(tmp_path):
    run = ROOT / "runs" / "_no_such_run_for_tests"
    assert not run.exists()
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_status.py"), "--is-running", run.name],
        capture_output=True, text=True,
    )
    assert result.returncode == 1


def test_is_running_needs_a_run_name():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_status.py"), "--is-running"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "usage" in result.stderr


def test_job_files_on_disk_still_parse(tmp_path):
    """Guard the assumption the check rests on: .job.json carries a pid."""
    # not-archive-aware: a .job.json is live-training state; an archived run has
    # no job to check and its stale file says nothing about the assumption.
    for job_path in (ROOT / "runs").glob("*/.job.json"):
        job = json.loads(job_path.read_text())
        assert isinstance(job.get("pid"), int), job_path
