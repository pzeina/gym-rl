#!/usr/bin/env bash
# Block until a detached run finishes, then print a one-screen digest.
#
# Intended use: Bash tool with run_in_background=true. The harness re-invokes
# Claude exactly ONCE, when the run exits — instead of N polling turns. A
# sleeping wait costs nothing; a polling subagent costs a request per poll.
#
# Usage:  scripts/train_wait.sh <run-name> [poll-seconds]
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
PY=${PY:-$ROOT/.venv/bin/python}

RUN=${1:?usage: scripts/train_wait.sh <run-name> [poll-seconds]}
POLL=${2:-30}
JOB="$ROOT/runs/$RUN/.job.json"

[ -f "$JOB" ] || { echo "no job file for run '$RUN' (was it launched with scripts/train.sh?)" >&2; exit 2; }
PID=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['pid'])" "$JOB")

# Not `kill -0 $PID`: once training exits, the OS can hand that pid to an
# unrelated process and this loop would sit waiting on a stranger — stalling
# the campaign queue, which blocks here between jobs.
while "$PY" "$ROOT/scripts/train_status.py" --is-running "$RUN"; do
  sleep "$POLL"
done

echo "run '$RUN' finished (pid $PID gone)."
exec "$PY" "$ROOT/scripts/train_status.py" "$RUN"
