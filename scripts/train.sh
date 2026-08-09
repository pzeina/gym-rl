#!/usr/bin/env bash
# Launch a cohort training run FULLY DETACHED from the Claude session.
#
# Why this exists: training is a shell process, not a reasoning task. It must
# cost zero model tokens while it runs. This script nohup's the run so it
# survives session exit (/clear, closing the terminal, ending the chat), and
# records a job file that scripts/train_status.py summarises in ~20 lines.
#
# NEVER run `python -m cohort.training.train` in the foreground of a chat turn.
# NEVER spawn a subagent to babysit a run — poll with scripts/train_status.py
# or block with scripts/train_wait.sh instead.
#
# Usage:
#   scripts/train.sh <run-name> [train args...]
#   scripts/train.sh fireteam_defend_v8 --scenario fireteam_defend --total-steps 3000000 --seed 12
#
# Env:
#   FORCE=1   allow appending to a run directory that already has metrics.csv
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
PY=${PY:-$ROOT/.venv/bin/python}

if [ $# -lt 1 ]; then
  echo "usage: scripts/train.sh <run-name> [train args...]" >&2
  exit 2
fi
RUN=$1
shift

RUN_DIR="$ROOT/runs/$RUN"
LOG="$ROOT/logs/$RUN.log"
JOB="$RUN_DIR/.job.json"

if [ -f "$RUN_DIR/metrics.csv" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "refusing: $RUN_DIR/metrics.csv exists — training would append and corrupt the curve." >&2
  echo "pick a new run name (e.g. ${RUN}b) or re-run with FORCE=1." >&2
  exit 1
fi

# Already-live job for this run? Ask train_status, which checks the pid is
# actually carrying --run-name "$RUN": pids are recycled, and a bare kill -0 on
# a months-old job file will refuse a perfectly valid launch once the OS hands
# that number to something else.
if [ -f "$JOB" ] && "$PY" "$ROOT/scripts/train_status.py" --is-running "$RUN"; then
  OLD_PID=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('pid',''))" "$JOB" 2>/dev/null || true)
  echo "refusing: run '$RUN' is already training (pid $OLD_PID). scripts/train_status.py $RUN" >&2
  exit 1
fi

mkdir -p "$RUN_DIR" "$ROOT/logs"

# Pull --total-steps out of the args so status/ETA can be computed without
# waiting for train.py to write config.json.
TOTAL_STEPS=500000
prev=""
for a in "$@"; do
  if [ "$prev" = "--total-steps" ]; then TOTAL_STEPS=$a; fi
  case "$a" in --total-steps=*) TOTAL_STEPS=${a#*=} ;; esac
  prev=$a
done

# Wrapped, not bare: the wrapper refreshes the fleet/program boards the moment
# the run lands, so a finished run is never waiting on a session to be visible.
# The recorded pid is the wrapper's — it outlives training by the few seconds
# the refresh takes, which is what train_wait.sh should wait for anyway.
nohup "$ROOT/scripts/train_then_boards.sh" --run-name "$RUN" "$@" >"$LOG" 2>&1 &
PID=$!
disown 2>/dev/null || true

"$PY" - "$JOB" "$RUN" "$PID" "$LOG" "$TOTAL_STEPS" "$@" <<'PYEOF'
import json, sys, time
job, run, pid, log, total = sys.argv[1:6]
args = sys.argv[6:]
json.dump({
    "run": run,
    "pid": int(pid),
    "log": log,
    "total_steps": int(total),
    "started": time.time(),
    "started_human": time.strftime("%Y-%m-%d %H:%M:%S"),
    "args": args,
}, open(job, "w"), indent=2)
PYEOF

cat <<EOF
launched detached: $RUN
  pid   $PID   (survives session exit; kill with: kill $PID)
  log   logs/$RUN.log
  steps $TOTAL_STEPS

This run now costs ZERO tokens. Safe to /clear or end the session.
Check back with:  scripts/train_status.py $RUN
EOF
