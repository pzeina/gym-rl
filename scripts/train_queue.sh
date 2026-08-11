#!/usr/bin/env bash
# Run a whole retrain CAMPAIGN detached, sequentially, in one background process.
#
# A campaign of 6 runs is 6x the wall-clock but still zero tokens — the whole
# queue is one nohup'd shell loop. Use this instead of asking Claude to launch
# runs one after another (each hand-off is a full expensive turn).
#
# Usage:
#   scripts/train_queue.sh <jobs-file>
#
# Jobs file: one job per line, blank lines and #-comments ignored.
#   <run-name> <train args...>
# e.g.
#   fireteam_defend_v8   --scenario fireteam_defend --total-steps 3000000 --seed 12
#   fireteam_defend_v8b  --scenario fireteam_defend --total-steps 3000000 --seed 13
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
PY=${PY:-$ROOT/.venv/bin/python}

JOBS=${1:?usage: scripts/train_queue.sh <jobs-file>}
[ -f "$JOBS" ] || { echo "no such jobs file: $JOBS" >&2; exit 2; }

# $$ as well as the clock: a multi-lane campaign launches its queues in the
# same second, and two of them sharing one log file interleaves the campaign
# transcript of both. Caught launching baseline v1.19 — lanes A and B both
# opened logs/queue_20260811_100314.log.
STAMP=$(date +%Y%m%d_%H%M%S)_$$
QLOG="$ROOT/logs/queue_$STAMP.log"
mkdir -p "$ROOT/logs"

# Validate up front: a typo'd scenario should fail now, not 40 minutes in.
while read -r line; do
  case "$line" in ''|\#*) continue ;; esac
  set -- $line
  run=$1; shift
  scen=""
  prev=""
  for a in "$@"; do
    [ "$prev" = "--scenario" ] && scen=$a
    case "$a" in --scenario=*) scen=${a#*=} ;; esac
    prev=$a
  done
  [ -n "$scen" ] && "$PY" -c "from cohort.config import get_scenario; get_scenario('$scen')" \
    || { echo "bad scenario '$scen' for run '$run'" >&2; exit 1; }
  if [ -f "runs/$run/metrics.csv" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "refusing: runs/$run/metrics.csv exists (job '$run')" >&2; exit 1
  fi
done < "$JOBS"

nohup bash -c '
  set -u
  jobs_file="$1"; root="$2"
  cd "$root"
  while read -r line; do
    case "$line" in ""|\#*) continue ;; esac
    set -- $line
    run=$1; shift
    echo "=== [$(date +%H:%M:%S)] START $run $* ==="
    FORCE="${FORCE:-0}" scripts/train.sh "$run" "$@" || echo "!!! FAILED to launch $run"
    # scripts/train.sh detaches; wait for that run to finish before the next.
    scripts/train_wait.sh "$run" >/dev/null 2>&1 || true
    echo "=== [$(date +%H:%M:%S)] DONE  $run ==="
  done < "$jobs_file"
  echo "=== campaign complete ==="
' _ "$JOBS" "$ROOT" >"$QLOG" 2>&1 &
QPID=$!
disown 2>/dev/null || true

N=$(grep -cvE '^\s*(#|$)' "$JOBS" || true)
cat <<EOF
campaign launched detached: $N job(s)
  pid   $QPID   (kill the campaign with: kill $QPID)
  log   ${QLOG#$ROOT/}

Zero tokens while it runs. Safe to /clear or end the session.
Check back with:  scripts/train_status.py
EOF
