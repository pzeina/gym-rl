#!/bin/bash
# The night watch's event stream: one line per training that ends (any
# outcome — landed or crashed, both leave the live set), plus one line when a
# detached probe/eval marked by a "*-DONE" sentinel in its log finishes.
#
# Meant to run under the session's Monitor tool (persistent), where every
# stdout line becomes a wake-up event. Cheap by construction: one
# train_status.py call per cycle, nothing else.
#
# Usage: night_watch_monitor.sh [sentinel_log ...]
#   Each sentinel_log is watched for a line containing "-DONE"; the first
#   match emits one event, then that log is not reported again.
set -u
cd "$(dirname "$0")/.."

prev=""
declare -A sentinel_fired
while true; do
  out=$(.venv/bin/python scripts/train_status.py 2>/dev/null)
  cur=$(echo "$out" | awk '/RUNNING/{print $2}')
  if [ -n "$prev" ]; then
    for r in $prev; do
      if ! echo "$cur" | grep -qx "$r"; then
        line=$(echo "$out" | grep -m1 " $r " | tr -s ' ')
        echo "TRAINING ENDED: $r ::${line:- see train_status}"
      fi
    done
  fi
  prev="$cur"
  for log in "$@"; do
    if [ -z "${sentinel_fired[$log]:-}" ] && grep -q -- "-DONE" "$log" 2>/dev/null; then
      echo "DETACHED JOB DONE: $log"
      sentinel_fired[$log]=1
    fi
  done
  sleep 120
done
