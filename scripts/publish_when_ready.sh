#!/usr/bin/env bash
# Score each baseline member at N=100 the moment it lands, not all at the end.
#
# A campaign's last run finishing is not the end of the work: eight members at
# two checkpoints each is another half hour of episodes, and doing it serially
# after the fact puts every minute of it on the critical path. This waits on
# each member in manifest order and publishes it as soon as it is down, so by
# the time the long pole lands, everything behind it is already scored.
#
# Detached, zero model tokens. Safe to run while training continues: it refuses
# a run that is still RUNNING, and scripts/publish_baseline.py refuses to
# overwrite an evaluation at least as large as the one it would write.
#
# Usage:  nohup scripts/publish_when_ready.sh > logs/publish_when_ready.log 2>&1 &
set -u

cd "$(dirname "$0")/.."
ROOT=$PWD
PY=${PY:-$ROOT/.venv/bin/python}
POLL=${POLL:-90}
DEADLINE=$(( $(date +%s) + ${MAX_WAIT:-28800} ))   # 8h, so a stuck lane cannot hang this forever

members=$("$PY" -c "
import json
print(' '.join(json.load(open('runs/BASELINE.json'))['runs'].values()))
")

# Publish whichever member is READY, never in a fixed order. The first version
# of this waited on the manifest in order and blocked on fireteam_defend_v20 —
# the LAST job of its lane — while four runs that had already landed sat
# unscored behind it. Waiting on a named run is only correct when you know the
# order things finish in, and across three parallel lanes you do not.
echo "=== watching: $members"
remaining="$members"
while [ -n "${remaining// /}" ]; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "!!! deadline reached; never scored: $remaining"
    exit 2
  fi
  progressed=0
  next=""
  for run in $remaining; do
    # Landed = the directory exists, a final checkpoint is on disk, and no
    # training process carries this run name. All three: a queued job has no
    # directory, and a directory mid-run has a checkpoint but a live pid.
    if [ -f "runs/$run/ckpt_latest.pt" ] && ! "$PY" scripts/train_status.py --is-running "$run"; then
      echo "=== [$(date +%H:%M:%S)] $run landed — scoring at N=100"
      "$PY" scripts/publish_baseline.py "$run" || echo "!!! $run: publish reported problems"
      progressed=1
    else
      next="$next $run"
    fi
  done
  remaining=$next
  [ "$progressed" = "1" ] || sleep "$POLL"
done

echo "=== every member scored; refreshing boards and the README table"
"$PY" scripts/update_boards.py --quiet || echo "board refresh failed (non-fatal)"
"$PY" scripts/results_table.py --write || echo "table regeneration failed (non-fatal)"
"$PY" scripts/baseline.py
