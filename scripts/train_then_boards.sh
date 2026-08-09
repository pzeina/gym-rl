#!/usr/bin/env bash
# Internal: run one training job, then refresh the boards. Not called by hand —
# scripts/train.sh wraps every launch in this, so "a run landed" and "the boards
# are current" are the same event and neither costs a model token.
#
# The refresh runs whether training succeeded, crashed, or was killed: a run
# that died still changes what the fleet board should say about it, and a board
# that still shows a dead run as in-flight is worse than one showing a failure.
#
# Usage (from train.sh):  scripts/train_then_boards.sh --run-name <run> [args...]
set -u

cd "$(dirname "$0")/.."
ROOT=$PWD
PY=${PY:-$ROOT/.venv/bin/python}

"$PY" -m cohort.training.train "$@"
status=$?

echo "=== training exited ($status); refreshing boards ==="
"$PY" "$ROOT/scripts/update_boards.py" --quiet || echo "board refresh failed (non-fatal)"

exit $status
