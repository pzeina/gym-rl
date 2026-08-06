---
description: Cheap check-in on detached training runs (no raw logs, no metrics.csv)
allowed-tools: Bash(.venv/bin/python scripts/train_status.py:*)
---

Run `.venv/bin/python scripts/train_status.py $ARGUMENTS` and relay the output.

Do not read `metrics.csv`, `logs/*.log`, or any checkpoint. If the run is still
training, say so in one line and stop — do not speculate about the numbers
mid-run, and do not offer to keep watching. If it has finished, say so and
suggest `/train-report <run>`.
