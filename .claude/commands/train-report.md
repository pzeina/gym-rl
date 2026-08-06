---
description: Analyse a finished run from a compact digest (this is the part worth big-model tokens)
allowed-tools: Bash, Read, Glob, Grep, Agent
---

Analyse the finished run(s): $ARGUMENTS

1. Get the digest via the **run-digest** agent (or, for a single run,
   `.venv/bin/python scripts/run_report.py <run> [--vs <baseline>]` directly —
   it is already compact). Never pull `metrics.csv`, `tb/`, or a full log into
   this context.
2. Then do the actual thinking here, at full model strength:
   - did it learn, plateau, or regress — and against which baseline?
   - which reward component drifted, and does that match the change under test?
   - does the behavioral suite (obedience latency, report P/R, false DONE,
     doctrine preference) agree with the success rate, or contradict it?
   - is the result inside the CI of the baseline, i.e. is it a real effect?
3. Recommend exactly one next action: publish, retrain with a named parameter
   change, or investigate a specific hypothesis. Say which run is the baseline
   for the next comparison.

If a follow-up retrain is warranted, propose the exact `scripts/train.sh` line
but do not launch it until I say go.
