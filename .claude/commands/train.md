---
description: Launch a training run (or campaign) fully detached — zero tokens while it trains
allowed-tools: Bash, Read, Write, Glob, Agent
---

Launch training for: $ARGUMENTS

Rules:
- Delegate to the **train-ops** agent (haiku). Do not launch from this context,
  and do not use general-purpose.
- One run → `scripts/train.sh`; several → a jobs file + `scripts/train_queue.sh`.
- Do not wait for it, do not poll it, do not analyse it.

After the agent confirms the launch, tell me the run name, pid, log path and
rough ETA in **three lines or fewer**, then remind me the run survives `/clear`
and that I should clear the context now if this session's work is done.
