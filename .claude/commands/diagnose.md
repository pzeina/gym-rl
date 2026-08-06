---
description: Oracle-diagnose a checkpoint's behavior before changing rewards (this repo's rule: diagnose first)
allowed-tools: Bash, Read, Glob, Grep, Agent
---

Oracle-diagnose: $ARGUMENTS

`CLAUDE.md` requires diagnosis before any reward or scenario change. This is
that step. Arguments are a run name (or checkpoint path), optionally `vs
<baseline-run>`.

1. Get the fact-sheet via the **oracle-diagnose** agent (haiku). Default
   protocol is `--episodes 30 --seed 500`, matching every previous campaign's
   seed block — **keep it** unless I say otherwise, or the numbers stop being
   comparable to the diagnoses already in the ROADMAP.
   Always pass `--vs` when a baseline exists; a rate with nothing to compare
   against is nearly useless here.

2. Then do the thinking here, at full strength. The question is always
   **mechanism**, not score:
   - *When it could have fought, did it?* — fire rate under threat, split
     human / leader / rifleman. A gap between the commander and its own
     riflemen is the signature that broke v6 open (0.005 vs 0.97).
   - *Where did the fight happen?* — cover occupancy and distance from the
     root objective under threat, vs what the mission presumes.
   - *What were they doing while threatened?* — the mission mix. An ADVANCE
     share during a DEFEND is the v7 signature.
   - *Where did they die?* — at the objective vs in the open.
   - For preparation-period scenarios: in-cover-at-objective during prep is
     the prep-period metric; off-objective fight distance is the
     occupancy-pay metric. **Those two separate the v1.10 defend cycle's two
     variables** — report them separately, never as one verdict.

3. State the mechanism as a falsifiable claim, and name the measurement that
   would refute it. If the evidence does not support a single mechanism, say
   so — "no mechanism identified" is a valid and useful result, and far better
   than a plausible story.

4. Only then propose a change, at most one, and say which measurement should
   move if it works. Do not launch anything.

Never pull `metrics.csv`, `tb/`, or a full training log into this context —
use `/train-report` for the learning curve, this for behavior.
