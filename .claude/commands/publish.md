---
description: Publish a finished run — evaluate at N=100, then DRAFT the README row and ROADMAP entry for approval (never commits)
allowed-tools: Bash, Read, Edit, Glob, Grep, Agent
---

Publish: $ARGUMENTS

The publication ritual, which is otherwise five manual steps repeated per
scenario. **This command never commits.** Published numbers are the repo's
credibility; I read the draft before it lands.

1. **Mechanics** — delegate to the **publish-ops** agent (haiku): checkpoint
   loadability, the N=100 evaluation, `behavior.json`, `probe.json`. It returns
   numbers only.
   - If the checkpoint cannot load under the current spaces, stop and say so.
     After a breaking cycle that is the expected answer, not a failure to
     work around.

2. **Judge before drafting** — at full strength, and say it plainly:
   - Does this beat, match, or miss the run it replaces? Compare against the
     **CI**, not the point estimate: overlapping CIs are not an improvement.
   - Do the behavioral metrics agree with the success rate? A success gain
     with `false_complete_rate` or `human_death_rate` worsening is a trade to
     state, not to bury.
   - Do the regression gates pass? A gate failure means it does not publish.
   - **If it misses, say so and stop.** This repo's standard is that misses
     ship with numbers and a diagnosis in the progress log — not silence, and
     not a quiet republication of the old number.

3. **Draft, do not apply**, and show me both as diffs:
   - the README results-table row (N, success, CI, in the existing format)
   - the ROADMAP progress-log entry, dated, in the existing voice: what was
     tested, what the number is, what moved in the behavior suite, and the
     honest caveat if there is one
   - if a previous checkpoint is being superseded, what happens to it

4. Then **apply it and commit**, and show me the diff in your reply. This step
   used to stop and ask; it does not need to any more, because the things the
   ask was protecting are now enforced by machinery: the README table is
   GENERATED from the committed evaluations (`scripts/results_table.py`, with
   `tests/test_results_table.py` failing on drift), `scripts/baseline.py` gates
   the fleet on N, gates, stability, provenance and per-artifact digests, and
   the digests catch a number changing under a published claim.

   What still stops and asks: a MISS. This repo's standard is that misses ship
   with numbers and a diagnosis, and deciding whether a miss supersedes an
   incumbent is a judgement about the project's claims, not about a number.

Never pull `metrics.csv`, `tb/`, or a full training log into this context.
