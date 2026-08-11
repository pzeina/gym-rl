# Baseline v1.19 — the homogeneous fleet

**What this campaign is.** One run per doctrine scenario, all trained from the
same commit, all on the shipped reward defaults, all on the same seed. It exists
because the fleet it replaces was not one thing: every champion sat at a
different commit, and four of them only reproduce with a `--reward` override
that was the experiment's variable rather than the product's setting.

Before (published champions, `economics.json:git_commit`):

    fireteam_v8              9933a3a   fireteam_defend_v19   e91b753
    squad_v9                 48716cc   defend_brique_v14     e91b753
    squad_recon_v7           4395c12   patrol_brique_v5      0e3cf43
    squad_screen_fallen_v2   a0649de   platoon_v5            6571b70

Seven commits across eight scenarios. A number from one and a number from
another were never quite comparable, and `run_report --vs` could not see the
difference because it diffs `economics.json`'s prices — which a code change
never touches.

## The rules this campaign holds to

1. **One environment.** Every run's `cohort/` tree — resolved from its recorded
   `economics.json:git_commit` — is identical, and `scripts/baseline.py` fails
   the set if it is not. Not commit equality: `fireteam_v9` was pulled out of
   lane A and relaunched three tooling commits later, and the `cohort/` tree was
   byte-identical (`5f848fb`) across all of them. A gate that failed on that
   would be teaching its readers to ignore it.
2. **No overrides.** No `--reward` anywhere in these files. The three settings
   the defend family used to pass on the command line
   (`defend_survivor_scale=0.35`, `root_done_bonus_first_claim_only=false`,
   `done_false=-0.5`) are the tree's defaults now, so the published policy and
   the shipped config are the same object. If a scenario needs an override to
   work, that is a finding about the defaults, not a launch flag.
3. **One seed** (12). Not because one seed is enough evidence — it is not — but
   because the seed should not be a free parameter that quietly varies with how
   a scenario happened to be tuned. Where a run lands weak, the answer is a
   named confirmation seed, logged.
4. **Steps sized to the scenario, and stated.** `fireteam` goes to 3.5M rather
   than the 2.5M that left `fireteam_v8` 12 points short of its own peak;
   `squad_screen` to 2.5M; everything else keeps the budget that produced its
   published result.

## Lanes

Three queues in parallel on a 10-core box, balanced to ~2.5h each rather than
~7.5h serially. Lane A leads with `platoon` because at 16 agents it is the long
pole and everything else can hide behind it.

    lane A   platoon_v6 (3.0M)  →  fireteam_v9 (3.5M)
    lane B   squad_v10 (3.0M)   →  patrol_brique_v6 (3.0M)  →  squad_screen_v11 (2.5M)
    lane C   squad_recon_v8 (3.0M) →  defend_brique_v15 (3.0M) →  fireteam_defend_v20 (3.5M)

Launch:

    scripts/train_queue.sh campaigns/baseline_v1_19_laneA.jobs
    scripts/train_queue.sh campaigns/baseline_v1_19_laneB.jobs
    scripts/train_queue.sh campaigns/baseline_v1_19_laneC.jobs

## What it is measured against

Each run's published predecessor, at both checkpoints, at N=100. The v1.19
change under test is the ENDEX guarantee — rollout-neutral by construction, so
success rates are expected to *match* their predecessors and the announcement
column is expected to go to 100% everywhere. A success-rate move on any scenario
is therefore a finding, not a win: it means something other than the announcement
changed.
