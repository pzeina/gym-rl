# ASSURANCE-SYNC — instructions for the Claude session working on this repo

You are the session developing **gym-rl** (this repository). An external
assurance layer — EPISTREAM-RL at `~/Documents/epistream-rl`, holding a fork
of this repo at `~/Documents/gym-rl-fork` (branch `assurance-integration`) —
monitors this system from the outside, reviews it, files feature requests as
GitHub issues, and re-measures after fixes. This file defines your side of
that workflow. Read it once, do §1 now, then follow §2 and §3 for the rest
of the project's life.

## 1. One-time setup — install the sync hook (do this now)

A ready-made Stop hook exists at
`/Users/pzeinaty/Documents/epistream-rl/scripts/gymrl-task-hook.sh`
(versioned there; read it — it is short). It does two things at each of your
turn ends: queues a "task done" event for the assurance layer when new
commits exist, and tells you — once per issue — when new GitHub issues need
implementing.

Create (or merge into) `.claude/settings.local.json` **in the project root
directory this session actually runs in** — if `/hooks` said "0 hooks
configured", the likeliest cause is that a previous install landed in a
directory that is not your session's root; installing it yourself from
inside the session fixes that:

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/Users/pzeinaty/Documents/epistream-rl/scripts/gymrl-task-hook.sh",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

Then ask your user to restart the session (or approve the hook via
`/hooks`) — hook configs load at session start. Verify afterwards: `/hooks`
must show one Stop hook. If the hook cannot be activated, use the manual
fallback in §4 instead — the workflow must not silently lapse.

## 2. Feature requests → one small dedicated parallel agent

When you learn of new open issues on `pzeina/gym-rl` (the hook will block
one of your stops with the list; or you find them via the check below):

- Spawn **ONE small dedicated parallel agent** (Agent tool,
  `run_in_background: true`) to implement them, so your main thread stays
  free for your user.
- The agent reads each issue, implements and tests the fix, and makes one
  clearly-scoped commit per issue whose message references it (e.g.
  `refs #5`).
- **Never close the issues** — closing is the assurance layer's job, done
  only after it has re-measured the fix from the outside. Comment on the
  issue if useful; do not close.

Currently open: **#3–#8**, all findings of the external assurance review —
full background in `~/Documents/gym-rl-fork/docs/assurance-review.md`
(worth reading before fixing; it contains measured evidence and fix
directions for each).

Check for open issues at any time:

```sh
curl -s "https://api.github.com/repos/pzeina/gym-rl/issues?state=open&per_page=100"
```

## 3. Task done → signal the assurance layer

The signal is **commit-based**: finishing a task means committing it. With
the hook active, nothing else is needed — the hook notices new commits and
queues the event; the assurance layer then pulls your changes into its fork,
re-runs its measurements, adapts to new features, and (for issue fixes)
verifies and closes the issues.

## 4. Manual fallback — only if the hook cannot be activated

After each completed (committed) task, append one event line yourself:

```sh
printf '{"ts":"%s","repo":"gym-rl","branch":"%s","head":"%s","subject":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(git rev-parse --abbrev-ref HEAD)" \
  "$(git rev-parse --short HEAD)" \
  "$(git log -1 --format=%s | tr -d '"\\' | cut -c1-120)" \
  >> /Users/pzeinaty/Documents/gym-rl-sync/queue.jsonl
```

And at session start (plus occasionally after tasks), run the issue check
from §2; compare against
`/Users/pzeinaty/Documents/gym-rl-sync/dispatched-issues` (one issue number
per line — create it if absent), append any new numbers there, and spawn the
§2 agent for them.

## 5. Contract summary

- **You own this repository** — change whatever it needs; the assurance
  layer adapts to you, never the other way around.
- **Do not touch `~/Documents/gym-rl-fork`** — the assurance layer merges
  from your repo into its fork itself.
- **Never close assurance issues** — fix, commit with `refs #N`, and leave
  the verdict to the re-measurement.
- **Do not delete `~/Documents/gym-rl-sync/`** — it is the durable mailbox
  between the two sessions (`queue.jsonl`: your task-done events;
  `dispatched-issues`: which issues were already handed to a fix agent;
  `state`: the hook's commit baseline).
