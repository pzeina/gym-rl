---
description: Refresh the three boards and republish whichever have drifted
allowed-tools: Bash(.venv/bin/python scripts/update_boards.py:*), Artifact
---

The HTML is refreshed automatically whenever a run lands (`scripts/train.sh`
wraps every launch in `scripts/train_then_boards.sh`). This command exists for
the one step a shell cannot do: pushing the boards to claude.ai.

1. Run `.venv/bin/python scripts/update_boards.py`.
2. If it prints "published artifacts are current", say so in one line and stop.
3. Otherwise publish each pending board with the **Artifact** tool, passing the
   `url` recorded in `runs/.boards.json` — without it a new artifact is minted
   instead of updating the existing one:
   - `fleet` → `runs/fleet_board.html`, favicon 📡
   - `program` → `runs/program_board.html`, favicon 🧭
   - `gallery` → `runs/scenario_gallery.html`, favicon 📻
   Keep each favicon and title exactly as they are; a changed favicon reads to
   the user as a different page.
4. Run `.venv/bin/python scripts/update_boards.py --mark-published`.
5. Report in two lines: what actually changed (which runs appeared, which
   numbers moved) and the URLs.

Do not read the generated HTML, `metrics.csv`, or any log — the digest line from
step 1 is the summary. The boards are generated: if a number looks wrong, fix
`scripts/fleet_board.py`, `scripts/program_board.py` or
`scripts/scenario_gallery.py`, never the HTML.
