#!/usr/bin/env python
"""The baseline fleet: what it is, and whether it still holds together.

    scripts/baseline.py              # audit the manifest, table + verdict
    scripts/baseline.py --seal       # stamp the commit the fleet was trained at

A *baseline* here is one run per doctrine scenario that a reader can treat as a
single system: same code, same prices, same evidence standard. The fleet this
replaces was not that. Its eight champions sat at seven different commits, four
of them only reproduced with a ``--reward`` override that had been an
experiment's variable, and one (``fireteam_v8``) was published with a flag
saying it did not clear the bar. Every individual number was honest; the set was
not a system.

The manifest (``runs/BASELINE.json``) names the members. This module is the gate
over it, and the checks are the definition:

* **coverage**   — every doctrine scenario has a member. A scenario that is not
                   in the manifest and not in ``NOT_BASELINE`` fails the audit,
                   so a new scenario cannot quietly escape the fleet.
* **provenance** — every member records the same ``git_commit``. This is the
                   check the fleet could not previously pass, and the reason
                   ``run_report --vs`` was blind to code confounds for so long.
* **purity**     — no member carries a ``--reward`` override. What ships is what
                   was trained; if a scenario needs an override to work, that is
                   a finding about the defaults, not a launch flag.
* **evidence**   — a FINAL-policy evaluation at N >= 100. Not the best
                   checkpoint: the headline is the policy the run ended with.
* **gates**      — no failed regression gate on any member.
* **stability**  — best-final give-back under ``PUBLISH_STABILITY_POINTS``, the
                   bar ``publish_audit.py --validate`` showed predicts signed
                   overstatement at r = 0.564, p = 0.015.
* **loadable**   — the checkpoint loads under the current spaces. A baseline
                   whose weights no longer load is a historical artifact.
* **announced**  — every success announced on the net (the v1.19 guarantee).
                   Complete by construction, so a miss here means the guarantee
                   broke, not that a policy got shy.

Exit status is 0 only if every check passes on every member — this is meant to
be run before publishing anything that calls itself the baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
MANIFEST = RUNS / "BASELINE.json"
sys.path.insert(0, str(ROOT))

from scripts.run_report import PUBLISH_STABILITY_POINTS  # noqa: E402

MIN_EPISODES = 100

# The scenarios a deployable baseline must cover: the eight that are doctrine
# rather than instrumentation.
DOCTRINE_SCENARIOS = [
    "fireteam",
    "fireteam_defend",
    "squad",
    "squad_recon",
    "squad_screen",
    "patrol_brique",
    "defend_brique",
    "platoon",
]

# The rest, each with the reason it is not a baseline member. These are arms of
# experiments: they exist to be compared against a doctrine scenario, and three
# of them are *designed* to score worse than it. Publishing them as part of a
# fleet would advertise a deliberately crippled cohort as a product.
NOT_BASELINE = {
    "squad_nomask": "B3 ablation arm — doctrine masks removed, exists to be worse",
    "squad_flat": "B3 ablation arm — no chain of command at all, exists to be worse",
    "squad_short_vision": "information-asymmetry probe, not a shipping configuration",
    "squad_screen_core": "observation-width bisect arm, superseded by the bisect's answer",
}


def load() -> dict:
    try:
        return json.loads(MANIFEST.read_text())
    except (OSError, json.JSONDecodeError):
        return {"version": None, "commit": None, "runs": {}}


def _run_facts(run: str) -> dict:
    """Everything the audit needs about one member, with absences named."""
    d = RUNS / run
    facts: dict = {"run": run, "exists": d.is_dir(), "problems": []}
    if not facts["exists"]:
        facts["problems"].append("no run directory")
        return facts

    econ = d / "economics.json"
    if econ.exists():
        e = json.loads(econ.read_text())
        facts["commit"] = e.get("git_commit")
        facts["overrides"] = list(e.get("reward_overrides") or [])
    else:
        facts["commit"] = None
        facts["overrides"] = []
        facts["problems"].append("no economics.json — provenance unknown")

    final = d / "behavior_final.json"
    if final.exists():
        b = json.loads(final.read_text())
        m = b.get("metrics", {})
        facts["episodes"] = b.get("episodes", 0)
        facts["success"] = m.get("success_rate")
        facts["ci"] = b.get("success_ci95")
        facts["gates_failed"] = [g["name"] for g in b.get("gates", []) if not g.get("passed", True)]
        facts["successes"] = m.get("successes")
        facts["announced"] = m.get("successes_announced")
    else:
        facts["episodes"] = 0
        facts["problems"].append("no behavior_final.json — the FINAL policy was never scored")

    if facts["overrides"]:
        facts["problems"].append(f"reward overrides: {', '.join(facts['overrides'])}")
    if facts.get("episodes", 0) < MIN_EPISODES:
        facts["problems"].append(f"evaluated at N={facts.get('episodes', 0)}, needs {MIN_EPISODES}")
    if facts.get("gates_failed"):
        facts["problems"].append(f"gate failed: {', '.join(facts['gates_failed'])}")

    ann, succ = facts.get("announced"), facts.get("successes")
    if ann is not None and succ is not None and ann < succ:
        facts["problems"].append(f"announced {ann}/{succ} — the v1.19 guarantee is broken")

    from scripts.publish_audit import audit_run

    a = audit_run(d)
    if a:
        facts["gap"] = a["gap"]
        if a["gap"] >= PUBLISH_STABILITY_POINTS:
            facts["problems"].append(
                f"gave back {a['gap']:.1f} pts between peak and final decile"
            )
    return facts


def _loadable(run: str) -> bool:
    from cohort.viz.dashboard import checkpoint_meta

    ckpt = RUNS / run / "ckpt_best.pt"
    if not ckpt.exists():
        return False
    try:
        return bool(checkpoint_meta(str(ckpt)).get("loadable"))
    except Exception:
        return False


def audit(check_loadable: bool = True) -> int:
    manifest = load()
    members = manifest.get("runs", {})
    where = MANIFEST if not MANIFEST.is_relative_to(ROOT) else MANIFEST.relative_to(ROOT)
    print(f"baseline {manifest.get('version') or '(unversioned)'} — "
          f"{len(members)} members, manifest {where}\n")

    failures: list[str] = []

    missing = [s for s in DOCTRINE_SCENARIOS if s not in members]
    if missing:
        failures.append(f"coverage: no member for {', '.join(missing)}")
    stray = [s for s in members if s not in DOCTRINE_SCENARIOS]
    if stray:
        failures.append(f"coverage: {', '.join(stray)} is not a doctrine scenario")

    rows = []
    for scenario in DOCTRINE_SCENARIOS:
        run = members.get(scenario)
        if not run:
            rows.append((scenario, "—", None, "MISSING"))
            continue
        f = _run_facts(run)
        if check_loadable and f["exists"] and not _loadable(run):
            f["problems"].append("checkpoint does not load under the current spaces")
        rows.append((scenario, run, f, "OK" if not f["problems"] else "FAIL"))
        for p in f["problems"]:
            failures.append(f"{run}: {p}")

    commits = {f["commit"] for _, _, f, _ in rows if f and f.get("commit")}
    if len(commits) > 1:
        failures.append(
            f"provenance: {len(commits)} distinct commits across the fleet — "
            "these runs are not one system"
        )
    sealed = manifest.get("commit")
    if sealed and commits and {sealed} != commits:
        failures.append(
            f"provenance: manifest is sealed at {sealed[:8]} but the runs carry "
            f"{', '.join(sorted(c[:8] for c in commits))}"
        )

    print(f"{'scenario':<18}{'run':<24}{'N':>5}{'success':>16}{'gap':>7}  status")
    for scenario, run, f, status in rows:
        if not f:
            print(f"{scenario:<18}{run:<24}{'':>5}{'':>16}{'':>7}  MISSING")
            continue
        gap = f"{f['gap']:.1f}" if f.get("gap") is not None else "—"
        print(f"{scenario:<18}{run:<24}{f.get('episodes', 0):>5}"
              f"{f.get('ci') or '—'!s:>16}{gap:>7}  {status}")

    print()
    if commits:
        print(f"commit: {'  '.join(sorted(c[:8] for c in commits))}"
              f"{'   (one system)' if len(commits) == 1 else '   (NOT one system)'}")
    if not failures:
        print("\nBASELINE OK — every member same commit, no overrides, N>=100 final "
              "policy, gates green, stable, loadable, every win announced.")
        return 0
    print(f"\nBASELINE NOT READY — {len(failures)} problem(s):")
    for f in failures:
        print(f"  · {f}")
    return 1


def seal(version: str | None = None) -> int:
    """Record the commit the members actually carry, so drift is detectable."""
    manifest = load()
    commits = set()
    for run in manifest.get("runs", {}).values():
        econ = RUNS / run / "economics.json"
        if econ.exists():
            c = json.loads(econ.read_text()).get("git_commit")
            if c:
                commits.add(c)
    if len(commits) != 1:
        print(f"refusing to seal: {len(commits)} distinct commits across the fleet")
        return 1
    manifest["commit"] = commits.pop()
    if version:
        manifest["version"] = version
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"sealed {manifest.get('version')} at {manifest['commit'][:8]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seal", action="store_true",
                    help="stamp the manifest with the commit its members carry")
    ap.add_argument("--version", help="set the baseline version while sealing")
    ap.add_argument("--no-loadable", action="store_true",
                    help="skip the checkpoint-load check (it imports torch)")
    args = ap.parse_args()
    if args.seal:
        return seal(args.version)
    return audit(check_loadable=not args.no_loadable)


if __name__ == "__main__":
    raise SystemExit(main())
