#!/usr/bin/env python
"""The baseline fleet: what it is, and whether it still holds together.

    scripts/baseline.py              # audit the manifest, table + verdict
    scripts/baseline.py --seal       # stamp the tree, commits and evaluations

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
                   The best checkpoint's evaluation is held to the same N,
                   because the README publishes it too (issue #45). It is the
                   *peak* column, honestly labelled as a peak — but a peak read
                   off five episodes and a peak read off a hundred are not the
                   same claim, and until this rule existed only the caption
                   could tell them apart.
* **sealed**     — every published evaluation still hashes to what ``--seal``
                   recorded. ``cohort_tree`` pins the environment and
                   ``checkpoint_sha256`` pins the weights; this pins the numbers
                   derived from them, which is the side the other two leave
                   open. A re-score is not an accusation — it is a reason to
                   re-seal.
* **gates**      — no failed regression gate on any member.
* **stability**  — best-final give-back under ``PUBLISH_STABILITY_POINTS``, the
                   bar ``publish_audit.py --validate`` showed predicts signed
                   overstatement at r = 0.564, p = 0.015.
* **loadable**   — both checkpoints load under the current spaces. A baseline
                   whose weights no longer load is a historical artifact.
* **committed**  — and both are in the repository. The headline is computed from
                   ``ckpt_latest.pt``, so a clone that does not carry those bytes
                   can read every published figure and reproduce none of them
                   (issue #44). A number without its weights is a claim.
* **announced**  — every success announced on the net (the v1.19 guarantee).
                   Complete by construction, so a miss here means the guarantee
                   broke, not that a policy got shy.

Exit status is 0 only if every check passes on every member — this is meant to
be run before publishing anything that calls itself the baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
MANIFEST = RUNS / "BASELINE.json"
sys.path.insert(0, str(ROOT))

from cohort.metrics import ROOT_REPORT_CLOSE_FLOOR, split_gates  # noqa: E402
from scripts.fleet_status import find_run  # noqa: E402
from scripts.run_report import PUBLISH_STABILITY_POINTS  # noqa: E402


def run_dir(name: str):
    """Resolve against THIS module's RUNS, which the tests point at a fixture."""
    return find_run(name, RUNS) or RUNS / name

MIN_EPISODES = 100

# The two evaluations the README publishes for every member, and therefore the
# two this module gates and seals. ``behavior_final.json`` is the headline (the
# policy the run ended with); ``behavior.json`` is the peak column, scored from
# ``ckpt_best.pt``. Both are *derived* — they are what the weights and the
# environment produce — and the derived side is where nothing was digested.
PUBLISHED_EVALUATIONS = ("behavior_final.json", "behavior.json")


def artifact_digest(path: Path) -> str | None:
    """sha256 of one published evaluation, or None when it is not there.

    Absence answers None rather than raising, so a missing artifact reads as a
    named finding at the call site instead of a traceback in a gate.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


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


def cohort_tree(commit: str | None) -> str | None:
    """The hash of ``cohort/`` as of a commit — the provenance that matters.

    "Same commit" is the wrong bar and this campaign proved it within an hour:
    ``fireteam_v9`` was launched three commits after its lane-mates, and all
    three commits were tooling — scripts, tests, a README table. The tree under
    ``cohort/`` was byte-identical across every one of them
    (``5f848fb`` throughout), so the four runs trained in the same environment
    and a commit-equality check would have failed the fleet for a reason that
    has nothing to do with the runs.

    The converse is what the check must catch: two members whose ``cohort/``
    trees differ trained against different environments and cannot be read as
    one system, however adjacent their commits look.

    Derived from git rather than recorded at train time, so it works for every
    run already on disk. A commit this clone does not have reads as None, which
    the audit reports as unknown and never as agreement.
    """
    if not commit:
        return None
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", f"{commit}:cohort"], cwd=ROOT,
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _seal_drift(run: str, sealed: dict | None) -> list[str]:
    """How this member's published evaluations differ from what was sealed.

    Issue #45. ``platoon_v6``'s ``behavior.json`` was overwritten at N=5 by a
    spot-check (``cohort.training.evaluate`` writes it by DEFAULT) and committed
    in ``a321329``. For that whole window ``runs/BASELINE.json`` was
    byte-identical — same ``cohort_tree``, same ``checkpoint_sha256`` — because
    the environment and the weights had not moved. Only the number derived from
    them had, and nothing in the tree digested that. This gate printed
    ``BASELINE OK`` with output byte-identical to the repaired tree's.

    So the seal now covers the derived side too, and drift in a sealed member is
    detectable from the tree alone at any later date, by anyone, without a
    running campaign to compare against.

    ``sealed`` is the manifest's whole stamp, or None for a manifest written
    before stamping existed — silence there, because an unstamped manifest has
    no opinion, which is not the same finding as disagreement.
    """
    if sealed is None:
        return []
    mine = sealed.get(run)
    if mine is None:
        return [f"the manifest is sealed and stamps no evaluation for {run} — "
                "this member joined the fleet after the seal; re-seal it"]
    problems = []
    for name, want in sorted(mine.items()):
        got = artifact_digest(run_dir(run) / name)
        if got is None:
            problems.append(f"{name} was sealed at {want[:12]} and is not on disk")
        elif got != want:
            problems.append(
                f"{name} changed since the seal ({want[:12]} -> {got[:12]}) — "
                "if the re-score was intended, re-seal; if it was a spot-check, "
                "it just overwrote a published number"
            )
    return problems


def _run_facts(run: str, sealed: dict | None = None) -> dict:
    """Everything the audit needs about one member, with absences named."""
    d = run_dir(run)
    facts: dict = {"run": run, "exists": d.is_dir(), "problems": []}
    if not facts["exists"]:
        facts["problems"].append("no run directory")
        return facts

    econ = d / "economics.json"
    if econ.exists():
        e = json.loads(econ.read_text())
        facts["commit"] = e.get("git_commit")
        facts["cohort_tree"] = cohort_tree(facts["commit"])
        facts["overrides"] = list(e.get("reward_overrides") or [])
    else:
        facts["commit"] = None
        facts["cohort_tree"] = None
        facts["overrides"] = []
        facts["problems"].append("no economics.json — provenance unknown")

    final = d / "behavior_final.json"
    if final.exists():
        b = json.loads(final.read_text())
        m = b.get("metrics", {})
        facts["episodes"] = b.get("episodes", 0)
        facts["success"] = m.get("success_rate")
        facts["ci"] = b.get("success_ci95")
        facts["gates_failed"], facts["gates_unmeasured"] = split_gates(b.get("gates", []))
        facts["successes"] = m.get("successes")
        facts["announced"] = m.get("successes_announced")
    else:
        facts["episodes"] = 0
        facts["problems"].append("no behavior_final.json — the FINAL policy was never scored")

    # The peak column the README publishes, held to the same evidence bar. Its
    # absence is a finding rather than a silence: audit_run returns None without
    # it, so the give-back gate below simply would not run.
    best = d / "behavior.json"
    if best.exists():
        facts["peak_episodes"] = json.loads(best.read_text()).get("episodes", 0)
        if facts["peak_episodes"] < MIN_EPISODES:
            facts["problems"].append(
                f"peak evaluated at N={facts['peak_episodes']}, needs {MIN_EPISODES} — "
                "the README publishes this cell"
            )
    else:
        facts["peak_episodes"] = 0
        facts["problems"].append(
            "no behavior.json — the peak the README publishes was never scored, "
            "and the give-back gate cannot run without it"
        )

    facts["problems"].extend(_seal_drift(run, sealed))

    if facts["overrides"]:
        facts["problems"].append(f"reward overrides: {', '.join(facts['overrides'])}")
    if facts.get("episodes", 0) < MIN_EPISODES:
        facts["problems"].append(f"evaluated at N={facts.get('episodes', 0)}, needs {MIN_EPISODES}")
    if facts.get("gates_failed"):
        facts["problems"].append(f"gate failed: {', '.join(facts['gates_failed'])}")
    # Still a problem for a member — "every gate green" cannot be claimed off a
    # gate that never read anything — but it is a different problem, and saying
    # "failed" for it would be the overstatement this gate exists to prevent.
    if facts.get("gates_unmeasured"):
        facts["problems"].append(f"gate unmeasured: {', '.join(facts['gates_unmeasured'])}")

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


HEADLINE_CKPT = "ckpt_latest.pt"  # the FINAL policy — what behavior_final.json scored
CHECKPOINTS = ("ckpt_best.pt", HEADLINE_CKPT)


def _reporting_gate(run: str) -> bool | None:
    """Does this run pass ``closed_on_root_report_rate`` on the FINAL policy?

    v1.21 fixed which checkpoint the reporting gate reads, and the answer is the
    artifact the project publishes. Measured against the 0.5 floor, the shipping
    v1.19 fleet fails its own bar on ``ckpt_best`` in two members of eight —
    ``patrol_brique_v6`` at 0.000, ``platoon_v6`` at 0.021 — and passes on the
    final policy at a minimum of 0.808. A bar that retroactively fails the fleet
    it was written to protect is reading the wrong artifact. ``ckpt_best`` was
    also, until 504e87b, selected by a rule that could latch on a 2%-success
    window (assurance #57).

    ``None`` when the rate was never measured, which is not the same as failed.
    """
    data = _json_or_empty(run_dir(run) / "behavior_final.json")
    rate = (data.get("metrics") or {}).get("closed_on_root_report_rate")
    return None if rate is None else rate >= ROOT_REPORT_CLOSE_FLOOR


def _json_or_empty(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def seed_search_facts(manifest: dict, scenario: str, member: str | None) -> dict | None:
    """The declared seed search behind one member, or None if none was declared.

    **Why this is in the manifest at all.** ``closed_on_root_report_rate`` is a
    per-run bar over a quantity that is bimodal in the seed: across 14 matched
    ``patrol_brique`` runs the commander reports in 6, and the two modes are
    0.750-1.000 and exactly 0.000 with nothing between them. Where a scenario
    behaves that way the member is necessarily chosen from several seeds — and a
    member picked as the best of four, published as though it were the only one,
    is precisely the overstatement this module exists to refuse. So the search is
    declared: the manifest lists the candidate runs, and every number here is
    derived from their committed artifacts rather than kept by hand.

    The manifest holds only the run names. ``k of K`` is counted, and NOT
    reported as a rate with an interval — at K=4 a confidence interval would be
    a decoration over four coin flips, and the 0.43 the campaign measured comes
    from a 14-run corpus, not from this fleet.
    """
    candidates = (manifest.get("seed_search") or {}).get(scenario)
    if not candidates:
        return None
    rows = []
    for run in candidates:
        d = run_dir(run)
        cfg = _json_or_empty(d / "config.json")
        econ = _json_or_empty(d / "economics.json")
        rows.append({
            "run": run,
            "exists": d.exists(),
            "seed": cfg.get("seed"),
            "tree": cohort_tree(econ.get("git_commit")),
            "overrides": list(econ.get("reward_overrides") or []),
            "reports": _reporting_gate(run),
        })
    reporting = sum(1 for r in rows if r["reports"] is True)
    return {
        "scenario": scenario,
        "runs": rows,
        "reporting": reporting,
        "total": len(rows),
        "member_searched": member is not None and any(r["run"] == member for r in rows),
    }


def _seed_search_problems(facts: dict, member: str | None) -> list[str]:
    """What makes a declared search worth believing.

    Each of these is a way the published ``k of K`` could be true of nothing:
    a candidate that does not exist, one trained against a different environment
    or at a price the fleet does not ship, one that was never scored — and a
    member that is not among the seeds it was supposedly selected from.
    """
    problems = []
    if not facts["member_searched"]:
        problems.append(
            f"seed_search[{facts['scenario']}] does not contain the member "
            f"{member} — a search the published run was not part of"
        )
    trees = {r["tree"] for r in facts["runs"] if r["tree"]}
    if len(trees) > 1:
        problems.append(
            f"seed_search[{facts['scenario']}]: {len(trees)} distinct cohort/ trees — "
            "a reporting rate over runs that trained against different environments"
        )
    for r in facts["runs"]:
        if not r["exists"]:
            problems.append(f"seed_search[{facts['scenario']}]: no run directory for {r['run']}")
        elif r["overrides"]:
            problems.append(
                f"seed_search[{facts['scenario']}]: {r['run']} carries "
                f"{', '.join(r['overrides'])} — not the configuration the fleet ships"
            )
        elif r["reports"] is None:
            problems.append(
                f"seed_search[{facts['scenario']}]: {r['run']} has no measured "
                "closed_on_root_report_rate, so it counts in neither direction"
            )
    return problems


def _loadable(run: str) -> bool:
    """Do this member's checkpoints load under the current spaces?

    ``checkpoint_meta`` takes a Path and calls ``path.stat()`` on it. This
    passed ``str(ckpt)``, which raises ``AttributeError`` — and a bare
    ``except Exception`` turned that into "does not load" for **all eight
    members at once**, on a fleet whose every checkpoint had just been loaded
    to score it at N=100. A gate that fails for a reason unrelated to what it
    gates is worse than no gate: this one would have been read as a spaces
    break and sent someone hunting a retrain.

    So the type is right, and the except is narrow. A torch/pickle failure is a
    genuine "does not load"; a TypeError or an AttributeError is a bug in this
    function and must surface as one.

    It also checks BOTH checkpoints. It used to check ``ckpt_best.pt`` alone —
    the one checkpoint the ``evidence`` rule three docstrings up explicitly says
    is *not* the headline. A fleet whose final weights no longer load is a
    historical artifact whatever its best-window snapshot does.
    """
    from cohort.viz.dashboard import checkpoint_meta

    for name in CHECKPOINTS:
        ckpt = run_dir(run) / name
        if not ckpt.is_file():
            return False
        try:
            if not checkpoint_meta(ckpt).get("loadable"):
                return False
        except (OSError, RuntimeError, ValueError, KeyError):
            return False
    return True


def _uncommitted(run: str) -> list[str]:
    """Which of this member's checkpoints are absent from the repository.

    Issue #44. ``.gitignore``'s ``runs/*/ckpt_latest.pt`` ignored the final
    policy fleet-wide, so from a fresh clone the eight headline figures could be
    *read* in ``behavior_final.json`` and not one of them re-derived — the
    weights that produce them were the single artifact not committed. Worse, a
    ``*`` does not cross ``/``, so when 96 superseded runs moved to
    ``runs/archive/`` the rule stopped covering them and the runs nobody needs
    to reproduce became the only ones carrying both checkpoints.

    A number whose weights are not in the repository is a claim, not a result,
    so this is a gate rather than a note. It answers with ``[]`` — silence, not
    a failure — whenever git cannot answer at all: a tarball export, or the
    tmp_path fixtures the audit's own tests run against. An environment with no
    index has no opinion about what is committed, and a gate that fires there
    would fire for a reason that has nothing to do with the fleet.
    """
    import subprocess

    d = run_dir(run)
    present = [n for n in CHECKPOINTS if (d / n).is_file()]
    if not present:
        return []
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", *(str(d / n) for n in present)],
            cwd=ROOT, capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    tracked = {line.rsplit("/", 1)[-1] for line in out.stdout.split()}
    return [n for n in present if n not in tracked]


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
        f = _run_facts(run, manifest.get("artifacts"))
        if check_loadable and f["exists"] and not _loadable(run):
            f["problems"].append("checkpoint does not load under the current spaces")
        if f["exists"]:
            absent = _uncommitted(run)
            if absent:
                f["problems"].append(
                    f"{', '.join(absent)} is not committed — a reader can see this "
                    "run's headline and cannot re-derive it"
                )
        rows.append((scenario, run, f, "OK" if not f["problems"] else "FAIL"))
        for p in f["problems"]:
            failures.append(f"{run}: {p}")

    searches = {}
    for scenario in DOCTRINE_SCENARIOS:
        s = seed_search_facts(manifest, scenario, members.get(scenario))
        if s is None:
            continue
        searches[scenario] = s
        failures.extend(_seed_search_problems(s, members.get(scenario)))

    commits = {f["commit"] for _, _, f, _ in rows if f and f.get("commit")}
    trees = {f["cohort_tree"] for _, _, f, _ in rows if f and f.get("cohort_tree")}
    unknown = [run for _, run, f, _ in rows
               if f and f.get("commit") and not f.get("cohort_tree")]
    if len(trees) > 1:
        failures.append(
            f"provenance: {len(trees)} distinct cohort/ trees across the fleet — "
            "these runs trained against different environments"
        )
    if unknown:
        failures.append(
            f"provenance: cannot resolve cohort/ for {', '.join(unknown)} — "
            "unknown is not the same finding as identical"
        )
    sealed = manifest.get("cohort_tree")
    if sealed and trees and {sealed} != trees:
        failures.append(
            f"provenance: manifest is sealed at cohort/ {sealed[:8]} but the runs "
            f"carry {', '.join(sorted(t[:8] for t in trees))}"
        )

    print(f"{'scenario':<18}{'run':<24}{'N':>5}{'success':>16}{'gap':>7}  status")
    for scenario, run, f, status in rows:
        if not f:
            print(f"{scenario:<18}{run:<24}{'':>5}{'':>16}{'':>7}  MISSING")
            continue
        gap = f"{f['gap']:.1f}" if f.get("gap") is not None else "—"
        print(f"{scenario:<18}{run:<24}{f.get('episodes', 0):>5}"
              f"{f.get('ci') or '—'!s:>16}{gap:>7}  {status}")

    # Printed for every member, not only the searched ones: "1 seed" and "1 of 4
    # seeds" are different claims, and a reader who sees the second only where it
    # is flattering learns nothing from its absence.
    print("\nreporting channel — closed_on_root_report_rate on the FINAL policy, "
          f"floor {ROOT_REPORT_CLOSE_FLOOR:g}")
    for scenario in DOCTRINE_SCENARIOS:
        s = searches.get(scenario)
        if s is None:
            member = members.get(scenario)
            passes = _reporting_gate(member) if member else None
            verdict = "—" if passes is None else "reports" if passes else "MUTE"
            print(f"  {scenario:<18} 1 seed, not searched      {verdict}")
            continue
        seeds = ", ".join(str(r["seed"]) for r in s["runs"])
        print(f"  {scenario:<18} {s['reporting']} of {s['total']} seeds report   (seeds {seeds})")
        for r in s["runs"]:
            mark = "  <- member" if r["run"] == members.get(scenario) else ""
            verdict = "—" if r["reports"] is None else "reports" if r["reports"] else "mute"
            print(f"      seed {r['seed']!s:<4} {r['run']:<32} {verdict}{mark}")

    print()
    if trees:
        verdict = "one environment" if len(trees) == 1 else "NOT one environment"
        print(f"cohort/ tree: {'  '.join(sorted(t[:8] for t in trees))}   ({verdict})")
    if commits:
        # Printed, never gated on. Tooling commits between two launches are
        # routine and say nothing about what the runs trained against.
        note = "" if len(commits) == 1 else "   (tooling-only differences are expected)"
        print(f"commits:      {'  '.join(sorted(c[:8] for c in commits))}{note}")
    if not failures:
        print("\nBASELINE OK — every member on the same cohort/ tree, no overrides, "
              "N>=100 on both published evaluations, unchanged since the seal, "
              "gates green, stable, loadable, committed, every win announced.")
        return 0
    print(f"\nBASELINE NOT READY — {len(failures)} problem(s):")
    for f in failures:
        print(f"  · {f}")
    return 1


def seal(version: str | None = None) -> int:
    """Record the environment the members trained against, so drift is detectable.

    The seal is the ``cohort/`` tree, not the commit: a member launched a
    tooling commit after its lane-mates is the same system, and a member
    launched across an env change is not, however close the shas look. The
    commits are recorded alongside as provenance, never as the gate.

    It also stamps a digest of every published evaluation (issue #45). The tree
    already pinned the environment and the weights; those two held perfectly
    through the one corruption this fleet actually had, because what moved was
    the *number derived from them*. Re-scoring a member is a normal thing to do
    and it invalidates the seal by design — ``--seal`` again and the manifest
    says so out loud, which is the whole point of a seal.
    """
    manifest = load()
    commits, trees = set(), set()
    for run in manifest.get("runs", {}).values():
        econ = run_dir(run) / "economics.json"
        if econ.exists():
            c = json.loads(econ.read_text()).get("git_commit")
            if c:
                commits.add(c)
                trees.add(cohort_tree(c))
    if len(trees) != 1 or None in trees:
        print(f"refusing to seal: {len(trees)} distinct cohort/ tree(s) across the fleet")
        return 1
    manifest["cohort_tree"] = trees.pop()
    manifest["commits"] = sorted(commits)
    manifest["commit"] = manifest["commits"][0] if len(commits) == 1 else None
    if version:
        manifest["version"] = version

    artifacts: dict[str, dict[str, str]] = {}
    for run in manifest.get("runs", {}).values():
        d = run_dir(run)
        artifacts[run] = {name: digest for name in PUBLISHED_EVALUATIONS
                          if (digest := artifact_digest(d / name)) is not None}
    manifest["artifacts"] = artifacts

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    stamped = sum(len(v) for v in artifacts.values())
    print(f"sealed {manifest.get('version')} at cohort/ {manifest['cohort_tree'][:8]} "
          f"({len(commits)} commit(s), {stamped} evaluation digest(s))")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seal", action="store_true",
                    help="stamp the manifest with its members' cohort/ tree, commits "
                         "and a digest of every published evaluation")
    ap.add_argument("--version", help="set the baseline version while sealing")
    ap.add_argument("--no-loadable", action="store_true",
                    help="skip the checkpoint-load check (it imports torch)")
    args = ap.parse_args()
    if args.seal:
        return seal(args.version)
    return audit(check_loadable=not args.no_loadable)


if __name__ == "__main__":
    raise SystemExit(main())
