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

One thing is disclosed and deliberately NOT gated on: any manifest run —
member or ``seed_search`` candidate — whose model tensors reproduce another
run's checkpoint bit-for-bit (assurance #60). Bit-deterministic training makes
that honest and even expected; what it forbids is reading such a pair as an
independent draw, which is how a pre-registered seed-carry test came to compare
five checkpoints with themselves.

The complement of the declared search is declared too (``seed_spread``, owner
decision 2026-08-18): every OTHER run of the record — live or archived — that
trained the member's exact config at a different seed, or the same seed on a
different ``cohort/`` tree. The search says which seeds the member was chosen
from; the spread says what else the record knows about that lottery, because a
board that prints "2 of 2 seeds report" while mute same-config draws sit in
the archive is the quiet half of a search (assurance #63: ``squad_v29_seed14``
failed the reporting gate on both checkpoints and was in neither block). The
audit dedupes the spread by model-tensor digest — a bit-identical re-execution
is ONE draw, never two — and FAILS on any same-config draw the corpus holds
that neither block names. Cross-tree draws are carried and annotated: they are
evidence about the seed, not about the sealed environment.

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
    # "exists to be worse" until 2026-08-18, when the three-seed measurement
    # showed the flat arm beating the shipped system on success. The arms exist
    # to MEASURE what the hierarchy buys, whichever way it falls.
    "squad_nomask": "B3 ablation arm — doctrine masks removed; measures, not ships",
    "squad_flat": "B3 ablation arm — no chain of command at all; measures, not ships",
    "platoon_nomask": "B3 ablation arm at platoon depth — measures, not ships",
    "platoon_flat": "B3 ablation arm at platoon depth — measures, not ships",
    "platoon_hard": (
        "harder-OpFor follow-up scenario (14-defender garrison) — an experiment "
        "axis until the owner decides it ships"
    ),
    "platoon_hard_nomask": "B3 ablation arm of platoon_hard — measures, not ships",
    "platoon_hard_flat": "B3 ablation arm of platoon_hard — measures, not ships",
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

    Memoized: a commit's tree is immutable, and the ``seed_spread`` audit asks
    about the same handful of commits once per run in the corpus — without the
    cache that is a ``git rev-parse`` subprocess per row, per board render.
    """
    if not commit:
        return None
    if commit in _COHORT_TREES:
        return _COHORT_TREES[commit]
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", f"{commit}:cohort"], cwd=ROOT,
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    tree = out.stdout.strip() if out.returncode == 0 else None
    _COHORT_TREES[commit] = tree
    return tree


_COHORT_TREES: dict[str, str | None] = {}


def config_matches(existing: dict, config: dict, *, modulo: tuple[str, ...] = ()) -> bool:
    """Exact training-config identity, optionally ignoring named keys.

    THE matcher — ``campaign_preflight`` uses it to refuse queueing a job the
    record already answers (``modulo=()``: bit-deterministic re-derivation) and
    the ``seed_spread`` audit uses it with ``modulo=("seed",)`` to recognise
    another draw of the same lottery. One definition, because two matchers is
    how one of them quietly stops matching.
    """
    def strip(c: dict) -> dict:
        return {k: v for k, v in c.items() if k not in modulo}

    return strip(existing) == strip(config)


def overrides_match(recorded: list | None, overrides: list[str]) -> bool:
    """Same prices, order-free. Different recorded prices are a different
    experiment, not another draw; an UNRECORDED price (pre-economics run) stays
    a suspect, because unknown is not the same finding as different."""
    return recorded is None or sorted(recorded) == sorted(overrides)


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


RECORDED_EVAL = {"ckpt_best.pt": "behavior.json", HEADLINE_CKPT: "behavior_final.json"}


def _recorded_file_hash(run: str, name: str) -> str | None:
    """The checkpoint's FILE sha256 as its own evaluation recorded it (#67).

    ``runs/archive/`` prunes ``ckpt_latest.pt``, but the evaluation that scored
    that checkpoint wrote ``checkpoint_sha256`` beside the metrics — at eval
    time, before the prune, and committed. So for 50 of the repository's 51
    pruned finals the identity is still in the record, and "settled at
    ckpt_best" was a limitation of the lookup, not of the evidence. Guarded on
    the recorded ``checkpoint`` actually naming this file, so a mis-keyed
    record cannot lend one checkpoint another's identity.

    This is a *file* hash — a different quantity from ``policy_digest``'s
    digest over the model tensors. A checkpoint serializes its
    ``reward_config``, so one policy can live in two byte-strings (the #61
    confound); comparing across the two namespaces reports "differs" for
    bit-identical runs. Callers compare file hash to file hash only.
    """
    rec = _json_or_empty(run_dir(run) / RECORDED_EVAL[name])
    sha = rec.get("checkpoint_sha256")
    recorded = rec.get("checkpoint")
    if not sha or not recorded or Path(recorded).name != name:
        return None
    return sha


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


def seed_spread_facts(manifest: dict, scenario: str, member: str | None,
                      digest=None) -> dict | None:
    """Every OTHER same-config draw the record holds for one scenario.

    **Why a second block.** ``seed_search`` declares the seeds a member was
    *chosen* from, and the board prints "k of K seeds report" off it. But the
    record can hold more draws of the same lottery than the search ever looked
    at: the 2026-08-18 measurement campaign put a seed-13 run beside each
    one-seed member, the archive holds same-config runs at other seeds and on
    earlier ``cohort/`` trees, and ``squad_v29_seed14`` — which fails the
    reporting gate on both checkpoints — sat in neither block while the board
    said "2 of 2 seeds report" (assurance #63). True of the search, and the
    quiet half of it. So the manifest's ``seed_spread`` block names those runs,
    and this function derives everything else from their committed artifacts.

    Three derivations keep the count honest:

    * **dedupe** — draws are counted by ``policy_digest`` over EVERY checkpoint
      both runs hold (assurance #65: ``runs/archive/`` prunes the final, so a
      key on ``ckpt_latest.pt`` alone left 36 of the 56 declared runs silently
      undedupable — absence-as-distinct inflates the count of independent
      draws, the exact failure the block exists to prevent). Two runs sharing
      at least one digestible checkpoint and disagreeing on none are ONE draw,
      annotated with which checkpoints settled it, so a bit-identical
      re-execution (``squad_v29_seed14`` == archived ``squad_v10c``) counts
      once, never twice (#60's lesson applied to counting). Where a checkpoint
      file is pruned, the file sha256 its own evaluation recorded stands in
      (#67) — file hash against file hash, never against a tensor digest — so
      an archived final still carries identity instead of quietly degrading
      the merge to ``ckpt_best``.
    * **cross-tree** — each draw's ``cohort/`` tree is derived from its
      recorded commit; a draw with no run on the member's tree is annotated,
      because it is evidence about the seed, not about the sealed environment.
    * **completeness** — the corpus (``runs/`` and ``runs/archive/``) is
      scanned for same-config-modulo-seed draws in NEITHER block, and the
      audit fails naming them. That failure is the whole point: the block
      cannot go quietly stale the way the board's caption did.

    Returns None when the scenario has no declared spread and the scan finds
    nothing — silence there means the search really is the whole record.
    ``digest`` is injectable for tests and for ``--no-loadable``; unmeasured
    reporting rates count in neither direction but are disclosed (archived
    draws are not re-scored to make a count look complete).
    """
    if not member:
        return None
    declared = list((manifest.get("seed_spread") or {}).get(scenario) or [])
    search = list((manifest.get("seed_search") or {}).get(scenario) or [])
    member_dir = run_dir(member)
    member_cfg = _json_or_empty(member_dir / "config.json")
    member_tree = cohort_tree(_json_or_empty(member_dir / "economics.json").get("git_commit"))

    rows = []
    for run in declared:
        d = run_dir(run)
        cfg = _json_or_empty(d / "config.json")
        econ = _json_or_empty(d / "economics.json")
        rows.append({
            "run": run,
            "exists": d.is_dir(),
            "seed": cfg.get("seed"),
            "tree": cohort_tree(econ.get("git_commit")),
            "overrides": econ.get("reward_overrides"),  # None = unrecorded, [] = none
            "config_matches": bool(cfg and member_cfg
                                   and config_matches(cfg, member_cfg, modulo=("seed",))),
            "reports": _reporting_gate(run),
        })

    undeclared = []
    if member_cfg:
        from scripts.fleet_status import run_dirs

        listed = {member, *search, *declared}
        for d in run_dirs(RUNS):
            if d.name in listed:
                continue
            cfg = _json_or_empty(d / "config.json")
            if not cfg or not config_matches(cfg, member_cfg, modulo=("seed",)):
                continue
            recorded = _json_or_empty(d / "economics.json").get("reward_overrides")
            if not overrides_match(recorded, []):
                continue  # a different price is a different experiment
            undeclared.append(d.name)

    if not rows and not undeclared:
        return None

    digests_enabled = digest is None
    if digest is None:
        from scripts.publish_audit import policy_digest as digest

    # Member + search + valid spread runs, grouped into DRAWS by model-tensor
    # identity over EVERY checkpoint both runs hold (assurance #65). Keying on
    # the final alone could not dedupe an archived run — `runs/archive/` prunes
    # `ckpt_latest.pt` and keeps `ckpt_best.pt` — so absence silently degraded
    # to "distinct", inflating the independent-draw count. Two runs are one
    # draw when they share at least one comparable checkpoint and disagree on
    # none; a run with no derivable identity at all stands alone (unknown
    # identity is not the same as shared) and is disclosed — as a FAILURE when
    # its files are on disk and the digests are real, because that is the key
    # going quietly unavailable, not an honest absence.
    kinds: dict[str, str] = {}
    entries = ([("declared", member)] + [("declared", r) for r in search]
               + [("spread", r["run"]) for r in rows
                  if r["exists"] and r["config_matches"] and not r["overrides"]])
    order: list[str] = []
    for kind, run in entries:
        if run not in kinds:
            kinds[run] = kind
            order.append(run)
    keys = {run: {name: dg for name in CHECKPOINTS
                  if (dg := digest(run_dir(run) / name))}
            for run in order}
    # The pruned final's recorded identity (#67). `runs/archive/` prunes
    # `ckpt_latest.pt`, so #65's every-checkpoint key still settled its merged
    # groups at ckpt_best alone — annotated as if the final were unknowable,
    # when each run's own behavior_final.json IS the evaluation of that
    # checkpoint and records its file sha256. Runs are compared per checkpoint
    # tensor-to-tensor where both digests exist, file-to-file otherwise, and
    # never across the two namespaces (a file hash splits one policy in two
    # whenever only the serialized price differs — the #61 confound).
    file_keys = {run: {name: sha for name in CHECKPOINTS
                       if (sha := _recorded_file_hash(run, name))}
                 for run in order}

    def _same(a: str, b: str, name: str) -> bool | None:
        if name in keys[a] and name in keys[b]:
            return keys[a][name] == keys[b][name]
        if name in file_keys[a] and name in file_keys[b]:
            if file_keys[a][name] == file_keys[b][name]:
                return True  # equal FILE hashes entail equal tensors — sound
            if name in keys[a] or name in keys[b]:
                return False  # a tensor anchors one side; the record splits them
            # Differing FILE hashes with no tensor on either side prove
            # nothing (#68): a checkpoint serializes its reward_config, so one
            # policy can live in two byte-strings (#61), and with both files
            # pruned no tensor can overrule the record. Unresolved, not
            # distinct — `False` here flowed into `all(verdicts)` and split
            # tensor-identical pairs into extra independent draws the moment
            # their archives lost the files.
            return None
        return None

    def _unadjudicated(a: str, b: str, name: str) -> bool:
        """The #68 shape: recorded file hashes disagree, no tensor to consult."""
        return (name not in keys[a] and name not in keys[b]
                and name in file_keys[a] and name in file_keys[b]
                and file_keys[a][name] != file_keys[b][name])

    parent = {run: run for run in order}

    def _draw(r: str) -> str:
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    for i, a in enumerate(order):
        for b in order[i + 1:]:
            verdicts = [v for name in CHECKPOINTS
                        if (v := _same(a, b, name)) is not None]
            if verdicts and all(verdicts):
                parent[_draw(b)] = _draw(a)

    draws: list[dict] = []
    by_key: dict[str, dict] = {}
    for run in order:
        key = _draw(run)
        group = by_key.get(key)
        if group is None:
            group = {"runs": [], "kinds": set(), "seeds": [], "trees": [], "verdicts": []}
            by_key[key] = group
            draws.append(group)
        group["runs"].append(run)
        group["kinds"].add(kinds[run])
        group["seeds"].append(_json_or_empty(run_dir(run) / "config.json").get("seed"))
        group["trees"].append(
            cohort_tree(_json_or_empty(run_dir(run) / "economics.json").get("git_commit")))
        group["verdicts"].append(_reporting_gate(run))

    # Which checkpoints carry each group's identity — and, through a merge via
    # an intermediary that holds only one of them, whether any two members of a
    # group actually disagree somewhere. That shape is two final policies in
    # one count and cannot be resolved mechanically, so it is surfaced as a
    # problem rather than silently counted either way.
    conflicts: list[tuple[list[str], list[str]]] = []
    unresolved: list[tuple[list[str], list[str]]] = []
    for g in draws:
        settled, disputed = [], []
        for name in CHECKPOINTS:
            held = [v for i, a in enumerate(g["runs"])
                    for b in g["runs"][i + 1:]
                    if (v := _same(a, b, name)) is not None]
            if held:
                (settled if all(held) else disputed).append(name)
        g["settled"] = settled
        if disputed:
            conflicts.append((list(g["runs"]), disputed))
        # The pairs the record could not adjudicate (#68): recorded file
        # hashes that disagree at a checkpoint neither side still holds. The
        # group merged on what the tensors do settle — the #65 behaviour —
        # and the difference between "the final agrees" and "the final could
        # not be read" is stated rather than resolved by whichever answer the
        # surviving artifact happens to support.
        g["unresolved"] = [name for name in CHECKPOINTS
                           if any(_unadjudicated(a, b, name)
                                  for i, a in enumerate(g["runs"])
                                  for b in g["runs"][i + 1:])]
        if g["unresolved"]:
            unresolved.append((list(g["runs"]), list(g["unresolved"])))
    undigested = [r for r in order if not keys[r] and not file_keys[r]]
    unreadable = [r for r in order if not keys[r] and digests_enabled
                  and any((run_dir(r) / n).is_file() for n in CHECKPOINTS)]
    # The one case the recovery does not reach (#67): a final that is neither
    # on disk nor recorded anywhere — its identity is unrecoverable by anyone,
    # and any merge it takes part in rests on ckpt_best alone. Disclosed, not
    # failed: the record cannot be completed retroactively.
    final_unknown = [r for r in order if r not in undigested
                     and HEADLINE_CKPT not in keys[r]
                     and HEADLINE_CKPT not in file_keys[r]]

    for g in draws:
        measured = [v for v in g["verdicts"] if v is not None]
        g["reports"] = True if any(measured) else False if measured else None
        resolved = [t for t in g["trees"] if t]
        g["cross_tree"] = bool(member_tree and resolved
                               and all(t != member_tree for t in resolved))

    spread = [g for g in draws if g["kinds"] == {"spread"}]
    return {
        "scenario": scenario,
        "runs": rows,
        "undeclared": undeclared,
        "draws": draws,
        "undigested": undigested,
        "unreadable": unreadable,
        "final_unknown": final_unknown,
        "conflicts": conflicts,
        "unresolved": unresolved,
        "spread_draws": len(spread),
        "spread_reporting": sum(1 for g in spread if g["reports"] is True),
        "spread_mute": sum(1 for g in spread if g["reports"] is False),
        "spread_unmeasured": sum(1 for g in spread if g["reports"] is None),
        "cross_tree": sum(1 for g in spread if g["cross_tree"]),
        "known_total": len(draws),
        "known_reporting": sum(1 for g in draws if g["reports"] is True),
    }


def _seed_spread_problems(facts: dict, member: str | None, search: list[str]) -> list[str]:
    """What makes the disclosed spread worth believing — and the failure the
    block exists for: a same-config draw the manifest does not carry."""
    sc = facts["scenario"]
    problems = []
    names = [r["run"] for r in facts["runs"]]
    for run in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"seed_spread[{sc}]: {run} is listed twice — one draw, two counts")
    for run in names:
        if run == member or run in search:
            problems.append(
                f"seed_spread[{sc}]: {run} is already the member or a seed_search "
                "candidate — one draw counted in two blocks")
    for r in facts["runs"]:
        if not r["exists"]:
            problems.append(f"seed_spread[{sc}]: no run directory for {r['run']}")
        elif r["overrides"]:
            problems.append(
                f"seed_spread[{sc}]: {r['run']} carries {', '.join(r['overrides'])} — "
                "a different experiment, not a draw of the shipped configuration")
        elif not r["config_matches"]:
            problems.append(
                f"seed_spread[{sc}]: {r['run']}'s training config is not the member's "
                "modulo seed — not a draw of the same lottery")
    for g in facts["draws"]:
        if len({v for v in g["verdicts"] if v is not None}) > 1:
            problems.append(
                f"seed_spread[{sc}]: bit-identical runs {' = '.join(g['runs'])} carry "
                "disagreeing measured reporting verdicts — a re-score moved one of them")
    for runs, names in facts.get("conflicts", ()):
        problems.append(
            f"seed_spread[{sc}]: {' = '.join(runs)} group through a shared checkpoint "
            f"but disagree at {', '.join(n.removesuffix('.pt') for n in names)} — two "
            "final policies in one draw group; the identity is ambiguous and needs a "
            "human read")
    for run in facts.get("unreadable", ()):
        problems.append(
            f"seed_spread[{sc}]: {run} holds checkpoint files but none can be "
            "digested — unknown identity silently counts as a distinct draw, which "
            "inflates the spread (assurance #65)")
    if facts["undeclared"]:
        problems.append(
            f"seed_spread[{sc}]: same-config draw(s) in neither seed_search nor "
            f"seed_spread: {', '.join(facts['undeclared'])} — the record holds more "
            "draws than the manifest declares")
    return problems


def _print_spread(sp: dict, declared_names: set[str]) -> None:
    """The spread under one scenario's reporting lines: counts, then draws."""
    extra = f", {sp['spread_unmeasured']} unmeasured" if sp["spread_unmeasured"] else ""
    cross = f"; {sp['cross_tree']} cross-tree" if sp["cross_tree"] else ""
    print(f"      spread: +{sp['spread_draws']} distinct draws over {len(sp['runs'])} "
          f"more runs — {sp['spread_reporting']} report, {sp['spread_mute']} mute"
          f"{extra}{cross}")
    for g in sp["draws"]:
        spread_runs = [r for r in g["runs"] if r not in declared_names]
        if not spread_runs:
            continue
        # Which checkpoints the identity rests on — said out loud whenever it
        # is fewer than all of them, i.e. whenever a pruned final made
        # ckpt_best carry the match (assurance #65).
        note = ""
        if len(g["runs"]) > 1 and set(g.get("settled") or ()) < set(CHECKPOINTS):
            note = (" — settled at "
                    + " + ".join(n.removesuffix(".pt") for n in g["settled"]))
        if "declared" in g["kinds"]:
            # Folded into a member/search line above — one draw, said out loud.
            print(f"          {' = '.join(spread_runs)}  ==  {g['runs'][0]} "
                  f"(the same draw, counted once{note})")
            continue
        label = " = ".join(g["runs"])
        verdict = "—" if g["reports"] is None else "reports" if g["reports"] else "mute"
        where = ""
        if g["cross_tree"]:
            trees = sorted({(t or "?")[:8] for t in g["trees"]})
            where = f"  cross-tree {'/'.join(trees)}"
        seed = g["seeds"][0]
        print(f"          seed {seed!s:<4} {label:<40} {verdict:<8}{where}{note}")
    if sp.get("undigested"):
        print(f"      identity underived for {len(sp['undigested'])} run(s) — no "
              "digestible checkpoint, each counted as its own draw: "
              f"{', '.join(sp['undigested'])}")
    if sp.get("final_unknown"):
        print(f"      final policy unrecoverable for {len(sp['final_unknown'])} "
              "run(s) — ckpt_latest pruned and no evaluation recorded its hash, "
              "so identity rests on ckpt_best alone: "
              f"{', '.join(sp['final_unknown'])}")
    for runs, names in sp.get("unresolved", ()):
        print(f"      unresolved at {' + '.join(n.removesuffix('.pt') for n in names)} "
              f"for {' = '.join(runs)} — recorded file hashes disagree and neither "
              "file survives to compare tensors; a file hash splits one policy in "
              "two whenever only the serialized price differs (#61), so the "
              "disagreement is unadjudicated, not a second draw (#68)")
    unmeasured = f" ({sp['spread_unmeasured']} unmeasured)" if sp["spread_unmeasured"] else ""
    print(f"      known same-config draws: {sp['known_reporting']} of "
          f"{sp['known_total']} report{unmeasured}")


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


def tracked_files(run: str) -> set[str] | None:
    """Basenames of this run's files the repository tracks; None when git
    cannot answer (a tarball export, a tmp_path fixture).

    Issue #66. The manifest checks used to read the *filesystem*, so a declared
    run that was on the authoring disk but never ``git add``-ed passed every
    gate there and failed in any clone — ``platoon_v10_seed12`` did exactly
    that. "Declared" must mean "in the repository", and only git can say so;
    presence on one machine's disk is what the gate exists to see through.
    """
    import subprocess

    d = run_dir(run)
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", str(d)],
            cwd=ROOT, capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return {line.rsplit("/", 1)[-1] for line in out.stdout.splitlines() if line}


def _policy_reproductions(manifest: dict) -> list[tuple[str, str, list[str]]]:
    """(run, other_run, checkpoints) wherever a manifest run's model tensors are
    bit-identical to another run's on disk.

    Assurance #60. Training is bit-deterministic in (seed, scenario, steps, lr,
    price), so a re-launch across commits that never touch the trajectory
    reproduces its predecessor exactly — and all TWELVE v1.21 campaign runs
    did, spanning four prior ``cohort/`` trees. The runs were real and the seal
    honest. What broke was a claim: the campaign pre-registered "if seeds
    18/19/14 report again here the seed carries; if they do not, the draw is
    re-rolled by the tree", and every one of the five cited comparisons was a
    checkpoint against a bit-identical copy of itself. "Re-rolled" was
    unreachable, so five-for-five was an identity, not a measurement.

    Hence a DISCLOSURE, never a failure: reproducing a draw is what determinism
    is for, and the sealed fleet it happened to is correct. The thing that must
    not happen again is describing such a pair as evidence that anything
    between the two launches moved — or spared — the weights, so the audit says
    which manifest runs re-execute an existing policy *before* any claim about
    them is written. Members AND ``seed_search`` candidates: the seed-carry
    identity lived in the candidates, and four of the twelve were nothing else.

    The identity is ``publish_audit.policy_digest`` — the model tensors, not
    the file — shared with ``--validate``'s deduplication. A file hash splits
    one policy in two whenever only the serialized price differs, which is
    precisely the comparison a price experiment makes (#60 §3).
    """
    from scripts.fleet_status import run_dirs
    from scripts.publish_audit import policy_digest

    watched = list(dict.fromkeys(
        [r for r in (manifest.get("runs") or {}).values() if r]
        + [r for c in (manifest.get("seed_search") or {}).values() for r in c]))
    if not any((run_dir(r) / n).is_file() for r in watched for n in CHECKPOINTS):
        return []          # nothing to digest, and no torch import on the way out
    holders: dict[str, list[str]] = {}          # digest -> runs carrying it
    digests: dict[tuple[str, str], str] = {}    # (run, checkpoint) -> digest
    for d in run_dirs(RUNS):
        for name in CHECKPOINTS:
            digest = policy_digest(d / name)
            if digest:
                digests[(d.name, name)] = digest
                if d.name not in holders.setdefault(digest, []):
                    holders[digest].append(d.name)
    findings: list[tuple[str, str, list[str]]] = []
    for run in watched:
        matches: dict[str, list[str]] = {}
        for name in CHECKPOINTS:
            digest = digests.get((run, name))
            if digest is None:
                continue
            for other in holders[digest]:
                if other != run and name not in matches.setdefault(other, []):
                    matches[other].append(name)
        findings.extend((run, other, names) for other, names in sorted(matches.items()))
    return findings


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

    spreads = {}
    for scenario in DOCTRINE_SCENARIOS:
        member = members.get(scenario)
        sp = seed_spread_facts(manifest, scenario, member,
                               digest=None if check_loadable else (lambda p: None))
        if sp is None:
            continue
        spreads[scenario] = sp
        failures.extend(_seed_spread_problems(
            sp, member, (manifest.get("seed_search") or {}).get(scenario) or []))

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
        else:
            seeds = ", ".join(str(r["seed"]) for r in s["runs"])
            print(f"  {scenario:<18} {s['reporting']} of {s['total']} seeds report   (seeds {seeds})")
            for r in s["runs"]:
                mark = "  <- member" if r["run"] == members.get(scenario) else ""
                verdict = "—" if r["reports"] is None else "reports" if r["reports"] else "mute"
                print(f"      seed {r['seed']!s:<4} {r['run']:<32} {verdict}{mark}")
        sp = spreads.get(scenario)
        if sp:
            declared_names = {members.get(scenario),
                              *((manifest.get("seed_search") or {}).get(scenario) or [])}
            _print_spread(sp, declared_names)

    # Disclosed, never gated on (assurance #60): a bit-deterministic re-run
    # reproducing its predecessor is honest — but a comparison inside such a
    # pair is a checkpoint against itself, and no seed-carry or tree-change
    # claim can rest on one. Say so before the claim gets written.
    if check_loadable:
        repro = _policy_reproductions(manifest)
        if repro:
            print("\npolicy reproductions — model tensors bit-identical to another "
                  "run's (assurance #60)")
            for run, other, names in repro:
                what = " + ".join(n.removesuffix(".pt") for n in names)
                print(f"  {run:<26} {what}  ==  {other}")
            print("  a re-executed draw is not a defect and gates on nothing; but "
                  "any comparison\n  within such a pair is an identity, not a "
                  "measurement — it can support no claim\n  about what changed "
                  "between the two launches")

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
