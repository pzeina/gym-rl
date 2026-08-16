"""A confirmed root claim is the LAST root claim of its episode.

The assurance layer's standing negative control (issue #33): across 86 corpora
tapped from the net alone, the confirmed root claim is always the last claim of
its episode — 0 violations. It holds structurally rather than statistically. A
truthful root-mission COMPLETE sets ``_root_close_step`` inside
``_report_done``, and the terminal check later in the same ``step`` reads that
as ``root_reported`` and ends the episode. Nothing can follow it, so a corpus
showing two confirmed root claims in one episode — or any root claim after a
confirmed one — is a broken measurement, not a strange policy.

That makes it worth asserting rather than merely believing, because it is
exactly the property our own ``scripts/done_probe.py`` reported violating. The
probe keyed "who held the root at this step" by ``_step_count`` as read
*before* ``CohortEnv.step`` increments it, while ``_say`` stamps the
incremented value, so every claim made on an episode's LAST step fell out of
the root's count — and the last step is precisely where the confirmed ones
live. Measured on ``defend_brique_v13``/latest, 40 episodes from seed 500: 55
root claims / 0 confirmed under the old keying against 87 / 32 under the fixed
one. The assurance layer's independent net-only tap read 87 / 32 and its
``root_rejects`` of 55 = 87 - 32 pinned the defect's scope from the outside —
the bug removed the confirmed claims and nothing else.

**Honest scope note, because it changes what this file is worth.** The ratio
form of the invariant would NOT have caught that particular bug from the
probe's own output: dropping a whole final step removes the claim, its
confirmation and nothing else, so 55 - 55 = 0 confirmed still satisfies
"at most one". What catches a *keying* defect is a coverage guard — every
adjudicated message's step must be attributable to some root — and that guard
now lives inline in ``done_probe.py`` beside this invariant. The invariant here
catches the other half of the class: mis-attribution that invents a claim after
the close, and any future change to the terminal branch that lets an episode
run on past a confirmed claim.

Two forms, both cheap:

* **env-level** — drive an episode to a confirmed root claim and assert nothing
  root-claimed follows it, in this or any later step;
* **data-level** — over every committed evaluation carrying the root-split
  fields, ``done_reports_root - done_rejected_root`` is 0 or 1 per episode.
  That difference IS the confirmed count: ``cohort_env._report_done`` answers
  every DONE with a CONFIRM or a REJECT and never with neither.

Scoped to the ROOT's claims throughout. A subordinate's COMPLETE closes its own
task and not the operation, so several may be confirmed in one episode; only
the root's ends it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cohort import make_env
from cohort.core.orders import MessageKind
from cohort.env.actions import CATALOG

ROOT = Path(__file__).resolve().parents[1]

STAY = 0
DONE = next(spec.index for spec in CATALOG if spec.kind == "done")

# One per root-mission shape: SEIZE, a horizon DEFEND, a BRIQUE DEFEND and a
# team-adjudicated RECON. The invariant is about the close, which every shape
# reaches by its own route.
SCENARIOS = ["fireteam", "fireteam_defend", "defend_brique", "squad_recon"]

#: Roots that may declare an end state, so a confirmed claim is reachable.
CLAIMING = ["fireteam", "squad_recon"]
#: DEFEND roots. Since v1.17 (owner's decision) their MISSION COMPLETE is masked
#: shut at any ``defend_horizon``, so the invariant holds VACUOUSLY on them —
#: which is worth asserting as its own statement rather than letting the shared
#: parametrization quietly stop testing anything. Their close is the SITREP,
#: and their announcement is COMMAND's ENDEX.
CONTINUOUS = ["fireteam_defend", "defend_brique"]


def _win_now(env) -> None:
    """Put the world into the root-mission end state, whatever that state is.

    Blunt on purpose: the band is destroyed, the latched defend failure is
    cleared and everyone stands on the objective. Which of those the scenario's
    ``_check_success`` actually consults is its business — this only has to make
    the answer true.
    """
    for enemy in env.enemies:
        enemy.alive = False
    env._defend_lost_step = None
    if env.spec_cfg.root_objective:
        obj = env.world.objective_by_name(env.spec_cfg.root_objective)
        for soldier in env.roster.living:
            soldier.pos = obj.pos


def _claim_whenever_admissible(scenario: str, seed: int, win_after_rejections: int | None):
    """Run one episode claiming COMPLETE on every step the mask allows it.

    Returns the root's claim ledger as ``(kind, step)`` pairs in transcript
    order, plus the step count. The root is resolved BEFORE each ``env.step``
    and the traffic is read as the slice that step appended, so attribution
    never goes through a step number at all — the one way to key it that the
    probe's bug cannot reach.

    ``win_after_rejections`` engineers the end state once the root has been
    turned down that many times, which reproduces the shape the fleet actually
    files (``defend_brique_v13`` episode 0: three root claims, two rejected).
    ``None`` never wins, which is the ENDEX-only arm.
    """
    env = make_env(scenario)
    obs, _ = env.reset(seed=seed)
    env._h_hour = 0  # the preparation period is over; the criterion is live
    ledger: list[tuple[str, int]] = []
    steps = 0
    rejections = 0
    won = False

    while env.agents:
        if not won and win_after_rejections is not None and rejections >= win_after_rejections:
            _win_now(env)
            won = True
        root = env.roster.root()
        root_id = root.id if root is not None else None
        before = len(env.transcript.messages)
        actions = {
            cs: (DONE if obs[cs]["action_mask"][DONE] else STAY) for cs in env.agents
        }
        obs, _, _, _, _ = env.step(actions)
        steps += 1
        for msg in env.transcript.messages[before:]:
            if msg.kind is MessageKind.DONE and msg.sender_id == root_id:
                ledger.append(("claim", steps))
            elif msg.kind is MessageKind.DONE_CONFIRM and msg.recipient_id == root_id:
                ledger.append(("confirm", steps))
            elif msg.kind is MessageKind.DONE_REJECT and msg.recipient_id == root_id:
                ledger.append(("reject", steps))
                rejections += 1

    return env, ledger, steps


@pytest.mark.parametrize("scenario", CLAIMING)
def test_a_confirmed_root_claim_ends_the_episode_in_the_same_step(scenario):
    """The structural reason the invariant holds, asserted where it is made.

    Confirmation closes the grace window; the terminal check reads the closed
    window as success in that same step. So the confirmed claim lands on the
    episode's final step and there is no later step for a second one to occupy.

    Scoped to claiming roots since v1.17 — the defend arm of this assertion
    moved to ``test_a_defend_root_files_no_claim_at_all_and_is_announced_anyway``
    below, because on those scenarios the mask now makes it unreachable.
    """
    env, ledger, steps = _claim_whenever_admissible(scenario, seed=1, win_after_rejections=2)
    kinds = [kind for kind, _ in ledger]

    assert kinds.count("confirm") == 1, f"expected exactly one confirmed root claim: {ledger}"
    assert ledger[-1][0] == "confirm", f"something root-claimed after the close: {ledger}"
    assert ledger[-1][1] == steps, "the confirmed claim was not on the episode's last step"
    assert not env.agents, "the episode ran on past a confirmed root claim"
    assert env.outcome == "success"

    # ...and the transcript agrees, read forwards rather than through the ledger
    last_confirm = max(
        i for i, m in enumerate(env.transcript.messages) if m.kind is MessageKind.DONE_CONFIRM
    )
    root_id = env.roster.root().id
    after = env.transcript.messages[last_confirm + 1:]
    assert not [m for m in after if m.kind is MessageKind.DONE and m.sender_id == root_id]


@pytest.mark.parametrize("scenario", CONTINUOUS)
def test_a_defend_root_files_no_claim_at_all_and_is_announced_anyway(scenario):
    """v1.17: the invariant is vacuous on a defend root, and says so out loud.

    Claim on every step the mask allows and the ledger comes back empty — not
    because the policy declined, but because the bit was never set. What must
    NOT go with it is the announcement: COMMAND's ENDEX is gated on
    ``command_closes_the_operation``, which v1.16 split off from completability
    precisely so this change could be made without repeating v1.13's silent
    loss of the whole channel.
    """
    env, ledger, _ = _claim_whenever_admissible(scenario, seed=1, win_after_rejections=0)

    assert ledger == [], f"a defend root reached the DONE channel: {ledger}"
    assert env.outcome == "success"
    endexes = [m for m in env.transcript.messages if m.kind is MessageKind.ENDEX]
    assert len(endexes) == 1, f"expected exactly one ENDEX, got {len(endexes)}"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_an_operation_that_is_never_won_files_rejections_and_never_a_confirm(scenario):
    """The other arm: claim on every admissible step, win nothing, confirm nothing.

    This is the arm that makes the ledger form meaningful — a policy CAN file
    an unbounded number of root claims in one episode. What it cannot do is get
    two of them confirmed, or file one after the confirmation.
    """
    _, ledger, _ = _claim_whenever_admissible(scenario, seed=1, win_after_rejections=None)
    kinds = [kind for kind, _ in ledger]

    assert "confirm" not in kinds, "a claim was confirmed in an operation nobody won"
    assert kinds.count("claim") == kinds.count("reject"), "a claim went unadjudicated"


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("win_after_rejections", [None, 0, 2])
def test_root_claims_minus_rejections_is_zero_or_one(scenario, win_after_rejections):
    """The ledger form, in the arithmetic the behavior suite records it in.

    ``done_reports_root - done_rejected_root`` is what a corpus exposes, and it
    must never exceed 1 however hard the root spams the channel.
    """
    _, ledger, _ = _claim_whenever_admissible(scenario, seed=3, win_after_rejections=win_after_rejections)
    kinds = [kind for kind, _ in ledger]
    confirmed = kinds.count("claim") - kinds.count("reject")

    assert confirmed in (0, 1), f"{confirmed} confirmed root claims in one episode: {ledger}"
    assert confirmed == kinds.count("confirm"), "claims minus rejections is not the confirmed count"


def _behavior_corpora():
    """Committed evaluations carrying the root-split fields, newest era first.

    Runs predating the split (refs #13) are skipped rather than failed — they
    have no numerator to check. A file that will not parse is skipped too: a
    training run writing its evaluation while the suite runs is not a
    regression, and the count assertion below keeps that from hollowing the
    test out.

    Enumerated through ``run_dirs`` so the archive counts (refs #58) — a
    data-level invariant that stops seeing the older half of the corpus the day
    it is filed away is not an invariant, it is a shrinking sample.
    """
    from scripts.fleet_status import run_dirs

    for path in sorted(p for d in run_dirs(ROOT / "runs") for p in d.glob("behavior*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        episodes = payload.get("per_episode")
        if not isinstance(episodes, list) or not episodes:
            continue
        if "done_reports_root" not in episodes[0]:
            continue
        yield path, episodes


def test_committed_evaluations_carry_at_most_one_confirmed_root_claim_per_episode():
    """The data-level form, over every corpus in this working copy.

    Verified at 0 violations over 1800 episodes in 18 files at the time of
    writing, ``defend_brique_v13``/final among them (100 episodes, 321 root
    claims, 94 confirmed). This keeps it holding — and would fail loudly on any
    future evaluation whose root attribution drops or duplicates a claim.

    **Scoped to episodes with no succession, because that is exactly as far as
    the proxy is exact** (found by ``squad_v14d_nobonus``, 2026-08-12, the two
    episodes that made this fail). The invariant is about the root's *OPORD*
    claim: ``cohort_env._report_done`` closes the operation only when
    ``is_root_opord_claim`` holds. But ``metrics._done_traffic`` counts
    ``done_reports_root`` as *any* DONE whose sender held the root at that step
    — deliberately, per its own comment. The two agree while the root is one
    soldier for the whole episode, and diverge the moment a successor is
    promoted: the promoted commander still carries its personal SEIZE/ADVANCE
    mission and may truthfully complete **that**, which is confirmed by
    ``is_complete`` and counted here, while the operation correctly runs on.
    Both failing episodes were succession episodes with a dead commander
    (``squad_v14d_nobonus``: 3 confirmed over 2 successions, and 2 over 1).

    So this is a limit of the recorded quantity, not of the invariant, and the
    env-level form above — which drives a real episode and reads the actual
    close — is unaffected and still exact. The honest fix in the corpus would
    be a root-*mission* claim counter; until one exists, excluding succession
    episodes keeps the guard strict where it is sound rather than loosening the
    bound everywhere. The exclusions are asserted to stay rare, so a regression
    that starts orphaning roots cannot hide inside the exemption.
    """
    corpora = list(_behavior_corpora())
    if not corpora:
        pytest.skip("no committed evaluation in this working copy carries the root split")

    def _succeeded_mid_episode(ep):
        return bool(ep.get("succession_events"))

    violations = [
        (str(path.relative_to(ROOT)), i, ep["done_reports_root"], ep["done_rejected_root"])
        for path, episodes in corpora
        for i, ep in enumerate(episodes)
        if not _succeeded_mid_episode(ep)
        and ep["done_reports_root"] - ep["done_rejected_root"] not in (0, 1)
    ]
    assert not violations, f"episodes with a second confirmed root claim: {violations[:10]}"

    checked = sum(
        1 for _, episodes in corpora for ep in episodes if not _succeeded_mid_episode(ep)
    )
    assert checked >= 100, f"only {checked} episodes checked — the corpora went missing"

    # The exemption must stay an exemption. If succession episodes ever became
    # the majority of the corpus, this test would be asserting almost nothing —
    # and a chart-orphaning regression (the `_fill_vacancy` defect, ⚑ in
    # ROADMAP) is precisely a change that would drive successions up.
    total = sum(len(episodes) for _, episodes in corpora)
    assert checked >= total // 2, (
        f"only {checked} of {total} episodes are succession-free — the proxy's "
        "sound domain has shrunk to where this guard no longer says much"
    )


def test_the_anchor_corpus_still_shows_the_split_the_early_close_reading_rests_on():
    """``defend_brique_v13``/final is the run issue #33's §3 was answered from.

    94 successes carrying a confirmed root claim, 6 closed by ENDEX alone, all
    100 announced. Pinned because the early-close verdict in ROADMAP quotes
    those group sizes — if the file is ever re-scored, the reading has to be
    re-derived rather than silently inherited.
    """
    from scripts.fleet_status import find_run

    # Resolved, not hard-pathed: this run is archived, and a data-level
    # invariant that switches itself off when its corpus is filed away is
    # worth nothing.
    run = find_run("defend_brique_v13", ROOT / "runs")
    path = (run / "behavior_final.json") if run else ROOT / "nonexistent"
    if not path.is_file():
        pytest.skip("defend_brique_v13/behavior_final.json not present in this working copy")
    episodes = json.loads(path.read_text())["per_episode"]

    successes = [ep for ep in episodes if ep["outcome"] == "success"]
    confirmed = [
        ep for ep in successes if ep["done_reports_root"] - ep["done_rejected_root"] == 1
    ]
    assert len(successes) == 100
    assert len(confirmed) == 94
    assert len(successes) - len(confirmed) == 6
    assert sum(ep["close_announced"] for ep in episodes) == 100, "an operation closed unannounced"
