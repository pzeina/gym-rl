"""``oracle_probe.py``'s root rows are positional and cannot see a present,
silent root (refs #52).

Issue #52: the assurance layer's net-only reconstruction found 8 of 20 silent
squad arms sitting inside our own *reporting* occupancy band — roots that
stand on the objective (positionally healthy) and never claim it. The two
mechanisms behind "root filed zero DONE" — never arrived, vs. arrived and
declined to claim — share one net signature on a single positional mean, so
the fact-sheet's "dist from OBJ (all steps)" / "time within OBJ radius" rows
read a present-but-silent root as indistinguishable from a reporting one.

The fix: ``scripts/oracle_probe.py`` now tracks, per episode, whether the
root emitted at least one DONE, and buckets the same two positional
quantities into a "claimed" and a "silent" cluster instead of one pooled
mean (``Accum.root_claim_*`` / ``root_silent_*``). These tests drive real
``fireteam`` episodes (SEIZE-rooted, so DONE is admissible from the first
step) with a controlled action-picker monkeypatched in at
``oracle_probe._pick_actions`` — the same seam ``probe()`` calls through —
so the claim-detection and end-of-episode routing run unmodified.
"""

from __future__ import annotations

from cohort.env.actions import CATALOG
from scripts import oracle_probe

STAY = next(s.index for s in CATALOG if s.kind == "stay")
DONE = next(s.index for s in CATALOG if s.kind == "done")

SCENARIO = "fireteam"  # SEIZE-rooted: DONE is admissible, not masked shut


def _park_root_on_objective(env) -> None:
    """Kill every enemy and put every living soldier on the root objective.

    Mirrors ``tests/test_confirmed_claim_is_last.py``'s ``_win_now``: makes
    the root PRESENT at the objective from the first step, with nothing left
    to move it off or kill it, so occupancy isolates "present" from "claims".
    """
    for enemy in env.enemies:
        enemy.alive = False
    obj = env.world.objective_by_name(env.spec_cfg.root_objective)
    for soldier in env.roster.living:
        soldier.pos = obj.pos


def _stay_never_claim(env, obs):
    return {a: STAY for a in env.agents}


def _claim_when_admissible(env, obs):
    """Everyone holds; the root fires DONE the moment the mask allows it."""
    actions = {a: STAY for a in env.agents}
    root = env.roster.root()
    if root is not None and root.callsign in actions and obs[root.callsign]["action_mask"][DONE]:
        actions[root.callsign] = DONE
    return actions


def _run_one_probed_episode(monkeypatch, picker, *, park: bool, seed: int = 500):
    """Run ``oracle_probe.probe()`` for one episode under a controlled policy.

    Hooks in at ``_pick_actions`` — the exact name ``probe()`` calls — so the
    picker controls every action while ``probe()``'s own loop, claim
    detection and end-of-episode routing run untouched.
    """
    parked = {"done": not park}

    def fake_pick_actions(env, obs, net, rng, *, greedy=False):
        if not parked["done"]:
            _park_root_on_objective(env)
            parked["done"] = True
        return picker(env, obs)

    monkeypatch.setattr(oracle_probe, "_pick_actions", fake_pick_actions)
    acc, scenario = oracle_probe.probe(
        checkpoint=None, scenario=SCENARIO, episodes=1, first_seed=seed, greedy=False
    )
    assert scenario == SCENARIO
    return acc


def test_present_but_silent_root_lands_in_the_silent_cluster_with_high_occupancy(monkeypatch):
    """The exact case #52 names: parked on OBJ, never claims."""
    acc = _run_one_probed_episode(monkeypatch, _stay_never_claim, park=True)

    assert acc.root_claim_episodes == 0
    assert acc.root_silent_episodes == 1
    assert acc.root_silent_steps > 0
    # parked ON the objective for the whole episode: reads as healthy on the
    # (still-present, unconditioned) positional rows despite never claiming.
    assert acc.root_silent_at_objective / acc.root_silent_steps > 0.99
    assert acc.root_silent_dist_all / acc.root_silent_steps < 1.0
    # and the split says so explicitly: zero of this block's episodes claimed
    assert acc.root_claim_steps == 0


def test_a_root_that_claims_lands_in_the_claim_cluster(monkeypatch):
    acc = _run_one_probed_episode(monkeypatch, _claim_when_admissible, park=True)

    assert acc.root_claim_episodes == 1
    assert acc.root_silent_episodes == 0
    assert acc.root_claim_steps > 0
    assert acc.root_silent_steps == 0


def test_absent_root_that_never_arrives_is_silent_too_but_reads_differently(monkeypatch):
    """The OTHER silent mechanism: never reaches the objective at all.

    Both this and the present-but-silent case land in ``root_silent_*``
    (neither claims) — the split's job is only to separate claimed from
    silent, not absent from present-but-silent. What tells THOSE two apart is
    occupancy read within the silent cluster, exactly as the assurance
    layer's own corpus sorted its 20 silent arms against the two bands.
    """
    acc = _run_one_probed_episode(monkeypatch, _stay_never_claim, park=False)

    assert acc.root_claim_episodes == 0
    assert acc.root_silent_episodes == 1
    assert acc.root_silent_steps > 0
    assert acc.root_silent_at_objective == 0
    assert acc.root_silent_dist_all / acc.root_silent_steps > 10.0
