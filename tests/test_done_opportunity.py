"""Zero DONE reports must say WHICH silence it is (refs issue #13).

``done_reports = 0`` is the same number for two opposite findings:

* **absence** — MISSION COMPLETE was never admissible, so no price was ever
  consulted. That was the ``is_root_opord_claim`` mask bug (cc07199): on a
  DEFEND-rooted scenario the root's claim was hard-masked and the silence read
  as a taught behaviour for a whole training generation.
* **suppression** — the act was admissible on thousands of agent-steps and the
  policy declined it, which is a statement about ``done_false``, not about
  reachability. ``scripts/done_probe.py`` measured squad_v6 at 11,528
  admissible agent-steps over 10 episodes with 0 claims transmitted, against
  an oracle regime that took 57 confirmed completions on the same seeds.

The behavior suite could not tell them apart, because the claim count had no
denominator. It has one now: ``done_admissible`` (and the root's share of it),
recorded off the mask's own predicate so the two cannot drift.
"""

import numpy as np

from cohort import make_env
from cohort.env.actions import CATALOG, is_done_admissible
from cohort.metrics import TraceRecorder, aggregate_behavior, episode_behavior

STAY = 0
DONE = next(s.index for s in CATALOG if s.kind == "done")


def _record(env, actions_for, steps=40, seed=5, after_reset=None):
    """Drive ``steps`` ticks with a caller-chosen action map, recording."""
    rec = TraceRecorder()
    obs, _ = env.reset(seed=seed)
    if after_reset is not None:
        after_reset(env)
    rec.on_reset(env)
    for _ in range(steps):
        if not env.agents:
            break
        rec.before_step(env)
        obs, _, _, _, _ = env.step(actions_for(env, obs))
        rec.after_step(env)
    return aggregate_behavior([episode_behavior(rec.trace)])


def _all_stay(env, obs):
    return {a: STAY for a in env.agents}


def test_predicate_and_mask_cannot_drift():
    """``is_done_admissible`` is the mask's DONE bit, not a second opinion."""
    env = make_env("fireteam")
    obs, _ = env.reset(seed=11)
    rng = np.random.default_rng(11)
    for _ in range(60):
        if not env.agents:
            break
        for cs in env.agents:
            soldier = env.roster.by_callsign[cs]
            assert is_done_admissible(
                soldier,
                env.roster,
                root_mission=env.spec_cfg.root_mission,
                root_objective_id=env._root_objective_id(),
                step=env._step_count,
                done_cooldown=env.spec_cfg.done_cooldown,
            ) is bool(obs[cs]["action_mask"][DONE])
        actions = {
            a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents
        }
        obs, _, _, _, _ = env.step(actions)


def test_silence_with_an_open_channel_reads_as_suppression():
    """A tasked cohort that never claims: many opportunities, no claims."""
    env = make_env("fireteam")
    agg = _record(env, _all_stay)
    assert agg["done_reports"] == 0
    assert agg["done_admissible"] > 0, "the channel was open on tasked, completable missions"
    assert agg["done_admissible_root"] > 0, "including the root's own OPORD claim"
    assert agg["done_claim_rate"] == 0.0


def test_silence_with_a_shut_channel_reads_as_absence():
    """An untasked cohort has nothing to report: no opportunity, no claim.

    Same ``done_reports = 0``, opposite finding — and the two are now
    distinguishable without re-running a probe.
    """
    env = make_env("fireteam")

    def strip(e):
        for s in e.roster.soldiers:
            s.mission = None

    def strip_missions(e, obs):
        strip(e)
        return {a: STAY for a in e.agents}

    agg = _record(env, strip_missions, after_reset=strip)
    assert agg["done_reports"] == 0
    assert agg["done_admissible"] == 0
    assert agg["done_admissible_root"] == 0
    # no opportunity ever arose, so the rate is undefined rather than 0.0 —
    # the distinction the whole metric exists to preserve
    assert agg["done_claim_rate"] is None


def test_a_taken_opportunity_shows_up_in_the_rate():
    """The denominator is the act's opportunity, so a claim moves the rate."""
    env = make_env("fireteam")

    def claim_once(e, obs):
        acts = {a: STAY for a in e.agents}
        for a in e.agents:
            if obs[a]["action_mask"][DONE] and e._step_count == 3:
                acts[a] = DONE
                break
        return acts

    agg = _record(env, claim_once)
    assert agg["done_reports"] == 1
    assert agg["done_admissible"] > 1
    assert 0.0 < agg["done_claim_rate"] < 1.0
