"""Can a soldier acquire a new commander without the net being told? (refs #49)

The assurance layer reconstructs the command tree from radio traffic alone: an
ORDER establishes an edge, a succession broadcast re-points the edges under the
replaced callsign. Any command-state transition that happens *in state with no
radio act* is invisible to that reconstruction, so its residual "orphaned
branch" checks could be either a real surviving defect or a ``¬K``. #49 asks
the question that separates the two:

    After a commander dies, is there any path by which its subordinates acquire
    a new commander without a TAKING_COMMAND (or other) message being emitted?
    Concretely: when no eligible successor exists, what happens to the vacated
    branch, and is that outcome announced?

**The answer this module pins is: no such path exists.**

1. ``leader_id`` — the *only* representation of "who commands me" — is written
   in exactly two places in the whole package, both inside
   ``Roster._fill_vacancy``, and both on the code path that appends the event
   the environment turns into a broadcast. That is checked structurally, so a
   third write site fails the suite rather than opening a silent path.
2. When no eligible successor exists, ``_fill_vacancy`` returns ``None`` having
   changed **nothing**: the branch is not reattached to a grandparent, not
   re-homed, not touched. There is no transition to announce, because there is
   no transition — ``_pick_successor`` returns ``None`` only when the vacated
   leader has no *living direct subordinates* at all.
3. Over every death ordering of a squad, the chart the two succession
   formatters describe is exactly the chart in state.

What the module also pins is the honest other half: the net-only chart and the
state chart **do** diverge, in both directions, and neither divergence is
silence. Both are reachable in one tick of the shipping ``squad`` scenario and
both are measured here against ``cohort.probe.NetPredictor``, the repo's own
transcript-only reconstruction:

* **A real orphan the net hides.** Two leaders on the same limb fall together;
  the env devolves them one at a time against alive-flags that already count
  both deaths, so the lower leader's successor inherits a superior who is
  already gone. Every message goes out, but a replay must process the two
  casualties in sequence, and at the moment the first succession is replayed the
  second casualty has not been spoken — so the net rebuilds the branch as
  repaired when it is not.
* **A false orphan the net invents.** ``_assume`` re-points a vacated slot's
  *downward* edges but never files the successor under its new superior, so a
  later succession into that superior's slot does not sweep the promoted agent
  up — while state, since #42, does. The monitor is then holding an orphaned
  branch that no longer exists. This one is the shape of an orphaned-branch
  residual, it needs no new radio act, and it is fixable on the monitor's side.
"""

from __future__ import annotations

import ast
import itertools
import pathlib
from dataclasses import replace

import pytest

from cohort.core.missions import Mission, MissionType
from cohort.core.ranks import Rank
from cohort.core.units import Roster, Soldier

COHORT = pathlib.Path(__file__).resolve().parent.parent / "cohort"


def _squad() -> Roster:
    """SL1 leads TL1 (RFN1, RFN2) and TL2 (RFN3, RFN4)."""
    return Roster(
        [
            Soldier(id=0, callsign="SL1", rank=Rank.SL, pos=(0, 0), subordinate_ids=[1, 4]),
            Soldier(id=1, callsign="TL1", rank=Rank.TL, pos=(1, 0), leader_id=0, subordinate_ids=[2, 3]),
            Soldier(id=2, callsign="RFN1", rank=Rank.RFN, pos=(2, 0), leader_id=1),
            Soldier(id=3, callsign="RFN2", rank=Rank.RFN, pos=(3, 0), leader_id=1),
            Soldier(id=4, callsign="TL2", rank=Rank.TL, pos=(4, 0), leader_id=0, subordinate_ids=[5, 6]),
            Soldier(id=5, callsign="RFN3", rank=Rank.RFN, pos=(5, 0), leader_id=4),
            Soldier(id=6, callsign="RFN4", rank=Rank.RFN, pos=(6, 0), leader_id=4),
        ]
    )


def _chart(roster: Roster) -> dict[int, int | None]:
    return {s.id: s.leader_id for s in roster.soldiers}


def _subs(roster: Roster) -> dict[int, list[int]]:
    return {s.id: list(s.subordinate_ids) for s in roster.soldiers}


def _replay_announcements(
    roster: Roster,
    before_leader: dict[int, int | None],
    before_subs: dict[int, list[int]],
    events: list[tuple[Soldier, Soldier]],
) -> dict[int, int | None]:
    """The parent map the succession *broadcasts* describe, replayed from ``before``.

    Deliberately a re-statement of ``core/language.py``'s two formatters and
    nothing else — the same devolution rules ``cohort.probe.NetPredictor``
    applies to real transcript text, expressed over event tuples so the replay
    is isolated from message *ordering* (which the env-level tests below cover
    separately):

    * ``format_taking_command`` — "X IS DOWN. I AM ASSUMING COMMAND": the
      successor takes the dead leader's slot, so it inherits that slot's
      superior and that slot's living subordinates.
    * ``format_assuming_position`` — "ASSUMING X'S POSITION": the filler takes
      the slot X vacated moving up, so it reports to X and inherits the
      subordinates X used to lead.

    "Takes the slot" is read in full, in both directions: the successor becomes
    a *child of the slot's superior* as well as the parent of the slot's
    subordinates. That second half is the rule #42 put into state, and it is
    what makes a later succession into the superior's own slot sweep the
    promoted agent up with it. A replay that keeps only the downward half
    reports a branch as orphaned that the announcements repaired — see
    ``test_a_net_replay_that_drops_the_upward_link_invents_an_orphan``.
    """
    leader = dict(before_leader)
    subs = {i: list(v) for i, v in before_subs.items()}
    vacated_slot: dict[int, list[int]] = {}
    for successor, replaced in events:
        if not replaced.alive:  # I AM ASSUMING COMMAND
            slot_leader, slot_subs = leader[replaced.id], list(subs[replaced.id])
            subs[replaced.id] = []
        else:  # ASSUMING X'S POSITION
            slot_leader, slot_subs = replaced.id, list(vacated_slot.get(replaced.id, []))
        vacated_slot[successor.id] = list(subs[successor.id])
        leader[successor.id] = slot_leader
        if slot_leader is not None and successor.id not in subs[slot_leader]:
            subs[slot_leader].append(successor.id)
        kept = [i for i in slot_subs if i != successor.id and roster.by_id[i].alive]
        subs[successor.id] = kept
        for i in kept:
            leader[i] = successor.id
    return leader


# --------------------------------------------------------------------- #
# 1. there is one code path, and it is the announced one
# --------------------------------------------------------------------- #


def test_a_commander_only_ever_changes_inside_the_announced_path():
    """``leader_id`` is assigned in exactly two places, both in ``_fill_vacancy``.

    This is the structural half of #49's answer: "is there ANY path" is a
    question about the whole package, not about the paths one happens to think
    of. Both write sites sit after ``_pick_successor`` returned a successor and
    before ``events.append``, i.e. on the branch ``CohortEnv.step`` turns into a
    TAKING_COMMAND broadcast — so a commander change with no radio act would
    have to be a *third* write site, and adding one fails here.
    """
    sites: list[tuple[str, str]] = []
    for path in sorted(COHORT.rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        owner: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                owner[child] = node
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets: list[ast.expr] = list(node.targets)
            elif isinstance(node, ast.AugAssign | ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr == "leader_id":
                    scope: ast.AST | None = node
                    while scope is not None and not isinstance(
                        scope, ast.FunctionDef | ast.AsyncFunctionDef
                    ):
                        scope = owner.get(scope)
                    name = scope.name if isinstance(scope, ast.FunctionDef) else "<module>"
                    sites.append((path.relative_to(COHORT.parent).as_posix(), name))
    assert sites == [
        ("cohort/core/units.py", "_fill_vacancy"),
        ("cohort/core/units.py", "_fill_vacancy"),
    ], (
        "a soldier's commander changed somewhere other than the succession path "
        f"that announces it (#49): {sites}"
    )


# --------------------------------------------------------------------- #
# 2. the case #49 names by hand: no eligible successor
# --------------------------------------------------------------------- #


def test_no_eligible_successor_reattaches_nothing_and_therefore_says_nothing():
    """The direct answer to #49's second sentence.

    ``_pick_successor`` returns ``None`` only when the vacated leader has **no
    living direct subordinates**, so the "vacated branch" the question asks
    about is empty by construction. ``_fill_vacancy`` then returns immediately:
    the branch is not reattached to the grandparent, not re-homed anywhere else,
    and not modified at all. Nothing is announced because nothing happened — the
    only traffic the death produces is HQ's CASUALTY broadcast.
    """
    roster = _squad()
    tl2 = roster.by_callsign["TL2"]
    for callsign in ("RFN3", "RFN4"):  # TL2's whole team is already down
        roster.by_callsign[callsign].alive = False
    before = _chart(roster)
    tl2.alive = False
    assert roster.succeed(tl2) == [], "no living direct subordinate: nobody to promote"
    assert _chart(roster) == before, (
        "an unfilled vacancy must not move ANY soldier's commander — a silent "
        "reattachment here is exactly the observability gap #49 asks about"
    )
    assert tl2.subordinate_ids == [5, 6], "the dead leader's slot is left as it stood"


def test_an_unfilled_vacancy_never_re_homes_the_branch_onto_the_grandparent():
    """"What happens to the vacated branch?" — nothing, and that is the answer.

    Both team leaders and the squad leader fall in one tick. ``succeed(SL1)``
    finds no living *direct* subordinate and returns without touching anything,
    so the two riflemen elements — alive, and two levels below the vacancy — are
    not lifted onto the squad slot. There is no grandparent reattachment, silent
    or otherwise, and the cohort is left with no root at all while six of seven
    of its soldiers still stand.
    """
    roster = _squad()
    for callsign in ("SL1", "TL1", "TL2"):
        roster.by_callsign[callsign].alive = False
    before = _chart(roster)
    assert roster.succeed(roster.by_callsign["SL1"]) == [], "no living direct subordinate"
    assert _chart(roster) == before, "the squad slot's vacancy moved nobody"

    events: list[tuple[Soldier, Soldier]] = []
    for callsign in ("TL1", "TL2"):
        events += roster.succeed(roster.by_callsign[callsign])
    sl1 = roster.by_callsign["SL1"]
    assert [(s.callsign, v.callsign) for s, v in events] == [("RFN1", "TL1"), ("RFN3", "TL2")]
    assert roster.by_callsign["RFN1"].leader_id == sl1.id
    assert roster.by_callsign["RFN3"].leader_id == sl1.id
    assert roster.root() is None and len(roster.living) == 4, (
        "the announced chart is the whole chart: four living soldiers under a "
        "dead squad leader, and the net was told exactly that"
    )


# --------------------------------------------------------------------- #
# 3. every ordering: the announced chart IS the chart
# --------------------------------------------------------------------- #


def test_every_commander_change_in_every_death_ordering_is_announced():
    """Exhaustive: 5040 sequential orderings + every same-step pair and triple.

    For each, the parent map rebuilt from the succession *announcements* alone
    must equal the parent map in state. Equality in both directions is the
    point: a change with no announcement is the silent transition #49 asks
    about, and an announcement with no change would be traffic the net cannot
    reconcile.
    """
    for order in itertools.permutations(range(7)):
        roster = _squad()
        for dead_id in order:
            before_leader, before_subs = _chart(roster), _subs(roster)
            roster.by_id[dead_id].alive = False
            events = roster.succeed(roster.by_id[dead_id])
            announced = _replay_announcements(roster, before_leader, before_subs, events)
            assert announced == _chart(roster), (
                f"deaths in order {order}: the net's chart and the state's chart "
                f"disagree after {roster.by_id[dead_id].callsign} fell"
            )
            if announced == before_leader:
                assert events == [], "a chart that did not move must produce no traffic"

    for size in (2, 3):
        for batch in itertools.permutations(range(7), size):
            roster = _squad()
            before_leader, before_subs = _chart(roster), _subs(roster)
            for dead_id in batch:  # the env marks every death of a step, then devolves
                roster.by_id[dead_id].alive = False
            events: list[tuple[Soldier, Soldier]] = []
            for dead_id in batch:
                events += roster.succeed(roster.by_id[dead_id])
            announced = _replay_announcements(roster, before_leader, before_subs, events)
            assert announced == _chart(roster), (
                f"same-step deaths {batch}: a commander changed without an announcement"
            )


def test_succession_still_carries_the_mission_it_announces():
    """Guard on the other half of what the broadcast implies: mission continuity.

    A net-only monitor reads "I AM ASSUMING COMMAND" as "the slot's standing
    order came with it". If state stopped doing that, the traffic would be
    describing a chart the cohort is not flying.
    """
    roster = _squad()
    sl1 = roster.by_callsign["SL1"]
    sl1.mission = Mission(MissionType.SEIZE, 0, (9, 9), issuer_id=-1, step_assigned=0)
    sl1.alive = False
    (successor, replaced), *_ = roster.succeed(sl1)
    assert replaced is sl1
    assert successor.mission is sl1.mission


# --------------------------------------------------------------------- #
# 4. the finding: a branch CAN end up under a dead commander
# --------------------------------------------------------------------- #


def test_same_step_deaths_leave_a_branch_under_a_dead_commander():
    """#49 finding, asserted as current behaviour rather than fixed here.

    ``CohortEnv.step`` collects every death of the tick into ``player_deaths``
    and only then devolves them, one at a time, against alive-flags that already
    count all of them. So when SL1 and TL1 fall together, ``succeed(SL1)``
    cannot see TL1 as a candidate and hands the squad to TL2; ``succeed(TL1)``
    then promotes RFN1 into TL1's slot — and that slot's superior is SL1, who is
    already gone. RFN1 and RFN2 are left as a live element under a dead squad
    leader.

    Nothing about this is silent: both CASUALTY broadcasts and both succession
    broadcasts go out. The branch is genuinely orphaned in state, which is why
    an outside monitor that flags it is right to.
    """
    roster = _squad()
    for callsign in ("SL1", "TL1"):
        roster.by_callsign[callsign].alive = False
    events: list[tuple[Soldier, Soldier]] = []
    for callsign in ("SL1", "TL1"):  # env order: the casualty list, as it filled
        events += roster.succeed(roster.by_callsign[callsign])

    rfn1, sl1 = roster.by_callsign["RFN1"], roster.by_callsign["SL1"]
    assert roster.root() is roster.by_callsign["TL2"], "the squad devolved to the intact team"
    assert rfn1.leader_id == sl1.id and not sl1.alive, (
        "the second succession of the step promotes into a slot whose superior "
        "is already dead — the branch is headless (#49)"
    )
    assert roster.leader_of(rfn1) is None
    assert roster.by_callsign["RFN2"].leader_id == rfn1.id, "…with a live subordinate under it"
    assert [(s.callsign, v.callsign) for s, v in events] == [
        ("TL2", "SL1"),
        ("RFN3", "TL2"),
        ("RFN1", "TL1"),
    ], "every one of those moves was announced"


def _double_casualty_step(first: str, second: str) -> tuple:
    """One tick of the shipping ``squad`` scenario in which two leaders fall.

    Returns ``(env, state_chart, net_chart)`` — the callsign→superior map as the
    roster holds it, and the same map as ``cohort.probe.NetPredictor`` rebuilds
    it from the transcript alone. The predictor is the repo's own net-only
    reconstruction and applies exactly the rules #49 describes, so it stands in
    for any outside monitor here.

    Lethality is made deterministic (every shot that is taken lands) and the two
    named leaders are put on 1 HP beside a different enemy each, in the order the
    OpFor fires — nobody else is within one hit of dying.
    """
    from cohort import make_env
    from cohort.probe import Briefing, NetPredictor

    env = make_env("squad")
    env.reset(seed=0)
    # `env.combat` is the SCENARIO's shared CombatParams instance — replace it,
    # never mutate it, or every other test in the process inherits the change.
    env.combat = replace(env.combat, min_hit=1.0, max_hit=1.0, cover_multiplier=1.0)

    def _callsign(soldier_id: int | None) -> str | None:
        return env.roster.by_id[soldier_id].callsign if soldier_id is not None else None

    net = NetPredictor(
        Briefing(
            scenario="squad",
            objectives={o.name: o.pos for o in env.world.objectives},
            spawn=(0, 0),
            org={s.callsign: _callsign(s.leader_id) for s in env.roster.soldiers},
        )
    )

    def _feed(messages) -> None:
        net.observe(
            env._step_count,
            [
                {
                    "kind": m.kind.value,
                    "from": _callsign(m.sender_id) if m.sender_id >= 0 else "HQ",
                    "to": _callsign(m.recipient_id) if (m.recipient_id or -1) >= 0 else None,
                    "text": m.text,
                }
                for m in messages
            ],
        )

    _feed(env.transcript.messages)
    enemies = [e for e in env.enemies if e.alive]
    for callsign, enemy in zip((first, second), enemies, strict=False):
        soldier = env.roster.by_callsign[callsign]
        soldier.health = 1
        soldier.pos = (enemy.pos[0] + 1, enemy.pos[1])
    env.step(dict.fromkeys(env.agents, 0))  # everyone holds
    _feed(env.last_messages)

    assert not env.roster.by_callsign[first].alive and not env.roster.by_callsign[second].alive, (
        "both leaders must fall in the same tick for this fixture to mean anything"
    )
    kinds = [m.kind.value for m in env.last_messages]
    assert kinds.count("casualty") == 2, "both deaths were broadcast"
    assert kinds.count("taking_command") >= 2, "every succession move was broadcast"

    living = [s for s in env.roster.soldiers if s.alive]
    state_chart = {s.callsign: _callsign(s.leader_id) for s in living}
    net_chart = {s.callsign: net.leader.get(s.callsign) for s in living}
    return env, state_chart, net_chart


def test_the_env_reaches_that_state_and_the_net_reads_it_as_repaired():
    """The same state in a real episode, and what the net makes of it (#49).

    SL1 then TL1 fall in one tick of the shipping ``squad`` scenario. State
    leaves RFN1 under the dead SL1. The net-only chart puts RFN1 under TL2
    instead: when the predictor replays "TL2: SL1 IS DOWN, I AM ASSUMING
    COMMAND", the CASUALTY for TL1 has not been spoken yet, so TL1 is still a
    live subordinate of SL1's slot and is swept up to TL2 — and RFN1 inherits
    that when it takes TL1's slot a moment later.

    The reconstruction is therefore *more* optimistic than the state: this path
    hides a real orphan from an outside monitor, it does not manufacture one.
    """
    env, state_chart, net_chart = _double_casualty_step("SL1", "TL1")
    dead = {s.callsign for s in env.roster.soldiers if not s.alive}
    assert dead == {"SL1", "TL1"}
    assert state_chart["RFN1"] == "SL1", "state: RFN1 hangs off a dead squad leader"
    assert env.roster.leader_of(env.roster.by_callsign["RFN1"]) is None
    assert net_chart["RFN1"] == "TL2", "the net rebuilt the branch as repaired"
    assert [c for c, ldr in state_chart.items() if ldr in dead] == ["RFN1"]
    assert [c for c, ldr in net_chart.items() if ldr in dead] == []


def test_a_net_replay_that_drops_the_upward_link_invents_an_orphan():
    """The other direction, and the likelier explanation of a monitor's residual.

    TL2 then SL1 fall in one tick. In state the chain ends up **whole**: RFN3
    takes TL2's slot — which, since #42, makes it one of SL1's subordinates — and
    when TL1 then assumes SL1's command it sweeps RFN3 up with the rest of that
    slot's team. Both moves are broadcast.

    ``NetPredictor._assume`` re-points the *downward* edges of a vacated slot but
    never adds the successor to its new superior's subordinate list, so its
    replay does not know RFN3 belongs to SL1's slot and leaves it hanging off the
    dead SL1. That is a **false** orphaned branch, produced by a reconstruction
    that has not mirrored #42 — no silence, no missing radio act, and fixable
    entirely on the monitor's side.

    Asserted as current behaviour, deliberately: ``cohort/`` is frozen for the
    v1.20 campaign, and what the layer needs first is to know the gap is theirs.
    """
    env, state_chart, net_chart = _double_casualty_step("TL2", "SL1")
    dead = {s.callsign for s in env.roster.soldiers if not s.alive}
    assert dead == {"SL1", "TL2"}
    assert state_chart["RFN3"] == "TL1", "state: RFN3 was swept up to the new squad leader"
    assert env.roster.leader_of(env.roster.by_callsign["RFN3"]) is not None
    assert [c for c, ldr in state_chart.items() if ldr in dead] == [], "state: no orphan at all"
    assert net_chart["RFN3"] == "SL1", (
        "the net-only replay leaves RFN3 under the dead SL1 — an orphaned branch "
        "that does not exist in state (#49)"
    )


# --------------------------------------------------------------------- #
# 5. a defect found while answering #49 — #42's link is applied twice
# --------------------------------------------------------------------- #


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#42 defect, found while answering #49: _fill_vacancy links the "
        "backfilled agent into its new leader's subordinate_ids TWICE — once at "
        "the general `parent.subordinate_ids.append` #42 added, once at the "
        "pre-existing `successor.subordinate_ids.append(promoted.id)` that #42 "
        "made redundant. Fix is one line in cohort/core/units.py, which is "
        "frozen for the v1.20 campaign; remove this marker with the fix."
    ),
)
def test_a_backfilled_agent_is_linked_into_its_new_leader_exactly_once():
    """The commonest succession in the game duplicates a subordinate slot.

    SL1 falls, TL1 takes the squad, RFN1 backfills TL1's team. RFN1 is then
    appended to TL1's ``subordinate_ids`` twice, so ``living_subordinates``
    returns it twice — and that list is what ``env/observations.py`` writes into
    the four subordinate slots and what ``env/actions.py`` indexes with
    ``order_slot``. The new *root* therefore spends an observation slot on a
    duplicate and carries two distinct ORDER action indices that address the
    same agent, from the moment it takes command.

    Pre-#42 (``56ada9a^``) the same cascade produced ``[4, 2]``.
    """
    roster = _squad()
    sl1 = roster.by_callsign["SL1"]
    sl1.alive = False
    roster.succeed(sl1)
    tl1 = roster.by_callsign["TL1"]
    assert tl1.subordinate_ids == [4, 2], f"duplicated chart slot: {tl1.subordinate_ids}"
    subs = [s.callsign for s in tl1.living_subordinates(roster)]
    assert len(subs) == len(set(subs)), f"a subordinate is listed twice: {subs}"
