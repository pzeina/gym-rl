"""Chain-of-command succession: command devolves, recursively, with the mission."""

from cohort.core.missions import Mission, MissionType
from cohort.core.ranks import Rank
from cohort.core.units import Roster, Soldier


def _squad() -> Roster:
    """CDG1 leads CAP1 (SLD1, SLD2) and CAP2 (SLD3, SLD4)."""
    soldiers = [
        Soldier(id=0, callsign="CDG1", rank=Rank.CDG, pos=(0, 0), subordinate_ids=[1, 4]),
        Soldier(id=1, callsign="CAP1", rank=Rank.CAP, pos=(1, 0), leader_id=0, subordinate_ids=[2, 3]),
        Soldier(id=2, callsign="SLD1", rank=Rank.SLD, pos=(2, 0), leader_id=1),
        Soldier(id=3, callsign="SLD2", rank=Rank.SLD, pos=(3, 0), leader_id=1),
        Soldier(id=4, callsign="CAP2", rank=Rank.CAP, pos=(4, 0), leader_id=0, subordinate_ids=[5, 6]),
        Soldier(id=5, callsign="SLD3", rank=Rank.SLD, pos=(5, 0), leader_id=4),
        Soldier(id=6, callsign="SLD4", rank=Rank.SLD, pos=(6, 0), leader_id=4),
    ]
    return Roster(soldiers)


def test_leader_death_promotes_senior_subordinate_recursively():
    roster = _squad()
    cdg = roster.by_callsign["CDG1"]
    cdg.mission = Mission(MissionType.SEIZE, 0, (9, 9), issuer_id=-1, step_assigned=0)
    cdg.alive = False
    events = roster.succeed(cdg)

    cap1 = roster.by_callsign["CAP1"]
    # CAP1 (senior, lowest id among equals) assumes squad command
    assert cap1.effective_rank is Rank.CDG
    assert cap1.leader_id is None
    assert cap1.mission is not None and cap1.mission.type is MissionType.SEIZE, "mission continuity"
    assert roster.root() is cap1

    # CAP2 now reports to CAP1
    assert roster.by_callsign["CAP2"].leader_id == cap1.id
    # a rifleman from CAP1's old team was promoted to acting CAP over that team
    promoted = roster.by_callsign["SLD1"]
    assert promoted.effective_rank is Rank.CAP
    assert promoted.leader_id == cap1.id
    assert promoted.id in cap1.subordinate_ids
    assert roster.by_callsign["SLD2"].leader_id == promoted.id
    assert len(events) == 2


def test_deputy_preferred_over_higher_authority_subordinate():
    soldiers = [
        Soldier(id=0, callsign="CDG1", rank=Rank.CDG, pos=(0, 0), subordinate_ids=[1, 2], deputy_id=2),
        Soldier(id=1, callsign="CAP1", rank=Rank.CAP, pos=(1, 0), leader_id=0),
        Soldier(id=2, callsign="SLD1", rank=Rank.SLD, pos=(2, 0), leader_id=0),
    ]
    roster = Roster(soldiers)
    cdg = soldiers[0]
    cdg.alive = False
    roster.succeed(cdg)
    deputy = roster.by_callsign["SLD1"]
    assert deputy.effective_rank is Rank.CDG, "designated deputy takes over despite lower rank"
    assert roster.by_callsign["CAP1"].leader_id == deputy.id


def test_dead_deputy_falls_back_to_seniority():
    soldiers = [
        Soldier(id=0, callsign="CDG1", rank=Rank.CDG, pos=(0, 0), subordinate_ids=[1, 2], deputy_id=2),
        Soldier(id=1, callsign="CAP1", rank=Rank.CAP, pos=(1, 0), leader_id=0),
        Soldier(id=2, callsign="SLD1", rank=Rank.SLD, pos=(2, 0), leader_id=0, alive=False),
    ]
    roster = Roster(soldiers)
    soldiers[0].alive = False
    roster.succeed(soldiers[0])
    assert roster.by_callsign["CAP1"].effective_rank is Rank.CDG


def test_last_man_standing_no_successor():
    soldiers = [
        Soldier(id=0, callsign="CAP1", rank=Rank.CAP, pos=(0, 0), subordinate_ids=[1]),
        Soldier(id=1, callsign="SLD1", rank=Rank.SLD, pos=(1, 0), leader_id=0, alive=False),
    ]
    roster = Roster(soldiers)
    soldiers[0].alive = False
    assert roster.succeed(soldiers[0]) == []


def test_succession_announced_on_the_net():
    """In the live env, a death must produce CASUALTY + TAKING COMMAND traffic."""
    from cohort import make_env

    env = make_env("fireteam")
    env.reset(seed=5)
    cap = env.roster.by_callsign["CAP1"]
    cap.health = 1
    # put the leader alone next to the garrison so OpFor guns them down
    enemy = next(e for e in env.enemies if e.alive)
    cap.pos = (enemy.pos[0] + 1, enemy.pos[1])
    for _ in range(30):
        if not cap.alive:
            break
        env.step({a: 0 for a in env.agents})  # everyone holds
    assert not cap.alive, "leader should have been killed by the adjacent garrison"
    kinds = [m.kind.value for m in env.transcript.messages]
    assert "casualty" in kinds
    assert "taking_command" in kinds
    new_root = env.roster.root()
    assert new_root is not None
    assert new_root.effective_rank is Rank.CAP
    assert new_root.rank is Rank.SLD
