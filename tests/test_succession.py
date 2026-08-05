"""Chain-of-command succession: command devolves, recursively, with the mission."""

from cohort.core.missions import Mission, MissionType
from cohort.core.ranks import Rank
from cohort.core.units import Roster, Soldier


def _squad() -> Roster:
    """SL1 leads TL1 (RFN1, RFN2) and TL2 (RFN3, RFN4)."""
    soldiers = [
        Soldier(id=0, callsign="SL1", rank=Rank.SL, pos=(0, 0), subordinate_ids=[1, 4]),
        Soldier(id=1, callsign="TL1", rank=Rank.TL, pos=(1, 0), leader_id=0, subordinate_ids=[2, 3]),
        Soldier(id=2, callsign="RFN1", rank=Rank.RFN, pos=(2, 0), leader_id=1),
        Soldier(id=3, callsign="RFN2", rank=Rank.RFN, pos=(3, 0), leader_id=1),
        Soldier(id=4, callsign="TL2", rank=Rank.TL, pos=(4, 0), leader_id=0, subordinate_ids=[5, 6]),
        Soldier(id=5, callsign="RFN3", rank=Rank.RFN, pos=(5, 0), leader_id=4),
        Soldier(id=6, callsign="RFN4", rank=Rank.RFN, pos=(6, 0), leader_id=4),
    ]
    return Roster(soldiers)


def test_leader_death_promotes_senior_subordinate_recursively():
    roster = _squad()
    cdg = roster.by_callsign["SL1"]
    cdg.mission = Mission(MissionType.SEIZE, 0, (9, 9), issuer_id=-1, step_assigned=0)
    cdg.alive = False
    events = roster.succeed(cdg)

    tl1 = roster.by_callsign["TL1"]
    # TL1 (senior, lowest id among equals) assumes squad command
    assert tl1.effective_rank is Rank.SL
    assert tl1.leader_id is None
    assert tl1.mission is not None and tl1.mission.type is MissionType.SEIZE, "mission continuity"
    assert roster.root() is tl1

    # TL2 now reports to TL1
    assert roster.by_callsign["TL2"].leader_id == tl1.id
    # a rifleman from TL1's old team was promoted to acting TL over that team
    promoted = roster.by_callsign["RFN1"]
    assert promoted.effective_rank is Rank.TL
    assert promoted.leader_id == tl1.id
    assert promoted.id in tl1.subordinate_ids
    assert roster.by_callsign["RFN2"].leader_id == promoted.id
    assert len(events) == 2


def test_deputy_preferred_over_higher_authority_subordinate():
    soldiers = [
        Soldier(id=0, callsign="SL1", rank=Rank.SL, pos=(0, 0), subordinate_ids=[1, 2], deputy_id=2),
        Soldier(id=1, callsign="TL1", rank=Rank.TL, pos=(1, 0), leader_id=0),
        Soldier(id=2, callsign="RFN1", rank=Rank.RFN, pos=(2, 0), leader_id=0),
    ]
    roster = Roster(soldiers)
    cdg = soldiers[0]
    cdg.alive = False
    roster.succeed(cdg)
    deputy = roster.by_callsign["RFN1"]
    assert deputy.effective_rank is Rank.SL, "designated deputy takes over despite lower rank"
    assert roster.by_callsign["TL1"].leader_id == deputy.id


def test_dead_deputy_falls_back_to_seniority():
    soldiers = [
        Soldier(id=0, callsign="SL1", rank=Rank.SL, pos=(0, 0), subordinate_ids=[1, 2], deputy_id=2),
        Soldier(id=1, callsign="TL1", rank=Rank.TL, pos=(1, 0), leader_id=0),
        Soldier(id=2, callsign="RFN1", rank=Rank.RFN, pos=(2, 0), leader_id=0, alive=False),
    ]
    roster = Roster(soldiers)
    soldiers[0].alive = False
    roster.succeed(soldiers[0])
    assert roster.by_callsign["TL1"].effective_rank is Rank.SL


def test_last_man_standing_no_successor():
    soldiers = [
        Soldier(id=0, callsign="TL1", rank=Rank.TL, pos=(0, 0), subordinate_ids=[1]),
        Soldier(id=1, callsign="RFN1", rank=Rank.RFN, pos=(1, 0), leader_id=0, alive=False),
    ]
    roster = Roster(soldiers)
    soldiers[0].alive = False
    assert roster.succeed(soldiers[0]) == []


def test_succession_announced_on_the_net():
    """In the live env, a death must produce CASUALTY + TAKING COMMAND traffic."""
    from cohort import make_env

    env = make_env("fireteam")
    env.reset(seed=5)
    cap = env.roster.by_callsign["TL1"]
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
    assert new_root.effective_rank is Rank.TL
    assert new_root.rank is Rank.RFN


def test_casualty_from_hq_and_succession_text_from_language():
    """CASUALTY comes from HQ (the dead don't transmit) and every succession
    announcement — both shapes — is built by a core/language formatter."""
    from cohort import make_env
    from cohort.core import language as lang
    from cohort.core.orders import HQ_ID

    env = make_env("squad")
    env.reset(seed=5)
    sl = env.roster.by_callsign["SL1"]
    sl.health = 1
    enemy = next(e for e in env.enemies if e.alive)
    sl.pos = (enemy.pos[0] + 1, enemy.pos[1])
    for _ in range(40):
        if not sl.alive:
            break
        env.step({a: 0 for a in env.agents})
    assert not sl.alive, "squad leader should have been killed by the adjacent garrison"
    casualty = next(m for m in env.transcript.messages if m.kind.value == "casualty")
    assert casualty.sender_id == HQ_ID, "the dead do not transmit"
    takes = [m for m in env.transcript.messages if m.kind.value == "taking_command"]
    assert len(takes) >= 2, "SL death must cascade (TL moves up, RFN fills the TL slot)"
    for m in takes:
        p = m.payload
        expected = (
            lang.format_taking_command(p["successor"], p["replaced"])
            if p["assumed_command"]
            else lang.format_assuming_position(p["successor"], p["replaced"])
        )
        assert m.text == expected, "env must not assemble radio text inline"
    texts = [m.text for m in takes]
    assert any("ASSUMING COMMAND" in t for t in texts)
    assert any("'S POSITION" in t for t in texts)
