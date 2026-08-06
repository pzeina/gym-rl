#!/usr/bin/env python
"""Oracle behavioural fact-sheet for a checkpoint — FIXED columns, no opinions.

CLAUDE.md says "diagnose with the oracle BEFORE changing rewards", and the
project's best findings came from exactly that: the v6 defend TL firing on
0.5% of its opportunities, v4's deaths all landing at the objective, combat pay
out-earning compliance. Every one of those was hand-written throwaway analysis,
rewritten per campaign — so the numbers were never quite comparable run to run.

This script fixes the questions so the answers compare. Same columns every
time, over a seeded episode block, straight from ``env.oracle()`` (read-only,
RNG-free, never feeds observations or rewards).

    scripts/oracle_probe.py runs/<run>/ckpt_best.pt --episodes 30
    scripts/oracle_probe.py runs/<a>/ckpt_best.pt --vs runs/<b>/ckpt_best.pt

The central concept is **under threat**: an agent-step where a living enemy
stands within weapon range with line of sight — i.e. a step where firing was
physically possible. Rates conditioned on it answer "when it could have fought,
did it?" rather than "how often did it fight", which is what diagnosis needs.

Prints facts and exits. It does not interpret, recommend, or compare against a
DoD — that is the caller's job (see .claude/agents/oracle-diagnose.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cohort.core.missions import IN_POSITION_RADIUS
from cohort.core.world import dist
from cohort.env.cohort_env import make_env
from cohort.training.evaluate import _pick_actions
from cohort.training.train import load_policy


class Accum:
    """Running sums for one checkpoint's episode block."""

    def __init__(self) -> None:
        self.episodes = 0
        self.outcomes: Counter[str] = Counter()
        self.steps = 0
        # agent-steps under threat, split by role
        self.threat = Counter()          # role -> threatened agent-steps
        self.threat_fired = Counter()    # role -> ...of which fired
        self.threat_cover = Counter()    # role -> ...of which in cover
        self.threat_dist = Counter()     # role -> sum of dist to root objective
        self.threat_mission: Counter[str] = Counter()   # mission name under threat
        self.deaths_at_objective = 0
        self.deaths_in_the_open = 0
        self.human_deaths = 0
        self.enemy_kills = 0
        self.enemies_total = 0
        # preparation period (v1.10), when the scenario has one
        self.prep_steps = 0
        self.prep_in_cover_at_obj = 0
        self.prep_agent_steps = 0

    # -- rates -------------------------------------------------------- #
    def rate(self, num: Counter, role: str) -> float | None:
        d = self.threat[role]
        return (num[role] / d) if d else None


def _role(soldier_dict: dict, human_cs: str | None) -> str:
    """Bucket an agent: the human commander, any other leader, or a rifleman.

    The v6 diagnosis turned on the human TL behaving differently from its own
    riflemen, so that split is a fixed column rather than something to
    rediscover each campaign.
    """
    if human_cs is not None and soldier_dict["cs"] == human_cs:
        return "human"
    return "rifleman" if soldier_dict["eff"] == "RFN" else "leader"


def probe(
    checkpoint: str | None,
    scenario: str | None,
    episodes: int,
    first_seed: int,
    greedy: bool,
) -> tuple[Accum, str]:
    net = None
    if checkpoint is not None:
        net, ckpt = load_policy(checkpoint)
        scenario = scenario or ckpt.get("scenario")
    if scenario is None:
        raise SystemExit("need --scenario when probing without a checkpoint")

    env = make_env(scenario)
    acc = Accum()
    combat = env.combat
    root_name = env.spec_cfg.root_objective
    radius = IN_POSITION_RADIUS[env.spec_cfg.root_mission]

    for k in range(episodes):
        ep_seed = first_seed + k
        torch.manual_seed(ep_seed)
        rng = np.random.default_rng(ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        human_cs = next((s.callsign for s in env.roster.soldiers if s.human), None)
        root_obj = env.world.objective_by_name(root_name) if root_name else None
        obj_pos = root_obj.pos if root_obj is not None else None
        alive_before = {s.callsign: True for s in env.roster.soldiers}
        acc.episodes += 1
        acc.enemies_total += env.spec_cfg.n_enemies

        while env.agents:
            in_prep = env._in_preparation()
            actions = _pick_actions(env, obs, net, rng, greedy=greedy)
            obs, _, term, trunc, _ = env.step(actions)
            snap = env.oracle()
            acc.steps += 1

            living_enemy_pos = [e["pos"] for e in snap["enemies"] if e["alive"]]
            for sd in snap["soldiers"]:
                if not sd["alive"]:
                    # death accounting, once, on the transition
                    if alive_before.get(sd["cs"]):
                        alive_before[sd["cs"]] = False
                        if obj_pos is not None and dist(sd["pos"], obj_pos) <= radius:
                            acc.deaths_at_objective += 1
                        else:
                            acc.deaths_in_the_open += 1
                        if sd["cs"] == human_cs:
                            acc.human_deaths += 1
                    continue

                role = _role(sd, human_cs)
                if in_prep:
                    acc.prep_agent_steps += 1
                    if (
                        obj_pos is not None
                        and dist(sd["pos"], obj_pos) <= radius
                        and sd["cover"]
                    ):
                        acc.prep_in_cover_at_obj += 1

                # under threat: could this agent have fired this step?
                threatened = any(
                    dist(sd["pos"], p) <= combat.weapon_range
                    and env.world.line_of_sight(tuple(sd["pos"]), tuple(p))
                    for p in living_enemy_pos
                )
                if not threatened:
                    continue
                acc.threat[role] += 1
                acc.threat["team"] += 1
                if sd["fired"]:
                    acc.threat_fired[role] += 1
                    acc.threat_fired["team"] += 1
                if sd["cover"]:
                    acc.threat_cover[role] += 1
                    acc.threat_cover["team"] += 1
                if obj_pos is not None:
                    d = dist(sd["pos"], obj_pos)
                    acc.threat_dist[role] += d
                    acc.threat_dist["team"] += d
                acc.threat_mission[sd["mission"] or "NONE"] += 1

            if in_prep:
                acc.prep_steps += 1
            if all(term.values()) or all(trunc.values()):
                break

        acc.outcomes[env.outcome or "timeout"] += 1
        acc.enemy_kills += sum(1 for e in env.enemies if not e.alive)

    return acc, scenario


# ---------------------------------------------------------------------- #
# reporting
# ---------------------------------------------------------------------- #


def _fmt(v: float | None, spec: str = "{:.3f}") -> str:
    return "  n/a" if v is None else spec.format(v)


def _row(label: str, a: float | None, b: float | None, spec: str = "{:.3f}") -> str:
    if b is None:
        return f"  {label:<26} {_fmt(a, spec):>8}"
    return f"  {label:<26} {_fmt(a, spec):>8}  {_fmt(b, spec):>8}"


def report(acc: Accum, name: str, other: Accum | None, other_name: str | None) -> None:
    def pair(fn):
        return fn(acc), (fn(other) if other is not None else None)

    hdr = f"{'':<28}{name[:8]:>8}"
    if other_name:
        hdr += f"  {other_name[:8]:>8}"
    print(hdr)
    print("-" * len(hdr))

    print("UNDER THREAT (a living enemy in weapon range with LOS)")
    for role in ("team", "human", "leader", "rifleman"):
        a, b = pair(lambda x, r=role: x.rate(x.threat_fired, r))
        print(_row(f"fire rate [{role}]", a, b))
    for role in ("team", "human"):
        a, b = pair(lambda x, r=role: x.rate(x.threat_cover, r))
        print(_row(f"cover occupancy [{role}]", a, b))
    a, b = pair(
        lambda x: (x.threat_dist["team"] / x.threat["team"]) if x.threat["team"] else None
    )
    print(_row("dist from root OBJ", a, b, "{:.2f}"))
    a, b = pair(lambda x: x.threat["team"] / x.episodes if x.episodes else None)
    print(_row("threatened steps/ep", a, b, "{:.1f}"))

    print("MISSION MIX UNDER THREAT (share of threatened agent-steps)")
    names = sorted(
        acc.threat_mission, key=lambda k: -acc.threat_mission[k]
    )[:5]
    for m in names:
        tot = sum(acc.threat_mission.values()) or 1
        a = acc.threat_mission[m] / tot
        b = None
        if other is not None:
            otot = sum(other.threat_mission.values()) or 1
            b = other.threat_mission[m] / otot
        print(_row(f"{m}", a, b))

    print("CASUALTIES")
    a, b = pair(lambda x: x.deaths_at_objective / x.episodes if x.episodes else None)
    print(_row("friendly deaths at OBJ/ep", a, b, "{:.2f}"))
    a, b = pair(lambda x: x.deaths_in_the_open / x.episodes if x.episodes else None)
    print(_row("friendly deaths open/ep", a, b, "{:.2f}"))
    a, b = pair(lambda x: x.human_deaths / x.episodes if x.episodes else None)
    print(_row("human death rate", a, b))
    a, b = pair(
        lambda x: x.enemy_kills / x.enemies_total if x.enemies_total else None
    )
    print(_row("enemies killed (share)", a, b))

    if acc.prep_agent_steps or (other and other.prep_agent_steps):
        print("PREPARATION PERIOD (v1.10 scenarios only)")
        a, b = pair(
            lambda x: (x.prep_in_cover_at_obj / x.prep_agent_steps)
            if x.prep_agent_steps
            else None
        )
        print(_row("in cover at OBJ (prep)", a, b))
        a, b = pair(lambda x: x.prep_steps / x.episodes if x.episodes else None)
        print(_row("prep steps/ep", a, b, "{:.1f}"))

    print("OUTCOMES")
    keys = sorted(set(acc.outcomes) | set(other.outcomes if other else {}))
    for k in keys:
        a = acc.outcomes[k] / acc.episodes if acc.episodes else None
        b = (other.outcomes[k] / other.episodes) if other and other.episodes else None
        print(_row(k, a, b))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Oracle behavioural fact-sheet (fixed columns, no interpretation)."
    )
    p.add_argument("checkpoint", nargs="?", default=None)
    p.add_argument("--vs", default=None, help="baseline checkpoint to compare against")
    p.add_argument("--scenario", default=None)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument(
        "--seed", type=int, default=500, help="first episode seed (block is seed..seed+n-1)"
    )
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--json-out", default=None, help="also write the raw counters here")
    args = p.parse_args()

    acc, scenario = probe(
        args.checkpoint, args.scenario, args.episodes, args.seed, args.greedy
    )
    other = other_name = None
    if args.vs:
        other, _ = probe(args.vs, args.scenario, args.episodes, args.seed, args.greedy)
        other_name = Path(args.vs).parent.name

    name = Path(args.checkpoint).parent.name if args.checkpoint else "random"
    print(
        f"oracle probe: {scenario}  episodes={args.episodes} "
        f"seeds={args.seed}..{args.seed + args.episodes - 1} "
        f"{'greedy' if args.greedy else 'sampled'}"
    )
    print()
    report(acc, name, other, other_name)

    if args.json_out:
        payload = {
            "scenario": scenario,
            "episodes": args.episodes,
            "first_seed": args.seed,
            "run": name,
            "threat": dict(acc.threat),
            "threat_fired": dict(acc.threat_fired),
            "threat_cover": dict(acc.threat_cover),
            "threat_mission": dict(acc.threat_mission),
            "deaths_at_objective": acc.deaths_at_objective,
            "deaths_in_the_open": acc.deaths_in_the_open,
            "human_deaths": acc.human_deaths,
            "outcomes": dict(acc.outcomes),
            "prep_agent_steps": acc.prep_agent_steps,
            "prep_in_cover_at_obj": acc.prep_in_cover_at_obj,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nraw counters → {args.json_out}")


if __name__ == "__main__":
    main()
