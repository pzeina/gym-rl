"""Interactive commander console: speak to the cohort in its own language.

You act as HQ (or as any commander callsign) and type orders exactly as the
agents transmit them to each other; trained policies drive every soldier.

    python -m cohort.play --checkpoint runs/fireteam_v2/ckpt_best.pt
    python -m cohort.play --scenario squad --as SL1

Console commands:
    TL1, seize obj bravo       inject an order (any command-language line)
    <enter> or s [n]           advance n steps (default 5)
    m                          show the map
    net [n]                    show the last n radio messages (default 15)
    status                     roster: rank, position, health, mission
    help                       this list
    q                          quit
"""

from __future__ import annotations

import argparse

import numpy as np

from cohort.core.language import OrderParseError
from cohort.env.cohort_env import make_env
from cohort.training.evaluate import _pick_actions

HELP = __doc__.split("Console commands:")[1]


def _print_status(env) -> None:
    for s in env.roster.soldiers:
        state = "KIA" if not s.alive else f"{s.health:>3}hp {s.ammo:>2}rds"
        mission = s.mission.type.name if (s.alive and s.mission) else "STANDBY"
        acting = f" (acting {s.effective_rank.name})" if s.effective_rank is not s.rank else ""
        human = " HUMAN" if s.human else ""
        leader = env.roster.by_id[s.leader_id].callsign if s.leader_id is not None else "HQ"
        print(f"  {s.callsign:>5} [{s.rank.name}{acting}{human}] → {leader:<5} {state}  {mission}  @{s.pos}")


def _advance(env, obs, net, rng, n: int) -> dict:
    log_start = len(env.transcript)
    for _ in range(n):
        if not env.agents:
            break
        actions = _pick_actions(env, obs, net, rng)
        obs, *_ = env.step(actions)
    for m in env.transcript.since(log_start):
        print(f"  [t={m.step:>3}] {m.text}")
    return obs


def main() -> None:
    """Run the commander console."""
    parser = argparse.ArgumentParser(description="Command the cohort interactively.")
    parser.add_argument("--checkpoint", default=None, help="trained policy (.pt); omit for masked-random agents")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--as", dest="issuer", default="HQ", help="speak as HQ (default) or a commander callsign")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    net = None
    scenario = args.scenario
    if args.checkpoint:
        from cohort.training.train import load_policy

        net, ckpt = load_policy(args.checkpoint)
        scenario = scenario or ckpt["scenario"]
    scenario = scenario or "fireteam"

    env = make_env(scenario)
    rng = np.random.default_rng(args.seed)
    obs, _ = env.reset(seed=args.seed)
    print(f"scenario: {scenario} — you are {args.issuer}. Type 'help' for commands.\n")
    print(env._render_ansi())
    print(f"  [t=  0] {env.transcript.messages[0].text}")

    while True:
        if not env.agents:
            print(f"\n*** episode over: {env._episode_outcome} ***")
            _print_status(env)
            break
        try:
            line = input(f"\n{args.issuer}> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        low = line.lower()
        if low in ("q", "quit", "exit"):
            break
        if low in ("help", "h", "?"):
            print(HELP)
        elif low in ("m", "map"):
            print(env._render_ansi())
        elif low == "status":
            _print_status(env)
        elif low.startswith("net"):
            parts = low.split()
            n = int(parts[1]) if len(parts) > 1 else 15
            for m in env.transcript.messages[-n:]:
                print(f"  [t={m.step:>3}] {m.text}")
        elif low == "" or low.startswith("s"):
            parts = low.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            obs = _advance(env, obs, net, rng, n)
            print(env._render_ansi())
        else:
            try:
                msg = env.inject_order(line, issuer=args.issuer)
                print(f"  [t={msg.step:>3}] {msg.text}")
            except (OrderParseError, PermissionError) as exc:
                print(f"  !! {exc}")


if __name__ == "__main__":
    main()
