"""Assurance event tap: serialize episodes as (observable, truth) JSONL streams.

This module exists for the benefit of an *external* assurance layer (see the
EPISTREAM-RL project). It runs seeded episodes with a fixed checkpoint and
writes two strictly separated streams:

* the **observable stream** (``--out``): every radio message on the net, plus
  per-message observer sets. This is everything a monitor attached to the
  radio net may legitimately consume.
* the **truth stream** (``--truth``): per-step ground truth (positions, alive
  sets, standing missions, enemy count, episode outcome). This stream is for
  *scoring* a monitor's verdicts only; feeding it into a belief computation
  would make any knowledge claim circular.

The tap alters nothing about the environment or the policy: it drives the
same rollout loop as ``cohort.training.evaluate`` and reads public state.

Observer convention (the tap's encoding obligation): a message at step ``t``
is observed by ``HQ`` and by every soldier alive at the *end* of step ``t``.
The net is a single shared channel — cohort has no per-listener radio
propagation — so partial observability lives in private sightings, not in
comms. Casualties therefore stop observing from their death step onward.

Usage:
    python -m cohort.tap runs/squad_v1/ckpt_best.pt --episodes 30 --seed 500 \
        --out events.jsonl.gz --truth truth.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from typing import IO

import numpy as np
import torch

from cohort.core.orders import HQ_ID, Message
from cohort.env.cohort_env import CohortEnv, make_env
from cohort.training.evaluate import _pick_actions

TAP_SCHEMA = "1.0.0"


def _open(path: str) -> IO[str]:
    if path.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def _callsign(env: CohortEnv, agent_id: int) -> str:
    if agent_id == HQ_ID:
        return "HQ"
    return env.roster.by_id[agent_id].callsign


#: Canonical mission phrase, mirroring core.language.mission_phrase exactly.
_PHRASE_RE = re.compile(r"(?P<task>[A-Z]+)(?: OBJ (?P<obj>[A-Z]+)| ON ME| POSITION)")
_CONTACT_RE = re.compile(r"CONTACT, GRID (?P<grid>\d{4}), (?P<n>\d+) x ENEMY")
_SITREP_RE = re.compile(r"SITREP, GRID (?P<grid>\d{4}), HEALTH (?P<health>\d+)%, AMMO (?P<ammo>\d+)")
_CASUALTY_RE = re.compile(r"ALL STATIONS: (?P<cs>[A-Z]+\d+) IS DOWN")
_SUCCESSION_RE = re.compile(r"(?:(?P<dead>[A-Z]+\d+) IS DOWN\. I AM ASSUMING COMMAND|ASSUMING (?P<dead2>[A-Z]+\d+)'S POSITION)")


def _parse_phrase(text: str) -> dict:
    """Extract {task, objective} from a canonical mission phrase in ``text``.

    Round-trip check: re-assembling the phrase from the parsed parts must
    reproduce a substring of the original text, so a silent format change
    upstream fails here instead of shipping wrong payloads.
    """
    m = _PHRASE_RE.search(text)
    if m is None:
        raise ValueError(f"unparseable mission phrase in {text!r}")
    task, obj = m.group("task"), m.group("obj")
    rebuilt = f"{task} OBJ {obj}" if obj else (f"{task} ON ME" if task == "RALLY" else f"{task} POSITION")
    if rebuilt not in text:
        raise ValueError(f"round-trip mismatch: {rebuilt!r} not in {text!r}")
    return {"task": task.lower(), "objective": obj}


def _check_native(kind: str, native: dict, parsed: dict) -> None:
    """Cross-check the env's native payload (issue #5 fix) against the tap's
    independent text parsers.

    The two were built from opposite ends -- the env emits structure at the
    source, the tap re-derives it from the canonical radio text -- so their
    agreement on every message is an external verification of the #5 fix
    (and of the text formats staying faithful). Divergence crashes corpus
    generation loudly. Empty native payload (pre-fix code) skips the check.
    """
    if not native:
        return
    mismatches = []
    if kind in ("opord", "order", "done"):
        if str(native.get("mission", "")).lower() != parsed.get("task"):
            mismatches.append("mission")
        if native.get("objective") != parsed.get("objective"):
            mismatches.append("objective")
    elif kind == "contact":
        g = native.get("grid", [None, None])
        if f"{int(g[0]):02d}{int(g[1]):02d}" != parsed.get("grid"):
            mismatches.append("grid")
        if native.get("count") != parsed.get("count"):
            mismatches.append("count")
    elif kind == "sitrep":
        g = native.get("grid", [None, None])
        if f"{int(g[0]):02d}{int(g[1]):02d}" != parsed.get("grid"):
            mismatches.append("grid")
        if native.get("health") != parsed.get("health") or native.get("ammo") != parsed.get("ammo"):
            mismatches.append("health/ammo")
    elif kind == "casualty":
        if native.get("callsign") != parsed.get("casualty"):
            mismatches.append("callsign")
    elif kind == "taking_command":
        if native.get("replaced") != parsed.get("replaced"):
            mismatches.append("replaced")
    elif kind == "done_confirm":
        if str(native.get("mission", "")).lower() != parsed.get("task"):
            mismatches.append("mission")
        if native.get("verdict") != "confirmed":
            mismatches.append("verdict")
    elif kind == "done_reject":
        if native.get("verdict") != "rejected":
            mismatches.append("verdict")
    if mismatches:
        raise ValueError(f"native payload disagrees with parsed text for {kind}: {mismatches} ({native} vs {parsed})")


def _payload(m: Message) -> dict:
    """Structured content of a message, parsed from its canonical text form.

    Cross-checked against the env's native payload where present (#5)."""
    kind = m.kind.value
    if kind in ("opord", "order", "done"):
        parsed = _parse_phrase(m.text)
    elif kind == "contact":
        c = _CONTACT_RE.search(m.text)
        parsed = {"grid": c.group("grid"), "count": int(c.group("n"))} if c else {}
    elif kind == "sitrep":
        s = _SITREP_RE.search(m.text)
        parsed = {"grid": s.group("grid"), "health": int(s.group("health")), "ammo": int(s.group("ammo"))} if s else {}
    elif kind == "casualty":
        c = _CASUALTY_RE.search(m.text)
        parsed = {"casualty": c.group("cs")} if c else {}
    elif kind == "taking_command":
        s = _SUCCESSION_RE.search(m.text)
        parsed = {"replaced": (s.group("dead") or s.group("dead2"))} if s else {}
    elif kind == "done_confirm":
        # "RFN1, THIS IS TL1: ROGER, SEIZE OBJ ALPHA CONFIRMED. OUT."
        parsed = {**_parse_phrase(m.text), "verdict": "confirmed"}
    elif kind == "done_reject":
        # "RFN1, THIS IS TL1: NEGATIVE, CONTINUE MISSION. OUT." -- the text
        # names no mission; the claimant's stands (that is the point).
        parsed = {"verdict": "rejected"}
    else:
        parsed = {}
    _check_native(kind, dict(m.payload), parsed)
    return parsed


def _msg_record(env: CohortEnv, episode: int, m: Message, observers: list[str]) -> dict:
    return {
        "rec": "msg",
        "episode": episode,
        "step": m.step,
        "kind": m.kind.value,
        "sender": _callsign(env, m.sender_id),
        "recipient": None if m.recipient_id is None else _callsign(env, m.recipient_id),
        "text": m.text,
        "payload": _payload(m),
        "observers": observers,
    }


def _state_record(env: CohortEnv, episode: int, step: int) -> dict:
    mission = {}
    for s in env.roster.soldiers:
        if s.mission is None:
            mission[s.callsign] = None
        else:
            obj_id = s.mission.objective_id
            mission[s.callsign] = {
                "type": s.mission.type.value,
                "objective": None if obj_id is None else env.world.objectives[obj_id].name,
            }
    return {
        "rec": "state",
        "episode": episode,
        "step": step,
        "alive": sorted(s.callsign for s in env.roster.living),
        "pos": {s.callsign: [int(s.pos[0]), int(s.pos[1])] for s in env.roster.soldiers},
        "effective_rank": {s.callsign: s.effective_rank.name for s in env.roster.soldiers},
        "mission": mission,
        "enemies_alive": sum(e.alive for e in env.enemies),
    }


def tap_episodes(
    checkpoint: str | None,
    scenario: str | None,
    episodes: int,
    seed: int,
    out_path: str,
    truth_path: str,
    *,
    greedy: bool = False,
) -> dict:
    """Run seeded episodes and write the two streams. Returns summary counts."""
    net = None
    if checkpoint is not None:
        from cohort.training.train import load_policy

        net, ckpt = load_policy(checkpoint)
        scenario = scenario or ckpt["scenario"]
    if scenario is None:
        raise ValueError("Need a scenario when tapping the random baseline.")

    env = make_env(scenario)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    n_msgs = 0
    outcomes: dict[str, int] = {}
    with _open(out_path) as out, _open(truth_path) as truth:
        header = {
            "rec": "header",
            "tap_schema": TAP_SCHEMA,
            "scenario": scenario,
            "checkpoint": checkpoint,
            "episodes": episodes,
            "base_seed": seed,
            "greedy": greedy,
            "policy": "checkpoint" if net is not None else "masked-random",
        }
        out.write(json.dumps(header, sort_keys=True) + "\n")
        truth.write(json.dumps(header, sort_keys=True) + "\n")

        for ep in range(episodes):
            ep_seed = seed + ep
            obs, _ = env.reset(seed=ep_seed)
            out.write(json.dumps({"rec": "episode", "episode": ep, "seed": ep_seed}, sort_keys=True) + "\n")
            truth.write(json.dumps({"rec": "episode", "episode": ep, "seed": ep_seed}, sort_keys=True) + "\n")
            truth.write(json.dumps(_state_record(env, ep, 0), sort_keys=True) + "\n")

            watermark = 0
            steps = 0
            while env.agents:
                actions = _pick_actions(env, obs, net, rng, greedy=greedy)
                obs, _rew, _term, _trunc, _info = env.step(actions)
                steps += 1
                observers = ["HQ"] + sorted(s.callsign for s in env.roster.living)
                for m in env.transcript.since(watermark):
                    out.write(json.dumps(_msg_record(env, ep, m, observers), sort_keys=True) + "\n")
                    n_msgs += 1
                watermark = len(env.transcript)
                truth.write(json.dumps(_state_record(env, ep, steps), sort_keys=True) + "\n")

            outcome = env.outcome or "timeout"
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            end = {
                "rec": "end",
                "episode": ep,
                "outcome": outcome,
                "steps": steps,
                "survivors": sum(s.alive for s in env.roster.soldiers),
            }
            out.write(json.dumps(end, sort_keys=True) + "\n")
            truth.write(json.dumps(end, sort_keys=True) + "\n")

    return {"episodes": episodes, "messages": n_msgs, "outcomes": outcomes}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Tap cohort episodes into assurance event streams.")
    parser.add_argument("checkpoint", nargs="?", default=None)
    parser.add_argument("--random", action="store_true", help="masked-random baseline instead of a checkpoint")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--out", required=True, help="observable stream path (.jsonl or .jsonl.gz)")
    parser.add_argument("--truth", required=True, help="truth stream path (.jsonl or .jsonl.gz)")
    parser.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    args = parser.parse_args()
    checkpoint = None if args.random else args.checkpoint
    if checkpoint is None and not args.random:
        parser.error("Provide a checkpoint path or --random.")
    summary = tap_episodes(
        checkpoint,
        args.scenario,
        args.episodes,
        args.seed,
        args.out,
        args.truth,
        greedy=args.greedy,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
