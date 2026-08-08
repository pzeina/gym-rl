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
import hashlib
import json
import pathlib
import re
import subprocess
from dataclasses import replace
from typing import IO

import numpy as np
import torch

from cohort.config import briefing, get_scenario
from cohort.core.language import parse_opord, parse_order, parse_sitrep
from cohort.core.oracle import observe as oracle_observe
from cohort.core.orders import HQ_ID, Message
from cohort.env.cohort_env import CohortEnv, make_env
from cohort.training.evaluate import _pick_actions

#: 1.1.0 (v1.10 era): SITREP payloads gain the self-reported ``in_cover``
#: posture, OPORD payloads gain ``announced_assault_step`` where the scenario
#: has a preparation period, and the header gains the ``briefing`` overlay.
#: All three are additive -- consumers of 1.0.0 corpora see missing keys, not
#: changed ones -- but the schema is bumped so a corpus states what it can be
#: asked for rather than leaving the layer to probe.
TAP_SCHEMA = "1.2.0"


def _open(path: str) -> IO[str]:
    if path.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def _file_sha256(path: str) -> str:
    """SHA-256 of a checkpoint file, read in chunks (weights run to ~700 KB).

    Stamped into the tap header so a corpus identifies the weights that
    produced it by content rather than by path -- see the header comment.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cohort_provenance() -> dict[str, str | bool | None]:
    """Which `cohort` package this process actually imported, and at what commit.

    Recorded because it is NOT the directory this script lives in, and that
    surprise invalidated a verification (assurance PLAN.md 12.82). Running
    `python cohort/tap.py` puts sys.path[0] at the SCRIPT'S directory, so
    `import cohort.config` misses the sibling package and falls through to
    whatever `cohort` the interpreter has installed -- here an editable
    install pointing at the upstream working tree. Consequence: checking out
    an older commit and re-tapping does not roll the environment back. Two
    taps from two different checkouts both execute the upstream tree and come
    out byte-identical, which reads as "the change is inert" when in fact the
    change was never exercised. Pin the import with PYTHONPATH to compare
    checkouts.

    Stamping the resolved path and commit into the header makes that failure
    self-evident instead of silent: two corpora claiming to bracket a change
    while carrying the same `cohort_commit` did not bracket anything.
    """
    import cohort

    root = str(pathlib.Path(cohort.__file__).resolve().parent.parent)
    commit: str | None = None
    dirty: bool | None = None
    try:
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", root, "status", "--porcelain"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
        )
    except (subprocess.SubprocessError, OSError):
        pass  # not a checkout, or no git -- absent beats a wrong value
    return {"cohort_source": root, "cohort_commit": commit, "cohort_dirty": dirty}


def _callsign(env: CohortEnv, agent_id: int) -> str:
    if agent_id == HQ_ID:
        return "HQ"
    return env.roster.by_id[agent_id].callsign


#: Canonical mission phrase, mirroring core.language.mission_phrase exactly.
#: MICAT forms (v1.3): 'SUPPORT TL1' (unit-targeted), 'COVER FLANK OBJ X';
#: objective names carry no digits, callsigns always do -- the last
#: alternative is unambiguous.
_PHRASE_RE = re.compile(
    r"(?P<task>[A-Z]+)"
    r"(?: FLANK OBJ (?P<obj2>[A-Z]+)| TO (?P<cmk>WP|PL) (?P<cm>[A-Z]+)| OBJ (?P<obj>[A-Z]+)"
    r"| ON ME| POSITION| (?P<unit>[A-Z]+\d+))"
)
#: A5-3 formation orders: an element stance, not a mission.
_FORMATION_RE = re.compile(r"FORMATION (?P<f>COLUMN|LINE|WEDGE)")
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
    # Scope the search to the message body (after 'RECIPIENT, THIS IS X: ');
    # the MICAT unit-targeted form ('SUPPORT TL1') would otherwise false-match
    # the 'IS SL1' inside the address preamble -- and round-trip cleanly,
    # because the false match is a genuine substring (bug found by the
    # squad_screen random corpus, PLAN.md 12.26).
    body = text.split(": ", 1)[1] if ": " in text else text
    f = _FORMATION_RE.search(body)
    if f is not None:
        # stance order (A5-3): shapes movement, never a mission
        return {"formation": f.group("f").lower()}
    m = _PHRASE_RE.search(body)
    if m is None:
        raise ValueError(f"unparseable mission phrase in {text!r}")
    task = m.group("task")
    obj = m.group("obj") or m.group("obj2")
    unit = m.group("unit")
    cm = m.group("cm")
    if unit:
        rebuilt = f"{task} {unit}"
    elif m.group("obj2"):
        rebuilt = f"{task} FLANK OBJ {obj}"
    elif cm:
        rebuilt = f"{task} TO {m.group('cmk')} {cm}"
    elif obj:
        rebuilt = f"{task} OBJ {obj}"
    else:
        rebuilt = f"{task} ON ME" if task == "RALLY" else f"{task} POSITION"
    if rebuilt not in body:
        raise ValueError(f"round-trip mismatch: {rebuilt!r} not in {text!r}")
    # ADVANCE targets a control measure; the truth stream stores the bare
    # measure name (extra['control']), so the WP/PL prefix is presentation
    # -- strip it here to keep hypotheses comparable.
    return {"task": task.lower(), "objective": cm or obj, "unit": unit}


def _payload(m: Message) -> dict:
    """Structured content of a message, parsed from its canonical text form.

    The net carries text only -- structured payloads on messages are
    forbidden by upstream design (owner decision at 12f54dd; the earlier
    native-payload cross-check era is recorded in this file's history). The
    tap's parsers, with their round-trip self-checks, are therefore the
    single re-derivation of structure from what was actually said, which is
    exactly the epistemically honest arrangement: consumers get structure
    derived from the observable, never a side-channel."""
    kind = m.kind.value
    if kind in ("opord", "order", "done"):
        parsed = _parse_phrase(m.text)
        # v1.10 preparation period: HQ announces when the assault is due. The
        # clause sits after the task statement, so _parse_phrase ignores it --
        # dropping it silently was the defect reported as assurance issue #12.
        # It is the net's first FORWARD-LOOKING content: not a report of what
        # happened but an announced expectation, which is what makes
        # time-bounded readiness properties expressible at all.
        if kind == "opord":
            spoken = parse_opord(m.text)
            step = spoken.get("announced_assault_step") if spoken else None
            if step is not None:
                parsed = {**parsed, "announced_assault_step": step}
        # A5-2 timing qualifiers. An AT MY COMMAND order is STAGED: the
        # mission is assigned at emission but pending until the issuer's
        # EXECUTE, so the interval in between is not disobedience. Without
        # these fields a monitor sees the EXECUTE traffic but cannot tell
        # which orders were staged, which is exactly what upstream's
        # obedience-latency hypothesis turns on (EPISTREAM PLAN.md 12.53).
        if kind in ("opord", "order"):
            try:
                spoken_order = parse_order(m.text)
            except Exception:
                spoken_order = None
            if spoken_order is not None:
                if spoken_order.delay is not None:
                    parsed = {**parsed, "delay": spoken_order.delay}
                if spoken_order.at_my_command:
                    parsed = {**parsed, "at_my_command": True}
    elif kind == "contact":
        c = _CONTACT_RE.search(m.text)
        parsed = {"grid": c.group("grid"), "count": int(c.group("n"))} if c else {}
    elif kind == "sitrep":
        # Parsed by the system's own shipped inverse (upstream 6b75cce) rather
        # than a hand-rolled regex -- exactly the drift the ops-overlay commit
        # set out to prevent. The local regex stays as a cross-check: the two
        # must agree on the fields they share, or the format moved under us.
        spoken = parse_sitrep(m.text)
        s = _SITREP_RE.search(m.text)
        if spoken is None or s is None:
            parsed = {}
        else:
            grid = f"{spoken['grid'][0]:02d}{spoken['grid'][1]:02d}"
            if (grid, spoken["health"], spoken["ammo"]) != (
                s.group("grid"), int(s.group("health")), int(s.group("ammo"))
            ):
                raise ValueError(f"SITREP parser disagreement on {m.text!r}")
            parsed = {
                "grid": grid,
                "health": spoken["health"],
                "ammo": spoken["ammo"],
                # Self-reported terrain posture (issue #10). NOT a readout of
                # the ground: it is what the soldier says, so it stays
                # radio-legitimate and belongs in the observable stream.
                "in_cover": spoken["in_cover"],
            }
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
    elif kind == "support_end":
        # "TL1, THIS IS RFN1: SUPPORT ENDED, RFN2 IS DOWN. STANDING BY. OVER."
        s = re.search(r"SUPPORT ENDED, (?P<cs>[A-Z]+\d+) IS DOWN", m.text)
        parsed = {"supported": s.group("cs")} if s else {}
    elif kind == "trap":
        # "ALL STATIONS: RFN1 HIT A DEVICE AT GRID 1407. OUT." (BRIQUE)
        s = re.search(r"(?P<cs>[A-Z]+\d+) HIT A DEVICE AT GRID (?P<grid>\d{4})", m.text)
        parsed = {"victim": s.group("cs"), "grid": s.group("grid")} if s else {}
    else:
        parsed = {}
    return parsed


def _msg_record(env: CohortEnv, episode: int, m: Message) -> dict:
    """Serialize one message with its per-listener observer set.

    Observers = HQ plus every soldier alive at end of step that the env's
    own audibility model says can hear the sender (``_audible_to`` — always
    true under ``comm_model="global"``, range-gated under ``"range"``).
    Positions are those at step end, the same approximation as the alive
    set (documented tap convention).
    """
    observers = ["HQ"] + sorted(
        s.callsign for s in env.roster.living if env._audible_to(s, m.sender_id)
    )
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
            # `target` is the normalized order target: objective name, or the
            # supported unit's callsign for unit-targeted SUPPORT (MICAT).
            target = None if obj_id is None else env.world.objectives[obj_id].name
            sup_id = s.mission.extra.get("supported_id")
            if sup_id is not None and sup_id in env.roster.by_id:
                target = env.roster.by_id[sup_id].callsign
            # A5: ADVANCE targets a named control measure ('WP GOLD' /
            # 'PL AMBER'), stored in extra -- surface it as the target so
            # truth matches the order text's phrase.
            if s.mission.extra.get("control"):
                target = s.mission.extra["control"]
            mission[s.callsign] = {
                "type": s.mission.type.value,
                "objective": None if obj_id is None else env.world.objectives[obj_id].name,
                "target": target,
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
        # The sanctioned ground-truth channel (cohort.core.oracle, upstream
        # 8ca106e): behavior observables incl. the OpFor side. Scoring-only,
        # like everything else in the truth stream.
        "oracle": oracle_observe(env),
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
    comm_model: str | None = None,
    comm_range: float | None = None,
) -> dict:
    """Run seeded episodes and write the two streams. Returns summary counts.

    ``comm_model``/``comm_range`` override the scenario's audibility settings.
    Under the shipped default (``"global"``) every living station hears every
    message, so all living agents share one observation history and the
    assurance layer's per-observer projection is degenerate. ``"range"`` gates
    audibility by euclidean distance, which is what gives the layer a
    non-trivial ``obs_a`` to monitor.

    This changes the *system*, not merely the recording: agents genuinely miss
    traffic, so a checkpoint trained under the global net is off-distribution
    here. Corpora produced this way are a comms-degradation arm and must be
    labelled as such -- never a substitute for a range-trained checkpoint.
    """
    net = None
    if checkpoint is not None:
        from cohort.training.train import load_policy

        net, ckpt = load_policy(checkpoint)
        scenario = scenario or ckpt["scenario"]
    if scenario is None:
        raise ValueError("Need a scenario when tapping the random baseline.")

    spec = get_scenario(scenario)
    if comm_model is not None or comm_range is not None:
        if comm_model is not None and comm_model not in ("global", "range"):
            raise ValueError(f"comm_model must be 'global' or 'range', got {comm_model!r}")
        spec = replace(
            spec,
            comm_model=comm_model if comm_model is not None else spec.comm_model,
            comm_range=comm_range if comm_range is not None else spec.comm_range,
        )
    env = make_env(spec)

    n_msgs = 0
    outcomes: dict[str, int] = {}
    with _open(out_path) as out, _open(truth_path) as truth:
        from cohort.core.missions import MissionType

        header = {
            "rec": "header",
            "tap_schema": TAP_SCHEMA,
            # Self-describing mission catalog: the ordered task list of the
            # code that generated this corpus. The assurance layer selects
            # its doctrine premise by this, never by guessing from content.
            "missions": [m.value for m in MissionType],
            "scenario": scenario,
            "checkpoint": checkpoint,
            # Content identity of the weights, not merely where they sat
            # (assurance PLAN.md 12.80). `checkpoint` is an absolute path into
            # a working tree, and upstream republishes runs under the same
            # directory name (`v4` vs `v4b`), so the path alone cannot say
            # WHICH weights produced a corpus. At 12.51 three corpora were
            # tapped from uncommitted checkpoints, and when those files were
            # finally committed the provenance could only be argued
            # behaviourally, never checked. A digest settles it: the corpus
            # carries the identity of the policy that generated it,
            # independently of the tree it was read from.
            #
            # None on the masked-random arm, which has no weights -- absent
            # rather than a default, so a consumer can tell "no policy" from
            # "policy not recorded" (the `briefing_anchor` rule).
            "checkpoint_sha256": _file_sha256(checkpoint) if checkpoint is not None else None,
            # Identity of the CODE that produced this corpus, for the same
            # reason as the digest above -- see `_cohort_provenance`.
            **_cohort_provenance(),
            "episodes": episodes,
            "base_seed": seed,
            "greedy": greedy,
            "policy": "checkpoint" if net is not None else "masked-random",
            # Self-describing audibility, for the same reason as `missions`:
            # the observer sets in this stream mean something different under
            # each model, and the assurance layer must read the regime rather
            # than infer it from observer-set sizes.
            "comm_model": spec.comm_model,
            "comm_range": spec.comm_range,
            # Operations overlay (upstream 6b75cce, assurance issue #10): the
            # static mission facts a real monitor reads off the overlay before
            # H-hour -- objective coordinates, control-measure geometry, map
            # size, terrain guarantees and the engagement envelope. Static and
            # per-scenario, so no per-episode truth leaks; publishing it here
            # retires the hand-pinned, silently era-sensitive coordinate table
            # the layer was carrying (EPISTREAM PLAN.md 12.42/12.45).
            "briefing": briefing(scenario),
        }
        out.write(json.dumps(header, sort_keys=True) + "\n")
        truth.write(json.dumps(header, sort_keys=True) + "\n")

        for ep in range(episodes):
            # Self-contained per-episode seeding (upstream F8 doctrine):
            # episode k reproduces standalone; its sampling streams do not
            # depend on how many draws earlier episodes consumed, so a
            # change to one episode's length never scrambles the others.
            ep_seed = seed + ep
            rng = np.random.default_rng(ep_seed)
            torch.manual_seed(ep_seed)
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
                for m in env.transcript.since(watermark):
                    out.write(json.dumps(_msg_record(env, ep, m), sort_keys=True) + "\n")
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
    parser.add_argument(
        "--comm-model",
        default=None,
        choices=("global", "range"),
        help="override scenario audibility; 'range' gates by distance (comms-degradation arm)",
    )
    parser.add_argument(
        "--comm-range", type=float, default=None, help="audible radius under --comm-model range"
    )
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
        comm_model=args.comm_model,
        comm_range=args.comm_range,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
