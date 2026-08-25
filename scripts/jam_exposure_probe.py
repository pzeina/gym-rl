#!/usr/bin/env python
"""Why does the human stop dying under jamming? — a denominator check.

The jamming arm's most surprising number replicated at both seeds: **human
death FALLS**, 0.450 -> 0.050 (seed 12) and 0.400 -> 0.000 (seed 13), against
matched clear-net controls on the same commit. A degraded net that makes the
cohort *safer* would be a remarkable result, and it is exactly the shape this
project's own reading rule distrusts: *a metric that improves because its
denominator vanished has not improved.*

**The claim under test.** Under jamming the human is **not brought forward**.
Deaths fall because its exposure disappears, not because anything protects it.

**What would refute it, stated before the numbers were read.**

1. ``human_ring_entries`` and ``human_mean_enemy_dist`` — the exposure
   denominator. If the human closes with the enemy as often and as near as in
   the control, exposure did NOT fall and the death drop is a genuine safety
   gain. **Unchanged exposure refutes the claim.**
2. ``enemies_seen`` — is this human-specific, or did the whole team disengage?
   If team contact collapses too, the story is not "the human is left behind"
   but "nobody fights", which would also explain the deaths and is a different
   finding. **A collapse in team contact refutes the human-specific reading.**
3. ``human_mean_objective_dist`` — the mechanism itself. If the human advances
   as far as in the control, it is not being left behind, whatever the death
   rate does. **Unchanged advance refutes the mechanism.**

**Zero rollouts.** Everything here is arithmetic over the ``per_episode`` blocks
already committed in each run's ``behavior_final.json``, so it re-derives from
published evidence rather than generating new evidence to explain old evidence.
It also means the probe cannot drift from what the boards show.

**Counters that are blank under this comm model** — ``orders_delivered``,
``orders_lost``, ``opfor_investigation_steps`` read 0.00 in both arms because
they are recorded by the store-and-forward and OPFOR-investigation paths, which
``global`` and ``jammed`` do not exercise. They are printed as ``blank`` rather
than as zero: reporting "orders lost: 0.00" for a jammed net would be a claim,
and a false one.

    scripts/jam_exposure_probe.py                    # both seeds
    scripts/jam_exposure_probe.py --seeds 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_report import run_dir

#: (field, label, "up" | "down" = the direction the claim predicts under jamming)
EXPOSURE = (
    ("human_ring_entries", "human entries into enemy ring", "down"),
    ("human_mean_enemy_dist", "human mean distance to enemy", "up"),
    ("threat_pairs", "agent-steps under threat", "down"),
    ("human_mean_objective_dist", "human mean distance to OBJ", "up"),
)
CONTEXT = (
    ("enemies_seen", "enemies seen (team)"),
    ("contacts", "contact reports"),
    ("orders_issued", "orders issued"),
    ("length", "episode length"),
)
#: Recorded only by paths `global`/`jammed` never take — printed as blank.
BLANK_UNDER_THIS_COMM_MODEL = ("orders_delivered", "orders_lost",
                               "opfor_investigation_steps")


def per_episode(run: str) -> list[dict]:
    """The committed FINAL-policy evaluation of one run.

    Resolved through ``run_report.run_dir`` rather than by joining onto
    ``runs/``: archiving is a move, so a run cited here may live under
    ``runs/archive/`` and a direct join would report it missing. Enforced by
    ``tests/test_run_archive.py``, which caught this probe doing exactly that.
    """
    path = run_dir(run) / "behavior_final.json"
    if not path.exists():
        raise SystemExit(f"no committed evaluation for {run}")
    return json.loads(path.read_text())["per_episode"]


def death_rate(eps: list[dict]) -> float:
    scored = [e for e in eps if e.get("human_died") is not None]
    return sum(bool(e["human_died"]) for e in scored) / len(scored) if scored else float("nan")


def avg(eps: list[dict], field: str) -> float | None:
    vals = [e[field] for e in eps if e.get(field) is not None]
    return mean(vals) if vals else None


def compare(seed: int) -> dict:
    clear = per_episode(f"squad_ctrl_v2_seed{seed}")
    jam = per_episode(f"squad_jammed_control_v1_seed{seed}")
    print(f"\n== seed {seed}  (N={len(clear)} clear, {len(jam)} jammed, FINAL policy) ==")
    dc, dj = death_rate(clear), death_rate(jam)
    print(f"  {'human death rate':32} {dc:9.3f} -> {dj:9.3f}")

    print("  -- exposure (the denominator) --")
    moved: dict[str, tuple] = {}
    for field, label, direction in EXPOSURE:
        a, b = avg(clear, field), avg(jam, field)
        if a is None or b is None:
            continue
        rel = None if a == 0 else (b - a) / a * 100
        arrow = "" if rel is None else f"  ({rel:+.0f}%)"
        print(f"  {label:32} {a:9.2f} -> {b:9.2f}{arrow}")
        as_predicted = (b < a) if direction == "down" else (b > a)
        moved[field] = (a, b, as_predicted)

    print("  -- context (is the team still fighting?) --")
    for field, label in CONTEXT:
        a, b = avg(clear, field), avg(jam, field)
        if a is None or b is None:
            continue
        rel = None if a == 0 else (b - a) / a * 100
        arrow = "" if rel is None else f"  ({rel:+.0f}%)"
        print(f"  {label:32} {a:9.2f} -> {b:9.2f}{arrow}")
    for field in BLANK_UNDER_THIS_COMM_MODEL:
        print(f"  {field:32} {'blank':>9}    {'blank':>9}   (not recorded for this comm model)")
    return {"death": (dc, dj), "moved": moved,
            "enemies_seen": (avg(clear, "enemies_seen"), avg(jam, "enemies_seen"))}


def verdicts(results: list[dict]) -> None:
    print("\n-- pre-registered checks, both seeds --")

    ring = all(r["moved"].get("human_ring_entries", (0, 0, False))[2] for r in results)
    dist = all(r["moved"].get("human_mean_enemy_dist", (0, 0, False))[2] for r in results)
    if ring and dist:
        print("  1. exposure fell: SUPPORTED — the human closes with the enemy "
              "less often and stays further away at every seed")
    else:
        print("  1. exposure fell: REFUTED — exposure held up, so the lower death "
              "rate is a real safety gain and not a vanished denominator")

    seen_flat = all(
        r["enemies_seen"][0] and abs(r["enemies_seen"][1] - r["enemies_seen"][0])
        / r["enemies_seen"][0] < 0.25
        for r in results
    )
    if seen_flat:
        print("  2. team disengagement: REFUTED — enemies seen is flat, so the "
              "team still finds and fights; this is specific to the human")
    else:
        print("  2. team disengagement: SUPPORTED — team contact moved too, so "
              "the effect is not human-specific and the reading above is wrong")

    adv = all(r["moved"].get("human_mean_objective_dist", (0, 0, False))[2] for r in results)
    if adv:
        print("  3. the human is left behind: SUPPORTED — it ends up further "
              "from the objective at every seed")
    else:
        print("  3. the human is left behind: REFUTED — it advances as far as in "
              "the control, so exclusion is not the mechanism")

    print("\n  READING: where 1 and 3 hold and 2 is refuted, `human_death_rate` "
          "did NOT improve —\n  its denominator did. The metric must not be quoted "
          "as a safety result for this arm.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[12, 13])
    args = ap.parse_args()
    verdicts([compare(s) for s in args.seeds])


if __name__ == "__main__":
    main()
