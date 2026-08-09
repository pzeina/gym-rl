#!/usr/bin/env python
"""Render the program board: where the campaign stands, with the evidence.

    scripts/program_board.py                   # → runs/program_board.html
    scripts/program_board.py --out /tmp/pb.html

The fleet board answers "what have we got"; this one answers "what did we learn,
and what is it resting on". Companion, same visual system — it imports the fleet
board's tokens so the two read as one set.

**Every number on the page is read off disk at render time** — from each run's
committed ``behavior_final.json`` / ``behavior.json``, from ``economics.json``,
from ``endex_rescore.json`` where a policy had to be re-scored under a rule it
did not learn under, from git, and from the live job files. Only the *claims*
are written here, next to the runs that test them, so a retrain updates the
evidence without anyone retyping it. The narrative lists at the foot carry the
ROADMAP date they were taken from, because those are the owner's calls and this
file does not own them.

Two mechanisms exist because the page was caught overstating (refs #24): a
thread that leads with a *level* renders the same metric's spread across that
scenario's other generations beside it (``_family``), and a row whose number did
not come from a run's own behavior file must print either its own N or the word
"quoted" (``_panel`` references).
"""

from __future__ import annotations

import argparse
import html
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cohort.env.actions import N_ACTIONS
from cohort.env.observations import OBS_DIM
from scripts.fleet_board import CSS as BASE_CSS
from scripts.fleet_status import _half_width, _json, _rate, collect

ROOT = Path(__file__).resolve().parent.parent
VERSION = "v1.13"
ROADMAP_AS_OF = "2026-08-09"

EXTRA_CSS = """
.sheet{max-width:920px}
.lede{font-size:16.5px;line-height:1.6;color:var(--ink);max-width:64ch;margin:0}
.lede b{font-weight:640}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:2px;
  padding:20px 22px;display:flex;flex-direction:column;gap:14px}
.card + .card{margin-top:12px}
.card > header{display:flex;flex-direction:column;gap:5px}
.card h3{margin:0;font-size:19px;font-weight:640;letter-spacing:-.01em;text-wrap:balance;
  line-height:1.25}
.card .q{margin:0;color:var(--muted);font-size:14px;max-width:70ch}
.tagrow{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.verdict{margin:0;font-size:14.5px;line-height:1.6;max-width:70ch;
  border-left:2px solid var(--accent);padding-left:14px}
.verdict.null{border-left-color:var(--muted)}
.verdict b{font-weight:640}

.panel{display:flex;flex-direction:column;gap:9px;background:var(--sunk);
  border-radius:2px;padding:14px 16px}
.panel .cap{font:600 10.5px/1.4 var(--mono);text-transform:uppercase;letter-spacing:.11em;
  color:var(--muted)}
.rows{display:flex;flex-direction:column;gap:9px}
.brow{display:grid;grid-template-columns:minmax(96px,auto) 1fr minmax(84px,auto);
  gap:12px;align-items:center}
.brow .lab{font:12px/1.35 var(--mono);color:var(--muted);white-space:nowrap}
.brow .lab b{color:var(--ink);font-weight:600;display:block}
.brow .num{font:13px/1.3 var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap;
  display:flex;flex-direction:column;align-items:flex-end;gap:3px}
.brow .num .ep{font-size:11px;color:var(--muted)}
.plot{position:relative;height:12px;background:var(--track);border-radius:2px}
.plot .fill{position:absolute;left:0;top:0;bottom:0;border-radius:0 4px 4px 0;
  background:var(--accent)}
.plot .fill.b{background:var(--compare)}
.plot .fill.full{border-radius:2px}
.plot .ci{position:absolute;top:-3px;bottom:-3px;border-left:1.5px solid var(--ink);
  border-right:1.5px solid var(--ink);opacity:.5}
/* the bound has to stay legible where it crosses a filled bar, so it carries a
   surface ring rather than relying on contrast against whatever is beneath it */
.plot .bound{position:absolute;top:-5px;bottom:-5px;width:2px;background:var(--fail);
  border-radius:1px;box-shadow:0 0 0 1px var(--sunk);transform:translateX(-1px)}
.keys{display:flex;gap:8px 18px;flex-wrap:wrap;font:11.5px/1.4 var(--mono);color:var(--muted)}
.keys span{display:flex;align-items:center;gap:6px}
.keys i{width:10px;height:10px;border-radius:2px;display:inline-block;flex:none}
.keys i.dash{width:2px;height:12px;border-radius:1px;background:var(--fail)}
.keys i.br{width:8px;height:11px;border-radius:0;border-left:1.5px solid var(--ink);
  border-right:1.5px solid var(--ink);opacity:.5}

.threads{display:flex;flex-direction:column;gap:0}
.thread{display:grid;grid-template-columns:minmax(0,1fr) minmax(180px,auto);gap:10px 22px;
  padding:15px 0;border-bottom:1px solid var(--hair);align-items:start}
.thread:last-child{border-bottom:none}
.thread h4{margin:0 0 4px;font-size:14.5px;font-weight:640}
.thread p{margin:0;color:var(--muted);font-size:13.5px;line-height:1.55;max-width:62ch}
.probe{display:flex;flex-direction:column;gap:3px;font:12px/1.45 var(--mono);
  font-variant-numeric:tabular-nums;color:var(--muted)}
.probe b{color:var(--ink);font-weight:600}

ol.next{margin:0;padding-left:0;list-style:none;counter-reset:step;
  display:flex;flex-direction:column;gap:11px}
ol.next li{counter-increment:step;display:grid;grid-template-columns:26px 1fr;gap:12px;
  font-size:14.5px;line-height:1.55}
ol.next li::before{content:counter(step);font:600 11px/22px var(--mono);text-align:center;
  border:1px solid var(--rule);border-radius:2px;color:var(--muted);height:22px}
ol.next b{font-weight:640}
ol.next span{color:var(--muted);display:block;font-size:13.5px;margin-top:2px}
"""

# ── the claims, each pinned to the runs that test it ────────────────────────
# A panel is (caption, metric, scale, bound) over a list of (run, label, arm).
# arm "a" is the arm that was adopted, "b" the one it was measured against.
CAMPAIGNS = [
    {
        "title": "A shared terminal only one side could collect",
        "chips": ["d44ee8d", "squad_screen", "clean A/B"],
        "question": (
            "The terminal payout read <code>for s in roster.living</code>, so a soldier who "
            "died at step 50 of an episode that succeeded at step 200 collected none of it. "
            "One shared policy updates every agent at once, and a per-agent advantage only "
            "sees that hanging back cuts P(die). Does paying the fallen dissolve the collapse?"
        ),
        "panels": [
            {
                "cap": "success rate · final policy",
                "metric": None,
                "scale": 1.0,
                "runs": [
                    ("squad_screen_fallen_v1", "fallen paid · seed 3", "a"),
                    ("squad_screen_fallen_v2", "fallen paid · seed 5", "a"),
                    ("squad_screen_v9", "forfeited · seed 3", "b"),
                    ("squad_screen_v10", "forfeited · seed 5", "b"),
                ],
            }
        ],
        "verdict": (
            "<b>Solved, and it is not a subtle effect.</b> Both seeds move from total "
            "collapse to a perfect score with non-overlapping intervals and "
            "<code>done_false</code> held fixed. The 220-input observation space is "
            "exonerated by the same evidence — the arms that recover run the identical space "
            "that collapsed."
        ),
    },
    {
        "title": "…and the same fix made bodies cheap where there is no fast win",
        "chips": ["defend_brique", "clean A/B", "regression"],
        "question": (
            "Where a decisive objective exists, engaging ends the episode sooner, so survival "
            "and cover rise as instruments. A defend mission has no fast win — it is to still "
            "be there later. What does removing forfeiture buy there?"
        ),
        "panels": [
            {
                "cap": "success rate · final policy · N=100",
                "metric": None,
                "scale": 1.0,
                "runs": [
                    ("defend_brique_v3", "forfeited (before)", "b"),
                    ("defend_brique_v4", "fallen paid (after)", "a"),
                ],
            },
            {
                "cap": "distance from the objective under threat · gate bound 5.0 cells",
                "metric": "mean_distance_from_objective_under_threat",
                "scale": 8.0,
                "bound": 5.0,
                "runs": [
                    ("defend_brique_v3", "forfeited (before)", "b"),
                    ("defend_brique_v4", "fallen paid (after)", "a"),
                ],
            },
        ],
        "verdict": (
            "<b>Success says nothing; the gate says everything.</b> The two success intervals "
            "overlap completely — on that axis the change is invisible. But the defenders "
            "walked off their ground and broke an encoded regression gate the earlier policy "
            "passes. This is what forced a reward decision rather than a patch."
        ),
        "null": True,
    },
    {
        "title": "Option 4 — the defend terminal is scaled by the force that held it",
        "chips": ["f39b5a9", "defend_survivor_scale=0.35", "owner's call"],
        "question": (
            "On DEFEND/DENY roots the terminal is multiplied by "
            "<code>(1&minus;s) + s·surviving_weight/starting_weight</code>, rank-weighted, "
            "identically for every agent <i>including the fallen</i> — so a death is a shared "
            "loss, not a private one, and the free-ride asymmetry cannot re-form. "
            "s = 0.35 is fixed by the dominance invariant, not by taste."
        ),
        "panels": [
            {
                "cap": "fireteam_defend · success · final policy · N=100",
                "metric": None,
                "scale": 1.0,
                "runs": [
                    ("fireteam_defend_v11", "flat terminal", "b"),
                    ("fireteam_defend_v12", "survivor-scaled 0.35", "a"),
                ],
            },
            {
                "cap": "defend_brique · success · final policy · N=100",
                "metric": None,
                "scale": 1.0,
                "runs": [
                    ("defend_brique_v7", "flat terminal", "b"),
                    ("defend_brique_v6", "survivor-scaled 0.35", "a"),
                ],
            },
            # refs #24: "passes all four gates" is true and reads as a clean bill
            # of health. No gate bounds commander survival, so the number that
            # would qualify it goes on the page rather than in a footnote.
            {
                "cap": "the root's own death rate · final policy · no gate covers this",
                "metric": "human_death_rate",
                "scale": 0.5,
                "runs": [
                    ("fireteam_defend_v11", "fireteam_defend · flat", "b"),
                    ("fireteam_defend_v12", "fireteam_defend · scaled", "a"),
                    ("defend_brique_v7", "defend_brique · flat", "b"),
                    ("defend_brique_v6", "defend_brique · scaled", "a"),
                ],
            },
        ],
        "verdict": (
            "<b>Confirmed on both defend scenarios; keep 0.35.</b> On fireteam_defend the "
            "progress-log A/B reads p=0.034 on success and p=0.001 on root deaths; on "
            "defend_brique, p=0.027 — and that scenario had to be repaired first, because it "
            "declared prepared positions and never gave the fire team time to occupy them. "
            "Every arm above passes all four behavior gates — <b>and no gate covers commander "
            "survival</b>, which is the panel above: the option-4 arm more than halves the "
            "root's death rate on fireteam_defend and still buries a commander in about one "
            "episode in seven. Improvement, not a clean bill (refs #24). The pre-authorised "
            "fallback to option 1 is not indicated."
        ),
    },
]

# A thread's claim has to be about the run it names. Where the number it leads
# with is a LEVEL, the family band beside it says whether that level is this
# run's doing or the scenario's — both were being read as findings about one
# arm until an outside series said otherwise (refs #24).
THREADS = [
    {
        "title": "platoon_v5 answers slowly and stages orders it never releases",
        "body": (
            "It scores a perfect success rate while its command traffic degrades: obedience "
            "latency multiplies several-fold against <code>platoon_v4</code>, and staged "
            "orders are issued and then abandoned rather than released. Its <code>MISSION "
            "COMPLETE</code> silence is <i>not</i> the finding — <code>platoon_v3</code> filed "
            "none either and still succeeded, and where this family did claim, most claims "
            "were rejected on the net (band, right). Suspect is exploration, not price, and it "
            "is refutable: if exploration, the command traffic recovers at a higher "
            "<code>ent_coef</code> with <code>done_false</code> unchanged."
        ),
        "probes": [
            ("obedience latency", "platoon_v4", "obedience_latency_mean", "best",
             "platoon_v5", "obedience_latency_mean", "final"),
            ("DONE reports", "platoon_v4", "done_reports", "best", "platoon_v5", "done_reports", "final"),
        ],
        "family": {
            "cap": "claims rejected · earlier platoon generations",
            "prefix": "platoon_v",
            "metric": "false_complete_rate",
            "exclude": ("platoon_v5",),
        },
    },
    {
        "title": "fireteam_v8 reports worse the longer it trains",
        "body": (
            "The finding is the <b>within-run</b> movement: between the rolling-best "
            "checkpoint and the final policy its contact recall falls by more than half, and "
            "the share of its completion claims the net rejects rises. The <i>level</i> is not "
            "the finding — every earlier fireteam generation on disk sits in the band on the "
            "right, so a rejected-claim share near 0.9 is what this scenario has always done "
            "rather than something this run invented. Nor is it claiming at every opportunity: "
            "it takes the act at a few percent of the agent-steps where the mask offers it. "
            "(The assurance layer measured the same family level independently, over "
            "generations this repo does not hold.)"
        ),
        "probes": [
            ("false MISSION COMPLETE", "fireteam_v8", "false_complete_rate", "best",
             "fireteam_v8", "false_complete_rate", "final"),
            ("contact report recall", "fireteam_v8", "report_recall", "best",
             "fireteam_v8", "report_recall", "final"),
            ("claims / admissible step", "fireteam_v8", "done_claim_rate", "best",
             "fireteam_v8", "done_claim_rate", "final"),
        ],
        "family": {
            "cap": "false-COMPLETE rate · earlier fireteam generations",
            "prefix": "fireteam_v",
            "metric": "false_complete_rate",
            "exclude": ("fireteam_v8",),
        },
    },
]

NEXT = [
    ("Publish <code>endex_v1_13</code> at N=100.",
     "Both arms landed and close every operation on the root's own report. What their exit "
     "evaluations cannot settle is success: they are N=20 against N=100 baselines. The bar "
     "they are read against is v12's own policy re-scored under the new rule — both "
     "checkpoints, in runs/fireteam_defend_v12/endex_rescore.json, because one run has two "
     "and they are 2.5&times; apart."),
    ("Re-publish the fleet at N=100 off FINAL numbers.",
     "README and the v1.9 table are superseded twice over. scripts/publish_audit.py is the "
     "gate — 11 of 18 older published runs fail it."),
    ("Disentangle the five confounded arms.",
     "One run, squad_v9, at done_false &minus;2.0 with the fix. The CLI blocker is gone: reward "
     "weights are --reward KEY=VALUE now, so this is one flag rather than a tree edit."),
    ("Land the single-legal-action sampling fix.",
     "An agent with one legal action should take it without drawing. Held on purpose — it "
     "shifts the RNG stream, and landing it earlier would have desynchronised the A/Bs above."),
    ("Then the transparency probe, then directional vision.",
     "The probe still trails the OPORD-only baseline by 0.090 at best. Vision is designed and "
     "decided but breaking: PolicyNet is a memoryless MLP, so an explicit remembered-contact "
     "block is mandatory and its stale-track invariant is a first-class exploit hazard."),
]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _behavior(run: str, prefer: str) -> dict:
    """``prefer`` is "final" or "best"; fall back to the other rather than lie."""
    final = _json(ROOT / "runs" / run / "behavior_final.json")
    best = _json(ROOT / "runs" / run / "behavior.json")
    first, second = (final, best) if prefer == "final" else (best, final)
    return first or second


def _measure(run: str, metric: str | None, prefer: str = "final") -> dict:
    beh = _behavior(run, prefer)
    if not beh:
        return {}
    if metric is None:
        return {
            "value": _rate(beh.get("success_ci95")),
            "ci": _half_width(beh.get("success_ci95")),
            "text": beh.get("success_ci95"),
            "episodes": beh.get("episodes"),
        }
    value = beh.get("metrics", {}).get(metric)
    if value is None:
        return {}
    return {"value": value, "ci": None, "text": f"{value:.2f}", "episodes": beh.get("episodes")}


def _plot(m: dict, scale: float, arm: str, bound: float | None) -> str:
    pct = max(0.0, min(1.0, (m["value"] or 0) / scale)) * 100
    cls = "fill b" if arm == "b" else "fill"
    if pct >= 99.5:
        cls += " full"
    out = f'<div class="plot"><div class="{cls}" style="width:{pct:.1f}%"></div>'
    if m.get("ci"):
        lo = max(0.0, (m["value"] - m["ci"]) / scale) * 100
        hi = min(1.0, (m["value"] + m["ci"]) / scale) * 100
        out += f'<div class="ci" style="left:{lo:.1f}%;width:{max(hi - lo, 1.2):.1f}%"></div>'
    if bound:
        out += f'<div class="bound" style="left:{bound / scale * 100:.1f}%"></div>'
    return out + "</div>"


def _panel(panel: dict) -> str:
    rows, arms, breached, quoted = [], set(), False, False
    bound = panel.get("bound")
    entries = [{"name": run, "note": label, "arm": arm} for run, label, arm in panel["runs"]]
    # A reference is a number that did not come from a run's own behavior file:
    # v12's checkpoints re-scored under the ENDEX rule, say. It earns a place
    # beside the runs only if the row says on its face where it came from —
    # its own N when it was measured into a committed file (refs #24), and
    # "quoted" when nothing on disk backs it.
    for ref in panel.get("references", []):
        entries.append({**ref, "name": ref["label"], "arm": ref.get("arm", "b")})
    for entry in entries:
        name, label, arm = entry["name"], entry["note"], entry["arm"]
        literal = entry.get("value")
        m = (
            {"value": literal, "ci": None, "text": f"{literal:.2f}",
             "episodes": entry.get("episodes")}
            if literal is not None
            else _measure(name, panel["metric"])
        )
        if not m:
            continue
        arms.add(arm)
        quoted = quoted or (literal is not None and not m["episodes"])
        # a bound is a "max" gate: over it is a breach, and it is said in words —
        # the dashed marker alone is color doing a label's job
        flag = ""
        if bound is not None and m["value"] > bound:
            breached = True
            flag = '<span class="chip bad">✕ gate failed</span>'
        # N belongs on the row, not only in the caption: an N=20 arm sitting
        # beside an N=100 arm is the comparison this project keeps getting wrong
        episodes = f'<span class="ep">N={m["episodes"]}</span>' if m.get("episodes") else ""
        if literal is not None and not m["episodes"]:
            episodes = '<span class="ep">quoted</span>'
        rows.append(
            f'<div class="brow" title="{html.escape(name)}">'
            f'<span class="lab"><b>{html.escape(name)}</b>{html.escape(label)}</span>'
            f'{_plot(m, panel["scale"], arm, bound)}'
            f'<span class="num">{html.escape(str(m["text"]))}{episodes}{flag}</span></div>'
        )
    keys = []
    if "a" in arms:
        keys.append('<span><i style="background:var(--accent)"></i> the arm under test</span>')
    if "b" in arms:
        keys.append(
            '<span><i style="background:var(--compare)"></i> what it was measured against</span>'
        )
    if any(r for r in panel["runs"] if _measure(r[0], panel["metric"]).get("ci")):
        keys.append('<span><i class="br"></i> 95% CI</span>')
    if bound is not None:
        keys.append(
            '<span><i class="dash"></i> gate bound'
            f'{" · breached" if breached else ""}</span>'
        )
    if quoted:
        keys.append("<span>“quoted” = measured but not committed to a run dir</span>")
    return (
        f'<div class="panel"><div class="cap">{html.escape(panel["cap"])}</div>'
        f'<div class="rows">{"".join(rows)}</div>'
        f'<div class="keys">{"".join(keys)}</div></div>'
    )


ENDEX_ARMS = [
    ("fireteam_defend_v15", "fireteam_defend", "fireteam_defend_v12"),
    # both defend_brique seeds, because that is how it published: quoting the
    # better of a pair is the failure publish_audit.py exists to catch
    ("defend_brique_v9", "defend_brique · seed 12", "defend_brique_v6"),
    ("defend_brique_v10", "defend_brique · seed 13", "defend_brique_v6"),
]
#: v12's own checkpoints re-scored under the ENDEX rule, committed beside the
#: run it describes. This was published for a while as a bare **0.22** naming
#: no checkpoint (refs #24) — and one run has two, which here read 0.19 and
#: 0.47 at N=100. A single unlabelled figure lets the same retrain be read as
#: improvement or as regression on that choice alone, so both rows go on.
ENDEX_RESCORE = ROOT / "runs" / "fireteam_defend_v12" / "endex_rescore.json"


def _endex_baseline() -> list[dict]:
    """Panel reference rows for the re-scored baseline, weakest checkpoint first.

    Empty when nothing on disk backs it — the card then makes no baseline claim
    at all, which is the only honest fallback for a number with no source.
    """
    data = _json(ENDEX_RESCORE)
    rows = []
    for ckpt, m in (data.get("checkpoints") or {}).items():
        rate = m.get("closed_on_root_report_rate")
        if rate is None:
            continue
        policy = m.get("policy") or ckpt
        rows.append(
            {
                "label": f"fireteam_defend_v12 · {policy}",
                "policy": policy,
                "note": "old-rule policy, re-scored",
                "value": rate,
                "episodes": m.get("episodes"),
                "arm": "b",
            }
        )
    return sorted(rows, key=lambda r: r["value"])


def _baseline_phrase(baseline: list[dict]) -> str:
    """"0.19 at the rolling-best checkpoint and 0.47 at the final checkpoint"."""
    parts = [f"<b>{r['value']:.2f}</b> at the {r['policy']} checkpoint" for r in baseline]
    if len(parts) < 3:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Score interval for a pooled rate — the CI a ± figure cannot be added into."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def _endex_success_verdict() -> str:
    """Say what the success comparison currently supports — read, not asserted.

    This sentence claimed "not settled yet, these are the N=20 exit
    evaluations" for three hours after both arms were re-scored at N=100 and
    `defend_brique` published as a priced regression. A verdict about evidence
    has to be computed from the evidence, or it goes stale exactly like the
    board this project spent the day rebuilding.
    """
    # Seeds sharing a baseline are POOLED, because that is how the result
    # published. Judging each seed against the baseline on its own says "held"
    # for defend_brique — both seeds' intervals graze v6's — while the pooled
    # comparison is p=0.024 and non-overlapping. A board that contradicts the
    # README it summarises is worse than no board.
    groups: dict[str, list[tuple[str, str]]] = {}
    for arm, scen, base in ENDEX_ARMS:
        groups.setdefault(base, []).append((arm, scen))

    verdicts = []
    for base, arms in groups.items():
        b = _measure(base, None)
        got = [(scen, _measure(arm, None)) for arm, scen in arms]
        got = [(scen, m) for scen, m in got if m]
        if not (b and got):
            continue
        family = got[0][0].split(" · ")[0]
        thin = [f"N={m.get('episodes')}" for _, m in got if (m.get("episodes") or 0) < 100]
        if thin or (b.get("episodes") or 0) < 100:
            verdicts.append((family, "unsettled", f"{family} is still on an {', '.join(thin) or 'N<100'} evaluation"))
            continue
        wins = sum(round(m["value"] * m["episodes"]) for _, m in got)
        n = sum(m["episodes"] for _, m in got)
        lo, hi = _wilson(wins, n)
        held = hi >= b["value"] - (b["ci"] or 0)
        seeds = f" pooled over {len(got)} seeds" if len(got) > 1 else ""
        verdicts.append((
            family,
            "held" if held else "below",
            f"{family} {wins}/{n} = {wins / n:.3f} [{lo:.3f}, {hi:.3f}]{seeds} "
            f"against {b['text']}",
        ))
    if not verdicts:
        return "No arm has been evaluated yet."
    if any(v == "unsettled" for _, v, _ in verdicts):
        pending = "; ".join(t for _, v, t in verdicts if v == "unsettled")
        return (
            f"<b>The success comparison is not settled yet</b>: {pending}, set against "
            "N=100 numbers. <code>/publish</code> at N=100 is what decides it."
        )
    held = "; ".join(t for _, v, t in verdicts if v == "held")
    below = "; ".join(t for _, v, t in verdicts if v == "below")
    if held and below:
        return (
            f"<b>Success held on one scenario and was paid for on the other.</b> {held} — "
            f"intervals overlap, so a hold. But {below}: intervals do not overlap, and it "
            "publishes as a priced regression rather than a win."
        )
    if below:
        return (
            f"<b>Success was paid for.</b> {below} — intervals do not overlap; published "
            "as a priced regression."
        )
    return f"<b>Success held.</b> {held} — intervals overlap, so no difference is established."


def _endex(rows: list[dict]) -> dict:
    """The v1.13 card, written from whatever state the campaign is actually in.

    It has three: still training, landed, or not launched. A card that says
    "under test" after the answer arrived is the exact staleness these boards
    were rebuilt to stop.
    """
    by_run = {r["run"]: r for r in rows}
    training = [a for a, _, _ in ENDEX_ARMS if by_run.get(a, {}).get("state") == "RUNNING"]
    scored = [a for a, _, _ in ENDEX_ARMS if _measure(a, "closed_on_root_report_rate")]
    baseline = _endex_baseline()
    bar = _baseline_phrase(baseline)

    question = (
        '<p class="q">A DEFEND is not a task with an end state its holder may declare — it '
        "is held until relieved or re-tasked, so the order that ends it comes down the "
        "chain. <code>MISSION COMPLETE</code> is now masked shut on a continuous posture: "
        "the root reports the situation and COMMAND transmits <b>ENDEX</b>. Spaces are "
        "unchanged, so the whole fleet still loads — but every defend checkpoint on disk "
        "learned under the old rule.</p>"
    )
    chips = ['<span class="chip">16cb2a6</span>', '<span class="chip">masking only</span>']

    if not scored:
        chips.append(
            '<span class="chip on">training</span>' if training else '<span class="chip">queued</span>'
        )
        verdict = (
            "The signal to read is <code>closed_on_root_report_rate</code> — of the "
            "operations COMMAND closed, how many the root's own report closed early. "
            + (
                f"fireteam_defend_v12's old-rule policy re-scores at {bar} under the new "
                "rule (N=100 each); a policy trained on this loop should beat both, while "
                "success and root deaths hold at the v1.12 levels."
                if baseline
                else "There is no re-scored baseline on disk to beat it against yet."
            )
        )
        heading = "Under test — v1.13"
        panels = ""
    else:
        chips.append('<span class="chip ok">landed</span>')
        heading = "Landed — v1.13, reading the ENDEX retrain"
        closed = {
            "cap": "operations the root's own report closed early · final policy",
            "metric": "closed_on_root_report_rate",
            "scale": 1.0,
            "runs": [(a, f"{scen} · trained on the new loop", "a") for a, scen, _ in ENDEX_ARMS],
            "references": baseline,
        }
        success = {
            "cap": "success · final policy · the v1.12 arms these replace",
            "metric": None,
            "scale": 1.0,
            "runs": [
                r
                for arm, scen, base in ENDEX_ARMS
                for r in ((base, f"{scen} · old close rule", "b"), (arm, f"{scen} · ENDEX", "a"))
            ],
        }
        panels = _panel(closed) + _panel(success)
        verdict = (
            "<b>The close rule works, and it is not marginal.</b> Every arm closes its "
            "operations on the root's own report"
            + (f", against {bar} for the policy that learned under the old rule" if baseline else "")
            + " — with all four behavior gates passing on each. "
            + _endex_success_verdict()
        )

    card = (
        f'<article class="card"><header><div class="tagrow">{"".join(chips)}</div>'
        "<h3>COMMAND ends a defense, not the section holding the ground</h3>"
        f'{question}</header>{panels}<p class="verdict">{verdict}</p></article>'
    )
    return {
        "heading": heading,
        "card": card,
        "standfirst": (
            "Three questions this campaign settled, what each one is resting on, and the "
            + ("fourth, just landed." if scored else "one now under test.")
        ),
        "lede_tail": (
            "The question just answered is narrower and more doctrinal: "
            "<b>who is allowed to end a defense.</b>"
            if scored
            else "What is under test today is narrower and more doctrinal: "
            "<b>who is allowed to end a defense.</b>"
        ),
    }


def _campaign(c: dict) -> str:
    chips = "".join(f'<span class="chip">{html.escape(t)}</span>' for t in c["chips"])
    panels = "".join(_panel(p) for p in c["panels"])
    vcls = "verdict null" if c.get("null") else "verdict"
    return (
        f'<article class="card"><header><div class="tagrow">{chips}</div>'
        f'<h3>{c["title"]}</h3><p class="q">{c["question"]}</p></header>'
        f'{panels}<p class="{vcls}">{c["verdict"]}</p></article>'
    )


def _family(spec: dict) -> dict:
    """The spread of one metric across the sibling runs that committed it.

    refs #24. A level every generation of a scenario shows is a property of the
    family, not a finding about whichever run is being discussed — two threads
    on this page read as regressions for exactly that reason. The band is read
    off disk like everything else here, so it widens by itself as runs land,
    and a thread that leads with a level can be checked against it on sight.
    """
    exclude = set(spec.get("exclude", ()))
    values = {}
    for run in sorted((ROOT / "runs").glob(f"{spec['prefix']}*")):
        if not run.is_dir() or run.name in exclude:
            continue
        m = _measure(run.name, spec["metric"], "best")
        if m and m.get("value") is not None:
            values[run.name] = m["value"]
    if len(values) < 2:  # one sibling is an anecdote, not a family
        return {}
    return {"lo": min(values.values()), "hi": max(values.values()), "n": len(values)}


def _thread(t: dict) -> str:
    probes = []
    for label, run_a, key_a, pref_a, run_b, key_b, pref_b in t["probes"]:
        a = _measure(run_a, key_a, pref_a)
        b = _measure(run_b, key_b, pref_b)
        if not a or not b:
            continue
        probes.append(
            f"<div>{html.escape(label)}<br><b>{a['value']:.3g} → {b['value']:.3g}</b></div>"
        )
    band = _family(t["family"]) if t.get("family") else {}
    if band:
        probes.append(
            f'<div>{html.escape(t["family"]["cap"])}<br>'
            f'<b>{band["lo"]:.3g}&ndash;{band["hi"]:.3g}</b> over {band["n"]} runs</div>'
        )
    return (
        f'<div class="thread"><div><h4>{t["title"]}</h4><p>{t["body"]}</p></div>'
        f'<div class="probe">{"".join(probes)}</div></div>'
    )


def render(rows: list[dict], *, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "?"
    ahead = _git("rev-list", "--count", "origin/main..HEAD") or "?"
    tag = _git("describe", "--tags", "--abbrev=0") or "?"
    training = [r for r in rows if r["state"] == "RUNNING"]
    loadable = [r for r in rows if r["loadable"]]

    flight = []
    for t in training:
        roll = f" · rolling success {t['rolling']:.0%}" if t["rolling"] is not None else ""
        eta = f" · eta {t['eta']}" if t["eta"] else ""
        flight.append(
            '<div class="note on"><div class="note-h"><span class="dot"></span>'
            f'<span>{html.escape(t["run"])} is training</span>'
            f'<span class="chip on">{html.escape(t["scenario"] or "?")}</span></div>'
            f'<div class="runbar"><div class="meter">'
            f'<div class="fill" style="width:{t["progress"]:.1f}%"></div></div>'
            f'<span>{t.get("steps_done", 0):,} / {t.get("steps_total", 0):,} steps · '
            f'{t["progress"]:.0f}%{roll}{eta}</span></div></div>'
        )
    if not flight:
        flight.append(
            '<div class="note"><div class="note-h"><span>Nothing is training.</span></div>'
            "<p>The board is a snapshot of committed evaluations only.</p></div>"
        )
    endex = _endex(rows)

    return f"""<title>cohort · program board</title>
<style>{BASE_CSS}{EXTRA_CSS}</style>
<div class="sheet">
  <header class="masthead">
    <div class="eyebrow">cohort — chain of command, multi-agent RL</div>
    <h1>Where the program stands</h1>
    <p class="standfirst">{endex["standfirst"]}</p>
    <div class="specs">
      <span>version <b>{VERSION}</b></span>
      <span>branch <b>{html.escape(branch)}</b></span>
      <span><b>{ahead}</b> commits ahead of origin/main</span>
      <span>last tag <b>{html.escape(tag)}</b></span>
      <span>spaces <b>Discrete({N_ACTIONS}) / Box({OBS_DIM})</b></span>
      <span>generated <b>{stamp}</b></span>
    </div>
  </header>

  <p class="lede">A collapse that had haunted this repo since v1.0 turned out to be one
    shared policy free-riding on a terminal its casualties could not collect. Paying the
    fallen dissolved it — and quietly broke the defend family, where there is no fast win
    to buy. The fix for <i>that</i> is now confirmed on both defend scenarios.
    {endex["lede_tail"]}</p>

  <div class="strip">{"".join(flight)}</div>

  <section>
    <h2 class="sec">{endex["heading"]}</h2>
    {endex["card"]}
  </section>

  <section>
    <h2 class="sec">Settled, with the numbers</h2>
    {"".join(_campaign(c) for c in CAMPAIGNS)}
  </section>

  <section>
    <h2 class="sec">Open, and not closed by any of the above</h2>
    <div class="threads">
      {"".join(_thread(t) for t in THREADS)}
      <div class="thread"><div><h4>Only two of the seven v1.11 arms are single-variable</h4>
        <p>The two <code>squad_screen</code> pairs and <code>defend_brique</code> are clean.
        <code>squad</code>, <code>squad_recon</code>, <code>platoon</code>,
        <code>fireteam_defend</code> and <code>patrol_brique</code> all moved
        <code>done_false</code> from &minus;2.0 to &minus;0.5 in the same step as the fix, so the
        collapse being gone on those five is <i>consistent with</i> the fix generalising —
        not established by them. <code>run_report.py --vs</code> now prints this from
        <code>economics.json</code> rather than leaving it to memory.</p></div>
        <div class="probe">clean pairs<br><b>2 of 7</b></div></div>
    </div>
  </section>

  <section>
    <h2 class="sec">Next, in order — ROADMAP handoff, {ROADMAP_AS_OF}</h2>
    <ol class="next">
      {"".join(f"<li><div><b>{t}</b><span>{d}</span></div></li>" for t, d in NEXT)}
    </ol>
  </section>

  <footer>
    <p><b>Where these numbers come from.</b> Every figure is read at render time from each
      run's committed <code>behavior_final.json</code> (final policy) or
      <code>behavior.json</code> (rolling-best checkpoint), from
      <code>economics.json</code>, from <code>endex_rescore.json</code> where a policy was
      re-scored under a rule it did not learn under, and from the live job files — not
      retyped. A band beside a thread is that same metric across the scenario's other
      generations, so a level can be told apart from a finding. The p-values
      are quoted from the ROADMAP progress log, which is where the significance tests were
      run. {len(loadable)} runs load under the current spaces; the full fleet with its
      confidence intervals and behavior gates is on the fleet board.</p>
    <p>Regenerate with <code>scripts/program_board.py</code>. The claims and the ordered
      next steps are editorial and carry their ROADMAP date; the evidence under them does
      not — it updates itself on every retrain.</p>
  </footer>
</div>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Render the program board to static HTML.")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="runs/program_board.html")
    args = p.parse_args()
    out = Path(args.out)
    out.write_text(render(collect(Path(args.runs_dir))))
    print(f"program board → {out}")


if __name__ == "__main__":
    main()
