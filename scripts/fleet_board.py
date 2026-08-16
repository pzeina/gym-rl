#!/usr/bin/env python
"""Render the fleet board: every run, what it scored, and whether it still loads.

    scripts/fleet_board.py                     # → runs/fleet_board.html
    scripts/fleet_board.py --out /tmp/fb.html

Static, self-contained HTML — no scripts, no external assets, both themes.

Two rules this board exists to keep, both learned from getting them wrong:

* **Say which policy and at what N.** A run dir holds two evaluations —
  ``behavior_final.json`` (the FINAL policy, what publication quotes) and
  ``behavior.json`` (the rolling-best checkpoint) — at whatever episode count
  they were run. The board prints the source and the N on every row instead of
  captioning the whole page "N=100".
* **Every number carries its uncertainty.** The success bar draws its 95% CI as
  a whisker, because overlapping intervals are not a difference.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cohort.env.actions import N_ACTIONS
from cohort.env.observations import OBS_DIM
from scripts import baseline
from scripts.fleet_status import collect

BASELINE_VERSION = (baseline.load().get("version") or "").strip() or "(unversioned)"

# Palette: APP-6 friendly blue is the data hue, on a plotting-sheet slate ground.
# Both themes validated for the lightness band, chroma floor, CVD separation and
# contrast against their own surface — accent/compare are the only two
# categorical hues, and pass/fail are reserved status colors that never carry data.
CSS = """
:root{
  --ground:#DDE3E5; --panel:#F0F3F3; --sunk:#E7ECEC; --ink:#101A1E; --muted:#54636A;
  --rule:#C4CDD1; --hair:#D5DCDE; --accent:#2C6E9E; --track:#C8DAE7; --compare:#A67B22;
  --pass:#2F7D52; --fail:#A6382F; --pass-bg:#DFEBE3; --fail-bg:#F2DEDB;
}
@media (prefers-color-scheme:dark){
  :root{
    --ground:#0D1417; --panel:#161E22; --sunk:#1C262B; --ink:#DCE4E7; --muted:#8B9BA2;
    --rule:#25323A; --hair:#1F2B31; --accent:#3E90C8; --track:#22394A; --compare:#B98A2E;
    --pass:#5CB07E; --fail:#E08074; --pass-bg:#172C22; --fail-bg:#2E1D1B;
  }
}
:root[data-theme="dark"]{
  --ground:#0D1417; --panel:#161E22; --sunk:#1C262B; --ink:#DCE4E7; --muted:#8B9BA2;
  --rule:#25323A; --hair:#1F2B31; --accent:#3E90C8; --track:#22394A; --compare:#B98A2E;
  --pass:#5CB07E; --fail:#E08074; --pass-bg:#172C22; --fail-bg:#2E1D1B;
}
:root[data-theme="light"]{
  --ground:#DDE3E5; --panel:#F0F3F3; --sunk:#E7ECEC; --ink:#101A1E; --muted:#54636A;
  --rule:#C4CDD1; --hair:#D5DCDE; --accent:#2C6E9E; --track:#C8DAE7; --compare:#A67B22;
  --pass:#2F7D52; --fail:#A6382F; --pass-bg:#DFEBE3; --fail-bg:#F2DEDB;
}
:root{
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font:15px/1.55 var(--sans);
  padding:40px 20px 72px;-webkit-font-smoothing:antialiased}
.sheet{max-width:1040px;margin:0 auto;display:flex;flex-direction:column;gap:28px}

.eyebrow{font:600 11px/1.4 var(--mono);text-transform:uppercase;letter-spacing:.16em;
  color:var(--muted)}
.masthead{display:flex;flex-direction:column;gap:8px;padding-bottom:16px;
  border-bottom:2px solid var(--ink)}
.masthead h1{margin:0;font-size:31px;font-weight:660;letter-spacing:-.015em;
  text-wrap:balance;line-height:1.1}
.standfirst{margin:0;color:var(--muted);max-width:62ch}
.specs{display:flex;flex-wrap:wrap;gap:6px 20px;font:12px/1.4 var(--mono);
  color:var(--muted);padding-top:4px}
.specs b{color:var(--ink);font-weight:600}

.strip{display:flex;flex-direction:column;gap:10px}
.note{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--muted);
  border-radius:2px;padding:12px 16px;display:flex;flex-direction:column;gap:3px}
.note.on{border-left-color:var(--accent)}
.note-h{font-weight:600;display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}
.note p{margin:0;color:var(--muted);font-size:13.5px;max-width:74ch}
.dot{width:8px;height:8px;border-radius:50%;background:var(--accent);flex:none;
  align-self:center;box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 22%,transparent)}
.runbar{display:flex;align-items:center;gap:12px;font:12.5px/1 var(--mono);color:var(--muted);
  flex-wrap:wrap}
.runbar .meter{flex:1 1 220px;min-width:160px}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:10px}
.tile{background:var(--panel);border:1px solid var(--rule);border-radius:2px;
  padding:14px 16px;display:flex;flex-direction:column;gap:2px}
.tile .k{font:600 10.5px/1.4 var(--mono);text-transform:uppercase;letter-spacing:.11em;
  color:var(--muted)}
.tile .v{font-size:27px;font-weight:640;line-height:1.15;font-variant-numeric:tabular-nums}
.tile .d{font:12px/1.4 var(--mono);color:var(--muted)}

h2.sec{margin:0 0 10px;font:600 12px/1.4 var(--mono);text-transform:uppercase;
  letter-spacing:.15em;color:var(--muted)}
tr.grp td{background:var(--sunk);padding:8px 14px;border-bottom:1px solid var(--rule);
  border-top:1px solid var(--rule)}
tr.grp:first-child td{border-top:none}
tr.grp b{font:600 12.5px/1.3 var(--mono);letter-spacing:.02em}
tr.grp span{font:11.5px/1.3 var(--mono);color:var(--muted);margin-left:10px}

.grid{overflow-x:auto;border:1px solid var(--rule);border-radius:2px;background:var(--panel)}
table{width:100%;border-collapse:collapse;min-width:700px}
th{font:600 10px/1.4 var(--mono);text-transform:uppercase;letter-spacing:.12em;
  color:var(--muted);text-align:left;padding:9px 14px;border-bottom:1px solid var(--rule);
  white-space:nowrap}
td{padding:10px 14px;border-bottom:1px solid var(--hair);font-size:13.5px;vertical-align:middle}
tr:last-child td{border-bottom:none}
th.n,td.n{text-align:right}
td.name{white-space:nowrap;min-width:190px}
td.name .r{font:550 13px/1.3 var(--mono)}
td.name .ov{display:block;font:11px/1.4 var(--mono);color:var(--muted);margin-top:2px}
td.val{font:13px/1 var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
td.dim{color:var(--muted)}
.mcell{width:210px;min-width:150px}

.meter{position:relative;height:10px;background:var(--track);border-radius:2px}
.meter .fill{position:absolute;left:0;top:0;bottom:0;background:var(--accent);
  border-radius:0 4px 4px 0}
.meter .fill.full{border-radius:2px}
.meter .ci{position:absolute;top:-3px;bottom:-3px;border-left:1.5px solid var(--ink);
  border-right:1.5px solid var(--ink);opacity:.5}
.meter.idle{background:var(--sunk)}

.chip{display:inline-block;font:600 10px/1.6 var(--mono);text-transform:uppercase;
  letter-spacing:.09em;padding:1px 6px;border-radius:2px;border:1px solid var(--rule);
  color:var(--muted);white-space:nowrap}
.chip.ok{color:var(--pass);border-color:currentColor;background:var(--pass-bg)}
.chip.bad{color:var(--fail);border-color:currentColor;background:var(--fail-bg)}
.chip.on{color:var(--accent);border-color:currentColor}
.chip.base{color:var(--ink);border-color:var(--ink);font-weight:700}
.chip + .chip{margin-left:5px}

details.arch{background:var(--panel);border:1px solid var(--rule);border-radius:2px}
details.arch > summary{cursor:pointer;padding:13px 16px;font:600 12px/1.4 var(--mono);
  text-transform:uppercase;letter-spacing:.12em;color:var(--muted);list-style:none;
  display:flex;justify-content:space-between;gap:12px;align-items:center}
details.arch > summary::-webkit-details-marker{display:none}
details.arch > summary::after{content:"▾";font-size:13px;transition:transform .15s}
details.arch[open] > summary::after{transform:rotate(180deg)}
details.arch > summary:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
details.arch .grid{border:none;border-top:1px solid var(--rule);border-radius:0}
.arch p.lede{margin:0;padding:0 16px 12px;color:var(--muted);font-size:13.5px;max-width:74ch}
section > p.lede{margin:-4px 0 12px;color:var(--muted);font-size:13.5px;max-width:74ch}

footer{border-top:1px solid var(--rule);padding-top:16px;display:flex;flex-direction:column;
  gap:10px;color:var(--muted);font-size:12.5px}
.legend{display:flex;gap:8px 22px;flex-wrap:wrap;align-items:center}
.legend span{display:flex;align-items:center;gap:7px}
.legend i{width:11px;height:11px;border-radius:2px;display:inline-block;flex:none}
.legend i.br{width:9px;border:none;border-left:1.5px solid var(--ink);
  border-right:1.5px solid var(--ink);opacity:.5;border-radius:0}
footer code{font:12px/1.5 var(--mono);color:var(--ink)}
footer p{margin:0;max-width:78ch}
@media (max-width:620px){
  body{padding:26px 14px 48px}
  .masthead h1{font-size:26px}
}
"""

ERA_NOTE = {
    131: "pre-PROTERRE",
    137: "PROTERRE v1.4",
    166: "comms/control v1.9",
}


def _nat(name: str) -> tuple:
    """Sort ``v9`` before ``v10`` — version order, not lexical order."""
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))


def _meter(rate: float | None, ci: float | None, *, idle: bool = False) -> str:
    if rate is None:
        return '<div class="meter idle"></div>'
    pct = max(0.0, min(1.0, rate)) * 100
    cls = "fill full" if pct >= 99.5 else "fill"
    out = f'<div class="meter{" idle" if idle else ""}"><div class="{cls}" style="width:{pct:.1f}%"></div>'
    if ci:
        lo, hi = max(0.0, rate - ci) * 100, min(1.0, rate + ci) * 100
        out += f'<div class="ci" style="left:{lo:.1f}%;width:{max(hi - lo, 1.2):.1f}%"></div>'
    return out + "</div>"


def _gate_cell(row: dict) -> str:
    gates, failed = row["gates"], row["gates_failed"]
    unmeasured = row["gates_unmeasured"]
    if not gates:
        return '<span class="chip">none</span>'
    if not failed:
        # "n/n pass" must not count a gate that had nothing to read.
        scored = len(gates) - len(unmeasured)
        if unmeasured:
            tip = html.escape(", ".join(unmeasured))
            return (
                f'<span class="chip ok">{scored}/{scored} pass</span>'
                f'<span class="chip" title="{tip}">{len(unmeasured)} unmeasured</span>'
            )
        return f'<span class="chip ok">{scored}/{scored} pass</span>'
    short = failed[0].replace("_under_threat", "").replace("mean_distance_from_objective", "distance")
    extra = f" +{len(failed) - 1}" if len(failed) > 1 else ""
    return f'<span class="chip bad" title="{html.escape(", ".join(failed))}">✕ {html.escape(short)}{extra}</span>'


def _steps(row: dict) -> str:
    return f"{row['env_steps'] / 1e6:.2f}M" if row.get("env_steps") else "—"


def _tip(row: dict) -> str:
    """Native tooltip — never clipped by the table's scroll container."""
    bits = [row["run"]]
    if row["final_ci95"]:
        bits.append(f"final policy {row['final_ci95']} (N={row['final_episodes']})")
    if row["best_ci95"]:
        bits.append(f"best ckpt {row['best_ci95']} (N={row['best_episodes']})")
    for g in row["gates"]:
        if g.get("passed") is None:
            bits.append(f"{g['name']} unmeasured ({g['direction']} {g['bound']:g})")
            continue
        mark = "pass" if g["passed"] else "FAIL"
        bits.append(f"{g['name']} {g['value']:.3g} ({g['direction']} {g['bound']:g}) {mark}")
    if row["overrides"]:
        bits.append("rewards: " + ", ".join(row["overrides"]))
    return html.escape(" · ".join(bits))


def _row(row: dict) -> str:
    name = f'<span class="r">{html.escape(row["run"])}</span>'
    chips = ""
    if row.get("baseline"):
        chips = ' <span class="chip base">baseline</span>'
    if row["state"] == "RUNNING":
        chips += ' <span class="chip on">training</span>'
    ov = (
        f'<span class="ov">{html.escape(" ".join(row["overrides"]))}</span>'
        if row["overrides"]
        else ""
    )
    if row["success"] is None:
        if row["state"] == "RUNNING":
            bar = _meter((row.get("progress") or 0) / 100, None, idle=True)
            val = f'{row["progress"]:.0f}% trained'
        else:
            bar, val = _meter(None, None), "not evaluated"
        return (
            f'<tr title="{_tip(row)}"><td class="name">{name}{chips}{ov}</td>'
            f'<td class="mcell">{bar}</td><td class="val dim">{val}</td>'
            f'<td class="val dim n">—</td><td class="val dim">—</td>'
            f'<td class="val n">{_steps(row)}</td><td>{_gate_cell(row)}</td></tr>'
        )
    policy = "final" if row["policy"] == "final" else "best ckpt"
    return (
        f'<tr title="{_tip(row)}"><td class="name">{name}{chips}{ov}</td>'
        f'<td class="mcell">{_meter(row["success"], row["success_ci"])}</td>'
        f'<td class="val">{html.escape(row["success_ci95"])}</td>'
        f'<td class="val n">{row["episodes"]}</td>'
        f'<td class="val dim">{policy}</td>'
        f'<td class="val n">{_steps(row)}</td>'
        f"<td>{_gate_cell(row)}</td></tr>"
    )


HEAD = (
    "<tr><th>run</th><th>success</th><th>rate ± 95% CI</th><th class='n'>N</th>"
    "<th>policy</th><th class='n'>steps</th><th>behavior gates</th></tr>"
)
# One grid, not eight: separate tables size their columns independently, so the
# success bars stop lining up down the page and the eye loses the comparison.
COLS = (
    '<colgroup><col style="width:23%"><col style="width:20%"><col style="width:14%">'
    '<col style="width:6%"><col style="width:11%"><col style="width:10%">'
    '<col style="width:16%"></colgroup>'
)


def _archive(rows: list[dict]) -> str:
    if not rows:
        return ""
    trs = []
    for r in sorted(rows, key=lambda r: (-(r["obs_dim"] or 0), _nat(r["run"]))):
        era = ERA_NOTE.get(r["obs_dim"], "")
        succ = r["success_ci95"] or "not evaluated"
        trs.append(
            f'<tr title="{_tip(r)}"><td class="name"><span class="r">{html.escape(r["run"])}</span></td>'
            f'<td class="val dim">{html.escape(r["scenario"] or "?")}</td>'
            f'<td><span class="chip">Box({r["obs_dim"] or "?"})</span></td>'
            f'<td class="val dim">{html.escape(era)}</td>'
            f'<td class="val{" dim" if not r["success_ci95"] else ""}">{html.escape(succ)}</td>'
            f'<td class="val n">{r["episodes"] or "—"}</td></tr>'
        )
    head = (
        "<tr><th>run</th><th>scenario</th><th>layout</th><th>era</th>"
        "<th>rate ± 95% CI</th><th class='n'>N</th></tr>"
    )
    return (
        f'<details class="arch"><summary><span>Superseded observation eras'
        f"</span><span>{len(rows)} runs</span></summary>"
        "<p class=\"lede\">These checkpoints do not load under the current spaces. Their "
        "numbers were measured on an older observation layout and are kept for provenance "
        "only — they are not comparable with the fleet above.</p>"
        f'<div class="grid"><table><thead>{head}</thead>'
        f"<tbody>{''.join(trs)}</tbody></table></div></details>"
    )


def reporting_channel(manifest: dict) -> str:
    """The seed search behind each member, or the absence of one, as HTML.

    The one claim on this board that a reader cannot check from the success
    column. ``closed_on_root_report_rate`` is a per-run bar over a quantity
    bimodal in the seed — across 14 matched ``patrol_brique`` runs the commander
    reports in 6, at 0.750-1.000, and is silent in the other 8 at exactly 0.000 —
    so where a scenario behaves that way the member is chosen from several
    seeds. Publishing that member without saying how many were tried is the
    overstatement the manifest's ``seed_search`` block exists to prevent, and a
    declaration nobody renders is not a disclosure.

    Every number is read from the members' committed evaluations through
    ``baseline``; nothing here is kept by hand.
    """
    members = manifest.get("runs", {})
    if not members:
        return ""
    lines = []
    searched = False
    for scenario in baseline.DOCTRINE_SCENARIOS:
        member = members.get(scenario)
        if not member:
            continue
        facts = baseline.seed_search_facts(manifest, scenario, member)
        if facts is None:
            passes = baseline._reporting_gate(member)
            verdict = ("not measured" if passes is None
                       else "reports" if passes else "MUTE")
            cell = f'one seed · <span class="dim">{verdict}</span>'
        else:
            searched = True
            seeds = ", ".join(str(r["seed"]) for r in facts["runs"])
            cell = (f'<b>{facts["reporting"]} of {facts["total"]} seeds report</b> '
                    f'<span class="dim">· seeds {html.escape(seeds)}</span>')
        lines.append(f"<li>{html.escape(scenario)} — {cell}</li>")
    note = (
        "Where a scenario's commander is bimodal in the seed the member is chosen "
        "from a declared search, and the count says how many were tried."
        if searched else
        "No member here was chosen from a seed search — each scenario ran one seed."
    )
    return (
        '<div class="note"><div class="note-h">'
        "<span>Does the commander close its own operations?</span></div>"
        f"<p>Measured on the FINAL policy at a floor of "
        f"{baseline.ROOT_REPORT_CLOSE_FLOOR:g}, the artifact this board publishes. "
        f"{note}</p><ul>{''.join(lines)}</ul></div>"
    )


def render(rows: list[dict], *, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M")
    live = [r for r in rows if r["loadable"]]
    stale = [r for r in rows if not r["loadable"]]
    evaluated = [r for r in rows if r["success"] is not None]
    publishable = [r for r in rows if r["policy"] == "final" and r["episodes"] == 100]
    failing = [r for r in live if r["gates_failed"]]
    training = [r for r in rows if r["state"] == "RUNNING"]

    strip = []
    for t in training:
        done, total = t.get("steps_done") or 0, t.get("steps_total") or 0
        roll = f"rolling success {t['rolling']:.0%}" if t["rolling"] is not None else "no rolling yet"
        eta = f" · eta {t['eta']}" if t["eta"] else ""
        strip.append(
            '<div class="note on"><div class="note-h"><span class="dot"></span>'
            f'<span>{html.escape(t["run"])} is training</span>'
            f'<span class="chip on">{html.escape(t["scenario"] or "?")}</span></div>'
            f'<div class="runbar"><div class="meter">'
            f'<div class="fill" style="width:{t["progress"]:.1f}%"></div></div>'
            f'<span>{done:,} / {total:,} steps · {t["progress"]:.0f}% · {roll}{eta}</span></div></div>'
        )
    strip.append(
        '<div class="note"><div class="note-h">'
        f"<span>{len(live)} of {len(rows)} runs load under this build</span></div>"
        f"<p>The other {len(stale)} predate the current observation layout — their "
        "checkpoints cannot be loaded, evaluated, or compared against anything below. "
        "They are folded into the archive at the foot of the page.</p></div>"
    )

    tiles = [
        (f"{len(live)}", "Loadable now", f"of {len(rows)} runs on disk"),
        (f"{len(evaluated)}", "Evaluated", "with a committed behavior file"),
        (f"{len(publishable)}", "Final policy at N=100", "publication-grade evidence"),
        (f"{len(failing)}", "Failing a gate", "among loadable runs"),
    ]
    tilehtml = "".join(
        f'<div class="tile"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{v}</div><div class="d">{html.escape(d)}</div></div>'
        for v, k, d in tiles
    )

    # The baseline is the answer to "what does this project ship"; the other 90
    # runs are how it got there. Leading with the fleet-as-directory-listing was
    # the board's own version of the mistake the manifest exists to fix.
    members = [r for r in rows if r.get("baseline")]
    ordered = sorted(members, key=lambda r: baseline.DOCTRINE_SCENARIOS.index(r["baseline"])
                     if r["baseline"] in baseline.DOCTRINE_SCENARIOS else 99)
    base_rows = "".join(_row(r) for r in ordered)
    covered = {r["baseline"] for r in members}
    absent = [s for s in baseline.DOCTRINE_SCENARIOS if s not in covered]
    for scenario in absent:
        base_rows += (
            f'<tr><td class="name"><span class="r dim">{html.escape(scenario)}</span></td>'
            '<td class="mcell">' + _meter(None, None) + '</td>'
            '<td class="val dim" colspan="5">no member on disk yet</td></tr>'
        )
    baseline_section = (
        f'<section><h2 class="sec">Baseline {html.escape(BASELINE_VERSION)} — '
        f"the {len(baseline.DOCTRINE_SCENARIOS)} doctrine scenarios</h2>"
        '<p class="lede">One run per scenario, all trained from the same commit on the '
        "shipped reward defaults, all scored on the FINAL policy. This is the fleet the "
        "README describes; everything below is the record of getting here.</p>"
        f'<div class="grid"><table>{COLS}<thead>{HEAD}</thead>'
        f"<tbody>{base_rows}</tbody></table></div>"
        f"{reporting_channel(baseline.load())}</section>"
        if base_rows
        else ""
    )

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in live:
        if r.get("baseline"):
            continue  # already shown above; a run appearing twice reads as two runs
        groups[r["scenario"] or "unknown"].append(r)
    body = []
    for scenario in sorted(groups):
        rs = sorted(groups[scenario], key=lambda r: _nat(r["run"]))
        n_eval = sum(1 for r in rs if r["success"] is not None)
        body.append(
            f'<tr class="grp"><td colspan="7"><b>{html.escape(scenario)}</b>'
            f'<span>{len(rs)} run{"s" if len(rs) != 1 else ""} · '
            f"{n_eval} evaluated</span></td></tr>"
        )
        body.extend(_row(r) for r in rs)
    fleet = (
        f'<div class="grid"><table>{COLS}<thead>{HEAD}</thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
        if body
        else ""
    )

    return f"""<title>cohort · fleet board</title>
<style>{CSS}</style>
<div class="sheet">
  <header class="masthead">
    <div class="eyebrow">cohort — chain of command, multi-agent RL</div>
    <h1>Fleet board</h1>
    <p class="standfirst">Every trained run on disk: what it scored, how sure we are of
      that, and whether this build can still load it.</p>
    <div class="specs">
      <span>spaces <b>Discrete({N_ACTIONS}) / Box({OBS_DIM})</b></span>
      <span>generated <b>{stamp}</b></span>
      <span>source <b>runs/*/behavior_final.json · behavior.json</b></span>
    </div>
  </header>

  <div class="strip">{"".join(strip)}</div>

  <div class="tiles">{tilehtml}</div>

  {baseline_section}

  <section>
    <h2 class="sec">The record — {len(live) - len(members)} further runs on this build</h2>
    {fleet}
  </section>

  {_archive(stale)}

  <footer>
    <div class="legend">
      <span><i style="background:var(--accent)"></i> success rate</span>
      <span><i class="br"></i> 95% confidence interval</span>
      <span><i style="background:var(--pass)"></i> behavior gates pass</span>
      <span><i style="background:var(--fail)"></i> a gate failed</span>
      <span><i style="background:var(--track)"></i> not evaluated</span>
    </div>
    <p><b>Read the N.</b> A run holds two evaluations: the <b>final</b> policy —
      what publication quotes — and the rolling-<b>best</b> checkpoint. Each row says
      which one it shows and over how many episodes. Bars are the point estimate and
      the bracket is the 95% CI: <b>overlapping intervals are not a difference</b>.</p>
    <p>Behavior gates are the encoded regression hazards for that scenario (cover under
      threat, distance from the objective under threat, timeout rate). Hover any row for
      its gate values, both evaluations, and any reward overrides it trained under.</p>
    <p>Regenerate with <code>scripts/fleet_board.py</code>. No episodes are simulated
      and no <code>metrics.csv</code> is read — the board is a view over what each run
      already committed.</p>
  </footer>
</div>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Render the fleet board to static HTML.")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="runs/fleet_board.html")
    args = p.parse_args()
    out = Path(args.out)
    out.write_text(render(collect(Path(args.runs_dir))))
    print(f"fleet board → {out}")


if __name__ == "__main__":
    main()
