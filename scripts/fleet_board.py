#!/usr/bin/env python
"""Render the fleet board: every evaluated run, its CI, and whether it loads.

    scripts/fleet_board.py                     # → runs/fleet_board.html
    scripts/fleet_board.py --out /tmp/fb.html

Static HTML, self-contained. The success bar carries its 95% confidence
interval as a whisker rather than only printing it: this repo's standard is
that every published number carries its uncertainty, so the board shows it.
"""

from __future__ import annotations

import argparse
import html
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cohort.env.actions import N_ACTIONS
from cohort.env.observations import OBS_DIM
from scripts.fleet_status import collect

CSS = """
:root{
  --paper:#E9EAE4; --card:#F4F4F0; --ink:#1A1D1B; --dim:#63685E; --rule:#D2D4CB;
  --accent:#2F5D8C; --live:#3F7D4E; --stale:#B8863B; --track:#DCDED5;
}
@media (prefers-color-scheme: dark){
  :root{
    --paper:#141715; --card:#1C201D; --ink:#E4E6DF; --dim:#8D9486; --rule:#2C312D;
    --accent:#6E9FD4; --live:#5FA972; --stale:#D2A054; --track:#272C28;
  }
}
:root[data-theme="dark"]{
  --paper:#141715; --card:#1C201D; --ink:#E4E6DF; --dim:#8D9486; --rule:#2C312D;
  --accent:#6E9FD4; --live:#5FA972; --stale:#D2A054; --track:#272C28;
}
:root[data-theme="light"]{
  --paper:#E9EAE4; --card:#F4F4F0; --ink:#1A1D1B; --dim:#63685E; --rule:#D2D4CB;
  --accent:#2F5D8C; --live:#3F7D4E; --stale:#B8863B; --track:#DCDED5;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  padding:32px 20px 64px}
.wrap{max-width:900px;margin:0 auto;display:flex;flex-direction:column;gap:22px}
header{display:flex;flex-direction:column;gap:6px;border-bottom:2px solid var(--ink);
  padding-bottom:14px}
h1{font-size:26px;font-weight:650;letter-spacing:.01em;margin:0;text-wrap:balance}
.sub{color:var(--dim);font-size:14px}
.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.14em;color:var(--dim);
  font-weight:600}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
.banner{border-left:4px solid var(--stale);background:var(--card);border-radius:0 8px 8px 0;
  padding:12px 16px;display:flex;flex-direction:column;gap:4px}
.banner.ok{border-left-color:var(--live)}
.banner b{font-size:15px}
.banner p{margin:0;color:var(--dim);font-size:13.5px}
.stats{display:flex;gap:10px;flex-wrap:wrap}
.stat{flex:1 1 150px;background:var(--card);border:1px solid var(--rule);border-radius:8px;
  padding:11px 13px;display:flex;flex-direction:column;gap:3px}
.stat .v{font-size:23px;font-weight:650;line-height:1.1}
.stat .k{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--dim)}
.group{display:flex;flex-direction:column;gap:0}
.group h2{font-size:12px;text-transform:uppercase;letter-spacing:.13em;color:var(--dim);
  margin:18px 0 7px;font-weight:650}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;background:var(--card)}
table{width:100%;border-collapse:collapse;min-width:620px}
td,th{padding:8px 12px;text-align:left;border-bottom:1px solid var(--rule);font-size:13.5px}
th{font-size:10.5px;text-transform:uppercase;letter-spacing:.11em;color:var(--dim);
  font-weight:600;background:transparent}
tr:last-child td{border-bottom:none}
td.run{font-weight:550;white-space:nowrap}
td.run .stripe{display:inline-block;width:3px;height:13px;border-radius:2px;
  margin-right:8px;vertical-align:-2px;background:var(--stale)}
tr.live td.run .stripe{background:var(--live)}
.chip{display:inline-block;font-size:10.5px;padding:1px 7px;border-radius:20px;
  border:1px solid var(--rule);color:var(--dim);letter-spacing:.05em}
.chip.cur{border-color:var(--live);color:var(--live)}
.barcell{width:190px;min-width:150px}
.bar{position:relative;height:9px;border-radius:5px;background:var(--track);overflow:visible}
.bar .fill{position:absolute;left:0;top:0;bottom:0;border-radius:5px;background:var(--accent);
  opacity:.75}
.bar .ci{position:absolute;top:-3px;bottom:-3px;border-left:1.5px solid var(--ink);
  border-right:1.5px solid var(--ink);opacity:.55}
.num{text-align:right;white-space:nowrap}
footer{color:var(--dim);font-size:12.5px;border-top:1px solid var(--rule);padding-top:14px;
  display:flex;flex-direction:column;gap:5px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--dim)}
.legend span{display:flex;align-items:center;gap:6px}
.legend i{width:10px;height:10px;border-radius:2px;display:inline-block}
"""


def _bar(rate: float | None, ci: float | None) -> str:
    if rate is None:
        return '<span style="color:var(--dim)">—</span>'
    pct = max(0.0, min(1.0, rate)) * 100
    out = f'<div class="bar"><div class="fill" style="width:{pct:.1f}%"></div>'
    if ci:
        lo = max(0.0, rate - ci) * 100
        hi = min(1.0, rate + ci) * 100
        out += f'<div class="ci" style="left:{lo:.1f}%;width:{max(hi - lo, 1.2):.1f}%"></div>'
    return out + "</div>"


def _parse_ci(text: str | None) -> float | None:
    if not text or "±" not in text:
        return None
    try:
        return float(text.split("±")[1].strip())
    except ValueError:
        return None


def render(rows: list[dict]) -> str:
    rows = [r for r in rows if r["success"] is not None]
    live = [r for r in rows if r["loadable"]]
    eras = sorted({r["obs_dim"] for r in rows if r["obs_dim"]})
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["scenario"] or "unknown"].append(r)

    banner = (
        '<div class="banner"><b>Open breaking cycle — the whole fleet is stale.</b>'
        f"<p>No checkpoint on disk loads under the current spaces "
        f"(Discrete({N_ACTIONS})/Box({OBS_DIM})). Every result below was measured on an "
        "older observation layout and is kept as the standing baseline until the fleet "
        "is retrained.</p></div>"
        if not live
        else f'<div class="banner ok"><b>{len(live)} of {len(rows)} runs load under the '
        f"current build.</b><p>Spaces: Discrete({N_ACTIONS})/Box({OBS_DIM}).</p></div>"
    )

    body = []
    for scenario in sorted(groups):
        rs = sorted(groups[scenario], key=lambda r: r["run"])
        trs = []
        for r in rs:
            ci = _parse_ci(r["success_ci95"])
            cur = r["obs_dim"] == OBS_DIM
            steps = f"{r['env_steps'] / 1e6:.2f}M" if r["env_steps"] else "—"
            trs.append(
                f'<tr class="{"live" if r["loadable"] else ""}">'
                f'<td class="run"><span class="stripe"></span>{html.escape(r["run"])}</td>'
                f'<td><span class="chip {"cur" if cur else ""}">Box({r["obs_dim"] or "?"})</span></td>'
                f'<td class="barcell">{_bar(r["success"], ci)}</td>'
                f'<td class="num mono">{html.escape(r["success_ci95"] or "—")}</td>'
                f'<td class="num mono">{steps}</td></tr>'
            )
        body.append(
            f'<div class="group"><h2>{html.escape(scenario)}</h2><div class="tablewrap">'
            "<table><thead><tr><th>run</th><th>layout</th><th>success (N=100)</th>"
            "<th class='num'>rate ± 95% CI</th><th class='num'>steps</th></tr></thead>"
            f"<tbody>{''.join(trs)}</tbody></table></div></div>"
        )

    worst = min(rows, key=lambda r: r["success"]) if rows else None
    stats = (
        f'<div class="stats">'
        f'<div class="stat"><div class="v mono">{len(rows)}</div><div class="k">evaluated runs</div></div>'
        f'<div class="stat"><div class="v mono" style="color:var(--live)">{len(live)}</div>'
        f'<div class="k">loadable now</div></div>'
        f'<div class="stat"><div class="v mono">{len(eras)}</div><div class="k">layout eras</div></div>'
        f'<div class="stat"><div class="v mono">{worst["success"]:.2f}</div>'
        f'<div class="k">weakest · {html.escape(worst["scenario"] or "?")}</div></div>'
        "</div>"
        if rows
        else ""
    )

    return f"""<title>cohort · fleet board</title>
<style>{CSS}</style>
<div class="wrap">
  <header>
    <div class="eyebrow">cohort · chain-of-command multi-agent RL</div>
    <h1>Fleet board</h1>
    <div class="sub">Every evaluated run, its confidence interval, and whether this
      build can still load it.</div>
  </header>
  {banner}
  {stats}
  {"".join(body)}
  <footer>
    <div class="legend">
      <span><i style="background:var(--live)"></i> loads under the current spaces</span>
      <span><i style="background:var(--stale)"></i> stale — needs a retrain</span>
      <span><i style="background:var(--accent);opacity:.75"></i> success rate</span>
      <span>│ 95% confidence interval</span>
    </div>
    <div>Success is sampled-policy evaluation over N=100 episodes. Bars show the point
      estimate; the brackets show the 95% CI — overlapping intervals are not a
      difference. Generated by <span class="mono">scripts/fleet_board.py</span>
      from each run's committed <span class="mono">behavior.json</span>.</div>
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
