#!/usr/bin/env python
"""Render the eight doctrine scenarios as what they actually are: radio traffic.

    scripts/scenario_gallery.py                  # → runs/scenario_gallery.html
    scripts/scenario_gallery.py --out /tmp/g.html

The fleet board answers "how good is it" and the program board answers "what did
we learn". Neither shows the thing this project is *for*: a chain of command
whose every decision is a sentence a human can read. A success rate cannot show
that an OPORD came down, was acknowledged, produced doctrine-valid subordinate
tasks, survived its commander being killed, and ended with HQ closing the
operation on the net — and that is the whole claim.

So this is the third board, and it is the cheap one: a scenario's briefing, its
numbers, and the radio transcript of one evaluated episode, read straight out of
``runs/<member>/eval_transcript.txt`` — the file ``cohort.training.evaluate``
already writes beside every checkpoint. The GIF sits next to it on disk and is
named rather than embedded: eight animations are ~16MB, and a page nobody can
load demonstrates nothing.

Traffic is colored by act, not by sender: the OPORD that starts it, orders and
their WILCOs, contact and situation reports, casualties and successions, and the
close — the root's MISSION COMPLETE and HQ's ENDEX. Reading down one column you
can see which cohort reported its own win and which one had HQ close for it.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cohort.config import get_scenario  # noqa: E402
from scripts import baseline  # noqa: E402
from scripts.fleet_board import CSS as BASE_CSS  # noqa: E402

EXTRA_CSS = """
.scn{background:var(--panel);border:1px solid var(--rule);border-radius:2px;
  display:flex;flex-direction:column;gap:0;overflow:hidden}
.scn > header{padding:16px 18px 14px;border-bottom:1px solid var(--rule);
  display:flex;flex-direction:column;gap:7px}
.scn h3{margin:0;font-size:19px;font-weight:640;letter-spacing:-.01em}
.scn .brief{margin:0;color:var(--muted);font-size:13.5px;max-width:76ch}
.facts{display:flex;flex-wrap:wrap;gap:4px 18px;font:12px/1.5 var(--mono);color:var(--muted)}
.facts b{color:var(--ink);font-weight:600}
.net{margin:0;padding:14px 18px 16px;font:12px/1.75 var(--mono);overflow-x:auto;
  background:var(--sunk);white-space:pre;tab-size:2}
.net .t{color:var(--muted)}
.net .opord{color:var(--ink);font-weight:700}
.net .order{color:var(--accent)}
.net .rep{color:var(--compare)}
.net .cas{color:var(--fail)}
.net .close{color:var(--pass);font-weight:700}
.net .elide{color:var(--muted);font-style:italic;display:block;padding:3px 0}
.legend-net{display:flex;flex-wrap:wrap;gap:6px 16px;padding:10px 18px;
  border-top:1px solid var(--hair);font:11.5px/1.6 var(--mono);color:var(--muted)}
.legend-net i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;
  vertical-align:-1px}
.gallery{display:flex;flex-direction:column;gap:18px}
.missing{padding:16px 18px;color:var(--muted);font-size:13.5px}
"""

#: (regex over the message text, css class). First match wins, so the close
#: rules sit above the generic order rule.
ACTS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"OPORD"), "opord"),
    (re.compile(r"ENDEX"), "close"),
    (re.compile(r"COMPLETE\.|CONFIRMED|NEGATIVE, CONTINUE"), "close"),
    (re.compile(r"IS DOWN|ASSUMING COMMAND"), "cas"),
    (re.compile(r"CONTACT|SITREP|NO CHANGE|IN POSITION"), "rep"),
    (re.compile(r"WILCO|ROGER"), "order"),
)


def _classify(text: str) -> str:
    for pattern, cls in ACTS:
        if pattern.search(text):
            return cls
    return "order"


def _transcript(path: Path, *, head: int = 14, tail: int = 16) -> str:
    """The episode's net, elided in the middle so the shape stays readable.

    Head and tail rather than the first N lines: the OPORD cascade at the top
    and the close at the bottom are the two ends this page exists to show, and
    the fifty CONTACT reports between them are the part a reader skims.
    """
    try:
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    except OSError:
        return ""
    if not lines:
        return ""
    kept: list[str | None] = []
    if len(lines) <= head + tail:
        kept = list(lines)
    else:
        kept = [*lines[:head], None, *lines[-tail:]]
    out = []
    for ln in kept:
        if ln is None:
            out.append(f'<span class="elide">     … {len(lines) - head - tail} '
                       "more transmissions …</span>")
            continue
        stamp, _, rest = ln.partition("] ")
        if rest:
            out.append(f'<span class="t">{html.escape(stamp)}]</span> '
                       f'<span class="{_classify(rest)}">{html.escape(rest)}</span>')
        else:
            out.append(html.escape(ln))
    return "\n".join(out)


def _facts(scenario: str, run: str, row: dict | None) -> str:
    spec = get_scenario(scenario)
    bits = [
        f"org <b>{html.escape(str(spec.org))}</b>",
        f"root <b>{spec.root_mission.name}</b>",
        f"objective <b>{html.escape(str(spec.root_objective or '—'))}</b>",
        f"clock <b>{spec.max_steps}</b> steps",
    ]
    if spec.defend_horizon:
        bits.append(f"horizon <b>{spec.defend_horizon}</b>")
    if row and row.get("success_ci95"):
        bits.insert(0, f"success <b>{html.escape(row['success_ci95'])}</b> "
                       f"(N={row.get('episodes', '?')}, {row.get('policy', '?')})")
    bits.append(f"run <b>{html.escape(run)}</b>")
    return "".join(f"<span>{b}</span>" for b in bits)


def _scenario(scenario: str, run: str, rows: dict) -> str:
    spec = get_scenario(scenario)
    d = baseline.run_dir(run)
    net = _transcript(d / "eval_transcript.txt")
    gif = d / "eval.gif"
    body = (
        f'<pre class="net">{net}</pre>'
        if net
        else '<p class="missing">No evaluated episode on disk yet — '
             "<code>scripts/publish_baseline.py</code> writes the transcript beside "
             "the checkpoint.</p>"
    )
    footer = (
        '<div class="legend-net">'
        '<span><i style="background:var(--ink)"></i>OPORD</span>'
        '<span><i style="background:var(--accent)"></i>orders &amp; acknowledgements</span>'
        '<span><i style="background:var(--compare)"></i>contact &amp; situation reports</span>'
        '<span><i style="background:var(--fail)"></i>casualties &amp; succession</span>'
        '<span><i style="background:var(--pass)"></i>the close — COMPLETE, CONFIRMED, ENDEX</span>'
        + (f"<span>animation: <code>{html.escape(str(gif.relative_to(ROOT)))}</code></span>"
           if gif.is_file() else "")
        + "</div>"
    )
    return (
        f'<article class="scn"><header><h3>{html.escape(scenario)}</h3>'
        f'<p class="brief">{html.escape(spec.description)}</p>'
        f'<div class="facts">{_facts(scenario, run, rows.get(run))}</div></header>'
        f"{body}{footer}</article>"
    )


def render(rows: list[dict], *, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M")
    by_run = {r["run"]: r for r in rows}
    members = baseline.load().get("runs", {})
    cards = "".join(
        _scenario(scenario, members[scenario], by_run)
        for scenario in baseline.DOCTRINE_SCENARIOS
        if scenario in members
    )
    return f"""<title>cohort · the eight scenarios</title>
<style>{BASE_CSS}{EXTRA_CSS}</style>
<div class="sheet">
  <header class="masthead">
    <div class="eyebrow">cohort — chain of command, multi-agent RL</div>
    <h1>Eight scenarios, as radio traffic</h1>
    <p class="standfirst">One evaluated episode from each member of the baseline fleet,
      read off the net. A success rate cannot show that the OPORD came down, that it was
      acknowledged, that a rifleman took over a dead leader's fire team, or that HQ closed
      the operation — the transcript can.</p>
    <div class="specs">
      <span>generated <b>{stamp}</b></span>
      <span>source <b>runs/&lt;member&gt;/eval_transcript.txt</b></span>
      <span>episodes shown <b>1 of 100 evaluated</b></span>
    </div>
  </header>

  <div class="gallery">{cards}</div>

  <footer>
    <p><b>What to look for at the bottom of each net.</b> The root's
      <code>MISSION COMPLETE</code> is a REPORT and HQ's <code>ENDEX</code> is the FACT
      that the operation is over. Since v1.19 the ENDEX is transmitted on every scenario,
      so a win is never something only the scoreboard knows about — but a net that ends in
      an ENDEX with no COMPLETE before it is a cohort that won <i>without saying so</i>,
      and that difference is measured as <code>closed_on_root_report_rate</code>.</p>
    <p>The transcript is elided in the middle, never at the ends: the OPORD cascade and the
      close are what this page is for. The full file and the episode animation sit in the
      run directory. Regenerate with <code>scripts/scenario_gallery.py</code>.</p>
  </footer>
</div>
"""


def main() -> None:
    from scripts.fleet_status import collect

    p = argparse.ArgumentParser(description="Render the scenario gallery.")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default=str(ROOT / "runs" / "scenario_gallery.html"))
    args = p.parse_args()
    out = Path(args.out)
    out.write_text(render(collect(Path(args.runs_dir))))
    print(f"scenario gallery → {out}")


if __name__ == "__main__":
    main()
