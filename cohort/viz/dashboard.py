"""Interactive dashboard: training curves + episode explorer in the browser.

    python -m cohort.viz.dashboard              # serves http://localhost:8787
    python -m cohort.viz.dashboard --port 9000 --no-browser

Zero dependencies beyond the package itself: a stdlib HTTP server exposes
JSON endpoints and a single self-contained HTML page (dashboard.html).

Endpoints:
    /                → the dashboard page
    /api/state       → available runs (with config + latest metrics) and scenarios
    /api/metrics     → ?run=NAME: full metrics.csv as JSON arrays
    /api/episode     → ?scenario=S&policy=random|run:NAME:best|run:NAME:latest
                        &seed=N&greedy=0|1 : simulate one episode server-side
                        and return the full step-by-step trace for playback

The episode trace records everything the frontend needs for debugging:
per-step positions, health/ammo, missions and anchors, actions taken,
per-agent reward component breakdowns, visibility, the team's known-enemy
picture, radio messages, and the evolving chain of command.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from cohort.config import SCENARIOS
from cohort.core.language import OrderParseError
from cohort.core.missions import MissionType
from cohort.env.actions import CATALOG
from cohort.env.cohort_env import CohortEnv, make_env

MISSION_NAMES = [m.name for m in MissionType]


# ---------------------------------------------------------------------- #
# episode trace recording
# ---------------------------------------------------------------------- #


def _soldier_rec(env: CohortEnv, s, action_name: str | None, reward: float | None, components: dict | None) -> dict:
    mission = None
    if s.mission is not None:
        obj = env.world.objectives[s.mission.objective_id] if s.mission.objective_id is not None else None
        anchor = s.mission.anchor
        target = None  # supported unit's callsign (SUPPORT only)
        if s.mission.type is MissionType.RALLY:
            leader = env.roster.leader_of(s)
            if leader is not None:
                anchor = leader.pos
        elif s.mission.type is MissionType.SUPPORT:
            supported = env.roster.by_id.get(s.mission.extra.get("supported_id"))
            if supported is not None:
                target = supported.callsign
                if supported.alive:
                    anchor = supported.pos
        if s.mission.type is MissionType.ADVANCE and s.mission.extra.get("control"):
            cm = env.world.control_by_name(s.mission.extra["control"])
            if cm is not None and hasattr(cm, "nearest_point"):
                anchor = cm.nearest_point(s.pos)
        mission = {
            "type": s.mission.type.name,
            "obj": obj.name if obj else None,
            "target": target,
            "control": s.mission.extra.get("control"),
            "anchor": [float(anchor[0]), float(anchor[1])],
            "since": s.mission.step_assigned,
        }
    visible = [e.id for e in env._visible_enemies(s)] if s.alive else []
    return {
        "cs": s.callsign,
        "x": s.pos[0],
        "y": s.pos[1],
        "hp": s.health,
        "ammo": s.ammo,
        "alive": s.alive,
        "human": s.human,
        "rank": s.rank.name,
        "eff": s.effective_rank.name,
        "leader": env.roster.by_id[s.leader_id].callsign if s.leader_id is not None else None,
        "subs": [env.roster.by_id[i].callsign for i in s.subordinate_ids if env.roster.by_id[i].alive],
        "mission": mission,
        "formation": s.formation.name if s.formation is not None else None,
        "act": action_name,
        "r": None if reward is None else round(float(reward), 4),
        "rc": {k: round(v, 4) for k, v in components.items() if v} if components else {},
        "sees": visible,
    }


def _callsign_of(env: CohortEnv, agent_id: int) -> str:
    if agent_id == -1:
        return "HQ"
    soldier = env.roster.by_id.get(agent_id)
    return soldier.callsign if soldier is not None else f"#{agent_id}"


def record_episode(
    scenario: str,
    policy_path: str | None,
    seed: int,
    *,
    greedy: bool = False,
    max_steps: int | None = None,
) -> dict:
    """Simulate one episode and return the full trace for the frontend."""
    from cohort.training.evaluate import _pick_actions

    net = None
    if policy_path is not None:
        import torch

        from cohort.training.train import load_policy

        net, _ = load_policy(policy_path)
        torch.manual_seed(seed)  # same seed → same sampled episode, reproducibly

    env = make_env(scenario)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    static = {
        "scenario": scenario,
        "description": SCENARIOS[scenario].description,
        "width": env.world.width,
        "height": env.world.height,
        "grid": env.world.grid.tolist(),
        "objectives": [
            {"name": o.name, "x": o.pos[0], "y": o.pos[1], "r": o.radius} for o in env.world.objectives
        ],
        "waypoints": [
            {"name": w.name, "x": w.pos[0], "y": w.pos[1], "r": w.radius}
            for w in env.world.waypoints
        ],
        "phase_lines": [
            {"name": p.name, "x1": p.a[0], "y1": p.a[1], "x2": p.b[0], "y2": p.b[1]}
            for p in env.world.phase_lines
        ],
        "max_steps": env.spec_cfg.max_steps,
        "opord": env.transcript.messages[0].text if env.transcript.messages else "",
        "roster": [
            {"cs": s.callsign, "rank": s.rank.name, "id": s.id, "human": s.human}
            for s in env.roster.soldiers
        ],
        "policy": policy_path or "random (masked)",
        "seed": seed,
        "missions": MISSION_NAMES,
    }

    steps = [_initial_record(env)]

    limit = max_steps or env.spec_cfg.max_steps
    while env.agents and len(steps) <= limit:
        actions = _pick_actions(env, obs, net, rng, greedy=greedy)
        act_names = {a: CATALOG[i].name for a, i in actions.items()}
        obs, rewards, _terms, _truncs, infos = env.step(actions)
        steps.append(_step_record(env, act_names, rewards, infos))

    return {**static, "steps": steps, "outcome": env.outcome or "timeout", "length": len(steps) - 1}


class LiveSession:
    """One live, commandable episode: env + policy stepped on demand.

    The dashboard's Command tab drives this: advance the simulation a few
    steps at a time and inject human orders between advances — the browser
    equivalent of ``cohort.play``. A single session exists at a time; access
    is serialized by ``lock`` (the HTTP server is threaded).
    """

    def __init__(self, scenario: str, policy_path: str | None, seed: int) -> None:
        self.lock = threading.Lock()
        self.scenario = scenario
        self.seed = seed
        self.net = None
        if policy_path is not None:
            import torch

            from cohort.training.train import load_policy

            self.net, _ = load_policy(policy_path)
            torch.manual_seed(seed)
        self.env = make_env(scenario)
        self.rng = np.random.default_rng(seed)
        self.obs, _ = self.env.reset(seed=seed)
        self.static = {
            "scenario": scenario,
            "description": SCENARIOS[scenario].description,
            "width": self.env.world.width,
            "height": self.env.world.height,
            "grid": self.env.world.grid.tolist(),
            "objectives": [
                {"name": o.name, "x": o.pos[0], "y": o.pos[1], "r": o.radius}
                for o in self.env.world.objectives
            ],
            "waypoints": [
                {"name": w.name, "x": w.pos[0], "y": w.pos[1], "r": w.radius}
                for w in self.env.world.waypoints
            ],
            "phase_lines": [
                {"name": p.name, "x1": p.a[0], "y1": p.a[1], "x2": p.b[0], "y2": p.b[1]}
                for p in self.env.world.phase_lines
            ],
            "max_steps": self.env.spec_cfg.max_steps,
            "opord": self.env.transcript.messages[0].text if self.env.transcript.messages else "",
            "roster": [
                {"cs": s.callsign, "rank": s.rank.name, "id": s.id, "human": s.human}
                for s in self.env.roster.soldiers
            ],
            "policy": policy_path or "random (masked)",
            "seed": seed,
            "missions": MISSION_NAMES,
            "commanders": ["HQ"] + [
                s.callsign for s in self.env.roster.soldiers if s.effective_authority > 0
            ],
        }
        self.steps = [_initial_record(self.env)]

    def snapshot(self) -> dict:
        """Full state for a fresh page: static data + all steps so far."""
        return {
            **self.static,
            "steps": self.steps,
            "outcome": self.env.outcome,
            "length": len(self.steps) - 1,
        }

    def advance(self, n: int) -> dict:
        """Advance up to ``n`` policy steps; returns the new step records."""
        from cohort.training.evaluate import _pick_actions

        new: list[dict] = []
        for _ in range(n):
            if not self.env.agents:
                break
            actions = _pick_actions(self.env, self.obs, self.net, self.rng)
            act_names = {a: CATALOG[i].name for a, i in actions.items()}
            self.obs, rewards, _t, _tr, infos = self.env.step(actions)
            record = _step_record(self.env, act_names, rewards, infos)
            new.append(record)
            self.steps.append(record)
        return {"steps": new, "outcome": self.env.outcome, "length": len(self.steps) - 1}

    def order(self, text: str, issuer: str) -> dict:
        """Inject a human order; returns the resulting radio traffic.

        The ORDER/WILCO pair is appended to the latest step record so the
        canonical trace (and any late-joining client) includes human traffic.
        """
        before = len(self.env.transcript)
        self.env.inject_order(text, issuer=issuer)
        injected = _messages_of(self.env, self.env.transcript.since(before))
        self.steps[-1]["messages"].extend(injected)
        return {"ok": True, "messages": injected}


def _messages_of(env: CohortEnv, messages) -> list[dict]:
    return [
        {"kind": m.kind.value, "from": _callsign_of(env, m.sender_id),
         "to": _callsign_of(env, m.recipient_id) if m.recipient_id is not None else "ALL",
         "text": m.text}
        for m in messages
    ]


def _enemies_of(env: CohortEnv) -> list[dict]:
    return [
        {"id": e.id, "x": e.pos[0], "y": e.pos[1], "hp": e.health, "alive": e.alive}
        for e in env.enemies
    ]


def _traps_of(env: CohortEnv) -> list[dict]:
    """Ground-truth trap states for the trace (like enemies, the trace is
    omniscient); the frontend only draws the revealed ones."""
    return [
        {"x": t.pos[0], "y": t.pos[1], "revealed": t.revealed}
        for t in getattr(env, "traps", [])
    ]


def _initial_record(env: CohortEnv) -> dict:
    return {
        "t": 0,
        "soldiers": [_soldier_rec(env, s, None, None, None) for s in env.roster.soldiers],
        "enemies": _enemies_of(env),
        "traps": _traps_of(env),
        "messages": _messages_of(env, env.transcript.messages),
        "known": [],
    }


def _step_record(env: CohortEnv, act_names: dict, rewards: dict, infos: dict) -> dict:
    return {
        "t": env._step_count,
        "soldiers": [
            _soldier_rec(
                env,
                s,
                act_names.get(s.callsign),
                rewards.get(s.callsign),
                infos.get(s.callsign, {}).get("components"),
            )
            for s in env.roster.soldiers
        ],
        "enemies": _enemies_of(env),
        "traps": _traps_of(env),
        "messages": _messages_of(env, env.last_messages),
        "known": [[round(x, 1), round(y, 1)] for (x, y, _t) in env._known_enemies.values()],
    }


# ---------------------------------------------------------------------- #
# run discovery + metrics
# ---------------------------------------------------------------------- #


def scan_runs(runs_dir: Path) -> list[dict]:
    """Discover training runs with their config and latest metrics row."""
    runs = []
    if not runs_dir.is_dir():
        return runs
    for run in sorted(runs_dir.iterdir()):
        metrics = run / "metrics.csv"
        if not metrics.is_file():
            continue
        config = {}
        cfg_file = run / "config.json"
        if cfg_file.is_file():
            try:
                config = json.loads(cfg_file.read_text())
            except json.JSONDecodeError:
                config = {}
        last = {}
        try:
            with metrics.open() as f:
                for row in csv.DictReader(f):
                    last = row
        except OSError:
            continue
        runs.append(
            {
                "name": run.name,
                "scenario": config.get("scenario"),
                "config": config,
                "last": last,
                "checkpoints": [
                    kind for kind in ("best", "latest") if (run / f"ckpt_{kind}.pt").is_file()
                ],
                "behavior": (run / "behavior.json").is_file(),
            }
        )
    return runs


def load_behavior(runs_dir: Path, run_name: str) -> dict:
    """behavior.json of a run (the B2 behavioral metrics suite), parsed."""
    path = runs_dir / run_name / "behavior.json"
    if not path.is_file():
        msg = f"no behavior.json in run {run_name!r} — run an evaluation first"
        raise ValueError(msg)
    return json.loads(path.read_text())


def load_metrics(runs_dir: Path, run_name: str) -> dict:
    """metrics.csv → {fields: [...], rows: {field: [floats]}} for charting."""
    path = runs_dir / run_name / "metrics.csv"
    with path.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        columns: dict[str, list[float]] = {k: [] for k in fields}
        for row in reader:
            for k in fields:
                try:
                    columns[k].append(float(row[k]))
                except (TypeError, ValueError):
                    columns[k].append(0.0)
    return {"fields": fields, "columns": columns}


# ---------------------------------------------------------------------- #
# HTTP server
# ---------------------------------------------------------------------- #

_HTML_PATH = Path(__file__).with_name("dashboard.html")


class DashboardHandler(BaseHTTPRequestHandler):
    """Serves the SPA and the JSON API."""

    runs_dir: Path = Path("runs")
    live: LiveSession | None = None  # the single live commander-mode session

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: object, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _resolve_policy(self, policy: str) -> str | None:
        """'random' → None; 'run:<name>:<best|latest>' → validated ckpt path."""
        if policy in ("", "random"):
            return None
        parts = policy.split(":")
        if len(parts) != 3 or parts[0] != "run" or parts[2] not in ("best", "latest"):
            msg = f"bad policy spec {policy!r}"
            raise ValueError(msg)
        known = {r["name"] for r in scan_runs(self.runs_dir)}
        if parts[1] not in known:
            msg = f"unknown run {parts[1]!r}"
            raise ValueError(msg)
        path = self.runs_dir / parts[1] / f"ckpt_{parts[2]}.pt"
        if not path.is_file():
            msg = f"no ckpt_{parts[2]}.pt in run {parts[1]!r}"
            raise ValueError(msg)
        return str(path)

    def do_GET(self) -> None:
        """Route one request."""
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                self._send(200, _HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._json(
                    {
                        "runs": scan_runs(self.runs_dir),
                        "scenarios": {
                            name: spec.description for name, spec in SCENARIOS.items()
                        },
                    }
                )
            elif url.path == "/api/metrics":
                self._json(load_metrics(self.runs_dir, self._safe_run(query["run"])))
            elif url.path == "/api/behavior":
                self._json(load_behavior(self.runs_dir, self._safe_run(query["run"])))
            elif url.path == "/api/episode":
                trace = record_episode(
                    self._safe_scenario(query.get("scenario", "fireteam")),
                    self._resolve_policy(query.get("policy", "random")),
                    int(query.get("seed", 0)),
                    greedy=query.get("greedy", "0") == "1",
                )
                self._json(trace)
            elif url.path == "/api/live/start":
                session = LiveSession(
                    self._safe_scenario(query.get("scenario", "fireteam")),
                    self._resolve_policy(query.get("policy", "random")),
                    int(query.get("seed", 0)),
                )
                type(self).live = session
                self._json(session.snapshot())
            elif url.path.startswith("/api/live/"):
                session = type(self).live
                if session is None:
                    self._json({"error": "no live session — start one first"}, 400)
                    return
                with session.lock:
                    if url.path == "/api/live/step":
                        self._json(session.advance(min(50, int(query.get("n", 1)))))
                    elif url.path == "/api/live/order":
                        try:
                            self._json(
                                session.order(query["text"], query.get("issuer", "HQ"))
                            )
                        except (OrderParseError, PermissionError) as exc:
                            self._json({"error": str(exc)}, 400)
                    elif url.path == "/api/live/state":
                        self._json(session.snapshot())
                    else:
                        self._json({"error": "not found"}, 404)
            else:
                self._json({"error": "not found"}, 404)
        except (KeyError, ValueError, OSError) as exc:
            self._json({"error": str(exc)}, 400)

    def _safe_run(self, name: str) -> str:
        if name not in {r["name"] for r in scan_runs(self.runs_dir)}:
            msg = f"unknown run {name!r}"
            raise ValueError(msg)
        return name

    @staticmethod
    def _safe_scenario(name: str) -> str:
        if name not in SCENARIOS:
            msg = f"unknown scenario {name!r}"
            raise ValueError(msg)
        return name

    def log_message(self, fmt: str, *args: object) -> None:
        """Quiet request logging (episode generation prints are enough)."""


def serve(port: int = 8787, runs_dir: str = "runs", *, open_browser: bool = True) -> ThreadingHTTPServer:
    """Start the dashboard server (blocking)."""
    DashboardHandler.runs_dir = Path(runs_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    url = f"http://localhost:{server.server_address[1]}"
    print(f"cohort dashboard → {url}   (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye.")
    finally:
        server.server_close()
    return server


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Serve the interactive cohort dashboard.")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    serve(port=args.port, runs_dir=args.runs_dir, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
