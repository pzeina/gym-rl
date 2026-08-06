"""Dashboard server: episode traces, run discovery, HTTP endpoints."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from cohort.viz.dashboard import DashboardHandler, load_behavior, record_episode, scan_runs


def test_episode_trace_structure():
    trace = record_episode("fireteam", None, seed=5, max_steps=40)
    assert trace["scenario"] == "fireteam"
    assert trace["width"] == 36 and trace["height"] == 36
    assert len(trace["grid"]) == 36
    assert [o["name"] for o in trace["objectives"]] == ["ALPHA", "BRAVO"]
    assert trace["outcome"] in ("success", "defeat", "timeout")
    assert trace["steps"], "trace must contain steps"
    assert trace["steps"][0]["t"] == 0

    step = trace["steps"][1]
    assert len(step["soldiers"]) == 4
    soldier = step["soldiers"][0]
    for key in ("cs", "x", "y", "hp", "ammo", "alive", "rank", "eff", "mission", "act", "r", "rc", "sees"):
        assert key in soldier
    assert soldier["act"] is not None, "actions taken must be recorded"
    # the OPORD must be on the net at t=0
    assert any(m["kind"] == "opord" for m in trace["steps"][0]["messages"])
    # traces must be JSON-serializable end to end
    json.dumps(trace)


def test_episode_trace_reproducible():
    a = record_episode("fireteam", None, seed=42, max_steps=30)
    b = record_episode("fireteam", None, seed=42, max_steps=30)
    assert a["length"] == b["length"]
    assert a["outcome"] == b["outcome"]
    last_a = [(s["cs"], s["x"], s["y"]) for s in a["steps"][-1]["soldiers"]]
    last_b = [(s["cs"], s["x"], s["y"]) for s in b["steps"][-1]["soldiers"]]
    assert last_a == last_b, "same seed must reproduce the same episode"


def test_scan_runs(tmp_path):
    run = tmp_path / "myrun"
    run.mkdir()
    (run / "metrics.csv").write_text("iteration,env_steps,success_rate_rolling\n1,1024,0.5\n")
    (run / "config.json").write_text('{"scenario": "fireteam"}')
    (run / "ckpt_best.pt").write_bytes(b"x")
    (tmp_path / "not_a_run").mkdir()

    runs = scan_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["name"] == "myrun"
    assert runs[0]["scenario"] == "fireteam"
    assert runs[0]["last"]["env_steps"] == "1024"
    assert runs[0]["checkpoints"] == ["best"]
    assert runs[0]["behavior"] is False


def test_behavior_json_discovery_and_load(tmp_path):
    run = tmp_path / "myrun"
    run.mkdir()
    (run / "metrics.csv").write_text("iteration,env_steps\n1,1024\n")
    (run / "behavior.json").write_text(json.dumps({"episodes": 30, "metrics": {"coverage_time": 0.9}}))

    runs = scan_runs(tmp_path)
    assert runs[0]["behavior"] is True
    payload = load_behavior(tmp_path, "myrun")
    assert payload["metrics"]["coverage_time"] == 0.9
    try:
        load_behavior(tmp_path, "no_behavior_here")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_http_endpoints(tmp_path):
    DashboardHandler.runs_dir = tmp_path
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def get(path):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
                return r.status, r.read()

        status, body = get("/api/state")
        assert status == 200
        state = json.loads(body)
        assert "fireteam" in state["scenarios"]
        assert state["runs"] == []

        status, body = get("/")
        assert status == 200
        assert b"cohort" in body and b"<canvas" in body

        try:
            get("/api/metrics?run=nope")
            raise AssertionError("expected 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        server.shutdown()
        server.server_close()
        DashboardHandler.runs_dir = None
