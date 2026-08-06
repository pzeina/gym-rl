"""Dashboard server: episode traces, run discovery, HTTP endpoints."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

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
    # v1.10: each checkpoint carries the spaces it was trained on, so the UI
    # can refuse an incompatible one up front instead of failing inside a
    # forward pass. A stub file is reported unloadable, never raised on --
    # scan_runs walks whatever happens to be in runs/.
    assert [c["kind"] for c in runs[0]["checkpoints"]] == ["best"]
    assert runs[0]["checkpoints"][0]["loadable"] is False
    assert "unreadable checkpoint" in runs[0]["checkpoints"][0]["reason"]
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


def test_checkpoint_meta_flags_incompatible_spaces(tmp_path):
    """A breaking cycle must surface as a sentence, not a matmul error.

    Before v1.10 an old checkpoint failed deep inside a forward pass with
    "mat1 and mat2 shapes cannot be multiplied", the handler did not catch
    RuntimeError, the connection died, and the dashboard showed nothing at
    all. The spaces are checked up front now.
    """
    import torch

    from cohort.env.actions import N_ACTIONS
    from cohort.env.observations import OBS_DIM
    from cohort.viz.dashboard import checkpoint_meta

    stale = tmp_path / "stale.pt"
    torch.save({"obs_dim": OBS_DIM - 54, "n_actions": N_ACTIONS, "model": {}}, stale)
    meta = checkpoint_meta(stale)
    assert meta["loadable"] is False
    assert "incompatible" in meta["reason"]
    assert str(OBS_DIM) in meta["reason"], "the reason names the build's own spaces"

    current = tmp_path / "current.pt"
    torch.save({"obs_dim": OBS_DIM, "n_actions": N_ACTIONS, "model": {}}, current)
    assert checkpoint_meta(current)["loadable"] is True


def test_checkpoint_meta_never_raises_on_junk(tmp_path):
    """It walks whatever sits in runs/ — junk is unloadable, not fatal."""
    from cohort.viz.dashboard import checkpoint_meta

    junk = tmp_path / "truncated.pt"
    junk.write_bytes(b"not a torch file")
    assert checkpoint_meta(junk)["loadable"] is False
    assert checkpoint_meta(tmp_path / "does_not_exist.pt")["loadable"] is False


def test_scenario_facets_uniquely_identify_every_scenario():
    """The picker resolves a scenario from (task, echelon), so the pair must be
    unique — otherwise two scenarios collide behind one menu selection."""
    from cohort.config import SCENARIOS
    from cohort.viz.dashboard import scenario_facets

    seen: dict[tuple[str, str], str] = {}
    for name, spec in SCENARIOS.items():
        f = scenario_facets(spec)
        key = (f["task"], f["echelon"])
        assert key not in seen, f"{name} collides with {seen[key]} on {key}"
        seen[key] = name
        assert f["echelon"] == spec.org
    # the threat qualifies the task: defending a position against a mechanised
    # assault and against an irregular band are different problems
    tasks = {n: scenario_facets(s)["task"] for n, s in SCENARIOS.items()}
    assert tasks["fireteam_defend"] != tasks["defend_brique"]
    assert tasks["squad_nomask"].startswith("Ablation")


def test_recorded_trace_serves_a_legacy_checkpoint(tmp_path):
    """A checkpoint from a previous era replays from a recorded trace instead
    of erroring — and a loadable one is never served from a trace."""
    import json

    import torch

    from cohort.env.actions import N_ACTIONS
    from cohort.env.observations import OBS_DIM
    from cohort.viz.dashboard import DashboardHandler

    run = tmp_path / "legacy_v1"
    (run / "traces").mkdir(parents=True)
    (run / "metrics.csv").write_text("iteration,env_steps\n1,1024\n")
    (run / "config.json").write_text('{"scenario": "fireteam"}')
    torch.save({"obs_dim": OBS_DIM - 54, "n_actions": N_ACTIONS, "model": {}},
               run / "ckpt_best.pt")
    (run / "traces" / "fireteam_best_seed1.json").write_text(
        json.dumps({"outcome": "success", "length": 12, "steps": []})
    )

    handler = DashboardHandler.__new__(DashboardHandler)
    handler.runs_dir = tmp_path
    trace = handler._recorded_trace("run:legacy_v1:best", "fireteam", 1)
    assert trace["replayed_from_trace"] is True
    assert trace["outcome"] == "success"

    # a seed with no recorded trace explains how to record one
    with pytest.raises(ValueError, match=r"legacy_trace\.py"):
        handler._recorded_trace("run:legacy_v1:best", "fireteam", 99)

    # a current-era checkpoint is always simulated live, never replayed
    current = tmp_path / "current_v1"
    (current / "traces").mkdir(parents=True)
    torch.save({"obs_dim": OBS_DIM, "n_actions": N_ACTIONS, "model": {}},
               current / "ckpt_best.pt")
    (current / "traces" / "fireteam_best_seed1.json").write_text('{"outcome": "x"}')
    assert handler._recorded_trace("run:current_v1:best", "fireteam", 1) is None
