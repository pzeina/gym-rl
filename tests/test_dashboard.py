"""Dashboard server: episode traces, run discovery, HTTP endpoints."""

import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from cohort.viz.dashboard import (
    DashboardHandler,
    load_behavior,
    load_metrics,
    record_episode,
    scan_runs,
)


def test_episode_trace_structure():
    trace = record_episode("fireteam", None, seed=5, max_steps=40)
    assert trace["scenario"] == "fireteam"
    assert trace["width"] == 36 and trace["height"] == 36
    assert len(trace["grid"]) == 36
    assert [o["name"] for o in trace["objectives"]] == ["ALPHA", "BRAVO"]
    assert trace["outcome"] in ("success", "defeat", "timeout")
    assert trace["steps"], "trace must contain steps"
    assert trace["steps"][0]["t"] == 0

    # the env's real sensor parameters ride in the static payload, so the
    # frontend never hard-codes a range the env has moved away from
    assert trace["combat"]["vision"] > 0 and trace["combat"]["weapon"] > 0
    assert trace["comms"]["model"] == "global"

    step = trace["steps"][1]
    assert len(step["soldiers"]) == 4
    soldier = step["soldiers"][0]
    for key in ("cs", "x", "y", "hp", "ammo", "alive", "rank", "eff", "mission", "act", "r", "rc", "sees", "sensors"):
        assert key in soldier
    assert soldier["act"] is not None, "actions taken must be recorded"
    assert soldier["sensors"] is not None and "cover" in soldier["sensors"]
    # leader-side heard-window slots (read-back / DONE / interrogative STATUS)
    # are keyed per direct-subordinate callsign — the commander's-picture
    # panel and its claim-evidence stamp read exactly these
    leader = next(s for s in step["soldiers"] if s["subs"])
    for key in ("rb_heard", "done_heard", "status_heard"):
        heard = leader["sensors"][key]
        assert heard and set(heard) <= set(leader["subs"])
        assert all(isinstance(v, bool) for v in heard.values())
    # the OPORD must be on the net at t=0
    assert any(m["kind"] == "opord" for m in trace["steps"][0]["messages"])
    # traces must be JSON-serializable end to end
    json.dumps(trace)


def test_trace_sensors_follow_the_comm_model():
    """The sensor record mirrors what the observation builder is given.

    Acoustic cues exist only under sound_model="tactical"; garble / say-again
    state only under comm_model="range"; local friendly perception only under
    voice_only; per-agent enemy pictures only when pictures are local. A
    global-net trace stays lean — absent sensors are absent, not zero-filled.
    """
    tr = record_episode("squad_range_control", None, seed=3, max_steps=10)
    sen = tr["steps"][2]["soldiers"][0]["sensors"]
    assert "cues" in sen and "garble" in sen and "say_again" in sen and "known" in sen
    for c in sen["cues"]:
        assert c["kind"] and 0 <= c["brg"] <= 7 and c["band"] in (0, 1, 2)
    assert tr["comms"] == {
        "model": "range", "range": 12.0, "voice_range": 6.0,
        "sound": "tactical", "liaison": False,
    }

    tr = record_episode("squad_voice_direct", None, seed=3, max_steps=10)
    sen = tr["steps"][2]["soldiers"][0]["sensors"]
    assert "mates" in sen and "garble" not in sen
    for rec in sen["mates"].values():
        assert set(rec) == {"seen", "x", "y", "age"}

    tr = record_episode("squad_jammed_control", None, seed=3, max_steps=10)
    assert "jammed" in tr["steps"][2], "jam state is umpire-view step data"

    tr = record_episode("fireteam", None, seed=3, max_steps=10)
    sen = tr["steps"][2]["soldiers"][0]["sensors"]
    assert "cues" not in sen and "garble" not in sen and "mates" not in sen
    # dead soldiers carry no sensors (nothing senses)
    assert all(
        s["sensors"] is not None or not s["alive"]
        for st in tr["steps"] for s in st["soldiers"]
    )


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


def test_scan_runs_marks_live_training(tmp_path):
    """The dashboard's red 'live' marker: a run whose job pid is still alive.

    Same liveness test as scripts/train_status.py, so the two never disagree.
    A live run is listed *before* its first metrics row — that is exactly the
    moment you want to see it — and its metrics read as empty, not as an error.
    """
    run = tmp_path / "live_run"
    run.mkdir()
    (run / ".job.json").write_text(
        json.dumps({
            "pid": os.getpid(),
            "total_steps": 3_000_000,
            "started_human": "2026-08-07 09:00:00",
            "args": ["--scenario", "platoon", "--total-steps", "3000000"],
        })
    )

    runs = scan_runs(tmp_path)
    assert [r["name"] for r in runs] == ["live_run"]
    assert runs[0]["job"]["live"] is True
    assert runs[0]["job"]["total_steps"] == 3_000_000
    # config.json lands only after train.py finishes setting up; until then the
    # scenario is read off the launch args so the list entry is not just "?"
    assert runs[0]["scenario"] == "platoon"
    assert load_metrics(tmp_path, "live_run")["columns"] == {}

    # a finished run's job file names a pid that is gone: not live, and with no
    # metrics of its own it is not a run at all
    (run / ".job.json").write_text(json.dumps({"pid": 2**30}))
    assert scan_runs(tmp_path) == []
    (run / "metrics.csv").write_text("iteration,env_steps\n1,1024\n")
    assert scan_runs(tmp_path)[0]["job"]["live"] is False

    # a corrupt or half-written job file is "not live", never an exception
    (run / ".job.json").write_text("{not json")
    assert scan_runs(tmp_path)[0]["job"]["live"] is False


def test_metrics_nan_serializes_as_null(tmp_path):
    """An iteration that completed no episode logs NaN — which is not JSON.

    Python's json emits a bare `NaN` token; JSON.parse rejects it, and the
    browser then rendered *no charts at all* for any run containing one such
    iteration (most of them). NaN is missing data, so it goes out as null.
    """
    run = tmp_path / "myrun"
    run.mkdir()
    (run / "metrics.csv").write_text(
        "iteration,env_steps,ep_return\n1,1024,nan\n2,2048,3.5\n3,3072,\n"
    )
    m = load_metrics(tmp_path, "myrun")
    assert m["columns"]["ep_return"] == [None, 3.5, None]
    assert m["columns"]["env_steps"] == [1024.0, 2048.0, 3072.0]
    assert "NaN" not in json.dumps(m), "payload must be valid JSON for the browser"
    json.loads(json.dumps(m))


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

def test_live_order_form_speaks_the_doctrine():
    """The Command tab's order composer offers only lines the net accepts.

    The form's admissibility must mirror inject_order's (direct subordination,
    per-mission minimum authority), and every line composed from its templates
    must round-trip through the real parser to the same recipient and mission
    — the composer and the radio share one formatter, so this is the drift
    alarm.
    """
    import cohort.core.language as lang
    from cohort.core.missions import MissionType
    from cohort.viz.dashboard import LiveSession

    s = LiveSession("squad", None, seed=1)
    form = s.order_form("HQ")
    all_cs = {r["cs"] for r in form["recipients"]}
    assert "SL1" in all_cs and any(cs.startswith("RFN") for cs in all_cs)
    sl = next(r for r in form["recipients"] if r["cs"] == "SL1")
    rfn = next(r for r in form["recipients"] if r["cs"].startswith("RFN"))
    assert "DENY" in sl["missions"], "DENY is a section mission — SL holds it"
    assert "DENY" not in rfn["missions"], "a rifleman can never hold DENY"
    assert sl["leads"] and not rfn["leads"]

    # a team leader may only address its own direct subordinates
    tl_form = s.order_form("TL1")
    tl_cs = {r["cs"] for r in tl_form["recipients"]}
    assert tl_cs and all(cs.startswith("RFN") for cs in tl_cs)

    # every composable line parses back to exactly what the dropdowns said
    targets_of = {
        "objective": form["objectives"],
        "control": form["controls"],
        "unit": form["support_targets"],
        None: [None],
    }
    composed = 0
    for r in form["recipients"]:
        for mname in r["missions"]:
            t = form["templates"][mname]
            targets = [x for x in targets_of[t["target"]] if x != r["cs"]]
            if not targets:
                continue
            phrase = t["phrase"].replace("{T}", targets[0]) if targets[0] else t["phrase"]
            parsed = lang.parse_order(f"{r['cs']}, {phrase} AT T PLUS 3")
            assert parsed.recipient_callsign == r["cs"]
            assert parsed.mission is MissionType[mname]
            assert parsed.delay == 3
            composed += 1
    assert composed >= len(form["recipients"]) * 5, "the sweep must actually cover the grid"

    # EXECUTE lands on the trace like any other traffic
    out = s.execute("HQ")
    assert out["ok"] and out["messages"], "EXECUTE must appear on the net"
