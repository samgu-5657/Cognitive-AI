"""Local web UI tests (isolated tmp DBs; one localhost smoke test)."""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import web_ui
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology


def fresh(tmp_path, tag="ui"):
    m = Memory(str(tmp_path / f"{tag}.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    o.add_concept("bowl", "碗", parents=["container"])
    o.add_attribute("cup", "material", "ceramic", status="verified")
    return m


def counts(m):
    return (len(m.list_concepts()),
            len(m.query("SELECT * FROM attributes")),
            len(m.query("SELECT * FROM relations")),
            len(m.list_claims()),
            len(m.query("SELECT * FROM rules")),
            len(m.list_pending_candidates()))


def test_index_and_health(tmp_path):
    m = fresh(tmp_path)
    try:
        code, payload = web_ui.dispatch(m, "", "GET", "/nope", {})
        assert code == 404 and payload["ok"] is False
        code, payload = web_ui.dispatch(m, "", "GET", "/api/health", {})
        assert code == 200 and payload["ok"] is True
    finally:
        m.close()


def test_ask_real_path(tmp_path):
    m = fresh(tmp_path, tag="ui2")
    try:
        code, out = web_ui.dispatch(m, "", "POST", "/api/ask",
                                    {"text": "杯子是不是容器？"})
        assert code == 200, out
        assert out["verdict"]["verdict"] == "supported", out
        assert out["answer"].startswith("是的"), out
        code, out = web_ui.dispatch(m, "", "POST", "/api/ask",
                                    {"text": "   "})
        assert code == 400, out
    finally:
        m.close()


def test_analyze_real_call(tmp_path, monkeypatch):
    from cognitive_ai import semmeaning as meaning_mod
    m = fresh(tmp_path, tag="ui3")
    try:
        calls = {"n": 0}
        real = meaning_mod.understand_compositional_statement

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(meaning_mod, "understand_compositional_statement",
                            spy)
        code, out = web_ui.dispatch(m, "", "POST", "/api/analyze",
                                    {"text": "杯子是容器"})
        assert code == 200 and calls["n"] == 1, out
        assert out["report"].root.status == "supported", out
    finally:
        m.close()


def test_context_real_mechanism(tmp_path):
    m = fresh(tmp_path, tag="ui4")
    try:
        code, out = web_ui.dispatch(
            m, "", "POST", "/api/context",
            {"priors": ["杯子是容器"], "text": "它是容器"})
        assert code == 200, out
        resolutions = out["report"].resolutions
        assert resolutions[0].status == "resolved", resolutions
        assert resolutions[0].resolved_ref.ref_id == "cup"
        code, out = web_ui.dispatch(
            m, "", "POST", "/api/context",
            {"priors": [], "text": "它可以盛水"})
        assert out["report"].resolutions[0].status == "unresolved", out
        code, out = web_ui.dispatch(
            m, "", "POST", "/api/context",
            {"priors": ["杯子是容器", "碗是容器"], "text": "它可以盛水"})
        assert out["report"].resolutions[0].status == "ambiguous", out
    finally:
        m.close()


def test_propose_confirm_reject(tmp_path):
    m = fresh(tmp_path, tag="ui5")
    try:
        code, out = web_ui.dispatch(m, "", "POST", "/api/propose",
                                    {"text": "杯子有把手"})
        assert code == 200, out
        assert out["candidate"]["status"] == "proposed", out
        pid = out["candidate"]["id"]
        assert len(m.query("SELECT * FROM attributes")) == 1
        code, out = web_ui.dispatch(m, "", "POST", "/api/confirm",
                                    {"id": pid})
        assert code == 200 and out["ok"] is True, out
        assert len(m.query("SELECT * FROM attributes")) == 2
        code, out = web_ui.dispatch(m, "", "POST", "/api/propose",
                                    {"text": "杯子有颜色"})
        pid2 = out["candidate"]["id"]
        code, out = web_ui.dispatch(m, "", "POST", "/api/reject",
                                    {"id": pid2})
        assert code == 200 and out["ok"] is True, out
        assert m.get_pending_candidate(pid2)["status"] == "rejected"
        assert len(m.query("SELECT * FROM attributes")) == 2
    finally:
        m.close()


def test_invalid_id_changes_nothing(tmp_path):
    m = fresh(tmp_path, tag="ui6")
    try:
        n0 = counts(m)
        code, out = web_ui.dispatch(m, "", "POST", "/api/confirm",
                                    {"id": 987})
        assert code == 404, out
        code, out = web_ui.dispatch(m, "", "POST", "/api/confirm",
                                    {"id": "x"})
        assert code == 400, out
        assert counts(m) == n0
    finally:
        m.close()


def test_read_only_ops_stable(tmp_path):
    m = fresh(tmp_path, tag="ui7")
    try:
        n0 = counts(m)
        web_ui.dispatch(m, "", "POST", "/api/ask", {"text": "杯子是不是容器？"})
        web_ui.dispatch(m, "", "POST", "/api/analyze",
                        {"text": "杯子是容器"})
        web_ui.dispatch(m, "", "POST", "/api/context",
                        {"priors": ["杯子是容器"], "text": "它是容器"})
        web_ui.dispatch(m, "", "GET", "/api/status", {})
        web_ui.dispatch(m, "", "GET", "/api/pending", {})
        assert counts(m) == n0
    finally:
        m.close()


def test_no_shared_session_state(tmp_path):
    m = fresh(tmp_path, tag="ui8")
    try:
        assert not hasattr(web_ui, "sessions")
        _code, one = web_ui.dispatch(
            m, "", "POST", "/api/context",
            {"priors": ["杯子是容器"], "text": "它是容器"})
        _code, two = web_ui.dispatch(
            m, "", "POST", "/api/context",
            {"priors": [], "text": "它是容器"})
        assert one["report"].resolutions[0].status == "resolved"
        assert two["report"].resolutions[0].status == "unresolved"
    finally:
        m.close()


def test_json_safe_strict(tmp_path):
    from types import MappingProxyType
    from cognitive_ai.models import WorldObjectKind
    m = fresh(tmp_path, tag="ui9")
    try:
        code, out = web_ui.dispatch(m, "", "POST", "/api/analyze",
                                    {"text": "杯子是容器"})
        raw = web_ui.json_safe(out)
        assert json.loads(json.dumps(raw)) == raw
        try:
            web_ui.json_safe({"x": object()})
        except TypeError:
            pass
        else:
            raise AssertionError("str() fallback must not exist")
        assert web_ui.json_safe(WorldObjectKind.CLASS) == "class"
        assert web_ui.json_safe(MappingProxyType({"a": 1})) == {"a": 1}
    finally:
        m.close()


def test_live_localhost_smoke(tmp_path, monkeypatch):
    from cognitive_ai import cli as cli_mod
    data = tmp_path / "srv"
    data.mkdir()
    m = Memory(str(data / "cognitive.db"))
    try:
        o = Ontology(m)
        o.add_concept("entity", "实体")
        o.add_concept("container", "容器", parents=["entity"])
        o.add_concept("cup", "杯子", parents=["container"])
    finally:
        m.close()
    old = (cli_mod.DATA, cli_mod.DB, cli_mod.CORPUS, cli_mod.SEED)
    cli_mod.DATA, cli_mod.DB = str(data), str(data / "cognitive.db")
    cli_mod.CORPUS, cli_mod.SEED = str(data), str(data / "seed.json")
    server = None
    try:
        server, _thread = web_ui.run_in_thread(port=0)
        port = server.server_address[1]
        base = f"http://127.0.0.1:{port}"

        def call(method, path, body=None):
            req = urllib.request.Request(
                base + path, method=method,
                data=json.dumps(body or {}).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read().decode("utf-8"))

        with urllib.request.urlopen(base + "/") as r:
            index = r.read().decode("utf-8")
            assert r.status == 200 and "本地认知工作台" in index, \
                index[:200]
        status, health = call("GET", "/api/health")
        assert status == 200 and health["ok"] is True
        status, ans = call("POST", "/api/ask", {"text": "杯子是不是容器？"})
        assert status == 200 and ans["answer"].startswith("是的"), ans
    finally:
        if server is not None:
            server.shutdown()
        cli_mod.DATA, cli_mod.DB, cli_mod.CORPUS, cli_mod.SEED = old


def test_cli_ui_help():
    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "cognitive_ai.cli", "ui", "--help"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."))
    assert r.returncode == 0, r.stderr[-800:]
    assert "--port" in r.stdout and "--host" in r.stdout
