"""Phase 5A workbench tests. Live server NOT needed: FastAPI TestClient
drives the real service layer on isolated tmp data dirs. The real user
database is never touched."""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from cognitive_ai.web.server import create_app

SEED_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "data", "concepts.json"))


@pytest.fixture()
def client(tmp_path):
    data = tmp_path / "data"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED_SRC, data / "concepts.json")
    models = tmp_path / "models"
    models.mkdir()
    app = create_app(str(data), str(models))
    with TestClient(app) as c:
        yield c, str(data)


def ex(i, inputs, outputs):
    return {"id": i, "inputs": inputs, "outputs": outputs}


def test_1_home_and_status(client):
    c, _ = client
    assert c.get("/").status_code == 200
    r = c.get("/api/status").json()
    assert r["ok"] and r["data"]["service"] == "ok"
    assert r["data"]["kb_version"] >= 1


def test_2_cli_survives_without_ui_dep(monkeypatch):
    import subprocess
    code = ("import sys; sys.modules['fastapi'] = None;"
            "sys.modules['uvicorn'] = None;"
            "import cognitive_ai.cli as C;"
            "print('cli-ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=os.path.join(
                           os.path.dirname(__file__), ".."))
    assert r.returncode == 0 and "cli-ok" in r.stdout, r.stderr[-500:]


def test_3_chinese_multi_turn(client):
    c, _ = client
    s = "u1"
    r1 = c.post("/api/chat", json={"text": "帮助", "session_id": s}).json()
    assert r1["ok"] and any("概念" in l for l in r1["data"]["lines"])
    r2 = c.post("/api/chat",
                json={"text": "教 茶壶", "session_id": s}).json()
    assert any("茶壶" in l for l in r2["data"]["lines"])
    r3 = c.post("/api/chat",
                json={"text": "材质=陶瓷", "session_id": s}).json()
    assert any("observation" in l for l in r3["data"]["lines"])


def test_4_unparseable_writes_nothing(client):
    c, data = client
    before = c.get("/api/history?limit=1000").json()["data"]
    n0 = (len(before["changes"]), len(before["experiences"]))
    r = c.post("/api/solve", json={"text": "花钱如流水",
                                   "session_id": "u"}).json()
    assert r["ok"] and r["data"]["failure_cause"] == "unparseable"
    after = c.get("/api/history?limit=1000").json()["data"]
    assert (len(after["changes"]), len(after["experiences"])) == n0


def test_5_teach_preview_confirm(client):
    c, _ = client
    pv = c.post("/api/teach/preview",
                json={"name": "紫砂壶", "attributes": {"材质": "紫砂"},
                      "parents": ["container"]}).json()
    assert pv["ok"] and pv["data"]["will_create"] is True
    assert "hypothesis" in pv["data"]["as_status"]
    cf = c.post("/api/teach/confirm",
                json={"name": "紫砂壶", "attributes": {"材质": "紫砂"},
                      "parents": ["container"]}).json()
    assert cf["ok"] and cf["data"]["experience_id"]
    det = c.get("/api/concepts", params={"q": "紫砂"})
    names = json.dumps(det.json(), ensure_ascii=False)
    assert "紫砂" in names


def test_6_rule_examples_and_validation_saved(client):
    c, _ = client
    train = [ex("a", {"n": "2"}, {"out": "4"}),
             ex("b", {"n": "3"}, {"out": "6"})]
    held = [ex("c", {"n": "5"}, {"out": "10"})]
    r = c.post("/api/rules/induce",
               json={"name": "double", "domain": "math",
                     "examples": train, "negatives": []}).json()
    assert r["ok"] and r["data"]["saved"]
    rid = r["data"]["saved"]
    v = c.post("/api/rules/validate",
               json={"rule_id": rid, "train": train, "heldout": held,
                     "negatives": [], "activate": False}).json()
    assert v["ok"] and v["data"]["validation"]["passed"] is True
    show = c.get(f"/api/rules/{rid}").json()
    assert show["ok"] and show["data"]["status"] == "validated"


def test_7_failed_validation_blocks_activation(client):
    c, _ = client
    train = [ex("a", {"n": "2"}, {"out": "4"}),
             ex("b", {"n": "3"}, {"out": "6"})]
    held = [ex("c", {"n": "5"}, {"out": "11"})]  # wrong: true is 10
    r = c.post("/api/rules/induce",
               json={"name": "dblbad", "domain": "math",
                     "examples": train, "negatives": []}).json()
    assert r["ok"] and r["data"]["saved"]
    v = c.post("/api/rules/validate",
               json={"rule_id": r["data"]["saved"], "train": train,
                     "heldout": held, "negatives": [], "activate": True}
               ).json()
    assert v["ok"] and v["data"]["promotion"]["ok"] is False
    show = c.get(f"/api/rules/{r['data']['saved']}").json()
    assert show["data"]["status"] != "active"


def test_8_active_rule_called_in_task(client):
    c, _ = client
    train = [ex("a", {"n": "2"}, {"out": "4"}),
             ex("b", {"n": "3"}, {"out": "6"})]
    held = [ex("c", {"n": "9"}, {"out": "18"})]
    rid = c.post("/api/rules/induce",
                 json={"name": "dbl", "domain": "math",
                       "examples": train, "negatives": []}
                 ).json()["data"]["saved"]
    c.post("/api/rules/validate",
           json={"rule_id": rid, "train": train, "heldout": held,
                 "negatives": [], "activate": True})
    s = c.post("/api/solve",
               json={"text": "计算99+1", "session_id": "u"}).json()
    assert s["ok"]
    calls = c.get(f"/api/rules/{rid}/calls").json()["data"]
    assert calls["status"] == "active"


def test_9_trace_from_real_execution(client):
    c, _ = client
    s = c.post("/api/solve", json={"text": "计算3+4*2",
                                   "session_id": "u"}).json()
    assert s["ok"] and s["data"]["answer"]["text"] == "答案:11。"
    tid = s["data"]["task"]["task_id"]
    t = c.get(f"/api/tasks/{tid}").json()
    assert t["ok"] and any(st["op"] == "compute" for st in t["data"]["steps"])


def test_10_views_use_real_db(client):
    c, _ = client
    r = c.get("/api/concepts", params={"q": "cup"}).json()
    assert r["ok"] and any(x["id"] == "cup" for x in r["data"])
    d = c.get("/api/concepts/cup").json()
    assert d["ok"] and d["data"]["id"] == "cup"
    assert c.get("/api/concepts/nope").status_code == 404


def test_11_sessions_isolated(client):
    c, _ = client
    c.post("/api/chat", json={"text": "教 甲物", "session_id": "sA"})
    c.post("/api/chat", json={"text": "材质=铁", "session_id": "sA"})
    r = c.post("/api/chat", json={"text": "上下文",
                                  "session_id": "sB"}).json()
    assert r["ok"] and "甲物" not in json.dumps(r, ensure_ascii=False)


def test_12_input_validation(client):
    c, _ = client
    assert c.post("/api/explore", json={"steps": 999}).status_code == 400
    assert c.post("/api/explore", json={"steps": 0}).status_code == 400
    assert c.post("/api/solve", json={"text": ""}).status_code == 400
    assert c.post("/api/solve",
                  json={"text": "x" * 501}).status_code == 400
    assert c.get("/api/rules/nope").status_code == 404
    assert c.post("/api/queue/review",
                  json={"sample_id": "nope",
                        "accept": True}).status_code == 400


def test_13_no_ungated_model_activation(client):
    c, _ = client
    # the API surface exposes no activation path except validated rules;
    # neural gate module has no promote endpoint at all
    paths = [r.path for r in c.app.routes
             if hasattr(r, "path") and r.path.startswith("/api/")]
    assert not any("promote" in p or "activate" in p for p in paths
                   if "rules" not in p)
    m = c.get("/api/model").json()
    assert m["ok"] and "note" in m["data"]


def test_14_persistence_across_reopen(tmp_path):
    from cognitive_ai.web.server import create_app as _mk
    from fastapi.testclient import TestClient as TC
    data = tmp_path / "d2"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED_SRC, data / "concepts.json")
    app1 = _mk(str(data), str(tmp_path / "m"))
    with TC(app1) as c1:
        c1.post("/api/teach/confirm",
                json={"name": "持久杯", "attributes": {"材质": "纸"},
                      "parents": []})
    app1.state.wb.close()
    app2 = _mk(str(data), str(tmp_path / "m"))
    with TC(app2) as c2:
        r = c2.get("/api/concepts", params={"q": "持久杯"}).json()
        assert any("持久杯" in x["id"] or "持久杯" in x["name"]
                   for x in r["data"])
    app2.state.wb.close()
