"""Phase 6B web end-to-end tests: the browser HTTP interface goes through
the unified dialogue pipeline. All calls drive the real service layer via
FastAPI TestClient on isolated tmp data dirs (never dialog.process
directly). No pretrained LLM anywhere in the loop."""
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
    data = tmp_path / "wdata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED_SRC, data / "concepts.json")
    models = tmp_path / "wmodels"
    models.mkdir()
    app = create_app(str(data), str(models))
    with TestClient(app) as c:
        yield c


def lines_of(r):
    assert r["ok"], r
    return "\n".join(r["data"]["lines"])


def test_1_web_greeting_answers(client):
    r = client.post("/api/chat",
                    json={"text": "你好", "session_id": "w1"}).json()
    t = lines_of(r)
    assert "Cognitive AI" in t
    assert r["data"]["route"] == "smalltalk"


def test_2_web_paraphrases_agree_on_seed_knowledge(client):
    a = lines_of(client.post(
        "/api/chat", json={"text": "杯子有什么用途",
                           "session_id": "w2"}).json())
    b = lines_of(client.post(
        "/api/chat", json={"text": "杯子可以拿来做什么",
                           "session_id": "w2"}).json())
    assert "drinking" in a or "饮用" in a, a
    assert "drinking" in b or "饮用" in b, b


def test_3_web_math_still_solves(client):
    r = client.post("/api/solve", json={"text": "计算3+4*2",
                                        "session_id": "w3"}).json()
    assert r["ok"] and r["data"]["answer"]["text"] == "答案:11。"
    assert r["data"]["task"]["status"] == "verified"


def test_4_web_unknown_not_fabricated(client):
    r = client.post("/api/chat", json={"text": "花钱如流水般哗啦",
                                       "session_id": "w4"}).json()
    t = lines_of(r)
    assert r["data"]["certainty"] == "unknown"
    assert "drinking" not in t and "陶瓷" not in t


def test_5_web_why_without_evidence_stays_honest(client):
    r = client.post("/api/chat", json={"text": "为什么杯子会漏水",
                                       "session_id": "w5"}).json()
    assert r["data"]["certainty"] == "unknown"
    t = lines_of(r)
    assert "drinking" not in t, t


def test_6_web_teach_and_rules_no_regression(client):
    r1 = client.post("/api/chat", json={"text": "帮助",
                                        "session_id": "w6"}).json()
    assert any("概念" in l for l in r1["data"]["lines"])
    r2 = client.post("/api/chat", json={"text": "教 茶壶",
                                        "session_id": "w6"}).json()
    assert any("茶壶" in l for l in r2["data"]["lines"])
    r3 = client.post("/api/chat", json={"text": "材质=陶瓷",
                                        "session_id": "w6"}).json()
    assert any("observation" in l for l in r3["data"]["lines"])
    train = [{"id": "a", "inputs": {"n": "2"}, "outputs": {"out": "4"}},
             {"id": "b", "inputs": {"n": "3"}, "outputs": {"out": "6"}}]
    held = [{"id": "c", "inputs": {"n": "5"}, "outputs": {"out": "10"}}]
    rid = client.post("/api/rules/induce",
                      json={"name": "double", "domain": "math",
                            "examples": train, "negatives": []}).json()
    assert rid["ok"] and rid["data"]["saved"]
    v = client.post("/api/rules/validate",
                    json={"rule_id": rid["data"]["saved"], "train": train,
                          "heldout": held, "negatives": [],
                          "activate": False}).json()
    assert v["ok"] and v["data"]["validation"]["passed"] is True


def test_7_unknown_writes_no_verified_knowledge(client):
    before = client.get("/api/concepts", params={"q": "cup"}).json()
    n0 = len(before["data"])
    r = client.post("/api/chat", json={"text": "紫砂壶为什么会发光",
                                       "session_id": "w7"}).json()
    assert r["ok"] and r["data"]["certainty"] == "unknown"
    after = client.get("/api/concepts", params={"q": "cup"}).json()
    assert len(after["data"]) == n0
    det = client.get("/api/concepts/cup").json()
    assert det["ok"]
    assert not any(a.get("status") == "verified" and
                   a.get("value") == "发光"
                   for a in det["data"]["attributes"])


def test_8_web_send_button_routes_math(client):
    # "发送" == POST /api/chat (unified dialog); math must reach the
    # existing engine, not the unknown-prose rejection.
    r = client.post("/api/chat", json={"text": "计算3+4*2",
                                       "session_id": "w8"}).json()
    assert r["ok"] and r["data"]["route"] == "qa"
    assert "答案:11。" in lines_of(r)
    assert r["data"]["certainty"] == "certain"
    r = client.post("/api/chat", json={"text": "求解2x+3=7",
                                       "session_id": "w8"}).json()
    assert r["ok"] and r["data"]["route"] == "qa"
    assert "x=2" in lines_of(r), r["data"]
    # the 解题 button path is untouched and agrees
    s = client.post("/api/solve", json={"text": "计算3+4*2",
                                        "session_id": "w8"}).json()
    assert s["ok"] and s["data"]["answer"]["text"] == "答案:11。"
