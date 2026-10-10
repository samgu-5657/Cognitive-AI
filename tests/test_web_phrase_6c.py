"""Phase 6C web end-to-end (task 5): learned phrasings work through the
real browser entry (/api/chat). Teaching, induction, validation and
activation all go through /api/chat commands; the final probe is an
unseen entity in the learned construction answered from the real KB.
No direct dialog.process() calls in assertions paths (service only)."""
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
    data = tmp_path / "pdata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED_SRC, data / "concepts.json")
    models = tmp_path / "pmodels"
    models.mkdir()
    app = create_app(str(data), str(models))
    with TestClient(app) as c:
        yield c


def chat(c, text, sid="p6c"):
    r = c.post("/api/chat", json={"text": text, "session_id": sid}).json()
    assert r["ok"], r
    return r["data"]


def test_web_learned_phrasing_answers_from_kb(client):
    d0 = chat(client, "告诉我杯子的用途")
    assert d0["route"] == "unknown"
    assert chat(client, "说法 告诉我碗的用途 => 碗有什么用途")["lines"]
    assert chat(client, "说法 告诉我瓶子的用途 => 瓶子有什么用途")["lines"]
    assert chat(client,
                "说法 告诉我杯子的用途 => 杯子有什么用途 --留存")["lines"]
    ind = chat(client, "说法归纳")
    assert any("slot" in l for l in ind["lines"]), ind
    rid = [l.split()[0] for l in ind["lines"] if l.strip().startswith("pr_")]
    assert rid, ind
    val = chat(client, f"说法验证 {rid[0]}")
    assert any("验证通过" in l for l in val["lines"]), val
    act = chat(client, f"说法激活 {rid[0]}")
    assert any("已激活" in l for l in act["lines"]), act
    # unseen entity, learned construction, real KB answer
    d1 = chat(client, "告诉我茶壶的用途")
    assert d1["route"] == "qa", d1
    # unlearned construction still unknown, nothing fabricated
    d2 = chat(client, "请告诉我茶壶的材质")
    assert d2["route"] == "unknown", d2
    assert d2["certainty"] == "unknown"
    # /api/solve compatibility untouched
    s = client.post("/api/solve", json={"text": "计算3+4*2",
                                        "session_id": "p6c"}).json()
    assert s["ok"] and s["data"]["answer"]["text"] == "答案:11。"
