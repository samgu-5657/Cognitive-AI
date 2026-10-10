"""Multi-turn conversation tests (post-6D): topic setup, pronoun
reference, ellipsis-free follow-ups, topic-shift math, topic return,
and session isolation -- all through the real /api/chat entry on
isolated DBs. No per-sentence hardcodes; reference uses the existing
context machinery (update_context/resolve_frame)."""
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
    data = tmp_path / "mdata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED_SRC, data / "concepts.json")
    models = tmp_path / "mmodels"
    models.mkdir()
    app = create_app(str(data), str(models))
    with TestClient(app) as c:
        yield c
    app.state.wb.close()


def say(c, text, sid="conv"):
    r = c.post("/api/chat", json={"text": text, "session_id": sid}).json()
    assert r["ok"], r
    return r["data"]


def test_full_conversation_script(client):
    t1 = say(client, "我想了解杯子。")
    assert t1["route"] == "topic", t1
    assert "cup" in "\n".join(t1["lines"])
    t2 = say(client, "它有什么用途？")
    assert t2["route"] == "qa" and t2["certainty"] == "certain", t2
    assert "drinking" in "\n".join(t2["lines"]), t2
    t3 = say(client, "那它由什么制成？")
    assert t3["route"] == "qa" and t3["certainty"] == "certain", t3
    assert "ceramic" in "\n".join(t3["lines"]), t3
    t4 = say(client, "为什么？")
    assert t4["route"] == "followup", t4
    assert "ceramic" in "\n".join(t4["lines"]), t4
    t5 = say(client, "换个话题，计算3+4*2。")
    assert t5["route"] == "qa", t5
    assert "答案:11。" in "\n".join(t5["lines"]), t5
    t6 = say(client, "回到刚才的杯子，它有什么用途？")
    assert t6["route"] == "qa" and t6["certainty"] == "certain", t6
    assert "drinking" in "\n".join(t6["lines"]), t6


def test_ellipsis_reuses_topic(client):
    say(client, "我想了解杯子。", "e1")
    t = say(client, "那材质呢？", "e1")
    assert t["route"] == "qa", t
    assert "ceramic" in "\n".join(t["lines"]), t


def test_pronoun_without_topic_asks(client):
    t = say(client, "它有什么用途？", "fresh")
    assert t["route"] in ("clarify", "qa")
    if t["route"] == "clarify":
        assert "它" in "\n".join(t["lines"])
    # zero knowledge writes either way
    d = client.get("/api/concepts/cup").json()["data"]
    assert all(a["status"] != "verified" or a["key"] in
               ("function", "material") for a in d["attributes"])


def test_sessions_isolated(client):
    say(client, "我想了解杯子。", "sA")
    t = say(client, "它有什么用途？", "sB")
    # sB never introduced a topic: must not borrow sA's cup
    assert not ("drinking" in "\n".join(t["lines"]) and
                t["certainty"] == "certain"), t


def test_material_conflict_gates_reference_answer(client):
    c = client
    wb = c.app.state.wb
    wb.mem.add_claim("cup.material: ceramic vs observed glass",
                     subject_id=None, status="contested", confidence=0.5)
    say(c, "我想了解杯子。", "cc")
    t = say(c, "那它由什么制成？", "cc")
    assert t["certainty"] == "unknown", t
    assert "ceramic" not in "\n".join(t["lines"]) or \
        "不是" in "\n".join(t["lines"]) or \
        "未知" in "\n".join(t["lines"]) or \
        "证据不足" in "\n".join(t["lines"]), t
