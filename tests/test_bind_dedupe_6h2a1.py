"""Phase 6H-2A.1 duplicate-binding regression tests.

Root cause (proven in Experiment-1): semantics._fill appended the
subject once in the generic slot prologue AND again in the
query/verify/reason branches, so one question produced two identical
subject entities, two identical verified queries and a doubled answer.

Fix: the generic prologue is the single subject writer; redundant
branch appends removed (object slots untouched). These tests pin frame
shape, trace query counts and end-to-end answers.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from cognitive_ai import dialog as D
from cognitive_ai.cli import seed_from_json
from cognitive_ai.executor import load_trace
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory
from cognitive_ai.semantics import parse_sentence
from cognitive_ai.web.server import create_app
import shutil

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def make_session(tmp_path, tag="bd"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    return Session(mem, corp), mem


def test_1_function_query_single_subject(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        fr = parse_sentence("杯子有什么用途", source="user",
                            lexicon=sess._lexicon())
        subs = [e for e in fr.entities if e.get("role") == "subject"]
        assert [e["text"] for e in subs] == ["杯子"], fr.entities
    finally:
        mem.close()


def test_2_attr_query_single_subject(tmp_path):
    sess, mem = make_session(tmp_path, tag="bd2")
    try:
        fr = parse_sentence("杯子由什么制成", source="user",
                            lexicon=sess._lexicon())
        subs = [e["text"] for e in fr.entities if e.get("role") == "subject"]
        assert subs == ["杯子"], fr.entities
    finally:
        mem.close()


def test_3_two_capabilities_kept(tmp_path):
    sess, mem = make_session(tmp_path, tag="bd3")
    try:
        turn = D.process(
            sess, "我发明了一种叫光盒的东西，它能够储存光能，并在黑暗中释放光能。")
        assert turn["route"] == "teach_propose", turn["route"]
        caps = (sess.pending_define or {}).get("capabilities", [])
        assert len(caps) == 2, caps
        assert caps[0]["action"] == "储存光能"
        assert "黑暗" in caps[1]["action"] and caps[1]["condition"] == "黑暗"
    finally:
        mem.close()


def test_4_legitimate_slots_kept(tmp_path):
    sess, mem = make_session(tmp_path, tag="bd4")
    try:
        fr = parse_sentence("杯子是不是容器", source="user",
                            lexicon=sess._lexicon()).to_dict()
        roles = sorted((e["text"], e["role"]) for e in fr["entities"])
        assert ("杯子", "subject") in roles, roles
        assert any(r == "object" for _, r in roles), roles
        fr2 = parse_sentence("杯子是容器", source="user",
                             lexicon=sess._lexicon()).to_dict()
        assert len(fr2["entities"]) == 2, fr2["entities"]
    finally:
        mem.close()


def test_5_web_trace_single_query_single_answer(tmp_path):
    data = tmp_path / "bdata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    app = create_app(str(data), str(tmp_path / "bmodels"))
    with TestClient(app) as c:
        r = c.post("/api/chat", json={"text": "杯子有什么用途",
                                      "session_id": "b"}).json()
        assert r["ok"]
        lines = r["data"]["lines"]
        assert sum("drinking" in l for l in lines) == 1, lines
        tid = c.get("/api/tasks?limit=1").json()["data"][0]["task_id"]
        tr = c.get(f"/api/tasks/{tid}").json()["data"]
        queries = [s for s in tr["steps"]
                   if s.get("op") == "query_attribute"]
        assert len(queries) == 1, [(s.get("inputs"), s.get("status"))
                                   for s in tr["steps"]]
        assert tr["task"]["goal"]["entities"] == ["杯子"], \
            tr["task"]["goal"]
    app.state.wb.close()


def test_6_states_still_honest(tmp_path):
    sess, mem = make_session(tmp_path, tag="bd6")
    try:
        mem.add_claim("cup.material: ceramic vs observed glass",
                      subject_id=None, status="contested", confidence=0.5)
        assert D.process(sess, "花钱如流水般哗啦")["answer"][
            "certainty"] == "unknown"
        assert D.process(sess, "杯子是不是陶瓷做的")["answer"][
            "certainty"] == "unknown"
        t = D.process(sess, "杯子有什么用途")
        assert "drinking" in t["answer"]["text"] and \
            t["answer"]["text"].count("drinking") == 1, t["answer"]
    finally:
        mem.close()
