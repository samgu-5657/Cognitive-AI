"""Post-6D conflict-scope regression tests (isolated DBs only).

Scenario mirrors the real database (cup.material contested claim) WITHOUT
touching it: pure-math tasks must stay verified, unrelated attribute
queries must answer, related material judgments stay gated, and failure
texts must not double-wrap refusals.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from cognitive_ai import dialog as D
from cognitive_ai.cli import seed_from_json
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory
from cognitive_ai.web.server import create_app

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))
SEED_SRC = SEED


def make_conflicted(tmp_path, tag="cs"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    mem.add_claim("cup.material: ceramic vs observed glass",
                  subject_id=None, status="contested", confidence=0.5)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    return Session(mem, corp), mem


def test_1_math_verified_despite_material_conflict(tmp_path):
    sess, mem = make_conflicted(tmp_path)
    try:
        turn = D.process(sess, "计算3+4*2")
        assert turn["route"] == "qa"
        assert "答案:11。" in turn["answer"]["text"], turn["answer"]
        assert turn["answer"]["certainty"] == "certain"
    finally:
        mem.close()


def test_2_material_judgment_stays_gated(tmp_path):
    sess, mem = make_conflicted(tmp_path)
    try:
        turn = D.process(sess, "杯子是不是陶瓷做的")
        assert turn["answer"]["certainty"] == "unknown", turn["answer"]
        assert "drinking" not in turn["answer"]["text"]
    finally:
        mem.close()


def test_3_function_query_not_blocked_by_material_conflict(tmp_path):
    sess, mem = make_conflicted(tmp_path)
    try:
        for q in ("杯子有什么用途", "杯子可以拿来做什么"):
            turn = D.process(sess, q)
            assert "drinking" in turn["answer"]["text"], (q, turn["answer"])
        # reason questions must NOT borrow the function answer, even
        # though verified function knowledge exists.
        turn = D.process(sess, "为什么人们会使用杯子")
        assert "drinking" not in turn["answer"]["text"], turn["answer"]
        assert turn["answer"]["certainty"] == "unknown", turn["answer"]
    finally:
        mem.close()


def test_4_no_credible_function_knowledge_stays_honest(tmp_path):
    db = str(tmp_path / "nofunc.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    try:
        mem.execute("DELETE FROM attributes WHERE concept_id='cup' "
                    "AND key='function'")
        sess = Session(mem, str(tmp_path))
        turn = D.process(sess, "杯子有什么用途")
        # cup.function deleted, but drinkware.function=drinking (verified)
        # is inherited: the honest answer is qualified, never certain.
        assert turn["answer"]["certainty"] in ("unknown", "qualified")
        assert turn["answer"]["certainty"] != "certain", turn["answer"]
        assert "继承" in turn["answer"]["text"] or \
            "假设" in turn["answer"]["text"] or \
            "drinking" not in turn["answer"]["text"], turn["answer"]
    finally:
        mem.close()


def test_5_why_without_causal_evidence_makes_no_claim(tmp_path):
    sess, mem = make_conflicted(tmp_path, "cs5")
    try:
        turn = D.process(sess, "为什么杯子会漏水")
        assert turn["answer"]["certainty"] == "unknown"
        assert "drinking" not in turn["answer"]["text"]
        assert "ceramic" not in turn["answer"]["text"]
    finally:
        mem.close()


def test_6_unknown_writes_no_verified_knowledge(tmp_path):
    sess, mem = make_conflicted(tmp_path, "cs6")
    try:
        n0 = mem.query("SELECT COUNT(*) AS n FROM attributes")[0]["n"]
        turn = D.process(sess, "花钱如流水般哗啦")
        assert turn["answer"]["certainty"] == "unknown"
        assert mem.query("SELECT COUNT(*) AS n FROM attributes")[0]["n"] \
            == n0
        rows = mem.query("SELECT * FROM attributes WHERE status='verified'"
                         " AND source NOT IN ('seed','validated')")
        assert rows == [], rows
    finally:
        mem.close()


def test_7_no_double_wrapped_refusal(tmp_path):
    sess, mem = make_conflicted(tmp_path, "cs7")
    try:
        turn = D.process(sess, "杯子是不是陶瓷做的")
        text = turn["answer"]["text"]
        assert text.count("证据不足") <= 1, text
        assert text.count("结论未定") <= 1, text
    finally:
        mem.close()


def test_8_web_chat_and_solve_math_under_conflict(tmp_path):
    data = tmp_path / "cdata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED_SRC, data / "concepts.json")
    app = create_app(str(data), str(tmp_path / "cmodels"))
    with TestClient(app) as c:
        wb = app.state.wb
        wb.mem.add_claim("cup.material: ceramic vs observed glass",
                         subject_id=None, status="contested",
                         confidence=0.5)
        r = c.post("/api/chat", json={"text": "计算3+4*2",
                                      "session_id": "cs"}).json()
        assert r["ok"] and "答案:11。" in "\n".join(r["data"]["lines"]), r
        s = c.post("/api/solve", json={"text": "计算3+4*2",
                                       "session_id": "cs"}).json()
        assert s["ok"] and s["data"]["answer"]["text"] == "答案:11。", s
        q = c.post("/api/chat", json={"text": "杯子有什么用途",
                                      "session_id": "cs"}).json()
        assert q["ok"] and "drinking" in "\n".join(q["data"]["lines"]), q
    app.state.wb.close()


def test_9_reason_question_keeps_type_through_web_stack(tmp_path):
    """Structural regression: parse type + route + trace, not strings."""
    from cognitive_ai.semantics import parse_sentence, resolve_question
    data = tmp_path / "rdata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED_SRC, data / "concepts.json")
    app = create_app(str(data), str(tmp_path / "rmodels"))
    with TestClient(app) as c:
        wb = app.state.wb
        wb.mem.add_claim("cup.material: ceramic vs observed glass",
                         subject_id=None, status="contested",
                         confidence=0.5)
        # 1. parse type stays reason, never function
        fr = parse_sentence("为什么人们会使用杯子",
                            source="web").to_dict()
        assert fr["intent"] == "query"
        assert any(p.get("type") == "query_reason"
                   for p in fr["predicates"]), fr
        assert not any(p.get("type") == "query_function"
                       for p in fr["predicates"]), fr
        rq = resolve_question(fr, {})
        assert rq["kind"] == "reason_q", rq
        # 2. route + answer through the real entry
        r = c.post("/api/chat", json={"text": "为什么人们会使用杯子",
                                      "session_id": "rs"}).json()
        assert r["ok"] and r["data"]["route"] == "qa", r
        assert "drinking" not in "\n".join(r["data"]["lines"]), r
        assert r["data"]["certainty"] == "unknown", r
        # 3. execution trace: reason gate, no function answer laundered
        tasks = c.get("/api/tasks?limit=5").json()["data"]
        assert tasks, "dialog QA must persist its trace"
        tr = c.get(f"/api/tasks/{tasks[0]['task_id']}").json()["data"]
        assert tr["task"]["goal"].get("ask_about") == "reason", \
            tr["task"]["goal"]
        assert not any(s.get("inputs", {}).get("key") == "function" and
                       s.get("status") == "verified"
                       for s in tr["steps"]
                       if s.get("op") == "query_attribute"), \
            "no verified function read may back a reason answer"
        # 4. function question still answers drinking on the same stack
        q = c.post("/api/chat", json={"text": "杯子有什么用途",
                                      "session_id": "rs"}).json()
        assert q["ok"] and "drinking" in "\n".join(q["data"]["lines"]), q
    app.state.wb.close()
