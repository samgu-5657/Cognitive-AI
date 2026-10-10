"""Phase 6H-1 unified cycle tests: real returns, states, traces and
write counts through dialog.process (and one API/CLI compat check).

States asserted from turn["stages"]["cycle"]["state"]: answer |
clarification | abstention | failure. No answer hardcoding beyond the
seed KB's own verified values (drinking/ceramic/11), which the trace
must cite.
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
from cognitive_ai.cognitive_cycle import run_cycle
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory
from cognitive_ai.web.server import create_app

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def make_session(tmp_path, tag="cy"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    return Session(mem, corp), mem


def cycle_state(turn):
    return (turn.get("stages", {}).get("cycle") or {}).get("state")


def test_simple_qa_is_answer(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        turn = D.process(sess, "杯子有什么用途")
        assert turn["route"] == "qa"
        assert cycle_state(turn) == "answer", turn["stages"]
        assert "drinking" in turn["answer"]["text"]
        assert turn["answer"]["certainty"] == "certain"
    finally:
        mem.close()


def test_math_and_multi_domain_task(tmp_path):
    sess, mem = make_session(tmp_path, tag="cy2")
    try:
        turn = D.process(sess, "计算3+4*2")
        assert cycle_state(turn) == "answer"
        assert "答案:11。" in turn["answer"]["text"]
        tr = turn["stages"]["cycle"]["stages"]
        names = [s["stage"] for s in tr]
        assert names[:3] == ["parse", "build", "execute"], names
        assert any(s.get("task_status") == "verified" for s in tr)
    finally:
        mem.close()


def test_unknown_abstains_with_gap(tmp_path):
    sess, mem = make_session(tmp_path, tag="cy3")
    try:
        n0 = len(mem.list_questions())
        turn = D.process(sess, "紫砂壶有什么功能")
        assert cycle_state(turn) in ("abstention", "failure"), turn["stages"]
        assert turn["answer"]["certainty"] == "unknown"
        assert len(mem.list_questions()) == n0 + 1
    finally:
        mem.close()


def test_ambiguous_clarify_and_resume(tmp_path):
    sess, mem = make_session(tmp_path, tag="cy4")
    try:
        t1 = D.process(sess, "它有什么用途")
        assert t1["route"] == "clarify", (t1["route"], t1["answer"])
        assert sess.pending_clarify is not None
        t2 = D.process(sess, "杯子")
        assert t2["route"] == "qa", (t2["route"], t2["answer"])
        assert cycle_state(t2) == "answer", t2["stages"]
        assert "drinking" in t2["answer"]["text"], t2["answer"]
        assert sess.pending_clarify is None
    finally:
        mem.close()


def test_sessions_do_not_share_pending(tmp_path):
    s1, m1 = make_session(tmp_path, tag="cy5a")
    s2, m2 = make_session(tmp_path, tag="cy5b")
    try:
        D.process(s1, "它有什么用途")
        assert s1.pending_clarify is not None
        assert s2.pending_clarify is None
        t = D.process(s2, "杯子有什么用途")
        assert cycle_state(t) == "answer"
    finally:
        m1.close()
        m2.close()


def test_verify_failure_is_failure_not_answer(tmp_path):
    sess, mem = make_session(tmp_path, tag="cy6")
    try:
        # undecidable computation: verified math re-derivation fails,
        # so the cycle must report failure, never a forced answer.
        turn = D.process(sess, "计算2++3")
        assert cycle_state(turn) == "failure", turn["stages"]
        assert turn["answer"]["certainty"] == "unknown"
        assert "11" not in turn["answer"]["text"] and \
            "答案:" not in turn["answer"]["text"], turn["answer"]
    finally:
        mem.close()


def test_conflict_scope_inside_cycle(tmp_path):
    sess, mem = make_session(tmp_path, tag="cy7")
    try:
        mem.add_claim("cup.material: ceramic vs observed glass",
                      subject_id=None, status="contested", confidence=0.5)
        m = D.process(sess, "计算3+4*2")
        assert cycle_state(m) == "answer" and "答案:11。" in \
            m["answer"]["text"]
        f = D.process(sess, "杯子有什么用途")
        assert cycle_state(f) == "answer" and "drinking" in \
            f["answer"]["text"]
        v = D.process(sess, "杯子是不是陶瓷做的")
        assert cycle_state(v) in ("abstention", "failure"), v["stages"]
    finally:
        mem.close()


def test_single_experience_per_cycle(tmp_path):
    import json as _json
    sess, mem = make_session(tmp_path, tag="cy8")
    try:
        n0 = len(mem.list_experiences(limit=100000))
        D.process(sess, "杯子有什么用途")
        rows = mem.list_experiences(limit=100000)
        new = rows[:len(rows) - n0]
        ops = []
        for r in new:
            try:
                d = r.get("detail")
                d = _json.loads(d) if isinstance(d, str) else (d or {})
                ops.append(d.get("op"))
            except Exception:
                ops.append(None)
        assert ops.count("cycle") == 1, ops
        assert "dialog" not in ops, ops
    finally:
        mem.close()


def test_trace_persisted_and_complete(tmp_path):
    sess, mem = make_session(tmp_path, tag="cy9")
    try:
        turn = D.process(sess, "杯子有什么用途")
        tid = turn["answer"].get("task_id") or \
            (sess.last_answer or {}).get("task_id")
        assert tid
        from cognitive_ai.executor import load_trace
        tr = load_trace(mem, tid)
        assert tr and tr["task"]["status"] == "verified"
        assert any(s["op"] == "query_attribute" for s in tr["steps"])
        assert tr.get("verification", {}).get("verdict") == "verified"
    finally:
        mem.close()


def test_api_and_cli_compat(tmp_path):
    data = tmp_path / "cdata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    app = create_app(str(data), str(tmp_path / "cmodels"))
    with TestClient(app) as c:
        r = c.post("/api/chat", json={"text": "杯子有什么用途",
                                      "session_id": "k"}).json()
        assert r["ok"] and "drinking" in "\n".join(r["data"]["lines"])
        assert set(("lines", "route", "certainty", "citations")) <= \
            set(r["data"]), r["data"]
        s = c.post("/api/solve", json={"text": "计算3+4*2",
                                       "session_id": "k"}).json()
        assert s["ok"] and s["data"]["answer"]["text"] == "答案:11。"
    app.state.wb.close()
    sess, mem = make_session(tmp_path, tag="cy10")
    try:
        out = sess.handle("对话 杯子有什么用途")
        assert any("drinking" in l for l in out), out
        assert sess.last_answer is not None
    finally:
        mem.close()


def test_direct_cycle_entry_states(tmp_path):
    sess, mem = make_session(tmp_path, tag="cy11")
    try:
        from cognitive_ai.semantics import parse_sentence
        fr = parse_sentence("杯子有什么用途", source="user").to_dict()
        res = run_cycle(sess, "杯子有什么用途", fr)
        assert res["state"] == "answer"
        assert [s["stage"] for s in res["stages"]][:2] == \
            ["parse", "build"]
        assert res["trace"]["task"]["status"] == "verified"
    finally:
        mem.close()
