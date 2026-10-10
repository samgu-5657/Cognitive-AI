"""Phase 6H-1a control-ownership tests: one coordinator, counted calls.

Wraps plan/verify/decide/persist to prove a plain request plans once,
verifies once, persists once and logs one experience; that repair stays
bounded in the single loop; and that solve_text/CLI/Web entries behave
identically through the controller.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from cognitive_ai import dialog as D
from cognitive_ai import executor as E
from cognitive_ai import metacog as M
from cognitive_ai import verify as V
from cognitive_ai.cli import seed_from_json
from cognitive_ai.cognitive_cycle import run_cycle
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory
from cognitive_ai.web.server import create_app

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def make_session(tmp_path, tag="cc"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    return Session(mem, corp), mem


def test_1_simple_request_completes(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        turn = D.process(sess, "杯子有什么用途")
        assert turn["stages"]["cycle"]["state"] == "answer"
        assert "drinking" in turn["answer"]["text"]
    finally:
        mem.close()


def test_2_3_single_pass_counts(tmp_path, monkeypatch):
    sess, mem = make_session(tmp_path, tag="cc2")
    calls = {"plan": 0, "run": 0, "verify": 0, "decide": 0, "persist": 0}
    orig_plan, orig_run = E._make_plan, E._run_plan
    orig_verify, orig_decide = V.verify_task, M.decide
    orig_persist = E._persist

    def plan(t):
        calls["plan"] += 1
        return orig_plan(t)

    def run(t, p, c, tr):
        calls["run"] += 1
        return orig_run(t, p, c, tr)

    def verify(t, s, m, o=None):
        calls["verify"] += 1
        return orig_verify(t, s, m, o)

    def decide(t, s, v, m, a=0, b=2):
        calls["decide"] += 1
        return orig_decide(t, s, v, m, a, b)

    def persist(m, t, tr):
        calls["persist"] += 1
        return orig_persist(m, t, tr)

    monkeypatch.setattr(E, "_make_plan", plan)
    monkeypatch.setattr(E, "_run_plan", run)
    monkeypatch.setattr(V, "verify_task", verify)
    monkeypatch.setattr(M, "decide", decide)
    monkeypatch.setattr(E, "_persist", persist)
    try:
        n0 = len(mem.list_experiences(limit=100000))
        turn = D.process(sess, "杯子有什么用途")
        assert turn["stages"]["cycle"]["state"] == "answer"
        assert calls == {"plan": 1, "run": 1, "verify": 1,
                         "decide": 0, "persist": 1}, calls
        import json as _json
        ops = []
        for r in mem.list_experiences(limit=100000)[:100000 - n0]:
            try:
                d = r.get("detail")
                ops.append((_json.loads(d) if isinstance(d, str)
                            else (d or {})).get("op"))
            except Exception:
                ops.append(None)
        assert ops == ["cycle"], ops
    finally:
        mem.close()


def test_4_5_bounded_repair_and_honest_failure(tmp_path):
    sess, mem = make_session(tmp_path, tag="cc3")
    try:
        turn = D.process(sess, "计算2++3")
        st = turn["stages"]["cycle"]
        assert st["state"] == "failure", st
        assert turn["answer"]["certainty"] == "unknown"
        reps = [s for s in st["stages"] if s["stage"] == "execute"]
        assert reps, st
        from cognitive_ai.executor import load_trace
        tid = (sess.last_answer or {}).get("task_id")
        if tid:
            tr = load_trace(mem, tid)
            assert len(tr.get("repairs", [])) <= 2, tr.get("repairs")
    finally:
        mem.close()


def test_6_entries_stay_compatible(tmp_path):
    sess, mem = make_session(tmp_path, tag="cc4")
    try:
        from cognitive_ai.executor import solve_text
        from cognitive_ai.semantics import parse_sentence
        tr = solve_text("计算3+4*2",
                        parse_sentence("计算3+4*2", source="t").to_dict(),
                        mem, sess.onto, sess.reasoner)
        assert tr["task"]["status"] == "verified"
        assert tr["task"]["task_type"] == "math"
        out = sess.handle("对话 杯子有什么用途")
        assert any("drinking" in l for l in out), out
    finally:
        mem.close()
    data = tmp_path / "wdata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    app = create_app(str(data), str(tmp_path / "wmodels"))
    with TestClient(app) as c:
        r = c.post("/api/chat", json={"text": "杯子有什么用途",
                                      "session_id": "k"}).json()
        assert r["ok"] and "drinking" in "\n".join(r["data"]["lines"])
        s = c.post("/api/solve", json={"text": "计算3+4*2",
                                       "session_id": "k"}).json()
        assert s["ok"] and s["data"]["answer"]["text"] == "答案:11。"
        assert s["data"]["task"]["status"] == "verified"
    app.state.wb.close()
