"""Phase 4C end-to-end tests: one shared execution flow, three domains.

Isolated tmp DBs. Math samples are independent of any fixture (engine
computes; tests assert computed values). No hardcoded question->answer
mapping exists anywhere in src/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.cli import seed_from_json
from cognitive_ai.executor import load_trace, solve_text
from cognitive_ai.explainer import explain
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.reasoner import Reasoner
from cognitive_ai.semantics import parse_sentence

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def fresh_db(tmp_path, tag):
    mem = Memory(str(tmp_path / f"{tag}.db"))
    seed_from_json(mem, SEED)
    onto = Ontology(mem)
    onto.add_concept("mycup", "我的杯子", parents=["cup"])
    onto.add_concept("mycup2", "旧杯子", parents=["cup"])
    onto.add_attribute("mycup2", "function", "decoration",
                       status="hypothesis")
    return mem, onto, Reasoner(mem)


def solve(mem, onto, rsn, text):
    frame = parse_sentence(text, source="user").to_dict()
    trace = solve_text(text, frame, mem, onto, rsn)
    return trace, explain(trace, style="short")


def test_1_task_representation_from_nl(tmp_path):
    mem, onto, rsn = fresh_db(tmp_path, "t1")
    try:
        frame = parse_sentence("计算7*6-8/4").to_dict()
        from cognitive_ai.cogtask import build_task
        t = build_task("计算7*6-8/4", frame, mem)
        assert t.task_type == "math" and t.is_actionable()
        assert t.goal["expression"] == "7*6-8/4"
        assert "compute" in t.required_operations
    finally:
        mem.close()


def test_2_composed_ops_and_unseen_math(tmp_path):
    mem, onto, rsn = fresh_db(tmp_path, "t2")
    try:
        # unseen expressions (not in any fixture): compute + verify composed
        tr, an = solve(mem, onto, rsn, "计算7*6-8/4")
        assert tr["task"]["status"] == "verified"
        assert an["text"] == "答案:40。" and an["certainty"] == "certain"
        tr, an = solve(mem, onto, rsn, "解方程3x-5=10")
        assert tr["task"]["status"] == "verified"
        assert "x=5" in an["text"]
        tr, an = solve(mem, onto, rsn, "求导x^3+2x")
        assert tr["task"]["status"] == "verified"
        assert "3*x^2" in an["text"]  # unseen composition, power rule
        ops = [s["op"] for s in tr["steps"]]
        assert "compute" in ops and "check_result" in ops
    finally:
        mem.close()


def test_3_trace_supports_answer(tmp_path):
    mem, onto, rsn = fresh_db(tmp_path, "t3")
    try:
        tr, an = solve(mem, onto, rsn, "这个杯子有什么用途")
        assert tr["task"]["status"] == "verified"
        assert an["citations"], "every claim must cite trace entries"
        assert all(c.startswith("[") for c in an["citations"])
        assert "drinking" in an["text"]
    finally:
        mem.close()


def test_4_missing_premise_refuses_conclusion(tmp_path):
    mem, onto, rsn = fresh_db(tmp_path, "t4")
    try:
        tr, an = solve(mem, onto, rsn, "如果下雨地面就会湿。")
        assert tr["task"]["status"] == "failed"
        assert an["certainty"] == "unknown"
        assert "缺失" in an["text"] or "前提" in an["text"]
    finally:
        mem.close()


def test_5_logic_success_and_conflict(tmp_path):
    mem, onto, rsn = fresh_db(tmp_path, "t5")
    try:
        tr, an = solve(mem, onto, rsn, "如果下雨地面就会湿。已知下雨。")
        assert tr["task"]["status"] == "verified", tr
        assert an["certainty"] == "certain" and "会湿" in an["text"]
        mem.add_claim("下雨存疑", subject_id=None, status="contested",
                      confidence=0.5)
        tr2, an2 = solve(mem, onto, rsn, "如果下雨地面就会湿。已知下雨。")
        assert tr2["task"]["status"] == "failed"
        assert an2["certainty"] == "unknown"
    finally:
        mem.close()


def test_6_bad_operation_detected(tmp_path):
    mem, onto, rsn = fresh_db(tmp_path, "t6")
    try:
        tr, an = solve(mem, onto, rsn, "计算2++3")
        assert tr["task"]["status"] == "failed"
        assert tr["failure_cause"] == "compute_failed"
        from cognitive_ai.operations import ApplyRule
        rec = ApplyRule().run(
            {"rule_id": "no_such_rule"},
            {"memory": mem, "ontology": onto, "reasoner": rsn})
        assert rec["status"] == "failed"
    finally:
        mem.close()


def test_7_hypothesis_not_fact(tmp_path):
    mem, onto, rsn = fresh_db(tmp_path, "t7")
    try:
        tr, an = solve(mem, onto, rsn, "碗的材质是什么")
        assert an["certainty"] == "qualified"
        assert "假设" in an["text"] or "未验证" in an["text"]
        assert "已验证" not in an["text"]
    finally:
        mem.close()


def test_8_failure_yields_concrete_gap(tmp_path):
    mem, onto, rsn = fresh_db(tmp_path, "t8")
    try:
        n0 = len(mem.list_questions())
        tr, _ = solve(mem, onto, rsn, "紫砂壶有什么功能")
        assert tr["task"]["status"] == "failed"
        assert tr["gap"] and tr["gap"]["question_id"]
        assert len(mem.list_questions()) == n0 + 1
        assert tr["failure_cause"] in ("missing_concept", "verify_failed")
    finally:
        mem.close()


def test_9_new_knowledge_fixes_and_regresses(tmp_path):
    mem, onto, rsn = fresh_db(tmp_path, "t9")
    try:
        # before learning the process verifies but the answer is unknown
        tr, an = solve(mem, onto, rsn, "杯子放在哪里")
        assert tr["task"]["status"] == "verified"
        assert an["certainty"] in ("unknown", "qualified")
        assert "kitchen" not in an["text"]
        from cognitive_ai.schema_evolver import SchemaEvolver
        from cognitive_ai.learner import default_regression
        ev = SchemaEvolver(mem, default_regression)
        r = ev.direct_apply("add_attribute",
                            {"concept": "cup", "key": "location",
                             "value": "kitchen", "confidence": 0.9,
                             "source": "validated", "status": "verified"},
                            reason="simulated validated learning")
        assert r["ok"]
        tr2, an2 = solve(mem, onto, rsn, "杯子放在哪里")
        assert tr2["task"]["status"] == "verified"
        assert "kitchen" in an2["text"] and an2["certainty"] == "certain"
        # old capability intact
        tr3, _ = solve(mem, onto, rsn, "计算3+4*2")
        assert tr3["task"]["status"] == "verified"
    finally:
        mem.close()


def test_10_inheritance_not_blind(tmp_path):
    mem, onto, rsn = fresh_db(tmp_path, "t10")
    try:
        _, an1 = solve(mem, onto, rsn, "我的杯子有什么功能")
        assert "drinking" in an1["text"] and an1["certainty"] == "qualified"
        assert "继承" in an1["text"]
        _, an2 = solve(mem, onto, rsn, "旧杯子有什么功能")
        assert "decoration" in an2["text"]  # instance override wins
        assert "drinking" not in an2["text"]
    finally:
        mem.close()


def test_11_restart_keeps_traces(tmp_path):
    db = str(tmp_path / "t11.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    onto, rsn = Ontology(mem), Reasoner(mem)
    solve(mem, onto, rsn, "计算3+4*2")
    rows = mem.query("SELECT task_id FROM cog_tasks")
    assert rows
    tid = rows[0]["task_id"]
    mem.close()
    mem2 = Memory(db)
    try:
        tr = load_trace(mem2, tid)
        assert tr and tr["task"]["status"] == "verified"
        assert any(s["op"] == "compute" for s in tr["steps"])
    finally:
        mem2.close()


def test_12_shared_flow_across_domains(tmp_path):
    mem, onto, rsn = fresh_db(tmp_path, "t12")
    try:
        traces = []
        for q in ("计算1+2", "杯子的材质是什么",
                  "如果下雨地面就会湿。已知下雨。"):
            frame = parse_sentence(q, source="user").to_dict()
            traces.append(solve_text(q, frame, mem, onto, rsn))
        types = {t["task"]["task_type"] for t in traces}
        assert types == {"math", "conceptual", "logic"}
        for tr in traces:
            for s in tr["steps"]:
                assert {"op", "status", "inputs", "outputs"} <= set(s), s
            assert tr["task"]["status"] == "verified"
    finally:
        mem.close()
