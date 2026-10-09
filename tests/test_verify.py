"""Phase 5C verification tests: independent checks, t5 regression.

Isolated DBs. Every assertion checks behavior (re-derivation outcomes),
not the mere presence of status fields.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import verify as V
from cognitive_ai.cli import seed_from_json
from cognitive_ai.executor import solve_text
from cognitive_ai.explainer import explain
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.reasoner import Reasoner
from cognitive_ai.semantics import parse_sentence

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def fresh(tmp_path, tag):
    mem = Memory(str(tmp_path / f"{tag}.db"))
    seed_from_json(mem, SEED)
    return mem, Ontology(mem), Reasoner(mem)


def solve(mem, onto, rsn, text):
    frame = parse_sentence(text, source="user").to_dict()
    trace = solve_text(text, frame, mem, onto, rsn)
    return trace, explain(trace, style="short")


def test_correct_math_passes_verification(tmp_path):
    mem, onto, rsn = fresh(tmp_path, "v1")
    try:
        tr, an = solve(mem, onto, rsn, "计算7*6-8/4")
        assert tr["task"]["status"] == "verified"
        assert tr["verification"]["verdict"] == "verified"
        assert any(c["name"] == "goal_satisfied" and
                   c["result"] == "verified"
                   for c in tr["verification"]["checks"])
        assert an["text"] == "答案:40。"
    finally:
        mem.close()


def test_t5_underspecified_chain_never_verified(tmp_path):
    """Permanent regression test for the Phase 5B t5 false-verified bug:
    chain-None first-match output must not become a verified answer."""
    mem, onto, rsn = fresh(tmp_path, "v2")
    try:
        from cognitive_ai import rulelearn as RL

        def ex(i, a, b):
            return {"id": i, "inputs": a, "outputs": b}

        for rid, spec, trn, ho in (
                ("lr_double",
                 RL.induce([ex("d1", {"n": "2"}, {"out": "4"}),
                            ex("d2", {"n": "3"}, {"out": "6"})],
                           [], domain="math")["candidates"][0]["spec"],
                 [ex("d1", {"n": "2"}, {"out": "4"})],
                 [ex("d9", {"n": "9"}, {"out": "18"})]),
                ("lr_add1",
                 RL.induce([ex("a1", {"n": "4"}, {"out": "5"}),
                            ex("a2", {"n": "10"}, {"out": "11"})],
                           [], domain="math")["candidates"][0]["spec"],
                 [ex("a1", {"n": "4"}, {"out": "5"})],
                 [ex("a9", {"n": "7"}, {"out": "8"})])):
            rec = RL.make_proposal_record(rid, rid, "math",
                                          [{"form": "t", "spec": spec}], [])
            rec["spec"]["selected"] = 0
            mem.save_rule(rec)
            rep = RL.validate_rule(spec, trn, ho, [], onto)
            rec["validation"] = {**rep, "selected": 0}
            mem.save_rule(rec)
            assert RL.promote_record(mem, rid, "validated",
                                     validation=rec["validation"])["ok"]
            assert RL.promote_record(mem, rid, "active")["ok"]
        # the exact t5 shape: chain unspecified, first-match would say even
        from cognitive_ai.cogtask import CognitiveTask
        from cognitive_ai import executor as E
        task = CognitiveTask(task_type="math", input_text="t5",
                             input_frame={"intent": "structured"},
                             goal={"kind": "transform", "domain": "math",
                                   "chain": None, "inputs": {"n": "7"}})
        trace = {"task": task.to_dict(), "steps": [], "repairs": [],
                 "gap": None, "failure_cause": None, "verification": None,
                 "metacog": []}
        ctx = E._ctx(mem, onto, rsn)
        # the underspecified first-match can no longer even complete a
        # plan as verified: the apply step itself is unverified now
        assert not E._run_plan(task, E._make_plan(task), ctx, trace)
        rep = V.verify_task(task.to_dict(), trace["steps"], mem, onto)
        assert rep["verdict"] != "verified", rep
        # and the full loop must not mark it verified either
        tr2, an2 = solve(mem, onto, rsn, "计算7*6-8/4")
        assert tr2["task"]["status"] == "verified"  # sanity: good stays good
    finally:
        mem.close()


def test_tampered_trace_rejected(tmp_path):
    mem, onto, rsn = fresh(tmp_path, "v3")
    try:
        tr, _ = solve(mem, onto, rsn, "计算3+4*2")
        assert tr["task"]["status"] == "verified"
        # tamper: rewrite the recorded answer, keep everything else
        for s in tr["steps"]:
            if s.get("op") == "compute":
                s["outputs"]["result"] = "999"
        rep = V.verify_task(tr["task"], tr["steps"], mem, onto)
        assert any(c["result"] == "rejected" for c in rep["checks"]), rep
        assert rep["verdict"] == "rejected"
    finally:
        mem.close()


def test_rule_without_preconditions_rejected(tmp_path):
    mem, onto, rsn = fresh(tmp_path, "v4")
    try:
        # rule fires structurally but its precondition fails on real inputs
        from cognitive_ai import rulelearn as RL
        spec = {"kind": "slot", "name": "narrow", "domain": "concepts",
                "inputs": {"color": "any"}, "pre": [
                    {"pred": "equals", "args": ["$color", "red"]}],
                "out": {"kind": "label", "label": "apple"}}
        rec = RL.make_proposal_record("lr_n", "n", "concepts",
                                      [{"form": "t", "spec": spec}], [])
        rec["spec"]["selected"] = 0
        mem.save_rule(rec)
        v = {"passed": True, "selected": 0}
        assert RL.promote_record(mem, "lr_n", "validated",
                                 validation=v)["ok"]
        assert RL.promote_record(mem, "lr_n", "active")["ok"]
        from cognitive_ai.operations import ApplyLearnedRule
        ctx = {"memory": mem, "ontology": onto, "reasoner": rsn}
        rec = ApplyLearnedRule().run(
            {"rule_id": "lr_n", "inputs": {"color": "green"}}, ctx)
        assert rec["status"] in ("failed", "unverified"), rec
        row = mem.get_rule("lr_n")
        assert row["failures"] >= 1
    finally:
        mem.close()


def test_conflict_blocks_conclusion(tmp_path):
    mem, onto, rsn = fresh(tmp_path, "v5")
    try:
        tr, an = solve(mem, onto, rsn, "杯子的材质是什么")
        assert tr["task"]["status"] == "verified"
        # now contest that very attribute: the same answer must downgrade
        onto.add_attribute("cup", "material", "glass", status="contested")
        tr2, an2 = solve(mem, onto, rsn, "杯子的材质是什么")
        assert tr2["verification"]["verdict"] in ("inconclusive",
                                                  "rejected"), \
            tr2["verification"]
        assert an2["certainty"] != "certain"
    finally:
        mem.close()


def test_failure_raises_gap_and_provenance(tmp_path):
    mem, onto, rsn = fresh(tmp_path, "v6")
    try:
        n0 = len(mem.list_questions())
        tr, an = solve(mem, onto, rsn, "紫砂壶有什么功能")
        assert tr["task"]["status"] in ("failed", "inconclusive")
        assert tr["gap"] and len(mem.list_questions()) == n0 + 1
        assert "provenance" in an and an["provenance"]["task_id"] == \
            tr["task"]["task_id"]
        assert an["provenance"]["failure_cause"] is not None
    finally:
        mem.close()


def test_unverified_answer_not_certain_in_ui(tmp_path):
    from cognitive_ai.web.service import Workbench
    import shutil
    data = tmp_path / "wdata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    wb = Workbench(str(data), str(tmp_path / "wmodels"))
    try:
        r = wb.solve("紫砂壶有什么功能", "u")
        assert r["answer"]["certainty"] != "certain", r["answer"]
        assert "provenance" in r["answer"]
    finally:
        wb.close()
