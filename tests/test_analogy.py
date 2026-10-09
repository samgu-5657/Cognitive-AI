"""Phase 5B analogy tests: structure matching, transfer flow, A/B/C.

Isolated DBs. Memory cases are REAL verified executor traces (never the
test's own rows: exclude lists + disjoint inputs enforced).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import analogy as A
from cognitive_ai import rulelearn as RL
from cognitive_ai.cli import seed_from_json
from cognitive_ai.executor import solve_task_with_analogy
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.reasoner import Reasoner

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def ex(i, inputs, outputs):
    return {"id": i, "inputs": inputs, "outputs": outputs}


def setup_world(tmp_path, tag):
    """Seed KB + three accepted rules (double/add1/parity)."""
    mem = Memory(str(tmp_path / f"{tag}.db"))
    seed_from_json(mem, SEED)
    onto, rsn = Ontology(mem), Reasoner(mem)
    dbl = RL.induce([ex("d1", {"n": "2"}, {"out": "4"}),
                     ex("d2", {"n": "3"}, {"out": "6"}),
                     ex("d3", {"n": "5"}, {"out": "10"})],
                    [], domain="math")["candidates"][0]["spec"]
    add = RL.induce([ex("a1", {"n": "4"}, {"out": "5"}),
                     ex("a2", {"n": "10"}, {"out": "11"}),
                     ex("a3", {"n": "0"}, {"out": "1"})],
                    [], domain="math")["candidates"][0]["spec"]
    par = RL.induce([ex("p1", {"n": "2"}, {"out": "even"}),
                     ex("p2", {"n": "4"}, {"out": "even"}),
                     ex("p3", {"n": "6"}, {"out": "even"}),
                     ex("p4", {"n": "1"}, {"out": "odd"}),
                     ex("p5", {"n": "3"}, {"out": "odd"})],
                    [], domain="math")["candidates"][0]["spec"]
    for rid, spec, tr, ho in (
            ("lr_double", dbl,
             [ex("d1", {"n": "2"}, {"out": "4"})],
             [ex("d9", {"n": "9"}, {"out": "18"})]),
            ("lr_add1", add,
             [ex("a1", {"n": "4"}, {"out": "5"})],
             [ex("a9", {"n": "7"}, {"out": "8"})]),
            ("lr_parity", par,
             [ex("p1", {"n": "2"}, {"out": "even"})],
             [ex("p9", {"n": "12"}, {"out": "even"})])):
        rec = RL.make_proposal_record(rid, rid, "math",
                                      [{"form": "t", "spec": spec}], [])
        rec["spec"]["selected"] = 0
        mem.save_rule(rec)
        rep = RL.validate_rule(spec, tr, ho, [], onto)
        assert rep["passed"], (rid, rep)
        rec["validation"] = {**rep, "selected": 0}
        mem.save_rule(rec)
        assert RL.promote_record(mem, rid, "validated",
                                 validation=rec["validation"])["ok"]
        assert RL.promote_record(mem, rid, "active")["ok"]
    return mem, onto, rsn


def struct_task(chain, inputs, text):
    from cognitive_ai.cogtask import CognitiveTask
    return CognitiveTask(task_type="math", input_text=text,
                         input_frame={"intent": "structured"},
                         goal={"kind": "transform", "domain": "math",
                               "chain": chain, "inputs": inputs})


def memorize(mem, onto, rsn, chain, inputs, text):
    """Run a verified structured task into memory (a retrievable case).

    Uses the DIRECT executor path (no analogy): bootstrapping cases
    cannot be retrieved before they exist.
    """
    from cognitive_ai import executor as E
    from cognitive_ai.cogtask import CognitiveTask
    task = CognitiveTask(task_type="math", input_text=text,
                         input_frame={"intent": "structured"},
                         goal={"kind": "transform", "domain": "math",
                               "chain": chain, "inputs": inputs})
    trace = {"task": task.to_dict(), "steps": [], "repairs": [],
             "gap": None, "failure_cause": None}
    ctx = E._ctx(mem, onto, rsn)
    assert E._run_plan(task, E._make_plan(task), ctx, trace), trace
    task.status = "verified"
    E._sync(task, trace)
    E._persist(mem, task, trace)
    return task.task_id


def memory_cases(mem):
    import json
    return [A.case_from_trace(json.loads(r["trace"]))
            for r in mem.query("SELECT trace FROM cog_tasks "
                              "WHERE status='verified'")]


def test_structural_match_beats_surface():
    cur = {"task_id": "c", "task_type": "math", "goal_kind": "transform",
           "op_chain": ["apply_learned", "check_result"],
           "rule_chain": ["double", "add1"], "slots": {"rule": "x"},
           "input_text": "double seven then add one", "verified": False}
    same = {"task_id": "m1", "task_type": "math", "goal_kind": "transform",
            "op_chain": ["apply_learned", "check_result"],
            "rule_chain": ["double", "add1"], "slots": {"rule": "x"},
            "input_text": "totally different words here", "verified": True}
    diff = {"task_id": "m2", "task_type": "math", "goal_kind": "arithmetic",
            "op_chain": ["compute", "check_result"],
            "rule_chain": [], "slots": {"expression": "x"},
            "input_text": "double seven then add one", "verified": True}
    ranked = A.retrieve(cur, [same, diff])
    assert ranked[0]["candidate"]["task_id"] == "m1"
    assert ranked[0]["structural"]["total"] > ranked[0]["surface"]
    assert ranked[1]["surface"] > ranked[1]["structural"]["total"]
    assert ranked[1]["accepted"] is False


def test_A_renamed_surface_same_structure(tmp_path):
    mem, onto, rsn = setup_world(tmp_path, "tA")
    try:
        src = memorize(mem, onto, rsn, ["lr_double", "lr_add1"],
                       {"n": "6"}, "double six then add one")
        tr = solve_task_with_analogy(
            struct_task(["lr_double", "lr_add1"], {"n": "7"},
                        "seven doubled with one added"),
            mem, onto, rsn,
            step_inputs={"apply_learned": {
                "domain": "math", "chain": ["lr_double", "lr_add1"],
                "inputs": {"n": "7"}}})
        assert tr["analogy"]["outcome"] == "verified", tr["analogy"]
        assert tr["analogy"]["source_task_id"] == src
        assert tr["analogy"]["surface"] < \
            tr["analogy"]["structural"]["total"]
        final = [s for s in tr["steps"]
                 if s.get("op") == "apply_learned"]
        assert final and final[0]["outputs"]["final"].get("_prev") == "15"
    finally:
        mem.close()


def test_B_distractor_picks_right_source(tmp_path):
    mem, onto, rsn = setup_world(tmp_path, "tB")
    try:
        memorize(mem, onto, rsn, ["lr_double", "lr_add1"], {"n": "6"},
                 "double six then add one")
        par = memorize(mem, onto, rsn, ["lr_parity"], {"n": "12"},
                       "parity of twelve please double check")
        tr = solve_task_with_analogy(
            struct_task(["lr_parity"], {"n": "14"},
                        "double check parity of fourteen"),
            mem, onto, rsn,
            step_inputs={"apply_learned": {
                "domain": "math", "chain": ["lr_parity"],
                "inputs": {"n": "14"}}})
        # structural winner must be the parity case despite word overlap
        assert tr["analogy"]["source_task_id"] == par, tr["analogy"]
        assert tr["analogy"]["outcome"] == "verified"
    finally:
        mem.close()


def test_C_missing_precondition_refuses(tmp_path):
    mem, onto, rsn = setup_world(tmp_path, "tC")
    try:
        src = memorize(mem, onto, rsn, ["lr_double", "lr_add1"],
                       {"n": "6"}, "double six then add one")
        tr = solve_task_with_analogy(
            struct_task(["lr_double", "lr_add1"], {},
                        "double then add one"),
            mem, onto, rsn,
            step_inputs={"apply_learned": {
                "domain": "math", "chain": ["lr_double", "lr_add1"],
                "inputs": {}}})
        assert tr["analogy"]["outcome"] == "refused"
        assert "apply_learned.inputs" in str(tr["analogy"]["reason"])
        assert tr["task"]["status"] == "failed"
        assert tr["gap"] is not None
    finally:
        mem.close()


def test_D_compose_two_cases(tmp_path):
    mem, onto, rsn = setup_world(tmp_path, "tD")
    try:
        memorize(mem, onto, rsn, ["lr_double"], {"n": "9"}, "double nine")
        memorize(mem, onto, rsn, ["lr_add1"], {"n": "10"}, "add one to ten")
        # goal chain never demonstrated as a whole; compose from parts
        tr = solve_task_with_analogy(
            struct_task(["lr_double", "lr_add1"], {"n": "6"},
                        "double six then add one"),
            mem, onto, rsn,
            step_inputs={"apply_learned": {
                "domain": "math",
                "chain": ["lr_double", "lr_add1"], "inputs": {"n": "6"}}})
        assert tr["analogy"]["outcome"] == "verified", tr["analogy"]
        assert len(tr["analogy"].get("composed_sources", []) or []) == 2
        final = [s for s in tr["steps"]
                 if s.get("op") == "apply_learned"][0]
        assert final["outputs"]["final"].get("_prev") == "13", final
    finally:
        mem.close()


def test_E_conflict_tie_refuses_then_feedback_resolves(tmp_path):
    mem, onto, rsn = setup_world(tmp_path, "tE")
    try:
        memorize(mem, onto, rsn, ["lr_add1"], {"n": "10"}, "process five")
        memorize(mem, onto, rsn, ["lr_double"], {"n": "9"}, "process five")
        tr = solve_task_with_analogy(
            struct_task(None, {"n": "5"}, "process five"), mem, onto, rsn,
            step_inputs={"apply_learned": {
                "domain": "math", "chain": None, "inputs": {"n": "5"}}})
        assert tr["analogy"]["outcome"] == "refused"
        assert "tie" in tr["analogy"]["reason"]
        cands = tr["analogy"]["candidates"]
        mem.add_feedback(tr["task"]["task_id"], cands[0],
                         {"candidate_task_id": cands[0], "effect": "boost"})
        tr2 = solve_task_with_analogy(
            struct_task(None, {"n": "5"}, "process five"), mem, onto, rsn,
            step_inputs={"apply_learned": {
                "domain": "math", "chain": None, "inputs": {"n": "5"}}})
        assert tr2["analogy"]["outcome"] == "verified", tr2["analogy"]
    finally:
        mem.close()


def test_F_no_memory_refuses_cleanly(tmp_path):
    mem = Memory(str(tmp_path / "tF.db"))
    seed_from_json(mem, SEED)
    onto, rsn = Ontology(mem), Reasoner(mem)
    try:
        tr = solve_task_with_analogy(
            struct_task(["lr_double"], {"n": "5"}, "double five"),
            mem, onto, rsn)
        assert tr["analogy"]["outcome"] == "refused"
        assert tr["gap"] is not None
        assert tr["task"]["status"] == "failed"
    finally:
        mem.close()
