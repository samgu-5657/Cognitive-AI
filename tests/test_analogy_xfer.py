"""A/B/C transfer experiment (Phase 5B, Task 6).

Same KB, same frozen tasks, same budget in three isolated worlds:
  A: plain executor, no analogy.
  B: surface (token-overlap) retrieval + unchecked transplant.
  C: structural analogy + role mapping + precondition checks + verify.

Frozen tasks (inputs disjoint from memory cases):
  t1 chain[double,add1] n=7   -> 15
  t2 chain[parity] n=14       -> even   (text lures keyword retrieval)
  t3 chain[retired_x] n=5     -> must refuse (unknown rule, no cover)
  t4 chain[double,add1] {}    -> must refuse with missing-precondition reason

C's specified behavior is asserted; B is characterized (its traces
structurally lack precondition checks); A is the no-analogy control.
Old-task regression checked in every world.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import analogy as A
from cognitive_ai import rulelearn as RL
from cognitive_ai.cli import seed_from_json
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.reasoner import Reasoner

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def ex(i, inputs, outputs):
    return {"id": i, "inputs": inputs, "outputs": outputs}


def build_world(tmp_path, tag):
    mem = Memory(str(tmp_path / f"x_{tag}.db"))
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
        rec["validation"] = {**rep, "selected": 0}
        mem.save_rule(rec)
        assert RL.promote_record(mem, rid, "validated",
                                 validation=rec["validation"])["ok"]
        assert RL.promote_record(mem, rid, "active")["ok"]
    return mem, onto, rsn


def memorize(mem, onto, rsn, chain, inputs, text):
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


TASKS = [
    {"id": "t1", "chain": ["lr_double", "lr_add1"], "inputs": {"n": "7"},
     "text": "seven doubled with one added", "want": "15"},
    {"id": "t2", "chain": ["lr_parity"], "inputs": {"n": "14"},
     "text": "double check parity of fourteen", "want": "even"},
    {"id": "t3", "chain": ["retired_x"], "inputs": {"n": "5"},
     "text": "retired doubling of five", "want": None},
    {"id": "t4", "chain": ["lr_double", "lr_add1"], "inputs": {},
     "text": "double then add one", "want": None},
    # t5: chain unspecified. Direct execution first-matches (hazard);
    # analogy must refuse the three-way tie instead of guessing.
    {"id": "t5", "chain": None, "inputs": {"n": "7"},
     "text": "double seven then add one", "want": None},
]


def run_A(mem, onto, rsn, t):
    from cognitive_ai import executor as E
    from cognitive_ai.cogtask import CognitiveTask
    task = CognitiveTask(task_type="math", input_text=t["text"],
                         input_frame={"intent": "structured"},
                         goal={"kind": "transform", "domain": "math",
                               "chain": t["chain"], "inputs": t["inputs"]})
    trace = {"task": task.to_dict(), "steps": [], "repairs": [],
             "gap": None, "failure_cause": None}
    ok = E._run_plan(task, E._make_plan(task), E._ctx(mem, onto, rsn),
                     trace)
    task.status = "verified" if ok else "failed"
    ans = None
    for s in trace["steps"]:
        if s.get("op") == "apply_learned" and s.get("status") == "verified":
            outs = s["outputs"].get("applied", [])
            if outs:
                o = outs[-1]["output"]
                ans = o if isinstance(o, str) else o.get("label")
    return {"ok": ok, "trace": trace, "answer": ans}


def run_B(mem, onto, rsn, t):
    """Naive baseline: surface retrieval + unchecked transplant."""
    import json
    from cognitive_ai.operations import REGISTRY
    cases = [A.case_from_trace(json.loads(r["trace"]))
             for r in mem.query("SELECT trace FROM cog_tasks "
                               "WHERE status='verified'")]
    scored = sorted(((A.surface_score(t["text"], c["input_text"]), c)
                     for c in cases), key=lambda x: -x[0])
    trace = {"task": {"task_id": t["id"]}, "steps": [], "outcome": None}
    if not scored:
        return {"ok": False, "trace": trace, "answer": None}
    top = scored[0][1]
    from cognitive_ai import executor as E
    src = E.load_trace(mem, top["task_id"])
    plan = A.transplant_plan(src, {})
    ctx = E._ctx(mem, onto, rsn)
    answer = None
    # transplant the SOURCE's rule chain (not every active rule)
    chain = []
    for s in src.get("steps", []):
        if s.get("op") == "apply_learned":
            for app in (s.get("outputs", {}) or {}).get("applied", []):
                if app.get("rule_id") not in chain:
                    chain.append(app["rule_id"])
    for step in plan:
        op_cls = REGISTRY.get(step["op"])
        if op_cls is None:
            continue
        if step["op"] == "apply_learned":
            rec = op_cls().run(
                {"domain": "math", "chain": chain or None,
                 "inputs": dict(t["inputs"])}, ctx)
        elif step["op"] == "check_result":
            continue  # B skips verification
        else:
            rec = op_cls().run(step.get("inputs", {}), ctx)
        trace["steps"].append(rec)
        if step["op"] == "apply_learned" and rec["status"] == "verified":
            outs = rec["outputs"].get("applied", [])
            if outs:
                answer = outs[-1]["output"] \
                    if isinstance(outs[-1].get("output"), str) \
                    else outs[-1]["output"].get("label")
    return {"ok": answer is not None, "trace": trace, "answer": answer,
            "source": top["task_id"], "surface": scored[0][0]}


def run_C(mem, onto, rsn, t):
    from cognitive_ai import executor as E
    from cognitive_ai.cogtask import CognitiveTask
    task = CognitiveTask(task_type="math", input_text=t["text"],
                         input_frame={"intent": "structured"},
                         goal={"kind": "transform", "domain": "math",
                               "chain": t["chain"], "inputs": t["inputs"]})
    return E.solve_task_with_analogy(
        task, mem, onto, rsn,
        step_inputs={"apply_learned": {
            "domain": "math", "chain": t["chain"],
            "inputs": dict(t["inputs"])}})


def test_abc_experiment(tmp_path):
    worlds = {}
    for arm in ("A", "B", "C"):
        mem, onto, rsn = build_world(tmp_path, arm)
        memorize(mem, onto, rsn, ["lr_double", "lr_add1"], {"n": "6"},
                 "double check addition")
        memorize(mem, onto, rsn, ["lr_parity"], {"n": "12"},
                 "parity of twelve")
        memorize(mem, onto, rsn, ["lr_add1"], {"n": "10"}, "add one to ten")
        worlds[arm] = (mem, onto, rsn)
    report = {}
    for arm, (mem, onto, rsn) in worlds.items():
        rows = []
        for t in TASKS:
            if arm == "A":
                r = run_A(mem, onto, rsn, t)
                ans, ok = r["answer"], r["ok"]
            elif arm == "B":
                r = run_B(mem, onto, rsn, t)
                ans, ok = r["answer"], r["ok"]
            else:
                tr = run_C(mem, onto, rsn, t)
                ans = None
                if tr["analogy"] and tr["analogy"].get("outcome") == \
                        "verified":
                    for s in tr["steps"]:
                        if s.get("op") == "apply_learned" and \
                                s.get("status") == "verified":
                            outs = s["outputs"]["applied"]
                            o = outs[-1]["output"]
                            ans = o if isinstance(o, str) else o.get("label")
                ok = tr["task"]["status"] == "verified"
                r = {"trace": tr, "answer": ans, "ok": ok}
            rows.append({"id": t["id"], "ok": ok, "answer": ans,
                         "want": t["want"],
                         "correct": (ans == t["want"]) if t["want"] else None})
        # old-task regression in the same world
        from cognitive_ai.executor import solve_text
        from cognitive_ai.semantics import parse_sentence
        reg_ok = True
        for q in ("计算3+4*2", "杯子的材质是什么"):
            fr = parse_sentence(q, source="user").to_dict()
            if solve_text(q, fr, mem, onto, rsn)["task"]["status"] != \
                    "verified":
                reg_ok = False
        report[arm] = {"rows": rows, "regression_ok": reg_ok,
                       "score": sum(1 for r in rows if r["correct"])}
        mem.close()
    print("\nANALOGY A/B/C (5 frozen tasks + old regression):")
    for arm in ("A", "B", "C"):
        r = report[arm]
        print(f"  {arm}: score={r['score']}/2 regression={r['regression_ok']}")
        for row in r["rows"]:
            print(f"    {row['id']}: ok={row['ok']} answer={row['answer']} "
                  f"want={row['want']}")
    # C's specified behavior on each task
    assert report["C"]["rows"][0]["correct"] is True   # t1 transfer
    assert report["C"]["rows"][1]["correct"] is True   # t2 right source
    assert report["C"]["rows"][2]["correct"] is None   # t3 must refuse
    assert report["C"]["rows"][2]["ok"] is False
    assert report["C"]["rows"][3]["ok"] is False       # t4 missing input
    # t5: chain unspecified. A first-matches a wrong-but-verified answer
    # (documented hazard of chain-less goals); C must refuse the tie.
    assert report["C"]["rows"][4]["ok"] is False
    t5a = [r for r in report["A"]["rows"] if r["id"] == "t5"][0]
    assert t5a["ok"] is True and t5a["answer"] != "15", t5a
    assert report["A"]["regression_ok"] and report["C"]["regression_ok"]
    # B must exhibit at least one wrong transfer (honest naive baseline)
    assert report["B"]["score"] <= 1, report["B"]
