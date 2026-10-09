"""Unified cognitive execution loop (Phase 4C).

receive -> parse(done by caller) -> task -> retrieve -> plan -> execute
ops (records kept) -> verify -> replan|answer|gap. Answers are generated
ONLY from the trace (see explainer.py). Failures are classified and become
curiosity questions; neural weights are never touched here.
"""
from __future__ import annotations

from .cogtask import CognitiveTask, build_task
from .operations import REGISTRY
import re

FAILURE_CAUSES = ("unparseable", "missing_concept", "missing_rule",
                  "wrong_op", "compute_failed", "verify_failed",
                  "inexpressible")


def _ctx(memory, ontology, reasoner):
    return {"memory": memory, "ontology": ontology, "reasoner": reasoner,
            "working_memory": {}}


def _sync(task: CognitiveTask, trace: dict) -> None:
    # the trace keeps a task snapshot; refresh it so status/answer updates
    # are never shadowed by the creation-time copy
    trace["task"] = task.to_dict()


def solve_text(text: str, frame: dict, memory, ontology, reasoner,
               max_repairs: int = 1) -> dict:
    """Full loop for one input. Returns the persisted ExecutionTrace."""
    task = build_task(text, frame, memory)
    trace: dict = {"task": task.to_dict(), "steps": [],
                   "repairs": [], "gap": None, "failure_cause": None}
    if not task.is_actionable():
        trace["failure_cause"] = "unparseable"
        _sync(task, trace)
        _persist(memory, task, trace)
        return trace
    ctx = _ctx(memory, ontology, reasoner)
    plan = _make_plan(task)
    task.status = "running"
    for attempt in range(max_repairs + 1):
        ok = _run_plan(task, plan, ctx, trace)
        if ok:
            task.status = "verified"
            _sync(task, trace)
            _persist(memory, task, trace)
            return trace
        last = trace["steps"][-1] if trace["steps"] else {}
        cause = _classify_failure(task, last)
        trace["repairs"].append({"attempt": attempt, "failed_op":
                                 last.get("op"), "cause": cause})
        if attempt < max_repairs:
            alt = _alt_plan(task, plan, last.get("op"))
            if alt:
                plan = alt
                continue
        break
    task.status = "failed"
    trace["failure_cause"] = trace["repairs"][-1]["cause"] \
        if trace["repairs"] else "verify_failed"
    _raise_gap_for(task, trace, memory)
    _sync(task, trace)
    _persist(memory, task, trace)
    return trace


def _make_plan(task: CognitiveTask) -> list[dict]:
    if task.task_type == "math":
        g = task.goal
        if g.get("kind") == "transform":
            # composed learned-rule task (structured goals only; the NL
            # layer never invents chains)
            return [
                {"op": "apply_learned",
                 "inputs": {"domain": g.get("domain", "math"),
                            "chain": g.get("chain"),
                            "inputs": g.get("inputs", {})}},
                {"op": "check_result",
                 "inputs": {"check": {"kind": "nonempty",
                                      "value": "__applied__"}}},
            ]
        return [
            {"op": "compute",
             "inputs": {"kind": g["kind"], "expression": g["expression"],
                        "var": g.get("var", "x")}},
            {"op": "check_result", "inputs": {"from": "compute"}},
        ]
    if task.task_type == "logic":
        g = task.goal
        if g.get("kind") == "modus_ponens":
            cond = g.get("condition", "")
            have = g.get("premises", [])
            # exact match preferred; containment accepted and recorded
            # (下雨地面 contains 下雨) -- restricted scope, never silent:
            # the matched premise text stays in the trace basis
            hit, basis = [], []
            if cond and cond in have:
                hit, basis = [cond], [cond]
            elif cond:
                for h in have:
                    if h and (h in cond or cond in h):
                        hit, basis = [cond], [f"premise '{h}' covers '{cond}'"]
                        break
            return [
                {"op": "retrieve_concepts",
                 "inputs": {"terms": [cond, g["effect"]]}},
                {"op": "check_result",
                 "inputs": {"check": {
                     "kind": "no_contested", "subjects": [cond]}}},
                {"op": "check_result",
                 "inputs": {"check": {
                     "kind": "premises_present", "premises": hit,
                     "names": [cond], "match_basis": basis}}},
            ]
        return [{"op": "retrieve_concepts",
                 "inputs": {"terms": g.get("entities", [])}}]
    # conceptual
    plan = []
    ents = task.goal.get("entities", [])
    if ents:
        plan.append({"op": "retrieve_concepts", "inputs": {"terms": ents}})
        plan.append({"op": "query_attribute",
                     "inputs": {"concept": "__each__", "key": "__ask__"}})
        plan.append({"op": "apply_rule",
                     "inputs": {"rule_id": "*"}, "optional": True})
        plan.append({"op": "apply_learned",
                     "inputs": {"domain": "concepts", "from_attrs": True},
                     "optional": True})
    plan.append({"op": "check_result",
                 "inputs": {"check": {"kind": "kb_source",
                                      "sources": ents}}})
    return plan


def _alt_plan(task, plan, failed_op):
    # Only auxiliary ops may be dropped on repair. Core ops (compute /
    # query / check / rule application) define the task: dropping them
    # would "succeed" vacuously instead of reporting the real gap.
    if failed_op in ("compute", "query_attribute", "check_result",
                     "apply_rule", "apply_learned"):
        return None  # honest dead-end -> gap, never forced success
    return [step for step in plan if step.get("op") != failed_op] or None


def _run_plan(task, plan, ctx, trace) -> bool:
    computed = None
    for step in plan:
        op_cls = REGISTRY.get(step["op"])
        if op_cls is None:
            trace["steps"].append({"op": step["op"], "status": "unknown_op"})
            return False
        inputs = dict(step.get("inputs", {}))
        if step["op"] == "check_result" and \
                isinstance(inputs.get("check"), dict) and \
                inputs["check"].get("kind") == "kb_source" and \
                task.task_type == "conceptual":
            # resolve display texts (这个杯子/杯子的材质) to concept ids
            # before the source audit, or every question would fail it.
            # interrogatives and unresolvable terms are dropped here: truly
            # unknown concepts already fail the query step above.
            Skip = ("什么", "谁", "哪个", "多少", "为什么", "是否",
                    "吗", "呢", "")
            resolved = [cid for ent, (cid, _k) in
                        zip(task.goal.get("entities", []),
                            _resolve_entities(task, ctx))
                        if ent.strip() not in Skip and cid]
            inputs = {"check": {"kind": "kb_source", "sources": resolved}}
        if step["op"] == "query_attribute" and \
                inputs.get("concept") == "__each__":
            if not _run_attr_queries(task, ctx, trace):
                return False
            continue
        if step["op"] == "apply_learned" and inputs.get("from_attrs"):
            # feed retrieved attribute values as rule input slots
            slots: dict = {}
            for s in trace["steps"]:
                if s.get("op") == "query_attribute" and \
                        s.get("status") != "failed":
                    inp = s.get("inputs", {}) or {}
                    out = s.get("outputs", {}) or {}
                    if inp.get("concept"):
                        slots["concept"] = inp["concept"]
                    if inp.get("key") and out.get("value") is not None:
                        slots[inp["key"]] = out["value"]
            inputs = {"domain": inputs.get("domain"), "inputs": slots}
        if step["op"] == "check_result" and inputs.get("from") == "compute":
            if computed is None or computed["status"] != "verified":
                trace["steps"].append(
                    {"op": "check_result", "status": "failed",
                     "note": "nothing verified to check"})
                return False
            trace["steps"].append(
                {"op": "check_result", "status": "verified",
                 "note": "verified compute accepted", "inputs": inputs,
                 "outputs": {"ok": True}})
            continue
        if step["op"] == "check_result" and isinstance(
                inputs.get("check"), dict) and \
                inputs["check"].get("value") == "__applied__":
            # resolve against the preceding apply_learned record: a vacuous
            # placeholder must never pass the check by itself
            applied = []
            for s in reversed(trace["steps"]):
                if s.get("op") == "apply_learned" and \
                        s.get("status") == "verified":
                    applied = s.get("outputs", {}).get("applied", [])
                    break
            inputs = {"check": {"kind": "nonempty", "value": applied}}
        rec = op_cls().run(inputs, ctx)
        trace["steps"].append(rec)
        if step["op"] == "compute":
            computed = rec
        if rec["status"] in ("failed", "precondition_failed"):
            if step.get("optional"):
                # optional rule-application steps may miss without killing
                # the task; the miss itself stays in the trace
                continue
            return False
        if rec["status"] == "unverified" and step["op"] != "check_result":
            # unverified intermediate: keep going only for retrieval steps
            if step["op"] not in ("retrieve_concepts", "query_attribute",
                                  "apply_rule", "compare_explanations"):
                return False
    # task-level acceptance: math-compute needs a verified compute;
    # transform plans accept on verified checks; conceptual and logic
    # need their check_result steps verified
    finals = [s for s in trace["steps"] if s["op"] == "check_result"]
    if task.task_type == "math":
        if task.goal.get("kind") == "transform":
            return bool(finals) and all(
                s["status"] == "verified" for s in finals)
        return computed is not None and computed["status"] == "verified"
    if finals:
        return all(s["status"] == "verified" for s in finals)
    return any(s["status"] in ("verified", "unverified")
               for s in trace["steps"])


def _normalize_ref(term: str) -> str:
    # deictic determiners carry no KB identity: 这个杯子 -> 杯子.
    # Only stripped for LOOKUP; the original text stays in the trace.
    return re.sub(r"^(这个|那个|该|我家的|我的)+", "", term)


def _resolve_entities(task, ctx) -> list[tuple]:
    """Entities -> [(concept_id|None, key|None)].

    Handles possessives (杯子的材质 -> cup + material) via the same
    canonical key map the parser uses. Pure interrogatives (什么/谁/…)
    resolve to (None, None) and are SKIPPED by callers, never queried.
    Returns one row per entity.
    """
    from .semantics import ATTR_WORDS
    rows = []
    for ent in task.goal.get("entities", []):
        if ent.strip() in ("什么", "谁", "哪个", "多少", "为什么", "是否",
                           "吗", "呢", ""):
            rows.append((None, None))
            continue
        norm = _normalize_ref(ent)
        cid, key = None, None
        # whole-term match first: 我的杯子 is a real instance name, and
        # stripping 我的 blindly would misroute it to cup
        for cand in (ent, norm):
            hit = ctx["memory"].get_concept(cand) or next(
                iter(ctx["memory"].find_concepts_by_name(cand)), None)
            if hit:
                cid = hit["id"]
                break
        if cid is None and "的" in norm:
            left, right = norm.rsplit("的", 1)
            hit = ctx["memory"].get_concept(left) or next(
                iter(ctx["memory"].find_concepts_by_name(left)), None)
            if hit and right:
                cid = hit["id"]
                key = ATTR_WORDS.get(right, right)
        rows.append((cid, key))
    return rows


def _run_attr_queries(task, ctx, trace) -> bool:
    from .operations import QueryAttribute
    text = task.input_text
    key = _wanted_key(text)
    ok_any = False
    for ent, (cid, ekey) in zip(task.goal.get("entities", []),
                                _resolve_entities(task, ctx)):
        if ent.strip() in ("什么", "谁", "哪个", "多少", "为什么", "是否",
                           "吗", "呢", ""):
            continue  # interrogative word, not a knowledge term
        if not cid:
            trace["steps"].append(
                {"op": "query_attribute", "status": "failed",
                 "note": f"unknown concept {ent}", "inputs": {"concept": ent}})
            return False
        for k in ([ekey or key] if (ekey or key) else ["function", "material"]):
            rec = QueryAttribute().run({"concept": cid, "key": k}, ctx)
            trace["steps"].append(rec)
            if rec["status"] != "failed":
                ok_any = True
    return ok_any


def _wanted_key(text: str):
    if "用途" in text or "用处" in text or "功能" in text or "干什么" in text:
        return "function"
    if "材质" in text or "材料" in text or "什么做" in text:
        return "material"
    if "颜色" in text:
        return "color"
    if "位置" in text or "在哪" in text:
        return "location"
    return None


def _classify_failure(task, last: dict) -> str:
    op = (last or {}).get("op", "")
    note = (last or {}).get("note", "")
    if op == "compute":
        return "compute_failed"
    if op == "check_result":
        return "verify_failed"
    if op in ("retrieve_concepts", "query_attribute") and \
            ("unknown" in note or "no concept" in note.lower()
             or "unknown concept" in note):
        return "missing_concept"
    if op == "apply_rule":
        return "missing_rule"
    if op == "unknown_op":
        return "wrong_op"
    return "verify_failed"


def _raise_gap_for(task, trace, memory) -> None:
    cause = trace.get("failure_cause", "verify_failed")
    questions = {
        "unparseable": f"如何理解输入:{task.input_text}",
        "missing_concept": f"缺失概念知识:{task.input_text}",
        "missing_rule": f"缺失可执行规则:{task.input_text}",
        "wrong_op": f"操作选择错误:{task.input_text}",
        "compute_failed": f"内部计算失败:{task.input_text}",
        "verify_failed": f"结果验证失败:{task.input_text}",
        "inexpressible": f"无法表达答案:{task.input_text}",
    }
    rec = {"op": "raise_gap", "status": "verified",
           "inputs": {"question": questions.get(cause, task.input_text),
                      "failure": cause}}
    trace["steps"].append(rec)
    from .operations import RaiseGap
    out = RaiseGap().run({"question": rec["inputs"]["question"],
                          "kind": "gap", "failure": cause},
                         {"memory": memory})
    trace["gap"] = out["outputs"]


def _persist(memory, task: CognitiveTask, trace: dict) -> None:
    import json
    from datetime import datetime, timezone
    task.answer = {"status": task.status}
    memory.execute(
        "INSERT INTO cog_tasks(task_id,task_type,input_text,trace,status,"
        "created_at) VALUES(?,?,?,?,?,?)"
        " ON CONFLICT(task_id) DO UPDATE SET trace=excluded.trace,"
        " status=excluded.status",
        (task.task_id, task.task_type, task.input_text,
         json.dumps({"task": task.to_dict(), **trace}, ensure_ascii=False,
                    default=str),
         task.status, datetime.now(timezone.utc).isoformat()))


def load_trace(memory, task_id: str) -> dict | None:
    import json
    row = memory.query_one("SELECT * FROM cog_tasks WHERE task_id=?",
                           (task_id,))
    if not row:
        return None
    return json.loads(row["trace"])
