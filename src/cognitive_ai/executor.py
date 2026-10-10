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
                   "repairs": [], "gap": None, "failure_cause": None,
                   "verification": None, "metacog": []}
    if not task.is_actionable():
        trace["failure_cause"] = "unparseable"
        _sync(task, trace)
        _persist(memory, task, trace)
        return trace
    from . import metacog, verify
    ctx = _ctx(memory, ontology, reasoner)
    plan = _make_plan(task)
    task.status = "running"
    budget = max_repairs + 1
    attempts = 0
    verification = None
    while True:
        ok = _run_plan(task, plan, ctx, trace)
        attempts += 1
        if ok:
            verification = verify.verify_task(
                task.to_dict(), trace["steps"], memory, ontology)
            trace["verification"] = verification
            if verification["verdict"] == "verified":
                task.status = "verified"
                _finalize(memory, task, trace)
                return trace
            if verification["verdict"] == "inconclusive":
                task.status = "inconclusive"
                trace["failure_cause"] = "verify_failed"
                _raise_gap_for(task, trace, memory)
                _finalize(memory, task, trace)
                return trace
            ok = False  # rejected: fall through to failure handling
        last = trace["steps"][-1] if trace["steps"] else {}
        cause = _classify_failure(task, last)
        if verification is not None and \
                verification.get("verdict") == "rejected":
            cause = "verify_failed"
        decision = metacog.decide(task.to_dict(), trace["steps"],
                                  verification, memory, attempts, budget)
        trace["metacog"].append({**decision, "cause": cause,
                                 "attempt": attempts})
        if verification is not None and \
                verification.get("verdict") == "rejected":
            cause = "verify_failed"
        trace["repairs"].append({"attempt": attempts, "failed_op":
                                 last.get("op"), "cause": cause,
                                 "decision": decision["action"]})
        if decision["action"] == "switch_rule":
            alt = _alt_plan(task, plan, last.get("op"))
            if alt and attempts < budget:
                plan = alt
                verification = None
                continue
        if decision["action"] == "try_analogy":
            tried = _solve_via_analogy_once(task, trace, ctx, memory,
                                            ontology, reasoner)
            if tried == "verified":
                task.status = "verified"
                _finalize(memory, task, trace)
                return trace
            attempts += 1
            verification = None
            if attempts < budget:
                continue
        break
    if task.status != "inconclusive":
        task.status = "failed"
    if not trace["failure_cause"]:
        trace["failure_cause"] = trace["repairs"][-1]["cause"] \
            if trace["repairs"] else "verify_failed"
    if trace["gap"] is None:
        _raise_gap_for(task, trace, memory)
    _finalize(memory, task, trace)
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
    if (task.goal or {}).get("ask_about") == "reason":
        # causal-reason questions have no attribute to look up: refuse
        # honestly instead of guessing an unrelated attribute (e.g.
        # function) and presenting it as the "reason".
        trace["steps"].append(
            {"op": "query_attribute", "status": "failed",
             "note": "reason explanation unsupported; no causal model",
             "inputs": {"concept": (task.goal.get("entities") or [None])[0]}})
        return False
    text = task.input_text
    key = (task.goal or {}).get("ask_key") or _wanted_key(text)
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
    if "用途" in text or "用处" in text or "功能" in text or "干什么" in text \
            or "拿来" in text or "做什么" in text:
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
                      "failure": cause},
           "basis": [], "outputs": {}}
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


def solve_with_analogy(text: str, frame: dict, memory, ontology, reasoner,
                       exclude_task_ids: list | None = None,
                       step_inputs: dict | None = None) -> dict:
    """Analogy-driven solving on top of the standard machinery.

    1. Build the task + a dry plan skeleton for its structural case.
    2. Retrieve verified memory cases by STRUCTURE (surface reported only).
    3. Multiple supporters -> compare preconditions/evidence; ties refuse.
    4. Transplant the winner's plan, execute with current inputs, verify.
    5. Persist everything (mapping, checks, outcome) or refuse with a gap.
    Default executor plans are untouched; this is the opt-in analogy path.
    """
    from . import analogy as A
    from .operations import Analogize
    task = build_task(text, frame, memory)
    return solve_task_with_analogy(task, memory, ontology, reasoner,
                                   exclude_task_ids, step_inputs)


def solve_task_with_analogy(task: CognitiveTask, memory, ontology,
                            reasoner, exclude_task_ids=None,
                            step_inputs=None) -> dict:
    """Analogy core over a ready-made task (structured goals welcome)."""
    from . import analogy as A
    from .operations import Analogize
    trace: dict = {"task": task.to_dict(), "steps": [],
                   "repairs": [], "gap": None, "failure_cause": None,
                   "analogy": None}
    if not task.is_actionable():
        trace["failure_cause"] = "unparseable"
        _sync(task, trace)
        _persist(memory, task, trace)
        return trace
    ctx = _ctx(memory, ontology, reasoner)
    dry = _make_plan(task)
    cur = {"task_id": task.task_id, "task_type": task.task_type,
           "goal_kind": (task.goal or {}).get("kind", ""),
           "op_chain": [s.get("op") for s in dry],
           "rule_chain": list((task.goal or {}).get("chain") or []),
           "slots": {k: "any" for k in _task_slots(task)},
           "input_text": task.input_text, "verified": False}
    rec = Analogize().run({"current_case": cur,
                           "exclude_task_ids": exclude_task_ids or []}, ctx)
    trace["steps"].append(rec)
    ranked = [r for r in rec["outputs"].get("ranked", []) if r["accepted"]]
    if not ranked:
        trace["failure_cause"] = "missing_rule"
        trace["analogy"] = {"outcome": "refused",
                            "reason": "no structurally related case",
                            "ranked": rec["outputs"].get("ranked", [])}
        _raise_gap_for(task, trace, memory)
        task.status = "failed"
        _sync(task, trace)
        _persist(memory, task, trace)
        return trace
    # exact single cover: one candidate's rule chain IS the goal chain
    top = ranked[0]
    goal_chain = list((task.goal or {}).get("chain") or [])
    exact = [r for r in ranked
             if r["candidate"].get("rule_chain", []) == goal_chain
             and goal_chain]
    if len(exact) == 1:
        return _transfer_rules(task, trace, ctx, memory,
                               exact[0]["candidate"]["task_id"],
                               exact[0]["candidate"].get("rule_chain", []),
                               top, step_inputs)
    if len(exact) > 1:
        return _refuse(task, trace, memory, "duplicate covering cases",
                       [r["candidate"]["task_id"] for r in exact])
    # composed cover from several accepted cases (novel combination)
    composed = compose_for_goal(ranked, goal_chain)
    if composed:
        return _transfer_rules(task, trace, ctx, memory,
                               composed["sources"][0],
                               composed["rules"], top, step_inputs,
                               composed_sources=composed["sources"])
    # conflicting supporters: compare evidence; unbreakable ties refuse
    rivals = [r for r in ranked[1:]
              if abs(r["score"] - top["score"]) < 0.05
              and _plan_differs(r, top)]
    if rivals:
        return _refuse(
            task, trace, memory, "conflicting supporters; tie kept",
            [top["candidate"]["task_id"]] +
            [r["candidate"]["task_id"] for r in rivals])
    if goal_chain:
        # explicit chain, no full cover: partial application refused
        return _refuse(task, trace, memory,
                       "no verified case covers the goal chain",
                       [r["candidate"]["task_id"] for r in ranked[:3]])
    # no explicit chain: transfer the winner's own solution
    winner_chain = top["candidate"].get("rule_chain", [])
    if winner_chain and task.task_type == "math":
        return _transfer_rules(task, trace, ctx, memory,
                               top["candidate"]["task_id"], winner_chain,
                               top, step_inputs)
    src_id = top["candidate"]["task_id"]
    src_trace = load_trace(memory, src_id)
    plan = A.transplant_plan(src_trace, top["mapping"])
    auto_inputs = _inputs_for(task, plan)
    if step_inputs:
        # caller-supplied current inputs win (e.g. resolved concepts)
        auto_inputs = {**auto_inputs, **step_inputs}
    trec = A.run_transfer(task, plan, auto_inputs, ctx, src_id,
                          top["mapping"])
    trace["analogy"] = trec
    trace["steps"].extend(trec["steps"])
    if trec["outcome"] == "verified":
        task.status = "verified"
    else:
        task.status = "failed"
        trace["failure_cause"] = "verify_failed"
        _raise_gap_for(task, trace, memory)
    memory.add_experience(
        "evaluation", f"analogy {trec['outcome']}: {src_id} -> "
        f"{task.task_id}",
        {"source_task": src_id, "mapping": top["mapping"],
         "structural": top["structural"], "surface": top["surface"]})
    _sync(task, trace)
    _persist(memory, task, trace)
    return trace


def _task_slots(task: CognitiveTask) -> list[str]:
    slots = []
    for ent in task.goal.get("entities", []):
        slots.append("entity")
    g = task.goal
    if g.get("kind") in ("arithmetic", "linear", "derivative"):
        slots.append("expression" if g.get("expression") else "number")
    if g.get("kind") == "transform":
        # canonical apply_learned roles (matches case signatures)
        slots += ["rule", "input"]
    return slots or ["none"]


def _plan_differs(a: dict, b: dict) -> bool:
    if a["candidate"].get("op_chain") != b["candidate"].get("op_chain"):
        return True
    return a["candidate"].get("rule_chain", []) != \
        b["candidate"].get("rule_chain", [])


def _inputs_for(task: CognitiveTask, plan: list[dict]) -> dict:
    """Current-task inputs keyed by op for transplanted steps."""
    per_op: dict[str, dict] = {}
    g = task.goal
    if g.get("kind") in ("arithmetic", "linear", "derivative"):
        per_op["compute"] = {"kind": g["kind"],
                             "expression": g.get("expression", ""),
                             "var": g.get("var", "x")}
    if g.get("kind") == "transform":
        per_op["apply_learned"] = {"domain": g.get("domain", "math"),
                                   "chain": g.get("chain"),
                                   "inputs": g.get("inputs", {})}
    return per_op


def compose_for_goal(ranked_accepted, goal_chain):
    from .analogy import compose_for_goal as _compose
    return _compose(ranked_accepted, goal_chain)


def _refuse(task, trace, memory, reason, candidates):
    trace["analogy"] = {"outcome": "refused", "reason": reason,
                        "candidates": candidates}
    trace["failure_cause"] = "verify_failed"
    _raise_gap_for(task, trace, memory)
    task.status = "failed"
    _sync(task, trace)
    _persist(memory, task, trace)
    return trace


def _transfer_rules(task, trace, ctx, memory, src_id, rules, top,
                    step_inputs, composed_sources=None):
    """Execute a rule chain through the standard op + verification."""
    from . import analogy as A
    from .operations import ApplyLearnedRule, CheckResult
    mem = ctx["memory"]
    per_op = {"apply_learned": ((step_inputs or {}).get("apply_learned")
                                or {})}
    ok_pre, missing = A.check_transfer_preconditions(
        {}, [{"op": "apply_learned"}], per_op)
    if not ok_pre:
        trec = {"source_task_id": src_id,
                "composed_sources": composed_sources,
                "current_task_id": task.task_id, "mapping": top["mapping"],
                "structural": top["structural"], "surface": top["surface"],
                "rules": rules, "checks": [{"preconditions": False,
                                            "missing": missing}],
                "outcome": "refused",
                "reason": f"missing preconditions: {missing}"}
        trace["analogy"] = trec
        task.status = "failed"
        trace["failure_cause"] = "verify_failed"
        _raise_gap_for(task, trace, memory)
        mem.add_experience(
            "evaluation", f"analogy refused: {src_id} -> {task.task_id}",
            {"source_task": src_id, "missing": missing})
        _sync(task, trace)
        _persist(memory, task, trace)
        return trace
    rec = ApplyLearnedRule().run(
        {"domain": (task.goal or {}).get("domain", "math"), "chain": rules,
         "inputs": ((step_inputs or {}).get("apply_learned") or {}).get(
             "inputs", {})},
        ctx)
    trace["steps"].append(rec)
    chk = CheckResult().run(
        {"check": {"kind": "nonempty",
                   "value": rec.get("outputs", {}).get("applied", [])}},
        ctx)
    trace["steps"].append(chk)
    ok = rec["status"] == "verified" and chk["status"] == "verified"
    verdict = None
    if ok:
        # independent goal verification before any verified claim
        from . import verify
        verdict = verify.verify_task(
            task.to_dict(), trace["steps"], memory,
            ctx.get("ontology"))["verdict"]
        ok = verdict == "verified"
    if ok:
        outcome = "verified"
    elif verdict == "inconclusive":
        outcome = "inconclusive"
    else:
        outcome = "failed"
    trec = {"source_task_id": src_id, "composed_sources": composed_sources,
            "current_task_id": task.task_id, "mapping": top["mapping"],
            "structural": top["structural"], "surface": top["surface"],
            "rules": rules, "checks": [{"rules_verified": ok}],
            "outcome": outcome,
            "reason": "composed/transferred rule chain verified"
                      if ok else f"rule chain not proven ({outcome})"}
    trace["analogy"] = trec
    if outcome == "verified":
        task.status = "verified"
    else:
        task.status = outcome if outcome == "inconclusive" else "failed"
        trace["failure_cause"] = "verify_failed"
        _raise_gap_for(task, trace, memory)
    mem.add_experience(
        "evaluation", f"analogy {trec['outcome']}: {src_id} -> "
        f"{task.task_id}",
        {"source_task": src_id, "mapping": top["mapping"],
         "structural": top["structural"], "surface": top["surface"],
         "rules": rules})
    _finalize(memory, task, trace)
    return trace


def _finalize(memory, task, trace) -> None:
    from .explainer import provenance
    trace["provenance"] = provenance(trace)
    _sync(task, trace)
    _persist(memory, task, trace)


def _solve_via_analogy_once(task, trace, ctx, memory, ontology,
                            reasoner) -> str:
    """One controller-approved analogy attempt inside a repair loop.

    Quiet on gaps (the outer loop raises the single gap) but persists
    the attempt trace for audit.
    """
    from . import analogy as A
    dry = _make_plan(task)
    cur = {"task_id": task.task_id + "+an",
           "task_type": task.task_type,
           "goal_kind": (task.goal or {}).get("kind", ""),
           "op_chain": [s.get("op") for s in dry],
           "rule_chain": list((task.goal or {}).get("chain") or []),
           "slots": {k: "any" for k in _task_slots(task)},
           "input_text": task.input_text, "verified": False}
    from .operations import Analogize
    rec = Analogize().run({"current_case": cur,
                           "exclude_task_ids": [task.task_id]}, ctx)
    trace["steps"].append(rec)
    ranked = [r for r in rec["outputs"].get("ranked", []) if r["accepted"]]
    if not ranked:
        return "refused"
    top = ranked[0]
    goal_chain = list((task.goal or {}).get("chain") or [])
    exact = [r for r in ranked
             if r["candidate"].get("rule_chain", []) == goal_chain
             and goal_chain]
    if len(exact) == 1:
        rules = exact[0]["candidate"].get("rule_chain", [])
    else:
        composed = A.compose_for_goal(ranked, goal_chain)
        if not composed:
            return "refused"
        rules = composed["rules"]
    from .operations import ApplyLearnedRule, CheckResult
    auto = _inputs_for(task, [{"op": "apply_learned"}])
    rec2 = ApplyLearnedRule().run(
        {"domain": (task.goal or {}).get("domain", "math"), "chain": rules,
         "inputs": (auto.get("apply_learned") or {}).get("inputs", {})},
        ctx)
    trace["steps"].append(rec2)
    if rec2["status"] != "verified":
        return "failed"
    chk = CheckResult().run(
        {"check": {"kind": "nonempty",
                   "value": rec2.get("outputs", {}).get("applied", [])}},
        ctx)
    trace["steps"].append(chk)
    if chk["status"] == "verified":
        trace["analogy"] = {"outcome": "verified",
                            "source_task_id": top["candidate"]["task_id"],
                            "rules": rules, "mapping": top["mapping"]}
        return "verified"
    return "failed"
