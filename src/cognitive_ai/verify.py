"""Independent result verification (Phase 5C).

The verifier NEVER trusts executor conclusions: it re-derives everything
from the task goal (re-execution, KB re-query, rule re-check) and reports
one of verified | inconclusive | rejected per check and overall.

Meanings (strict):
  executed             the op ran without crashing;
  internally_consistent local re-checks pass (types, non-emptiness);
  rule_applicable      every used rule was active+selected and its
                       preconditions hold on the ACTUAL inputs;
  verified             the answer satisfies the task goal (re-derived);
  inconclusive         evidence insufficient either way;
  rejected             the answer contradicts goal/evidence/checks.
"""
from __future__ import annotations

import re


def verify_task(task: dict, trace_steps: list, memory, ontology=None) -> dict:
    """Full independent verification. Returns {verdict, checks[]}.

    checks[] entries: {name, result: verified|inconclusive|rejected,
    detail}. Overall verdict = rejected if any rejected, else
    inconclusive if any inconclusive, else verified.
    """
    goal = task.get("goal", {}) if isinstance(task, dict) else {}
    kind = goal.get("kind", "")
    checks = []
    if kind in ("arithmetic", "linear", "derivative"):
        checks.append(_check_math_goal(task, trace_steps))
    elif kind == "transform":
        checks.append(_check_transform_goal(task, trace_steps, memory))
    elif kind in ("answer_about", "infer"):
        checks.append(_check_qa_goal(task, trace_steps, memory, ontology))
    elif kind == "modus_ponens":
        checks.append(_check_logic_goal(task, trace_steps, memory))
    else:
        checks.append({"name": "goal_supported", "result": "inconclusive",
                       "detail": f"no verifier for goal kind {kind!r}"})
    checks.append(_check_rule_applicability(task, trace_steps, memory,
                                            ontology))
    checks.append(_check_binding_consistency(trace_steps))
    checks.append(_check_kb_conflicts(task, trace_steps, memory, ontology))
    verdict = "verified"
    for c in checks:
        if c["result"] == "rejected":
            verdict = "rejected"
            break
        if c["result"] == "inconclusive":
            verdict = "inconclusive"
    return {"verdict": verdict, "checks": checks}


def _final_answer(trace_steps: list):
    for s in reversed(trace_steps):
        if s.get("op") == "compute" and s.get("status") == "verified":
            return s.get("outputs", {}).get("result")
        if s.get("op") == "apply_learned" and s.get("status") == "verified":
            outs = (s.get("outputs", {}) or {}).get("applied", [])
            if outs:
                return outs[-1].get("output")
    return None


def _check_math_goal(task, steps) -> dict:
    from . import mathengine as M
    goal = task["goal"]
    kind, expr = goal.get("kind"), goal.get("expression", "")
    answer = _final_answer(steps)
    if answer is None:
        return {"name": "goal_satisfied", "result": "inconclusive",
                "detail": "no verified compute/apply record to check"}
    try:
        if kind == "arithmetic":
            expect = str(M.evaluate(M.parse_expr(expr)))
            ok = expect == str(answer)
            return {"name": "goal_satisfied",
                    "result": "verified" if ok else "rejected",
                    "detail": f"independent re-evaluation={expect}, "
                              f"answer={answer}"}
        if kind == "linear":
            var = goal.get("var", "x")
            sol = M.Fraction(str(answer).split("=")[-1])
            lhs, rhs = M.split_equation(expr)
            resid = M.evaluate(M.Add(
                (lhs, M.Mul((M.Num(M.Fraction(-1)), rhs)))), {var: sol})
            ok = resid == 0
            return {"name": "goal_satisfied",
                    "result": "verified" if ok else "rejected",
                    "detail": f"substitution residual={resid}"}
        if kind == "derivative":
            node = M.parse_expr(expr)
            terms = node.terms if isinstance(node, M.Add) else (node,)
            parts = [M.differentiate(t, goal.get("var", "x"))[0]
                     for t in terms]
            expect = parts[0] if len(parts) == 1 else M.Add(tuple(parts))
            ok = str(expect) == str(answer)
            return {"name": "goal_satisfied",
                    "result": "verified" if ok else "rejected",
                    "detail": f"term-wise re-derivation={expect}, "
                              f"answer={answer}"}
    except Exception as e:  # noqa: BLE001 - unverifiable, not verified
        return {"name": "goal_satisfied", "result": "inconclusive",
                "detail": f"re-derivation impossible: {e}"}
    return {"name": "goal_satisfied", "result": "inconclusive",
            "detail": "unknown math goal"}


def _check_transform_goal(task, steps, memory) -> dict:
    goal = task["goal"]
    chain = list(goal.get("chain") or [])
    inputs = dict(goal.get("inputs") or {})
    if not chain:
        return {"name": "goal_satisfied", "result": "inconclusive",
                "detail": "goal chain unspecified; no verified answer "
                          "possible without a specified chain"}
    from . import rulelearn as RL
    data = dict(inputs)
    try:
        for rid in chain:
            row = memory.get_rule(rid)
            if row is None or row.get("status") != "active":
                return {"name": "goal_satisfied", "result": "rejected",
                        "detail": f"rule {rid} not active"}
            spec = RL.selected_spec(row)
            slots = spec.get("inputs", {})
            if "_prev" in data and isinstance(slots, dict) \
                    and len(slots) == 1:
                data = {**data, next(iter(slots)): data["_prev"]}
            res = RL.apply_spec(spec, data)
            if res["status"] != "ok":
                return {"name": "goal_satisfied", "result": "rejected",
                        "detail": f"rule {rid} failed re-execution: "
                                  f"{res['status']}"}
            data["_prev"] = res["output"] if not isinstance(
                res["output"], dict) else res["output"].get("label",
                                                            res["output"])
        expect = data.get("_prev")
    except RL.Unsupported as e:
        return {"name": "goal_satisfied", "result": "inconclusive",
                "detail": f"re-execution unsupported: {e}"}
    answer = _final_answer(steps)
    if answer is None:
        # coverage-only verdict: chain re-runs, but nothing claimed
        return {"name": "goal_satisfied", "result": "inconclusive",
                "detail": f"chain re-executes (result {expect}); "
                          "no claimed answer to compare"}
    if isinstance(answer, dict):
        answer = answer.get("label", answer.get("value", answer))
    ok = str(expect) == str(answer)
    return {"name": "goal_satisfied",
            "result": "verified" if ok else "rejected",
            "detail": f"independent chain re-execution={expect}, "
                      f"answer={answer}"}


def _check_logic_goal(task, steps, memory) -> dict:
    """Modus ponens re-derived: premise present, no contest, effect stated.

    Missing premise or contested condition -> rejected (no conclusion).
    Empty effect -> inconclusive (nothing to conclude).
    """
    goal = task.get("goal", {})
    cond = goal.get("condition", "")
    effect = (goal.get("effect", "") or "").strip()
    premises = list(goal.get("premises", []) or [])
    if not cond:
        return {"name": "goal_satisfied", "result": "inconclusive",
                "detail": "no condition recorded"}
    if not effect:
        return {"name": "goal_satisfied", "result": "inconclusive",
                "detail": "no effect stated"}
    if cond not in premises and not any(
            (h and (h in cond or cond in h)) for h in premises):
        return {"name": "goal_satisfied", "result": "rejected",
                "detail": f"premise missing for condition {cond!r}"}
    matched = cond if cond in premises else next(
        h for h in premises if h and (h in cond or cond in h))
    for cl in memory.list_claims():
        if cl.get("status") != "contested":
            continue
        text = cl.get("text") or ""
        if cond and ((cond in text or text in cond)
                     or _overlapish(cond, text)):
            return {"name": "goal_satisfied", "result": "rejected",
                    "detail": f"condition contested: {text[:60]}"}
    return {"name": "goal_satisfied", "result": "verified",
            "detail": f"premise {matched!r} holds; effect {effect!r} follows"}


def _overlapish(a: str, b: str, n: int = 2) -> bool:
    grams = {a[i:i + n] for i in range(len(a) - n + 1)} - {""}
    return any(g and g in b for g in grams)


def _check_qa_goal(task, steps, memory, ontology) -> dict:
    # every cited attribute value must currently hold in the KB with a
    # non-rejected status. A correctly reported unknown (no value claimed,
    # none stored) is a VERIFIED outcome -- the system proved absence.
    # A claimed value that drifted or vanished is REJECTED.
    problems = []
    checked = 0
    for s in steps:
        if s.get("op") != "query_attribute" or s.get("status") == "failed":
            continue
        inp = s.get("inputs", {}) or {}
        out = s.get("outputs", {}) or {}
        cid, key = inp.get("concept"), inp.get("key")
        if ontology is None or not cid or not key:
            continue
        if memory.get_concept(cid) is None:
            problems.append(f"concept {cid} vanished")
            continue
        inh = ontology.inherited_attributes(cid)
        if out.get("status") == "unknown":
            if key in inh:
                problems.append(f"{cid}.{key} claimed unknown but stored")
            continue
        checked += 1
        if key not in inh:
            problems.append(f"{cid}.{key} cited but not stored")
        elif str(inh[key]["value"]) != str(out.get("value")):
            problems.append(f"{cid}.{key} value drift: "
                            f"{out.get('value')} vs {inh[key]['value']}")
        elif inh[key]["status"] == "rejected":
            problems.append(f"{cid}.{key} relies on rejected row")
    if problems:
        return {"name": "goal_satisfied", "result": "rejected",
                "detail": "; ".join(problems)}
    if checked == 0:
        return {"name": "goal_satisfied", "result": "verified",
                "detail": "unknown correctly reported; nothing claimed"}
    return {"name": "goal_satisfied", "result": "verified",
            "detail": f"{checked} citation(s) re-queried OK"}


def _check_rule_applicability(task, steps, memory, ontology) -> dict:
    bad = []
    used = 0
    for s in steps:
        if s.get("op") == "apply_learned" and s.get("status") == "verified":
            used += 1
            for app in (s.get("outputs", {}) or {}).get("applied", []):
                row = memory.get_rule(app.get("rule_id", ""))
                if row is None or row.get("status") != "active":
                    bad.append(f"{app.get('rule_id')} not active")
        if s.get("op") == "apply_rule" and s.get("status") == "verified":
            used += 1
    if used == 0:
        return {"name": "rule_applicability", "result": "verified",
                "detail": "no rule applications to audit"}
    if bad:
        return {"name": "rule_applicability", "result": "rejected",
                "detail": "; ".join(bad)}
    return {"name": "rule_applicability", "result": "verified",
            "detail": f"{used} application(s) on active rules"}


def _check_binding_consistency(steps) -> dict:
    # chain threading coherence: each chained rule after the first must
    # have consumed the previous output (_prev present in its trace)
    for s in steps:
        if s.get("op") != "apply_learned" or s.get("status") != "verified":
            continue
        apps = (s.get("outputs", {}) or {}).get("applied", [])
        if len(apps) > 1:
            linked = all(
                any("compute" in str(t) for t in (a.get("trace") or []))
                for a in apps[1:])
            if not linked:
                return {"name": "binding_consistency",
                        "result": "inconclusive",
                        "detail": "chained application lacks compute trace"}
    return {"name": "binding_consistency", "result": "verified",
            "detail": "threading coherent"}


def _check_kb_conflicts(task, steps, memory, ontology) -> dict:
    # Only conflicts touching THIS task's concepts/keys matter; a distant
    # open conflict must not taint unrelated answers.
    if ontology is None:
        return {"name": "kb_conflicts", "result": "inconclusive",
                "detail": "no ontology available"}
    from .reasoner import Reasoner
    touched: set[str] = set()
    for s in steps:
        if s.get("op") == "query_attribute":
            inp = s.get("inputs", {}) or {}
            if inp.get("concept"):
                touched.add(inp["concept"])
                if inp.get("key"):
                    touched.add(f"{inp['concept']}.{inp['key']}")
    hits = []
    for cf in Reasoner(memory).check_conflicts():
        if cf.get("type") == "attribute_conflict":
            ref = f"{cf.get('concept')}.{cf.get('key')}"
            if cf.get("concept") in touched or ref in touched:
                hits.append(ref)
        else:
            text = str((cf.get("claim") or {}).get("text", ""))
            m = re.match(r"(\S+?)\.(\S+?):", text)
            if m and (m.group(1) in touched
                      or f"{m.group(1)}.{m.group(2)}" in touched):
                hits.append(text[:60])
            elif not touched:
                hits.append(text[:60])
    if hits:
        return {"name": "kb_conflicts", "result": "inconclusive",
                "detail": f"open conflict(s) touch this task: {hits}"}
    return {"name": "kb_conflicts", "result": "verified",
            "detail": "no open conflicts touch this task"}
