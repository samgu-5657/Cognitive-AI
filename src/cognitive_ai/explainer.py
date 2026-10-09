"""Answer generation FROM execution traces (Phase 4C).

This module does not think: it renders what the trace records. Every
factual claim cites a trace entry; unverified/incomplete traces can only
produce uncertainty or failure explanations, never conclusions.
"""
from __future__ import annotations


def explain(trace: dict, style: str = "short") -> dict:
    """Returns {text, certainty, citations}. certainty in
    {certain, qualified, unknown} derived from the trace, not asserted."""
    task = trace.get("task", {})
    steps = trace.get("steps", [])
    status = task.get("status", "failed")
    if status != "verified":
        return _explain_failure(trace)
    if task.get("task_type") == "math":
        return _explain_math(trace, style)
    if task.get("task_type") == "conceptual":
        return _explain_conceptual(trace, style)
    if task.get("task_type") == "logic":
        return _explain_logic(trace, style)
    return {"text": "任务类型未知,无法生成答案。", "certainty": "unknown",
            "citations": []}


def _cite(step: dict) -> str:
    return f"[{step.get('op')}:{step.get('status')}]"


def _explain_math(trace, style) -> dict:
    comp = next((s for s in trace["steps"] if s.get("op") == "compute"
                 and s.get("status") == "verified"), None)
    if comp is None:
        return {"text": "计算未经验证,不能给出结论。",
                "certainty": "unknown", "citations": []}
    result = comp["outputs"]["result"]
    cites = [_cite(comp)]
    if style == "short":
        text = f"答案:{result}。"
    else:
        lines = [f"答案:{result}。", "推导:"]
        for st in comp["outputs"].get("steps", []):
            lines.append(f"  {st.get('rule')}: {st.get('before')} "
                         f"-> {st.get('after')}")
        lines.append("验证: 独立复算/回代已通过(见轨迹)。")
        text = "\n".join(lines)
    return {"text": text, "certainty": "certain", "citations": cites}


def _explain_conceptual(trace, style) -> dict:
    attrs = [s for s in trace["steps"] if s.get("op") == "query_attribute"
             and s.get("status") != "failed"]
    learned = [s for s in trace["steps"] if s.get("op") == "apply_learned"
               and s.get("status") == "verified"]
    rules = [s for s in trace["steps"] if s.get("op") == "apply_rule"
             and s.get("status") == "verified"]
    if not attrs and not learned:
        return {"text": "没有查到相关知识,无法回答。",
                "certainty": "unknown", "citations": []}
    lines, cites, certain = [], [], True
    for s in attrs:
        o = s["outputs"]
        inp = s.get("inputs", {}) or {}
        subj = inp.get("concept", "?")
        key = inp.get("key", "?")
        st, val = o.get("status"), o.get("value")
        cites.append(_cite(s))
        if st == "verified":
            lines.append(f"{subj}的{key}是{val}。(已验证)")
        elif isinstance(st, str) and st.startswith("inherited"):
            lines.append(f"{subj}的{key}是{val}。(继承自"
                         f"{(o.get('basis') or ['?'])[0]},非实例实测)")
            certain = False
        elif st == "hypothesis":
            lines.append(f"{subj}的{key}可能是{val}。(暂定假设,未验证)")
            certain = False
        else:
            lines.append(f"{subj}的{key}未知。")
            certain = False
    if style == "short" and lines:
        text = lines[0]
    else:
        text = "\n".join(["结论:"] + [f"  {l}" for l in lines])
    for s in learned:
        for app in s.get("outputs", {}).get("applied", []):
            o = app.get("output", {}) or {}
            label = o.get("label", o.get("value", o))
            lines.append(f"归纳规则{app['rule_id']}给出:{label}"
                         f"(候选知识,来自已验证规则,非断言事实)")
            cites.append(_cite(s))
    for s in rules:
        for f in s.get("outputs", {}).get("fired", []):
            lines.append(f"已验证规则{f.get('rule')}给出:{f.get('conclusion')}")
            cites.append(_cite(s))
    if style == "short" and lines:
        text = lines[0]
    elif lines:
        text = "\n".join(["结论:"] + [f"  {l}" for l in lines])
    else:
        text = "没有查到相关知识,无法回答。"
    return {"text": text,
            "certainty": "certain" if certain and not learned else
            ("qualified" if (lines and (not certain or learned)) else "unknown"),
            "citations": cites}


def _explain_logic(trace, style) -> dict:
    checks = [s for s in trace["steps"] if s.get("op") == "check_result"]
    if not checks or not all(c["status"] == "verified" for c in checks):
        missing = []
        for c in checks:
            missing += (c.get("outputs") or {}).get("missing", [])
        detail = f"缺失:{missing}" if missing else (checks[-1].get("note", "")
                                                    if checks else "")
        return {"text": "前提不足或未经证实,不能推出结论。"
                        + (f"({detail})" if detail else ""),
                "certainty": "unknown",
                "citations": [_cite(c) for c in checks]}
    goal = trace.get("task", {}).get("goal", {})
    return {"text": f"前提成立,可以推出:{goal.get('effect', goal)}。",
            "certainty": "certain",
            "citations": [_cite(c) for c in checks]}


def _explain_failure(trace) -> dict:
    cause = trace.get("failure_cause", "verify_failed")
    gap = (trace.get("gap") or {})
    reasons = {
        "unparseable": "输入无法可靠解析,已记录需要澄清。",
        "missing_concept": "缺少相关概念知识,已记为知识缺口。",
        "missing_rule": "缺少可执行规则,已记为知识缺口。",
        "wrong_op": "没有合适的内部操作,已记录。",
        "compute_failed": "内部计算超出范围或不合法,未猜测结果。",
        "verify_failed": "结果未能通过独立验证,不输出结论。",
        "inexpressible": "暂无合适的表达方式。",
    }
    text = reasons.get(cause, "任务失败,原因已记录。")
    miss = []
    for s in trace.get("steps", []):
        if s.get("op") == "check_result":
            miss += (s.get("outputs") or {}).get("missing", [])
    if miss:
        text += f"(缺失:{miss})"
    qid = gap.get("question_id")
    if qid:
        text += f"(缺口问题#{qid}已进入学习队列)"
    return {"text": text, "certainty": "unknown",
            "citations": [f"[failure:{cause}]"]}
