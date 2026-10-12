"""Answer generation FROM execution traces (Phase 4C).

This module does not think: it renders what the trace records. Every
factual claim cites a trace entry; unverified/incomplete traces can only
produce uncertainty or failure explanations, never conclusions.
"""
from __future__ import annotations

import re


def explain(trace: dict, style: str = "short") -> dict:
    """Returns {text, certainty, citations, provenance}. certainty in
    {certain, qualified, unknown} derived from the trace, not asserted."""
    task = trace.get("task", {})
    status = task.get("status", "failed")
    base = {"provenance": provenance(trace)}
    if status == "inconclusive":
        out = _explain_failure(trace)
        out["text"] = "证据不足，结论未定：" + out["text"]
        out.update(base)
        return out
    if status != "verified":
        out = _explain_failure(trace)
        out.update(base)
        return out
    if task.get("task_type") == "math":
        out = _explain_math(trace, style)
    if task.get("task_type") == "conceptual":
        out = _explain_conceptual(trace, style)
    if task.get("task_type") == "logic":
        out = _explain_logic(trace, style)
    if task.get("task_type") not in ("math", "conceptual", "logic"):
        out = {"text": "任务类型未知,无法生成答案。", "certainty": "unknown",
               "citations": []}
    out.update(base)
    return out


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
    chained = [s for s in trace["steps"] if s.get("op") == "chain_infer"
               and s.get("status") == "verified"]
    if not attrs and not learned and not chained:
        return {"text": "没有查到相关知识,无法回答。",
                "certainty": "unknown", "citations": []}
    lines, cites, certain = [], [], True
    vlines = []
    for s in attrs:
        o = s["outputs"]
        inp = s.get("inputs", {}) or {}
        subj = inp.get("concept", "?")
        key = inp.get("key", "?")
        # display-only localization for atomic siblings (trace keeps raw):
        # function#2 reads as 功能2. Values (with conditions) untouched.
        disp = re.sub(r"^function#(\d+)$", r"功能\1", str(key))
        st, val = o.get("status"), o.get("value")
        cites.append(_cite(s))
        if "expected" in inp:
            # A-not-A verification (Phase 6D): report match, not value.
            # Verification conclusions lead the short answer (they are the
            # question asked); supporting reads follow in steps style.
            exp, match = inp.get("expected"), o.get("match")
            if key == "is_a":
                if st == "verified" and match:
                    vlines.append(f"是的,{subj}属于{exp}。(已验证)")
                elif st == "verified":
                    vlines.append(f"不,{subj}不属于{exp}。(已验证)")
                else:
                    vlines.append(f"{subj}是否属于{exp}未知。")
                    certain = False
                lines.append(vlines[-1])
                continue
            if st == "verified" and match:
                vlines.append(f"是的,{subj}的{disp}是{exp}。(已验证)")
            elif st == "verified":
                vlines.append(f"不是,{subj}的{disp}是{val},不是{exp}。(已验证)")
            elif st == "unverified":
                vlines.append(f"{subj}的{disp}可能是{val},"
                              f"是否为{exp}尚无验证。(暂定假设,未验证)")
                certain = False
            else:
                vlines.append(f"{subj}的{disp}未知,无法验证是否为{exp}。")
                certain = False
            lines.append(vlines[-1])
            continue
        if st == "verified":
            lines.append(f"{subj}的{disp}是{val}。(已验证)")
        elif isinstance(st, str) and st.startswith("inherited"):
            lines.append(f"{subj}的{disp}是{val}。(继承自"
                         f"{(o.get('basis') or ['?'])[0]},非实例实测)")
            certain = False
        elif st == "hypothesis":
            lines.append(f"{subj}的{disp}可能是{val}。(暂定假设,未验证)")
            certain = False
        else:
            lines.append(f"{subj}的{disp}未知。")
            certain = False
    text = None
    if style == "short" and lines:
        # verification conclusions lead; otherwise join sibling reads so
        # atomic capabilities (function/function#2…) all surface instead
        # of only the first line. Still strictly trace-backed.
        text = vlines[0] if vlines else "；".join(lines)
    elif lines:
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
    for s in chained:
        for proof in (s.get("outputs", {}) or {}).get("proofs", []):
            chain = ">".join(
                [proof.get("rule_id", "kb")] +
                [p.get("rule_id", "kb")
                 for p in proof.get("premises", []) or []])
            val = proof.get("value", "")
            st = proof.get("status", "unknown")
            cites.append(_cite(s))
            if st == "verified":
                lines.append(f"多步推导[{chain}]得出:{val}(已验证,有证明链)")
            elif st == "inherited(verified)":
                lines.append(f"多步推导[{chain}]得出:{val}(继承依据,非直接验证)")
                certain = False
            elif st == "hypothesis":
                lines.append(f"多步推导[{chain}]可能得出:{val}(暂定假设,未验证)")
                certain = False
            elif st == "contested":
                lines.append(f"多步推导[{chain}]存在冲突,无确定结论。")
                certain = False
    if text is None:
        if style == "short" and lines:
            text = vlines[0] if vlines else lines[0]
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


def provenance(trace: dict) -> dict:
    """Structured answer lineage: task, conditions, evidence, checks,
    unresolved items and the status timeline. Everything points back
    into the persisted trace; nothing here is generated prose."""
    task = trace.get("task", {})
    facts, rules, unresolved = [], [], []
    for s in trace.get("steps", []):
        if s.get("op") == "query_attribute" and s.get("status") != "failed":
            facts.append({"op": "query_attribute",
                          "inputs": s.get("inputs", {}),
                          "outputs": s.get("outputs", {})})
        if s.get("op") in ("apply_rule", "apply_learned") and \
                s.get("status") == "verified":
            rules.append({"op": s["op"],
                          "applied": (s.get("outputs", {}) or {}).get(
                              "applied", s.get("outputs", {}).get("fired",
                                                                  []))})
    unresolved = list((task.get("unresolved_items") or []))
    for g in (task.get("goal") or {}).get("premises", []) or []:
        facts.append({"premise": g})
    timeline = [(r.get("attempt"), r.get("decision"), r.get("cause"))
                for r in trace.get("repairs", [])]
    return {"task_id": task.get("task_id"),
            "task_type": task.get("task_type"),
            "goal": task.get("goal", {}),
            "facts": facts, "rules": rules,
            "checks": (trace.get("verification") or {}).get("checks", []),
            "unresolved": unresolved,
            "status": task.get("status"),
            "status_timeline": timeline,
            "failure_cause": trace.get("failure_cause"),
            "gap_question": (trace.get("gap") or {}).get("question_id")}
