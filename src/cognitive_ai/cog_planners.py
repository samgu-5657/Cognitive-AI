"""Frame -> CognitiveTask planners (Phase 4C). One shared entry, domain
branches only where the goal/operations genuinely differ."""
from __future__ import annotations

import re

from .semantics import parse_sentence


def plan_for_frame(task, frame: dict, memory) -> None:
    text = task.input_text
    route = _detect_math(text)
    if route:
        _plan_math(task, frame, route)
        return
    if _detect_logic(frame, text):
        _plan_logic(task, frame)
        return
    if _detect_conceptual(frame, text):
        _plan_conceptual(task, frame)
        return
    task.task_type = "unknown"


def _detect_math(text: str):
    s = text.replace(" ", "").replace("×", "*").replace("÷", "/")
    m = re.match(r"^(?:计算|求|算|请计算|请问)?(.+)$", s)
    body = m.group(1) if m else s
    if re.match(r"^(?:解方程|解|求解)(.+)$", s):
        eq = re.sub(r"^(?:解方程|解|求解)", "", s)
        return {"kind": "linear" if "=" in eq else None, "expr": eq}
    if re.match(r"^(?:求导|求|对)(.+)(?:的导数|求导|导数)$", s) or \
            re.match(r"^(?:求导|导数)(.+)$", s):
        expr = re.sub(r"^(?:求导|导数)", "", s)
        expr = re.sub(r"(?:的导数|求导|导数)$", "", expr)
        return {"kind": "derivative", "expr": expr}
    if re.fullmatch(r"[0-9xX+\-*/^()\.]+", body) and \
            any(c.isdigit() for c in body) and \
            any(op in body for op in ("+", "-", "*", "/", "^")):
        if "=" in body:
            return {"kind": "linear", "expr": body}
        return {"kind": "arithmetic", "expr": body}
    return None


def _plan_math(task, frame, route) -> None:
    if not route.get("kind"):
        task.task_type = "unknown"
        return
    task.task_type = "math"
    task.goal = {"kind": route["kind"], "expression": route["expr"],
                 "var": "x"}
    task.known_facts = [{"fact": f"表达式:{route['expr']}",
                         "source": "user"}]
    task.required_operations = ["compute", "check_result"]
    task.constraints = ["仅限内部数学引擎(无外部求解器)",
                        f"表达式必须合法,否则缺项"]


def _detect_logic(frame: dict, text: str) -> bool:
    if frame.get("intent") == "causal":
        return True
    s = text.replace(" ", "")
    return ("如果" in s or "要是" in s or "假如" in s) and \
        any(w in s for w in ("推出", "那么", "因此", "所以", "得出"))


def _plan_logic(task, frame) -> None:
    task.task_type = "logic"
    clauses = [c for c in re.split(r"[。；？?]", task.input_text) if c.strip()]
    causal, premises = None, []
    for cl in clauses:
        if _has_conditional(cl):
            f = parse_sentence(cl)
            causal = next((p for p in f.predicates
                           if p.get("type") == "causal"), causal)
        premises += _extract_premises(cl)
    if frame and not causal:
        preds = frame.get("predicates", [])
        causal = next((p for p in preds if p.get("type") == "causal"), None)
    if causal:
        task.goal = {"kind": "modus_ponens",
                     "condition": causal.get("condition", ""),
                     "effect": causal.get("effect", ""),
                     "premises": premises}
        task.known_facts = [
            {"fact": f"规则:若{causal.get('condition')}则{causal.get('effect')}",
             "source": "user"}]
        task.known_facts += [{"fact": f"前提:{p}", "source": "user"}
                             for p in premises]
        if causal.get("condition", "") not in premises:
            task.constraints.append(
                "条件是否成立尚无证据:结论只能是有条件的,不能断言")
    else:
        task.goal = {"kind": "infer", "text": task.input_text}
    task.required_operations = ["retrieve_concepts", "apply_rule",
                                "check_result"]
    task.constraints = ["前提缺失/反例/冲突时不得下肯定结论"]


def _has_conditional(clause: str) -> bool:
    return any(m in clause for m in ("如果", "要是", "假如"))


def _extract_premises(text: str) -> list[str]:
    """已知X[且Y]... -> [X, Y]. Pure splitter, no semantic claims."""
    m = re.search(r"已知(.+?)(?:[，。]|能推出|推出|因此|所以|$)", text)
    if not m:
        return []
    chunk = m.group(1)
    return [c.strip("，,、 ") for c in re.split(r"[且和与、,，]", chunk)
            if c.strip("，,、 ")]


def _detect_conceptual(frame: dict, text: str) -> bool:
    if frame.get("intent") in ("query", "classify", "attribute", "function",
                               "observe"):
        return True
    return any(w in text for w in ("什么", "用途", "功能", "属性", "为什么",
                                   "是否", "吗", "呢"))


def _plan_conceptual(task, frame) -> None:
    task.task_type = "conceptual"
    ents = [e["text"] for e in frame.get("entities", [])]
    task.goal = {"kind": "answer_about", "entities": ents,
                 "raw": task.input_text}
    task.known_facts = [{"fact": f"提及实体:{e}", "source": "parse"}
                        for e in ents]
    if not ents:
        task.unresolved_items.append("缺少可查询的实体,需要澄清")
    task.required_operations = ["retrieve_concepts", "query_attribute",
                                "apply_rule", "compare_explanations",
                                "check_result"]
    task.constraints = ["区分verified/继承/hypothesis/unknown",
                        "不得把同类特征无条件继承给实例"]
