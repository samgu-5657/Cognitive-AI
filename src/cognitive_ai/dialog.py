"""Unified dialogue pipeline (Phase 6B, Task 1).

One entry: process(session, text) -> DialogueTurn with per-stage results.
Stages: context -> parse -> route -> execute -> verify -> answer -> log.
Routes reuse existing modules (Session commands, executor, explainer);
smalltalk answers report REAL capabilities/counts, never invented ones.
"""
from __future__ import annotations

import re

GREETINGS = ("你好", "您好", "hi", "hello", "嗨", "早上好", "晚上好")
IDENTITY_Q = ("你是谁", "你叫什么", "你的名字", "介绍一下你自己")
CAPABILITY_Q = ("你能做什么", "你会什么", "你有什么功能", "功能介绍",
                "你能干什么", "帮助")
FOLLOW_WHY = ("为什么", "为啥", "什么原因", "理由是什么")
FOLLOW_EXAMPLE = ("能举个例子吗", "举个例子", "例如", "比如说", "比如")


def _is_smalltalk_command(text: str) -> str | None:
    s = text.strip()
    if s in GREETINGS or s.startswith(tuple(g + "，" for g in GREETINGS)):
        return "greet"
    if any(q in s for q in IDENTITY_Q):
        return "identity"
    if any(q in s for q in CAPABILITY_Q):
        return "capability"
    return None


def _is_followup(text: str) -> str | None:
    s = text.strip()
    if any(w in s for w in FOLLOW_WHY):
        return "why"
    if any(w in s for w in FOLLOW_EXAMPLE):
        return "example"
    return None


def process(session, text: str) -> dict:
    """Run the full pipeline. Never raises on user input."""
    turn: dict = {"input": text, "stages": {}, "route": None,
                  "answer": None, "error": None}
    try:
        # stage 1: context (session id always exists on Session)
        turn["stages"]["context"] = {"session": session.session_id}
        # stage 2+3: parse intent of the *dialogue act* + route
        stripped = (text or "").strip()
        if not stripped:
            turn["route"] = "empty"
            turn["answer"] = {"text": "请输入内容，或输入“帮助”查看可用的说法。",
                              "certainty": "unknown", "citations": []}
            _log(session, turn)
            return turn
        small = _is_smalltalk_command(stripped)
        if small and not (getattr(session, "pending_teach", None) or
                          getattr(session, "pending_rule", None)):
            turn["route"] = "smalltalk"
            turn["stages"]["parse"] = {"dialogue_act": small}
            turn["answer"] = _answer_smalltalk(session, small)
            _log(session, turn)
            return turn
        follow = _is_followup(stripped)
        # Bare follow-ups only: threshold 6 chars keeps "为什么" (3) and
        # "能举个例子吗" (6) while longer content questions containing
        # follow-up words (e.g. 8-char 为什么杯子会漏水) stay fresh QA.
        if follow and len(stripped) <= 6 and getattr(session, "last_answer", None) \
                and not (getattr(session, "pending_teach", None) or
                         getattr(session, "pending_rule", None)):
            # bare follow-ups only ("为什么", "能举个例子吗"): longer
            # inputs mentioning entities are fresh questions, even if
            # they contain a follow-up word (e.g. 为什么人们会使用杯子).
            turn["route"] = "followup"
            turn["stages"]["parse"] = {"dialogue_act": follow}
            turn["answer"] = _answer_followup(session, follow)
            _log(session, turn)
            return turn
        # stage 3b: staged teaching/rule flows and existing commands keep
        # their exact behavior (pending_teach answers must reach
        # Session._teach_answer, never the NL parser)
        if getattr(session, "pending_teach", None) or \
                getattr(session, "pending_rule", None):
            turn["route"] = "command"
            lines = session.handle(stripped)
            turn["answer"] = {"text": "\n".join(
                l for l in lines if l != "__quit__"),
                "certainty": "unknown", "citations": []}
            _log(session, turn)
            return turn
        first = stripped.split()[0]
        from .interact import ALIASES
        if first in ALIASES or first.lower() in (
                "help", "quit", "concepts", "inspect", "predict", "observe",
                "teach", "label", "hypothesis", "correct", "explore",
                "experiences", "history", "model", "queue", "review",
                "state", "cancel", "understand", "confirm", "context",
                "solve", "task", "rule_learn", "lex"):
            turn["route"] = "command"
            lines = session.handle(stripped)
            turn["answer"] = {"text": "\n".join(
                l for l in lines if l != "__quit__"),
                "certainty": "unknown", "citations": []}
            _log(session, turn)
            return turn
        # stage 3c: natural language -> parse -> QA or pending teach.
        # The compositional parser has no math patterns (by design: math
        # detection lives in cog_planners._detect_math over the raw text,
        # shared with the executor). A math-detectable input must route to
        # QA so the existing engine runs; never reject it as unknown prose.
        from .cog_planners import _detect_math
        from .semantics import parse_sentence, resolve_question
        _ph = getattr(session, "_phrase_rules", None)
        frame = parse_sentence(stripped, source="user",
                               lexicon=session._lexicon(),
                               phrase_rules=_ph() if _ph else None)
        turn["stages"]["parse"] = frame.to_dict()
        if frame.intent in ("query", "classify", "attribute", "function") \
                or _detect_math(stripped):
            rq = resolve_question(frame.to_dict(), session._lexicon())
            turn["stages"]["adjudication"] = rq
            if rq.get("needs_clarify") and rq.get("kind") == "unknown_q":
                # understood enough to know what is missing, not enough
                # to answer: ask, don't guess. Zero writes.
                turn["route"] = "clarify"
                turn["answer"] = {
                    "text": "我不太确定你在问什么："
                            + "; ".join(rq.get("defects", [])) + "。"
                            "请带上属性词重问（如 材质/用途/颜色），或用"
                            "“观察 <名称> 属性=值”结构化输入。"
                            "本次未写入任何内容。",
                    "certainty": "unknown", "citations": []}
                _log(session, turn)
                return turn
            turn["route"] = "qa"
            turn["answer"] = _answer_qa(session, stripped, frame)
        elif frame.intent in ("observe", "correct", "negate", "hedge",
                              "causal"):
            turn["route"] = "teach_pending"
            session.pending_frame = frame.to_dict()
            turn["answer"] = {
                "text": f"我理解为：意图={frame.intent}，实体="
                        f"{[e['text'] for e in frame.entities]}。"
                        f"输入“确认”执行，“取消”放弃；确认前不写入任何内容。",
                "certainty": "unknown", "citations": []}
        else:
            turn["route"] = "unknown"
            turn["answer"] = {
                "text": "这句话我暂时无法理解（只支持帮助中列出的命令与"
                        "常见陈述/疑问句式）。请换一种表达，或用“观察 "
                        "<名称> 属性=值”结构化输入。本次未写入任何内容。",
                "certainty": "unknown", "citations": []}
        _log(session, turn)
        return turn
    except Exception as e:  # noqa: BLE001 - dialogue never crashes
        turn["error"] = f"{type(e).__name__}: {e}"
        turn["answer"] = {"text": "处理出错，未写入任何内容。",
                          "certainty": "unknown", "citations": []}
        return turn


def _answer_smalltalk(session, kind: str) -> dict:
    if kind == "greet":
        return {"text": "你好！我是本地运行的 Cognitive AI，不联网、不调用"
                        "大语言模型。你可以问我已学过的概念、做数学计算、"
                        "教我新知识，或输入“帮助”查看说法示例。",
                "certainty": "certain", "citations": []}
    if kind == "identity":
        try:
            tag = session.adapter().version if session.adapter() else "未加载"
        except Exception:
            tag = "未加载"
        n_concepts = len(session.mem.list_concepts())
        return {"text": "我是 Cognitive AI，一个本地认知系统原型：概念网络"
                        f"中有 {n_concepts} 个概念，神经网络模型为{tag}。"
                        "我的回答都来自内部知识库与执行记录，查不到就直说不知道。",
                "certainty": "certain", "citations": []}
    # capability: enumerate ONLY implemented, tested abilities
    from .operations import REGISTRY
    from .semantics import INTENTS
    ops = sorted(REGISTRY)
    return {"text": "我目前真正会做的事：\n"
            "1. 回答已学概念的属性/用途/关系（有证据等级标注）\n"
            "2. 整数/分数算术、一元一次方程、多项式求导（内部引擎逐步推导）\n"
            "3. 按 fixed 句式理解中文陈述并记录观察/假设/纠正\n"
            "4. 从示例归纳候选规则（需验证后才激活）\n"
            f"5. 识别意图约 {len(INTENTS)} 类，内部操作约 {len(ops)} 种\n"
            "不会做的事：开放闲聊、联网搜索、理解任意长难句、写代码运行。"
            "能力边界见 README。",
            "certainty": "certain", "citations": []}


def _answer_followup(session, kind: str) -> dict:
    last = session.last_answer or {}
    if kind == "why":
        prov = last.get("provenance") or {}
        facts = prov.get("facts") or []
        if not facts:
            return {"text": "上一条回答没有留下可解释的依据，我无法说明原因。",
                    "certainty": "unknown", "citations": []}
        parts = []
        for f in facts[:3]:
            if "inputs" in f:
                parts.append(f"{f['inputs'].get('concept')}的"
                             f"{f['inputs'].get('key')}="
                             f"{f.get('outputs', {}).get('value')}")
        return {"text": "因为：" + "；".join(parts) + "。（来自执行记录）",
                "certainty": last.get("certainty", "unknown"),
                "citations": last.get("citations", [])}
    prov = last.get("provenance") or {}
    facts = prov.get("facts") or []
    if facts and "inputs" in facts[0]:
        c = facts[0]["inputs"].get("concept", "?")
        return {"text": f"例如：关于{c}，我上次查到的就是上面这些有来源的"
                        "结论。换个具体概念我可以再查。",
                "certainty": "unknown", "citations": []}
    return {"text": "上一条回答没有具体实例可举。请先问一个具体概念。",
            "certainty": "unknown", "citations": []}


def _answer_qa(session, text: str, frame) -> dict:
    from .executor import solve_text
    from .explainer import explain
    trace = solve_text(text, frame.to_dict(), session.mem, session.onto,
                       session.reasoner)
    ans = explain(trace, style="short")
    from .answers import render
    out = render(ans, trace)
    session.last_answer = {"text": out["text"],
                           "certainty": ans["certainty"],
                           "citations": ans["citations"],
                           "provenance": ans.get("provenance"),
                           "task_id": trace["task"]["task_id"]}
    session.mem.add_experience(
        "evaluation", f"dialog qa -> {trace['task']['status']}",
        {"task_id": trace["task"]["task_id"],
         "certainty": ans["certainty"], "op": "dialog"})
    return out


def _log(session, turn: dict) -> None:
    try:
        session.mem.add_experience(
            "evaluation", f"dialog turn route={turn['route']}",
            {"op": "dialog", "route": turn["route"],
             "stages": list(turn["stages"])})
    except Exception:
        pass
