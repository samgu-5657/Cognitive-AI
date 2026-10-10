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


def _is_bare_followup(text: str) -> bool:
    """Bare follow-up test by CONTENT, not length.

    Strips follow-up markers, discourse glue and punctuation; only an
    empty remainder counts ("为什么", "那为什么呢", "能举个例子吗").
    Anything naming an object or action ("为什么杯子会漏水", "那材质呢")
    parses fresh. Replaces the old fixed char-count threshold.
    """
    s = text.strip().rstrip("？?！!。")
    for w in sorted(list(FOLLOW_WHY) + list(FOLLOW_EXAMPLE) + ["那", "那么"],
                    key=len, reverse=True):
        s = s.replace(w, "")
    s = s.strip().strip("？?！!吗呢，,、 ")
    return not s


def _ellipsis_key(text: str, lexicon: dict) -> tuple[str, str] | None:
    """Short attribute-word fragment -> (word, canonical key).

    Matches 那材质呢 / 颜色? / 用途吗 etc. Generic over the attribute
    vocabulary (built-ins + learned mappings), never per-sentence.
    """
    s = text.strip().strip("。！")
    m = re.match(r"^(那|那么)?(.+?)(呢|吗|？|\?)?$", s)
    if not m:
        return None
    word = m.group(2).strip().strip("？?！!呢吗，,、 ")
    if not word or len(word) > 6:
        return None
    from .semantics import ATTR_WORDS
    key = ATTR_WORDS.get(word) or (lexicon or {}).get(word)
    if not key:
        return None
    return word, key


def _bind_topic(session, topic: str) -> None:
    """Restore a named topic to the front of the session entities."""
    try:
        st = session.mem.get_context(session.session_id)
        ents = [{"text": topic, "turn": int(st.get("turn", 0)) + 1}] + \
            [e for e in st.get("entities", []) if e.get("text") != topic]
        st["entities"] = ents[:10]
        session.mem.set_context(session.session_id, st)
    except Exception:
        pass


def _answer_topic(session, name: str) -> dict:
    """Read-only concept overview for topic introductions / returns."""
    name = (name or "").strip()
    if not name:
        return {"text": "想了解哪个对象？请说出名称。",
                "certainty": "unknown", "citations": []}
    hit = session.mem.get_concept(name)
    if hit is None:
        found = session.mem.find_concepts_by_name(name)
        hit = found[0] if found else None
    if hit is None:
        _bind_topic(session, name)
        return {"text": f"“{name}”暂未收录，我先把它记为本轮话题；"
                        "用“观察 <名称> 属性=值”教我，或换个已知概念。",
                "certainty": "unknown", "citations": []}
    try:
        d = session.onto.describe(hit["id"])
    except Exception:
        return {"text": f"“{name}”暂无法展开，稍后再试。",
                "certainty": "unknown", "citations": []}
    lines = [f"我们来聊{d['name']}（{d['id']}）。"]
    if d.get("parents"):
        lines.append(f"它属于：{', '.join(d['parents'])}。")
    known = [(a["key"], a["value"], a["status"])
             for a in d.get("attributes", [])]
    if known:
        lines.append("已知属性：" + "；".join(
            f"{k}={v}（{st}）" for k, v, st in known[:6]) + "。")
    lines.append("可以继续问它的用途、材质等，或用“观察”教我新知识。")
    _bind_topic(session, name)
    return {"text": "\n".join(lines), "certainty": "qualified",
            "citations": []}


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
        # (follow-ups are handled after parsing in stage 3c, where bare
        # "为什么" can be told apart from content questions naming an
        # entity -- see _is_bare_followup. No char-count threshold.)
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
        from .semantics import (ATTR_WORDS, parse_sentence, resolve_frame,
                                resolve_question, update_context)
        work = stripped
        # Sentence-final punctuation carries no meaning for routing
        # (计算3+4*2。 must still hit math detection); the parser strips
        # it again internally. Question particles 吗/呢 are kept.
        work = work.rstrip("。！？；;!?")
        # topic-shift prefix (换个话题，…) splits from the real content:
        # the marker is recorded, the content after it is fully parsed
        # (never dropped with the marker). History is kept -- recency
        # decides reference, explicit 回到 restores older topics.
        shifted = False
        for mark in ("换个话题", "换个主题", "换话题", "换一题", "换个问题"):
            if work.startswith(mark):
                work = work[len(mark):].lstrip("，,、 ").strip()
                shifted = True
                break
        turn["stages"]["shift"] = {"shifted": shifted}
        if shifted and not work:
            turn["route"] = "clarify"
            turn["answer"] = {
                "text": "想换话题可以直说新问题（如“计算3+4*2”或"
                        "“介绍一下茶壶”）。本次未写入任何内容。",
                "certainty": "unknown", "citations": []}
            _log(session, turn)
            return turn
        # return-to-topic marker (回到刚才的X，…) binds X as current
        # topic, then the remainder is parsed normally.
        mret = re.match(r"^回到(刚才|之前)的(.+?)([，,].*)?$", work)
        if mret:
            topic, rest = mret.group(2).strip(), \
                (mret.group(3) or "").lstrip("，, ").strip()
            if topic and rest:
                _bind_topic(session, topic)
                work = rest
                turn["stages"]["shift"] = {"shifted": shifted,
                                           "returned": topic}
            elif topic:
                turn["route"] = "topic"
                turn["answer"] = _answer_topic(session, topic)
                _log(session, turn)
                return turn
        _ph = getattr(session, "_phrase_rules", None)
        frame = parse_sentence(work, source="user",
                               lexicon=session._lexicon(),
                               phrase_rules=_ph() if _ph else None)
        # session context: resolve pronouns against real history, then
        # persist (per-session isolation via session_id; entities only,
        # most-recent-first, capped -- see semantics.update_context).
        # The dialog layer additionally keeps last-turn bookkeeping
        # under the "dialog" key (frame kind, entities, route).
        state = session.mem.get_context(session.session_id)
        frame = resolve_frame(frame, state)
        state = update_context(state, frame)
        try:
            dlg = state.get("dialog", {})
            dlg["last_entities"] = [e.get("text", "")
                                    for e in frame.entities]
            dlg["last_intent"] = frame.intent
            state["dialog"] = dlg
            session.mem.set_context(session.session_id, state)
        except Exception:
            pass
        turn["stages"]["parse"] = frame.to_dict()
        turn["stages"]["context"] = {
            "entities": [e.get("text", "")
                         for e in state.get("entities", [])]}
        # bare follow-up ("为什么", "那为什么呢"): every content word is
        # a follow-up marker -> reuse the last answer; anything with real
        # content left parses fresh (content test, not char count).
        if _is_bare_followup(work):
            follow = _is_followup(work)
            if follow and getattr(session, "last_answer", None):
                turn["route"] = "followup"
                turn["stages"]["parse"] = {"dialogue_act": follow}
                turn["answer"] = _answer_followup(session, follow)
                _log(session, turn)
                return turn
        # ellipsis ("那材质呢"): short attribute-word fragment reuses the
        # current topic entity with that key. Checked before pronoun
        # clarification: the 那 here is glue, not a reference. No topic
        # -> clarify.
        ell = _ellipsis_key(work, session._lexicon())
        if ell and (frame.intent == "unknown" or
                    (frame.intent == "query" and not frame.predicates)):
            topic = (state.get("entities") or [{}])[0].get("text", "")
            if topic:
                frame = parse_sentence(f"{topic}的{ell[1]}是什么",
                                       source="user-ellipsis",
                                       lexicon=session._lexicon())
                frame.trace.append(f"ellipsis: {work!r} -> topic "
                                   f"{topic!r} + key {ell[1]!r}")
                turn["stages"]["parse"] = frame.to_dict()
            else:
                turn["route"] = "clarify"
                turn["answer"] = {
                    "text": f"你问的“{ell[0]}”我缺少对象：请先说清是哪个"
                            "东西（如“介绍一下杯子”），再问它的属性。"
                            "本次未写入任何内容。",
                    "certainty": "unknown", "citations": []}
                _log(session, turn)
                return turn
        # unresolved pronouns with several/no candidates: ask, don't guess.
        if frame.unresolved_references and frame.intent == "query" \
                and frame.predicates:
            cands = [e.get("text", "")
                     for e in state.get("entities", [])]
            turn["route"] = "clarify"
            extra = ("（候选：" + "、".join(cands[:3]) + "）") if cands \
                else ""
            turn["answer"] = {
                "text": f"你说的“{frame.unresolved_references[0]}”我无法"
                        f"确定所指{extra}。请直接说出对象名称。"
                        "本次未写入任何内容。",
                "certainty": "unknown", "citations": []}
            _log(session, turn)
            return turn
        if frame.intent == "query" and any(
                p.get("type") == "query_topic" for p in frame.predicates):
            turn["route"] = "topic"
            turn["answer"] = _answer_topic(
                session, frame.predicates[0].get("subj", ""))
            _log(session, turn)
            return turn
        if frame.intent in ("query", "classify", "attribute", "function") \
                or _detect_math(work):
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
            turn["answer"] = _answer_qa(session, work, frame)
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
