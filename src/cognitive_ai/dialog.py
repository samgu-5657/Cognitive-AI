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


def _answer_paraphrase(session, text: str) -> dict:
    """Render the same question roles through learned + fixed surfaces.

    Input direction reuses parse/resolve (+semlearn fallback for taught
    forms); output direction uses verbalize over the SAME (kind, key)
    with the entity slot refilled. Roundtrip-checked by tests, not by
    construction. Zero writes.
    """
    from . import verbalize as _VB
    from .semantics import parse_sentence, resolve_question
    if not text:
        return {"text": "请在后面加上要换说法的问题（如：复述某物的材质疑问句）。",
                "certainty": "unknown", "citations": []}
    _ph = getattr(session, "_sem_rules", None)
    rules = _ph() if _ph else []
    frame = parse_sentence(text, source="user",
                           lexicon=session._lexicon())
    if frame.intent == "unknown" and rules:
        from . import semlearn as _SL
        try:
            built = _SL.construct(text, rules)
        except Exception:
            built = None
        if built is not None:
            frame = _frame_from_dict(built)
    rq = resolve_question(frame.to_dict(), session._lexicon())
    if rq.get("kind") in ("unknown_q", "unsupported_q") or \
            rq.get("needs_clarify"):
        return {"text": "这句的提问类型不明确，无法换说法："
                        + "; ".join(rq.get("defects", ["无法解析"])) + "。",
                "certainty": "unknown", "citations": []}
    subjects = rq.get("subjects", [])
    entity = subjects[0] if subjects else ""
    key = rq.get("key")
    kind = rq.get("kind")
    if kind == "classify_q" and "的" in entity:
        # possessive question (考拉的材质是什么): same normalization
        # the executor uses -- split off a known attribute word so the
        # learned attr_q rules can fire on it. Generic, not per-sentence.
        from .semantics import ATTR_WORDS
        left, right = entity.rsplit("的", 1)
        mapped = ATTR_WORDS.get(right, (session._lexicon() or {}).get(right))
        if left and mapped:
            kind, entity, key = "attr_q", left, mapped
    if key:
        for suffix in ("的" + _VB.zh_key(key), "的" + str(key)):
            if entity.endswith(suffix) and len(entity) > len(suffix):
                entity = entity[:-len(suffix)]
                break
    outs = _VB.render_question(kind, entity, key, rules,
                               exclude=text)
    if not outs:
        return {"text": "这个问法没有已知的替代表达。",
                "certainty": "unknown", "citations": []}
    lines = ["同一问题的其他说法："]
    for o in outs:
        lines.append(f"  {o['text']}（来源：{o['source']}）")
    return {"text": "\n".join(lines), "certainty": "unknown",
            "citations": []}


def _answer_generate(session, text: str) -> dict:
    """Generate (entity, value) surfaces (Phase 6H-2B).

    Input is explicit roles ("生成 海狸 筑坝"), not a parsed
    sentence: the learning lives in the TEMPLATE. Active
    constructions render first (source learned:id); the fixed
    canonical fallback (X可以V) always applies. Zero writes.
    """
    from . import constructions as _CX
    parts = text.split()
    if len(parts) != 2 or not all(parts):
        return {"text": "用法: 生成 <实体> <功能>（如：生成 海狸 筑坝）。",
                "certainty": "unknown", "citations": []}
    ent, val = parts
    _cx = getattr(session, '_constructions', None)
    outs = _CX.render(ent, val, _cx() if _cx else [])
    outs = list(outs) + [{"text": f"{ent}可以{val}", "source": "fixed"}]
    lines = ['同一语义的不同说法：']
    for o in outs:
        lines.append(f"  {o['text']}（来源：{o['source']}）")
    return {"text": "\n".join(lines), "certainty": "unknown",
            "citations": []}


def _frame_from_dict(d: dict):
    """Rebuild a SemanticFrame from a stored dict (resume path)."""
    from .semantics import SemanticFrame as _SF
    return _SF(**{k: d.get(k) for k in
                   ("intent", "entities", "predicates", "modality",
                    "scope", "source", "confidence",
                    "unresolved_references", "ambiguities", "raw",
                    "trace")})


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
        # pending concept-definition confirmation (Phase 6G-1): handled
        # before everything else so 确认/取消/修改 bind to the candidate.
        if getattr(session, "pending_define", None):
            turn["route"] = "teach_confirm"
            turn["answer"] = _define_followup(session, stripped)
            _log(session, turn)
            return turn
        # pending auto-discovery proposal (Phase 6H-2C): 确认 records a
        # trusted pair + runs the auto step; 取消 drops it with an audit;
        # anything else abandons the proposal and flows through normally.
        if getattr(session, "pending_cx", None):
            ans = _cx_followup(session, stripped)
            if ans is not None:
                turn["route"] = "teach_confirm"
                turn["answer"] = ans
                _log(session, turn)
                return turn
        # post-confirm capability amendment (Phase 6G-1收尾): 修改光盒2=…
        # or 修改功能2=… (topic concept). Touches one row/claim only.
        mam = re.match(r"^修改\s*(.+)$", stripped)
        if mam and not getattr(session, "pending_define", None):
            amended = _amend_saved(session, mam.group(1).strip())
            if amended is not None:
                turn["route"] = "teach_confirm"
                turn["answer"] = amended
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
        # paraphrase (Phase 6H-2A): 复述/换个说法 + question renders the
        # same roles through learned and fixed surfaces. Read-only.
        if first in ("复述", "换个说法"):
            turn["route"] = "paraphrase"
            turn["answer"] = _answer_paraphrase(
                session, stripped[len(first):].strip())
            _log(session, turn)
            return turn
        # generate (Phase 6H-2B): 生成 <实体> <功能> renders (entity,
        # value) through the SAME active constructions the parser uses,
        # plus the fixed canonical fallback. Read-only; unknown when no
        # construction or fallback covers the roles.
        if first == "生成":
            turn["route"] = "generate"
            turn["answer"] = _answer_generate(
                session, stripped[len(first):].strip())
            _log(session, turn)
            return turn
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
        # concept definition (Phase 6G-1): 我定义了一种叫X的… proposes a
        # candidate (parsed per clause, unclear parts kept verbatim) and
        # waits for explicit 确认. Nothing is written before confirmation.
        from .semantics import extract_definition
        if extract_definition(work) is not None:
            turn["route"] = "teach_propose"
            turn["answer"] = _propose_define(session, work)
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
        # Phase 6E direct-frame learning: only when the fixed parser
        # admits unknown. construct() builds the frame field-by-field
        # (no rewrite-and-reparse); a confident direct parse is never
        # replaced. No active rule -> baseline behavior, untouched.
        if frame.intent == "unknown":
            from . import semlearn as _SL
            _sem = getattr(session, "_sem_rules", None)
            try:
                built = _SL.construct(work, _sem() if _sem else [])
            except Exception:
                built = None
            if built is not None:
                from .semantics import SemanticFrame as _SF
                frame = _SF(**{k: built[k]
                               for k in ("intent", "entities", "predicates",
                                         "modality", "scope", "source",
                                         "confidence",
                                         "unresolved_references",
                                         "ambiguities", "raw", "trace")})
                turn["stages"]["parse"] = frame.to_dict()
                turn["stages"]["semlearn"] = True
                # the constructed entity joins the session topic like any
                # parsed one (pronouns later in the turn resolve to it)
                try:
                    state = update_context(state, frame)
                    session.mem.set_context(session.session_id, state)
                    turn["stages"]["context"] = {
                        "entities": [e.get("text", "") for e in
                                     state.get("entities", [])]}
                except Exception:
                    pass
        # Phase 6H-2B constructions: only when everything older is
        # unknown (fixed parser, phrase rewrites, semlearn frames).
        # A unique match yields an observation-grade frame for the
        # existing teach-confirm flow; ambiguity stays unknown.
        if frame.intent == "unknown":
            from . import constructions as _CX
            _cx = getattr(session, "_constructions", None)
            try:
                cfr = _CX.parse_frame(
                    work, _cx() if _cx else [], raw=work)
            except Exception:
                cfr = None
            if cfr is not None:
                from .semantics import SemanticFrame as _SF
                frame = _SF(**{k: cfr[k]
                               for k in ("intent", "entities", "predicates",
                                         "modality", "scope", "source",
                                         "confidence",
                                         "unresolved_references",
                                         "ambiguities", "raw", "trace")})
                turn["stages"]["parse"] = frame.to_dict()
                turn["stages"]["construction"] = True
                try:
                    state = update_context(state, frame)
                    session.mem.set_context(session.session_id, state)
                    turn["stages"]["context"] = {
                        "entities": [e.get("text", "") for e in
                                     state.get("entities", [])]}
                except Exception:
                    pass
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
        # clarification resume (Phase 6H-1): a waiting pronoun/question
        # is retried with the new input merged in; anything else flows
        # through normally (a later success or clarification replaces it).
        from .cognitive_cycle import park_for_clarify, run_cycle, try_resume
        resumed = try_resume(session, frame.to_dict(), work)
        if resumed is not None:
            frame = _frame_from_dict(resumed)
            turn["stages"]["parse"] = frame.to_dict()
            turn["stages"]["resume"] = True
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
            park_for_clarify(session, frame.to_dict(),
                             turn["answer"]["text"])
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
                # to answer: ask, don't guess. Zero writes. Parked for a
                # later resume with the missing piece.
                turn["route"] = "clarify"
                turn["answer"] = {
                    "text": "我不太确定你在问什么："
                            + "; ".join(rq.get("defects", [])) + "。"
                            "请带上属性词重问（如 材质/用途/颜色），或用"
                            "“观察 <名称> 属性=值”结构化输入。"
                            "本次未写入任何内容。",
                    "certainty": "unknown", "citations": []}
                park_for_clarify(session, frame.to_dict(),
                                 turn["answer"]["text"])
                _log(session, turn)
                return turn
            # ordinary request: one unified cognitive cycle (Phase 6H-1).
            # The controller logs exactly one experience; dialog skips its
            # own turn log on this path (no double counting).
            cyc = run_cycle(session, work, frame.to_dict())
            turn["route"] = "qa"
            turn["answer"] = cyc["answer"]
            turn["stages"]["cycle"] = {"state": cyc["state"],
                                       "stages": cyc["stages"]}
            if cyc["state"] == "answer":
                try:
                    session.pending_clarify = None
                except Exception:
                    pass
            return turn
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
            # auto-discovery (Phase 6H-2C): unknown + aligned known
            # entity/value -> propose an interpretation for confirmation.
            # Anything unalignable keeps the honest unknown answer.
            from . import constructions as _CXD
            try:
                prop = _CXD.propose_alignment(work, session.mem)
            except Exception:
                prop = None
            if prop is not None:
                session.pending_cx = {"src": work, **prop}
                turn["route"] = "propose"
                turn["answer"] = {
                    "text": f"我不太确定这句话，但其中提到了“{prop['entity']}”"
                            f"和“{prop['value']}”（已有知识中有此关联）。"
                            f"你是想说“{prop['entity']}可以{prop['value']}”吗？"
                            "输入“确认”记录这条理解，“取消”放弃；"
                            "确认前不写入任何内容。",
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


def _cx_followup(session, text: str) -> dict | None:
    """Confirm / cancel an auto-discovery proposal (Phase 6H-2C).

    确认: writes the observation through the existing hypothesis gates,
    records a TRUSTED pair (user-confirmed + KB-aligned, never a guess),
    then runs the automatic cluster/induce/validate step and reports
    what happened (candidate induced? validated? still collecting?).
    Activation always stays an explicit 构式激活 step.
    取消: drops the proposal with an audit record.
    Anything else: None (proposal abandoned, input flows through).
    """
    prop = session.pending_cx or {}
    s = text.strip()
    if s in ("确认", "确认保存", "好的", "保存", "是的", "对"):
        session.pending_cx = None
        ent, val = prop.get("entity", ""), prop.get("value", "")
        res = session.learner.learn_from_observation(
            {"name": ent, "attributes": {"function": val},
             "source": "chat-cx-confirm"})
        from . import constructions as _CXF
        _CXF.record_pair(session.mem, prop.get("src", ""), ent, val,
                         prop.get("basis", ""))
        try:
            rep = _CXF.auto_learn_step(session.mem)
        except Exception as e:  # noqa: BLE001 - report, never crash
            rep = {"clusters": 0, "induced": [], "validated": [],
                   "error": str(e)}
        lines = [f"已记录理解：{ent}可以{val}（用户确认，状态hypothesis）。"]
        if rep.get("induced"):
            lines.append("根据多条确认经验归纳出候选构式："
                         + "、".join(rep["induced"]) + "（未验证未激活）。")
        else:
            lines.append("确认经验已积累；还缺更多不同主体的例子才能归纳。")
        if rep.get("validated"):
            lines.append("以下候选已通过留存验证（仍待显式激活）："
                         + "、".join(rep["validated"]) + "。")
        if rep.get("auto_activated"):
            lines.append("以下构式已满足证据门槛并自动激活，可直接使用："
                         + "、".join(rep["auto_activated"]) + "。")
        if rep.get("conflicts"):
            lines.append("发现与已激活构式矛盾的新证据，已将其降级停用："
                         + "、".join(c["rule"] for c in rep["conflicts"])
                         + "（双方证据均保留，未做取舍）。")
        lines.append("注意：自动激活只发生在留存证据达标时；"
                     "用“构式激活 <id>”仍可手动激活已验证候选。")
        return {"text": "\n".join(lines), "certainty": "unknown",
                "citations": []}
    if s in ("取消", "算了", "不对", "不是", "不要", "放弃"):
        session.pending_cx = None
        try:
            session.mem.add_experience(
                "evaluation", "auto-discovery proposal cancelled",
                {"op": "cx_cancel", "src": prop.get("src", "")})
        except Exception:
            pass
        return {"text": "已放弃这条理解，未写入任何知识。",
                "certainty": "unknown", "citations": []}
    session.pending_cx = None
    return None


def _propose_define(session, text: str) -> dict:
    """Build a teach candidate from a definition sentence (Phase 6G-1).

    Clauses parse independently: capable->function, 是->category hint,
    有->attribute; failures stay verbatim as unclear parts. Modal
    inheritance: when a clause parsed as (non-denied) capable and the
    next clause is coordinator-led (并/且/而/还), otherwise unparsable,
    negation-free and matches [在X中]action, the modal is inherited and
    the remainder becomes its own capability -- condition preserved in
    both the action text and a structured field (schema limit noted in
    the preview). Ambiguous inheritances stay unclear. Stores
    session.pending_define, binds the name as topic, writes NOTHING.
    """
    import re as _re
    from .semantics import extract_definition, parse_sentence
    got = extract_definition(text, session._lexicon())
    name = got["name"]
    cand: dict = {"name": name, "category": got.get("category", ""),
                  "attrs": {}, "capabilities": [], "clear": [],
                  "unclear": [], "raw": text}

    def _subj_of(subj: str) -> str | None:
        if subj == name:
            return name
        if subj in ("它", "他", "她", "这", "那", "这个", "那个"):
            return name
        return None

    def _add_capability(action: str, cond: str | None, raw: str,
                        modal: str, inherited: bool) -> None:
        for c in cand["capabilities"]:
            if c["action"] == action:
                return
        cand["capabilities"].append(
            {"subject": name, "action": action, "condition": cond,
             "raw": raw, "modal": modal, "inherited": inherited})
        cand["clear"].append(raw)

    def _has_neg(s: str) -> bool:
        return any(w in s for w in ("不", "没", "别", "并非", "无法", "不能"))

    modal: str | None = None
    for cl in got.get("clauses", []):
        fr = parse_sentence(cl, source="define-clause",
                            lexicon=session._lexicon())
        handled = False
        if fr.intent == "function":
            for p in fr.predicates:
                if p.get("type") == "capable" and not p.get("denied"):
                    subj = _subj_of(p.get("subj", ""))
                    if subj and p.get("action"):
                        _add_capability(p["action"], None, cl,
                                        _cap_modal(cl), False)
                        modal = _cap_modal(cl) or modal
                        handled = True
        elif fr.intent == "attribute":
            for p in fr.predicates:
                if p.get("type") == "attr" and p.get("key") and \
                        not p.get("denied"):
                    cand["attrs"][p["key"]] = p.get("value")
                    cand["clear"].append(cl)
                    handled = True
        elif fr.intent == "classify":
            for p in fr.predicates:
                if p.get("type") == "is_a" and not p.get("denied"):
                    cand["category"] = cand["category"] or p.get("obj", "")
                    cand["clear"].append(cl)
                    handled = True
        if handled:
            continue
        # modal inheritance across 并-coordination (generic, logged).
        # Only when this clause itself parsed to nothing usable
        # (unknown): anything with its own reading keeps that reading.
        stripped = _re.sub(r"^(并|且|而|还|并且)+", "", cl).strip()
        mcond = _re.match(r"^在(.+?)中", stripped)
        cond = mcond.group(1) if mcond else None
        rest = stripped[mcond.end():].strip() if mcond else stripped
        if modal and fr.intent == "unknown" and not _has_neg(cl) and \
                len(rest) >= 2:
            _add_capability(stripped, cond, cl, modal, True)
        else:
            if cl not in cand["unclear"]:
                cand["unclear"].append(cl)
    session.pending_define = cand
    _bind_topic(session, cand["name"])
    parts = [f"我理解到一个新概念：{cand['name']}。"]
    if cand["category"]:
        parts.append(f"类别说法是：{cand['category']}（待核对，不虚构归属）。")
    for i, c in enumerate(cand["capabilities"], 1):
        line = f"能力{i}：{c['action']}"
        if c["condition"]:
            line += f"（条件：在{c['condition']}中）"
        if c["inherited"]:
            line += f"（模态“{c['modal']}”继承自上一分句）"
        parts.append(line + "。")
    for k, v in cand["attrs"].items():
        parts.append(f"属性{k}={v}。")
    if cand["unclear"]:
        parts.append("以下部分我没能可靠识别（原文保留，未解释）："
                     + "；".join(cand["unclear"]) + "。")
    parts.append("表示限制：条件只以文本+结构化字段保留，知识库尚无条件"
                 "语义，回答时会如实标注。")
    parts.append("输入“确认”保存理解到的内容，输入“取消”放弃；"
                 "用“修改 …”调整（如：修改 2=在白天释放光能）。"
                 "确认前不写入任何知识。")
    return {"text": "\n".join(parts), "certainty": "unknown",
            "citations": []}


def _cap_modal(clause: str) -> str | None:
    """Modal trigger that licensed a capable parse (generic list)."""
    import re as _re
    m = _re.search(r"(能够|可以|用来|用于|适合|能|会)", clause)
    return m.group(1) if m else None


def _define_followup(session, text: str) -> dict:
    """Confirm / cancel / amend a pending definition (Phase 6G-1)."""
    cand = session.pending_define or {}
    s = text.strip()
    if s in ("确认", "确认保存", "好的", "保存", "是的", "对"):
        return _apply_define(session)
    if s in ("取消", "算了", "不对", "不是", "不要", "放弃"):
        session.pending_define = None
        try:
            session.mem.add_experience(
                "evaluation", "define teaching cancelled by user",
                {"op": "define_cancel",
                 "name": cand.get("name", "")})
        except Exception:
            pass
        return {"text": "已取消本次教学，未写入任何知识。",
                "certainty": "unknown", "citations": []}
    body = s
    for mark in ("修改", "改成", "改为", "应该是", "改一下"):
        if body.startswith(mark):
            body = body[len(mark):].strip("，,、 ").strip()
            break
    from .semantics import ATTR_WORDS
    lex = session._lexicon()

    def _canon_key(k: str) -> str:
        if lex and k in lex:
            return lex[k]
        return ATTR_WORDS.get(k, k)

    def _split_cond(text: str) -> tuple[str, str | None]:
        import re as _re
        m = _re.match(r"^在(.+?)中", text.strip())
        if m and m.end() < len(text.strip()):
            return text.strip(), m.group(1)
        return text.strip(), None

    amended = False
    for tok in body.replace("：", "=").split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k or not v:
            continue
        import re as _re2
        mind = _re2.fullmatch(r"(?:功能|能力)?(\d+)", k)
        if mind:
            idx = int(mind.group(1)) - 1
            caps = cand.setdefault("capabilities", [])
            action, cond = _split_cond(v)
            if 0 <= idx < len(caps):
                caps[idx].update(action=action, condition=cond,
                                 raw=v, modal=None, inherited=False)
            else:
                caps.append({"subject": cand.get("name", ""),
                             "action": action, "condition": cond,
                             "raw": v, "modal": None, "inherited": False})
            amended = True
            continue
        ck = _canon_key(k)
        if ck == "function":
            action, cond = _split_cond(v)
            cand.setdefault("capabilities", []).append(
                {"subject": cand.get("name", ""), "action": action,
                 "condition": cond, "raw": v, "modal": None,
                 "inherited": False})
            amended = True
        else:
            cand["attrs"][ck] = v
            amended = True
    if not amended:
        from .semantics import parse_sentence
        fr = parse_sentence(body, source="define-amend",
                            lexicon=session._lexicon())
        for p in fr.predicates:
            if p.get("type") == "capable" and not p.get("denied") \
                    and p.get("action"):
                action, cond = _split_cond(p["action"])
                cand.setdefault("capabilities", []).append(
                    {"subject": cand.get("name", ""), "action": action,
                     "condition": cond, "raw": body, "modal": None,
                     "inherited": False})
                amended = True
            elif p.get("type") == "attr" and p.get("key") and \
                    not p.get("denied"):
                cand["attrs"][p["key"]] = p.get("value")
                amended = True
    if not amended:
        return {"text": "没能从修改中提取出属性。请用“修改 序号=新能力”"
                        "（如：修改 2=在白天释放光能）或“修改 属性=值”的格式，"
                        "或“取消”放弃。",
                "certainty": "unknown", "citations": []}
    session.pending_define = cand
    lines = [f"已更新候选：{cand['name']}。"]
    for i, c in enumerate(cand.get("capabilities", []), 1):
        line = f"  能力{i}：{c['action']}"
        if c.get("condition"):
            line += f"（条件：在{c['condition']}中）"
        lines.append(line)
    for k, v in cand["attrs"].items():
        lines.append(f"  {k}={v}")
    lines.append("输入“确认”保存，“取消”放弃。")
    return {"text": "\n".join(lines), "certainty": "unknown",
            "citations": []}


def _apply_define(session) -> dict:
    """Write a confirmed candidate through the existing gates.

    Atomicity: EVERY capability becomes its own Claim (hypothesis, with
    condition/source/raw) and its own attribute row (function for #1,
    function#N for the rest, hypothesis). Scalars go through
    learn_from_observation as before. Nothing is joined into one string;
    nothing is verified here.
    """
    cand = session.pending_define or {}
    session.pending_define = None
    name = cand.get("name", "")
    caps = list(cand.get("capabilities", []))
    scalars = dict(cand.get("attrs", {}))
    parents: list = []
    cat = (cand.get("category") or "").strip()
    if cat:
        hit = session.mem.get_concept(cat) or next(
            iter(session.mem.find_concepts_by_name(cat)), None)
        if hit:
            parents.append(hit["id"])
    first = caps[0]["action"] if caps else None
    attrs = dict(scalars)
    if first is not None:
        attrs["function"] = first
    res = session.learner.learn_from_observation(
        {"name": name, "parents": parents, "attributes": attrs,
         "source": "chat-define"})
    matches = session.mem.find_concepts_by_name(name)
    cid = matches[0]["id"] if matches else name
    note = "parents=未知(未虚构)" if not parents else f"parents={parents}"
    claim_ids: list = []
    for i, c in enumerate(caps, 1):
        key = "function" if i == 1 else f"function#{i}"
        if i > 1:
            try:
                session.onto.add_attribute(
                    cid, key, c["action"], confidence=0.55,
                    source="chat-define", status="hypothesis")
            except Exception:
                pass
        cond_txt = f"条件：在{c['condition']}中" if c.get("condition") \
            else "条件：无"
        modal_txt = f"（模态“{c['modal']}”继承）" if c.get("inherited") \
            else ""
        try:
            claim_id = session.mem.add_claim(
                f"能力:{name}#{i} {c['action']}（{cond_txt}）{modal_txt}"
                f"[用户确认定义，非独立验证事实]",
                subject_id=cid, status="hypothesis", confidence=0.5)
            session.mem.add_evidence(
                claim_id, f"原文片段：{c['raw']}", source="chat-define",
                credibility=0.5)
            claim_ids.append(claim_id)
        except Exception:
            pass
    try:
        session.mem.add_experience(
            "observation", f"define confirmed '{name}' -> {claim_ids}",
            {"op": "define_confirm", "experience_id": res.get(
                "experience_id"), "claim_ids": claim_ids,
             "capabilities": len(caps), "raw": cand.get("raw", "")})
    except Exception:
        pass
    lines = [f"[define] 已保存对'{name}'的理解（用户确认的定义，"
             f"状态hypothesis，非独立验证事实），{note}。"]
    lines.append(f"已存{len(caps)}条独立能力候选，各有来源记录。")
    if cand.get("unclear"):
        lines.append("未识别部分未写入：" + "；".join(cand["unclear"]) + "。")
    lines.append("可用“光盒有什么作用”这类问法查询；回答会标注来源。")
    _bind_topic(session, name)
    return {"text": "\n".join(lines), "certainty": "unknown",
            "citations": []}


def _amend_saved(session, body: str) -> dict | None:
    """Amend one persisted capability (Phase 6G-1收尾).

    Forms: 修改光盒2=新能力 / 修改功能2=新能力 (topic concept).
    Returns None when the input is not such an amendment (caller falls
    through to normal routing). Only the addressed row + claim change;
    status/source of every row stay untouched.
    """
    import re as _re
    m = _re.match(r"^(?:功能|能力)(\d+)\s*=\s*(.+)$", body)
    name, idx, new = None, None, None
    if m:
        try:
            st = session.mem.get_context(session.session_id)
            ents = st.get("entities", [])
            name = ents[0].get("text", "") if ents else ""
        except Exception:
            name = ""
        idx, new = int(m.group(1)), m.group(2).strip()
    else:
        m2 = _re.match(r"^(.+?)(\d+)\s*=\s*(.+)$", body)
        if not m2:
            return None
        name, idx, new = m2.group(1).strip(), int(m2.group(2)), \
            m2.group(3).strip()
    if not name or not new or idx < 1:
        return None
    hit = session.mem.get_concept(name) or next(
        iter(session.mem.find_concepts_by_name(name)), None)
    if hit is None:
        return None
    cid = hit["id"]
    key = "function" if idx == 1 else f"function#{idx}"
    rows = [a for a in session.mem.get_attributes(cid)
            if a.get("key") == key]
    if not rows:
        return None
    other_keys = sorted({a.get("key") for a in
                         session.mem.get_attributes(cid)
                         if str(a.get("key", "")).startswith("function")})
    session.mem.update_attribute_value(rows[0]["id"], new)
    n_updated_claims = 0
    for cl in session.mem.list_claims():
        if cl.get("text", "").startswith(f"能力:{name}#{idx} "):
            cond_m = _re.match(r"^在(.+?)中", new.strip())
            cond_txt = f"（条件：在{cond_m.group(1)}中）" if cond_m and \
                cond_m.end() < len(new.strip()) else "（条件：无）"
            session.mem.update_claim_text(
                cl["id"], f"能力:{name}#{idx} {new}{cond_txt}"
                f"[用户确认定义+修正，非独立验证事实]")
            n_updated_claims += 1
    try:
        session.mem.add_experience(
            "evaluation", f"define amend saved {name}#{idx}",
            {"op": "define_amend_saved", "concept": cid, "key": key,
             "claims": n_updated_claims})
    except Exception:
        pass
    return {"text": f"已修正{name}的能力{idx}为：{new}。"
                    f"其他能力（{', '.join(other_keys)}）未动，"
                    f"验证状态不变。",
            "certainty": "unknown", "citations": []}


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


def _log(session, turn: dict) -> None:
    try:
        session.mem.add_experience(
            "evaluation", f"dialog turn route={turn['route']}",
            {"op": "dialog", "route": turn["route"],
             "stages": list(turn["stages"])})
    except Exception:
        pass
