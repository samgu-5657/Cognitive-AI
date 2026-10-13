"""Deterministic Chinese answer realization (Phase M3E).

Builds AnswerSemantics ONLY from an answer_question() verdict dict,
then composes fixed inventories keyed by semantic fields (subject,
relation, object, polarity, modality, status, evidence kind) --
never per-sentence templates. roundtrip() re-parses supported kinds
with the existing parser and compares roles; anything else is
reported, never faked.
"""
from __future__ import annotations


def build_answer_semantics(verdict: dict) -> dict:
    """Verdict dict -> AnswerSemantics dict. No text invented here."""
    q = verdict.get("question", {}) or {}
    order = {"verified": 4, "inherited(verified)": 4, "hypothesis": 3,
             "inherited(hypothesis)": 3, "contested": 2,
             "inherited(contested)": 2, "inherited": 1}
    best, rank = "unknown", -1
    for e in verdict.get("evidence", []) or []:
        r = order.get(str(e.get("status", "")), 0)
        if r > rank:
            rank, best = r, str(e.get("status", "unknown"))
    queries = verdict.get("queries", []) or []
    limited = bool(verdict.get("limited", False) or any(
        isinstance(qq, dict) and qq.get("op") == "chain_limited"
        for qq in queries))
    sem = {
        "kind": q.get("qkind", "unknown"),
        "subject": q.get("subject", ""),
        "key": q.get("key", ""),
        "object": q.get("object", ""),
        "expected": q.get("expected", ""),
        "negated": bool(q.get("negated", False)),
        "modal": q.get("modal"),
        "is_yesno": bool(q.get("is_yesno", False)),
        "verdict": verdict.get("verdict", "unknown"),
        "answer": verdict.get("answer", ""),
        "evidence": verdict.get("evidence", []) or [],
        "queries": list(queries),
        "detail": verdict.get("detail", ""),
        "ambiguity": verdict.get("ambiguity", ""),
        "status": best,
        "limited": limited,
    }
    return sem


def _answer_empty(ans) -> bool:
    return ans is None or (isinstance(ans, str) and ans.strip() == "")


def _status_note(evidences: list) -> str:
    sts = {str(e.get("status", "")) for e in evidences}
    if "verified" in sts or "inherited(verified)" in sts:
        return "已验证"
    if "contested" in sts or "inherited(contested)" in sts:
        return "存疑"
    if "hypothesis" in sts or "inherited(hypothesis)" in sts:
        return "暂定假设，未验证"
    if "inherited" in sts:
        # Bare "inherited" marks a transitive category step whose own
        # credibility is not stored (parent links carry no status):
        # report provenance, never upgrade to verified.
        return "通过类别继承得到"
    return "未知"


def _ev_src(evidences: list) -> str:
    kinds = []
    for e in evidences:
        k = e.get("kind")
        if k == "rule":
            kinds.append("规则推导")
        elif k in ("attribute", "parents", "is_a", "relation"):
            kinds.append("知识库记录")
    seen = list(dict.fromkeys(kinds))
    return "＋".join(seen) if seen else "暂无直接证据"


def realize(sem: dict) -> dict:
    """Compose Chinese text from AnswerSemantics. Returns
    {text, constructions[], roundtrippable}."""
    used = []
    v = sem.get("verdict", "unknown")
    limited = bool(sem.get("limited", False))
    if v == "ambiguous":
        return {"text": "没能确定你指的是哪个，请澄清：" +
                (sem.get("ambiguity") or "主体或对象不唯一") + "。",
                "constructions": ["clarify:ambiguity"],
                "roundtrippable": False}
    if v == "failure":
        return {"text": "查询执行失败：" +
                (sem.get("detail") or "内部错误") + "。未写入任何内容。",
                "constructions": ["failure:engine"],
                "roundtrippable": False}
    if v == "supported" and _answer_empty(sem.get("answer", "")):
        if limited:
            return {"text": "推理搜索受限，尚不能确定；查询结果也不完整，"
                    "未能形成确定结论，未作断言。",
                    "constructions": ["guard:empty-answer", "limit:budget"],
                    "roundtrippable": False}
        return {"text": "查询结果不完整，未能形成确定结论，暂不作断言。",
                "constructions": ["guard:empty-answer"],
                "roundtrippable": False}
    if v == "unknown":
        if limited:
            return {"text": "推理搜索受限，尚不能确定"
                    + (sem.get("subject") or "相关情况")
                    + "的情况。未证明不存在，也未完成验证。",
                    "constructions": ["limit:budget"],
                    "roundtrippable": False}
        return {"text": "不知道" +
                (sem.get("subject") or "这个") + "的相关情况。证据不足。",
                "constructions": ["unknown:greeting-gap"],
                "roundtrippable": False}
    if v == "conflicted":
        return {"text": "证据冲突，双方都保留："
                + (sem.get("answer") or sem.get("detail") or "结论未定")
                + "。",
                "constructions": ["conflict:both-kept"],
                "roundtrippable": False}
    # supported from here on (non-empty answer guaranteed by guard above)
    out = _realize_supported(sem, used)
    if limited:
        out = dict(out)
        out["text"] = out.get("text", "") + "注意：推理搜索受限，结论可能不完整，尚不能确定。"
        out["constructions"] = list(out.get("constructions", [])) + ["limit:budget"]
    return out


def _realize_supported(sem: dict, used: list) -> dict:
    s, o = sem.get("subject", ""), sem.get("object", "")
    note = _status_note(sem.get("evidence", []))
    src = _ev_src(sem.get("evidence", []))
    tail = f"（{note}，依据：{src}）"
    kind = sem.get("kind", "")
    neg = sem.get("negated", False)
    yesno = sem.get("is_yesno", False)
    if kind == "category" and not yesno:
        used.append("fact:category-list")
        if any(str(e.get("status", "")) == "inherited"
               for e in sem.get("evidence", [])):
            used.append("inheritance:transitive")
        return {"text": f"{s}属于{sem.get('answer', '')}。{tail}",
                "constructions": used, "roundtrippable": False}
    if kind == "is_a_verify":
        used.append("verify:is_a")
        if sem.get("answer") == "是" and not neg:
            return {"text": f"是的，{s}是{o}。{tail}",
                    "constructions": used, "roundtrippable": True}
        if (sem.get("answer") == "否" and not neg) or neg:
            return {"text": f"不，{s}不是{o}。{tail}",
                    "constructions": used, "roundtrippable": True}
    if kind == "capability" and not yesno:
        used.append("fact:capable")
        val = sem.get("answer") or o
        if neg:
            return {"text": f"{s}不能{val}。{tail}",
                    "constructions": used, "roundtrippable": True}
        return {"text": f"{s}可以{val}。{tail}",
                "constructions": used, "roundtrippable": True}
    if kind == "capability" and yesno:
        used.append("verify:capable")
        if neg:
            return {"text": f"{s}不能{o}。{tail}",
                    "constructions": used, "roundtrippable": True}
        ans = sem.get("answer", "")
        if ans == "是":
            return {"text": f"是的，{s}可以{o}。{tail}",
                    "constructions": used, "roundtrippable": True}
        return {"text": f"不，{s}不可以{o}。{tail}",
                "constructions": used, "roundtrippable": True}
    if kind == "has" and not yesno:
        used.append("fact:has-list")
        return {"text": f"{s}有记录的方面：{sem.get('answer', '')}。{tail}",
                "constructions": used, "roundtrippable": False}
    if kind == "has" and yesno:
        used.append("verify:has")
        if neg:
            return {"text": f"{s}没有{o}。{tail}",
                    "constructions": used, "roundtrippable": True}
        if sem.get("answer") == "是":
            return {"text": f"是的，{s}有{o}。{tail}",
                    "constructions": used, "roundtrippable": True}
        return {"text": f"不，{s}没有{o}。{tail}",
                "constructions": used, "roundtrippable": True}
    if kind == "attr_open":
        used.append("fact:attr")
        # o already is the Chinese attribute word (材质/用途/…);
        # no key mapping is invented here.
        disp = o
        st = sem.get("status", "unknown")
        if st == "verified":
            return {"text": f"{s}的{disp}是{sem.get('answer', '')}。"
                            f"{tail}",
                    "constructions": used, "roundtrippable": False}
        return {"text": f"{s}的{disp}可能是{sem.get('answer', '')}。"
                        f"{tail}",
                "constructions": used, "roundtrippable": False}
    used.append("fallback:supported-generic")
    return {"text": f"关于{s}：{sem.get('answer', '')}。{tail}",
            "constructions": used, "roundtrippable": False}


def _strip_answer_particles(text: str) -> str:
    for head in ("是的，", "是,", "不，", "不,"):
        if text.startswith(head):
            return text[len(head):].strip()
    return text


def check_roundtrip(text: str, mem, expect: dict) -> dict:
    """Re-parse generated text, compare roles against expectation.

    Returns {ok: True|False|None, detail}. None = kind not covered by
    both parser and realizer (honest limit, not a pass).
    Expected keys: subject/predicate/object/negated/modal.
    """
    from .semparse import parse_question, parse_statement
    main = _strip_answer_particles(text.split("。")[0])
    fr = parse_statement(main, mem).to_dict()
    if fr.get("predicate", "unknown") == "unknown":
        frq = parse_question(main, mem).to_dict()
        got = {"subject": frq.get("subject", ""),
               "predicate": frq.get("qkind", ""),
               "object": frq.get("object", ""),
               "negated": bool(frq.get("negated", False)),
               "modal": frq.get("modal")}
    else:
        got = {"subject": fr.get("subject", ""),
               "predicate": fr.get("predicate", ""),
               "object": fr.get("object", ""),
               "negated": bool(fr.get("negated", False)),
               "modal": fr.get("modal")}
    diffs = [k for k in ("subject", "predicate", "object", "negated",
                         "modal") if got.get(k) != expect.get(k)]
    if diffs:
        return {"ok": False, "detail": f"role mismatch: {diffs}",
                "got": got}
    return {"ok": True, "detail": "roles match", "got": got}
