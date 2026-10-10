"""Bidirectional compositional verbalization (Phase 6H-2A).

Input direction (utt -> roles) already exists: semantics.parse_sentence
+ semlearn.construct + resolve_question. This module adds the reverse
(roles -> utt) over the SAME role inventory, so one learned regularity
serves both directions:

- A sem rule (kind, key, pre, suf) learned from teaching renders a
  question surface pre+ENTITY+suf for any entity (source: learned).
- A small FIXED canonical inventory renders role-faithful Chinese for
  facts and questions (source: fixed). Fixed = documented protocol,
  never extended per sentence: adding a new canonical form needs a
  code change plus a test, exactly like parser patterns.

Faithfulness rules (no fabrication for fluency):
- hypothesis rows render as 可能/暂定假设, never as verified fact;
- contested rows render both sides;
- denied renders 不是/没有; conditional text passes through verbatim
  and is labeled as condition (schema limit, same as teaching);
- unknown key with no value and no rule -> abstain (None), never an
  invented sentence.
"""
from __future__ import annotations

KEY_ZH = {"function": "用途", "material": "材质", "color": "颜色",
          "shape": "形状", "location": "位置", "size": "尺寸",
          "temperature": "温度", "weight": "重量",
          "has_handle": "把手", "can_contain": "容纳能力"}


def zh_key(key: str) -> str:
    import re as _re
    m = _re.fullmatch(r"function#(\d+)", str(key))
    if m:
        return f"用途{m.group(1)}"
    return KEY_ZH.get(str(key), str(key))


def render_question(kind: str, entity: str, key: str | None,
                    rules: list | None = None,
                    exclude: str = "") -> list[dict]:
    """Question surfaces for (kind, entity, key). Returns a list of
    {text, source} with source in {learned, fixed}. Empty list means
    abstain: no known surface for this question."""
    out: list[dict] = []
    for r in rules or []:
        if r.get("status") != "active":
            continue
        if (r.get("kind"), r.get("key")) != (kind, key):
            continue
        text = f"{r.get('pre', '')}{entity}{r.get('suf', '')}"
        if text and text != exclude:
            out.append({"text": text, "source": f"learned:{r['id']}"})
    fixed = _canonical_question(kind, entity, key)
    if fixed and all(f["text"] != fixed for f in out):
        out.append({"text": fixed, "source": "fixed"})
    return out


def _canonical_question(kind: str, entity: str,
                        key: str | None) -> str | None:
    if not entity:
        return None
    if kind == "function_q":
        return f"{entity}有什么用途"
    if kind == "attr_q" and key == "material":
        return f"{entity}的材质是什么"
    if kind == "attr_q" and key and key in KEY_ZH:
        return f"{entity}的{KEY_ZH[key]}是什么"
    if kind == "has_q" and key:
        return f"{entity}有{key}吗"
    return None


def render_fact(concept: str, key: str, value, status: str,
                denied: bool = False, condition: str | None = None,
                ) -> dict | None:
    """Fact surface for one attribute row. Returns {text, source} or
    None (abstain) when there is nothing statable."""
    if value is None:
        return None
    k = zh_key(key)
    base = f"{concept}的{k}是{value}"
    if denied:
        text = f"{concept}的{k}不是{value}。(已记录否定，未写入属性)"
        return {"text": text, "source": "fixed"}
    if condition:
        base += f"（条件：在{condition}中）"
    if status == "verified":
        return {"text": base + "。(已验证)", "source": "fixed"}
    if status == "contested":
        return {"text": base + "。(存疑：冲突双方已保留，未定)",
                "source": "fixed"}
    return {"text": f"{concept}的{k}可能是{value}。(暂定假设，未验证)",
            "source": "fixed"}


def roundtrip_kind(text: str, lexicon: dict | None = None,
                   rules: list | None = None) -> dict:
    """Parse a generated surface back to (kind, key, entity) for the
    role-preservation check. Uses only existing machinery: direct parse
    first, then the same semlearn fallback dialog.process itself uses
    (learned surfaces are only parseable with their rules active)."""
    from .semantics import parse_sentence, resolve_question
    fr = parse_sentence(text, source="verbalize-check",
                        lexicon=lexicon or {}).to_dict()
    if fr.get("intent") == "unknown" and rules:
        from . import semlearn as _SL
        try:
            built = _SL.construct(text, rules)
        except Exception:
            built = None
        if built is not None:
            fr = built
    rq = resolve_question(fr, lexicon or {})
    ents = [e.get("text", "") for e in fr.get("entities", [])]
    kind, key = rq.get("kind"), rq.get("key")
    entity = ents[0] if ents else ""
    if kind == "classify_q" and "的" in entity:
        # same possessive normalization the dialogue path uses, so a
        # fixed surface round-trips to the roles it was rendered from.
        from .semantics import ATTR_WORDS
        left, right = entity.rsplit("的", 1)
        mapped = ATTR_WORDS.get(right, (lexicon or {}).get(right))
        if left and mapped:
            kind, entity, key = "attr_q", left, mapped
    return {"kind": kind, "key": key, "entity": entity,
            "intent": fr.get("intent")}
