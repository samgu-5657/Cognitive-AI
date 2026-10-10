"""Answer organization layer (Phase 6B, Task 6).

Renders structured explainer output into readable Chinese WITHOUT
inventing facts: every sentence template fills slots from citations /
provenance. Display dictionary maps internal keys to Chinese for
readability; unknown tokens pass through verbatim with no guessing.
"""
from __future__ import annotations

KEY_ZH = {"function": "功能", "material": "材质", "color": "颜色",
          "shape": "形状", "location": "位置", "size": "尺寸",
          "temperature": "温度", "weight": "重量", "has_handle": "把手",
          "can_contain": "容纳能力"}
VAL_ZH = {"drinking": "饮用", "ceramic": "陶瓷", "glass": "玻璃",
          "true": "有", "false": "无", "holding": "盛放"}


def zh_key(k) -> str:
    return KEY_ZH.get(str(k), str(k))


def zh_val(v) -> str:
    if isinstance(v, bool):
        return VAL_ZH["true"] if v else VAL_ZH["false"]
    return VAL_ZH.get(str(v), str(v))


def render(explainer_out: dict, trace: dict) -> dict:
    """Translate one explainer result. Same facts in -> same facts out
    (wording may differ, facts never)."""
    certainty = explainer_out.get("certainty", "unknown")
    task = (trace.get("task") or {})
    if task.get("status") == "inconclusive":
        return {"text": "证据不足，结论未定：" + explainer_out.get("text", ""),
                "certainty": "unknown",
                "citations": explainer_out.get("citations", []),
                "provenance": explainer_out.get("provenance")}
    if certainty == "unknown" or task.get("status") != "verified":
        return {"text": explainer_out.get("text", ""),
                "certainty": "unknown",
                "citations": explainer_out.get("citations", []),
                "provenance": explainer_out.get("provenance")}
    text = explainer_out.get("text", "")
    # localize known internal tokens; leave the rest untouched
    for en, zh in list(KEY_ZH.items()) + list(VAL_ZH.items()):
        text = text.replace(f"的{en}是", f"的{zh}是").replace(
            f"={en}", f"={zh}")
    if certainty == "qualified":
        text += "（注意：部分结论来自继承或假设，不是直接验证的事实。）"
    return {"text": text, "certainty": certainty,
            "citations": explainer_out.get("citations", []),
            "provenance": explainer_out.get("provenance")}


def render_form(kind: str, facts: list) -> str:
    """Basic answer forms built ONLY from supplied fact dicts."""
    if kind == "definition" and facts:
        f = facts[0]
        return f"{f.get('concept', '?')}是{f.get('value', '?')}。"
    if kind == "use" and facts:
        vals = "、".join(str(f.get("value", "?")) for f in facts)
        return f"用途是：{vals}。"
    if kind == "steps" and facts:
        return "\n".join(f"{i + 1}. {f.get('rule', '?')}："
                         f"{f.get('before', '?')} → {f.get('after', '?')}"
                         for i, f in enumerate(facts))
    if kind == "compare" and len(facts) >= 2:
        return (f"{facts[0].get('concept', '?')}的"
                f"{facts[0].get('key', '?')}是{facts[0].get('value', '?')}，"
                f"而{facts[1].get('concept', '?')}的是"
                f"{facts[1].get('value', '?')}。")
    if kind == "reason" and facts:
        return "因为：" + "；".join(str(f.get("value", "?")) for f in facts) \
            + "。"
    return "证据不足，无法组织答案。"
