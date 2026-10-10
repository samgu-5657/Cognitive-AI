"""Direct semantic-frame learning (Phase 6E).

Unlike phraselearn.py (which rewrites utterances into canonical sentences
for the fixed parser), this module CONSTRUCTS SemanticFrame-compatible
dicts straight from learned (affix, kind, key) structures. It NEVER calls
parse_sentence on a rewritten sentence: the frame below is built field by
field, in the exact shapes semantics.py itself emits, and every frame
carries a `learned_semantic_frame:<rule_id>` trace entry as provenance.

Scope: the three query kinds the system already defines --
  function_q (predicate query_function, key function),
  attr_q     (predicate query_attr, key e.g. material),
  has_q      (predicate has, feature e.g. 把手/has_handle).
No new frame schema: outputs must satisfy resolve_question() with the
same kind/key the label claims, or validation fails.

Gates (same principles as phrase/rule learning):
- one example -> stored example only, never a generalizable claim;
- >=2 examples with DISTINCT entities -> slot candidate (the slot must
  EQUAL the labeled entity: role check, not substring coincidence);
- held-out examples (disjoint entities) must reproduce kind/entity/key
  exactly; 100% required -> validated;
- only explicit activation makes a rule usable; callers try it ONLY when
  the fixed parser returns unknown (never overrides a confident parse);
- several matching rules, empty slot, or failed role check -> abstain.
A constructed frame is a QUERY: it can only read knowledge through the
existing executor gates and can never write verified facts by itself.
"""
from __future__ import annotations

import time

KINDS = ("function_q", "attr_q", "has_q")

_counter = 0


def new_rule_id(name: str = "sem") -> str:
    global _counter
    _counter += 1
    return f"sr_{name}_{int(time.time() * 1000) % 100000}_{_counter}"


def check_label(kind: str, entity: str, key: str) -> dict:
    """Validate a taught label against the canonical definitions."""
    if kind not in KINDS:
        return {"ok": False,
                "reason": f"kind must be one of {KINDS}, got {kind!r}"}
    if not entity.strip():
        return {"ok": False, "reason": "entity required"}
    if kind == "function_q" and key != "function":
        return {"ok": False, "reason": "function_q requires key=function"}
    if kind in ("attr_q", "has_q") and not key.strip():
        return {"ok": False, "reason": f"{kind} requires a key"}
    return {"ok": True}


def build_frame(kind: str, entity: str, key: str, raw: str,
                rule_id: str) -> dict:
    """Construct the frame directly. No parsing involved."""
    ent = {"text": entity, "role": "subject"}
    if kind == "function_q":
        preds = [{"type": "query_function", "subj": entity}]
    elif kind == "attr_q":
        preds = [{"type": "query_attr", "subj": entity, "key": key}]
    else:  # has_q
        preds = [{"type": "has", "subj": entity, "feature": key}]
    return {"intent": "query", "entities": [ent], "predicates": preds,
            "modality": "unknown", "scope": "unknown", "source": "semlearn",
            "confidence": 0.7, "unresolved_references": [],
            "ambiguities": [], "raw": raw,
            "trace": [f"learned_semantic_frame:{rule_id}"]}


def _common_affix(texts: list[str]) -> tuple[str, str]:
    pre, suf = texts[0], texts[0]
    for t in texts[1:]:
        i = 0
        while i < min(len(pre), len(t)) and pre[i] == t[i]:
            i += 1
        pre = pre[:i]
        j = 0
        while j < min(len(suf), len(t)) and suf[-1 - j] == t[-1 - j]:
            j += 1
        suf = suf[len(suf) - j:] if j else ""
    if pre and suf and len(pre) + len(suf) >= min(len(t) for t in texts):
        return "", ""
    return pre, suf


def induce(examples: list[dict]) -> dict:
    """Induce slot candidates from labeled examples.

    Example: {"src","kind","entity","key"}. Groups by (kind,key); a slot
    needs >=2 examples with DISTINCT entities whose labeled entity EQUALS
    the affix middle (role check). Anything weaker stays un-induced with
    a recorded reason -- never a silent literal generalization.
    """
    groups: dict[tuple, list[dict]] = {}
    rejected = []
    for ex in examples:
        chk = check_label(ex.get("kind", ""), ex.get("entity", ""),
                          ex.get("key", ""))
        if not chk["ok"]:
            rejected.append({"example": ex, **chk})
            continue
        if not ex.get("src", "").strip():
            rejected.append({"example": ex, "ok": False,
                             "reason": "empty src"})
            continue
        groups.setdefault((ex["kind"], ex["key"]), []).append(ex)
    candidates = []
    for (kind, key), rows in groups.items():
        ents = {r["entity"].strip() for r in rows}
        if len(rows) < 2 or len(ents) < 2:
            rejected.append({"kind": kind, "key": key, "ok": False,
                             "reason": "needs >=2 examples with distinct "
                             "entities for a generalizable structure; "
                             "examples kept, nothing induced"})
            continue
        srcs = [r["src"].strip() for r in rows]
        pre, suf = _common_affix(srcs)
        if not (pre or suf):
            rejected.append({"kind": kind, "key": key, "ok": False,
                             "reason": "no shared affix"})
            continue
        mids = [s[len(pre):len(s) - len(suf) if suf else len(s)]
                for s in srcs]
        # role check: the middle must BE the labeled entity
        if not all(m == r["entity"].strip()
                   for m, r in zip(mids, rows)):
            rejected.append({"kind": kind, "key": key, "ok": False,
                             "reason": "affix middle is not the labeled "
                             "entity; no slot induced"})
            continue
        candidates.append({
            "kind": kind, "key": key, "pre": pre, "suf": suf,
            "confidence": 0.7,
            "support": [{k: r[k] for k in ("src", "entity")} for r in rows],
            "note": f"slot from {len(rows)} examples: {pre!r} + ENTITY "
                    f"+ {suf!r} -> {kind}/{key}"})
    return {"candidates": candidates, "groups": len(groups),
            "rejected": rejected}


def construct(text: str, rules: list[dict]) -> dict | None:
    """Build a frame directly from active rules, or abstain (None).

    Never rewrites-and-reparses: the ONLY frame source here is
    build_frame(). Several matching rules -> abstain (ambiguity).
    """
    s = text.strip()
    hits = []
    for r in rules:
        if r.get("status") != "active":
            continue
        pre, suf = r.get("pre", ""), r.get("suf", "")
        if not s.startswith(pre):
            continue
        if suf and not s.endswith(suf):
            continue
        mid = s[len(pre):len(s) - len(suf) if suf else len(s)]
        if not mid:
            continue
        hits.append((r.get("kind"), r.get("key"), r, mid))
    if not hits:
        return None
    if len({(k, kk) for k, kk, _, _ in hits}) > 1:
        return None  # two readings, two kinds: abstain, never guess
    kind, key, rule, mid = hits[0]
    return build_frame(kind, mid, key, text, rule["id"])


def validate_rule(rule: dict, heldout: list[dict]) -> dict:
    """Held-out check on raw labels: kind/entity/key must match exactly."""
    if not heldout:
        return {"passed": False, "reason": "no held-out examples",
                "checked": 0}
    detail = []
    for h in heldout:
        fr = construct(h["src"], [{**rule, "status": "active"}])
        ok = fr is not None and any(
            p.get("type") in ("query_function", "query_attr", "has")
            for p in fr["predicates"]) and \
            (fr["entities"] and
             fr["entities"][0].get("text") == h["entity"].strip())
        if ok:
            from .semantics import resolve_question
            rq = resolve_question(fr, {})
            ok = rq.get("kind") == rule["kind"] and \
                rq.get("key") == rule["key"]
        detail.append({"src": h["src"], "ok": bool(ok)})
    passed = all(d["ok"] for d in detail)
    return {"passed": passed, "checked": len(detail), "detail": detail,
            "reason": "" if passed else "held-out mismatch"}


def activate_rule(mem, rule_id: str) -> dict:
    row = mem.get_sem_rule(rule_id)
    if not row:
        return {"ok": False, "error": "rule not found"}
    if row["status"] != "validated":
        return {"ok": False,
                "error": f"only validated rules activate (now "
                f"{row['status']})"}
    row["status"] = "active"
    mem.save_sem_rule(row)
    mem.add_experience("evaluation", f"sem rule {rule_id} activated",
                       {"op": "sem_activate", "rule_id": rule_id})
    return {"ok": True, "rule_id": rule_id}
