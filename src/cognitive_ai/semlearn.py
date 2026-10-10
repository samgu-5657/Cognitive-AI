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


def check_vocab_consistency(rule: dict) -> dict:
    """Kind-vs-affix cross-check (Phase 6F, Task 2/3).

    The rule claims affix (pre/suf) expresses (kind,key). If the affix
    itself contains a KNOWN vocabulary word for a different key, the
    label contradicts established evidence:
      affix has 材质/材料... (ATTR_WORDS -> material) but rule is not
        attr_q/material  -> conflict;
      affix has 用途/用处/功能 (function words) but rule key is not
        function          -> conflict;
      affix names the has-feature itself (把手 in affix, feature 把手)
        -> consistent.
    Affixes with no known words carry no signal -> pass with a note
    (honest limit: form-consistent mislabels without known words are
    NOT detectable; the report says so instead of claiming otherwise).
    """
    from .semantics import ATTR_WORDS
    affix = (rule.get("pre", "") or "") + "|" + (rule.get("suf", "") or "")
    kind, key = rule.get("kind"), rule.get("key")
    for word, mapped in ATTR_WORDS.items():
        if word and word in affix:
            # the affix names a known attribute word: the rule key must
            # be exactly what that word means (用途->function,
            # 材质->material, ...), whatever the claimed kind.
            if key == mapped:
                return {"ok": True,
                        "reason": f"affix names {word}->{mapped}, matches"}
            if kind == "has_q" and key and key in affix:
                return {"ok": True,
                        "reason": "affix consistent with has-feature"}
            return {"ok": False,
                    "reason": f"label conflict: affix contains known word "
                    f"{word!r} (-> {mapped}) but rule claims "
                    f"{kind}/{key}"}
    if kind == "has_q" and key and key in affix:
        return {"ok": True, "reason": "affix names the feature itself"}
    return {"ok": True,
            "reason": "no known vocabulary in affix; consistency "
            "undecidable from form alone (mislabels here are NOT "
            "detectable -- reported, not claimed)"}


def validate_rule(rule: dict, heldout: list[dict],
                  contrast: list | None = None) -> dict:
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
    if not all(d["ok"] for d in detail):
        return {"passed": False, "checked": len(detail), "detail": detail,
                "contrast": [], "reason": "held-out mismatch"}
    # contrast silence: other-kind held-outs must NOT fire this rule.
    # A rule that also accepts a different question type is ambiguous
    # and stays a candidate until more evidence arrives.
    contra = []
    for c in contrast or []:
        hit = construct(c["src"], [{**rule, "status": "active"}])
        if hit is not None:
            contra.append({"src": c["src"], "rule": rule["id"],
                           "fired_kind": rule["kind"]})
    if contra:
        return {"passed": False, "checked": len(detail), "detail": detail,
                "contrast": contra,
                "reason": "contrast conflict: rule also fires on "
                "other-kind probes; needs more evidence"}
    return {"passed": True, "checked": len(detail), "detail": detail,
            "contrast": [], "reason": ""}


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
