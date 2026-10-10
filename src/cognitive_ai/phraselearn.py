"""Learnable expression -> canonical-form mappings (Phase 6C).

Problem: the compositional parser (semantics.py) only understands fixed
slot patterns. New phrasings of already-supported meanings (e.g. an
imperative "告诉我X的用途" for the known "X有什么用途") are rejected as
unknown. This module learns REWRITES from user-taught (src => dst) pairs:

- dst must already parse (rule-free) to a supported frame; otherwise the
  example is rejected with a reason -- the system never learns a mapping
  into an unparseable or out-of-scope meaning.
- >=2 examples sharing one dst-frame signature induce a slot template
  from the common prefix/suffix; the varying middle must carry over into
  dst (it is the entity slot). 1 example yields a literal mapping only.
- Held-out examples (never used for induction) must reproduce the dst
  frame signature exactly; 100% required, then status validated.
- Only explicit activation makes a rule usable by the parser. Learned
  rules apply as a FALLBACK (direct parse succeeded -> untouched), at
  most once, and ambiguous matches abstain instead of guessing.

Why not a neural net here: a handful of user examples cannot train a
from-scratch model (see neural/ baselines: hundreds of samples for
0.42 accuracy); affix induction is inspectable, persistent, gated, and
its wins/losses are measurable in tests/test_phrase_generalize.py.
"""
from __future__ import annotations

import json

ALLOWED_DST_INTENTS = ("query", "attribute", "function", "classify",
                       "observe")

_counter = 0


def new_rule_id(name: str = "phrase") -> str:
    global _counter
    _counter += 1
    import time
    return f"pr_{name}_{int(time.time() * 1000) % 100000}_{_counter}"


def _sig_key(sig: dict) -> str:
    """Canonical signature comparison key.

    JSON-normalized (tuples become arrays on DB round-trips), so a
    signature computed fresh from induce() compares equal to the same
    signature loaded back from storage. repr() would NOT survive that
    round-trip -- this exact bug was caught by the held-out gate."""
    return json.dumps(sig, sort_keys=True, ensure_ascii=False, default=str)


def parse_for_filter(dst: str, lexicon: dict | None) -> dict:
    """Rule-free dst signature for matching held-out examples to rules."""
    return frame_signature(_parse_rule_free(dst.strip(), lexicon).to_dict())


def consider_learned_override(direct: dict, text: str, rules: list[dict],
                              lexicon: dict | None = None) -> dict | None:
    """Task-3 gate: may a learned rewrite replace a DIRECT parse?

    Conditions (all required):
    1. the direct frame admits doubt (non-empty ambiguities);
    2. at least one rule is active AND validated-or-active... (active);
    3. the rewrite parses rule-free with NO ambiguities;
    4. the rewrite reproduces a frame the current parser understands.
    Two reasonable readings -> None (caller keeps direct + its doubt
    note, i.e. clarify/abstain downstream). Direct parses without doubt
    are NEVER overridden -- prove it in tests, not prose.
    Returns the rewritten frame dict (with override trace) or None.
    """
    from .semantics import parse_sentence as _parse
    lexicon = lexicon or {}
    if not direct.get("ambiguities"):
        return None
    tried = apply_rules(text.strip(), [r for r in rules
                                       if r.get("status") == "active"],
                        lexicon)
    if not tried.get("rewritten"):
        return None  # includes the disagree-abstain case
    inner = _parse(tried["rewritten"], "phrase-override", lexicon, None)
    if inner.intent == "unknown" or inner.ambiguities:
        return None
    d = inner.to_dict()
    d["trace"] = list(d.get("trace", [])) + [
        f"phrase_override:{tried['rule_id']} replaced doubtful direct "
        f"parse (doubt was: {direct.get('ambiguities')})"]
    d["confidence"] = round(max(0.0, float(d.get("confidence", 0)) - 0.1),
                            3)
    return d


def frame_signature(frame: dict) -> dict:
    """Comparable meaning skeleton: intent + predicate types/keys.

    Entity TEXT is excluded on purpose: different objects asking the
    same kind of question share one signature (that is the generalization
    we want to measure)."""
    preds = []
    for p in frame.get("predicates", []) or []:
        preds.append((p.get("type"), p.get("key"), p.get("action"),
                      p.get("feature")))
    return {"intent": frame.get("intent"),
            "modality": frame.get("modality"),
            "preds": sorted(preds, key=str)}


def _parse_rule_free(text: str, lexicon: dict | None):
    from .semantics import parse_sentence
    return parse_sentence(text, source="phrase-teach", lexicon=lexicon,
                          phrase_rules=None)


def check_example(src: str, dst: str, lexicon: dict | None) -> dict:
    """Validate one taught pair without storing anything."""
    if not src.strip() or not dst.strip():
        return {"ok": False, "reason": "empty src or dst"}
    if src.strip() == dst.strip():
        return {"ok": False, "reason": "src and dst identical"}
    fr = _parse_rule_free(dst.strip(), lexicon)
    if fr.intent == "unknown":
        return {"ok": False, "reason": "dst not parseable by current parser",
                "dst_intent": fr.intent}
    if fr.intent not in ALLOWED_DST_INTENTS:
        return {"ok": False, "reason": f"dst intent {fr.intent} out of "
                "learning scope (query/attribute/function/classify/observe)",
                "dst_intent": fr.intent}
    return {"ok": True, "signature": frame_signature(fr.to_dict())}


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
        # affixes overlap: no room for a slot
        return "", ""
    return pre, suf


def induce(examples: list[dict], lexicon: dict | None = None) -> dict:
    """Induce candidate rules from train examples.

    Each example: {"src":..., "dst":...}. Returns {"candidates": [...],
    "groups": n, "rejected": [...]}. Candidates are NOT stored and NOT
    active; callers save them explicitly as status=candidate.
    """
    lexicon = lexicon or {}
    groups: dict[str, list[dict]] = {}
    rejected = []
    for ex in examples:
        chk = check_example(ex.get("src", ""), ex.get("dst", ""), lexicon)
        if not chk["ok"]:
            rejected.append({"example": ex, **chk})
            continue
        key = _sig_key(chk["signature"])
        groups.setdefault(key, []).append({**ex, "_sig": chk["signature"]})
    candidates = []
    for key, rows in groups.items():
        if len(rows) == 1:
            ex = rows[0]
            candidates.append({
                "kind": "literal", "confidence": 0.5,
                "src_tmpl": {"text": ex["src"].strip()},
                "dst_tmpl": {"text": ex["dst"].strip()},
                "signature": ex["_sig"],
                "support": [ex], "note": "single example: exact match only"})
            continue
        srcs = [r["src"].strip() for r in rows]
        pre, suf = _common_affix(srcs)
        slot_ok = bool(pre or suf)
        mids = [s[len(pre):len(s) - len(suf) if suf else len(s)]
                for s in srcs] if slot_ok else []
        carry = slot_ok and all(m and m in r["dst"] for m, r in
                               zip(mids, rows))
        if not carry:
            rejected.append({"examples": [r["src"] for r in rows],
                             "ok": False,
                             "reason": "no shared affix with carried-over "
                             "slot; kept as separate literals"})
            for r in rows:
                candidates.append({
                    "kind": "literal", "confidence": 0.5,
                    "src_tmpl": {"text": r["src"].strip()},
                    "dst_tmpl": {"text": r["dst"].strip()},
                    "signature": r["_sig"], "support": [r],
                    "note": "no common slot: exact match only"})
            continue
        dst_t0 = rows[0]["dst"].strip().replace(mids[0], "{}", 1)
        if dst_t0.count("{}") != 1 or any(
                r["dst"].strip().replace(m, "{}", 1) != dst_t0
                for m, r in zip(mids, rows)):
            rejected.append({"examples": [r["src"] for r in rows],
                             "ok": False,
                             "reason": "slot maps to different dst positions"})
            continue
        candidates.append({
            "kind": "slot", "confidence": 0.75,
            "src_tmpl": {"pre": pre, "suf": suf},
            "dst_tmpl": {"template": dst_t0},
            "signature": rows[0]["_sig"],
            "support": [{k: r[k] for k in ("src", "dst")} for r in rows],
            "note": f"slot from {len(rows)} examples: {pre!r} + X + {suf!r}"})
    return {"candidates": candidates, "groups": len(groups),
            "rejected": rejected}


def apply_rules(text: str, rules: list[dict],
                lexicon: dict | None = None) -> dict:
    """Try active rules on text. Returns {"rewritten", "rule_id"} or
    {"rewritten": None, "reason": ...}. At most one application; the
    rewritten text must parse rule-free to a non-unknown frame; ties
    abstain."""
    s = text.strip()
    hits = []
    for r in rules:
        if r.get("status") != "active":
            continue
        rw = _rewrite_one(s, r)
        if rw is None or rw == s:
            continue
        fr = _parse_rule_free(rw, lexicon or {})
        if fr.intent == "unknown":
            continue
        hits.append((r, rw, frame_signature(fr.to_dict())))
    if not hits:
        return {"rewritten": None, "reason": "no active rule applies"}
    sigs = {_sig_key(h[2]) for h in hits}
    if len(sigs) > 1:
        return {"rewritten": None, "reason": "ambiguous: rules disagree",
                "rules": [h[0]["id"] for h in hits]}
    best = sorted(hits, key=lambda h: -len(h[0].get("id", "")))[0]
    return {"rewritten": best[1], "rule_id": best[0]["id"],
            "signature": best[2]}


def _rewrite_one(s: str, rule: dict) -> str | None:
    tmpl = rule.get("src_tmpl") or {}
    dst = rule.get("dst_tmpl") or {}
    if rule.get("kind") == "literal":
        return dst.get("text") if s == tmpl.get("text") else None
    pre, suf = tmpl.get("pre", ""), tmpl.get("suf", "")
    if not s.startswith(pre):
        return None
    if suf and not s.endswith(suf):
        return None
    mid = s[len(pre):len(s) - len(suf) if suf else len(s)]
    if not mid:
        return None
    return (dst.get("template") or "").replace("{}", mid)


def validate_rule(rule: dict, heldout: list[dict],
                  lexicon: dict | None = None) -> dict:
    """Held-out check: every held-out src must rewrite and reproduce the
    rule's dst signature. Strict 100% (small data, honest bar)."""
    lexicon = lexicon or {}
    if not heldout:
        return {"passed": False, "reason": "no held-out examples",
                "checked": 0}
    want = _sig_key(rule["signature"])
    detail = []
    for h in heldout:
        rw = _rewrite_one(h["src"].strip(), rule)
        if rw is None:
            detail.append({"src": h["src"], "ok": False,
                           "reason": "rule does not match"})
            continue
        fr = _parse_rule_free(rw, lexicon)
        got = _sig_key(frame_signature(fr.to_dict()))
        detail.append({"src": h["src"], "rewritten": rw, "ok": got == want,
                       "want": want, "got": got})
    passed = all(d["ok"] for d in detail)
    return {"passed": passed, "checked": len(detail), "detail": detail,
            "reason": "" if passed else "held-out mismatch"}


def activate_rule(mem, rule_id: str) -> dict:
    row = mem.get_phrase_rule(rule_id)
    if not row:
        return {"ok": False, "error": "rule not found"}
    if row["status"] != "validated":
        return {"ok": False,
                "error": f"only validated rules activate (now "
                f"{row['status']})"}
    row["status"] = "active"
    mem.save_phrase_rule(row)
    mem.add_experience("evaluation", f"phrase rule {rule_id} activated",
                       {"op": "phrase_activate", "rule_id": rule_id})
    return {"ok": True, "rule_id": rule_id}
