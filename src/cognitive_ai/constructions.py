"""Bidirectional constructions (Phase 6H-2B).

A construction binds TWO slots (entity + value) between fixed affixes
for one statement kind (this phase: capability only). Parse and generate
read the SAME stored row -- there are no separate per-direction
templates, handwritten or otherwise.

  parse:    pre + ENTITY + mid + VALUE + suf  ->  observation-grade frame
            {intent observe, capable(entity, value)} for the existing
            teach-confirm flow (hypothesis gates, never verified here).
  generate: (entity, value) -> pre + entity + mid + value + suf.

Learning is affix induction with role checks, never sentence memos:
- >=2 examples with DISTINCT entities;
- the entity slot must equal the labeled entity, the value slot the
  labeled value (or stay fixed when all values agree);
- held-out examples (disjoint entities, incl. novel entity x value
  combos) must parse to exact (entity, value) AND regenerate the
  source sentence byte-for-byte;
- ambiguity (several matching rows) abstains in both directions.

Lifecycle: candidate -> validated (held-out 100%) -> explicit
activate. Only active rows are consulted, and only when the fixed
parser returns unknown (parse) or when asked to generate.
"""
from __future__ import annotations

import time

_counter = 0


def new_rule_id(name: str = "cx") -> str:
    global _counter
    _counter += 1
    return f"cx_{name}_{int(time.time() * 1000) % 100000}_{_counter}"


def _lcs_all(strs: list[str]) -> str:
    """Longest common substring of all strings (short inputs only)."""
    if not strs:
        return ""
    best = ""
    base = min(strs, key=len)
    for i in range(len(base)):
        for j in range(i + 1, len(base) + 1):
            cand = base[i:j]
            if len(cand) > len(best) and all(cand in s for s in strs):
                best = cand
    return best


def _split_affix(srcs: list[str]) -> tuple[str, str, str] | None:
    """Common (pre, mid, suf): pre = common prefix, suf = common suffix
    (non-overlapping), mid = longest common substring of the remainders.
    None when no slot-bearing structure exists.
    """
    pre = srcs[0]
    for t in srcs[1:]:
        i = 0
        while i < min(len(pre), len(t)) and pre[i] == t[i]:
            i += 1
        pre = pre[:i]
    suf = srcs[0]
    for t in srcs[1:]:
        j = 0
        while j < min(len(suf), len(t)) and suf[-1 - j] == t[-1 - j]:
            j += 1
        suf = suf[len(suf) - j:] if j else ""
    if len(pre) + len(suf) >= min(len(t) for t in srcs):
        return None
    cores = [s[len(pre):len(s) - len(suf) if suf else len(s)]
             for s in srcs]
    mid = _lcs_all(cores)
    if not mid:
        return None
    # mid must leave room for both slots in every core
    for c in cores:
        parts = c.split(mid, 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
    return pre, mid, suf


def induce(examples: list[dict]) -> dict:
    """Induce construction candidates. Example: {"src","entity","value"}.

    Needs >=2 examples with distinct entities. The entity part of each
    core must equal the labeled entity and the value part the labeled
    value; when all values agree the value stays fixed (reported).
    Anything weaker -> rejected with a reason, never a silent memo.
    """
    candidates, rejected = [], []
    rows = [r for r in examples if r.get("src", "").strip()
            and r.get("entity", "").strip() and r.get("value", "").strip()]
    if len(rows) < 2:
        return {"candidates": candidates, "groups": 0,
                "rejected": [{"ok": False, "reason": "needs >=2 labeled "
                              "examples with distinct entities; examples "
                              "kept, nothing induced"}]}
    if len({r["entity"].strip() for r in rows}) < 2:
        return {"candidates": candidates, "groups": 1,
                "rejected": [{"ok": False, "reason": "entities not "
                              "distinct; no generalizable slot"}]}
    aff = _split_affix([r["src"].strip() for r in rows])
    if aff is None:
        return {"candidates": candidates, "groups": 1,
                "rejected": [{"ok": False, "reason": "no shared "
                              "pre/mid/suf structure"}]}
    pre, mid, suf = aff
    ents_ok = True
    for r in rows:
        core = r["src"].strip()[len(pre):len(r["src"].strip())
                                - len(suf) if suf
                                else len(r["src"].strip())]
        if mid not in core:
            ents_ok = False
            break
        left, right = core.split(mid, 1)
        if left != r["entity"].strip() or not right:
            ents_ok = False
            break
    if not ents_ok:
        return {"candidates": candidates, "groups": 1,
                "rejected": [{"ok": False, "reason": "affix middle does "
                              "not isolate the labeled (entity, value) "
                              "pair"}]}
    # value slot variable only if values differ across examples
    values = [r["value"].strip() for r in rows]
    fixed = values[0] if len(set(values)) == 1 else ""
    if not fixed:
        # each value must sit exactly after mid in its own source
        for r in rows:
            core = r["src"].strip()[len(pre):len(r["src"].strip())
                                    - len(suf) if suf else
                                    len(r["src"].strip())]
            if core != r["entity"].strip() + mid + r["value"].strip():
                return {"candidates": candidates, "groups": 1,
                        "rejected": [{"ok": False, "reason": "value slot "
                                      "does not align; no slot induced"}]}
    candidates.append({
        "kind": "capability", "pre": pre, "mid": mid, "suf": suf,
        "fixed_value": fixed, "confidence": 0.7,
        "support": [{k: r[k] for k in ("src", "entity", "value")}
                    for r in rows],
        "note": f"construction from {len(rows)} examples: {pre!r} + "
                f"ENTITY + {mid!r} + "
                f"{'VALUE' if not fixed else repr(fixed)} + {suf!r}"})
    return {"candidates": candidates, "groups": 1, "rejected": rejected}


def match(text: str, rules: list[dict]) -> list[tuple]:
    """All (rule, entity, value) readings of text. Empty = no match."""
    s = text.strip()
    hits = []
    for r in rules:
        if r.get("status") != "active" or r.get("kind") != "capability":
            continue
        pre, mid, suf = r.get("pre", ""), r.get("mid", ""), r.get("suf", "")
        if not s.startswith(pre):
            continue
        if suf and not s.endswith(suf):
            continue
        core = s[len(pre):len(s) - len(suf) if suf else len(s)]
        if mid not in core:
            continue
        ent, rest = core.split(mid, 1)
        if not ent:
            continue
        if r.get("fixed_value"):
            if rest != r["fixed_value"]:
                continue
            val = rest
        elif not rest:
            continue
        else:
            val = rest
        hits.append((r, ent, val))
    return hits


def parse_frame(text: str, rules: list[dict], raw: str = "") -> dict | None:
    """Parse direction: unique match -> observation-grade frame.

    Multiple matches -> None (ambiguity abstains). The frame reuses the
    exact capable/observe shapes the fixed parser emits, so the existing
    teach-confirm flow (hypothesis only) applies unchanged.
    """
    hits = match(text, rules)
    if len(hits) != 1:
        return None
    r, ent, val = hits[0]
    return {"intent": "observe",
            "entities": [{"text": ent, "role": "subject"}],
            "predicates": [{"type": "capable", "subj": ent,
                            "action": val}],
            "modality": "assert", "scope": "generic",
            "source": "construction",
            "confidence": 0.65, "unresolved_references": [],
            "ambiguities": [], "raw": raw or text,
            "trace": [f"construction:{r['id']} parsed "
                      f"{text!r} -> ({ent}, {val})"]}


def render(entity: str, value: str, rules: list[dict]) -> list[dict]:
    """Generate direction: (entity, value) -> surfaces from the SAME rows.

    Returns [{text, source}] or [] (abstain). Fixed values constrain:
    a fixed-value row only renders its own value.
    """
    out = []
    for r in rules:
        if r.get("status") != "active" or r.get("kind") != "capability":
            continue
        if not entity:
            continue
        if r.get("fixed_value"):
            if value != r["fixed_value"]:
                continue
            out.append({"text": f"{r['pre']}{entity}{r['mid']}"
                                f"{r['fixed_value']}{r['suf']}",
                        "source": f"learned:{r['id']}"})
        elif value:
            out.append({"text": f"{r['pre']}{entity}{r['mid']}{value}"
                                f"{r['suf']}",
                        "source": f"learned:{r['id']}"})
    return out


def validate_rule(rule: dict, heldout: list[dict]) -> dict:
    """Held-out check both directions: parse exactness AND regeneration.

    Each held-out {src, entity, value} must parse (rule alone) to the
    exact pair, and rendering that pair must reproduce src exactly.
    Strict 100%.
    """
    if not heldout:
        return {"passed": False, "reason": "no held-out examples",
                "checked": 0}
    probe = [{**rule, "status": "active"}]
    detail = []
    for h in heldout:
        fr = parse_frame(h["src"], probe)
        ok = fr is not None and \
            fr["entities"][0]["text"] == h["entity"].strip() and \
            fr["predicates"][0]["action"] == h["value"].strip()
        regen = ""
        if ok:
            outs = render(h["entity"].strip(), h["value"].strip(), probe)
            regen = outs[0]["text"] if outs else ""
            ok = regen == h["src"].strip()
        detail.append({"src": h["src"], "ok": bool(ok), "regen": regen})
    passed = all(d["ok"] for d in detail)
    return {"passed": passed, "checked": len(detail), "detail": detail,
            "reason": "" if passed else "held-out parse/regenerate mismatch"}


def activate_rule(mem, rule_id: str) -> dict:
    row = mem.get_construction(rule_id)
    if not row:
        return {"ok": False, "error": "rule not found"}
    if row["status"] != "validated":
        return {"ok": False,
                "error": f"only validated rules activate (now "
                f"{row['status']})"}
    row["status"] = "active"
    mem.save_construction(row)
    mem.add_experience("evaluation", f"construction {rule_id} activated",
                       {"op": "construction_activate", "rule_id": rule_id})
    return {"ok": True, "rule_id": rule_id}
