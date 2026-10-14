"""Read-only semantic grounding (Phase M3B).

Turns a ParsedUtterance into a query plan against the live KB and
judges it WITHOUT writing anything: no concepts, attributes,
relations, claims or evidence are created here.

Verdicts: supported | conflicted | unknown | ambiguous.
- supported: real evidence backs the statement (KB row, derived
  conclusion, or a denial whose affirmative core is refuted).
- conflicted: evidence directly contradicts the statement.
- unknown: not enough evidence either way. NEVER read as false.
- ambiguous: the utterance itself cannot be pinned down.

Negation is never dropped and never flipped silently: the affirmative
core is evaluated first, then the polarity is applied (supported core
+ denial = conflicted; refuted core + denial = supported; unknown
core + denial = unknown).
"""
from __future__ import annotations

from .memory import Memory
from .models import GroundingVerdict
from .ontology import Ontology
from .reasoner import Reasoner


def _ev(kind: str, ref, status: str, detail: str = "") -> dict:
    return {"kind": kind, "ref": ref, "status": status, "detail": detail}


def ground_statement(text: str, mem: Memory) -> GroundingVerdict:
    """Parse + ground one statement. Pure reads, zero writes."""
    from .semparse import parse_statement
    onto = Ontology(mem)
    rsn = Reasoner(mem)
    u = parse_statement(text, mem)
    out = GroundingVerdict(
        sentence=text, parse=u.to_dict(), subject_match={},
        object_match={}, queries=[], verdict="unknown", evidence=[],
        detail="")
    if u.ambiguity:
        out.ambiguity = u.ambiguity
        out.verdict = "ambiguous"
        out.detail = "utterance ambiguous, no queries run"
        return out
    if u.predicate == "unknown":
        out.detail = "no resolvable predicate"
        return out
    subj = _resolve_concept(mem, u.subject) if u.subject_known else None
    if subj is None:
        # unknown subject: still resolve the object so candidate
        # building can see it, but run no subject-bound queries.
        out.object_match = {"text": u.object,
                            "concept_id": _resolve_concept(mem, u.object)}
        out.detail = f"subject not in KB: {u.subject}"
        return out
    out.subject_match = {"text": u.subject, "concept_id": subj}
    queries, evidence = [], []

    if u.predicate == "is_a":
        queries.append({"op": "is_a", "concept": subj,
                        "ancestor": u.object})
        target = _resolve_concept(mem, u.object)
        if target is None:
            out.object_match = {"text": u.object, "concept_id": None}
            out.queries = queries
            out.detail = f"object not in KB: {u.object}"
            return out
        out.object_match = {"text": u.object, "concept_id": target}
        r = rsn.is_a(subj, target)
        evidence.append(_ev("is_a", f"{subj}->{target}", "verified",
                            str(r.get("trace", []))))
        core = bool(r["conclusion"]) if r["conclusion"] != "unknown" \
            else None
    elif u.predicate == "has":
        queries.append({"op": "has_attr_scan", "concept": subj,
                        "feature": u.object})
        core, hits = _has_feature(mem, onto, subj, u.object)
        evidence.extend(hits)
        out.object_match = {"text": u.object,
                            "concept_id": _resolve_concept(mem, u.object)}
    elif u.predicate == "capable":
        queries.append({"op": "capability", "concept": subj,
                        "action": u.object, "modal": u.modal})
        core, hits = _capability(mem, onto, subj, u.object)
        evidence.extend(hits)
        out.object_match = {"text": u.object,
                            "concept_id": _resolve_concept(mem, u.object)}
    else:  # unspecified and anything future: no query invented
        out.queries = queries
        out.detail = f"predicate '{u.predicate}' has no KB mapping"
        return out

    out.queries = queries
    out.evidence = evidence
    # rule-derived conclusions count as extra support with their own
    # source; they never override direct rows, only add to them.
    for e in _derived_support(mem, onto, subj, u):
        e["supports"] = True
        evidence.append(e)
    if core is None and any("supports" in e for e in evidence):
        core = True
    # verdict follows the best supporting status, never upgrades it:
    # contested-only support is a live dispute (conflicted), hypothesis
    # stays visibly weak. Polarity applies last, never silently.
    best = "unknown"
    order = {"verified": 3, "inherited(verified)": 3, "hypothesis": 2,
             "inherited(hypothesis)": 2, "contested": 1,
             "inherited(contested)": 1}
    for e in evidence:
        if order.get(e.get("status", ""), 0) > order.get(best, 0):
            best = e.get("status", "unknown")
    if core is True and best in ("contested", "inherited(contested)"):
        out.verdict = "supported" if u.negated else "conflicted"
    elif core is True:
        out.verdict = "conflicted" if u.negated else "supported"
    elif core is False:
        out.verdict = "supported" if u.negated else "conflicted"
    else:
        out.verdict = "unknown"
    out.detail = ("negated statement over "
                  f"{'supported' if core else 'refuted' if core is False else 'unknown'}"
                  " core") if u.negated else f"core: {core}"
    return out


def answer_question(text: str, mem: Memory):
    """Answer one parsed question. Pure reads, zero writes.

    Returns a GroundingVerdict with verdict in supported | conflicted
    | unknown | ambiguous | failure plus question/answer/is_yesno.
    Rules: yes/no needs decisive evidence (no record is never a 否);
    contested evidence on both sides is conflicted; hypothesis rows
    keep their label (never upgraded); derived conclusions cite rule,
    bindings, status and depth; engine errors are failure, never
    silent empties.
    """
    from .semparse import parse_question
    onto = Ontology(mem)
    out = GroundingVerdict(sentence=text)
    try:
        q = parse_question(text, mem)
    except Exception as e:
        out.verdict = "failure"
        out.detail = f"parser error: {e}"
        return out
    out.question = q.to_dict()
    out.is_yesno = q.is_yesno
    if q.ambiguity:
        out.ambiguity = q.ambiguity
        out.verdict = "ambiguous"
        out.detail = "question ambiguous, no queries run"
        return out
    if q.qkind == "unknown" or not q.subject_known:
        out.detail = "unresolvable subject or question form"
        return out
    subj = _resolve_concept(mem, q.subject)
    if subj is None:
        out.detail = f"subject not in KB: {q.subject}"
        return out
    out.subject_match = {"text": q.subject, "concept_id": subj}
    try:
        return _answer_dispatch(out, q, subj, mem, onto)
    except Exception as e:
        out.verdict = "failure"
        out.detail = f"engine error: {e}"
        return out


def _answer_dispatch(out, q, subj, mem, onto):
    rsn = Reasoner(mem)
    kind = q.qkind
    if kind == "category":
        direct = sorted(mem.get_parents(subj))
        try:
            chain = sorted(set(onto.ancestors(subj)) - set(direct) -
                           {subj})
        except Exception:
            chain = []
        if not direct and not chain:
            out.detail = f"no category recorded for {subj}"
            return out
        for p in direct:
            out.evidence.append(_ev("parents", p, "verified",
                                    f"{subj} direct parent"))
        for p in chain:
            out.evidence.append(_ev("parents", p, "inherited",
                                    f"{subj} inherited parent"))
        out.queries.append({"op": "parents", "concept": subj})
        out.verdict = "supported"
        out.answer = "、".join(direct + chain)
        out.detail = (f"direct: {direct}; inherited: {chain}")
        return out
    if kind == "capability" and not q.is_yesno:
        return _answer_open_attr(out, q, subj, mem, onto, "function")
    if kind == "attr_open" and not q.is_yesno:
        key = FEATURE_KEYS.get(q.object, q.object)
        if not key:
            out.detail = "attribute word missing"
            return out
        return _answer_open_attr(out, q, subj, mem, onto, key)
    if kind == "has" and not q.is_yesno:
        return _answer_open_has(out, q, subj, mem, onto)
    if kind == "has" and q.is_yesno:
        return _answer_verify_has(out, q, subj, mem, onto)
    if kind == "is_a_verify":
        return _answer_verify_isa(out, q, subj, mem, onto, rsn)
    if kind == "capability" and q.is_yesno:
        return _answer_verify_capable(out, q, subj, mem, onto)
    out.detail = f"question kind '{kind}' has no query mapping"
    return out


def _answer_open_attr(out, q, subj, mem, onto, key):
    out.queries.append({"op": "get_attribute", "concept": subj,
                        "key": key})
    try:
        inh = onto.inherited_attributes(subj)
    except Exception:
        inh = {}
    vals = []
    if key in inh:
        row = inh[key]
        vals.append((row.get("value"), row.get("status", "hypothesis"),
                     row.get("id")))
        out.evidence.append(_ev("attribute", row.get("id"),
                                row.get("status", "hypothesis"),
                                f"{row.get('concept_id')}.{key}="
                                f"{row.get('value')}"))
        # rival rows the nearest-wins lookup hides (same key, other
        # values, own or nearest-ancestor level): they make the answer
        # contested instead of silently picked.
        for other in _rival_attr_values(mem, onto, subj, key,
                                        row.get("value")):
            vals.append(other)
            out.evidence.append(_ev("attribute", other[2], "contested",
                                    f"rival value kept: {other[0]}"))
    for f in _derived_for(mem, subj, key):
        if f.get("limited"):
            out.queries.append({"op": "chain_limited", "concept": subj})
            out.limited = True
            out.detail = (out.detail + "；推理搜索受限，未完成验证" if out.detail
                          else "推理搜索受限，未完成验证")
            continue
        vals.append((f["value"], f["status"], f["ref"]))
        out.evidence.append(_ev("rule", f["ref"], f["status"], f["why"]))
    if not vals:
        if out.limited:
            out.detail = out.detail + f"；no {key} recorded for {subj}"
            return out
        out.detail = f"no {key} recorded for {subj}"
        return out
    uniq = {str(v) for v, _, _ in vals}
    if len(uniq) > 1:
        out.verdict = "conflicted"
        out.answer = "；".join(f"{v}（{s}）" for v, s, _ in vals)
        out.detail = "rival values kept, no silent pick"
        return out
    v, s, _ = vals[0]
    out.verdict = "supported"
    out.answer = str(v)
    if out.limited:
        out.detail = out.detail + f"；value status kept as {s}"
    else:
        out.detail = f"value status kept as {s}"
    return out


def _answer_open_has(out, q, subj, mem, onto):
    out.queries.append({"op": "list_attributes", "concept": subj})
    try:
        inh = onto.inherited_attributes(subj)
    except Exception:
        inh = {}
    try:
        rels = mem.get_relations(src_id=subj)
    except Exception:
        rels = []
    _all_derived = _derived_for(mem, subj, "")
    if any(f.get("limited") for f in _all_derived):
        out.queries.append({"op": "chain_limited", "concept": subj})
        out.limited = True
        out.detail = "推理搜索受限，未完成验证"
    derived = [f for f in _all_derived if not f.get("limited")]
    for f in derived:
        out.evidence.append(_ev("rule", f["ref"], f["status"], f["why"]))
    if not inh and not rels and not derived:
        if out.limited:
            out.detail = out.detail + f"；nothing recorded for {subj}"
            return out
        out.detail = f"nothing recorded for {subj}"
        return out
    for key, row in inh.items():
        out.evidence.append(_ev("attribute", row.get("id"),
                                row.get("status", "hypothesis"),
                                f"{row.get('concept_id')}.{key}="
                                f"{row.get('value')}"))
    for r in rels:
        out.evidence.append(_ev("relation", r.get("id"),
                                r.get("status", "hypothesis"),
                                f"{subj}-{r.get('rel_type')}->"
                                f"{r.get('dst_id')}"))
    out.verdict = "supported"
    out.answer = "、".join(sorted(inh)) + (
        "；" + "、".join(f"{r.get('rel_type')}->{r.get('dst_id')}"
                         for r in rels) if rels else "") + (
        "；" + "、".join(f"推导:{f['value']}@{f['ref']}" for f in derived)
        if derived else "")
    if not out.detail.startswith("推理搜索受限"):
        out.detail = f"{len(inh)} attribute(s), {len(rels)} relation(s), " \
            f"{len(derived)} derived"
    return out


# Fixed Chinese -> canonical key vocabulary for has-questions
# (tiny, documented; not per-sentence logic).
FEATURE_KEYS = {"功能": "function", "作用": "function", "用途": "function",
                "用处": "function", "材质": "material", "材料": "material",
                "颜色": "color", "形状": "shape", "位置": "location"}


def _canon_feature(feat: str) -> str:
    return FEATURE_KEYS.get(feat, feat)


def _answer_verify_has(out, q, subj, mem, onto):
    feat = _canon_feature(q.object)
    out.queries.append({"op": "has_attr_scan", "concept": subj,
                        "feature": feat, "negated": q.negated})
    try:
        inh = onto.inherited_attributes(subj)
    except Exception:
        inh = {}
    pros, cons = [], []
    for key, row in inh.items():
        st = row.get("status", "hypothesis")
        if key == feat or key == q.object or row.get("value") == feat:
            (pros if "contested" not in str(st) else cons).append(
                (key, row))
            out.evidence.append(_ev(
                "attribute", row.get("id"), st,
                f"{row.get('concept_id')}.{key}={row.get('value')}"))
    if pros and cons:
        out.verdict = "conflicted"
        out.answer = "存疑"
        out.detail = "supporting and rival evidence both kept"
        return out
    if cons and not pros:
        out.verdict = "conflicted"
        out.answer = "存疑"
        out.detail = "contested-only evidence kept, no silent unknown"
        return out
    hit = bool(pros)
    if q.negated:
        # denial of a supported feature is contradicted; denial with
        # no record at all stays unknown (absence is not proof).
        if hit:
            out.verdict = "conflicted"
            out.answer = "否"
            out.detail = "denial contradicted by stored evidence"
        else:
            out.detail = f"no record of {feat} on {subj}"
        return out
    if hit:
        out.verdict = "supported"
        out.answer = "是"
        out.detail = "feature recorded"
        return out
    out.detail = f"no record of {feat} on {subj}; not proven false"
    return out


def _answer_verify_isa(out, q, subj, mem, onto, rsn):
    target = _resolve_concept(mem, q.object)
    out.object_match = {"text": q.object, "concept_id": target}
    out.queries.append({"op": "is_a", "concept": subj,
                        "ancestor": q.object, "negated": q.negated})
    if target is None:
        out.detail = f"object not in KB: {q.object}"
        return out
    r = rsn.is_a(subj, target)
    out.evidence.append(_ev("is_a", f"{subj}->{target}", "verified",
                            str(r.get("trace", []))))
    holds = bool(r["conclusion"]) if r["conclusion"] != "unknown" else None
    if holds is None:
        out.detail = "hierarchy lookup failed"
        return out
    if q.negated:
        out.verdict = "conflicted" if holds else "supported"
        out.answer = "否" if holds else "是"
        out.detail = "negated membership question over decided hierarchy"
        return out
    out.verdict = "supported"
    out.answer = "是" if holds else "否"
    out.detail = "decided hierarchy"
    return out


def _answer_verify_capable(out, q, subj, mem, onto):
    action = q.object
    out.queries.append({"op": "capability", "concept": subj,
                        "action": action, "modal": q.modal,
                        "negated": q.negated})
    try:
        inh = onto.inherited_attributes(subj)
    except Exception:
        inh = {}
    pros, cons = [], []
    for key, row in inh.items():
        st = row.get("status", "hypothesis")
        if key == "function" and row.get("value") == action:
            (pros if "contested" not in str(st) else cons).append(row)
            out.evidence.append(_ev(
                "attribute", row.get("id"), st,
                f"{row.get('concept_id')}.function={action}"))
    if pros and cons:
        out.verdict = "conflicted"
        out.answer = "存疑"
        out.detail = "supporting and rival evidence both kept"
        return out
    if cons and not pros:
        out.verdict = "conflicted"
        out.answer = "存疑"
        out.detail = "contested-only evidence kept, no silent unknown"
        return out
    if q.negated:
        if pros:
            out.verdict = "conflicted"
            out.answer = "否"
            out.detail = "denied capability contradicted by evidence"
        else:
            out.detail = f"no record of capability {action} on {subj}"
        return out
    if pros:
        out.verdict = "supported"
        out.answer = "是"
        out.detail = "capability recorded"
        return out
    out.detail = f"no record of capability {action} on {subj}; " \
        "not proven false"
    return out


def _rival_attr_values(mem, onto, cid, key, value):
    """Other stored values for the applicable rows of cid.key."""
    import json as _json
    try:
        own = [a.get("value") for a in mem.get_attributes(cid)
               if a.get("key") == key]
        pool = own if own else []
        ids = {a.get("id"): a.get("value")
               for a in mem.get_attributes(cid) if a.get("key") == key}
        if not pool:
            for anc in onto.ancestors(cid):
                rows = [a for a in mem.get_attributes(anc)
                        if a.get("key") == key]
                if rows:
                    pool = [a.get("value") for a in rows]
                    ids = {a.get("id"): a.get("value") for a in rows}
                    break
        out = []
        for aid, v in ids.items():
            if _json.dumps(v, sort_keys=True, default=str) != \
                    _json.dumps(value, sort_keys=True, default=str):
                out.append((v, "contested", aid))
        return out
    except Exception:
        return []


def _derived_for(mem, subj, key):
    """Rule-derived (concept, key) values with rule/binding/status.

    A budget/cap marker from the engine is kept as a pseudo-entry
    {"limited": True} so callers can tell truncated search apart from
    genuine absence of evidence.
    """
    out = []
    try:
        from .reasoner import Reasoner
        fired = Reasoner(mem).forward_chain()
    except Exception:
        return out
    if any(f.get("limited") for f in fired):
        out.append({"limited": True})
    for f in fired:
        if f.get("limited"):
            continue
        conc = f.get("conclusion")
        if not isinstance(conc, dict) or conc.get("type") != "has_attr":
            continue
        if conc.get("concept") != subj:
            continue
        if key and conc.get("key") != key:
            continue
        out.append({"value": conc.get("value"),
                    "status": f.get("status", "hypothesis"),
                    "ref": f.get("rule_id", f.get("rule")),
                    "why": f"depth={f.get('depth', 1)} "
                    f"bindings={f.get('bindings', {})}"})
    return out


def _resolve_concept(mem: Memory, text: str):
    if not text:
        return None
    if mem.get_concept(text) is not None:
        return text
    hits = mem.find_concepts_by_name(text)
    ids = {c["id"] for c in hits}
    if len(ids) == 1:
        return next(iter(ids))
    return None


def _has_feature(mem: Memory, onto: Ontology, cid: str, feat: str):
    """Possession evidence: exact key, exact value, or relation.

    Returns (core True/False/None, evidence list). No match anywhere
    is None (unknown), never False: absence of a record is not proof
    of absence.
    """
    ev = []
    try:
        inh = onto.inherited_attributes(cid)
    except Exception:
        inh = {}
    if feat in inh:
        row = inh[feat]
        ev.append(_ev("attribute", row.get("id"), row.get("status",
                                                          "hypothesis"),
                      f"{row.get('concept_id')}.{feat}={row.get('value')}"))
        return True, ev
    for key, row in inh.items():
        if row.get("value") == feat:
            ev.append(_ev("attribute", row.get("id"),
                          row.get("status", "hypothesis"),
                          f"{row.get('concept_id')}.{key}={feat}"))
            return True, ev
    try:
        rels = mem.get_relations(src_id=cid)
    except Exception:
        rels = []
    for r in rels:
        if r.get("dst_id") == feat or r.get("rel_type") == feat:
            ev.append(_ev("relation", r.get("id"),
                          r.get("status", "hypothesis"),
                          f"{cid}-{r.get('rel_type')}->{r.get('dst_id')}"))
            return True, ev
    return None, ev


def _capability(mem: Memory, onto: Ontology, cid: str, action: str):
    """Capability evidence: exact function value or explicit relation.

    A merely related value (drinking vs 盛水) is NOT counted: support
    needs equality, otherwise unknown. Modal is recorded by the caller.
    """
    ev = []
    try:
        inh = onto.inherited_attributes(cid)
    except Exception:
        inh = {}
    row = inh.get("function")
    if row is not None and row.get("value") == action:
        ev.append(_ev("attribute", row.get("id"),
                      row.get("status", "hypothesis"),
                      f"{row.get('concept_id')}.function={action}"))
        return True, ev
    try:
        rels = mem.get_relations(src_id=cid)
    except Exception:
        rels = []
    for r in rels:
        if r.get("dst_id") == action or r.get("rel_type") == action:
            ev.append(_ev("relation", r.get("id"),
                          r.get("status", "hypothesis"),
                          f"{cid}-{r.get('rel_type')}->{r.get('dst_id')}"))
            return True, ev
    return None, ev


def _derived_support(mem: Memory, onto: Ontology, subj: str, u) -> list:
    """Rule-derived conclusions backing this utterance, if any.

    Reuses forward_chain entries (active rules only, per M2 gates);
    each item keeps rule id, bindings, status and depth. Direct rows
    always win ties; derivation only ADDS support, never overwrites.
    """
    out = []
    try:
        fired = Reasoner(mem).forward_chain()
    except Exception:
        return out
    for f in fired:
        conc = f.get("conclusion")
        if not isinstance(conc, dict):
            continue
        if conc.get("type") == "has_attr" and \
                conc.get("concept") == subj and u.predicate in (
                    "has", "capable"):
            if u.predicate == "has" and conc.get("key") != u.object and \
                    conc.get("value") != u.object:
                continue
            if u.predicate == "capable" and conc.get("key") != "function":
                continue
            out.append(_ev("rule", f.get("rule_id", f.get("rule")),
                           f.get("status", "hypothesis"),
                           f"depth={f.get('depth', 1)} "
                           f"bindings={f.get('bindings', {})}"))
            for e in out:
                e["supports"] = True
    return out
