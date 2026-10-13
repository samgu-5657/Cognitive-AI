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
