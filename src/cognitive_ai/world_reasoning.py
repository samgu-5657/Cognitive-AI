"""State-scoped logical reasoning (Phase M4-E).

Pure in-memory, deterministic inference over one caller-supplied
WorldState. Reads state facts only; never touches Memory, SQLite, the
legacy knowledge tables, other world instances, or the network. Never
modifies states, writes events, executes actions, or upgrades
knowledge credibility: entailed means derivable here, not verified in
reality.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from . import matching
from .models import (WorldInferenceRule, WorldLiteralPattern,
                     WorldObjectRef, WorldPatternTerm, WorldProofNode,
                     WorldProposition, WorldReasoningResult, WorldRuleView,
                     WorldRuleDiagnostic, WorldState, WorldStateFact,
                     WorldTerm, WorldModality, WorldPolarity)
from .policy import MAX_CHAIN_DEPTH

POSITIVE = "positive"
NEGATIVE = "negative"


def _pattern_terms_complete(pattern: WorldLiteralPattern) -> Optional[str]:
    if not pattern.predicate:
        return "predicate missing"
    for term in (pattern.subject, pattern.object):
        if not term.is_variable() and term.ref is None \
                and not term.has_literal and not term.unresolved:
            return "pattern term carries nothing to match"
    return None


def validate_inference_rule(rule: WorldInferenceRule) -> tuple:
    """Structural check only; () means usable. No IO."""
    problems = []
    if not rule.rule_id:
        problems.append("rule_id missing")
    if not rule.premises:
        problems.append("no premises")
    premise_vars: set = set()
    for i, prem in enumerate(rule.premises):
        bad = _pattern_terms_complete(prem)
        if bad:
            problems.append(f"premise {i}: {bad}")
        for term in (prem.subject, prem.object):
            if term.is_variable():
                premise_vars.add(term.variable)
    bad = _pattern_terms_complete(rule.conclusion)
    if bad:
        problems.append(f"conclusion: {bad}")
    else:
        for term in (rule.conclusion.subject, rule.conclusion.object):
            if term.is_variable() and term.variable not in premise_vars:
                problems.append(
                    f"conclusion variable {term.variable!r} "
                    f"never appears in premises")
    return tuple(problems)


def _legacy_term(value: Any) -> tuple:
    """Legacy value slot -> (WorldPatternTerm, refusal reason)."""
    if isinstance(value, str) and value.startswith("?") and len(value) > 1:
        return WorldPatternTerm(variable=value[1:]), ""
    if isinstance(value, str) and value == "?":
        return WorldPatternTerm(), "lone '?' is not a value"
    if value is None:
        return WorldPatternTerm(), "slot has no value"
    if isinstance(value, bool):
        return WorldPatternTerm(literal=value, has_literal=True), ""
    if isinstance(value, (str, int, float)):
        return WorldPatternTerm(literal=value, has_literal=True), ""
    return WorldPatternTerm(), f"unsupported value {value!r}"


def _legacy_ref(value: Any) -> tuple:
    """Legacy concept slot -> reference term (id used verbatim)."""
    if isinstance(value, str) and value.startswith("?") and len(value) > 1:
        return WorldPatternTerm(variable=value[1:]), ""
    if not isinstance(value, str) or not value or value == "?":
        return WorldPatternTerm(), f"bad concept slot {value!r}"
    return WorldPatternTerm(
        ref=WorldObjectRef(ref_id=value, name=value,
                           kind="legacy_concept")), ""


def _legacy_pattern(cond: dict) -> tuple:
    """One legacy condition/conclusion -> (WorldLiteralPattern, reason).

    Only has_attr / relation shapes convert losslessly. is_a has no
    world counterpart (states carry no concept graph), exceptions
    cannot be expressed, and both are refused instead of guessed.
    """
    ctype = cond.get("type")
    if ctype == "has_attr":
        concept, key = cond.get("concept"), cond.get("key")
        if "value" not in cond:
            return WorldLiteralPattern(), "has_attr without value"
        if not isinstance(key, str) or not key or key.startswith("?"):
            return WorldLiteralPattern(), f"bad key {key!r}"
        subj, bad = _legacy_ref(concept)
        if bad:
            return WorldLiteralPattern(), bad
        obj, bad = _legacy_term(cond.get("value"))
        if bad:
            return WorldLiteralPattern(), bad
        return WorldLiteralPattern(
            subject=subj, predicate=key, object=obj,
            polarity=WorldPolarity.POSITIVE), ""
    if ctype == "relation":
        src, rel, dst = (cond.get("src"), cond.get("rel_type"),
                         cond.get("dst"))
        if not isinstance(rel, str) or not rel:
            return WorldLiteralPattern(), f"bad rel_type {rel!r}"
        subj, bad = _legacy_ref(src)
        if bad:
            return WorldLiteralPattern(), bad
        obj, bad = _legacy_ref(dst)
        if bad:
            return WorldLiteralPattern(), bad
        return WorldLiteralPattern(
            subject=subj, predicate=rel, object=obj,
            polarity=WorldPolarity.POSITIVE), ""
    return WorldLiteralPattern(), \
        f"type {ctype!r} has no lossless world form"


def compile_world_rules(views) -> dict:
    """Compile stored rule views; only active rules may participate.

    Lossless conversions only. Every refusal keeps rule_id + reason.
    complete is False when an *active* rule could not be compiled.
    """
    rules = []
    refused = []
    complete = True
    for view in views:
        if isinstance(view, dict):
            rid = view.get("rule_id", "")
            status = view.get("status") or "active"
            conds = view.get("conditions", [])
            conc = view.get("conclusion", {})
            excs = view.get("exceptions", [])
        else:
            rid = view.rule_id
            status = view.status or "active"
            conds = list(view.conditions)
            conc = dict(view.conclusion)
            excs = list(view.exceptions)
        if status != "active":
            refused.append(WorldRuleDiagnostic(
                rule_id=str(rid),
                reason=f"status {status!r} is not inferable"))
            continue
        if excs:
            refused.append(WorldRuleDiagnostic(
                rule_id=str(rid),
                reason="exceptions have no lossless world form"))
            complete = False
            continue
        premises = []
        bad = ""
        for cond in conds:
            if not isinstance(cond, Mapping):
                bad = f"condition {cond!r} is not a mapping"
                break
            pat, why = _legacy_pattern(cond)
            if why:
                bad = f"condition not convertible: {why}"
                break
            premises.append(pat)
        if bad or not premises:
            refused.append(WorldRuleDiagnostic(
                rule_id=str(rid),
                reason=bad or "no premises convertible"))
            complete = False
            continue
        if not isinstance(conc, Mapping):
            refused.append(WorldRuleDiagnostic(
                rule_id=str(rid), reason="conclusion is not a mapping"))
            complete = False
            continue
        target, why = _legacy_pattern(conc)
        if why:
            refused.append(WorldRuleDiagnostic(
                rule_id=str(rid),
                reason=f"conclusion not convertible: {why}"))
            complete = False
            continue
        try:
            rule = WorldInferenceRule(
                rule_id=str(rid), premises=tuple(premises),
                conclusion=target,
                source_ref=f"rule:{rid}", status="active")
        except ValueError as e:
            refused.append(WorldRuleDiagnostic(
                rule_id=str(rid), reason=str(e)))
            complete = False
            continue
        problems = validate_inference_rule(rule)
        if problems:
            refused.append(WorldRuleDiagnostic(
                rule_id=str(rid), reason="; ".join(problems)))
            complete = False
            continue
        rules.append(rule)
    return {"rules": tuple(rules), "refused": tuple(refused),
            "complete": complete}


def _fact_key(prop: WorldProposition) -> tuple:
    subj = prop.subject
    obj = prop.object
    skey = ("ref", subj.ref.ref_id) if subj.ref is not None else (
        ("lit", repr(subj.literal)) if subj.has_literal else
        ("unres", subj.unresolved, subj.missing))
    okey = ("ref", obj.ref.ref_id) if obj.ref is not None else (
        ("lit", repr(obj.literal)) if obj.has_literal else
        ("unres", obj.unresolved, obj.missing))
    return (skey, prop.predicate, okey)


def _literal_proposition(pattern: WorldLiteralPattern) -> WorldProposition:
    def term_of(t: WorldPatternTerm) -> WorldTerm:
        if t.ref is not None:
            return WorldTerm(ref=t.ref)
        if t.has_literal:
            return WorldTerm(literal=t.literal, has_literal=True)
        return WorldTerm(unresolved="unbound")

    return WorldProposition(
        subject=term_of(pattern.subject), predicate=pattern.predicate,
        object=term_of(pattern.object), polarity=pattern.polarity)


def _match_term(term: WorldPatternTerm, fact_term: WorldTerm,
                bindings: dict) -> Optional[dict]:
    """One slot via the shared kernel; world policy: only refs bind."""
    kb = {name: ("ref", ref.ref_id)
          for name, ref in bindings.items()}
    out = matching.unify(
        matching.pattern_term(term), matching.fact_term(fact_term), kb,
        bindable=lambda fact: fact[0] == "ref")
    if out is None:
        return None
    merged = dict(bindings)
    for name in out:
        if name not in kb:
            assert fact_term.ref is not None
            merged[name] = fact_term.ref
    return merged


def _match_premise(premise: WorldLiteralPattern, facts: list,
                   bindings: dict) -> list:
    out = []
    for fact in facts:
        prop = fact.proposition
        if prop.predicate != premise.predicate:
            continue
        b1 = _match_term(premise.subject, prop.subject, bindings)
        if b1 is None:
            continue
        b2 = _match_term(premise.object, prop.object, b1)
        if b2 is None:
            continue
        out.append((b2, fact))
    return out


def _instantiate(pattern: WorldLiteralPattern,
                 bindings: dict) -> Optional[WorldLiteralPattern]:
    def ground(term: WorldPatternTerm) -> Optional[WorldPatternTerm]:
        if term.is_variable():
            bound = bindings.get(term.variable)
            if bound is None:
                return None
            return WorldPatternTerm(ref=bound)
        return term

    subj, obj = ground(pattern.subject), ground(pattern.object)
    if subj is None or obj is None:
        return None
    return WorldLiteralPattern(subject=subj, predicate=pattern.predicate,
                               object=obj, polarity=pattern.polarity)


def _rule_bindings(rule: WorldInferenceRule, pool: list) -> list:
    results: list = [{}]
    for premise in rule.premises:
        stepped = []
        for bindings in results:
            for new_bindings, _fact in _match_premise(premise, pool,
                                                      bindings):
                stepped.append(new_bindings)
        results = stepped
        if not results:
            break
    seen = []
    for b in results:
        if b not in seen:
            seen.append(b)
    return seen


def _premise_proofs(rule: WorldInferenceRule, bindings: dict,
                    state: WorldState, derived: dict) -> Optional[list]:
    subs = []
    for premise in rule.premises:
        found = None
        for i, fact in enumerate(state.facts):
            if fact.proposition.predicate != premise.predicate:
                continue
            if _match_term(premise.subject, fact.proposition.subject,
                           bindings) is None:
                continue
            if _match_term(premise.object, fact.proposition.object,
                           bindings) is None:
                continue
            found = WorldProofNode(
                kind="fact", state_id=state.state_id, fact_index=i,
                proposition=fact.proposition, depth=0)
            break
        if found is None:
            for (_prop, node, _d) in derived.values():
                prop = node.proposition
                if prop.predicate != premise.predicate:
                    continue
                if _match_term(premise.subject, prop.subject,
                               bindings) is None:
                    continue
                if _match_term(premise.object, prop.object,
                               bindings) is None:
                    continue
                found = node
                break
        if found is None:
            return None
        subs.append(found)
    return subs


def reason_about_state(state: WorldState, query: WorldLiteralPattern,
                       rules, max_depth: Optional[int] = None) -> Any:
    """Reason about one query inside one state. Read-only, bounded."""
    if max_depth is None:
        max_depth = MAX_CHAIN_DEPTH
    compiled = [r for r in rules
                if isinstance(r, WorldInferenceRule)]
    views = [r for r in rules if not isinstance(r, WorldInferenceRule)]
    uncompilable: list = []
    complete = True
    if views:
        outcome = compile_world_rules(views)
        compiled.extend(outcome["rules"])
        uncompilable.extend(outcome["refused"])
        if not outcome["complete"]:
            complete = False
    for rule in compiled:
        if validate_inference_rule(rule):
            return WorldReasoningResult(
                query=query, state_id=state.state_id, verdict="invalid",
                complete=False, uncompilable=tuple(uncompilable),
                source_refs=(f"state:{state.state_id}",))
    if _pattern_terms_complete(query) is not None:
        return WorldReasoningResult(
            query=query, state_id=state.state_id, verdict="invalid",
            complete=False, uncompilable=tuple(uncompilable),
            source_refs=(f"state:{state.state_id}",))
    for term in (query.subject, query.object):
        if term.is_variable():
            return WorldReasoningResult(
                query=query, state_id=state.state_id, verdict="invalid",
                complete=False, uncompilable=tuple(uncompilable),
                source_refs=(f"state:{state.state_id}",))
    if not isinstance(max_depth, int) or max_depth < 0:
        return WorldReasoningResult(
            query=query, state_id=state.state_id, verdict="invalid",
            complete=False, uncompilable=tuple(uncompilable),
            source_refs=(f"state:{state.state_id}",))

    qkey = _fact_key(_literal_proposition(query))
    qpol = query.polarity
    oppos = NEGATIVE if qpol == POSITIVE else POSITIVE
    positives: list = []
    negatives: list = []
    for i, fact in enumerate(state.facts):
        if _fact_key(fact.proposition) != qkey:
            continue
        leaf = WorldProofNode(
            kind="fact", state_id=state.state_id, fact_index=i,
            proposition=fact.proposition, depth=0)
        if fact.proposition.polarity == qpol:
            positives.append(leaf)
        elif fact.proposition.polarity == oppos:
            negatives.append(leaf)

    derived: dict = {}
    fired: set = set()
    capped = False
    depth_reached = 0
    for depth in range(1, max_depth + 1):
        pool_facts = list(state.facts) + [
            _PropFact(node.proposition) for (_p, node, _d)
            in derived.values()]
        fresh: dict = {}
        for rule in compiled:
            for bindings in _rule_bindings(rule, pool_facts):
                inst = _instantiate(rule.conclusion, bindings)
                if inst is None:
                    continue
                key = (_fact_key(_literal_proposition(inst)),
                       inst.polarity)
                if key in derived or key in fresh:
                    continue
                subs = _premise_proofs(rule, bindings, state, derived)
                if subs is None:
                    continue
                node = WorldProofNode(
                    kind="rule", state_id=state.state_id,
                    proposition=_literal_proposition(inst),
                    rule_id=rule.rule_id,
                    bindings=tuple(sorted(bindings.items(),
                                           key=lambda kv: kv[0])),
                    conclusion=inst, premises=tuple(subs), depth=depth)
                fresh[key] = (node, rule.rule_id)
                fired.add(rule.rule_id)
        if not fresh:
            break
        for key, (node, _rid) in fresh.items():
            derived[key] = (node.proposition, node, depth)
        depth_reached = depth
        if depth == max_depth and fresh:
            capped = True
    for key, (_prop, node, _d) in derived.items():
        if key[0] != qkey:
            continue
        if key[1] == qpol:
            positives.append(node)
        elif key[1] == oppos:
            negatives.append(node)

    if positives and negatives:
        verdict = "conflicted"
    elif positives or negatives:
        verdict = "limited" if (capped or not complete) else (
            "entailed" if positives else "refuted")
    else:
        verdict = "limited" if (capped or not complete) else "unknown"
    return WorldReasoningResult(
        query=query, state_id=state.state_id, verdict=verdict,
        positive_proofs=tuple(positives),
        negative_proofs=tuple(negatives),
        rules_used=tuple(sorted(fired)),
        max_derivation_depth=depth_reached,
        complete=complete and not capped,
        uncompilable=tuple(uncompilable),
        source_refs=(f"state:{state.state_id}",))


class _PropFact:
    """Minimal .proposition carrier so derived nodes join the pool."""

    def __init__(self, proposition):
        self.proposition = proposition


def verify_world_proof(state: WorldState, query: WorldLiteralPattern,
                       proof: Any, rules, max_depth: Optional[int] = None,
                       _trail: tuple = ()) -> dict:
    """Independently verify one proof tree. Read-only.

    The result says whether the proof structure is valid, never
    whether the proposition is verified in reality ("valid" here is
    deliberately distinct from the knowledge status "verified").
    """
    if max_depth is None:
        max_depth = MAX_CHAIN_DEPTH
    compiled = [r for r in rules if isinstance(r, WorldInferenceRule)]
    views = [r for r in rules if not isinstance(r, WorldInferenceRule)]
    if views:
        outcome = compile_world_rules(views)
        compiled.extend(outcome["rules"])
    by_id = {r.rule_id: r for r in compiled}
    if not isinstance(proof, WorldProofNode):
        return {"result": "invalid", "detail": "not a proof node"}
    if proof.kind == "fact":
        return _verify_leaf(state, proof)
    if proof.kind != "rule":
        return {"result": "invalid",
                "detail": f"bad node kind {proof.kind!r}"}
    rule = by_id.get(proof.rule_id)
    if rule is None:
        return {"result": "invalid",
                "detail": f"rule {proof.rule_id!r} not available"}
    bindings = dict(proof.bindings)
    if set(bindings) != _rule_vars(rule):
        return {"result": "invalid",
                "detail": "bindings do not match rule variables"}
    if len(proof.premises) != len(rule.premises):
        return {"result": "invalid",
                "detail": f"{len(proof.premises)} subproofs for "
                f"{len(rule.premises)} premises"}
    node_key = (proof.rule_id, _fact_key(proof.proposition))
    if node_key in _trail:
        return {"result": "invalid", "detail": "proof cycle detected"}
    inst = _instantiate(rule.conclusion, bindings)
    if inst is None:
        return {"result": "invalid",
                "detail": "bindings leave conclusion ungrounded"}
    if _fact_key(_literal_proposition(inst)) != \
            _fact_key(proof.proposition):
        return {"result": "invalid",
                "detail": "node proposition differs from rule conclusion"}
    if _fact_key(_literal_proposition(proof.conclusion)) != \
            _fact_key(proof.proposition):
        return {"result": "invalid",
                "detail": "node conclusion differs from proposition"}
    if proof.depth > max_depth:
        return {"result": "invalid",
                "detail": f"depth {proof.depth} exceeds {max_depth}"}
    for cond, sub in zip(rule.premises, proof.premises):
        inst_cond = _instantiate(cond, bindings)
        if inst_cond is None:
            return {"result": "invalid",
                    "detail": "bindings leave a premise ungrounded"}
        if _fact_key(_literal_proposition(inst_cond)) != \
                _fact_key(_node_pattern(sub)):
            return {"result": "invalid",
                    "detail": "subproof supports a different proposition"}
        sub_res = verify_world_proof(
            state, query, sub, compiled, max_depth,
            _trail + (node_key,))
        if sub_res["result"] != "valid":
            return {"result": "invalid",
                    "detail": f"subproof failed: {sub_res['detail']}"}
    expect = 1 + max([s.depth for s in proof.premises]) \
        if proof.premises else 1
    if proof.depth != expect:
        return {"result": "invalid",
                "detail": f"depth {proof.depth} != 1+max(premises) "
                f"{expect}"}
    return {"result": "valid",
            "detail": f"rule {proof.rule_id}, "
            f"{len(proof.premises)} premise(s) re-derived"}


def _rule_vars(rule: WorldInferenceRule) -> set:
    found: set = set()
    for prem in list(rule.premises) + [rule.conclusion]:
        for term in (prem.subject, prem.object):
            if term.is_variable():
                found.add(term.variable)
    return found


def _node_pattern(node: WorldProofNode) -> WorldLiteralPattern:
    prop = node.proposition

    def term_of(t: WorldTerm) -> WorldPatternTerm:
        if t.ref is not None:
            return WorldPatternTerm(ref=t.ref)
        if t.has_literal:
            return WorldPatternTerm(literal=t.literal, has_literal=True)
        return WorldPatternTerm()

    return WorldLiteralPattern(
        subject=term_of(prop.subject), predicate=prop.predicate,
        object=term_of(prop.object), polarity=prop.polarity)


def _verify_leaf(state: WorldState, proof: WorldProofNode) -> dict:
    idx = proof.fact_index
    if not isinstance(idx, int) or idx < 0 or idx >= len(state.facts):
        return {"result": "invalid",
                "detail": "fact index outside this state"}
    if proof.state_id != state.state_id:
        return {"result": "invalid",
                "detail": "leaf cites another state"}
    fact = state.facts[idx]
    if _fact_key(fact.proposition) != _fact_key(proof.proposition):
        return {"result": "invalid",
                "detail": "no matching base fact at that index"}
    if fact.proposition.polarity != proof.proposition.polarity:
        return {"result": "invalid",
                "detail": "polarity differs from stored fact"}
    return {"result": "valid", "detail": "base fact holds"}
