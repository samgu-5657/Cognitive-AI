"""Prediction review & diagnosis (Phase M4-G).

Reads forecasts, comparisons and states; re-verifies every claimed
proof with the real verifier; classifies differences; emits
pending-review candidates. Pure memory, deterministic, no IO: no
SQLite, Memory, Learner, SchemaEvolver, files, network, LLM or
randomness. Never writes knowledge, never confirms a root cause.
"""
from __future__ import annotations

from typing import Any, Optional

from . import world_reasoning as _reasoning
from .models import (WorldForecast, WorldLiteralPattern,
                     WorldPredictionComparison, WorldDiagnosticCandidate,
                     WorldDiagnosticHypothesis, WorldHypothesisKind,
                     WorldProofNode, WorldReviewClassification,
                     WorldReviewItemResult, WorldReviewReport, WorldState)

CONSISTENT = WorldReviewClassification.CONSISTENT.value
CONFIRMED_MISMATCH = WorldReviewClassification.CONFIRMED_MISMATCH.value
REFERENCE_UNKNOWN = WorldReviewClassification.REFERENCE_UNKNOWN.value
FORECAST_INCONCLUSIVE = WorldReviewClassification.FORECAST_INCONCLUSIVE.value
CONFLICTED = WorldReviewClassification.CONFLICTED.value
LIMITED = WorldReviewClassification.LIMITED.value
INVALID = WorldReviewClassification.INVALID.value

_OVERALL_PRIORITY = (INVALID, CONFLICTED, LIMITED, CONFIRMED_MISMATCH,
                     REFERENCE_UNKNOWN, FORECAST_INCONCLUSIVE, CONSISTENT)


def _key_of(proposition) -> tuple:
    subj = proposition.subject
    obj = proposition.object
    skey = ("ref", subj.ref.ref_id) if subj.ref is not None else (
        ("lit", repr(subj.literal)) if subj.has_literal else
        ("unres", subj.unresolved, subj.missing))
    okey = ("ref", obj.ref.ref_id) if obj.ref is not None else (
        ("lit", repr(obj.literal)) if obj.has_literal else
        ("unres", obj.unresolved, obj.missing))
    return (skey, proposition.predicate, okey)


def _pattern_proposition(query) -> tuple:
    from .models import WorldProposition, WorldTerm

    def term_of(t):
        if t.ref is not None:
            return WorldTerm(ref=t.ref)
        if t.has_literal:
            return WorldTerm(literal=t.literal, has_literal=True)
        return WorldTerm(unresolved="unbound")

    return WorldProposition(
        subject=term_of(query.subject), predicate=query.predicate,
        object=term_of(query.object), polarity=query.polarity)


def _query_key(query: WorldLiteralPattern) -> tuple:
    return _key_of(_pattern_proposition(query))


def _proof_ref(proof: WorldProofNode) -> str:
    if proof.kind == "fact":
        return f"fact:{proof.state_id}#{proof.fact_index}"
    return f"rule:{proof.rule_id}@{proof.depth}"


def _valid_proofs(state: WorldState, query: WorldLiteralPattern,
                  proofs, rules, max_depth) -> tuple:
    """Split claimed proofs into (valid_positive, valid_negative).

    Positive means concluding the query key with the query polarity,
    negative the same key with the opposite polarity. Every proof is
    re-verified; verdict labels are never trusted.
    """
    qkey = _query_key(query)
    qpol = query.polarity
    oppos = "negative" if qpol == "positive" else "positive"
    pos, neg = [], []
    for proof in proofs:
        if not isinstance(proof, WorldProofNode):
            continue
        if proof.state_id != state.state_id:
            continue
        if _key_of(proof.proposition) != qkey:
            continue
        res = _reasoning.verify_world_proof(
            state, query, proof, rules, max_depth)
        if res.get("result") != "valid":
            continue
        if proof.proposition.polarity == qpol:
            pos.append(proof)
        elif proof.proposition.polarity == oppos:
            neg.append(proof)
    return tuple(pos), tuple(neg)


def _standard_hypotheses(action_id: str, rule_ids: tuple,
                         state_ids: tuple) -> tuple:
    action_refs = (action_id,) if action_id else ()
    return (
        WorldDiagnosticHypothesis(
            kind=WorldHypothesisKind.ACTION_MODEL_MAY_BE_WRONG.value,
            statement="check the action premises and effects",
            refs=action_refs + state_ids),
        WorldDiagnosticHypothesis(
            kind=WorldHypothesisKind.
            INFERENCE_RULE_MAY_BE_WRONG_OR_MISSING.value,
            statement="check the inference rules involved",
            refs=tuple(rule_ids) + state_ids),
        WorldDiagnosticHypothesis(
            kind=WorldHypothesisKind.UNMODELED_WORLD_CHANGE.value,
            statement="check for state changes the action model "
            "does not represent",
            refs=state_ids),
        WorldDiagnosticHypothesis(
            kind=WorldHypothesisKind.
            REFERENCE_SCOPE_OR_PROVENANCE_UNCERTAIN.value,
            statement="check the reference state's source and scope",
            refs=state_ids),
    )


def review_prediction(forecast: WorldForecast,
                      comparison: WorldPredictionComparison,
                      reference_state: WorldState, rules,
                      max_depth: Optional[int] = None) -> WorldReviewReport:
    """Review one forecast against an independent reference state."""
    if max_depth is None:
        from .policy import MAX_CHAIN_DEPTH
        max_depth = MAX_CHAIN_DEPTH
    limitations = ["reference provenance must be checked by the caller; "
                   "a different id never proves real-world origin"]

    def invalid_report(reason: str, start="", forecast_id="",
                       reference_id="") -> WorldReviewReport:
        return WorldReviewReport(
            start_state_id=start, forecast_state_id=forecast_id,
            reference_state_id=reference_id,
            items=(WorldReviewItemResult(classification=INVALID,
                                         detail=reason),),
            overall=INVALID, complete=False,
            limitations=tuple(limitations + [reason]))

    if not isinstance(forecast, WorldForecast) or \
            not isinstance(comparison, WorldPredictionComparison) or \
            not isinstance(reference_state, WorldState):
        return invalid_report("forecast, comparison or reference invalid")
    try:
        rule_list = list(rules)
    except TypeError:
        return invalid_report("rules not iterable")
    predicted = forecast.predicted_state
    if predicted is None or not isinstance(predicted, WorldState):
        return invalid_report("forecast has no predicted state")
    if comparison.start_state_id != forecast.start_state_id or \
            comparison.forecast_state_id != predicted.state_id or \
            comparison.reference_state_id != reference_state.state_id:
        return invalid_report("id linkage broken between forecast, "
                              "comparison and reference")
    if reference_state.state_id == predicted.state_id:
        return invalid_report("reference must differ from predicted state")
    if len(comparison.items) != len(forecast.queries) or \
            len(forecast.queries) != len(forecast.results) or \
            not forecast.queries:
        return invalid_report("query/result association broken or empty")

    items = []
    candidates = []
    for index, (query, fres, item) in enumerate(zip(
            forecast.queries, forecast.results, comparison.items)):
        items.append(_review_item(
            index, query, fres, item, forecast, comparison,
            predicted, reference_state, rule_list, max_depth,
            candidates))
    present = {it.classification for it in items}
    overall = next(c for c in _OVERALL_PRIORITY if c in present)
    complete = overall == CONSISTENT
    return WorldReviewReport(
        start_state_id=forecast.start_state_id,
        forecast_state_id=predicted.state_id,
        reference_state_id=reference_state.state_id,
        items=tuple(items), candidates=tuple(candidates),
        overall=overall, complete=complete,
        limitations=tuple(limitations))


def _review_item(index: int, query, fres, item, forecast, comparison,
                 predicted: WorldState, reference_state: WorldState,
                 rules: list, max_depth: int,
                 candidates: list) -> WorldReviewItemResult:
    if item.query != query or item.forecast_verdict != fres.verdict:
        return WorldReviewItemResult(
            query=query, classification=INVALID,
            detail="comparison item does not match forecast query/result")
    fpos, fneg = _valid_proofs(predicted, query, item.forecast_proofs,
                               rules, max_depth)
    rpos, rneg = _valid_proofs(reference_state, query,
                               item.reference_proofs, rules, max_depth)
    f_decided = fres.verdict in ("entailed", "refuted")
    f_side = fpos if fres.verdict == "entailed" else (
        fneg if fres.verdict == "refuted" else ())
    r_side = rpos if item.reference_verdict == "entailed" else (
        rneg if item.reference_verdict == "refuted" else ())

    def mismatch_pair() -> bool:
        return {fres.verdict, item.reference_verdict} == \
            {"entailed", "refuted"}

    # Invalid: decided claims without valid proofs, or conflict claims
    # without both sides proven.
    need_invalid = False
    if fres.verdict in ("entailed", "refuted") and not f_side:
        need_invalid = True
    if item.reference_verdict in ("entailed", "refuted") and not r_side:
        need_invalid = True
    if fres.verdict == "conflicted" and not (fpos and fneg):
        need_invalid = True
    if item.reference_verdict == "conflicted" and not (rpos and rneg):
        need_invalid = True
    if need_invalid:
        return WorldReviewItemResult(
            query=query, classification=INVALID,
            detail="a decided claim lacks a valid proof")
    if (fpos and fneg) or (rpos and rneg):
        detail = "valid proofs on both polarities"
        if mismatch_pair():
            detail += "; an opposite decided pair is also present"
        return WorldReviewItemResult(
            query=query, classification=CONFLICTED, detail=detail)
    if fres.verdict == "limited" or \
            item.reference_verdict == "limited":
        return WorldReviewItemResult(
            query=query, classification=LIMITED,
            detail="bounded or incomplete reasoning on some side")
    if mismatch_pair():
        if not fres.complete:
            return WorldReviewItemResult(
                query=query, classification=LIMITED,
                detail="opposite verdicts but forecast reasoning "
                "incomplete")
        candidate = WorldDiagnosticCandidate(
            query=query, forecast_verdict=fres.verdict,
            reference_verdict=item.reference_verdict,
            comparison_category=item.category,
            classification=CONFIRMED_MISMATCH,
            hypotheses=_standard_hypotheses(
                forecast.action_id, fres.rules_used,
                (forecast.start_state_id,
                 forecast.predicted_state.state_id,
                 reference_state.state_id)),
            forecast_proof_refs=tuple(_proof_ref(p) for p in
                                        (fpos + fneg)),
            reference_proof_refs=tuple(_proof_ref(p) for p in
                                       (rpos + rneg)),
            action_id=forecast.action_id, rule_ids=fres.rules_used,
            state_ids=(forecast.start_state_id,
                       forecast.predicted_state.state_id,
                       reference_state.state_id),
            reason="opposite complete verified conclusions on both sides",
            status="pending_review", simulated=True)
        candidates.append(candidate)
        return WorldReviewItemResult(
            query=query, classification=CONFIRMED_MISMATCH,
            detail="opposite complete verified conclusions",
            candidate=candidate)
    if fres.verdict == "unknown":
        return WorldReviewItemResult(
            query=query, classification=FORECAST_INCONCLUSIVE,
            detail="forecast derives nothing; not a prediction error")
    if item.reference_verdict == "unknown":
        return WorldReviewItemResult(
            query=query, classification=REFERENCE_UNKNOWN,
            detail="reference state lacks supporting derivation; "
            "no rule blamed")
    if fres.verdict == item.reference_verdict and \
            fres.verdict in ("entailed", "refuted"):
        if not fres.complete or not comparison.complete:
            return WorldReviewItemResult(
                query=query, classification=LIMITED,
                detail="matching verdicts but reasoning incomplete")
        return WorldReviewItemResult(
            query=query, classification=CONSISTENT,
            detail=f"both {fres.verdict} with valid proofs")
    return WorldReviewItemResult(
        query=query, classification=INVALID,
        detail="verdict combination not classifiable")
