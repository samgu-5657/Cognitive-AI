"""Deterministic symbolic action simulation (Phase M4-C).

Pure functions over immutable snapshots. No database, file, network,
randomness, LLM, Memory, Ontology or Learner access. Unknown premises
never pass silently; conflicts are kept and reported; the input state
is never modified.
"""
from __future__ import annotations

from typing import Any, Optional

from .models import (WorldActionSchema, WorldEffect, WorldEvent,
                     WorldLiteralPattern, WorldObjectRef, WorldPatternTerm,
                     WorldPreconditionCheck, WorldProposition,
                     WorldState, WorldStateFact, WorldTerm,
                     WorldTransitionOutcome, WorldTransitionResult,
                     WorldModality, WorldPolarity, _term_key)

TRUE = "true"
FALSE = "false"
UNKNOWN = "unknown"
CONFLICTED = "conflicted"
INVALID = "invalid"


def _pattern_variables(action: WorldActionSchema) -> set:
    found: set = set()

    def collect(term: WorldPatternTerm) -> None:
        if term.is_variable():
            found.add(term.variable)

    for prem in action.premises:
        collect(prem.subject)
        collect(prem.object)
    for eff in action.effects:
        collect(eff.target.subject)
        collect(eff.target.object)
    return found


def validate_action_schema(action: WorldActionSchema) -> tuple:
    """Check an action definition; () means valid. Pure inspection."""
    problems = []
    if not action.action_id:
        problems.append("action_id missing")
    params = list(action.params)
    if len(set(params)) != len(params):
        problems.append("duplicate parameter names")
    for p in params:
        if not p:
            problems.append("empty parameter name")
    used = _pattern_variables(action)
    for v in sorted(used):
        if v not in params:
            problems.append(f"unbound variable {v!r}")
    if not action.effects:
        problems.append("no effects defined")
    seen_targets: dict = {}
    for eff in action.effects:
        tgt = eff.target
        key = (_ground_term_key(tgt.subject, {}),
               tgt.predicate,
               _ground_term_key(tgt.object, {}),
               str(tgt.polarity))
        if key in seen_targets and seen_targets[key] != str(tgt.polarity):
            problems.append(
                f"contradictory effects on {tgt.predicate}")
        seen_targets.setdefault(key, str(tgt.polarity))
    return tuple(problems)


def _ground_term_key(term: WorldPatternTerm, bindings: dict) -> tuple:
    if term.is_variable():
        bound = bindings.get(term.variable)
        if bound is None:
            return ("unbound", term.variable)
        return ("ref", bound.ref_id)
    if term.ref is not None:
        return ("ref", term.ref.ref_id)
    if term.has_literal:
        return ("lit", repr(term.literal))
    return ("unres", "")


def _match_term(term: WorldPatternTerm, fact_term: WorldTerm,
                bindings: dict) -> bool:
    if term.is_variable():
        bound = bindings.get(term.variable)
        if bound is None:
            return False
        return fact_term.ref is not None and \
            fact_term.ref.ref_id == bound.ref_id
    if term.ref is not None:
        return fact_term.ref is not None and \
            fact_term.ref.ref_id == term.ref.ref_id
    if term.has_literal:
        return fact_term.has_literal and fact_term.literal == term.literal
    return False


def _fact_key_of(prop: WorldProposition) -> tuple:
    return (_term_key(prop.subject), prop.predicate,
            _term_key(prop.object))


def evaluate_state_literal(state: WorldState,
                           literal: WorldLiteralPattern) -> str:
    """Four-valued lookup: true | false | unknown | conflicted.

    Positive-only facts mean true, negative-only mean false, neither
    means unknown, both mean conflicted. Absence is never negation.
    """
    key = (_ground_term_key(literal.subject, {}),
           literal.predicate,
           _ground_term_key(literal.object, {}))
    if ("unbound",) == key[0][:1] or ("unbound",) == key[2][:1]:
        return INVALID
    if ("unres",) == key[0][:1] or ("unres",) == key[2][:1]:
        return INVALID
    pos = neg = False
    for f in state.facts:
        if _fact_key_of(f.proposition) != key:
            continue
        if f.proposition.polarity == WorldPolarity.POSITIVE:
            pos = True
        elif f.proposition.polarity == WorldPolarity.NEGATIVE:
            neg = True
    if pos and neg:
        return CONFLICTED
    if pos:
        return TRUE
    if neg:
        return FALSE
    return UNKNOWN


def _check_one(state: WorldState, premise: WorldLiteralPattern,
               bindings: dict) -> WorldPreconditionCheck:
    ground_sub = _ground_term_key(premise.subject, bindings)
    ground_obj = _ground_term_key(premise.object, bindings)
    if ground_sub[0] in ("unbound", "unres") or \
            ground_obj[0] in ("unbound", "unres"):
        return WorldPreconditionCheck(
            premise=premise, outcome=INVALID, matched=(),
            reason="pattern term did not resolve to a bound value")
    key = (ground_sub, premise.predicate, ground_obj)
    pos_facts: list = []
    neg_facts: list = []
    for f in state.facts:
        if _fact_key_of(f.proposition) != key:
            continue
        if f.proposition.polarity == WorldPolarity.POSITIVE:
            pos_facts.append(f)
        elif f.proposition.polarity == WorldPolarity.NEGATIVE:
            neg_facts.append(f)
    want_pos = premise.polarity == WorldPolarity.POSITIVE
    if pos_facts and neg_facts:
        return WorldPreconditionCheck(
            premise=premise, outcome=CONFLICTED,
            matched=tuple(pos_facts + neg_facts),
            reason="affirming and denying state facts both present")
    if want_pos:
        if pos_facts:
            return WorldPreconditionCheck(
                premise=premise, outcome=TRUE, matched=tuple(pos_facts),
                reason="affirming state fact present")
        if neg_facts:
            return WorldPreconditionCheck(
                premise=premise, outcome=FALSE, matched=tuple(neg_facts),
                reason="only denying state facts present")
        return WorldPreconditionCheck(
            premise=premise, outcome=UNKNOWN, matched=(),
            reason="no state record for this proposition")
    if neg_facts:
        return WorldPreconditionCheck(
            premise=premise, outcome=TRUE, matched=tuple(neg_facts),
            reason="denying state fact present")
    if pos_facts:
        return WorldPreconditionCheck(
            premise=premise, outcome=FALSE, matched=tuple(pos_facts),
            reason="only affirming state facts present")
    return WorldPreconditionCheck(
        premise=premise, outcome=UNKNOWN, matched=(),
        reason="no state record for this proposition")


def check_action_preconditions(
        state: WorldState, action: WorldActionSchema,
        bindings: dict) -> tuple:
    """Check every premise; one result per premise, in order."""
    return tuple(_check_one(state, p, bindings) for p in action.premises)


def _validate_bindings(action: WorldActionSchema,
                       bindings: dict) -> Optional[str]:
    params = list(action.params)
    if not isinstance(bindings, dict):
        return "bindings must be a mapping"
    missing = [p for p in params if p not in bindings]
    if missing:
        return f"missing bindings for {missing}"
    extra = [k for k in bindings if k not in params]
    if extra:
        return f"undeclared parameters {extra}"
    for name, ref in bindings.items():
        if not isinstance(ref, WorldObjectRef) or not ref.ref_id:
            return f"parameter {name!r} did not resolve to a stable id"
    return None


def _copy_fact(fact: WorldStateFact) -> WorldStateFact:
    prop = fact.proposition
    return WorldStateFact(
        proposition=WorldProposition(
            subject=prop.subject, object=prop.object,
            predicate=prop.predicate, polarity=prop.polarity,
            modality=prop.modality,
            representation_origin=prop.representation_origin,
            source_ref=prop.source_ref, raw=dict(prop.raw)),
        event_ref=fact.event_ref, origin=fact.origin)


def make_state(state_id: str, step_index: int, facts: list,
               parent_state_id: str = "") -> WorldState:
    """Build a snapshot; caller containers are copied, never aliased."""
    return WorldState(
        state_id=state_id, step_index=step_index,
        facts=tuple(_copy_fact(f) for f in facts),
        parent_state_id=parent_state_id)


def _effect_fact(effect: WorldEffect, bindings: dict,
                 event_id: str) -> WorldStateFact:
    tgt = effect.target

    def ground(term: WorldPatternTerm) -> WorldTerm:
        if term.is_variable():
            return WorldTerm(ref=bindings[term.variable])
        if term.ref is not None:
            return WorldTerm(ref=term.ref)
        return WorldTerm(literal=term.literal, has_literal=True)

    prop = WorldProposition(
        subject=ground(tgt.subject), predicate=tgt.predicate,
        object=ground(tgt.object), polarity=tgt.polarity,
        modality=WorldModality.ASSERTED, representation_origin="simulation",
        source_ref=f"event:{event_id}", raw={})
    return WorldStateFact(proposition=prop, event_ref=event_id,
                          origin="simulation")


def simulate_action(state: WorldState, action: WorldActionSchema,
                    bindings: dict, new_state_id: str = "",
                    event_id: str = "") -> WorldTransitionResult:
    """Simulate one action. Input state is never modified; no IO."""
    problems = validate_action_schema(action)
    if problems:
        return WorldTransitionResult(
            outcome=WorldTransitionOutcome.INVALID, checks=(),
            reason="; ".join(problems))
    bind_error = _validate_bindings(action, bindings)
    if bind_error is not None:
        return WorldTransitionResult(
            outcome=WorldTransitionOutcome.INVALID, checks=(),
            reason=bind_error)
    checks = check_action_preconditions(state, action, bindings)
    outcomes = [c.outcome for c in checks]
    if any(o == INVALID for o in outcomes):
        return WorldTransitionResult(
            outcome=WorldTransitionOutcome.INVALID, checks=checks,
            reason="a premise term failed to resolve")
    if any(o == CONFLICTED for o in outcomes):
        return WorldTransitionResult(
            outcome=WorldTransitionOutcome.CONFLICTED, checks=checks,
            reason="a premise is explicitly contested in this state")
    if any(o == FALSE for o in outcomes):
        return WorldTransitionResult(
            outcome=WorldTransitionOutcome.BLOCKED, checks=checks,
            reason="a premise is denied by this state")
    if any(o == UNKNOWN for o in outcomes):
        return WorldTransitionResult(
            outcome=WorldTransitionOutcome.UNKNOWN, checks=checks,
            reason="a premise has no supporting state record")
    step = state.step_index + 1
    new_id = new_state_id or \
        f"{state.state_id}::{action.action_id}::{step}"
    ev_id = event_id or f"ev::{new_id}"
    bound_names = sorted(bindings)
    effect_facts = [_effect_fact(e, bindings, ev_id)
                    for e in action.effects]
    drop_keys = {_fact_key_of(f.proposition) for f in effect_facts}
    kept = [_copy_fact(f) for f in state.facts
            if _fact_key_of(f.proposition) not in drop_keys]
    new_state = WorldState(
        state_id=new_id, step_index=step,
        facts=tuple(kept + effect_facts),
        parent_state_id=state.state_id)
    event = WorldEvent(
        event_id=ev_id, action_id=action.action_id,
        bindings=tuple((n, bindings[n]) for n in bound_names),
        from_state=state.state_id, to_state=new_id, step_index=step,
        effects=tuple(action.effects), simulated=True)
    return WorldTransitionResult(
        outcome=WorldTransitionOutcome.APPLIED, checks=checks,
        reason=f"applied {len(effect_facts)} effect(s)",
        new_state=new_state, event=event)
