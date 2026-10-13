"""Phase M4-G review & diagnosis (pure memory, no DB/files)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import world_dynamics as wd
from cognitive_ai import world_prediction as wp
from cognitive_ai import world_review as wv
from cognitive_ai.models import (WorldActionSchema, WorldEffect,
                                 WorldLiteralPattern, WorldObjectRef,
                                 WorldPatternTerm, WorldProposition,
                                 WorldStateFact, WorldTerm,
                                 WorldObjectKind, WorldPolarity)

CUP = WorldObjectRef("cupA", "杯子A", WorldObjectKind.INDIVIDUAL)
WATER = WorldObjectRef("waterA", "水A", WorldObjectKind.INDIVIDUAL)
TABLE = WorldObjectRef("tableB", "桌子B", WorldObjectKind.INDIVIDUAL)


def ref_term(ref):
    return WorldTerm(ref=ref)


def var(name):
    return WorldPatternTerm(variable=name)


def const(ref):
    return WorldPatternTerm(ref=ref)


def lit_const(value):
    return WorldPatternTerm(literal=value, has_literal=True)


def fact(subj, pred, obj, polarity):
    return WorldStateFact(
        proposition=WorldProposition(
            subject=ref_term(subj), predicate=pred,
            object=ref_term(obj), polarity=polarity),
        origin="test")


def pat(subj, pred, obj, polarity=WorldPolarity.POSITIVE):
    s = const(subj) if isinstance(subj, WorldObjectRef) else subj
    o = const(obj) if isinstance(obj, WorldObjectRef) else obj
    return WorldLiteralPattern(subject=s, predicate=pred, object=o,
                               polarity=polarity)


def pour_action():
    return WorldActionSchema(
        action_id="pour", name="倒入", params=("cup", "liquid"),
        premises=(WorldLiteralPattern(
            subject=var("liquid"), predicate="inside",
            object=var("cup"), polarity=WorldPolarity.NEGATIVE),),
        effects=(WorldEffect(WorldLiteralPattern(
            subject=var("liquid"), predicate="inside",
            object=var("cup"),
            polarity=WorldPolarity.POSITIVE)),))


def wet_rule():
    from cognitive_ai.models import WorldInferenceRule
    return WorldInferenceRule(
        rule_id="wet", premises=(WorldLiteralPattern(
            subject=var("liquid"), predicate="inside",
            object=var("cup"), polarity=WorldPolarity.POSITIVE),),
        conclusion=WorldLiteralPattern(
            subject=var("cup"), predicate="wet",
            object=lit_const(True), polarity=WorldPolarity.POSITIVE),
        source_ref="test:wet", status="active")


def start_state():
    return wd.make_state("s0", 0, [fact(WATER, "inside", CUP,
                                        WorldPolarity.NEGATIVE)])


def queries():
    return [pat(WATER, "inside", CUP)]


def run_case(reference):
    s0 = start_state()
    fc = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                            queries(), [wet_rule()],
                            new_state_id="s1", event_id="e1")
    assert fc.transition.outcome == "applied"
    cmp = wp.compare_forecast(fc, reference, [wet_rule()])
    return fc, cmp


# ---- A. consistent ----
def test_review_consistent():
    ref = wd.make_state("ref", 0, [fact(WATER, "inside", CUP,
                                        WorldPolarity.POSITIVE)])
    fc, cmp = run_case(ref)
    assert cmp.overall == "consistent"
    rep = wv.review_prediction(fc, cmp, ref, [wet_rule()])
    assert rep.overall == "consistent" and rep.complete is True
    assert rep.candidates == ()
    assert rep.items[0].classification == "consistent"
    assert rep.start_state_id == "s0"
    assert rep.forecast_state_id == "s1"
    assert rep.reference_state_id == "ref"


# ---- B. confirmed mismatch ----
def test_review_confirmed_mismatch():
    ref = wd.make_state("ref", 0, [fact(WATER, "inside", CUP,
                                        WorldPolarity.NEGATIVE)])
    fc, cmp = run_case(ref)
    assert cmp.overall == "mismatch"
    rep = wv.review_prediction(fc, cmp, ref, [wet_rule()])
    assert rep.overall == "confirmed_mismatch"
    assert rep.complete is False
    assert len(rep.candidates) == 1
    cand = rep.candidates[0]
    assert cand.status == "pending_review" and cand.simulated is True
    assert cand.classification == "confirmed_mismatch"
    assert cand.forecast_verdict == "entailed"
    assert cand.reference_verdict == "refuted"
    assert cand.forecast_proof_refs and cand.reference_proof_refs
    assert len(cand.hypotheses) == 4
    assert all(h.is_hypothesis is True for h in cand.hypotheses)
    kinds = {h.kind for h in cand.hypotheses}
    assert kinds == {"action_model_may_be_wrong",
                     "inference_rule_may_be_wrong_or_missing",
                     "unmodeled_world_change",
                     "reference_scope_or_provenance_uncertain"}
    assert "pour" in cand.action_id
    assert "wet" in cand.rule_ids


# ---- C. reference unknown ----
def test_review_reference_unknown():
    ref = wd.make_state("ref", 0, [])
    fc, cmp = run_case(ref)
    rep = wv.review_prediction(fc, cmp, ref, [wet_rule()])
    assert rep.items[0].classification == "reference_unknown"
    assert rep.candidates == ()
    assert rep.overall == "reference_unknown"


# ---- D. forecast inconclusive ----
def test_review_forecast_inconclusive():
    s0 = start_state()
    fc = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                            [pat(CUP, "on", TABLE)], [wet_rule()],
                            new_state_id="s1", event_id="e1")
    assert [r.verdict for r in fc.results] == ["unknown"]
    ref = wd.make_state("ref", 0, [fact(CUP, "on", TABLE,
                                        WorldPolarity.POSITIVE)])
    cmp = wp.compare_forecast(fc, ref, [wet_rule()])
    rep = wv.review_prediction(fc, cmp, ref, [wet_rule()])
    assert rep.items[0].classification == "forecast_inconclusive"
    assert rep.candidates == ()


# ---- E. conflict & limited ----
def test_review_conflict_and_limited():
    ref = wd.make_state("ref", 0, [
        fact(WATER, "inside", CUP, WorldPolarity.POSITIVE),
        fact(WATER, "inside", CUP, WorldPolarity.NEGATIVE)])
    fc, cmp = run_case(ref)
    rep = wv.review_prediction(fc, cmp, ref, [wet_rule()])
    assert rep.items[0].classification == "conflicted"
    assert rep.candidates == ()
    assert rep.overall == "conflicted"
    from cognitive_ai.models import WorldRuleView
    isa = WorldRuleView(
        rule_id="isa", name="i", status="active", source="seed",
        conditions=({"type": "is_a", "concept": "?x",
                     "ancestor": "container"},),
        conclusion={"type": "has_attr", "concept": "?x", "key": "holds",
                    "value": True},
        exceptions=())
    s0 = start_state()
    fc2 = wp.forecast_action(s0, pour_action(),
                             {"cup": CUP, "liquid": WATER},
                             queries(), [isa],
                             new_state_id="s2", event_id="e2")
    ref2 = wd.make_state("ref2", 0, [])
    cmp2 = wp.compare_forecast(fc2, ref2, [isa])
    rep2 = wv.review_prediction(fc2, cmp2, ref2, [isa])
    assert rep2.items[0].classification == "limited"
    assert rep2.candidates == ()


# ---- F. tampered proofs ----
def wet_case():
    s0 = start_state()
    q = pat(CUP, "wet", lit_const(True))
    fc = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                            [q], [wet_rule()],
                            new_state_id="s1", event_id="e1")
    ref = wd.make_state("ref", 0, [WorldStateFact(
        proposition=WorldProposition(
            subject=ref_term(CUP), predicate="wet",
            object=WorldTerm(literal=True, has_literal=True),
            polarity=WorldPolarity.NEGATIVE),
        origin="test")])
    cmp = wp.compare_forecast(fc, ref, [wet_rule()])
    assert cmp.overall == "mismatch"
    assert fc.results[0].positive_proofs[0].kind == "rule"
    return fc, cmp, ref


def tampered_review(mutate):
    import dataclasses
    fc, cmp, ref = wet_case()
    item = cmp.items[0]
    bad_proofs = tuple(mutate(p) for p in item.forecast_proofs)
    bad_item = dataclasses.replace(item, forecast_proofs=bad_proofs)
    bad_cmp = dataclasses.replace(cmp, items=(bad_item,))
    return wv.review_prediction(fc, bad_cmp, ref, [wet_rule()])


def test_tampered_proofs_rejected():
    import dataclasses

    def with_leaf(mutate_leaf):
        def go(p):
            leaf = p.premises[0]
            assert leaf.kind == "fact", leaf
            return dataclasses.replace(p, premises=(mutate_leaf(leaf),))
        return go

    assert tampered_review(
        lambda p: dataclasses.replace(p, rule_id="nope")
        ).items[0].classification == "invalid"
    assert tampered_review(with_leaf(
        lambda leaf: dataclasses.replace(leaf, fact_index=99))
        ).items[0].classification == "invalid"
    assert tampered_review(
        lambda p: dataclasses.replace(p, bindings=(("liquid", TABLE),))
        ).items[0].classification == "invalid"
    assert tampered_review(
        lambda p: dataclasses.replace(
            p, conclusion=WorldLiteralPattern(
                subject=var("liquid"), predicate="inside",
                object=var("cup"),
                polarity=WorldPolarity.NEGATIVE))
        ).items[0].classification == "invalid"
    assert tampered_review(
        lambda p: dataclasses.replace(p, depth=99)
        ).items[0].classification == "invalid"
    assert tampered_review(
        lambda p: dataclasses.replace(p, state_id="elsewhere")
        ).items[0].classification == "invalid"
    rep = tampered_review(lambda p: p)
    assert rep.items[0].classification == "confirmed_mismatch"


# ---- G. bad inputs ----
def test_bad_inputs_diagnosed():
    import dataclasses
    ref = wd.make_state("ref", 0, [fact(WATER, "inside", CUP,
                                        WorldPolarity.POSITIVE)])
    fc, cmp = run_case(ref)
    bad_ref = dataclasses.replace(cmp, reference_state_id="ghost")
    rep = wv.review_prediction(fc, bad_ref, ref, [wet_rule()])
    assert rep.overall == "invalid" and rep.candidates == ()
    rep2 = wv.review_prediction(fc, cmp, fc.predicted_state, [wet_rule()])
    assert rep2.overall == "invalid"
    bad_q = dataclasses.replace(
        cmp.items[0],
        query=pat(CUP, "on", TABLE))
    bad_cmp = dataclasses.replace(cmp, items=(bad_q,))
    rep3 = wv.review_prediction(fc, bad_cmp, ref, [wet_rule()])
    assert rep3.items[0].classification == "invalid"
    nop = dataclasses.replace(cmp.items[0], forecast_proofs=())
    rep4 = wv.review_prediction(
        fc, dataclasses.replace(cmp, items=(nop,)), ref, [wet_rule()])
    assert rep4.items[0].classification == "invalid"


# ---- H. immutability & side effects ----
def test_review_immutable_and_pure():
    ref = wd.make_state("ref", 0, [fact(WATER, "inside", CUP,
                                        WorldPolarity.NEGATIVE)])
    fc, cmp = run_case(ref)
    before = (fc, cmp, ref, fc.predicted_state)
    rep = wv.review_prediction(fc, cmp, ref, [wet_rule()])
    assert (fc, cmp, ref, fc.predicted_state) == before
    with pytest.raises(AttributeError):
        rep.items[0].candidate.hypotheses[0].refs = ()
    with pytest.raises(AttributeError):
        rep.items[0].query.predicate = "other"
    with pytest.raises(AttributeError):
        rep.candidates = ()
    with pytest.raises(AttributeError):
        rep.limitations += ("x",)


def test_review_has_no_io_paths():
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "src",
                            "cognitive_ai", "world_review.py"),
               encoding="utf-8").read()
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    for banned in ("sqlite3", "open(", "Memory", "Learner",
                   "SchemaEvolver", "socket", "random", "urllib", "torch",
                   "neural", "http"):
        assert banned not in src, banned


# ---- I. real verifier ----
def test_review_calls_real_verifier(monkeypatch):
    from cognitive_ai import world_reasoning as rsn
    calls = {"n": 0}
    real = rsn.verify_world_proof

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(rsn, "verify_world_proof", spy)
    ref = wd.make_state("ref", 0, [fact(WATER, "inside", CUP,
                                        WorldPolarity.NEGATIVE)])
    fc, cmp = run_case(ref)
    rep = wv.review_prediction(fc, cmp, ref, [wet_rule()])
    assert calls["n"] > 0, calls
    assert rep.overall == "confirmed_mismatch"
