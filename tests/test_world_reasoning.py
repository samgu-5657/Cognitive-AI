"""Phase M4-E state-scoped reasoning (pure memory, no DB/files)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import world_dynamics as wd
from cognitive_ai import world_reasoning as wr
from cognitive_ai.models import (WorldActionSchema, WorldEffect,
                                 WorldInferenceRule, WorldLiteralPattern,
                                 WorldObjectRef, WorldPatternTerm,
                                 WorldProofNode, WorldProposition,
                                 WorldReasoningResult, WorldRuleDiagnostic,
                                 WorldState, WorldStateFact, WorldTerm,
                                 WorldObjectKind, WorldPolarity)
from cognitive_ai.policy import MAX_CHAIN_DEPTH

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


def query(subj, pred, obj, polarity=WorldPolarity.POSITIVE):
    s = const(subj) if isinstance(subj, WorldObjectRef) else lit_const(subj)
    o = const(obj) if isinstance(obj, WorldObjectRef) else lit_const(obj)
    return WorldLiteralPattern(subject=s, predicate=pred, object=o,
                               polarity=polarity)


def rule(rid, premises, conclusion):
    return WorldInferenceRule(rule_id=rid, premises=tuple(premises),
                              conclusion=conclusion,
                              source_ref=f"test:{rid}", status="active")


def chain_rules():
    r1 = rule("r1", (WorldLiteralPattern(
        subject=var("x"), predicate="on", object=const(TABLE),
        polarity=WorldPolarity.POSITIVE),), WorldLiteralPattern(
        subject=var("x"), predicate="held", object=const(TABLE),
        polarity=WorldPolarity.POSITIVE))
    r2 = rule("r2", (WorldLiteralPattern(
        subject=var("x"), predicate="held", object=const(TABLE),
        polarity=WorldPolarity.POSITIVE),), WorldLiteralPattern(
        subject=var("x"), predicate="ready", object=lit_const(True),
        polarity=WorldPolarity.POSITIVE))
    r3 = rule("r3", (WorldLiteralPattern(
        subject=var("x"), predicate="ready", object=lit_const(True),
        polarity=WorldPolarity.POSITIVE),), WorldLiteralPattern(
        subject=var("x"), predicate="served", object=lit_const(True),
        polarity=WorldPolarity.POSITIVE))
    return r1, r2, r3


# ---- A. direct facts ----
def test_direct_four_cases():
    base = [fact(CUP, "on", TABLE, WorldPolarity.POSITIVE)]
    both = base + [fact(CUP, "on", TABLE, WorldPolarity.NEGATIVE)]
    neg = [fact(CUP, "on", TABLE, WorldPolarity.NEGATIVE)]
    q = query(CUP, "on", TABLE)
    assert wr.reason_about_state(
        wd.make_state("s", 0, base), q, []).verdict == "entailed"
    assert wr.reason_about_state(
        wd.make_state("s", 0, neg), q, []).verdict == "refuted"
    assert wr.reason_about_state(
        wd.make_state("s", 0, []), q, []).verdict == "unknown"
    assert wr.reason_about_state(
        wd.make_state("s", 0, both), q, []).verdict == "conflicted"


# ---- B. three-step chain ----
def test_three_step_chain():
    r1, r2, r3 = chain_rules()
    s0 = wd.make_state("s0", 0, [fact(CUP, "on", TABLE,
                                       WorldPolarity.POSITIVE)])
    res = wr.reason_about_state(
        s0, query(CUP, "served", True), [r1, r2, r3])
    assert res.verdict == "entailed", res
    assert res.rules_used == ("r1", "r2", "r3"), res
    assert res.max_derivation_depth == 3, res
    assert res.complete is True and res.uncompilable == ()
    assert len(res.positive_proofs) == 1
    top = res.positive_proofs[0]
    assert top.kind == "rule" and top.rule_id == "r3"
    assert top.depth == 3
    assert dict(top.bindings) == {"x": CUP}
    mid = top.premises[0]
    assert mid.rule_id == "r2" and mid.depth == 2
    leaf = mid.premises[0].premises[0]
    assert leaf.kind == "fact" and leaf.fact_index == 0
    assert leaf.proposition == s0.facts[0].proposition
    assert wr.verify_world_proof(
        s0, query(CUP, "served", True), top,
        [r1, r2, r3]) == {"result": "valid",
                          "detail": "rule r3, 1 premise(s) re-derived"}


def test_state_untouched_by_reasoning():
    r1, r2, r3 = chain_rules()
    facts = [fact(CUP, "on", TABLE, WorldPolarity.POSITIVE)]
    s0 = wd.make_state("s0", 0, facts)
    before = s0.facts
    wr.reason_about_state(s0, query(CUP, "served", True), [r1, r2, r3])
    assert s0.facts == before


# ---- C. missing premise ----
def test_missing_premise_no_forgery():
    r1, _r2, r3 = chain_rules()
    s0 = wd.make_state("s0", 0, [])
    res = wr.reason_about_state(s0, query(CUP, "served", True), [r1, r3])
    assert res.verdict == "unknown", res
    assert res.positive_proofs == () and res.negative_proofs == ()


# ---- D. conflict ----
def test_derived_conflict():
    r1, r2, r3 = chain_rules()
    deny = rule("deny", (WorldLiteralPattern(
        subject=var("x"), predicate="on", object=const(TABLE),
        polarity=WorldPolarity.POSITIVE),), WorldLiteralPattern(
        subject=var("x"), predicate="served", object=lit_const(True),
        polarity=WorldPolarity.NEGATIVE))
    s0 = wd.make_state("s0", 0, [fact(CUP, "on", TABLE,
                                       WorldPolarity.POSITIVE)])
    res = wr.reason_about_state(
        s0, query(CUP, "served", True), [r1, r2, r3, deny])
    assert res.verdict == "conflicted", res
    assert len(res.positive_proofs) == 1
    assert len(res.negative_proofs) == 1


# ---- E. bindings ----
def test_repeated_variable_consistency():
    r = rule("same", (WorldLiteralPattern(
        subject=var("x"), predicate="likes", object=var("x"),
        polarity=WorldPolarity.POSITIVE),), WorldLiteralPattern(
        subject=var("x"), predicate="selfish", object=lit_const(True),
        polarity=WorldPolarity.POSITIVE))
    s = wd.make_state("s", 0, [fact(CUP, "likes", TABLE,
                                    WorldPolarity.POSITIVE)])
    res = wr.reason_about_state(s, query(CUP, "selfish", True), [r])
    assert res.verdict == "unknown", res
    s2 = wd.make_state("s2", 0, [fact(CUP, "likes", CUP,
                                      WorldPolarity.POSITIVE)])
    res2 = wr.reason_about_state(s2, query(CUP, "selfish", True), [r])
    assert res2.verdict == "entailed", res2
    assert dict(res2.positive_proofs[0].bindings) == {"x": CUP}


def test_multi_param_and_literals():
    r = rule("mix", (WorldLiteralPattern(
        subject=var("a"), predicate="near", object=var("b"),
        polarity=WorldPolarity.POSITIVE),
        WorldLiteralPattern(subject=var("b"), predicate="hot",
                             object=lit_const(100),
                             polarity=WorldPolarity.POSITIVE),),
        WorldLiteralPattern(subject=var("a"), predicate="warned",
                            object=lit_const("烫"),
                            polarity=WorldPolarity.POSITIVE))
    s = wd.make_state("s", 0, [
        WorldStateFact(proposition=WorldProposition(
            subject=ref_term(CUP), predicate="near",
            object=ref_term(WATER), polarity=WorldPolarity.POSITIVE),
            origin="test"),
        WorldStateFact(proposition=WorldProposition(
            subject=ref_term(WATER), predicate="hot",
            object=WorldTerm(literal=100, has_literal=True),
            polarity=WorldPolarity.POSITIVE),
            origin="test")])
    q = WorldLiteralPattern(subject=const(CUP), predicate="warned",
                            object=lit_const("烫"),
                            polarity=WorldPolarity.POSITIVE)
    res = wr.reason_about_state(s, q, [r])
    assert res.verdict == "entailed", res
    assert dict(res.positive_proofs[0].bindings) == {"a": CUP,
                                                    "b": WATER}


def test_illegal_query_and_rules():
    s = wd.make_state("s", 0, [])
    bad_q = WorldLiteralPattern(subject=var("x"), predicate="on",
                                object=const(TABLE),
                                polarity=WorldPolarity.POSITIVE)
    assert wr.reason_about_state(s, bad_q, []).verdict == "invalid"
    with pytest.raises(ValueError):
        WorldInferenceRule(rule_id="", premises=(
            WorldLiteralPattern(subject=var("x"), predicate="on",
                                object=const(TABLE),
                                polarity=WorldPolarity.POSITIVE),),
            conclusion=WorldLiteralPattern(
                subject=var("x"), predicate="p",
                object=lit_const(1), polarity=WorldPolarity.POSITIVE))
    with pytest.raises(ValueError):
        WorldInferenceRule(
            rule_id="dangling", premises=(WorldLiteralPattern(
                subject=var("x"), predicate="on", object=const(TABLE),
                polarity=WorldPolarity.POSITIVE),),
            conclusion=WorldLiteralPattern(
                subject=var("ghost"), predicate="p",
                object=lit_const(1), polarity=WorldPolarity.POSITIVE))


# ---- F. cycles & depth ----
def test_self_loop_terminates():
    r = rule("loop", (WorldLiteralPattern(
        subject=var("x"), predicate="p", object=lit_const(1),
        polarity=WorldPolarity.POSITIVE),), WorldLiteralPattern(
        subject=var("x"), predicate="p", object=lit_const(1),
        polarity=WorldPolarity.POSITIVE))
    s = wd.make_state("s", 0, [WorldStateFact(
        proposition=WorldProposition(
            subject=ref_term(CUP), predicate="p",
            object=WorldTerm(literal=1, has_literal=True),
            polarity=WorldPolarity.POSITIVE),
        origin="test")])
    res = wr.reason_about_state(
        s, query(CUP, "p", 1), [r])
    assert res.verdict == "entailed", res


def test_mutual_rules_terminate():
    r_ab = rule("ab", (WorldLiteralPattern(
        subject=var("x"), predicate="pa", object=lit_const(1),
        polarity=WorldPolarity.POSITIVE),), WorldLiteralPattern(
        subject=var("x"), predicate="pb", object=lit_const(1),
        polarity=WorldPolarity.POSITIVE))
    r_ba = rule("ba", (WorldLiteralPattern(
        subject=var("x"), predicate="pb", object=lit_const(1),
        polarity=WorldPolarity.POSITIVE),), WorldLiteralPattern(
        subject=var("x"), predicate="pa", object=lit_const(1),
        polarity=WorldPolarity.POSITIVE))
    s = wd.make_state("s", 0, [WorldStateFact(
        proposition=WorldProposition(
            subject=ref_term(CUP), predicate="pa",
            object=WorldTerm(literal=1, has_literal=True),
            polarity=WorldPolarity.POSITIVE),
        origin="test")])
    res = wr.reason_about_state(s, query(CUP, "pb", 1), [r_ab, r_ba])
    assert res.verdict == "entailed", res


def test_depth_cap_reports_limited():
    r1, r2, r3 = chain_rules()
    s0 = wd.make_state("s0", 0, [fact(CUP, "on", TABLE,
                                       WorldPolarity.POSITIVE)])
    res = wr.reason_about_state(s0, query(CUP, "served", True),
                                [r1, r2, r3], max_depth=2)
    assert res.verdict == "limited", res
    assert res.complete is False
    assert res.positive_proofs == ()


# ---- G. proof tampering ----
def good_proof():
    r1, r2, r3 = chain_rules()
    s0 = wd.make_state("s0", 0, [fact(CUP, "on", TABLE,
                                       WorldPolarity.POSITIVE)])
    res = wr.reason_about_state(s0, query(CUP, "served", True),
                                [r1, r2, r3])
    assert res.verdict == "entailed"
    return s0, [r1, r2, r3], res.positive_proofs[0]


def test_proof_tampering_rejected():
    import dataclasses
    s0, rules, top = good_proof()
    q = query(CUP, "served", True)
    assert wr.verify_world_proof(s0, q, top, rules)["result"] == \
        "valid"
    bad_id = dataclasses.replace(top, rule_id="nope")
    assert wr.verify_world_proof(s0, q, bad_id, rules)["result"] == \
        "invalid"
    bad_bind = dataclasses.replace(top, bindings=(("x", TABLE),))
    assert wr.verify_world_proof(s0, q, bad_bind, rules)["result"] == \
        "invalid"
    leaf = top.premises[0].premises[0].premises[0]
    assert leaf.kind == "fact", leaf
    bad_leaf = dataclasses.replace(leaf, fact_index=7)
    bad_r1 = dataclasses.replace(top.premises[0].premises[0],
                                 premises=(bad_leaf,))
    bad_mid = dataclasses.replace(top.premises[0], premises=(bad_r1,))
    bad_tree = dataclasses.replace(top, premises=(bad_mid,))
    assert wr.verify_world_proof(s0, q, bad_tree, rules)["result"] == \
        "invalid"
    bad_drop = dataclasses.replace(top, premises=())
    assert wr.verify_world_proof(s0, q, bad_drop, rules)["result"] == \
        "invalid"
    bad_conc = dataclasses.replace(
        top, proposition=top.proposition,
        conclusion=WorldLiteralPattern(
            subject=const(CUP), predicate="served",
            object=lit_const(False), polarity=WorldPolarity.POSITIVE))
    assert wr.verify_world_proof(s0, q, bad_conc, rules)["result"] == \
        "invalid"
    bad_deep = dataclasses.replace(top, depth=99)
    assert wr.verify_world_proof(s0, q, bad_deep, rules,
                                 max_depth=4)["result"] == "invalid"


def test_proof_cycle_rejected():
    s0, rules, top = good_proof()
    q = query(CUP, "served", True)
    loop = dataclasses_replace_kind_rule(top)
    assert wr.verify_world_proof(s0, q, loop, rules)["result"] == \
        "invalid"


def dataclasses_replace_kind_rule(top):
    import dataclasses
    inner = dataclasses.replace(top.premises[0], premises=(top,))
    return dataclasses.replace(top, premises=(inner,))


# ---- H. adapter & completeness ----
def test_legacy_adapter_lossless_subset():
    from cognitive_ai.models import WorldRuleView
    good = WorldRuleView(
        rule_id="g", name="g", status="active", source="seed",
        conditions=({"type": "has_attr", "concept": "?x", "key": "holds",
                     "value": True},),
        conclusion={"type": "has_attr", "concept": "?x", "key": "stores",
                    "value": True},
        exceptions=())
    out = wr.compile_world_rules([good])
    assert out["complete"] is True and out["refused"] == ()
    assert len(out["rules"]) == 1
    assert out["rules"][0].conclusion.predicate == "stores"


def test_legacy_adapter_refusals():
    from cognitive_ai.models import WorldRuleView
    isa = WorldRuleView(
        rule_id="isa", name="i", status="active", source="seed",
        conditions=({"type": "is_a", "concept": "?x",
                     "ancestor": "container"},),
        conclusion={"type": "has_attr", "concept": "?x", "key": "holds",
                    "value": True},
        exceptions=())
    off = WorldRuleView(
        rule_id="off", name="o", status="disabled", source="seed",
        conditions=({"type": "has_attr", "concept": "?x", "key": "k",
                     "value": 1},),
        conclusion={"type": "has_attr", "concept": "?x", "key": "j",
                    "value": 1},
        exceptions=())
    exc = WorldRuleView(
        rule_id="exc", name="e", status="active", source="seed",
        conditions=({"type": "has_attr", "concept": "?x", "key": "k",
                     "value": 1},),
        conclusion={"type": "has_attr", "concept": "?x", "key": "j",
                    "value": 1},
        exceptions=({"type": "has_attr", "concept": "?x", "key": "bad",
                     "value": 1},))
    out = wr.compile_world_rules([isa, off, exc])
    assert out["complete"] is False
    by_id = {d.rule_id: d.reason for d in out["refused"]}
    assert set(by_id) == {"isa", "off", "exc"}, by_id
    assert all(by_id.values())


def test_uncompilable_active_forces_limited():
    from cognitive_ai.models import WorldRuleView
    isa = WorldRuleView(
        rule_id="isa", name="i", status="active", source="seed",
        conditions=({"type": "is_a", "concept": "?x",
                     "ancestor": "container"},),
        conclusion={"type": "has_attr", "concept": "?x", "key": "holds",
                    "value": True},
        exceptions=())
    s0 = wd.make_state("s0", 0, [fact(CUP, "on", TABLE,
                                       WorldPolarity.POSITIVE)])
    res = wr.reason_about_state(s0, query(CUP, "on", TABLE), [isa])
    assert res.verdict == "limited", res
    assert res.complete is False
    assert [d.rule_id for d in res.uncompilable] == ["isa"]
    assert len(res.positive_proofs) == 1


def test_inactive_rules_do_not_limit():
    from cognitive_ai.models import WorldRuleView
    off = WorldRuleView(
        rule_id="off", name="o", status="disabled", source="seed",
        conditions=({"type": "has_attr", "concept": "?x", "key": "k",
                     "value": 1},),
        conclusion={"type": "has_attr", "concept": "?x", "key": "j",
                    "value": 1},
        exceptions=())
    s0 = wd.make_state("s0", 0, [fact(CUP, "on", TABLE,
                                       WorldPolarity.POSITIVE)])
    res = wr.reason_about_state(s0, query(CUP, "on", TABLE), [off])
    assert res.verdict == "entailed", res
    assert res.complete is True


# ---- I. isolation ----
def test_reasoning_has_no_io():
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "src",
                            "cognitive_ai", "world_reasoning.py"),
               encoding="utf-8").read()
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    for banned in ("sqlite3", "open(", "Memory", "socket", "random",
                   "urllib", "torch", "semground", "learner", "Ontology",
                   "Reasoner"):
        assert banned not in src, banned


def test_result_types():
    r1, _r2, _r3 = chain_rules()
    s0 = wd.make_state("s0", 0, [fact(CUP, "on", TABLE,
                                       WorldPolarity.POSITIVE)])
    res = wr.reason_about_state(s0, query(CUP, "served", True), [r1])
    assert isinstance(res, WorldReasoningResult)
    assert res.state_id == "s0" and res.query.predicate == "served"
    assert isinstance(res.rules_used, tuple)
