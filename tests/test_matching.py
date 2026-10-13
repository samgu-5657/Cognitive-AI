"""Shared unification kernel + both callers (isolated DBs, no Temp/)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import matching
from cognitive_ai.memory import Memory
from cognitive_ai.models import (WorldObjectRef, WorldPatternTerm,
                                 WorldTerm, WorldObjectKind,
                                 WorldInferenceRule, WorldLiteralPattern,
                                 WorldPolarity)


def ref_term(ref):
    return WorldTerm(ref=ref)


CUP = WorldObjectRef("cupA", "杯子A", WorldObjectKind.INDIVIDUAL)
WATER = WorldObjectRef("waterA", "水A", WorldObjectKind.INDIVIDUAL)


def var(name):
    return ("var", name)


def test_first_binding():
    out = matching.unify(var("x"), ("lit", 1), {})
    assert out == {"x": ("lit", 1)}, out


def test_repeated_variable_consistency():
    kb = matching.unify(var("x"), ("lit", 1), {})
    assert matching.unify(var("x"), ("lit", 1), kb) == kb
    assert matching.unify(var("x"), ("lit", 2), kb) is None


def test_alias_and_self_alias():
    kb = matching.unify(var("x"), ("var", "y"), {})
    assert kb == {"x": ("var", "y")}, kb
    assert matching.unify(var("x"), ("var", "y"), kb) == kb
    assert matching.unify(var("x"), ("var", "z"), kb) is None
    assert matching.unify(var("x"), ("const", 1), kb) is None
    assert matching.unify(var("q"), ("var", "q"), {}) == {}


def test_stable_id_and_literal_compare():
    assert matching.unify(("ref", "a"), ("ref", "a"), {}) == {}
    assert matching.unify(("ref", "a"), ("ref", "b"), {}) is None
    assert matching.unify(("lit", 1), ("lit", 1), {}) == {}
    assert matching.unify(("lit", 1), ("lit", 2), {}) is None
    assert matching.unify(("ref", "a"), ("lit", "a"), {}) is None


def test_bad_terms_never_match():
    assert matching.unify(("bad", "x"), ("const", 1), {}) is None
    assert matching.unify(("const", 1), ("bad", "x"), {}) is None
    assert matching.compatible(("bad", "x"), ("var", "y")) is False
    assert matching.compatible(("var", "x"), ("const", 1)) is True
    assert matching.compatible(("const", 1), ("const", 1)) is True
    assert matching.compatible(("const", 1), ("const", 2)) is False


def test_legacy_adapters_roundtrip():
    assert matching.legacy_term("?x") == ("var", "x")
    assert matching.legacy_term("?") == ("lit", "?")
    assert matching.legacy_term(None) == ("lit", None)
    assert matching.legacy_term(True) == ("lit", True)
    assert matching.to_kernel_bindings({"?x": "?y"}) == \
        {"x": ("var", "y")}
    assert matching.from_kernel_value(("var", "y")) == "?y"
    assert matching.from_kernel_value(("lit", 3)) == 3
    assert matching.resolve_legacy({"?x": "?y", "?y": 1}, "?x") == 1
    assert matching.is_variable_name("?x") is True
    assert matching.is_variable_name("?") is False
    assert matching.is_variable_name(1) is False


def test_bindable_policy():
    kb = matching.unify(var("x"), ("lit", 1), {},
                        bindable=lambda f: f[0] == "ref")
    assert kb is None
    kb = matching.unify(var("x"), ("ref", "a"), {},
                        bindable=lambda f: f[0] == "ref")
    assert kb == {"x": ("ref", "a")}


def test_world_pattern_adapters():
    from cognitive_ai.models import WorldPatternTerm as PT
    assert matching.pattern_term(PT(variable="c")) == ("var", "c")
    assert matching.pattern_term(
        PT(ref=CUP)) == ("ref", "cupA")
    assert matching.pattern_term(
        PT(literal=2, has_literal=True)) == ("lit", 2)
    assert matching.pattern_term(PT())[0] == "bad"
    assert matching.fact_term(ref_term(CUP)) == ("ref", "cupA")
    assert matching.fact_term(WorldTerm(unresolved="z"))[0] == "bad"


def test_reasoner_calls_kernel(tmp_path, monkeypatch):
    from cognitive_ai.ontology import Ontology
    from cognitive_ai.reasoner import Reasoner
    m = Memory(str(tmp_path / "spya.db"))
    try:
        o = Ontology(m)
        o.add_concept("entity", "实体")
        o.add_concept("container", "容器", parents=["entity"])
        o.add_concept("cup", "杯子", parents=["container"])
        m.upsert_rule({
            "id": "r_spy", "conditions": [{"type": "is_a",
                                           "concept": "?x",
                                           "ancestor": "container"}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "holds", "value": True},
            "exceptions": []})
        m.upsert_rule({
            "id": "r_spy2", "conditions": [{"type": "has_attr",
                                            "concept": "?x",
                                            "key": "holds",
                                            "value": True}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "stores", "value": True},
            "exceptions": []})
        calls = {"unify": 0, "compatible": 0, "resolve": 0}
        real_unify, real_compat = matching.unify, matching.compatible
        real_resolve = matching.resolve_legacy

        def spy_unify(*a, **k):
            calls["unify"] += 1
            return real_unify(*a, **k)

        def spy_compat(*a):
            calls["compatible"] += 1
            return real_compat(*a)

        def spy_resolve(*a):
            calls["resolve"] += 1
            return real_resolve(*a)

        monkeypatch.setattr(matching, "unify", spy_unify)
        monkeypatch.setattr(matching, "compatible", spy_compat)
        monkeypatch.setattr(matching, "resolve_legacy", spy_resolve)
        fired = Reasoner(m).forward_chain()
        assert any(f.get("rule_id") == "r_spy2" for f in fired), fired
        assert calls["unify"] > 0, calls
        good = {"goal": {"type": "has_attr", "concept": "cup",
                         "key": "holds", "value": True},
                "rule_id": "r_spy",
                "bindings": {"?x": "cup"},
                "premises": [{"goal": {"type": "is_a",
                                       "concept": "cup",
                                       "ancestor": "container"},
                              "bindings": {}}]}
        assert Reasoner(m).verify_proof(good)["result"] == "verified"
        assert calls["compatible"] > 0 and calls["resolve"] > 0, calls
    finally:
        m.close()


def test_world_reasoning_calls_kernel(monkeypatch):
    from cognitive_ai import world_dynamics as wd
    from cognitive_ai import world_reasoning as wr
    from cognitive_ai.models import (WorldActionSchema, WorldEffect,
                                     WorldLiteralPattern, WorldStateFact,
                                     WorldProposition, WorldPolarity)
    s0 = wd.make_state("s0", 0, [WorldStateFact(
        proposition=WorldProposition(
            subject=ref_term(CUP), predicate="on",
            object=ref_term(WATER),
            polarity=WorldPolarity.POSITIVE),
        origin="test")])
    rule_q = WorldLiteralPattern(
        subject=WorldPatternTerm(variable="x"), predicate="on",
        object=WorldPatternTerm(variable="y"),
        polarity=WorldPolarity.POSITIVE)
    calls = {"n": 0}
    real = matching.unify

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(matching, "unify", spy)
    res = wr.reason_about_state(
        s0,
        WorldLiteralPattern(subject=WorldPatternTerm(ref=CUP),
                            predicate="on",
                            object=WorldPatternTerm(ref=WATER),
                            polarity=WorldPolarity.POSITIVE),
        [WorldInferenceRule(
            rule_id="r", premises=(rule_q,),
            conclusion=WorldLiteralPattern(
                subject=WorldPatternTerm(variable="x"),
                predicate="near",
                object=WorldPatternTerm(variable="y"),
                polarity=WorldPolarity.POSITIVE))])
    assert calls["n"] > 0, calls
    assert res.verdict in ("entailed", "unknown", "limited"), res
