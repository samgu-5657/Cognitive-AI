"""Phase M4-C deterministic world dynamics (pure memory, no DB/files).

Objects, states and actions are built explicitly in each test. No
database is created, Temp/ is untouched, and no production QA/learn
path is modified.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import world_dynamics as wd
from cognitive_ai.models import (WorldActionSchema, WorldEffect,
                                 WorldEvent, WorldLiteralPattern,
                                 WorldObjectRef, WorldPatternTerm,
                                 WorldPreconditionCheck, WorldProposition,
                                 WorldState, WorldStateFact, WorldTerm,
                                 WorldTransitionResult,
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


def fact(subj, pred, obj, polarity):
    return WorldStateFact(
        proposition=WorldProposition(
            subject=ref_term(subj), predicate=pred,
            object=ref_term(obj), polarity=polarity),
        origin="test")


def lit(subj, pred, obj, polarity):
    return WorldLiteralPattern(subject=subj, predicate=pred, object=obj,
                               polarity=polarity)


def pour_action():
    return WorldActionSchema(
        action_id="pour", name="倒入",
        params=("cup", "liquid"),
        premises=(lit(var("liquid"), "inside", var("cup"),
                      WorldPolarity.NEGATIVE),),
        effects=(WorldEffect(lit(var("liquid"), "inside", var("cup"),
                                  WorldPolarity.POSITIVE)),))


def test_four_valued_lookup():
    pos = wd.make_state("s-pos", 0, [fact(WATER, "inside", CUP,
                                          WorldPolarity.POSITIVE)])
    neg = wd.make_state("s-neg", 0, [fact(WATER, "inside", CUP,
                                          WorldPolarity.NEGATIVE)])
    empty = wd.make_state("s-empty", 0, [])
    both = wd.make_state("s-both", 0,
                         [fact(WATER, "inside", CUP,
                               WorldPolarity.POSITIVE),
                          fact(WATER, "inside", CUP,
                               WorldPolarity.NEGATIVE)])
    probe = lit(const(WATER), "inside", const(CUP),
                WorldPolarity.POSITIVE)
    assert wd.evaluate_state_literal(pos, probe) == "true"
    assert wd.evaluate_state_literal(neg, probe) == "false"
    assert wd.evaluate_state_literal(empty, probe) == "unknown"
    assert wd.evaluate_state_literal(both, probe) == "conflicted"


def test_rejects_unknown_polarity_facts():
    with pytest.raises(ValueError):
        WorldStateFact(proposition=WorldProposition(
            subject=ref_term(CUP), predicate="inside",
            object=ref_term(WATER), polarity=WorldPolarity.UNKNOWN))
    with pytest.raises(ValueError):
        lit(const(WATER), "inside", const(CUP), WorldPolarity.UNKNOWN)


def test_normal_transition():
    s0 = wd.make_state("s0", 0, [fact(WATER, "inside", CUP,
                                      WorldPolarity.NEGATIVE)])
    res = wd.simulate_action(s0, pour_action(),
                             {"cup": CUP, "liquid": WATER},
                             new_state_id="s1", event_id="e1")
    assert res.outcome == "applied", res
    assert isinstance(res, WorldTransitionResult)
    assert res.new_state.state_id == "s1"
    assert res.new_state.parent_state_id == "s0"
    assert res.new_state.step_index == 1
    assert res.event.event_id == "e1"
    assert res.event.action_id == "pour"
    assert res.event.from_state == "s0"
    assert res.event.to_state == "s1"
    assert res.event.step_index == 1
    assert res.event.simulated is True
    assert dict(res.event.bindings) == {"cup": CUP, "liquid": WATER}
    assert wd.evaluate_state_literal(
        res.new_state,
        lit(const(WATER), "inside", const(CUP),
            WorldPolarity.POSITIVE)) == "true"
    assert len(res.checks) == 1 and res.checks[0].outcome == "true"


def test_original_state_immutable():
    raw = [{"subj": CUP, "pred": "on", "obj": TABLE, "pol": True}]
    facts = [fact(WATER, "inside", CUP, WorldPolarity.NEGATIVE)]
    s0 = wd.make_state("s0", 0, facts)
    before = s0.facts
    facts.append(fact(TABLE, "on", CUP, WorldPolarity.POSITIVE))
    raw[0]["pol"] = False
    res = wd.simulate_action(s0, pour_action(),
                             {"cup": CUP, "liquid": WATER},
                             new_state_id="s1", event_id="e1")
    assert res.outcome == "applied"
    assert s0.facts == before and len(s0.facts) == 1
    assert s0.state_id == "s0" and s0.step_index == 0
    assert res.new_state is not s0


def test_state_dedupes_same_polarity():
    s = WorldState(state_id="s", step_index=0, facts=(
        fact(WATER, "inside", CUP, WorldPolarity.POSITIVE),
        fact(WATER, "inside", CUP, WorldPolarity.POSITIVE)))
    assert len(s.facts) == 1


def test_unknown_blocked_conflicted():
    action = pour_action()
    good = {"cup": CUP, "liquid": WATER}
    s_unknown = wd.make_state("su", 0, [])
    assert wd.simulate_action(
        s_unknown, action, good,
        new_state_id="x", event_id="y").outcome == "unknown"
    s_blocked = wd.make_state("sb", 0, [fact(WATER, "inside", CUP,
                                              WorldPolarity.POSITIVE)])
    assert wd.simulate_action(
        s_blocked, action, good,
        new_state_id="x", event_id="y").outcome == "blocked"
    s_conf = wd.make_state("sc", 0,
                           [fact(WATER, "inside", CUP,
                                 WorldPolarity.POSITIVE),
                            fact(WATER, "inside", CUP,
                                 WorldPolarity.NEGATIVE)])
    assert wd.simulate_action(
        s_conf, action, good,
        new_state_id="x", event_id="y").outcome == "conflicted"
    for outcome, res in (
            ("unknown", wd.simulate_action(s_unknown, action, good)),
            ("blocked", wd.simulate_action(s_blocked, action, good)),
            ("conflicted", wd.simulate_action(s_conf, action, good))):
        assert res.outcome == outcome
        assert res.new_state is None and res.event is None
        assert len(res.checks) == 1


def test_unknown_among_satisfied():
    action = WorldActionSchema(
        action_id="move", name="移动", params=("cup", "dst"),
        premises=(
            lit(var("cup"), "on", const(TABLE), WorldPolarity.POSITIVE),
            lit(var("cup"), "held", const(TABLE), WorldPolarity.POSITIVE),
        ),
        effects=(WorldEffect(lit(var("cup"), "on", var("dst"),
                                  WorldPolarity.POSITIVE)),))
    s = wd.make_state("s", 0, [fact(CUP, "on", TABLE,
                                    WorldPolarity.POSITIVE)])
    res = wd.simulate_action(s, action, {"cup": CUP, "dst": WATER})
    assert res.outcome == "unknown", res
    assert [c.outcome for c in res.checks] == ["true", "unknown"]
    assert res.new_state is None and res.event is None


def test_binding_problems_invalid():
    action = pour_action()
    s0 = wd.make_state("s0", 0, [fact(WATER, "inside", CUP,
                                       WorldPolarity.NEGATIVE)])
    assert wd.simulate_action(
        s0, action, {"cup": CUP}).outcome == "invalid"
    assert wd.simulate_action(
        s0, action, {"cup": CUP, "liquid": WATER,
                     "extra": TABLE}).outcome == "invalid"
    assert wd.simulate_action(
        s0, action, {"cup": CUP,
                     "liquid": "waterA"}).outcome == "invalid"
    bad = WorldActionSchema(
        action_id="bad", name="坏", params=("cup",),
        premises=(lit(var("ghost"), "inside", var("cup"),
                      WorldPolarity.POSITIVE),),
        effects=(WorldEffect(lit(var("ghost"), "inside", var("cup"),
                                  WorldPolarity.POSITIVE)),))
    assert wd.validate_action_schema(bad) != ()
    assert wd.simulate_action(
        s0, bad, {"cup": CUP}).outcome == "invalid"
    dup = WorldActionSchema(
        action_id="dup", name="重", params=("a", "a"), premises=(),
        effects=(WorldEffect(lit(var("a"), "on", const(TABLE),
                                  WorldPolarity.POSITIVE)),))
    assert wd.validate_action_schema(dup) != ()
    assert wd.validate_action_schema(pour_action()) == ()


def test_effect_consistency():
    s0 = wd.make_state("s0", 0,
                       [fact(WATER, "inside", CUP,
                             WorldPolarity.NEGATIVE),
                        fact(CUP, "on", TABLE, WorldPolarity.POSITIVE)])
    res = wd.simulate_action(s0, pour_action(),
                             {"cup": CUP, "liquid": WATER},
                             new_state_id="s1", event_id="e1")
    assert res.outcome == "applied"
    s1 = res.new_state
    probe = lit(const(WATER), "inside", const(CUP),
                WorldPolarity.POSITIVE)
    assert wd.evaluate_state_literal(s1, probe) == "true"
    assert fact(CUP, "on", TABLE, WorldPolarity.POSITIVE) in s1.facts
    assert len(s1.facts) == 2
    take = WorldActionSchema(
        action_id="take", name="取出", params=("cup", "liquid"),
        premises=(lit(var("liquid"), "inside", var("cup"),
                      WorldPolarity.POSITIVE),),
        effects=(WorldEffect(lit(var("liquid"), "inside", var("cup"),
                                  WorldPolarity.NEGATIVE)),))
    res2 = wd.simulate_action(s1, take, {"cup": CUP, "liquid": WATER},
                              new_state_id="s2", event_id="e2")
    assert res2.outcome == "applied"
    assert wd.evaluate_state_literal(res2.new_state, probe) == "false"


def test_unrelated_conflict_survives():
    s0 = wd.make_state("s0", 0,
                       [fact(WATER, "inside", CUP,
                             WorldPolarity.NEGATIVE),
                        fact(CUP, "on", TABLE, WorldPolarity.POSITIVE),
                        fact(CUP, "on", TABLE, WorldPolarity.NEGATIVE)])
    res = wd.simulate_action(s0, pour_action(),
                             {"cup": CUP, "liquid": WATER},
                             new_state_id="s1", event_id="e1")
    assert res.outcome == "applied"
    assert wd.evaluate_state_literal(
        res.new_state,
        lit(const(CUP), "on", const(TABLE),
            WorldPolarity.POSITIVE)) == "conflicted"


def test_traceability_and_determinism():
    s0 = wd.make_state("s0", 0, [fact(WATER, "inside", CUP,
                                       WorldPolarity.NEGATIVE)])
    kw = {"bindings": {"cup": CUP, "liquid": WATER},
          "new_state_id": "s1", "event_id": "e1"}
    r1 = wd.simulate_action(s0, pour_action(), **kw)
    r2 = wd.simulate_action(s0, pour_action(), **kw)
    assert r1 == r2
    assert r1.new_state.parent_state_id == "s0"
    assert r1.event.from_state == "s0"
    assert r1.event.to_state == r1.new_state.state_id == "s1"
    assert isinstance(r1.event, WorldEvent)
    assert all(isinstance(c, WorldPreconditionCheck) for c in r1.checks)


def _code_text(rel):
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "src",
                            "cognitive_ai", rel), encoding="utf-8").read()
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    return src


def test_no_io_paths():
    for rel in ("world_dynamics.py", "models.py"):
        src = _code_text(rel)
        for banned in ("sqlite3", "open(", "socket", "torch", "urllib",
                       "http", "llm", "random"):
            assert banned not in src, (rel, banned)


def test_isolated_from_legacy_paths():
    src = _code_text("world_dynamics.py")
    for banned in ("semground", "semrealize", "learner", "Memory",
                   "Reasoner", "Ontology", "forward_chain"):
        assert banned not in src, banned


def _nested_fact():
    return WorldStateFact(
        proposition=WorldProposition(
            subject=ref_term(CUP), predicate="inside",
            object=ref_term(WATER), polarity=WorldPolarity.POSITIVE,
            raw={"tag": {"deep": [1, 2]}, "note": "n"}),
        origin="test")


def test_input_containers_do_not_leak_into_snapshot():
    facts = [_nested_fact()]
    s = wd.make_state("s", 0, facts)
    facts.append(_nested_fact())
    assert len(s.facts) == 1
    assert len(wd.make_state("s2", 0, []).facts) == 0


def test_public_nested_fields_deep_frozen():
    s = wd.make_state("s", 0, [_nested_fact()])
    raw = s.facts[0].proposition.raw
    with pytest.raises(TypeError):
        raw["note"] = "changed"
    with pytest.raises(TypeError):
        raw["tag"]["deep"] = []
    with pytest.raises(AttributeError):
        raw["tag"]["deep"].append(3)
    with pytest.raises(TypeError):
        raw["new"] = 1


def test_event_and_result_nested_frozen():
    s0 = wd.make_state("s0", 0, [fact(WATER, "inside", CUP,
                                       WorldPolarity.NEGATIVE)])
    res = wd.simulate_action(s0, pour_action(),
                             {"cup": CUP, "liquid": WATER},
                             new_state_id="s1", event_id="e1")
    assert res.outcome == "applied"
    with pytest.raises(TypeError):
        res.event.bindings[0] = ("cup", CUP)
    with pytest.raises(AttributeError):
        res.event.bindings += ((("x", CUP)),)
    with pytest.raises(TypeError):
        res.new_state.facts[0].proposition.raw["k"] = 1
    assert isinstance(res.checks, tuple)
    assert isinstance(res.new_state.facts, tuple)


def test_snapshots_share_no_mutable_state():
    src = [_nested_fact()]
    s1 = wd.make_state("s1", 0, src)
    s2 = wd.make_state("s2", 0, src)
    assert s1.facts is not s2.facts
    assert s1.facts == s2.facts
    for s in (s1, s2):
        with pytest.raises(TypeError):
            s.facts[0].proposition.raw["note"] = "x"


def test_rule_view_nested_frozen():
    from cognitive_ai.models import WorldRuleView
    rule = WorldRuleView(
        rule_id="r", name="r", status="active", source="seed",
        conditions=({"type": "is_a"},),
        conclusion={"type": "has_attr", "key": "k"},
        exceptions=[])
    assert isinstance(rule.conditions, tuple)
    assert isinstance(rule.exceptions, tuple)
    with pytest.raises(TypeError):
        rule.conclusion["key"] = "other"
    with pytest.raises(TypeError):
        rule.conditions[0]["type"] = "other"
