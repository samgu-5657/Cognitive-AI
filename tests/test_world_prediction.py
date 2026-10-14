"""Phase M4-F forecast & compare (pure memory, no DB/files)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import world_dynamics as wd
from cognitive_ai import world_prediction as wp
from cognitive_ai import world_reasoning as wr
from cognitive_ai.models import (WorldActionSchema, WorldEffect,
                                 WorldForecast,
                                 WorldLiteralPattern, WorldObjectRef,
                                 WorldPatternTerm, WorldPredictionComparison,
                                 WorldProposition, WorldStateFact,
                                 WorldTerm, WorldObjectKind, WorldPolarity)

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
    return [pat(WATER, "inside", CUP),
            pat(CUP, "wet", lit_const(True))]


# ---- A. basic forecast ----
def test_basic_forecast():
    s0 = start_state()
    fc = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                            queries(), [wet_rule()],
                            new_state_id="s1", event_id="e1")
    assert isinstance(fc, WorldForecast)
    assert fc.transition.outcome == "applied", fc
    assert fc.predicted_state.state_id == "s1"
    assert fc.start_state_id == "s0" and fc.action_id == "pour"
    assert dict(fc.bindings) == {"cup": CUP, "liquid": WATER}
    assert [r.verdict for r in fc.results] == ["entailed", "entailed"]
    assert fc.complete is True
    assert len(fc.results[1].positive_proofs) == 1
    assert fc.results[1].positive_proofs[0].rule_id == "wet"


def test_forecast_calls_real_chain(monkeypatch):
    from cognitive_ai import world_dynamics as dyn
    from cognitive_ai import world_reasoning as rsn
    calls = {"sim": 0, "reason": 0}
    real_sim, real_reason = dyn.simulate_action, rsn.reason_about_state

    def spy_sim(*a, **k):
        calls["sim"] += 1
        return real_sim(*a, **k)

    def spy_reason(*a, **k):
        calls["reason"] += 1
        return real_reason(*a, **k)

    monkeypatch.setattr(dyn, "simulate_action", spy_sim)
    monkeypatch.setattr(rsn, "reason_about_state", spy_reason)
    fc = wp.forecast_action(start_state(), pour_action(),
                            {"cup": CUP, "liquid": WATER},
                            queries(), [wet_rule()])
    assert calls["sim"] == 1 and calls["reason"] == 2, calls
    assert fc.transition.outcome == "applied"


# ---- B. consistent ----
def test_compare_consistent():
    s0 = start_state()
    fc = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                            queries(), [wet_rule()],
                            new_state_id="s1", event_id="e1")
    ref = wd.make_state("ref", 0, [
        fact(WATER, "inside", CUP, WorldPolarity.POSITIVE),
        WorldStateFact(proposition=WorldProposition(
            subject=ref_term(CUP), predicate="wet",
            object=WorldTerm(literal=True, has_literal=True),
            polarity=WorldPolarity.POSITIVE),
            origin="test")])
    cmp = wp.compare_forecast(fc, ref, [wet_rule()])
    assert isinstance(cmp, WorldPredictionComparison)
    assert [it.category for it in cmp.items] == ["consistent", "consistent"]
    assert cmp.overall == "consistent" and cmp.complete is True
    assert cmp.forecast_state_id == "s1"
    assert cmp.reference_state_id == "ref"


# ---- C. mismatch ----
def test_compare_mismatch_both_directions():
    s0 = start_state()
    fc = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                            [pat(WATER, "inside", CUP)], [wet_rule()],
                            new_state_id="s1", event_id="e1")
    ref_deny = wd.make_state("ref", 0, [fact(WATER, "inside", CUP,
                                             WorldPolarity.NEGATIVE)])
    c1 = wp.compare_forecast(fc, ref_deny, [wet_rule()])
    assert [it.category for it in c1.items] == ["mismatch"]
    assert c1.overall == "mismatch" and c1.complete is False
    assert "entailed" in c1.items[0].detail
    assert c1.items[0].forecast_proofs and c1.items[0].reference_proofs
    fc3 = wp.forecast_action(start_state(), pour_action(),
                             {"cup": CUP, "liquid": WATER},
                             [pat(CUP, "wet", lit_const(True))],
                             [wet_rule()],
                             new_state_id="s3", event_id="e3")
    c3 = wp.compare_forecast(
        fc3,
        wd.make_state("ref3", 0, [WorldStateFact(
            proposition=WorldProposition(
                subject=ref_term(CUP), predicate="wet",
                object=WorldTerm(literal=True, has_literal=True),
                polarity=WorldPolarity.NEGATIVE),
            origin="test")]), [wet_rule()])
    assert [it.category for it in c3.items] == ["mismatch"]
    assert c3.overall == "mismatch"


# ---- D. unknown & limited ----
def test_reference_unknown():
    s0 = start_state()
    fc = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                            [pat(WATER, "inside", CUP)], [wet_rule()],
                            new_state_id="s1", event_id="e1")
    ref = wd.make_state("ref", 0, [])
    cmp = wp.compare_forecast(fc, ref, [wet_rule()])
    assert [it.category for it in cmp.items] == ["reference_unknown"]
    assert cmp.overall == "reference_unknown"
    assert cmp.complete is False


def test_forecast_unknown_inconclusive():
    s0 = start_state()
    fc = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                            [pat(CUP, "on", TABLE)], [wet_rule()],
                            new_state_id="s1", event_id="e1")
    assert fc.transition.outcome == "applied"
    assert [r.verdict for r in fc.results] == ["unknown"]
    ref = wd.make_state("ref", 0, [fact(CUP, "on", TABLE,
                                        WorldPolarity.POSITIVE)])
    cmp = wp.compare_forecast(fc, ref, [wet_rule()])
    assert [it.category for it in cmp.items] == ["inconclusive"]
    assert cmp.overall == "inconclusive"


def test_limited_propagates():
    from cognitive_ai.models import WorldRuleView
    r1 = wr_chain_rule("r1", "held", "on", const(TABLE))
    r2 = wr_chain_rule("r2", "ready", "held", lit_const(True))
    r3 = wr_chain_rule("r3", "served", "ready", lit_const(True))
    s0 = wd.make_state("s0", 0, [fact(CUP, "on", TABLE,
                                       WorldPolarity.POSITIVE),
                                 fact(WATER, "inside", CUP,
                                      WorldPolarity.NEGATIVE)])
    fc = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                            [pat(CUP, "served", lit_const(True))],
                            [r1, r2, r3], max_depth=1,
                            new_state_id="s1", event_id="e1")
    assert fc.transition.outcome == "applied"
    assert [r.verdict for r in fc.results] == ["limited"]
    ref = wd.make_state("ref", 0, [fact(CUP, "on", TABLE,
                                        WorldPolarity.POSITIVE)])
    cmp = wp.compare_forecast(fc, ref, [r1, r2, r3], max_depth=1)
    assert [it.category for it in cmp.items] == ["limited"]
    assert cmp.overall == "limited" and cmp.complete is False
    isa = WorldRuleView(
        rule_id="isa", name="i", status="active", source="seed",
        conditions=({"type": "is_a", "concept": "?x",
                     "ancestor": "container"},),
        conclusion={"type": "has_attr", "concept": "?x", "key": "holds",
                    "value": True},
        exceptions=())
    fc2 = wp.forecast_action(s0, pour_action(),
                             {"cup": CUP, "liquid": WATER},
                             [pat(WATER, "inside", CUP)], [isa],
                             new_state_id="s2", event_id="e2")
    assert [r.verdict for r in fc2.results] == ["limited"]
    assert fc2.complete is False


def wr_chain_rule(rid, pred, over_pred, over_obj):
    from cognitive_ai.models import WorldInferenceRule
    return WorldInferenceRule(
        rule_id=rid,
        premises=(WorldLiteralPattern(
            subject=var("x"), predicate=over_pred, object=over_obj,
            polarity=WorldPolarity.POSITIVE),),
        conclusion=WorldLiteralPattern(
            subject=var("x"), predicate=pred, object=lit_const(True),
            polarity=WorldPolarity.POSITIVE))


# ---- E. conflict ----
def test_reference_conflict():
    s0 = start_state()
    fc = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                            [pat(WATER, "inside", CUP)], [wet_rule()],
                            new_state_id="s1", event_id="e1")
    ref = wd.make_state("ref", 0, [
        fact(WATER, "inside", CUP, WorldPolarity.POSITIVE),
        fact(WATER, "inside", CUP, WorldPolarity.NEGATIVE)])
    cmp = wp.compare_forecast(fc, ref, [wet_rule()])
    assert [it.category for it in cmp.items] == ["conflicted"]
    assert cmp.overall == "conflicted"


# ---- F. input failures ----
def test_failed_forecasts_and_bad_compare():
    s_blocked = wd.make_state("sb", 0, [fact(WATER, "inside", CUP,
                                              WorldPolarity.POSITIVE)])
    assert wp.forecast_action(
        s_blocked, pour_action(), {"cup": CUP, "liquid": WATER},
        [pat(WATER, "inside", CUP)], []).transition.outcome == "blocked"
    assert wp.forecast_action(
        wd.make_state("su", 0, []), pour_action(),
        {"cup": CUP, "liquid": WATER},
        [pat(WATER, "inside", CUP)], []).transition.outcome == "unknown"
    s_conf = wd.make_state("sc", 0, [
        fact(WATER, "inside", CUP, WorldPolarity.POSITIVE),
        fact(WATER, "inside", CUP, WorldPolarity.NEGATIVE)])
    assert wp.forecast_action(
        s_conf, pour_action(), {"cup": CUP, "liquid": WATER},
        [pat(WATER, "inside", CUP)], []).transition.outcome == "conflicted"
    bad_bind = wp.forecast_action(
        start_state(), pour_action(), {"cup": CUP},
        [pat(WATER, "inside", CUP)], [])
    assert bad_bind.transition.outcome == "invalid"
    assert bad_bind.predicted_state is None
    bad_q = wp.forecast_action(start_state(), pour_action(),
                               {"cup": CUP, "liquid": WATER}, ["nope"], [])
    assert bad_q.transition.outcome == "invalid"
    assert bad_q.predicted_state is None
    for bad in (bad_bind, bad_q):
        assert bad.complete is False
    s0 = start_state()
    fc = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                            [pat(WATER, "inside", CUP)], [],
                            new_state_id="s1", event_id="e1")
    assert wp.compare_forecast(
        fc, fc.predicted_state, []).overall == "invalid"
    same_id = wd.make_state("s1", 5, [])
    assert wp.compare_forecast(fc, same_id, []).overall == "invalid"
    empty_q = wp.forecast_action(s0, pour_action(),
                                 {"cup": CUP, "liquid": WATER}, [], [])
    assert wp.compare_forecast(
        empty_q, wd.make_state("r", 0, []), []).overall == "invalid"


# ---- G. deep immutability ----
def test_forecast_results_frozen():
    s0 = start_state()
    fc = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                            queries(), [wet_rule()],
                            new_state_id="s1", event_id="e1")
    with pytest.raises(TypeError):
        fc.predicted_state.facts[0].proposition.raw["k"] = 1
    with pytest.raises(TypeError):
        fc.bindings[0] = ("cup", CUP)
    with pytest.raises(AttributeError):
        fc.results = ()
    proof = fc.results[1].positive_proofs[0]
    with pytest.raises(TypeError):
        proof.premises[0].proposition.raw["k"] = 1
    ref = wd.make_state("ref", 0, [fact(WATER, "inside", CUP,
                                        WorldPolarity.POSITIVE)])
    cmp = wp.compare_forecast(
        fc,
        wd.make_state("ref2", 0, [
            fact(WATER, "inside", CUP, WorldPolarity.POSITIVE),
            WorldStateFact(proposition=WorldProposition(
                subject=ref_term(CUP), predicate="wet",
                object=WorldTerm(literal=True, has_literal=True),
                polarity=WorldPolarity.POSITIVE),
                origin="test")]),
        [wet_rule()])
    assert cmp.overall == "consistent"
    with pytest.raises(AttributeError):
        cmp.items[0].detail = "x"
    with pytest.raises(AttributeError):
        cmp.items += ()


# ---- H. no side effects ----
def test_no_side_effects_and_determinism():
    s0 = start_state()
    ref = wd.make_state("ref", 0, [fact(WATER, "inside", CUP,
                                        WorldPolarity.POSITIVE)])
    kw = {"new_state_id": "s1", "event_id": "e1"}
    fc1 = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                             queries(), [wet_rule()], **kw)
    c1 = wp.compare_forecast(fc1, ref, [wet_rule()])
    assert s0 == start_state() and ref == wd.make_state(
        "ref", 0, [fact(WATER, "inside", CUP, WorldPolarity.POSITIVE)])
    fc2 = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                             queries(), [wet_rule()], **kw)
    assert fc1 == fc2
    assert wp.compare_forecast(fc2, ref, [wet_rule()]) == c1


def test_no_io_or_random_paths():
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "src",
                            "cognitive_ai", "world_prediction.py"),
               encoding="utf-8").read()
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    for banned in ("sqlite3", "open(", "Memory", "socket", "random",
                   "urllib", "torch", "semground", "learner", "Ontology",
                   "Reasoner", "neural", "http"):
        assert banned not in src, banned
