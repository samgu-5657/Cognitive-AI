"""Phase M4-H review audit & learning gate (pure memory, no DB)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import world_dynamics as wd
from cognitive_ai import world_learning_gate as wg
from cognitive_ai import world_prediction as wp
from cognitive_ai import world_review as wv
from cognitive_ai.models import (WorldActionSchema, WorldEffect,
                                 WorldInferenceRule, WorldLiteralPattern,
                                 WorldObjectRef, WorldPatternTerm,
                                 WorldProposition, WorldStateFact,
                                 WorldTerm, WorldObjectKind, WorldPolarity,
                                 WorldReviewEvidence)

CUP = WorldObjectRef("cupA", "杯子A", WorldObjectKind.INDIVIDUAL)
WATER = WorldObjectRef("waterA", "水A", WorldObjectKind.INDIVIDUAL)


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
    return WorldInferenceRule(
        rule_id="wet", premises=(WorldLiteralPattern(
            subject=var("liquid"), predicate="inside",
            object=var("cup"), polarity=WorldPolarity.POSITIVE),),
        conclusion=WorldLiteralPattern(
            subject=var("cup"), predicate="wet",
            object=lit_const(True), polarity=WorldPolarity.POSITIVE),
        source_ref="test:wet", status="active")


def mismatch_candidate():
    s0 = wd.make_state("s0", 0, [fact(WATER, "inside", CUP,
                                       WorldPolarity.NEGATIVE)])
    fc = wp.forecast_action(s0, pour_action(), {"cup": CUP, "liquid": WATER},
                            [pat(WATER, "inside", CUP)], [wet_rule()],
                            new_state_id="s1", event_id="e1")
    ref = wd.make_state("ref", 0, [fact(WATER, "inside", CUP,
                                        WorldPolarity.NEGATIVE)])
    cmp = wp.compare_forecast(fc, ref, [wet_rule()])
    rep = wv.review_prediction(fc, cmp, ref, [wet_rule()])
    assert rep.overall == "confirmed_mismatch"
    assert len(rep.candidates) == 1
    return rep.candidates[0]


def ev(eid, group, kind="independent_observation",
       direction="supports"):
    return WorldReviewEvidence(
        evidence_id=eid, source_id=f"src-{group}",
        independence_group=group, kind=kind, direction=direction,
        proposition_ref="rule:wet conclusion holds",
        scope_note=f"lab run {group}")


def reg(pid, passed=True):
    return {"check_id": "rg-7", "scope": "tests/test_world_reasoning.py",
            "passed": passed, "source": "pytest 2026-10-13",
            "proposal_id": pid}


def full_ready():
    cand = mismatch_candidate()
    dec = wg.review_candidate(cand, "accepted_for_investigation",
                              "operator-li", "worth investigating")
    prop = wg.create_change_proposal(
        cand, dec, "inference_rule", "wet", "revise",
        expected_before=wet_rule(), proposed_after=wet_rule(),
        reason="recheck conclusion scope", regression_note="run M4-E suite",
        proposal_id="chg-wet-1")
    res = wg.evaluate_change_gate(
        dec, prop, [ev("e1", "g1"), ev("e2", "g2")],
        current_target=wet_rule(), regression=reg("chg-wet-1"))
    return cand, dec, prop, res


# ---- A. review states ----
def test_accepted_is_not_confirmation():
    cand = mismatch_candidate()
    dec = wg.review_candidate(cand, "accepted_for_investigation",
                              "op", "look closer")
    assert dec.decision == "accepted_for_investigation"
    assert dec.candidate_classification == "confirmed_mismatch"
    assert dec.candidate_ref == wg.candidate_ref(cand)
    for other in ("rejected", "deferred", "needs_more_evidence"):
        d = wg.review_candidate(cand, other, "op", "not now")
        assert d.decision == other
    with pytest.raises(ValueError):
        wg.review_candidate(cand, "accepted_for_investigation", "", "why")
    with pytest.raises(ValueError):
        wg.review_candidate(cand, "accepted_for_investigation", "op", "")
    with pytest.raises(ValueError):
        wg.review_candidate(cand, "approved", "op", "why")


def test_candidate_ref_stable():
    cand = mismatch_candidate()
    assert wg.candidate_ref(cand) == wg.candidate_ref(cand)
    assert wg.candidate_ref(cand).startswith("cand-")


# ---- B. concrete proposals ----
def test_proposal_shapes():
    cand = mismatch_candidate()
    dec = wg.review_candidate(cand, "accepted_for_investigation",
                              "op", "go")
    new_rule = wg.create_change_proposal(
        cand, dec, "inference_rule", "wet2", "add",
        proposed_after=wet_rule(), reason="add variant",
        regression_note="run suite", proposal_id="chg-add-1")
    assert new_rule.target_id == "wet2"
    dis = wg.create_change_proposal(
        cand, dec, "inference_rule", "wet", "disable",
        expected_before=wet_rule(), reason="retire",
        regression_note="run suite", proposal_id="chg-dis-1")
    assert dis.proposed_after is None
    with pytest.raises(ValueError):
        wg.create_change_proposal(
            cand, dec, "inference_rule", "wet", "revise",
            expected_before=wet_rule(), proposed_after=None,
            reason="half", regression_note="run", proposal_id="x")
    with pytest.raises(ValueError):
        wg.create_change_proposal(
            cand, dec, "spaceship", "wet", "revise",
            expected_before=wet_rule(), proposed_after=wet_rule(),
            reason="half", regression_note="run", proposal_id="x")
    other_dec = wg.review_candidate(cand, "rejected", "op", "no")
    with pytest.raises(ValueError):
        wg.create_change_proposal(
            cand, other_dec, "inference_rule", "wet", "revise",
            expected_before=wet_rule(), proposed_after=wet_rule(),
            reason="r", regression_note="n", proposal_id="y")
    with pytest.raises(ValueError):
        wg.create_change_proposal(
            cand, dec, "inference_rule", "wet", "revise")
    with pytest.raises(ValueError):
        from cognitive_ai.models import WorldChangeProposal
        WorldChangeProposal(
            proposal_id="p", candidate_ref="c", target_type="inference_rule",
            target_id="wet", operation="revise",
            expected_before=wet_rule(), proposed_after=None,
            reason="r", regression_note="n")


# ---- C. independent evidence ----
def test_two_groups_pass():
    _c, dec, prop, res = full_ready()
    assert res.status == "ready_for_application", res
    assert res.qualifying_groups == ("g1", "g2")
    assert res.regression_ok is True and res.blockers == ()


def test_evidence_gates():
    cand = mismatch_candidate()
    dec = wg.review_candidate(cand, "accepted_for_investigation",
                              "op", "go")
    prop = wg.create_change_proposal(
        cand, dec, "inference_rule", "wet", "revise",
        expected_before=wet_rule(), proposed_after=wet_rule(),
        reason="r", regression_note="n", proposal_id="chg-e-1")
    one = wg.evaluate_change_gate(
        dec, prop, [ev("e1", "g1")], current_target=wet_rule(),
        regression=reg("chg-e-1"))
    assert one.status == "blocked_insufficient_evidence", one
    dup = wg.evaluate_change_gate(
        dec, prop, [ev("e1", "g1"), ev("e2", "g1")],
        current_target=wet_rule(), regression=reg("chg-e-1"))
    assert dup.status == "blocked_insufficient_evidence", dup
    assert dup.qualifying_groups == ("g1",)
    sim = wg.evaluate_change_gate(
        dec, prop, [ev("e1", "g1"), ev("e2", "g2", kind="simulated")],
        current_target=wet_rule(), regression=reg("chg-e-1"))
    assert sim.status == "blocked_insufficient_evidence", sim
    unk = wg.evaluate_change_gate(
        dec, prop, [ev("e1", "g1"), ev("e2", "g2", kind="unknown")],
        current_target=wet_rule(), regression=reg("chg-e-1"))
    assert unk.status == "blocked_insufficient_evidence", unk
    nogroup = wg.evaluate_change_gate(
        dec, prop, [WorldReviewEvidence(
            evidence_id="e1", source_id="s1", independence_group="",
            kind="independent_observation", direction="supports",
            proposition_ref="rule:wet", scope_note="run")],
        current_target=wet_rule(), regression=reg("chg-e-1"))
    assert nogroup.status == "blocked_insufficient_evidence", nogroup
    opp = wg.evaluate_change_gate(
        dec, prop, [ev("e1", "g1"), ev("e2", "g2"),
                    ev("e3", "g3", direction="refutes")],
        current_target=wet_rule(), regression=reg("chg-e-1"))
    assert opp.status == "blocked_insufficient_evidence", opp
    assert opp.opposing == ("e3",), opp


# ---- D. regression gate ----
def test_regression_gate():
    _c, dec, prop, _r = full_ready()
    pid = prop.proposal_id
    assert wg.evaluate_change_gate(
        dec, prop, [ev("e1", "g1"), ev("e2", "g2")],
        current_target=wet_rule(), regression=None).status == \
        "blocked_regression"
    bad = dict(reg(pid))
    bad["passed"] = False
    assert wg.evaluate_change_gate(
        dec, prop, [ev("e1", "g1"), ev("e2", "g2")],
        current_target=wet_rule(), regression=bad).status == \
        "blocked_regression"
    unlink = dict(reg("other"))
    assert wg.evaluate_change_gate(
        dec, prop, [ev("e1", "g1"), ev("e2", "g2")],
        current_target=wet_rule(), regression=unlink).status == \
        "blocked_regression"
    noid = dict(reg(pid))
    del noid["check_id"]
    assert wg.evaluate_change_gate(
        dec, prop, [ev("e1", "g1"), ev("e2", "g2")],
        current_target=wet_rule(), regression=noid).status == \
        "blocked_regression"


# ---- E. remove each condition ----
def test_each_condition_required():
    cand = mismatch_candidate()
    accept = wg.review_candidate(cand, "accepted_for_investigation",
                                 "op", "go")
    rej = wg.review_candidate(cand, "rejected", "op", "no")
    prop = wg.create_change_proposal(
        cand, accept, "inference_rule", "wet", "revise",
        expected_before=wet_rule(), proposed_after=wet_rule(),
        reason="r", regression_note="n", proposal_id="chg-q-1")
    good_evs = [ev("e1", "g1"), ev("e2", "g2")]
    good_reg = reg("chg-q-1")
    assert wg.evaluate_change_gate(
        rej, prop, good_evs, current_target=wet_rule(),
        regression=good_reg).status == "blocked_review"
    stale = wg.create_change_proposal(
        cand, accept, "inference_rule", "wet", "revise",
        expected_before=wet_rule(), proposed_after=wet_rule(),
        reason="r", regression_note="n", proposal_id="chg-q-2")
    other_rule = wet_rule()
    import dataclasses
    drifted = dataclasses.replace(other_rule, source_ref="drifted")
    assert wg.evaluate_change_gate(
        accept, stale, good_evs, current_target=drifted,
        regression=dict(good_reg, proposal_id="chg-q-2")).status == \
        "blocked_invalid_proposal"


def test_unsupported_basis():
    import dataclasses
    cand = mismatch_candidate()
    dec = wg.review_candidate(cand, "accepted_for_investigation",
                              "op", "go")
    prop = wg.create_change_proposal(
        cand, dec, "inference_rule", "wet", "revise",
        expected_before=wet_rule(), proposed_after=wet_rule(),
        reason="r", regression_note="n", proposal_id="chg-u-1")
    vague = dataclasses.replace(dec, candidate_classification="limited")
    out = wg.evaluate_change_gate(
        vague, prop, [ev("e1", "g1"), ev("e2", "g2")],
        current_target=wet_rule(),
        regression=dict(reg("chg-u-1")))
    assert out.status == "blocked_unsupported_change", out


def test_add_needs_absence_proof():
    cand = mismatch_candidate()
    dec = wg.review_candidate(cand, "accepted_for_investigation",
                              "op", "go")
    prop = wg.create_change_proposal(
        cand, dec, "inference_rule", "wet2", "add",
        proposed_after=wet_rule(), reason="r", regression_note="n",
        proposal_id="chg-a-1")
    assert wg.evaluate_change_gate(
        dec, prop, [ev("e1", "g1"), ev("e2", "g2")],
        regression=reg("chg-a-1")).status == "blocked_invalid_proposal"
    assert wg.evaluate_change_gate(
        dec, prop, [ev("e1", "g1"), ev("e2", "g2")],
        absent_proof="registry lookup: no rule wet2",
        regression=reg("chg-a-1")).status == "ready_for_application"


# ---- F. deep immutability ----
def test_gate_objects_frozen():
    _c, dec, prop, res = full_ready()
    with pytest.raises(AttributeError):
        dec.reviewer = "mallory"
    with pytest.raises(AttributeError):
        prop.reason = "other"
    with pytest.raises(AttributeError):
        res.blockers = ()
    with pytest.raises(TypeError):
        res.regression_check["passed"] = False
    with pytest.raises(AttributeError):
        prop.proposed_after.rule_id = "hacked"


# ---- G. no side effects ----
def test_gate_module_has_no_io_paths():
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "src",
                            "cognitive_ai", "world_learning_gate.py"),
               encoding="utf-8").read()
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    for banned in ("sqlite3", "open(", "Memory", "Learner",
                   "SchemaEvolver", "pending_candidates", "socket",
                   "random", "urllib", "torch", "neural", "http"):
        assert banned not in src, banned


def test_flow_leaves_inputs_untouched():
    cand = mismatch_candidate()
    snapshot = (cand, cand.query, cand.hypotheses)
    dec = wg.review_candidate(cand, "accepted_for_investigation",
                              "op", "go")
    prop = wg.create_change_proposal(
        cand, dec, "inference_rule", "wet", "revise",
        expected_before=wet_rule(), proposed_after=wet_rule(),
        reason="r", regression_note="n", proposal_id="chg-s-1")
    evs = [ev("e1", "g1"), ev("e2", "g2")]
    wg.evaluate_change_gate(dec, prop, evs, current_target=wet_rule(),
                            regression=reg("chg-s-1"))
    assert (cand, cand.query, cand.hypotheses) == snapshot
    assert [e.evidence_id for e in evs] == ["e1", "e2"]
