"""Phase M4-I controlled change application (isolated tmp DBs)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import world_dynamics as wd
from cognitive_ai import world_learning_gate as wg
from cognitive_ai import world_prediction as wp
from cognitive_ai import world_review as wv
from cognitive_ai.memory import Memory
from cognitive_ai.models import (WorldActionSchema, WorldEffect,
                                 WorldInferenceRule, WorldLiteralPattern,
                                 WorldObjectRef, WorldPatternTerm,
                                 WorldProposition, WorldStateFact,
                                 WorldTerm, WorldObjectKind, WorldPolarity)
from cognitive_ai.world_change_apply import (
    apply_world_change, authorize_change_application,
    record_post_apply_validation, world_rule_to_legacy)

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


def holds_rule_content(key="stores"):
    return WorldInferenceRule(
        rule_id="r_hold", premises=(WorldLiteralPattern(
            subject=var("x"), predicate="holds",
            object=lit_const(True),
            polarity=WorldPolarity.POSITIVE),),
        conclusion=WorldLiteralPattern(
            subject=var("x"), predicate=key, object=lit_const(True),
            polarity=WorldPolarity.POSITIVE),
        source_ref="test", status="active")


def seed_rule(mem):
    mem.upsert_rule({
        "id": "r_hold",
        "conditions": [{"type": "has_attr", "concept": "?x",
                        "key": "holds", "value": True}],
        "conclusion": {"type": "has_attr", "concept": "?x",
                       "key": "stores", "value": True},
        "exceptions": []})


def candidate():
    s0 = wd.make_state("s0", 0, [fact(WATER, "inside", CUP,
                                       WorldPolarity.NEGATIVE)])
    pour = WorldActionSchema(
        action_id="pour", name="倒入", params=("cup", "liquid"),
        premises=(WorldLiteralPattern(
            subject=var("liquid"), predicate="inside",
            object=var("cup"), polarity=WorldPolarity.NEGATIVE),),
        effects=(WorldEffect(WorldLiteralPattern(
            subject=var("liquid"), predicate="inside",
            object=var("cup"),
            polarity=WorldPolarity.POSITIVE)),))
    q = WorldLiteralPattern(subject=const(WATER), predicate="inside",
                            object=const(CUP),
                            polarity=WorldPolarity.POSITIVE)
    fc = wp.forecast_action(s0, pour, {"cup": CUP, "liquid": WATER},
                            [q], [], new_state_id="s1", event_id="e1")
    ref = wd.make_state("ref", 0, [fact(WATER, "inside", CUP,
                                        WorldPolarity.NEGATIVE)])
    cmp = wp.compare_forecast(
        fc,
        wd.make_state("ref2", 0, [fact(WATER, "inside", CUP,
                                       WorldPolarity.NEGATIVE)]),
        [])
    assert cmp.overall == "mismatch"
    from cognitive_ai.world_review import review_prediction as _rp
    rep = _rp(fc, cmp, wd.make_state(
        "ref2", 0, [fact(WATER, "inside", CUP,
                          WorldPolarity.NEGATIVE)]), [])
    assert rep.overall == "confirmed_mismatch"
    assert len(rep.candidates) == 1
    return rep.candidates[0]


def ev(eid, group, kind="independent_observation",
       direction="supports"):
    from cognitive_ai.models import WorldReviewEvidence
    return WorldReviewEvidence(
        evidence_id=eid, source_id=f"src-{group}",
        independence_group=group, kind=kind, direction=direction,
        proposition_ref="rule:r_hold stores scope",
        scope_note=f"run {group}")


def reg(pid, passed=True):
    return {"check_id": "rg-9", "scope": "tests/test_world_reasoning.py",
            "passed": passed, "source": "pytest 2026-10-13",
            "proposal_id": pid}


def ready_package(tmp_path, tag, operation="revise", target="r_hold",
                  after_key="stockpiles", pid="chg-1"):
    m = Memory(str(tmp_path / f"{tag}.db"))
    seed_rule(m)
    cand = candidate()
    dec = wg.review_candidate(cand, "accepted_for_investigation",
                              "op", "go")
    kw = {"expected_before": holds_rule_content(),
          "proposed_after": holds_rule_content(key=after_key)}
    if operation == "add":
        kw = {"proposed_after": holds_rule_content()}
    if operation == "disable":
        kw = {"expected_before": holds_rule_content()}
    prop = wg.create_change_proposal(
        cand, dec, "inference_rule", target, operation, reason="r",
        regression_note="n", proposal_id=pid, **kw)
    evs = [ev("e1", "g1"), ev("e2", "g2")]
    gate = wg.evaluate_change_gate(dec, prop, evs,
                                   current_target=holds_rule_content()
                                   if operation != "add" else None,
                                   absent_proof="lookup: absent"
                                   if operation == "add" else "",
                                   regression=reg(pid))
    assert gate.status == "ready_for_application", gate
    auth = authorize_change_application(prop, gate, "chief", "ship it",
                                        auth_id="auth-1")
    return m, cand, dec, prop, gate, evs, reg(pid), auth


# ---- A. authorization ----
def test_no_auth_no_apply(tmp_path):
    m, cand, dec, prop, gate, evs, rg, auth = ready_package(
        tmp_path, "apA1")
    try:
        bad = dict(vars(auth))
        import dataclasses
        forged = dataclasses.replace(auth, proposal_fingerprint="0" * 64)
        out = apply_world_change(forged, prop, dec, gate, evs, rg, m)
        assert out.status == "blocked_authorization", out
        assert [r["id"] for r in m.list_rules()] == ["r_hold"]
        with pytest.raises(ValueError):
            authorize_change_application(prop, gate, "", "why",
                                         auth_id="a")
        with pytest.raises(ValueError):
            authorize_change_application(prop, gate, "chief", "",
                                         auth_id="a")
    finally:
        m.close()


def test_acceptance_is_not_authorization(tmp_path):
    import dataclasses
    m, cand, dec, prop, gate, evs, rg, auth = ready_package(
        tmp_path, "apA2")
    try:
        forged = dataclasses.replace(auth, proposal_fingerprint="0" * 64)
        out = apply_world_change(forged, prop, dec, gate, evs, rg, m)
        assert out.status == "blocked_authorization", out
        rej = wg.review_candidate(cand, "rejected", "op2", "no")
        out2 = apply_world_change(auth, prop, rej, gate, evs, rg, m)
        assert out2.status == "blocked_gate", out2
        assert [r["id"] for r in m.list_rules()] == ["r_hold"]
    finally:
        m.close()


# ---- B. gate re-validation ----
def test_stale_gate_and_drift_blocked(tmp_path):
    m, cand, dec, prop, gate, evs, rg, auth = ready_package(
        tmp_path, "apB1")
    try:
        prop_other = wg.create_change_proposal(
            cand, dec, "inference_rule", "r_hold", "revise",
            expected_before=holds_rule_content(),
            proposed_after=holds_rule_content(key="other"),
            reason="rival", regression_note="n", proposal_id="chg-other-1")
        gate_other = wg.evaluate_change_gate(
            dec, prop_other, evs, current_target=holds_rule_content(),
            regression=reg("chg-other-1"))
        assert gate_other.status == "ready_for_application", gate_other
        out = apply_world_change(auth, prop, dec, gate_other, evs, rg, m)
        assert out.status == "blocked_gate", out
        thin = [e for e in evs if e.evidence_id == "e1"]
        out2 = apply_world_change(auth, prop, dec, gate, thin, rg, m)
        assert out2.status == "blocked_gate", out2
        bad_reg = dict(rg)
        bad_reg["passed"] = False
        out3 = apply_world_change(auth, prop, dec, gate, evs, bad_reg, m)
        assert out3.status == "blocked_gate", out3
        m.upsert_rule({
            "id": "r_hold",
            "conditions": [{"type": "has_attr", "concept": "?x",
                            "key": "holds", "value": False}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "stores", "value": True},
            "exceptions": []})
        out4 = apply_world_change(auth, prop, dec, gate, evs, rg, m)
        assert out4.status == "blocked_stale_target", out4
        assert [r["id"] for r in m.list_rules()] == ["r_hold"]
        assert m.list_lifecycle("r_hold") == []
    finally:
        m.close()


# ---- C. rule changes ----
def test_add_revise_disable(tmp_path):
    m, _c, dec, _p, _g, _e, _r, _a = ready_package(tmp_path, "apC0")
    m.close()
    m, cand, dec, prop, gate, evs, rg, auth = ready_package(
        tmp_path, "apC1", operation="add", target="r_new",
        pid="chg-add-1")
    try:
        out = apply_world_change(auth, prop, dec, gate, evs, rg, m)
        assert out.status == "applied_pending_regression", out
        row = [r for r in m.list_rules() if r["id"] == "r_new"][0]
        assert row["status"] == "active", row
        assert row["source"] == "world-change:chg-add-1", row
    finally:
        m.close()
    m, cand, dec, prop, gate, evs, rg, auth = ready_package(
        tmp_path, "apC2", pid="chg-rev-1")
    try:
        out = apply_world_change(auth, prop, dec, gate, evs, rg, m)
        assert out.status == "applied_pending_regression", out
        by_id = {r["id"]: r for r in m.list_rules()}
        assert by_id["r_hold"]["status"] == "disabled", by_id
        assert by_id["r_hold@chg-rev-1"]["status"] == "active", by_id
        assert by_id["r_hold@chg-rev-1"]["source"] == \
            "world-change:chg-rev-1"
        lcs = [e for e in m.list_lifecycle("r_hold")
               if "chg-rev-1" in (e.get("reason") or "")]
        assert lcs and lcs[0]["to_state"] == "disabled", lcs
    finally:
        m.close()
    m, cand, dec, prop, gate, evs, rg, auth = ready_package(
        tmp_path, "apC3", operation="disable", pid="chg-dis-1")
    try:
        out = apply_world_change(auth, prop, dec, gate, evs, rg, m)
        assert out.status == "applied_pending_regression", out
        row = [r for r in m.list_rules() if r["id"] == "r_hold"][0]
        assert row["status"] == "disabled", row
        assert row["conclusion"]["key"] == "stores", row
    finally:
        m.close()


def test_unsupported_shapes(tmp_path):
    m, _c, _d, _p, _g, _e, _r, _a = ready_package(tmp_path, "apC4")
    try:
        neg = WorldInferenceRule(
            rule_id="neg", premises=(WorldLiteralPattern(
                subject=var("x"), predicate="holds",
                object=lit_const(True),
                polarity=WorldPolarity.POSITIVE),),
            conclusion=WorldLiteralPattern(
                subject=var("x"), predicate="stores",
                object=lit_const(True),
                polarity=WorldPolarity.NEGATIVE))
        ok, _why = world_rule_to_legacy(
            neg, rule_id="neg", source="s")
        assert ok is False
        cand = candidate()
        dec = wg.review_candidate(cand, "accepted_for_investigation",
                                  "op", "go")
        prop = wg.create_change_proposal(
            cand, dec, "inference_rule", "r_new", "add",
            proposed_after=neg, reason="r", regression_note="n",
            proposal_id="chg-neg-1")
        gate = wg.evaluate_change_gate(
            dec, prop, [ev("e1", "g1"), ev("e2", "g2")],
            absent_proof="lookup absent", regression=reg("chg-neg-1"))
        assert gate.status == "ready_for_application", gate
        auth = authorize_change_application(prop, gate, "chief", "go",
                                            auth_id="a-neg")
        out = apply_world_change(auth, prop, dec, gate,
                                 [ev("e1", "g1"), ev("e2", "g2")],
                                 reg("chg-neg-1"), m)
        assert out.status == "blocked_unsupported_change", out
        action_prop = wg.create_change_proposal(
            cand, dec, "action_schema", "pour", "disable",
            expected_before=pour_schema(), reason="r",
            regression_note="n", proposal_id="chg-act-1")
        from cognitive_ai.models import WorldActionSchema as _AS
        assert isinstance(action_prop.expected_before, _AS)
    finally:
        m.close()


def pour_schema():
    from cognitive_ai.models import WorldActionSchema as _AS
    from cognitive_ai.models import WorldEffect as _WE
    return _AS(action_id="pour", name="倒入", params=("cup",),
               premises=(), effects=(_WE(),))


def test_status_disallows_change(tmp_path):
    m, _c, _d, _p, _g, _e, _r, _a = ready_package(tmp_path, "apC5")
    try:
        m.set_rule_status("r_hold", "disabled", "test park")
        cand = candidate()
        dec = wg.review_candidate(cand, "accepted_for_investigation",
                                  "op", "go")
        prop = wg.create_change_proposal(
            cand, dec, "inference_rule", "r_hold", "disable",
            expected_before=holds_rule_content(), reason="r",
            regression_note="n", proposal_id="chg-park-1")
        gate = wg.evaluate_change_gate(
            dec, prop, [ev("e1", "g1"), ev("e2", "g2")],
            current_target=holds_rule_content(),
            regression=reg("chg-park-1"))
        assert gate.status == "ready_for_application", gate
        auth = authorize_change_application(prop, gate, "chief", "go",
                                            auth_id="a-park")
        out = apply_world_change(auth, prop, dec, gate,
                                 [ev("e1", "g1"), ev("e2", "g2")],
                                 reg("chg-park-1"), m)
        assert out.status == "blocked_unsupported_change", out
    finally:
        m.close()


# ---- D. atomicity & audit ----
def test_failed_write_leaves_nothing(tmp_path, monkeypatch):
    m, cand, dec, prop, gate, evs, rg, auth = ready_package(
        tmp_path, "apD1", pid="chg-fail-1")
    try:
        real = m.apply_rule_change

        def boom(**_k):
            raise RuntimeError("disk gone")

        monkeypatch.setattr(m, "apply_rule_change", boom)
        out = apply_world_change(auth, prop, dec, gate, evs, rg, m)
        assert out.status == "failed_atomic_write", out
        assert [r["id"] for r in m.list_rules()] == ["r_hold"]
        assert m.list_lifecycle("r_hold") == []
        changes = m.query("SELECT * FROM schema_changes")
        assert len(changes) == 1, changes
        assert changes[0]["status"] == "proposed", changes
        assert m.query("SELECT * FROM experiences") == []
    finally:
        m.close()


def test_repeat_apply_idempotent(tmp_path):
    m, cand, dec, prop, gate, evs, rg, auth = ready_package(
        tmp_path, "apD2", pid="chg-rep-1")
    try:
        first = apply_world_change(auth, prop, dec, gate, evs, rg, m)
        assert first.status == "applied_pending_regression", first
        n_rules = len(m.list_rules())
        n_life = len(m.query("SELECT * FROM lifecycle_events"))
        second = apply_world_change(auth, prop, dec, gate, evs, rg, m)
        assert second.status == "applied_pending_regression", second
        assert "already applied" in second.reason
        assert len(m.list_rules()) == n_rules
        assert len(m.query("SELECT * FROM lifecycle_events")) == n_life
    finally:
        m.close()


def test_conflicting_proposal_blocked(tmp_path):
    m, cand, dec, prop, gate, evs, rg, auth = ready_package(
        tmp_path, "apD3", pid="chg-first-1")
    try:
        assert apply_world_change(
            auth, prop, dec, gate, evs, rg,
            m).status == "applied_pending_regression"
        prop2 = wg.create_change_proposal(
            cand, dec, "inference_rule", "r_hold", "revise",
            expected_before=holds_rule_content(),
            proposed_after=holds_rule_content(key="other"),
            reason="rival", regression_note="n", proposal_id="chg-rival-1")
        gate2 = wg.evaluate_change_gate(
            dec, prop2, evs, current_target=holds_rule_content(),
            regression=reg("chg-rival-1"))
        assert gate2.status == "ready_for_application", gate2
        auth2 = authorize_change_application(prop2, gate2, "chief", "go",
                                             auth_id="a-rival")
        out = apply_world_change(auth2, prop2, dec, gate2, evs,
                                 reg("chg-rival-1"), m)
        assert out.status == "blocked_stale_target", out
        by_id = {r["id"]: r for r in m.list_rules()}
        assert by_id["r_hold@chg-first-1"]["status"] == "active"
    finally:
        m.close()


# ---- E. post-apply validation ----
def validation_for(app, passed=True):
    return {"validation_id": "val-1", "proposal_id": app.proposal_id,
            "target_ref": app.after_ref,
            "target_fingerprint": app.applied_fingerprint,
            "scope": "tests/test_world_reasoning.py", "passed": passed,
            "report_ref": "pytest run 9", "source": "operator"}


def test_pending_then_verified(tmp_path):
    from cognitive_ai.world_change_apply import \
        record_post_apply_validation
    m, cand, dec, prop, gate, evs, rg, auth = ready_package(
        tmp_path, "apE1", pid="chg-val-1")
    try:
        app = apply_world_change(auth, prop, dec, gate, evs, rg, m)
        assert app.status == "applied_pending_regression", app
        assert app.needs_regression_followup is True
        incomplete = dict(validation_for(app))
        del incomplete["scope"]
        same, info = record_post_apply_validation(app, incomplete, m)
        assert same.status == "applied_pending_regression"
        assert "error" in info
        done, info = record_post_apply_validation(
            app, validation_for(app), m)
        assert done.status == "applied_verified", done
        assert done.needs_regression_followup is False
    finally:
        m.close()


def test_regression_failure_compensates(tmp_path):
    from cognitive_ai.world_change_apply import \
        record_post_apply_validation
    m, cand, dec, prop, gate, evs, rg, auth = ready_package(
        tmp_path, "apE2", pid="chg-comp-1")
    try:
        app = apply_world_change(auth, prop, dec, gate, evs, rg, m)
        assert app.status == "applied_pending_regression", app
        bad = validation_for(app, passed=False)
        done, info = record_post_apply_validation(app, bad, m)
        assert done.status == "compensation_required", done
        by_id = {r["id"]: r for r in m.list_rules()}
        assert by_id["r_hold"]["status"] == "disabled"
        assert by_id["r_hold"]["conclusion"]["key"] == "stores"
        assert by_id["r_hold@chg-comp-1"]["status"] == "disabled"
        assert by_id["r_hold@chg-comp-1"]["conclusion"]["key"] == \
            "stockpiles"
        hist = [e for e in m.list_lifecycle("r_hold@chg-comp-1")]
        assert hist, "new version history kept"
        assert info.get("compensated"), info
    finally:
        m.close()


# ---- F. purity & freezing ----
def test_checks_without_apply_write_nothing(tmp_path):
    m = Memory(str(tmp_path / "apF1.db"))
    seed_rule(m)
    try:
        from cognitive_ai.world_learning_gate import evaluate_change_gate
        n0 = (len(m.list_rules()),
              len(m.query("SELECT * FROM lifecycle_events")),
              len(m.query("SELECT * FROM experiences")),
              len(m.query("SELECT * FROM schema_changes")))
        cand = candidate()
        dec = wg.review_candidate(cand, "accepted_for_investigation",
                                  "op", "go")
        prop = wg.create_change_proposal(
            cand, dec, "inference_rule", "r_hold", "revise",
            expected_before=holds_rule_content(),
            proposed_after=holds_rule_content(key="other"),
            reason="r", regression_note="n", proposal_id="chg-ck-1")
        evs = [ev("e1", "g1"), ev("e2", "g2")]
        gate = evaluate_change_gate(
            dec, prop, evs, current_target=holds_rule_content(),
            regression=reg("chg-ck-1"))
        assert gate.status == "ready_for_application", gate
        auth = authorize_change_application(prop, gate, "chief", "go",
                                            auth_id="a-ck")
        assert (len(m.list_rules()),
                len(m.query("SELECT * FROM lifecycle_events")),
                len(m.query("SELECT * FROM experiences")),
                len(m.query("SELECT * FROM schema_changes"))) == n0
    finally:
        m.close()


def test_apply_objects_frozen():
    m = Memory(":memory:")
    m.upsert_rule({
        "id": "r_hold",
        "conditions": [{"type": "has_attr", "concept": "?x",
                        "key": "holds", "value": True}],
        "conclusion": {"type": "has_attr", "concept": "?x",
                       "key": "stores", "value": True},
        "exceptions": []})
    try:
        cand = candidate()
        dec = wg.review_candidate(cand, "accepted_for_investigation",
                                  "op", "go")
        prop = wg.create_change_proposal(
            cand, dec, "inference_rule", "r_hold", "revise",
            expected_before=holds_rule_content(),
            proposed_after=holds_rule_content(key="other"),
            reason="r", regression_note="n", proposal_id="chg-fr-1")
        gate = wg.evaluate_change_gate(
            dec, prop, [ev("e1", "g1"), ev("e2", "g2")],
            current_target=holds_rule_content(),
            regression=reg("chg-fr-1"))
        auth = authorize_change_application(prop, gate, "chief", "go",
                                            auth_id="a-fr")
        app = apply_world_change(auth, prop, dec, gate,
                                 [ev("e1", "g1"), ev("e2", "g2")],
                                 reg("chg-fr-1"), m)
        assert app.status == "applied_pending_regression", app
        with pytest.raises(AttributeError):
            auth.reason = "forged"
        with pytest.raises(AttributeError):
            app.audit_refs = ()
        val = {"validation_id": "v", "proposal_id": app.proposal_id,
               "target_ref": app.after_ref,
               "target_fingerprint": app.applied_fingerprint,
               "scope": "s", "passed": True, "report_ref": "r",
               "source": "op"}
        from cognitive_ai.world_change_apply import \
            record_post_apply_validation
        done, _info = record_post_apply_validation(app, val, m)
        assert done.status == "applied_verified", done
    finally:
        m.close()
