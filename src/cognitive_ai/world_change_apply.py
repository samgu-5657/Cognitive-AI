"""Controlled change application (Phase M4-I).

Coordinates authorization checks, gate re-validation, atomic rule
writes and audit trails. Pure orchestration over formal interfaces:
Memory (single-transaction bundles), SchemaEvolver (change intent),
lifecycle helpers. No direct SQLite, no invented rule semantics.

Only inference_rule targets with lossless world mapping are
supported; action schemas have no persistence and fail closed.
Simulation results are never written as knowledge.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Optional

from . import world_learning_gate as _gate
from .memory import Memory
from .models import (WorldActionSchema, WorldApplicationAuthorization,
                     WorldApplicationStatus, WorldChangeApplicationResult,
                     WorldChangeOperation, WorldChangeProposal,
                     WorldChangeTarget, WorldInferenceRule,
                     WorldLearningGateResult, WorldPostApplyValidation,
                     WorldReviewDecision)
from .schema_evolver import SchemaEvolver

APPLIED_PENDING = WorldApplicationStatus.APPLIED_PENDING_REGRESSION.value
APPLIED_VERIFIED = WorldApplicationStatus.APPLIED_VERIFIED.value
BLOCKED_AUTH = WorldApplicationStatus.BLOCKED_AUTHORIZATION.value
BLOCKED_GATE = WorldApplicationStatus.BLOCKED_GATE.value
BLOCKED_STALE = WorldApplicationStatus.BLOCKED_STALE_TARGET.value
BLOCKED_STALE_TARGET = BLOCKED_STALE
BLOCKED_UNSUPPORTED = WorldApplicationStatus.BLOCKED_UNSUPPORTED_CHANGE.value
BLOCKED_INVALID = WorldApplicationStatus.BLOCKED_INVALID_PROPOSAL.value
FAILED_WRITE = WorldApplicationStatus.FAILED_ATOMIC_WRITE.value
NEED_COMPENSATION = WorldApplicationStatus.COMPENSATION_REQUIRED.value

READY = "ready_for_application"
ACCEPTED = "accepted_for_investigation"
REVISE_STATUSES = ("active", "validated")

ABSENT_MARK = "absent"


def _canonical(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    from enum import Enum
    if isinstance(obj, Enum):
        return obj.value
    from types import MappingProxyType
    if isinstance(obj, MappingProxyType):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in sorted(
            obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if dataclasses.is_dataclass(obj):
        return {"__type__": type(obj).__name__,
                **{f.name: _canonical(getattr(obj, f.name))
                   for f in sorted(dataclasses.fields(obj),
                                   key=lambda f: f.name)}}
    raise TypeError(f"unserializable: {type(obj).__name__}")


def _fingerprint(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(_canonical(obj), ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()


def _proposal_fingerprint(proposal: WorldChangeProposal) -> str:
    return _fingerprint({
        "proposal_id": proposal.proposal_id,
        "candidate_ref": proposal.candidate_ref,
        "target_type": proposal.target_type,
        "target_id": proposal.target_id,
        "operation": proposal.operation,
        "expected_before": _canonical(proposal.expected_before)
        if proposal.expected_before is not None else None,
        "proposed_after": _canonical(proposal.proposed_after)
        if proposal.proposed_after is not None else None,
        "reason": proposal.reason,
        "status": proposal.status,
    })


def _gate_fingerprint(gate: WorldLearningGateResult) -> str:
    return _fingerprint({
        "proposal_id": gate.proposal_id,
        "status": gate.status,
        "qualifying_groups": list(gate.qualifying_groups),
        "opposing": list(gate.opposing),
        "regression": _canonical(dict(gate.regression_check)),
    })


def _expected_target_fingerprint(proposal: WorldChangeProposal) -> str:
    if proposal.operation == WorldChangeOperation.ADD.value:
        return f"{ABSENT_MARK}:{proposal.target_id}"
    return _fingerprint(_canonical(proposal.expected_before))


def authorize_change_application(proposal: WorldChangeProposal,
                                 gate: WorldLearningGateResult,
                                 authorized_by: str, reason: str,
                                 auth_id: str = "") -> Any:
    """Authorize one exact proposal. Investigation and gate readiness
    never substitute for this authorization."""
    if not isinstance(proposal, WorldChangeProposal):
        raise TypeError("not a change proposal")
    if not isinstance(gate, WorldLearningGateResult):
        raise TypeError("not a gate result")
    if gate.proposal_id != proposal.proposal_id:
        raise ValueError("gate belongs to another proposal")
    if gate.status != READY:
        raise ValueError(f"gate is {gate.status}, not ready")
    return WorldApplicationAuthorization(
        auth_id=auth_id, proposal_id=proposal.proposal_id,
        proposal_fingerprint=_proposal_fingerprint(proposal),
        gate_fingerprint=_gate_fingerprint(gate),
        authorized_by=authorized_by, reason=reason,
        operation=proposal.operation, target_id=proposal.target_id,
        expected_target_fingerprint=_expected_target_fingerprint(
            proposal))


def _legacy_slot_to_pattern_note(*_a, **_k):
    raise NotImplementedError


def world_rule_to_legacy(rule: WorldInferenceRule, *, rule_id: str,
                         source: str) -> tuple:
    """World rule -> legacy rule row. (ok, row-or-reason).

    Shape rule: reference objects become relation endpoints, literals
    become has_attr values, variables stay variables. Anything else
    (negative polarity, literal subjects, variable objects, empty
    predicates) has no lossless legacy form and is refused.
    """
    problems = []
    try:
        from .world_reasoning import validate_inference_rule
        problems = list(validate_inference_rule(rule))
    except Exception as e:
        return False, f"rule invalid: {e}"
    if problems:
        return False, "; ".join(problems)

    def convert(pattern, where: str):
        if pattern.polarity != "positive":
            return None, f"{where}: non-positive polarity"
        subj = pattern.subject
        if subj.is_variable():
            src = "?" + subj.variable
        elif subj.ref is not None:
            src = subj.ref.ref_id
        else:
            return None, f"{where}: literal subject"
        if not pattern.predicate:
            return None, f"{where}: empty predicate"
        obj = pattern.object
        if obj.is_variable():
            return None, f"{where}: variable object has no legacy shape"
        if obj.ref is not None:
            return {"type": "relation", "src": src,
                    "rel_type": pattern.predicate,
                    "dst": obj.ref.ref_id}, ""
        if obj.has_literal:
            return {"type": "has_attr", "concept": src,
                    "key": pattern.predicate, "value": obj.literal}, ""
        return None, f"{where}: empty object"

    conditions = []
    for i, prem in enumerate(rule.premises):
        mapped, why = convert(prem, f"premise {i}")
        if why:
            return False, why
        conditions.append(mapped)
    target, why = convert(rule.conclusion, "conclusion")
    if why:
        return False, why
    return True, {"id": rule_id, "name": rule_id,
                  "conditions": conditions, "conclusion": target,
                  "exceptions": [], "source": source, "status": "active"}


def _legacy_semantic(row: dict) -> dict:
    import json as _json
    def load(value, default):
        if isinstance(value, (list, dict)):
            return value
        try:
            return _json.loads(value)
        except Exception:
            return default

    return {"conditions": load(row.get("conditions"), []),
            "conclusion": load(row.get("conclusion"), {}),
            "exceptions": load(row.get("exceptions"), [])}


def _recheck_gate_inputs(proposal, gate, evidences, regression):
    """Evidence and regression must be unchanged since the gate ran."""
    pid = proposal.proposal_id
    if not isinstance(evidences, (list, tuple)):
        return _blocked(pid, proposal.target_id, proposal.operation,
                        BLOCKED_GATE, "evidences changed shape")
    groups, opposing, bad = _gate.group_qualified_evidences(evidences)
    if bad is not None:
        return _blocked(pid, proposal.target_id, proposal.operation,
                        BLOCKED_GATE, f"bad evidence entry {bad!r}")
    if tuple(sorted(groups)) != tuple(gate.qualifying_groups) or \
            tuple(sorted(set(opposing))) != tuple(sorted(set(
                gate.opposing))):
        return _blocked(pid, proposal.target_id, proposal.operation,
                        BLOCKED_GATE,
                        "evidence set changed since gate qualification")
    if not isinstance(regression, dict) or \
            dict(regression) != dict(gate.regression_check):
        return _blocked(pid, proposal.target_id, proposal.operation,
                        BLOCKED_GATE,
                        "regression record changed since gate qualification")
    return None


def record_post_apply_validation(application: WorldChangeApplicationResult,
                                   validation: dict,
                                   mem: Memory) -> tuple:
    """Record post-apply validation; compensate on regression failure.

    Returns (updated result, info). The stored validation must name the
    actually written target fingerprint; compensation uses the formal
    status API and keeps every audit row.
    """
    import dataclasses
    if not isinstance(application, WorldChangeApplicationResult):
        return application, {"error": "not an application result"}
    if application.status != APPLIED_PENDING:
        return application, {
            "error": f"nothing pending ({application.status})"}
    if not isinstance(validation, dict):
        return application, {"error": "validation record missing"}
    for key in ("validation_id", "proposal_id", "target_ref",
                "target_fingerprint", "scope", "report_ref", "source"):
        if not validation.get(key):
            return application, {"error": f"validation {key} missing"}
    if "passed" not in validation:
        return application, {"error": "validation passed flag missing"}
    if validation["proposal_id"] != application.proposal_id:
        return application, {"error": "validation linked elsewhere"}
    after_id = (application.after_ref or "").split("rule:", 1)[-1]
    rows = [r for r in mem.list_rules() if r.get("id") == after_id]
    if not rows:
        return application, {"error": "applied target no longer present"}
    fresh_fp = _fingerprint(_legacy_semantic(rows[0]))
    if fresh_fp != validation["target_fingerprint"] or \
            fresh_fp != application.applied_fingerprint:
        return application, {
            "error": "validation targets another version"}
    mem.add_experience(
        "evaluation",
        f"post-apply validation {validation['validation_id']} for "
        f"{application.proposal_id}: "
        f"{'passed' if validation['passed'] is True else 'failed'}",
        {"proposal_id": application.proposal_id,
         "target_ref": validation["target_ref"],
         "target_fingerprint": fresh_fp,
         "scope": validation["scope"],
         "report_ref": validation["report_ref"],
         "source": validation["source"]})
    if validation["passed"] is True:
        done = dataclasses.replace(
            application, status=APPLIED_VERIFIED,
            needs_regression_followup=False,
            reason=application.reason + "; regression verified")
        return done, {"validated": True,
                      "validation_id": validation["validation_id"]}
    info: dict = {"validated": False,
                  "validation_id": validation["validation_id"]}
    try:
        if application.operation in (
                WorldChangeOperation.ADD.value,
                WorldChangeOperation.REVISE.value):
            mem.set_rule_status(
                after_id, "disabled",
                f"compensate failed regression for "
                f"{application.proposal_id}")
            info["compensated"] = f"disabled {after_id}"
        else:
            prior = None
            for ev in mem.list_lifecycle(application.target_id):
                if application.proposal_id in (ev.get("reason") or "") \
                        and ev.get("to_state") == "disabled":
                    prior = ev.get("from_state")
            if prior not in ("active", "validated"):
                info["compensated"] = False
                info["manual"] = "previous status not recoverable here"
            else:
                mem.set_rule_status(
                    application.target_id, prior,
                    f"compensate failed regression for "
                    f"{application.proposal_id}")
                info["compensated"] = \
                    f"restored {application.target_id} to {prior}"
    except Exception as e:
        info["compensated"] = False
        info["manual"] = f"compensation failed: {e}"
    mem.add_experience(
        "evaluation",
        f"compensation for {application.proposal_id}: "
        f"{info.get('compensated')}",
        {"proposal_id": application.proposal_id, "info": info})
    compensated = dataclasses.replace(
        application, status=NEED_COMPENSATION,
        needs_regression_followup=False,
        reason=application.reason + "; regression failed")
    return compensated, info


def _blocked(proposal_id: str, target_id: str, operation: str,
             status: str, reason: str,
             expected_fp: str = "") -> WorldChangeApplicationResult:
    return WorldChangeApplicationResult(
        proposal_id=proposal_id, target_id=target_id, operation=operation,
        before_ref=f"rule:{target_id}" if target_id else "",
        after_ref="", status=status, reason=reason,
        needs_regression_followup=False,
        expected_fingerprint=expected_fp, applied_fingerprint="")


def apply_world_change(auth: WorldApplicationAuthorization,
                       proposal: WorldChangeProposal,
                       decision: Any,
                       gate: WorldLearningGateResult,
                       evidences, regression: dict,
                       mem: Memory) -> WorldChangeApplicationResult:
    """Re-validate everything against live reads, then apply atomically.

    propose-intent goes through SchemaEvolver; the mutation plus all
    audit rows commit in one Memory transaction. Any failure leaves no
    partial rule or audit behind.
    """
    pid = getattr(proposal, "proposal_id", "")
    tid = getattr(proposal, "target_id", "")
    op = getattr(proposal, "operation", "")
    if not isinstance(auth, WorldApplicationAuthorization) or \
            not isinstance(proposal, WorldChangeProposal) or \
            not isinstance(gate, WorldLearningGateResult):
        return _blocked(pid, tid, op, BLOCKED_INVALID,
                        "authorization/proposal/gate mistyped")
    if auth.proposal_id != proposal.proposal_id or \
            _proposal_fingerprint(proposal) != auth.proposal_fingerprint:
        return _blocked(pid, tid, op, BLOCKED_AUTH,
                        "authorization does not bind this proposal content")
    if auth.operation != proposal.operation or \
            auth.target_id != proposal.target_id:
        return _blocked(pid, tid, op, BLOCKED_AUTH,
                        "authorization scope mismatch")
    if _gate_fingerprint(gate) != auth.gate_fingerprint or \
            gate.proposal_id != proposal.proposal_id:
        return _blocked(pid, tid, op, BLOCKED_GATE,
                        "gate result does not match authorization")
    if gate.status != READY:
        return _blocked(pid, tid, op, BLOCKED_GATE,
                        f"gate is {gate.status}")
    if not isinstance(decision, WorldReviewDecision) or \
            decision.candidate_ref != proposal.candidate_ref:
        return _blocked(pid, tid, op, BLOCKED_GATE,
                        "decision does not cover this proposal")
    if decision.decision != ACCEPTED:
        return _blocked(pid, tid, op, BLOCKED_GATE,
                        f"review state is {decision.decision}")
    rechecked = _recheck_gate_inputs(proposal, gate, evidences,
                                      regression)
    if rechecked is not None:
        return rechecked
    if proposal.target_type == WorldChangeTarget.ACTION_SCHEMA.value:
        return _blocked(pid, tid, op, BLOCKED_UNSUPPORTED,
                        "action schemas have no persistence",
                        auth.expected_target_fingerprint)
    if proposal.target_type != WorldChangeTarget.INFERENCE_RULE.value:
        return _blocked(pid, tid, op, BLOCKED_INVALID,
                        "unknown target type")
    expected_fp = _expected_target_fingerprint(proposal)
    if expected_fp != auth.expected_target_fingerprint:
        return _blocked(pid, tid, op, BLOCKED_AUTH,
                        "expected target drifted from authorization",
                        expected_fp)

    rows = [r for r in mem.list_rules() if r.get("id") == tid]
    current = rows[0] if rows else None
    marker = f"world-change:{pid}"
    if op == WorldChangeOperation.ADD.value:
        if current is not None:
            if current.get("source") == marker:
                content_ok, _legacy = world_rule_to_legacy(
                    proposal.proposed_after, rule_id=tid, source=marker)
                if content_ok and _legacy_semantic(current) == {
                        "conditions": _legacy["conditions"],
                        "conclusion": _legacy["conclusion"],
                        "exceptions": []}:
                    return _applied(pid, tid, op, f"rule:{tid}",
                                    f"rule:{tid}", expected_fp,
                                    _fingerprint(_legacy_semantic(current)),
                                    (f"change:already-recorded",),
                                    "already applied", False)
            return _blocked(pid, tid, op, BLOCKED_STALE_TARGET,
                            "target id already in use", expected_fp)
        ok, legacy = world_rule_to_legacy(
            proposal.proposed_after, rule_id=tid, source=marker)
        if not ok:
            return _blocked(pid, tid, op, BLOCKED_UNSUPPORTED, legacy,
                            expected_fp)
        return _commit(mem, pid, tid, op, None, legacy, None,
                       proposal, expected_fp, evidences, regression,
                       decision, marker)
    if current is None:
        return _blocked(pid, tid, op, BLOCKED_STALE_TARGET,
                        "target rule missing", expected_fp)
    ok, want = world_rule_to_legacy(
        proposal.expected_before, rule_id=tid, source=current.get(
            "source", ""))
    if not ok:
        return _blocked(pid, tid, op, BLOCKED_INVALID,
                        f"expected content not representable: {want}",
                        expected_fp)
    if _legacy_semantic(current) != {"conditions": want["conditions"],
                                     "conclusion": want["conclusion"],
                                     "exceptions": []}:
        return _blocked(pid, tid, op, BLOCKED_STALE_TARGET,
                        "current target differs from expected content",
                        expected_fp)
    if op == WorldChangeOperation.DISABLE.value:
        prior = _already_applied_disable(mem, pid, tid, expected_fp)
        if prior is not None:
            return prior
        successor = _superseded_by(mem, tid)
        if successor is not None:
            return _blocked(pid, tid, op, BLOCKED_STALE_TARGET,
                            f"superseded by {successor}", expected_fp)
        if current.get("status") not in REVISE_STATUSES:
            return _blocked(pid, tid, op, BLOCKED_UNSUPPORTED,
                            f"status {current.get('status')} disallows it",
                            expected_fp)
        return _commit(mem, pid, tid, op, current, None, "disabled",
                       proposal, expected_fp, evidences, regression,
                       decision, marker)
    if op == WorldChangeOperation.REVISE.value:
        new_id = f"{tid}@{pid}"
        ok, legacy = world_rule_to_legacy(
            proposal.proposed_after, rule_id=new_id, source=marker)
        if not ok:
            return _blocked(pid, tid, op, BLOCKED_UNSUPPORTED, legacy,
                            expected_fp)
        prior = _already_applied_revise(mem, pid, tid, new_id, marker,
                                        legacy, expected_fp)
        if prior is not None:
            return prior
        successor = _superseded_by(mem, tid)
        if successor is not None:
            return _blocked(pid, tid, op, BLOCKED_STALE_TARGET,
                            f"superseded by {successor}", expected_fp)
        if current.get("status") not in REVISE_STATUSES:
            return _blocked(pid, tid, op, BLOCKED_UNSUPPORTED,
                            f"status {current.get('status')} disallows it",
                            expected_fp)
        return _commit(mem, pid, tid, op, current, legacy, "disabled",
                       proposal, expected_fp, evidences, regression,
                       decision, marker, new_id=new_id)
    return _blocked(pid, tid, op, BLOCKED_INVALID, "unknown operation",
                    expected_fp)


def _superseded_by(mem, tid) -> Optional[str]:
    """Newer version id recorded as superseding tid, if any."""
    for ev in mem.list_lifecycle(tid):
        if ev.get("to_state") != "disabled":
            continue
        for item in _evidence_items(ev):
            if isinstance(item, dict) and item.get("new_rule_id") \
                    and str(item["new_rule_id"]) != tid:
                return str(item["new_rule_id"])
    return None


def _evidence_items(ev: dict) -> list:
    import json as _json
    try:
        items = _json.loads(ev.get("evidence") or "[]")
        return items if isinstance(items, list) else []
    except Exception:
        return []
    rows = [r for r in mem.list_rules() if r.get("id") == new_id]
    if not rows or rows[0].get("source") != marker:
        return None
    if _legacy_semantic(rows[0]) != {
            "conditions": legacy["conditions"],
            "conclusion": legacy["conclusion"], "exceptions": []}:
        return None
    return _applied(
        pid, tid, WorldChangeOperation.REVISE.value, f"rule:{tid}",
        f"rule:{new_id}", expected_fp,
        _fingerprint(_legacy_semantic(rows[0])),
        (f"rule:{new_id}",), "already applied", True)


def _already_applied_revise(mem, pid, tid, new_id, marker, legacy,
                             expected_fp):
    rows = [r for r in mem.list_rules() if r.get("id") == new_id]
    if not rows or rows[0].get("source") != marker:
        return None
    if _legacy_semantic(rows[0]) != {
            "conditions": legacy["conditions"],
            "conclusion": legacy["conclusion"], "exceptions": []}:
        return None
    return _applied(
        pid, tid, WorldChangeOperation.REVISE.value, f"rule:{tid}",
        f"rule:{new_id}", expected_fp,
        _fingerprint(_legacy_semantic(rows[0])),
        (f"rule:{new_id}",), "already applied", True)


def _already_applied_disable(mem, pid, tid, expected_fp):
    rows = [r for r in mem.list_rules() if r.get("id") == tid]
    if not rows or rows[0].get("status") != "disabled":
        return None
    for ev in mem.list_lifecycle(tid):
        if pid in (ev.get("reason") or "") and \
                ev.get("to_state") == "disabled":
            return _applied(
                pid, tid, WorldChangeOperation.DISABLE.value,
                f"rule:{tid}", f"rule:{tid}", expected_fp,
                _fingerprint(_legacy_semantic(rows[0])),
                (f"rule:{tid}", f"lifecycle:{ev.get('id')}"),
                "already applied", True)
    return None


def _applied(pid, tid, op, before_ref, after_ref, expected_fp,
             applied_fp, audits, reason, needs_reg):
    return WorldChangeApplicationResult(
        proposal_id=pid, target_id=tid, operation=op,
        before_ref=before_ref, after_ref=after_ref,
        status=APPLIED_PENDING, audit_refs=audits, reason=reason,
        needs_regression_followup=needs_reg,
        expected_fingerprint=expected_fp,
        applied_fingerprint=applied_fp)


def _commit(mem, pid, tid, op, current, legacy, disable_to,
            proposal, expected_fp, evidences, regression, decision,
            marker, new_id=None):
    from .schema_evolver import SchemaEvolver
    write_id = new_id or tid
    payload = {"proposal_id": pid, "target_id": tid, "operation": op,
               "write_id": write_id}
    change_id = SchemaEvolver(mem).propose(
        "world_rule_change", payload,
        f"apply {op} {tid} for {pid}",
        [{"proposal_id": pid}])
    upserts, statuses, lifecycles = [], [], []
    if legacy is not None:
        upserts.append(legacy)
        lifecycles.append(("rule", write_id, "none", "active",
                           f"world-change {pid}: create {write_id}",
                           [{"proposal_id": pid}]))
    if disable_to is not None and current is not None:
        statuses.append((tid, disable_to))
        lifecycles.append(("rule", tid, current.get("status", "active"),
                           disable_to,
                           f"world-change {pid}: {op} {tid}", [
                               {"proposal_id": pid,
                                "new_rule_id": write_id}]))
    experiences = [("world_change",
                    f"applied {op} {tid} for proposal {pid}",
                    {"proposal_id": pid, "candidate_ref":
                     proposal.candidate_ref, "old_rule_id": tid,
                     "new_rule_id": write_id,
                     "expected_fingerprint": expected_fp,
                     "applied_fingerprint": _fingerprint(
                         _legacy_semantic(
                             {"conditions": legacy["conditions"],
                              "conclusion": legacy["conclusion"],
                              "exceptions": []} if legacy else {}))})]
    marks = [(change_id, "applied")]
    try:
        mem.apply_rule_change(
            rule_upserts=upserts, rule_statuses=statuses,
            lifecycles=lifecycles, experiences=experiences,
            change_marks=marks)
    except Exception as e:
        return WorldChangeApplicationResult(
            proposal_id=pid, target_id=tid, operation=op,
            before_ref=f"rule:{tid}", after_ref="",
            status=FAILED_WRITE,
            reason=f"atomic write failed, rolled back: {e}",
            needs_regression_followup=False,
            expected_fingerprint=expected_fp, applied_fingerprint="")
    applied_fp = _fingerprint(_legacy_semantic(
        {"conditions": legacy["conditions"],
         "conclusion": legacy["conclusion"],
         "exceptions": []} if legacy else _legacy_semantic(
            next(r for r in mem.list_rules() if r.get("id") == tid))))
    audits = [f"change:{change_id}", f"rule:{write_id}"]
    for ev in mem.list_lifecycle(write_id if legacy is not None else tid):
        if pid in (ev.get("reason") or ""):
            audits.append(f"lifecycle:{ev.get('id')}")
    return _applied(pid, tid, op, f"rule:{tid}", f"rule:{write_id}",
                    expected_fp, applied_fp, tuple(audits),
                    f"applied {op} {tid}", True)
