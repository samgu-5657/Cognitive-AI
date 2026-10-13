"""Review audit & learning gate (Phase M4-H).

Pure memory, deterministic, no IO: no SQLite, Memory, Learner,
SchemaEvolver, files, network, LLM or randomness. Three separated
concerns only: auditing a diagnostic candidate (investigation, never
confirmation), pinning a concrete change proposal (never derived from
hypothesis prose), and checking application qualification (never an
application). Nothing here writes knowledge or rules.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from .models import (WorldActionSchema, WorldChangeOperation,
                     WorldChangeProposal, WorldChangeTarget,
                     WorldDiagnosticCandidate, WorldEvidenceDirection,
                     WorldEvidenceKind, WorldGateStatus,
                     WorldInferenceRule, WorldLearningGateResult,
                     WorldReviewDecision, WorldReviewDecisionKind,
                     WorldReviewEvidence)

ACCEPTED = WorldReviewDecisionKind.ACCEPTED_FOR_INVESTIGATION.value
READY = WorldGateStatus.READY_FOR_APPLICATION.value
BLOCKED_EVIDENCE = WorldGateStatus.BLOCKED_INSUFFICIENT_EVIDENCE.value
BLOCKED_REVIEW = WorldGateStatus.BLOCKED_REVIEW.value
BLOCKED_REGRESSION = WorldGateStatus.BLOCKED_REGRESSION.value
BLOCKED_INVALID = WorldGateStatus.BLOCKED_INVALID_PROPOSAL.value
BLOCKED_UNSUPPORTED = WorldGateStatus.BLOCKED_UNSUPPORTED_CHANGE.value

CONFIRMED_MISMATCH = "confirmed_mismatch"
QUALIFYING_KINDS = (
    WorldEvidenceKind.INDEPENDENT_OBSERVATION.value,
    WorldEvidenceKind.USER_ASSERTION.value,
)


def _term_key(term) -> tuple:
    if term.ref is not None:
        return ("ref", term.ref.ref_id)
    if term.has_literal:
        return ("lit", repr(term.literal))
    return ("unres", term.unresolved, term.missing)


def candidate_ref(candidate: WorldDiagnosticCandidate) -> str:
    """Deterministic stable reference (sha256, no cross-process hash()).

    Covers the query key, both verdicts, classification, action, rule
    and state ids, and the reason: identical candidates re-identify,
    different ones never collide in practice.
    """
    if not isinstance(candidate, WorldDiagnosticCandidate):
        raise TypeError("not a diagnostic candidate")
    prop = candidate.query
    key = {
        "subject": list(_term_key(prop.subject)),
        "predicate": prop.predicate,
        "object": list(_term_key(prop.object)),
        "forecast": candidate.forecast_verdict,
        "reference": candidate.reference_verdict,
        "classification": candidate.classification,
        "action": candidate.action_id,
        "rules": sorted(candidate.rule_ids),
        "states": sorted(candidate.state_ids),
        "reason": candidate.reason,
    }
    digest = hashlib.sha256(
        json.dumps(key, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
    return f"cand-{digest}"


def review_candidate(candidate: WorldDiagnosticCandidate, decision: str,
                     reviewer: str, reason: str) -> WorldReviewDecision:
    """Audit one candidate. Accepted means investigation only."""
    if not isinstance(candidate, WorldDiagnosticCandidate):
        raise TypeError("not a diagnostic candidate")
    if candidate.status != "pending_review" or not candidate.reason or \
            not candidate.query.predicate:
        raise ValueError("candidate incomplete: cannot be audited")
    ref = candidate_ref(candidate)
    return WorldReviewDecision(
        candidate_ref=ref, decision=decision,
        candidate_classification=candidate.classification,
        reviewer=reviewer, reason=reason,
        simulated=candidate.simulated)


def create_change_proposal(candidate: WorldDiagnosticCandidate,
                           decision: WorldReviewDecision,
                           target_type: str, target_id: str,
                           operation: str, expected_before: Any = None,
                           proposed_after: Any = None, reason: str = "",
                           regression_note: str = "",
                           proposal_id: str = "") -> WorldChangeProposal:
    """Pin a concrete proposal. Content comes only from the caller.

    Hypothesis prose is never read here: target, operation and both
    contents are required explicit arguments.
    """
    if not isinstance(candidate, WorldDiagnosticCandidate):
        raise TypeError("not a diagnostic candidate")
    if not isinstance(decision, WorldReviewDecision):
        raise TypeError("not a review decision")
    if decision.candidate_ref != candidate_ref(candidate):
        raise ValueError("decision refers to another candidate")
    if decision.decision != ACCEPTED:
        raise ValueError("only accepted candidates proceed to proposals")
    if candidate.classification != CONFIRMED_MISMATCH:
        raise ValueError("only confirmed mismatches ground rule proposals")
    return WorldChangeProposal(
        proposal_id=proposal_id, candidate_ref=decision.candidate_ref,
        target_type=target_type, target_id=target_id, operation=operation,
        expected_before=expected_before, proposed_after=proposed_after,
        reason=reason, regression_note=regression_note,
        status="pending_review", simulated=True)


def _proposal_problems(proposal: WorldChangeProposal) -> Optional[str]:
    if not isinstance(proposal, WorldChangeProposal):
        return "not a change proposal"
    try:
        WorldChangeProposal(
            proposal_id=proposal.proposal_id or "x",
            candidate_ref=proposal.candidate_ref or "x",
            target_type=proposal.target_type,
            target_id=proposal.target_id or "x",
            operation=proposal.operation,
            expected_before=proposal.expected_before,
            proposed_after=proposal.proposed_after,
            reason=proposal.reason or "x",
            regression_note=proposal.regression_note or "x",
            status=proposal.status, simulated=proposal.simulated)
    except (ValueError, TypeError) as e:
        return str(e)
    if not proposal.proposal_id or not proposal.candidate_ref or \
            not proposal.target_id or not proposal.reason or \
            not proposal.regression_note or \
            proposal.status != "pending_review" or \
            proposal.simulated is not True:
        return "proposal fields incomplete or already applied"
    return None


def _target_model(proposal: WorldChangeProposal):
    if proposal.target_type == WorldChangeTarget.ACTION_SCHEMA.value:
        return WorldActionSchema
    if proposal.target_type == WorldChangeTarget.INFERENCE_RULE.value:
        return WorldInferenceRule
    return None


def evaluate_change_gate(decision: WorldReviewDecision,
                         proposal: WorldChangeProposal,
                         evidences, current_target: Any = None,
                         absent_proof: str = "",
                         regression: Optional[dict] = None,
                         ) -> WorldLearningGateResult:
    """Check whether a proposal qualifies. Read-only, never applies."""
    if not isinstance(decision, WorldReviewDecision):
        return _blocked("", "blocked: decision invalid",
                        BLOCKED_INVALID, "decision not audited")
    if not isinstance(proposal, WorldChangeProposal):
        return _blocked("", "blocked: proposal invalid",
                        BLOCKED_INVALID, "proposal not well-formed")
    if not isinstance(evidences, (list, tuple)):
        return _blocked(proposal.proposal_id, decision.decision,
                        BLOCKED_INVALID, "evidences not a sequence")
    if not isinstance(regression, dict):
        regression = {}
    structural = _proposal_problems(proposal)
    if structural is not None:
        return _blocked(proposal.proposal_id, decision.decision,
                        BLOCKED_INVALID, structural)
    if decision.candidate_ref != proposal.candidate_ref:
        return _blocked(proposal.proposal_id, decision.decision,
                        BLOCKED_INVALID,
                        "decision and proposal refer apart")
    if decision.decision != ACCEPTED:
        return _blocked(proposal.proposal_id, decision.decision,
                        BLOCKED_REVIEW,
                        f"review state is {decision.decision}")
    if decision.candidate_classification != CONFIRMED_MISMATCH:
        return _blocked(proposal.proposal_id, decision.decision,
                        BLOCKED_UNSUPPORTED,
                        "only confirmed mismatches ground proposals")
    model = _target_model(proposal)
    if model is None:
        return _blocked(proposal.proposal_id, decision.decision,
                        BLOCKED_INVALID, "unknown target type")
    if proposal.operation == WorldChangeOperation.ADD.value:
        if current_target is not None:
            return _blocked(proposal.proposal_id, decision.decision,
                            BLOCKED_INVALID, "target id already in use")
        if not absent_proof:
            return _blocked(proposal.proposal_id, decision.decision,
                            BLOCKED_INVALID,
                            "absence of target id not proven")
    else:
        if not isinstance(current_target, model):
            return _blocked(proposal.proposal_id, decision.decision,
                            BLOCKED_INVALID, "no current target view")
        if current_target != proposal.expected_before:
            return _blocked(proposal.proposal_id, decision.decision,
                            BLOCKED_INVALID,
                            "current target differs from expected content")

    groups, opposing, bad_entry = group_qualified_evidences(evidences)
    if bad_entry is not None:
        return _blocked(proposal.proposal_id, decision.decision,
                        BLOCKED_INVALID,
                        f"bad evidence entry {bad_entry!r}")
    if opposing:
        return _gate(proposal.proposal_id, decision.decision, (),
                     tuple(opposing), regression, False,
                     (f"opposing evidence blocks: "
                      f"{sorted(set(opposing))}",),
                     BLOCKED_EVIDENCE)
    qualifying = tuple(sorted(groups))
    if len(qualifying) < 2:
        return _gate(proposal.proposal_id, decision.decision, qualifying,
                     (), regression, False,
                     ("fewer than two independent groups",),
                     BLOCKED_EVIDENCE)

    reg_ok, reg_problem = _check_regression(proposal, regression)
    if not reg_ok:
        return _gate(proposal.proposal_id, decision.decision, qualifying,
                     (), regression, False, (reg_problem,),
                     BLOCKED_REGRESSION)
    return _gate(proposal.proposal_id, decision.decision, qualifying, (),
                 regression, True, (), READY)


def group_qualified_evidences(evidences) -> tuple:
    """Split evidences into ({group: [ids]}, [opposing ids], bad entry).

    Shared by the gate and the apply-time recheck so both enforce one
    independence policy. Unknown/unmarked groups, non-qualifying kinds
    and non-support directions never count as independent support.
    """
    groups: dict = {}
    opposing = []
    for ev in evidences:
        if not isinstance(ev, WorldReviewEvidence):
            return {}, [], ev
        if not ev.independence_group or not ev.scope_note:
            continue
        if ev.kind not in QUALIFYING_KINDS:
            continue
        if ev.direction == WorldEvidenceDirection.REFUTES.value:
            opposing.append(ev.evidence_id)
            continue
        if ev.direction != WorldEvidenceDirection.SUPPORTS.value:
            continue
        groups.setdefault(ev.independence_group, []).append(ev.evidence_id)
    return groups, opposing, None


def _check_regression(proposal: WorldChangeProposal,
                      regression: dict) -> tuple:
    for key in ("check_id", "scope", "source"):
        if not regression.get(key):
            return False, f"regression {key} missing"
    if regression.get("proposal_id") != proposal.proposal_id:
        return False, "regression linked to another proposal"
    if regression.get("passed") is not True:
        return False, "regression did not pass"
    return True, ""


def _gate(proposal_id: str, decision: str, groups: tuple,
          opposing: tuple, regression: dict, reg_ok: bool,
          blockers: tuple, status: str) -> WorldLearningGateResult:
    return WorldLearningGateResult(
        proposal_id=proposal_id, review_decision=decision,
        qualifying_groups=groups, opposing=opposing,
        regression_check=dict(regression), regression_ok=reg_ok,
        blockers=blockers, status=status,
        diagnostics=(f"proposal {proposal_id}",
                     f"groups={list(groups)}"))


def _blocked(proposal_id: str, decision: str, status: str,
             reason: str) -> WorldLearningGateResult:
    return WorldLearningGateResult(
        proposal_id=proposal_id, review_decision=decision,
        status=status, blockers=(reason,),
        diagnostics=(f"proposal {proposal_id}",))
