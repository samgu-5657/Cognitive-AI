"""Core data structures. Pure dataclasses, no IO, no reasoning."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Optional
import json


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConceptKind(str, Enum):
    ENTITY = "Entity"
    EVENT = "Event"
    ACTION = "Action"
    ATTRIBUTE = "Attribute"
    ABSTRACT = "AbstractConcept"


class KnowledgeStatus(str, Enum):
    VERIFIED = "verified"
    HYPOTHESIS = "hypothesis"
    CONTESTED = "contested"
    UNKNOWN = "unknown"


class QuestionStatus(str, Enum):
    PROPOSED = "proposed"
    QUEUED = "queued"
    RESEARCHING = "researching"
    AWAITING_EVIDENCE = "awaiting_evidence"
    LEARNED = "learned"
    INCONCLUSIVE = "inconclusive"
    REJECTED = "rejected"


class RuleStatus(str, Enum):
    """Lifecycle states for rows in the rules table.

    Only ACTIVE rules may fire in deterministic inference. CANDIDATE
    and VALIDATED rows are stored knowledge waiting for promotion;
    DISABLED / REJECTED rows stay in the database for audit but never
    participate. Transitions go through set_rule_status() so every
    change leaves a lifecycle event.
    """

    ACTIVE = "active"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    DISABLED = "disabled"
    REJECTED = "rejected"


@dataclass
class Concept:
    id: str
    name: str
    kind: str = ConceptKind.ENTITY
    parents: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = "seed"
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AttributeFact:
    concept_id: str
    key: str
    value: Any
    confidence: float = 0.8
    source: str = "observation"
    status: str = KnowledgeStatus.HYPOTHESIS


@dataclass
class RelationFact:
    src_id: str
    rel_type: str
    dst_id: str
    confidence: float = 0.8
    source: str = "observation"
    status: str = KnowledgeStatus.HYPOTHESIS


@dataclass
class Evidence:
    content: str
    source: str
    credibility: float = 0.5
    created_at: str = field(default_factory=utcnow)


@dataclass
class Claim:
    text: str
    subject_id: Optional[str] = None
    status: str = KnowledgeStatus.HYPOTHESIS
    confidence: float = 0.5
    evidence_ids: list[int] = field(default_factory=list)


@dataclass
class Experience:
    kind: str  # observation | inference | research | schema_change | evaluation
    summary: str
    detail: dict = field(default_factory=dict)
    prediction: dict = field(default_factory=dict)
    outcome: dict = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow)


@dataclass
class Hypothesis:
    text: str
    target_id: Optional[str] = None
    confidence: float = 0.5
    status: str = KnowledgeStatus.HYPOTHESIS


@dataclass
class ResearchQuestion:
    text: str
    kind: str = "gap"  # gap | conflict | unexplained | counterexample | link | improvement
    target_id: Optional[str] = None
    score: float = 0.0
    status: str = QuestionStatus.PROPOSED
    detail: dict = field(default_factory=dict)




@dataclass
class ParsedQuestion:
    """One compositionally parsed question (Phase M3D).

    qkind in category | attribute | capability | has | is_a_verify.
    is_yesno marks yes/no questions (answer 是/否 with evidence);
    open questions expect a value. expected carries the asked-about
    value for verification questions. Pure data, no IO.
    """

    raw: str = ""
    subject: str = ""
    subject_known: bool = False
    qkind: str = "unknown"
    key: str = ""
    expected: str = ""
    object: str = ""
    object_known: bool = False
    negated: bool = False
    modal: str | None = None
    is_yesno: bool = False
    unknown_spans: list = field(default_factory=list)
    ambiguity: str = ""
    spans: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PendingCandidate:
    """A not-yet-accepted learning candidate (Phase M3C).

    Stored in the pending_candidates table with status proposed |
    accepted | rejected. Pending rows are invisible to reasoning and
    never count as accepted facts.
    """

    id: int = 0
    kind: str = ""
    utterance: str = ""
    parse: dict = field(default_factory=dict)
    observation: dict = field(default_factory=dict)
    status: str = "proposed"
    reason: str = ""
    created_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GroundingVerdict:
    """Read-only grounding of one utterance against the KB (Phase M3B).

    verdict in supported | conflicted | unknown | ambiguous. Pure
    data, no IO. evidence entries carry kind/ref/status/detail so every
    judgment points at real rows, rules or traces.
    """

    sentence: str = ""
    parse: dict = field(default_factory=dict)
    subject_match: dict = field(default_factory=dict)
    object_match: dict = field(default_factory=dict)
    queries: list = field(default_factory=list)
    verdict: str = "unknown"
    evidence: list = field(default_factory=list)
    ambiguity: str = ""
    detail: str = ""
    question: dict = field(default_factory=dict)
    answer: str = ""
    is_yesno: bool = False
    limited: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class ParsedUtterance:
    """One compositionally parsed statement (Phase M3A).

    Pure data, no IO. The parser fills subject/predicate/object from
    independent pieces (known-concept lookup, function-word tokens);
    anything unrecognized stays in unknown_spans instead of being
    dropped or invented. An input the parser cannot structure has
    predicate "unknown".
    """

    raw: str = ""
    subject: str = ""
    subject_known: bool = False
    predicate: str = "unknown"  # is_a | has | capable | unspecified | unknown
    object: str = ""
    object_known: bool = False
    negated: bool = False
    modal: Optional[str] = None  # "can" for 可以/能够/能, else None
    unknown_spans: list = field(default_factory=list)
    ambiguity: str = ""
    spans: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SchemaChangeProposal:
    kind: str  # add_concept | add_attribute | add_relation | reparent | split_category | merge_categories | update_rule
    payload: dict = field(default_factory=dict)
    reason: str = ""
    evidence: list[dict] = field(default_factory=list)
    status: str = "proposed"  # proposed | applied | rejected | rolled_back
    version_before: Optional[int] = None
    version_after: Optional[int] = None


# ---- Unified world-model core (Phase M4-B). Pure data, no IO. ----
#
# These classes describe one shared semantic view over the legacy
# stores (concepts / attributes / relations / claims / evidences /
# rules). They never decide truth: WorldProposition carries no truth
# value, and WorldFactView keeps the stored knowledge status verbatim
# in a field separate from the proposition and from any query verdict.
#
# Deep immutability (Phase M4-C-FIX-01): every container entering
# these classes is converted at construction time -- dicts become
# read-only proxies (recursively), lists/sets become tuples/frozen
# sets, public sequences become tuples. Frozen dataclasses alone only
# block field rebinding; the conversion below also blocks mutation
# through nested references, in both directions (caller input and
# public-field access share no mutable object).


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {k: _freeze_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(v) for v in value)
    return value


class WorldObjectKind(str, Enum):
    INDIVIDUAL = "individual"
    CLASS = "class"
    LITERAL = "literal"
    EVENT = "event"
    STATE = "state"
    LEGACY_CONCEPT = "legacy_concept"


class WorldPolarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class WorldModality(str, Enum):
    ASSERTED = "asserted"
    CAPABLE = "capable"
    POSSIBLE = "possible"
    UNKNOWN = "unknown"


class WorldEvidenceDirection(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class WorldObjectRef:
    """Stable reference to one world-model object.

    Legacy concept rows always map to LEGACY_CONCEPT: the old table
    cannot reliably tell individuals from classes, and names must not
    be used to guess. EVENT / STATE exist only as future kinds.
    """

    ref_id: str = ""
    name: str = ""
    kind: str = WorldObjectKind.LEGACY_CONCEPT


@dataclass(frozen=True)
class WorldTerm:
    """One end of a proposition. Exactly one slot is active.

    ref: resolved object reference. literal (+has_literal): a value
    kept as-is. unresolved: raw text kept verbatim when resolution was
    not possible. missing: True only when lookup ran and proved the
    referenced object absent. unresolved and missing are never merged:
    the first means "kept, not resolved", the second "looked up, gone".
    """

    ref: Optional[WorldObjectRef] = None
    literal: Any = None
    has_literal: bool = False
    unresolved: str = ""
    missing: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "literal", _freeze_value(self.literal))


@dataclass(frozen=True)
class WorldProposition:
    """One proposition: subject / predicate / object + polarity/modality.

    Carries no truth value. Negation/capability marks come from parsed
    input or the codebase's own function-key convention; anything the
    old record does not say stays UNKNOWN. raw keeps the original
    fields (attribute key/value, relation rel_type/dst_id) so every
    semantic field traces back to stored data.
    """

    subject: WorldTerm = field(default_factory=WorldTerm)
    predicate: str = ""
    object: WorldTerm = field(default_factory=WorldTerm)
    polarity: str = WorldPolarity.UNKNOWN
    modality: str = WorldModality.UNKNOWN
    representation_origin: str = ""  # e.g. attribute | relation
    source_ref: str = ""  # e.g. attribute:12 | relation:5
    raw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", _freeze_value(dict(self.raw)))


@dataclass(frozen=True)
class WorldEvidenceRef:
    """Reference to one stored evidence row.

    claim_ref uses the evidences.claim_id foreign key when present.
    direction stays UNKNOWN unless the stored data explicitly states
    support/refutation (the current schema never does).
    """

    evidence_id: int = 0
    source: str = ""
    claim_ref: str = ""
    direction: str = WorldEvidenceDirection.UNKNOWN


@dataclass(frozen=True)
class WorldFactView:
    """One stored fact read as a world-model view.

    status is the stored knowledge status verbatim (verified /
    hypothesis / contested / ...), independent of the proposition and
    of any query verdict vocabulary. derivation is only filled from
    real, traceable derivation records, never invented.
    """

    proposition: WorldProposition = field(
        default_factory=WorldProposition)
    record_id: str = ""
    status: str = ""
    evidence: tuple = ()
    derivation: Optional[dict] = None
    store: str = ""  # e.g. attribute | relation

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if self.derivation is not None:
            object.__setattr__(
                self, "derivation", _freeze_value(dict(self.derivation)))


@dataclass(frozen=True)
class WorldRuleView:
    """Read-only view of one stored rule (stable reference, no firing)."""

    rule_id: str = ""
    name: str = ""
    status: str = ""
    source: str = ""
    conditions: tuple = ()
    conclusion: dict = field(default_factory=dict)
    exceptions: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "conditions", tuple(_freeze_value(c)
                                      for c in self.conditions))
        object.__setattr__(
            self, "conclusion", _freeze_value(dict(self.conclusion)))
        object.__setattr__(
            self, "exceptions", tuple(_freeze_value(e)
                                      for e in self.exceptions))


@dataclass(frozen=True)
class WorldClaimView:
    """One stored claim, text kept verbatim (never re-structured).

    subject_ref uses the claims.subject_id link when it resolves;
    evidence uses the evidences.claim_id link. There is no reliable
    stored link from a claim to an attribute/relation row, so none is
    fabricated here.
    """

    claim_id: int = 0
    text: str = ""
    status: str = ""
    subject_ref: str = ""
    evidence: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))


@dataclass(frozen=True)
class WorldSnapshot:
    """One immutable read of objects + facts + rules.

    unmapped holds entries the adapter read but could not map
    ({kind, ref, reason}); nothing is dropped silently.
    """

    objects: tuple = ()
    facts: tuple = ()
    rules: tuple = ()
    claims: tuple = ()
    unmapped: tuple = ()

    def __post_init__(self) -> None:
        for name in ("objects", "facts", "rules", "claims", "unmapped"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def filter_facts(self, subject: str = "", predicate: str = "",
                     store: str = "") -> tuple:
        """Filter facts by subject ref_id, predicate and store type."""
        out = []
        for f in self.facts:
            prop = f.proposition
            subj = prop.subject.ref.ref_id \
                if prop.subject.ref is not None else ""
            if subject and subj != subject:
                continue
            if predicate and prop.predicate != predicate:
                continue
            if store and f.store != store:
                continue
            out.append(f)
        return tuple(out)


# ---- Dynamic world-model state (Phase M4-C). Pure data + local checks.
#
# Static knowledge (M4-B views) and dynamic state are kept apart: only
# explicitly supplied state facts enter a WorldState, never arbitrary
# attribute rows. States are immutable snapshots; simulation builds
# new states and never touches the database.


class WorldTransitionOutcome(str, Enum):
    APPLIED = "applied"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    CONFLICTED = "conflicted"
    INVALID = "invalid"


@dataclass(frozen=True)
class WorldPatternTerm:
    """One parameter slot inside an action pattern.

    Exactly one slot is active: a named ``variable`` to be bound, a
    fixed ``ref``, or a fixed ``literal`` (+has_literal). Variables are
    explicit names (no "?"), never guessed from strings.
    """

    variable: str = ""
    ref: Optional[WorldObjectRef] = None
    literal: Any = None
    has_literal: bool = False

    def is_variable(self) -> bool:
        return bool(self.variable)

    def __post_init__(self) -> None:
        object.__setattr__(self, "literal", _freeze_value(self.literal))


@dataclass(frozen=True)
class WorldLiteralPattern:
    """One premise/effect pattern: subject/predicate/object + polarity.

    Polarity must be explicitly POSITIVE or NEGATIVE; UNKNOWN is never
    a usable premise or effect and is rejected at construction.
    """

    subject: WorldPatternTerm = field(default_factory=WorldPatternTerm)
    predicate: str = ""
    object: WorldPatternTerm = field(default_factory=WorldPatternTerm)
    polarity: str = WorldPolarity.POSITIVE

    def __post_init__(self) -> None:
        if self.polarity not in (WorldPolarity.POSITIVE,
                                 WorldPolarity.NEGATIVE):
            raise ValueError(
                f"pattern polarity must be positive/negative, "
                f"got {self.polarity!r}")


@dataclass(frozen=True)
class WorldStateFact:
    """One explicit state proposition inside a snapshot.

    The proposition must carry definite POSITIVE or NEGATIVE polarity:
    unknown polarity is never a known state fact.
    """

    proposition: WorldProposition = field(
        default_factory=WorldProposition)
    event_ref: str = ""
    origin: str = ""

    def __post_init__(self) -> None:
        if self.proposition.polarity not in (WorldPolarity.POSITIVE,
                                             WorldPolarity.NEGATIVE):
            raise ValueError(
                "state facts need definite polarity, got "
                f"{self.proposition.polarity!r}")


@dataclass(frozen=True)
class WorldState:
    """One immutable world-state snapshot.

    Same-polarity duplicates are dropped and facts are deterministically
    ordered at construction, so equal content always builds equal
    states. Construction never mutates its inputs.
    """

    state_id: str = ""
    step_index: int = 0
    facts: tuple = ()
    parent_state_id: str = ""

    def __post_init__(self) -> None:
        seen: set = set()
        ordered: list = []
        for f in self.facts:
            key = (repr(_term_key(f.proposition.subject)),
                   f.proposition.predicate,
                   repr(_term_key(f.proposition.object)),
                   str(f.proposition.polarity))
            if key in seen:
                continue
            seen.add(key)
            ordered.append(f)
        ordered.sort(key=lambda f: (
            repr(_term_key(f.proposition.subject)),
            f.proposition.predicate,
            repr(_term_key(f.proposition.object)),
            str(f.proposition.polarity)))
        object.__setattr__(self, "facts", tuple(ordered))


def _term_key(term: WorldTerm) -> tuple:
    if term.ref is not None:
        return ("ref", term.ref.ref_id)
    if term.has_literal:
        return ("lit", repr(term.literal))
    return ("unres", term.unresolved, term.missing)


@dataclass(frozen=True)
class WorldEffect:
    """One state-change effect: assert a definite polarity.

    Only definite assertion is supported: no arithmetic, no continuous
    simulation, no callbacks. The pattern polarity is the polarity
    being set.
    """

    target: WorldLiteralPattern = field(
        default_factory=WorldLiteralPattern)


@dataclass(frozen=True)
class WorldActionSchema:
    """One parameterized action rule: params + premises + effects."""

    action_id: str = ""
    name: str = ""
    params: tuple = ()
    premises: tuple = ()
    effects: tuple = ()

    def __post_init__(self) -> None:
        for name in ("params", "premises", "effects"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class WorldPreconditionCheck:
    """Result of checking one premise: outcome + matched facts + reason.

    outcome is one of true | false | unknown | conflicted | invalid.
    A missing state record is reported as unknown, never as false.
    """

    premise: WorldLiteralPattern = field(
        default_factory=WorldLiteralPattern)
    outcome: str = "unknown"
    matched: tuple = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched", tuple(self.matched))


@dataclass(frozen=True)
class WorldEvent:
    """One successful simulated action. Identifiable as simulation.

    No wall-clock timestamps: logical step + state links make the event
    reproducible. Simulation results are never real observations.
    """

    event_id: str = ""
    action_id: str = ""
    bindings: tuple = ()
    from_state: str = ""
    to_state: str = ""
    step_index: int = 0
    effects: tuple = ()
    simulated: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "bindings",
            tuple((p[0], p[1]) for p in self.bindings))
        object.__setattr__(self, "effects", tuple(self.effects))


@dataclass(frozen=True)
class WorldTransitionResult:
    """Outcome of one simulation attempt (see WorldTransitionOutcome)."""

    outcome: str = WorldTransitionOutcome.INVALID
    checks: tuple = ()
    reason: str = ""
    new_state: Optional[WorldState] = None
    event: Optional[WorldEvent] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))


# ---- Forecast & comparison (Phase M4-F). Pure data, no IO. ----
#
# A forecast says what a given action model derives in a simulated
# state; a comparison says whether an independent reference state
# derives the same. Neither ever claims reality verification.


class WorldComparisonOutcome(str, Enum):
    CONSISTENT = "consistent"
    MISMATCH = "mismatch"
    REFERENCE_UNKNOWN = "reference_unknown"
    CONFLICTED = "conflicted"
    LIMITED = "limited"
    INCONCLUSIVE = "inconclusive"
    INVALID = "invalid"


@dataclass(frozen=True)
class WorldForecast:
    """One action forecast with per-query reasoning on the new state."""

    start_state_id: str = ""
    action_id: str = ""
    bindings: tuple = ()
    transition: WorldTransitionResult = field(
        default_factory=WorldTransitionResult)
    predicted_state: Optional[WorldState] = None
    queries: tuple = ()
    results: tuple = ()
    complete: bool = False
    diagnostics: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "bindings",
            tuple((p[0], p[1]) for p in self.bindings))
        for name in ("queries", "results", "diagnostics"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class WorldPredictionComparisonItem:
    """One query compared between forecast and reference reasoning."""

    query: WorldLiteralPattern = field(
        default_factory=WorldLiteralPattern)
    forecast_verdict: str = ""
    reference_verdict: str = ""
    category: str = ""
    detail: str = ""
    forecast_proofs: tuple = ()
    reference_proofs: tuple = ()

    def __post_init__(self) -> None:
        for name in ("forecast_proofs", "reference_proofs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class WorldPredictionComparison:
    """Whole-forecast comparison. Items are never aggregated away."""

    start_state_id: str = ""
    forecast_state_id: str = ""
    reference_state_id: str = ""
    items: tuple = ()
    overall: str = ""
    complete: bool = False
    diagnostics: tuple = ()

    def __post_init__(self) -> None:
        for name in ("items", "diagnostics"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


# ---- Review & diagnosis (Phase M4-G). Pure data, no IO. ----
#
# A review never decides a root cause: confirmed_mismatch means both
# sides derived opposite, complete, independently verified conclusions.
# Hypotheses are review directions (is_hypothesis=True always), never
# confirmed faults, scores, or learning orders.


class WorldReviewClassification(str, Enum):
    CONSISTENT = "consistent"
    CONFIRMED_MISMATCH = "confirmed_mismatch"
    REFERENCE_UNKNOWN = "reference_unknown"
    FORECAST_INCONCLUSIVE = "forecast_inconclusive"
    CONFLICTED = "conflicted"
    LIMITED = "limited"
    INVALID = "invalid"


class WorldHypothesisKind(str, Enum):
    ACTION_MODEL_MAY_BE_WRONG = "action_model_may_be_wrong"
    INFERENCE_RULE_MAY_BE_WRONG_OR_MISSING = \
        "inference_rule_may_be_wrong_or_missing"
    UNMODELED_WORLD_CHANGE = "unmodeled_world_change"
    REFERENCE_SCOPE_OR_PROVENANCE_UNCERTAIN = \
        "reference_scope_or_provenance_uncertain"
    INSUFFICIENT_INFORMATION = "insufficient_information"


@dataclass(frozen=True)
class WorldDiagnosticHypothesis:
    """One possible explanation direction, explicitly not a verdict."""

    kind: str = ""
    statement: str = ""
    is_hypothesis: bool = True
    refs: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "refs", tuple(self.refs))


@dataclass(frozen=True)
class WorldDiagnosticCandidate:
    """One pending-review difference candidate. Never a confirmed fault:
    no confirmed rule id, no prescribed fix, no auto-confirm flag."""

    query: WorldLiteralPattern = field(
        default_factory=WorldLiteralPattern)
    forecast_verdict: str = ""
    reference_verdict: str = ""
    comparison_category: str = ""
    classification: str = ""
    hypotheses: tuple = ()
    forecast_proof_refs: tuple = ()
    reference_proof_refs: tuple = ()
    action_id: str = ""
    rule_ids: tuple = ()
    state_ids: tuple = ()
    reason: str = ""
    status: str = "pending_review"
    simulated: bool = True

    def __post_init__(self) -> None:
        for name in ("hypotheses", "forecast_proof_refs",
                     "reference_proof_refs", "rule_ids", "state_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class WorldReviewItemResult:
    """Per-query diagnosis inside a review report."""

    query: WorldLiteralPattern = field(
        default_factory=WorldLiteralPattern)
    classification: str = ""
    detail: str = ""
    candidate: Optional[WorldDiagnosticCandidate] = None


@dataclass(frozen=True)
class WorldReviewReport:
    """Whole-forecast review: items, candidates, overall state."""

    start_state_id: str = ""
    forecast_state_id: str = ""
    reference_state_id: str = ""
    items: tuple = ()
    candidates: tuple = ()
    overall: str = ""
    complete: bool = False
    limitations: tuple = ()

    def __post_init__(self) -> None:
        for name in ("items", "candidates", "limitations"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


# ---- Review audit & learning gate (Phase M4-H). Pure data, no IO. ----
#
# Three strictly separated concerns: a review decision only accepts
# further investigation; a change proposal pins exact before/after
# contents; a gate result only reports whether the evidence,
# regression and target checks pass. Nothing here writes knowledge,
# and ready_for_application never means verified.


class WorldReviewDecisionKind(str, Enum):
    ACCEPTED_FOR_INVESTIGATION = "accepted_for_investigation"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"


class WorldEvidenceKind(str, Enum):
    INDEPENDENT_OBSERVATION = "independent_observation"
    USER_ASSERTION = "user_assertion"
    DERIVED = "derived"
    SIMULATED = "simulated"
    UNKNOWN = "unknown"


class WorldChangeTarget(str, Enum):
    ACTION_SCHEMA = "action_schema"
    INFERENCE_RULE = "inference_rule"


class WorldChangeOperation(str, Enum):
    ADD = "add"
    REVISE = "revise"
    DISABLE = "disable"


class WorldGateStatus(str, Enum):
    READY_FOR_APPLICATION = "ready_for_application"
    BLOCKED_INSUFFICIENT_EVIDENCE = "blocked_insufficient_evidence"
    BLOCKED_REVIEW = "blocked_review"
    BLOCKED_REGRESSION = "blocked_regression"
    BLOCKED_INVALID_PROPOSAL = "blocked_invalid_proposal"
    BLOCKED_UNSUPPORTED_CHANGE = "blocked_unsupported_change"


@dataclass(frozen=True)
class WorldReviewDecision:
    """Audit of one diagnostic candidate. Accepted means only that
    further investigation is worthwhile, never that a hypothesis holds."""

    candidate_ref: str = ""
    decision: str = ""
    candidate_classification: str = ""
    reviewer: str = ""
    reason: str = ""
    simulated: bool = True

    def __post_init__(self) -> None:
        if self.decision not in tuple(k.value for k in
                                      WorldReviewDecisionKind):
            raise ValueError(f"bad review decision {self.decision!r}")
        if not self.candidate_ref:
            raise ValueError("candidate reference missing")
        if not self.reviewer:
            raise ValueError("reviewer identity required")
        if not self.reason:
            raise ValueError("review reason required")
        if self.simulated is not True:
            raise ValueError("only simulated-source candidates audited")


@dataclass(frozen=True)
class WorldReviewEvidence:
    """One evidence statement for a concrete change proposal."""

    evidence_id: str = ""
    source_id: str = ""
    independence_group: str = ""
    kind: str = ""
    direction: str = ""
    proposition_ref: str = ""
    scope_note: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.source_id:
            raise ValueError("evidence needs id and source id")
        if self.kind not in tuple(k.value for k in WorldEvidenceKind):
            raise ValueError(f"bad evidence kind {self.kind!r}")
        if self.direction not in tuple(k.value for k in
                                       WorldEvidenceDirection):
            raise ValueError(f"bad evidence direction {self.direction!r}")
        if not self.proposition_ref:
            raise ValueError("evidence must name its proposition")


@dataclass(frozen=True)
class WorldChangeProposal:
    """One concrete, reviewable model change (never applied here)."""

    proposal_id: str = ""
    candidate_ref: str = ""
    target_type: str = ""
    target_id: str = ""
    operation: str = ""
    expected_before: Any = None
    proposed_after: Any = None
    reason: str = ""
    regression_note: str = ""
    status: str = "pending_review"
    simulated: bool = True

    def __post_init__(self) -> None:
        if self.operation not in tuple(k.value for k in
                                       WorldChangeOperation):
            raise ValueError(f"bad operation {self.operation!r}")
        if self.target_type not in tuple(k.value for k in
                                         WorldChangeTarget):
            raise ValueError(f"bad target type {self.target_type!r}")
        if not self.proposal_id or not self.candidate_ref or \
                not self.target_id:
            raise ValueError("proposal/candidate/target ids required")
        if not self.reason or not self.regression_note:
            raise ValueError("reason and regression note required")
        if self.status != "pending_review":
            raise ValueError("this module never applies proposals")
        if self.simulated is not True:
            raise ValueError("only simulated-source proposals handled")
        want = (WorldActionSchema
                if self.target_type ==
                WorldChangeTarget.ACTION_SCHEMA.value
                else WorldInferenceRule)
        if self.operation == WorldChangeOperation.ADD.value:
            if self.expected_before is not None:
                raise ValueError("add must not carry expected content")
            if not isinstance(self.proposed_after, want):
                raise ValueError("add needs complete new content")
        elif self.operation == WorldChangeOperation.REVISE.value:
            if not isinstance(self.expected_before, want) or \
                    not isinstance(self.proposed_after, want):
                raise ValueError("revise needs full before/after content")
        elif self.operation == WorldChangeOperation.DISABLE.value:
            if not isinstance(self.expected_before, want):
                raise ValueError("disable needs the expected content")
            if self.proposed_after is not None:
                raise ValueError("disable carries no new content")


@dataclass(frozen=True)
class WorldLearningGateResult:
    """Outcome of one qualification check (never an application)."""

    proposal_id: str = ""
    review_decision: str = ""
    qualifying_groups: tuple = ()
    opposing: tuple = ()
    regression_check: dict = field(default_factory=dict)
    regression_ok: bool = False
    blockers: tuple = ()
    status: str = ""
    diagnostics: tuple = ()

    def __post_init__(self) -> None:
        for name in ("qualifying_groups", "opposing", "blockers",
                     "diagnostics"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "regression_check",
                           _freeze_value(dict(self.regression_check)))


# ---- Controlled application (Phase M4-I). Pure data, no IO. ----
#
# Authorization binds one proposal's exact content, its gate result
# and its expected target: any drift voids it. Application results
# never imply knowledge verification; applied_verified additionally
# needs a post-apply validation record tied to the written version.


class WorldApplicationStatus(str, Enum):
    APPLIED_PENDING_REGRESSION = "applied_pending_regression"
    APPLIED_VERIFIED = "applied_verified"
    BLOCKED_AUTHORIZATION = "blocked_authorization"
    BLOCKED_GATE = "blocked_gate"
    BLOCKED_STALE_TARGET = "blocked_stale_target"
    BLOCKED_UNSUPPORTED_CHANGE = "blocked_unsupported_change"
    BLOCKED_INVALID_PROPOSAL = "blocked_invalid_proposal"
    FAILED_ATOMIC_WRITE = "failed_atomic_write"
    COMPENSATION_REQUIRED = "compensation_required"


@dataclass(frozen=True)
class WorldApplicationAuthorization:
    """Independent authorization to apply one exact proposal.

    Binds proposal content, gate result and expected target by
    fingerprint. Investigation acceptance and gate readiness never
    substitute for this authorization.
    """

    auth_id: str = ""
    proposal_id: str = ""
    proposal_fingerprint: str = ""
    gate_fingerprint: str = ""
    authorized_by: str = ""
    reason: str = ""
    operation: str = ""
    target_id: str = ""
    expected_target_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.auth_id or not self.proposal_id:
            raise ValueError("authorization and proposal ids required")
        if not self.proposal_fingerprint or not self.gate_fingerprint \
                or not self.expected_target_fingerprint:
            raise ValueError("authorization fingerprints required")
        if not self.authorized_by or not self.reason:
            raise ValueError("authorizer identity and reason required")
        if self.operation not in tuple(k.value for k in
                                       WorldChangeOperation):
            raise ValueError(f"bad operation {self.operation!r}")
        if not self.target_id:
            raise ValueError("authorization target required")


@dataclass(frozen=True)
class WorldChangeApplicationResult:
    """Outcome of one application attempt. Pending until a linked
    post-apply validation passes; never a knowledge credibility claim."""

    proposal_id: str = ""
    target_id: str = ""
    operation: str = ""
    before_ref: str = ""
    after_ref: str = ""
    status: str = ""
    audit_refs: tuple = ()
    reason: str = ""
    needs_regression_followup: bool = False
    expected_fingerprint: str = ""
    applied_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.status not in tuple(k.value for k in
                                    WorldApplicationStatus):
            raise ValueError(f"bad application status {self.status!r}")
        object.__setattr__(self, "audit_refs", tuple(self.audit_refs))


@dataclass(frozen=True)
class WorldPostApplyValidation:
    """Validation record for an applied change. Pass counts only with
    a report reference, a test scope and a fingerprint of the version
    actually written."""

    validation_id: str = ""
    proposal_id: str = ""
    target_ref: str = ""
    target_fingerprint: str = ""
    scope: str = ""
    passed: bool = False
    report_ref: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if not self.validation_id or not self.proposal_id or \
                not self.target_ref:
            raise ValueError("validation/proposal/target refs required")


# ---- Batch knowledge import (Phase M4-J). Pure data, no IO. ----
#
# Preview categories describe import handling only: they are neither
# knowledge credibility states nor candidate audit states. Source
# references are kept verbatim; a shared file group never counts as
# several independent sources.


class KnowledgeImportCategory(str, Enum):
    READY_FOR_CANDIDATE = "ready_for_candidate"
    ALREADY_KNOWN = "already_known"
    DUPLICATE_IN_BATCH = "duplicate_in_batch"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    CONFLICTED = "conflicted"
    UNKNOWN = "unknown"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class KnowledgeImportItem:
    """One input line: source, text, parse/grounding outcome, handling."""

    line_no: int = 0
    text: str = ""
    source_id: str = ""
    source_group: str = ""
    category: str = ""
    parse: dict = field(default_factory=dict)
    grounding: dict = field(default_factory=dict)
    reason: str = ""
    candidate_id: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "parse",
                           _freeze_value(dict(self.parse)))
        object.__setattr__(self, "grounding",
                           _freeze_value(dict(self.grounding)))


@dataclass(frozen=True)
class KnowledgeImportPreview:
    """Whole-file preview. Read-only; creating it writes nothing."""

    file_id: str = ""
    format: str = ""
    total_lines: int = 0
    counts: dict = field(default_factory=dict)
    items: tuple = ()
    error: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "counts",
                           _freeze_value(dict(self.counts)))


# ---- Compositional semantics (Phase M5-A). Pure data, no IO. ----
#
# A SemanticExpression composes existing atomic parses into ordered
# trees. Completeness is computed from the real parse: a result is
# complete only when its tree exists and holds no unsupported node.


class SemanticExpressionKind(str, Enum):
    ATOM = "atom"
    CONJUNCTION = "conjunction"
    CONDITIONAL = "conditional"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SemanticExpression:
    """One node of a semantic tree.

    atom: holds a real ParsedUtterance dict, no children.
    conjunction: >= 2 ordered children, connective recorded.
    conditional: exactly [antecedent, consequent], roles recorded.
    unsupported: raw text plus a reason, nothing silently dropped.
    """

    kind: str = ""
    raw_text: str = ""
    atom: dict = field(default_factory=dict)
    children: tuple = ()
    connective: str = ""
    roles: tuple = ()
    diagnosis: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(self, "roles", tuple(self.roles))
        object.__setattr__(self, "atom",
                           _freeze_value(dict(self.atom)))
        kind = self.kind
        kinds = tuple(k.value for k in SemanticExpressionKind)
        if kind not in kinds:
            raise ValueError(f"bad expression kind {kind!r}")
        if kind == SemanticExpressionKind.ATOM.value:
            if not self.atom or self.children or self.connective \
                    or self.roles or self.diagnosis:
                raise ValueError("atom nodes carry only an atom parse")
        elif kind == SemanticExpressionKind.CONJUNCTION.value:
            if len(self.children) < 2 or self.atom or self.diagnosis:
                raise ValueError("conjunctions need >= 2 children")
        elif kind == SemanticExpressionKind.CONDITIONAL.value:
            if len(self.children) != 2 or \
                    tuple(self.roles) != ("antecedent", "consequent") \
                    or self.atom or self.diagnosis:
                raise ValueError("conditionals need antecedent+consequent")
        else:
            if not self.diagnosis or not self.raw_text or self.children \
                    or self.atom:
                raise ValueError("unsupported nodes keep text + reason")


def _expression_supported(node) -> bool:
    if node.kind == SemanticExpressionKind.UNSUPPORTED.value:
        return False
    return all(_expression_supported(c) for c in node.children)


def _count_kinds(node) -> tuple:
    atoms, composite = 0, 0
    if node.kind == SemanticExpressionKind.ATOM.value:
        atoms = 1
    elif node.kind in (SemanticExpressionKind.CONJUNCTION.value,
                       SemanticExpressionKind.CONDITIONAL.value):
        composite = 1
    for child in node.children:
        sub_atoms, sub_composite = _count_kinds(child)
        atoms += sub_atoms
        composite += sub_composite
    return atoms, composite


@dataclass(frozen=True)
class SemanticParseResult:
    """Whole-statement parse: tree (if any), real completeness, counts."""

    raw_input: str = ""
    tree: Optional[SemanticExpression] = None
    complete: bool = False
    diagnostics: tuple = ()
    atom_count: int = 0
    composite_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics",
                           tuple(self.diagnostics))
        if self.tree is None:
            if self.complete:
                raise ValueError("no tree cannot be complete")
            return
        atoms, composite = _count_kinds(self.tree)
        if self.atom_count != atoms or \
                self.composite_count != composite:
            raise ValueError("counts must match the real tree")
        real = _expression_supported(self.tree)
        if self.complete != real:
            raise ValueError("completeness must follow the real parse")


# ---- Compositional understanding (Phase M5-B). Pure data, no IO. ----
#
# SemanticNodeStatus classifies how one tree node relates to current
# knowledge. It is neither a knowledge credibility state nor a query
# verdict: unknown/conflicted here are complete judgments, while
# failure/ambiguity/unsupported/limits mark incomplete processing.


class SemanticNodeStatus(str, Enum):
    SUPPORTED = "supported"
    CONFLICTED = "conflicted"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    FAILURE = "failure"
    CONDITIONAL_UNASSERTED = "conditional_unasserted"


_INCOMPLETE_NODE_STATUSES = ("failure", "unsupported", "ambiguous")


@dataclass(frozen=True)
class SemanticNodeEvaluation:
    """Analysis of one semantic-tree node.

    Atom nodes carry the real grounding dict; conjunctions keep every
    child in order; conditionals keep antecedent/consequent roles;
    unsupported nodes keep text plus reason. limited marks an explicit
    query/engine restriction carried by the node.
    """

    kind: str = ""
    raw_text: str = ""
    status: str = ""
    grounding: dict = field(default_factory=dict)
    children: tuple = ()
    roles: tuple = ()
    diagnostics: tuple = ()
    source_refs: tuple = ()
    limited: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(self, "roles", tuple(self.roles))
        object.__setattr__(self, "diagnostics",
                           tuple(self.diagnostics))
        object.__setattr__(self, "source_refs",
                           tuple(self.source_refs))
        object.__setattr__(self, "grounding",
                           _freeze_value(dict(self.grounding)))
        statuses = tuple(s.value for s in SemanticNodeStatus)
        if self.status not in statuses:
            raise ValueError(f"bad node status {self.status!r}")
        kinds = tuple(k.value for k in SemanticExpressionKind)
        if self.kind not in kinds:
            raise ValueError(f"bad node kind {self.kind!r}")
        atom = SemanticExpressionKind.ATOM.value
        conj = SemanticExpressionKind.CONJUNCTION.value
        cond = SemanticExpressionKind.CONDITIONAL.value
        unsup = SemanticExpressionKind.UNSUPPORTED.value
        if self.kind == atom and not self.grounding:
            raise ValueError("atom evaluations need real grounding")
        if self.kind == conj and not self.children:
            raise ValueError("conjunction evaluations keep children")
        if self.kind == cond and (
                len(self.children) != 2 or
                tuple(self.roles) != ("antecedent", "consequent")):
            raise ValueError("conditional evaluations keep both roles")
        if self.kind == unsup and not self.diagnostics:
            raise ValueError("unsupported evaluations keep the reason")


def _evaluation_ok(node) -> bool:
    if node.status in _INCOMPLETE_NODE_STATUSES or node.limited:
        return False
    return all(_evaluation_ok(c) for c in node.children)


def _evaluation_counts(node) -> dict:
    counts = {s.value: 0 for s in SemanticNodeStatus}
    counts[node.status] += 1
    for child in node.children:
        for key, value in _evaluation_counts(child).items():
            counts[key] += value
    return counts


@dataclass(frozen=True)
class CompositionalUnderstanding:
    """Whole-sentence understanding report.

    complete means fully structured and fully queried: unknown and
    conflicted verdicts are complete outcomes, while failure,
    ambiguity, unsupported spans and recorded limits are not.
    """

    raw_input: str = ""
    parse: Optional[SemanticParseResult] = None
    root: Optional[SemanticNodeEvaluation] = None
    grounded_atoms: int = 0
    counts: dict = field(default_factory=dict)
    complete: bool = False
    diagnostics: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts",
                           _freeze_value(dict(self.counts)))
        object.__setattr__(self, "diagnostics",
                           tuple(self.diagnostics))
        if self.root is None:
            if self.complete:
                raise ValueError("no evaluation cannot be complete")
            return
        if self.parse is not None and not self.parse.complete and \
                self.complete:
            raise ValueError("unparsed input cannot be complete")
        if self.counts != _evaluation_counts(self.root):
            raise ValueError("counts must match the evaluation tree")
        if self.complete != _evaluation_ok(self.root):
            raise ValueError("completeness must follow the evaluation")


# ---- Explicit semantic context (Phase M5-C). Pure data, no IO. ----
#
# A context only records what was mentioned and how it resolved: it
# never confirms knowledge, and legacy_concept refs are never silently
# upgraded to individuals.


class ReferenceResolutionStatus(str, Enum):
    NOT_REFERENCE = "not_reference"
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


@dataclass(frozen=True)
class SemanticContextMention:
    """One registered object mention. The ref keeps its real kind:
    legacy_concept stays legacy_concept here."""

    object_ref: Optional[WorldObjectRef] = None
    mention_text: str = ""
    role: str = "subject"
    source_statement: str = ""
    order: int = 0
    source_ref: str = ""

    def __post_init__(self) -> None:
        if self.object_ref is None:
            raise ValueError("mention needs an object reference")


@dataclass(frozen=True)
class SemanticContext:
    """Caller-managed explicit context. Appends return new contexts;
    trimming keeps the newest mentions and records that it happened."""

    mentions: tuple = ()
    sequence: int = 0
    trimmed: bool = False
    diagnostics: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "mentions", tuple(self.mentions))
        object.__setattr__(self, "diagnostics",
                           tuple(self.diagnostics))


@dataclass(frozen=True)
class SemanticReferenceResolution:
    """One reference attempt. resolved carries a ref only when the
    status is resolved; failed attempts keep candidates and reasons."""

    mention: str = ""
    status: str = ""
    resolved_ref: Optional[WorldObjectRef] = None
    candidates: tuple = ()
    source_ref: str = ""
    reason: str = ""
    diagnostics: tuple = ()

    def __post_init__(self) -> None:
        statuses = tuple(s.value for s in ReferenceResolutionStatus)
        if self.status not in statuses:
            raise ValueError(f"bad resolution status {self.status!r}")
        object.__setattr__(self, "candidates",
                           tuple(self.candidates))
        object.__setattr__(self, "diagnostics",
                           tuple(self.diagnostics))
        if self.status == ReferenceResolutionStatus.RESOLVED.value:
            if self.resolved_ref is None:
                raise ValueError("resolved needs a reference")
        elif self.resolved_ref is not None:
            raise ValueError("only resolved carries a reference")


@dataclass(frozen=True)
class ContextualUnderstanding:
    """Understanding with explicit context. understanding is absent
    (or incomplete) whenever a reference could not be safely handled."""

    raw_input: str = ""
    context_seq: int = 0
    resolutions: tuple = ()
    normalized_text: str = ""
    understanding: Optional[CompositionalUnderstanding] = None
    complete: bool = False
    diagnostics: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolutions",
                           tuple(self.resolutions))
        object.__setattr__(self, "diagnostics",
                           tuple(self.diagnostics))
        if self.understanding is None and self.complete:
            raise ValueError("no understanding cannot be complete")


# ---- Clarification interaction (Phase M5-D). Pure data, no IO. ----
#
# A clarification asks the user to pick among real candidates; an
# answer only records the pick. Neither asserts knowledge, and option
# ids are deterministic content references, never list positions.


class SemanticClarificationKind(str, Enum):
    AMBIGUOUS_REFERENCE = "ambiguous_reference"
    MISSING_ANTECEDENT = "missing_antecedent"
    UNSUPPORTED_IDENTITY = "unsupported_identity"
    UNSUPPORTED_EXPRESSION = "unsupported_expression"


class SemanticInteractionState(str, Enum):
    UNDERSTOOD = "understood"
    NEEDS_CLARIFICATION = "needs_clarification"
    CLARIFICATION_RESOLVED = "clarification_resolved"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


@dataclass(frozen=True)
class SemanticClarificationOption:
    """One selectable reading. Unselectable options keep their reason."""

    option_id: str = ""
    ref: Optional[WorldObjectRef] = None
    display_name: str = ""
    mention_ref: str = ""
    selectable: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.option_id:
            raise ValueError("option needs a stable id")
        if self.selectable and self.ref is None:
            raise ValueError("selectable options need a reference")


@dataclass(frozen=True)
class SemanticClarificationRequest:
    """One question to the user. IDs are deterministic content hashes."""

    request_id: str = ""
    raw_input: str = ""
    span: str = ""
    kind: str = ""
    question: str = ""
    options: tuple = ()
    context_fingerprint: str = ""
    diagnostics: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", tuple(self.options))
        object.__setattr__(self, "diagnostics",
                           tuple(self.diagnostics))
        if not self.request_id or not self.raw_input:
            raise ValueError("request needs id and input")
        kinds = tuple(k.value for k in SemanticClarificationKind)
        if self.kind not in kinds:
            raise ValueError(f"bad clarification kind {self.kind!r}")
        if self.kind == SemanticClarificationKind.AMBIGUOUS_REFERENCE \
                .value and not self.options:
            raise ValueError("ambiguous requests need options")
        if not self.question:
            raise ValueError("clarification needs a question")


@dataclass(frozen=True)
class SemanticClarificationAnswer:
    """One recorded user pick. A pick, not a knowledge claim."""

    request_id: str = ""
    option_id: str = ""
    context_fingerprint: str = ""
    origin: str = "user"

    def __post_init__(self) -> None:
        if not self.request_id or not self.option_id:
            raise ValueError("answer needs request and option ids")
        if not self.context_fingerprint:
            raise ValueError("answer needs a context fingerprint")


@dataclass(frozen=True)
class SemanticInteractionResult:
    """One interaction step: understood, clarifying, resolved, or not."""

    state: str = ""
    raw_input: str = ""
    context_ref: str = ""
    request: Optional[SemanticClarificationRequest] = None
    answer: Optional[SemanticClarificationAnswer] = None
    understanding: Optional[CompositionalUnderstanding] = None
    complete: bool = False
    diagnostics: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics",
                           tuple(self.diagnostics))
        states = tuple(s.value for s in SemanticInteractionState)
        if self.state not in states:
            raise ValueError(f"bad interaction state {self.state!r}")
        if self.complete:
            ok_states = (
                SemanticInteractionState.UNDERSTOOD.value,
                SemanticInteractionState.CLARIFICATION_RESOLVED.value,
            )
            if self.state not in ok_states or \
                    self.understanding is None or \
                    not self.understanding.complete:
                raise ValueError(
                    "completion needs a complete understanding")


# ---- State-scoped logical reasoning (Phase M4-E). Pure data, no IO. ----
#
# WorldInferenceRule differs from WorldActionSchema on purpose: rules
# derive conclusions without touching any state, while actions produce
# new states. An entailed verdict means "derivable from this state and
# these rules", never "verified in reality".


@dataclass(frozen=True)
class WorldRuleDiagnostic:
    """Why one stored rule view was refused compilation."""

    rule_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class WorldInferenceRule:
    """One immutable logic rule over world-state propositions."""

    rule_id: str = ""
    premises: tuple = ()
    conclusion: WorldLiteralPattern = field(
        default_factory=WorldLiteralPattern)
    source_ref: str = ""
    status: str = ""

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("inference rule needs a rule_id")
        if not self.premises:
            raise ValueError("inference rule needs premises")
        object.__setattr__(self, "premises", tuple(self.premises))
        premise_vars: set = set()
        for prem in self.premises:
            if not prem.predicate:
                raise ValueError("premise needs a predicate")
            for term in (prem.subject, prem.object):
                if term.is_variable():
                    premise_vars.add(term.variable)
        if not self.conclusion.predicate:
            raise ValueError("conclusion needs a predicate")
        for term in (self.conclusion.subject, self.conclusion.object):
            if term.is_variable() and term.variable not in premise_vars:
                raise ValueError(
                    f"conclusion variable {term.variable!r} "
                    f"never appears in premises")


@dataclass(frozen=True)
class WorldProofNode:
    """One node of a reasoning proof tree.

    Leaf (kind "fact"): a state fact cited by index into an immutable
    snapshot. Rule node (kind "rule"): one rule application with the
    bindings used, the ground conclusion, one subproof per premise,
    and its depth. Validators must re-check everything; a node never
    trusts its own claims.
    """

    kind: str = ""
    state_id: str = ""
    fact_index: int = -1
    proposition: WorldProposition = field(
        default_factory=WorldProposition)
    rule_id: str = ""
    bindings: tuple = ()
    conclusion: WorldLiteralPattern = field(
        default_factory=WorldLiteralPattern)
    premises: tuple = ()
    depth: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("fact", "rule"):
            raise ValueError(f"proof node kind must be fact/rule, "
                             f"got {self.kind!r}")
        object.__setattr__(
            self, "bindings",
            tuple((p[0], p[1]) for p in self.bindings))
        object.__setattr__(self, "premises", tuple(self.premises))


@dataclass(frozen=True)
class WorldReasoningResult:
    """Verdict of reasoning about one query in one state.

    verdict is one of entailed | refuted | conflicted | unknown |
    limited | invalid. positive_proofs support the query as stated,
    negative_proofs support its opposite. complete is False whenever
    an applicable rule could not be compiled or the depth cap may
    have cut the search short.
    """

    query: WorldLiteralPattern = field(
        default_factory=WorldLiteralPattern)
    state_id: str = ""
    verdict: str = "unknown"
    positive_proofs: tuple = ()
    negative_proofs: tuple = ()
    rules_used: tuple = ()
    max_derivation_depth: int = 0
    complete: bool = True
    uncompilable: tuple = ()
    source_refs: tuple = ()

    def __post_init__(self) -> None:
        for name in ("positive_proofs", "negative_proofs", "rules_used",
                     "uncompilable", "source_refs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


def to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def from_json(s: str) -> Any:
    if not s:
        return None
    return json.loads(s)
