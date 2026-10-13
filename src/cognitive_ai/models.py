"""Core data structures. Pure dataclasses, no IO, no reasoning."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
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


def to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def from_json(s: str) -> Any:
    if not s:
        return None
    return json.loads(s)
