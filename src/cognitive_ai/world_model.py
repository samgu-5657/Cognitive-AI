"""Read-only world-model adapter (Phase M4-B).

Maps the legacy stores (concepts / attributes / relations / claims /
evidences / rules) to the unified World* views in models.py.

Hard limits, enforced by construction and tests:

- Reads go through public ``memory.Memory`` read APIs only. This
  module never opens a connection and never writes.
- Legacy concept rows always map to ``legacy_concept``: the old table
  cannot tell individuals from classes, and names are never used to
  guess. ``event`` / ``state`` kinds exist for the future only.
- Stored rows are positive ``key=value`` records, so polarity is
  positive; the stores keep no negation column. Modality stays
  unknown, except the ``function`` key, which the codebase itself
  reads as capability (cf. semground capability queries); the raw key
  is always kept, so the mapping stays traceable.
- Knowledge status is copied verbatim into ``WorldFactView.status``;
  ``inherited`` is never upgraded to ``verified``.
- Links are only built where the schema stores them
  (``evidences.claim_id`` → claim, ``claims.subject_id`` → concept).
  Claim text is never matched to attribute rows, so facts carry no
  fabricated evidence and unmappable rows land in
  ``snapshot.unmapped`` with a reason instead of being dropped.
- Rule derivations are never materialized as observations: rules are
  exposed as read-only views (stable references, no firing).
"""
from __future__ import annotations

from typing import Any, Optional

from .memory import Memory
from .models import (WorldClaimView, WorldEvidenceRef, WorldFactView,
                     WorldObjectKind, WorldObjectRef, WorldProposition,
                     WorldRuleView, WorldSnapshot, WorldTerm,
                     WorldEvidenceDirection, WorldModality, WorldPolarity)

# The codebase's own convention: the `function` attribute key expresses
# a capability (see semground capability handling). Everywhere else,
# modality is unknown rather than guessed.
CAPABILITY_KEY = "function"


def object_ref_for_concept(row: dict) -> WorldObjectRef:
    """Map one concept row to a stable object reference.

    Always ``legacy_concept``: neither the row nor the legacy ``kind``
    column can prove individual vs class, and promotion by name or
    guesswork is forbidden.
    """
    return WorldObjectRef(
        ref_id=str(row.get("id", "")),
        name=str(row.get("name", row.get("id", ""))),
        kind=WorldObjectKind.LEGACY_CONCEPT)


def resolve_term(mem: Memory, text: str) -> WorldTerm:
    """Resolve raw reference text to a term without guessing.

    Exact id hit or a single name hit resolves; several name hits stay
    ``unresolved`` (ambiguous, must not pick); no hit at all means the
    object is ``missing`` while the raw text is still kept.
    """
    if not text:
        return WorldTerm()
    if mem.get_concept(text) is not None:
        row = mem.get_concept(text)
        return WorldTerm(ref=object_ref_for_concept(row))
    hits = mem.find_concepts_by_name(text)
    ids = {c["id"] for c in hits}
    if len(ids) == 1:
        row = mem.get_concept(next(iter(ids)))
        return WorldTerm(ref=object_ref_for_concept(row))
    if ids:
        return WorldTerm(unresolved=text)
    return WorldTerm(unresolved=text, missing=True)


def attribute_to_fact(mem: Memory, row: dict) -> Optional[WorldFactView]:
    """Map one attribute row. None only when its subject row is gone."""
    cid = row.get("concept_id", "")
    concept = mem.get_concept(cid) if cid else None
    if concept is None:
        return None
    key = row.get("key", "")
    value = row.get("value")
    modality = WorldModality.CAPABLE if key == CAPABILITY_KEY \
        else WorldModality.UNKNOWN
    prop = WorldProposition(
        subject=WorldTerm(ref=object_ref_for_concept(concept)),
        predicate=str(key),
        object=WorldTerm(literal=value, has_literal=True),
        polarity=WorldPolarity.POSITIVE,
        modality=modality,
        representation_origin="attribute",
        source_ref=f"attribute:{row.get('id')}",
        raw={"key": key, "value": value})
    return WorldFactView(
        proposition=prop,
        record_id=f"attribute:{row.get('id')}",
        status=str(row.get("status", "")),
        evidence=(),
        derivation=None,
        store="attribute")


def relation_to_fact(mem: Memory, row: dict) -> Optional[WorldFactView]:
    """Map one relation row. None only when its subject row is gone."""
    src = row.get("src_id", "")
    concept = mem.get_concept(src) if src else None
    if concept is None:
        return None
    rel_type = row.get("rel_type", "")
    dst = row.get("dst_id", "")
    prop = WorldProposition(
        subject=WorldTerm(ref=object_ref_for_concept(concept)),
        predicate=str(rel_type),
        object=resolve_term(mem, str(dst)),
        polarity=WorldPolarity.POSITIVE,
        modality=WorldModality.UNKNOWN,
        representation_origin="relation",
        source_ref=f"relation:{row.get('id')}",
        raw={"rel_type": rel_type, "dst_id": dst})
    return WorldFactView(
        proposition=prop,
        record_id=f"relation:{row.get('id')}",
        status=str(row.get("status", "")),
        evidence=(),
        derivation=None,
        store="relation")


def evidence_refs_for_claim(mem: Memory, claim_id: int) -> tuple:
    """Evidence rows linked by the stored evidences.claim_id key."""
    out = []
    for e in mem.list_evidences(claim_id):
        out.append(WorldEvidenceRef(
            evidence_id=int(e.get("id", 0)),
            source=str(e.get("source", "")),
            claim_ref=f"claim:{claim_id}",
            direction=WorldEvidenceDirection.UNKNOWN))
    return tuple(out)


def claim_view(mem: Memory, row: dict) -> WorldClaimView:
    """Map one claim row; text is kept verbatim, never re-structured."""
    subject_id = row.get("subject_id") or ""
    subject_ref = ""
    if subject_id and mem.get_concept(subject_id) is not None:
        subject_ref = str(subject_id)
    return WorldClaimView(
        claim_id=int(row.get("id", 0)),
        text=str(row.get("text", "")),
        status=str(row.get("status", "")),
        subject_ref=subject_ref,
        evidence=evidence_refs_for_claim(mem, int(row.get("id", 0))))


def rule_view(row: dict) -> WorldRuleView:
    """Read-only view of one stored rule (reference, no firing)."""
    conds = row.get("conditions", []) or []
    excs = row.get("exceptions", []) or []
    conc = row.get("conclusion", {}) or {}
    return WorldRuleView(
        rule_id=str(row.get("id", "")),
        name=str(row.get("name", row.get("id", ""))),
        status=str(row.get("status", "")),
        source=str(row.get("source", "")),
        conditions=tuple(conds),
        conclusion=dict(conc),
        exceptions=tuple(excs))


def build_snapshot(mem: Memory) -> WorldSnapshot:
    """Read the whole KB into one immutable snapshot. Pure reads."""
    objects = []
    for c in mem.list_concepts():
        objects.append(object_ref_for_concept(c))
    facts = []
    unmapped = []
    for c in mem.list_concepts():
        for a in mem.get_attributes(c["id"]):
            fact = attribute_to_fact(mem, a)
            if fact is None:
                unmapped.append({"kind": "attribute",
                                 "ref": f"attribute:{a.get('id')}",
                                 "reason": "subject concept row absent"})
            else:
                facts.append(fact)
    for r in mem.get_relations():
        fact = relation_to_fact(mem, r)
        if fact is None:
            unmapped.append({"kind": "relation",
                             "ref": f"relation:{r.get('id')}",
                             "reason": "subject concept row absent"})
        else:
            facts.append(fact)
    rules = tuple(rule_view(r) for r in mem.list_rules())
    claims = tuple(claim_view(mem, c) for c in mem.list_claims())
    return WorldSnapshot(
        objects=tuple(objects),
        facts=tuple(facts),
        rules=rules,
        claims=claims,
        unmapped=tuple(unmapped))
