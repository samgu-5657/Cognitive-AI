"""Explicit-context reference handling (Phase M5-C). Pure, read-only.

Contexts are caller-managed immutable objects: extending returns a
new context, never mutates. Pronoun "它" resolves only against
subject mentions with stable, re-identifiable references; ambiguity
is kept, never guessed. Demonstratives need real individual
identity, which legacy data cannot supply. Nothing here writes
knowledge, learns, or changes state.
"""
from __future__ import annotations

from typing import Optional

from . import policy
from .models import (ContextualUnderstanding, ReferenceResolutionStatus,
                     SemanticContext, SemanticContextMention,
                     SemanticReferenceResolution, WorldObjectKind,
                     WorldObjectRef)
from .world_model import object_ref_for_concept

NOT_REFERENCE = ReferenceResolutionStatus.NOT_REFERENCE.value
RESOLVED = ReferenceResolutionStatus.RESOLVED.value
AMBIGUOUS = ReferenceResolutionStatus.AMBIGUOUS.value
UNRESOLVED = ReferenceResolutionStatus.UNRESOLVED.value
UNSUPPORTED = ReferenceResolutionStatus.UNSUPPORTED.value
INVALID = ReferenceResolutionStatus.INVALID.value

PRONOUN = "它"
_DEMONSTRATIVE_PREFIXES = ("这个", "那个")
_MENTION_PREFIX = "前面提到的"


def empty_context() -> SemanticContext:
    """One fresh, empty explicit context."""
    return SemanticContext(mentions=(), sequence=0, trimmed=False,
                           diagnostics=())


def _walk_pairs(expression, evaluation):
    yield expression, evaluation
    for expr_child, eval_child in zip(expression.children,
                                      evaluation.children):
        yield from _walk_pairs(expr_child, eval_child)


def extend_semantic_context(context: SemanticContext,
                            understanding, memory: object,
                            max_mentions: Optional[int] = None
                            ) -> SemanticContext:
    """Register reliably known subject mentions. Returns a new context.

    Only atoms whose grounding carries a resolvable subject reference
    are registered, judged on subject_match alone. Ambiguous or
    reference-less subjects are skipped. Nothing is confirmed.
    """
    if max_mentions is None:
        max_mentions = policy.MAX_SEMANTIC_CONTEXT_MENTIONS
    if understanding is None or understanding.parse is None or \
            understanding.parse.tree is None or \
            understanding.root is None:
        return SemanticContext(
            mentions=context.mentions, sequence=context.sequence + 1,
            trimmed=context.trimmed,
            diagnostics=tuple(context.diagnostics) + (
                "nothing registrable in this understanding",))
    mentions = list(context.mentions)
    order = len(mentions)
    diagnostics = list(context.diagnostics)
    for expr, node in _walk_pairs(understanding.parse.tree,
                                  understanding.root):
        if expr.kind != "atom" or node.kind != "atom":
            continue
        parse = node.grounding.get("parse", {}) or {}
        if not parse.get("subject_known"):
            continue
        match = node.grounding.get("subject_match", {}) or {}
        concept_id = match.get("concept_id")
        if not concept_id:
            continue
        row = memory.get_concept(concept_id)
        if row is None:
            diagnostics.append(
                f"subject reference {concept_id!r} no longer stored")
            continue
        mentions.append(SemanticContextMention(
            object_ref=object_ref_for_concept(row),
            mention_text=str(parse.get("subject", "")),
            role="subject", source_statement=expr.raw_text,
            order=order,
            source_ref=f"context-seq:{context.sequence + 1}"))
        order += 1
    trimmed = context.trimmed
    if len(mentions) > max_mentions:
        dropped = len(mentions) - max_mentions
        mentions = mentions[dropped:]
        trimmed = True
        diagnostics.append(f"trimmed {dropped} oldest mention(s)")
    return SemanticContext(mentions=tuple(mentions),
                           sequence=context.sequence + 1,
                           trimmed=trimmed,
                           diagnostics=tuple(diagnostics))


def _reidentify(ref: WorldObjectRef, memory: object) -> tuple:
    """Check one reference still resolves uniquely. Returns
    (ok, reason): ok means exactly this stable id re-identifies."""
    if memory.get_concept(ref.ref_id) is None:
        return False, f"reference {ref.ref_id!r} no longer stored"
    hits = memory.find_concepts_by_name(ref.name)
    ids = {c["id"] for c in hits}
    if ref.ref_id in ids and len(ids) == 1:
        direct = memory.get_concept(ref.name)
        if direct is None or direct["id"] == ref.ref_id:
            return True, "stable id re-identified uniquely"
    return False, (f"name {ref.name!r} maps to {sorted(ids)}; "
                   f"ref {ref.ref_id!r} cannot pass silently")


def check_reference_identifiable(ref: WorldObjectRef,
                                   memory: object) -> tuple:
    """Public re-identification check for interaction code.

    Same rule the resolver itself enforces; exposed so clarification
    options can be marked selectable without duplicating the logic.
    """
    return _reidentify(ref, memory)


def resolve_subject_reference(subject: str, context: SemanticContext,
                              memory: object) -> SemanticReferenceResolution:
    """Resolve one subject span. Pure function, no writes."""
    if not isinstance(subject, str) or not subject:
        return SemanticReferenceResolution(
            mention=str(subject), status=INVALID,
            reason="empty subject cannot resolve")
    if subject != PRONOUN and not subject.startswith(
            _DEMONSTRATIVE_PREFIXES) and not subject.startswith(
            _MENTION_PREFIX):
        return SemanticReferenceResolution(
            mention=subject, status=NOT_REFERENCE,
            reason="not a handled reference form")
    seq_ref = f"context-seq:{context.sequence}"
    if subject == PRONOUN:
        return _resolve_pronoun(subject, context, memory, seq_ref)
    return _resolve_demonstrative(subject, context, memory, seq_ref)


def _subject_candidates(context: SemanticContext) -> list:
    seen: list = []
    for mention in context.mentions:
        if mention.role != "subject":
            continue
        if mention.object_ref.ref_id not in \
                [r.ref_id for r in seen]:
            seen.append(mention.object_ref)
    return seen


def _resolve_pronoun(subject: str, context: SemanticContext,
                     memory: object, seq_ref: str
                     ) -> SemanticReferenceResolution:
    candidates = _subject_candidates(context)
    if not candidates:
        return SemanticReferenceResolution(
            mention=subject, status=UNRESOLVED, source_ref=seq_ref,
            reason="no subject candidates in context")
    if len(candidates) > 1:
        return SemanticReferenceResolution(
            mention=subject, status=AMBIGUOUS, candidates=tuple(candidates),
            source_ref=seq_ref,
            reason=f"{len(candidates)} distinct references; no recency pick",
            diagnostics=("recency does not disambiguate",))
    ref = candidates[0]
    ok, reason = _reidentify(ref, memory)
    if not ok:
        if memory.get_concept(ref.ref_id) is None:
            return SemanticReferenceResolution(
                mention=subject, status=INVALID, candidates=(ref,),
                source_ref=seq_ref, reason=reason)
        return SemanticReferenceResolution(
            mention=subject, status=AMBIGUOUS, candidates=(ref,),
            source_ref=seq_ref, reason=reason)
    return SemanticReferenceResolution(
        mention=subject, status=RESOLVED, resolved_ref=ref,
        candidates=(ref,), source_ref=seq_ref, reason=reason)


def _demonstrative_noun(subject: str) -> tuple:
    for prefix in _DEMONSTRATIVE_PREFIXES:
        if subject.startswith(prefix):
            noun = subject[len(prefix):]
            return noun, "demonstrative"
    if subject.startswith(_MENTION_PREFIX):
        noun = subject[len(_MENTION_PREFIX):]
        if noun.startswith("的"):
            noun = noun[1:]
        return noun, "mentioned"
    return "", ""


def _resolve_demonstrative(subject: str, context: SemanticContext,
                           memory: object, seq_ref: str
                           ) -> SemanticReferenceResolution:
    noun, _flavor = _demonstrative_noun(subject)
    if not noun:
        return SemanticReferenceResolution(
            mention=subject, status=INVALID, source_ref=seq_ref,
            reason="demonstrative carries no usable noun")
    individuals = [m.object_ref for m in context.mentions
                   if m.object_ref.name == noun and
                   m.object_ref.kind == WorldObjectKind.INDIVIDUAL]
    if not individuals:
        return SemanticReferenceResolution(
            mention=subject, status=UNSUPPORTED, source_ref=seq_ref,
            reason="no individual identity available; legacy concepts "
            "cannot stand in for a specific individual",
            diagnostics=("system cannot yet tell individuals apart "
                         "from legacy concepts",))
    if len({r.ref_id for r in individuals}) > 1:
        return SemanticReferenceResolution(
            mention=subject, status=AMBIGUOUS,
            candidates=tuple(individuals), source_ref=seq_ref,
            reason="several individuals match")
    ref = individuals[0]
    ok, reason = _reidentify(ref, memory)
    if not ok:
        return SemanticReferenceResolution(
            mention=subject, status=AMBIGUOUS, candidates=(ref,),
            source_ref=seq_ref, reason=reason)
    return SemanticReferenceResolution(
        mention=subject, status=RESOLVED, resolved_ref=ref,
        candidates=(ref,), source_ref=seq_ref, reason=reason)


def _leaf_subjects(tree) -> list:
    """Atom leaf (raw_text, subject) pairs in tree order."""
    out = []

    def visit(node) -> None:
        if node.kind == "atom":
            subject = (node.atom.get("subject", "") or "")
            out.append((node.raw_text, subject))
        for child in node.children:
            visit(child)

    visit(tree)
    return out


def _locate_replacements(raw: str, leaves: list):
    """Verifiable leaf span positions, applied back to front.

    Every replaced leaf must start with its subject and be found in
    order; anything else aborts the rewrite with a reason.
    """
    spans = []
    offset = 0
    for raw_text, subject, name in leaves:
        at = raw.find(raw_text, offset)
        if at < 0:
            return None, f"leaf {raw_text!r} not locatable in input"
        stripped = raw_text.strip()
        if not stripped.startswith(subject):
            return None, (f"subject {subject!r} not at the head of "
                          f"{raw_text!r}")
        spans.append((at, raw_text, subject, name))
        offset = at + len(raw_text)
    return spans, ""


def understand_with_context(text: str, memory: object,
                            context: SemanticContext,
                            max_depth=None,
                            max_nodes=None):
    """Understand with explicit context. Pure reads, zero writes."""
    from .semmeaning import understand_compositional_statement
    from .semcompose import parse_compositional_statement
    from .models import ContextualUnderstanding
    raw = text if isinstance(text, str) else ""
    parsed = parse_compositional_statement(text, memory, max_depth,
                                           max_nodes)
    if parsed.tree is None:
        return ContextualUnderstanding(
            raw_input=raw, context_seq=context.sequence, resolutions=(),
            normalized_text="", understanding=None, complete=False,
            diagnostics=tuple(parsed.diagnostics))
    jobs = []
    resolutions = []
    for raw_text, subject in _leaf_subjects(parsed.tree):
        if not (subject == PRONOUN or subject.startswith(
                _DEMONSTRATIVE_PREFIXES) or subject.startswith(
                _MENTION_PREFIX)):
            continue
        resolution = resolve_subject_reference(subject, context, memory)
        resolutions.append(resolution)
        if resolution.status != RESOLVED:
            return ContextualUnderstanding(
                raw_input=raw, context_seq=context.sequence,
                resolutions=tuple(resolutions), normalized_text="",
                understanding=None, complete=False,
                diagnostics=(f"reference {subject!r}: "
                             f"{resolution.status}: "
                             f"{resolution.reason}",))
        jobs.append((raw_text, subject,
                     resolution.resolved_ref.name))
    normalized, locate_reason = _apply_replacements(raw, jobs)
    if normalized is None:
        return ContextualUnderstanding(
            raw_input=raw, context_seq=context.sequence,
            resolutions=tuple(resolutions), normalized_text="",
            understanding=None, complete=False,
            diagnostics=(locate_reason or
                         "leaf spans not safely locatable",))
    understanding = understand_compositional_statement(normalized, memory,
                                                       max_depth,
                                                       max_nodes)
    diags = list(understanding.diagnostics)
    diags.append(f"normalized {raw!r} to {normalized!r}")
    from .models import CompositionalUnderstanding as _CU
    understanding = _CU(
        raw_input=understanding.raw_input, parse=understanding.parse,
        root=understanding.root,
        grounded_atoms=understanding.grounded_atoms,
        counts=dict(understanding.counts), complete=understanding.complete,
        diagnostics=tuple(diags))
    return ContextualUnderstanding(
        raw_input=raw, context_seq=context.sequence,
        resolutions=tuple(resolutions), normalized_text=normalized,
        understanding=understanding,
        complete=understanding.complete,
        diagnostics=tuple(diags))


def _apply_replacements(raw, jobs):
    located, reason = _locate_replacements(raw, jobs)
    if located is None:
        return None, reason
    out = raw
    for at, raw_text, subject, name in sorted(located, reverse=True):
        head = raw_text[:len(raw_text) - len(raw_text.lstrip())]
        tail = raw_text[len(subject):]
        out = out[:at] + head + name + tail + \
            out[at + len(raw_text):]
    return out, ""
