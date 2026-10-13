"""Clarification interaction loop (Phase M5-D). Pure memory, no IO.

Understood inputs pass through; ambiguous or missing references
produce structured clarification requests; a validated user pick
rebuilds a local reference context and re-runs real understanding and
grounding. No guessing, no learning writes, no global sessions: the
browser resubmits the request it was given.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

from . import semcontext as _ctx
from .models import (SemanticClarificationAnswer,
                     SemanticClarificationKind,
                     SemanticClarificationOption,
                     SemanticClarificationRequest,
                     SemanticContext, SemanticContextMention,
                     SemanticInteractionResult, SemanticInteractionState)

UNDERSTOOD = SemanticInteractionState.UNDERSTOOD.value
NEEDS_CLARIFICATION = SemanticInteractionState.NEEDS_CLARIFICATION.value
RESOLVED = SemanticInteractionState.CLARIFICATION_RESOLVED.value
UNSUPPORTED = SemanticInteractionState.UNSUPPORTED.value
INVALID = SemanticInteractionState.INVALID.value

AMBIGUOUS_REF = SemanticClarificationKind.AMBIGUOUS_REFERENCE.value
MISSING = SemanticClarificationKind.MISSING_ANTECEDENT.value
BAD_IDENTITY = SemanticClarificationKind.UNSUPPORTED_IDENTITY.value
BAD_EXPRESSION = SemanticClarificationKind.UNSUPPORTED_EXPRESSION.value


def context_fingerprint(context: SemanticContext) -> str:
    """Deterministic context identity (sha256, never hash())."""
    payload = {"sequence": context.sequence,
               "mentions": [[m.object_ref.ref_id, m.order,
                             m.mention_text, m.source_statement]
                            for m in context.mentions]}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _stable_id(*parts: str) -> str:
    return hashlib.sha256(
        "|".join(parts).encode("utf-8")).hexdigest()[:16]


def create_clarification_request(raw_input: str, span: str, kind: str,
                                 question: str, options,
                                 context_fingerprint: str,
                                 diagnostics=()) -> \
        SemanticClarificationRequest:
    """Build a request with deterministic option and request ids.

    options: (ref, mention_text, selectable, reason) tuples. The
    context fingerprint is passed explicitly so id computation never
    depends on hidden session state.
    """
    norm_options = []
    for ref, mention_text, selectable, reason in options:
        norm_options.append(SemanticClarificationOption(
            option_id=f"opt-{ref.ref_id}",
            ref=ref if selectable else None,
            display_name=ref.name,
            mention_ref=mention_text,
            selectable=bool(selectable),
            reason=reason if not selectable else ""))
    norm_options.sort(key=lambda o: o.option_id)
    request_id = "clr-" + _stable_id(
        raw_input, span, kind,
        ",".join(o.option_id for o in norm_options),
        context_fingerprint)
    return SemanticClarificationRequest(
        request_id=request_id, raw_input=raw_input, span=span, kind=kind,
        question=question, options=tuple(norm_options),
        context_fingerprint=context_fingerprint,
        diagnostics=tuple(diagnostics))


def _request_intact(request: SemanticClarificationRequest) -> bool:
    """Recompute the id from stored fields: tampering breaks the hash."""
    expect = "clr-" + _stable_id(
        request.raw_input, request.span, request.kind,
        ",".join(sorted(o.option_id for o in request.options)),
        request.context_fingerprint)
    if expect != request.request_id:
        return False
    for option in request.options:
        if option.selectable:
            if option.ref is None:
                return False
            if option.option_id != f"opt-{option.ref.ref_id}":
                return False
            if option.display_name != option.ref.name:
                return False
    return True


def begin_semantic_interaction(text: str, memory: object,
                               context: SemanticContext, **_kw):
    """Understand; clarify only when truly needed. Read-only."""
    from .semcontext import understand_with_context
    understood = understand_with_context(text, memory, context)
    raw = text if isinstance(text, str) else ""
    if understood.complete and understood.understanding is not None:
        return SemanticInteractionResult(
            state=UNDERSTOOD, raw_input=raw,
            context_ref=context_fingerprint(context),
            understanding=understood.understanding, complete=True,
            diagnostics=tuple(understood.diagnostics))
    failed = [r for r in understood.resolutions
              if r.status != "resolved"]
    if not failed:
        return SemanticInteractionResult(
            state=UNSUPPORTED, raw_input=raw,
            context_ref=context_fingerprint(context),
            understanding=understood.understanding, complete=False,
            diagnostics=tuple(understood.diagnostics) + (
                "understanding incomplete without reference cause",))
    first = failed[0]
    if first.status == "ambiguous":
        options = []
        for ref in first.candidates:
            ok, reason = _ctx.check_reference_identifiable(ref, memory)
            options.append((ref, first.mention, ok,
                            "" if ok else reason))
        mention = first.mention
        request = create_clarification_request(
            raw, first.mention, AMBIGUOUS_REF,
            f"“{mention}”可能指代多个对象，请选择一个：",
            options, context_fingerprint(context),
            (first.reason,))
        return SemanticInteractionResult(
            state=NEEDS_CLARIFICATION, raw_input=raw,
            context_ref=context_fingerprint(context), request=request,
            complete=False,
            diagnostics=tuple(understood.diagnostics))
    if first.status == "unresolved":
        request = create_clarification_request(
            raw, first.mention, MISSING,
            "缺少可指代的先前对象，请补充一条先前陈述或重新描述对象。",
            [], context_fingerprint(context), (first.reason,))
        return SemanticInteractionResult(
            state=NEEDS_CLARIFICATION, raw_input=raw,
            context_ref=context_fingerprint(context), request=request,
            complete=False,
            diagnostics=tuple(understood.diagnostics))
    if first.status in ("unsupported", "invalid"):
        request = create_clarification_request(
            raw, first.mention, BAD_IDENTITY,
            "当前系统无法可靠区分该对象身份，请重新描述。",
            [], context_fingerprint(context), (first.reason,))
        return SemanticInteractionResult(
            state=NEEDS_CLARIFICATION, raw_input=raw,
            context_ref=context_fingerprint(context), request=request,
            complete=False,
            diagnostics=tuple(understood.diagnostics))
    return SemanticInteractionResult(
        state=INVALID, raw_input=raw,
        context_ref=context_fingerprint(context), complete=False,
        diagnostics=("unclassifiable reference outcome",))


def answer_semantic_clarification(request: SemanticClarificationRequest,
                                  context: SemanticContext,
                                  answer: SemanticClarificationAnswer,
                                  memory: object, **_kw):
    """Validate a user pick, rebuild a local context, re-understand."""
    from .semcontext import understand_with_context
    if not isinstance(request, SemanticClarificationRequest) or \
            not isinstance(answer, SemanticClarificationAnswer):
        return SemanticInteractionResult(
            state=INVALID, raw_input=getattr(request, "raw_input", ""),
            complete=False, diagnostics=("bad request/answer",))
    if answer.request_id != request.request_id:
        return _invalid(request, "answer targets another request")
    if answer.context_fingerprint != request.context_fingerprint or \
            answer.context_fingerprint != context_fingerprint(context):
        return _invalid(request, "context changed; re-run understanding")
    if not _request_intact(request):
        return _invalid(request, "request content does not match its id")
    options = {o.option_id: o for o in request.options}
    option = options.get(answer.option_id)
    if option is None:
        return _invalid(request, "unknown option id")
    if not option.selectable or option.ref is None:
        return _invalid(request, "option not selectable")
    ok, reason = _ctx.check_reference_identifiable(option.ref, memory)
    if not ok:
        return _invalid(request, f"reference no longer safe: {reason}")
    local = SemanticContext(
        mentions=(SemanticContextMention(
            object_ref=option.ref, mention_text=request.span,
            role="subject",
            source_statement=f"clarification:{request.request_id}",
            order=0,
            source_ref=f"context-seq:{context.sequence}"),),
        sequence=context.sequence, trimmed=False, diagnostics=())
    understood = understand_with_context(request.raw_input, memory,
                                         local)
    if understood.understanding is None:
        return SemanticInteractionResult(
            state=UNSUPPORTED, raw_input=request.raw_input,
            context_ref=context_fingerprint(context), request=request,
            answer=answer, complete=False,
            diagnostics=tuple(understood.diagnostics))
    return SemanticInteractionResult(
        state=RESOLVED, raw_input=request.raw_input,
        context_ref=context_fingerprint(context), request=request,
        answer=answer, understanding=understood.understanding,
        complete=understood.complete,
        diagnostics=tuple(understood.diagnostics))


def _invalid(request, reason: str) -> SemanticInteractionResult:
    raw = request.raw_input if isinstance(
        request, SemanticClarificationRequest) else ""
    return SemanticInteractionResult(
        state=INVALID, raw_input=raw, request=request
        if isinstance(request, SemanticClarificationRequest) else None,
        complete=False, diagnostics=(reason,))


def _ref_from_json(data: dict):
    from .models import WorldObjectRef
    if not isinstance(data, dict):
        raise ValueError("bad object reference")
    ref_id = data.get("ref_id", "")
    name = data.get("name", ref_id)
    kind = data.get("kind", "legacy_concept")
    if not ref_id:
        raise ValueError("bad object reference")
    return WorldObjectRef(ref_id=str(ref_id), name=str(name),
                          kind=str(kind))


def clarification_request_from_json(data: dict):
    """Rebuild a request echoed by the browser. Strict shapes only;
    integrity is still re-checked by content hash on use."""
    from .models import SemanticClarificationOption
    if not isinstance(data, dict):
        raise ValueError("bad request")
    options = []
    for entry in data.get("options", []) or []:
        if not isinstance(entry, dict):
            raise ValueError("bad option")
        ref = None
        if entry.get("ref") is not None:
            ref = _ref_from_json(entry["ref"])
        options.append(SemanticClarificationOption(
            option_id=str(entry.get("option_id", "")),
            ref=ref,
            display_name=str(entry.get("display_name", "")),
            mention_ref=str(entry.get("mention_ref", "")),
            selectable=bool(entry.get("selectable", False)),
            reason=str(entry.get("reason", ""))))
    return SemanticClarificationRequest(
        request_id=str(data.get("request_id", "")),
        raw_input=str(data.get("raw_input", "")),
        span=str(data.get("span", "")),
        kind=str(data.get("kind", "")),
        question=str(data.get("question", "")),
        options=tuple(options),
        context_fingerprint=str(data.get("context_fingerprint", "")),
        diagnostics=tuple(data.get("diagnostics", []) or []))
