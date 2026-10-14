"""Compositional grounding (Phase M5-B). Pure reads, zero writes.

Walks an M5-A semantic tree, grounds each atom leaf with the real
ground_statement, and aggregates fixed composition rules. Conditionals
keep antecedent/consequent roles and stay unasserted: recognizing an
"if A then B" shape never asserts A, B, or any rule. No candidates,
no Learner calls, no Memory writes.
"""
from __future__ import annotations

from typing import Optional

from . import semcompose as _parse
from . import semground as _ground
from .models import (CompositionalUnderstanding, SemanticExpression,
                     SemanticExpressionKind, SemanticNodeEvaluation,
                     SemanticNodeStatus, SemanticParseResult)

SUPPORTED = SemanticNodeStatus.SUPPORTED.value
CONFLICTED = SemanticNodeStatus.CONFLICTED.value
UNKNOWN = SemanticNodeStatus.UNKNOWN.value
AMBIGUOUS = SemanticNodeStatus.AMBIGUOUS.value
PARTIAL = SemanticNodeStatus.PARTIALLY_SUPPORTED.value
UNSUPPORTED = SemanticNodeStatus.UNSUPPORTED.value
FAILURE = SemanticNodeStatus.FAILURE.value
UNASSERTED = SemanticNodeStatus.CONDITIONAL_UNASSERTED.value

_ATOM = SemanticExpressionKind.ATOM.value
_CONJ = SemanticExpressionKind.CONJUNCTION.value
_COND = SemanticExpressionKind.CONDITIONAL.value
_UNSUPPORTED = SemanticExpressionKind.UNSUPPORTED.value


def _verdict_status(verdict: str) -> str:
    return {"supported": SUPPORTED, "conflicted": CONFLICTED,
            "unknown": UNKNOWN, "ambiguous": AMBIGUOUS,
            "failure": FAILURE}.get(verdict, UNSUPPORTED)


def understand_compositional_statement(text: str, memory,
                                       max_depth=None,
                                       max_nodes=None
                                       ) -> CompositionalUnderstanding:
    """Parse, ground each atom once, aggregate. Read-only throughout."""
    parsed = _parse.parse_compositional_statement(text, memory,
                                                  max_depth, max_nodes)
    raw = text if isinstance(text, str) else ""
    if parsed.tree is None:
        return CompositionalUnderstanding(
            raw_input=raw, parse=parsed, root=None, grounded_atoms=0,
            counts={}, complete=False,
            diagnostics=tuple(parsed.diagnostics))
    cache: dict = {}
    calls = {"n": 0}

    def ground_once(node: SemanticExpression) -> dict:
        key = node.raw_text
        cached = cache.get(key)
        if cached is not None and cached[0] == node.atom:
            return cached[1]
        calls["n"] += 1
        verdict = _ground.ground_statement(node.raw_text, memory)
        doc = verdict.to_dict()
        cache[key] = (node.atom, doc)
        return doc

    root = _evaluate(parsed.tree, memory, ground_once)
    counts = _count_statuses(root)
    diags = list(parsed.diagnostics)
    for extra in _collect_diagnostics(root):
        if extra not in diags:
            diags.append(extra)
    complete = parsed.complete and _tree_ok(root)
    return CompositionalUnderstanding(
        raw_input=raw, parse=parsed, root=root,
        grounded_atoms=calls["n"], counts=counts, complete=complete,
        diagnostics=tuple(diags))


def _evidence_refs(doc: dict) -> tuple:
    refs = []
    for entry in doc.get("evidence", []) or []:
        kind, ref = entry.get("kind", ""), entry.get("ref", "")
        if kind or ref:
            refs.append(f"{kind}:{ref}")
    return tuple(refs)


def _evaluate(node: SemanticExpression, memory, ground_once
              ) -> SemanticNodeEvaluation:
    if node.kind == _ATOM:
        return _evaluate_atom(node, ground_once)
    if node.kind == _CONJ:
        children = tuple(_evaluate(c, memory, ground_once)
                         for c in node.children)
        return SemanticNodeEvaluation(
            kind=_CONJ, raw_text=node.raw_text,
            status=_aggregate_conjunction([c.status for c in children]),
            children=children,
            diagnostics=_child_diagnostics(children),
            limited=any(c.limited for c in children))
    if node.kind == _COND:
        children = tuple(_evaluate(c, memory, ground_once)
                         for c in node.children)
        failed = any(_has_failure(c) for c in children)
        return SemanticNodeEvaluation(
            kind=_COND, raw_text=node.raw_text,
            status=FAILURE if failed else UNASSERTED,
            children=children, roles=tuple(node.roles),
            diagnostics=_child_diagnostics(children),
            limited=any(c.limited for c in children))
    if node.diagnosis.startswith("ambiguous:"):
        status = AMBIGUOUS
    else:
        status = UNSUPPORTED
    return SemanticNodeEvaluation(
        kind=_UNSUPPORTED, raw_text=node.raw_text, status=status,
        diagnostics=(node.diagnosis or "unsupported span",))


def _evaluate_atom(node: SemanticExpression, ground_once
                   ) -> SemanticNodeEvaluation:
    doc = ground_once(node)
    status = _verdict_status(doc.get("verdict", ""))
    limited = bool(doc.get("limited", False))
    diags = []
    if doc.get("detail"):
        diags.append(str(doc["detail"]))
    if limited:
        diags.append("query hit an explicit engine limit")
    if status == UNSUPPORTED:
        diags.append(f"unexpected verdict {doc.get('verdict')!r}")
    return SemanticNodeEvaluation(
        kind=_ATOM, raw_text=node.raw_text, status=status,
        grounding=doc, diagnostics=tuple(diags),
        source_refs=_evidence_refs(doc) or (f"atom:{node.raw_text}",),
        limited=limited)


def _aggregate_conjunction(statuses: list) -> str:
    if FAILURE in statuses:
        return FAILURE
    if CONFLICTED in statuses:
        return CONFLICTED
    if AMBIGUOUS in statuses:
        return AMBIGUOUS
    if statuses and all(s == SUPPORTED for s in statuses):
        return SUPPORTED
    if SUPPORTED in statuses:
        return PARTIAL
    if UNKNOWN in statuses:
        return UNKNOWN
    return UNSUPPORTED


def _has_failure(node: SemanticNodeEvaluation) -> bool:
    if node.status == FAILURE:
        return True
    return any(_has_failure(c) for c in node.children)


def _child_diagnostics(children: tuple) -> tuple:
    out = []
    for child in children:
        for diag in child.diagnostics:
            if diag not in out:
                out.append(diag)
    return tuple(out)


def _count_statuses(node: SemanticNodeEvaluation) -> dict:
    counts = {s.value: 0 for s in SemanticNodeStatus}
    counts[node.status] += 1
    for child in node.children:
        for key, value in _count_statuses(child).items():
            counts[key] += value
    return counts


def _collect_diagnostics(node: SemanticNodeEvaluation) -> list:
    out = [f"{node.raw_text!r}: {d}" for d in node.diagnostics]
    for child in node.children:
        out.extend(_collect_diagnostics(child))
    return out


def _tree_ok(node: SemanticNodeEvaluation) -> bool:
    if node.status in (FAILURE, UNSUPPORTED, AMBIGUOUS) or node.limited:
        return False
    return all(_tree_ok(c) for c in node.children)
