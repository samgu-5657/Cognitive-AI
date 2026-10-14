"""Compositional statement parsing (Phase M5-A). Pure, read-only.

Builds recursive semantic trees from explicit connectives (并且/而且/
同时) and restricted conditionals (如果 A，那么 B). Leaf nodes are
parsed ONLY by the existing semparse atom parser; this module never
re-implements atomic lexical or semantic judgments. Unparseable spans
stay as unsupported nodes with reasons, never dropped, never guessed.
No learning, no reasoning, no writes of any kind.
"""
from __future__ import annotations

from typing import Optional

from . import policy
from .models import (SemanticExpression, SemanticExpressionKind,
                     SemanticParseResult)
from .semparse import parse_statement

ATOM_PREDICATES = ("is_a", "has", "capable")
CONJUNCTIVES = ("并且", "而且", "同时")
CONDITION_OPEN = "如果"
CONDITION_THEN = "那么"

_QUOTE_CLOSE = {"“": "”", "‘": "’", "「": "」", "『": "』",
                '"': '"', "'": "'"}
_PAREN_OPEN = set("（(")
_PAREN_CLOSE = {"）": "（", ")": "("}


def _scan_top_level(text: str, tokens: tuple) -> list:
    """Positions of tokens at depth 0 and outside quotes.

    Only structural level is considered here, never word meaning:
    anything inside quotes or parentheses is left untouched.
    """
    hits = []
    in_quote = None
    depth = 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_quote is not None:
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in _QUOTE_CLOSE:
            in_quote = _QUOTE_CLOSE[ch]
            i += 1
            continue
        if ch in _PAREN_OPEN:
            depth += 1
            i += 1
            continue
        if ch in _PAREN_CLOSE:
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0:
            for tok in tokens:
                if text.startswith(tok, i):
                    hits.append((i, tok))
                    break
        i += 1
    return hits


def _split_conditional(text: str):
    """Strict 如果 A，那么 B split or None.

    Returns (antecedent, consequent, ok). Empty sides are reported,
    never guessed.
    """
    stripped = text.strip()
    if not stripped.startswith(CONDITION_OPEN):
        return None
    rest = stripped[len(CONDITION_OPEN):]
    hits = _scan_top_level(rest, (CONDITION_THEN,))
    if not hits:
        return None
    pos = hits[0][0]
    antecedent = rest[:pos].strip().rstrip("，,").strip()
    consequent = rest[pos + len(CONDITION_THEN):].strip()
    return antecedent, consequent, bool(antecedent and consequent)


def _split_conjunction(text: str):
    """Top-level connective split, or None when absent.

    Returns the segment list, or ("",) marker semantics via empty
    segments: any empty segment means malformed input.
    """
    hits = _scan_top_level(text, CONJUNCTIVES)
    if not hits:
        return None
    segments = []
    prev = 0
    for pos, tok in hits:
        segments.append(text[prev:pos].strip())
        prev = pos + len(tok)
    segments.append(text[prev:].strip())
    return segments


def _tally(node: SemanticExpression) -> tuple:
    atoms, composite = 0, 0
    if node.kind == SemanticExpressionKind.ATOM.value:
        atoms = 1
    elif node.kind in (SemanticExpressionKind.CONJUNCTION.value,
                       SemanticExpressionKind.CONDITIONAL.value):
        composite = 1
    for child in node.children:
        sub_atoms, sub_composite = _tally(child)
        atoms += sub_atoms
        composite += sub_composite
    return atoms, composite


def _supported(node: SemanticExpression) -> bool:
    if node.kind == SemanticExpressionKind.UNSUPPORTED.value:
        return False
    return all(_supported(c) for c in node.children)


def _collect_diagnostics(node: SemanticExpression) -> list:
    out = []
    if node.kind == SemanticExpressionKind.UNSUPPORTED.value:
        out.append(f"{node.raw_text!r}: {node.diagnosis}")
    for child in node.children:
        out.extend(_collect_diagnostics(child))
    return out


def _limited_node(raw: str, reason: str, state: dict) -> SemanticExpression:
    state["diagnostics"].append(f"{raw!r}: {reason}")
    return SemanticExpression(
        kind=SemanticExpressionKind.UNSUPPORTED.value, raw_text=raw,
        diagnosis=reason)


def _make_node(state: dict, max_nodes: int, raw: str,
               build) -> SemanticExpression:
    state["count"] += 1
    if state["count"] > max_nodes:
        return _limited_node(raw, "node budget exceeded", state)
    return build()


def _parse_atom(text: str, mem) -> SemanticExpression:
    utterance = parse_statement(text, mem)
    if utterance.ambiguity:
        return SemanticExpression(
            kind=SemanticExpressionKind.UNSUPPORTED.value, raw_text=text,
            diagnosis=f"ambiguous: {utterance.ambiguity}")
    if utterance.predicate not in ATOM_PREDICATES:
        return SemanticExpression(
            kind=SemanticExpressionKind.UNSUPPORTED.value, raw_text=text,
            diagnosis=f"predicate {utterance.predicate!r} not composable")
    return SemanticExpression(
        kind=SemanticExpressionKind.ATOM.value, raw_text=text,
        atom=utterance.to_dict())


def _parse_node(text: str, mem, depth: int, max_depth: int,
                max_nodes: int, state: dict) -> SemanticExpression:
    if depth > max_depth:
        return _limited_node(text, "depth limit exceeded", state)
    cond = _split_conditional(text)
    if cond is not None:
        antecedent, consequent, ok = cond
        if not ok:
            return _make_node(
                state, max_nodes, text,
                lambda: SemanticExpression(
                    kind=SemanticExpressionKind.UNSUPPORTED.value,
                    raw_text=text,
                    diagnosis="empty antecedent or consequent"))
        children = (
            _parse_node(antecedent, mem, depth + 1, max_depth, max_nodes,
                        state),
            _parse_node(consequent, mem, depth + 1, max_depth, max_nodes,
                        state),
        )
        used = _connectives_used(text)
        return _make_node(
            state, max_nodes, text,
            lambda: SemanticExpression(
                kind=SemanticExpressionKind.CONDITIONAL.value,
                raw_text=text, children=children,
                connective="如果…那么",
                roles=("antecedent", "consequent")))
    segments = _split_conjunction(text)
    if segments is not None:
        if any(not seg for seg in segments):
            return _make_node(
                state, max_nodes, text,
                lambda: SemanticExpression(
                    kind=SemanticExpressionKind.UNSUPPORTED.value,
                    raw_text=text,
                    diagnosis="empty segment around connective"))
        children = tuple(
            _parse_node(seg, mem, depth + 1, max_depth, max_nodes, state)
            for seg in segments)
        return _make_node(
            state, max_nodes, text,
            lambda: SemanticExpression(
                kind=SemanticExpressionKind.CONJUNCTION.value,
                raw_text=text, children=children,
                connective=_connectives_used(text)))
    return _make_node(
        state, max_nodes, text,
        lambda: _parse_atom(text, mem))


def _connectives_used(text: str) -> str:
    seen = []
    for _, tok in _scan_top_level(text, CONJUNCTIVES):
        if tok not in seen:
            seen.append(tok)
    return "+".join(seen)


def parse_compositional_statement(text: str, mem,
                                  max_depth: Optional[int] = None,
                                  max_nodes: Optional[int] = None
                                  ) -> SemanticParseResult:
    """Parse one statement into a recursive semantic tree.

    Conditional first, then conjunction, then the existing atom
    parser for leaves. Read-only: mem is only read by the atom
    parser, nothing is ever written.
    """
    if max_depth is None:
        max_depth = policy.MAX_SEMCOMPOSE_DEPTH
    if max_nodes is None:
        max_nodes = policy.MAX_SEMCOMPOSE_NODES
    raw = text if isinstance(text, str) else ""
    stripped = raw.strip()
    if not stripped:
        return SemanticParseResult(
            raw_input=raw, tree=None, complete=False,
            diagnostics=("empty input",), atom_count=0,
            composite_count=0)
    state: dict = {"count": 0, "diagnostics": []}
    tree = _parse_node(stripped, mem, 0, max_depth, max_nodes, state)
    diags = list(state["diagnostics"])
    diags.extend(d for d in _collect_diagnostics(tree)
                 if d not in diags)
    atoms, composite = _tally(tree)
    return SemanticParseResult(
        raw_input=raw, tree=tree,
        complete=_supported(tree), diagnostics=tuple(diags),
        atom_count=atoms, composite_count=composite)
