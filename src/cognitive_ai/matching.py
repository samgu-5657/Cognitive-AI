"""Shared unification kernel (M4-E-FIX-01). Pure, IO-free, deterministic.

One representation for both callers::

    KTerm = ("ref", id) | ("lit", value) | ("var", name) | ("bad", reason)

- Legacy reasoner slots (``"?x"`` strings, raw KB values) enter via
  :func:`legacy_term`: variables keep their name, everything else is
  a literal compared with ``==`` -- exactly the old engine's rule,
  including ``None`` and lone ``"?"``.
- Explicit world slots enter via :func:`pattern_term` /
  :func:`fact_term`: stable references, literals, or explicit
  variables. Empty or unresolvable slots become ``bad`` and never
  match; the two domains never mix inside one call.
- :func:`unify` does first binding, repeat-variable consistency and
  bound-candidate checks. The only caller-side policy is ``bindable``:
  the legacy engine binds variables to anything, the world layer only
  to stable references. Everything else is shared.

- Legacy reasoner slots (``"?x"`` strings, raw KB values) enter via
  :func:`legacy_term`, which treats every non-variable as a constant
  exactly like the old engine did (including ``None`` and lone ``"?"``).
- Explicit world slots enter via :func:`pattern_term` /
  :func:`fact_term`; empty or unresolvable slots become ``bad`` and
  never match.
- :func:`unify` does first binding, repeat-variable consistency and
  bound-candidate checks. The only caller-side policy is ``bindable``:
  the legacy engine binds variables to anything, the world layer only
  to stable references. Everything else is shared.

Slot semantics mirror the legacy ``_match_derived`` cascade exactly:
a constant pattern against an open (variable) fact constrains nothing;
an alias locked to one variable name rejects anything else, including
constants; a variable never binds to itself.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .models import WorldPatternTerm, WorldTerm


def is_variable_name(value: Any) -> bool:
    """Legacy ``"?x"`` variable predicate (exact old rule)."""
    return isinstance(value, str) and value.startswith("?") and \
        len(value) > 1


def legacy_term(value: Any) -> tuple:
    """Legacy slot value -> KTerm. Never invents, never drops."""
    if is_variable_name(value):
        return ("var", value[1:])
    return ("lit", value)


def pattern_term(term: WorldPatternTerm) -> tuple:
    """Explicit world pattern slot -> KTerm."""
    if term.is_variable():
        return ("var", term.variable)
    if term.ref is not None:
        return ("ref", term.ref.ref_id)
    if term.has_literal:
        return ("lit", term.literal)
    return ("bad", "empty pattern slot")


def fact_term(term: WorldTerm) -> tuple:
    """World fact slot -> KTerm. Unresolved/missing never match."""
    if term.ref is not None:
        return ("ref", term.ref.ref_id)
    if term.has_literal:
        return ("lit", term.literal)
    if term.unresolved:
        return ("bad", f"unresolved {term.unresolved!r}")
    return ("bad", "empty fact slot")


def resolve_legacy(bindings: dict, value: Any) -> Any:
    """Follow a legacy binding chain (exact old ``_resolve`` loop)."""
    while is_variable_name(value) and value in bindings:
        value = bindings[value]
    return value


def to_kernel_bindings(legacy_bindings: dict) -> dict:
    """``{"?x": raw}`` -> ``{"x": KTerm}`` for kernel threading."""
    out = {}
    for key, value in legacy_bindings.items():
        name = key[1:] if is_variable_name(key) else key
        out[name] = legacy_term(value)
    return out


def from_kernel_value(term: tuple) -> Any:
    """KTerm back to a legacy raw value."""
    if term[0] == "var":
        return "?" + term[1]
    return term[1]


def compatible(left: tuple, right: tuple) -> bool:
    """Check-only consistency: variables match anything, bad never."""
    if left[0] == "bad" or right[0] == "bad":
        return False
    if left[0] == "var" or right[0] == "var":
        return True
    return left == right


def unify(pattern: tuple, fact: tuple, bindings: dict,
          bindable: Optional[Callable[[tuple], bool]] = None):
    """Unify one pattern slot against one fact slot.

    Returns the extended bindings dict, the unchanged dict when the
    slot constrains nothing, or None on conflict. Never mutates its
    input. ``bindable`` gates first-time variable binding; the default
    allows anything (legacy engine semantics).
    """
    if pattern[0] == "bad" or fact[0] == "bad":
        return None
    if bindable is None:
        def bindable(_fact: tuple) -> bool:
            return True
    if pattern[0] == "var":
        name = pattern[1]
        if name in bindings:
            bound = bindings[name]
            if bound[0] == "var":
                if fact[0] == "var":
                    return bindings if bound[1] == fact[1] else None
                return None
            if fact[0] == "var":
                return bindings
            return bindings if bound == fact else None
        if fact[0] == "var" and fact[1] == name:
            return bindings
        if not bindable(fact):
            return None
        return {**bindings, name: fact}
    if fact[0] == "var":
        return bindings
    return bindings if pattern == fact else None
