"""Compositional Chinese statement parsing (Phase M3A).

One fixed, small inventory of composable pieces -- never a new regex
per sentence:

- subject: longest leading span matching a known concept id or name
  (memory lookup). Unresolvable subjects are kept as unknown spans.
- function words scanned as independent tokens: copula 是, possession
  有, capability modals 可以/能够/能, negation markers 不是/并非/
  没有/不能/不会/无法/不可/别/勿 and bare 不/没 directly before the
  predicate phrase, sentence particles.
- predicate from the marker found: is_a | has | capable | unspecified.
  Bare 不/没 yields an *unspecified* negated predication: the negation
  is recorded, the verbhood of the rest is NOT invented.
- object/value: raw text plus a known flag (known concept or not).
  Unknown words are kept in unknown_spans, never dropped, never filled.

Lexicalized traps (不锈钢 and the like) are safe because subject
matching runs first on whole known names, and bare 不/没 only fires
at the predicate boundary, never inside an already-matched span.
Ambiguity (several concepts for one span) and unparseable input yield
unknown/ambiguity instead of guesses.
"""
from __future__ import annotations

from .memory import Memory
from .models import ParsedUtterance

COPULA = "是"
HAVE = "有"
MODALS = ("可以", "能够", "能")
# negation markers tried longest-first. Markers that already name the
# predicate (不是->is_a, 没有->has, 不能/不会/无法/不可->capable) keep
# it: stripping them must not erase what kind of claim this is.
# Bare 不/没/别/勿 only deny without naming a predicate.
NEG_TOKENS = ("不是", "并非", "没有", "不能", "不会", "无法", "不可",
              "别", "勿")
NEG_PRED = {"不是": "is_a", "并非": "is_a", "没有": "has",
            "不能": "capable", "不会": "capable", "无法": "capable",
            "不可": "capable"}
BARE_NEG = ("不", "没")
PARTICLES = ("吗", "呢", "吧", "啊", "呀", "了", "着", "过", "的")
CLOSERS = ("。", "；", ";", "？", "?", "！", "!")


def _strip_outer(text: str) -> str:
    s = text.strip()
    while s and s[-1] in CLOSERS:
        s = s[:-1].strip()
    return s


def _known_spans(mem: Memory, text: str) -> list[tuple[str, str]]:
    """All (matched_text, concept_id) for known names in text."""
    hits = []
    try:
        concepts = mem.list_concepts()
    except Exception:
        return hits
    for c in concepts:
        for name in (c.get("id", ""), c.get("name", "")):
            if name and len(name) >= 1 and name in text:
                hits.append((name, c["id"]))
    return hits


def _resolve_subject(mem: Memory, text: str):
    """Longest leading known-concept span.

    Returns (matched_text|None, concept_id|None, [candidate_ids]).
    Several different concepts on the same span -> ambiguity.
    """
    cands: dict[str, set] = {}
    try:
        concepts = mem.list_concepts()
    except Exception:
        return None, None, []
    for c in concepts:
        for name in (c.get("id", ""), c.get("name", "")):
            if name and text.startswith(name):
                cands.setdefault(name, set()).add(c["id"])
    if not cands:
        return None, None, []
    longest = max(len(k) for k in cands)
    top = [k for k in cands if len(k) == longest]
    if len(top) > 1:
        ids: set = set()
        for k in top:
            ids |= cands[k]
        return None, None, sorted(ids)
    name = top[0]
    ids = sorted(cands[name])
    if len(ids) > 1:
        return None, None, ids
    return name, ids[0], ids


def _split_negation(rest: str):
    """Split leading negation marker. Returns (negated, kind, rest).

    kind mirrors the marker family so the caller can keep modal
    information (不能 still carries 可以-style capability meaning).
    Bare 不/没 only count directly before the predicate phrase.
    """
    for tok in NEG_TOKENS:
        if rest.startswith(tok):
            return True, tok, rest[len(tok):].strip()
    for tok in BARE_NEG:
        if rest.startswith(tok):
            return True, tok, rest[len(tok):].strip()
    return False, "", rest


def _strip_particles(text: str) -> str:
    s = text.strip()
    changed = True
    while changed and s:
        changed = False
        for p in PARTICLES:
            if s.endswith(p) and len(s) > len(p):
                s = s[:-len(p)].strip()
                changed = True
    return s


def parse_statement(text: str, mem: Memory) -> ParsedUtterance:
    """Parse one simple statement compositionally. Never guesses."""
    raw = text
    s = _strip_outer(text)
    if not s:
        return ParsedUtterance(raw=raw, ambiguity="empty input")
    out = ParsedUtterance(raw=raw)
    out.spans["input"] = s

    subj_text, subj_id, cand = _resolve_subject(mem, s)
    if cand and subj_text is None:
        out.ambiguity = ("subject span matches several concepts: " +
                         ", ".join(cand))
        out.unknown_spans.append(s)
        return out
    if subj_text is None:
        out.unknown_spans.append(s)
        return out
    out.subject, out.subject_known = subj_text, True
    out.spans["subject"] = subj_text
    rest = s[len(subj_text):].strip()
    if not rest:
        out.ambiguity = "subject without predication"
        out.unknown_spans.append(s)
        return out

    negated, neg_kind, rest = _split_negation(rest)
    out.negated = negated
    if neg_kind:
        out.spans["negation"] = neg_kind
    rest = _strip_particles(rest)

    # capability modal comes next when present (kept even under denial:
    # 不能 still asks about ability, it does not become a bare verb).
    modal = None
    for m in MODALS:
        if rest.startswith(m):
            modal = "can"
            out.spans["modal"] = m
            rest = rest[len(m):].strip()
            break
    if modal is None and neg_kind in ("不能", "不会", "无法", "不可"):
        modal = "can"
        out.spans["modal"] = neg_kind
    out.modal = modal
    rest = _strip_particles(rest)
    if not rest:
        out.ambiguity = "marker without content"
        out.unknown_spans.append(s)
        return out

    # copula / possession heads (only when they lead the remainder, so
    # values containing 是/有 are never split). A negation marker that
    # already named the predicate (不是/没有/不能…) keeps it: stripping
    # must not erase what kind of claim this is.
    named = NEG_PRED.get(neg_kind, "")
    if rest.startswith(COPULA) or named == "is_a":
        out.predicate = "is_a"
        body = rest[len(COPULA):].strip() if rest.startswith(COPULA) \
            else rest
        obj = _strip_particles(body)
        out.spans["copula"] = COPULA
    elif (rest.startswith(HAVE) or named == "has") and not any(
            rest.startswith(w) for w in ("意思", "有趣", "有空", "有点",
                                         "有些", "有所")):
        out.predicate = "has"
        body = rest[len(HAVE):].strip() if rest.startswith(HAVE) else rest
        obj = _strip_particles(body)
        out.spans["possession"] = HAVE
    elif modal is not None or named == "capable":
        out.predicate = "capable"
        obj = rest
        out.spans["capability"] = rest
    else:
        # no recognizable predicate head: keep the negation that was
        # found, but do not invent a verb reading for the rest.
        out.predicate = "unspecified"
        obj = rest
        out.unknown_spans.append(rest)
    if not obj:
        out.ambiguity = "predicate without object"
        out.unknown_spans.append(s)
        return out
    out.object = obj
    out.spans["object"] = obj
    hits = _known_spans(mem, obj)
    out.object_known = any(t == obj for t, _ in hits)
    if not out.object_known:
        # keep even unknown values; only the span text is new, the
        # fact that something is unknown is explicit.
        out.unknown_spans.append(obj)
    # order-preserving dedup: the same span may be noted twice (e.g.
    # once as unparsed rest, once as unknown value).
    out.unknown_spans = list(dict.fromkeys(out.unknown_spans))
    return out
