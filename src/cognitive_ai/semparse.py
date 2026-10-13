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


Q_WORDS = ("什么", "谁", "哪个", "哪些", "多少", "几", "是否",
           "怎么", "为什么")
Q_ENDS = ("吗", "呢")


def _has_qword(text: str) -> bool:
    return any(w in text for w in Q_WORDS)


def parse_question(text: str, mem: Memory):
    """Parse one question compositionally into ParsedQuestion.

    Same building blocks as statements (known-subject lookup, negation
    and modal tokens, heads), plus interrogative detection. A form that
    is not a supported question shape yields unknown/ambiguity with a
    reason -- never a guessed query, never an auto-created concept.
    """
    from .models import ParsedQuestion
    raw = text
    s = _strip_outer(text)
    q = ParsedQuestion(raw=raw)
    q.spans["input"] = s
    if not s:
        q.ambiguity = "empty input"
        return q
    subj_text, _subj_id, cand = _resolve_subject(mem, s)
    if cand and subj_text is None:
        q.ambiguity = ("subject span matches several concepts: " +
                       ", ".join(cand))
        q.unknown_spans.append(s)
        return q
    if subj_text is None:
        # unknown subject: marker-led split like statements, but the
        # question stays unanswerable until the subject is known.
        hit = None
        for mk in ("是不是", "能不能", "有没有", "不是", "并非",
                   "没有", "不能", "不会", "无法", "不可", "可以",
                   "能够", "能", "是", "有"):
            i = s.find(mk)
            if i > 0 and (hit is None or i < hit[0]):
                hit = (i, mk)
        if hit is None:
            q.unknown_spans.append(s)
            return q
        subj_text = s[:hit[0]].strip()
        if not subj_text:
            q.unknown_spans.append(s)
            return q
        q.subject = subj_text
        q.subject_known = False
        q.unknown_spans.append(subj_text)
        q.spans["subject"] = subj_text
        q.spans["subject_unresolved"] = True
        rest = s[hit[0]:].strip()
    else:
        q.subject = subj_text
        q.subject_known = True
        q.spans["subject"] = subj_text
        rest = s[len(subj_text):].strip()
    if not rest:
        q.ambiguity = "subject without question"
        return q
    negated, neg_kind, rest = _split_negation(rest)
    q.negated = negated
    if neg_kind:
        q.spans["negation"] = neg_kind
    modal = None
    for m in MODALS:
        if rest.startswith(m):
            modal = "can"
            q.spans["modal"] = m
            rest = rest[len(m):].strip()
            break
    if modal is None and neg_kind in ("不能", "不会", "无法", "不可"):
        modal = "can"
        q.spans["modal"] = neg_kind
    q.modal = modal
    # trailing 做/进行 light verbs after a modal carry no new role
    if modal is not None:
        for lv in ("做", "进行", "加以"):
            if rest.startswith(lv):
                q.spans["light_verb"] = lv
                rest = rest[len(lv):].strip()
                break
    named = NEG_PRED.get(neg_kind, "")
    # A-not-A verifications first (是不是/能不能/有没有 name both the
    # predicate and the yes/no form).
    m = _match_ana(rest)
    if m is not None:
        kind, body = m
        return _finish_question(q, mem, kind, body, True,
                                named or kind)
    # trailing question particles mark yes/no on complete propositions.
    qmark = False
    for end in Q_ENDS:
        if rest.endswith(end) and len(rest) > len(end):
            qmark = True
            q.spans["qmark"] = end
            rest = rest[:-len(end)].strip()
            break
    rest = _strip_particles(rest)
    import re as _re
    mposs = _re.match(r"^的(.+?)是什么$", rest)
    if mposs and mposs.group(1).strip():
        # possessive open question (X的材质是什么): checked before the
        # generic interrogative branch, which would otherwise swallow
        # it. The attribute word travels as the object; key mapping
        # happens at answer time.
        q.qkind = "attr_open"
        q.object = mposs.group(1).strip()
        q.spans["object"] = q.object
        q.unknown_spans.append(q.object)
        q.is_yesno = False
        return q
    if _has_qword(rest):
        return _finish_open(q, mem, rest, named, modal)
    if not rest:
        q.ambiguity = "marker without content"
        return q
    # a negated head that already named the kind (没有X / 不能X)
    # verifies that kind; bare negation without a named kind stays out.
    if q.negated and named in ("is_a", "has", "capable"):
        return _finish_question(q, mem, named, rest, True, named)
    # complete proposition (+optional 吗): yes/no verification only
    # when a question particle was actually seen; a bare assertion
    # (杯子可以盛水) is not a question at all.
    if not qmark:
        q.unknown_spans.append(rest)
        return q
    if rest.startswith(COPULA):
        return _finish_question(q, mem, "is_a",
                                _strip_particles(rest[1:].strip()),
                                True, named or "is_a")
    if rest.startswith(HAVE):
        return _finish_question(q, mem, "has",
                                _strip_particles(rest[1:].strip()),
                                True, named or "has")
    if modal is not None:
        return _finish_question(q, mem, "capable", rest, True,
                                named or "capable")
    q.unknown_spans.append(rest)
    return q


def _match_ana(rest: str):
    """A-not-A heads: (kind, body) or None. Generic slots only."""
    import re as _re
    m = _re.match(r"^是不是(.+)$", rest)
    if m:
        return ("is_a", m.group(1).strip())
    m = _re.match(r"^能不能(.+)$", rest)
    if m:
        return ("capable", m.group(1).strip())
    m = _re.match(r"^有没有(.+)$", rest)
    if m:
        return ("has", m.group(1).strip())
    return None


def _finish_open(q, mem, rest: str, named: str, modal):
    """Open (wh-) question from the interrogative remainder."""
    # X有什么功能/作用 -> capability; X有什么 -> open possession;
    # X是什么 -> category; X可以做什么 -> capability.
    for key, words in (("function", ("功能", "作用", "用途", "用处")),):
        for w in words:
            if rest == f"有什么{w}":
                q.qkind = "capability"
                q.key = "function"
                q.is_yesno = False
                q.spans["target"] = w
                return q
    if rest in ("有什么", "有啥", "有些什么"):
        q.qkind = "has"
        q.is_yesno = False
        return q
    if rest == "是什么":
        q.qkind = "category"
        q.is_yesno = False
        return q
    if modal is not None and rest in ("做什么", "干什么", "做些什么"):
        q.qkind = "capability"
        q.key = "function"
        q.is_yesno = False
        return q
    q.unknown_spans.append(rest)
    return q


def _finish_question(q, mem, kind: str, body: str, yesno: bool,
                     named: str):
    """Yes/no verification frame for a complete proposition."""
    body = _strip_particles(body.strip())
    if not body:
        q.ambiguity = "marker without content"
        return q
    if kind == "is_a":
        q.qkind = "is_a_verify"
    elif kind == "has":
        q.qkind = "has"
    elif kind == "capable":
        q.qkind = "capability"
    else:
        q.unknown_spans.append(body)
        return q
    q.object = body
    q.spans["object"] = body
    q.is_yesno = yesno
    if kind == "capable":
        q.key = "function"
    if kind == "has" and body in ("功能", "作用", "用途", "用处"):
        q.key = "function"
    q.unknown_spans.append(body)
    return q


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
        # unknown subject: try a marker-led split so X是Y / X有Y /
        # X可以Z with a new X still yields a structured frame (the
        # subject stays unknown and uncreated). Bare 不/没 are NOT
        # splitters here: too easy to cut inside unknown words.
        FB_MARKERS = ("是不是", "能不能", "有没有", "不是", "并非",
                      "没有", "不能", "不会", "无法", "不可", "可以",
                      "能够", "能", "是", "有")
        hit = None
        for mk in FB_MARKERS:
            i = s.find(mk)
            if i > 0 and (hit is None or i < hit[0]):
                hit = (i, mk)
        if hit is None:
            out.unknown_spans.append(s)
            return out
        i, mk = hit
        subj_text = s[:i].strip()
        if not subj_text:
            out.unknown_spans.append(s)
            return out
        out.subject = subj_text
        out.subject_known = False
        out.unknown_spans.append(subj_text)
        out.spans["subject"] = subj_text
        out.spans["subject_unresolved"] = True
        rest = s[i:].strip()
        if not rest:
            out.ambiguity = "subject without predication"
            return out
        return _parse_predicate(out, mem, rest, s)
    out.subject, out.subject_known = subj_text, True
    out.spans["subject"] = subj_text
    rest = s[len(subj_text):].strip()
    if not rest:
        out.ambiguity = "subject without predication"
        out.unknown_spans.append(s)
        return out
    return _parse_predicate(out, mem, rest, s)


def _parse_predicate(out: ParsedUtterance, mem: Memory, rest: str,
                     s: str) -> ParsedUtterance:
    """Shared predicate half: negation, modal, heads, object."""
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
