"""Compositional Chinese understanding (Phase 4B).

Produces SemanticFrame candidates from natural input WITHOUT pretrained
models: a small function-word lexicon + slot-based compositional patterns
({NP}是{NP}, {NP}不是{NP}做的, 如果{COND}…{EFF}…), never whole-sentence
hardcodes. Paraphrase tests prove generality.

A frame is ALWAYS a candidate: modality/negation/hedge are preserved, and
nothing here writes verified facts (callers route through observation /
hypothesis / correction / research flows).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

# ---- lexicons (closed function words, not sentence templates) ----
# Negation as STRUCTURAL patterns (lookahead), never bare substrings:
# bare 不 would misfire inside 不锈钢/不同. Same for bare 没 (没有 handled).
# 不+verb complements cover common verb negations (不吃/不来/...).
NEG_VERBS = "吃喝长爱来去买看听说读写跑走飞游打做用睡醒哭笑飞"
NEG_PATTERNS = (r"不是", r"没有", r"并非", r"别", r"勿", r"无法",
                r"不(?=是|有|可以|能|会|属于|对|行|用|[" + NEG_VERBS + r"])",
                r"没(?!有)")
HEDGES = ("可能", "也许", "或许", "好像", "似乎", "我觉得", "我认为",
          "大概", "应该", "八成")
CORRECTION_MARKS = ("刚才说错了", "说错了", "不对", "其实", "更正", "纠正")
GENERIC_MARKS = ("通常", "一般", "往往", "都是", "总是")
INSTANCE_MARKS = ("我买了", "我有一个", "我刚买了", "今天买了", "这个",
                  "我这个", "我家的")
PRONOUNS = ("它", "他", "她", "它们", "这个", "那个", "这", "那")
COND_MARKS = ("如果", "要是", "假如")
COND_TAILS = ("就", "那么", "会", "可能")
QUERY_MARKS_END = ("吗", "呢")
QUERY_MARKS_ANY = ("什么", "为什么", "是否为", "是不是", "多少",
                   "哪个", "是否")

# Chinese attribute words -> canonical KB keys (unknown words pass through
# raw with an ambiguity flag; never silently dropped).
# Leading quantifiers stripped from NP slots (一个/一种/...), deictics kept.
# Leading politeness markers carry no meaning (请问X... asks about X):
# stripped for slot matching ONLY, always logged in the frame trace.
POLITENESS_LEAD = ("麻烦问一下", "请问一下", "请教一下", "打扰一下",
                   "请问")
QUESTION_WORDS = ("什么", "谁", "哪个", "哪些", "多少", "几", "是否",
                  "是否为", "是不是", "能否", "能不能", "有没有", "吗")
QUANTIFIERS = ("一种", "一类", "一个", "一只", "一位", "一些", "许多",
               "很多", "一辆", "一台", "一架", "一部", "一艘", "一匹",
               "一张", "一件", "一条", "一棵", "一朵", "一本", "一支",
               "一块", "一片", "一杯", "一碗", "一瓶", "一把", "一顶",
               "一双", "一套", "一群", "一堆")
ATTR_WORDS = {
    "材质": "material", "材料": "material", "质地": "material",
    "质料": "material", "做": "material", "做的": "material",
    "功能": "function", "用途": "function", "用处": "function",
    "颜色": "color", "形状": "shape", "位置": "location",
    "产地": "location", "温度": "temperature", "尺寸": "size",
    "大小": "size", "重量": "weight",
}

INTENTS = ("classify", "attribute", "function", "negate", "hedge",
           "correct", "causal", "observe", "query", "unknown")

MODALITIES = ("assert", "negate", "possible", "hypothesis", "correction",
              "conditional", "unknown")


@dataclass
class SemanticFrame:
    intent: str = "unknown"
    entities: list = field(default_factory=list)
    predicates: list = field(default_factory=list)
    modality: str = "unknown"
    scope: str = "unknown"  # generic | instance | unknown
    source: str = "user"
    confidence: float = 0.0
    unresolved_references: list = field(default_factory=list)
    ambiguities: list = field(default_factory=list)
    raw: str = ""
    trace: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _strip_punct(s: str) -> str:
    return s.strip(" \t\n。！？；;!?，,、")


def _has_any(s: str, words) -> str | None:
    for w in words:
        if w in s:
            return w
    return None


def _ent(text: str, role: str) -> dict:
    return {"text": text.strip(), "role": role}


def _clean_np(np: str) -> str:
    np = np.strip()
    for q in sorted(QUANTIFIERS, key=len, reverse=True):
        if np.startswith(q):
            np = np[len(q):].strip()
            break
    return np


def _strip_qmark(np: str) -> str:
    """Drop ONE trailing interrogative particle from a slot value.

    吗/呢 mark the sentence as a question (already recorded); leaving
    them inside feature/action slots (把手吗) would poison lexicon
    lookups and KB keys. Trailing-only, logged by callers via trace."""
    return re.sub(r"(吗|呢)$", "", np.strip())


def parse_sentence(sentence: str, source: str = "user",
                   lexicon: dict | None = None,
                   phrase_rules: list[dict] | None = None) -> SemanticFrame:
    """Compositional parse. Slot patterns only; no sentence hardcodes.

    lexicon: learned word->key mappings (experience first, built-ins
    as fallback). None behaves exactly like the built-in map.
    phrase_rules: ACTIVE learned rewrite rules (Phase 6C). They apply
    ONLY when the direct parse is unknown (fallback): a directly parsed
    sentence is never reshaped. At most one rewrite; ambiguous matches
    abstain with an explicit note.
    """
    raw = sentence
    s = _strip_punct(sentence)
    fr = SemanticFrame(raw=raw, source=source)
    if not s:
        fr.trace.append("empty input")
        return fr
    if phrase_rules:
        # Learned rewrites are strictly a FALLBACK: a directly parseable
        # sentence is never reshaped. Consult the direct (rule-free) parse
        # first; only an unknown direct result may use one rewrite.
        from .phraselearn import apply_rules
        direct = parse_sentence(s, source, lexicon, None)
        if direct.intent != "unknown":
            # Task-3 gate: a doubtful direct parse (parser recorded its
            # own ambiguities) may defer to a clean learned rewrite;
            # confident parses are never overridden.
            if direct.ambiguities:
                from .phraselearn import consider_learned_override
                alt = consider_learned_override(
                    direct.to_dict(), s, phrase_rules, lexicon or {})
                if alt is not None:
                    fr2 = SemanticFrame(raw=raw, source=source)
                    for k, v in alt.items():
                        if hasattr(fr2, k):
                            setattr(fr2, k, v)
                    fr2.raw = raw
                    return fr2
            return direct
        tried = apply_rules(s, phrase_rules, lexicon or {})
        if tried.get("rewritten"):
            inner = parse_sentence(tried["rewritten"], source,
                                   lexicon, None)
            if inner.intent != "unknown":
                inner.trace.append(
                    f"phrase_rule:{tried['rule_id']} rewrote "
                    f"{s!r} -> {tried['rewritten']!r}")
                inner.confidence = round(max(0.0, inner.confidence - 0.05),
                                         3)
                inner.raw = raw
                return inner
            fr.trace.append(f"phrase_rule:{tried['rule_id']} rewrite "
                            "did not parse; kept direct result")
        elif tried.get("reason") == "ambiguous: rules disagree":
            fr.ambiguities.append(
                f"learned rewrites disagree {tried.get('rules')}; abstaining")
            fr.trace.append("phrase_rule ambiguity -> no rewrite")
    # 1. interrogative? 吗/呢 only sentence-finally (的话呢哈哈 is not a
    # question); content question words count anywhere. Recorded FIRST but
    # applied LAST: slots still parse (X的材质是什么 keeps entities),
    # the question form only shapes the final intent.
    interrogative = s.endswith(QUERY_MARKS_END) or \
        _has_any(s, QUERY_MARKS_ANY)
    if interrogative:
        fr.trace.append("interrogative form noted")
    # 2. correction? (handled before modality so 其实 inside corrections wins)
    mark = _has_any(s, CORRECTION_MARKS)
    if mark:
        rest = s.split(mark, 1)[1]
        # further discourse markers inside are stripped, not nested:
        # 其实 introduces the corrected content, not another correction
        clean = rest
        for m in CORRECTION_MARKS:
            clean = clean.replace(m, "")
        clean = clean.strip()
        fr.intent, fr.modality = "correct", "correction"
        fr.trace.append(f"correction marker: {mark}")
        inner = parse_sentence(clean, source) if clean else None
        fr.predicates.append({"type": "correction", "mark": mark,
                              "content": inner.to_dict() if inner else None})
        fr.confidence = 0.8 if clean else 0.4
        if not clean:
            fr.ambiguities.append("missing corrected content")
        else:
            fr.entities = inner.entities
            fr.unresolved_references = inner.unresolved_references
        return fr
    # 3. conditional?
    cmark = _has_any(s, COND_MARKS)
    if cmark:
        rest = s.split(cmark, 1)[1]
        cond, eff = rest, ""
        # comma-first split (如果下雨，地面就会湿): condition is the
        # clause before ，; otherwise fall back to 就/那么
        mcomma = re.search(r"[，,]", rest)
        mj = re.search(r"就|那么", rest)
        if mcomma and (not mj or mcomma.start() < mj.start()):
            cond, eff = rest[:mcomma.start()], rest[mcomma.end():]
            eff = re.sub(r"^(就|那么)+", "", eff).strip()
            cond = cond.strip()
        else:
            for tail in ("就", "那么"):
                if tail in rest:
                    cond, eff = rest.split(tail, 1)
                    break
        fr.intent, fr.modality, fr.scope = \
            "causal", "conditional", "generic"
        fr.predicates.append({"type": "causal", "condition": cond.strip(),
                              "effect": eff.strip()})
        fr.trace.append(f"conditional: {cmark}")
        fr.confidence = 0.7 if eff.strip() else 0.4
        if not eff.strip():
            fr.ambiguities.append("missing effect clause")
        return fr
    # 4. modality scan: structural negation (never bare substrings),
    # then hedges; generic/instance scope markers. 不可能/没可能 is
    # negation, not hedging; otherwise a hedge scopes widest
    # (大概不会来 = possibly-not, not a flat denial).
    # Leading conditional-time adjuncts (在…的时候/时/情况下，) scope the
    # whole sentence and carry no slot meaning: lift them here (before
    # the modality scan) so 没有 inside them cannot fake a denial.
    # The raw text keeps them; the strip is logged in the trace.
    m_when = re.match(r"^在(.+?)(的时候|时|情况下)[，,、]?", s)
    if m_when and m_when.end() < len(s):
        s = s[m_when.end():].strip()
        fr.trace.append(f"time-adjunct stripped: {m_when.group(0)!r}")
    neg = any(re.search(p, s) for p in NEG_PATTERNS)
    hedge = _has_any(s, HEDGES)
    if re.search(r"(不|没|别)可能", s):
        neg, hedge = True, False
    generic = _has_any(s, GENERIC_MARKS)
    instance = _has_any(s, INSTANCE_MARKS)
    fr.scope = "generic" if generic else ("instance" if instance else "unknown")
    # 5. compositional slot patterns (generic, longest first).
    # Hedge/scope words are lifted out for slot matching ONLY (modality
    # already recorded above): 可能是陶土做的 -> 是陶土做的; 都是 carries
    # the copula so it becomes 是 instead of vanishing.
    # Leading politeness markers likewise (请问+X的材质是什么 asks
    # about X的材质是什么): raw text stays in fr.raw, strip is logged.
    # Never for meaning -- bare 请问 with nothing after stays unknown.
    slots = s.replace("都是", "是")
    # Generic modal/discourse normalization for slot matching ONLY:
    # 能够->能 (same modal), leading 并/且/而/还 (clause glue).
    slots = slots.replace("能够", "能")
    slots = re.sub(r"^(并|且|而|还)+", "", slots)
    for h in sorted(list(HEDGES) + [g for g in GENERIC_MARKS if g != "都是"],
                    key=len, reverse=True):
        slots = slots.replace(h, "")
    for p in sorted(POLITENESS_LEAD, key=len, reverse=True):
        if slots.startswith(p):
            slots = slots[len(p):]
            fr.trace.append(f"politeness stripped: {p}")
            break
    # Leading 那/那么 before a pronoun is discourse glue (那它… means
    # 它…): strip it for slot matching so reference resolution sees the
    # real pronoun. Logged; bare 那… with no pronoun behind stays as-is.
    m_na = re.match(r"^(那么|那)(?=[它他她])", slots)
    if m_na:
        slots = slots[m_na.end():]
        fr.trace.append("discourse 那 stripped before pronoun")
    pats = [
        # ---- question templates (generic NP slots, interrogative first)
        # X有什么用途/功能/用处 -> query function
        (re.compile(r"^(?P<a>.+?)有什[么麽](?P<b>用途|功能|用处|作用)$"),
         "query_function"),
        # X可以拿来干什么 / 拿X能干什么 -> query function
        (re.compile(r"^(?P<a>.+?)可以拿来(?P<b>干什么|做什么)$"),
         "query_function"),
        (re.compile(r"^拿(?P<a>.+?)能(?P<b>干什么|做什么)$"),
         "query_function"),
        # 用X通常能做什么 / 用X能干什么 -> query function
        (re.compile(r"^用(?P<a>.+?)(通常)?能(?P<b>干什么|做什么|什么)$"),
         "query_function"),
        # X是干什么用的 / X是做什么的 -> query function
        (re.compile(r"^(?P<a>.+?)是(?P<b>干什么|做什么)用的?$"),
         "query_function"),
        # NOTE (post-6D fix): "为什么人们会使用X" is a WHY-question,
        # not a function question. It falls through to the generic
        # query_reason pattern below and abstains honestly without
        # causal evidence -- never borrows the function answer.
        # 为什么X会V -> query reason
        (re.compile(r"^为什么(?P<a>.+?)会(?P<b>.+?)$"), "query_reason"),
        # ---- topic introduction (generic object slot, no hardcodes).
        # 我想了解X / 介绍一下X names X the current topic; the dialogue
        # layer answers from the concept record (read-only overview).
        (re.compile(r"^(?:我想了解|我想知道|介绍一下|讲讲|说说|聊聊)"
                     r"(?P<b>.+?)$"),
         "topic"),
        # ---- A-not-A verification questions (yes/no about KB facts).
        # Generic slots, NOT per-sentence hardcodes. Placed BEFORE every
        # negate pattern: 是不是 contains 不是 as a substring, and 不+是
        # fires the negation scanner, but A-not-A asks -- it never denies.
        # X是不是Y做的 -> verify material=Y about X
        (re.compile(r"^(?P<a>.+?)是不是(?P<b>.+?)"
                     r"(做|制成|构成|组成|所制)的?(吗)?$"),
         "verify_attr"),
        # X是Y色吗 / X是Y的吗 (short Y) -> verify attribute about X.
        # Same generic key rule as attr_copula (color/material/learned
        # mapping); unclear keys stay unverifiable, never guessed.
        (re.compile(r"^(?P<a>.+?)是(?P<b>.+?)(色|的)吗$"),
         "verify_attr_cop"),
        # X是不是Y -> verify class=Y about X
        (re.compile(r"^(?P<a>.+?)是不是(?P<b>.+?)(吗)?$"),
         "verify_class"),
        # X有没有Y -> verify possession of Y
        (re.compile(r"^(?P<a>.+?)有没有(?P<b>.+?)(吗)?$"), "verify_has"),
        # X能不能V -> verify capability V
        (re.compile(r"^(?P<a>.+?)能不能(?P<b>.+?)(吗)?$"),
         "verify_capable"),
        # X不是Y做的 / X不是金属的 / X并非Y所制 -> negate attribute
        (re.compile(r"^(?P<a>.+?)(?:不是|并非)(?P<b>.+?)"
                    r"(做|制成|构成|所制)的?$"),
         "negate_attr"),
        # X没有Y -> negate possession
        (re.compile(r"^(?P<a>.+?)没有(?P<b>.+?)$"), "negate_has"),
        # X不/没长Y -> negate possession (lacks)
        (re.compile(r"^(?P<a>.+?)(?:不|没)长(?P<b>.+?)$"), "negate_has"),
        # X无法V -> negate capability
        (re.compile(r"^(?P<a>.+?)无法(?P<b>.+?)$"), "negate_capable"),
        # X是Y做的 / X由Y制成 -> attribute (material).
        # The copula must not be the 是 inside 可能是/可是.
        (re.compile(r"^(?P<a>.+?)((?<!可能)(?<!可)是|由)(?P<b>.+?)"
                    r"(做|制成|构成|组成|所制)的?$"),
         "made_of"),
        # X不是Y / X并非Y / X不属于Y -> negate classify
        (re.compile(r"^(?P<a>.+?)(?:不是|并非|不属于)(?P<b>.+?)$"),
         "negate_class"),
        # X是Y色/ X是Y的 (attribute copula; key inferred or left open)
        (re.compile(r"^(?P<a>.+?)((?<!可能)(?<!可)是)(?P<b>.+?)(色|的)$"),
         "attr_copula"),
        # X是Y / X属于Y -> classify (same copula guard)
        (re.compile(r"^(?P<a>.+?)((?<!可能)(?<!可)是|属于)(?P<b>.+?)$"),
         "is_a"),
        # 我买了X / 我有一个X / 我家有X -> instance observation.
        # Specific BEFORE generic 有 (我家有旧椅子 must not become has).
        # 我有一… also matches abstract possessions (我有忧愁): recorded as
        # episodic observations, never verified facts (documented trade-off).
        (re.compile(r"^(?:我(?:今天|刚)?买了|我有一个|我有一|今天买了|我家有|家里有)"
                    r"(?P<b>.+?)$"),
         "got_instance"),
        # X有Y -> possession/feature. 有 inside lexicalized words
        # (有意思/有趣/有空/有点/有些/有所) is not possession.
        (re.compile(r"^(?P<a>.+?)有(?!意思|趣|空|点|些|所)(?P<b>.+?)$"),
         "has"),
        # X可以V / X能V / X用来V -> function
        (re.compile(r"^(?P<a>.+?)(可以|能|用来|用于|适合)(?P<b>.+?)$"),
         "capable"),
        # X会V (2+ char complement; bare 会议/误会 must not match)
        (re.compile(r"^(?P<a>.+?)会(?P<b>.{2,})$"), "capable"),
        # X放在哪里 / X在哪里 -> location question
        (re.compile(r"^(?P<a>.+?)(放在哪里|在哪里|在哪儿|位于哪里)$"),
         "where"),
        # 我买了X / 我有一个X / 我家有X -> instance observation
        (re.compile(r"^(?:我(?:今天|刚)?买了|我有一个|今天买了|我家有|家里有)"
                    r"(?P<b>.+?)$"),
         "got_instance"),
    ]
    for rx, kind in pats:
        m = rx.match(slots)
        if not m:
            continue
        fr.trace.append(f"pattern: {kind}")
        _fill(fr, kind, m, neg=neg, hedge=hedge, lexicon=lexicon)
        break
    else:
        # honest fallback: a lone hedge/negation marker still tells us the
        # modality, but contributes no structure (never fabricated slots)
        if hedge:
            fr.intent, fr.modality, fr.confidence = \
                "hedge", "possible", 0.4
            fr.ambiguities.append("hedged content did not match a pattern")
            fr.trace.append("hedge fallback (no structure claimed)")
        elif neg:
            fr.intent, fr.modality, fr.confidence = \
                "negate", "negate", 0.4
            fr.ambiguities.append("negated content did not match a pattern")
            fr.trace.append("negation fallback (no structure claimed)")
        else:
            fr.trace.append("no pattern matched")
    # interrogative applied LAST: a parsed statement-question keeps its
    # slots and becomes a query intent (X的材质是什么 asks, not asserts)
    if interrogative and fr.intent in ("classify", "attribute", "function"):
        fr.intent = "query"
        fr.trace.append("interrogative: statement intent -> query")
        fr.confidence = round(max(0.0, fr.confidence - 0.1), 3)
    elif interrogative and fr.intent == "unknown":
        fr.intent, fr.modality = "query", "unknown"
        fr.trace.append("interrogative without slots")
        fr.confidence = 0.5
    # 6. pronouns -> unresolved (resolution happens against context later).
    # Flag a pronoun unless its only occurrences sit inside longer entity
    # words (e.g. 这个 inside 这个壶 is a determiner, not a bare reference).
    # A 那 consumed as discourse glue (trace "discourse 那 stripped") is
    # not a reference either: only the pronoun behind it counts.
    glue_na = any(t.startswith("discourse 那 stripped")
                  for t in fr.trace)
    for p in PRONOUNS:
        if p == "那" and glue_na:
            continue
        if p not in s:
            continue
        buried = any(p in e["text"] and len(e["text"]) > len(p)
                     for e in fr.entities)
        if not buried and p not in fr.unresolved_references:
            fr.unresolved_references.append(p)
    # 7. confidence from fill completeness
    if fr.intent != "unknown":
        fr.confidence = max(0.0, fr.confidence
                            - 0.15 * (len(fr.unresolved_references)
                                      + len(fr.ambiguities)))
        fr.confidence = round(fr.confidence, 3)
    return fr


def _fill(fr: SemanticFrame, kind: str, m, neg, hedge,
            lexicon: dict | None = None) -> None:
    a = m.groupdict().get("a", "").strip() if "a" in m.groupdict() else ""
    b = m.groupdict().get("b", "").strip()
    # discourse fillers at subject start carry no meaning
    a = _strip_punct(re.sub(r"^(一下子|一下|请)+", "", a))
    if a:
        fr.entities.append(_ent(a, "subject"))
    modality = "possible" if hedge else ("negate" if neg else "assert")
    if kind == "is_a":
        fr.entities.append(_ent(_clean_np(b), "object"))
        fr.predicates.append({"type": "is_a", "subj": a, "obj": _clean_np(b)})
        if modality == "negate":
            fr.intent, fr.modality = "negate", "negate"
            fr.confidence = 0.85
        elif modality == "possible":
            fr.intent, fr.modality = "hedge", "possible"
            fr.confidence = 0.7
        else:
            fr.intent, fr.modality = "classify", "assert"
            fr.confidence = 0.9
    elif kind == "negate_class":
        fr.entities.append(_ent(_clean_np(b), "object"))
        fr.predicates.append({"type": "is_a", "subj": a, "obj": _clean_np(b),
                              "denied": True})
        fr.intent, fr.modality, fr.confidence = "negate", "negate", 0.85
    elif kind == "made_of":
        key = _canon_attr("材质", lexicon)
        fr.predicates.append({"type": "attr", "subj": a, "key": key,
                              "value": b})
        if modality == "negate":
            fr.intent, fr.modality = "negate", "negate"
            fr.predicates[-1]["denied"] = True
            fr.confidence = 0.85
        elif modality == "possible":
            fr.intent, fr.modality = "hedge", "possible"
            fr.confidence = 0.7
        else:
            fr.intent, fr.modality = "attribute", "assert"
            fr.confidence = 0.85
    elif kind == "query_function":
        fr.entities.append(_ent(a, "subject"))
        fr.predicates.append({"type": "query_function", "subj": a})
        fr.intent, fr.modality = "query", "unknown"
        fr.confidence = 0.8
    elif kind == "topic":
        subj = _clean_np(b)
        fr.entities.append(_ent(subj, "subject"))
        fr.predicates.append({"type": "query_topic", "subj": subj})
        fr.intent, fr.modality = "query", "unknown"
        fr.confidence = 0.8
    elif kind == "verify_attr":
        key = _canon_attr("材质", lexicon)
        fr.entities.append(_ent(a, "subject"))
        fr.predicates.append({"type": "verify_attr", "subj": a, "key": key,
                              "expected": b})
        fr.intent, fr.modality = "query", "unknown"
        fr.confidence = 0.8
        fr.trace.append("A-not-A question: asks, never denies")
    elif kind == "verify_attr_cop":
        if b.endswith("色"):
            key = "color"
        elif len(b) <= 4:
            key = "material"
        elif lexicon and b in lexicon:
            key = lexicon[b]
        else:
            key = None
        fr.entities.append(_ent(a, "subject"))
        fr.predicates.append({"type": "verify_attr", "subj": a, "key": key,
                              "expected": b})
        fr.intent, fr.modality = "query", "unknown"
        fr.confidence = 0.8
        if key is None:
            fr.ambiguities.append(f"attribute key for '{b}' unclear; "
                                  "cannot verify")
        fr.trace.append("A-not-A question: asks, never denies")
    elif kind == "verify_class":
        fr.entities.append(_ent(a, "subject"))
        fr.entities.append(_ent(_clean_np(b), "object"))
        fr.predicates.append({"type": "verify_class", "subj": a,
                              "expected": _clean_np(b)})
        fr.intent, fr.modality = "query", "unknown"
        fr.confidence = 0.8
        fr.trace.append("A-not-A question: asks, never denies")
    elif kind == "verify_has":
        feat = _strip_qmark(_clean_np(b))
        if lexicon and feat in lexicon:
            feat = lexicon[feat]
        fr.entities.append(_ent(a, "subject"))
        fr.predicates.append({"type": "verify_has", "subj": a,
                              "feature": feat})
        fr.intent, fr.modality = "query", "unknown"
        fr.confidence = 0.8
        fr.trace.append("A-not-A question: asks, never denies")
    elif kind == "verify_capable":
        fr.entities.append(_ent(a, "subject"))
        fr.predicates.append({"type": "verify_capable", "subj": a,
                              "action": _strip_qmark(b)})
        fr.intent, fr.modality = "query", "unknown"
        fr.confidence = 0.8
        fr.trace.append("A-not-A question: asks, never denies")
    elif kind == "query_reason":
        fr.entities.append(_ent(a, "subject"))
        fr.predicates.append({"type": "query_reason", "subj": a,
                              "about": b})
        fr.intent, fr.modality = "query", "unknown"
        fr.confidence = 0.75
    elif kind == "negate_attr":
        fr.predicates.append({"type": "attr", "subj": a,
                              "key": _canon_attr("材质", lexicon), "value": b,
                              "denied": True})
        fr.intent, fr.modality, fr.confidence = "negate", "negate", 0.85
    elif kind in ("negate_has",):
        fr.predicates.append({"type": "has", "subj": a,
                              "feature": _strip_qmark(_clean_np(b)),
                              "denied": True})
        fr.intent, fr.modality, fr.confidence = "negate", "negate", 0.8
    elif kind == "negate_capable":
        fr.predicates.append({"type": "capable", "subj": a,
                              "action": _strip_qmark(b),
                              "denied": True})
        fr.intent, fr.modality, fr.confidence = "negate", "negate", 0.8
    elif kind == "attr_copula":
        # X是Y色 -> color=Y; X是Y的 (short Y) -> material=Y (X是铁的);
        # learned word mappings resolve longer values; otherwise the key
        # stays open with an explicit ambiguity.
        if b.endswith("色"):
            key, val, amb = "color", b, None
        elif len(b) <= 4:
            key, val, amb = "material", b, None
        elif lexicon and b in lexicon:
            key, val, amb = lexicon[b], b, None
        else:
            key, val, amb = None, b, (f"attribute key for '{b}' unclear; "
                                      "needs mapping")
        pred = {"type": "attr", "subj": a, "key": key, "value": val}
        fr.predicates.append(pred)
        if modality == "negate":
            fr.intent, fr.modality = "negate", "negate"
            pred["denied"] = True
            fr.confidence = 0.8
        elif modality == "possible":
            fr.intent, fr.modality = "hedge", "possible"
            fr.confidence = 0.65
        else:
            fr.intent, fr.modality = "attribute", "assert"
            fr.confidence = 0.8
            if amb:
                fr.ambiguities.append(amb)
    elif kind == "has":
        feat = _strip_qmark(_clean_np(b))
        if lexicon and feat in lexicon:
            feat = lexicon[feat]
        fr.predicates.append({"type": "has", "subj": a, "feature": feat})
        if modality == "negate":
            fr.intent, fr.modality = "negate", "negate"
            fr.predicates[-1]["denied"] = True
            fr.confidence = 0.8
        elif modality == "possible":
            fr.intent, fr.modality = "hedge", "possible"
            fr.confidence = 0.65
        else:
            fr.intent, fr.modality = "attribute", "assert"
            fr.confidence = 0.8
    elif kind == "capable":
        fr.predicates.append({"type": "capable", "subj": a,
                              "action": _strip_qmark(b)})
        if modality == "negate":
            fr.intent, fr.modality = "negate", "negate"
            fr.predicates[-1]["denied"] = True
            fr.confidence = 0.8
        elif modality == "possible":
            fr.intent, fr.modality = "hedge", "possible"
            fr.confidence = 0.65
        else:
            fr.intent, fr.modality = "function", "assert"
            fr.confidence = 0.85
    elif kind == "got_instance":
        fr.entities.append(_ent(_clean_np(b), "object"))
        fr.predicates.append({"type": "new_instance",
                              "obj": _clean_np(b)})
        fr.intent, fr.modality = "observe", "assert"
        if fr.scope == "unknown":
            fr.scope = "instance"
        fr.confidence = 0.85
    elif kind == "where":
        fr.predicates.append({"type": "query_attr", "subj": a,
                              "key": "location"})
        fr.intent, fr.modality = "query", "unknown"
        fr.confidence = 0.85


def extract_definition(text: str, lexicon: dict | None = None) -> dict | None:
    """Split a concept-introduction sentence (Phase 6G-1).

    Generic head pattern only: [我/新][定义/发明/创造/设计/制作/做]
    [了]一种叫<name>的<rest>. <rest> splits at the first clause boundary
    into a category word and raw statement clauses. Each clause is parsed
    separately by the caller (capable->function, 是->category, 有->
    attribute); nothing here invents meanings. Returns None when the
    head does not match.
    """
    s = _strip_punct(text.strip())
    m = re.match(r"^(?:我|新)?(?:定义|发明|创造|设计|制作|做)?了?"
                 r"一种叫(?P<name>.+?)的(?P<rest>.+)$", s)
    if not m:
        return None
    name = _clean_np(m.group("name").strip())
    rest = m.group("rest").strip()
    if not name or not rest:
        return None
    parts = [p.strip() for p in re.split(r"[，。；;]", rest) if p.strip()]
    category, clauses = (parts[0], parts[1:]) if parts else ("", [])
    return {"name": name, "category": category, "clauses": clauses,
            "raw": text}


def _canon_attr(word: str, lex: dict | None = None) -> str:
    if lex and word in lex:
        return lex[word]
    return ATTR_WORDS.get(word, word)


def canonicalize_predicate(pred: dict,
                             lex: dict | None = None) -> dict:
    """Map Chinese attr words inside a predicate to canonical KB keys.

    Learned mappings (lex) win over built-ins: experience beats defaults.
    """
    p = dict(pred)
    for k in ("key", "action", "feature"):
        if k in p and isinstance(p[k], str):
            if lex and p[k] in lex:
                p[k] = lex[p[k]]
                continue
            for zh, en in ATTR_WORDS.items():
                if p[k] == zh:
                    p[k] = en
    return p


def update_context(state: dict, frame: SemanticFrame) -> dict:
    """Fold a parsed frame's real entities into the session context.

    Only non-pronoun entity texts are stored, most-recent-first, capped.
    Pronouns themselves are never stored as antecedents.
    """
    st = {"entities": list(state.get("entities", [])),
          "last_observation": state.get("last_observation"),
          "turn": int(state.get("turn", 0)) + 1}
    for e in frame.entities:
        t = (e.get("text") or "").strip()
        if not t or t in PRONOUNS:
            continue
        if not any(x["text"] == t for x in st["entities"]):
            st["entities"].insert(0, {"text": t, "turn": st["turn"]})
    st["entities"] = st["entities"][:10]
    if frame.intent in ("observe", "classify", "attribute", "function",
                        "negate", "hedge") and frame.entities:
        st["last_observation"] = {
            "entities": [e["text"] for e in frame.entities],
            "predicates": frame.predicates, "turn": st["turn"],
            "raw": frame.raw}
    return st


def _has_question_word(text: str) -> bool:
    return any(w in (text or "") for w in QUESTION_WORDS)


def resolve_question(frame: dict, lexicon: dict | None = None) -> dict:
    """Adjudicate what a parsed frame is asking (Phase 6D, Task 1).

    Pure function over an already-parsed frame: separates "parsed" from
    "understood". Returns {kind, key, expected, verify, subjects,
    defects, needs_clarify, evidence}. kind in {function_q, attr_q,
    has_q, verify_attr, verify_class, verify_has, verify_capable,
    classify_q, unsupported_q, unknown_q}.

    Confidence is deliberately NOT a calibrated probability: the parser
    emits ordinal grades (fixed per-pattern constants). Callers must
    treat them as grades, never as P(correct).
    """
    lexicon = lexicon or {}
    preds = frame.get("predicates", []) or []
    subjects = [e.get("text", "") for e in frame.get("entities", []) or []]
    intent = frame.get("intent", "unknown")
    out = {"kind": "unknown_q", "key": None, "expected": None, "verify": [],
           "subjects": subjects, "defects": [], "needs_clarify": False,
           "evidence": list(frame.get("trace", []) or [])}
    if intent == "unknown":
        out["defects"].append("unparseable: no pattern matched")
        return out
    if intent in ("negate", "hedge", "correct", "causal", "observe",
                  "classify", "attribute", "function"):
        # assertions / teachings / corrections are not questions
        out["kind"] = "unsupported_q"
        out["defects"].append(f"not a question (intent={intent})")
        return out
    # intent == "query" from here on
    for p in preds:
        t = p.get("type")
        if t == "query_function":
            out.update(kind="function_q", key="function")
            return out
        if t == "capable" and not p.get("denied") and \
                _has_question_word(str(p.get("action", ""))):
            # 光盒能做什么: ability asked with an interrogative action
            # is a function question (was: unrecognized -> clarify).
            out.update(kind="function_q", key="function")
            out["evidence"].append("capable predicate with "
                                   "interrogative action")
            return out
        if t == "query_attr" and p.get("key"):
            out.update(kind="attr_q", key=p["key"])
            return out
        if t == "attr" and not p.get("denied") and p.get("key") and \
                _has_question_word(str(p.get("value", ""))):
            # X由什么制成 / X是什么颜色的: the interrogative sits in the
            # value slot; the KEY is the question (was: ignored, and the
            # executor guessed an unrelated default attribute).
            out.update(kind="attr_q", key=p["key"])
            out["evidence"].append(f"attr-key {p['key']} from predicate")
            return out
        if t == "has" and not p.get("denied"):
            feat = p.get("feature", "")
            key = lexicon.get(feat, feat) if feat else None
            if key:
                out.update(kind="has_q", key=key)
                out["evidence"].append(f"feature {feat!r} -> key {key!r}")
                return out
            out["defects"].append("has-question without feature")
        if t == "query_reason":
            # 为什么X会V: no causal model exists; the planner marks it
            # ask_about=reason and the executor refuses honestly. It is a
            # recognized question, never a clarification case.
            out.update(kind="reason_q")
            return out
        if t == "query_topic":
            # 我想了解X: topic introduction answered from the concept
            # record (read-only). Never a knowledge write.
            out.update(kind="topic_q")
            return out
        if t in ("verify_attr", "verify_class", "verify_has",
                 "verify_capable"):
            out.update(kind=t, key=p.get("key"),
                       expected=p.get("expected", p.get("feature",
                                                        p.get("action"))),
                       verify=[p])
            return out
        if t == "is_a" and not p.get("denied") and \
                _has_question_word(str(p.get("obj", ""))):
            # X的材质是什么: possessive entity carries the key; the
            # executor resolves it. Bare 杯子是什么 cannot name a key.
            out.update(kind="classify_q", key=None)
            return out
    out["defects"].append("query frame with no recognizable question role")
    if subjects:
        out["needs_clarify"] = True
        out["defects"].append("cannot determine question type: ask with "
                              "an attribute word (材质/用途/…) or a "
                              "supported pattern")
    return out


def resolve_frame(frame: SemanticFrame, state: dict) -> SemanticFrame:
    """Replace bare-pronoun entity texts using session context.

    Exactly one recent candidate -> substitute (records resolved_from).
    Several same-turn candidates -> ambiguity entry, reference kept.
    None -> stays in unresolved_references. Never guesses silently:
    every substitution is recorded on the entity.
    """
    import copy
    fr = copy.deepcopy(frame)
    cands = [e["text"] for e in state.get("entities", [])]
    if not cands:
        return fr
    for e in fr.entities:
        if e["text"] not in PRONOUNS:
            continue
        if len(cands) == 1:
            e["resolved_from"] = e["text"]
            e["text"] = cands[0]
            fr.unresolved_references = [
                r for r in fr.unresolved_references if r != e["resolved_from"]]
            for p in fr.predicates:
                for k in ("subj",):
                    if p.get(k) == e["resolved_from"]:
                        p[k] = e["text"]
        else:
            fr.ambiguities.append(
                f"pronoun '{e['text']}' has {len(cands)} candidates "
                f"{cands[:3]}; please clarify")
    return fr
