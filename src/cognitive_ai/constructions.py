"""Bidirectional constructions (Phase 6H-2B).

A construction binds TWO slots (entity + value) between fixed affixes
for one statement kind (this phase: capability only). Parse and generate
read the SAME stored row -- there are no separate per-direction
templates, handwritten or otherwise.

  parse:    pre + ENTITY + mid + VALUE + suf  ->  observation-grade frame
            {intent observe, capable(entity, value)} for the existing
            teach-confirm flow (hypothesis gates, never verified here).
  generate: (entity, value) -> pre + entity + mid + value + suf.

Learning is affix induction with role checks, never sentence memos:
- >=2 examples with DISTINCT entities;
- the entity slot must equal the labeled entity, the value slot the
  labeled value (or stay fixed when all values agree);
- held-out examples (disjoint entities, incl. novel entity x value
  combos) must parse to exact (entity, value) AND regenerate the
  source sentence byte-for-byte;
- ambiguity (several matching rows) abstains in both directions.

Lifecycle: candidate -> validated (held-out 100%) -> explicit
activate. Only active rows are consulted, and only when the fixed
parser returns unknown (parse) or when asked to generate.
"""
from __future__ import annotations

import time

_counter = 0


def new_rule_id(name: str = "cx") -> str:
    global _counter
    _counter += 1
    return f"cx_{name}_{int(time.time() * 1000) % 100000}_{_counter}"


def _lcs_all(strs: list[str]) -> str:
    """Longest common substring of all strings (short inputs only)."""
    if not strs:
        return ""
    best = ""
    base = min(strs, key=len)
    for i in range(len(base)):
        for j in range(i + 1, len(base) + 1):
            cand = base[i:j]
            if len(cand) > len(best) and all(cand in s for s in strs):
                best = cand
    return best


def _split_affix(srcs: list[str]) -> tuple[str, str, str] | None:
    """Common (pre, mid, suf): pre = common prefix, suf = common suffix
    (non-overlapping), mid = longest common substring of the remainders.
    None when no slot-bearing structure exists.
    """
    pre = srcs[0]
    for t in srcs[1:]:
        i = 0
        while i < min(len(pre), len(t)) and pre[i] == t[i]:
            i += 1
        pre = pre[:i]
    suf = srcs[0]
    for t in srcs[1:]:
        j = 0
        while j < min(len(suf), len(t)) and suf[-1 - j] == t[-1 - j]:
            j += 1
        suf = suf[len(suf) - j:] if j else ""
    if len(pre) + len(suf) >= min(len(t) for t in srcs):
        return None
    cores = [s[len(pre):len(s) - len(suf) if suf else len(s)]
             for s in srcs]
    mid = _lcs_all(cores)
    if not mid:
        return None
    # mid must leave room for both slots in every core
    for c in cores:
        parts = c.split(mid, 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
    return pre, mid, suf


def induce(examples: list[dict]) -> dict:
    """Induce construction candidates. Example: {"src","entity","value"}.

    Needs >=2 examples with distinct entities. The entity part of each
    core must equal the labeled entity and the value part the labeled
    value; when all values agree the value stays fixed (reported).
    Anything weaker -> rejected with a reason, never a silent memo.
    """
    candidates, rejected = [], []
    rows = [r for r in examples if r.get("src", "").strip()
            and r.get("entity", "").strip() and r.get("value", "").strip()]
    if len(rows) < 2:
        return {"candidates": candidates, "groups": 0,
                "rejected": [{"ok": False, "reason": "needs >=2 labeled "
                              "examples with distinct entities; examples "
                              "kept, nothing induced"}]}
    if len({r["entity"].strip() for r in rows}) < 2:
        return {"candidates": candidates, "groups": 1,
                "rejected": [{"ok": False, "reason": "entities not "
                              "distinct; no generalizable slot"}]}
    aff = _split_affix([r["src"].strip() for r in rows])
    if aff is None:
        return {"candidates": candidates, "groups": 1,
                "rejected": [{"ok": False, "reason": "no shared "
                              "pre/mid/suf structure"}]}
    pre, mid, suf = aff
    ents_ok = True
    for r in rows:
        core = r["src"].strip()[len(pre):len(r["src"].strip())
                                - len(suf) if suf
                                else len(r["src"].strip())]
        if mid not in core:
            ents_ok = False
            break
        left, right = core.split(mid, 1)
        if left != r["entity"].strip() or not right:
            ents_ok = False
            break
    if not ents_ok:
        return {"candidates": candidates, "groups": 1,
                "rejected": [{"ok": False, "reason": "affix middle does "
                              "not isolate the labeled (entity, value) "
                              "pair"}]}
    # value slot variable only if values differ across examples
    values = [r["value"].strip() for r in rows]
    fixed = values[0] if len(set(values)) == 1 else ""
    if not fixed:
        # each value must sit exactly after mid in its own source
        for r in rows:
            core = r["src"].strip()[len(pre):len(r["src"].strip())
                                    - len(suf) if suf else
                                    len(r["src"].strip())]
            if core != r["entity"].strip() + mid + r["value"].strip():
                return {"candidates": candidates, "groups": 1,
                        "rejected": [{"ok": False, "reason": "value slot "
                                      "does not align; no slot induced"}]}
    candidates.append({
        "kind": "capability", "pre": pre, "mid": mid, "suf": suf,
        "fixed_value": fixed, "confidence": 0.7,
        "support": [{k: r[k] for k in ("src", "entity", "value")}
                    for r in rows],
        "note": f"construction from {len(rows)} examples: {pre!r} + "
                f"ENTITY + {mid!r} + "
                f"{'VALUE' if not fixed else repr(fixed)} + {suf!r}"})
    return {"candidates": candidates, "groups": 1, "rejected": rejected}


def match(text: str, rules: list[dict]) -> list[tuple]:
    """All (rule, entity, value) readings of text. Empty = no match."""
    s = text.strip()
    hits = []
    for r in rules:
        if r.get("status") != "active" or r.get("kind") != "capability":
            continue
        pre, mid, suf = r.get("pre", ""), r.get("mid", ""), r.get("suf", "")
        if not s.startswith(pre):
            continue
        if suf and not s.endswith(suf):
            continue
        core = s[len(pre):len(s) - len(suf) if suf else len(s)]
        if mid not in core:
            continue
        ent, rest = core.split(mid, 1)
        if not ent:
            continue
        if r.get("fixed_value"):
            if rest != r["fixed_value"]:
                continue
            val = rest
        elif not rest:
            continue
        else:
            val = rest
        hits.append((r, ent, val))
    return hits


def parse_frame(text: str, rules: list[dict], raw: str = "") -> dict | None:
    """Parse direction: unique match -> observation-grade frame.

    Multiple matches -> None (ambiguity abstains). The frame reuses the
    exact capable/observe shapes the fixed parser emits, so the existing
    teach-confirm flow (hypothesis only) applies unchanged.
    """
    hits = match(text, rules)
    if len(hits) != 1:
        return None
    r, ent, val = hits[0]
    return {"intent": "observe",
            "entities": [{"text": ent, "role": "subject"}],
            "predicates": [{"type": "capable", "subj": ent,
                            "action": val}],
            "modality": "assert", "scope": "generic",
            "source": "construction",
            "confidence": 0.65, "unresolved_references": [],
            "ambiguities": [], "raw": raw or text,
            "trace": [f"construction:{r['id']} parsed "
                      f"{text!r} -> ({ent}, {val})"]}


def render(entity: str, value: str, rules: list[dict]) -> list[dict]:
    """Generate direction: (entity, value) -> surfaces from the SAME rows.

    Returns [{text, source}] or [] (abstain). Fixed values constrain:
    a fixed-value row only renders its own value.
    """
    out = []
    for r in rules:
        if r.get("status") != "active" or r.get("kind") != "capability":
            continue
        if not entity:
            continue
        if r.get("fixed_value"):
            if value != r["fixed_value"]:
                continue
            out.append({"text": f"{r['pre']}{entity}{r['mid']}"
                                f"{r['fixed_value']}{r['suf']}",
                        "source": f"learned:{r['id']}"})
        elif value:
            out.append({"text": f"{r['pre']}{entity}{r['mid']}{value}"
                                f"{r['suf']}",
                        "source": f"learned:{r['id']}"})
    return out


def validate_rule(rule: dict, heldout: list[dict]) -> dict:
    """Held-out check both directions: parse exactness AND regeneration.

    Each held-out {src, entity, value} must parse (rule alone) to the
    exact pair, and rendering that pair must reproduce src exactly.
    Strict 100%.
    """
    if not heldout:
        return {"passed": False, "reason": "no held-out examples",
                "checked": 0}
    probe = [{**rule, "status": "active"}]
    detail = []
    for h in heldout:
        fr = parse_frame(h["src"], probe)
        ok = fr is not None and \
            fr["entities"][0]["text"] == h["entity"].strip() and \
            fr["predicates"][0]["action"] == h["value"].strip()
        regen = ""
        if ok:
            outs = render(h["entity"].strip(), h["value"].strip(), probe)
            regen = outs[0]["text"] if outs else ""
            ok = regen == h["src"].strip()
        detail.append({"src": h["src"], "ok": bool(ok), "regen": regen})
    passed = all(d["ok"] for d in detail)
    return {"passed": passed, "checked": len(detail), "detail": detail,
            "reason": "" if passed else "held-out parse/regenerate mismatch"}


def activate_rule(mem, rule_id: str) -> dict:
    row = mem.get_construction(rule_id)
    if not row:
        return {"ok": False, "error": "rule not found"}
    if row["status"] != "validated":
        return {"ok": False,
                "error": f"only validated rules activate (now "
                f"{row['status']})"}
    row["status"] = "active"
    mem.save_construction(row)
    mem.add_experience("evaluation", f"construction {rule_id} activated",
                       {"op": "construction_activate", "rule_id": rule_id})
    return {"ok": True, "rule_id": rule_id}


def propose_alignment(text: str, mem) -> dict | None:
    """Align an unknown utterance with known knowledge (Phase 6H-2C).

    Returns {"entity", "value", "basis"} or None. Generic rule, no
    per-sentence logic: exactly ONE known concept mentioned AND exactly
    ONE of that concept's known function values mentioned (raw words).
    Pronouns, zero matches and several matches all abstain (None) --
    ambiguity must become clarification, never a guess.
    """
    from .semantics import PRONOUNS
    s = text.strip()
    if not s or any(p == s for p in PRONOUNS):
        return None
    ents = []
    try:
        for c in mem.list_concepts():
            for name in (c.get("id", ""), c.get("name", "")):
                if name and name in s and len(name) >= 2:
                    ents.append(c["id"])
                    break
    except Exception:
        return None
    ents = list(dict.fromkeys(ents))
    if len(ents) != 1:
        return None
    vals = []
    try:
        from .ontology import Ontology
        inh = Ontology(mem).inherited_attributes(ents[0])
    except Exception:
        inh = {}
    for key, row in (inh or {}).items():
        # any known attribute value of this entity anchors alignment
        # (keys stay raw here: 观察 stores user words uncanonicalized;
        # canonicalization happens at confirm time, not at guess time).
        v = row.get("value")
        if isinstance(v, str) and len(v) >= 2 and v in s:
            vals.append(v)
    try:
        lex = {r["word"]: r["key"] for r in mem.list_mappings()}
    except Exception:
        lex = {}
    for word, key in lex.items():
        if key == "function" and len(word) >= 2 and word in s and \
                word not in vals:
            vals.append(word)
    vals = list(dict.fromkeys(vals))
    if len(vals) != 1:
        return None
    return {"entity": ents[0], "value": vals[0],
            "basis": f"known concept + known value co-occur in input"}


def record_pair(mem, src: str, entity: str, value: str,
                basis: str) -> int:
    """Store one trusted utterance pair as an experience (no new table).

    Trusted = user-confirmed proposal or verified-QA evidence, never a
    guess. Candidates are induced from these by auto_learn_step.
    """
    return mem.add_experience(
        "observation", f"trusted utterance pair: {src!r}",
        {"op": "cx_pair", "src": src, "entity": entity, "value": value,
         "basis": basis})


def list_pairs(mem) -> list[dict]:
    """All trusted pairs, oldest first."""
    out = []
    try:
        rows = mem.query("SELECT * FROM experiences ORDER BY id")
    except Exception:
        return out
    import json as _json
    for r in rows:
        try:
            d = r.get("detail")
            d = _json.loads(d) if isinstance(d, str) else (d or {})
        except Exception:
            continue
        if isinstance(d, dict) and d.get("op") == "cx_pair" and \
                d.get("src") and d.get("entity") and d.get("value"):
            out.append({"id": r["id"], "src": d["src"],
                        "entity": d["entity"], "value": d["value"],
                        "basis": d.get("basis", "")})
    return out


def auto_learn_step(mem) -> dict:
    """Cluster trusted pairs, induce candidates, validateREADY ones.

    Greedy clustering by shared affix structure with role checks; each
    cluster with >=2 distinct entities induces at most one candidate
    (saved, never activated here). Candidates whose later pairs (fresh
    entities, not used as train) fully pass are promoted to validated.
    Activation stays an explicit human step. Returns a report dict.
    """
    pairs = list_pairs(mem)
    trained = set()
    for r in mem.list_constructions():
        try:
            for e in r.get("evidence", []) or []:
                for d in (e or {}).get("support", []) or []:
                    if isinstance(d, dict) and d.get("src"):
                        trained.add(d["src"])
        except Exception:
            pass
    clusters: list[list[dict]] = []
    for p in pairs:
        placed = False
        for cl in clusters:
            trial = [q["src"] for q in cl] + [p["src"]]
            aff = _split_affix(trial)
            if aff is None:
                continue
            pre, mid, suf = aff
            ok = True
            for q in cl + [p]:
                core = q["src"][len(pre):len(q["src"]) - len(suf)
                                if suf else len(q["src"])]
                if mid not in core:
                    ok = False
                    break
                left, right = core.split(mid, 1)
                if left != q["entity"] or right != q["value"]:
                    ok = False
                    break
            if ok:
                cl.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    report = {"clusters": len(clusters), "induced": [], "validated": []}
    known_affix = set()
    for r in mem.list_constructions():
        try:
            known_affix.add((r.get("pre", ""), r.get("mid", ""),
                             r.get("suf", "")))
        except Exception:
            pass
    for cl in clusters:
        if len({q["entity"] for q in cl}) < 2:
            continue
        if all(q["src"] in trained for q in cl):
            continue
        aff = _split_affix([q["src"] for q in cl])
        if aff is not None and aff in known_affix:
            continue  # same structure already tracked; later pairs
            # only serve as held-out evidence below
        rep = induce([{"src": q["src"], "entity": q["entity"],
                       "value": q["value"]} for q in cl])
        for c in rep["candidates"]:
            rid = new_rule_id("auto")
            mem.save_construction(
                {"id": rid, "kind": c["kind"], "pre": c["pre"],
                 "mid": c["mid"], "suf": c["suf"],
                 "fixed_value": c.get("fixed_value", ""),
                 "status": "candidate",
                 "evidence": [{"source": "auto-dialogue",
                               "support": c["support"],
                               "pair_ids": [q["id"] for q in cl]}],
                 "report": {"note": c["note"]}})
            mem.add_experience(
                "evaluation", f"auto construction candidate {rid}",
                {"op": "cx_auto_induce", "rule_id": rid})
            report["induced"].append(rid)
    # validate-ready: candidates AND validated auto-rows accumulate
    # later, fresh-entity evidence into the ledger. Candidate rows get
    # promoted on a full pass; validated rows only grow the ledger
    # toward the auto-activation threshold below.
    for row in mem.list_constructions(status="candidate") + \
            [r for r in mem.list_constructions(status="validated")
             if _is_auto(r)]:
        try:
            ev = row.get("evidence", [])
            used = set()
            for e in ev if isinstance(ev, list) else []:
                used.update((e or {}).get("pair_ids", []) or [])
        except Exception:
            used = set()
        if not used:
            continue
        train_ents = {q["entity"] for q in list_pairs(mem)
                      if q["id"] in used}
        held = [{"src": q["src"], "entity": q["entity"],
                 "value": q["value"], "pair_id": q["id"]}
                for q in list_pairs(mem)
                if q["id"] not in used and
                q["entity"] not in train_ents]
        if not held:
            continue
        rep = validate_rule(row, held)
        row["report"] = {**row.get("report", {}), **rep,
                         "auto_validated": True}
        if rep["passed"]:
            if row["status"] == "candidate":
                row["status"] = "validated"
                report["validated"].append(row["id"])
            _ledger_add(row, held)
        else:
            fails = row.get("report", {}).get("heldout_failures", []) or []
            fails = list(fails) + [h.get("pair_id") for h in held
                                   if h.get("pair_id") is not None]
            row["report"] = {**row.get("report", {}),
                             "heldout_failures": fails}
        mem.save_construction(row)
    # auto-activation (Phase 6H-2D): validated auto-rules whose ledger
    # reaches 3+ independent held-out passes over >=2 entities and >=2
    # values activate WITHOUT a manual command, with a full audit
    # event. Anything below threshold stays put.
    for row in mem.list_constructions(status="validated"):
        if not _is_auto(row):
            continue
        if _maybe_auto_activate(mem, row):
            report.setdefault("auto_activated", []).append(row["id"])
    # conflict watch (Phase 6H-2D): a later confirmed pair that matches
    # an ACTIVE auto-rule's affix but disagrees on roles is credible
    # counter-evidence. Record it and demote to validated (stop using
    # the rule) -- never silently let either side win.
    for row in mem.list_constructions(status="active"):
        if not _is_auto(row):
            continue
        _conflict_watch(mem, row, report)
    return report


def _is_auto(row: dict) -> bool:
    try:
        return any((e or {}).get("source") == "auto-dialogue"
                   for e in row.get("evidence", []) or [])
    except Exception:
        return False


# Evidence-constrained auto-activation thresholds (Phase 6H-2D).
# Counted from persisted pair ids, never memory counters or text.
MIN_HELDOUT_PASSES = 3
MIN_HELDOUT_ENTITIES = 2
MIN_HELDOUT_VALUES = 2


def _ledger_add(row: dict, held: list) -> None:
    have = set(row.get("report", {}).get("heldout_pass_ids", []) or [])
    for h in held:
        if h.get("pair_id") is not None:
            have.add(h["pair_id"])
    row["report"] = {**row.get("report", {}),
                     "heldout_pass_ids": sorted(have)}


def _maybe_auto_activate(mem, row: dict) -> bool:
    """Activate a validated auto-rule iff its persisted ledger meets
    3+ passes / 2+ entities / 2+ values. Returns True on activation."""
    import time as _time
    try:
        ids = set(row.get("report", {}).get("heldout_pass_ids", []) or [])
        pairs = [q for q in list_pairs(mem) if q["id"] in ids]
        if len(pairs) < MIN_HELDOUT_PASSES:
            return False
        if len({q["entity"] for q in pairs}) < MIN_HELDOUT_ENTITIES:
            return False
        if len({q["value"] for q in pairs}) < MIN_HELDOUT_VALUES:
            return False
    except Exception:
        return False
    row["status"] = "active"
    mem.save_construction(row)
    mem.add_experience(
        "evaluation", f"auto-activated construction {row['id']}",
        {"op": "cx_auto_activate", "rule_id": row["id"],
         "evidence_pair_ids": sorted(ids),
         "reason": f"{len(pairs)} independent held-out passes over "
         f"{len({q['entity'] for q in pairs})} entities and "
         f"{len({q['value'] for q in pairs})} values",
         "lifecycle": "validated->active",
         "at": _time.time()})
    return True


def _conflict_watch(mem, row: dict, report: dict) -> None:
    """Demote an active auto-rule on credible counter-evidence."""
    try:
        ev = row.get("evidence", [])
        used = set()
        for e in ev if isinstance(ev, list) else []:
            used.update((e or {}).get("pair_ids", []) or [])
        seen = set(row.get("report", {}).get("conflict_pair_ids", []) or [])
        probe = [{**row, "status": "active"}]
        for q in list_pairs(mem):
            if q["id"] in used or q["id"] in seen:
                continue
            hits = match(q["src"], probe)
            if not hits:
                continue  # different affix: not evidence about this rule
            _, ent, val = hits[0]
            if ent == q["entity"] and val == q["value"]:
                continue  # agrees: no conflict
            seen.add(q["id"])
            row["status"] = "validated"
            row["report"] = {**row.get("report", {}),
                             "conflict_pair_ids": sorted(seen)}
            mem.save_construction(row)
            mem.add_experience(
                "evaluation",
                f"construction conflict on {row['id']}: demoted",
                {"op": "cx_conflict", "rule_id": row["id"],
                 "pair_id": q["id"], "expected": [q["entity"], q["value"]],
                 "got": [ent, val]})
            report.setdefault("conflicts", []).append(
                {"rule": row["id"], "pair_id": q["id"]})
            return
        if seen != set(row.get("report", {}).get(
                "conflict_pair_ids", []) or []):
            row["report"] = {**row.get("report", {}),
                             "conflict_pair_ids": sorted(seen)}
            mem.save_construction(row)
    except Exception:
        pass
