"""Continual learning: observation -> research -> KB update, all recorded."""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from .curiosity import Curiosity
from .memory import Memory
from .novelty import NoveltyDetector
from .ontology import Ontology
from .policy import (INCONCLUSIVE_COOLDOWN_S, MAX_RESEARCH_ATTEMPTS,
                     MIN_CORROBORATING_SOURCES, MIN_EVIDENCE_CREDIBILITY)
from .reasoner import Reasoner
from .research import LocalCorpusProvider, _mentions
from .schema_evolver import SchemaEvolver


def default_regression(memory: Memory) -> tuple[bool, str]:
    """Built-in regression gate: no orphan core concepts, no cycles, cup intact."""
    onto = Ontology(memory)
    try:
        cup = memory.get_concept("cup")
        if cup is None:
            return False, "core concept 'cup' missing"
        if "container" not in onto.ancestors("cup"):
            return False, "cup lost ancestor 'container'"
        # cycle sanity: no concept is its own ancestor
        for c in memory.list_concepts():
            if c["id"] in onto.ancestors(c["id"]):
                return False, f"cycle at {c['id']}"
    except Exception as e:
        return False, str(e)
    return True, "ok"


class Learner:
    """Type-A (episodic) + Type-B (structural) learning. No neural nets here."""

    def __init__(self, memory: Memory, corpus_dir: str):
        self.mem = memory
        self.onto = Ontology(memory)
        self.reasoner = Reasoner(memory)
        self.detector = NoveltyDetector(memory)
        self.curiosity = Curiosity(memory)
        self.provider = LocalCorpusProvider(corpus_dir)
        self.evolver = SchemaEvolver(memory, regression_check=default_regression)

    # ---- one observation ----
    def observe(self, obs: dict) -> dict:
        """obs: {"name","parents","attributes","relations","source"}."""
        report = self.detector.check_observation(obs)
        exp_id = self.mem.add_experience(
            "observation", f"observed {obs.get('name')}",
            {"observation": obs, "novelty": dict(report)})
        return {"experience_id": exp_id, "novelty": dict(report)}

    def learn_from_observation(self, obs: dict) -> dict:
        """Full mini-cycle for one structured observation."""
        result: dict[str, Any] = {"observation": obs}
        seen = self.observe(obs)
        result["novelty"] = seen["novelty"]
        name = obs.get("name", "")
        matches = self.mem.find_concepts_by_name(name)
        if not matches:
            # create provisional sub-concept under first known parent (or entity)
            parents = [p for p in obs.get("parents", [])
                       if self.mem.get_concept(p) is not None] or ["entity"]
            cid = name.lower().strip().replace(" ", "_") or "unknown_item"
            if self.mem.get_concept(cid) is None:
                r = self.evolver.direct_apply(
                    "add_concept",
                    {"id": cid, "name": name, "kind": "Entity",
                     "parents": parents,
                     "attributes": obs.get("attributes", {}),
                     "confidence": 0.5, "source": "learning"},
                    reason=f"novel observation '{name}' with no matching concept",
                    evidence=[{"observation": obs}])
                result["new_concept"] = {"id": cid, **r}
                for rel in obs.get("relations", []):
                    try:
                        self.evolver.direct_apply(
                            "add_relation",
                            {"src": cid, "rel_type": rel.get("rel_type", "related_to"),
                             "dst": rel.get("dst"), "confidence": 0.5,
                             "source": "learning"}, reason="observed relation")
                    except Exception:
                        pass
            else:
                result["new_concept"] = {"id": cid, "note": "already exists"}
        else:
            cid = matches[0]["id"]
            # add genuinely new attribute keys as hypotheses (never overwrite verified)
            inherited = self.onto.inherited_attributes(cid)
            added = []
            for k, v in obs.get("attributes", {}).items():
                if k not in inherited:
                    r = self.evolver.direct_apply(
                        "add_attribute",
                        {"concept": cid, "key": k, "value": v,
                         "confidence": 0.55, "source": "learning",
                         "status": "hypothesis"},
                        reason=f"extend concept {cid} with observed attribute {k}",
                        evidence=[{"observation": obs}])
                    added.append({k: r})
                elif inherited[k]["value"] != v:
                    # conflict: record contested claim + evidence, do NOT overwrite.
                    # Keep the verified row untouched AND store the rival
                    # observation as a second `contested` attribute row so
                    # check_conflicts() can see both sides.
                    cl = self.mem.add_claim(
                        f"{cid}.{k}: {inherited[k]['value']} vs observed {v}",
                        subject_id=cid, status="contested", confidence=0.5)
                    self.mem.add_evidence(
                        cl, f"observed {k}={v} for {name}", source="observation",
                        credibility=0.5)
                    self.evolver.direct_apply(
                        "add_attribute",
                        {"concept": cid, "key": k, "value": v,
                         "confidence": 0.5, "source": "observation",
                         "status": "contested"},
                        reason=f"contested evidence for {cid}.{k}",
                        evidence=[{"observation": obs}])
                    added.append({k: {"status": "contested", "claim_id": cl}})
            result["updated_concept"] = {"id": cid, "added": added}
        # evaluate: did conflicts grow?
        result["conflicts"] = self.reasoner.check_conflicts()
        self.mem.add_experience("evaluation",
                                f"learned from {name}",
                                {"result": _jsonable(result)})
        return result

    # ---- research one question ----
    def research_question(self, qid: int) -> dict:
        """Research a single question. Binding preserved end to end.

        Returns a fixed-shape binding including question_id / target_id /
        key / evidence_sources / action / applied_change_id so the loop log
        always matches the actual modification (or the lack of one).

        Rules enforced:
        - inconclusive NEVER writes semantic knowledge or schema versions;
        - evidence must mention the target concept;
        - at least MIN_CORROBORATING_SOURCES independent hits must agree
          on the same cleaned value before any add_attribute proposal;
        - promoted values are always `hypothesis`, never `verified`.
        """
        import re
        import time
        q = self.mem.query_one("SELECT * FROM questions WHERE id=?", (qid,))
        if q is None:
            return {"ok": False, "error": "question not found",
                    "question_id": qid, "target_id": None, "key": None,
                    "question": None, "result": "error",
                    "evidence_sources": [], "action": "none",
                    "applied_change_id": None}
        detail = _detail_dict(q.get("detail"))
        attempts = int(detail.get("attempts", 0) or 0)
        target = q["target_id"]
        key, concept_in_text = _parse_gap_question(q["text"])
        base = {"ok": True, "question_id": qid, "target_id": target,
                "key": key, "question": q["text"]}
        if attempts >= MAX_RESEARCH_ATTEMPTS:
            self.mem.add_experience(
                "research", f"Q{qid}: skipped, max attempts reached",
                {"question": q["text"], "attempts": attempts})
            return {**base, "result": "inconclusive",
                    "evidence_sources": [], "action": "skipped_max_attempts",
                    "applied_change_id": None}
        # The question text names its target; a mismatch means the queue row
        # is stale — never write to a concept the question is not about.
        if key and concept_in_text and target and concept_in_text != target:
            return self._record_inconclusive(
                qid, q, detail, attempts, "target_mismatch",
                f"text names '{concept_in_text}' but target is '{target}'",
                [], base)
        self.mem.update_question(qid, status="researching")
        focus = _focus_terms(self.mem, target)
        hits = self.provider.search(q["text"], limit=5,
                                    focus_terms=tuple(focus))
        sources = [h["source"] for h in hits]
        # Only target-relevant, credible hits count as evidence.
        relevant = [h for h in hits
                    if h.get("credibility", 0) >= MIN_EVIDENCE_CREDIBILITY
                    and _mentions(h.get("excerpt", ""), tuple(focus))]
        rel_sources = [h["source"] for h in relevant]
        applied = None
        applied_change_id = None
        value = None
        if (key and target and self.mem.get_concept(target) is not None
                and q["kind"] == "gap"):
            inh = self.onto.inherited_attributes(target)
            if key not in inh:
                # corroboration: >=2 relevant hits, same cleaned value
                votes: dict[str, list[str]] = {}
                for h in relevant:
                    v = _clean_value(_extract_value(h.get("excerpt", ""), key))
                    if v:
                        votes.setdefault(v, []).append(h["source"])
                agreed = [(v, s) for v, s in votes.items()
                          if len(s) >= MIN_CORROBORATING_SOURCES]
                if agreed:
                    agreed.sort(key=lambda kv: (-len(kv[1]), kv[0]))
                    value = agreed[0][0]
                    supporting = [h for h in relevant
                                  if h["source"] in agreed[0][1]]
                    r = self.evolver.direct_apply(
                        "add_attribute",
                        {"concept": target, "key": key, "value": value,
                         "confidence": round(
                             min(h["credibility"] for h in supporting)
                             * 0.8, 3),
                         "source": supporting[0]["source"],
                         "status": "hypothesis"},
                        reason=f"research Q{qid}: {q['text']}",
                        evidence=supporting)
                    if r.get("ok"):
                        applied = r
                        applied_change_id = r.get("change_id")
        if applied:
            self.mem.update_question(
                qid, status="learned",
                detail={**detail, "attempts": attempts + 1,
                        "evidence": rel_sources, "value": value})
            self.mem.add_experience(
                "research", f"Q{qid}: learned",
                {"question": q["text"], "target": target, "key": key,
                 "value": value, "hits": relevant, "applied": applied})
            return {**base, "result": "learned",
                    "evidence_sources": rel_sources,
                    "action": "add_attribute",
                    "applied_change_id": applied_change_id}
        reason = _inconclusive_reason(hits, relevant, key, target)
        return self._record_inconclusive(qid, q, detail, attempts,
                                         reason[0], reason[1], sources, base)

    def _record_inconclusive(self, qid: int, q: dict, detail: dict,
                             attempts: int, code: str, note: str,
                             sources: list, base: dict) -> dict:
        """Mark inconclusive with attempts + cooldown. NEVER writes schema."""
        import time
        new_detail = {**detail, "attempts": attempts + 1,
                      "cooldown_until": time.time() + INCONCLUSIVE_COOLDOWN_S,
                      "last_reason": code, "last_note": note,
                      "last_evidence": sources}
        self.mem.update_question(qid, status="inconclusive",
                                 detail=new_detail)
        self.mem.add_experience("research", f"Q{qid}: inconclusive ({code})",
                                {"question": q["text"], "reason": note,
                                 "evidence_sources": sources})
        return {**base, "result": "inconclusive",
                "evidence_sources": sources, "action": "none",
                "applied_change_id": None}


def _detail_dict(raw: Any) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def _parse_gap_question(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract (key, concept) from "What is the 'key' of concept 'c'?".

    Returns (None, None) for non-gap question kinds.
    """
    import re
    m = re.search(r"'([^']+)' of concept '([^']+)'", text or "")
    if m:
        return m.group(1), m.group(2)
    return None, None


def _focus_terms(mem: Memory, target: Optional[str]) -> list[str]:
    """Terms proving a hit is about the target: id + concept name."""
    if not target:
        return []
    terms = [target]
    c = mem.get_concept(target)
    if c and c.get("name") and c["name"] != target:
        terms.append(c["name"])
    return terms


def _inconclusive_reason(hits: list, relevant: list,
                         key: Optional[str],
                         target: Optional[str]) -> tuple[str, str]:
    if not hits:
        return ("no_evidence", "no corpus document matched the question")
    if not relevant:
        return ("no_target_relevant_evidence",
                f"hits exist but none credibly discuss target '{target}'")
    if not key or not target:
        return ("not_a_gap_question",
                "only missing-attribute gaps can propose attributes")
    return ("no_corroborated_value",
            "no value agreed upon by enough independent sources")


# Fragments that prove an extracted string is a sentence remnant, not a value.
_VALUE_POISON = (" 的 ", " 是 ", "：", ":", "；", ";", "。", ".", "/", "\\",
                 "concept", "material", "function", "location", "relation",
                 "attribute")


def _clean_value(raw: Optional[str]) -> Optional[str]:
    """Accept only short atomic values; reject sentence fragments."""
    if not raw:
        return None
    v = raw.strip().strip("'\"“”‘’").strip()
    v = re.sub(r"[。；;.\s]+$", "", v).strip()
    if not v or len(v) > 24:
        return None
    if any(p in v for p in _VALUE_POISON):
        return None
    if len(v.split()) > 2:
        return None
    return v or None


def _jsonable(o: Any) -> Any:
    try:
        json.dumps(o, ensure_ascii=False, default=str)
        return json.loads(json.dumps(o, ensure_ascii=False, default=str))
    except Exception:
        return str(o)


def _extract_value(excerpt: str, key: str) -> Optional[str]:
    """Deterministic tiny extractor: 'key: value' or 'key是value' in excerpt.

    Capture stops at sentence/field boundaries so only the adjacent value
    fragment is returned; _clean_value() validates it further.
    """
    import re
    stop = r"\n；;。.，,"
    pats = [re.compile(rf"{re.escape(key)}\s*[:：]\s*([^{stop}]{{1,40}})"),
            re.compile(rf"{re.escape(key)}\s*是\s*([^{stop}]{{1,40}})")]
    for p in pats:
        m = p.search(excerpt)
        if m:
            return m.group(1).strip()
    return None
