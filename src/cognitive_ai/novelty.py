"""Novelty / knowledge-gap detection.

Honest boundary: without a trained text-understanding model this module
cannot reliably understand arbitrary Chinese sentences. It works on
structured observations {"name","parents","attributes","relations"} plus a
small deterministic Chinese pattern parser for `X是Y`-style input.
Everything else is recorded as unexplained, never guessed.
"""
from __future__ import annotations

import json
import re

from .memory import Memory
from .ontology import Ontology
from .reasoner import Reasoner

# Very small deterministic patterns, e.g. "杯子是容器" / "杯子有把手"
_PATTERNS = [
    (re.compile(r"^\s*(?P<subj>.+?)\s*是\s*(?P<obj>.+?)\s*[。；;]?\s*$"),
     "is_a"),
    (re.compile(r"^\s*(?P<subj>.+?)\s*有(?!意思|趣|空|点|些|所)"
                r"\s*(?P<obj>.+?)\s*[。；;]?\s*$"),
     "has_feature"),
    (re.compile(r"^\s*(?P<subj>.+?)\s*(?:可以|能|用于|用来)\s*(?P<obj>.+?)\s*[。；;]?\s*$"),
     "capability"),
]


def parse_chinese_simple(sentence: str, mem=None) -> dict:
    """Best-effort deterministic parse. Returns {"parsed": bool, ...}.

    With a memory handle this runs the compositional parser
    (semparse.parse_statement) and maps the result back onto the
    legacy shape, plus negation/modality/ambiguity extras. Without one
    it keeps the three legacy whole-sentence patterns so old callers
    keep working.
    """
    if mem is None:
        return _parse_legacy(sentence)
    from .semparse import parse_statement
    u = parse_statement(sentence, mem)
    kind_map = {"is_a": "is_a", "has": "has_feature",
                "capable": "capability"}
    out = {"parsed": u.predicate in kind_map,
           "kind": kind_map.get(u.predicate, u.predicate),
           "subj": u.subject, "obj": u.object,
           "negated": u.negated, "modal": u.modal,
           "ambiguity": u.ambiguity,
           "unknown_spans": u.unknown_spans, "spans": u.spans}
    if not out["parsed"]:
        out["note"] = ("unsupported sentence pattern; needs structured "
                       "input")
        out["raw"] = sentence
    return out


def _parse_legacy(sentence: str) -> dict:
    for rx, kind in _PATTERNS:
        m = rx.match(sentence.strip())
        if m:
            return {"parsed": True, "kind": kind,
                    "subj": m.group("subj").strip(),
                    "obj": m.group("obj").strip()}
    return {"parsed": False, "raw": sentence,
            "note": "unsupported sentence pattern; needs structured input"}


class NoveltyReport(dict):
    pass


class NoveltyDetector:
    # attributes every physical object is expected to clarify eventually
    EXPECTED_KEYS = ["material", "function", "location"]

    def __init__(self, memory: Memory):
        self.mem = memory
        self.onto = Ontology(memory)
        self.reasoner = Reasoner(memory)

    def check_observation(self, obs: dict) -> NoveltyReport:
        """obs: {"name","parents"=[], "attributes"={}, "relations"=[],
        "source"="observation"}."""
        name = obs.get("name", "").strip()
        parents = obs.get("parents", [])
        attrs: dict = obs.get("attributes", {})
        rels: list = obs.get("relations", [])
        matches = self.mem.find_concepts_by_name(name) if name else []
        report: NoveltyReport = NoveltyReport({
            "name": name, "matched": [c["id"] for c in matches],
            "is_new_name": len(matches) == 0,
            "unexplained_attributes": [], "conflicts": [],
            "missing_expected": [], "parent_problems": [],
            "needs_new_concept": False, "candidates": [],
        })
        if not matches:
            report["needs_new_concept"] = True
            report["candidates"].append(
                {"type": "add_concept", "name": name,
                 "reason": "no concept with this name exists"})
            # name alone is NOT enough to declare a brand-new category:
            # candidate is provisional until distinguishing evidence arrives.
            report["candidates"][-1]["provisional"] = True
            for p in parents:
                if self.mem.get_concept(p) is None:
                    report["parent_problems"].append(f"unknown parent: {p}")
            return report

        cid = matches[0]["id"]
        inherited = self.onto.inherited_attributes(cid)
        for k, v in attrs.items():
            if k not in inherited:
                report["unexplained_attributes"].append(k)
            else:
                old = inherited[k]["value"]
                if old != v:
                    report["conflicts"].append(
                        {"key": k, "old": old, "new": v})
        for k in self.EXPECTED_KEYS:
            if k not in inherited and k not in attrs:
                report["missing_expected"].append(k)
        for p in parents:
            if self.mem.get_concept(p) is None:
                report["parent_problems"].append(f"unknown parent: {p}")
            elif not self.onto.is_a(cid, p):
                report["candidates"].append(
                    {"type": "reparent_candidate", "concept": cid,
                     "parent": p, "reason": f"observation suggests {cid} is_a {p}"})
        # category too broad? same parent children with divergent attr values
        for p in self.mem.get_parents(cid):
            divergent = self._divergent_siblings(p)
            if divergent:
                report["candidates"].append(
                    {"type": "split_candidate", "category": p,
                     "reason": f"children diverge on {divergent}"})
        if report["unexplained_attributes"] and len(matches) == 1:
            report["candidates"].append(
                {"type": "extend_concept", "concept": cid,
                 "attrs": report["unexplained_attributes"],
                 "reason": "observation has attributes the concept lacks"})
        return report

    def _divergent_siblings(self, parent: str) -> list[str]:
        kids = self.mem.get_children(parent)
        if len(kids) < 2:
            return []
        keys: dict[str, set] = {}
        for k in kids:
            inh = self.onto.inherited_attributes(k)
            for key, row in inh.items():
                keys.setdefault(key, set()).add(
                    json.dumps(row["value"], sort_keys=True))
        return sorted([k for k, v in keys.items() if len(v) > 1])

    def knowledge_gaps(self) -> list[dict]:
        """Scan whole KB for gaps: missing expected attrs, conflicts,孤立 concepts."""
        gaps: list[dict] = []
        for c in self.mem.list_concepts():
            inh = self.onto.inherited_attributes(c["id"])
            for k in self.EXPECTED_KEYS:
                if k not in inh and c["id"] not in (
                        "entity", "event", "action", "attribute",
                        "abstract", "object", "container"):
                    gaps.append({"type": "missing_attribute",
                                 "concept": c["id"], "key": k})
            if not self.mem.get_parents(c["id"]) and c["id"] not in (
                    "entity", "event", "action", "attribute", "abstract"):
                gaps.append({"type": "orphan_concept", "concept": c["id"]})
            if not inh and c["id"] not in (
                    "entity", "event", "action", "attribute", "abstract"):
                gaps.append({"type": "no_attributes", "concept": c["id"]})
        for cf in self.reasoner.check_conflicts():
            gaps.append({"type": cf["type"], **cf})
        # relations missing: concepts with no relations at all
        for c in self.mem.list_concepts():
            if not self.mem.relations_involving(c["id"]) and c["id"] not in (
                    "entity", "event", "action", "attribute", "abstract"):
                gaps.append({"type": "no_relations", "concept": c["id"]})
        return gaps
