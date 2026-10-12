"""Explainable rule reasoner. Returns structured results, never guesses unknown."""
from __future__ import annotations

import json
from typing import Optional

from .memory import Memory
from .ontology import Ontology


class ReasoningResult(dict):
    pass


class Reasoner:
    def __init__(self, memory: Memory):
        self.mem = memory
        self.onto = Ontology(memory)

    # ---- primitive queries ----
    def is_a(self, cid: str, ancestor: str) -> ReasoningResult:
        exists = self.mem.get_concept(cid) is not None
        anc_exists = self.mem.get_concept(ancestor) is not None
        if not exists or not anc_exists:
            return ReasoningResult({
                "query": {"type": "is_a", "concept": cid, "ancestor": ancestor},
                "conclusion": "unknown",
                "confidence": 0.0,
                "trace": [{"rule": "concept_lookup",
                           "detail": "one or both concepts not found"}],
            })
        val = self.onto.is_a(cid, ancestor)
        return ReasoningResult({
            "query": {"type": "is_a", "concept": cid, "ancestor": ancestor},
            "conclusion": bool(val),
            "confidence": 1.0 if val else 0.9,
            "trace": [{"rule": "transitive_is_a",
                       "facts": [f"{cid} ancestors={self.onto.ancestors(cid)}"]}],
        })

    def get_attribute(self, cid: str, key: str) -> ReasoningResult:
        if self.mem.get_concept(cid) is None:
            return ReasoningResult({
                "query": {"type": "get_attribute", "concept": cid, "key": key},
                "conclusion": "unknown", "confidence": 0.0,
                "trace": [{"rule": "concept_lookup",
                           "detail": f"concept not found: {cid}"}],
            })
        inherited = self.onto.inherited_attributes(cid)
        if key in inherited:
            a = inherited[key]
            return ReasoningResult({
                "query": {"type": "get_attribute", "concept": cid, "key": key},
                "conclusion": a["value"], "confidence": a["confidence"],
                "status": a["status"],
                "trace": [{"rule": "attribute_inheritance",
                           "facts": [f"{a['concept_id']}.{key}={a['value']} "
                                     f"status={a['status']} source={a['source']}"]}],
            })
        return ReasoningResult({
            "query": {"type": "get_attribute", "concept": cid, "key": key},
            "conclusion": "unknown", "confidence": 0.0,
            "trace": [{"rule": "attribute_inheritance",
                       "detail": f"no value for {key} on {cid} or ancestors"}],
        })

    def query_relations(self, src: str, rel_type: Optional[str] = None) -> ReasoningResult:
        rels = self.mem.get_relations(src_id=src, rel_type=rel_type)
        return ReasoningResult({
            "query": {"type": "relations", "src": src, "rel_type": rel_type},
            "conclusion": rels, "confidence": 1.0,
            "trace": [{"rule": "relation_lookup",
                       "facts": [f"found {len(rels)} relation(s)"]}],
        })

    # ---- rule engine ----
    def _cond_holds(self, cond: dict) -> tuple[bool | None, str]:
        """Returns (True/False/None-unknown, explanation)."""
        t = cond.get("type")
        if t == "is_a":
            r = self.is_a(cond["concept"], cond["ancestor"])
            if r["conclusion"] == "unknown":
                return None, f"unknown is_a {cond['concept']}->{cond['ancestor']}"
            return bool(r["conclusion"]), f"is_a {cond['concept']}->{cond['ancestor']}={r['conclusion']}"
        if t == "has_attr":
            r = self.get_attribute(cond["concept"], cond["key"])
            if r["conclusion"] == "unknown":
                return None, f"unknown attr {cond['concept']}.{cond['key']}"
            if "value" in cond:
                return (r["conclusion"] == cond["value"],
                        f"{cond['concept']}.{cond['key']}={r['conclusion']} expected={cond['value']}")
            return True, f"{cond['concept']} has {cond['key']}"
        if t == "relation":
            rels = self.mem.get_relations(src_id=cond.get("src"),
                                          rel_type=cond.get("rel_type"))
            want = cond.get("dst")
            if want is None:
                return (len(rels) > 0), f"relation exists: {len(rels)}"
            hit = any(r["dst_id"] == want for r in rels)
            return hit, f"relation {cond.get('src')} -{cond.get('rel_type')}-> {want} = {hit}"
        return None, f"unsupported condition type: {t}"

    def forward_chain(self, max_steps: int = 50) -> list[dict]:
        """Apply all rules once (with exception handling). Returns fired conclusions."""
        fired: list[dict] = []
        for rule in self.mem.list_rules():
            conds = rule.get("conditions", [])
            excs = rule.get("exceptions", [])
            # exceptions: if any holds -> skip
            skipped = False
            for e in excs:
                v, _ = self._cond_holds(e)
                if v is True:
                    fired.append({"rule": rule["id"], "skipped": True,
                                  "reason": f"exception holds: {e}"})
                    skipped = True
                    break
            if skipped:
                continue
            results = [self._cond_holds(c) for c in conds]
            if any(v is None for v, _ in results):
                fired.append({"rule": rule["id"], "conclusion": "unknown",
                              "reason": "missing evidence: " +
                              "; ".join(d for _, d in results)})
                continue
            if all(v is True for v, _ in results):
                fired.append({"rule": rule["id"],
                              "conclusion": rule.get("conclusion"),
                              "trace": [d for _, d in results]})
            else:
                fired.append({"rule": rule["id"], "conclusion": "not_fired",
                              "trace": [d for _, d in results]})
            if len(fired) >= max_steps:
                break
        return fired

    def check_conflicts(self) -> list[dict]:
        """Find same concept+key with different values, or contested claims."""
        conflicts: list[dict] = []
        for c in self.mem.list_concepts():
            by_key: dict[str, set] = {}
            for a in self.mem.get_attributes(c["id"]):
                by_key.setdefault(a["key"], set()).add(json.dumps(a["value"], sort_keys=True))
            for k, vals in by_key.items():
                if len(vals) > 1:
                    conflicts.append({"type": "attribute_conflict",
                                      "concept": c["id"], "key": k,
                                      "values": sorted(vals)})
        for cl in self.mem.list_claims():
            if cl["status"] == "contested":
                conflicts.append({"type": "contested_claim", "claim": cl})
        return conflicts

    def answer(self, query: dict) -> ReasoningResult:
        qtype = query.get("type")
        if qtype == "is_a":
            return self.is_a(query["concept"], query["ancestor"])
        if qtype == "get_attribute":
            return self.get_attribute(query["concept"], query["key"])
        if qtype == "relations":
            return self.query_relations(query.get("src"), query.get("rel_type"))
        if qtype == "forward_chain":
            return ReasoningResult({"query": query,
                                    "conclusion": self.forward_chain(),
                                    "confidence": 1.0, "trace": []})
        if qtype == "conflicts":
            return ReasoningResult({"query": query,
                                    "conclusion": self.check_conflicts(),
                                    "confidence": 1.0, "trace": []})
        return ReasoningResult({"query": query, "conclusion": "unknown",
                                "confidence": 0.0,
                                "trace": [{"detail": "unsupported query type"}]})
