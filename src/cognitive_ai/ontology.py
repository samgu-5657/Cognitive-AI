"""Concept system: CRUD, multiple inheritance, cycle detection, inheritance queries."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .memory import Memory
from .models import utcnow


class OntologyError(ValueError):
    pass


class Ontology:
    def __init__(self, memory: Memory):
        self.mem = memory

    # ---- create / read ----
    def add_concept(self, cid: str, name: str, kind: str = "Entity",
                    parents: list[str] | None = None,
                    confidence: float = 1.0, source: str = "seed") -> dict:
        parents = parents or []
        for p in parents:
            if self.mem.get_concept(p) is None:
                raise OntologyError(f"parent concept not found: {p}")
        now = utcnow()
        existing = self.mem.get_concept(cid)
        version = (existing["version"] + 1) if existing else 1
        created = existing["created_at"] if existing else now
        self.mem.upsert_concept({
            "id": cid, "name": name, "kind": kind, "confidence": confidence,
            "source": source, "created_at": created, "updated_at": now,
            "version": version,
        })
        self.mem.set_parents(cid, [])
        for p in parents:
            self.add_parent(cid, p)
        return self.describe(cid)

    def get(self, cid: str) -> Optional[dict]:
        c = self.mem.get_concept(cid)
        if c is None:
            return None
        c["parents"] = self.mem.get_parents(cid)
        return c

    def find_by_name(self, name: str) -> list[dict]:
        return self.mem.find_concepts_by_name(name)

    def list(self) -> list[dict]:
        out = []
        for c in self.mem.list_concepts():
            c["parents"] = self.mem.get_parents(c["id"])
            out.append(c)
        return out

    def describe(self, cid: str) -> dict:
        c = self.get(cid)
        if c is None:
            raise OntologyError(f"concept not found: {cid}")
        c["attributes"] = self.mem.get_attributes(cid)
        c["relations"] = self.mem.get_relations(src_id=cid)
        c["children"] = self.mem.get_children(cid)
        c["ancestors"] = self.ancestors(cid)
        c["inherited_attributes"] = self.inherited_attributes(cid)
        return c

    # ---- parents with cycle check ----
    def add_parent(self, cid: str, pid: str) -> None:
        if self.mem.get_concept(cid) is None:
            raise OntologyError(f"concept not found: {cid}")
        if self.mem.get_concept(pid) is None:
            raise OntologyError(f"parent not found: {pid}")
        if cid == pid:
            raise OntologyError("concept cannot be its own parent")
        # cycle: pid must not be a descendant of cid
        if cid in self.ancestors(pid):
            raise OntologyError(
                f"illegal cycle: {pid} is already a descendant of {cid}")
        self.mem.add_parent(cid, pid)

    def remove_parent(self, cid: str, pid: str) -> None:
        self.mem.remove_parent(cid, pid)

    def ancestors(self, cid: str) -> list[str]:
        seen: list[str] = []
        stack = list(self.mem.get_parents(cid))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.append(cur)
            stack.extend(self.mem.get_parents(cur))
        return sorted(seen)

    def descendants(self, pid: str) -> list[str]:
        seen: list[str] = []
        stack = list(self.mem.get_children(pid))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.append(cur)
            stack.extend(self.mem.get_children(cur))
        return sorted(seen)

    def is_a(self, cid: str, ancestor: str) -> bool:
        if cid == ancestor:
            return True
        return ancestor in self.ancestors(cid)

    def inherited_attributes(self, cid: str) -> dict[str, dict]:
        """Nearest definition wins. Returns {key: attr_row}."""
        merged: dict[str, dict] = {}
        # walk from farthest ancestors to self so nearer overrides
        order = list(reversed(self.ancestors(cid))) + [cid]
        for node in order:
            for a in self.mem.get_attributes(node):
                merged[a["key"]] = a
        return merged

    # ---- attributes / relations ----
    def add_attribute(self, cid: str, key: str, value: Any,
                      confidence: float = 0.8, source: str = "observation",
                      status: str = "hypothesis") -> int:
        if self.mem.get_concept(cid) is None:
            raise OntologyError(f"concept not found: {cid}")
        return self.mem.add_attribute(cid, key, value, confidence, source, status)

    def add_relation(self, src: str, rel: str, dst: str,
                     confidence: float = 0.8, source: str = "observation",
                     status: str = "hypothesis") -> int:
        for c in (src, dst):
            if self.mem.get_concept(c) is None:
                raise OntologyError(f"concept not found: {c}")
        return self.mem.add_relation(src, rel, dst, confidence, source, status)
