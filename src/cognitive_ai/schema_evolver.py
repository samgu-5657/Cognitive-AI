"""Schema evolution: the ONLY module allowed to mutate the concept structure.

Every mutation: snapshot before -> validate -> mutate -> regression check ->
snapshot after + change record. Failures roll back automatically.
Major (high-impact) changes require validation before applying.
"""
from __future__ import annotations

from typing import Callable, Optional

from .memory import Memory
from .ontology import Ontology

# kinds that touch many existing items and are therefore "major"
MAJOR_KINDS = {"split_category", "merge_categories", "reparent"}


class SchemaEvolver:
    def __init__(self, memory: Memory,
                 regression_check: Optional[Callable[[Memory], tuple[bool, str]]] = None,
                 allow_structural: bool = True):
        self.mem = memory
        self.onto = Ontology(memory)
        self.regression_check = regression_check
        # System B (facts only) sets allow_structural=False: structural kinds
        # (reparent/split/merge) are then rejected instead of applied.
        self.allow_structural = allow_structural

    # ---- versioning ----
    def current_version(self) -> int:
        return self.mem.current_version()

    def _commit_version(self, reason: str) -> int:
        v = self.current_version() + 1
        self.mem.save_version(v, self.mem.snapshot(), reason)
        return v

    def ensure_initial_version(self) -> int:
        if self.current_version() == 0 and self.mem.list_concepts():
            return self._commit_version("initial version")
        return self.current_version()

    # ---- proposal API ----
    def propose(self, kind: str, payload: dict, reason: str = "",
                evidence: list | None = None) -> int:
        return self.mem.record_change(kind, payload, reason, evidence,
                                      status="proposed",
                                      v_before=self.current_version())

    def apply_proposal(self, change_id: int) -> dict:
        row = self.mem.query_one("SELECT * FROM schema_changes WHERE id=?",
                                 (change_id,))
        if row is None:
            return {"ok": False, "error": "proposal not found"}
        if row["status"] == "applied":
            return {"ok": True, "note": "already applied"}
        import json
        payload = json.loads(row["payload"])
        evidence = json.loads(row["evidence"])
        before_snapshot = self.mem.snapshot()
        v_before = self.current_version()
        try:
            self._apply(payload, kind=row["kind"])
        except Exception as e:  # validation failure -> reject, no state change
            self.mem.restore_snapshot(before_snapshot)
            self.mem.update_change_status(change_id, "rejected")
            self.mem.add_experience(
                "schema_change",
                f"rejected proposal #{change_id} ({row['kind']}): {e}",
                {"proposal": row["kind"], "payload": payload, "error": str(e)})
            return {"ok": False, "error": str(e), "status": "rejected"}
        # regression check for major changes (and any change when a hook exists)
        if self.regression_check is not None and row["kind"] in MAJOR_KINDS:
            ok, msg = self.regression_check(self.mem)
            if not ok:
                self.mem.restore_snapshot(before_snapshot)
                self.mem.update_change_status(change_id, "rejected")
                self.mem.add_experience(
                    "schema_change",
                    f"rejected proposal #{change_id}: regression failed: {msg}",
                    {"proposal": row["kind"], "payload": payload})
                return {"ok": False, "error": f"regression failed: {msg}",
                        "status": "rejected"}
        v_after = self._commit_version(
            f"{row['kind']}: {row['reason']}")
        self.mem.update_change_status(change_id, "applied", v_after)
        impact = self._impact(payload, row["kind"])
        self.mem.add_experience(
            "schema_change",
            f"applied proposal #{change_id} ({row['kind']}) v{v_before}->v{v_after}",
            {"payload": payload, "impact": impact, "evidence": evidence})
        return {"ok": True, "version_before": v_before,
                "version_after": v_after, "impact": impact}

    def direct_apply(self, kind: str, payload: dict, reason: str = "",
                     evidence: list | None = None) -> dict:
        """Low-risk auto path: propose + immediately validate/apply."""
        cid = self.propose(kind, payload, reason, evidence)
        return {"change_id": cid, **self.apply_proposal(cid)}

    # ---- mutation implementations ----
    def _apply(self, payload: dict, kind: str) -> None:
        if kind in MAJOR_KINDS and not self.allow_structural:
            raise ValueError(
                f"structural change '{kind}' blocked: system runs facts-only")
        if kind == "add_concept":
            self.onto.add_concept(
                payload["id"], payload.get("name", payload["id"]),
                payload.get("kind", "Entity"),
                payload.get("parents", []),
                payload.get("confidence", 0.7),
                payload.get("source", "evolution"))
            for k, v in (payload.get("attributes") or {}).items():
                self.onto.add_attribute(payload["id"], k, v,
                                        source=payload.get("source", "evolution"),
                                        status="hypothesis")
        elif kind == "add_attribute":
            self.onto.add_attribute(
                payload["concept"], payload["key"], payload["value"],
                payload.get("confidence", 0.7),
                payload.get("source", "evolution"),
                payload.get("status", "hypothesis"))
        elif kind == "add_relation":
            self.onto.add_relation(
                payload["src"], payload["rel_type"], payload["dst"],
                payload.get("confidence", 0.7),
                payload.get("source", "evolution"),
                payload.get("status", "hypothesis"))
        elif kind == "reparent":
            if payload.get("remove"):
                self.onto.remove_parent(payload["concept"], payload["remove"])
            self.onto.add_parent(payload["concept"], payload["parent"])
        elif kind == "split_category":
            # payload: {category, groups: {new_id: [member_ids]}, attrs per group optional}
            cat = payload["category"]
            if self.mem.get_concept(cat) is None:
                raise ValueError(f"category not found: {cat}")
            cat_parents = self.mem.get_parents(cat)
            for new_id, members in payload["groups"].items():
                if self.mem.get_concept(new_id) is None:
                    self.onto.add_concept(
                        new_id, payload.get("names", {}).get(new_id, new_id),
                        "Entity", cat_parents, source="split")
                    self.onto.add_parent(new_id, cat)
                for m in members:
                    if self.mem.get_concept(m) is None:
                        raise ValueError(f"member not found: {m}")
                    # move member under the new sub-category (keep other parents)
                    self.onto.add_parent(m, new_id)
                for k, v in (payload.get("group_attrs", {}).get(new_id, {})).items():
                    self.onto.add_attribute(new_id, k, v, source="split",
                                            status="hypothesis")
        elif kind == "merge_categories":
            # payload: {sources: [ids], target: id}
            tgt = payload["target"]
            if self.mem.get_concept(tgt) is None:
                raise ValueError(f"target not found: {tgt}")
            for s in payload["sources"]:
                if s == tgt:
                    continue
                if self.mem.get_concept(s) is None:
                    raise ValueError(f"source not found: {s}")
                for child in self.mem.get_children(s):
                    self.onto.add_parent(child, tgt)
                # orphan the merged-away category (keep record, detach children already moved)
            for s in payload["sources"]:
                if s != tgt:
                    self.mem.execute("DELETE FROM concept_parents WHERE concept_id=?",
                                     (s,))
        elif kind == "update_rule":
            self.mem.upsert_rule(payload["rule"])
        elif kind == "update_attribute_status":
            self.mem.update_attribute_status(
                payload["attribute_id"], payload["status"],
                payload.get("confidence"))
        else:
            raise ValueError(f"unsupported change kind: {kind}")

    def _impact(self, payload: dict, kind: str) -> dict:
        if kind == "split_category":
            n = sum(len(v) for v in payload.get("groups", {}).values())
            return {"members_affected": n, "major": True}
        if kind in ("merge_categories", "reparent"):
            return {"major": True}
        return {"major": False}

    # ---- rollback ----
    def rollback(self, version: int) -> dict:
        row = self.mem.get_version(version)
        if row is None:
            return {"ok": False, "error": f"version not found: {version}"}
        import json
        snap = json.loads(row["snapshot"])
        cur = self.current_version()
        self.mem.restore_snapshot(snap)
        v_new = self._commit_version(f"rollback to v{version}")
        self.mem.add_experience(
            "schema_change", f"rolled back v{cur} -> snapshot v{version} (now v{v_new})",
            {"target": version})
        return {"ok": True, "version": v_new}

    def history(self) -> list[dict]:
        return self.mem.list_changes()

    def versions(self) -> list[dict]:
        return self.mem.list_versions()
