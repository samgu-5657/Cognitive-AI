"""SQLite-backed memory. The ONLY persistence entry point."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS concepts(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'Entity',
  confidence REAL NOT NULL DEFAULT 1.0,
  source TEXT NOT NULL DEFAULT 'seed',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS concept_parents(
  concept_id TEXT NOT NULL,
  parent_id TEXT NOT NULL,
  PRIMARY KEY(concept_id, parent_id)
);
CREATE TABLE IF NOT EXISTS attributes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  concept_id TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.8,
  source TEXT NOT NULL DEFAULT 'observation',
  status TEXT NOT NULL DEFAULT 'hypothesis'
);
CREATE TABLE IF NOT EXISTS relations(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  src_id TEXT NOT NULL,
  rel_type TEXT NOT NULL,
  dst_id TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.8,
  source TEXT NOT NULL DEFAULT 'observation',
  status TEXT NOT NULL DEFAULT 'hypothesis'
);
CREATE TABLE IF NOT EXISTS rules(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  conditions TEXT NOT NULL DEFAULT '[]',
  conclusion TEXT NOT NULL DEFAULT '{}',
  exceptions TEXT NOT NULL DEFAULT '[]',
  source TEXT NOT NULL DEFAULT 'seed',
  status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiences(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  summary TEXT NOT NULL,
  detail TEXT NOT NULL DEFAULT '{}',
  prediction TEXT NOT NULL DEFAULT '{}',
  outcome TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS claims(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  subject_id TEXT,
  status TEXT NOT NULL DEFAULT 'hypothesis',
  confidence REAL NOT NULL DEFAULT 0.5,
  evidence_ids TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidences(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  claim_id INTEGER,
  content TEXT NOT NULL,
  source TEXT NOT NULL,
  credibility REAL NOT NULL DEFAULT 0.5,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS questions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'gap',
  target_id TEXT,
  score REAL NOT NULL DEFAULT 0.0,
  status TEXT NOT NULL DEFAULT 'proposed',
  detail TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_versions(
  version INTEGER PRIMARY KEY,
  snapshot TEXT NOT NULL,
  reason TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_changes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL DEFAULT '{}',
  reason TEXT NOT NULL DEFAULT '',
  evidence TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'proposed',
  version_before INTEGER,
  version_after INTEGER,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- Phase M3C: not-yet-accepted learning candidates. Rows here are
-- invisible to reasoning; only proposed/accepted/rejected bookkeeping.
CREATE TABLE IF NOT EXISTS pending_candidates(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL DEFAULT '',
  utterance TEXT NOT NULL DEFAULT '',
  parse TEXT NOT NULL DEFAULT '{}',
  observation TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'proposed',
  reason TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
-- Phase 2: knowledge lifecycle history (additive; old DBs gain it on open)
CREATE TABLE IF NOT EXISTS lifecycle_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  target_type TEXT NOT NULL,
  target_ref TEXT NOT NULL,
  from_state TEXT NOT NULL,
  to_state TEXT NOT NULL,
  reason TEXT NOT NULL DEFAULT '',
  evidence TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);
"""

# Additive per-column migrations for databases created before Phase 2.
# Only ADD COLUMN with defaults: old rows stay readable, new code treats
# '' as 'unknown/ungraded'.
EVIDENCE_MIGRATION_COLUMNS = (
    ("content_hash", "TEXT NOT NULL DEFAULT ''"),
    ("relevance", "TEXT NOT NULL DEFAULT ''"),
    ("origin_note", "TEXT NOT NULL DEFAULT ''"),
)

# Phase M4-D: world-instance / state-snapshot / simulation-event storage.
# Separate tables; existing knowledge rows are never migrated or touched.
WORLD_MODEL_STORAGE_VERSION = "world-model-storage-v1"

WORLD_MODEL_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS worlds(
  world_id TEXT PRIMARY KEY,
  name TEXT NOT NULL DEFAULT '',
  storage_format_version TEXT NOT NULL DEFAULT 'world-model-storage-v1'
);
CREATE TABLE IF NOT EXISTS world_states(
  state_id TEXT PRIMARY KEY,
  world_id TEXT NOT NULL,
  parent_state_id TEXT,
  step_index INTEGER NOT NULL DEFAULT 0 CHECK(step_index >= 0),
  state_format_version TEXT NOT NULL DEFAULT 'world-state-v1'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_world_states_initial
  ON world_states(world_id) WHERE parent_state_id IS NULL;
CREATE TABLE IF NOT EXISTS world_state_facts(
  state_id TEXT NOT NULL,
  fact_index INTEGER NOT NULL,
  content TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY(state_id, fact_index)
);
CREATE TABLE IF NOT EXISTS world_events(
  event_id TEXT PRIMARY KEY,
  world_id TEXT NOT NULL,
  action_id TEXT NOT NULL DEFAULT '',
  pre_state_id TEXT NOT NULL,
  post_state_id TEXT NOT NULL UNIQUE,
  step_index INTEGER NOT NULL DEFAULT 0 CHECK(step_index >= 0),
  bindings TEXT NOT NULL DEFAULT '[]',
  effects TEXT NOT NULL DEFAULT '[]',
  simulated INTEGER NOT NULL DEFAULT 1
);
"""


class Memory:
    """Thin SQLite wrapper. All writes commit immediately."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # check_same_thread=False + a lock so one Memory can be shared
        # across threads; all access goes through execute()/query().
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.RLock()
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()
        self._migrate()
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", "0")
        if self.get_meta("current_version") is None:
            self.set_meta("current_version", "0")

    def _migrate(self) -> None:
        """Backward-compatible migrations: additive only, never rewrite data."""
        cols = {r["name"] for r in self.query("PRAGMA table_info(evidences)")}
        with self.lock:
            for name, ddl in EVIDENCE_MIGRATION_COLUMNS:
                if name not in cols:
                    self.conn.execute(
                        f"ALTER TABLE evidences ADD COLUMN {name} {ddl}")
            # Phase M2: rules gain a lifecycle status. Old rows keep
            # working exactly as before: missing status reads as active.
            rule_cols = {r["name"] for r in
                         self.query("PRAGMA table_info(rules)")}
            if "status" not in rule_cols:
                self.conn.execute(
                    "ALTER TABLE rules ADD COLUMN status "
                    "TEXT NOT NULL DEFAULT 'active'")
            self.conn.commit()

    # ---- generic helpers ----
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self.lock:
            cur = self.conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def query_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        with self.lock:
            cur = self.conn.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None

    # ---- meta ----
    def get_meta(self, key: str) -> Optional[str]:
        row = self.query_one("SELECT value FROM meta WHERE key=?", (key,))
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def current_version(self) -> int:
        return int(self.get_meta("current_version") or "0")

    def set_current_version(self, v: int) -> None:
        self.set_meta("current_version", str(v))

    # ---- concepts ----
    def upsert_concept(self, c: dict) -> None:
        self.execute(
            "INSERT INTO concepts(id,name,kind,confidence,source,created_at,updated_at,version)"
            " VALUES(?,?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET name=excluded.name, kind=excluded.kind,"
            " confidence=excluded.confidence, source=excluded.source,"
            " updated_at=excluded.updated_at, version=excluded.version",
            (
                c["id"], c["name"], c.get("kind", "Entity"),
                float(c.get("confidence", 1.0)), c.get("source", "seed"),
                c.get("created_at", _utcnow()), c.get("updated_at", _utcnow()),
                int(c.get("version", 1)),
            ),
        )

    def get_concept(self, cid: str) -> Optional[dict]:
        return self.query_one("SELECT * FROM concepts WHERE id=?", (cid,))

    def find_concepts_by_name(self, name: str) -> list[dict]:
        return self.query(
            "SELECT * FROM concepts WHERE lower(name)=lower(?)", (name,)
        )

    def list_concepts(self) -> list[dict]:
        return self.query("SELECT * FROM concepts ORDER BY id")

    def set_parents(self, cid: str, parents: list[str]) -> None:
        self.execute("DELETE FROM concept_parents WHERE concept_id=?", (cid,))
        for p in parents:
            self.execute(
                "INSERT OR IGNORE INTO concept_parents(concept_id,parent_id) VALUES(?,?)",
                (cid, p),
            )

    def add_parent(self, cid: str, pid: str) -> None:
        self.execute(
            "INSERT OR IGNORE INTO concept_parents(concept_id,parent_id) VALUES(?,?)",
            (cid, pid),
        )

    def remove_parent(self, cid: str, pid: str) -> None:
        self.execute(
            "DELETE FROM concept_parents WHERE concept_id=? AND parent_id=?", (cid, pid)
        )

    def get_parents(self, cid: str) -> list[str]:
        rows = self.query(
            "SELECT parent_id FROM concept_parents WHERE concept_id=? ORDER BY parent_id",
            (cid,),
        )
        return [r["parent_id"] for r in rows]

    def get_children(self, pid: str) -> list[str]:
        rows = self.query(
            "SELECT concept_id FROM concept_parents WHERE parent_id=? ORDER BY concept_id",
            (pid,),
        )
        return [r["concept_id"] for r in rows]

    # ---- attributes ----
    def add_attribute(self, concept_id: str, key: str, value: Any,
                      confidence: float = 0.8, source: str = "observation",
                      status: str = "hypothesis") -> int:
        cur = self.execute(
            "INSERT INTO attributes(concept_id,key,value,confidence,source,status)"
            " VALUES(?,?,?,?,?,?)",
            (concept_id, key, json.dumps(value, ensure_ascii=False),
             confidence, source, status),
        )
        return int(cur.lastrowid)

    def get_attributes(self, concept_id: str) -> list[dict]:
        rows = self.query(
            "SELECT * FROM attributes WHERE concept_id=? ORDER BY key", (concept_id,)
        )
        for r in rows:
            try:
                r["value"] = json.loads(r["value"])
            except Exception:
                pass
        return rows

    def update_attribute_status(self, aid: int, status: str,
                                confidence: Optional[float] = None) -> None:
        if confidence is None:
            self.execute("UPDATE attributes SET status=? WHERE id=?", (status, aid))
        else:
            self.execute(
                "UPDATE attributes SET status=?, confidence=? WHERE id=?",
                (status, confidence, aid),
            )

    # ---- relations ----
    def add_relation(self, src_id: str, rel_type: str, dst_id: str,
                     confidence: float = 0.8, source: str = "observation",
                     status: str = "hypothesis") -> int:
        cur = self.execute(
            "INSERT INTO relations(src_id,rel_type,dst_id,confidence,source,status)"
            " VALUES(?,?,?,?,?,?)",
            (src_id, rel_type, dst_id, confidence, source, status),
        )
        return int(cur.lastrowid)

    def get_relations(self, src_id: Optional[str] = None,
                      rel_type: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM relations WHERE 1=1"
        params: list[Any] = []
        if src_id is not None:
            sql += " AND src_id=?"
            params.append(src_id)
        if rel_type is not None:
            sql += " AND rel_type=?"
            params.append(rel_type)
        sql += " ORDER BY id"
        return self.query(sql, tuple(params))

    def relations_involving(self, cid: str) -> list[dict]:
        return self.query(
            "SELECT * FROM relations WHERE src_id=? OR dst_id=? ORDER BY id", (cid, cid)
        )

    # ---- rules ----
    def upsert_rule(self, rule: dict) -> None:
        # Status compat: a fresh row without explicit status starts
        # active (previous behavior: every stored rule was usable). A
        # re-upsert never silently flips an existing lifecycle state.
        existing = self.query_one("SELECT status FROM rules WHERE id=?",
                                  (rule["id"],))
        status = rule.get("status")
        if status is None:
            status = existing["status"] if existing and \
                existing.get("status") else "active"
        self.execute(
            "INSERT INTO rules(id,name,conditions,conclusion,exceptions,source,status,created_at)"
            " VALUES(?,?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET name=excluded.name, conditions=excluded.conditions,"
            " conclusion=excluded.conclusion, exceptions=excluded.exceptions,"
            " source=excluded.source, status=excluded.status",
            (
                rule["id"], rule.get("name", rule["id"]),
                json.dumps(rule.get("conditions", []), ensure_ascii=False),
                json.dumps(rule.get("conclusion", {}), ensure_ascii=False),
                json.dumps(rule.get("exceptions", []), ensure_ascii=False),
                rule.get("source", "seed"),
                status,
                rule.get("created_at", _utcnow()),
            ),
        )

    def set_rule_status(self, rule_id: str, status: str,
                        reason: str = "") -> str:
        """Move one rule through its lifecycle with an audit record.

        Returns the previous status. Only the RuleStatus vocabulary is
        accepted; unknown ids raise KeyError.
        """
        from .models import RuleStatus
        allowed = {s.value for s in RuleStatus}
        if status not in allowed:
            raise ValueError(f"unknown rule status: {status}")
        row = self.query_one("SELECT status FROM rules WHERE id=?",
                             (rule_id,))
        if row is None:
            raise KeyError(f"rule not found: {rule_id}")
        old = row.get("status") or "active"
        if old != status:
            self.execute("UPDATE rules SET status=? WHERE id=?",
                         (status, rule_id))
            self.record_lifecycle("rule", rule_id, old, status, reason)
        return old

    def list_rules(self) -> list[dict]:
        rows = self.query("SELECT * FROM rules ORDER BY id")
        for r in rows:
            for k in ("conditions", "conclusion", "exceptions"):
                try:
                    r[k] = json.loads(r[k])
                except Exception:
                    r[k] = [] if k != "conclusion" else {}
        return rows

    def delete_rule(self, rid: str) -> None:
        self.execute("DELETE FROM rules WHERE id=?", (rid,))

    # ---- experiences ----
    def add_experience(self, kind: str, summary: str, detail: dict | None = None,
                       prediction: dict | None = None,
                       outcome: dict | None = None) -> int:
        cur = self.execute(
            "INSERT INTO experiences(kind,summary,detail,prediction,outcome,created_at)"
            " VALUES(?,?,?,?,?,?)",
            (kind, summary,
             json.dumps(detail or {}, ensure_ascii=False),
             json.dumps(prediction or {}, ensure_ascii=False),
             json.dumps(outcome or {}, ensure_ascii=False),
             _utcnow()),
        )
        return int(cur.lastrowid)

    def list_experiences(self, limit: int = 50) -> list[dict]:
        return self.query(
            "SELECT * FROM experiences ORDER BY id DESC LIMIT ?", (limit,)
        )

    # ---- pending learning candidates (Phase M3C) ----
    def add_pending_candidate(self, kind: str, utterance: str,
                              parse: dict, observation: dict,
                              reason: str = "") -> int:
        cur = self.execute(
            "INSERT INTO pending_candidates(kind,utterance,parse,"
            "observation,status,reason,created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (kind, utterance, json.dumps(parse, ensure_ascii=False),
             json.dumps(observation, ensure_ascii=False), "proposed",
             reason, _utcnow()))
        return int(cur.lastrowid)

    def get_pending_candidate(self, pid: int) -> Optional[dict]:
        return self.query_one("SELECT * FROM pending_candidates WHERE id=?",
                              (pid,))

    def list_pending_candidates(self, status: Optional[str] = None) -> list[dict]:
        if status:
            return self.query("SELECT * FROM pending_candidates WHERE status=? "
                              "ORDER BY id", (status,))
        return self.query("SELECT * FROM pending_candidates ORDER BY id")

    def set_pending_status(self, pid: int, status: str) -> None:
        if status not in ("proposed", "accepted", "rejected"):
            raise ValueError(f"unknown pending status: {status}")
        self.execute("UPDATE pending_candidates SET status=? WHERE id=?",
                     (status, pid))

    # ---- claims / evidences ----
    def add_claim(self, text: str, subject_id: Optional[str] = None,
                  status: str = "hypothesis", confidence: float = 0.5) -> int:
        cur = self.execute(
            "INSERT INTO claims(text,subject_id,status,confidence,evidence_ids,created_at)"
            " VALUES(?,?,?,?,?,?)",
            (text, subject_id, status, confidence, "[]", _utcnow()),
        )
        return int(cur.lastrowid)

    def add_evidence(self, claim_id: Optional[int], content: str, source: str,
                     credibility: float = 0.5, content_hash: str = "",
                     relevance: str = "", origin_note: str = "") -> int:
        cur = self.execute(
            "INSERT INTO evidences(claim_id,content,source,credibility,"
            "content_hash,relevance,origin_note,created_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (claim_id, content, source, credibility, content_hash,
             relevance, origin_note, _utcnow()),
        )
        return int(cur.lastrowid)

    def list_evidences(self, claim_id: Optional[int] = None) -> list[dict]:
        if claim_id is None:
            return self.query("SELECT * FROM evidences ORDER BY id")
        return self.query("SELECT * FROM evidences WHERE claim_id=? ORDER BY id",
                          (claim_id,))

    def list_claims(self) -> list[dict]:
        return self.query("SELECT * FROM claims ORDER BY id")

    # ---- lifecycle events ----
    def record_lifecycle(self, target_type: str, target_ref: str,
                         from_state: str, to_state: str, reason: str = "",
                         evidence: list | None = None) -> int:
        cur = self.execute(
            "INSERT INTO lifecycle_events(target_type,target_ref,from_state,"
            "to_state,reason,evidence,created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (target_type, target_ref, from_state, to_state, reason,
             json.dumps(evidence or [], ensure_ascii=False), _utcnow()),
        )
        return int(cur.lastrowid)

    def list_lifecycle(self, target_ref: Optional[str] = None,
                       limit: int = 200) -> list[dict]:
        if target_ref is None:
            return self.query(
                "SELECT * FROM lifecycle_events ORDER BY id DESC LIMIT ?",
                (limit,))
        return self.query(
            "SELECT * FROM lifecycle_events WHERE target_ref=? ORDER BY id",
            (target_ref,))

    def transact(self, ops: list[tuple[str, tuple]]) -> None:
        """Run several writes in one transaction; any failure rolls back."""
        with self.lock:
            try:
                cur = self.conn.cursor()
                for sql, params in ops:
                    cur.execute(sql, params)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def apply_rule_change(self, *, rule_upserts=(), rule_statuses=(),
                          lifecycles=(), experiences=(),
                          change_marks=()) -> None:
        """Apply one rule change with all audit rows atomically.

        Each entry mirrors its single-purpose counterpart (upsert_rule,
        set_rule_status parts, record_lifecycle, add_experience,
        update_change_status) so the bundle changes no semantics, only
        the transaction boundary. Any failure rolls everything back.
        """
        ops = []
        for r in rule_upserts:
            ops.append((
                "INSERT INTO rules(id,name,conditions,conclusion,exceptions,"
                "source,status,created_at) VALUES(?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET name=excluded.name,"
                " conditions=excluded.conditions,"
                " conclusion=excluded.conclusion,"
                " exceptions=excluded.exceptions,"
                " source=excluded.source, status=excluded.status",
                (r["id"], r.get("name", r["id"]),
                 json.dumps(r.get("conditions", []), ensure_ascii=False),
                 json.dumps(r.get("conclusion", {}), ensure_ascii=False),
                 json.dumps(r.get("exceptions", []), ensure_ascii=False),
                 r.get("source", "seed"), r.get("status", "active"),
                 r.get("created_at", _utcnow())),
            ))
        for rule_id, status in rule_statuses:
            ops.append(("UPDATE rules SET status=? WHERE id=?",
                        (status, rule_id)))
        for entry in lifecycles:
            target_type, target_ref, from_state, to_state, reason, \
                evidence = entry
            ops.append((
                "INSERT INTO lifecycle_events(target_type,target_ref,"
                "from_state,to_state,reason,evidence,created_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (target_type, target_ref, from_state, to_state, reason,
                 json.dumps(evidence or [], ensure_ascii=False),
                 _utcnow()),
            ))
        for entry in experiences:
            kind, summary, detail = entry
            ops.append((
                "INSERT INTO experiences(kind,summary,detail,prediction,"
                "outcome,created_at) VALUES(?,?,?,?,?,?)",
                (kind, summary,
                 json.dumps(detail or {}, ensure_ascii=False),
                 json.dumps({}, ensure_ascii=False),
                 json.dumps({}, ensure_ascii=False), _utcnow()),
            ))
        for change_id, status in change_marks:
            ops.append(("UPDATE schema_changes SET status=? WHERE id=?",
                        (status, change_id)))
        self.transact(ops)

    # ---- world-model storage (Phase M4-D) ----
    def ensure_world_tables(self) -> str:
        """Create world tables idempotently; record the storage version.

        Re-running is safe: tables use IF NOT EXISTS and the version
        marker is only set, never used to rewrite data. Statements run
        one by one because cursor.execute() takes single statements;
        DDL autocommits in SQLite, so crash-safety comes from the
        marker (set last: absent marker means retry) rather than from
        rolling back table creation.
        """
        statements = [s.strip() for s in WORLD_MODEL_TABLES_SQL.split(";")
                      if s.strip()]
        self.transact([(s, ()) for s in statements])
        self.set_meta("world_model_storage_version",
                      WORLD_MODEL_STORAGE_VERSION)
        return WORLD_MODEL_STORAGE_VERSION

    def get_world_storage_version(self) -> Optional[str]:
        return self.get_meta("world_model_storage_version")

    def insert_world(self, world_id: str, name: str,
                     storage_version: str) -> None:
        self.execute(
            "INSERT INTO worlds(world_id,name,storage_format_version)"
            " VALUES(?,?,?)", (world_id, name, storage_version))

    def get_world(self, world_id: str) -> Optional[dict]:
        return self.query_one("SELECT * FROM worlds WHERE world_id=?",
                              (world_id,))

    def list_worlds(self) -> list[dict]:
        return self.query("SELECT * FROM worlds ORDER BY world_id")

    def insert_world_state(self, state_id: str, world_id: str,
                           parent_state_id: Optional[str],
                           step_index: int,
                           state_version: str) -> None:
        self.execute(
            "INSERT INTO world_states(state_id,world_id,parent_state_id,"
            "step_index,state_format_version) VALUES(?,?,?,?,?)",
            (state_id, world_id, parent_state_id, step_index,
             state_version))

    def get_world_state(self, state_id: str) -> Optional[dict]:
        return self.query_one("SELECT * FROM world_states WHERE state_id=?",
                              (state_id,))

    def list_world_states(self, world_id: str) -> list[dict]:
        return self.query(
            "SELECT * FROM world_states WHERE world_id=? "
            "ORDER BY step_index, state_id", (world_id,))

    def get_initial_world_state(self, world_id: str) -> Optional[dict]:
        return self.query_one(
            "SELECT * FROM world_states WHERE world_id=? "
            "AND parent_state_id IS NULL", (world_id,))

    def insert_world_fact(self, state_id: str, fact_index: int,
                          content: str) -> None:
        self.execute(
            "INSERT INTO world_state_facts(state_id,fact_index,content)"
            " VALUES(?,?,?)", (state_id, fact_index, content))

    def get_world_fact_rows(self, state_id: str) -> list[dict]:
        return self.query(
            "SELECT * FROM world_state_facts WHERE state_id=? "
            "ORDER BY fact_index", (state_id,))

    def insert_world_event(self, event_id: str, world_id: str,
                           action_id: str, pre_state_id: str,
                           post_state_id: str, step_index: int,
                           bindings: str, effects: str,
                           simulated: int) -> None:
        self.execute(
            "INSERT INTO world_events(event_id,world_id,action_id,"
            "pre_state_id,post_state_id,step_index,bindings,effects,"
            "simulated) VALUES(?,?,?,?,?,?,?,?,?)",
            (event_id, world_id, action_id, pre_state_id, post_state_id,
             step_index, bindings, effects, simulated))

    def get_world_event(self, event_id: str) -> Optional[dict]:
        return self.query_one("SELECT * FROM world_events WHERE event_id=?",
                              (event_id,))

    def get_world_event_by_post(self, post_state_id: str) -> Optional[dict]:
        return self.query_one(
            "SELECT * FROM world_events WHERE post_state_id=?",
            (post_state_id,))

    def list_world_events(self, world_id: str) -> list[dict]:
        return self.query(
            "SELECT * FROM world_events WHERE world_id=? "
            "ORDER BY step_index, event_id", (world_id,))

    # ---- questions ----
    def find_question(self, text: str,
                      target_id: Optional[str] = None) -> Optional[dict]:
        """Exact (text, target) lookup across ALL statuses (incl. learned)."""
        return self.query_one(
            "SELECT * FROM questions WHERE text=? AND "
            "COALESCE(target_id,'')=COALESCE(?, '')",
            (text, target_id),
        )

    def add_question(self, text: str, kind: str = "gap",
                     target_id: Optional[str] = None, score: float = 0.0,
                     status: str = "proposed", detail: dict | None = None) -> int:
        # dedupe: the same (text, target) gap is one question, forever.
        # A resolved (learned) question is never duplicated; it simply is
        # not re-created. Callers use find_question + update_question to
        # refresh scores of existing rows.
        row = self.find_question(text, target_id)
        if row:
            return int(row["id"])
        cur = self.execute(
            "INSERT INTO questions(text,kind,target_id,score,status,detail,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (text, kind, target_id, score, status,
             json.dumps(detail or {}, ensure_ascii=False), _utcnow(), _utcnow()),
        )
        return int(cur.lastrowid)

    def list_questions(self, status: Optional[str] = None,
                       limit: int = 100) -> list[dict]:
        if status:
            return self.query(
                "SELECT * FROM questions WHERE status=? ORDER BY score DESC, id",
                (status,),
            )
        return self.query(
            "SELECT * FROM questions ORDER BY score DESC, id LIMIT ?", (limit,)
        )

    def update_question(self, qid: int, **fields: Any) -> None:
        allowed = {"text", "kind", "target_id", "score", "status", "detail"}
        sets, params = [], []
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k == "detail":
                v = json.dumps(v, ensure_ascii=False)
            sets.append(f"{k}=?")
            params.append(v)
        if not sets:
            return
        sets.append("updated_at=?")
        params.append(_utcnow())
        params.append(qid)
        self.execute(f"UPDATE questions SET {', '.join(sets)} WHERE id=?",
                     tuple(params))

    # ---- schema versions / changes ----
    def save_version(self, version: int, snapshot: dict, reason: str = "") -> None:
        self.execute(
            "INSERT INTO schema_versions(version,snapshot,reason,created_at)"
            " VALUES(?,?,?,?)"
            " ON CONFLICT(version) DO UPDATE SET snapshot=excluded.snapshot,"
            " reason=excluded.reason",
            (version, json.dumps(snapshot, ensure_ascii=False), reason, _utcnow()),
        )
        self.set_current_version(str(version))

    def get_version(self, version: int) -> Optional[dict]:
        return self.query_one(
            "SELECT * FROM schema_versions WHERE version=?", (version,)
        )

    def list_versions(self) -> list[dict]:
        return self.query("SELECT version, reason, created_at FROM schema_versions ORDER BY version")

    def record_change(self, kind: str, payload: dict, reason: str = "",
                      evidence: list | None = None, status: str = "proposed",
                      v_before: Optional[int] = None,
                      v_after: Optional[int] = None) -> int:
        cur = self.execute(
            "INSERT INTO schema_changes(kind,payload,reason,evidence,status,"
            "version_before,version_after,created_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (kind, json.dumps(payload, ensure_ascii=False), reason,
             json.dumps(evidence or [], ensure_ascii=False), status,
             v_before, v_after, _utcnow()),
        )
        return int(cur.lastrowid)

    def list_changes(self, limit: int = 100) -> list[dict]:
        return self.query(
            "SELECT * FROM schema_changes ORDER BY id DESC LIMIT ?", (limit,)
        )

    def update_change_status(self, cid: int, status: str,
                             v_after: Optional[int] = None) -> None:
        if v_after is None:
            self.execute("UPDATE schema_changes SET status=? WHERE id=?",
                         (status, cid))
        else:
            self.execute(
                "UPDATE schema_changes SET status=?, version_after=? WHERE id=?",
                (status, v_after, cid),
            )

    # ---- snapshot ----
    def snapshot(self) -> dict:
        return {
            "concepts": self.list_concepts(),
            "parents": self.query("SELECT * FROM concept_parents ORDER BY concept_id"),
            "attributes": self.query("SELECT * FROM attributes ORDER BY id"),
            "relations": self.query("SELECT * FROM relations ORDER BY id"),
            "rules": self.query("SELECT * FROM rules ORDER BY id"),
        }

    def restore_snapshot(self, snap: dict) -> None:
        self.execute("DELETE FROM concept_parents")
        self.execute("DELETE FROM attributes")
        self.execute("DELETE FROM relations")
        self.execute("DELETE FROM rules")
        self.execute("DELETE FROM concepts")
        for c in snap.get("concepts", []):
            self.execute(
                "INSERT INTO concepts(id,name,kind,confidence,source,created_at,updated_at,version)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (c["id"], c["name"], c.get("kind", "Entity"),
                 c.get("confidence", 1.0), c.get("source", "seed"),
                 c.get("created_at", _utcnow()), c.get("updated_at", _utcnow()),
                 c.get("version", 1)),
            )
        for p in snap.get("parents", []):
            self.execute(
                "INSERT OR IGNORE INTO concept_parents(concept_id,parent_id) VALUES(?,?)",
                (p["concept_id"], p["parent_id"]),
            )
        for a in snap.get("attributes", []):
            self.execute(
                "INSERT INTO attributes(id,concept_id,key,value,confidence,source,status)"
                " VALUES(?,?,?,?,?,?,?)",
                (a["id"], a["concept_id"], a["key"], a["value"],
                 a["confidence"], a["source"], a["status"]),
            )
        for r in snap.get("relations", []):
            self.execute(
                "INSERT INTO relations(id,src_id,rel_type,dst_id,confidence,source,status)"
                " VALUES(?,?,?,?,?,?,?)",
                (r["id"], r["src_id"], r["rel_type"], r["dst_id"],
                 r["confidence"], r["source"], r["status"]),
            )
        for r in snap.get("rules", []):
            self.execute(
                "INSERT INTO rules(id,name,conditions,conclusion,exceptions,source,created_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (r["id"], r["name"], r["conditions"], r["conclusion"],
                 r["exceptions"], r["source"], r["created_at"]),
            )

    def close(self) -> None:
        self.conn.close()
