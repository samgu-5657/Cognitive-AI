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
-- Phase 4B: per-session dialogue context (additive)
CREATE TABLE IF NOT EXISTS dialogue_context(
  session TEXT PRIMARY KEY,
  state TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL
);
-- Phase 4C: persisted cognitive execution traces (additive)
CREATE TABLE IF NOT EXISTS cog_tasks(
  task_id TEXT PRIMARY KEY,
  task_type TEXT NOT NULL DEFAULT 'unknown',
  input_text TEXT NOT NULL DEFAULT '',
  trace TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'created',
  created_at TEXT NOT NULL
);
-- Phase 4D: learnable rules with versions + invocation counters (additive)
CREATE TABLE IF NOT EXISTS learnable_rules(
  id TEXT PRIMARY KEY,
  version INTEGER NOT NULL DEFAULT 1,
  name TEXT NOT NULL DEFAULT '',
  domain TEXT NOT NULL DEFAULT '',
  spec TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'candidate',
  validation TEXT NOT NULL DEFAULT '{}',
  calls INTEGER NOT NULL DEFAULT 0,
  successes INTEGER NOT NULL DEFAULT 0,
  failures INTEGER NOT NULL DEFAULT 0,
  previous TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
-- Phase 5B: user corrections to analogy mappings (additive)
CREATE TABLE IF NOT EXISTS analogy_feedback(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL DEFAULT '',
  source_task_id TEXT NOT NULL DEFAULT '',
  correction TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
-- Phase 6A: learned word -> canonical key mappings (additive)
CREATE TABLE IF NOT EXISTS lexicon(
  word TEXT PRIMARY KEY,
  key TEXT NOT NULL,
  evidence TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);
-- Phase 6C: learned expression -> canonical form mappings (additive).
-- Examples are user-taught (src => dst) pairs; rules are induced templates
-- (literal or prefix/suffix slot) with candidate/validated/active gating.
-- Nothing here is knowledge: rules only reshape input text before the
-- fixed compositional parser, and every application is logged in the
-- frame trace. Activation is always explicit, never automatic.
CREATE TABLE IF NOT EXISTS phrase_examples(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  src TEXT NOT NULL,
  dst TEXT NOT NULL,
  heldout INTEGER NOT NULL DEFAULT 0,
  evidence TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS phrase_rules(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL DEFAULT 'literal',
  src_tmpl TEXT NOT NULL DEFAULT '{}',
  dst_tmpl TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'candidate',
  evidence TEXT NOT NULL DEFAULT '[]',
  report TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
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


class Memory:
    """Thin SQLite wrapper. All writes commit immediately."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # check_same_thread=False + a lock: the server serves requests on
        # worker threads sharing this connection. All access goes through
        # execute()/query() below, both locked.
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
        self.execute(
            "INSERT INTO rules(id,name,conditions,conclusion,exceptions,source,created_at)"
            " VALUES(?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET name=excluded.name, conditions=excluded.conditions,"
            " conclusion=excluded.conclusion, exceptions=excluded.exceptions,"
            " source=excluded.source",
            (
                rule["id"], rule.get("name", rule["id"]),
                json.dumps(rule.get("conditions", []), ensure_ascii=False),
                json.dumps(rule.get("conclusion", {}), ensure_ascii=False),
                json.dumps(rule.get("exceptions", []), ensure_ascii=False),
                rule.get("source", "seed"),
                rule.get("created_at", _utcnow()),
            ),
        )

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

    # ---- learnable rules (Phase 4D; structured specs, never prose) ----
    def save_rule(self, rule: dict) -> None:
        self.execute(
            "INSERT INTO learnable_rules(id,version,name,domain,spec,status,"
            "validation,calls,successes,failures,previous,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET version=excluded.version,"
            " name=excluded.name, domain=excluded.domain, spec=excluded.spec,"
            " status=excluded.status, validation=excluded.validation,"
            " calls=excluded.calls, successes=excluded.successes,"
            " failures=excluded.failures, previous=excluded.previous,"
            " updated_at=excluded.updated_at",
            (rule["id"], int(rule.get("version", 1)), rule.get("name", ""),
             rule.get("domain", ""),
             json.dumps(rule.get("spec", {}), ensure_ascii=False),
             rule.get("status", "candidate"),
             json.dumps(rule.get("validation", {}), ensure_ascii=False),
             int(rule.get("calls", 0)), int(rule.get("successes", 0)),
             int(rule.get("failures", 0)),
             json.dumps(rule.get("previous", ""), ensure_ascii=False),
             rule.get("created_at", _utcnow()), _utcnow()))

    def get_rule(self, rid: str) -> Optional[dict]:
        row = self.query_one("SELECT * FROM learnable_rules WHERE id=?",
                             (rid,))
        return self._decode_rule(row) if row else None

    def list_rules_learned(self, status: Optional[str] = None,
                           domain: Optional[str] = None) -> list[dict]:
        sql, params = "SELECT * FROM learnable_rules WHERE 1=1", []
        if status:
            sql += " AND status=?"
            params.append(status)
        if domain:
            sql += " AND domain=?"
            params.append(domain)
        sql += " ORDER BY id"
        return [self._decode_rule(r) for r in self.query(sql, tuple(params))]

    @staticmethod
    def _decode_rule(row: dict) -> dict:
        row = dict(row)
        for k in ("spec", "validation", "previous"):
            try:
                row[k] = json.loads(row[k]) if isinstance(row[k], str) \
                    else row[k]
            except Exception:
                row[k] = {} if k != "previous" else ""
        return row

    # ---- analogy feedback (Phase 5B) ----
    def add_feedback(self, task_id: str, source_task_id: str,
                     correction: dict) -> int:
        cur = self.execute(
            "INSERT INTO analogy_feedback(task_id,source_task_id,correction,"
            "created_at) VALUES(?,?,?,?)",
            (task_id, source_task_id,
             json.dumps(correction, ensure_ascii=False), _utcnow()))
        return int(cur.lastrowid)

    def list_feedback(self) -> list[dict]:
        return self.query("SELECT * FROM analogy_feedback ORDER BY id")

    def feedback_adjustments(self) -> dict:
        """User corrections as candidate score adjustments.

        A correction {candidate_task_id, effect: boost|penalize} shifts
        future retrieval transparently (reported per candidate).
        """
        adj: dict[str, float] = {}
        for row in self.list_feedback():
            try:
                corr = json.loads(row["correction"]) \
                    if isinstance(row["correction"], str) \
                    else row["correction"]
            except Exception:
                continue
            cid = corr.get("candidate_task_id") or row["source_task_id"]
            eff = corr.get("effect", "")
            if eff == "boost":
                adj[cid] = adj.get(cid, 0.0) + 0.2
            elif eff == "penalize":
                adj[cid] = adj.get(cid, 0.0) - 0.3
        return adj

    # ---- learned lexicon (Phase 6A): word -> canonical key ----
    def learn_mapping(self, word: str, key: str, evidence: str = "") -> None:
        word, key = word.strip(), key.strip()
        if not word or not key:
            raise ValueError("word and key required")
        row = self.query_one("SELECT * FROM lexicon WHERE word=?", (word,))
        ev = []
        if row:
            try:
                ev = json.loads(row["evidence"])
            except Exception:
                ev = []
        if evidence and evidence not in ev:
            ev.append(evidence)
        self.execute(
            "INSERT INTO lexicon(word,key,evidence,created_at)"
            " VALUES(?,?,?,?)"
            " ON CONFLICT(word) DO UPDATE SET key=excluded.key,"
            " evidence=excluded.evidence",
            (word, key, json.dumps(ev, ensure_ascii=False), _utcnow()))

    def lookup_mapping(self, word: str) -> Optional[str]:
        row = self.query_one("SELECT key FROM lexicon WHERE word=?",
                             (word.strip(),))
        return row["key"] if row else None

    def list_mappings(self) -> list[dict]:
        return self.query("SELECT * FROM lexicon ORDER BY word")

    # ---- learned phrase rules (Phase 6C): expression -> canonical form ----
    def add_phrase_example(self, src: str, dst: str, heldout: bool = False,
                           evidence: str = "") -> int:
        src, dst = src.strip(), dst.strip()
        if not src or not dst:
            raise ValueError("src and dst required")
        if src == dst:
            raise ValueError("src and dst are identical; nothing to learn")
        cur = self.execute(
            "INSERT INTO phrase_examples(src,dst,heldout,evidence,"
            "created_at) VALUES(?,?,?,?,?)",
            (src, dst, 1 if heldout else 0,
             json.dumps([evidence] if evidence else [], ensure_ascii=False),
             _utcnow()))
        return cur.lastrowid

    def list_phrase_examples(self, heldout=None) -> list[dict]:
        if heldout is None:
            return self.query("SELECT * FROM phrase_examples ORDER BY id")
        return self.query("SELECT * FROM phrase_examples WHERE heldout=? "
                          "ORDER BY id", (1 if heldout else 0,))

    def save_phrase_rule(self, rule: dict) -> None:
        now = _utcnow()
        self.execute(
            "INSERT INTO phrase_rules(id,kind,src_tmpl,dst_tmpl,status,"
            "evidence,report,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET kind=excluded.kind,"
            " src_tmpl=excluded.src_tmpl, dst_tmpl=excluded.dst_tmpl,"
            " status=excluded.status, evidence=excluded.evidence,"
            " report=excluded.report, updated_at=excluded.updated_at",
            (rule["id"], rule.get("kind", "literal"),
             json.dumps(rule.get("src_tmpl"), ensure_ascii=False),
             json.dumps(rule.get("dst_tmpl"), ensure_ascii=False),
             rule.get("status", "candidate"),
             json.dumps(rule.get("evidence", []), ensure_ascii=False),
             json.dumps(rule.get("report", {}), ensure_ascii=False), now,
             now))

    def get_phrase_rule(self, rid: str) -> dict | None:
        row = self.query_one("SELECT * FROM phrase_rules WHERE id=?", (rid,))
        if not row:
            return None
        return self._decode_phrase_rule(row)

    def list_phrase_rules(self, status: str | None = None) -> list[dict]:
        if status:
            rows = self.query("SELECT * FROM phrase_rules WHERE status=? "
                              "ORDER BY id", (status,))
        else:
            rows = self.query("SELECT * FROM phrase_rules ORDER BY id")
        return [self._decode_phrase_rule(r) for r in rows]

    @staticmethod
    def _decode_phrase_rule(row: dict) -> dict:
        out = dict(row)
        for k in ("src_tmpl", "dst_tmpl", "evidence", "report"):
            try:
                out[k] = json.loads(out[k]) if isinstance(out[k], str) \
                    else out[k]
            except Exception:
                pass
        # the dst-frame signature lives inside the report (no extra
        # column): promote it so validation/application see one shape
        # whether the rule is fresh from induce() or loaded from disk.
        if "signature" not in out and isinstance(out.get("report"), dict) \
                and "signature" in out["report"]:
            out["signature"] = out["report"]["signature"]
        return out

    # ---- dialogue context (per session; never shared across sessions) ----
    def get_context(self, session: str) -> dict:
        row = self.query_one("SELECT * FROM dialogue_context WHERE session=?",
                             (session,))
        if not row:
            return {"entities": [], "last_observation": None, "turn": 0}
        try:
            st = json.loads(row["state"])
            st.setdefault("entities", [])
            st.setdefault("turn", 0)
            return st
        except Exception:
            return {"entities": [], "last_observation": None, "turn": 0}

    def set_context(self, session: str, state: dict) -> None:
        self.execute(
            "INSERT INTO dialogue_context(session,state,updated_at)"
            " VALUES(?,?,?)"
            " ON CONFLICT(session) DO UPDATE SET state=excluded.state,"
            " updated_at=excluded.updated_at",
            (session, json.dumps(state, ensure_ascii=False), _utcnow()))

    def clear_context(self, session: str) -> None:
        self.execute("DELETE FROM dialogue_context WHERE session=?",
                     (session,))

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
