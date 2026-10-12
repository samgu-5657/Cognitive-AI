"""Migration test: a database created before the evidence-fingerprint columns
and the lifecycle_events table existed must open read/write without losing data.

Builds that old layout in tmp instead of depending on a machine-specific backup.
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.memory import Memory

OLD_SCHEMA = """
CREATE TABLE concepts(
  id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'Entity',
  confidence REAL NOT NULL DEFAULT 1.0, source TEXT NOT NULL DEFAULT 'seed',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1);
CREATE TABLE attributes(
  id INTEGER PRIMARY KEY AUTOINCREMENT, concept_id TEXT NOT NULL,
  key TEXT NOT NULL, value TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.8,
  source TEXT NOT NULL DEFAULT 'observation',
  status TEXT NOT NULL DEFAULT 'hypothesis');
CREATE TABLE evidences(
  id INTEGER PRIMARY KEY AUTOINCREMENT, claim_id INTEGER,
  content TEXT NOT NULL, source TEXT NOT NULL,
  credibility REAL NOT NULL DEFAULT 0.5, created_at TEXT NOT NULL);
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _build_old_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.execute("INSERT INTO concepts(id,name,created_at,updated_at) "
                 "VALUES('cup','杯子','t0','t0')")
    conn.execute("INSERT INTO attributes(concept_id,key,value,status) "
                 "VALUES('cup','material',?, 'verified')",
                 (json.dumps("ceramic"),))
    conn.execute("INSERT INTO evidences(content,source,created_at) "
                 "VALUES('cup material: ceramic','local:cup.txt','t0')")
    conn.execute("INSERT INTO meta(key,value) VALUES('current_version','3')")
    conn.commit()
    conn.close()


def test_old_database_migrates_compatibly(tmp_path):
    path = str(tmp_path / "old.db")
    _build_old_db(path)
    mem = Memory(path)  # must not raise; migration is additive
    try:
        assert [c["id"] for c in mem.list_concepts()] == ["cup"]
        mats = [a for a in mem.get_attributes("cup") if a["key"] == "material"]
        assert mats[0]["value"] == "ceramic"
        assert mats[0]["status"] == "verified"  # old knowledge untouched
        assert mem.current_version() == 3
        cols = {r["name"] for r in mem.query("PRAGMA table_info(evidences)")}
        assert {"content_hash", "relevance", "origin_note"} <= cols
        old_ev = mem.list_evidences()[0]
        assert old_ev["content"] == "cup material: ceramic"
        assert old_ev["content_hash"] == ""  # new column defaults
        eid = mem.record_lifecycle("attribute", "cup.material", "verified",
                                   "contested", "migration check", [])
        assert eid and mem.list_lifecycle("cup.material")
    finally:
        mem.close()
    # reopening an already-migrated DB is a no-op
    Memory(path).close()
