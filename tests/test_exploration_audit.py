"""Audit regression tests for exploration + continual learning.

All tests use isolated temp DBs / temp corpus dirs and never touch the
user's real database.

Rules under test:
  R1 inconclusive must not add/overwrite semantic knowledge.
  R2 no supporting evidence -> no learned attribute, no schema version, no change record.
  R3 log question ID / target / key must match the applied modification.
  R4 evidence must mention the target concept; single weak source is not enough.
  R5 no unbounded duplicate questions; cooling questions are not re-studied.
  R6 failed/unsafe modifications are rejected or rolled back.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.curiosity import Curiosity
from cognitive_ai.learner import Learner
from cognitive_ai.loop import ExplorationLoop
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.schema_evolver import SchemaEvolver


def seed_minimal(mem: Memory) -> None:
    o = Ontology(mem)
    o.add_concept("entity", "entity")
    o.add_concept("object", "object", parents=["entity"])
    o.add_concept("artifact", "artifact", parents=["object"])
    o.add_concept("container", "container", parents=["artifact"])
    o.add_concept("cup", "cup", parents=["container"])
    o.add_attribute("cup", "material", "ceramic", confidence=0.9,
                    source="seed", status="verified")
    SchemaEvolver(mem).ensure_initial_version()


def snapshot_state(mem: Memory) -> dict:
    return {
        "version": mem.current_version(),
        "attrs": sorted((a["concept_id"], a["key"], a["value"])
                        for c in mem.list_concepts()
                        for a in mem.get_attributes(c["id"])),
        "relations": len(mem.query("SELECT * FROM relations")),
        "changes": len(mem.list_changes(limit=10000)),
    }


def write_corpus(tmp_path, files: dict) -> str:
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    for name, text in files.items():
        (d / name).write_text(text, encoding="utf-8")
    return str(d)


# R1/R2: no evidence -> inconclusive, zero structural side effects
def test_no_evidence_means_no_structural_change(tmp_path):
    mem = Memory(str(tmp_path / "a.db"))
    seed_minimal(mem)
    corpus = write_corpus(tmp_path, {})  # empty corpus
    before = snapshot_state(mem)
    learner = Learner(mem, corpus)
    Curiosity(mem).generate()
    open_qs = [q for q in mem.list_questions()
               if q["status"] in ("proposed", "queued")]
    assert open_qs, "expected at least one gap question"
    res = learner.research_question(int(open_qs[0]["id"]))
    assert res["result"] == "inconclusive"
    assert res.get("applied") is None
    after = snapshot_state(mem)
    assert after == before, f"structural drift without evidence: {before} -> {after}"
    mem.close()


# R3: log binding must match the applied modification exactly
def test_log_binding_matches_applied_change(tmp_path):
    mem = Memory(str(tmp_path / "b.db"))
    seed_minimal(mem)
    corpus = write_corpus(tmp_path, {
        "m1.txt": "The cup is discussed here. material: clay. cup location: shelf.",
        "m2.txt": "More notes about cup. material: clay. cup is a container.",
    })
    loop = ExplorationLoop(mem, corpus)
    out = loop.run(steps=1, max_seconds=60)
    assert out["steps_executed"] == 1
    entry = out["log"][0]
    for field in ("question_id", "target_id", "key", "question",
                  "result", "evidence_sources", "action"):
        assert field in entry, f"log entry missing binding field: {field}"
    if entry["result"] == "learned":
        assert entry["action"] == "add_attribute"
        assert entry["applied_change_id"] is not None
        ch = mem.query_one("SELECT * FROM schema_changes WHERE id=?",
                           (entry["applied_change_id"],))
        assert ch is not None
        payload = json.loads(ch["payload"])
        assert payload["concept"] == entry["target_id"]
        assert payload["key"] == entry["key"]
        assert f"Q{entry['question_id']}" in ch["reason"]
        assert entry["target_id"] in ch["reason"] or \
            entry["target_id"] in entry["question"]
    else:
        assert entry["applied_change_id"] is None
    mem.close()


# R1/R2 continued: verified knowledge untouched without valid evidence
def test_no_verified_pollution_without_evidence(tmp_path):
    mem = Memory(str(tmp_path / "c.db"))
    seed_minimal(mem)
    corpus = write_corpus(tmp_path, {
        # mentions a DIFFERENT concept only; must not leak into cup
        "other.txt": "bowl notes. material: glass. bowl location: table.",
    })
    before = snapshot_state(mem)
    learner = Learner(mem, corpus)
    Curiosity(mem).generate()
    for q in [x for x in mem.list_questions()
              if x["status"] in ("proposed", "queued")]:
        learner.research_question(int(q["id"]))
    cup_vals = [a["value"] for a in mem.get_attributes("cup")
                if a["key"] == "material"]
    assert cup_vals == ["ceramic"], f"verified value polluted: {cup_vals}"
    assert not [a for a in mem.get_attributes("cup")
                if a["status"] == "verified" and a["value"] != "ceramic"]
    assert snapshot_state(mem)["version"] == before["version"]
    mem.close()


# R4: supporting evidence -> candidate recorded, validated, applied with sources
def test_supported_evidence_applies_through_validation(tmp_path):
    mem = Memory(str(tmp_path / "d.db"))
    seed_minimal(mem)
    corpus = write_corpus(tmp_path, {
        "e1.txt": "Field notes on the artifact storeroom. location: pantry. artifact kept here.",
        "e2.txt": "Second survey of artifact holdings. location: pantry. artifact catalog.",
    })
    learner = Learner(mem, corpus)
    Curiosity(mem).generate()
    q = mem.query_one(
        "SELECT * FROM questions WHERE target_id='artifact' AND text LIKE '%location%'")
    assert q is not None
    v0 = mem.current_version()
    res = learner.research_question(int(q["id"]))
    assert res["result"] == "learned", f"expected learned, got {res}"
    assert res["applied_change_id"] is not None
    rows = [a for a in mem.get_attributes("artifact") if a["key"] == "location"]
    assert rows and rows[0]["value"] == "pantry"
    assert rows[0]["status"] == "hypothesis"  # never auto-verified
    assert mem.current_version() == v0 + 1
    ch = mem.query_one("SELECT * FROM schema_changes WHERE id=?",
                       (res["applied_change_id"],))
    ev = json.loads(ch["evidence"])
    assert len(ev) >= 2, "single source must not suffice"
    mem.close()


# R4b: single source alone must NOT promote to knowledge
def test_single_source_insufficient(tmp_path):
    mem = Memory(str(tmp_path / "d2.db"))
    seed_minimal(mem)
    corpus = write_corpus(tmp_path, {
        "only.txt": "Notes on artifact. location: pantry. artifact room.",
    })
    learner = Learner(mem, corpus)
    Curiosity(mem).generate()
    q = mem.query_one(
        "SELECT * FROM questions WHERE target_id='artifact' AND text LIKE '%location%'")
    v0 = mem.current_version()
    res = learner.research_question(int(q["id"]))
    assert res["result"] == "inconclusive"
    assert res.get("applied") is None
    assert mem.current_version() == v0
    assert not [a for a in mem.get_attributes("artifact") if a["key"] == "location"]
    mem.close()


# R5: multi-step runs must not inflate the queue with duplicates
def test_no_duplicate_question_inflation(tmp_path):
    mem = Memory(str(tmp_path / "e.db"))
    seed_minimal(mem)
    corpus = write_corpus(tmp_path, {})
    loop = ExplorationLoop(mem, corpus)
    out = loop.run(steps=4, max_seconds=60)
    rows = mem.list_questions(limit=10000)
    keys = [(r["text"], r["target_id"] or "") for r in rows]
    assert len(keys) == len(set(keys)), "duplicate (text, target) questions generated"
    assert out["steps_executed"] <= 4
    for entry in out["log"]:
        assert "new_questions" in entry
    # second identical run must not grow the queue
    n0 = len(mem.list_questions(limit=10000))
    Curiosity(mem).generate()
    Curiosity(mem).generate()
    assert len(mem.list_questions(limit=10000)) == n0
    mem.close()


# R5b: cooling/inconclusive questions are not immediately re-studied
def test_cooling_questions_not_restudied(tmp_path):
    mem = Memory(str(tmp_path / "f.db"))
    seed_minimal(mem)
    corpus = write_corpus(tmp_path, {})
    loop = ExplorationLoop(mem, corpus)
    out1 = loop.run(steps=3, max_seconds=60)
    studied = [e.get("question_id") for e in out1["log"] if "question_id" in e]
    out2 = loop.run(steps=3, max_seconds=60)
    studied2 = [e.get("question_id") for e in out2["log"] if "question_id" in e]
    assert not (set(studied) & set(studied2)), \
        f"cooling questions re-studied immediately: {set(studied) & set(studied2)}"
    mem.close()


# R6: failed validation -> rejected, state intact
def test_failed_modification_rejected_and_rolled_back(tmp_path):
    mem = Memory(str(tmp_path / "g.db"))
    seed_minimal(mem)
    ev = SchemaEvolver(mem, lambda m: (False, "forced regression failure"))
    v0 = ev.current_version()
    cid = ev.propose("split_category",
                     {"category": "container", "groups": {"g1": ["cup"]}},
                     reason="audit check")
    out = ev.apply_proposal(cid)
    assert out["ok"] is False
    assert ev.current_version() == v0
    row = mem.query_one("SELECT * FROM schema_changes WHERE id=?", (cid,))
    assert row["status"] == "rejected"
    # explore must never auto-apply structural (major) modifications
    mem2 = Memory(str(tmp_path / "g2.db"))
    seed_minimal(mem2)
    corpus = write_corpus(tmp_path, {
        "s1.txt": "cup notes. material: clay. cup shelf.",
        "s2.txt": "more cup notes. material: clay. cup rack.",
    })
    ExplorationLoop(mem2, corpus).run(steps=3, max_seconds=60)
    kinds = {c["kind"] for c in mem2.list_changes(limit=10000)}
    assert kinds <= {"add_attribute", "add_concept", "add_relation", "update_rule",
                     "update_attribute_status"}, f"unexpected auto change kinds: {kinds}"
    assert "split_category" not in kinds and "merge_categories" not in kinds
    mem.close()
    mem2.close()
