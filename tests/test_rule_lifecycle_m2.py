"""Phase M2 rule lifecycle + premise credibility tests (isolated DBs).

Every test reads real DB rows, traces and behaviors: candidate rules
stay silent, validated-but-inactive rules stay silent, disabled rules
stop firing, relation rows keep their stored status, conflicts are
reported, migrations preserve old data, and restarts behave identically.
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.memory import Memory
from cognitive_ai.models import RuleStatus
from cognitive_ai.ontology import Ontology
from cognitive_ai.reasoner import Reasoner


def fresh(tmp_path, tag="m2"):
    m = Memory(str(tmp_path / f"{tag}.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("vessel", "容器", parents=["entity"])
    o.add_concept("potA", "铁锅", parents=["vessel"])
    o.add_concept("potB", "铁壶", parents=["vessel"])
    m.upsert_rule({
        "id": "r_builtin", "conditions": [{"type": "is_a",
                                           "concept": "?x",
                                           "ancestor": "vessel"}],
        "conclusion": {"type": "has_attr", "concept": "?x",
                       "key": "can_heat", "value": True},
        "exceptions": [], "source": "seed"})
    return m


def test_1_builtin_rules_fire(tmp_path):
    m = fresh(tmp_path)
    try:
        hits = [f for f in Reasoner(m).forward_chain()
                if f.get("rule_id") == "r_builtin" and
                f.get("bindings", {}).get("?x") == "potA" and
                isinstance(f.get("conclusion"), dict)]
        assert len(hits) == 1 and hits[0]["status"] == "verified", hits
        assert hits[0]["rule_status"] == "active"
        assert hits[0]["rule_source"] == "seed"
    finally:
        m.close()


def test_2_candidate_rule_stays_silent(tmp_path):
    m = fresh(tmp_path, tag="m2b")
    try:
        m.upsert_rule({
            "id": "r_cand", "conditions": [{"type": "is_a",
                                            "concept": "?x",
                                            "ancestor": "vessel"}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "fancy", "value": True},
            "exceptions": [], "source": "learning",
            "status": "candidate"})
        fired = Reasoner(m).forward_chain()
        assert not [f for f in fired if f.get("rule_id") == "r_cand" and
                    isinstance(f.get("conclusion"), dict)], fired
        skips = [f for f in fired if f.get("rule_id") == "r_cand" and
                 f.get("skipped")]
        assert skips and "candidate" in skips[0]["reason"], fired
    finally:
        m.close()


def test_3_validated_activated_rule_chains(tmp_path):
    m = fresh(tmp_path, tag="m2c")
    try:
        m.upsert_rule({
            "id": "r_learned", "conditions": [{"type": "has_attr",
                                               "concept": "?x",
                                               "key": "can_heat",
                                               "value": True}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "stove_ok", "value": True},
            "exceptions": [], "source": "learning",
            "status": "candidate"})
        assert not [f for f in Reasoner(m).forward_chain()
                    if f.get("rule_id") == "r_learned" and
                    isinstance(f.get("conclusion"), dict)]
        assert m.set_rule_status("r_learned", "validated",
                                 reason="held-out 2/2") == "candidate"
        assert not [f for f in Reasoner(m).forward_chain()
                    if f.get("rule_id") == "r_learned" and
                    isinstance(f.get("conclusion"), dict)]
        assert m.set_rule_status("r_learned", "active",
                                 reason="reviewer: alice") == "validated"
        hits = [f for f in Reasoner(m).forward_chain()
                if f.get("rule_id") == "r_learned" and
                f.get("bindings", {}).get("?x") == "potB" and
                isinstance(f.get("conclusion"), dict)]
        assert len(hits) == 1 and hits[0]["status"] == "verified", hits
        evs = m.list_lifecycle("r_learned")
        assert [(e["from_state"], e["to_state"]) for e in evs] == [
            ("candidate", "validated"), ("validated", "active")], evs
        assert any("alice" in (e.get("reason") or "") for e in evs)
    finally:
        m.close()


def test_4_disabled_rejected_rules_stop(tmp_path):
    m = fresh(tmp_path, tag="m2d")
    try:
        assert any(f.get("rule_id") == "r_builtin" and
                   isinstance(f.get("conclusion"), dict)
                   for f in Reasoner(m).forward_chain())
        m.set_rule_status("r_builtin", "disabled", reason="under review")
        fired = Reasoner(m).forward_chain()
        assert not [f for f in fired if f.get("rule_id") == "r_builtin" and
                    isinstance(f.get("conclusion"), dict)], fired
        assert any(f.get("rule_id") == "r_builtin" and f.get("skipped")
                   for f in fired)
        m.set_rule_status("r_builtin", "rejected", reason="wrong")
        fired = Reasoner(m).forward_chain()
        assert not [f for f in fired if f.get("rule_id") == "r_builtin" and
                    isinstance(f.get("conclusion"), dict)], fired
    finally:
        m.close()


def test_5_verified_facts_chain(tmp_path):
    m = fresh(tmp_path, tag="m2e")
    try:
        o = Ontology(m)
        o.add_attribute("potA", "seasoned", True, status="verified")
        m.upsert_rule({
            "id": "r_sea", "conditions": [{"type": "has_attr",
                                           "concept": "?x", "key": "seasoned",
                                           "value": True}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "ready", "value": True},
            "exceptions": []})
        hits = [f for f in Reasoner(m).forward_chain()
                if f.get("rule_id") == "r_sea" and
                isinstance(f.get("conclusion"), dict)]
        assert len(hits) == 1 and hits[0]["status"] == "verified", hits
    finally:
        m.close()


def test_6_hypothesis_relation_no_upgrade(tmp_path):
    m = fresh(tmp_path, tag="m2f")
    try:
        o = Ontology(m)
        o.add_concept("chefA", "厨师", parents=["entity"])
        o.add_relation("chefA", "owns", "potA")  # default hypothesis
        m.upsert_rule({
            "id": "r_own", "conditions": [{"type": "relation",
                                           "src": "?p", "rel_type": "owns",
                                           "dst": "?x"}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "cared_for", "value": True},
            "exceptions": []})
        hits = [f for f in Reasoner(m).forward_chain()
                if f.get("rule_id") == "r_own" and
                isinstance(f.get("conclusion"), dict)]
        assert len(hits) == 1, hits
        assert hits[0]["status"] == "hypothesis", hits
        assert "hypothesis" in str(hits[0]["trace"]), hits
    finally:
        m.close()


def test_7_contested_relation_not_merged(tmp_path):
    m = fresh(tmp_path, tag="m2g")
    try:
        o = Ontology(m)
        o.add_concept("chefA", "厨师", parents=["entity"])
        mem_add = m.execute(
            "INSERT INTO relations(src_id,rel_type,dst_id,confidence,source,status)"
            " VALUES(?,?,?,?,?,?)",
            ("chefA", "owns", "potA", 0.8, "observation", "contested"))
        assert mem_add.lastrowid
        m.upsert_rule({
            "id": "r_own2", "conditions": [{"type": "relation",
                                            "src": "?p", "rel_type": "owns",
                                            "dst": "?x"}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "cared_for", "value": True},
            "exceptions": []})
        fired = Reasoner(m).forward_chain()
        assert not [f for f in fired if f.get("rule_id") == "r_own2" and
                    isinstance(f.get("conclusion"), dict)], fired
        assert any(f.get("conclusion") == "contested" for f in fired), fired
    finally:
        m.close()


def test_8_layers_keep_source_and_status(tmp_path):
    m = fresh(tmp_path, tag="m2h")
    try:
        o = Ontology(m)
        o.add_attribute("vessel", "thick", True, status="verified")
        m.upsert_rule({
            "id": "r_thick", "conditions": [{"type": "has_attr",
                                             "concept": "?x", "key": "thick",
                                             "value": True}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "sturdy", "value": True},
            "exceptions": [], "source": "learning"})
        hits = [f for f in Reasoner(m).forward_chain()
                if f.get("rule_id") == "r_thick" and
                f.get("bindings", {}).get("?x") == "potA" and
                isinstance(f.get("conclusion"), dict)]
        assert len(hits) == 1, hits
        assert hits[0]["rule_source"] == "learning"
        prem = hits[0]["premises"][0]
        assert "vessel.thick" in str(prem.get("basis", [])), prem
    finally:
        m.close()


def test_9_migration_keeps_old_knowledge(tmp_path):
    db = str(tmp_path / "old.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE concepts(id TEXT PRIMARY KEY, name TEXT)")
    con.execute("CREATE TABLE concept_parents(concept_id TEXT, parent_id TEXT)")
    con.execute("CREATE TABLE attributes(id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "concept_id TEXT, key TEXT, value TEXT, confidence REAL,"
                "source TEXT, status TEXT)")
    con.execute("CREATE TABLE rules(id TEXT PRIMARY KEY, name TEXT,"
                "conditions TEXT, conclusion TEXT, exceptions TEXT,"
                "source TEXT, created_at TEXT)")
    con.execute("INSERT INTO concepts VALUES('potA','铁锅')")
    con.execute("INSERT INTO attributes(concept_id,key,value,confidence,source,status)"
                " VALUES('potA','seasoned','true',0.9,'seed','verified')")
    con.execute("INSERT INTO rules VALUES('r_old','old',"
                "'[{\"type\": \"has_attr\", \"concept\": \"potA\","
                " \"key\": \"seasoned\", \"value\": true}]',"
                "'{\"type\": \"has_attr\", \"concept\": \"potA\","
                " \"key\": \"ready\", \"value\": true}',"
                "'[]','seed','2026-01-01')")
    con.commit()
    con.close()
    m = Memory(db)
    try:
        assert m.get_concept("potA") is not None
        rows = m.query("SELECT status FROM rules WHERE id='r_old'")
        assert rows and rows[0]["status"] == "active", rows
        attrs = m.get_attributes("potA")
        assert any(a["key"] == "seasoned" and a["status"] == "verified"
                   for a in attrs), attrs
        fired = Reasoner(m).forward_chain()
        assert any(f.get("rule_id") == "r_old" and
                   isinstance(f.get("conclusion"), dict) for f in fired)
    finally:
        m.close()


def test_10_restart_consistent(tmp_path):
    db = str(tmp_path / "m2j.db")
    m = Memory(db)
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("vessel", "容器", parents=["entity"])
    o.add_concept("potA", "铁锅", parents=["vessel"])
    m.upsert_rule({
        "id": "r_heat", "conditions": [{"type": "is_a", "concept": "?x",
                                        "ancestor": "vessel"}],
        "conclusion": {"type": "has_attr", "concept": "?x",
                       "key": "can_heat", "value": True}, "exceptions": []})
    m.set_rule_status("r_heat", "disabled", reason="test")
    m.close()
    m2 = Memory(db)
    try:
        row = m2.query_one("SELECT status FROM rules WHERE id=?",
                           ("r_heat",))
        assert row["status"] == "disabled"
        fired = Reasoner(m2).forward_chain()
        assert not [f for f in fired if f.get("rule_id") == "r_heat" and
                    isinstance(f.get("conclusion"), dict)]
        m2.set_rule_status("r_heat", "active", reason="test")
        fired = Reasoner(m2).forward_chain()
        assert any(f.get("rule_id") == "r_heat" and
                   isinstance(f.get("conclusion"), dict) for f in fired)
    finally:
        m2.close()


def test_11_verifier_sees_status_changes(tmp_path):
    m = fresh(tmp_path, tag="m2k")
    try:
        good = [f for f in Reasoner(m).forward_chain()
                if f.get("rule_id") == "r_builtin" and
                f.get("bindings", {}).get("?x") == "potA" and
                isinstance(f.get("conclusion"), dict)][0]
        assert Reasoner(m).verify_proof(good)["result"] == "verified"
        m.set_rule_status("r_builtin", "disabled", reason="test")
        bad = Reasoner(m).verify_proof(good)
        assert bad["result"] == "rejected", bad
        assert "disabled" in bad["detail"], bad
    finally:
        m.close()


def test_12_old_engine_behavior_intact(tmp_path):
    m = fresh(tmp_path, tag="m2l")
    try:
        fired = Reasoner(m).forward_chain()
        assert not any(f.get("limited") for f in fired)
        assert any(f.get("rule_id") == "r_builtin" for f in fired)
        assert Reasoner(m).check_conflicts() == []
    finally:
        m.close()
