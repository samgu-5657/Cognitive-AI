"""Reasoner tests: traceability, unknown handling, conflicts, exceptions."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.reasoner import Reasoner


def setup(tmp_path):
    m = Memory(str(tmp_path / "r.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    o.add_attribute("container", "can_contain", True, status="verified")
    m.upsert_rule({
        "id": "r1", "name": "containers hold",
        "conditions": [{"type": "is_a", "concept": "cup", "ancestor": "container"}],
        "conclusion": {"type": "has_attr", "concept": "cup",
                       "key": "can_contain", "value": True},
        "exceptions": [{"type": "has_attr", "concept": "cup",
                        "key": "is_broken", "value": True}],
    })
    return m


def test_inheritance_trace(tmp_path):
    m = setup(tmp_path)
    r = Reasoner(m).get_attribute("cup", "can_contain")
    assert r["conclusion"] is True
    assert any("attribute_inheritance" in str(t) for t in r["trace"])
    m.close()


def test_unknown_not_guessed(tmp_path):
    m = setup(tmp_path)
    r = Reasoner(m).get_attribute("cup", "nonexistent_key")
    assert r["conclusion"] == "unknown"
    r2 = Reasoner(m).is_a("cup", "ghost")
    assert r2["conclusion"] == "unknown"
    fired = Reasoner(m).forward_chain()
    assert any(x["rule"] == "r1" and "conclusion" in x for x in fired)
    m.close()


def test_exception_blocks_rule(tmp_path):
    m = setup(tmp_path)
    Ontology(m).add_attribute("cup", "is_broken", True, status="verified")
    fired = Reasoner(m).forward_chain()
    hit = [x for x in fired if x["rule"] == "r1"][0]
    assert hit.get("skipped") is True
    m.close()


def test_conflict_detected(tmp_path):
    m = setup(tmp_path)
    Ontology(m).add_attribute("cup", "material", "ceramic", status="verified")
    Ontology(m).add_attribute("cup", "material", "glass", status="hypothesis")
    conflicts = Reasoner(m).check_conflicts()
    assert any(c["type"] == "attribute_conflict" and c["key"] == "material"
               for c in conflicts)
    m.close()
