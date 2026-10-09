"""Memory tests: restart persistence, retrieval."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.memory import Memory


def test_restart_keeps_knowledge(tmp_path):
    p = str(tmp_path / "m.db")
    m = Memory(p)
    m.upsert_concept({"id": "cup", "name": "杯子"})
    m.set_parents("cup", [])
    m.add_attribute("cup", "material", "ceramic", status="verified")
    m.add_experience("observation", "saw a cup", {"name": "cup"})
    v = m.current_version()
    m.save_version(v + 1, m.snapshot(), "test")
    m.close()

    m2 = Memory(p)
    assert m2.get_concept("cup")["name"] == "杯子"
    assert m2.get_attributes("cup")[0]["status"] == "verified"
    assert len(m2.list_experiences()) == 1
    assert m2.current_version() == v + 1
    m2.close()


def test_retrieval_by_concept_relation(tmp_path):
    m = Memory(str(tmp_path / "m2.db"))
    m.upsert_concept({"id": "a", "name": "a"})
    m.upsert_concept({"id": "b", "name": "b"})
    m.add_relation("a", "used_for", "b")
    assert len(m.get_relations(src_id="a")) == 1
    assert len(m.relations_involving("b")) == 1
    assert len(m.find_concepts_by_name("A")) == 1
    m.close()
