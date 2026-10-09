"""Ontology tests: CRUD, multiple inheritance, cycle detection, persistence."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology, OntologyError


def make_mem(tmp_path):
    return Memory(str(tmp_path / "t.db"))


def seed_basic(mem):
    o = Ontology(mem)
    o.add_concept("entity", "实体")
    o.add_concept("object", "物体", parents=["entity"])
    o.add_concept("artifact", "人造物品", parents=["object"])
    o.add_concept("container", "容器", parents=["artifact"])
    o.add_concept("drinkware", "饮用器具", parents=["artifact"])
    o.add_concept("cup", "杯子",
                  parents=["container", "drinkware", "object", "artifact", "entity"])
    return o


def test_create_query_persist(tmp_path):
    mem = make_mem(tmp_path)
    o = seed_basic(mem)
    d = o.describe("cup")
    assert set(["container", "drinkware"]).issubset(set(d["parents"]))
    assert "entity" in d["ancestors"]
    mem.close()
    # reopen: persistence
    mem2 = Memory(str(tmp_path / "t.db"))
    o2 = Ontology(mem2)
    assert o2.is_a("cup", "container")
    assert o2.is_a("cup", "drinkware")
    mem2.close()


def test_multiple_inheritance(tmp_path):
    mem = make_mem(tmp_path)
    o = seed_basic(mem)
    assert o.is_a("cup", "container") and o.is_a("cup", "drinkware")
    # attribute inheritance
    o.add_attribute("container", "can_contain", True, status="verified")
    assert o.inherited_attributes("cup")["can_contain"]["value"] is True


def test_cycle_rejected(tmp_path):
    mem = make_mem(tmp_path)
    o = seed_basic(mem)
    with pytest.raises(OntologyError):
        o.add_parent("entity", "cup")  # would create cycle
    with pytest.raises(OntologyError):
        o.add_parent("cup", "cup")
    # still acyclic
    assert "cup" not in o.ancestors("entity")
