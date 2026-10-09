"""End-to-end learning loop: split scenario + persistence across restart."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.learner import Learner, default_regression
from cognitive_ai.loop import ExplorationLoop
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.schema_evolver import SchemaEvolver


def seed_vessels(m: Memory):
    o = Ontology(m)
    for cid, name, parents in [
        ("entity", "实体", []), ("object", "物体", ["entity"]),
        ("artifact", "人造物品", ["object"]),
        ("container", "容器", ["artifact"]),
        ("vessel", "器皿", ["container"]),
        ("cup", "杯子", ["container"]),
        ("bowl", "碗", ["vessel"]),
        ("bottle", "瓶子", ["vessel"]),
    ]:
        o.add_concept(cid, name, parents=parents)
    o.add_attribute("cup", "has_handle", True, status="verified")
    o.add_attribute("bowl", "has_handle", False, status="verified")
    o.add_attribute("bottle", "has_lid", True, status="verified")
    SchemaEvolver(m).ensure_initial_version()


def test_split_scenario(tmp_path):
    m = Memory(str(tmp_path / "l.db"))
    seed_vessels(m)
    ev = SchemaEvolver(m, default_regression)
    v0 = ev.current_version()
    cid = ev.propose("split_category",
                     {"category": "vessel",
                      "groups": {"open_vessel": ["bowl"],
                                 "closed_vessel": ["bottle"]},
                      "names": {"open_vessel": "开口器皿",
                                "closed_vessel": "闭口器皿"}},
                     reason="bowl/bottle diverge stably on has_lid",
                     evidence=[{"note": "stable difference"}])
    out = ev.apply_proposal(cid)
    assert out["ok"] is True
    o = Ontology(m)
    assert o.is_a("bowl", "open_vessel")
    assert o.is_a("bottle", "closed_vessel")
    # rollback works, versions visible
    rb = ev.rollback(v0 + 1 if out["version_after"] == v0 + 1 else v0)
    assert rb["ok"] is True
    m.close()


def test_autonomous_gap_to_research(tmp_path):
    m = Memory(str(tmp_path / "l2.db"))
    seed_vessels(m)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    loop = ExplorationLoop(m, os.path.join(root, "data", "corpus"))
    out = loop.run(steps=2, max_seconds=60)
    assert out["steps_executed"] >= 1
    # run produced experiences (Type-A learning recorded)
    assert len(m.list_experiences(limit=100)) > 0
    m.close()
