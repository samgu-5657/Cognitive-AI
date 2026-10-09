"""Schema evolution: validated apply, rejection, rollback, no silent overwrite."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.learner import default_regression
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.schema_evolver import SchemaEvolver


def setup(tmp_path):
    m = Memory(str(tmp_path / "s.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    SchemaEvolver(m).ensure_initial_version()
    return m


def test_apply_and_history(tmp_path):
    m = setup(tmp_path)
    ev = SchemaEvolver(m, default_regression)
    v0 = ev.current_version()
    r = ev.direct_apply("add_attribute",
                        {"concept": "cup", "key": "material", "value": "ceramic"},
                        reason="test")
    assert r["ok"] and r["version_after"] == v0 + 1
    assert len(ev.history()) >= 1
    m.close()


def test_failed_change_rejected_and_rolled_back(tmp_path):
    m = setup(tmp_path)
    ev = SchemaEvolver(m, default_regression)
    v0 = ev.current_version()
    # illegal: parent does not exist
    r = ev.direct_apply("add_concept",
                        {"id": "x", "name": "x", "parents": ["ghost"]},
                        reason="bad")
    assert r["ok"] is False
    assert ev.current_version() == v0
    assert m.get_concept("x") is None
    m.close()


def test_regression_failure_rejects_major_change(tmp_path):
    m = setup(tmp_path)
    # regression hook that always fails
    ev = SchemaEvolver(m, lambda mem: (False, "forced fail"))
    cid = ev.propose("split_category",
                     {"category": "container",
                      "groups": {"g1": ["cup"]}}, reason="forced")
    out = ev.apply_proposal(cid)
    assert out["ok"] is False and "regression" in out["error"]
    m.close()


def test_rollback_restores(tmp_path):
    m = setup(tmp_path)
    ev = SchemaEvolver(m, default_regression)
    v0 = ev.current_version()
    ev.direct_apply("add_concept",
                    {"id": "plate", "name": "盘子", "parents": ["entity"]},
                    reason="add plate")
    assert m.get_concept("plate") is not None
    out = ev.rollback(v0)
    assert out["ok"] is True
    assert m.get_concept("plate") is None
    m.close()


def test_conflict_does_not_overwrite_verified(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from cognitive_ai.learner import Learner
    import tempfile
    m = setup(tmp_path)
    Ontology(m).add_attribute("cup", "material", "ceramic", status="verified")
    with tempfile.TemporaryDirectory() as td:
        learner = Learner(m, td)  # empty corpus
        out = learner.learn_from_observation(
            {"name": "杯子", "attributes": {"material": "glass"}})
    # verified value must survive; conflict recorded instead
    vals = [a["value"] for a in m.get_attributes("cup") if a["key"] == "material"]
    assert "ceramic" in vals
    assert any(c["type"] == "attribute_conflict"
               for c in out["conflicts"])
    m.close()
