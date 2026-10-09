"""Curiosity + research + learning loop tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.curiosity import Curiosity
from cognitive_ai.loop import ExplorationLoop
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.research import LocalCorpusProvider
from cognitive_ai.schema_evolver import SchemaEvolver


def seed(tmp_path, corpus_dir=None):
    m = Memory(str(tmp_path / "c.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    # deliberately missing 'location' and 'function' -> gaps expected
    SchemaEvolver(m).ensure_initial_version()
    return m


def test_curiosity_finds_gap(tmp_path):
    m = seed(tmp_path)
    qs = Curiosity(m).generate()
    assert len(qs) > 0
    assert any("cup" in q["text"] for q in qs)
    assert all(q["score"] is not None for q in qs)
    m.close()


def test_local_research_finds_seed_corpus(tmp_path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    p = LocalCorpusProvider(os.path.join(root, "data", "corpus"))
    hits = p.search("杯子 material ceramic", limit=3)
    assert len(hits) > 0
    assert hits[0]["credibility"] > 0
    assert "excerpt" in hits[0]


def test_explore_budget_stops(tmp_path):
    m = seed(tmp_path)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    loop = ExplorationLoop(m, os.path.join(root, "data", "corpus"))
    out = loop.run(steps=3, max_seconds=60)
    assert out["steps_executed"] <= 3
    m.close()


def test_explore_invalid_budget_rejected(tmp_path):
    m = seed(tmp_path)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    loop = ExplorationLoop(m, os.path.join(root, "data", "corpus"))
    try:
        loop.run(steps=0)
        assert False, "should have raised"
    except ValueError:
        pass
    try:
        loop.run(steps=1000)
        assert False, "should have raised"
    except ValueError:
        pass
    m.close()


def test_novelty_novel_item(tmp_path):
    from cognitive_ai.learner import Learner
    import tempfile
    m = seed(tmp_path)
    with tempfile.TemporaryDirectory() as td:
        out = Learner(m, td).learn_from_observation(
            {"name": "新茶壶", "parents": ["container"],
             "attributes": {"material": "clay"}})
    assert "new_concept" in out
    m.close()
