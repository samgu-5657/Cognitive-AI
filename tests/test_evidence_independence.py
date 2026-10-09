"""Evidence independence tests: fingerprints, grading, duplicate collapse."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.curiosity import Curiosity
from cognitive_ai.learner import Learner
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.research import (content_hash, grade_relevance,
                                   normalize_text, text_similarity)
from cognitive_ai.schema_evolver import SchemaEvolver


def test_content_hash_stable_and_order_insensitive_to_case():
    assert content_hash("  Cup Notes.  ") == content_hash("cup notes.")
    assert content_hash("abc") != content_hash("abd")


def test_similarity_identical_and_unrelated():
    assert text_similarity("the cup is ceramic", "the cup is ceramic") == 1.0
    assert text_similarity("the cup is ceramic", "quantum field theory") < 0.3


def test_grade_relevance_levels():
    assert grade_relevance("the cup is here. material: ceramic",
                           ("cup",), "material") == "direct"
    assert grade_relevance("the cup is here, nothing else",
                           ("cup",), "material") == "indirect"
    assert grade_relevance("material: ceramic. a bowl note.",
                           ("cup",), "material") == "unrelated"


def test_provider_hit_carries_hash(tmp_path):
    from cognitive_ai.research import LocalCorpusProvider
    d = tmp_path / "c"
    d.mkdir()
    (d / "a.txt").write_text("cup notes. material: clay. the cup.", encoding="utf-8")
    hits = LocalCorpusProvider(str(d)).search("cup material", limit=5)
    assert hits and hits[0]["content_hash"]


def seed(tmp_path, name):
    mem = Memory(str(tmp_path / name))
    o = Ontology(mem)
    o.add_concept("entity", "e")
    o.add_concept("cup", "cup", parents=["entity"])
    from cognitive_ai.schema_evolver import SchemaEvolver as SE
    SE(mem).ensure_initial_version()
    return mem


def write(tmp_path, name, text):
    d = tmp_path / "corp"
    d.mkdir(exist_ok=True)
    (d / name).write_text(text, encoding="utf-8")
    return str(d)


def ask(mem, corp, target="cup", key="location"):
    Curiosity(mem).generate()
    q = mem.query_one(
        "SELECT * FROM questions WHERE target_id=? AND text LIKE ?",
        (target, f"%'{key}'%"))
    return Learner(mem, corp).research_question(int(q["id"]))


def test_identical_copies_collapse_to_one_origin(tmp_path):
    mem = seed(tmp_path, "e1.db")
    try:
        body = "cup field notes. location: shelf. the cup stays here. more cup words."
        corp = write(tmp_path, "a.txt", body)
        write(tmp_path, "b.txt", body)  # byte-identical copy
        v0 = mem.current_version()
        res = ask(mem, corp)
        assert res["result"] == "inconclusive", res
        assert mem.current_version() == v0
        assert "location" not in Ontology(mem).inherited_attributes("cup")
    finally:
        mem.close()


def test_reworded_near_duplicate_collapses(tmp_path):
    mem = seed(tmp_path, "e2.db")
    try:
        a = "cup field notes here. location: shelf. the cup stays here for winter. more cup words about storage."
        b = "cup field notes here. location: shelf. the cup stays here for summer. more cup words about storage."
        assert text_similarity(a, b) >= 0.85
        corp = write(tmp_path, "a.txt", a)
        write(tmp_path, "b.txt", b)
        v0 = mem.current_version()
        res = ask(mem, corp)
        assert res["result"] == "inconclusive", res
        assert mem.current_version() == v0
    finally:
        mem.close()


def test_unrelated_hits_do_not_promote(tmp_path):
    mem = seed(tmp_path, "e3.db")
    try:
        corp = write(tmp_path, "a.txt",
                     "bowl notes. material: glass. the bowl location: table.")
        write(tmp_path, "b.txt",
              "bowl notes two. material: glass. bowl location: table.")
        res = ask(mem, corp, target="cup", key="location")
        assert res["result"] == "inconclusive", res
        assert "location" not in Ontology(mem).inherited_attributes("cup")
    finally:
        mem.close()
