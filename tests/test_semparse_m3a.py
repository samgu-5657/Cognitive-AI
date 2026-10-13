"""Phase M3A compositional parsing tests (isolated DBs).

Entities and values live only in tmp databases: replacing them needs
no new sentence regex, which is exactly what these tests prove.
Assertions check structured roles, never mere substring presence.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.memory import Memory
from cognitive_ai.novelty import parse_chinese_simple
from cognitive_ai.ontology import Ontology
from cognitive_ai.semparse import parse_statement


def fresh(tmp_path, tag="sp"):
    m = Memory(str(tmp_path / f"{tag}.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("cup", "杯子", parents=["entity"])
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("water", "水", parents=["entity"])
    return m


def test_legacy_three_forms(tmp_path):
    m = fresh(tmp_path)
    try:
        assert parse_statement("杯子是容器", m).predicate == "is_a"
        assert parse_statement("杯子有把手", m).predicate == "has"
        assert parse_statement("杯子可以盛水", m).predicate == "capable"
        r = parse_chinese_simple("杯子是容器", m)
        assert r["parsed"] is True and r["kind"] == "is_a"
        assert r["subj"] == "杯子" and r["obj"] == "容器"
    finally:
        m.close()


def test_can_variants_map(tmp_path):
    m = fresh(tmp_path, tag="sp2")
    try:
        for t in ("杯子可以盛水", "杯子能够盛水", "杯子能盛水"):
            u = parse_statement(t, m)
            assert (u.predicate, u.modal) == ("capable", "can"), (t, u)
    finally:
        m.close()


def test_negation_is_explicit(tmp_path):
    m = fresh(tmp_path, tag="sp3")
    try:
        cases = [("杯子不能盛水", "capable"), ("杯子不是容器", "is_a"),
                 ("杯子没有把手", "has")]
        for t, pred in cases:
            u = parse_statement(t, m)
            assert u.negated is True, (t, u)
            assert u.predicate == pred, (t, u)
        # negation must not be reclassified as its affirmative twin
        aff = parse_statement("杯子可以盛水", m)
        neg = parse_statement("杯子不能盛水", m)
        assert aff.negated is False and neg.negated is True
        assert aff.object == neg.object == "盛水"
        # adapter carries the flag instead of dropping it
        r = parse_chinese_simple("杯子不能盛水", m)
        assert r["parsed"] is True and r["negated"] is True
    finally:
        m.close()


def test_roles_compose(tmp_path):
    m = fresh(tmp_path, tag="sp4")
    try:
        u = parse_statement("杯子不能盛水", m)
        assert (u.subject, u.subject_known) == ("杯子", True)
        assert u.modal == "can" and u.object == "盛水"
        assert u.spans.get("negation") == "不能"
        u = parse_statement("杯子没有把手", m)
        assert u.spans.get("possession") == "有"
        assert u.spans.get("negation") == "没有"
    finally:
        m.close()


def test_new_entities_without_new_regex(tmp_path):
    m = fresh(tmp_path, tag="sp5")
    try:
        o = Ontology(m)
        o.add_concept("fox", "狐狸", parents=["entity"])
        o.add_concept("swim", "游泳", parents=["entity"])
        u = parse_statement("狐狸游泳", m)
        assert (u.subject, u.subject_known) == ("狐狸", True)
        assert (u.object, u.object_known) == ("游泳", True)
        u = parse_statement("狐狸不能游泳", m)
        assert (u.predicate, u.negated) == ("capable", True)
        assert (u.object, u.object_known) == ("游泳", True)
    finally:
        m.close()


def test_unknown_and_ambiguity(tmp_path):
    m = fresh(tmp_path, tag="sp6")
    try:
        u = parse_statement("花钱如流水", m)
        assert u.predicate == "unknown" and u.subject_known is False
        u = parse_statement("杯子", m)
        assert u.predicate == "unknown" and u.ambiguity != ""
        r = parse_chinese_simple("花钱如流水", m)
        assert r["parsed"] is False
        # two concepts sharing one name: no silent pick
        o = Ontology(m)
        o.add_concept("cup2", "杯子", parents=["entity"])
        u = parse_statement("杯子是容器", m)
        assert u.ambiguity != "" and u.subject_known is False, u
    finally:
        m.close()


def test_parse_never_upgrades_knowledge(tmp_path):
    m = fresh(tmp_path, tag="sp7")
    try:
        n0 = len(m.query("SELECT * FROM attributes"))
        parse_statement("杯子是金子做的", m)
        parse_statement("杯子不能盛水", m)
        assert len(m.query("SELECT * FROM attributes")) == n0
        assert m.list_claims() == []
    finally:
        m.close()


def test_unknown_words_kept(tmp_path):
    m = fresh(tmp_path, tag="sp8")
    try:
        u = parse_statement("杯子有把手", m)
        assert "把手" in u.unknown_spans, u
        assert u.object == "把手"
        u = parse_statement("杯子不能盛水喝", m)
        assert u.object == "盛水喝" and "盛水喝" in u.unknown_spans
    finally:
        m.close()


def test_public_behavior_intact(tmp_path):
    m = fresh(tmp_path, tag="sp9")
    try:
        from cognitive_ai.reasoner import Reasoner
        o = Ontology(m)
        o.add_attribute("cup", "material", "ceramic", status="verified")
        assert Reasoner(m).get_attribute("cup", "material")[
            "conclusion"] == "ceramic"
        assert Reasoner(m).answer(
            {"type": "forward_chain"})["conclusion"] == []
        import shutil
        import subprocess
        seed_dir = tmp_path / "clid"
        seed_dir.mkdir()
        shutil.copy(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "data", "concepts.json")),
            seed_dir / "concepts.json")
        r = subprocess.run(
            [sys.executable, "-m", "cognitive_ai.cli", "parse", "杯子是容器"],
            capture_output=True, text=True, cwd=os.path.join(
                os.path.dirname(__file__), ".."),
            env={**os.environ, "PYTHONPATH": os.path.join(
                os.path.dirname(__file__), "..", "src"),
                "COGNITIVE_AI_DATA": str(seed_dir)})
        assert r.returncode == 0 and "is_a" in r.stdout, r.stderr[-500:]
    finally:
        m.close()
