"""Phase M3B read-only grounding tests (isolated DBs).

Every test builds its own concepts/attributes/relations/rules in a tmp
database, then checks structured verdicts, evidence refs and that the
row counts are identical before and after. Nothing here writes to the
production database.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.semground import ground_statement


def fresh(tmp_path, tag="sg"):
    m = Memory(str(tmp_path / f"{tag}.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    o.add_concept("water", "水", parents=["entity"])
    o.add_attribute("cup", "material", "ceramic", status="verified")
    return m


def counts(m):
    return (len(m.query("SELECT * FROM attributes")),
            len(m.list_claims()),
            len(m.query("SELECT * FROM relations")),
            len(m.query("SELECT * FROM concepts")))


def test_1_supported_affirmative(tmp_path):
    m = fresh(tmp_path)
    try:
        v = ground_statement("杯子是容器", m)
        assert v.verdict == "supported", v.to_dict()
        assert v.subject_match.get("concept_id") == "cup"
        assert any(e["kind"] == "is_a" for e in v.evidence), v.evidence
        v = ground_statement("杯子的材质是陶瓷", m)
        assert v.verdict in ("supported", "unknown"), v.to_dict()
    finally:
        m.close()


def test_2_unknown_attr_not_fact(tmp_path):
    m = fresh(tmp_path, tag="sg2")
    try:
        v = ground_statement("杯子有把手", m)
        assert v.verdict == "unknown", v.to_dict()
        assert "把手" in str(v.parse.get("unknown_spans", []) +
                             [v.parse.get("object", "")])
    finally:
        m.close()


def test_3_unknown_concept_never_created(tmp_path):
    m = fresh(tmp_path, tag="sg3")
    try:
        before = counts(m)
        v = ground_statement("雾铃是容器", m)
        assert v.verdict == "unknown", v.to_dict()
        assert m.get_concept("雾铃") is None
        assert counts(m) == before
    finally:
        m.close()


def test_4_direct_contradiction(tmp_path):
    m = fresh(tmp_path, tag="sg4")
    try:
        v = ground_statement("杯子不是容器", m)
        assert v.parse["negated"] is True
        assert v.verdict == "conflicted", v.to_dict()
    finally:
        m.close()


def test_5_unsupported_denial_not_proven(tmp_path):
    m = fresh(tmp_path, tag="sg5")
    try:
        v = ground_statement("杯子不是玻璃做的", m)
        assert v.verdict == "unknown", v.to_dict()
        assert "玻璃" in str(v.parse.get("unknown_spans", []) +
                             [v.parse.get("object", "")])
    finally:
        m.close()


def test_6_modality_affirm_neg(tmp_path):
    m = fresh(tmp_path, tag="sg6")
    try:
        a = ground_statement("杯子可以盛水", m)
        n = ground_statement("杯子不能盛水", m)
        assert a.parse["negated"] is False and a.parse["modal"] == "can"
        assert n.parse["negated"] is True and n.parse["modal"] == "can"
        assert a.parse["object"] == n.parse["object"] == "盛水"
        assert a.parse["predicate"] == n.parse["predicate"] == "capable"
    finally:
        m.close()


def test_7_no_cross_mismatch(tmp_path):
    m = fresh(tmp_path, tag="sg7")
    try:
        o = Ontology(m)
        o.add_concept("bowl", "碗", parents=["entity"])
        o.add_attribute("bowl", "material", "clay", status="verified")
        v = ground_statement("杯子是容器", m)
        assert v.subject_match.get("concept_id") == "cup"
        assert "bowl" not in str(v.evidence) and "clay" not in str(v.evidence)
    finally:
        m.close()


def test_8_ambiguous_or_unresolved(tmp_path):
    m = fresh(tmp_path, tag="sg8")
    try:
        o = Ontology(m)
        o.add_concept("cup2", "杯子", parents=["entity"])
        v = ground_statement("杯子是容器", m)
        assert v.verdict == "ambiguous", v.to_dict()
        assert v.ambiguity != ""
        v = ground_statement("花钱如流水", m)
        assert v.verdict == "unknown", v.to_dict()
    finally:
        m.close()


def test_9_derived_keeps_provenance(tmp_path):
    m = fresh(tmp_path, tag="sg9")
    try:
        o = Ontology(m)
        o.add_attribute("container", "holds", True, status="verified")
        m.upsert_rule({
            "id": "r_hold", "conditions": [{"type": "is_a",
                                            "concept": "?x",
                                            "ancestor": "container"}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "holds", "value": True},
            "exceptions": []})
        v = ground_statement("杯子有holds", m)
        assert v.verdict == "supported", v.to_dict()
        assert any(e["kind"] == "rule" for e in v.evidence), v.evidence
    finally:
        m.close()


def test_10_db_unchanged(tmp_path):
    m = fresh(tmp_path, tag="sg10")
    try:
        before = counts(m)
        for t in ("杯子是容器", "杯子不是容器", "雾铃是容器",
                  "杯子有把手", "杯子不能盛水"):
            ground_statement(t, m)
        assert counts(m) == before
    finally:
        m.close()


def test_11_unseen_entities_no_hardcode(tmp_path):
    m = fresh(tmp_path, tag="sg11")
    try:
        o = Ontology(m)
        o.add_concept("teapot", "茶壶", parents=["container"])
        v = ground_statement("茶壶是容器", m)
        assert v.verdict == "supported", v.to_dict()
        assert v.subject_match.get("concept_id") == "teapot"
    finally:
        m.close()


def test_12_cli_understand_readonly(tmp_path):
    import json
    import shutil
    import subprocess
    data = tmp_path / "udata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "concepts.json")),
        data / "concepts.json")
    env = {**os.environ,
           "PYTHONPATH": os.path.abspath(os.path.join(
               os.path.dirname(__file__), "..", "src")),
           "COGNITIVE_AI_DATA": str(data)}
    r = subprocess.run(
        [sys.executable, "-m", "cognitive_ai.cli", "understand",
         "杯子有把手"], capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."), env=env)
    assert r.returncode == 0, r.stderr[-500:]
    doc = json.loads(r.stdout)
    assert doc["verdict"] in ("supported", "conflicted", "unknown",
                              "ambiguous"), doc
    assert "parse" in doc and "evidence" in doc, doc
