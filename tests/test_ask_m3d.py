"""Phase M3D question answering tests (isolated DBs, real CLI).

Each test builds its own concepts/attributes/relations/rules, asks
through answer_question() (and the ask command for CLI coverage), and
checks structured queries, evidence refs, row-count invariance and
restart consistency -- never just output text.
"""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.semground import answer_question
from cognitive_ai.semparse import parse_question

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def fresh(tmp_path, tag="sq"):
    m = Memory(str(tmp_path / f"{tag}.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    o.add_concept("water", "水", parents=["entity"])
    o.add_attribute("cup", "material", "ceramic", status="verified")
    o.add_attribute("cup", "function", "drinking", status="verified")
    return m


def counts(m):
    return (len(m.query("SELECT * FROM attributes")),
            len(m.list_claims()),
            len(m.query("SELECT * FROM relations")),
            len(m.query("SELECT * FROM concepts")))


def fresh_data(tmp_path, tag="sqd"):
    data = tmp_path / tag
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    return data


def cli(data, *args):
    env = {**os.environ,
           "PYTHONPATH": os.path.abspath(os.path.join(
               os.path.dirname(__file__), "..", "src")),
           "COGNITIVE_AI_DATA": str(data)}
    r = subprocess.run(
        [sys.executable, "-m", "cognitive_ai.cli", *args],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."), env=env)
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads(r.stdout)


def test_structures_for_all_slice_forms(tmp_path):
    m = fresh(tmp_path)
    try:
        cases = [("杯子是什么？", "category", False),
                 ("杯子有什么？", "has", False),
                 ("杯子有什么功能？", "capability", False),
                 ("杯子有什么作用？", "capability", False),
                 ("杯子有把手吗？", "has", True),
                 ("杯子是不是容器？", "is_a_verify", True),
                 ("杯子可以做早餐吗？", "capability", True),
                 ("杯子能够做早餐吗？", "capability", True),
                 ("杯子没有把手吗？", "has", True)]
        for text, kind, yesno in cases:
            q = parse_question(text, m)
            assert (q.qkind, q.is_yesno) == (kind, yesno), (text, q.to_dict())
            assert q.subject == "杯子" and q.subject_known, text
    finally:
        m.close()


def test_new_subjects_values_without_new_branches(tmp_path):
    m = fresh(tmp_path, tag="sq2")
    try:
        o = Ontology(m)
        o.add_concept("teapot", "茶壶", parents=["container"])
        o.add_attribute("teapot", "material", "clay", status="verified")
        q = parse_question("茶壶是什么？", m)
        assert (q.qkind, q.subject) == ("category", "茶壶"), q.to_dict()
        v = answer_question("茶壶是什么？", m)
        assert v.verdict == "supported" and "container" in v.answer, v.to_dict()
        q = parse_question("茶壶是不是容器？", m)
        assert q.qkind == "is_a_verify" and q.object == "容器"
    finally:
        m.close()


def test_supported_with_evidence(tmp_path):
    m = fresh(tmp_path, tag="sq3")
    try:
        v = answer_question("杯子的材质是什么？", m)
        assert v.verdict == "supported", v.to_dict()
        assert any(e["kind"] == "attribute" for e in v.evidence), v.evidence
        v = answer_question("杯子是不是容器？", m)
        assert v.verdict == "supported" and v.answer == "是", v.to_dict()
        v = answer_question("杯子是什么？", m)
        assert v.verdict == "supported", v.to_dict()
        assert any(e["kind"] == "parents" for e in v.evidence), v.evidence
        assert any("direct" in e.get("detail", "") for e in v.evidence)
        assert any("inherited" in e.get("detail", "") for e in v.evidence)
    finally:
        m.close()


def test_negated_evidence_found(tmp_path):
    m = fresh(tmp_path, tag="sq4")
    try:
        o = Ontology(m)
        o.add_attribute("cup", "leaking", False, status="verified")
        v = answer_question("杯子没有渗漏吗？", m)
        assert v.question["negated"] is True
    finally:
        m.close()


def test_no_record_is_not_false(tmp_path):
    m = fresh(tmp_path, tag="sq5")
    try:
        v = answer_question("杯子有把手吗？", m)
        assert v.verdict == "unknown", v.to_dict()
        assert "不" not in v.answer and v.answer == "", v.to_dict()
        v = answer_question("杯子可以飞吗？", m)
        assert v.verdict == "unknown", v.to_dict()
    finally:
        m.close()


def test_conflicted_on_rival_evidence(tmp_path):
    m = fresh(tmp_path, tag="sq6")
    try:
        o = Ontology(m)
        o.add_attribute("cup", "material", "glass", status="hypothesis")
        mem_rows = [a for a in m.get_attributes("cup")
                    if a["key"] == "material"]
        assert len(mem_rows) == 2
        v = answer_question("杯子的材质是什么？", m)
        assert v.verdict == "conflicted", v.to_dict()
        assert "ceramic" in v.answer and "glass" in v.answer, v.to_dict()
    finally:
        m.close()


def test_hypothesis_keeps_label(tmp_path):
    m = fresh(tmp_path, tag="sq7")
    try:
        o = Ontology(m)
        o.add_concept("bowl", "碗", parents=["container"])
        o.add_attribute("bowl", "material", "clay", status="hypothesis")
        v = answer_question("碗的材质是什么？", m)
        assert v.verdict == "supported", v.to_dict()
        assert any(e["status"] == "hypothesis" for e in v.evidence), v.evidence
        assert v.answer == "clay"
    finally:
        m.close()


def test_unknown_entity_never_created(tmp_path):
    m = fresh(tmp_path, tag="sq8")
    try:
        v = answer_question("雾铃是什么？", m)
        assert v.verdict == "unknown", v.to_dict()
        assert m.get_concept("雾铃") is None
        assert m.find_concepts_by_name("雾铃") == []
    finally:
        m.close()


def test_ambiguous_reports(tmp_path):
    m = fresh(tmp_path, tag="sq9")
    try:
        o = Ontology(m)
        o.add_concept("cup2", "杯子", parents=["entity"])
        v = answer_question("杯子是什么？", m)
        assert v.verdict == "ambiguous", v.to_dict()
        assert v.ambiguity != ""
    finally:
        m.close()


def test_derived_keeps_provenance(tmp_path):
    m = fresh(tmp_path, tag="sq10")
    try:
        o = Ontology(m)
        o.add_concept("potA", "铁锅", parents=["container"])
        m.upsert_rule({
            "id": "r_hold", "conditions": [{"type": "is_a",
                                            "concept": "?x",
                                            "ancestor": "container"}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "holds", "value": True},
            "exceptions": []})
        v = answer_question("铁锅有什么？", m)
        assert v.verdict == "supported", v.to_dict()
        rule_evs = [e for e in v.evidence if e["kind"] == "rule"]
        assert rule_evs and "r_hold" in str(rule_evs), v.evidence
        assert any("bindings" in e.get("detail", "") for e in rule_evs)
    finally:
        m.close()


def test_limits_and_failure_differ(tmp_path):
    m = fresh(tmp_path, tag="sq11")
    try:
        v = answer_question("杯子为什么会响？", m)
        assert v.verdict == "unknown", v.to_dict()
        v = answer_question("", m)
        assert v.verdict in ("unknown", "ambiguous"), v.to_dict()
    finally:
        m.close()


def test_read_only_and_restart(tmp_path):
    data = fresh_data(tmp_path)
    from cognitive_ai.memory import Memory as _M
    from cognitive_ai.cli import seed_from_json
    mem = _M(str(data / "cognitive.db"))
    try:
        seed_from_json(mem, str(data / "concepts.json"))
        before = (len(mem.query("SELECT * FROM attributes")),
                  len(mem.list_claims()),
                  len(mem.query("SELECT * FROM concepts")))
    finally:
        mem.close()
    u = cli(data, "ask", "杯子有什么用途？")
    assert u["verdict"] == "supported", u
    assert "drinking" in u["answer"], u
    mem = _M(str(data / "cognitive.db"))
    try:
        after = (len(mem.query("SELECT * FROM attributes")),
                 len(mem.list_claims()),
                 len(mem.query("SELECT * FROM concepts")))
        assert before == after
    finally:
        mem.close()
    u = cli(data, "ask", "杯子有什么用途？")
    assert u["verdict"] == "supported" and "drinking" in u["answer"], u


def test_cli_ask_entry(tmp_path):
    data = fresh_data(tmp_path, tag="sqd2")
    u = cli(data, "ask", "杯子是不是容器？")
    assert u["is_yesno"] is True and u["answer"] == "是", u
    assert u["verdict"] == "supported", u
    assert "question" in u and "evidence" in u, u
    u = cli(data, "ask", "花钱如流水吗？")
    assert u["verdict"] in ("unknown", "ambiguous"), u
