"""Phase M5-B compositional grounding (isolated tmp DBs, real APIs)."""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import semmeaning
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.semmeaning import understand_compositional_statement


def fresh(tmp_path, tag="sm"):
    m = Memory(str(tmp_path / f"{tag}.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    o.add_attribute("cup", "material", "ceramic", status="verified")
    o.add_attribute("cup", "function", "喝水", status="verified")
    return m


def counts(m):
    return (len(m.list_concepts()),
            len(m.query("SELECT * FROM attributes")),
            len(m.query("SELECT * FROM relations")),
            len(m.list_claims()),
            len(m.query("SELECT * FROM rules")),
            len(m.list_pending_candidates()),
            len(m.list_experiences(limit=100000)))


# ---- A. atoms ----
def test_atom_statuses(tmp_path):
    m = fresh(tmp_path)
    try:
        sup = understand_compositional_statement("杯子是容器", m)
        assert sup.root.status == "supported"
        assert sup.root.grounding["verdict"] == "supported"
        assert sup.complete is True and sup.grounded_atoms == 1
        unk = understand_compositional_statement("杯子有盖子", m)
        assert unk.root.status == "unknown"
        assert unk.complete is True
        assert unk.grounded_atoms == 1
        con = understand_compositional_statement("杯子不是容器", m)
        assert con.root.status == "conflicted"
        assert con.complete is True
    finally:
        m.close()


def test_ambiguous_atom(tmp_path):
    m = fresh(tmp_path, tag="smA2")
    try:
        Ontology(m).add_concept("cup2", "杯子", parents=["entity"])
        rep = understand_compositional_statement("杯子是容器", m)
        assert rep.root.status == "ambiguous"
        assert rep.complete is False
    finally:
        m.close()


def test_atom_matches_direct_grounding(tmp_path):
    import json
    from cognitive_ai.semground import ground_statement
    m = fresh(tmp_path, tag="smA3")
    try:
        rep = understand_compositional_statement("杯子是容器", m)
        direct = ground_statement("杯子是容器", m).to_dict()
        norm = json.loads(json.dumps(rep.root.grounding,
                                     ensure_ascii=False, default=dict))
        assert norm == json.loads(json.dumps(direct, ensure_ascii=False))
    finally:
        m.close()


# ---- B. conjunction ----
def test_conjunction_aggregation(tmp_path):
    m = fresh(tmp_path, tag="smB1")
    try:
        both = understand_compositional_statement("杯子是容器而且杯子可以喝水",
                                                  m)
        assert both.root.status == "supported", both.root
        assert both.complete is True
        mixed = understand_compositional_statement(
            "杯子是容器而且杯子有盖子", m)
        assert mixed.root.status == "partially_supported", mixed.root
        assert mixed.complete is True
        neither = understand_compositional_statement("杯子有盖子而且杯子可以飞",
                                                     m)
        assert neither.root.status == "unknown", neither.root
        assert neither.complete is True
        clash = understand_compositional_statement("杯子是容器而且杯子不是容器",
                                                   m)
        assert clash.root.status == "conflicted", clash.root
        assert [c.raw_text for c in clash.root.children] == \
            ["杯子是容器", "杯子不是容器"]
    finally:
        m.close()


def test_unsupported_child_kept(tmp_path):
    m = fresh(tmp_path, tag="smB2")
    try:
        rep = understand_compositional_statement("杯子是容器而且杯子为什么会响",
                                                 m)
        kinds = [c.status for c in rep.root.children]
        assert kinds == ["supported", "unsupported"], kinds
        assert rep.complete is False
        assert rep.diagnostics
    finally:
        m.close()


# ---- C. conditionals ----
def test_conditional_stays_unasserted(tmp_path):
    m = fresh(tmp_path, tag="smC1")
    try:
        rep = understand_compositional_statement("如果杯子是容器，那么杯子可以喝水",
                                                 m)
        assert rep.root.status == "conditional_unasserted", rep.root
        assert rep.root.roles == ("antecedent", "consequent")
        ant, cons = rep.root.children
        assert (ant.status, cons.status) == ("supported", "supported")
        assert ant.grounding["verdict"] == "supported"
        assert rep.complete is True
        mixed = understand_compositional_statement(
            "如果杯子是容器，那么杯子有盖子", m)
        assert mixed.root.status == "conditional_unasserted"
        assert mixed.root.children[1].status == "unknown"
        assert mixed.complete is True
        flipped = understand_compositional_statement(
            "如果杯子有盖子，那么杯子是容器", m)
        assert flipped.root.status == "conditional_unasserted"
        assert flipped.complete is True
    finally:
        m.close()


def test_conditional_writes_nothing(tmp_path):
    m = fresh(tmp_path, tag="smC2")
    try:
        n0 = counts(m)
        understand_compositional_statement("如果杯子是容器，那么杯子可以喝水",
                                           m)
        assert counts(m) == n0
    finally:
        m.close()


# ---- D. depth & partial ----
def test_limits_propagate(tmp_path, monkeypatch):
    import cognitive_ai.policy as policy
    m = fresh(tmp_path, tag="smD1")
    try:
        rep = understand_compositional_statement("杯子是容器并且", m)
        assert rep.complete is False
        monkeypatch.setattr(policy, "MAX_SEMCOMPOSE_NODES", 2)
        rep2 = understand_compositional_statement("杯子是容器而且杯子可以喝水",
                                                  m)
        assert rep2.complete is False
        assert rep2.diagnostics
    finally:
        m.close()


# ---- E. real call path ----
def test_spies_prove_real_calls(tmp_path, monkeypatch):
    from cognitive_ai import semcompose as compose_mod
    from cognitive_ai import semground as ground_mod
    m = fresh(tmp_path, tag="smE1")
    try:
        calls = {"parse": [], "ground": []}
        real_parse = compose_mod.parse_compositional_statement
        real_ground = ground_mod.ground_statement

        def spy_parse(text, mem, *a, **k):
            calls["parse"].append(text)
            return real_parse(text, mem, *a, **k)

        def spy_ground(text, mem):
            calls["ground"].append(text)
            return real_ground(text, mem)

        monkeypatch.setattr(compose_mod, "parse_compositional_statement",
                            spy_parse)
        monkeypatch.setattr(ground_mod, "ground_statement", spy_ground)
        rep = understand_compositional_statement(
            "如果杯子是容器，那么杯子可以喝水而且杯子是容器", m)
        assert calls["parse"] == ["如果杯子是容器，那么杯子可以喝水而且杯子是容器"]
        assert sorted(set(calls["ground"])) == \
            sorted(["杯子是容器", "杯子可以喝水"])
        assert rep.grounded_atoms == 2
        assert "Learner" not in dir(semmeaning)
    finally:
        m.close()


# ---- F. frozen ----
def test_report_frozen(tmp_path):
    m = fresh(tmp_path, tag="smF1")
    try:
        rep = understand_compositional_statement("杯子是容器而且杯子可以喝水",
                                                 m)
        with pytest.raises(AttributeError):
            rep.complete = False
        with pytest.raises(TypeError):
            rep.root.children[0].grounding["verdict"] = "x"
        with pytest.raises(AttributeError):
            rep.root.children += ()
        with pytest.raises(TypeError):
            rep.counts["supported"] = 0
        again = understand_compositional_statement("杯子是容器而且杯子可以喝水",
                                                   m)
        assert again == rep
    finally:
        m.close()


# ---- G. read-only ----
def test_understand_writes_nothing(tmp_path, monkeypatch):
    m = fresh(tmp_path, tag="smG1")
    try:
        n0 = counts(m)
        seen = {"pending": 0, "attr": 0, "rule": 0}
        real_pending = Memory.add_pending_candidate
        real_attr = Memory.add_attribute
        real_rule = Memory.upsert_rule

        def spy_pending(*a, **k):
            seen["pending"] += 1
            return real_pending(*a, **k)

        def spy_attr(*a, **k):
            seen["attr"] += 1
            return real_attr(*a, **k)

        def spy_rule(*a, **k):
            seen["rule"] += 1
            return real_rule(*a, **k)

        monkeypatch.setattr(Memory, "add_pending_candidate", spy_pending)
        monkeypatch.setattr(Memory, "add_attribute", spy_attr)
        monkeypatch.setattr(Memory, "upsert_rule", spy_rule)
        for text in ("杯子是容器", "杯子是容器而且杯子可以喝水",
                     "如果杯子是容器，那么杯子可以喝水",
                     "杯子不是容器", "杯子为什么会响"):
            understand_compositional_statement(text, m)
        assert seen == {"pending": 0, "attr": 0, "rule": 0}, seen
        assert counts(m) == n0
    finally:
        m.close()


def test_cli_semantic_analyze(tmp_path):
    import shutil
    data = tmp_path / "cli"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(os.path.join(os.path.dirname(__file__), "..", "data",
                             "concepts.json"),
                data / "concepts.json")
    env = {**os.environ,
           "PYTHONPATH": os.path.abspath(os.path.join(
               os.path.dirname(__file__), "..", "src")),
           "COGNITIVE_AI_DATA": str(data)}
    r = subprocess.run(
        [sys.executable, "-m", "cognitive_ai.cli", "semantic-analyze",
         "杯子是容器而且杯子可以喝水"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."), env=env)
    assert r.returncode == 0, r.stderr[-800:]
    assert "conjunction" in r.stdout, r.stdout
    assert "conditional_unasserted" not in r.stdout
    assert "partially_supported" in r.stdout, r.stdout
