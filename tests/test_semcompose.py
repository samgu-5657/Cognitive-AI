"""Phase M5-A compositional semantics (isolated tmp DBs, real parser)."""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.memory import Memory
from cognitive_ai.models import SemanticParseResult
from cognitive_ai.ontology import Ontology
from cognitive_ai.semcompose import parse_compositional_statement
from cognitive_ai.semparse import parse_question, parse_statement


def fresh(tmp_path, tag="sc"):
    m = Memory(str(tmp_path / f"{tag}.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    o.add_attribute("cup", "material", "ceramic", status="verified")
    o.add_attribute("cup", "function", "drinking", status="verified")
    return m


def counts(m):
    return (len(m.list_concepts()),
            len(m.query("SELECT * FROM attributes")),
            len(m.query("SELECT * FROM relations")),
            len(m.list_claims()),
            len(m.list_pending_candidates()))


# ---- A. atom compatibility ----
def test_atoms_match_direct_parser(tmp_path):
    m = fresh(tmp_path)
    try:
        for text in ("杯子是容器", "杯子有把手", "杯子可以盛水",
                     "杯子不能飞", "杯子不是动物", "杯子没有把手"):
            res = parse_compositional_statement(text, m)
            assert isinstance(res, SemanticParseResult)
            assert res.tree.kind == "atom", (text, res.tree)
            direct = parse_statement(text, m).to_dict()
            norm = json.loads(json.dumps(res.tree.atom, ensure_ascii=False,
                                         default=dict))
            assert norm == json.loads(json.dumps(direct,
                                                 ensure_ascii=False))
            assert res.complete is True, res
            assert res.atom_count == 1 and res.composite_count == 0
    finally:
        m.close()


def test_unsupported_atom_kept(tmp_path):
    m = fresh(tmp_path, tag="scA2")
    try:
        res = parse_compositional_statement("杯子为什么会响", m)
        assert res.tree.kind == "unsupported", res.tree
        assert "杯子为什么会响" in res.tree.raw_text
        assert res.complete is False
        assert res.diagnostics
        q = parse_question("杯子是什么？", m)
        assert q.qkind == "category"
        from cognitive_ai.learner import Learner
        cand = Learner(m, "").classify_utterance("杯子可以盛水")
        assert cand["kind"] == "attribute" and cand["can_confirm"]
    finally:
        m.close()


# ---- B. conjunction ----
def test_two_and_three_clauses(tmp_path):
    m = fresh(tmp_path, tag="scB1")
    try:
        res = parse_compositional_statement("杯子是容器而且杯子可以盛水",
                                            m)
        assert res.tree.kind == "conjunction", res.tree
        assert [c.raw_text for c in res.tree.children] == \
            ["杯子是容器", "杯子可以盛水"]
        assert all(c.kind == "atom" for c in res.tree.children)
        assert res.complete is True
        assert res.atom_count == 2 and res.composite_count == 1
        res3 = parse_compositional_statement(
            "杯子是容器并且杯子有把手而且杯子可以盛水", m)
        assert res3.tree.kind == "conjunction"
        assert [c.raw_text for c in res3.tree.children] == \
            ["杯子是容器", "杯子有把手", "杯子可以盛水"]
        assert res3.tree.connective == "并且+而且", res3.tree
        assert res3.complete is True
    finally:
        m.close()


def test_bad_clause_kept_not_dropped(tmp_path):
    m = fresh(tmp_path, tag="scB2")
    try:
        res = parse_compositional_statement("杯子是容器而且杯子为什么会响",
                                            m)
        assert res.tree.kind == "conjunction"
        kinds = [c.kind for c in res.tree.children]
        assert kinds == ["atom", "unsupported"], kinds
        assert res.complete is False
        assert res.diagnostics
    finally:
        m.close()


# ---- C. conditionals ----
def test_basic_conditional(tmp_path):
    m = fresh(tmp_path, tag="scC1")
    try:
        res = parse_compositional_statement("如果杯子是容器，那么杯子可以盛水",
                                            m)
        assert res.tree.kind == "conditional", res.tree
        assert res.tree.roles == ("antecedent", "consequent")
        ant, cons = res.tree.children
        assert (ant.raw_text, cons.raw_text) == ("杯子是容器", "杯子可以盛水")
        assert ant.kind == cons.kind == "atom"
        assert res.complete is True
        assert res.atom_count == 2 and res.composite_count == 1
    finally:
        m.close()


def test_conjunctive_branches(tmp_path):
    m = fresh(tmp_path, tag="scC2")
    try:
        res = parse_compositional_statement(
            "如果杯子是容器而且杯子有把手，那么杯子可以盛水", m)
        ant, cons = res.tree.children
        assert ant.kind == "conjunction" and cons.kind == "atom"
        assert res.complete is True
        res2 = parse_compositional_statement(
            "如果杯子是容器，那么杯子有把手而且杯子可以盛水", m)
        ant2, cons2 = res2.tree.children
        assert ant2.kind == "atom" and cons2.kind == "conjunction"
        assert res2.complete is True
    finally:
        m.close()


def test_conditional_edges(tmp_path):
    m = fresh(tmp_path, tag="scC3")
    try:
        res = parse_compositional_statement("如果杯子为什么会响，那么杯子是容器",
                                            m)
        assert res.tree.kind == "conditional"
        assert res.tree.children[0].kind == "unsupported"
        assert res.complete is False
        empty = parse_compositional_statement("如果，那么杯子是容器", m)
        assert empty.tree.kind == "unsupported"
        assert empty.complete is False
        bare = parse_compositional_statement("如果杯子是容器", m)
        assert bare.tree.kind != "conditional", bare.tree
    finally:
        m.close()


# ---- D. nesting & priority ----
def test_nested_and_quotes(tmp_path):
    m = fresh(tmp_path, tag="scD1")
    try:
        res = parse_compositional_statement(
            "杯子是容器而且如果杯子有把手，那么杯子可以盛水", m)
        assert res.tree.kind == "conjunction"
        assert res.tree.children[1].kind == "conditional"
        assert res.complete is True
        res2 = parse_compositional_statement(
            "如果杯子是容器，那么杯子有把手而且杯子可以盛水", m)
        assert res2.tree.kind == "conditional"
        assert res2.tree.children[1].kind == "conjunction"
        quoted = parse_compositional_statement(
            "他说“杯子是容器而且杯子可以盛水”", m)
        assert quoted.tree.kind != "conjunction", quoted.tree
    finally:
        m.close()


# ---- E. limits & errors ----
def test_depth_and_node_limits(tmp_path, monkeypatch):
    import cognitive_ai.policy as policy
    m = fresh(tmp_path, tag="scE1")
    try:
        def nest(n):
                text = "杯子可以盛水"
                for _ in range(n):
                    text = f"如果杯子是容器，那么{text}"
                return text

        ok = parse_compositional_statement(nest(3), m)
        assert ok.tree.kind == "conditional" and ok.complete is True
        over = parse_compositional_statement(nest(6), m)
        assert over.complete is False
        assert any("depth" in d for d in over.diagnostics), over
        monkeypatch.setattr(policy, "MAX_SEMCOMPOSE_NODES", 4)
        many = parse_compositional_statement(
            "杯子是容器而且杯子有把手而且杯子可以盛水而且杯子是容器",
            m)
        assert many.complete is False
        assert any("budget" in d for d in many.diagnostics), many
        assert parse_compositional_statement("", m).complete is False
        assert parse_compositional_statement("   ", m).tree is None
        trailing = parse_compositional_statement("杯子是容器并且", m)
        assert trailing.tree.kind == "unsupported"
        assert trailing.complete is False
        cause = parse_compositional_statement("因为杯子是容器，所以杯子可以盛水",
                                              m)
        assert cause.tree.kind != "conditional", cause.tree
    finally:
        m.close()


# ---- F. deep immutability ----
def test_tree_frozen(tmp_path):
    m = fresh(tmp_path, tag="scF1")
    try:
        res = parse_compositional_statement("杯子是容器而且杯子可以盛水",
                                            m)
        with pytest.raises(AttributeError):
            res.tree.raw_text = "x"
        with pytest.raises(TypeError):
            res.tree.children[0].atom["subject"] = "y"
        with pytest.raises(AttributeError):
            res.tree.children += ()
        again = parse_compositional_statement("杯子是容器而且杯子可以盛水",
                                              m)
        assert again == res
    finally:
        m.close()


# ---- G. no side effects ----
def test_parse_writes_nothing(tmp_path, monkeypatch):
    m = fresh(tmp_path, tag="scG1")
    try:
        n0 = counts(m)
        calls = {"n": 0}
        real = Memory.add_pending_candidate

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(Memory, "add_pending_candidate", spy)
        for text in ("杯子是容器", "杯子是容器而且杯子可以盛水",
                     "如果杯子是容器，那么杯子可以盛水",
                     "杯子为什么会响"):
            parse_compositional_statement(text, m)
        assert calls["n"] == 0
        assert counts(m) == n0
    finally:
        m.close()


# ---- H. importer still works ----
def test_importer_unaffected(tmp_path):
    from cognitive_ai.knowledge_import import preview_import
    m = fresh(tmp_path, tag="scH1")
    try:
        path = tmp_path / "h.txt"
        path.write_text("杯子有盖子\n", encoding="utf-8")
        pv = preview_import(str(path), m)
        assert pv.items[0].category == "ready_for_candidate"
    finally:
        m.close()


def test_cli_semantic_parse(tmp_path):
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
        [sys.executable, "-m", "cognitive_ai.cli", "semantic-parse",
         "杯子是容器而且杯子可以盛水"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."), env=env)
    assert r.returncode == 0, r.stderr[-800:]
    assert "conjunction" in r.stdout and "complete: True" in r.stdout, \
        r.stdout
