"""Phase M5-C explicit context (isolated tmp DBs, real APIs)."""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import semcontext as sc
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.semmeaning import understand_compositional_statement


def fresh(tmp_path, tag="sx"):
    m = Memory(str(tmp_path / f"{tag}.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    o.add_concept("bowl", "碗", parents=["container"])
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


def context_with(m, *priors):
    context = sc.empty_context()
    for prior in priors:
        analyzed = understand_compositional_statement(prior, m)
        context = sc.extend_semantic_context(context, analyzed, m)
    return context


# ---- A. unique reference ----
def test_unique_pronoun(tmp_path):
    m = fresh(tmp_path)
    try:
        context = context_with(m, "杯子是容器")
        assert context.sequence == 1
        assert [x.object_ref.ref_id for x in context.mentions] == ["cup"]
        assert context.mentions[0].object_ref.kind == "legacy_concept"
        rep = sc.understand_with_context("它可以喝水", m, context)
        assert rep.raw_input == "它可以喝水"
        assert len(rep.resolutions) == 1
        resolution = rep.resolutions[0]
        assert resolution.status == "resolved", resolution
        assert resolution.resolved_ref.ref_id == "cup"
        assert rep.normalized_text == "杯子可以喝水", rep
        assert rep.understanding is not None
        assert rep.understanding.grounded_atoms >= 1
        assert rep.complete is True
    finally:
        m.close()


# ---- B. no antecedent ----
def test_empty_context_unresolved(tmp_path):
    m = fresh(tmp_path, tag="sxB")
    try:
        rep = sc.understand_with_context("它可以喝水", m,
                                         sc.empty_context())
        assert rep.resolutions[0].status == "unresolved"
        assert rep.understanding is None
        assert rep.complete is False
        assert rep.normalized_text == ""
    finally:
        m.close()


# ---- C. several candidates ----
def test_two_candidates_ambiguous(tmp_path):
    m = fresh(tmp_path, tag="sxC")
    try:
        context = context_with(m, "杯子是容器", "碗是容器")
        rep = sc.understand_with_context("它可以喝水", m, context)
        resolution = rep.resolutions[0]
        assert resolution.status == "ambiguous", resolution
        assert {c.ref_id for c in resolution.candidates} == {"cup",
                                                             "bowl"}
        assert rep.understanding is None
        assert rep.complete is False
    finally:
        m.close()


# ---- D. same candidate repeated ----
def test_repeated_mention_deduped(tmp_path):
    m = fresh(tmp_path, tag="sxD")
    try:
        context = context_with(m, "杯子是容器", "杯子有陶瓷")
        assert len(context.mentions) == 2
        resolution = sc.resolve_subject_reference("它", context, m)
        assert resolution.status == "resolved", resolution
        assert [c.ref_id for c in resolution.candidates] == ["cup"]
    finally:
        m.close()


def test_ambiguous_name_not_merged(tmp_path):
    m = fresh(tmp_path, tag="sxE")
    try:
        context = context_with(m, "杯子是容器")
        assert [x.object_ref.ref_id for x in context.mentions] == ["cup"]
        Ontology(m).add_concept("cup2", "杯子", parents=["entity"])
        resolution = sc.resolve_subject_reference("它", context, m)
        assert resolution.status == "ambiguous", resolution
    finally:
        m.close()


# ---- F. individual boundary ----
def test_demonstratives_need_individuals(tmp_path):
    m = fresh(tmp_path, tag="sxF")
    try:
        context = context_with(m, "杯子是容器")
        rep = sc.understand_with_context("这个杯子可以喝水", m, context)
        assert rep.resolutions[0].status == "unsupported", rep.resolutions
        assert rep.understanding is None
        rep2 = sc.understand_with_context("前面提到的物体可以喝水", m,
                                          context)
        assert rep2.resolutions[0].status in ("unsupported", "ambiguous")
    finally:
        m.close()


def test_plain_noun_not_a_reference(tmp_path):
    m = fresh(tmp_path, tag="sxF2")
    try:
        context = context_with(m, "杯子是容器")
        resolution = sc.resolve_subject_reference("杯子", context, m)
        assert resolution.status == "not_reference"
    finally:
        m.close()


# ---- G. composition ----
def test_conjunction_pronouns(tmp_path):
    m = fresh(tmp_path, tag="sxG")
    try:
        context = context_with(m, "杯子是容器")
        rep = sc.understand_with_context("它有陶瓷，而且它可以喝水", m,
                                         context)
        assert rep.complete is True, rep
        assert rep.normalized_text == "杯子有陶瓷，而且杯子可以喝水", rep
        assert [r.status for r in rep.resolutions] == ["resolved",
                                                      "resolved"]
        assert rep.understanding.grounded_atoms == 2
    finally:
        m.close()


def test_conditional_pronouns(tmp_path):
    m = fresh(tmp_path, tag="sxG2")
    try:
        context = context_with(m, "杯子是容器")
        rep = sc.understand_with_context("如果它是容器，那么它可以喝水", m,
                                         context)
        assert rep.complete is True, rep
        assert rep.normalized_text == "如果杯子是容器，那么杯子可以喝水"
        assert rep.understanding.root.status == \
            "conditional_unasserted"
    finally:
        m.close()


# ---- H. read-only & semantics kept ----
def test_context_read_only(tmp_path, monkeypatch):
    m = fresh(tmp_path, tag="sxH")
    try:
        n0 = counts(m)
        before = understand_compositional_statement("杯子是容器", m)
        context = context_with(m, "杯子是容器")
        sc.understand_with_context("它可以喝水", m, context)
        sc.understand_with_context("杯子是容器", m, context)
        assert counts(m) == n0
        assert understand_compositional_statement("杯子是容器",
                                                  m) == before
        seen = {"pending": 0}
        real = Memory.add_pending_candidate

        def spy(*a, **k):
            seen["pending"] += 1
            return real(*a, **k)

        monkeypatch.setattr(Memory, "add_pending_candidate", spy)
        sc.understand_with_context("它可以喝水", m, context)
        assert seen["pending"] == 0
        assert counts(m) == n0
    finally:
        m.close()


def test_trimming_recorded(tmp_path):
    import cognitive_ai.policy as policy
    m = fresh(tmp_path, tag="sxH2")
    try:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(policy, "MAX_SEMANTIC_CONTEXT_MENTIONS", 1)
        try:
            context = context_with(m, "杯子是容器", "碗是容器")
            assert context.trimmed is True
            assert [x.object_ref.ref_id for x in context.mentions] == \
                ["bowl"]
        finally:
            monkeypatch.undo()
    finally:
        m.close()


# ---- I. frozen ----
def test_context_frozen(tmp_path):
    m = fresh(tmp_path, tag="sxI")
    try:
        context = context_with(m, "杯子是容器")
        with pytest.raises(AttributeError):
            context.sequence = 9
        with pytest.raises(AttributeError):
            context.mentions[0].mention_text = "x"
        rep = sc.understand_with_context("它可以喝水", m, context)
        with pytest.raises(AttributeError):
            rep.complete = False
        with pytest.raises(AttributeError):
            rep.resolutions[0].candidates += ()
        with pytest.raises(TypeError):
            rep.understanding.root.grounding["verdict"] = "x"
        items = list(context.mentions)
        items.append(context.mentions[0])
        assert len(context.mentions) == 1
    finally:
        m.close()


# ---- J. CLI & regression ----
def test_cli_context(tmp_path):
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

    def run(*args):
        r = subprocess.run(
            [sys.executable, "-m", "cognitive_ai.cli"] + list(args),
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."), env=env)
        assert r.returncode == 0, r.stderr[-800:]
        return r.stdout

    out = run("semantic-context", "--prior", "杯子是容器", "它有什么用途？")
    assert "resolved" in out, out
    assert "cup" in out, out
    empty = run("semantic-context", "它有什么用途？")
    assert "unresolved" in empty, empty
    clash = run("semantic-context", "--prior", "杯子是容器",
                "--prior", "碗是容器", "它有什么用途？")
    assert "ambiguous" in clash, clash
