"""Phase M4-J batch import (isolated tmp DBs, never Temp/)."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.knowledge_import import (classify_preview_item,
                                           preview_import, stage_import)
from cognitive_ai.learner import Learner
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology


def fresh(tmp_path, tag="ki"):
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
    return (len(m.list_concepts()),
            len(m.query("SELECT * FROM attributes")),
            len(m.query("SELECT * FROM relations")),
            len(m.list_claims()),
            len(m.query("SELECT * FROM rules")),
            len(m.list_pending_candidates()),
            len(m.list_experiences(limit=100000)))


def write(path, data, binary=False):
    mode = "wb" if binary else "w"
    kw = {} if binary else {"encoding": "utf-8"}
    with open(str(path), mode, **kw) as fh:
        fh.write(data)
    return str(path)


# ---- A. file parsing ----
def test_txt_lines_and_empty(tmp_path):
    m = fresh(tmp_path, tag="kiA1")
    try:
        p = write(tmp_path / "a.txt", "杯子是容器\n\n杯子有盖子\n")
        pv = preview_import(p, m)
        assert pv.format == "txt" and pv.error == ""
        assert pv.total_lines == 3
        assert [it.line_no for it in pv.items] == [1, 3]
        assert pv.items[0].text == "杯子是容器"
    finally:
        m.close()


def test_jsonl_mixed_and_empty_file(tmp_path):
    m = fresh(tmp_path, tag="kiA2")
    try:
        p = write(tmp_path / "b.jsonl",
                  '{"text":"杯子是容器"}\nnot json\n\n'
                  '{"text":123}\n{}\n{"nope":1}\n')
        pv = preview_import(p, m)
        assert pv.format == "jsonl"
        cats = [it.category for it in pv.items]
        assert cats[0] == "already_known", pv.items[0]
        assert cats[1:] == ["invalid_input"] * 4, cats
        assert pv.counts["invalid_input"] == 4
        p2 = write(tmp_path / "empty.txt", "")
        pv2 = preview_import(p2, m)
        assert pv2.items == () and pv2.error == ""
    finally:
        m.close()


def test_unreadable_and_bad_encoding(tmp_path):
    m = fresh(tmp_path, tag="kiA3")
    try:
        pv = preview_import(str(tmp_path / "nope.txt"), m)
        assert pv.error != "" and pv.items == ()
        p = write(tmp_path / "bad.txt", b"\xff\xfe\x00bad",
                  binary=True)
        pv2 = preview_import(p, m)
        assert "UTF-8" in pv2.error and pv2.items == ()
        p4 = write(tmp_path / "weird.csv", "a,b\n")
        pv3 = preview_import(p4, m)
        assert "unsupported format" in pv3.error
    finally:
        m.close()


def test_unicode_kept(tmp_path):
    m = fresh(tmp_path, tag="kiA4")
    try:
        p = write(tmp_path / "u.txt", "马克杯可以盛水\n")
        pv = preview_import(p, m)
        assert "马克杯" in pv.items[0].text
    finally:
        m.close()


# ---- B. semantics from real APIs ----
def test_semantic_categories(tmp_path):
    m = fresh(tmp_path, tag="kiB1")
    try:
        o = Ontology(m)
        o.add_concept("cup2", "杯子", parents=["entity"])
        p = write(tmp_path / "s.txt", "杯子是容器\n杯子不是容器\n")
        pv = preview_import(p, m)
        cats = [it.category for it in pv.items]
        assert cats == ["ambiguous", "ambiguous"], cats
    finally:
        m.close()
    m2 = fresh(tmp_path, tag="kiB1b")
    try:
        p2 = write(tmp_path / "s_b.txt", "杯子为什么会响\n")
        pv2 = preview_import(p2, m2)
        assert pv2.items[0].category == "unsupported", pv2.items[0]
    finally:
        m2.close()


def test_ready_known_conflict(tmp_path):
    m = fresh(tmp_path, tag="kiB2")
    try:
        p = write(tmp_path / "s2.txt",
                  "杯子有盖子\n杯子是容器\n杯子不是容器\n")
        pv = preview_import(p, m)
        cats = [it.category for it in pv.items]
        assert cats == ["ready_for_candidate", "already_known",
                        "conflicted"], cats
    finally:
        m.close()


def test_unknown_category_contract():
    cat, _reason = classify_preview_item(
        {"parse": {"predicate": "has", "ambiguity": ""},
         "grounding": {"verdict": "unknown"},
         "kind": "mystery", "can_confirm": False, "reason": "x"})
    assert cat == "unknown"


# ---- C. provenance ----
def test_sources_kept(tmp_path):
    m = fresh(tmp_path, tag="kiC1")
    try:
        p = write(tmp_path / "facts.txt", "杯子有盖子\n杯子有盖子\n")
        pv = preview_import(p, m)
        a, b = pv.items
        assert (a.line_no, b.line_no) == (1, 2)
        assert a.source_id == "facts.txt:1"
        assert b.source_id == "facts.txt:2"
        assert a.source_group == b.source_group == "facts.txt"
        assert b.category == "duplicate_in_batch"
        p2 = write(tmp_path / "f.jsonl",
                   '{"text":"杯子有盖子","source_id":"book-a:p12",'
                   '"source_group":"book-a"}\n'
                   '{"text":"杯子是容器"}\n')
        pv2 = preview_import(p2, m)
        assert pv2.items[0].source_id == "book-a:p12"
        assert pv2.items[0].source_group == "book-a"
        assert pv2.items[1].source_id == "f.jsonl:2"
        assert pv2.items[1].source_group == "f.jsonl"
    finally:
        m.close()


# ---- D. preview is read-only ----
def test_preview_writes_nothing(tmp_path, monkeypatch):
    m = fresh(tmp_path, tag="kiD1")
    try:
        n0 = counts(m)
        calls = {"n": 0}
        real = Memory.add_pending_candidate

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(Memory, "add_pending_candidate", spy)
        p = write(tmp_path / "r.txt", "杯子有盖子\n杯子是容器\n")
        pv = preview_import(p, m)
        assert len(pv.items) == 2
        assert calls["n"] == 0
        assert counts(m) == n0
    finally:
        m.close()


# ---- E. staging ----
def test_stage_only_ready(tmp_path):
    m = fresh(tmp_path, tag="kiE1")
    try:
        p = write(tmp_path / "w.txt",
                  "杯子有盖子\n杯子有盖子\n杯子是容器\n"
                  "杯子不是容器\n杯子为什么会响\n")
        learner = Learner(m, "")
        out = stage_import(preview_import(p, m), learner, m)
        assert out["staged"] == 1 and out["failed"] == 0, out
        staged = [r for r in out["results"]
                  if r["outcome"] == "staged"]
        assert len(staged) == 1 and staged[0]["candidate_id"] >= 1
        pending = m.list_pending_candidates(status="proposed")
        assert len(pending) == 1
        assert pending[0]["utterance"] == "杯子有盖子"
        import json as _json
        obs = _json.loads(pending[0]["observation"])
        assert obs["import_source"]["source_group"] == "w.txt", obs
        assert "[src w.txt:1]" in pending[0]["reason"]
        assert len(m.query("SELECT * FROM attributes")) == 2
        out2 = stage_import(preview_import(p, m), learner, m)
        assert out2["staged"] == 0, out2
        assert len(m.list_pending_candidates(
            status="proposed")) == 1
    finally:
        m.close()


# ---- CLI ----
def test_cli_preview_default_and_stage(tmp_path):
    import shutil
    data = tmp_path / "cli"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(os.path.join(os.path.dirname(__file__), "..", "data",
                             "concepts.json"),
                data / "concepts.json")
    from cognitive_ai.cli import seed_from_json
    from cognitive_ai.memory import Memory as _M
    mem = _M(str(data / "cognitive.db"))
    try:
        seed_from_json(mem, str(data / "concepts.json"))
    finally:
        mem.close()
    facts = write(data / "facts.txt", "杯子有把手\n")
    env = {**os.environ,
           "PYTHONPATH": os.path.abspath(os.path.join(
               os.path.dirname(__file__), "..", "src")),
           "COGNITIVE_AI_DATA": str(data)}
    r = subprocess.run(
        [sys.executable, "-m", "cognitive_ai.cli", "import-data", facts],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."), env=env)
    assert r.returncode == 0, r.stderr[-800:]
    assert "ready_for_candidate=1" in r.stdout, r.stdout
    mem = _M(str(data / "cognitive.db"))
    try:
        assert mem.list_pending_candidates() == []
    finally:
        mem.close()
    r = subprocess.run(
        [sys.executable, "-m", "cognitive_ai.cli", "import-data", "stage",
         facts],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."), env=env)
    assert r.returncode == 0, r.stderr[-800:]
    assert "staged 1" in r.stdout, r.stdout
    mem = _M(str(data / "cognitive.db"))
    try:
        rows = mem.list_pending_candidates(status="proposed")
        assert len(rows) == 1 and rows[0]["utterance"] == "杯子有把手"
    finally:
        mem.close()
