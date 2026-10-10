"""Phase 6A NLU generalization tests: learned lexicon, paraphrases,
QA before/after teaching, abstention, and a no-hardcode audit proving
test sentences are not embedded in the parser source.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.cli import seed_from_json
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.semantics import canonicalize_predicate, parse_sentence

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))
SEM_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "src", "cognitive_ai",
                                       "semantics.py"))


def make_session(tmp_path, tag="g"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    return Session(mem, corp, models_root=str(tmp_path / f"m_{tag}"),
                   queue_path=str(tmp_path / f"q_{tag}.jsonl")), mem


def test_lexicon_learning_changes_parse(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        before = parse_sentence("杯子有把手")
        feat0 = [p for p in before.predicates if p.get("type") == "has"][0]
        assert feat0["feature"] == "把手"
        out = sess.handle("词义 把手 has_handle")
        assert any("lexical" in l for l in out)
        after = parse_sentence("杯子有把手",
                               lexicon={r["word"]: r["key"]
                                        for r in mem.list_mappings()})
        feat1 = [p for p in after.predicates if p.get("type") == "has"][0]
        assert feat1["feature"] == "has_handle", after.predicates
        # canonicalizer honors learned mappings over raw words too
        assert canonicalize_predicate(
            {"type": "has", "feature": "把手"},
            {"把手": "has_handle"})["feature"] == "has_handle"
    finally:
        mem.close()


def test_paraphrases_share_frame_after_teaching(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        sess.handle("词义 料子 material")
        lex = {r["word"]: r["key"] for r in mem.list_mappings()}
        fa = parse_sentence("杯子是陶瓷的", lexicon=lex)
        fb = parse_sentence("杯子是料子的", lexicon=lex)
        assert fa.intent == fb.intent == "attribute"
        ka = [p for p in fa.predicates if p.get("type") == "attr"][0]["key"]
        kb = [p for p in fb.predicates if p.get("type") == "attr"][0]["key"]
        assert ka == kb == "material"
    finally:
        mem.close()


def test_qa_before_after_teaching(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        # before: unknown handle question has no verified answer
        out0 = sess.handle("状态 杯子")
        assert any("未知" in l or "unknown" in l.lower() or "缺口" in l
                   for l in out0)
        sess.handle("词义 把手 has_handle")
        sess.handle("观察 杯子 把手=有")
        out1 = sess.handle("状态 杯子")
        assert any("把手" in l or "has_handle" in l for l in out1)
    finally:
        mem.close()


def test_abstention_preserved(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        out = sess.handle("理解 花钱如流水")
        assert any("无法理解" in l for l in out)
        out = sess.handle("确认")
        assert any("没有待确认" in l for l in out)
    finally:
        mem.close()


def test_no_hardcoded_test_sentences():
    """The parser source must not contain full test sentences (>=6 chars):
    compositional rules only, never per-sentence hardcodes."""
    src = open(SEM_SRC, encoding="utf-8").read()
    sentences = [
        "紫砂壶是一种茶壶", "紫砂壶通常是陶土做的", "茶壶可以用来泡茶",
        "这个壶不是金属做的", "我觉得这个壶可能是陶土做的",
        "刚才说错了，它其实是不锈钢的",
        "如果温度太高，某些材料可能会变形",
        "我今天买了一个新杯子", "它有一个把手",
        "杯子是玻璃做的", "杯子由玻璃制成", "瓶子并非玻璃所制",
        "我家有红杯子", "剪刀是工具", "鲤鱼是一种鱼",
        "这不可能是玻璃杯", "说错了，其实是塑料制品",
    ]
    leaked = [s for s in sentences if len(s) >= 6 and s in src]
    assert not leaked, f"hardcoded test sentences in parser: {leaked}"


def test_lexicon_persists_across_restart(tmp_path):
    db = str(tmp_path / "lex.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    try:
        s1 = Session(mem, str(tmp_path))
        s1.handle("词义 把手 has_handle")
    finally:
        mem.close()
    mem2 = Memory(db)
    try:
        assert mem2.lookup_mapping("把手") == "has_handle"
        f = parse_sentence("杯子有把手",
                           lexicon={r["word"]: r["key"]
                                    for r in mem2.list_mappings()})
        assert [p for p in f.predicates
                if p.get("type") == "has"][0]["feature"] == "has_handle"
    finally:
        mem2.close()
