"""Phase 6B dialogue tests: compositional question templates + unified
orchestrator. Isolated tmp DBs. No pretrained LLM anywhere in the loop:
dialog/answers only call existing local modules (asserted below).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.cli import seed_from_json
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory
from cognitive_ai.semantics import parse_sentence

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def make_session(tmp_path, tag="d6b"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    sess = Session(
        mem, corp, models_root=str(tmp_path / f"m_{tag}"),
        queue_path=str(tmp_path / f"q_{tag}.jsonl"))
    return sess, mem


def test_function_paraphrases_share_frame_and_answer(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        from cognitive_ai import dialog as D
        texts = ["杯子有什么用途", "杯子可以拿来干什么",
                 "拿杯子能干什么", "用杯子通常能做什么",
                 "杯子是干什么用的"]
        # NOTE: "为什么人们会使用杯子" is deliberately NOT here: it is a
        # reason question (reason_q), not a function paraphrase, and must
        # abstain without causal evidence instead of borrowing drinking.
        for t in texts:
            fr = parse_sentence(t, source="user")
            assert fr.intent == "query", (t, fr.intent)
            assert any(p.get("type") == "query_function"
                       for p in fr.predicates), (t, fr.predicates)
        for t in texts:
            turn = D.process(sess, t)
            assert turn["route"] == "qa", (t, turn["route"])
            assert "drinking" in turn["answer"]["text"] or \
                "饮用" in turn["answer"]["text"], (t, turn["answer"])
    finally:
        mem.close()


def test_reason_question_refuses_honestly(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        from cognitive_ai import dialog as D
        n_attr0 = len(mem.query(
            "SELECT * FROM attributes WHERE concept_id='cup'"))
        turn = D.process(sess, "为什么杯子会漏水")
        assert turn["route"] == "qa"
        assert turn["answer"]["certainty"] == "unknown"
        # must not present an attribute value as the "reason"
        assert "drinking" not in turn["answer"]["text"]
        n_attr1 = len(mem.query(
            "SELECT * FROM attributes WHERE concept_id='cup'"))
        assert n_attr0 == n_attr1, "reason question must not write attributes"
    finally:
        mem.close()


def test_smalltalk_reports_real_counts_and_writes_nothing(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        from cognitive_ai import dialog as D
        n0 = len(mem.list_concepts())
        turn = D.process(sess, "你好")
        assert turn["route"] == "smalltalk"
        assert "大语言模型" in turn["answer"]["text"]
        turn = D.process(sess, "你是谁")
        assert turn["route"] == "smalltalk"
        assert str(n0) in turn["answer"]["text"], turn["answer"]["text"]
        assert len(mem.list_concepts()) == n0
    finally:
        mem.close()


def test_followup_why_uses_provenance(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        from cognitive_ai import dialog as D
        turn = D.process(sess, "杯子有什么用途")
        assert turn["route"] == "qa"
        turn2 = D.process(sess, "为什么")
        assert turn2["route"] == "followup"
        assert "drinking" in turn2["answer"]["text"] or \
            "因为" in turn2["answer"]["text"], turn2["answer"]
    finally:
        mem.close()


def test_unknown_input_zero_writes(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        from cognitive_ai import dialog as D
        c0 = len(mem.list_concepts())
        e0 = len(mem.list_experiences(limit=10000))
        turn = D.process(sess, "花钱如流水般哗啦")
        assert turn["answer"]["certainty"] == "unknown"
        assert len(mem.list_concepts()) == c0
        # only evaluation-type dialog experiences allowed, no KB writes
        assert len(mem.query("SELECT * FROM claims")) == 0
        assert len(mem.list_experiences(limit=10000)) >= e0
    finally:
        mem.close()


def test_dialog_command_wired_in_session(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        assert sess.last_answer is None
        out = sess.handle("对话 你好")
        assert any("dialog" in l for l in out), out
        out = sess.handle("对话 杯子有什么用途")
        assert any("drinking" in l or "饮用" in l for l in out), out
        assert sess.last_answer is not None
    finally:
        mem.close()


def test_no_llm_dependency_in_dialog_stack():
    for mod in ("dialog", "answers"):
        src = open(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "src", "cognitive_ai",
            f"{mod}.py")), encoding="utf-8").read().lower()
        for banned in ("openai", "anthropic", "transformers", "llm",
                       "http://", "https://", "import torch", "from torch"):
            assert banned not in src, f"{mod}.py mentions {banned}"
