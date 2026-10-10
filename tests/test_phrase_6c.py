"""Phase 6C unit tests: phrase-example learning mechanics.

Isolated tmp DBs. Covers: example gating (dst must parse), slot vs
literal induction, held-out validation strictness, explicit activation,
fallback-only application, ambiguity abstention, restart persistence,
and a no-hardcode audit (test sentences must not leak into src/).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import phraselearn as PL
from cognitive_ai.cli import seed_from_json
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory
from cognitive_ai.semantics import parse_sentence

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))
SRC_FILES = [os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "src", "cognitive_ai", f))
    for f in ("semantics.py", "phraselearn.py", "dialog.py")]

# Sentences generated for THIS file's scenarios (also audited below).
S_USE_CUP = "告诉我杯子的用途"
S_USE_BOWL = "告诉我碗的用途"
S_USE_BOTTLE = "告诉我瓶子的用途"
S_USE_POT = "告诉我茶壶的用途"
S_MAT_BOWL = "请问碗的材质是什么"
S_MAT_BOTTLE = "请问瓶子的材质是什么"
ALL_SENT = [S_USE_CUP, S_USE_BOWL, S_USE_BOTTLE, S_USE_POT,
            S_MAT_BOWL, S_MAT_BOTTLE]


def make_session(tmp_path, tag="p6c"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    return Session(mem, corp), mem


def teach(sess, src, dst, held=False):
    flag = " --留存" if held else ""
    out = sess.handle(f"说法 {src} => {dst}{flag}")
    assert any("已记" in l for l in out), out


def test_dst_must_parse_or_example_rejected(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        out = sess.handle("说法 花钱如流水般哗啦 => 哗啦哗啦不知道")
        assert any("未收录" in l for l in out), out
        assert len(mem.list_phrase_examples()) == 0
    finally:
        mem.close()


def test_dst_out_of_scope_rejected(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        # dst parses as negate: outside the learning scope
        out = sess.handle("说法 杯子很结实 => 杯子不是纸做的")
        assert any("未收录" in l for l in out), out
    finally:
        mem.close()


def test_slot_induction_needs_two_examples(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        teach(sess, S_USE_BOWL, "碗有什么用途")
        out = sess.handle("说法归纳")
        assert any("literal" in l for l in out), out
        rows = mem.list_phrase_rules()
        assert len(rows) == 1 and rows[0]["kind"] == "literal"
    finally:
        mem.close()


def test_slot_rule_from_two_examples_and_validation(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        teach(sess, S_USE_BOWL, "碗有什么用途")
        teach(sess, S_USE_BOTTLE, "瓶子有什么用途")
        teach(sess, S_USE_CUP, "杯子有什么用途", held=True)
        out = sess.handle("说法归纳")
        assert any("[slot]" in l for l in out), out
        rid = mem.list_phrase_rules()[0]["id"]
        # not usable before validation+activation
        fr = parse_sentence(S_USE_POT, source="user",
                            phrase_rules=mem.list_phrase_rules("active"))
        assert fr.intent == "unknown"
        out = sess.handle(f"说法验证 {rid}")
        assert any("验证通过" in l for l in out), out
        out = sess.handle(f"说法激活 {rid}")
        assert any("已激活" in l for l in out), out
        fr = parse_sentence(S_USE_POT, source="user",
                            phrase_rules=mem.list_phrase_rules("active"))
        assert fr.intent == "query", fr.to_dict()
        assert any(p.get("type") == "query_function" for p in fr.predicates)
        assert any(f"phrase_rule:{rid}" in t for t in fr.trace)
    finally:
        mem.close()


def test_activation_requires_validation(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        teach(sess, S_USE_BOWL, "碗有什么用途")
        teach(sess, S_USE_BOTTLE, "瓶子有什么用途")
        sess.handle("说法归纳")
        rid = mem.list_phrase_rules()[0]["id"]
        out = sess.handle(f"说法激活 {rid}")
        assert any("失败" in l for l in out), out
    finally:
        mem.close()


def test_ambiguous_rules_abstain(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        teach(sess, "问杯子", "杯子有什么用途")
        teach(sess, "问杯子", "杯子的材质是什么")
        sess.handle("说法归纳")
        for r in mem.list_phrase_rules():
            r["status"] = "validated"
            mem.save_phrase_rule(r)
            assert PL.activate_rule(mem, r["id"])["ok"]
        out = PL.apply_rules("问杯子", mem.list_phrase_rules("active"), {})
        assert out["rewritten"] is None and "ambiguous" in out["reason"]
    finally:
        mem.close()


def test_rules_persist_across_restart(tmp_path):
    db = str(tmp_path / "persist.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    try:
        s = Session(mem, str(tmp_path))
        teach(s, S_USE_BOWL, "碗有什么用途")
        teach(s, S_USE_BOTTLE, "瓶子有什么用途")
        teach(s, S_USE_CUP, "杯子有什么用途", held=True)
        s.handle("说法归纳")
        rid = mem.list_phrase_rules()[0]["id"]
        s.handle(f"说法验证 {rid}")
        assert PL.activate_rule(mem, rid)["ok"]
    finally:
        mem.close()
    mem2 = Memory(db)
    try:
        rules = mem2.list_phrase_rules(status="active")
        assert len(rules) == 1
        fr = parse_sentence(S_USE_POT, source="user", phrase_rules=rules)
        assert fr.intent == "query"
    finally:
        mem2.close()


def test_no_hardcoded_test_sentences_in_src():
    blobs = [open(p, encoding="utf-8").read() for p in SRC_FILES]
    leaked = [s for s in ALL_SENT if len(s) >= 6
              and any(s in b for b in blobs)]
    assert not leaked, f"test sentences leaked into parser source: {leaked}"
