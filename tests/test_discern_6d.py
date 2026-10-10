"""Phase 6D discernment units: resolve_question adjudication, A-not-A
frames, politeness handling, and the learned-override gate.

No calibrated probabilities anywhere: confidences are ordinal grades.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import phraselearn as PL
from cognitive_ai.cli import seed_from_json
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory
from cognitive_ai.semantics import parse_sentence, resolve_question

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def rq(text, lex=None):
    return resolve_question(
        parse_sentence(text, source="user", lexicon=lex or {}).to_dict(),
        lex or {})


def test_attr_interrogative_names_key():
    r = rq("杯子由什么制成")
    assert (r["kind"], r["key"]) == ("attr_q", "material"), r


def test_has_question_names_feature():
    r = rq("杯子有把手吗")
    assert (r["kind"], r["key"]) == ("has_q", "把手"), r
    r = rq("杯子有把手吗", {"把手": "has_handle"})
    assert (r["kind"], r["key"]) == ("has_q", "has_handle"), r


def test_ana_questions_are_queries_not_denials():
    for t, kind in [("杯子是不是陶瓷做的", "verify_attr"),
                    ("杯子是不是容器", "verify_class"),
                    ("杯子有没有把手", "verify_has"),
                    ("杯子能不能装水", "verify_capable"),
                    ("杯子是陶瓷的吗", "verify_attr")]:
        f = parse_sentence(t, source="user").to_dict()
        assert f["intent"] == "query", (t, f["intent"])
        assert not any(p.get("denied") for p in f["predicates"]), t
        assert rq(t)["kind"] == kind, (t, rq(t))


def test_true_denial_still_negate():
    f = parse_sentence("杯子不是玻璃做的", source="user").to_dict()
    assert f["intent"] == "negate"
    assert rq("杯子不是玻璃做的")["kind"] == "unsupported_q"


def test_politeness_keeps_entity_and_question():
    f = parse_sentence("请问杯子的材质是什么",
                       source="user").to_dict()
    assert any("politeness stripped" in t for t in f["trace"]), f["trace"]
    assert f["entities"] and f["entities"][0]["text"] == "杯子的材质", f
    r = rq("请问杯子的材质是什么")
    assert r["kind"] == "classify_q", r


def test_assertions_are_not_questions():
    assert rq("杯子是陶瓷的")["kind"] == "unsupported_q"
    assert rq("杯子有把手")["kind"] == "unsupported_q"


def test_override_never_touches_confident_parse(tmp_path):
    db = str(tmp_path / "o.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    try:
        s = Session(mem, str(tmp_path))
        s.handle("说法 杯子有什么用途 => 碗的材质是什么")
        s.handle("说法归纳")
        rid = mem.list_phrase_rules()[0]["id"]
        mem.list_phrase_examples(heldout=True)  # none yet
        s.handle("说法 碗的颜色是什么 => 碗的材质是什么 --留存")
        s.handle(f"说法验证 {rid}")
        row = mem.get_phrase_rule(rid)
        row["status"] = "validated"
        mem.save_phrase_rule(row)
        assert PL.activate_rule(mem, rid)["ok"]
        direct = parse_sentence("杯子有什么用途", source="user").to_dict()
        assert direct["ambiguities"] == []
        assert PL.consider_learned_override(
            direct, "杯子有什么用途",
            mem.list_phrase_rules("active"), {}) is None
    finally:
        mem.close()


def test_override_rescues_doubtful_parse(tmp_path):
    db = str(tmp_path / "o2.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    try:
        s = Session(mem, str(tmp_path))
        src, dst = "杯子是用来喝热水的", "杯子可以喝热水"
        # single example -> literal rule; lifecycle promoted manually
        # here (induce/validate flow itself is covered in 6C tests)
        s.handle(f"说法 {src} => {dst}")
        s.handle("说法归纳")
        rid = mem.list_phrase_rules()[0]["id"]
        assert mem.list_phrase_rules()[0]["kind"] == "literal"
        row = mem.get_phrase_rule(rid)
        row["status"] = "validated"
        mem.save_phrase_rule(row)
        assert PL.activate_rule(mem, rid)["ok"]
        direct = parse_sentence(src, source="user").to_dict()
        assert direct["ambiguities"], "direct parse must admit doubt"
        alt = PL.consider_learned_override(
            direct, src, mem.list_phrase_rules("active"), {})
        assert alt is not None and not alt["ambiguities"], alt
        assert any("phrase_override" in t for t in alt["trace"]), alt
        full = parse_sentence(src, source="user",
                               phrase_rules=mem.list_phrase_rules("active"))
        assert full.intent == "function", full.to_dict()
    finally:
        mem.close()


def test_override_abstains_when_rewrite_also_doubtful():
    direct = {"intent": "attribute", "entities": [{"text": "杯子"}],
              "predicates": [{"type": "attr", "subj": "杯子",
                              "key": None, "value": "用来喝热水"}],
              "ambiguities": ["attribute key unclear"], "trace": []}
    rule = {"id": "pr_x", "kind": "literal", "status": "active",
            "src_tmpl": {"text": "杯子是用来喝热水的"},
            "dst_tmpl": {"text": "杯子是用来喝冷水的"},
            "signature": {}}
    assert PL.consider_learned_override(direct, "杯子是用来喝热水的",
                                        [rule], {}) is None
