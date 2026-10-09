"""Phase 4B tests: compositional semantics, context, NL teaching loop.

Isolated tmp DBs. Rule-vs-net comparison uses the frozen semantics sets.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.cli import seed_from_json
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory
from cognitive_ai.semantics import (parse_sentence, resolve_frame,
                                    update_context)

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))
SEMDATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "data", "semantics"))


def make_session(tmp_path, tag="n"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    return Session(mem, corp, models_root=str(tmp_path / f"m_{tag}"),
                   queue_path=str(tmp_path / f"q_{tag}.jsonl")), mem


def test_paraphrases_share_frame():
    pairs = [
        ("紫砂壶是一种茶壶", "紫砂壶属于茶壶", "classify"),
        ("杯子是玻璃做的", "杯子由玻璃制成", "attribute"),
        ("茶壶可以用来泡茶", "茶壶能泡茶", "function"),
        ("这个壶不是金属做的", "这个壶并非金属做的", "negate"),
        ("这可能是玻璃杯", "这或许是玻璃杯", "hedge"),
        ("如果下雨，地面就会变湿", "要是下雨，地面就会湿", "causal"),
        ("我今天买了一个新杯子", "我刚买了个新杯子", "observe"),
    ]
    for a, b, want in pairs:
        fa, fb = parse_sentence(a), parse_sentence(b)
        assert fa.intent == want, (a, fa.intent)
        assert fb.intent == want, (b, fb.intent)


def test_modality_not_confused():
    f_assert = parse_sentence("紫砂壶是陶土做的")
    f_neg = parse_sentence("这个壶不是金属做的")
    f_hedge = parse_sentence("我觉得这个壶可能是陶土做的")
    f_corr = parse_sentence("刚才说错了，它其实是不锈钢的")
    assert (f_assert.intent, f_assert.modality) == ("attribute", "assert")
    assert (f_neg.intent, f_neg.modality) == ("negate", "negate")
    assert f_neg.predicates[0].get("denied") is True
    assert (f_hedge.intent, f_hedge.modality) == ("hedge", "possible")
    assert (f_corr.intent, f_corr.modality) == ("correct", "correction")
    inner = f_corr.predicates[0]["content"]
    assert inner["modality"] == "assert"  # 不锈钢 affirmed, not negated


def test_correction_traces_antecedent(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        sess.handle("教 茶壶")
        sess.handle("材质=陶瓷")
        sess.handle("泡茶")
        sess.handle("容器")
        out = sess.handle("理解 刚才说错了，它其实是不锈钢的")
        assert any("correct" in l for l in out)
        out = sess.handle("确认")
        assert any("correction" in l.lower() or "纠正" in l for l in out)
        exps = [e for e in mem.list_experiences(limit=50)
                if "correction" in e["summary"].lower()
                or "correction" in str(e["detail"])]
        assert exps, "correction must leave a linked record"
    finally:
        mem.close()


def test_unresolved_pronoun_blocks_execution(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        # several candidates -> ambiguity; confirm must refuse, no 它-concept
        sess.handle("理解 我买了杯子")
        sess.handle("确认")
        sess.handle("理解 我买了茶壶")
        sess.handle("确认")
        out = sess.handle("理解 它是玻璃做的")
        assert any("歧义" in l or "澄清" in l or "哪" in l for l in out)
        v0 = mem.current_version()
        out = sess.handle("确认")
        assert any("拒绝" in l for l in out)
        assert mem.current_version() == v0
        assert not mem.find_concepts_by_name("它")
    finally:
        mem.close()


def test_pronoun_simple_and_ambiguous(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        out = sess.handle("理解 我今天买了一个新杯子")
        assert any("observe" in l for l in out)
        out = sess.handle("确认")
        out = sess.handle("理解 它有一个把手")
        joined = "\n".join(out)
        assert "杯" in joined  # resolved to 杯子, recorded
        # ambiguous: two entities, one pronoun
        sess.handle("清空上下文")
        st = mem.get_context(sess.session_id)
        st = update_context(st, parse_sentence("我买了杯子"))
        st = update_context(st, parse_sentence("我买了茶壶"))
        mem.set_context(sess.session_id, st)
        out = sess.handle("理解 它是红色的")
        assert any("澄清" in l or "clarif" in l.lower() or "歧义" in l
                   or "哪" in l for l in out)
    finally:
        mem.close()


def test_parse_failure_writes_nothing(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        before = (len(mem.list_concepts()),
                  len(mem.query("SELECT * FROM attributes")),
                  mem.current_version(),
                  len(mem.list_claims()))
        out = sess.handle("理解 花钱如流水")
        assert any("无法理解" in l for l in out)
        after = (len(mem.list_concepts()),
                 len(mem.query("SELECT * FROM attributes")),
                 mem.current_version(), len(mem.list_claims()))
        assert before == after
    finally:
        mem.close()


def test_nl_teaching_persists_across_restart(tmp_path):
    db = str(tmp_path / "nl.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    try:
        s1 = Session(mem, str(tmp_path), models_root=str(tmp_path / "m"),
                     queue_path=str(tmp_path / "q.jsonl"))
        assert any("attribute" in l for l in s1.handle("理解 碗是陶瓷做的"))
        s1.handle("确认")
        ctx = mem.get_context(s1.session_id)
        assert ctx["entities"], "context must persist in DB"
    finally:
        mem.close()
    mem2 = Memory(db)
    try:
        assert mem2.find_concepts_by_name("碗") or \
            [c for c in mem2.list_claims()]
        out = Session(mem2, str(tmp_path)).handle("查看 bowl")
        assert any("bowl" in l for l in out)
    finally:
        mem2.close()


def test_rule_vs_net_comparison_runs():
    from cognitive_ai.neural.intent import (check_group_isolation,
                                            compare_rule_vs_net, load_rows)
    tr = load_rows(os.path.join(SEMDATA, "train.jsonl"))
    va = load_rows(os.path.join(SEMDATA, "val.jsonl"))
    te = load_rows(os.path.join(SEMDATA, "test.jsonl"))
    assert check_group_isolation(("train", tr), ("val", va),
                                 ("test", te))["ok"]
    import glob
    ckpts = glob.glob(os.path.join(os.path.dirname(SEMDATA), "..",
                                   "models", "intent-v1", "best.pt"))
    if not ckpts:
        import pytest
        pytest.skip("intent checkpoint not trained on this machine")
    rep = compare_rule_vs_net(te, os.path.dirname(ckpts[0]))
    assert "_overall" in rep["per_intent"]
    assert rep["per_intent"]["_overall"]["n"] == len(te)


def test_context_persist_view_clear(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        sess.handle("理解 我买了杯子")
        out = sess.handle("上下文")
        assert any("杯子" in l for l in out)
        sess.handle("清空上下文")
        out = sess.handle("上下文")
        assert "entities=[]" in "".join(out)
        # sessions do not share context
        s2 = Session(mem, str(tmp_path))
        assert s2.mem.get_context(s2.session_id)["entities"] == []
    finally:
        mem.close()


def test_insufficient_verdict_not_novel(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        out = sess.handle("分类 杯子")
        joined = "\n".join(out)
        assert "规则归类" in joined
        # empty-attribute rule classify must not claim novelty
        res = sess.learner.classify({"name": "x", "attributes": {}})
        assert res["verdict"] == "insufficient" and res["best"] is None
    finally:
        mem.close()
