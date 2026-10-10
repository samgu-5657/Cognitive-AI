"""Phase 6D independent evaluation (Task 5): isolated battery with
train/test separation, near-miss intent pairs, politeness, negation
pairs, ambiguity, and unknowns. Reports frame/intent accuracy,
misroute rate, abstention, learning delta, and regressions.

Small-sample disclaimer: ~20 probes prove THESE cases, not general
NLU. Battery sentences are audited against src (no hardcodes).
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

# (text, expected_kind_or_intent, must_contain_in_answer_or_None)
# kind "Q:<resolve_kind>" checks adjudication; "I:<intent>" checks parse;
# "R:<route>" checks dialogue routing; answer checks are substring match
# ("*" skips), certainty checked separately where noted.
BATTERY = [
    # supported typical
    ("杯子有什么用途", "Q:function_q", "drinking"),
    ("杯子的材质是什么", "Q:classify_q", "ceramic"),
    # near-miss: similar surface, different intent (the 6D fixes)
    ("杯子由什么制成", "Q:attr_q", "ceramic"),
    ("杯子有把手吗", "Q:has_q", "*"),
    # negation vs A-not-A pairs
    ("杯子不是玻璃做的", "I:negate", "*"),
    ("杯子是不是玻璃做的", "Q:verify_attr", "不是"),
    ("杯子是不是陶瓷做的", "Q:verify_attr", "是的"),
    ("杯子是陶瓷的吗", "Q:verify_attr", "是的"),
    ("杯子有没有把手", "Q:verify_has", "*"),
    # politeness
    ("请问杯子的材质是什么", "Q:classify_q", "ceramic"),
    ("请问杯子的用途", "R:unknown", "*"),
    # ambiguity admitted, zero writes
    ("杯子是用来喝热水的", "I:attribute", "*"),
    # unknowns abstain
    ("花钱如流水般哗啦", "R:unknown", "*"),
    ("噼里啪啦丁冬", "R:unknown", "*"),
]


def make_session(tmp_path, tag="e6d"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    return Session(mem, corp), mem


def run_battery(sess, mem):
    from cognitive_ai import dialog as D
    stats = {"n": 0, "frame": 0, "intent": 0, "misroute": 0,
             "abstain_ok": 0, "abstain_n": 0}
    fails = []
    for text, exp, want in BATTERY:
        stats["n"] += 1
        fr = parse_sentence(text, source="user",
                            lexicon=sess._lexicon(),
                            phrase_rules=sess._phrase_rules()).to_dict()
        turn = D.process(sess, text)
        ans = (turn.get("answer") or {}).get("text", "")
        if exp.startswith("Q:"):
            got = resolve_question(fr, sess._lexicon())["kind"]
            ok = got == exp[2:]
            stats["frame"] += ok
            if not ok:
                fails.append((text, exp, got))
                stats["misroute"] += 1
        elif exp.startswith("I:"):
            ok = fr["intent"] == exp[2:]
            stats["intent"] += ok
            if not ok:
                fails.append((text, exp, fr["intent"]))
        else:
            ok = turn["route"] == exp[2:]
            stats["intent"] += ok
            if not ok:
                fails.append((text, exp, turn["route"]))
        if want != "*" and want not in ans:
            fails.append((text, "answer-missing:" + want, ans[:40]))
        if turn["route"] == "unknown":
            stats["abstain_n"] += 1
            if (turn["answer"] or {}).get("certainty") == "unknown":
                stats["abstain_ok"] += 1
    return stats, fails


def test_eval_battery(tmp_path, capsys):
    sess, mem = make_session(tmp_path)
    try:
        c0 = len(mem.list_concepts())
        stats, fails = run_battery(sess, mem)
        with capsys.disabled():
            print(f"\n[6D EVAL] n={stats['n']} frame_ok={stats['frame']} "
                  f"intent_ok={stats['intent']} misroute={stats['misroute']} "
                  f"abstain={stats['abstain_ok']}/{stats['abstain_n']} "
                  f"fails={fails}")
        assert stats["misroute"] == 0, fails
        assert stats["abstain_ok"] == stats["abstain_n"] == 3, stats
        # ambiguity + unknowns wrote no concepts
        assert len(mem.list_concepts()) == c0
        assert stats["frame"] >= 9, fails
    finally:
        mem.close()


def test_learning_delta_intact(tmp_path):
    """6C mechanism still works after 6D changes (delta, not removal)."""
    sess, mem = make_session(tmp_path, tag="e6d2")
    try:
        from cognitive_ai import dialog as D
        assert D.process(sess, "告诉我杯子的用途")["route"] == "unknown"
        sess.handle("说法 告诉我碗的用途 => 碗有什么用途")
        sess.handle("说法 告诉我瓶子的用途 => 瓶子有什么用途")
        sess.handle("说法 告诉我杯子的用途 => 杯子有什么用途 --留存")
        sess.handle("说法归纳")
        for r in mem.list_phrase_rules():
            sess.handle(f"说法验证 {r['id']}")
            sess.handle(f"说法激活 {r['id']}")
        turn = D.process(sess, "告诉我杯子的用途")
        assert turn["route"] == "qa" and "drinking" in turn["answer"]["text"]
    finally:
        mem.close()


def test_web_and_cli_entries(tmp_path):
    import shutil
    from fastapi.testclient import TestClient
    from cognitive_ai.web.server import create_app
    data = tmp_path / "vdata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    app = create_app(str(data), str(tmp_path / "vmodels"))
    with TestClient(app) as c:
        r = c.post("/api/chat", json={"text": "杯子是不是陶瓷做的",
                                      "session_id": "v"}).json()
        assert r["ok"] and "是的" in "\n".join(r["data"]["lines"]), r
        r = c.post("/api/chat", json={"text": "杯子由什么制成",
                                      "session_id": "v"}).json()
        assert r["ok"] and "ceramic" in "\n".join(r["data"]["lines"]), r
        r = c.post("/api/chat", json={"text": "杯子不是玻璃做的",
                                      "session_id": "v"}).json()
        assert r["ok"] and "negate" in "\n".join(r["data"]["lines"]).lower() \
            or "否定" in "\n".join(r["data"]["lines"]), r
    app.state.wb.close()
    sess, mem = make_session(tmp_path, tag="e6d3")
    try:
        out = sess.handle("对话 杯子是不是玻璃做的")
        assert any("不是" in l for l in out), out
    finally:
        mem.close()


def test_battery_not_hardcoded_in_src():
    blobs = []
    for f in ("semantics.py", "phraselearn.py", "dialog.py", "executor.py",
              "explainer.py", "cog_planners.py"):
        blobs.append(open(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "src", "cognitive_ai", f)),
            encoding="utf-8").read())
    texts = [t for t, _, _ in BATTERY]
    leaked = [s for s in texts if len(s) >= 6
              and any(s in b for b in blobs)]
    assert not leaked, f"battery leaked into source: {leaked}"
