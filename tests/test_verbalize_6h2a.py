"""Phase 6H-2A bidirectional tests: one learned regularity serves
parse (utt->roles) and generation (roles->utt).

Isolated DBs; fresh entities disjoint across train/held/test and from
6E/6F batteries; battery audited against src (no per-sentence rules,
no answer hardcodes). Roundtrip asserts roles, not strings.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from cognitive_ai import dialog as D
from cognitive_ai import verbalize as VB
from cognitive_ai.cli import seed_from_json
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory
from cognitive_ai.web.server import create_app

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))

TRN = ["狐狸", "浣熊"]
HLD = ["松鼠", "刺猬"]
TST = ["考拉", "袋鼠"]


def use_f(e):
    return f"请讲下{e}的用途"


def mat_f(e):
    return f"请讲下{e}的材质"


def make_session(tmp_path, tag="v6"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    return Session(mem, corp), mem


def teach_all(sess):
    for e in TRN:
        sess.handle(f"语义示例 {use_f(e)} => kind=function_q;entity={e};key=function")
        sess.handle(f"语义示例 {mat_f(e)} => kind=attr_q;entity={e};key=material")
    for e in HLD:
        sess.handle(f"语义示例 {use_f(e)} => kind=function_q;entity={e};key=function --留存")
        sess.handle(f"语义示例 {mat_f(e)} => kind=attr_q;entity={e};key=material --留存")
    out = sess.handle("语义归纳")
    assert "2 候选" in "\n".join(out), out
    for r in sess.mem.list_sem_rules():
        v = sess.handle(f"语义验证 {r['id']}")
        assert any("验证通过" in l for l in v), v
        a = sess.handle(f"语义激活 {r['id']}")
        assert any("已激活" in l for l in a), a


def test_learned_surface_for_unseen_entity(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        teach_all(sess)
        turn = D.process(sess, "复述 考拉的材质是什么")
        assert turn["route"] == "paraphrase", turn["route"]
        txt = turn["answer"]["text"]
        assert "请讲下考拉的材质" in txt, txt
        assert "learned:" in txt and "fixed" in txt, txt
    finally:
        mem.close()


def test_kinds_do_not_confuse(tmp_path):
    sess, mem = make_session(tmp_path, tag="v62")
    try:
        teach_all(sess)
        t = D.process(sess, "复述 考拉有什么用途")
        txt = t["answer"]["text"]
        assert "请讲下考拉的用途" in txt, txt
        assert "材质" not in txt, txt
    finally:
        mem.close()


def test_roundtrip_keeps_roles(tmp_path):
    sess, mem = make_session(tmp_path, tag="v63")
    try:
        teach_all(sess)
        rules = sess._sem_rules()
        for kind, ent, key in [("attr_q", "考拉", "material"),
                               ("function_q", "袋鼠", "function")]:
            for o in VB.render_question(kind, ent, key, rules):
                back = VB.roundtrip_kind(o["text"], sess._lexicon(), rules)
                assert (back["kind"], back["key"], back["entity"]) == \
                    (kind, key, ent), (o, back)
    finally:
        mem.close()


def test_fact_status_marking_and_abstention():
    f = VB.render_fact("杯子", "material", "ceramic", "verified")
    assert "已验证" in f["text"] and "可能" not in f["text"]
    f = VB.render_fact("杯子", "has_handle", True, "hypothesis")
    assert "可能是" in f["text"] and "已验证" not in f["text"], f
    assert VB.render_fact("杯子", "mystery", None, "unknown") is None
    assert VB.render_question("attr_q", "杯子", "mystery", []) == []


def test_unknown_input_clarifies(tmp_path):
    sess, mem = make_session(tmp_path, tag="v64")
    try:
        turn = D.process(sess, "复述 花钱如流水")
        assert turn["route"] in ("paraphrase", "unknown", "clarify")
        assert "drinking" not in turn["answer"]["text"]
        assert "ceramic" not in turn["answer"]["text"]
    finally:
        mem.close()


def test_restart_keeps_generation(tmp_path):
    db = str(tmp_path / "v.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    try:
        s = Session(mem, str(tmp_path))
        teach_all(s)
    finally:
        mem.close()
    mem2 = Memory(db)
    try:
        s2 = Session(mem2, str(tmp_path))
        turn = D.process(s2, "复述 袋鼠有什么用途")
        assert "请讲下袋鼠的用途" in turn["answer"]["text"], turn["answer"]
    finally:
        mem2.close()


def test_web_paraphrase(tmp_path):
    import shutil
    data = tmp_path / "vdata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    app = create_app(str(data), str(tmp_path / "vmodels"))
    with TestClient(app) as c:
        sid = "vp"
        c.post("/api/chat", json={"text": f"语义示例 {use_f('碗')} => kind=function_q;entity=碗;key=function", "session_id": sid})
        c.post("/api/chat", json={"text": f"语义示例 {use_f('瓶子')} => kind=function_q;entity=瓶子;key=function", "session_id": sid})
        c.post("/api/chat", json={"text": f"语义示例 {use_f('杯子')} => kind=function_q;entity=杯子;key=function --留存", "session_id": sid})
        ind = c.post("/api/chat", json={"text": "语义归纳", "session_id": sid}).json()
        rid = [l.split()[0] for l in ind["data"]["lines"] if l.strip().startswith("sr_")][0]
        c.post("/api/chat", json={"text": f"语义验证 {rid}", "session_id": sid})
        c.post("/api/chat", json={"text": f"语义激活 {rid}", "session_id": sid})
        r = c.post("/api/chat", json={"text": "复述 茶杯有什么用途", "session_id": sid}).json()
        assert r["ok"] and "请讲下茶杯的用途" in "\n".join(r["data"]["lines"]), r
    app.state.wb.close()


def test_battery_not_hardcoded_in_src():
    blobs = []
    for f in ("semantics.py", "phraselearn.py", "semlearn.py",
              "verbalize.py", "dialog.py", "interact.py", "executor.py"):
        blobs.append(open(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "src", "cognitive_ai", f)),
            encoding="utf-8").read())
    tests = ([f"请讲下{e}的用途" for e in TST] +
             [f"请讲下{e}的材质" for e in TST] +
             ["复述 考拉的材质是什么", "复述 茶杯有什么用途"])
    leaked = [s for s in tests if len(s) >= 6
              and any(s in b for b in blobs)]
    assert not leaked, f"battery leaked into source: {leaked}"
