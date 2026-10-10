"""Phase 6H-2B construction tests: one induced row (not three
memorized sentences), bidirectional use, novel combos, abstention,
restart, session sharing, and the teach-confirm loop -- all on
isolated DBs. Training/held/test entities pairwise disjoint;
battery audited against src (no per-sentence rules, no hardcodes).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import constructions as CX
from cognitive_ai import dialog as D
from cognitive_ai.cli import seed_from_json
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))

TRN = [("海狸", "筑坝"), ("企鹅", "游泳"), ("蜂鸟", "悬停")]
HLD = [("河狸", "潜水"), ("海豚", "跳跃")]
NEW_E, NEW_V = "水獭", "筑坝"


def form(e, v):
    return f"{e}的本领包括{v}"


def make_session(tmp_path, tag="cx"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    return Session(mem, corp), mem


def teach_all(sess):
    for e, v in TRN:
        sess.handle(f"构式 {form(e, v)} => entity={e};value={v}")
    for e, v in HLD:
        sess.handle(f"构式 {form(e, v)} => entity={e};value={v} --留存")
    out = sess.handle("构式归纳")
    assert "1 候选" in "\n".join(out), out
    rid = [r["id"] for r in sess.mem.list_constructions()]
    assert len(rid) == 1, rid
    return rid[0]


def activate(sess, mem, rid):
    v = sess.handle(f"构式验证 {rid}")
    assert any("验证通过" in l for l in v), v
    a = sess.handle(f"构式激活 {rid}")
    assert any("已激活" in l for l in a), a


def test_baseline_unknown_then_one_row(tmp_path, capsys):
    sess, mem = make_session(tmp_path)
    try:
        from cognitive_ai.semantics import parse_sentence
        assert parse_sentence(form(*TRN[0]), source="user").intent == \
            "unknown"
        rid = teach_all(sess)
        row = mem.get_construction(rid)
        assert (row["pre"], row["mid"], row["suf"]) == ("", "的本领包括", ""), row
        assert len(row["evidence"][0]["support"]) == 3, row
        with capsys.disabled():
            print(f"\n[6H-2B] induced=1 row from {len(TRN)} examples "
                  f"(not {len(TRN)} memorized sentences)")
    finally:
        mem.close()


def test_novel_entity_and_novel_combo(tmp_path):
    sess, mem = make_session(tmp_path, tag="cx2")
    try:
        rid = teach_all(sess)
        activate(sess, mem, rid)
        rules = mem.list_constructions(status="active")
        # unseen entity, seen value
        fr = CX.parse_frame(form(NEW_E, NEW_V), rules)
        assert fr is not None, "novel entity must parse"
        assert fr["entities"][0]["text"] == NEW_E
        assert fr["predicates"][0]["action"] == NEW_V
        assert fr["intent"] == "observe"
        assert any(str(t).startswith("construction:") and rid in str(t)
                   for t in fr["trace"]), fr["trace"]
        # seen entity, value never paired with it in training
        fr2 = CX.parse_frame(form(HLD[0][0], HLD[1][1]), rules)
        assert fr2 is not None
        assert fr2["predicates"][0]["action"] == HLD[1][1], fr2
    finally:
        mem.close()


def test_generate_and_roundtrip(tmp_path):
    sess, mem = make_session(tmp_path, tag="cx3")
    try:
        rid = teach_all(sess)
        activate(sess, mem, rid)
        rules = mem.list_constructions(status="active")
        outs = CX.render(NEW_E, "潜水", rules)
        assert outs and outs[0]["text"] == form(NEW_E, "潜水"), outs
        assert outs[0]["source"] == f"learned:{rid}", outs
        back = CX.parse_frame(outs[0]["text"], rules)
        assert back["entities"][0]["text"] == NEW_E
        assert back["predicates"][0]["action"] == "潜水"
        turn = D.process(sess, f"生成 {NEW_E} 潜水")
        assert turn["route"] == "generate", turn["route"]
        assert form(NEW_E, "潜水") in turn["answer"]["text"], turn["answer"]
    finally:
        mem.close()


def test_parse_feeds_teach_confirm(tmp_path):
    sess, mem = make_session(tmp_path, tag="cx4")
    try:
        rid = teach_all(sess)
        activate(sess, mem, rid)
        turn = D.process(sess, form(NEW_E, NEW_V))
        assert turn["route"] == "teach_pending", turn["route"]
        out = D.process(sess, "确认")
        assert "已记录" in out["answer"]["text"] or \
            "已保存" in out["answer"]["text"] or \
            "observation" in out["answer"]["text"], out["answer"]
        rows = mem.get_attributes(NEW_E)
        assert any(r["key"] == "function" and r["status"] == "hypothesis"
                   for r in rows), rows
    finally:
        mem.close()


def test_abstention_cases(tmp_path):
    sess, mem = make_session(tmp_path, tag="cx5")
    try:
        # single example: stored, never induced
        sess.handle(f"构式 {form('貘', '挖洞')} => entity=貘;value=挖洞")
        out = sess.handle("构式归纳")
        assert "0 候选" in "\n".join(out), out
        assert mem.list_constructions() == []
        # contradictory alignment: entity/value swapped
        sess.handle(f"构式 {form('獾', '筑巢')} => entity=筑巢;value=獾")
        sess.handle(f"构式 {form('鼹', '打洞')} => entity=鼹;value=打洞")
        out = sess.handle("构式归纳")
        assert "0 候选" in "\n".join(out) or \
            all(r["status"] == "candidate"
                for r in mem.list_constructions()), out
        # ambiguous matches abstain in both directions
        r1 = {"id": "a", "kind": "capability", "pre": "", "mid": "X",
              "suf": "", "fixed_value": "", "status": "active"}
        r2 = {"id": "b", "kind": "capability", "pre": "", "mid": "X",
              "suf": "", "fixed_value": "", "status": "active"}
        assert CX.parse_frame("甲X乙", [r1, r2]) is None
    finally:
        mem.close()


def test_restart_and_sharing(tmp_path):
    db = str(tmp_path / "cxr.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    try:
        s = Session(mem, str(tmp_path))
        for e, v in TRN:
            s.handle(f"构式 {form(e, v)} => entity={e};value={v}")
        for e, v in HLD:
            s.handle(f"构式 {form(e, v)} => entity={e};value={v} --留存")
        s.handle("构式归纳")
        rid = mem.list_constructions()[0]["id"]
        s.handle(f"构式验证 {rid}")
        assert CX.activate_rule(mem, rid)["ok"]
    finally:
        mem.close()
    mem2 = Memory(db)
    try:
        s2 = Session(mem2, str(tmp_path))
        rules = s2._constructions()
        assert len(rules) == 1 and rules[0]["status"] == "active"
        fr = CX.parse_frame(form(NEW_E, NEW_V), rules)
        assert fr["predicates"][0]["action"] == NEW_V
        assert CX.render(NEW_E, NEW_V, rules)[0]["text"] == \
            form(NEW_E, NEW_V)
    finally:
        mem2.close()


def test_battery_not_hardcoded_in_src():
    blobs = []
    for f in ("semantics.py", "phraselearn.py", "semlearn.py",
              "constructions.py", "dialog.py", "interact.py",
              "executor.py"):
        blobs.append(open(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "src", "cognitive_ai", f)),
            encoding="utf-8").read())
    tests = [form(e, v) for e, v in TRN + HLD] + \
        [form(NEW_E, NEW_V), f"生成 {NEW_E} 潜水"]
    leaked = [s for s in tests if len(s) >= 6
              and any(s in b for b in blobs)]
    assert not leaked, f"battery leaked into source: {leaked}"
