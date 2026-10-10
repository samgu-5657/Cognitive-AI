"""Phase 6E direct-frame experiment: semlearn constructs frames
field-by-field (never rewrite-and-reparse). Isolated DBs, dynamic
teaching, pairwise-disjoint train/held/test entities, combinatorial
battery audited against src.

Proves: three kinds construct correctly, types do not confuse, unknown
and ambiguous inputs abstain, restart persists, trace names the new
learner, baseline returns without rules, mislabels never reach the KB,
and the old phrase-rewrite experiment still passes (separate suite).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import semlearn as SL
from cognitive_ai.cli import seed_from_json
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory
from cognitive_ai.semantics import parse_sentence, resolve_question

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def use_f(e):
    return f"请讲下{e}的用途"


def mat_f(e):
    return f"请讲下{e}的材质"


def has_f(e):
    return f"{e}的把手情况如何"


USE_T, MAT_T, HAS_T = ["碗", "瓶子"], ["杯子", "水壶"], ["碗", "瓶子"]
USE_H, MAT_H, HAS_H = ["杯子", "水壶"], ["碗", "瓶子"], ["杯子", "水壶"]
TEST_E = ["茶壶", "茶杯"]

LAB_USE = "kind=function_q;entity={e};key=function"
LAB_MAT = "kind=attr_q;entity={e};key=material"
LAB_HAS = "kind=has_q;entity={e};key=把手"


def make_session(tmp_path, tag="s6e"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    return Session(mem, corp), mem


def teach_all(sess):
    for e in USE_T:
        sess.handle(f"语义示例 {use_f(e)} => {LAB_USE.format(e=e)}")
    for e in MAT_T:
        sess.handle(f"语义示例 {mat_f(e)} => {LAB_MAT.format(e=e)}")
    for e in HAS_T:
        sess.handle(f"语义示例 {has_f(e)} => {LAB_HAS.format(e=e)}")
    for e in USE_H:
        sess.handle(f"语义示例 {use_f(e)} => {LAB_USE.format(e=e)} --留存")
    for e in MAT_H:
        sess.handle(f"语义示例 {mat_f(e)} => {LAB_MAT.format(e=e)} --留存")
    for e in HAS_H:
        sess.handle(f"语义示例 {has_f(e)} => {LAB_HAS.format(e=e)} --留存")
    out = sess.handle("语义归纳")
    assert "3 候选" in "\n".join(out), out


def activate_all(sess, mem):
    for r in mem.list_sem_rules():
        v = sess.handle(f"语义验证 {r['id']}")
        assert any("验证通过" in l for l in v), v
        a = sess.handle(f"语义激活 {r['id']}")
        assert any("已激活" in l for l in a), a


def test_single_example_claims_nothing(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        sess.handle(f"语义示例 {use_f('碗')} => {LAB_USE.format(e='碗')}")
        out = sess.handle("语义归纳")
        assert "0 候选" in "\n".join(out), out
        assert mem.list_sem_rules() == []
    finally:
        mem.close()


def test_bad_labels_rejected(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        out = sess.handle("语义示例 杯子好用 => kind=nope;entity=杯子;key=x")
        assert any("未收录" in l for l in out), out
        out = sess.handle("语义示例 杯子好用 => kind=function_q;entity=杯子")
        assert any("格式不对" in l or "未收录" in l for l in out), out
        out = sess.handle("语义示例 杯子好用 => kind=function_q;"
                          "entity=杯子;key=material;extra=1")
        assert any("格式不对" in l or "未收录" in l for l in out), out
        assert mem.list_sem_examples() == []
    finally:
        mem.close()


def test_direct_frames_and_matrix_with_restart(tmp_path, capsys):
    sess, mem = make_session(tmp_path)
    try:
        # baseline: all test probes unknown to the fixed parser
        probes = ([(use_f(e), "function_q", "function") for e in TEST_E] +
                  [(mat_f(e), "attr_q", "material") for e in TEST_E] +
                  [(has_f(e), "has_q", "把手") for e in TEST_E])
        base = sum(1 for t, _, _ in probes
                   if parse_sentence(t, source="user").intent == "unknown")
        assert base == len(probes), "battery must start unknown"
        teach_all(sess)
        activate_all(sess, mem)
        rules = mem.list_sem_rules(status="active")
        assert len(rules) == 3
        kinds = {(r["kind"], r["key"]) for r in rules}
        assert kinds == {("function_q", "function"),
                         ("attr_q", "material"), ("has_q", "把手")}, kinds
        # direct construction: kind/key/entity exact, trace names semlearn
        hits, traced = 0, 0
        for t, k, key in probes:
            fr = SL.construct(t, rules)
            assert fr is not None, t
            rq = resolve_question(fr, {})
            if rq.get("kind") == k and rq.get("key") == key and \
                    fr["entities"][0].get("text") in t:
                hits += 1
            if any(str(x).startswith("learned_semantic_frame")
                   for x in fr.get("trace", [])):
                traced += 1
        # confusion: single-kind rule lists stay silent cross-type
        conf = 0
        for t, k, _ in probes:
            others = [r for r in rules if r["kind"] != k]
            if SL.construct(t, others) is not None:
                conf += 1
        # ambiguity: two same-affix kinds -> abstain
        amb = SL.construct(
            use_f("茶壶"),
            rules + [{"id": "sr_fake", "kind": "attr_q", "key": "material",
                      "pre": "请讲下", "suf": "的用途", "status": "active"}])
        # unknown: brand-new affix abstains
        unk = SL.construct("给介绍下茶壶的用途", rules)
        # reason probes untouched by sem rules
        rsn = sum(1 for t in ["为什么茶壶会漏水", "为什么人们会使用茶杯"]
                  if SL.construct(t, rules) is None)
        with capsys.disabled():
            print(f"\n[6E-direct] base_unknown={base}/{len(probes)} "
                  f"construct={hits}/{len(probes)} traced={traced} "
                  f"confusion={conf} amb_abstain={amb is None} "
                  f"unknown_abstain={unk is None} reason_kept={rsn}/2")
        assert hits == len(probes) and traced == len(probes)
        assert conf == 0 and amb is None and unk is None and rsn == 2
    finally:
        mem.close()
    mem2 = Memory(str(tmp_path / "s6e.db"))
    try:
        rules = mem2.list_sem_rules(status="active")
        assert len(rules) == 3
        for t, k, key in ([(use_f(e), "function_q", "function")
                           for e in TEST_E] +
                          [(mat_f(e), "attr_q", "material")
                           for e in TEST_E]):
            fr = SL.construct(t, rules)
            rq = resolve_question(fr, {})
            assert (rq.get("kind"), rq.get("key")) == (k, key), t
    finally:
        mem2.close()


def test_dialog_uses_constructed_frames(tmp_path):
    sess, mem = make_session(tmp_path, tag="s6e2")
    try:
        from cognitive_ai import dialog as D
        assert D.process(sess, use_f("杯子"))["route"] == "unknown"
        teach_all(sess)
        activate_all(sess, mem)
        turn = D.process(sess, use_f("杯子"))
        assert turn["route"] == "qa", turn["route"]
        assert "drinking" in turn["answer"]["text"], turn["answer"]
        assert turn["stages"].get("semlearn") is True, turn["stages"]
        turn = D.process(sess, mat_f("杯子"))
        assert "ceramic" in turn["answer"]["text"], turn["answer"]
        # semlearn never calls the rewrite path: no phrase trace
        assert not any("phrase_rule" in str(t)
                       for t in turn["stages"]["parse"].get("trace", []))
    finally:
        mem.close()


def test_mislabel_never_reaches_kb(tmp_path):
    sess, mem = make_session(tmp_path, tag="s6e3")
    try:
        from cognitive_ai import dialog as D
        n0 = mem.query("SELECT COUNT(*) AS n FROM attributes")[0]["n"]
        # consistently mislabeled: USE forms labeled as material questions
        for e in ["碗", "瓶子"]:
            sess.handle(f"语义示例 {use_f(e)} => "
                        f"kind=attr_q;entity={e};key=material")
        for e in ["杯子", "水壶"]:
            sess.handle(f"语义示例 {use_f(e)} => "
                        f"kind=attr_q;entity={e};key=material --留存")
        sess.handle("语义归纳")
        for r in mem.list_sem_rules():
            v = sess.handle(f"语义验证 {r['id']}")
            # 6F gate: the USE affix contradicts a material label, so
            # validation must refuse (not merely stay silent later)
            assert any("验证未通过" in l for l in v), v
            a = sess.handle(f"语义激活 {r['id']}")
            assert any("失败" in l for l in a), a
        D.process(sess, use_f("杯子"))
        assert mem.query("SELECT COUNT(*) AS n FROM attributes")[0]["n"] \
            == n0
        assert mem.query("SELECT * FROM attributes WHERE status='verified'"
                         " AND source NOT IN ('seed','validated')") == []
    finally:
        mem.close()


def test_battery_not_hardcoded_in_src():
    blobs = []
    for f in ("semantics.py", "phraselearn.py", "semlearn.py",
              "dialog.py", "interact.py", "executor.py"):
        blobs.append(open(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "src", "cognitive_ai", f)),
            encoding="utf-8").read())
    tests = ([use_f(e) for e in TEST_E] + [mat_f(e) for e in TEST_E] +
             [has_f(e) for e in TEST_E])
    leaked = [s for s in tests if len(s) >= 6
              and any(s in b for b in blobs)]
    assert not leaked, f"battery leaked into source: {leaked}"


def test_web_full_learn_loop(tmp_path):
    """End-to-end through /api/chat: teach -> induce -> validate ->
    activate -> unseen probe, all on the browser entry."""
    import shutil

    import pytest
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from cognitive_ai.web.server import create_app
    data = tmp_path / "sdata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    app = create_app(str(data), str(tmp_path / "smodels"))

    def chat(c, text, sid="sw"):
        r = c.post("/api/chat", json={"text": text,
                                      "session_id": sid}).json()
        assert r["ok"], r
        return r["data"]

    with TestClient(app) as c:
        assert chat(c, use_f("杯子"))["route"] == "unknown"
        chat(c, f"语义示例 {use_f('碗')} => {LAB_USE.format(e='碗')}")
        chat(c, f"语义示例 {use_f('瓶子')} => {LAB_USE.format(e='瓶子')}")
        chat(c, f"语义示例 {use_f('杯子')} => {LAB_USE.format(e='杯子')}"
                " --留存")
        ind = chat(c, "语义归纳")
        rid = [l.split()[0] for l in ind["lines"]
               if l.strip().startswith("sr_")]
        assert rid, ind
        val = chat(c, f"语义验证 {rid[0]}")
        assert any("验证通过" in l for l in val["lines"]), val
        act = chat(c, f"语义激活 {rid[0]}")
        assert any("已激活" in l for l in act["lines"]), act
        # unseen entity answers from the real KB through the web stack
        cup = chat(c, use_f("杯子"))
        assert cup["route"] == "qa", cup
        assert "drinking" in "\n".join(cup["lines"]), cup
        # unknown entity: honest gap, no fabrication
        pot = chat(c, use_f("茶壶"))
        assert pot["route"] == "qa", pot
        assert "drinking" not in "\n".join(pot["lines"]), pot
    app.state.wb.close()
