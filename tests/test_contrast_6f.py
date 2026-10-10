"""Phase 6F contrast experiment: correct triple validates, mislabeled
rules are refused with reasons, cross-type confusion is measured.

Isolated DBs, fresh entities (disjoint from 6E batteries and from each
other across train/held/test), combinatorial probes audited vs src.
Honest scope: only mislabels contradicting ESTABLISHED vocabulary
(材质/用途/...) are detectable; form-consistent mislabels without
known words are reported as undecidable, never claimed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import semlearn as SL
from cognitive_ai.cli import seed_from_json
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory
from cognitive_ai.semantics import resolve_question

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))

TRN = ["狐狸", "浣熊"]
HLD = ["松鼠", "刺猬"]
TST = ["考拉", "袋鼠"]


def use_f(e):
    return f"请讲下{e}的用途"


def mat_f(e):
    return f"请讲下{e}的材质"


def has_f(e):
    return f"{e}的把手情况如何"


def lab(kind, e, key):
    return f"kind={kind};entity={e};key={key}"


def make_session(tmp_path, tag="k6f"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    return Session(mem, corp), mem


def teach_triple(sess, trn, hld):
    for e in trn:
        sess.handle(f"语义示例 {use_f(e)} => {lab('function_q', e, 'function')}")
        sess.handle(f"语义示例 {mat_f(e)} => {lab('attr_q', e, 'material')}")
        sess.handle(f"语义示例 {has_f(e)} => {lab('has_q', e, '把手')}")
    for e in hld:
        sess.handle(f"语义示例 {use_f(e)} => {lab('function_q', e, 'function')}"
                    " --留存")
        sess.handle(f"语义示例 {mat_f(e)} => {lab('attr_q', e, 'material')}"
                    " --留存")
        sess.handle(f"语义示例 {has_f(e)} => {lab('has_q', e, '把手')}"
                    " --留存")
    out = sess.handle("语义归纳")
    assert "3 候选" in "\n".join(out), out


def test_correct_triple_activates_and_stays_clean(tmp_path, capsys):
    sess, mem = make_session(tmp_path)
    try:
        teach_triple(sess, TRN, HLD)
        ok = 0
        for r in mem.list_sem_rules():
            v = sess.handle(f"语义验证 {r['id']}")
            if any("验证通过" in l for l in v):
                sess.handle(f"语义激活 {r['id']}")
                ok += 1
        assert ok == 3, [r["id"] for r in mem.list_sem_rules()]
        rules = mem.list_sem_rules(status="active")
        cross, abst = 0, 0
        for e in TST:
            probes = [(use_f(e), "function_q"), (mat_f(e), "attr_q"),
                      (has_f(e), "has_q")]
            for t, k in probes:
                fr = SL.construct(t, rules)
                assert fr is not None, t
                assert resolve_question(fr, {})["kind"] == k, t
                others = [r for r in rules if r["kind"] != k]
                if SL.construct(t, others) is not None:
                    cross += 1
        for g in ["花钱如流水般哗啦", "给介绍下考拉的用途"]:
            if SL.construct(g, rules) is None:
                abst += 1
        with capsys.disabled():
            print(f"\n[6F] activate={ok}/3 cross_fires={cross} "
                  f"abstain={abst}/2")
        assert cross == 0 and abst == 2
    finally:
        mem.close()
    mem2 = Memory(str(tmp_path / "k6f.db"))
    try:
        rows = mem2.list_sem_rules(status="active")
        assert len(rows) == 3
        fr = SL.construct(use_f("考拉"), rows)
        assert resolve_question(fr, {})["kind"] == "function_q"
    finally:
        mem2.close()


def test_mislabeled_material_as_function_refused(tmp_path, capsys):
    sess, mem = make_session(tmp_path, tag="k6f2")
    try:
        n0 = mem.query("SELECT COUNT(*) AS n FROM attributes")[0]["n"]
        # MAT forms consistently mislabeled as function questions
        for e in TRN:
            sess.handle(f"语义示例 {mat_f(e)} => "
                        f"{lab('function_q', e, 'function')}")
        for e in HLD:
            sess.handle(f"语义示例 {mat_f(e)} => "
                        f"{lab('function_q', e, 'function')} --留存")
        sess.handle("语义归纳")
        rows = mem.list_sem_rules()
        assert len(rows) == 1
        v = sess.handle(f"语义验证 {rows[0]['id']}")
        assert any("验证未通过" in l for l in v), v
        assert any("材质" in l for l in v), v
        assert mem.get_sem_rule(rows[0]["id"])["status"] == "candidate"
        a = sess.handle(f"语义激活 {rows[0]['id']}")
        assert any("失败" in l for l in a), a
        # refused rule never answers, KB untouched
        from cognitive_ai import dialog as D
        turn = D.process(sess, mat_f("杯子"))
        assert "drinking" not in turn["answer"]["text"], turn["answer"]
        assert mem.query("SELECT COUNT(*) AS n FROM attributes")[0]["n"] \
            == n0
        with capsys.disabled():
            print(f"\n[6F-mislabel] refused=1/1 "
                  f"reason={v[0][:60]}")
    finally:
        mem.close()
    mem2 = Memory(str(tmp_path / "k6f2.db"))
    try:
        assert mem2.get_sem_rule(rows[0]["id"])["status"] == "candidate"
    finally:
        mem2.close()


def test_mislabeled_use_as_material_refused(tmp_path):
    sess, mem = make_session(tmp_path, tag="k6f3")
    try:
        for e in TRN:
            sess.handle(f"语义示例 {use_f(e)} => "
                        f"{lab('attr_q', e, 'material')}")
        for e in HLD:
            sess.handle(f"语义示例 {use_f(e)} => "
                        f"{lab('attr_q', e, 'material')} --留存")
        sess.handle("语义归纳")
        rid = mem.list_sem_rules()[0]["id"]
        v = sess.handle(f"语义验证 {rid}")
        assert any("验证未通过" in l for l in v), v
        assert any("用途" in l for l in v), v
    finally:
        mem.close()


def test_web_uses_contrast_gates(tmp_path):
    import shutil

    import pytest
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from cognitive_ai.web.server import create_app
    data = tmp_path / "kdata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    app = create_app(str(data), str(tmp_path / "kmodels"))

    def chat(c, text, sid="kw"):
        r = c.post("/api/chat", json={"text": text,
                                      "session_id": sid}).json()
        assert r["ok"], r
        return r["data"]

    with TestClient(app) as c:
        chat(c, f"语义示例 {use_f('碗')} => {lab('function_q', '碗', 'function')}")
        chat(c, f"语义示例 {use_f('瓶子')} => {lab('function_q', '瓶子', 'function')}")
        chat(c, f"语义示例 {use_f('杯子')} => {lab('function_q', '杯子', 'function')}"
                " --留存")
        ind = chat(c, "语义归纳")
        rid = [l.split()[0] for l in ind["lines"]
               if l.strip().startswith("sr_")][0]
        assert any("验证通过" in l
                   for l in chat(c, f"语义验证 {rid}")["lines"])
        chat(c, f"语义激活 {rid}")
        d = chat(c, use_f("杯子"))
        assert d["route"] == "qa" and "drinking" in "\n".join(d["lines"]), d
    app.state.wb.close()


def test_battery_not_hardcoded_in_src():
    blobs = []
    for f in ("semantics.py", "phraselearn.py", "semlearn.py",
              "dialog.py", "interact.py", "executor.py"):
        blobs.append(open(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "src", "cognitive_ai", f)),
            encoding="utf-8").read())
    tests = ([use_f(e) for e in TST] + [mat_f(e) for e in TST] +
             [has_f(e) for e in TST])
    leaked = [s for s in tests if len(s) >= 6
              and any(s in b for b in blobs)]
    assert not leaked, f"battery leaked into source: {leaked}"
