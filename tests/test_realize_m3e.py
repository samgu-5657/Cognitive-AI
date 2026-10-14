"""Phase M3E deterministic realization tests (isolated DBs, real ask).

Every test runs the actual ask entry (default Chinese, --json, --trace),
checks composed text against structured roles/status/evidence, verifies
roundtrips structurally, and diffs row counts before/after.
"""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.semground import answer_question
from cognitive_ai.semrealize import (build_answer_semantics, check_roundtrip,
                                     realize)

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def fresh(tmp_path, tag="sr"):
    m = Memory(str(tmp_path / f"{tag}.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    o.add_attribute("cup", "material", "ceramic", status="verified")
    o.add_attribute("cup", "function", "drinking", status="verified")
    return m


def counts(m):
    return (len(m.query("SELECT * FROM attributes")),
            len(m.list_claims()),
            len(m.query("SELECT * FROM relations")),
            len(m.query("SELECT * FROM concepts")))


def realize_of(text, mem):
    doc = answer_question(text, mem).to_dict()
    sem = build_answer_semantics(doc)
    return doc, sem, realize(sem)


def fresh_data(tmp_path, tag="srd"):
    data = tmp_path / tag
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    return data


def cli(data, *args):
    env = {**os.environ,
           "PYTHONPATH": os.path.abspath(os.path.join(
               os.path.dirname(__file__), "..", "src")),
           "COGNITIVE_AI_DATA": str(data)}
    r = subprocess.run(
        [sys.executable, "-m", "cognitive_ai.cli", *args],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."), env=env)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout


def test_supported_kinds(tmp_path):
    m = fresh(tmp_path)
    try:
        for text in ("杯子是什么？", "杯子的材质是什么？",
                     "杯子有什么用途？"):
            doc, sem, out = realize_of(text, m)
            assert doc["verdict"] == "supported", (text, doc)
            assert "杯子" in out["text"], (text, out)
            assert out["constructions"], out
    finally:
        m.close()


def test_new_entities_no_branches(tmp_path):
    m = fresh(tmp_path, tag="sr2")
    try:
        o = Ontology(m)
        o.add_concept("teapot", "茶壶", parents=["container"])
        o.add_attribute("teapot", "material", "clay", status="verified")
        doc, sem, out = realize_of("茶壶的材质是什么？", m)
        assert doc["verdict"] == "supported"
        assert "茶壶" in out["text"] and "clay" in out["text"], out
        doc, sem, out = realize_of("茶壶是不是容器？", m)
        assert doc["answer"] == "是" and out["text"].startswith("是的"), out
    finally:
        m.close()


def test_polarity_kept(tmp_path):
    m = fresh(tmp_path, tag="sr3")
    try:
        o = Ontology(m)
        o.add_concept("water", "水", parents=["entity"])
        _, _, out_yes = realize_of("杯子是不是容器？", m)
        assert out_yes["text"].startswith("是的"), out_yes
        doc_no, _, out_no = realize_of("杯子是水吗？", m)
        assert doc_no["verdict"] == "supported", doc_no
        assert out_no["text"].startswith("不，"), out_no
        _, _, out_deny = realize_of("杯子没有盖子吗？", m)
        assert "没有" in out_deny["text"] or "未知" in out_deny["text"] or \
            "不知道" in out_deny["text"], out_deny
    finally:
        m.close()


def test_hypothesis_marked(tmp_path):
    m = fresh(tmp_path, tag="sr4")
    try:
        o = Ontology(m)
        o.add_concept("bowl", "碗", parents=["container"])
        o.add_attribute("bowl", "material", "clay", status="hypothesis")
        doc, sem, out = realize_of("碗的材质是什么？", m)
        assert doc["verdict"] == "supported"
        assert "可能是" in out["text"] and "已验证" not in out["text"], out
    finally:
        m.close()


def test_states_differ(tmp_path):
    m = fresh(tmp_path, tag="sr5")
    try:
        o = Ontology(m)
        o.add_concept("water", "水", parents=["entity"])
        texts = {}
        for t in ("杯子的材质是什么？", "杯子有把手吗？",
                  "花钱如流水吗？", "杯子是水吗？"):
            doc, sem, out = realize_of(t, m)
            texts[t] = (doc["verdict"], out["text"])
        assert texts["杯子的材质是什么？"][0] == "supported"
        assert texts["杯子有把手吗？"][0] == "unknown"
        assert texts["花钱如流水吗？"][0] in ("unknown", "ambiguous")
        assert texts["杯子是水吗？"][0] == "supported"
        uniq = {v[1] for _, v in texts.items()}
        assert len(uniq) == 4, texts
    finally:
        m.close()


def test_derived_keeps_source(tmp_path):
    m = fresh(tmp_path, tag="sr6")
    try:
        o = Ontology(m)
        o.add_concept("potA", "铁锅", parents=["container"])
        m.upsert_rule({
            "id": "r_hold", "conditions": [{"type": "is_a",
                                            "concept": "?x",
                                            "ancestor": "container"}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "holds", "value": True},
            "exceptions": []})
        doc, sem, out = realize_of("铁锅有什么？", m)
        assert doc["verdict"] == "supported", doc
        assert any(e["kind"] == "rule" for e in doc["evidence"]), doc
        assert "规则推导" in out["text"], out
        assert "r_hold" in json.dumps(doc, ensure_ascii=False), doc
    finally:
        m.close()


def test_limited_vs_unknown(tmp_path):
    m = fresh(tmp_path, tag="sr7")
    try:
        import cognitive_ai.policy as policy
        o = Ontology(m)
        for i in range(3):
            o.add_concept(f"e{i}", f"实体{i}", parents=["entity"])
        m.upsert_rule({
            "id": "r_any", "conditions": [{"type": "is_a",
                                           "concept": "?x",
                                           "ancestor": "entity"}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "seen", "value": True},
            "exceptions": []})
        old = policy.MAX_CHAIN_WORK
        policy.MAX_CHAIN_WORK = 0
        try:
            doc, sem, out = realize_of("杯子有什么？", m)
        finally:
            policy.MAX_CHAIN_WORK = old
        assert "受限" in doc.get("detail", "") or \
            "受限" in json.dumps(doc.get("queries", []),
                                 ensure_ascii=False), doc
        plain = realize_of("杯子有什么？", m)[0]
        assert "受限" not in json.dumps(plain.get("queries", []),
                                        ensure_ascii=False), plain
        assert "受限" not in plain.get("detail", ""), plain
    finally:
        m.close()


def test_roundtrip_roles(tmp_path):
    m = Memory(str(tmp_path / "sr8.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    o.add_attribute("cup", "material", "ceramic", status="verified")
    o.add_attribute("cup", "function", "盛水", status="verified")
    o.add_attribute("cup", "lid", "盖子", status="verified")
    try:
        cases = [
            ("杯子是不是容器？",
             {"subject": "杯子", "predicate": "is_a", "object": "容器",
              "negated": False, "modal": None}),
            ("杯子可以盛水吗？",
             {"subject": "杯子", "predicate": "capable", "object": "盛水",
              "negated": False, "modal": "can"}),
            ("杯子有盖子吗？",
             {"subject": "杯子", "predicate": "has", "object": "盖子",
              "negated": False, "modal": None}),
        ]
        for text, expect in cases:
            doc, sem, out = realize_of(text, m)
            assert doc["verdict"] == "supported", (text, doc)
            assert out["roundtrippable"] is True, (text, out)
            back = check_roundtrip(out["text"], m, expect)
            assert back["ok"] is True, (text, back)
        doc, sem, out = realize_of("杯子的材质是什么？", m)
        assert out["roundtrippable"] is False, out
        # capability absent stays unknown and is never claimed roundtripped
        doc_u, _, out_u = realize_of("杯子可以飞吗？", m)
        assert doc_u["verdict"] == "unknown", doc_u
        assert out_u["roundtrippable"] is False, out_u
        assert "不知道" in out_u["text"] or "证据不足" in out_u["text"], out_u
    finally:
        m.close()


def test_unsupported_not_rewritten(tmp_path):
    m = fresh(tmp_path, tag="sr9")
    try:
        doc, sem, out = realize_of("杯子为什么会响？", m)
        assert doc["verdict"] == "unknown"
        assert "不知道" in out["text"], out
        assert "因为" not in out["text"], out
    finally:
        m.close()


def test_same_semantics_new_entity(tmp_path):
    m = fresh(tmp_path, tag="sr10")
    try:
        o = Ontology(m)
        o.add_concept("mug", "马克杯", parents=["container"])
        o.add_attribute("mug", "material", "steel", status="verified")
        _, _, a = realize_of("杯子的材质是什么？", m)
        _, _, b = realize_of("马克杯的材质是什么？", m)
        assert "ceramic" in a["text"] and "steel" in b["text"]
        assert a["text"].replace("杯子", "X").replace("ceramic", "V") == \
            b["text"].replace("马克杯", "X").replace("steel", "V")
    finally:
        m.close()


def test_default_json_trace_same_answer(tmp_path):
    data = fresh_data(tmp_path)
    from cognitive_ai.cli import seed_from_json
    from cognitive_ai.memory import Memory as _M
    mem = _M(str(data / "cognitive.db"))
    try:
        seed_from_json(mem, str(data / "concepts.json"))
    finally:
        mem.close()
    zh = cli(data, "ask", "杯子有什么用途？")
    assert "drinking" in zh or "用途" in zh, zh
    doc = json.loads(cli(data, "ask", "--json", "杯子有什么用途？"))
    assert doc["verdict"] == "supported", doc
    assert ("drinking" in zh) == ("drinking" in json.dumps(
        doc, ensure_ascii=False)), (zh, doc)
    tr = cli(data, "ask", "--trace", "杯子有什么用途？")
    assert "queries" in tr and "evidence" in tr, tr


def test_no_writes_and_unknown_entities(tmp_path):
    data = fresh_data(tmp_path, tag="srd2")
    from cognitive_ai.cli import seed_from_json
    from cognitive_ai.memory import Memory as _M
    mem = _M(str(data / "cognitive.db"))
    try:
        seed_from_json(mem, str(data / "concepts.json"))
        n0 = (len(mem.query("SELECT * FROM attributes")),
              len(mem.list_claims()))
    finally:
        mem.close()
    out = cli(data, "ask", "雾铃是什么？")
    assert "不知道" in out, out
    mem = _M(str(data / "cognitive.db"))
    try:
        assert mem.get_concept("雾铃") is None
        n1 = (len(mem.query("SELECT * FROM attributes")),
              len(mem.list_claims()))
        assert n0 == n1 or n1 == (0, 0), (n0, n1)
    finally:
        mem.close()


def test_m3e_fix01_plain_unknown_vs_limited(tmp_path):
    m = fresh(tmp_path, tag="fix01a")
    try:
        import cognitive_ai.policy as policy
        o = Ontology(m)
        for i in range(3):
            o.add_concept(f"e{i}", f"实体{i}", parents=["entity"])
        m.upsert_rule({
            "id": "r_any", "conditions": [{"type": "is_a",
                                           "concept": "?x",
                                           "ancestor": "entity"}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "seen", "value": True},
            "exceptions": []})
        # 1. plain unknown, no budget limit
        doc0, sem0, out0 = realize_of("杯子是什么颜色？", m)
        assert doc0["verdict"] == "unknown", doc0
        assert doc0.get("limited", False) is False, doc0
        assert sem0.get("limited", False) is False, sem0
        assert "不知道" in out0["text"] or "证据不足" in out0["text"], out0
        assert "受限" not in out0["text"], out0
        # 2. budget exhausted -> limited flag propagates
        old = policy.MAX_CHAIN_WORK
        policy.MAX_CHAIN_WORK = 0
        try:
            doc1, sem1, out1 = realize_of("杯子有什么？", m)
        finally:
            policy.MAX_CHAIN_WORK = old
        assert doc1.get("limited", False) is True, doc1
        assert any(isinstance(q, dict) and q.get("op") == "chain_limited"
                   for q in doc1.get("queries", [])), doc1
        assert sem1.get("limited", False) is True, sem1
        assert "受限" in out1["text"] and "尚不能确定" in out1["text"], out1
        assert "不存在" not in out1["text"], out1
        # 3. distinguishable
        assert out0["text"] != out1["text"], (out0, out1)
        # 8. queries/evidence still present
        assert "queries" in doc1 and "evidence" in doc1, doc1
    finally:
        m.close()


def test_m3e_fix01_empty_answer_guard(tmp_path):
    m = fresh(tmp_path, tag="fix01b")
    try:
        doc, _, _ = realize_of("杯子的材质是什么？", m)
        assert doc["verdict"] == "supported" and doc["answer"], doc
        verdict_before = doc["verdict"]
        ev_n = len(doc.get("evidence", []))
        # 4. supported + empty must not emit generic assertion
        doc_e = dict(doc)
        doc_e["answer"] = ""
        sem_e = build_answer_semantics(doc_e)
        out_e = realize(sem_e)
        assert "关于杯子：。" not in out_e["text"], out_e
        assert "未能形成确定结论" in out_e["text"], out_e
        assert "guard:empty-answer" in out_e["constructions"], out_e
        # 5. whitespace triggers the same guard
        doc_w = dict(doc)
        doc_w["answer"] = "   "
        out_w = realize(build_answer_semantics(doc_w))
        assert "未能形成确定结论" in out_w["text"], out_w
        assert "关于杯子：。" not in out_w["text"], out_w
        # 6. non-empty supported still renders normally
        _, _, out_ok = realize_of("杯子的材质是什么？", m)
        assert "ceramic" in out_ok["text"], out_ok
        # 7. realizer does not mutate the grounding verdict
        doc2, sem2, out2 = realize_of("杯子的材质是什么？", m)
        assert doc2["verdict"] == verdict_before, (doc2, verdict_before)
        assert len(doc2.get("evidence", [])) == ev_n, doc2
        assert sem2.get("verdict") == doc2["verdict"], (sem2, doc2)
        assert sem2.get("answer") == doc2["answer"], (sem2, doc2)
    finally:
        m.close()


def test_m3e_fix02_inherited_category_not_unknown(tmp_path):
    m = fresh(tmp_path, tag="fix02a")
    try:
        doc, sem, out = realize_of("杯子是什么？", m)
        assert doc["verdict"] == "supported", doc
        kinds = [e.get("kind") for e in doc.get("evidence", [])]
        statuses = [e.get("status") for e in doc.get("evidence", [])]
        assert "parents" in kinds, doc
        assert "verified" in statuses and "inherited" in statuses, doc
        assert "container" in doc["answer"], doc
        assert sem.get("status") != "unknown", sem
        assert "属于" in out["text"], out
        assert "未知" not in out["text"], out
        assert "inheritance:transitive" in out["constructions"], out
        assert "知识库记录" in out["text"], out
    finally:
        m.close()


def test_m3e_fix02_inherited_only_mapping(tmp_path):
    m = fresh(tmp_path, tag="fix02b")
    try:
        from cognitive_ai.semrealize import _status_note
        doc, _, _ = realize_of("杯子是什么？", m)
        # Pure transitive evidence: provenance reported, never verified.
        doc_i = dict(doc)
        doc_i["evidence"] = [{"kind": "parents", "ref": "entity",
                              "status": "inherited",
                              "detail": "cup inherited parent"}]
        sem_i = build_answer_semantics(doc_i)
        assert sem_i.get("status") == "inherited", sem_i
        out_i = realize(sem_i)
        assert "通过类别继承得到" in out_i["text"], out_i
        assert "已验证" not in out_i["text"], out_i
        assert "不知道" not in out_i["text"], out_i
        # Underlying credibility is kept when present, never upgraded.
        assert _status_note([{"status": "hypothesis"}]) == "暂定假设，未验证"
        assert _status_note([{"status": "inherited(hypothesis)"}]) == \
            "暂定假设，未验证"
        assert _status_note([{"status": "contested"}]) == "存疑"
        assert _status_note([{"status": "verified"}]) == "已验证"
        assert _status_note([]) == "未知"
    finally:
        m.close()


def test_m3e_fix02_no_inheritance_stays_unknown(tmp_path):
    m = fresh(tmp_path, tag="fix02c")
    try:
        o = Ontology(m)
        o.add_concept("loner", "孤子", parents=[])
        doc, sem, out = realize_of("孤子是什么？", m)
        assert doc["verdict"] == "unknown", doc
        assert sem.get("limited", False) is False, sem
        assert "不知道" in out["text"] or "证据不足" in out["text"], out
        assert "属于" not in out["text"], out
    finally:
        m.close()


def _seed_cli_data(tmp_path, tag):
    data = fresh_data(tmp_path, tag=tag)
    from cognitive_ai.cli import seed_from_json
    from cognitive_ai.memory import Memory as _M
    mem = _M(str(data / "cognitive.db"))
    try:
        seed_from_json(mem, str(data / "concepts.json"))
    finally:
        mem.close()
    return data


def test_m3e_fix03_cli_output_contract(tmp_path):
    data = _seed_cli_data(tmp_path, "fix03")
    q_ok, q_un = "杯子有什么用途？", "杯子是什么颜色？"
    zh = cli(data, "ask", q_ok)
    lines = [ln for ln in zh.splitlines() if ln.strip()]
    assert len(lines) == 1, zh
    assert '"verdict"' not in zh and "== trace ==" not in zh, zh
    assert "drinking" in zh, zh
    payload = json.loads(cli(data, "ask", "--json", q_ok))
    for k in ("question", "verdict", "answer", "evidence", "queries"):
        assert k in payload, payload
    assert payload["verdict"] == "supported", payload
    assert payload["evidence"] and payload["queries"], payload
    assert payload["realization"]["text"] == zh.strip(), (payload, zh)
    tr = cli(data, "ask", "--trace", q_ok)
    tlines = tr.splitlines()
    assert tlines[0] == zh.strip(), (tr, zh)
    assert tlines[1] == "== trace ==", tr
    diag = json.loads("\n".join(tlines[2:]))
    for k in ("question", "queries", "evidence", "constructions",
              "roundtrippable"):
        assert k in diag, diag
    assert diag["queries"] == payload["queries"], (diag, payload)
    assert diag["evidence"] == payload["evidence"], (diag, payload)
    both_raw = cli(data, "ask", "--json", "--trace", q_ok)
    assert "== trace ==" not in both_raw, both_raw
    both = json.loads(both_raw)
    assert both["verdict"] == "supported", both
    assert both["queries"] and both["evidence"], both
    assert both["realization"]["text"] == zh.strip(), (both, zh)
    # Unknown mode is invariant across output modes.
    zh_u = cli(data, "ask", q_un)
    assert "不知道" in zh_u, zh_u
    doc_u = json.loads(cli(data, "ask", "--json", q_un))
    assert doc_u["verdict"] == "unknown" and doc_u["answer"] == "", doc_u
    assert doc_u.get("limited", False) is False, doc_u
    assert "受限" not in doc_u.get("detail", ""), doc_u
    assert doc_u["realization"]["text"] == zh_u.strip(), (doc_u, zh_u)
    tr_u = cli(data, "ask", "--trace", q_un)
    assert tr_u.splitlines()[0] == zh_u.strip(), (tr_u, zh_u)


def test_m3e_fix03_cli_single_query(tmp_path, monkeypatch, capsys):
    import cognitive_ai.semground as sg
    from cognitive_ai import cli as cli_mod
    data = _seed_cli_data(tmp_path, "fix03b")
    old = (cli_mod.DATA, cli_mod.DB, cli_mod.CORPUS, cli_mod.SEED)
    cli_mod.DATA, cli_mod.DB = str(data), str(data / "cognitive.db")
    cli_mod.CORPUS, cli_mod.SEED = (str(data / "corpus"),
                                    str(data / "concepts.json"))
    real = sg.answer_question
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    try:
        monkeypatch.setattr(sg, "answer_question", counting)
        q = "杯子有什么用途？"
        for argv in (["ask", q], ["ask", "--json", q],
                     ["ask", "--trace", q],
                     ["ask", "--json", "--trace", q]):
            calls["n"] = 0
            capsys.readouterr()
            cli_mod.build_parser().parse_args(argv).fn(
                cli_mod.build_parser().parse_args(argv))
            assert calls["n"] == 1, argv
    finally:
        cli_mod.DATA, cli_mod.DB, cli_mod.CORPUS, cli_mod.SEED = old


def _fix04_db(tmp_path, tag, setup=None):
    from cognitive_ai.memory import Memory as _M
    db = str(tmp_path / f"{tag}.db")
    m = _M(db)
    try:
        o = Ontology(m)
        o.add_concept("entity", "实体")
        o.add_concept("container", "容器", parents=["entity"])
        o.add_concept("cup", "杯子", parents=["container"])
        o.add_attribute("cup", "material", "ceramic", status="verified")
        o.add_attribute("cup", "function", "drinking", status="verified")
        if setup is not None:
            setup(o, m)
    finally:
        m.close()
    return db


def _fix04_point(monkeypatch, cli_mod, db):
    monkeypatch.setattr(cli_mod, "DB", db)
    monkeypatch.setattr(cli_mod, "SEED", db + ".no-seed.json")


def _fix04_run(cli_mod, capsys, argv):
    capsys.readouterr()
    args = cli_mod.build_parser().parse_args(argv)
    args.fn(args)
    return capsys.readouterr().out


def _fix04_modes(cli_mod, capsys, q):
    zh = _fix04_run(cli_mod, capsys, ["ask", q])
    payload = json.loads(_fix04_run(cli_mod, capsys, ["ask", "--json", q]))
    tr = _fix04_run(cli_mod, capsys, ["ask", "--trace", q])
    both = json.loads(_fix04_run(cli_mod, capsys,
                                 ["ask", "--json", "--trace", q]))
    return zh, payload, tr, both


def test_m3e_fix04_supported_cli(tmp_path, monkeypatch, capsys):
    from cognitive_ai.memory import Memory as _M
    from cognitive_ai import cli as cli_mod
    db = _fix04_db(tmp_path, "fix04sup")
    _fix04_point(monkeypatch, cli_mod, db)
    m = _M(db)
    try:
        n0 = counts(m)
        zh, payload, tr, both = _fix04_modes(cli_mod, capsys,
                                             "杯子有什么用途？")
        assert payload["verdict"] == "supported", payload
        assert "drinking" in payload["answer"], payload
        assert payload["realization"]["text"] == zh.strip()
        assert tr.splitlines()[0] == zh.strip()
        assert both["realization"]["text"] == zh.strip()
        assert "drinking" in zh and "断言" not in zh, zh
        doc, sem, out = realize_of("杯子是不是容器？", m)
        assert doc["verdict"] == "supported" and doc["answer"] == "是"
        back = check_roundtrip(out["text"], m,
                               {"subject": "杯子", "predicate": "is_a",
                                "object": "容器", "negated": False,
                                "modal": None})
        assert back["ok"] is True, back
        assert counts(m) == n0
    finally:
        m.close()


def test_m3e_fix04_hypothesis_cli(tmp_path, monkeypatch, capsys):
    from cognitive_ai.memory import Memory as _M
    from cognitive_ai import cli as cli_mod

    def setup(o, m):
        o.add_concept("bowl", "碗", parents=["container"])
        o.add_attribute("bowl", "material", "clay", status="hypothesis")

    db = _fix04_db(tmp_path, "fix04hyp", setup)
    _fix04_point(monkeypatch, cli_mod, db)
    m = _M(db)
    try:
        n0 = counts(m)
        zh, payload, tr, both = _fix04_modes(cli_mod, capsys,
                                             "碗的材质是什么？")
        assert payload["verdict"] == "supported", payload
        assert payload["answer"] == "clay", payload
        assert any(e.get("status") == "hypothesis"
                   for e in payload["evidence"]), payload
        assert "可能是" in zh and "暂定假设" in zh, zh
        assert "已验证" not in zh, zh
        assert payload["realization"]["text"] == zh.strip()
        assert tr.splitlines()[0] == zh.strip()
        assert both["realization"]["text"] == zh.strip()
        assert counts(m) == n0
    finally:
        m.close()


def test_m3e_fix04_contested_cli(tmp_path, monkeypatch, capsys):
    from cognitive_ai.memory import Memory as _M
    from cognitive_ai import cli as cli_mod

    def setup(o, m):
        o.add_attribute("cup", "material", "glass", status="hypothesis")

    db = _fix04_db(tmp_path, "fix04con", setup)
    _fix04_point(monkeypatch, cli_mod, db)
    m = _M(db)
    try:
        n0 = counts(m)
        zh, payload, tr, both = _fix04_modes(cli_mod, capsys,
                                             "杯子的材质是什么？")
        assert payload["verdict"] == "conflicted", payload
        assert "ceramic" in payload["answer"] \
            and "glass" in payload["answer"], payload
        assert "冲突" in zh and "ceramic" in zh and "glass" in zh, zh
        assert payload["realization"]["text"] == zh.strip()
        assert tr.splitlines()[0] == zh.strip()
        assert both["realization"]["text"] == zh.strip()
        assert counts(m) == n0
    finally:
        m.close()


def test_m3e_fix04_inherited_cli(tmp_path, monkeypatch, capsys):
    from cognitive_ai.memory import Memory as _M
    from cognitive_ai import cli as cli_mod
    db = _fix04_db(tmp_path, "fix04inh")
    _fix04_point(monkeypatch, cli_mod, db)
    m = _M(db)
    try:
        n0 = counts(m)
        zh, payload, tr, both = _fix04_modes(cli_mod, capsys,
                                             "杯子是什么？")
        assert payload["verdict"] == "supported", payload
        assert "container" in payload["answer"], payload
        assert any(e.get("status") == "inherited"
                   for e in payload["evidence"]), payload
        assert "属于" in zh and "不知道" not in zh and "未知" not in zh, zh
        assert "inheritance:transitive" in \
            payload["realization"]["constructions"], payload
        assert payload["realization"]["text"] == zh.strip()
        assert tr.splitlines()[0] == zh.strip()
        assert both["realization"]["text"] == zh.strip()
        assert counts(m) == n0
    finally:
        m.close()


def test_m3e_fix04_unknown_cli(tmp_path, monkeypatch, capsys):
    from cognitive_ai.memory import Memory as _M
    from cognitive_ai import cli as cli_mod
    db = _fix04_db(tmp_path, "fix04unk")
    _fix04_point(monkeypatch, cli_mod, db)
    m = _M(db)
    try:
        n0 = counts(m)
        zh, payload, tr, both = _fix04_modes(cli_mod, capsys,
                                             "杯子是什么颜色？")
        assert payload["verdict"] == "unknown", payload
        assert payload["answer"] == "", payload
        assert payload.get("limited", False) is False, payload
        assert "不知道" in zh, zh
        assert payload["realization"]["text"] == zh.strip()
        assert tr.splitlines()[0] == zh.strip()
        assert both["realization"]["text"] == zh.strip()
        assert counts(m) == n0
    finally:
        m.close()


def test_m3e_fix04_limited_cli(tmp_path, monkeypatch, capsys):
    import cognitive_ai.policy as policy
    from cognitive_ai.memory import Memory as _M
    from cognitive_ai import cli as cli_mod

    def setup(o, m):
        for i in range(3):
            o.add_concept(f"e{i}", f"实体{i}", parents=["entity"])
        m.upsert_rule({
            "id": "r_any", "conditions": [{"type": "is_a",
                                           "concept": "?x",
                                           "ancestor": "entity"}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "seen", "value": True},
            "exceptions": []})

    db = _fix04_db(tmp_path, "fix04lim", setup)
    _fix04_point(monkeypatch, cli_mod, db)
    m = _M(db)
    try:
        n0 = counts(m)
        zh_p, pay_p, __, __ = _fix04_modes(cli_mod, capsys,
                                           "杯子的位置是什么？")
        assert pay_p["verdict"] == "unknown", pay_p
        assert pay_p.get("limited", False) is False, pay_p
        old = policy.MAX_CHAIN_WORK
        policy.MAX_CHAIN_WORK = 0
        try:
            zh_l, pay_l, tr_l, both_l = _fix04_modes(
                cli_mod, capsys, "杯子的位置是什么？")
            zh_s, pay_s, __, both_s = _fix04_modes(
                cli_mod, capsys, "杯子有什么？")
        finally:
            policy.MAX_CHAIN_WORK = old
        assert pay_l["verdict"] == "unknown", pay_l
        assert pay_l.get("limited", False) is True, pay_l
        assert "受限" in zh_l and "尚不能确定" in zh_l, zh_l
        assert "未证明不存在" in zh_l, zh_l
        assert pay_l["realization"]["text"] == zh_l.strip()
        assert tr_l.splitlines()[0] == zh_l.strip()
        assert pay_s["verdict"] == "supported", pay_s
        assert pay_s.get("limited", False) is True, pay_s
        assert pay_s["evidence"], pay_s
        assert "受限" in zh_s and "可能不完整" in zh_s, zh_s
        assert both_s["realization"]["text"] == zh_s.strip()
        assert both_l["queries"] and pay_l["queries"], pay_l
        assert counts(m) == n0
    finally:
        m.close()


def test_m3e_fix04_ambiguous_cli(tmp_path, monkeypatch, capsys):
    from cognitive_ai.memory import Memory as _M
    from cognitive_ai import cli as cli_mod

    def setup(o, m):
        o.add_concept("cup2", "杯子", parents=["entity"])

    db = _fix04_db(tmp_path, "fix04amb", setup)
    _fix04_point(monkeypatch, cli_mod, db)
    m = _M(db)
    try:
        n0 = counts(m)
        zh, payload, tr, both = _fix04_modes(cli_mod, capsys,
                                             "杯子是什么？")
        assert payload["verdict"] == "ambiguous", payload
        assert "请澄清" in zh, zh
        assert payload["realization"]["text"] == zh.strip()
        assert tr.splitlines()[0] == zh.strip()
        assert both["realization"]["text"] == zh.strip()
        assert counts(m) == n0
    finally:
        m.close()


def test_m3e_fix04_failure_mapping():
    from cognitive_ai.models import GroundingVerdict
    doc = GroundingVerdict(sentence="杯子是什么？", verdict="failure",
                           detail="engine error: boom").to_dict()
    sem = build_answer_semantics(doc)
    out = realize(sem)
    assert sem.get("verdict") == "failure", sem
    assert "失败" in out["text"] and "未写入" in out["text"], out
    assert out["roundtrippable"] is False, out
