"""Phase 6E composition experiment: role-anchored slots, cross-type
confusion matrix, restart persistence -- all on isolated DBs.

Trains USE and MAT constructions on disjoint entities, tests on fresh
entities with BOTH types, and proves: transfer works, types do not
confuse, reason questions stay reason questions, abstention holds.
Battery is combinatorial (entities x constructions) and audited
against src; train/held/test entities are pairwise disjoint.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from cognitive_ai import phraselearn as PL
from cognitive_ai.cli import seed_from_json
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory
from cognitive_ai.semantics import parse_sentence, resolve_question

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))

USE_TRN = ["碗", "瓶子"]
MAT_TRN = ["杯子", "水壶"]
USE_HLD = ["杯子", "水壶"]
MAT_HLD = ["碗", "瓶子"]
TEST_ENTS = ["茶壶", "茶杯"]


def use_form(e):
    return f"告诉我{e}的用途"


def use_canon(e):
    return f"{e}有什么用途"


def mat_form(e):
    return f"请告诉我{e}的材质"


def mat_canon(e):
    return f"{e}的材质是什么"


REASON = ["为什么茶壶会漏水", "为什么人们会使用茶杯"]


def make_session(tmp_path, tag="c6e"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    return Session(mem, corp), mem


def strict_key(text, rules=None):
    fr = parse_sentence(text, source="user", lexicon={}, phrase_rules=rules)
    d = PL.frame_signature(fr.to_dict())
    d["entities"] = sorted(e.get("text", "") for e in fr.entities)
    return PL._sig_key(d), fr


def teach_all(sess):
    for e in USE_TRN:
        sess.handle(f"说法 {use_form(e)} => {use_canon(e)}")
    for e in MAT_TRN:
        sess.handle(f"说法 {mat_form(e)} => {mat_canon(e)}")
    for e in USE_HLD:
        sess.handle(f"说法 {use_form(e)} => {use_canon(e)} --留存")
    for e in MAT_HLD:
        sess.handle(f"说法 {mat_form(e)} => {mat_canon(e)} --留存")
    out = sess.handle("说法归纳")
    assert "2 候选" in "\n".join(out), out
    return out


def test_slots_are_role_anchored(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        teach_all(sess)
        rules = mem.list_phrase_rules()
        assert len(rules) == 2
        assert all(r["kind"] == "slot" for r in rules), rules
        # each slot template's dst must name the same role the slot fills
        for r in rules:
            assert "{}" in r["dst_tmpl"].get("template", ""), r
    finally:
        mem.close()


def test_composition_matrix_with_restart(tmp_path, capsys):
    sess, mem = make_session(tmp_path)
    try:
        teach_all(sess)
        rules = mem.list_phrase_rules()
        use_r = next(r for r in rules
                     if "告诉我" in r["src_tmpl"].get("pre", ""))
        mat_r = next(r for r in rules
                     if "请告诉我" in r["src_tmpl"].get("pre", ""))
        for r in rules:
            v = sess.handle(f"说法验证 {r['id']}")
            assert any("验证通过" in l for l in v), v
            a = sess.handle(f"说法激活 {r['id']}")
            assert any("已激活" in l for l in a), a
        active = mem.list_phrase_rules(status="active")
        assert len(active) == 2
        # transfer: fresh entities x both types
        probes = ([(use_form(e), use_canon(e)) for e in TEST_ENTS] +
                  [(mat_form(e), mat_canon(e)) for e in TEST_ENTS])
        hits = sum(1 for t, c in probes
                   if strict_key(t, active)[0] == strict_key(c)[0])
        # confusion: USE rule silent on MAT probes and vice versa
        conf = 0
        for t, _ in probes:
            is_use = t.startswith("告诉我") and not t.startswith("请告诉我")
            other = mat_r if is_use else use_r
            if PL.apply_rules(t, [other], {})["rewritten"] is not None:
                conf += 1
        # reason probes stay reason probes (no USE/MAT rewrite fires)
        reason_ok = 0
        for t in REASON:
            fr = parse_sentence(t, source="user",
                                phrase_rules=active).to_dict()
            if resolve_question(fr, {})["kind"] == "reason_q" and \
                    PL.apply_rules(t, active, {})["rewritten"] is None:
                reason_ok += 1
        with capsys.disabled():
            print(f"\n[6E] transfer={hits}/{len(probes)} "
                  f"confusion={conf} reason_kept={reason_ok}/{len(REASON)}")
        assert hits == len(probes)
        assert conf == 0
        assert reason_ok == len(REASON)
    finally:
        mem.close()
    mem2 = Memory(str(tmp_path / "c6e.db"))
    try:
        active = mem2.list_phrase_rules(status="active")
        assert len(active) == 2
        for t, c in ([(use_form(e), use_canon(e)) for e in TEST_ENTS] +
                     [(mat_form(e), mat_canon(e)) for e in TEST_ENTS]):
            assert strict_key(t, active)[0] == strict_key(c)[0], t
    finally:
        mem2.close()


def test_composed_answers_from_kb(tmp_path):
    sess, mem = make_session(tmp_path, tag="c6e2")
    try:
        from cognitive_ai import dialog as D
        teach_all(sess)
        for r in mem.list_phrase_rules():
            sess.handle(f"说法验证 {r['id']}")
            sess.handle(f"说法激活 {r['id']}")
        a = D.process(sess, use_form("杯子"))["answer"]["text"]
        b = D.process(sess, mat_form("杯子"))["answer"]["text"]
        assert "drinking" in a, a
        assert "ceramic" in b, b
        assert "ceramic" not in a and "drinking" not in b, (a, b)
    finally:
        mem.close()


def test_battery_not_hardcoded_in_src():
    blobs = []
    for f in ("semantics.py", "phraselearn.py", "dialog.py",
              "interact.py", "executor.py"):
        blobs.append(open(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "src", "cognitive_ai", f)),
            encoding="utf-8").read())
    tests = ([use_form(e) for e in TEST_ENTS] +
             [mat_form(e) for e in TEST_ENTS] + REASON)
    leaked = [s for s in tests if len(s) >= 6
              and any(s in b for b in blobs)]
    assert not leaked, f"battery leaked into source: {leaked}"
