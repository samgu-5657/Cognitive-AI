"""M3F-FIX-01: one claim, one verdict + lifecycle audit for contested rows.

Uses fresh isolated SQLite DBs (never the M3F experiment DBs). The
contested fixture is built through the real NL flow
(propose_from_utterance -> confirm_candidate), never by hand-writing
verdicts.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.learner import Learner
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.semground import answer_question


def fresh(tmp_path, tag="m3ffix"):
    m = Memory(str(tmp_path / f"{tag}.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    o.add_attribute("cup", "material", "ceramic", status="verified")
    o.add_attribute("cup", "function", "drinking", status="verified")
    return m


def contest_cup(m):
    """Real flow: learn/confirm '杯子可以盛水' against drinking."""
    learner = Learner(m, "nowhere")
    cand = learner.propose_from_utterance("杯子可以盛水")
    assert cand["kind"] == "attribute" and cand["can_confirm"], cand
    out = learner.confirm_candidate(cand["id"])
    assert out["ok"] is True, out
    return cand["id"]


def cup_function_rows(m):
    return [a for a in m.get_attributes("cup") if a["key"] == "function"]


def cli_env(data):
    return {**os.environ,
            "PYTHONPATH": os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "src")),
            "COGNITIVE_AI_DATA": str(data)}


def test_open_and_verify_share_contested(tmp_path):
    m = fresh(tmp_path, tag="m3ffix01")
    try:
        contest_cup(m)
        v_open = answer_question("杯子有什么用途？", m)
        v_yesno = answer_question("杯子可以盛水吗？", m)
        assert v_open.verdict == "conflicted", v_open.to_dict()
        assert v_yesno.verdict == "conflicted", v_yesno.to_dict()
        assert "盛水" in v_open.answer, v_open.to_dict()
        assert v_yesno.answer == "存疑", v_yesno.to_dict()
        assert v_yesno.to_dict().get("limited", False) is False
        assert any(e.get("status") == "contested"
                   for e in v_yesno.evidence), v_yesno.to_dict()
    finally:
        m.close()


def test_no_record_stays_unknown(tmp_path):
    m = fresh(tmp_path, tag="m3ffix02")
    try:
        v = answer_question("杯子可以飞吗？", m)
        assert v.verdict == "unknown", v.to_dict()
        assert v.answer == "", v.to_dict()
    finally:
        m.close()


def test_both_sides_kept(tmp_path):
    m = fresh(tmp_path, tag="m3ffix03")
    try:
        contest_cup(m)
        rows = {a["value"]: a for a in cup_function_rows(m)}
        assert rows["drinking"]["status"] == "verified", rows
        assert rows["盛水"]["status"] == "contested", rows
        claims = [c for c in m.list_claims()
                  if c.get("status") == "contested"]
        assert claims, "rival claim missing"
        assert any(m.query(
            "SELECT * FROM evidences WHERE claim_id=?",
            (c["id"],)) for c in claims)
    finally:
        m.close()


def test_hypothesis_not_upgraded(tmp_path):
    m = fresh(tmp_path, tag="m3ffix04")
    try:
        o = Ontology(m)
        o.add_concept("bowl", "碗", parents=["container"])
        o.add_attribute("bowl", "material", "clay", status="hypothesis")
        contest_cup(m)
        for q in ("杯子可以盛水吗？", "杯子有什么用途？", "碗的材质是什么？"):
            answer_question(q, m)
        got = {(a["concept_id"], a["key"], a["value"]): a["status"]
               for cid in ("cup", "bowl") for a in m.get_attributes(cid)}
        assert got[("cup", "function", "drinking")] == "verified", got
        assert got[("cup", "function", "盛水")] == "contested", got
        assert got[("bowl", "material", "clay")] == "hypothesis", got
    finally:
        m.close()


def test_unrelated_contested_ignored(tmp_path):
    m = fresh(tmp_path, tag="m3ffix05")
    try:
        o = Ontology(m)
        o.add_concept("plate", "盘子", parents=["container"])
        o.add_attribute("plate", "function", "飞", status="contested")
        v = answer_question("杯子可以飞吗？", m)
        assert v.verdict == "unknown", v.to_dict()
        v2 = answer_question("盘子可以飞吗？", m)
        assert v2.verdict == "conflicted", v2.to_dict()
    finally:
        m.close()


def test_lifecycle_event_recorded(tmp_path):
    m = fresh(tmp_path, tag="m3ffix06")
    try:
        assert m.list_lifecycle("cup.function") == []
        contest_cup(m)
        evs = m.list_lifecycle("cup.function")
        assert len(evs) == 1, evs
        ev = evs[0]
        assert ev["target_type"] == "attribute", ev
        assert ev["from_state"] == "verified", ev
        assert ev["to_state"] == "contested", ev
        assert ev["reason"], ev
        assert json.loads(ev["evidence"]), ev
    finally:
        m.close()


def test_lifecycle_readable_via_cli(tmp_path):
    data = tmp_path / "m3ffix07"
    data.mkdir()
    m = Memory(str(data / "cognitive.db"))
    try:
        o = Ontology(m)
        o.add_concept("entity", "实体")
        o.add_concept("container", "容器", parents=["entity"])
        o.add_concept("cup", "杯子", parents=["container"])
        o.add_attribute("cup", "function", "drinking", status="verified")
        contest_cup(m)
    finally:
        m.close()
    r = subprocess.run(
        [sys.executable, "-m", "cognitive_ai.cli",
         "lifecycle", "--limit", "5"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        env=cli_env(data))
    assert r.returncode == 0, r.stderr[-800:]
    rows = json.loads(r.stdout)
    hits = [e for e in rows if e.get("target_ref") == "cup.function"
            and e.get("to_state") == "contested"]
    assert hits, rows


def test_restart_keeps_state_and_events(tmp_path):
    db = str(tmp_path / "m3ffix08.db")
    m = Memory(db)
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    o.add_attribute("cup", "function", "drinking", status="verified")
    contest_cup(m)
    m.close()
    m2 = Memory(db)
    try:
        rows = {a["value"]: a["status"]
                for a in m2.get_attributes("cup")
                if a["key"] == "function"}
        assert rows == {"drinking": "verified", "盛水": "contested"}, rows
        assert len(m2.list_lifecycle("cup.function")) == 1
        assert answer_question("杯子可以盛水吗？", m2).verdict == \
            "conflicted"
    finally:
        m2.close()


def test_repeat_learn_no_duplicates(tmp_path):
    m = fresh(tmp_path, tag="m3ffix09")
    try:
        contest_cup(m)
        contest_cup(m)
        same = [a for a in cup_function_rows(m) if a["value"] == "盛水"]
        assert len(same) == 1, [dict(a) for a in same]
        evs = [e for e in m.list_lifecycle("cup.function")
               if e["to_state"] == "contested"]
        assert len(evs) == 1, evs
    finally:
        m.close()
