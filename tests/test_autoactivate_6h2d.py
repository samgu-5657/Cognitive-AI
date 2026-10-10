"""Phase 6H-2D auto-activation experiment: evidence thresholds computed
from persisted pairs; activation without any manual command; conflict
demotion; restart persistence. All through real /api/chat on isolated
DBs (except the mechanism-level conflict probe, which still goes
through the real record_pair + auto_learn_step entries).
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from cognitive_ai.web.server import create_app

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))

TRN = [("狐狸", "筑坝"), ("浣熊", "游泳")]
HLD = [("松鼠", "潜水"), ("刺猬", "跳跃"), ("考拉", "筑坝")]


def form(e, v):
    return f"{e}擅长{v}"


@pytest.fixture()
def client(tmp_path):
    data = tmp_path / "ddata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    models = tmp_path / "dmodels"
    models.mkdir()
    app = create_app(str(data), str(models))
    with TestClient(app) as c:
        yield c
    app.state.wb.close()


def chat(c, text, sid="da"):
    r = c.post("/api/chat", json={"text": text, "session_id": sid}).json()
    assert r["ok"], r
    return r["data"]


def confirm_pair(c, e, v, sid="da"):
    d = chat(c, form(e, v), sid)
    assert d["route"] == "propose", (e, v, d)
    return chat(c, "确认", sid)


def test_threshold_and_auto_activation(client):
    c = client
    wb = c.app.state.wb
    assert chat(c, form("考拉", "筑坝"))["route"] == "unknown"
    for e, v in TRN:
        chat(c, f"观察 {e} 功能={v}")
    for e, v in TRN:
        confirm_pair(c, e, v)
    rows = wb.mem.list_constructions()
    assert len(rows) == 1 and rows[0]["status"] == "candidate", rows
    assert wb.mem.list_constructions(status="active") == []
    chat(c, f"观察 {HLD[0][0]} 功能={HLD[0][1]}")
    confirm_pair(c, *HLD[0])
    assert wb.mem.get_construction(rows[0]["id"])["status"] == "validated"
    assert wb.mem.list_constructions(status="active") == []
    for e, v in HLD[1:]:
        chat(c, f"观察 {e} 功能={v}")
    confirm_pair(c, *HLD[1])
    assert wb.mem.list_constructions(status="active") == []
    out = confirm_pair(c, *HLD[2])
    row = wb.mem.get_construction(rows[0]["id"])
    assert row["status"] == "active", row
    _ = out
    evs = [e for e in wb.mem.list_experiences(limit=100000)
           if "auto-activated construction" in e.get("summary", "")]
    assert len(evs) == 1, evs
    import json as _json
    det = _json.loads(evs[0]["detail"])
    assert det["op"] == "cx_auto_activate"
    assert det["lifecycle"] == "validated->active"
    assert len(det["evidence_pair_ids"]) >= 3, det
    # behavior: fresh entity parses + generates via the active rule
    d = chat(c, "袋鼠擅长游泳")
    assert d["route"] == "teach_pending", d
    g = chat(c, "生成 袋鼠 游泳")
    assert "袋鼠擅长游泳" in "\n".join(g["lines"]), g
    assert "learned:" in "\n".join(g["lines"]), g


def test_duplicate_and_ambiguous_never_activate(client):
    c = client
    wb = c.app.state.wb
    chat(c, "观察 狐狸 功能=筑坝", "dup")
    chat(c, "观察 浣熊 功能=游泳", "dup")
    chat(c, form("狐狸", "筑坝"), "dup")
    chat(c, "确认", "dup")
    chat(c, form("狐狸", "筑坝"), "dup")
    chat(c, "确认", "dup")
    assert wb.mem.list_constructions() == [] or all(
        r["status"] != "active"
        for r in wb.mem.list_constructions()), \
        wb.mem.list_constructions()
    d = chat(c, "狐狸浣熊擅长筑坝", "dup")
    assert d["route"] in ("unknown", "clarify"), d


def test_conflict_demotes_active_rule(tmp_path):
    import tempfile
    from cognitive_ai import constructions as CX
    from cognitive_ai.cli import seed_from_json
    from cognitive_ai.memory import Memory
    db = str(tmp_path / "cf.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    try:
        for e, v in TRN:
            mem.add_experience(
                "evaluation", "x",
                {"op": "cx_pair", "src": form(e, v), "entity": e,
                 "value": v, "basis": "t"})
        rep = CX.auto_learn_step(mem)
        assert len(rep["induced"]) == 1
        rid = rep["induced"][0]
        for e, v in HLD:
            mem.add_experience(
                "evaluation", "x",
                {"op": "cx_pair", "src": form(e, v), "entity": e,
                 "value": v, "basis": "t"})
        CX.auto_learn_step(mem)
        row = mem.get_construction(rid)
        row["status"] = "validated"
        mem.save_construction(row)
        assert CX.activate_rule(mem, rid)["ok"]
        # mislabeled pair: same affix, wrong roles
        mem.add_experience(
            "evaluation", "x",
            {"op": "cx_pair", "src": form("考拉", "筑坝"),
             "entity": "袋鼠", "value": "筑坝", "basis": "mislabeled"})
        rep = CX.auto_learn_step(mem)
        assert mem.get_construction(rid)["status"] == "validated", \
            mem.get_construction(rid)
        assert rep.get("conflicts"), rep
    finally:
        mem.close()


def test_restart_keeps_auto_active(tmp_path):
    from fastapi.testclient import TestClient as TC
    data = tmp_path / "ddata2"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    app1 = create_app(str(data), str(tmp_path / "dm"))
    with TC(app1) as c:
        for e, v in TRN + HLD:
            c.post("/api/chat", json={"text": f"观察 {e} 功能={v}",
                                      "session_id": "r"})
            c.post("/api/chat", json={"text": form(e, v),
                                      "session_id": "r"})
            c.post("/api/chat", json={"text": "确认", "session_id": "r"})
        rows = app1.state.wb.mem.list_constructions(status="active")
        assert len(rows) == 1, rows
    app1.state.wb.close()
    app2 = create_app(str(data), str(tmp_path / "dm"))
    with TC(app2) as c:
        rows = app2.state.wb.mem.list_constructions(status="active")
        assert len(rows) == 1, rows
        d = c.post("/api/chat", json={"text": "袋鼠擅长游泳",
                                      "session_id": "n"}).json()["data"]
        assert d["route"] == "teach_pending", d
    app2.state.wb.close()


def test_manual_activate_still_works(client):
    c = client
    wb = c.app.state.wb
    for e, v in TRN:
        chat(c, f"构式 {form(e, v)} => entity={e};value={v}", "m")
    for e, v in HLD[:1]:
        chat(c, f"构式 {form(e, v)} => entity={e};value={v} --留存", "m")
    ind = chat(c, "构式归纳", "m")
    rid = [l.split()[0] for l in ind["lines"]
           if l.strip().startswith("cx_")][0]
    assert any("验证通过" in l
               for l in chat(c, f"构式验证 {rid}", "m")["lines"])
    assert any("已激活" in l
               for l in chat(c, f"构式激活 {rid}", "m")["lines"])
    assert wb.mem.get_construction(rid)["status"] == "active"


def test_battery_not_hardcoded_in_src():
    blobs = []
    for f in ("semantics.py", "phraselearn.py", "semlearn.py",
              "constructions.py", "dialog.py", "interact.py",
              "executor.py"):
        blobs.append(open(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "src", "cognitive_ai", f)),
            encoding="utf-8").read())
    tests = [form(e, v) for e, v in TRN + HLD] + \
        ["袋鼠擅长游泳", "狐狸浣熊擅长筑坝"]
    leaked = [s for s in tests if len(s) >= 6
              and any(s in b for b in blobs)]
    assert not leaked, f"battery leaked into source: {leaked}"
