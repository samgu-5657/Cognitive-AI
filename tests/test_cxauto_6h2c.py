"""Phase 6H-2C automatic discovery: ordinary dialogue accumulates
trusted pairs, auto-induces candidates, auto-validates on later
pairs; activation stays explicit. All through real /api/chat on
isolated DBs. No 构式 teaching commands are used anywhere here.
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

F1, F2, F3 = "狐狸擅长筑坝", "浣熊擅长游泳", "松鼠擅长潜水"
FRESH = "考拉擅长筑坝"


@pytest.fixture()
def client(tmp_path):
    data = tmp_path / "adata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    models = tmp_path / "amodels"
    models.mkdir()
    app = create_app(str(data), str(models))
    with TestClient(app) as c:
        yield c
    app.state.wb.close()


def chat(c, text, sid="au"):
    r = c.post("/api/chat", json={"text": text, "session_id": sid}).json()
    assert r["ok"], r
    return r["data"]


def test_auto_loop_end_to_end(client):
    c = client
    # 6. unknown before learning
    d0 = chat(c, FRESH)
    assert d0["route"] == "unknown", d0
    # 1-2. ordinary dialogue accumulates trusted pairs (no 构式 cmds)
    for e, v in [("狐狸", "筑坝"), ("浣熊", "游泳")]:
        chat(c, f"观察 {e} 功能={v}")
    d1 = chat(c, F1)
    assert d1["route"] == "propose", d1
    c1 = chat(c, "确认")
    assert "已记录理解" in "\n".join(c1["lines"]), c1
    d2 = chat(c, F2)
    assert d2["route"] == "propose", d2
    c2 = chat(c, "确认")
    assert "候选构式" in "\n".join(c2["lines"]), c2
    # 3-4. auto candidate exists, unvalidated, from real pairs
    wb = c.app.state.wb
    rows = wb.mem.list_constructions()
    assert len(rows) == 1 and rows[0]["status"] == "candidate", rows
    assert rows[0]["evidence"][0]["source"] == "auto-dialogue"
    # 7. single unverified sample never yields an active rule
    assert wb.mem.list_constructions(status="active") == []
    # held-out third entity validates automatically
    chat(c, "观察 松鼠 功能=潜水")
    assert chat(c, F3)["route"] == "propose"
    c3 = chat(c, "确认")
    assert "验证" in "\n".join(c3["lines"]), c3
    assert wb.mem.get_construction(rows[0]["id"])["status"] == \
        "validated"
    # 5/6. explicit activation -> fresh entity + novel combo work
    chat(c, f"构式激活 {rows[0]['id']}")
    assert wb.mem.get_construction(rows[0]["id"])["status"] == "active"
    d3 = chat(c, FRESH)
    assert d3["route"] == "teach_pending", d3
    g = chat(c, "生成 考拉 筑坝")
    assert FRESH in "\n".join(g["lines"]), g
    assert "learned:" in "\n".join(g["lines"]), g


def test_cancel_and_ambiguity_write_nothing(client):
    c = client
    wb = c.app.state.wb
    chat(c, "观察 狐狸 功能=筑坝", "cx")
    d = chat(c, "狐狸擅长筑坝", "cx")
    assert d["route"] == "propose", d
    n0 = len(wb.mem.list_constructions())
    out = chat(c, "取消", "cx")
    assert "放弃" in "\n".join(out["lines"]), out
    assert len(wb.mem.list_constructions()) == n0
    assert [p for p in wb.mem.query("SELECT * FROM experiences")
            if "cx_pair" in str(p.get("detail", ""))] == []
    # ambiguous: two known entities in one unknown utterance
    chat(c, "观察 浣熊 功能=游泳", "cx")
    d = chat(c, "狐狸浣熊擅长筑坝", "cx")
    assert d["route"] in ("unknown", "clarify"), d


def test_restart_and_session_sharing(tmp_path):
    from fastapi.testclient import TestClient as TC
    data = tmp_path / "adata2"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    app1 = create_app(str(data), str(tmp_path / "am"))
    with TC(app1) as c:
        chat(c, "观察 狐狸 功能=筑坝", "s1")
        chat(c, "观察 浣熊 功能=游泳", "s1")
        chat(c, F1, "s1")
        chat(c, "确认", "s1")
        chat(c, F2, "s1")
        chat(c, "确认", "s1")
        # 10. pending proposal does not leak across sessions
        chat(c, "观察 松鼠 功能=潜水", "s1")
        chat(c, F3, "s1")
        other = chat(c, "确认", "s2")
        assert "待确认" in "\n".join(other["lines"]) or \
            "理解" not in "\n".join(other["lines"]), other
        chat(c, "确认", "s1")
        rows = app1.state.wb.mem.list_constructions()
        assert rows and rows[0]["status"] in ("candidate", "validated")
    app1.state.wb.close()
    # 9. restart keeps rules; shared DB serves both sessions
    app2 = create_app(str(data), str(tmp_path / "am"))
    with TC(app2) as c:
        rows = app2.state.wb.mem.list_constructions()
        assert len(rows) == 1, rows
        for r in rows:
            if r["status"] == "validated":
                chat(c, f"构式激活 {r['id']}", "s9")
        d = chat(c, FRESH, "s9")
        assert d["route"] in ("teach_pending", "qa", "unknown"), d
    app2.state.wb.close()


def test_battery_not_hardcoded_in_src():
    blobs = []
    for f in ("semantics.py", "phraselearn.py", "semlearn.py",
              "constructions.py", "dialog.py", "interact.py",
              "executor.py"):
        blobs.append(open(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "src", "cognitive_ai", f)),
            encoding="utf-8").read())
    tests = [F1, F2, F3, FRESH, "狐狸浣熊擅长筑坝"]
    leaked = [s for s in tests if len(s) >= 6
              and any(s in b for b in blobs)]
    assert not leaked, f"battery leaked into source: {leaked}"
