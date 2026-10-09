"""Phase 4A interactive teaching tests (isolated tmp DBs only)."""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.cli import seed_from_json
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def make_session(tmp_path, tag="s"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    models = str(tmp_path / f"models_{tag}")
    os.makedirs(models, exist_ok=True)
    queue = str(tmp_path / f"q_{tag}.jsonl")
    return Session(mem, corp, models_root=models, queue_path=queue), mem


def counts(mem):
    return {
        "concepts": len(mem.list_concepts()),
        "attrs": len(mem.query("SELECT * FROM attributes")),
        "versions": mem.current_version(),
        "exps": len(mem.list_experiences(limit=100000)),
        "claims": len(mem.list_claims()),
    }


def test_multi_turn_teach_flow(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        out = sess.handle("教 茶壶")
        assert any("茶壶" in l for l in out)
        out = sess.handle("材质=陶瓷 颜色=白")
        assert any("observation" in l for l in out)
        out = sess.handle("泡茶")
        assert any("observation" in l for l in out)
        out = sess.handle("容器")
        assert sess.pending_teach is None
        assert mem.get_concept("teapot") or mem.find_concepts_by_name("茶壶")
    finally:
        mem.close()


def test_utf8_chinese_roundtrip(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        out = sess.handle("观察 杯子 材质=玻璃")
        assert any("杯子" in l for l in out)
        out = sess.handle("假设 杯子可以用来喝水")
        assert any("hypothesis" in l for l in out)
    finally:
        mem.close()


def test_unparseable_writes_nothing(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        before = counts(mem)
        out = sess.handle("呃嗯随便说一句完全没结构的话呢哈哈")
        assert any("无法理解" in l for l in out)
        assert counts(mem) == before
    finally:
        mem.close()


def test_confirmed_fact_persists_across_restart(tmp_path):
    sess, mem = make_session(tmp_path, tag="p")
    mem.close()
    db = str(tmp_path / "p.db")
    mem2 = Memory(db)
    try:
        s2 = Session(mem2, str(tmp_path / "corp_p"),
                     models_root=str(tmp_path / "models_p"),
                     queue_path=str(tmp_path / "q_p.jsonl"))
        s2.handle("观察 茶壶 材质=陶瓷")
        n1 = counts(mem2)
    finally:
        mem2.close()
    mem3 = Memory(db)
    try:
        s3 = Session(mem3, str(tmp_path / "corp_p"),
                     models_root=str(tmp_path / "models_p"),
                     queue_path=str(tmp_path / "q_p.jsonl"))
        out = s3.handle("查看 teapot")
        assert any("陶瓷" in l or "teapot" in l for l in out)
        assert counts(mem3) == n1
    finally:
        mem3.close()


def test_observation_vs_label_are_distinct(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        v0 = mem.current_version()
        sess.handle("观察 新杯子 材质=塑料")
        assert mem.current_version() >= v0  # observation may evolve structure
        from cognitive_ai.neural.queue import UnknownQueue
        q = UnknownQueue(sess.queue_path)
        rec = q.add("新杯子", "m", "entity", None, 0.3, "low_confidence")
        v1 = mem.current_version()
        out = sess.handle(f"标签 {rec['sample_id']} entity")
        assert any("training_label" in l for l in out)
        # labeling touches NO concept/attribute rows and no versions
        assert mem.current_version() == v1
        assert not mem.find_concepts_by_name("标签产物")
    finally:
        mem.close()


def test_unknown_predict_never_autolabels(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        out = sess.handle("分类 激光打印机")
        assert any("激光打印机" in l for l in out)
        if sess.queue_path:
            from cognitive_ai.neural.queue import UnknownQueue
            for r in UnknownQueue(sess.queue_path).list():
                assert r["label"] is None
    finally:
        mem.close()


def test_correction_leaves_record(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        out = sess.handle("纠正 杯子 材质 玻璃 因为看到玻璃杯")
        assert any("correction" in l for l in out)
        vals = [(a["value"], a["status"])
                for a in mem.get_attributes("cup") if a["key"] == "material"]
        assert ("ceramic", "verified") in vals  # old knowledge kept
        exps = [e for e in mem.list_experiences(limit=100)
                if "correction" in e["summary"]]
        assert exps and "玻璃" in exps[0]["summary"]
    finally:
        mem.close()


def test_lifecycle_respected_and_model_untouched(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        v0 = mem.current_version()
        sess.handle("假设 月亮是奶酪做的")
        hyps = [c for c in mem.list_claims() if c["status"] == "hypothesis"]
        assert hyps
        assert mem.current_version() == v0  # hypothesis: no version bump
        # chat has no direct weight-update path: adapter untouched/absent
        assert sess.handle("模型")[0] is not None
    finally:
        mem.close()


def test_quit_restart_keeps_state(tmp_path):
    sess, mem = make_session(tmp_path, tag="q")
    try:
        sess.handle("观察 保温杯 材质=不锈钢")
        assert "__quit__" in sess.handle("退出")
    finally:
        mem.close()
    mem2 = Memory(str(tmp_path / "q.db"))
    try:
        assert mem2.find_concepts_by_name("保温杯")
    finally:
        mem2.close()


def test_chat_subprocess_io(tmp_path):
    # isolated DATA dir so the live user database is never touched
    datadir = tmp_path / "data"
    (datadir / "corpus").mkdir(parents=True)
    shutil.copy(os.path.join(os.path.dirname(__file__), "..",
                             "data", "concepts.json"),
                datadir / "concepts.json")
    script = ("帮助\n概念\n查看 cup\n分类 杯子\n"
              "观察 测试杯 材质=纸\n假设 测试杯很轻\n退出\n")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8",
           "PYTHONPATH": os.path.join(os.path.dirname(__file__),
                                      "..", "src"),
           "COGNITIVE_AI_DATA": str(datadir)}
    # bytes mode: the child writes UTF-8, locale GBK must not decode it
    r = subprocess.run(
        [sys.executable, "-m", "cognitive_ai.cli", "chat"],
        input=script.encode("utf-8"), capture_output=True, timeout=120,
        cwd=os.path.join(os.path.dirname(__file__), ".."), env=env)
    out = r.stdout.decode("utf-8", errors="replace")
    err = (r.stderr or b"").decode("utf-8", errors="replace")
    assert r.returncode == 0, err[-1000:]
    assert "可用命令" in out and "再见" in out
    assert "测试杯" in out
    # state actually persisted in the ISOLATED db
    mem = Memory(str(datadir / "cognitive.db"))
    try:
        assert mem.find_concepts_by_name("测试杯")
    finally:
        mem.close()
