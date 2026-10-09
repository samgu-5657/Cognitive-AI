"""Phase 3B tests: quarantine queue, review, incremental training, gate.

All tmp-isolated. Covers: no auto-labels, review history, duplicates,
candidate-checkpoint production, replay use, gate rejection, rollback,
leakage freedom, label extension, mini end-to-end cycle on CPU.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from cognitive_ai.neural import TORCH_AVAILABLE

NEED_TORCH = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch missing")
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "neural"))


def test_unknown_never_becomes_label(tmp_path):
    from cognitive_ai.neural.queue import UnknownQueue
    q = UnknownQueue(str(tmp_path / "q.jsonl"))
    r = q.add("激光打印机", "m1", "entity", "electronics", 0.31,
              "low_confidence")
    assert r["label"] is None and r["review"]["status"] == "pending"
    # accept REQUIRES an explicit L1 label
    bad = q.review(r["sample_id"], accept=True)
    assert not bad["ok"]
    assert q.get(r["sample_id"])["label"] is None


def test_review_history_and_dup_merge(tmp_path):
    from cognitive_ai.neural.queue import UnknownQueue
    q = UnknownQueue(str(tmp_path / "q.jsonl"))
    a = q.add("激光打印机", "m1", "entity", "electronics", 0.31,
              "low_confidence")
    b = q.add("激光打印机", "m1", "entity", "electronics", 0.29,
              "low_confidence")
    assert b["dup_of"] == a["sample_id"]
    rep = q.merge_duplicates()
    assert rep["merged"] >= 0
    ok = q.review(a["sample_id"], accept=True, l1="entity",
                  l2="electronics", reviewer="tester")
    assert ok["ok"] and ok["status"] == "accepted"
    rec = q.get(a["sample_id"])
    assert rec["label"] == {"l1": "entity", "l2": "electronics"}
    assert rec["review"]["history"][0]["action"] == "accept"
    # duplicates cannot be reviewed; accepted() skips them
    dup_review = q.review(b["sample_id"], accept=True, l1="entity")
    assert not dup_review["ok"]
    assert [x["sample_id"] for x in q.accepted()] == [a["sample_id"]]
    no = q.review(a["sample_id"], accept=False, reviewer="tester")
    assert no["ok"] and q.get(a["sample_id"])["review"]["status"] == "rejected"


def tiny_base(tmp_path, tag="base", n=40, epochs=8):
    from cognitive_ai.neural.dataset import load_jsonl
    from cognitive_ai.neural.trainer import train_model
    tr = load_jsonl(os.path.join(DATA, "train.jsonl"))[:n]
    va = load_jsonl(os.path.join(DATA, "val.jsonl"))[:12]
    return train_model(tr, va, str(tmp_path / tag), kind="cnn", seed=7,
                       epochs=epochs, batch=8)


@NEED_TORCH
def test_incremental_produces_candidate_and_uses_replay(tmp_path):
    from cognitive_ai.neural.dataset import load_jsonl
    from cognitive_ai.neural.trainer import incremental_train
    base = tiny_base(tmp_path)
    tr = load_jsonl(os.path.join(DATA, "train.jsonl"))
    new = [{"id": "n1", "group": "g-n1", "text": "卡车",
            "level1": "entity", "level2": "vehicle"},
           {"id": "n2", "group": "g-n2", "text": "滑雪",
            "level1": "action", "level2": "motion"}]
    meta = incremental_train(base["checkpoint"], base["vocab"], new, tr,
                             str(tmp_path / "cand"), epochs=5,
                             replay_ratio=2.0)
    assert meta["before"] != meta["after"]
    assert meta["new_n"] == 2 and meta["replay_n"] == 4
    assert os.path.exists(meta["checkpoint"])
    # unlabeled rows are refused, never silently skipped
    with pytest.raises(ValueError):
        incremental_train(base["checkpoint"], base["vocab"],
                          [{"id": "x", "text": "??"}], tr,
                          str(tmp_path / "bad"))
    # closed L1 space is enforced with a clear error
    with pytest.raises(ValueError):
        incremental_train(base["checkpoint"], base["vocab"],
                          [{"id": "x", "text": "??", "level1": "alien"}],
                          tr, str(tmp_path / "bad2"))


@NEED_TORCH
def test_label_extension_preserves_mapping(tmp_path):
    import torch
    from cognitive_ai.neural import LEVEL2
    from cognitive_ai.neural.trainer import extend_l2, load_model
    base = tiny_base(tmp_path)
    model, _, _ = load_model(base["checkpoint"], base["vocab"])
    before = model.net.fc2.weight.detach().clone()
    model, full = extend_l2(model, LEVEL2, LEVEL2 + ["sub_newthing"])
    assert full[-1] == "sub_newthing" and len(full) == len(LEVEL2) + 1
    assert torch.equal(model.net.fc2.weight[:len(LEVEL2)], before)
    # second extension stacks on the CURRENT map, old rows still exact
    w2 = model.net.fc2.weight.detach().clone()
    model, full2 = extend_l2(model, full, list(full) + ["sub_second"])
    assert full2[-2:] == ["sub_newthing", "sub_second"]
    assert torch.equal(model.net.fc2.weight[:len(full)], w2)
    # no-op when nothing new
    m2, full3 = extend_l2(model, full2, list(full2))
    assert full3 == full2


@NEED_TORCH
def test_gate_reject_keeps_active_and_promote_rolls_back(tmp_path):
    from cognitive_ai.neural.dataset import load_jsonl
    from cognitive_ai.neural.gate import (PromotionPolicy, decide,
                                          evaluate_pair, promote,
                                          read_active, rollback_active)
    from cognitive_ai.neural.trainer import incremental_train
    base = tiny_base(tmp_path)
    base_dir = os.path.dirname(base["checkpoint"])
    tr = load_jsonl(os.path.join(DATA, "train.jsonl"))
    va = [r for r in load_jsonl(os.path.join(DATA, "val.jsonl"))
          if r.get("level1")]
    new = [{"id": f"n{i}", "group": f"g{i}", "text": t,
            "level1": l1, "level2": l2} for i, (t, l1, l2) in enumerate([
                ("卡车", "entity", "vehicle"), ("大卡车", "entity", "vehicle"),
                ("滑雪", "action", "motion"), ("滑冰", "action", "motion"),
                ("钳子", "entity", "tool"), ("老虎钳", "entity", "tool")])]
    meta = incremental_train(base["checkpoint"], base["vocab"], new, tr,
                             str(tmp_path / "cand"), epochs=5)
    cand_dir = os.path.dirname(meta["checkpoint"])
    pair = evaluate_pair(base_dir, cand_dir, va, new)
    assert "delta_regression" in pair and "forgetting" in pair
    assert pair["candidate"]["new_n"] == 6
    # forced-reject branch: active pointer untouched
    root = tmp_path / "models"
    root.mkdir()
    (root / "active.json").write_text(json.dumps({"tag": "base"}),
                                      encoding="utf-8")
    dec = decide(pair, PromotionPolicy(regression_tolerance=-1.0,
                                       min_new_samples=3))
    assert not dec["promote"]
    assert read_active(str(root))["tag"] == "base"
    # forced-accept branch then rollback
    dec2 = decide(pair, PromotionPolicy(regression_tolerance=1.0,
                                        min_new_samples=3))
    assert dec2["promote"]
    promote(str(root), "cand", pair, dec2, data_version="test")
    assert read_active(str(root))["tag"] == "cand"
    rb = rollback_active(str(root))
    assert rb["ok"] and rb["active"]["tag"] == "base"


@NEED_TORCH
def test_new_data_leak_free():
    from cognitive_ai.neural.dataset import check_no_leakage, load_jsonl
    tr = load_jsonl(os.path.join(DATA, "train.jsonl"))
    new = [{"id": "n1", "group": "rev-new-truck", "text": "卡车",
            "level1": "entity"}]
    assert check_no_leakage(tr, [], new)["ok"]


@NEED_TORCH
def test_mini_end_to_end_cycle_cpu(tmp_path):
    from cognitive_ai.neural.adapter import NeuralAdapter
    from cognitive_ai.neural.cycle import cycle_predict, maybe_train
    from cognitive_ai.neural.gate import PromotionPolicy, read_active
    from cognitive_ai.neural.queue import UnknownQueue
    from cognitive_ai.memory import Memory
    base = tiny_base(tmp_path, n=30, epochs=5)
    base_dir = os.path.dirname(base["checkpoint"])
    models = tmp_path / "models"
    models.mkdir()
    import shutil
    shutil.copytree(base_dir, models / "tinybase")
    (models / "active.json").write_text(json.dumps({"tag": "tinybase"}),
                                        encoding="utf-8")
    ad = NeuralAdapter(str(models / "tinybase" / "best.pt"),
                       str(models / "tinybase" / "vocab.json"))
    mem = Memory(str(tmp_path / "c.db"))
    qpath = str(tmp_path / "queue.jsonl")
    try:
        q = UnknownQueue(qpath)
        out = cycle_predict(ad, q, mem, "@@@未知词xyz", source="test")
        assert out["candidate"]["verdict"] in ("known", "unknown")
        # hand-queue two reviewed samples, run the gated batch
        for t, l1 in (("卡车", "entity"), ("滑雪", "action")):
            r = q.add(t, ad.version, None, None, 0.2, "low_confidence",
                      source="test")
            assert q.review(r["sample_id"], accept=True, l1=l1)["ok"]
        res = maybe_train(
            mem, "tinybase", str(models), qpath,
            os.path.join(DATA, "train.jsonl"), "tinycand",
            batch_min=2, epochs=3,
            policy=PromotionPolicy(min_new_samples=2,
                                   regression_tolerance=1.0))
        assert res["ok"] and res["action"] in (
            "promoted", "rejected_by_gate")
        exps = mem.list_experiences(limit=100)
        assert any("continual cycle" in e["summary"] for e in exps)
    finally:
        mem.close()
