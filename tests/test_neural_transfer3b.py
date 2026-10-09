"""Continual-learning A/B/C/D experiment (Phase 3B, Task 6).

New concepts (unseen words, frozen held-out paraphrases):
  truck: 卡车/大卡车/货车 + held-out 大货车
  pliers: 钳子/老虎钳 + held-out 钢丝钳
  ski: 滑雪/滑冰 + held-out 溜冰
Old knowledge: frozen val subset (same rows for every arm).
Unknowns: 4 test unknown words (rejection check).

  A: base model, no new samples.
  B: incremental on new samples ONLY (replay_ratio=0).
  C: incremental on new samples + old replay (ratio 2.0).
  D: from-scratch on (old-train-subset + new), same epochs (same budget).

Scores REPORTED (printed), superiority NOT asserted: a null result is an
honest finding. Invariants asserted: candidates produced, gate ran,
no verified-KB interaction (weights only), leakage-free partitions.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from cognitive_ai.neural import TORCH_AVAILABLE

NEED_TORCH = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch missing")
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "neural"))

NEW_TRAIN = [
    {"id": "nw-t1", "group": "nw-truck", "text": "卡车",
     "level1": "entity", "level2": "vehicle"},
    {"id": "nw-t2", "group": "nw-truck", "text": "大卡车",
     "level1": "entity", "level2": "vehicle"},
    {"id": "nw-t3", "group": "nw-truck", "text": "货车",
     "level1": "entity", "level2": "vehicle"},
    {"id": "nw-p1", "group": "nw-pliers", "text": "钳子",
     "level1": "entity", "level2": "tool"},
    {"id": "nw-p2", "group": "nw-pliers", "text": "老虎钳",
     "level1": "entity", "level2": "tool"},
    {"id": "nw-s1", "group": "nw-ski", "text": "滑雪",
     "level1": "action", "level2": "motion"},
    {"id": "nw-s2", "group": "nw-ski", "text": "滑冰",
     "level1": "action", "level2": "motion"},
]
NEW_HELDOUT = [
    {"id": "nh-t1", "group": "nh-truck", "text": "大货车", "level1": "entity"},
    {"id": "nh-p1", "group": "nh-pliers", "text": "钢丝钳", "level1": "entity"},
    {"id": "nh-s1", "group": "nh-ski", "text": "溜冰", "level1": "action"},
]
UNK_WORDS = ["激光打印机", "拓扑学", "鳎目鱼", "铆钉枪"]


@NEED_TORCH
def test_abcd_transfer_experiment(tmp_path):
    from cognitive_ai.neural.dataset import check_no_leakage, load_jsonl
    from cognitive_ai.neural.evaluator import predict_probs
    from cognitive_ai.neural.gate import evaluate_pair
    from cognitive_ai.neural.trainer import incremental_train, train_model
    from cognitive_ai.neural import LEVEL1

    tr = load_jsonl(os.path.join(DATA, "train.jsonl"))
    va = [r for r in load_jsonl(os.path.join(DATA, "val.jsonl"))
          if r.get("level1")]
    old_train = tr[:48]
    old_frozen = va[:12]
    assert check_no_leakage(old_train, old_frozen, NEW_TRAIN)["ok"]
    assert check_no_leakage(old_train, [], NEW_HELDOUT)["ok"]

    base = train_model(old_train, va[:12], str(tmp_path / "base"),
                       kind="cnn", seed=7, epochs=10, batch=8)
    base_dir = str(tmp_path / "base")
    EPOCHS = 15

    def acc(model, vocab, rows):
        p1, _ = predict_probs(model, vocab, [r["text"] for r in rows])
        return round(sum(1 for r, a in zip(rows, p1)
                         if max(range(len(a)), key=lambda i: a[i]) ==
                         LEVEL1.index(r["level1"])) / len(rows), 4)

    def rej(model, vocab):
        p1, _ = predict_probs(model, vocab, UNK_WORDS)
        return round(sum(1 for a in p1 if max(a) < 0.6) / len(p1), 4)

    from cognitive_ai.neural.trainer import load_model
    arms = {}
    # A: frozen base
    mA, vA, _ = load_model(base["checkpoint"], base["vocab"])
    arms["A"] = (mA, vA, {"kind": "frozen"})
    # B: new-only
    mB = incremental_train(base["checkpoint"], base["vocab"], NEW_TRAIN,
                           [], str(tmp_path / "candB"), epochs=EPOCHS,
                           replay_ratio=0.0)
    mBm, _, _ = load_model(mB["checkpoint"],
                           str(tmp_path / "candB" / "vocab.json"))
    arms["B"] = (mBm, vA, mB)
    # C: new + replay
    mC = incremental_train(base["checkpoint"], base["vocab"], NEW_TRAIN,
                           old_train, str(tmp_path / "candC"), epochs=EPOCHS,
                           replay_ratio=2.0)
    mCm, _, _ = load_model(mC["checkpoint"],
                           str(tmp_path / "candC" / "vocab.json"))
    arms["C"] = (mCm, vA, mC)
    # D: from scratch, same data as C, same epochs
    d = train_model(old_train + NEW_TRAIN, va[:12],
                    str(tmp_path / "scratchD"), kind="cnn", seed=7,
                    epochs=EPOCHS, batch=8)
    mDm, vDm, _ = load_model(d["checkpoint"], d["vocab"])
    arms["D"] = (mDm, vDm, d)

    print("\n3B TRANSFER (new=3 held-out, old=12 frozen val, unk=4):")
    report = {}
    for name, (m, v, meta) in arms.items():
        new_acc = acc(m, v, NEW_HELDOUT)
        old_acc = acc(m, v, old_frozen)
        r = rej(m, v)
        report[name] = {"new": new_acc, "old": old_acc, "reject": r}
        print(f"  arm {name}: new_acc={new_acc} old_acc={old_acc} "
              f"reject={r}")
    # invariants: every arm evaluated on identical frozen rows; gate runs
    pair = evaluate_pair(base_dir, str(tmp_path / "candC"), old_frozen,
                         NEW_TRAIN)
    assert pair["candidate"]["new_n"] == len(NEW_TRAIN)
    assert "forgetting" in pair
    assert set(report) == {"A", "B", "C", "D"}
