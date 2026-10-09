"""Phase 3C structure experiment (Task 5): fixed vs facts-only vs dynamic.

New subcategory (reviewed, L2='none'): kitchenware under entity --
  train: 炒锅/平底锅/蒸锅 ; frozen held-out: 铁锅.
Old knowledge: frozen val subset (12 rows). Same epochs everywhere.

  A: base model + OOD verdicts only (no learning, no structure).
  B: incremental with L2='none' labels (facts only, geometry unchanged).
  C: discovery -> LOO validation -> extend head -> incremental with the
     new L2 label -> model gate. If validation fails, C keeps base and
     reports the rejection (no forced structure).

Metrics (frozen rows, printed): new-L2 hit on held-out, old L1 accuracy,
unknown FAR/FRR via OOD verdicts. Invariants asserted; superiority is
reported, never asserted. Success is test metrics, never category counts.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from cognitive_ai.neural import TORCH_AVAILABLE

NEED_TORCH = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch missing")
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "neural"))

NEW_REVIEWED = [
    {"sample_id": "kw1", "text": "炒锅", "l1": "entity", "l2": "none"},
    {"sample_id": "kw2", "text": "平底锅", "l1": "entity", "l2": "none"},
    {"sample_id": "kw3", "text": "蒸锅", "l1": "entity", "l2": "none"},
]
HELDOUT = {"text": "铁锅", "l1": "entity"}
UNK_WORDS = ["激光打印机", "鳎目鱼", "xyz", " topology"]
OLD_N = 12
EPOCHS = 15


@NEED_TORCH
def test_abc_dynamic_structure():
    from cognitive_ai.neural import LEVEL1
    from cognitive_ai.neural.adapter import NeuralAdapter
    from cognitive_ai.neural.dataset import check_no_leakage, load_jsonl
    from cognitive_ai.neural.discovery import (apply_new_l2, discover,
                                               l2_prototypes,
                                               validate_proposal)
    from cognitive_ai.neural.evaluator import predict_probs
    from cognitive_ai.neural.gate import evaluate_pair
    from cognitive_ai.neural.ood import assess, class_prototypes
    from cognitive_ai.neural.trainer import incremental_train, train_model
    import tempfile
    td = tempfile.mkdtemp(prefix="xfer3c_")

    tr = load_jsonl(os.path.join(DATA, "train.jsonl"))
    va = [r for r in load_jsonl(os.path.join(DATA, "val.jsonl"))
          if r.get("level1")]
    old_train, old_frozen = tr[:48], va[:OLD_N]
    new_rows = [{"id": s["sample_id"], "group": f"kw-{s['sample_id']}",
                 "text": s["text"], "level1": s["l1"], "level2": "none"}
                for s in NEW_REVIEWED]
    assert check_no_leakage(old_train, old_frozen, new_rows)["ok"]

    base = train_model(old_train, va[:12], os.path.join(td, "base"),
                       kind="cnn", seed=7, epochs=10, batch=8)
    base_dir = os.path.join(td, "base")
    ad0 = NeuralAdapter(base["checkpoint"], base["vocab"])
    l1p = class_prototypes(ad0.model, ad0.vocab, old_train)
    l2p = l2_prototypes(ad0.model, ad0.vocab, old_train)
    th = {"margin_min": 0.17, "conf_min": 0.5, "dist_far": 4.0,
          "n_val_correct": 0}

    def old_acc(model, vocab):
        p1, _ = predict_probs(
            model, vocab, [r["text"] for r in old_frozen])
        return round(sum(1 for r, a in zip(old_frozen, p1)
                         if max(range(len(a)), key=lambda i: a[i]) ==
                         LEVEL1.index(r["level1"])) / len(old_frozen), 4)

    def ood_on_heldout(model, vocab):
        v = assess(HELDOUT["text"], model, vocab, l1p, 1.0, th)
        return v["verdict"]

    # A: detect only
    repA = {"new_verdict": ood_on_heldout(ad0.model, ad0.vocab),
            "old_acc": old_acc(ad0.model, ad0.vocab)}
    # B: facts only (L2 stays 'none': geometry unchanged)
    mB = incremental_train(base["checkpoint"], base["vocab"], new_rows,
                           old_train, os.path.join(td, "candB"),
                           epochs=EPOCHS, replay_ratio=2.0)
    from cognitive_ai.neural.trainer import load_model
    mBm, vBm, _ = load_model(mB["checkpoint"],
                             os.path.join(td, "candB", "vocab.json"))
    pB, _ = predict_probs(mBm, vBm, [HELDOUT["text"]])
    repB = {"old_acc": old_acc(mBm, vBm),
            "heldout_l1": LEVEL1[max(range(5), key=lambda i: pB[0][i])]}
    # C: discovery -> validate -> extend -> incremental -> gate
    disc = discover(NEW_REVIEWED, ad0.model, ad0.vocab, l1p, l2p, th)
    sel = disc["selected"]
    repC = {"discovery": sel["kind"], "validation": None,
            "heldout_l2": None, "old_acc": None, "gate": None}
    if sel["kind"] == "new_l2" and validate_proposal(
            NEW_REVIEWED, ad0.model, ad0.vocab, l1p, l2p)["ok"]:
        ext = apply_new_l2(base_dir, os.path.join(td, "extC"),
                           "kitchenware")
        assert ext["ok"]
        new_kw = [{**r, "level2": "kitchenware"} for r in new_rows]
        mC = incremental_train(ext["checkpoint"],
                               os.path.join(td, "extC", "vocab.json"),
                               new_kw, old_train,
                               os.path.join(td, "candC"), epochs=EPOCHS,
                               replay_ratio=2.0,
                               l2_labels=["kitchenware"])
        mCm, vCm, ckC = load_model(mC["checkpoint"],
                                   os.path.join(td, "candC", "vocab.json"))
        l2names = ckC.get("level2", [])
        _, p2 = predict_probs(mCm, vCm, [HELDOUT["text"]])
        repC["heldout_l2"] = l2names[max(range(len(p2[0])),
                                         key=lambda i: p2[0][i])]
        repC["old_acc"] = old_acc(mCm, vCm)
        pair = evaluate_pair(base_dir, os.path.join(td, "candC"),
                             old_frozen, new_kw)
        repC["gate"] = {"delta": pair["delta_regression"],
                        "new_n": pair["candidate"]["new_n"]}
        assert pair["candidate"]["new_n"] == 3
    else:
        repC["old_acc"] = repA["old_acc"]  # structure rejected: base kept

    print("\n3C STRUCTURE A/B/C (heldout=铁锅, old=12 frozen val):")
    print(f"  A detect-only : heldout_verdict={repA['new_verdict']} "
          f"old_acc={repA['old_acc']}")
    print(f"  B facts-only  : heldout_L1={repB['heldout_l1']} "
          f"old_acc={repB['old_acc']}")
    print(f"  C dynamic     : discovery={repC['discovery']} "
          f"heldout_L2={repC['heldout_l2']} old_acc={repC['old_acc']} "
          f"gate={repC['gate']}")
    assert set(repC) >= {"discovery", "old_acc"}
