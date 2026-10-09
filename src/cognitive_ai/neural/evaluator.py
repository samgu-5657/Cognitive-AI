"""Independent test evaluation: accuracy, P/R/F1, confusion, rejection.

Also ships the keyword RULE BASELINE so every report compares the net
against the existing symbolic parser on identical inputs.
"""
from __future__ import annotations

from . import LEVEL1, LEVEL2, require_torch

# Generic character cues per level-1 class. Deliberately coarse: this is the
# honest symbolic baseline, not a tuned competitor.
BASELINE_CUES = {
    "entity": list("杯车书桌椅机衣房刀锤箱包灯镜碗瓶剪扇蕉床"),
    "event": list("雨雪风震会赛祸灾电考节婚礼雹"),
    "action": list("跑走游驾吃喝唱读写画睡爬舞步"),
    "attribute": list("红色蓝绿圆方冷热大小高矮明暗软硬"),
    "abstract": list("爱恨情理学义德法梦忆史观"),
}


def baseline_predict(text: str):
    scores = {k: sum(ch in text for ch in cues)
              for k, cues in BASELINE_CUES.items()}
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    if ranked[0] == 0 or (len(ranked) > 1 and ranked[0] == ranked[1]):
        return {"label": "unknown", "confidence": 0.0, "scores": scores}
    return {"label": best, "confidence": min(1.0, 0.3 + 0.1 * ranked[0]),
            "scores": scores}


def predict_probs(model, vocab, texts: list[str]):
    torch = require_torch()
    import torch.nn.functional as F
    x = torch.tensor([vocab.encode(t) for t in texts], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        o1, o2 = model(x)
    return (F.softmax(o1, 1).tolist(), F.softmax(o2, 1).tolist())


def prf_from_confusion(cm: list[list[int]]) -> dict:
    out = {}
    for i, name in enumerate(LEVEL1):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(cm))) - tp
        fn = sum(cm[i]) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[name] = {"precision": round(p, 3), "recall": round(r, 3),
                     "f1": round(f, 3), "support": sum(cm[i])}
    return out


def evaluate(model, vocab, rows: list[dict],
             unknown_threshold: float = 0.6) -> dict:
    """Evaluate on labeled rows + unknown rejection. No training here."""
    from .dataset import labeled, unknowns
    lab = labeled(rows)
    unk = unknowns(rows)
    p1, p2 = predict_probs(model, vocab, [r["text"] for r in lab])
    n = len(LEVEL1)
    cm = [[0] * n for _ in range(n)]
    correct_l1 = correct_l2 = 0
    for r, a, b in zip(lab, p1, p2):
        pred = int(max(range(n), key=lambda i: a[i]))
        true = LEVEL1.index(r["level1"])
        cm[true][pred] += 1
        correct_l1 += (pred == true)
        l2 = r.get("level2") or "none"
        if LEVEL2.index(l2) == int(max(range(len(LEVEL2)),
                                       key=lambda i: b[i])):
            correct_l2 += 1
    rej = 0
    if unk:
        pu, _ = predict_probs(model, vocab, [r["text"] for r in unk])
        rej = sum(1 for a in pu if max(a) < unknown_threshold)
    base_ok = sum(1 for r in lab
                  if baseline_predict(r["text"])["label"] == r["level1"])
    return {
        "n": len(lab), "n_unknown": len(unk),
        "accuracy_l1": round(correct_l1 / len(lab), 4) if lab else 0.0,
        "accuracy_l2": round(correct_l2 / len(lab), 4) if lab else 0.0,
        "per_class": prf_from_confusion(cm),
        "confusion": {"labels": LEVEL1, "matrix": cm},
        "unknown_rejection_rate": round(rej / len(unk), 4) if unk else None,
        "unknown_threshold": unknown_threshold,
        "baseline_accuracy_l1": round(base_ok / len(lab), 4) if lab else 0.0,
    }
