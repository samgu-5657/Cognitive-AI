"""Open-world unknown detection.

Signals (all from REAL model outputs, never hardcoded words):
  maxprob   - top-1 probability. A temperature is fit on VAL, but the CLI
              passes T=1.0 (raw probabilities): the fit lowered NLL without
              improving ECE or accept-precision, so it is diagnostic only;
  margin    - top1 minus top2 probability;
  dist      - Euclidean distance to the nearest L1 class prototype
              (mean penultimate feature of correctly-classified TRAIN rows);
  unk_rate  - fraction of input chars missing from the vocabulary.

Five verdicts:
  known_confident     strong, agreeing evidence;
  known_uncertain     weak margin/confidence but near a known prototype;
  similar_but_unproven conflicting evidence (predicted class != nearest
                      prototype, or L1/L2 heads disagree strongly);
  suspected_novel     far from every prototype with flat scores;
  unjudgeable         input quality too poor (empty / mostly-unknown chars).

Thresholds are tuned on VAL ROWS ONLY (tune_thresholds); TEST only reports.
Overconfident compositional errors (e.g. near-misses sharing all chars
with a known class) may still slip through -- reported as failures, not
patched with word lists.
"""
from __future__ import annotations

import json
import math
import os

from . import LEVEL1

UNJUDGEABLE_UNK_RATE = 1.0


def class_prototypes(model, vocab, train_rows: list[dict]) -> dict:
    """Mean feature per L1 class + spread. Train rows only."""
    from .dataset import labeled
    from .evaluator import predict_probs
    torch = _torch()
    rows = labeled(train_rows)
    p1, _ = predict_probs(model, vocab, [r["text"] for r in rows])
    feats = model.features(
        torch.tensor([vocab.encode(r["text"]) for r in rows])).tolist()
    out = {}
    for i, name in enumerate(LEVEL1):
        vecs = [f for r, f, a in zip(rows, feats, p1)
                if r["level1"] == name
                and max(range(len(a)), key=lambda k: a[k]) == i]
        if not vecs:
            continue
        mean = [sum(c) / len(vecs) for c in zip(*vecs)]
        spread = sum(_dist(f, mean) for f in vecs) / len(vecs)
        out[name] = {"mean": mean, "spread": spread, "n": len(vecs)}
    return out


def _torch():
    from . import require_torch
    return require_torch()


def _dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _softmax(logits):
    m = max(logits)
    ex = [math.exp(v - m) for v in logits]
    s = sum(ex)
    return [v / s for v in ex]


def fit_temperature(model, vocab, val_rows: list[dict]) -> dict:
    """Grid-search T minimizing NLL on VAL rows. Returns calibration report."""
    from .dataset import labeled
    torch = _torch()
    rows = labeled(val_rows)
    x = torch.tensor([vocab.encode(r["text"]) for r in rows])
    y = torch.tensor([LEVEL1.index(r["level1"]) for r in rows])
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
    logits = logits.tolist()

    def nll(t):
        return -sum(math.log(_softmax([v / t for v in lg])[yi] + 1e-12)
                    for lg, yi in zip(logits, y.tolist())) / len(logits)

    best_t, best_n = 1.0, nll(1.0)
    t = 0.2
    while t <= 5.0:
        v = nll(t)
        if v < best_n:
            best_n, best_t = v, t
        t = round(t + 0.1, 10)
    ece_before = _ece([_softmax(lg) for lg in logits],
                      y.tolist())
    ece_after = _ece([_softmax([v / best_t for v in lg]) for lg in logits],
                     y.tolist())
    return {"temperature": round(best_t, 2), "nll": round(best_n, 4),
            "ece_before": ece_before, "ece_after": ece_after,
            "n": len(rows)}


def _ece(probs, labels, bins: int = 5) -> float:
    n = len(probs)
    if not n:
        return 0.0
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo < max(p) <= hi]
        if not idx:
            continue
        acc = sum(1 for i in idx
                  if max(range(len(probs[i])),
                         key=lambda k: probs[i][k]) == labels[i]) / len(idx)
        conf = sum(max(probs[i]) for i in idx) / len(idx)
        ece += abs(acc - conf) * len(idx) / n
    return round(ece, 4)


def tune_thresholds(model, vocab, protos: dict,
                    val_rows: list[dict], ece: float = 0.0) -> dict:
    """Derive thresholds from VAL statistics (correctly-classified rows).

    margin_min: max(25th pct of correct-row margins, ECE). The ECE floor is
                principled: a top-1 lead smaller than the model's mean
                calibration error is noise, not evidence. (On tiny val sets
                the raw percentile collapses to ~0 and would never fire.)
    dist_far:   95th percentile of correct-row distances to the PREDICTED
                class prototype (farther = possibly novel).
    """
    from .dataset import labeled
    torch = _torch()
    rows = labeled(val_rows)
    x = torch.tensor([vocab.encode(r["text"]) for r in rows])
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
    feats = model.features(x).tolist()
    margins, dists = [], []
    for lg, f, r in zip(logits.tolist(), feats, rows):
        p = _softmax(lg)
        top = sorted(range(len(p)), key=lambda i: -p[i])
        if LEVEL1[top[0]] != r["level1"]:
            continue  # only correct rows set the bar
        margins.append(p[top[0]] - p[top[1]])
        dists.append(_dist(f, protos[LEVEL1[top[0]]]["mean"]))
    margins.sort()
    dists.sort()

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0
    return {"margin_min": round(max(pct(margins, 0.25), ece), 4),
            "conf_min": 0.5,
            "dist_far": round(pct(dists, 0.95), 4),
            "n_val_correct": len(margins)}


def unk_rate(text: str, vocab) -> float:
    if not text:
        return 1.0
    return sum(1 for ch in text if ch not in vocab.stoi) / len(text)


def assess(text: str, model, vocab, protos: dict, temperature: float,
           thresholds: dict, second: tuple | None = None) -> dict:
    """Full 5-way verdict with every intermediate score recorded.

    `second` is an optional (model, vocab) pair forming a 2-model ensemble:
    disagreement on the top-1 class vetoes known_confident. The two nets
    were trained independently (different architectures/seeds), so shared
    errors are memorized biases, flagged as failures in reports.
    """
    torch = _torch()
    ur = unk_rate(text, vocab)
    if not text or ur >= UNJUDGEABLE_UNK_RATE:
        return {"verdict": "unjudgeable", "maxprob": 0.0, "margin": 0.0,
                "dist_nearest": None, "nearest_proto": None,
                "unk_rate": round(ur, 3),
                "reasons": ["input quality too poor to judge"],
                "predicted": None}
    x = torch.tensor([vocab.encode(text)])
    model.eval()
    with torch.no_grad():
        logits, l2 = model(x)
    p = _softmax([v / temperature for v in logits.tolist()[0]])
    top = sorted(range(len(p)), key=lambda i: -p[i])
    margin = p[top[0]] - p[top[1]]
    f = model.features(x).tolist()[0]
    ds = sorted(((n, _dist(f, pr["mean"])) for n, pr in protos.items()),
                key=lambda kv: kv[1])
    nearest, dist = ds[0][0], ds[0][1]
    pred = LEVEL1[top[0]]
    reasons = [f"maxprob={p[top[0]]:.3f}", f"margin={margin:.3f}",
               f"nearest={nearest}:{dist:.2f}"]
    t = thresholds
    ensemble_ok = True
    if second is not None:
        m2, v2 = second
        x2 = torch.tensor([v2.encode(text)])
        m2.eval()
        with torch.no_grad():
            lg2, _ = m2(x2)
        p2 = _softmax([v / temperature for v in lg2.tolist()[0]])
        top2 = max(range(len(p2)), key=lambda i: p2[i])
        ensemble_ok = (LEVEL1[top2] == pred)
        reasons.append(f"ensemble_top={LEVEL1[top2]}:{p2[top2]:.3f} "
                       f"agree={ensemble_ok}")
    if margin < t["margin_min"] or p[top[0]] < t["conf_min"]:
        if dist > t["dist_far"]:
            verdict = "suspected_novel"
            reasons.append("flat scores AND far from every prototype")
        elif pred != nearest or not ensemble_ok:
            verdict = "similar_but_unproven"
            reasons.append("weak evidence with conflicting signals")
        else:
            verdict = "known_uncertain"
            reasons.append("weak evidence near a known prototype")
    elif (pred != nearest or not ensemble_ok) and dist <= t["dist_far"]:
        verdict = "similar_but_unproven"
        reasons.append("confident but counter-evidence exists "
                       "(twin prototype or ensemble split)")
    elif dist > t["dist_far"]:
        verdict = "suspected_novel"
        reasons.append("confident yet far from every prototype")
    else:
        verdict = "known_confident"
        reasons.append("strong agreeing evidence")
    return {"verdict": verdict, "predicted": pred,
            "maxprob": round(p[top[0]], 4), "margin": round(margin, 4),
            "dist_nearest": round(dist, 3), "nearest_proto": nearest,
            "unk_rate": round(ur, 3), "reasons": reasons,
            "l2": int(max(range(len(l2.tolist()[0])),
                           key=lambda i: l2.tolist()[0][i]))}


def save_artifacts(out_dir: str, protos: dict, calibration: dict,
                   thresholds: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, obj in (("prototypes.json", protos),
                      ("calibration.json", calibration),
                      ("ood_thresholds.json", thresholds)):
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False)
        paths[name] = p
    return paths


def load_artifacts(model_dir: str) -> dict:
    out = {}
    for name in ("prototypes.json", "calibration.json",
                 "ood_thresholds.json"):
        p = os.path.join(model_dir, name)
        out[name] = None
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                out[name] = json.load(fh)
    return out
