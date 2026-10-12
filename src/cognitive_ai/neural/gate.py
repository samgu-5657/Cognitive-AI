"""Promotion gate for candidate checkpoints.

Pipeline: candidate dir (from incremental_train) -> evaluate on the FROZEN
regression set (same rows for base and candidate) + new-sample set ->
PromotionPolicy decides -> pointer swap in models/active.json, never an
in-place overwrite. Previous pointer is kept for rollback.

Default policy protects old knowledge:
- regression L1 accuracy drop must stay within REGRESSION_TOLERANCE
  (default 0.02: val-sized frozen sets carry ~+/-2% sampling noise;
  the tolerance is a config, not a per-run invention);
- at least MIN_NEW_SAMPLES reviewed new samples must exist, else the
  evidence is insufficient and the candidate is rejected without promotion;
- new-sample accuracy is REPORTED, not gated (small new sets are noisy).

The FINAL held-out test set is never used here; only for end-of-phase
reporting. Checkpoint selection already used val; the gate reuses the same
frozen regression rows for both models so the comparison is apples-to-apples.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from . import LEVEL1

REGRESSION_TOLERANCE = 0.02
MIN_NEW_SAMPLES = 5


@dataclass
class PromotionPolicy:
    regression_tolerance: float = REGRESSION_TOLERANCE
    min_new_samples: int = MIN_NEW_SAMPLES
    # Rationale: frozen regression sets here have ~50 rows, so single-sample
    # flips move accuracy by ~2%. Tolerance 0.02 absorbs one noisy flip while
    # still rejecting real degradation (>=2 flips). min_new_samples=5 keeps
    # tiny anecdotal batches from triggering promotions.


def expected_calibration_error(probs: list[list[float]],
                               labels: list[int], bins: int = 5) -> float:
    n = len(probs)
    if not n:
        return 0.0
    edges = [i / bins for i in range(bins + 1)]
    ece = 0.0
    for b in range(bins):
        idx = [i for i, p in enumerate(probs)
               if edges[b] < max(p) <= edges[b + 1] or
               (b == 0 and max(p) == 0.0)]
        if not idx:
            continue
        acc = sum(1 for i in idx
                  if max(range(len(probs[i])),
                         key=lambda k: probs[i][k]) == labels[i]) / len(idx)
        conf = sum(max(probs[i]) for i in idx) / len(idx)
        ece += abs(acc - conf) * len(idx) / n
    return round(ece, 4)


def evaluate_pair(base_dir: str, cand_dir: str, regression_rows: list[dict],
                  new_rows: list[dict], unknown_rows: list[dict] | None = None,
                  threshold: float = 0.6) -> dict:
    """Evaluate base vs candidate checkpoints on identical frozen rows."""
    from .dataset import labeled, unknowns
    from .evaluator import evaluate, predict_probs
    from .trainer import load_model

    def _load(d):
        for name in ("candidate.pt", "best.pt"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                v = os.path.join(d, "vocab.json")
                if not os.path.exists(p) or not os.path.exists(v):
                    raise FileNotFoundError(f"incomplete model dir: {d}")
                return load_model(p, v)
        raise FileNotFoundError(f"no checkpoint in {d}")

    reg = labeled(regression_rows)
    new = labeled(new_rows)
    unk = list(unknown_rows or []) + unknowns(regression_rows)
    out = {}
    for tag, d in (("base", base_dir), ("candidate", cand_dir)):
        model, vocab, _ = _load(d)
        rep = evaluate(model, vocab, reg, unknown_threshold=threshold)
        p1, _ = predict_probs(
            model, vocab, [r["text"] for r in (new or reg[:1])])
        new_acc = None
        if new:
            new_acc = round(sum(
                1 for r, a in zip(new, p1)
                if max(range(len(a)), key=lambda i: a[i]) ==
                LEVEL1.index(r["level1"])
            ) / len(new), 4)
        pu, _ = predict_probs(model, vocab, [r["text"] for r in unk]) \
            if unk else ([], [])
        out[tag] = {
            "regression": rep,
            "new_acc": new_acc, "new_n": len(new),
        }
        pr, _ = predict_probs(model, vocab, [r["text"] for r in reg])
        out[tag]["regression_ece"] = expected_calibration_error(
            pr, [LEVEL1.index(r["level1"]) for r in reg])
        out[tag]["reject_new"] = round(sum(
            1 for a in pu if max(a) < threshold) / len(pu), 4) if pu else None
    base_acc = out["base"]["regression"]["accuracy_l1"]
    cand_acc = out["candidate"]["regression"]["accuracy_l1"]
    out["delta_regression"] = round(cand_acc - base_acc, 4)
    out["forgetting"] = _forgetting(
        out["base"]["regression"]["per_class"],
        out["candidate"]["regression"]["per_class"])
    return out


def _forgetting(before: dict, after: dict) -> dict:
    return {k: round(before[k]["f1"] - after[k]["f1"], 3) for k in before}


def decide(pair_report: dict, policy: PromotionPolicy | None = None) -> dict:
    policy = policy or PromotionPolicy()
    new_n = pair_report["candidate"]["new_n"]
    if new_n < policy.min_new_samples:
        return {"promote": False,
                "reason": f"insufficient evidence: {new_n} new samples "
                          f"< {policy.min_new_samples}"}
    drop = -pair_report["delta_regression"]
    if round(drop, 9) > policy.regression_tolerance:
        return {"promote": False,
                "reason": f"regression drop {round(drop, 4)} exceeds "
                          f"tolerance {policy.regression_tolerance}"}
    return {"promote": True,
            "reason": f"regression delta {pair_report['delta_regression']} "
                      f"within tolerance; new_n={new_n}"}


# ---- active pointer (swap, never overwrite) ----
def active_path(models_root: str) -> str:
    return os.path.join(models_root, "active.json")


def read_active(models_root: str) -> dict | None:
    p = active_path(models_root)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def promote(models_root: str, candidate_tag: str, report: dict,
            decision: dict, data_version: str) -> dict:
    prev = read_active(models_root)
    new = {"tag": candidate_tag,
           "promoted_at": datetime.now(timezone.utc).isoformat(),
           "data_version": data_version,
           "regression_acc": report["candidate"]["regression"]["accuracy_l1"],
           "delta_regression": report["delta_regression"],
           "reason": decision.get("reason", ""),
           "previous": prev}
    with open(active_path(models_root), "w", encoding="utf-8") as fh:
        json.dump(new, fh, ensure_ascii=False, indent=2)
    return {"ok": True, "active": new}


def rollback_active(models_root: str) -> dict:
    cur = read_active(models_root)
    if not cur or not cur.get("previous"):
        return {"ok": False, "error": "no previous stable checkpoint"}
    with open(active_path(models_root), "w", encoding="utf-8") as fh:
        json.dump(cur["previous"], fh, ensure_ascii=False, indent=2)
    return {"ok": True, "active": cur["previous"]}
