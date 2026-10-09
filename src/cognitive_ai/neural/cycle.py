"""Continual-learning cycle driver (Phase 3B, Task 7).

One cycle (never one training per unknown input):
  1. predict input via the adapter (active checkpoint);
  2. low-confidence/unknown -> UnknownQueue + episodic experience;
  3. when accepted queue >= batch_min AND train budget allows:
     incremental_train (reviewed + replay) -> candidate dir;
  4. gate.evaluate_pair on the FROZEN regression set -> decide;
  5. promote pointer swap only on pass, else keep active + record.

Three update kinds are recorded SEPARATELY, never conflated:
  - neural_params: model pointer change (models/active.json);
  - knowledge_graph: none in this loop (only via existing evidence gates);
  - episodic: experiences rows for every step above.
"""
from __future__ import annotations

import os
import time

from .adapter import NeuralAdapter, UNKNOWN_THRESHOLD
from .gate import decide, evaluate_pair, promote, read_active
from .queue import UnknownQueue


def cycle_predict(adapter: NeuralAdapter, queue: UnknownQueue, memory,
                  text: str, source: str = "cycle",
                  ood: dict | None = None) -> dict:
    """Step 1-2 + 4(partial): predict, quarantine unknowns, log experience.

    With ood artifacts (prototypes/thresholds/second model), the 5-way
    verdict decides: anything but known_confident is quarantined.
    Without them, falls back to the adapter maxprob threshold.
    """
    if ood and ood.get("protos") and ood.get("thresholds"):
        from .ood import assess
        second = ood.get("second")
        verdict = assess(text, adapter.model, adapter.vocab,
                         ood["protos"], ood.get("temperature", 1.0),
                         ood["thresholds"],
                         second=second)
        cand = {"concept_type": None, "candidate_category": None,
                "confidence": verdict.get("maxprob", 0.0),
                "model_version": adapter.version,
                "verdict": ("known" if verdict["verdict"] == "known_confident"
                            else "unknown"),
                "ood_verdict": verdict["verdict"],
                "ood_reasons": verdict.get("reasons", []),
                "predicted": verdict.get("predicted"),
                "input": text, "level2": None, "level2_confidence": 0.0}
    else:
        cand = adapter.predict(text)
    memory.add_experience(
        "observation", f"neural saw '{text}' -> {cand['verdict']}",
        {"candidate": cand, "model_version": adapter.version})
    queued = None
    if cand["verdict"] == "unknown":
        queued = queue.add(
            text, adapter.version, cand.get("predicted"), None,
            cand["confidence"],
            reason=cand.get("ood_verdict", "low_confidence"), source=source)
    return {"candidate": cand, "queued": queued}


def maybe_train(memory, base_tag: str, models_root: str, queue_path: str,
                train_path: str, out_tag: str, batch_min: int = 5,
                replay_ratio: float = 2.0, epochs: int = 30,
                seed: int = 11, max_train_seconds: float = 600.0,
                policy=None) -> dict:
    """Steps 3-5 + 8-10: gated batch training. Returns a full audit trail."""
    from .dataset import check_no_leakage, load_jsonl
    from .trainer import incremental_train
    t0 = time.time()
    queue = UnknownQueue(queue_path)
    accepted = queue.accepted()
    if len(accepted) < batch_min:
        return {"ok": False, "action": "wait_for_batch",
                "accepted": len(accepted), "batch_min": batch_min}
    new_rows = [{"id": r["sample_id"], "group": f"rev-{r['sample_id']}",
                 "text": r["text"], "level1": r["label"]["l1"],
                 "level2": r["label"]["l2"], "source": "reviewed"}
                for r in accepted]
    replay_rows = [r for r in load_jsonl(train_path) if r.get("level1")]
    leak = check_no_leakage(replay_rows, [], new_rows)
    if not leak["ok"]:
        return {"ok": False, "action": "rejected_leakage", "leak": leak}
    base_dir = os.path.join(models_root, base_tag)
    out_dir = os.path.join(models_root, out_tag)
    vocab = os.path.join(base_dir, "vocab.json")
    ckpt = os.path.join(base_dir, "best.pt")
    if not os.path.exists(ckpt):
        return {"ok": False, "action": "missing_base",
                "base": base_tag}
    meta = incremental_train(ckpt, vocab, new_rows, replay_rows, out_dir,
                             epochs=epochs, seed=seed,
                             replay_ratio=replay_ratio)
    elapsed = time.time() - t0
    if elapsed > max_train_seconds:
        return {"ok": False, "action": "budget_exceeded",
                "elapsed_s": round(elapsed, 1), "meta": meta}
    # gate on frozen regression rows (same rows both sides)
    reg_rows = [r for r in load_jsonl(train_path.replace(
        "train.jsonl", "val.jsonl")) if r.get("level1")]
    pair = evaluate_pair(base_dir, out_dir, reg_rows, new_rows)
    decision = decide(pair, policy)
    result = {"ok": True, "action": "promoted" if decision["promote"]
              else "rejected_by_gate",
              "meta": meta, "pair": pair, "decision": decision,
              "elapsed_s": round(elapsed, 1)}
    if decision["promote"]:
        result["promotion"] = promote(models_root, out_tag, pair,
                                      decision, data_version="reviewed+replay")
    memory.add_experience(
        "evaluation", f"continual cycle: {result['action']}",
        {"neural_params": result["action"],
         "knowledge_graph": "unchanged (separate gates only)",
         "base": base_tag, "candidate": out_tag,
         "delta_regression": pair["delta_regression"],
         "reason": decision["reason"]})
    return result
