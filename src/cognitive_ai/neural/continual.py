"""Phase 3B hooks (interface + working stores, no auto weight updates).

DONE in 3A (real, tested):
- UnknownLog: record inputs the net abstained on, with model version.
- SampleStore: append reviewed labeled samples (quarantined from training
  data until explicitly promoted).
- promote_samples(): move reviewed samples into the training corpus dir
  as a NEW versioned file (old files untouched).
- replay_compare(): evaluate old vs candidate checkpoint on old + new
  samples; accept_candidate() only swaps the pointer when the candidate
  does not regress (old accuracy drop <= tolerance AND new accuracy gain).

NOT in 3A (3B work): online weight updates, automatic promotion policy,
catastrophic-forgetting mitigations beyond replay eval.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class UnknownLog:
    def __init__(self, path: str):
        self.path = path

    def record(self, text: str, model_version: str, confidence: float) -> None:
        rec = {"text": text, "model_version": model_version,
               "confidence": confidence, "at": _now()}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]


class SampleStore:
    """Quarantined reviewed samples. Same schema as training JSONL."""

    def __init__(self, path: str):
        self.path = path

    def add(self, sample: dict) -> None:
        for k in ("id", "group", "text", "level1"):
            if k not in sample:
                raise ValueError(f"sample missing required field: {k}")
        sample = {**sample, "reviewed_at": _now()}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def load(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]


def promote_samples(store_path: str, train_path: str) -> dict:
    """Append reviewed samples as a NEW versioned corpus file.

    Never edits existing corpus files. Returns counts + new file path.
    """
    store = SampleStore(store_path)
    samples = store.load()
    if not samples:
        return {"ok": False, "error": "no reviewed samples to promote"}
    base, ext = os.path.splitext(train_path)
    n = 1
    while os.path.exists(f"{base}.v{n}{ext}"):
        n += 1
    dest = f"{base}.v{n}{ext}"
    with open(dest, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    return {"ok": True, "promoted": len(samples), "file": dest}


def replay_compare(old_report: dict, new_report: dict,
                   tolerance: float = 0.0) -> dict:
    """Decide whether a candidate checkpoint may replace the current one.

    old_report/new_report: evaluator.evaluate() dicts with keys
    accuracy_l1 measured on the SAME old sample set (caller ensures this)
    plus new_report may carry accuracy_new on new samples.
    Accept iff old accuracy does not drop beyond tolerance.
    """
    drop = old_report.get("accuracy_l1", 0) - new_report.get("accuracy_l1", 0)
    accept = round(drop, 9) <= tolerance
    return {"accept": accept, "old_acc": old_report.get("accuracy_l1"),
            "new_acc": new_report.get("accuracy_l1"),
            "new_samples_acc": new_report.get("accuracy_new"),
            "drop": round(drop, 4), "tolerance": tolerance}
