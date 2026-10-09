"""Unknown-sample quarantine queue (Phase 3B).

Layout: one JSON object per line in a queue file (default
`data/neural/queue/unknown.jsonl`, git-ignored runtime state).

Record schema:
  sample_id, text, norm_text, model_version, l1_pred, l2_pred,
  confidence, reason, source, created_at, dup_of,
  label (null until reviewed), review {status, l1, l2, reviewer,
  reviewed_at, history: [{action, at, by, detail}]}.

Rules enforced here, not by convention:
- model predictions are stored SEPARATELY from labels and can never
  become labels implicitly (`label` starts null; only review() sets it);
- normalized-text duplicates collapse to `dup_of` (first record wins);
- without a label a sample can never enter supervised training data
  (see trainer/CLI guards, tested).
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone

from . import LEVEL1, LEVEL2

PENDING, ACCEPTED, REJECTED = "pending", "accepted", "rejected"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def detect_reason(l1_pred, confidence: float, threshold: float,
                  text: str, vocab_size: int = 0, unk_rate: float = 0.0) -> str:
    if confidence < threshold:
        return "low_confidence"
    if unk_rate >= 0.5:
        return "mostly_unknown_chars"
    return "manual"


class UnknownQueue:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # ---- storage ----
    def _read_all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]

    def _write_all(self, rows: list[dict]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    # ---- intake ----
    def add(self, text: str, model_version: str, l1_pred, l2_pred,
            confidence: float, reason: str, source: str = "predict",
            sample_id: str | None = None) -> dict:
        rows = self._read_all()
        norm = normalize(text)
        dup_of = next((r["sample_id"] for r in rows
                       if r["norm_text"] == norm), None)
        rec = {
            "sample_id": sample_id or uuid.uuid4().hex[:12],
            "text": text, "norm_text": norm,
            "model_version": model_version,
            "l1_pred": l1_pred, "l2_pred": l2_pred,
            "confidence": round(float(confidence), 4),
            "reason": reason, "source": source,
            "created_at": _now(), "dup_of": dup_of,
            "label": None,  # NEVER auto-filled from predictions
            "review": {"status": PENDING, "l1": None, "l2": None,
                       "reviewer": None, "reviewed_at": None,
                       "history": []},
        }
        rows.append(rec)
        self._write_all(rows)
        return rec

    def get(self, sample_id: str) -> dict | None:
        return next((r for r in self._read_all()
                     if r["sample_id"] == sample_id), None)

    def list(self, status: str | None = None,
             limit: int = 50) -> list[dict]:
        rows = self._read_all()
        if status:
            rows = [r for r in rows if r["review"]["status"] == status]
        return rows[:limit]

    def merge_duplicates(self) -> dict:
        """Point repeats at the first record; returns merge report."""
        rows = self._read_all()
        seen: dict[str, str] = {}
        merged = 0
        for r in rows:
            first = seen.setdefault(r["norm_text"], r["sample_id"])
            if r["sample_id"] != first and r["dup_of"] != first:
                r["dup_of"] = first
                merged += 1
        self._write_all(rows)
        return {"merged": merged, "total": len(rows)}

    # ---- review (the ONLY path to labels) ----
    def review(self, sample_id: str, accept: bool,
              l1: str | None = None, l2: str | None = None,
              reviewer: str = "operator") -> dict:
        rows = self._read_all()
        rec = next((r for r in rows if r["sample_id"] == sample_id), None)
        if rec is None:
            return {"ok": False, "error": "sample not found"}
        if rec["dup_of"]:
            return {"ok": False,
                    "error": f"duplicate of {rec['dup_of']}; review the original"}
        hist = rec["review"]["history"]
        if accept:
            if l1 not in LEVEL1:
                return {"ok": False,
                        "error": f"explicit L1 label required, got {l1!r}"}
            if l2 is not None and l2 not in LEVEL2:
                return {"ok": False,
                        "error": f"unknown L2 label {l2!r}; extend labels first"}
            rec["label"] = {"l1": l1, "l2": l2 or "none"}
            rec["review"].update({"status": ACCEPTED, "l1": l1,
                                  "l2": l2 or "none", "reviewer": reviewer,
                                  "reviewed_at": _now()})
            hist.append({"action": "accept", "at": _now(), "by": reviewer,
                         "detail": {"l1": l1, "l2": l2 or "none"}})
        else:
            rec["review"].update({"status": REJECTED, "reviewer": reviewer,
                                  "reviewed_at": _now()})
            hist.append({"action": "reject", "at": _now(), "by": reviewer,
                         "detail": {}})
        self._write_all(rows)
        return {"ok": True, "sample_id": sample_id,
                "status": rec["review"]["status"]}

    def accepted(self) -> list[dict]:
        return [r for r in self._read_all()
                if r["review"]["status"] == ACCEPTED and r["label"]
                and not r["dup_of"]]
