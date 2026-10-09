"""JSONL sample loading + group-disjoint split validation (no leakage)."""
from __future__ import annotations

import json


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def groups(rows: list[dict]) -> set[str]:
    return {r["group"] for r in rows}


def texts(rows: list[dict]) -> set[str]:
    return {r["text"].strip() for r in rows}


def check_no_leakage(train: list[dict], val: list[dict],
                     test: list[dict]) -> dict:
    """Assert paraphrase groups AND exact texts are disjoint across splits."""
    gt, gv, gte = groups(train), groups(val), groups(test)
    tt, tv, tte = texts(train), texts(val), texts(test)
    report = {
        "group_overlap_train_val": sorted(gt & gv),
        "group_overlap_train_test": sorted(gt & gte),
        "group_overlap_val_test": sorted(gv & gte),
        "text_overlap_train_val": sorted(tt & tv),
        "text_overlap_train_test": sorted(tt & tte),
        "text_overlap_val_test": sorted(tv & tte),
    }
    report["ok"] = not any(report.values())
    return report


def labeled(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r.get("unknown") and r.get("level1")]


def unknowns(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("unknown")]
