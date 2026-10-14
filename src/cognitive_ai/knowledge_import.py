"""Batch knowledge import: preview (read-only) then explicit stage.

TXT: one statement per non-empty line. JSONL: one {"text": ...} object
per line with optional source_id/source_group. A whole file defaults
to one source group; lines never count as independent sources.

Preview only parses and grounds (Learner.classify_utterance, zero
writes). Staging calls Learner.propose_from_utterance for ready items
only -- never confirm, never direct knowledge writes.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from .learner import Learner
from .memory import Memory
from .models import (KnowledgeImportCategory, KnowledgeImportItem,
                     KnowledgeImportPreview)

READY = KnowledgeImportCategory.READY_FOR_CANDIDATE.value
KNOWN = KnowledgeImportCategory.ALREADY_KNOWN.value
DUP = KnowledgeImportCategory.DUPLICATE_IN_BATCH.value
UNSUP = KnowledgeImportCategory.UNSUPPORTED.value
AMBIG = KnowledgeImportCategory.AMBIGUOUS.value
CONFL = KnowledgeImportCategory.CONFLICTED.value
UNKNOWN = KnowledgeImportCategory.UNKNOWN.value
INVALID = KnowledgeImportCategory.INVALID_INPUT.value

CATEGORIES = (READY, KNOWN, DUP, UNSUP, AMBIG, CONFL, UNKNOWN, INVALID)


def classify_preview_item(cand: dict) -> tuple:
    """Map a classify_utterance result to (category, reason).

    Order matters: ambiguity and unstorable shapes first, then known,
    then live disputes, then confirmable kinds (denials included),
    then plain unknown. Unknown verdicts alone never force a block.
    """
    parse = cand.get("parse", {}) or {}
    grounding = cand.get("grounding", {}) or {}
    kind = cand.get("kind", "")
    verdict = grounding.get("verdict", "")
    if parse.get("ambiguity") or verdict == "ambiguous":
        return AMBIG, "input ambiguous, needs clarification"
    if parse.get("predicate") == "unknown" or kind == "unusable":
        return UNSUP, cand.get("reason") or "not safely storable"
    if kind == "already_known":
        return KNOWN, cand.get("reason") or "already in knowledge"
    if verdict == "conflicted":
        return CONFL, "evidence disputes this statement"
    if cand.get("can_confirm"):
        return READY, cand.get("reason") or "confirmable shape"
    if verdict == "unknown":
        return UNKNOWN, "no supporting or disputing evidence"
    return UNSUP, cand.get("reason") or "not safely storable"


def _read_lines(path: str) -> tuple:
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        return None, f"cannot read file: {e}"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return None, f"not valid UTF-8: {e}"
    return text.splitlines(), ""


def _parse_txt(lines: list, file_id: str, group: str) -> tuple:
    items = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        items.append({"line_no": lineno, "text": line.strip(),
                      "source_id": f"{file_id}:{lineno}",
                      "source_group": group})
    return items, ""


def _parse_jsonl(lines: list, file_id: str, group: str) -> tuple:
    items = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError as e:
            items.append({"line_no": lineno, "text": line.strip(),
                          "source_id": f"{file_id}:{lineno}",
                          "source_group": group,
                          "invalid": f"bad JSON: {e}"})
            continue
        if not isinstance(obj, dict):
            items.append({"line_no": lineno, "text": line.strip(),
                          "source_id": f"{file_id}:{lineno}",
                          "source_group": group,
                          "invalid": "line is not a JSON object"})
            continue
        text = obj.get("text")
        if not isinstance(text, str) or not text.strip():
            items.append({"line_no": lineno, "text": line.strip(),
                          "source_id": f"{file_id}:{lineno}",
                          "source_group": group,
                          "invalid": "text must be a non-empty string"})
            continue
        source_id = obj.get("source_id")
        source_group = obj.get("source_group")
        if source_id is not None and not isinstance(source_id, str):
            items.append({"line_no": lineno, "text": text.strip(),
                          "source_id": f"{file_id}:{lineno}",
                          "source_group": group,
                          "invalid": "source_id must be a string"})
            continue
        if source_group is not None and not isinstance(source_group,
                                                       str):
            items.append({"line_no": lineno, "text": text.strip(),
                          "source_id": f"{file_id}:{lineno}",
                          "source_group": group,
                          "invalid": "source_group must be a string"})
            continue
        items.append({"line_no": lineno, "text": text.strip(),
                      "source_id": source_id or f"{file_id}:{lineno}",
                      "source_group": source_group or group})
    return items, ""


def preview_import(path: str, memory: Memory) -> KnowledgeImportPreview:
    """Preview a batch file. Read-only: creates no candidates and
    writes nothing (parse + ground only)."""
    file_id = os.path.basename(path)
    group = file_id
    suffix = os.path.splitext(file_id)[1].lower()
    lines, error = _read_lines(path)
    if lines is None:
        return KnowledgeImportPreview(
            file_id=file_id, format="unknown", error=error)
    if suffix == ".txt":
        raw_items, error = _parse_txt(lines, file_id, group)
        fmt = "txt"
    elif suffix == ".jsonl":
        raw_items, error = _parse_jsonl(lines, file_id, group)
        fmt = "jsonl"
    else:
        return KnowledgeImportPreview(
            file_id=file_id, format="unknown",
            error=f"unsupported format {suffix!r}: use .txt or .jsonl")
    if error:
        return KnowledgeImportPreview(
            file_id=file_id, format=fmt, error=error)
    learner = Learner(memory, "")
    seen: dict = {}
    built = []
    counts = {c: 0 for c in CATEGORIES}
    for entry in raw_items:
        if "invalid" in entry:
            category, reason = INVALID, entry["invalid"]
            parse, grounding = {}, {}
        elif entry["text"] in seen:
            category = DUP
            reason = f"duplicate of line {seen[entry['text']]}"
            parse, grounding = {}, {}
        else:
            cand = learner.classify_utterance(entry["text"])
            category, reason = classify_preview_item(cand)
            parse, grounding = cand.get("parse", {}), \
                cand.get("grounding", {})
            seen[entry["text"]] = entry["line_no"]
        counts[category] += 1
        built.append(KnowledgeImportItem(
            line_no=entry["line_no"], text=entry["text"],
            source_id=entry["source_id"],
            source_group=entry["source_group"], category=category,
            parse=parse, grounding=grounding, reason=reason))
    return KnowledgeImportPreview(
        file_id=file_id, format=fmt, total_lines=len(lines),
        counts=counts, items=tuple(built))


def stage_import(preview: KnowledgeImportPreview, learner: Learner,
                 memory: Memory) -> dict:
    """Stage ready items through Learner.propose_from_utterance.

    Re-classifies every item against the live database (never trusts a
    stale preview), skips batch duplicates and already-pending
    utterances, never confirms, and records one result per item.
    """
    results = []
    staged = skipped = failed = 0
    for item in preview.items:
        if item.category != READY:
            results.append({"line_no": item.line_no, "text": item.text,
                            "source_id": item.source_id,
                            "outcome": "skipped",
                            "reason": f"not staged: {item.category}"})
            skipped += 1
            continue
        try:
            live = learner.classify_utterance(item.text)
            category, _reason = classify_preview_item(live)
            if category != READY:
                results.append({
                    "line_no": item.line_no, "text": item.text,
                    "source_id": item.source_id, "outcome": "skipped",
                    "reason": f"now {category}, not staged"})
                skipped += 1
                continue
            pending = [p for p in memory.list_pending_candidates(
                status="proposed") if p.get("utterance") == item.text]
            if pending:
                results.append({
                    "line_no": item.line_no, "text": item.text,
                    "source_id": item.source_id, "outcome": "skipped",
                    "reason": "already pending as candidate "
                    f"#{pending[0]['id']}"})
                skipped += 1
                continue
            cand = learner.propose_from_utterance(
                item.text,
                source={"source_id": item.source_id,
                        "source_group": item.source_group,
                        "origin": "import"})
            results.append({"line_no": item.line_no, "text": item.text,
                            "source_id": item.source_id,
                            "outcome": "staged",
                            "candidate_id": cand["id"],
                            "reason": cand.get("reason", "")})
            staged += 1
        except Exception as e:
            results.append({"line_no": item.line_no, "text": item.text,
                            "source_id": item.source_id,
                            "outcome": "failed", "reason": str(e)})
            failed += 1
    return {"file_id": preview.file_id, "results": results,
            "staged": staged, "skipped": skipped, "failed": failed}
