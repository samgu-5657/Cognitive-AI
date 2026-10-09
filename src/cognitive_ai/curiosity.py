"""Curiosity: propose research questions from the KB state and score them.

Score formula (centralized, configurable via SCORE_WEIGHTS / configure()):
    score = w_info*info_gain + w_progress*progress + w_relevance*relevance
            + w_novelty*novelty - w_cost*cost - w_risk*risk
All factors in [0,1]; weights non-negative. Higher score = explore first.
"""
from __future__ import annotations

from typing import Any

from .memory import Memory
from .novelty import NoveltyDetector
from .policy import MAX_NEW_QUESTIONS_PER_STEP, MAX_RESEARCH_ATTEMPTS

SCORE_WEIGHTS: dict[str, float] = {
    "w_info": 1.0,
    "w_progress": 0.8,
    "w_relevance": 0.7,
    "w_novelty": 0.6,
    "w_cost": 0.5,
    "w_risk": 0.8,
}

IMPORTANT_CONCEPTS = {"cup", "container", "vessel", "drinkware"}


def configure(**kwargs: float) -> dict[str, float]:
    for k, v in kwargs.items():
        if k in SCORE_WEIGHTS:
            SCORE_WEIGHTS[k] = float(v)
    return dict(SCORE_WEIGHTS)


def score_question(factors: dict[str, float],
                   weights: dict[str, float] | None = None) -> float:
    w = weights or SCORE_WEIGHTS
    return (w["w_info"] * factors.get("info_gain", 0.0)
            + w["w_progress"] * factors.get("progress", 0.0)
            + w["w_relevance"] * factors.get("relevance", 0.0)
            + w["w_novelty"] * factors.get("novelty", 0.0)
            - w["w_cost"] * factors.get("cost", 0.0)
            - w["w_risk"] * factors.get("risk", 0.0))


def _factors_for(kind: str, target: str | None) -> dict[str, float]:
    base = {"info_gain": 0.6, "progress": 0.5, "relevance": 0.4,
            "novelty": 0.5, "cost": 0.3, "risk": 0.1}
    if kind == "conflict":
        base.update({"info_gain": 0.9, "progress": 0.8, "risk": 0.3})
    elif kind == "missing_attribute":
        base.update({"info_gain": 0.7, "progress": 0.6, "cost": 0.2})
    elif kind == "unexplained":
        base.update({"info_gain": 0.8, "novelty": 0.8})
    elif kind == "counterexample":
        base.update({"info_gain": 0.85, "progress": 0.7})
    if target in IMPORTANT_CONCEPTS:
        base["relevance"] = min(1.0, base["relevance"] + 0.3)
    return base


TEXT_TEMPLATES = {
    "missing_attribute": "What is the '{key}' of concept '{concept}'?",
    "no_relations": "What relations does concept '{concept}' participate in?",
    "no_attributes": "What attributes distinguish concept '{concept}'?",
    "orphan_concept": "Where does concept '{concept}' belong in the category hierarchy?",
    "attribute_conflict": "Which value of '{key}' for '{concept}' is correct: {values}?",
    "contested_claim": "How to resolve contested claim: {text}?",
    "unexplained": "How should unexplained observation '{name}' be categorized?",
}


class Curiosity:
    def __init__(self, memory: Memory):
        self.mem = memory
        self.detector = NoveltyDetector(memory)

    def generate(self, max_new: int = MAX_NEW_QUESTIONS_PER_STEP) -> list[dict]:
        """Reconcile the question queue with current gaps.

        - Existing (text, target) rows are NEVER duplicated; their score is
          refreshed instead.
        - At most `max_new` genuinely new rows are persisted per call, so
          `new_questions` counts real additions, not re-scans.
        Each returned item carries `is_new` (True only if just inserted).
        """
        import json
        gaps = self.detector.knowledge_gaps()
        questions: list[dict] = []
        for g in gaps:
            q = self._gap_to_question(g)
            if q:
                questions.append(q)
        # unexplained observations from recent experiences
        for e in self.mem.list_experiences(limit=20):
            try:
                d = json.loads(e["detail"]) if isinstance(e["detail"], str) else e["detail"]
            except Exception:
                d = {}
            if e["kind"] == "observation" and d.get("unexplained"):
                q = {"text": TEXT_TEMPLATES["unexplained"].format(
                        name=d.get("name", "?")),
                     "kind": "unexplained",
                     "target_id": None,
                     "factors": _factors_for("unexplained", None)}
                questions.append(q)
        # score + persist with true dedupe and a per-step insertion cap
        out = []
        new_count = 0
        for q in questions:
            q["score"] = round(score_question(q["factors"]), 3)
            existing = self.mem.find_question(q["text"], q.get("target_id"))
            if existing is not None:
                if existing["status"] in ("proposed", "queued"):
                    self.mem.update_question(int(existing["id"]),
                                             score=q["score"])
                out.append({"id": int(existing["id"]), "is_new": False, **q})
                continue
            if new_count >= max_new:
                continue  # gap persists; retried on a later step, not duplicated
            qid = self.mem.add_question(q["text"], q["kind"],
                                        q.get("target_id"), q["score"],
                                        "queued", {"factors": q["factors"]})
            new_count += 1
            out.append({"id": qid, "is_new": True, **q})
        out.sort(key=lambda x: -x["score"])
        return out

    def eligible_open(self, limit: int = 50) -> list[dict]:
        """Open questions ready for research now.

        Excludes resolved rows and rows in cooldown or past the maximum
        number of inconclusive attempts. Cooled-down questions are kept
        (never deleted, never force-studied).
        """
        import json
        import time
        now = time.time()
        out = []
        for q in self.mem.list_questions():
            if q["status"] not in ("proposed", "queued"):
                continue
            try:
                d = json.loads(q["detail"]) if isinstance(q["detail"], str) \
                    else (q["detail"] or {})
            except Exception:
                d = {}
            if not isinstance(d, dict):
                d = {}
            if float(d.get("cooldown_until", 0) or 0) > now:
                continue
            if int(d.get("attempts", 0) or 0) >= MAX_RESEARCH_ATTEMPTS:
                continue
            out.append(q)
            if len(out) >= limit:
                break
        return out

    def _gap_to_question(self, g: dict) -> dict | None:
        t = g.get("type")
        if t == "missing_attribute":
            text = TEXT_TEMPLATES[t].format(key=g["key"], concept=g["concept"])
            return {"text": text, "kind": "gap", "target_id": g["concept"],
                    "factors": _factors_for("missing_attribute", g["concept"])}
        if t in ("no_relations", "no_attributes", "orphan_concept"):
            text = TEXT_TEMPLATES[t].format(concept=g["concept"])
            return {"text": text, "kind": "gap", "target_id": g["concept"],
                    "factors": _factors_for("gap", g["concept"])}
        if t == "attribute_conflict":
            text = TEXT_TEMPLATES[t].format(key=g["key"], concept=g["concept"],
                                            values=", ".join(g["values"]))
            return {"text": text, "kind": "conflict", "target_id": g["concept"],
                    "factors": _factors_for("conflict", g["concept"])}
        if t == "contested_claim":
            text = TEXT_TEMPLATES[t].format(text=g["claim"]["text"])
            return {"text": text, "kind": "conflict",
                    "target_id": g["claim"].get("subject_id"),
                    "factors": _factors_for("conflict", None)}
        return None

    def queue(self, limit: int = 50) -> list[dict]:
        return self.mem.list_questions()[:limit]

    def top(self, n: int = 1) -> list[dict]:
        return [q for q in self.mem.list_questions()
                if q["status"] in ("proposed", "queued")][:n]
