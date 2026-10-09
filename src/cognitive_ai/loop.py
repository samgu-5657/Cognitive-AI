"""Autonomous exploration loop with explicit budget. Never infinite."""
from __future__ import annotations

import time

from .curiosity import Curiosity
from .learner import Learner
from .memory import Memory
from .policy import MAX_NEW_QUESTIONS_PER_STEP


class ExplorationLoop:
    def __init__(self, memory: Memory, corpus_dir: str):
        self.mem = memory
        self.learner = Learner(memory, corpus_dir)
        self.curiosity = Curiosity(memory)

    def run(self, steps: int = 5, max_seconds: float = 120.0) -> dict:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if steps > 100:
            raise ValueError("steps budget exceeded (max 100 per run)")
        t0 = time.time()
        log: list[dict] = []
        for i in range(steps):
            if time.time() - t0 > max_seconds:
                log.append({"step": i, "stopped": "time budget reached"})
                break
            # 1. reconcile queue with current gaps (capped, deduplicated)
            gen = self.curiosity.generate(
                max_new=MAX_NEW_QUESTIONS_PER_STEP)
            new_count = sum(1 for g in gen if g.get("is_new"))
            # 2. pick top eligible question (skips cooling / exhausted ones)
            eligible = self.curiosity.eligible_open()
            if not eligible:
                log.append({"step": i,
                            "note": "no eligible open questions",
                            "new_questions": new_count})
                break
            top = eligible[0]
            res = self.learner.research_question(int(top["id"]))
            # 3. bind the log entry to the exact question + outcome
            log.append({"step": i,
                        "question_id": int(top["id"]),
                        "target_id": top["target_id"],
                        "key": res.get("key"),
                        "question": top["text"],
                        "result": res.get("status") or res.get("result"),
                        "evidence_sources": res.get("evidence_sources", []),
                        "action": res.get("action", "none"),
                        "applied_change_id": res.get("applied_change_id"),
                        "new_questions": new_count})
        self.mem.add_experience("evaluation",
                                f"exploration run: {len(log)} step(s)",
                                {"log": log})
        return {"steps_executed": len(log), "log": log,
                "elapsed_s": round(time.time() - t0, 2)}
