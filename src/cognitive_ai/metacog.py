"""Metacognitive controller (Phase 5C). Decides next actions from REAL
records -- never from a bare confidence number, never by infinite retry.

Actions: continue_plan | switch_rule | try_analogy | retrieve_more |
ask_user | raise_gap | refuse. Every replan carries an explicit reason
and counts against a budget (default 2 attempts per task).
"""
from __future__ import annotations

ACTIONS = ("continue_plan", "switch_rule", "try_analogy", "retrieve_more",
           "ask_user", "raise_gap", "refuse")

DEFAULT_BUDGET = 2


def past_failures(memory, task_text: str = "") -> list[dict]:
    """Previous failure experiences, newest first (for loop detection)."""
    out = []
    for e in memory.list_experiences(limit=200):
        import json
        try:
            d = json.loads(e["detail"]) if isinstance(e["detail"], str) \
                else e["detail"]
        except Exception:
            d = {}
        s = (e.get("summary") or "") + str(d)
        if "fail" in s or "gap" in s or "inconclusive" in s or \
                "refus" in s:
            out.append({"summary": e["summary"], "kind": e["kind"]})
    return out


def decide(task: dict, trace_steps: list, verification: dict,
           memory, attempts_used: int = 0,
           budget: int = DEFAULT_BUDGET) -> dict:
    """Return {action, reason, budget_left}. Pure function of records."""
    cause = _dominant_cause(trace_steps, verification)
    same_cause_runs = sum(
        1 for f in past_failures(memory)
        if cause.split(":")[0] in f["summary"]) if cause else 0
    left = budget - attempts_used
    # loop guard first: repeating the same failure is never a plan
    if same_cause_runs >= 2 and cause:
        return {"action": "refuse", "reason": f"same failure repeated "
                f"({cause}); refusing instead of retrying",
                "budget_left": left}
    if left <= 0:
        return {"action": "raise_gap" if cause != "unparseable" else "refuse",
                "reason": "repair budget exhausted", "budget_left": 0}
    if cause is None:
        return {"action": "continue_plan", "reason": "no failure recorded",
                "budget_left": left}
    head, _, _detail = cause.partition(":")
    if head == "unparseable":
        return {"action": "ask_user",
                "reason": "input needs clarification before any plan",
                "budget_left": left}
    if head == "missing_concept":
        return {"action": "ask_user",
                "reason": f"unknown concept {_detail}; user can teach it",
                "budget_left": left}
    if head == "missing_rule":
        return {"action": "try_analogy",
                "reason": "no direct rule; analogy may supply a plan",
                "budget_left": left}
    if head == "compute_failed":
        return {"action": "raise_gap",
                "reason": f"computation out of scope: {_detail}",
                "budget_left": left}
    if head == "wrong_op":
        return {"action": "switch_rule",
                "reason": f"drop {_detail} and retry once",
                "budget_left": left - 1}
    # verify_failed and the rest: try analogy once, else gap
    tried_analogy = any((s.get("op") == "analogize" or
                         s.get("analogy", {}).get("outcome"))
                        for s in trace_steps)
    if not tried_analogy:
        return {"action": "try_analogy",
                "reason": "verification failed; an alternative plan may help",
                "budget_left": left}
    return {"action": "raise_gap",
            "reason": "alternatives exhausted; recording gap",
            "budget_left": left}


def _dominant_cause(trace_steps: list, verification: dict) -> str | None:
    for s in reversed(trace_steps):
        if s.get("status") in ("failed", "precondition_failed"):
            op = s.get("op", "?")
            note = s.get("note", "") or \
                str((s.get("outputs") or {}).get("missing", ""))
            if op == "compute":
                return f"compute_failed:{note[:80]}"
            if op in ("retrieve_concepts", "query_attribute"):
                return f"missing_concept:{note[:80]}"
            if op in ("apply_rule", "apply_learned"):
                return f"missing_rule:{note[:80]}"
            return f"verify_failed:{op}:{note[:80]}"
    rej = [c for c in (verification or {}).get("checks", [])
           if c.get("result") == "rejected"]
    if rej:
        return f"verify_failed:{rej[0]['name']}:{rej[0]['detail'][:80]}"
    inc = [c for c in (verification or {}).get("checks", [])
           if c.get("result") == "inconclusive"]
    if inc:
        return f"verify_failed:{inc[0]['name']}"
    return None
