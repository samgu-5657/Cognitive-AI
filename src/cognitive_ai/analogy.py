"""Structural analogy engine (Phase 5B). Simulation of partial human
analogical reasoning -- not a claim of replicating thought.

A case = solved task reduced to typed structure (goal kind, op-chain
skeleton, slot types, relation edges). Names and surface words are kept
for display only and NEVER drive matching; a separate surface score is
reported contrastively so keyword-lookalikes can be caught and refused.

Pipeline: represent -> retrieve candidates -> align (role mapping) ->
precondition check -> transplant plan -> execute -> verify ->
record (or refuse with reason + gap).
"""
from __future__ import annotations

import re

from .policy import ANALOGY_STRUCT_MIN as _STRUCT_MIN  # noqa: F401 (re-export)

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+")


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall((text or "").lower()))


def surface_score(a_text: str, b_text: str) -> float:
    ta, tb = tokens(a_text), tokens(b_text)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return round(len(ta & tb) / len(ta | tb), 3)


def case_from_trace(trace: dict) -> dict:
    """Reduce a persisted execution trace to a matchable case structure."""
    task = trace.get("task", {})
    steps = [s for s in trace.get("steps", [])]
    ops = [(s.get("op"), s.get("status")) for s in steps]
    slots: dict[str, str] = {}
    for s in steps:
        for k in _canon_slot_keys(s.get("op", "")):
            slots[k] = "any"
    edges = []
    rule_chain = []
    for s in steps:
        if s.get("op") == "query_attribute":
            inp = s.get("inputs", {}) or {}
            if inp.get("concept"):
                edges.append(("has_attr", inp.get("concept"),
                              inp.get("key")))
        if s.get("op") == "apply_learned":
            for app in (s.get("outputs", {}) or {}).get("applied", []):
                edges.append(("rule_fired", app.get("rule_id"), "active"))
                if app.get("rule_id"):
                    rule_chain.append(app["rule_id"])
    return {"task_id": task.get("task_id", ""),
            "task_type": task.get("task_type", "unknown"),
            "goal_kind": (task.get("goal") or {}).get("kind", ""),
            "op_chain": [op for op, _ in ops],
            "rule_chain": rule_chain,
            "op_status": {op: st for op, st in ops},
            "slots": slots,
            "edges": edges,
            "input_text": task.get("input_text", ""),
            "verified": task.get("status") == "verified"}


def _canon_slot_keys(op: str) -> tuple:
    return {"compute": ("expression",),
            "apply_learned": ("rule", "input"),
            "query_attribute": ("concept", "key"),
            "retrieve_concepts": ("terms",),
            "apply_rule": ("rule",)}.get(op, ())


def _chain_similarity(a: list[str], b: list[str]) -> float:
    # ordered overlap: LCS/len ratio over kinds (order matters in plans).
    # Both empty is neutral (0.5), not perfect: absence of structure is
    # not evidence of sameness; one-sided emptiness scores 0.
    if not a and not b:
        return 0.5
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] \
                else max(dp[i + 1][j], dp[i][j + 1])
    return round(dp[m][n] / max(m, n), 3)


def structural_score(cur: dict, cand: dict) -> dict:
    """Decomposed structural match. No surface terms involved."""
    goal = 1.0 if cur.get("goal_kind") and \
        cur.get("goal_kind") == cand.get("goal_kind") else 0.0
    type_match = 1.0 if cur.get("task_type") and \
        cur.get("task_type") == cand.get("task_type") else 0.0
    chain = _chain_similarity(cur.get("op_chain", []),
                              cand.get("op_chain", []))
    rules = _chain_similarity(cur.get("rule_chain", []),
                              cand.get("rule_chain", []))
    sk, ck = set(cur.get("slots", {})), set(cand.get("slots", {}))
    slots = round(len(sk & ck) / len(sk | ck), 3) if (sk or ck) else 1.0
    total = round(0.3 * goal + 0.2 * type_match + 0.2 * chain
                  + 0.2 * rules + 0.1 * slots, 3)
    return {"goal": goal, "type": type_match, "chain": chain,
            "rules": rules, "slots": slots, "total": total}


def align_roles(cur: dict, cand: dict) -> dict:
    """Role mapping between cases: op kinds in order + shared slot keys.
    Names are never aligned (they carry no structural meaning)."""
    mapping = {"ops": [], "slots": {}}
    ci = 0
    for i, op in enumerate(cand.get("op_chain", [])):
        while ci < len(cur.get("op_chain", [])) and \
                cur["op_chain"][ci] != op:
            ci += 1
        if ci < len(cur.get("op_chain", [])):
            mapping["ops"].append({"from": i, "to": ci, "op": op})
            ci += 1
    for k in cur.get("slots", {}):
        if k in cand.get("slots", {}):
            mapping["slots"][k] = k
    return mapping


def retrieve(cur: dict, memory_cases: list[dict],
             struct_min: float | None = None,
             feedback: dict | None = None) -> list[dict]:
    """Rank verified memory cases by structural score.

    feedback: {candidate_task_id: adjustment} from user corrections;
    applied transparently and reported per candidate.
    """
    from .policy import ANALOGY_STRUCT_MIN
    struct_min = ANALOGY_STRUCT_MIN if struct_min is None else struct_min
    feedback = feedback or {}
    out = []
    for cand in memory_cases:
        if not cand.get("verified"):
            continue
        struct = structural_score(cur, cand)
        surf = surface_score(cur.get("input_text", ""),
                             cand.get("input_text", ""))
        adj = float(feedback.get(cand.get("task_id", ""), 0.0))
        total = round(struct["total"] + adj, 3)
        out.append({"candidate": cand,
                    "structural": struct,
                    "surface": surf,
                    "feedback_adj": adj,
                    "score": total,
                    "mapping": align_roles(cur, cand),
                    "accepted": total >= struct_min})
    out.sort(key=lambda r: -r["score"])
    return out


def check_transfer_preconditions(mapping: dict, plan_steps: list[dict],
                                 current_inputs: dict) -> tuple[bool, list]:
    """Every transplanted op must find its required input keys in the
    current task's per-op inputs. Returns (ok, missing[])."""
    need_keys = {"apply_learned": ("inputs",), "compute": ("expression",),
                 "query_attribute": ("concept", "key"),
                 "retrieve_concepts": ("terms",),
                 "apply_rule": ("rule_id",)}
    missing = []
    for step in plan_steps:
        have = (current_inputs.get(step["op"]) or {})
        for k in need_keys.get(step["op"], ()):
            if k == "inputs":
                if not isinstance(have.get("inputs"), dict) or \
                        not have.get("inputs"):
                    missing.append(f"{step['op']}.inputs")
            elif k not in have:
                missing.append(f"{step['op']}.{k}")
    return (not missing), sorted(set(missing))


def compose_for_goal(ranked_accepted: list[dict],
                     goal_chain: list[str]) -> dict | None:
    """Cover an explicit goal rule-chain by concatenating accepted
    candidates' rule chains in order. Returns {rules, sources} or None
    when coverage is incomplete -- partial application is refused."""
    if not goal_chain:
        return None
    rules, sources, i = [], [], 0
    for r in ranked_accepted:
        rc = r["candidate"].get("rule_chain", [])
        j = 0
        while i + j < len(goal_chain) and j < len(rc) and \
                rc[j] == goal_chain[i + j]:
            j += 1
        if j > 0:
            rules.extend(rc[:j])
            sources.append(r["candidate"]["task_id"])
            i += j
        if i >= len(goal_chain):
            return {"rules": rules, "sources": sources}
    return None


def transplant_plan(cand_trace: dict, mapping: dict) -> list[dict]:
    """Rebuild the candidate's verified steps as a fresh plan.

    Op kinds and relative order are transplanted; concrete inputs are NOT
    copied (they belong to the old task) -- the caller supplies current
    inputs per step. Steps that only made sense with old inputs are
    dropped and reported.
    """
    plan, dropped = [], []
    for s in cand_trace.get("steps", []):
        if s.get("status") != "verified":
            dropped.append({"op": s.get("op"),
                            "reason": "not verified in source"})
            continue
        if s.get("op") in ("raise_gap",):
            dropped.append({"op": s.get("op"),
                            "reason": "gap records do not transfer"})
            continue
        plan.append({"op": s.get("op"), "inputs": {},
                     "from_analogy": True})
    return plan


def run_transfer(current_task, plan_steps: list[dict], step_inputs: dict,
                 ctx: dict, source_task_id: str, mapping: dict) -> dict:
    """Execute transplanted steps with CURRENT inputs, verify, record.

    Returns a transfer record (never a bare answer). Source conclusions
    are never copied as facts.
    """
    from .operations import REGISTRY
    record = {"source_task_id": source_task_id,
              "current_task_id": current_task.task_id,
              "mapping": mapping, "steps": [], "checks": [],
              "outcome": "pending", "reason": ""}
    ok, missing = check_transfer_preconditions(
        mapping, plan_steps, step_inputs)
    record["checks"].append({"preconditions": ok, "missing": missing})
    if not ok:
        record.update(outcome="refused",
                      reason=f"missing preconditions: {missing}")
        return record
    for step in plan_steps:
        op_cls = REGISTRY.get(step["op"])
        if op_cls is None:
            record["steps"].append({"op": step["op"],
                                    "status": "unknown_op"})
            record.update(outcome="failed",
                          reason=f"unknown op {step['op']}")
            return record
        inputs = dict(step_inputs.get(step["op"], step.get("inputs", {})))
        rec = op_cls().run(inputs, ctx)
        record["steps"].append(rec)
        if rec["status"] in ("failed", "precondition_failed"):
            record.update(outcome="failed",
                          reason=f"{step['op']}: {rec.get('note')}")
            return record
    finals = [s for s in record["steps"] if s["op"] == "check_result"]
    if finals and not all(s["status"] == "verified" for s in finals):
        record.update(outcome="failed", reason="verification failed")
        return record
    record.update(outcome="verified",
                  reason="transplanted plan verified on current task")
    return record
