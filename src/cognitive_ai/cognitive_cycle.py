"""Unified cognitive cycle (Phase 6H-1).

One ordinary request, one controlled flow: build -> execute -> verify ->
decide -> render, with clarification requests that can actually resume.

Deliberately thin: parsing stays in semantics.py, planning in
cog_planners.py, execution/repair/persistence in executor.solve_text
(independent verify + metacog budget inside), rendering in
explainer.py + answers.py. This module only decides the NEXT step,
threads stage data, classifies the unified outcome, and keeps exactly
one experience record per cycle (no double logging with older paths).

Unified states: answer | clarification | abstention | failure.
  answer         verified conclusion with cited evidence;
  clarification  missing info or genuine ambiguity; resumable state kept;
  abstention     knowledge/evidence insufficient (gap recorded);
  failure        execution or internal processing failure.
"""
from __future__ import annotations

STATES = ("answer", "clarification", "abstention", "failure")

_ABSTAIN_CAUSES = ("missing_concept", "missing_rule")
_FAILURE_CAUSES = ("unparseable", "wrong_op", "compute_failed")


def run_cycle(session, text: str, frame: dict) -> dict:
    """Run one ordinary NL request. Never raises on user input.

    frame: already-parsed SemanticFrame dict (pronouns resolved by the
    caller). Returns {"state", "answer", "trace", "task", "stages",
    "pending"} where answer is the rendered {text, certainty,
    citations, provenance} dict.
    """
    stages: list[dict] = [{"stage": "parse", "status": "ok"}]
    try:
        from .cogtask import build_task
        task = build_task(text, frame, session.mem)
        stages.append({"stage": "build", "status": "ok",
                       "task_type": task.task_type,
                       "actionable": task.is_actionable()})
        if not task.is_actionable():
            return _clarify(session, task, stages,
                            "任务缺少可执行的目标或关键信息。")
        from .executor import solve_text
        trace = solve_text(text, frame, session.mem, session.onto,
                           session.reasoner)
        stages.append({"stage": "execute", "status": "ok",
                       "task_status": trace["task"]["status"],
                       "failure_cause": trace.get("failure_cause")})
        stages.append({"stage": "verify", "status": "ok",
                       "verdict": (trace.get("verification") or {}).get(
                           "verdict")})
        result = _classify(session, task, trace, stages)
        _record(session, result)
        return result
    except Exception as e:  # noqa: BLE001 - cycle never crashes
        task = {"task_id": None, "status": "failed"}
        result = {"state": "failure", "answer": {
            "text": "处理出错，未写入任何内容。", "certainty": "unknown",
            "citations": []}, "trace": {"task": task, "steps": []},
            "task": task,
            "stages": stages + [{"stage": "decide", "status": "failed",
                                 "error": f"{type(e).__name__}: {e}"}],
            "pending": None}
        _record(session, result)
        return result


def _classify(session, task, trace, stages: list) -> dict:
    from .answers import render
    from .explainer import explain
    status = trace["task"]["status"]
    cause = trace.get("failure_cause")
    verdict = (trace.get("verification") or {}).get("verdict")
    ans = render(explain(trace, style="short"), trace)
    if status == "verified":
        state = "answer"
    elif status == "inconclusive" or cause in _ABSTAIN_CAUSES:
        state = "abstention"
    elif cause in _FAILURE_CAUSES or verdict == "rejected":
        state = "failure"
    else:
        # failed for any other reason (e.g. verify_failed without a
        # contradiction): report failure, never a forced answer.
        state = "failure" if status == "failed" else "abstention"
    stages.append({"stage": "decide", "status": "ok", "state": state,
                   "cause": cause, "verdict": verdict})
    if state == "answer":
        try:
            session.last_answer = {
                "text": ans["text"], "certainty": ans["certainty"],
                "citations": ans["citations"],
                "provenance": ans.get("provenance"),
                "task_id": trace["task"]["task_id"]}
        except Exception:
            pass
    return {"state": state, "answer": ans, "trace": trace,
            "task": trace["task"], "stages": stages, "pending": None}


def park_for_clarify(session, frame_dict: dict, question: str) -> dict:
    """Store resumable clarification state (per-session)."""
    pending = {'kind': 'pronoun' if (frame_dict.get(
        'unresolved_references')) else 'question',
               'task': {'input_frame': frame_dict},
               'question': question}
    try:
        session.pending_clarify = pending
    except Exception:
        pass
    return pending


def _clarify(session, task, stages: list, why: str) -> dict:
    """Structured clarification with resumable state (per-session)."""
    pending = {"kind": "task",
               "task": task.to_dict() if hasattr(task, "to_dict")
               else dict(task),
               "question": why}
    try:
        session.pending_clarify = pending
    except Exception:
        pass
    stages.append({"stage": "decide", "status": "ok",
                   "state": "clarification"})
    result = {
        "state": "clarification",
        "answer": {"text": why + "请补充信息后继续；"
                   "本次未写入任何内容。", "certainty": "unknown",
                   "citations": []},
        "trace": {"task": pending["task"], "steps": []},
        "task": pending["task"], "stages": stages, "pending": pending}
    _record(session, result)
    return result


def try_resume(session, frame: dict, raw_text: str = "") -> dict | None:
    """Merge a clarification answer back into the waiting task.

    Pronoun case: the new input names a real entity (parsed frame or
    bare concept mention) -> substitute it into the stored frame's
    unresolved slots and return the merged frame for a fresh cycle.
    Anything else -> None (caller processes input normally; a later
    success or a new clarification replaces the stale pending state).
    """
    try:
        pend = getattr(session, "pending_clarify", None)
    except Exception:
        pend = None
    if not pend:
        return None
    stored = (pend.get("task") or {}).get("input_frame") or {}
    waiting = list(stored.get("unresolved_references", []) or [])
    if not waiting:
        return None
    named = [e.get("text", "") for e in frame.get("entities", []) or []
             if e.get("text", "") not in
             ("它", "他", "她", "它们", "这个", "那个", "这", "那", "")]
    if not named and (raw_text or "").strip():
        try:
            mem = session.mem
            t = raw_text.strip().strip("。！？；;!?，,、 ")
            if mem.get_concept(t) is not None:
                named = [t]
            else:
                hits = mem.find_concepts_by_name(t)
                if hits:
                    named = [t]
        except Exception:
            pass
    if not named:
        return None
    merged = {k: v for k, v in stored.items()}
    ents = []
    for e in stored.get("entities", []) or []:
        if e.get("text") in waiting:
            ents.append({"text": named[0], "role": e.get("role",
                                                         "subject"),
                         "resolved_from": e.get("text")})
        else:
            ents.append(dict(e))
    merged["entities"] = ents
    merged["unresolved_references"] = []
    merged["trace"] = list(merged.get("trace", [])) + \
        [f"clarify-resume: {waiting} -> {named[0]}"]
    try:
        session.pending_clarify = None
    except Exception:
        pass
    return merged


def _record(session, result: dict) -> None:
    """Exactly one experience row per cycle (no double logging)."""
    try:
        task = result.get("task") or {}
        session.mem.add_experience(
            "evaluation",
            f"cognitive cycle -> {result.get('state')}",
            {"op": "cycle", "state": result.get("state"),
             "task_id": task.get("task_id"),
             "stages": [s.get("stage") for s in result.get("stages",
                                                           [])]})
    except Exception:
        pass
