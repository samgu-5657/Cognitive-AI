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


class SolverSession:
    """Minimal session facade for session-less callers (Phase 6H-1a).

    Lets the compat entry executor.solve_text delegate to the unified
    controller without inventing dialogue state: no context, no pending
    teaching, no cross-turn memory. pending_clarify set here dies with
    the object, which is correct (nothing to resume to).
    """

    def __init__(self, memory, ontology, reasoner):
        self.mem = memory
        self.onto = ontology
        self.reasoner = reasoner
        self.last_answer = None
        self.pending_clarify = None
        self.session_id = "solver"


def run_cycle(session, text: str, frame: dict, *, record: bool = True,
              max_repairs: int = 1) -> dict:
    """Run one ordinary NL request. Never raises on user input.

    frame: already-parsed SemanticFrame dict (pronouns resolved by the
    caller). The controller builds the task, drives execution through
    the executor's plan/run segment functions (single bounded loop
    below -- the ONLY repair loop), classifies the unified outcome and
    renders. record=False skips the cycle experience (compat callers
    keep their own logging).
    Returns {"state", "answer", "trace", "task", "stages",
    "pending"} where answer is the rendered {text, certainty,
    citations, provenance} dict.
    """
    stages: list[dict] = [{"stage": "parse", "status": "ok"}]
    try:
        from .cogtask import build_task
        from .executor import _persist, _sync
        task = build_task(text, frame, session.mem)
        stages.append({"stage": "build", "status": "ok",
                       "task_type": task.task_type,
                       "actionable": task.is_actionable()})
        if not task.is_actionable():
            trace: dict = {"task": task.to_dict(), "steps": [],
                           "repairs": [], "gap": None,
                           "failure_cause": "unparseable",
                           "verification": None, "metacog": []}
            _sync(task, trace)
            _persist(session.mem, task, trace)
            result = _clarify(session, task, stages,
                              "任务缺少可执行的目标或关键信息。")
            result["trace"] = trace
            result["task"] = trace["task"]
            if record:
                _record(session, result)
            return result
        trace = {"task": task.to_dict(), "steps": [],
                 "repairs": [], "gap": None, "failure_cause": None,
                 "verification": None, "metacog": []}
        _drive_execution(task, session.mem, session.onto,
                         session.reasoner, trace, max_repairs, stages)
        result = _classify(session, task, trace, stages)
        if record:
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
        if record:
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
    return result


def _drive_execution(task, mem, onto, reasoner, trace: dict,
                     max_repairs: int, stages: list) -> None:
    """The single bounded execution loop (moved verbatim here, 6H-1a).

    Previously this lived inside executor.solve_text, which made the
    controller call a full plan/verify/repair cycle. Now the controller
    owns the loop and only borrows the executor's segment functions
    (plan builders, single-plan runner, repair helpers, persistence);
    the executor never runs a control loop of its own.
    """
    from . import metacog, verify
    from .executor import (_alt_plan, _classify_failure, _finalize,
                           _make_plan, _raise_gap_for, _run_plan,
                           _solve_via_analogy_once)
    ctx = {"memory": mem, "ontology": onto, "reasoner": reasoner,
           "working_memory": {}}
    plan = _make_plan(task)
    task.status = "running"
    budget = max_repairs + 1
    attempts = 0
    verification = None
    while True:
        ok = _run_plan(task, plan, ctx, trace)
        attempts += 1
        if ok:
            verification = verify.verify_task(
                task.to_dict(), trace["steps"], mem, onto)
            trace["verification"] = verification
            if verification["verdict"] == "verified":
                task.status = "verified"
                _finalize(mem, task, trace)
                break
            if verification["verdict"] == "inconclusive":
                task.status = "inconclusive"
                trace["failure_cause"] = "verify_failed"
                _raise_gap_for(task, trace, mem)
                _finalize(mem, task, trace)
                break
            ok = False  # rejected: fall through to failure handling
        last = trace["steps"][-1] if trace["steps"] else {}
        cause = _classify_failure(task, last)
        if verification is not None and \
                verification.get("verdict") == "rejected":
            cause = "verify_failed"
        decision = metacog.decide(task.to_dict(), trace["steps"],
                                  verification, mem, attempts, budget)
        trace["metacog"].append({**decision, "cause": cause,
                                 "attempt": attempts})
        if verification is not None and \
                verification.get("verdict") == "rejected":
            cause = "verify_failed"
        trace["repairs"].append({"attempt": attempts, "failed_op":
                                 last.get("op"), "cause": cause,
                                 "decision": decision["action"]})
        if decision["action"] == "switch_rule":
            alt = _alt_plan(task, plan, last.get("op"))
            if alt and attempts < budget:
                plan = alt
                verification = None
                continue
        if decision["action"] == "try_analogy":
            tried = _solve_via_analogy_once(task, trace, ctx, mem,
                                            onto, reasoner)
            if tried == "verified":
                task.status = "verified"
                _finalize(mem, task, trace)
                break
            attempts += 1
            verification = None
            if attempts < budget:
                continue
        break
    if task.status not in ("verified", "inconclusive"):
        task.status = "failed"
    _finish_execution(task, trace, mem)
    stages.append({"stage": "execute", "status": "ok",
                   "task_status": trace["task"]["status"],
                   "failure_cause": trace.get("failure_cause")})
    stages.append({"stage": "verify", "status": "ok",
                   "verdict": (trace.get("verification") or {}).get(
                       "verdict")})


def _finish_execution(task, trace: dict, mem) -> None:
    from .executor import _finalize, _raise_gap_for
    if task.status in ("verified", "inconclusive"):
        return  # already finalized inside the loop; touch nothing
    task.status = "failed"
    if not trace["failure_cause"]:
        trace["failure_cause"] = trace["repairs"][-1]["cause"] \
            if trace["repairs"] else "verify_failed"
    if trace["gap"] is None:
        _raise_gap_for(task, trace, mem)
    _finalize(mem, task, trace)


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
