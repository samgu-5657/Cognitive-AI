"""Phase M5-D clarification loop (isolated tmp DBs, real APIs)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import seminteraction as si
from cognitive_ai.memory import Memory
from cognitive_ai.models import SemanticClarificationAnswer
from cognitive_ai.ontology import Ontology
from cognitive_ai.semcontext import (empty_context,
                                     extend_semantic_context)
from cognitive_ai.semmeaning import understand_compositional_statement


def fresh(tmp_path, tag="si"):
    m = Memory(str(tmp_path / f"{tag}.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    o.add_concept("bowl", "碗", parents=["container"])
    o.add_attribute("cup", "material", "ceramic", status="verified")
    o.add_attribute("cup", "function", "喝水", status="verified")
    return m


def counts(m):
    return (len(m.list_concepts()),
            len(m.query("SELECT * FROM attributes")),
            len(m.query("SELECT * FROM relations")),
            len(m.query("SELECT * FROM rules")),
            len(m.list_pending_candidates()),
            len(m.query("SELECT * FROM lifecycle_events")))


def context_with(m, *priors):
    context = empty_context()
    for prior in priors:
        analyzed = understand_compositional_statement(prior, m)
        context = extend_semantic_context(context, analyzed, m)
    return context


def answer_for(request, context, option_id):
    return SemanticClarificationAnswer(
        request_id=request.request_id, option_id=option_id,
        context_fingerprint=si.context_fingerprint(context),
        origin="user")


# ---- A. understood ----
def test_understood_no_request(tmp_path, monkeypatch):
    from cognitive_ai import semcontext as ctx_mod
    m = fresh(tmp_path)
    try:
        calls = {"n": 0}
        real = ctx_mod.understand_with_context

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(ctx_mod, "understand_with_context", spy)
        context = context_with(m, "杯子是容器")
        rep = si.begin_semantic_interaction("杯子可以喝水", m, context)
        assert rep.state == "understood", rep
        assert rep.request is None
        assert rep.complete is True
        assert calls["n"] >= 1
    finally:
        m.close()


# ---- B. ambiguous loop ----
def test_ambiguous_clarify_and_resolve(tmp_path):
    m = fresh(tmp_path, tag="siB")
    try:
        context = context_with(m, "杯子是容器", "碗是容器")
        first = si.begin_semantic_interaction("它可以喝水", m, context)
        assert first.state == "needs_clarification", first
        request = first.request
        assert request.kind == "ambiguous_reference"
        assert sorted(o.option_id for o in request.options) == \
            ["opt-bowl", "opt-cup"]
        assert all(o.selectable for o in request.options)
        assert first.complete is False
        second = si.answer_semantic_clarification(
            request, context, answer_for(request, context, "opt-cup"), m)
        assert second.state == "clarification_resolved", second
        assert second.understanding is not None
        assert second.understanding.root.status == "supported"
        assert second.complete is True
    finally:
        m.close()


# ---- C. missing antecedent ----
def test_missing_antecedent(tmp_path):
    m = fresh(tmp_path, tag="siC")
    try:
        first = si.begin_semantic_interaction("它可以盛水", m,
                                              empty_context())
        assert first.state == "needs_clarification", first
        assert first.request.kind == "missing_antecedent"
        assert first.request.options == ()
        assert first.complete is False
    finally:
        m.close()


# ---- D. same-name identity ----
def test_unselectable_same_name(tmp_path):
    import dataclasses
    m = fresh(tmp_path, tag="siD")
    try:
        context = context_with(m, "杯子是容器")
        Ontology(m).add_concept("cup2", "杯子", parents=["entity"])
        first = si.begin_semantic_interaction("它可以喝水", m, context)
        assert first.state == "needs_clarification", first
        assert len(first.request.options) == 1
        option = first.request.options[0]
        assert option.selectable is False, option
        out = si.answer_semantic_clarification(
            first.request, context,
            answer_for(first.request, context, option.option_id), m)
        assert out.state == "invalid", out
        with pytest.raises(ValueError, match="selectable options need a reference"):
            dataclasses.replace(option, selectable=True)
    finally:
        m.close()


# ---- E. stale & forged picks ----
def test_stale_and_forged_picks(tmp_path):
    import dataclasses
    m = fresh(tmp_path, tag="siE")
    try:
        context = context_with(m, "杯子是容器", "碗是容器")
        first = si.begin_semantic_interaction("它可以喝水", m, context)
        request = first.request
        assert si.answer_semantic_clarification(
            request, context,
            SemanticClarificationAnswer(
                request_id="clr-deadbeef", option_id="opt-cup",
                context_fingerprint=si.context_fingerprint(context),
                origin="user"), m).state == "invalid"
        assert si.answer_semantic_clarification(
            request, context,
            answer_for(request, context, "opt-nobody"), m).state == \
            "invalid"
        moved = context_with(m, "杯子是容器", "碗是容器", "杯子有陶瓷")
        assert si.answer_semantic_clarification(
            request, moved, answer_for(request, moved, "opt-cup"),
            m).state == "invalid"
        Ontology(m).add_concept("cup2", "杯子", parents=["entity"])
        assert si.answer_semantic_clarification(
            request, context, answer_for(request, context, "opt-cup"),
            m).state == "invalid"
    finally:
        m.close()


# ---- F. composition ----
def test_conjunction_and_conditional(tmp_path):
    m = fresh(tmp_path, tag="siF")
    try:
        context = context_with(m, "杯子是容器", "碗是容器")
        first = si.begin_semantic_interaction("它有陶瓷而且它可以喝水", m,
                                              context)
        assert first.state == "needs_clarification", first
        second = si.answer_semantic_clarification(
            first.request, context,
            answer_for(first.request, context, "opt-bowl"), m)
        assert second.state == "clarification_resolved", second
        assert second.understanding.root.kind == "conjunction"
        assert second.understanding.root.children[0].raw_text == \
            "碗有陶瓷"
        third = si.begin_semantic_interaction("如果它是容器，那么它可以喝水", m,
                                              context)
        assert third.state == "needs_clarification", third
        fourth = si.answer_semantic_clarification(
            third.request, context,
            answer_for(third.request, context, "opt-cup"), m)
        assert fourth.state == "clarification_resolved", fourth
        assert fourth.understanding.root.status == \
            "conditional_unasserted"
        assert fourth.understanding.root.kind == "conditional"
        assert len(m.query("SELECT * FROM rules")) == 0
    finally:
        m.close()


# ---- G. UI ----
def test_ui_interact_and_answer(tmp_path):
    from cognitive_ai import web_ui
    m = fresh(tmp_path, tag="siG")
    try:
        code, out = web_ui.dispatch(
            m, "", "POST", "/api/interact",
            {"priors": ["杯子是容器", "碗是容器"], "text": "它可以喝水"})
        assert code == 200, out
        assert out["result"].state == "needs_clarification", out
        request = out["result"].request
        assert len(request.options) == 2
        code, out = web_ui.dispatch(
            m, "", "POST", "/api/answer",
            {"request": {"request_id": request.request_id,
                         "raw_input": request.raw_input,
                         "span": request.span, "kind": request.kind,
                         "question": request.question,
                         "options": [{"option_id": o.option_id,
                                      "ref": {"ref_id": o.ref.ref_id,
                                              "name": o.ref.name,
                                              "kind": o.ref.kind}
                                      if o.ref is not None else None,
                                      "display_name": o.display_name,
                                      "mention_ref": o.mention_ref,
                                      "selectable": o.selectable,
                                      "reason": o.reason}
                                     for o in request.options],
                         "context_fingerprint":
                         request.context_fingerprint,
                         "diagnostics": list(request.diagnostics)},
             "answer": {"request_id": request.request_id,
                        "option_id": "opt-bowl"},
             "priors": ["杯子是容器", "碗是容器"], "text": "它可以喝水"})
        assert code == 200, out
        assert out["result"].state == "clarification_resolved", out
        code, out = web_ui.dispatch(
            m, "", "POST", "/api/answer",
            {"request": {"request_id": "clr-0",
                         "raw_input": "它可以喝水", "span": "它",
                         "kind": "missing_antecedent",
                         "question": "请补充先前陈述", "options": [],
                         "context_fingerprint": "0", "diagnostics": []},
             "answer": {"request_id": "clr-0", "option_id": "opt-x"},
             "priors": [], "text": "它可以喝水"})
        assert code == 200, out
        assert out["result"].state == "invalid", out
    finally:
        m.close()


# ---- H. read-only & frozen ----
def test_interaction_read_only(tmp_path, monkeypatch):
    m = fresh(tmp_path, tag="siH")
    try:
        context = context_with(m, "杯子是容器", "碗是容器")
        n0 = counts(m)
        seen = {"pending": 0}
        real = Memory.add_pending_candidate

        def spy(*a, **k):
            seen["pending"] += 1
            return real(*a, **k)

        monkeypatch.setattr(Memory, "add_pending_candidate", spy)
        first = si.begin_semantic_interaction("它可以喝水", m, context)
        si.answer_semantic_clarification(
            first.request, context,
            answer_for(first.request, context, "opt-cup"), m)
        assert seen["pending"] == 0
        assert counts(m) == n0
        with pytest.raises(AttributeError):
            first.request.question = "x"
        with pytest.raises(AttributeError):
            first.request.options[0].display_name = "y"
        again = si.begin_semantic_interaction("它可以喝水", m, context)
        assert again.request.request_id == first.request.request_id
    finally:
        m.close()
