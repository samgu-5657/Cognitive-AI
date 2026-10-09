"""Metacognitive controller tests: budgeted, reasoned, non-looping."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import metacog
from cognitive_ai.cli import seed_from_json
from cognitive_ai.executor import solve_text
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.reasoner import Reasoner
from cognitive_ai.semantics import parse_sentence

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def fresh(tmp_path, tag):
    mem = Memory(str(tmp_path / f"{tag}.db"))
    seed_from_json(mem, SEED)
    return mem, Ontology(mem), Reasoner(mem)


def test_no_infinite_retry(tmp_path):
    mem, _, _ = fresh(tmp_path, "m1")
    try:
        d1 = metacog.decide({"input_text": "x"}, [], None, mem, 0, 2)
        d2 = metacog.decide({"input_text": "x"}, [], None, mem, 5, 2)
        assert d2["budget_left"] <= 0
        assert d2["action"] in ("raise_gap", "refuse")
        # repeated identical failures refuse instead of looping
        for _ in range(3):
            mem.add_experience("evaluation", "gap: compute_failed boo", {})
        d3 = metacog.decide({"input_text": "x"},
                             [{"op": "compute", "status": "failed",
                               "note": "boo"}], None, mem, 0, 5)
        assert d3["action"] == "refuse" and "repeated" in d3["reason"]
    finally:
        mem.close()


def test_ask_user_then_resume(tmp_path):
    mem, onto, rsn = fresh(tmp_path, "m2")
    try:
        tr = solve_text("紫砂壶有什么功能",
                        parse_sentence("紫砂壶有什么功能").to_dict(),
                        mem, onto, rsn)
        assert tr["task"]["status"] in ("failed", "inconclusive")
        decisions = [r["decision"] for r in tr.get("repairs", [])]
        assert decisions, "controller must record its decisions"
        # user teaches the missing concept; the same question now resolves
        onto.add_concept("zisha", "紫砂壶", parents=["entity"])
        onto.add_attribute("zisha", "function", "brewing",
                           status="verified")
        tr2 = solve_text("紫砂壶有什么功能",
                         parse_sentence("紫砂壶有什么功能").to_dict(),
                         mem, onto, rsn)
        assert tr2["task"]["status"] == "verified", tr2
    finally:
        mem.close()


def test_decisions_come_from_records_not_confidence(tmp_path):
    mem, _, _ = fresh(tmp_path, "m3")
    try:
        d = metacog.decide(
            {"input_text": "q"},
            [{"op": "retrieve_concepts", "status": "verified",
              "outputs": {}}], None, mem, 0, 2)
        assert d["action"] == "continue_plan"
        d = metacog.decide(
            {"input_text": "q"},
            [{"op": "compute", "status": "failed", "note": "bad syntax"}],
            None, mem, 0, 2)
        assert d["action"] == "raise_gap"
        assert "bad syntax" in d["reason"]
    finally:
        mem.close()
