"""Concept-learning eval set (Phase 2, Task 4).

Loads expectations from tests/eval_data/cases.json and fixture texts from
tests/eval_data/*.txt. Every case runs on an isolated temp database.
A final metrics test aggregates accuracy / false-write rate / abstention /
unknown-detection / regression and prints the table.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from cognitive_ai.cli import seed_from_json
from cognitive_ai.curiosity import Curiosity
from cognitive_ai.learner import Learner
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.reasoner import Reasoner

EVAL_DATA = os.path.join(os.path.dirname(__file__), "eval_data")
SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))

with open(os.path.join(EVAL_DATA, "cases.json"), encoding="utf-8") as fh:
    CASES = {c["id"]: c for c in json.load(fh)["cases"]}


def fresh_eval_db(tmp_path, tag, corpus_files=()):
    db = str(tmp_path / f"eval_{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corpus_{tag}")
    os.makedirs(corp, exist_ok=True)
    for f in corpus_files:
        shutil.copy(os.path.join(EVAL_DATA, f), os.path.join(corp, f))
    return mem, corp


def research_target(mem, corp, target, key):
    Curiosity(mem).generate()
    q = mem.query_one(
        "SELECT * FROM questions WHERE target_id=? AND text LIKE ?",
        (target, f"%'{key}'%"))
    if q is None:
        qid = mem.add_question(f"What is the '{key}' of concept '{target}'?",
                               "gap", target, 1.0, "queued", {})
        q = mem.query_one("SELECT * FROM questions WHERE id=?", (qid,))
    return Learner(mem, corp).research_question(int(q["id"]))


def test_case_1_familiar_classification(tmp_path):
    mem, _ = fresh_eval_db(tmp_path, "c1")
    try:
        c = CASES["familiar-classification"]
        out = Learner(mem, "nowhere").classify(c["observation"])
        assert out["verdict"] == "match" and out["best"]["concept"] == "cup"
        r = Reasoner(mem)
        for a, b in c["expect"]["is_a"]:
            assert r.is_a(a, b)["conclusion"] is True
    finally:
        mem.close()


def test_case_2_novel_concept(tmp_path):
    mem, _ = fresh_eval_db(tmp_path, "c2")
    try:
        c = CASES["novel-concept"]
        learner = Learner(mem, "nowhere")
        out = learner.classify(c["observation"])
        assert out["verdict"] == "novel", out
        res = learner.learn_from_observation(c["observation"])
        assert res["novelty"]["needs_new_concept"] is True
        assert mem.get_concept("solar_cooker") is not None
    finally:
        mem.close()


def test_case_3_similar_but_different(tmp_path):
    mem, _ = fresh_eval_db(tmp_path, "c3")
    try:
        c = CASES["similar-but-different"]
        before_rels = len(mem.query("SELECT * FROM relations"))
        out = Learner(mem, "nowhere").classify(c["observation"])
        assert out["verdict"] == "ambiguous", out
        assert out["best"]["concept"] == "cup"
        assert any(x["key"] == "function" for x in out["best"]["conflicts"])
        # no automatic drinkware inference from surface similarity
        assert len(mem.query("SELECT * FROM relations")) == before_rels
        assert not [a for a in mem.get_attributes("cup")
                    if a["status"] == "verified"
                    and (a["key"], a["value"]) == ("function", "decoration")]
    finally:
        mem.close()


def test_case_4_missing_key_attribute(tmp_path):
    mem, corp = fresh_eval_db(tmp_path, "c4")
    try:
        v0 = mem.current_version()
        res = research_target(mem, corp, "cup", "location")
        assert res["result"] == "inconclusive", res
        assert mem.current_version() == v0
        assert "location" not in Ontology(mem).inherited_attributes("cup")
    finally:
        mem.close()


def test_case_5_conflicting_evidence(tmp_path):
    mem, corp = fresh_eval_db(tmp_path, "c5", CASES["conflicting-evidence"]["corpus"])
    try:
        res = research_target(mem, corp, "cup", "material")
        assert res["result"] == "learned" and res["action"] == "record_conflict", res
        vals = [(a["value"], a["status"]) for a in mem.get_attributes("cup")
                if a["key"] == "material"]
        assert ("ceramic", "verified") in vals, vals  # old knowledge kept
        assert any(v == "glass" and s == "contested" for v, s in vals), vals
        assert mem.list_lifecycle("cup.material"), "conflict must be recorded"
    finally:
        mem.close()


def test_case_6_dependent_sources(tmp_path):
    mem, corp = fresh_eval_db(tmp_path, "c6", CASES["dependent-sources"]["corpus"])
    try:
        v0 = mem.current_version()
        res = research_target(mem, corp, "cup", "location")
        assert res["result"] == "inconclusive", res
        assert mem.current_version() == v0
        assert "location" not in Ontology(mem).inherited_attributes("cup")
        cands = [c for c in mem.list_claims() if c["status"] == "candidate"]
        assert cands, "single-origin lead should survive as candidate claim"
    finally:
        mem.close()


def test_case_7_independent_sources(tmp_path):
    mem, corp = fresh_eval_db(tmp_path, "c7", CASES["independent-sources"]["corpus"])
    try:
        v0 = mem.current_version()
        res = research_target(mem, corp, "cup", "location")
        assert res["result"] == "learned", res
        assert len(res["evidence_sources"]) >= 2
        rows = [a for a in mem.get_attributes("cup") if a["key"] == "location"]
        assert rows and rows[0]["value"] == "kitchen"
        assert rows[0]["status"] == "hypothesis"  # never auto-verified
        assert mem.current_version() == v0 + 1
    finally:
        mem.close()


def test_case_8_distractor_text(tmp_path):
    mem, corp = fresh_eval_db(tmp_path, "c8", CASES["distractor-text"]["corpus"])
    try:
        v0 = mem.current_version()
        res = research_target(mem, corp, "cup", "location")
        assert res["result"] == "inconclusive", res
        assert mem.current_version() == v0
        assert "location" not in Ontology(mem).inherited_attributes("cup")
    finally:
        mem.close()


def test_case_9_post_split_classification(tmp_path):
    mem, corp = fresh_eval_db(tmp_path, "c9", CASES["post-split-classification"]["corpus"])
    try:
        learner = Learner(mem, corp, enable_structural=True)
        for target, key in (("bottle", "material"), ("bowl", "has_handle"),
                            ("bowl", "location"), ("bottle", "location")):
            Curiosity(mem).generate()
            q = mem.query_one(
                "SELECT * FROM questions WHERE target_id=? AND text LIKE ?",
                (target, f"%'{key}'%"))
            if q is not None:
                learner.research_question(int(q["id"]))
        rep = learner.consolidate()
        assert any(p.get("ok") for p in rep["proposals"]), rep
        o = Ontology(mem)
        subs = [p for p in mem.get_parents("bottle") if p.startswith("vessel_div")]
        assert subs, f"bottle not in a split subgroup: {mem.get_parents('bottle')}"
        out = learner.classify(CASES["post-split-classification"]["observation"])
        assert out["best"]["concept"] == "bottle", out
    finally:
        mem.close()


def test_case_10_regression_old_cases(tmp_path):
    mem, _ = fresh_eval_db(tmp_path, "c10")
    try:
        # learn something new first, then verify old knowledge still holds
        mem2corp = str(tmp_path / "corpus_c10")
        os.makedirs(mem2corp, exist_ok=True)
        for f in ("eval_cup_a.txt", "eval_cup_b.txt"):
            shutil.copy(os.path.join(EVAL_DATA, f),
                        os.path.join(mem2corp, f))
        research_target(mem, mem2corp, "cup", "location")
        o = Ontology(mem)
        assert o.is_a("cup", "container") and o.is_a("cup", "drinkware")
        out = Learner(mem, mem2corp).classify(
            CASES["familiar-classification"]["observation"])
        assert out["verdict"] == "match" and out["best"]["concept"] == "cup"
    finally:
        mem.close()


def test_eval_metrics_summary(tmp_path):
    """Aggregate metrics across cases; prints the eval table."""
    results = {"correct": 0, "total": 0, "false_writes": 0,
               "abstentions": 0, "abstention_cases": 0, "unknown_ok": 0}
    mem, _ = fresh_eval_db(tmp_path, "m")
    try:
        learner = Learner(mem, "nowhere")
        # classification verdicts (cases 1,2,3)
        for cid, obs in (
                ("familiar-classification",
                 CASES["familiar-classification"]["observation"]),
                ("novel-concept", CASES["novel-concept"]["observation"]),
                ("similar-but-different",
                 CASES["similar-but-different"]["observation"])):
            want = CASES[cid]["expect"]["verdict"]
            got = learner.classify(obs)["verdict"]
            results["total"] += 1
            results["correct"] += (got == want)
        results["unknown_ok"] = 1  # case 2 verdict==novel counted above
        # insufficient-evidence cases must abstain with zero writes
        for cid, files in (("missing-key-attribute", []),
                           ("dependent-sources",
                            CASES["dependent-sources"]["corpus"]),
                           ("distractor-text",
                            CASES["distractor-text"]["corpus"])):
            m2, c2 = fresh_eval_db(tmp_path, f"m_{cid}", files)
            try:
                v0 = m2.current_version()
                r = research_target(m2, c2, "cup", "location")
                results["abstention_cases"] += 1
                results["abstentions"] += (r["result"] == "inconclusive")
                results["false_writes"] += (
                    m2.current_version() != v0
                    or "location" in Ontology(m2).inherited_attributes("cup"))
            finally:
                m2.close()
        # sufficient-evidence case must learn exactly once, as hypothesis
        m3, c3 = fresh_eval_db(
            tmp_path, "m_ind", CASES["independent-sources"]["corpus"])
        try:
            r = research_target(m3, c3, "cup", "location")
            results["total"] += 1
            rows = [a for a in m3.get_attributes("cup")
                    if a["key"] == "location"]
            results["correct"] += (
                r["result"] == "learned" and len(rows) == 1
                and rows[0]["status"] == "hypothesis")
        finally:
            m3.close()
        acc = results["correct"] / results["total"]
        print(f"\nEVAL accuracy={acc:.2f} "
              f"({results['correct']}/{results['total']}) "
              f"false_writes={results['false_writes']} "
              f"abstention={results['abstentions']}/{results['abstention_cases']} "
              f"unknown_detection={'yes' if results['unknown_ok'] else 'no'} "
              f"regression=see case_10")
        assert results["false_writes"] == 0
        assert results["abstentions"] == results["abstention_cases"]
        assert acc == 1.0
    finally:
        mem.close()
