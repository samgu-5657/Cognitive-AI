"""Phase 6A micro-world tests: infant-style stepwise learning.

The world simulator's dynamics are hidden from the learner; only
examples, feedback and counterexamples cross the boundary. The SAME
learner code path (induce/validate/promote) is used throughout.
Isolated DBs.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import rulelearn as RL
from cognitive_ai.learnloop import run_cycle
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.world import ACTION_ZH, DYNAMICS, ToyWorld


def ex(i, inputs, label):
    return {"id": i, "inputs": inputs, "outputs": {"label": label}}


def world_examples(world, actions=("heat", "cool")):
    return [{"id": f"{s}-{a}", "inputs": {"state": s, "action": a},
             "outputs": {"label": DYNAMICS[(s, a)]}}
            for s in ("固态", "液态", "气态") for a in actions]


def test_world_simulator_basics():
    w = ToyWorld()
    assert w.act("冰块", "heat")["after"] == "液态"
    assert w.act("冰块", "heat")["after"] == "气态"
    bad = w.act("冰块", "摇晃")
    assert not bad["ok"] and "unknown action" in bad["error"]
    assert w.act("不存在的东西", "heat")["ok"] is False
    day = ToyWorld().scripted_day()
    assert [e["after"] for e in day] == ["液态", "气态", "固态", "液态"]


def test_1_new_object_attribute_relation(tmp_path):
    mem = Memory(str(tmp_path / "w1.db"))
    try:
        w = ToyWorld()
        w.add_object("雪糕", "固态")  # new object, known state relation
        assert w.objects["雪糕"] == "固态"
        assert any(e["type"] == "new_object" for e in w.events)
    finally:
        mem.close()


def test_2_instance_vs_category(tmp_path):
    from cognitive_ai.cli import seed_from_json
    mem = Memory(str(tmp_path / "w2.db"))
    seed_from_json(mem, os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "concepts.json")))
    try:
        o = Ontology(mem)
        o.add_concept("ice1", "冰块一号", parents=["entity"])
        o.add_attribute("ice1", "state", "固态", status="verified")
        # instance carries its own state; the category rule (below)
        # applies by state value, never by object name
        assert o.inherited_attributes("ice1")["state"]["value"] == "固态"
    finally:
        mem.close()


def _learn_heat_rules(mem, onto):
    """One validated rule per world transition.

    Each transition is observed 3 independent times (distinct evidence);
    train uses repeats 0-1, heldout uses repeat 2. Same-transition
    repeats are independent evidence (corroboration rule); different
    transitions never share rows.
    """
    from cognitive_ai.world import ToyWorld as W
    w = W()
    all_rows = w.examples(repeats=3)
    by_t = {}
    for r in all_rows:
        by_t.setdefault((r["inputs"]["state"], r["inputs"]["action"]),
                        []).append(r)
    rids = []
    for (state, action), rows in sorted(by_t.items()):
        train, held = rows[:2], rows[2:]
        assert train and held and \
            {r["evidence"] for r in train}.isdisjoint(
                {r["evidence"] for r in held})
        # other transitions serve as negatives so each rule pins BOTH
        # state and action instead of an over-broad single attribute
        negs = [r for (s2, a2), rs in by_t.items() for r in rs
                if (s2, a2) != (state, action)]
        ledger = run_cycle(mem, onto, f"t_{state}_{action}", "concepts",
                           train, negs, held, auto_activate=True)
        assert ledger["outcome"] in ("rule", "rule-active"), ledger
        rids.append(ledger["stages"].get("candidate"))
    return rids


def assert_disjoint(a, b):
    ka = {(str(sorted(e["inputs"].items())), str(sorted(e["outputs"].items())))
          for e in a}
    kb = {(str(sorted(e["inputs"].items())), str(sorted(e["outputs"].items())))
          for e in b}
    assert not (ka & kb), "train/heldout leakage"


def test_3_new_rule_and_4_composition(tmp_path):
    mem = Memory(str(tmp_path / "w34.db"))
    from cognitive_ai.cli import seed_from_json
    seed_from_json(mem, os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "concepts.json")))
    onto = Ontology(mem)
    try:
        rids = _learn_heat_rules(mem, onto)
        assert len(rids) == 6, rids
        # compose two learned steps on an unseen start: 固态 -heat-> 液态
        # -heat-> 气态 (never demonstrated as one example)
        from cognitive_ai import rulelearn as R
        rules = {r["id"]: R.selected_spec(r) for r in
                 mem.list_rules_learned(status="active")}
        assert len(rules) == 6, list(rules)
        state, trace = "固态", []
        for _ in range(2):
            outs = [R.apply_spec(spec, {"state": state, "action": "heat"})
                    for spec in rules.values()]
            ok = [o for o in outs if o["status"] == "ok"]
            assert ok, "composition step must fire"
            state = ok[0]["output"]["label"]
            trace.append((state, ok[0]["trace"]))
        assert state == "气态", trace
        mem.add_experience("evaluation", "world composition heat∘heat",
                           {"op": "learn_cycle", "outcome": "composition",
                            "trace": str(trace)})
    finally:
        mem.close()


def test_5_counterexample_and_6_narrowed_scope(tmp_path):
    mem = Memory(str(tmp_path / "w56.db"))
    onto = Ontology(mem)
    try:
        # over-broad first hypothesis from ONE observation (heat->液态);
        # single evidence is inducible but must not validate
        pos = [ex("a", {"state": "固态", "action": "heat"},
                  "液态")]
        rep = RL.induce(pos, [], domain="concepts")
        assert rep["candidates"], "single-evidence hypothesis expected"
        neg = [ex("n", {"state": "气态", "action": "heat"},
                  "气态")]
        val = RL.validate_rule(rep["candidates"][0]["spec"], pos, pos,
                               neg, onto)
        assert not val["passed"], "counterexample must veto"
        assert val["checks"]["negatives_refused"] == "0/1"
        # with a second independent observation of the same transition
        # plus the counterexample as scope, the rule narrows and passes
        pos2 = pos + [ex("b", {"state": "固态", "action": "heat"},
                         "液态")]
        rep2 = RL.induce(pos2, neg, domain="concepts")
        assert rep2["candidates"]
        form = rep2["candidates"][0]["form"]
        assert "state" in form, form
        val2 = RL.validate_rule(rep2["candidates"][0]["spec"], pos2,
                                pos2[:1], neg, onto)
        assert val2["passed"], val2
    finally:
        mem.close()


def test_7_transfer_to_new_object_combo(tmp_path):
    mem = Memory(str(tmp_path / "w7.db"))
    from cognitive_ai.cli import seed_from_json
    seed_from_json(mem, os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "concepts.json")))
    onto = Ontology(mem)
    try:
        rids = _learn_heat_rules(mem, onto)
        assert len(rids) == 6
        from cognitive_ai import rulelearn as R
        specs = [R.selected_spec(r) for r in
                 mem.list_rules_learned(status="active")]
        # new object 雪糕 shares only the STATE value: transfer by state
        outs = [R.apply_spec(s, {"state": "固态", "action": "heat"})
                for s in specs]
        assert any(o["status"] == "ok" and o["output"]["label"] == "液态"
                   for o in outs)
    finally:
        mem.close()


def test_8_missing_action_reported(tmp_path):
    mem = Memory(str(tmp_path / "w8.db"))
    onto = Ontology(mem)
    try:
        _learn_heat_rules(mem, onto)
        w = ToyWorld()
        r = w.act("冰块", "摇晃")
        assert not r["ok"]
        assert "heat" in r["error"] and "cool" in r["error"]
    finally:
        mem.close()


def test_ledger_and_outcome_kinds(tmp_path):
    mem = Memory(str(tmp_path / "w9.db"))
    onto = Ontology(mem)
    try:
        _learn_heat_rules(mem, onto)
        exps = [e for e in mem.list_experiences(limit=100)
                if "learn cycle" in e["summary"]]
        assert exps
        import json
        for e in exps:
            d = json.loads(e["detail"])
            assert d.get("op") == "learn_cycle"
            assert d.get("outcome") in ("rule", "rule-active",
                                        "rejected", "insufficient",
                                        "needs_choice", "composition")
    finally:
        mem.close()


def test_qa_baseline_vs_after_learning(tmp_path):
    """Baseline (no rules) abstains; after learning, structured queries
    about world transitions are answered from active rules. Old KB
    questions keep working (regression spot-check)."""
    from cognitive_ai import rulelearn as R
    mem = Memory(str(tmp_path / "wqa.db"))
    from cognitive_ai.cli import seed_from_json
    seed_from_json(mem, os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "concepts.json")))
    onto = Ontology(mem)
    try:
        q = {"state": "固态", "action": "heat"}
        before = R.answer_query(q, mem, onto)
        assert before["status"] == "unknown"
        _learn_heat_rules(mem, onto)
        after = R.answer_query(q, mem, onto)
        assert after["status"] == "answered"
        assert after["answer"] == "液态", after
        assert after["rule_id"]
        # regression: seed knowledge untouched and queryable
        assert onto.inherited_attributes("cup")["material"]["value"] == \
            "ceramic"
    finally:
        mem.close()
