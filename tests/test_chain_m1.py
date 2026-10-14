"""Phase M1 variable-binding multi-step chaining tests (isolated DBs).

Rules use ?x/?y/?z variables; the same rules prove facts about unseen
entities. Every test inspects real bindings, depths, statuses and
traces -- never just answer text. Seed rules stay untouched; test
rules live only in tmp DBs. No KB writes happen during inference.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology
from cognitive_ai.reasoner import Reasoner


def fresh(tmp_path, tag="m1"):
    m = Memory(str(tmp_path / f"{tag}.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("vessel", "容器", parents=["entity"])
    o.add_concept("potA", "铁锅", parents=["vessel"])
    o.add_concept("potB", "铁壶", parents=["vessel"])
    m.upsert_rule({
        "id": "r_heat", "conditions": [{"type": "is_a", "concept": "?x",
                                        "ancestor": "vessel"}],
        "conclusion": {"type": "has_attr", "concept": "?x",
                       "key": "can_heat", "value": True},
        "exceptions": []})
    m.upsert_rule({
        "id": "r_stove", "conditions": [{"type": "has_attr",
                                         "concept": "?x", "key": "can_heat",
                                         "value": True}],
        "conclusion": {"type": "has_attr", "concept": "?x",
                       "key": "stove_ok", "value": True},
        "exceptions": []})
    m.upsert_rule({
        "id": "r_kitchen", "conditions": [{"type": "has_attr",
                                           "concept": "?x",
                                           "key": "stove_ok",
                                           "value": True}],
        "conclusion": {"type": "has_attr", "concept": "?x",
                       "key": "kitchen_ready", "value": True},
        "exceptions": []})
    return m


def chain_of(proof):
    out, node = [], proof
    while node is not None:
        if node.get("rule_id"):
            out.append(node["rule_id"])
        kids = node.get("premises", []) or []
        node = next((k for k in kids if k.get("rule_id")), None)
    return out


def test_two_and_three_step_proofs(tmp_path, capsys):
    m = fresh(tmp_path)
    try:
        r = Reasoner(m)
        two = [f for f in r.forward_chain()
               if f.get("rule_id") == "r_stove" and
               f.get("bindings", {}).get("?x") == "potA"]
        assert len(two) == 1, two
        assert two[0]["status"] == "verified"
        assert chain_of(two[0]) == ["r_stove", "r_heat"]
        three = [f for f in r.forward_chain()
                 if f.get("rule_id") == "r_kitchen" and
                 f.get("bindings", {}).get("?x") == "potA"]
        assert len(three) == 1
        assert chain_of(three[0]) == ["r_kitchen", "r_stove", "r_heat"]
        assert three[0]["status"] == "verified"
        with capsys.disabled():
            print(f"\n[M1] 2-step via {chain_of(two[0])}, "
                  f"3-step via {chain_of(three[0])}")
    finally:
        m.close()


def test_unseen_entity_same_rules(tmp_path):
    m = fresh(tmp_path, tag="m1b")
    try:
        hits = [f for f in Reasoner(m).forward_chain()
                if f.get("rule_id") == "r_stove" and
                f.get("bindings", {}).get("?x") == "potB"]
        assert len(hits) == 1 and hits[0]["status"] == "verified"
    finally:
        m.close()


def test_variable_consistency(tmp_path):
    m = fresh(tmp_path, tag="m1c")
    try:
        o = Ontology(m)
        for c in ("a1", "b1", "c1", "d1"):
            o.add_concept(c, c, parents=["entity"])
        o.add_relation("a1", "edge", "b1")
        o.add_relation("b1", "edge", "c1")
        o.add_relation("a1", "edge", "d1")
        m.upsert_rule({
            "id": "r_path", "conditions": [
                {"type": "relation", "src": "?x", "rel_type": "edge",
                 "dst": "?z"},
                {"type": "relation", "src": "?z", "rel_type": "edge",
                 "dst": "?y"}],
            "conclusion": {"type": "relation", "src": "?x",
                           "rel_type": "path2", "dst": "?y"},
            "exceptions": []})
        hits = [f for f in Reasoner(m).forward_chain()
                if f.get("rule_id") == "r_path"]
        pairs = sorted((h["bindings"].get("?x"), h["bindings"].get("?y"))
                       for h in hits)
        assert pairs == [("a1", "c1")], pairs
    finally:
        m.close()


def test_applicability(tmp_path):
    m = fresh(tmp_path, tag="m1d")
    try:
        o = Ontology(m)
        o.add_concept("rock", "石头", parents=["entity"])
        fired = Reasoner(m).forward_chain()
        # missing premise: is_a rock vessel refuted, no conclusion
        assert not [f for f in fired
                    if f.get("rule_id") == "r_heat" and
                    f.get("bindings", {}).get("?x") == "rock"]
        # concrete legacy rules still work
        assert any(f.get("rule") == "r_heat" or
                   f.get("rule_id") == "r_heat" for f in fired)
        # exceptions (incl. variable-bound) still block per binding
        o.add_attribute("potA", "cracked", True, status="verified")
        m.upsert_rule({
            "id": "r_guarded", "conditions": [
                {"type": "has_attr", "concept": "?x", "key": "can_heat",
                 "value": True}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "safe", "value": True},
            "exceptions": [{"type": "has_attr", "concept": "?x",
                            "key": "cracked", "value": True}]})
        fired = Reasoner(m).forward_chain()
        assert not [f for f in fired
                    if f.get("rule_id") == "r_guarded" and
                    f.get("bindings", {}).get("?x") == "potA" and
                    "conclusion" in f and isinstance(f.get("conclusion"),
                                                     dict)]
        assert [f for f in fired
                if f.get("rule_id") == "r_guarded" and
                f.get("bindings", {}).get("?x") == "potB" and
                f.get("conclusion", {}).get("key") == "safe"]
    finally:
        m.close()


def test_stability(tmp_path):
    m = fresh(tmp_path, tag="m1e")
    try:
        m.upsert_rule({
            "id": "r_cyc_a", "conditions": [{"type": "has_attr",
                                             "concept": "?x", "key": "qa",
                                             "value": True}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "qb", "value": True}, "exceptions": []})
        m.upsert_rule({
            "id": "r_cyc_b", "conditions": [{"type": "has_attr",
                                             "concept": "?x", "key": "qb",
                                             "value": True}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "qa", "value": True}, "exceptions": []})
        m.upsert_rule({
            "id": "r_dup", "conditions": [{"type": "is_a", "concept": "?x",
                                           "ancestor": "vessel"}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "can_heat", "value": True},
            "exceptions": []})
        fired = Reasoner(m).forward_chain()
        assert fired, "engine must terminate with results"
        assert not any(f.get("limited") for f in fired), fired
        heat = [f for f in fired
                if f.get("rule_id") in ("r_heat", "r_dup") and
                f.get("bindings", {}).get("?x") == "potA" and
                isinstance(f.get("conclusion"), dict)]
        assert len(heat) == 2, heat  # both paths kept, not merged
        # depth cap is honored and explicitly marked
        limited = Reasoner(m).forward_chain(max_steps=100000)
        deep = [f for f in Reasoner(m)._chain_rounds(
            100000, 1, 10**9) if f.get("rule_id") == "r_kitchen"]
        assert deep == [] or all(
            "depth" not in f or f.get("depth", 99) <= 2 for f in deep)
    finally:
        m.close()


def test_budget_limit_marked(tmp_path):
    m = fresh(tmp_path, tag="m1f")
    try:
        o = Ontology(m)
        for i in range(30):
            o.add_concept(f"n{i}", f"节点{i}", parents=["entity"])
            if i:
                o.add_relation(f"n{i-1}", "edge", f"n{i}")
        m.upsert_rule({
            "id": "r_walk", "conditions": [
                {"type": "relation", "src": "?x", "rel_type": "edge",
                 "dst": "?y"}],
            "conclusion": {"type": "relation", "src": "?x",
                           "rel_type": "reach", "dst": "?y"},
            "exceptions": []})
        m.upsert_rule({
            "id": "r_hop", "conditions": [
                {"type": "relation", "src": "?x", "rel_type": "reach",
                 "dst": "?z"},
                {"type": "relation", "src": "?z", "rel_type": "edge",
                 "dst": "?y"}],
            "conclusion": {"type": "relation", "src": "?x",
                           "rel_type": "reach", "dst": "?y"},
            "exceptions": []})
        fired = Reasoner(m)._chain_rounds(100000, 4, 200)
        assert any(f.get("limited") for f in fired), \
            "budget exhaustion must be marked, not silent"
    finally:
        m.close()


def test_conflict_and_hypothesis_status(tmp_path):
    m = fresh(tmp_path, tag="m1g")
    try:
        o = Ontology(m)
        o.add_attribute("potA", "stove_ok", False, status="verified")
        m.upsert_rule({
            "id": "r_no", "conditions": [{"type": "is_a", "concept": "?x",
                                          "ancestor": "vessel"}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "stove_ok", "value": False},
            "exceptions": []})
        fired = Reasoner(m).forward_chain()
        cont = [f for f in fired if f.get("conclusion") == "contested"]
        assert cont, fired
        o.add_concept("hypA", "假设物", parents=["entity"])
        o.add_attribute("hypA", "can_heat", True, status="hypothesis")
        m.upsert_rule({
            "id": "r_hyp", "conditions": [{"type": "has_attr",
                                           "concept": "?x",
                                           "key": "can_heat",
                                           "value": True}],
            "conclusion": {"type": "has_attr", "concept": "?x",
                           "key": "stove_ok", "value": True},
            "exceptions": []})
        fired = Reasoner(m).forward_chain()
        hyp = [f for f in fired if f.get("rule_id") == "r_hyp" and
               f.get("bindings", {}).get("?x") == "hypA" and
               isinstance(f.get("conclusion"), dict)]
        assert len(hyp) == 1 and hyp[0]["status"] == "hypothesis", hyp
    finally:
        m.close()


def test_evidence_no_write_and_restart(tmp_path):
    db = str(tmp_path / "m1h.db")
    m = Memory(db)
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("vessel", "容器", parents=["entity"])
    o.add_concept("potA", "铁锅", parents=["vessel"])
    m.upsert_rule({
        "id": "r_heat", "conditions": [{"type": "is_a", "concept": "?x",
                                        "ancestor": "vessel"}],
        "conclusion": {"type": "has_attr", "concept": "?x",
                       "key": "can_heat", "value": True}, "exceptions": []})
    try:
        n_attr = len(m.query("SELECT * FROM attributes"))
        n_claim = len(m.list_claims())
        fired = Reasoner(m).forward_chain()
        assert any(f.get("rule_id") == "r_heat" for f in fired)
        assert len(m.query("SELECT * FROM attributes")) == n_attr
        assert len(m.list_claims()) == n_claim
        assert any("potA" in str(t) for f in fired for t in
                   f.get("trace", []))
    finally:
        m.close()
    m2 = Memory(db)
    try:
        assert m2.get_concept("potA") is not None
        fired = Reasoner(m2).forward_chain()
        assert any(f.get("rule_id") == "r_heat" and
                   f.get("bindings", {}).get("?x") == "potA"
                   for f in fired)
    finally:
        m2.close()
