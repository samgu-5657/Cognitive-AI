"""Regression tests for the rule lifecycle state machine.

The Web UI bug: validating an already-validated rule failed with
'validated -> validated not allowed', aborting before the activation
step. Re-validation and re-activation are now explicit idempotent
no-ops (no version bump, no history rewrite). Isolated DBs only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import rulelearn as RL
from cognitive_ai.memory import Memory


def ex(i, inputs, outputs):
    return {"id": i, "inputs": inputs, "outputs": outputs}


TRAIN = [ex("a", {"n": "1"}, {"out": "3"}),
         ex("b", {"n": "2"}, {"out": "5"}),
         ex("c", {"n": "4"}, {"out": "9"})]
HELD = [ex("d", {"n": "7"}, {"out": "15"}),
        ex("e", {"n": "10"}, {"out": "21"})]


def _validated_rule(tmp_path=None, db=":memory:"):
    mem = Memory(str(tmp_path / db) if tmp_path else ":memory:")
    rep = RL.induce(TRAIN, [], domain="math")
    assert len(rep["candidates"]) == 1
    rec = RL.make_proposal_record("lr_d1", "double_plus_one", "math",
                                  rep["candidates"], [])
    rec["spec"]["selected"] = 0
    mem.save_rule(rec)
    spec = RL.selected_spec(rec)
    val = RL.validate_rule(spec, TRAIN, HELD, [], None)
    assert val["passed"], val
    assert val["checks"]["negatives_refused"] == "n/a"
    rec["validation"] = {**val, "selected": 0}
    mem.save_rule(rec)
    assert RL.promote_record(mem, "lr_d1", "validated",
                             validation=rec["validation"])["ok"]
    return mem, spec


def test_revalidate_validated_rule_is_idempotent():
    mem, _ = _validated_rule()
    try:
        v0 = mem.get_rule("lr_d1")["version"]
        out = RL.promote_record(mem, "lr_d1", "validated",
                                validation={"passed": True, "selected": 0,
                                            "note": "second run"})
        assert out["ok"] and "already validated" in out["note"]
        row = mem.get_rule("lr_d1")
        assert row["version"] == v0 and row["status"] == "validated"
        assert row["validation"]["note"] == "second run"
    finally:
        mem.close()


def test_revalidate_with_failure_changes_nothing():
    mem, _ = _validated_rule()
    try:
        v0 = mem.get_rule("lr_d1")["version"]
        old = mem.get_rule("lr_d1")["validation"]
        out = RL.promote_record(mem, "lr_d1", "validated",
                                validation={"passed": False})
        assert not out["ok"]
        row = mem.get_rule("lr_d1")
        assert row["version"] == v0 and row["status"] == "validated"
        assert row["validation"] == old
    finally:
        mem.close()


def test_candidate_cannot_jump_to_active():
    mem = Memory(":memory:")
    try:
        rec = RL.make_proposal_record(
            "lr_j", "j", "math",
            RL.induce(TRAIN, [], domain="math")["candidates"], [])
        rec["spec"]["selected"] = 0
        mem.save_rule(rec)
        out = RL.promote_record(mem, "lr_j", "active",
                                validation={"passed": True, "selected": 0})
        assert not out["ok"] and "not allowed" in out["error"]
        assert mem.get_rule("lr_j")["status"] == "candidate"
    finally:
        mem.close()


def test_failed_validation_blocks_activation():
    mem = Memory(":memory:")
    try:
        rec = RL.make_proposal_record(
            "lr_f", "f", "math",
            RL.induce(TRAIN, [], domain="math")["candidates"], [])
        rec["spec"]["selected"] = 0
        mem.save_rule(rec)
        out = RL.promote_record(mem, "lr_f", "active",
                                validation={"passed": False})
        assert not out["ok"]
        out = RL.promote_record(mem, "lr_f", "validated",
                                validation={"passed": False})
        assert not out["ok"]
        assert mem.get_rule("lr_f")["status"] == "candidate"
    finally:
        mem.close()


def test_repeat_activation_is_noop():
    mem, _ = _validated_rule()
    try:
        assert RL.promote_record(mem, "lr_d1", "active")["ok"]
        v1 = mem.get_rule("lr_d1")["version"]
        n_events = len(mem.list_lifecycle("lr_d1"))
        out = RL.promote_record(mem, "lr_d1", "active")
        assert out["ok"] and "already active" in out["note"]
        row = mem.get_rule("lr_d1")
        assert row["version"] == v1 and row["status"] == "active"
        assert len(mem.list_lifecycle("lr_d1")) == n_events
        assert (row["calls"], row["successes"], row["failures"]) == (0, 0, 0)
    finally:
        mem.close()


def test_activated_rule_executes_unseen_input():
    mem, spec = _validated_rule()
    try:
        assert RL.promote_record(mem, "lr_d1", "active")["ok"]
        from cognitive_ai.ontology import Ontology
        from cognitive_ai.operations import ApplyLearnedRule
        from cognitive_ai.reasoner import Reasoner
        onto, rsn = Ontology(mem), Reasoner(mem)
        rec = ApplyLearnedRule().run(
            {"rule_id": "lr_d1", "inputs": {"n": "25"}},
            {"memory": mem, "ontology": onto, "reasoner": rsn})
        assert rec["status"] == "verified", rec
        assert rec["outputs"]["final"].get("_prev") == "51", rec
        assert rec["outputs"]["applied"][0]["rule_id"] == "lr_d1"
        row = mem.get_rule("lr_d1")
        assert (row["calls"], row["successes"]) == (1, 1)
    finally:
        mem.close()


def test_rollback_restores_validated_and_disables_execution():
    mem, _ = _validated_rule()
    try:
        assert RL.promote_record(mem, "lr_d1", "active")["ok"]
        assert RL.rollback_record(mem, "lr_d1")["ok"]
        row = mem.get_rule("lr_d1")
        assert row["status"] == "validated"
        from cognitive_ai.ontology import Ontology
        from cognitive_ai.operations import ApplyLearnedRule
        from cognitive_ai.reasoner import Reasoner
        rec = ApplyLearnedRule().run(
            {"rule_id": "lr_d1", "inputs": {"n": "25"}},
            {"memory": mem, "ontology": Ontology(mem),
             "reasoner": Reasoner(mem)})
        assert rec["status"] == "failed"
    finally:
        mem.close()


def test_web_validate_twice_then_activate(tmp_path):
    from cognitive_ai.web.service import Workbench
    import shutil
    data = tmp_path / "wdata"
    (data / "corpus").mkdir(parents=True)
    shutil.copy(os.path.join(os.path.dirname(__file__), "..",
                             "data", "concepts.json"),
                data / "concepts.json")
    wb = Workbench(str(data), str(tmp_path / "wmodels"))
    try:
        train = [{"id": e["id"], "inputs": e["inputs"],
                  "outputs": e["outputs"]} for e in TRAIN]
        held = [{"id": e["id"], "inputs": e["inputs"],
                 "outputs": e["outputs"]} for e in HELD]
        rid = wb.rule_induce("double_plus_one", "math", train, [])["saved"]
        first = wb.rule_validate(rid, train, held, [], activate=True)
        assert first["validation"]["passed"] is True
        assert first["promotion"]["ok"] is True
        assert wb.mem.get_rule(rid)["status"] == "active"
        v1 = wb.mem.get_rule(rid)["version"]
        # the exact UI double-click sequence that used to fail
        second = wb.rule_validate(rid, train, held, [], activate=True)
        assert second["promotion"]["ok"] is True, second
        row = wb.mem.get_rule(rid)
        assert row["status"] == "active" and row["version"] == v1
        assert row["validation"]["checks"]["negatives_refused"] == "n/a"
    finally:
        wb.close()
