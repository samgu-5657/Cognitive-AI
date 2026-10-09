"""Lifecycle tests: transition shape, promotion gates, contest safety."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import lifecycle
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology


def ev(sources):
    return [{"source": s, "content_hash": f"hash{i}",
             "relevance": "direct"} for i, s in enumerate(sources)]


def test_transition_shape():
    assert lifecycle.can_transition("candidate", "hypothesis")
    assert lifecycle.can_transition("hypothesis", "supported")
    assert lifecycle.can_transition("supported", "verified")
    assert lifecycle.can_transition("verified", "contested")
    assert not lifecycle.can_transition("candidate", "verified")
    assert not lifecycle.can_transition("candidate", "supported")
    assert not lifecycle.can_transition("verified", "hypothesis")
    assert not lifecycle.can_transition("superseded", "hypothesis")


def test_two_files_never_verify():
    ok, msg = lifecycle.check_promotion(
        "supported", "verified", ev(["a", "b", "c"]))
    assert not ok and "confirmation" in msg
    ok, _ = lifecycle.check_promotion(
        "supported", "verified", ev(["a", "b", "c"]), confirmed_by="operator")
    assert ok
    ok, msg = lifecycle.check_promotion("hypothesis", "supported", ev(["a"]))
    assert not ok and "3" in msg


def test_promote_attribute_end_to_end(tmp_path):
    mem = Memory(str(tmp_path / "l.db"))
    try:
        o = Ontology(mem)
        o.add_concept("entity", "e")
        o.add_concept("cup", "cup", parents=["entity"])
        aid = o.add_attribute("cup", "location", "kitchen",
                              status="hypothesis")
        # denied: not enough sources for supported
        r = lifecycle.promote_attribute(mem, aid, "supported", ev(["a"]))
        assert not r["ok"]
        assert mem.query_one("SELECT * FROM attributes WHERE id=?",
                             (aid,))["status"] == "hypothesis"
        # allowed: 3 independent sources
        r = lifecycle.promote_attribute(mem, aid, "supported",
                                        ev(["a", "b", "c"]),
                                        reason="third source arrived")
        assert r["ok"] and r["to"] == "supported"
        # verified needs confirmation
        r = lifecycle.promote_attribute(mem, aid, "verified", ev(["a", "b", "c"]))
        assert not r["ok"]
        r = lifecycle.promote_attribute(mem, aid, "verified",
                                        ev(["a", "b", "c"]),
                                        confirmed_by="operator")
        assert r["ok"] and r["to"] == "verified"
        trail = [e for e in mem.list_lifecycle("cup.location")
                 if e["to_state"] in ("supported", "verified")]
        assert [e["to_state"] for e in trail] == ["supported", "verified"]
    finally:
        mem.close()


def test_contest_keeps_row_untouched(tmp_path):
    mem = Memory(str(tmp_path / "l2.db"))
    try:
        o = Ontology(mem)
        o.add_concept("entity", "e")
        o.add_concept("cup", "cup", parents=["entity"])
        aid = o.add_attribute("cup", "material", "ceramic", status="verified")
        r = lifecycle.contest_attribute(mem, aid, ev(["x", "y"]),
                                        reason="rival glass reported")
        assert r["ok"] and r["row_untouched"]
        rows = [a for a in mem.get_attributes("cup") if a["key"] == "material"]
        assert [(a["value"], a["status"]) for a in rows] == [("ceramic", "verified")]
        assert mem.list_lifecycle("cup.material")
    finally:
        mem.close()
