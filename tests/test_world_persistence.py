"""Phase M4-D world persistence (isolated tmp DBs, never Temp/)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import world_dynamics as wd
from cognitive_ai.memory import Memory
from cognitive_ai.models import (WorldObjectRef, WorldProposition,
                                 WorldState, WorldStateFact, WorldTerm,
                                 WorldObjectKind, WorldPolarity,
                                 WorldModality)
from cognitive_ai.world_persistence import (WorldPersistence, decode_event,
                                            decode_state, encode_event,
                                            encode_state, from_jsonable,
                                            to_jsonable)

CUP = WorldObjectRef("cupA", "杯子A", WorldObjectKind.INDIVIDUAL)
WATER = WorldObjectRef("waterA", "水A", WorldObjectKind.INDIVIDUAL)
TABLE = WorldObjectRef("tableB", "桌子B", WorldObjectKind.INDIVIDUAL)


def mem_with_seed(tmp_path, tag):
    m = Memory(str(tmp_path / f"{tag}.db"))
    from cognitive_ai.ontology import Ontology
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("cup", "杯子", parents=["entity"])
    o.add_attribute("cup", "material", "ceramic", status="verified")
    return m


def rich_fact():
    return WorldStateFact(
        proposition=WorldProposition(
            subject=WorldTerm(ref=CUP), predicate="contains",
            object=WorldTerm(ref=WATER), polarity=WorldPolarity.POSITIVE,
            modality=WorldModality.ASSERTED,
            representation_origin="test", source_ref="test:1",
            raw={"meta": {"tags": ["水", "hot"], "n": 3},
                 "flag": True, "nothing": None}),
        event_ref="", origin="test")


def neg_fact():
    return WorldStateFact(
        proposition=WorldProposition(
            subject=WorldTerm(ref=CUP), predicate="on",
            object=WorldTerm(ref=TABLE), polarity=WorldPolarity.NEGATIVE,
            modality=WorldModality.UNKNOWN,
            representation_origin="test", source_ref="test:2", raw={}),
        event_ref="", origin="test")


def neg_inside_fact():
    return WorldStateFact(
        proposition=WorldProposition(
            subject=WorldTerm(ref=WATER), predicate="inside",
            object=WorldTerm(ref=CUP),
            polarity=WorldPolarity.NEGATIVE,
            modality=WorldModality.UNKNOWN,
            representation_origin="test", source_ref="test:0", raw={}),
        event_ref="", origin="test")


def lit_fact():
    return WorldStateFact(
        proposition=WorldProposition(
            subject=WorldTerm(ref=CUP), predicate="level",
            object=WorldTerm(literal=2, has_literal=True),
            polarity=WorldPolarity.POSITIVE,
            representation_origin="test", source_ref="test:3", raw={}),
        event_ref="", origin="test")


def fresh_service(tmp_path, tag):
    m = Memory(str(tmp_path / f"{tag}.db"))
    svc = WorldPersistence(m)
    out = svc.ensure_storage()
    assert out["ok"] is True, out
    return m, svc


def pour_action():
    from cognitive_ai.models import (WorldActionSchema, WorldEffect,
                                     WorldLiteralPattern, WorldPatternTerm)
    return WorldActionSchema(
        action_id="pour", name="倒入",
        params=("cup", "liquid"),
        premises=(WorldLiteralPattern(
            subject=WorldPatternTerm(variable="liquid"),
            predicate="inside",
            object=WorldPatternTerm(variable="cup"),
            polarity=WorldPolarity.NEGATIVE),),
        effects=(WorldEffect(WorldLiteralPattern(
            subject=WorldPatternTerm(variable="liquid"),
            predicate="inside",
            object=WorldPatternTerm(variable="cup"),
            polarity=WorldPolarity.POSITIVE)),))


# ---- A. init & migration ----
def test_migration_repeatable(tmp_path):
    m, svc = fresh_service(tmp_path, "pA1")
    try:
        v0 = m.current_version()
        again = svc.ensure_storage()
        assert again == {"ok": True,
                         "migration_id": "world-model-storage-v1",
                         "applied": False}
        assert m.get_world_storage_version() == "world-model-storage-v1"
        assert m.current_version() == v0
    finally:
        m.close()


def test_initial_world_unique(tmp_path):
    m, svc = fresh_service(tmp_path, "pA2")
    try:
        s0 = wd.make_state("s0", 0, [rich_fact()])
        assert svc.create_world("w", "世界", s0)["ok"] is True
        assert svc.get_initial_state("w") == s0
        assert [s.state_id for s in svc.list_states("w")] == ["s0"]
        with pytest.raises(Exception):
            m.insert_world_state("sX", "w", None, 0, "world-state-v1")
    finally:
        m.close()


def test_legacy_data_untouched(tmp_path):
    m = mem_with_seed(tmp_path, "pA3")
    svc = WorldPersistence(m)
    before = (m.list_concepts(), m.get_attributes("cup"),
              m.list_claims(), m.list_rules(), m.current_version())
    try:
        svc.ensure_storage()
        s0 = wd.make_state("s0", 0, [rich_fact()])
        assert svc.create_world("w", "世界", s0)["ok"] is True
        after = (m.list_concepts(), m.get_attributes("cup"),
                 m.list_claims(), m.list_rules(), m.current_version())
        assert after == before
    finally:
        m.close()


def test_transact_rolls_back(tmp_path):
    m = Memory(str(tmp_path / "pA4.db"))
    try:
        m.execute("CREATE TABLE t_half(x TEXT PRIMARY KEY)", ())
        with pytest.raises(Exception):
            m.transact([
                ("INSERT INTO t_half(x) VALUES(?)", ("a",)),
                ("INSERT INTO t_half(x) VALUES(?)", ("a",)),
            ])
        assert m.query("SELECT * FROM t_half") == []
    finally:
        m.close()


def test_migration_partial_retry(tmp_path):
    m = Memory(str(tmp_path / "pA5.db"))
    svc = WorldPersistence(m)
    try:
        calls = {"n": 0}
        real = m.set_meta

        def flaky(key, value):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return real(key, value)

        m.set_meta = flaky
        with pytest.raises(RuntimeError):
            svc.ensure_storage()
        assert m.get_world_storage_version() is None
        m.set_meta = real
        out = svc.ensure_storage()
        assert out == {"ok": True,
                       "migration_id": "world-model-storage-v1",
                       "applied": True}
        assert m.get_world_storage_version() == "world-model-storage-v1"
    finally:
        m.close()


# ---- B. roundtrip ----
def test_state_roundtrip(tmp_path):
    m, svc = fresh_service(tmp_path, "pB1")
    try:
        s0 = wd.make_state("s0", 0, [rich_fact(), neg_fact(), lit_fact()])
        assert svc.create_world("w", "世界", s0)["ok"] is True
        back = svc.get_state("s0")
        assert back == s0
        assert back.facts[0].proposition.raw["meta"]["tags"][0] == "水"
        with pytest.raises(TypeError):
            back.facts[0].proposition.raw["meta"] = {}
    finally:
        m.close()


def test_event_roundtrip(tmp_path):
    m, svc = fresh_service(tmp_path, "pB2")
    try:
        from cognitive_ai.models import WorldProposition as _P
        s0 = wd.make_state("s0", 0, [WorldStateFact(
            proposition=_P(
                subject=WorldTerm(ref=WATER), predicate="inside",
                object=WorldTerm(ref=CUP),
                polarity=WorldPolarity.NEGATIVE),
            origin="test")])
        assert svc.create_world("w", "世界", s0)["ok"] is True
        res = wd.simulate_action(s0, pour_action(),
                                 {"cup": CUP, "liquid": WATER},
                                 new_state_id="s1", event_id="e1")
        assert res.outcome == "applied"
        assert svc.save_transition(res)["ok"] is True
        back = svc.get_event("e1")
        assert back == res.event
        assert back.simulated is True
        assert dict(back.bindings) == {"cup": CUP, "liquid": WATER}
        assert decode_event(encode_event(res.event)) == res.event
        assert decode_state(encode_state(res.new_state)) == res.new_state
    finally:
        m.close()


def test_codec_rejects_and_roundtrips():
    assert to_jsonable({"水": [1, True, None]}) == {"水": [1, True, None]}
    with pytest.raises(ValueError):
        from_jsonable({"__type__": "NoSuchThing"})
    with pytest.raises(TypeError):
        to_jsonable({"k": object()})
    with pytest.raises(TypeError):
        to_jsonable({1: "x"})
    with pytest.raises(ValueError):
        decode_state('{"format": "world-state-v9", "state": {}}')


# ---- C. branches ----
def test_branching_histories(tmp_path):
    m, svc = fresh_service(tmp_path, "pC1")
    try:
        other = WorldStateFact(
            proposition=WorldProposition(
                subject=WorldTerm(ref=TABLE), predicate="inside",
                object=WorldTerm(ref=CUP),
                polarity=WorldPolarity.NEGATIVE),
            origin="test")
        s0 = wd.make_state("s0", 0, [neg_inside_fact(), other])
        assert svc.create_world("w", "世界", s0)["ok"] is True
        r1 = wd.simulate_action(
            s0, pour_action(), {"cup": CUP, "liquid": WATER},
            new_state_id="s1", event_id="e1")
        r2 = wd.simulate_action(
            s0, pour_action(), {"cup": CUP, "liquid": TABLE},
            new_state_id="s2", event_id="e2")
        assert svc.save_transition(r1)["ok"] is True
        assert svc.save_transition(r2)["ok"] is True
        h1, h2 = svc.history("s1"), svc.history("s2")
        assert [s.state_id for s in h1["states"]] == ["s0", "s1"]
        assert [s.state_id for s in h2["states"]] == ["s0", "s2"]
        assert [e.event_id for e in h1["events"]] == ["e1"]
        assert [e.event_id for e in h2["events"]] == ["e2"]
        assert svc.get_state("s0") == s0
    finally:
        m.close()


# ---- D. transactions & duplicate ids ----
def test_duplicate_ids_refused(tmp_path):
    m, svc = fresh_service(tmp_path, "pD1")
    try:
        s0 = wd.make_state("s0", 0, [neg_inside_fact()])
        assert svc.create_world("w", "世界", s0)["ok"] is True
        dup = svc.create_world("w", "另一个", s0)
        assert dup["ok"] is False
        assert svc.get_world("w")["name"] == "世界"
        r = wd.simulate_action(s0, pour_action(),
                               {"cup": CUP, "liquid": WATER},
                               new_state_id="s1", event_id="e1")
        assert svc.save_transition(r)["ok"] is True
        r_again = wd.simulate_action(s0, pour_action(),
                                     {"cup": CUP, "liquid": TABLE},
                                     new_state_id="s1", event_id="e9")
        out = svc.save_transition(r_again)
        assert out["ok"] is False
        assert svc.get_state("s1").parent_state_id == "s0"
        assert svc.get_event("e9") is None
    finally:
        m.close()


def test_bad_links_refused(tmp_path):
    m, svc = fresh_service(tmp_path, "pD2")
    try:
        s0 = wd.make_state("s0", 0, [neg_inside_fact()])
        assert svc.create_world("w", "世界", s0)["ok"] is True
        r = wd.simulate_action(s0, pour_action(),
                               {"cup": CUP, "liquid": WATER},
                               new_state_id="s1", event_id="e1")
        import dataclasses
        bad_pre = dataclasses.replace(r.new_state, parent_state_id="ghost")
        bad = dataclasses.replace(r, new_state=bad_pre)
        assert svc.save_transition(bad)["ok"] is False
        bad_step = dataclasses.replace(
            r.event, step_index=r.event.step_index + 5)
        bad2 = dataclasses.replace(r, event=bad_step)
        assert svc.save_transition(bad2)["ok"] is False
        swapped = dataclasses.replace(r.event, from_state="s1",
                                      to_state="s0")
        bad3 = dataclasses.replace(r, event=swapped)
        assert svc.save_transition(bad3)["ok"] is False
        assert svc.get_state("s1") is None
        assert svc.get_event("e1") is None
    finally:
        m.close()


# ---- E. non-applied results ----
def test_non_applied_never_persist(tmp_path):
    m, svc = fresh_service(tmp_path, "pE1")
    try:
        s0 = wd.make_state("s0", 0, [])
        assert svc.create_world("w", "世界", s0)["ok"] is True
        action = pour_action()
        cases = [
            wd.simulate_action(s0, action, {"cup": CUP, "liquid": WATER}),
            wd.simulate_action(s0, action, {"cup": CUP}),
            wd.simulate_action(s0, action, {"cup": CUP, "liquid": WATER,
                                            "z": TABLE}),
        ]
        assert {c.outcome for c in cases} == {"unknown", "invalid"}
        for c in cases:
            assert svc.save_transition(c)["ok"] is False
        assert svc.list_states("w") == [s0]
        assert svc.list_events("w") == []
    finally:
        m.close()


# ---- F. read-only & isolation ----
def test_reads_change_nothing(tmp_path):
    m, svc = fresh_service(tmp_path, "pF1")
    try:
        s0 = wd.make_state("s0", 0, [neg_inside_fact()])
        assert svc.create_world("w", "世界", s0)["ok"] is True
        r = wd.simulate_action(s0, pour_action(),
                               {"cup": CUP, "liquid": WATER},
                               new_state_id="s1", event_id="e1")
        assert svc.save_transition(r)["ok"] is True

        def dump():
            return (m.query("SELECT * FROM worlds"),
                    m.query("SELECT * FROM world_states"),
                    m.query("SELECT * FROM world_state_facts"),
                    m.query("SELECT * FROM world_events"))

        before = dump()
        svc.get_world("w")
        svc.list_worlds()
        svc.get_state("s1")
        svc.get_initial_state("w")
        svc.list_states("w")
        svc.get_event("e1")
        svc.list_events("w")
        svc.history("s1")
        assert dump() == before
        wa = svc.create_world("wB", "乙", wd.make_state("b0", 0, []))
        assert wa["ok"] is True
        assert [s.state_id for s in svc.list_states("w")] == ["s0", "s1"]
    finally:
        m.close()


def test_persistence_touches_no_knowledge_tables(tmp_path):
    m = mem_with_seed(tmp_path, "pF2")
    svc = WorldPersistence(m)
    before = (m.list_concepts(), m.get_attributes("cup"),
              m.list_claims(), m.list_rules())
    try:
        svc.ensure_storage()
        s0 = wd.make_state("s0", 0, [neg_inside_fact()])
        assert svc.create_world("w", "世界", s0)["ok"] is True
        r = wd.simulate_action(s0, pour_action(),
                               {"cup": CUP, "liquid": WATER},
                               new_state_id="s1", event_id="e1")
        assert svc.save_transition(r)["ok"] is True
        svc.history("s1")
        after = (m.list_concepts(), m.get_attributes("cup"),
                 m.list_claims(), m.list_rules())
        assert after == before
    finally:
        m.close()


def test_codec_and_service_have_no_db_access():
    import re
    for rel in ("world_persistence.py",):
        src = open(os.path.join(os.path.dirname(__file__), "..", "src",
                                "cognitive_ai", rel), encoding="utf-8").read()
        assert ".conn" not in src, rel
        assert "sqlite3" not in src, rel
