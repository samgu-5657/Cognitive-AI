"""Phase M4-B world-model core + read-only adapter (isolated DBs).

Every test builds its own SQLite file, reads it through world_model,
and checks structure, provenance and read-only behavior. Nothing here
touches the production ask/learn paths except the compatibility test,
which only proves they still work alongside.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import world_model as wm
from cognitive_ai.memory import Memory
from cognitive_ai.models import (WorldObjectKind, WorldSnapshot,
                                 WorldEvidenceDirection, WorldModality,
                                 WorldPolarity)
from cognitive_ai.ontology import Ontology


def fresh(tmp_path, tag="wm"):
    m = Memory(str(tmp_path / f"{tag}.db"))
    o = Ontology(m)
    o.add_concept("entity", "实体")
    o.add_concept("container", "容器", parents=["entity"])
    o.add_concept("cup", "杯子", parents=["container"])
    o.add_concept("cup2", "杯子", parents=["entity"])
    o.add_attribute("cup", "material", "ceramic", status="verified")
    o.add_attribute("cup", "function", "drinking", status="verified")
    o.add_attribute("cup", "color", "white", status="hypothesis")
    o.add_relation("cup", "stored_in", "container", status="verified")
    return m


def counts(m):
    return (
        len(m.list_concepts()),
        len(m.query("SELECT * FROM attributes")),
        len(m.query("SELECT * FROM relations")),
        len(m.list_claims()),
        len(m.query("SELECT * FROM evidences")),
        len(m.list_rules()),
        len(m.query("SELECT * FROM lifecycle_events")),
        len(m.list_experiences(limit=100000)),
    )


def test_identity_stable_and_legacy(tmp_path):
    m = fresh(tmp_path)
    try:
        snap = wm.build_snapshot(m)
        by_id = {o.ref_id: o for o in snap.objects}
        assert by_id["cup"].name == "杯子"
        assert all(o.kind == WorldObjectKind.LEGACY_CONCEPT
                   for o in snap.objects), snap.objects
        cups = [o for o in snap.objects if o.name == "杯子"]
        assert {o.ref_id for o in cups} == {"cup", "cup2"}, cups
    finally:
        m.close()


def test_attribute_relation_mapping(tmp_path):
    m = fresh(tmp_path, tag="wm2")
    try:
        snap = wm.build_snapshot(m)
        attrs = snap.filter_facts(store="attribute")
        rels = snap.filter_facts(store="relation")
        assert attrs and rels, snap
        mat = [f for f in attrs if f.proposition.predicate == "material"]
        assert len(mat) == 1, attrs
        assert mat[0].proposition.object.literal == "ceramic"
        assert mat[0].proposition.object.has_literal is True
        assert mat[0].proposition.raw == {"key": "material",
                                         "value": "ceramic"}
        assert mat[0].record_id.startswith("attribute:")
        cap = [f for f in attrs if f.proposition.predicate == "function"]
        assert cap and cap[0].proposition.modality == WorldModality.CAPABLE
        assert mat[0].proposition.modality == WorldModality.UNKNOWN
        rel = rels[0]
        assert rel.proposition.predicate == "stored_in"
        assert rel.proposition.object.ref is not None
        assert rel.proposition.object.ref.ref_id == "container"
        assert rel.proposition.object.missing is False
        assert rel.proposition.raw["dst_id"] == "container"
        assert rel.record_id.startswith("relation:")
        assert rel.proposition.subject.ref.ref_id == "cup"
        assert mat[0].proposition.subject.ref.ref_id == "cup"
    finally:
        m.close()


def test_unresolvable_reference_not_faked(tmp_path):
    m = fresh(tmp_path, tag="wm3")
    try:
        o = Ontology(m)
        o.add_concept("ghost_holder", "容灵", parents=["entity"])
        m.execute("INSERT INTO relations(src_id,rel_type,dst_id) "
                  "VALUES(?,?,?)", ("ghost_holder", "holds", "no_such_thing"))
        snap = wm.build_snapshot(m)
        got = [f for f in snap.filter_facts(subject="ghost_holder")]
        assert len(got) == 1, got
        term = got[0].proposition.object
        assert term.ref is None, term
        assert term.missing is True and term.unresolved == "no_such_thing"
    finally:
        m.close()


def test_status_evidence_separation(tmp_path):
    m = fresh(tmp_path, tag="wm4")
    try:
        snap = wm.build_snapshot(m)
        assert not hasattr(snap.facts[0].proposition, "truth")
        assert not hasattr(snap.facts[0].proposition, "verdict")
        assert not hasattr(snap.facts[0], "verdict")
        by_pred = {}
        for f in snap.facts:
            by_pred.setdefault(f.proposition.predicate, f.status)
        assert by_pred["material"] == "verified"
        assert by_pred["color"] == "hypothesis"
        assert all(f.evidence == () for f in snap.facts), snap.facts
        assert all(f.derivation is None for f in snap.facts)
        assert all(e.direction == WorldEvidenceDirection.UNKNOWN
                   for c in snap.claims for e in c.evidence)
        colored = [f for f in snap.facts
                   if f.proposition.predicate == "color"][0]
        assert colored.proposition.polarity == WorldPolarity.POSITIVE
    finally:
        m.close()


def test_contested_never_upgraded(tmp_path):
    m = fresh(tmp_path, tag="wm5")
    try:
        m.add_attribute("cup", "material", "glass", status="contested")
        snap = wm.build_snapshot(m)
        mats = sorted(
            (f.proposition.object.literal, f.status)
            for f in snap.facts
            if f.proposition.predicate == "material")
        assert mats == [("ceramic", "verified"), ("glass", "contested")], mats
    finally:
        m.close()


def test_claim_evidence_links_only_where_stored(tmp_path):
    m = fresh(tmp_path, tag="wm6")
    try:
        cid = m.add_claim("cup.material 可疑", subject_id="cup",
                          status="contested")
        eid = m.add_evidence(cid, "rival observed", "observation")
        snap = wm.build_snapshot(m)
        assert len(snap.claims) == 1
        claim = snap.claims[0]
        assert claim.text == "cup.material 可疑" and claim.subject_ref == "cup"
        assert [(e.evidence_id, e.claim_ref) for e in claim.evidence] == \
            [(eid, f"claim:{cid}")]
        assert all(f.evidence == () for f in snap.facts)
    finally:
        m.close()


def test_read_only_snapshot(tmp_path):
    m = fresh(tmp_path, tag="wm7")
    try:
        n0 = counts(m)
        s1 = wm.build_snapshot(m)
        s2 = wm.build_snapshot(m)
        assert counts(m) == n0
        assert s1 == s2
        assert len(s1.objects) == 4 and len(s1.facts) == 4, s1
    finally:
        m.close()


def test_adapter_uses_only_memory_reads():
    src = open(os.path.join(os.path.dirname(__file__), "..", "src",
                            "cognitive_ai", "world_model.py"),
               encoding="utf-8").read()
    for banned in ("sqlite3", ".conn", "INSERT", "UPDATE", "DELETE",
                   "add_attribute", "add_claim", "add_evidence",
                   "set_pending_status", "upsert", "record_lifecycle",
                   "direct_apply", "forward_chain"):
        assert banned not in src, banned


def test_legacy_pipeline_still_works(tmp_path):
    from cognitive_ai.learner import Learner
    from cognitive_ai.semground import answer_question
    m = fresh(tmp_path, tag="wm8")
    try:
        v = answer_question("容器是什么？", m)
        assert v.verdict == "supported", v.to_dict()
        cand = Learner(m, "nowhere").propose_from_utterance("容器可以装水")
        assert cand["kind"] == "attribute" and cand["can_confirm"], cand
        assert isinstance(wm.build_snapshot(m), WorldSnapshot)
    finally:
        m.close()
