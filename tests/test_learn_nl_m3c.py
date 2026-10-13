"""Phase M3C candidate learning loop tests (isolated DBs, real CLI).

Every test runs the actual learn/confirm/reject/pending/understand
commands in a subprocess against a scratch COGNITIVE_AI_DATA dir and
checks real row counts, statuses, sources and retrieval results --
never just CLI output text.
"""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))


def fresh_data(tmp_path, tag="lc"):
    data = tmp_path / tag
    (data / "corpus").mkdir(parents=True)
    shutil.copy(SEED, data / "concepts.json")
    return data


def seed_db(data):
    """Force DB creation + seed via a read-only command."""
    cli(data, "doctor")
    return data


def cli(data, *args):
    env = {**os.environ,
           "PYTHONPATH": os.path.abspath(os.path.join(
               os.path.dirname(__file__), "..", "src")),
           "COGNITIVE_AI_DATA": str(data)}
    r = subprocess.run(
        [sys.executable, "-m", "cognitive_ai.cli", *args],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."), env=env)
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads(r.stdout)


def db_counts(data):
    from cognitive_ai.memory import Memory
    mem = Memory(str(data / "cognitive.db"))
    try:
        return (len(mem.query("SELECT * FROM attributes")),
                len(mem.list_claims()),
                len(mem.query("SELECT * FROM relations")),
                len(mem.query("SELECT * FROM concepts")),
                len(mem.list_pending_candidates()))
    finally:
        mem.close()


def test_1_propose_writes_no_accepted_knowledge(tmp_path):
    data = seed_db(fresh_data(tmp_path))
    before = db_counts(data)[:4]
    p = cli(data, "learn", "杯子有把手")
    assert p["can_confirm"] is True and p["kind"] == "attribute"
    assert db_counts(data)[:4] == before
    assert db_counts(data)[4] == 1


def test_2_confirm_writes_exact_records(tmp_path):
    data = fresh_data(tmp_path, tag="lc2")
    p = cli(data, "learn", "杯子有把手")
    out = cli(data, "confirm", str(p["id"]))
    assert out["ok"] is True
    from cognitive_ai.memory import Memory
    mem = Memory(str(data / "cognitive.db"))
    try:
        rows = [a for a in mem.get_attributes("cup")
                if a["key"] == "把手"]
        assert len(rows) == 1, rows
        assert rows[0]["status"] == "hypothesis"
        assert rows[0]["value"] is True
        pend = mem.get_pending_candidate(p["id"])
        assert pend["status"] == "accepted"
    finally:
        mem.close()


def test_3_retrieval_after_save(tmp_path):
    data = fresh_data(tmp_path, tag="lc3")
    p = cli(data, "learn", "杯子可以盛水")
    cli(data, "confirm", str(p["id"]))
    u = cli(data, "understand", "杯子可以盛水")
    # seed verified cup.function=drinking vs user-taught 盛水: a live
    # dispute, so conflicted -- both sides kept, nothing overwritten.
    assert u["verdict"] == "conflicted", u
    assert u["evidence"], u
    # seed already verified cup.function=drinking, so the rival value
    # must be kept as contested, never overwrite verified (AGENTS §3)
    from cognitive_ai.memory import Memory
    mem = Memory(str(data / "cognitive.db"))
    try:
        assert any("cup.function" in cl.get("text", "") and
                   cl.get("status") == "contested"
                   for cl in mem.list_claims()), mem.list_claims()
        rows = [a for a in mem.get_attributes("cup")
                if a["key"] == "function" and a["value"] == "drinking"]
        assert any(r["status"] == "verified" for r in rows), rows
    finally:
        mem.close()


def test_4_restart_retrieves(tmp_path):
    data = fresh_data(tmp_path, tag="lc4")
    p = cli(data, "learn", "杯子有把手")
    cli(data, "confirm", str(p["id"]))
    u = cli(data, "understand", "杯子有把手")
    assert u["verdict"] == "supported", u


def test_5_cancel_writes_nothing_accepted(tmp_path):
    data = seed_db(fresh_data(tmp_path, tag="lc5"))
    before = db_counts(data)[:4]
    p = cli(data, "learn", "杯子有把手")
    out = cli(data, "reject", str(p["id"]))
    assert out["ok"] is True
    assert db_counts(data)[:4] == before
    from cognitive_ai.memory import Memory
    mem = Memory(str(data / "cognitive.db"))
    try:
        assert mem.get_pending_candidate(p["id"])["status"] == "rejected"
    finally:
        mem.close()


def test_6_ambiguous_refused(tmp_path):
    data = fresh_data(tmp_path, tag="lc6")
    from cognitive_ai.memory import Memory
    from cognitive_ai.ontology import Ontology
    mem = Memory(str(data / "cognitive.db"))
    try:
        from cognitive_ai.cli import seed_from_json
        seed_from_json(mem, str(data / "concepts.json"))
        Ontology(mem).add_concept("cup2", "杯子", parents=["entity"])
    finally:
        mem.close()
    p = cli(data, "learn", "杯子是容器")
    assert p["can_confirm"] is False, p
    out = cli(data, "confirm", str(p["id"]))
    assert out["ok"] is False, out


def test_7_conflict_kept_both_sides(tmp_path):
    data = fresh_data(tmp_path, tag="lc7")
    from cognitive_ai.memory import Memory
    from cognitive_ai.ontology import Ontology
    mem = Memory(str(data / "cognitive.db"))
    try:
        from cognitive_ai.cli import seed_from_json
        seed_from_json(mem, str(data / "concepts.json"))
        Ontology(mem).add_attribute("cup", "把手", True, status="verified")
    finally:
        mem.close()
    p = cli(data, "learn", "杯子没有把手")
    assert p["kind"] == "denial", p
    cli(data, "confirm", str(p["id"]))
    mem = Memory(str(data / "cognitive.db"))
    try:
        rows = [a for a in mem.get_attributes("cup") if a["key"] == "把手"]
        assert any(r["status"] == "verified" and r["value"] is True
                   for r in rows), rows
        assert any("没有" in cl.get("text", "") or "杯子" in cl.get("text", "")
                   for cl in mem.list_claims()), mem.list_claims()
        u = cli(data, "understand", "杯子没有把手")
        assert u["verdict"] == "conflicted", u
    finally:
        mem.close()


def test_8_repeat_is_deduped(tmp_path):
    data = fresh_data(tmp_path, tag="lc8")
    p1 = cli(data, "learn", "杯子有把手")
    cli(data, "confirm", str(p1["id"]))
    n1 = db_counts(data)
    p2 = cli(data, "learn", "杯子有把手")
    out = cli(data, "confirm", str(p2["id"]))
    assert out["ok"] is True
    n2 = db_counts(data)
    assert n2[0] == n1[0] and n2[1] == n1[1], (n1, n2)


def test_9_negation_polarity_kept(tmp_path):
    data = fresh_data(tmp_path, tag="lc9")
    p = cli(data, "learn", "杯子不能盛水")
    assert p["parse"]["negated"] is True
    cli(data, "confirm", str(p["id"]))
    from cognitive_ai.memory import Memory
    mem = Memory(str(data / "cognitive.db"))
    try:
        assert not [a for a in mem.get_attributes("cup")
                    if a["key"] == "function" and a["value"] == "盛水"], \
            mem.get_attributes("cup")
        assert any("不能" in cl.get("text", "") or
                   "否定" in cl.get("text", "") or
                   "denial" in cl.get("text", "").lower() or
                   "杯子不能盛水" in cl.get("text", "")
                   for cl in mem.list_claims()), mem.list_claims()
    finally:
        mem.close()


def test_10_kinds_match_storage(tmp_path):
    data = fresh_data(tmp_path, tag="lc10")
    p = cli(data, "learn", "杯子可以盛水")
    assert p["kind"] == "attribute", p
    cli(data, "confirm", str(p["id"]))
    from cognitive_ai.memory import Memory
    mem = Memory(str(data / "cognitive.db"))
    try:
        rows = mem.get_attributes("cup")
        assert any(r["key"] == "function" and r["value"] == "盛水"
                   for r in rows), rows
    finally:
        mem.close()


def test_11_unknown_concept_created_on_confirm(tmp_path):
    data = fresh_data(tmp_path, tag="lc11")
    p = cli(data, "learn", "雾铃是容器")
    assert p["kind"] == "new_concept", p
    from cognitive_ai.memory import Memory
    mem = Memory(str(data / "cognitive.db"))
    try:
        assert mem.get_concept("雾铃") is None
    finally:
        mem.close()
    cli(data, "confirm", str(p["id"]))
    mem = Memory(str(data / "cognitive.db"))
    try:
        assert mem.get_concept("雾铃") is not None
        u = cli(data, "understand", "雾铃是容器")
        assert u["verdict"] == "supported", u
    finally:
        mem.close()


def test_12_no_pending_without_confirm(tmp_path):
    data = fresh_data(tmp_path, tag="lc12")
    from cognitive_ai.memory import Memory
    mem = Memory(str(data / "cognitive.db"))
    try:
        assert mem.list_pending_candidates() == []
    finally:
        mem.close()
    p = cli(data, "learn", "杯子有把手")
    assert p["status"] == "proposed"
    mem = Memory(str(data / "cognitive.db"))
    try:
        assert mem.get_concept("cup") is not None
        assert [a for a in mem.get_attributes("cup")
                if a["key"] == "把手"] == []
    finally:
        mem.close()
