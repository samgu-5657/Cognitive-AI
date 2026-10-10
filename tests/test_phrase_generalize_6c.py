"""Phase 6C generalization experiment (tasks 3+4): strict train/heldout
split, programmatic battery, before/after comparison with restart.

Construction under test: imperative USE-questions ("告诉我X的用途")
and POLITE material questions ("请问X的材质是什么"), neither parseable
by the fixed parser. Train entities and held-out/test entities are
disjoint; test sentences are built combinatorially and audited against
the parser source (no per-sentence hardcodes possible).

Honesty notes: small battery (12 probes); success here shows ONE
learned construction generalizes across entities, not general NLU.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai import phraselearn as PL
from cognitive_ai.cli import seed_from_json
from cognitive_ai.interact import Session
from cognitive_ai.memory import Memory
from cognitive_ai.semantics import parse_sentence

SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))

TRAIN_ENTS = ["碗", "瓶子"]
HELD_ENTS = ["杯子", "水壶"]
TEST_ENTS = ["茶壶", "茶杯"]
GARBAGE = ["花钱如流水般哗啦", "噼里啪啦丁冬"]


def use_form(ent):
    return f"告诉我{ent}的用途"


def use_canon(ent):
    return f"{ent}有什么用途"


def mat_form(ent):
    return f"请告诉我{ent}的材质"


def mat_canon(ent):
    return f"{ent}的材质是什么"


BATTERY = [use_form(e) for e in TEST_ENTS] + \
    [mat_form(e) for e in TEST_ENTS]


def canon_of(t):
    if t.startswith("告诉我"):
        return use_canon(t[3:-3])
    return mat_canon(t[4:-3])


def make_session(tmp_path, tag="g6c"):
    db = str(tmp_path / f"{tag}.db")
    mem = Memory(db)
    seed_from_json(mem, SEED)
    corp = str(tmp_path / f"corp_{tag}")
    os.makedirs(corp, exist_ok=True)
    return Session(mem, corp), mem


def sig_of(text, rules=None, lex=None):
    fr = parse_sentence(text, source="user", lexicon=lex or {},
                        phrase_rules=rules)
    return PL._sig_key(PL.frame_signature(fr.to_dict())), fr


def strict_of(text, rules=None):
    """Strict frame key INCLUDING entity texts.

    The coarse signature deliberately excludes entities (that is what
    lets one rule cover many objects). For the experiment we additionally
    require the right entity: baseline 请问-forms parse but mis-segment
    ("问茶壶的材质"), so only the learned rewrite reaches strict match.
    """
    fr = parse_sentence(text, source="user", lexicon={}, phrase_rules=rules)
    d = PL.frame_signature(fr.to_dict())
    d["entities"] = sorted(e.get("text", "") for e in fr.entities)
    return PL._sig_key(d), fr


def test_ab_experiment_with_restart(tmp_path, capsys):
    sess, mem = make_session(tmp_path)
    try:
        # ---- baseline: fixed parser only ----
        # NOTE: 请问-forms *parse* at baseline but mis-segment (entity
        # "问茶壶的材质"): the honest baseline metric is signature match
        # against the canonical form, which is 0/4.
        base_sig = 0
        for t in BATTERY:
            got, _ = strict_of(t)
            want, _ = strict_of(canon_of(t))
            if got == want:
                base_sig += 1
        assert base_sig == 0, "battery must not match canon at baseline"
        # entity extraction on directly-parseable controls still works
        fr = parse_sentence("杯子有什么用途", source="user")
        assert fr.entities and fr.entities[0]["text"] == "杯子"

        # ---- teach: two constructions, disjoint train entities ----
        for e in TRAIN_ENTS:
            sess.handle(f"说法 {use_form(e)} => {use_canon(e)}")
            sess.handle(f"说法 {mat_form(e)} => {mat_canon(e)}")
        for e in HELD_ENTS:
            sess.handle(f"说法 {use_form(e)} => {use_canon(e)} --留存")
            sess.handle(f"说法 {mat_form(e)} => {mat_canon(e)} --留存")
        out = sess.handle("说法归纳")
        assert "2 候选" in "\n".join(out), out
        for r in mem.list_phrase_rules():
            assert r["kind"] == "slot", r
            v = sess.handle(f"说法验证 {r['id']}")
            assert any("验证通过" in l for l in v), v
            a = sess.handle(f"说法激活 {r['id']}")
            assert any("已激活" in l for l in a), a

        # ---- test on unseen entities (never taught, never held-out) ----
        rules = mem.list_phrase_rules(status="active")
        sig_hits, ent_hits = 0, 0
        for t in BATTERY:
            got, fr = strict_of(t, rules)
            want, _ = strict_of(canon_of(t))
            if got == want:
                sig_hits += 1
            subj = next((p.get("subj") for p in fr.predicates
                         if p.get("subj")), "")
            if subj and subj in t:
                ent_hits += 1
        # ---- abstention preserved ----
        abst = sum(1 for g in GARBAGE
                   if parse_sentence(g, source="user",
                                     phrase_rules=rules).intent == "unknown")
        with capsys.disabled():
            print(f"\n[6C A/B] battery={len(BATTERY)} baseline_sig={base_sig}"
                  f" learned_sig={sig_hits} entity={ent_hits} "
                  f"abstain={abst}/{len(GARBAGE)}")
        assert sig_hits == len(BATTERY), "every unseen probe must match " \
            "its canonical frame signature"
        assert ent_hits == len(BATTERY)
        assert abst == len(GARBAGE)
    finally:
        mem.close()
    # ---- restart: rules persist and still apply ----
    mem2 = Memory(str(tmp_path / "g6c.db"))
    try:
        rules = mem2.list_phrase_rules(status="active")
        assert len(rules) == 2
        for t in BATTERY:
            fr = parse_sentence(t, source="user", phrase_rules=rules)
            assert fr.intent == "query", (t, fr.intent)
            got, _ = strict_of(t, rules)
            want, _ = strict_of(canon_of(t))
            assert got == want, t
    finally:
        mem2.close()


def test_learned_answer_matches_canonical_answer(tmp_path):
    """Same frame + same KB -> same answer (cup has verified function)."""
    sess, mem = make_session(tmp_path)
    try:
        from cognitive_ai import dialog as D
        assert D.process(sess, use_form("杯子"))["route"] == "unknown"
        for e in TRAIN_ENTS:
            sess.handle(f"说法 {use_form(e)} => {use_canon(e)}")
        for e in HELD_ENTS:
            sess.handle(f"说法 {use_form(e)} => {use_canon(e)} --留存")
        sess.handle("说法归纳")
        for r in mem.list_phrase_rules():
            sess.handle(f"说法验证 {r['id']}")
            sess.handle(f"说法激活 {r['id']}")
        a = D.process(sess, use_form("杯子"))["answer"]["text"]
        b = D.process(sess, use_canon("杯子"))["answer"]["text"]
        assert "drinking" in a and "drinking" in b, (a, b)
    finally:
        mem.close()


def test_wrong_lexicon_mapping_never_becomes_verified(tmp_path):
    sess, mem = make_session(tmp_path)
    try:
        sess.handle("词义 陶瓷 玻璃")
        from cognitive_ai import dialog as D
        turn = D.process(sess, "杯子的材质是什么")
        assert "玻璃" not in turn["answer"]["text"]
        rows = mem.query("SELECT * FROM attributes WHERE concept_id='cup'"
                         " AND status='verified'")
        assert all(r["value"] != "玻璃" for r in rows)
    finally:
        mem.close()


def test_battery_not_hardcoded_in_src():
    blobs = []
    for f in ("semantics.py", "phraselearn.py", "dialog.py",
              "interact.py"):
        blobs.append(open(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "src", "cognitive_ai", f)),
            encoding="utf-8").read())
    leaked = [s for s in BATTERY + GARBAGE
              if len(s) >= 6 and any(s in b for b in blobs)]
    assert not leaked, f"battery leaked into source: {leaked}"
