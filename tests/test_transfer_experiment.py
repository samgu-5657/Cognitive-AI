"""Transfer experiment (Phase 2, Task 5): does structural learning transfer?

Three systems face identical train cases and held-out test cases:
  A. baseline: seed knowledge only, no learning at all.
  B. facts only: research learns attributes; structural changes blocked
     (SchemaEvolver allow_structural=False, consolidate disabled).
  C. full: facts + validated structural changes (consolidate enabled).

Held-out tests (never in the train corpus):
  t1 familiar cup      -> expect match/cup
  t2 decorative cup    -> expect ambiguous + function conflict flagged
  t3 novel solar item  -> expect novel
  t4 lidded glass item -> expect best/bottle (+subgroup bonus only C can earn)
  t5 familiar bowl     -> expect match/bowl

Invariants asserted for every system (no superiority assumed):
  - seed verified rows unchanged (no verified pollution);
  - old ancestors (cup->container/drinkware) intact;
  - B never applies split/merge/reparent changes;
  - every C structural change has status applied (i.e. passed regression).
Scores are REPORTED (printed table), not asserted: if C shows no gain,
that is an honest finding, not a failure.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.cli import seed_from_json
from cognitive_ai.curiosity import Curiosity
from cognitive_ai.learner import Learner
from cognitive_ai.loop import ExplorationLoop
from cognitive_ai.memory import Memory
from cognitive_ai.ontology import Ontology

EVAL_DATA = os.path.join(os.path.dirname(__file__), "eval_data")
SEED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "concepts.json"))
TRAIN_FILES = ("eval_cup_a.txt", "eval_cup_b.txt",
               "eval_bowl_a.txt", "eval_bowl_b.txt",
               "eval_bottle_a.txt", "eval_bottle_b.txt")

TESTS = [
    {"id": "t1", "obs": {"name": "cup",
                         "attributes": {"material": "ceramic",
                                        "function": "drinking"}},
     "want_verdict": "match", "want_best": "cup"},
    {"id": "t2", "obs": {"name": "deco",
                         "attributes": {"material": "ceramic",
                                        "has_handle": True,
                                        "function": "decoration"}},
     "want_verdict": "ambiguous", "want_best": "cup",
     "want_conflict": "function"},
    {"id": "t3", "obs": {"name": "solar_cooker",
                         "attributes": {"power_source": "solar",
                                        "function": "heating"}},
     "want_verdict": "novel", "want_best": None},
    {"id": "t4", "obs": {"name": "new_jar",
                         "attributes": {"material": "glass",
                                        "has_lid": True}},
     "want_verdict": "match", "want_best": "bottle",
     "subgroup_bonus": True},
    {"id": "t5", "obs": {"name": "bowl",
                         "attributes": {"material": "ceramic",
                                        "has_handle": False}},
     "want_verdict": "match", "want_best": "bowl"},
]


def build_system(tmp_path, name, structural):
    d = tmp_path / f"xfer_{name}"
    d.mkdir(exist_ok=True)
    mem = Memory(str(d / "db.sqlite"))
    seed_from_json(mem, SEED)
    corp = str(d / "corpus")
    os.makedirs(corp, exist_ok=True)
    for f in TRAIN_FILES:
        shutil.copy(os.path.join(EVAL_DATA, f), os.path.join(corp, f))
    return mem, corp


TRAIN_QUESTIONS = (("cup", "location"), ("bowl", "has_handle"),
                   ("bowl", "location"), ("bottle", "material"),
                   ("bottle", "location"))


def train(name, mem, corp):
    """Identical train cases for B and C; A sees nothing.

    Targeted research of the same five gaps (deterministic order) so the
    comparison measures structural learning, not loop scheduling luck.
    """
    if name == "A":
        return {"trained": False}
    learner = Learner(mem, corp,
                      enable_structural=(name == "C"))
    studied = []
    for target, key in TRAIN_QUESTIONS:
        Curiosity(mem).generate()
        q = mem.query_one(
            "SELECT * FROM questions WHERE target_id=? AND text LIKE ?",
            (target, f"%'{key}'%"))
        if q is not None:
            studied.append(learner.research_question(int(q["id"])))
    con = learner.consolidate()
    return {"trained": True, "studied": studied, "consolidate": con}


def score_system(mem, corp):
    learner = Learner(mem, corp)
    onto = Ontology(mem)
    rows = []
    total = 0
    for t in TESTS:
        out = learner.classify(t["obs"])
        ok = (out["verdict"] == t["want_verdict"]
              and (t["want_best"] is None
                   or (out["best"] or {}).get("concept") == t["want_best"]))
        if ok and t.get("want_conflict"):
            ok = any(x["key"] == t["want_conflict"]
                     for x in (out["best"] or {}).get("conflicts", []))
        bonus = False
        if t.get("subgroup_bonus") and out["best"]:
            parents = mem.get_parents(out["best"]["concept"])
            bonus = any(p.startswith("vessel_div") for p in parents)
        total += ok
        rows.append({"id": t["id"], "verdict": out["verdict"],
                     "best": (out["best"] or {}).get("concept"),
                     "score": out["best"]["score"] if out["best"] else None,
                     "ok": ok, "subgroup_bonus": bonus})
    return {"total": total, "of": len(TESTS), "rows": rows,
            "bonus": sum(1 for r in rows if r["subgroup_bonus"])}


def check_invariants(name, mem):
    o = Ontology(mem)
    assert o.is_a("cup", "container") and o.is_a("cup", "drinkware")
    cup_mats = [(a["value"], a["status"]) for a in mem.get_attributes("cup")
                if a["key"] == "material"]
    assert ("ceramic", "verified") in cup_mats, cup_mats
    kinds = {c["kind"] for c in mem.list_changes(limit=10000)}
    if name == "B":
        assert not (kinds & {"split_category", "merge_categories",
                             "reparent"}), f"B applied structural changes: {kinds}"
    if name == "C":
        for c in mem.list_changes(limit=10000):
            if c["kind"] in ("split_category", "merge_categories",
                             "reparent"):
                assert c["status"] == "applied", c
    return kinds


def test_transfer_abc(tmp_path):
    report = {}
    for name, structural in (("A", False), ("B", False), ("C", True)):
        mem, corp = build_system(tmp_path, name, structural)
        try:
            train(name, mem, corp)
            kinds = check_invariants(name, mem)
            s = score_system(mem, corp)
            report[name] = {**s, "change_kinds": sorted(kinds),
                            "version": mem.current_version()}
        finally:
            mem.close()
    print("\nTRANSFER A/B/C (held-out tests, max 5 + subgroup bonus):")
    for name in ("A", "B", "C"):
        r = report[name]
        print(f"  system {name}: score={r['total']}/{r['of']} "
              f"bonus={r['bonus']} version=v{r['version']} "
              f"changes={r['change_kinds']}")
        for row in r["rows"]:
            print(f"    {row['id']}: verdict={row['verdict']} "
                  f"best={row['best']} ok={row['ok']} "
                  f"bonus={row['subgroup_bonus']}")
    # invariants only; superiority is reported, never asserted
    assert set(report) == {"A", "B", "C"}
