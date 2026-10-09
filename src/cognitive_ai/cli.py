"""Thin CLI wrapper. All logic lives in library modules."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cognitive_ai.curiosity import Curiosity, SCORE_WEIGHTS
from cognitive_ai.learner import Learner
from cognitive_ai.loop import ExplorationLoop
from cognitive_ai.memory import Memory
from cognitive_ai.novelty import NoveltyDetector, parse_chinese_simple
from cognitive_ai.ontology import Ontology
from cognitive_ai.reasoner import Reasoner
from cognitive_ai.schema_evolver import SchemaEvolver
from cognitive_ai.learner import default_regression

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
DB = os.path.join(DATA, "cognitive.db")
CORPUS = os.path.join(DATA, "corpus")
SEED = os.path.join(DATA, "concepts.json")


def get_memory() -> Memory:
    mem = Memory(DB)
    if mem.current_version() == 0 and not mem.list_concepts() \
            and os.path.exists(SEED):
        seed_from_json(mem, SEED)
    return mem


def seed_from_json(mem: Memory, path: str) -> None:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    onto = Ontology(mem)
    for c in data.get("concepts", []):
        onto.add_concept(c["id"], c.get("name", c["id"]),
                         c.get("kind", "Entity"), [],
                         c.get("confidence", 1.0), c.get("source", "seed"))
    for c in data.get("concepts", []):
        for p in c.get("parents", []):
            try:
                onto.add_parent(c["id"], p)
            except Exception as e:
                print(f"seed parent skip {c['id']}->{p}: {e}")
    for a in data.get("attributes", []):
        onto.add_attribute(a["concept"], a["key"], a["value"],
                           a.get("confidence", 0.9),
                           a.get("source", "seed"),
                           a.get("status", "verified"))
    for r in data.get("relations", []):
        onto.add_relation(r["src"], r["rel_type"], r["dst"],
                          r.get("confidence", 0.9),
                          r.get("source", "seed"),
                          r.get("status", "verified"))
    for rule in data.get("rules", []):
        mem.upsert_rule(rule)
    SchemaEvolver(mem).ensure_initial_version()


def cmd_concepts(_a):
    mem = get_memory()
    for c in Ontology(mem).list():
        print(f"{c['id']}  [{c['kind']}]  parents={c['parents']}  name={c['name']}")
    mem.close()


def cmd_inspect(a):
    mem = get_memory()
    try:
        d = Ontology(mem).describe(a.id)
    except Exception as e:
        print(f"error: {e}")
        return
    print(json.dumps(d, ensure_ascii=False, indent=2, default=str))
    mem.close()


def _load_json_arg(a) -> dict:
    if getattr(a, "json_file", None):
        with open(a.json_file, encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(a.json)


def cmd_observe(a):
    mem = get_memory()
    obs = _load_json_arg(a)
    out = Learner(mem, CORPUS).learn_from_observation(obs)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    mem.close()


def cmd_parse(a):
    print(json.dumps(parse_chinese_simple(a.sentence), ensure_ascii=False, indent=2))


def cmd_query(a):
    mem = get_memory()
    r = Reasoner(mem).answer(_load_json_arg(a))
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    mem.close()


def cmd_gaps(_a):
    mem = get_memory()
    gaps = NoveltyDetector(mem).knowledge_gaps()
    print(json.dumps(gaps, ensure_ascii=False, indent=2, default=str))
    mem.close()


def cmd_questions(_a):
    mem = get_memory()
    qs = Curiosity(mem).generate()
    new_n = sum(1 for q in qs if q.get("is_new"))
    for q in qs:
        print(f"[{q['id']}] (score={q['score']}) {q['text']}  kind={q['kind']} "
              f"target={q.get('target_id')} status=queued new={q.get('is_new')}")
    print(f"-- open={len(qs)} genuinely_new={new_n} --")
    mem.close()


def cmd_explore(a):
    mem = get_memory()
    out = ExplorationLoop(mem, CORPUS).run(steps=a.steps,
                                           max_seconds=a.max_seconds)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    mem.close()


def cmd_experiences(a):
    mem = get_memory()
    for e in mem.list_experiences(limit=a.limit):
        print(f"#{e['id']} [{e['kind']}] {e['summary']}  ({e['created_at']})")
    mem.close()


def cmd_history(_a):
    mem = get_memory()
    ev = SchemaEvolver(mem)
    print("== versions ==")
    for v in ev.versions():
        print(f"  v{v['version']}: {v['reason']} ({v['created_at']})")
    print("== changes ==")
    for c in ev.history():
        print(f"  #{c['id']} {c['kind']} status={c['status']} "
              f"v{c['version_before']}->v{c['version_after']} :: {c['reason']}")
    mem.close()


def cmd_rollback(a):
    mem = get_memory()
    print(json.dumps(SchemaEvolver(mem, default_regression).rollback(a.version),
                     ensure_ascii=False, indent=2))
    mem.close()


def cmd_doctor(_a):
    mem = get_memory()
    onto = Ontology(mem)
    problems = []
    for c in mem.list_concepts():
        if c["id"] in onto.ancestors(c["id"]):
            problems.append(f"cycle at {c['id']}")
    cup = mem.get_concept("cup")
    if cup is None:
        problems.append("core concept 'cup' missing")
    elif "container" not in onto.ancestors("cup"):
        problems.append("cup lost ancestor 'container'")
    ok, msg = default_regression(mem)
    print(json.dumps({
        "concepts": len(mem.list_concepts()),
        "rules": len(mem.list_rules()),
        "questions": len(mem.list_questions()),
        "experiences": len(mem.list_experiences(limit=100000)),
        "current_version": mem.current_version(),
        "score_weights": SCORE_WEIGHTS,
        "regression": {"ok": ok, "msg": msg},
        "problems": problems,
    }, ensure_ascii=False, indent=2))
    mem.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cognitive_ai",
                                description="Cognitive AI CLI (Phase 1)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("concepts").set_defaults(fn=cmd_concepts)
    s = sub.add_parser("inspect"); s.add_argument("id"); s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("observe"); s.add_argument("--json", default=None); s.add_argument("--json-file", default=None); s.set_defaults(fn=cmd_observe)
    s = sub.add_parser("parse"); s.add_argument("sentence"); s.set_defaults(fn=cmd_parse)
    s = sub.add_parser("query"); s.add_argument("--json", default=None); s.add_argument("--json-file", default=None); s.set_defaults(fn=cmd_query)
    sub.add_parser("gaps").set_defaults(fn=cmd_gaps)
    sub.add_parser("questions").set_defaults(fn=cmd_questions)
    s = sub.add_parser("explore"); s.add_argument("--steps", type=int, default=5)
    s.add_argument("--max-seconds", type=float, default=120.0); s.set_defaults(fn=cmd_explore)
    s = sub.add_parser("experiences"); s.add_argument("--limit", type=int, default=20)
    s.set_defaults(fn=cmd_experiences)
    sub.add_parser("history").set_defaults(fn=cmd_history)
    s = sub.add_parser("rollback"); s.add_argument("version", type=int); s.set_defaults(fn=cmd_rollback)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
