"""Thin CLI wrapper. All logic lives in library modules."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cognitive_ai.curiosity import Curiosity, SCORE_WEIGHTS
from cognitive_ai.learner import Learner
from cognitive_ai.lifecycle import describe as lifecycle_describe
from cognitive_ai.lifecycle import promote_attribute
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


def cmd_consolidate(_a):
    mem = get_memory()
    out = Learner(mem, CORPUS).consolidate()
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    mem.close()


def cmd_lifecycle(a):
    mem = get_memory()
    if a.target:
        rows = mem.list_lifecycle(a.target)
    else:
        rows = mem.list_lifecycle(limit=a.limit)
    if a.describe:
        print(json.dumps(lifecycle_describe(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    mem.close()


def cmd_promote(a):
    mem = get_memory()
    ev = json.loads(a.evidence) if a.evidence else []
    print(json.dumps(promote_attribute(
        mem, a.attr_id, a.to, evidence=ev,
        confirmed_by=a.confirmed_by or "", reason=a.reason or ""),
        ensure_ascii=False, indent=2, default=str))
    mem.close()


def _neural_paths(a):
    root = os.path.join(ROOT, "models", getattr(a, "tag", "cnn-v1"))
    return root


def cmd_neural_train(a):
    from cognitive_ai.neural.dataset import check_no_leakage, load_jsonl
    from cognitive_ai.neural.trainer import train_model
    data = os.path.join(DATA, "neural")
    tr = load_jsonl(os.path.join(data, "train.jsonl"))
    va = load_jsonl(os.path.join(data, "val.jsonl"))
    te = load_jsonl(os.path.join(data, "test.jsonl"))
    leak = check_no_leakage(tr, va, te)
    print(json.dumps({"leakage_check": leak}, ensure_ascii=False, indent=2))
    if not leak["ok"]:
        raise SystemExit("data leakage detected; refusing to train")
    out = train_model(tr, va, _neural_paths(a), kind=a.model,
                      seed=a.seed, epochs=a.epochs, batch=a.batch,
                      dropout=a.dropout, l2_weight=a.l2_weight,
                      weight_decay=a.weight_decay, emb_dim=a.emb,
                      filters=a.filters)
    print(json.dumps({k: v for k, v in out.items() if k != "history"},
                     ensure_ascii=False, indent=2))
    mem = get_memory()
    mem.add_experience("evaluation",
                       f"neural train {a.model} seed={a.seed}: "
                       f"best val_acc={out['best']['val_acc']} "
                       f"@epoch {out['best']['epoch']}",
                       {"out_dir": out["out_dir"], "best": out["best"]})
    mem.close()


def cmd_neural_eval(a):
    from cognitive_ai.neural.adapter import NeuralAdapter
    from cognitive_ai.neural.dataset import load_jsonl
    from cognitive_ai.neural.evaluator import evaluate
    data = os.path.join(DATA, "neural")
    te = load_jsonl(os.path.join(data, "test.jsonl"))
    root = _neural_paths(a)
    ad = NeuralAdapter(os.path.join(root, "best.pt"),
                       os.path.join(root, "vocab.json"))
    rep = evaluate(ad.model, ad.vocab, te)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    mem = get_memory()
    mem.add_experience("evaluation",
                       f"neural eval {ad.version}: "
                       f"acc_l1={rep['accuracy_l1']} "
                       f"baseline={rep['baseline_accuracy_l1']} "
                       f"reject={rep['unknown_rejection_rate']}",
                       {"report": rep, "model_version": ad.version})
    mem.close()


def cmd_neural_predict(a):
    from cognitive_ai.neural.adapter import NeuralAdapter
    root = _neural_paths(a)
    ad = NeuralAdapter(os.path.join(root, "best.pt"),
                       os.path.join(root, "vocab.json"))
    out = ad.predict(a.text)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.submit:
        mem = get_memory()
        print(json.dumps(ad.submit(mem, out), ensure_ascii=False, indent=2))
        mem.close()


def cmd_neural_compare(a):
    from cognitive_ai.neural.adapter import NeuralAdapter
    from cognitive_ai.neural.dataset import load_jsonl
    from cognitive_ai.neural.evaluator import evaluate
    data = os.path.join(DATA, "neural")
    te = load_jsonl(os.path.join(data, "test.jsonl"))
    for tag in a.tags.split(","):
        root = os.path.join(ROOT, "models", tag.strip())
        try:
            ad = NeuralAdapter(os.path.join(root, "best.pt"),
                               os.path.join(root, "vocab.json"))
            rep = evaluate(ad.model, ad.vocab, te)
            print(f"[{tag}] acc_l1={rep['accuracy_l1']} "
                  f"acc_l2={rep['accuracy_l2']} "
                  f"baseline={rep['baseline_accuracy_l1']} "
                  f"reject={rep['unknown_rejection_rate']}")
        except Exception as e:
            print(f"[{tag}] unavailable: {e}")


def _queue_path():
    return os.path.join(DATA, "neural", "queue", "unknown.jsonl")


def cmd_neural_queue(a):
    from cognitive_ai.neural.queue import UnknownQueue
    q = UnknownQueue(_queue_path())
    rows = q.list(status=a.status, limit=a.limit)
    for r in rows:
        print(f"[{r['sample_id']}] {r['text']!r} pred={r['l1_pred']}/"
              f"{r['l2_pred']} conf={r['confidence']} reason={r['reason']} "
              f"review={r['review']['status']} dup_of={r['dup_of']} "
              f"label={r['label']}")
    print(f"-- shown={len(rows)} --")


def cmd_neural_queue_show(a):
    from cognitive_ai.neural.queue import UnknownQueue
    print(json.dumps(UnknownQueue(_queue_path()).get(a.sample_id),
                     ensure_ascii=False, indent=2, default=str))


def cmd_neural_review(a):
    from cognitive_ai.neural.queue import UnknownQueue
    q = UnknownQueue(_queue_path())
    out = q.review(a.sample_id, accept=a.accept, l1=a.l1, l2=a.l2,
                   reviewer=a.reviewer)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if out.get("ok"):
        mem = get_memory()
        mem.add_experience("evaluation",
                           f"reviewed {a.sample_id}: {out['status']}",
                           {"sample_id": a.sample_id, **out})
        mem.close()


def cmd_neural_queue_dedup(_a):
    from cognitive_ai.neural.queue import UnknownQueue
    print(json.dumps(UnknownQueue(_queue_path()).merge_duplicates(),
                     ensure_ascii=False, indent=2))


def cmd_neural_queue_export(a):
    import os
    from cognitive_ai.neural.queue import UnknownQueue
    q = UnknownQueue(_queue_path())
    rows = [{"id": r["sample_id"], "group": f"rev-{r['sample_id']}",
             "text": r["text"], "level1": r["label"]["l1"],
             "level2": r["label"]["l2"], "source": "reviewed"}
            for r in q.accepted()]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"ok": True, "exported": len(rows), "file": a.out}))


def cmd_neural_incremental(a):
    from cognitive_ai.neural.dataset import load_jsonl
    from cognitive_ai.neural.queue import UnknownQueue
    from cognitive_ai.neural.trainer import incremental_train
    q = UnknownQueue(a.queue if a.queue else _queue_path())
    accepted = q.accepted()
    new_rows = [{"id": r["sample_id"], "group": f"rev-{r['sample_id']}",
                 "text": r["text"], "level1": r["label"]["l1"],
                 "level2": r["label"]["l2"], "source": "reviewed"}
                for r in accepted]
    if not new_rows:
        print(json.dumps({"ok": False, "error": "no accepted samples"}))
        return
    base = os.path.join(ROOT, "models", a.base)
    replay = [r for r in load_jsonl(a.train) if r.get("level1")]
    out = incremental_train(os.path.join(base, "best.pt"),
                            os.path.join(base, "vocab.json"),
                            new_rows, replay,
                            os.path.join(ROOT, "models", a.out),
                            epochs=a.epochs, seed=a.seed,
                            replay_ratio=a.replay_ratio)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


def cmd_neural_gate(a):
    from cognitive_ai.neural.dataset import load_jsonl
    from cognitive_ai.neural.gate import (PromotionPolicy, decide,
                                          evaluate_pair, promote)
    reg = [r for r in load_jsonl(a.regression) if r.get("level1")]
    new = [r for r in load_jsonl(a.new) if r.get("level1")] if a.new else []
    root = os.path.join(ROOT, "models")
    pair = evaluate_pair(os.path.join(root, a.base),
                         os.path.join(root, a.candidate), reg, new)
    dec = decide(pair, PromotionPolicy(
        regression_tolerance=a.tolerance, min_new_samples=a.min_new))
    print(json.dumps({"decision": dec, "delta": pair["delta_regression"],
                      "base_acc": pair["base"]["regression"]["accuracy_l1"],
                      "cand_acc": pair["candidate"]["regression"]["accuracy_l1"],
                      "new_acc": pair["candidate"]["new_acc"]},
                     ensure_ascii=False, indent=2))
    if a.promote and dec["promote"]:
        print(json.dumps(promote(root, a.candidate, pair, dec,
                                 data_version=a.data_version),
                         ensure_ascii=False, indent=2))
    elif a.promote:
        print(json.dumps({"promoted": False, "reason": dec["reason"]}))


def cmd_neural_active(_a):
    from cognitive_ai.neural.gate import read_active
    print(json.dumps(read_active(os.path.join(ROOT, "models")),
                     ensure_ascii=False, indent=2))


def cmd_neural_rollback(_a):
    from cognitive_ai.neural.gate import rollback_active
    print(json.dumps(rollback_active(os.path.join(ROOT, "models")),
                     ensure_ascii=False, indent=2))


def cmd_neural_cycle(a):
    from cognitive_ai.neural.adapter import NeuralAdapter
    from cognitive_ai.neural.cycle import cycle_predict, maybe_train
    from cognitive_ai.neural.gate import read_active
    from cognitive_ai.neural.queue import UnknownQueue
    root = os.path.join(ROOT, "models")
    act = read_active(root) or {"tag": "cnn-v1"}
    tag = act["tag"]
    ad = NeuralAdapter(os.path.join(root, tag, "best.pt"),
                       os.path.join(root, tag, "vocab.json"))
    mem = get_memory()
    q = UnknownQueue(_queue_path())
    out = cycle_predict(ad, q, mem, a.text, source="cli-cycle")
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    if a.train:
        res = maybe_train(
            mem, tag, root, _queue_path(),
            os.path.join(DATA, "neural", "train.jsonl"),
            a.out_tag, batch_min=a.batch_min, epochs=a.epochs,
            max_train_seconds=a.budget)
        print(json.dumps({k: v for k, v in res.items()
                          if k not in ("pair",)}, ensure_ascii=False,
                         indent=2, default=str))
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
    sub.add_parser("consolidate").set_defaults(fn=cmd_consolidate)
    s = sub.add_parser("lifecycle"); s.add_argument("--target", default=None)
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--describe", action="store_true"); s.set_defaults(fn=cmd_lifecycle)
    s = sub.add_parser("promote"); s.add_argument("attr_id", type=int)
    s.add_argument("to"); s.add_argument("--evidence", default="[]")
    s.add_argument("--confirmed-by", default=""); s.add_argument("--reason", default="")
    s.set_defaults(fn=cmd_promote)
    s = sub.add_parser("neural-train"); s.add_argument("--model",
        choices=["cnn", "gru"], default="cnn"); s.add_argument("--tag", default="cnn-v1")
    s.add_argument("--seed", type=int, default=7); s.add_argument("--epochs", type=int, default=300)
    s.add_argument("--batch", type=int, default=8); s.add_argument("--dropout", type=float, default=0.5)
    s.add_argument("--l2-weight", type=float, default=0.2); s.add_argument("--weight-decay", type=float, default=1e-3)
    s.add_argument("--emb", type=int, default=16); s.add_argument("--filters", type=int, default=16)
    s.set_defaults(fn=cmd_neural_train)
    s = sub.add_parser("neural-eval"); s.add_argument("--tag", default="cnn-v1")
    s.set_defaults(fn=cmd_neural_eval)
    s = sub.add_parser("neural-predict"); s.add_argument("text")
    s.add_argument("--tag", default="cnn-v1"); s.add_argument("--submit", action="store_true")
    s.set_defaults(fn=cmd_neural_predict)
    s = sub.add_parser("neural-compare"); s.add_argument("--tags", default="cnn-v1,gru-v1")
    s.set_defaults(fn=cmd_neural_compare)
    s = sub.add_parser("neural-queue"); s.add_argument("--status", default=None)
    s.add_argument("--limit", type=int, default=50); s.set_defaults(fn=cmd_neural_queue)
    s = sub.add_parser("neural-queue-show"); s.add_argument("sample_id")
    s.set_defaults(fn=cmd_neural_queue_show)
    s = sub.add_parser("neural-review"); s.add_argument("sample_id")
    s.add_argument("--l1", default=None); s.add_argument("--l2", default=None)
    s.add_argument("--accept", action="store_true"); s.add_argument("--reviewer", default="operator")
    s.set_defaults(fn=cmd_neural_review)
    sub.add_parser("neural-queue-dedup").set_defaults(fn=cmd_neural_queue_dedup)
    s = sub.add_parser("neural-queue-export"); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_neural_queue_export)
    s = sub.add_parser("neural-incremental"); s.add_argument("--base", default="cnn-v1")
    s.add_argument("--out", required=True); s.add_argument("--queue", default=None)
    s.add_argument("--train", default=os.path.join(DATA, "neural", "train.jsonl"))
    s.add_argument("--epochs", type=int, default=30); s.add_argument("--seed", type=int, default=11)
    s.add_argument("--replay-ratio", type=float, default=2.0)
    s.set_defaults(fn=cmd_neural_incremental)
    s = sub.add_parser("neural-gate"); s.add_argument("--base", default="cnn-v1")
    s.add_argument("--candidate", required=True)
    s.add_argument("--regression", default=os.path.join(DATA, "neural", "val.jsonl"))
    s.add_argument("--new", default=None); s.add_argument("--tolerance", type=float, default=0.02)
    s.add_argument("--min-new", type=int, default=5); s.add_argument("--promote", action="store_true")
    s.add_argument("--data-version", default="reviewed+replay")
    s.set_defaults(fn=cmd_neural_gate)
    sub.add_parser("neural-active").set_defaults(fn=cmd_neural_active)
    sub.add_parser("neural-rollback").set_defaults(fn=cmd_neural_rollback)
    s = sub.add_parser("neural-cycle"); s.add_argument("text")
    s.add_argument("--train", action="store_true"); s.add_argument("--out-tag", default="cnn-cand")
    s.add_argument("--batch-min", type=int, default=5); s.add_argument("--epochs", type=int, default=30)
    s.add_argument("--budget", type=float, default=600.0)
    s.set_defaults(fn=cmd_neural_cycle)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
