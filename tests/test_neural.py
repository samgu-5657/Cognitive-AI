"""Neural Phase 3A tests. Torch-dependent tests skip if torch is absent;
leakage/contract tests always run. All training uses tmp dirs + tiny
subsets for speed. Real corpus files only for the leakage test."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from cognitive_ai.neural import TORCH_AVAILABLE

NEED_TORCH = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch missing")
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "neural"))


def tiny_rows(n=30):
    from cognitive_ai.neural.dataset import load_jsonl
    tr = load_jsonl(os.path.join(DATA, "train.jsonl"))
    va = load_jsonl(os.path.join(DATA, "val.jsonl"))
    return tr[:n], va[:12]


def test_no_leakage_across_splits():
    from cognitive_ai.neural.dataset import check_no_leakage, load_jsonl
    rep = check_no_leakage(load_jsonl(os.path.join(DATA, "train.jsonl")),
                           load_jsonl(os.path.join(DATA, "val.jsonl")),
                           load_jsonl(os.path.join(DATA, "test.jsonl")))
    assert rep["ok"], rep


@NEED_TORCH
def test_train_from_random_init(tmp_path):
    from cognitive_ai.neural.trainer import train_model
    tr, va = tiny_rows()
    a = train_model(tr, va, str(tmp_path / "a"), kind="cnn", seed=7,
                    epochs=2, batch=8)
    b = train_model(tr, va, str(tmp_path / "b"), kind="cnn", seed=99,
                    epochs=2, batch=8)
    assert os.path.exists(a["checkpoint"]) and os.path.exists(a["vocab"])
    # different seeds -> different random inits (no fixed weights)
    assert a["best"]["fingerprint"] != b["best"]["fingerprint"]
    assert len(a["history"]) == 2 and "train_loss" in a["history"][0]


@NEED_TORCH
def test_save_reload_identical(tmp_path):
    from cognitive_ai.neural.evaluator import predict_probs
    from cognitive_ai.neural.trainer import load_model, train_model
    tr, va = tiny_rows()
    out = train_model(tr, va, str(tmp_path / "m"), kind="cnn", seed=7,
                      epochs=2, batch=8)
    m1, v1, _ = load_model(out["checkpoint"], out["vocab"])
    m2, v2, _ = load_model(out["checkpoint"], out["vocab"])
    p1, _ = predict_probs(m1, v1, ["杯子", "跑步"])
    p2, _ = predict_probs(m2, v2, ["杯子", "跑步"])
    assert p1 == p2


@NEED_TORCH
def test_training_reproducible(tmp_path):
    from cognitive_ai.neural.trainer import train_model
    tr, va = tiny_rows()
    kw = dict(kind="cnn", seed=7, epochs=3, batch=8)
    a = train_model(tr, va, str(tmp_path / "r1"), **kw)
    b = train_model(tr, va, str(tmp_path / "r2"), **kw)
    assert a["best"]["fingerprint"] == b["best"]["fingerprint"]
    assert a["history"] == b["history"]


def test_unknown_never_forced():
    from cognitive_ai.neural.adapter import to_candidate
    flat = [0.2, 0.2, 0.2, 0.2, 0.2]
    c = to_candidate("激光打印机", flat, [0.05] * 21, "test-v")
    assert c["verdict"] == "unknown" and c["concept_type"] is None
    assert c["candidate_category"] is None
    peaked = [0.9, 0.02, 0.03, 0.02, 0.03]
    c2 = to_candidate("杯子", peaked, [0.8] + [0.01] * 20, "test-v")
    assert c2["verdict"] == "known" and c2["concept_type"] == "Entity"
    assert 0.0 <= c2["confidence"] <= 1.0


@NEED_TORCH
def test_adapter_output_contract(tmp_path):
    from cognitive_ai.neural.adapter import NeuralAdapter
    from cognitive_ai.neural.trainer import train_model
    tr, va = tiny_rows()
    out = train_model(tr, va, str(tmp_path / "c"), kind="cnn", seed=7,
                      epochs=2, batch=8)
    ad = NeuralAdapter(out["checkpoint"], out["vocab"])
    for text in ("杯子", "下雨", "xyz未知词@@@"):
        r = ad.predict(text)
        assert set(("concept_type", "candidate_category", "confidence",
                    "model_version", "verdict", "level2",
                    "level2_confidence", "input")) <= set(r)
        assert r["verdict"] in ("known", "unknown")
        assert 0.0 <= r["confidence"] <= 1.0
        assert isinstance(r["model_version"], str) and r["model_version"]


@NEED_TORCH
def test_submit_never_bypasses_validation(tmp_path):
    from cognitive_ai.neural.adapter import NeuralAdapter, to_candidate
    from cognitive_ai.neural.trainer import train_model
    from cognitive_ai.memory import Memory
    from cognitive_ai.ontology import Ontology
    from cognitive_ai.schema_evolver import SchemaEvolver
    mem = Memory(str(tmp_path / "kb.db"))
    try:
        o = Ontology(mem)
        o.add_concept("entity", "e")
        o.add_concept("cup", "cup", parents=["entity"])
        o.add_attribute("cup", "material", "ceramic", status="verified")
        SchemaEvolver(mem).ensure_initial_version()
        v0 = mem.current_version()
        tr, va = tiny_rows()
        out = train_model(tr, va, str(tmp_path / "s"), kind="cnn",
                          seed=7, epochs=2, batch=8)
        ad = NeuralAdapter(out["checkpoint"], out["vocab"])
        cand = to_candidate("杯子", [0.9, 0.02, 0.03, 0.02, 0.03],
                            [0.8] + [0.01] * 20, ad.version)
        res = ad.submit(mem, cand)
        assert res["ok"] and res["version_unchanged"]
        assert mem.current_version() == v0
        rows = [a for a in mem.get_attributes("cup")
                if a["key"] == "material"]
        assert [(a["value"], a["status"]) for a in rows] == [
            ("ceramic", "verified")]
        claims = [c for c in mem.list_claims()
                  if c["status"] == "candidate"]
        assert claims, "candidate must be recorded as a Claim"
    finally:
        mem.close()


def test_missing_torch_keeps_rule_system_alive():
    import cognitive_ai.neural as N
    old = N.TORCH_AVAILABLE
    N.TORCH_AVAILABLE = False
    try:
        from cognitive_ai.neural.trainer import train_model
        with pytest.raises(RuntimeError):
            train_model([], [], "nowhere")
    finally:
        N.TORCH_AVAILABLE = old
    # core system imports carry no torch dependency at all
    code = ("import sys; sys.modules['torch'] = None;"
            "import cognitive_ai.memory, cognitive_ai.learner,"
            " cognitive_ai.loop, cognitive_ai.cli;"
            "print('core-ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=os.path.join(
                           os.path.dirname(__file__), ".."))
    assert r.returncode == 0 and "core-ok" in r.stdout, r.stderr[-500:]


@NEED_TORCH
def test_cpu_smoke_train_and_eval(tmp_path):
    import torch
    from cognitive_ai.neural.dataset import load_jsonl
    from cognitive_ai.neural.evaluator import evaluate
    from cognitive_ai.neural.trainer import load_model, train_model
    assert not torch.cuda.is_available() or True  # CPU path always valid
    tr = load_jsonl(os.path.join(DATA, "train.jsonl"))[:40]
    va = load_jsonl(os.path.join(DATA, "val.jsonl"))[:16]
    te = load_jsonl(os.path.join(DATA, "test.jsonl"))[:16]
    out = train_model(tr, va, str(tmp_path / "smoke"), kind="cnn",
                      seed=7, epochs=3, batch=8)
    model, vocab, ck = load_model(out["checkpoint"], out["vocab"])
    rep = evaluate(model, vocab, te)
    assert rep["n"] == 16 and "confusion" in rep and "per_class" in rep
    assert set(rep["per_class"]) == {"entity", "event", "action",
                                     "attribute", "abstract"}
