"""Phase 3C tests: OOD verdicts, discovery logic, structure wiring.

Deterministic logic tests use a stub model with canned features (discover
only calls model.features); behavior tests use real tiny models on tmp.
No test depends on committed weight files.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from cognitive_ai.neural import TORCH_AVAILABLE

NEED_TORCH = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch missing")
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "data", "neural"))
TH = {"margin_min": 0.17, "conf_min": 0.5, "dist_far": 3.82,
      "n_val_correct": 31}


class StubModel:
    """Canned feature rows, cycled to the batch size."""

    def __init__(self, rows):
        self._rows = rows
        self.training = False

    def features(self, x):
        import torch
        n = x.shape[0] if hasattr(x, "shape") else len(x)
        rep = (self._rows * ((n // len(self._rows)) + 1))[:n]
        return torch.tensor(rep, dtype=torch.float32)


def vec(dim, at, val=5.0):
    v = [0.1] * dim
    v[at] = val
    return v


def protos():
    return {
        "entity": {"mean": vec(4, 0), "spread": 1.0, "n": 9},
        "action": {"mean": vec(4, 1), "spread": 1.0, "n": 9},
        "abstract": {"mean": vec(4, 2), "spread": 1.0, "n": 9},
    }


def l2p():
    return {
        "tool": {"mean": vec(4, 0), "n": 3, "family": "entity", "spread": 0.8},
        "food": {"mean": [0.1, 0.1, 0.1, 5.0], "n": 3, "family": "entity",
                 "spread": 0.8},
    }


@NEED_TORCH
def test_zero_evidence_inputs_are_unjudgeable():
    from cognitive_ai.neural.ood import assess
    import torch

    class M:
        training = False

        def eval(self):
            return self

        def __call__(self, x):
            return (torch.zeros(len(x), 5), torch.zeros(len(x), 21))

        def features(self, x):
            return torch.zeros(len(x), 4)

    m = M()
    from cognitive_ai.neural.vocabulary import CharVocab
    v = CharVocab().build(["abc"])
    for t in ("xyz@@@", "@@@", ""):
        r = assess(t, m, v, protos(), 1.0, TH)
        assert r["verdict"] == "unjudgeable", (t, r)


@NEED_TORCH
def test_margin_and_distance_paths():
    from cognitive_ai.neural.ood import assess
    import torch

    class M:
        training = False

        def __init__(self, logits):
            self._lg = logits

        def eval(self):
            return self

        def __call__(self, x):
            n = len(x)
            return (torch.tensor(self._lg[:n]), torch.zeros(n, 21))

        def features(self, x):
            return torch.tensor([vec(4, 0)] * len(x))

    from cognitive_ai.neural.vocabulary import CharVocab
    v = CharVocab().build(["abcd"])
    flat = [[0.0] * 5]
    r = assess("abcd", M(flat), v, protos(), 1.0, TH)
    assert r["verdict"] in ("suspected_novel", "known_uncertain"), r
    peaked = [[10.0, 0.0, 0.0, 0.0, 0.0]]
    r2 = assess("abcd", M(peaked), v, protos(), 1.0, TH)
    assert r2["verdict"] == "known_confident", r2


@NEED_TORCH
def test_ensemble_veto_with_second_model():
    from cognitive_ai.neural.ood import assess
    import torch

    class M:
        training = False

        def __init__(self, top):
            self._top = top

        def eval(self):
            return self

        def __call__(self, x):
            n = len(x)
            lg = [[0.0] * 5 for _ in range(n)]
            for row in lg:
                row[self._top] = 10.0
            return (torch.tensor(lg), torch.zeros(n, 21))

        def features(self, x):
            return torch.tensor([vec(4, 0)] * len(x))

    from cognitive_ai.neural.vocabulary import CharVocab
    v = CharVocab().build(["abcd"])
    r = assess("abcd", M(0), v, protos(), 1.0, TH, second=(M(1), v))
    assert r["verdict"] == "similar_but_unproven", r
    assert any("ensemble" in s for s in r["reasons"])


@NEED_TORCH
def test_discovery_cohesive_proposes_insufficient_declines():
    from cognitive_ai.neural.discovery import discover, validate_proposal
    dim = 4
    # identical vectors: cohesion exactly 1.0, clear of every stub centroid
    tight = [[2.0] * dim for _ in range(3)]
    m = StubModel(tight)
    samps = [{"sample_id": f"k{i}", "text": f"w{i}", "l1": "entity",
              "l2": "none"} for i in range(3)]
    rep = discover(samps, m, None, protos(), l2p(), TH)
    assert rep["selected"]["kind"] == "new_l2", rep
    assert rep["selected"]["cohesion"] == pytest.approx(1.0)
    val = validate_proposal(samps, m, None, protos(), l2p())
    assert val["ok"] and val["rate"] == 1.0
    # dispersed vectors: same reviewer label, no shared direction
    disp = StubModel([vec(dim, 0), vec(dim, 1), vec(dim, 2)])
    rep2 = discover(samps, disp, None, protos(), l2p(), TH)
    assert rep2["selected"]["kind"] == "insufficient_evidence", rep2
    # too few samples: never proposes
    rep3 = discover(samps[:2], m, None, protos(), l2p(), TH)
    assert rep3["selected"]["kind"] == "insufficient_evidence"
    # explained group: near its own prototype -> existing_ok
    home = StubModel([vec(dim, 0) for _ in range(3)])
    rep4 = discover(samps, home, None, protos(), l2p(), TH)
    assert rep4["selected"]["kind"] in ("existing_ok",
                                        "insufficient_evidence"), rep4


@NEED_TORCH
def test_l1_new_is_candidate_only():
    from cognitive_ai.neural.discovery import discover, to_schema_payload
    dim = 4
    # cohesive cluster far from every L1 prototype
    far = [[9.0, 9.0, 9.0, 9.0] for _ in range(3)]
    m = StubModel(far)
    samps = [{"sample_id": f"f{i}", "text": f"w{i}", "l1": "entity",
              "l2": "none"} for i in range(3)]
    rep = discover(samps, m, None, protos(), {}, TH)
    kinds = {p["kind"] for p in rep["proposals"]}
    assert "new_l1_candidate" in kinds or "new_l2" in kinds, rep
    # no schema mapping exists for L1 candidates: must stay proposals
    with pytest.raises(ValueError):
        to_schema_payload({"kind": "new_l1_candidate"})
    # review path rejects unknown L1 labels outright
    from cognitive_ai.neural.queue import UnknownQueue
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        q = UnknownQueue(os.path.join(td, "q.jsonl"))
        r = q.add("???", "m", None, None, 0.1, "low_confidence")
        bad = q.review(r["sample_id"], accept=True, l1="alien")
        assert not bad["ok"]


@NEED_TORCH
def test_real_model_unknown_paths(tmp_path):
    """Real tiny model: flat/unknown inputs never verified as known."""
    from cognitive_ai.neural.dataset import load_jsonl
    from cognitive_ai.neural.ood import (assess, class_prototypes,
                                         tune_thresholds)
    from cognitive_ai.neural.trainer import train_model
    tr = load_jsonl(os.path.join(DATA, "train.jsonl"))[:30]
    va = [r for r in load_jsonl(os.path.join(DATA, "val.jsonl"))
          if r.get("level1")][:12]
    out = train_model(tr, va, str(tmp_path / "m"), kind="cnn", seed=7,
                      epochs=5, batch=8)
    from cognitive_ai.neural.trainer import load_model
    model, vocab, _ = load_model(out["checkpoint"], out["vocab"])
    pro = class_prototypes(model, vocab, tr)
    th = tune_thresholds(model, vocab, pro, va)
    for t in ("xyz@@@", "@@@"):
        r = assess(t, model, vocab, pro, 1.0, th)
        assert r["verdict"] in ("unjudgeable", "suspected_novel",
                                "known_uncertain",
                                "similar_but_unproven"), (t, r)


@NEED_TORCH
def test_extended_checkpoint_loads_and_predicts(tmp_path):
    from cognitive_ai.neural.adapter import NeuralAdapter
    from cognitive_ai.neural.dataset import load_jsonl
    from cognitive_ai.neural.discovery import apply_new_l2
    from cognitive_ai.neural.trainer import train_model
    tr = load_jsonl(os.path.join(DATA, "train.jsonl"))[:30]
    va = [r for r in load_jsonl(os.path.join(DATA, "val.jsonl"))
          if r.get("level1")][:12]
    out = train_model(tr, va, str(tmp_path / "m"), kind="cnn", seed=7,
                      epochs=5, batch=8)
    base = os.path.dirname(out["checkpoint"])
    rep = apply_new_l2(base, str(tmp_path / "ext"), "kitchenware_test")
    assert rep["ok"] and rep["new_label"] == "kitchenware_test"
    ad = NeuralAdapter(rep["checkpoint"],
                       os.path.join(str(tmp_path / "ext"), "vocab.json"))
    r = ad.predict("炒锅")
    assert r["model_version"] and r["verdict"] in ("known", "unknown")
    assert ad.l2_names[-1] == "kitchenware_test"
