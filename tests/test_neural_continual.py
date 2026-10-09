"""Phase 3B hook tests: unknown log, sample store, promotion, replay gate."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.neural.continual import (SampleStore, UnknownLog,
                                           promote_samples, replay_compare)


def test_unknown_log_roundtrip(tmp_path):
    log = UnknownLog(str(tmp_path / "unk.jsonl"))
    log.record("激光打印机", "cnn-seed7@x", 0.31)
    rows = log.load()
    assert len(rows) == 1 and rows[0]["text"] == "激光打印机"


def test_sample_store_validates_and_promotes(tmp_path):
    import pytest
    st = SampleStore(str(tmp_path / "reviewed.jsonl"))
    with pytest.raises(ValueError):
        st.add({"text": "杯子"})
    st.add({"id": "x1", "group": "g1", "text": "杯子", "level1": "entity",
            "level2": "container"})
    train = str(tmp_path / "train.jsonl")
    open(train, "w").close()
    out = promote_samples(str(tmp_path / "reviewed.jsonl"), train)
    assert out["ok"] and out["promoted"] == 1
    # original file untouched, new versioned file created
    assert open(train, encoding="utf-8").read() == ""
    assert out["file"].endswith(".v1.jsonl")
    # nothing left to promote
    st2 = SampleStore(str(tmp_path / "empty.jsonl"))
    assert promote_samples(str(tmp_path / "empty.jsonl"),
                           train)["ok"] is False


def test_replay_gate_blocks_regression():
    old = {"accuracy_l1": 0.8}
    same = {"accuracy_l1": 0.8, "accuracy_new": 0.9}
    worse = {"accuracy_l1": 0.7, "accuracy_new": 0.9}
    assert replay_compare(old, same)["accept"] is True
    assert replay_compare(old, worse)["accept"] is False
    assert replay_compare(old, worse, tolerance=0.1)["accept"] is True
