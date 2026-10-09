"""Seeded CPU training with best-checkpoint saving and metric logging."""
from __future__ import annotations

import hashlib
import json
import os
import random

from . import LEVEL1, LEVEL2, require_torch
from .dataset import labeled
from .model import build_model
from .vocabulary import CharVocab


def set_seed(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.set_num_threads(1)


def encode_split(vocab: CharVocab, rows: list[dict]):
    torch = require_torch()
    rows = labeled(rows)
    x = torch.tensor([vocab.encode(r["text"]) for r in rows], dtype=torch.long)
    y1 = torch.tensor([LEVEL1.index(r["level1"]) for r in rows],
                      dtype=torch.long)
    y2 = torch.tensor([LEVEL2.index(r.get("level2") or "none") for r in rows],
                      dtype=torch.long)
    return x, y1, y2, rows


def weights_fingerprint(model) -> str:
    import hashlib
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(model.state_dict()[k].cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def train_model(train_rows: list[dict], val_rows: list[dict],
                out_dir: str, kind: str = "cnn", seed: int = 7,
                epochs: int = 40, batch: int = 32, lr: float = 1e-3,
                max_len: int = 24, dropout: float = 0.3,
                emb_dim: int = 32, filters: int = 32,
                l2_weight: float = 0.5, weight_decay: float = 0.0) -> dict:
    """Train from RANDOM init. Returns paths + history + best val scores."""
    torch = require_torch()
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    vocab = CharVocab(max_len=max_len).build([r["text"] for r in train_rows])
    vocab.save(os.path.join(out_dir, "vocab.json"))
    model = build_model(kind, len(vocab), len(LEVEL1), len(LEVEL2),
                        emb_dim=emb_dim, filters=filters, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    import torch.nn as nn
    ce = nn.CrossEntropyLoss()
    xtr, y1tr, y2tr, _ = encode_split(vocab, train_rows)
    xva, y1va, y2va, _ = encode_split(vocab, val_rows)
    n = len(xtr)
    history = []
    best = {"val_acc": -1.0, "epoch": -1, "fingerprint": ""}
    ckpt = os.path.join(out_dir, "best.pt")
    hparams = {"emb_dim": emb_dim, "filters": filters, "dropout": dropout}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n).tolist()
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            o1, o2 = model(xtr[idx])
            loss = ce(o1, y1tr[idx]) + l2_weight * ce(o2, y2tr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            o1, _ = model(xva)
            acc = (o1.argmax(1) == y1va).float().mean().item()
        history.append({"epoch": ep, "train_loss": round(tot / n, 4),
                        "val_acc": round(acc, 4)})
        if acc > best["val_acc"]:
            best = {"val_acc": round(acc, 4), "epoch": ep,
                    "fingerprint": weights_fingerprint(model)}
            torch.save({"kind": kind, "seed": seed, "max_len": max_len,
                        "hparams": hparams, "level1": LEVEL1,
                        "level2": LEVEL2, "state": model.state_dict()}, ckpt)
    meta = {"kind": kind, "seed": seed, "epochs": epochs, "lr": lr,
            "batch": batch, "dropout": dropout, "emb_dim": emb_dim,
            "filters": filters, "l2_weight": l2_weight,
            "weight_decay": weight_decay, "vocab_size": len(vocab),
            "train_n": n, "val_n": len(xva), "best": best,
            "history": history, "device": "cpu",
            "init": "random (no pretrained weights)"}
    with open(os.path.join(out_dir, "metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return {"out_dir": out_dir, "checkpoint": ckpt,
            "vocab": os.path.join(out_dir, "vocab.json"),
            "metrics": os.path.join(out_dir, "metrics.json"), **meta}


def load_model(checkpoint: str, vocab_path: str):
    torch = require_torch()
    vocab = CharVocab.load(vocab_path)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    hp = ck.get("hparams", {})
    if ck.get("level1", LEVEL1) != LEVEL1:
        raise ValueError("checkpoint L1 label space differs; "
                         "L1 extension is not supported in Phase 3B")
    ck_l2 = ck.get("level2", LEVEL2)
    model = build_model(ck["kind"], len(vocab), len(LEVEL1), len(ck_l2),
                        emb_dim=hp.get("emb_dim", 32),
                        filters=hp.get("filters", 32),
                        dropout=hp.get("dropout", 0.3))
    model.load_state_dict(ck["state"])
    model.eval()
    return model, vocab, ck


def extend_l2(model, base_map: list[str],
              new_labels: list[str]) -> tuple:
    """Extend the L2 head with new subcategories.

    `base_map` is the model's CURRENT full L2 list (identity for the first
    extension: LEVEL2). Old output rows are copied EXACTLY (existing label
    mapping preserved); new rows are randomly initialized.
    Returns (model, full_level2_list).
    """
    torch = require_torch()
    fresh = [l for l in new_labels if l not in base_map]
    full = list(base_map) + fresh
    if not fresh:
        return model, list(base_map)
    import torch.nn as nn
    old = model.net.fc2
    if old.out_features != len(base_map):
        raise ValueError("base_map does not match model head width; "
                         "pass the model's current label list")
    new = nn.Linear(old.in_features, len(full))
    with torch.no_grad():
        new.weight[:len(base_map)] = old.weight
        new.bias[:len(base_map)] = old.bias
    model.net.fc2 = new
    return model, full


def _stratified_replay(old_rows: list[dict], n: int, seed: int) -> list[dict]:
    import random
    rng = random.Random(seed)
    by_label: dict[str, list] = {}
    for r in old_rows:
        by_label.setdefault(r.get("level1", "?"), []).append(r)
    per = max(1, n // max(1, len(by_label)))
    out = []
    for rows in by_label.values():
        out.extend(rng.sample(rows, min(per, len(rows))))
    return out[:n] if len(out) > n else out


def incremental_train(base_checkpoint: str, base_vocab: str,
                      new_rows: list[dict], replay_rows: list[dict],
                      out_dir: str, epochs: int = 30, batch: int = 8,
                      lr: float = 5e-4, seed: int = 11,
                      replay_ratio: float = 2.0,
                      l2_labels: list[str] | None = None) -> dict:
    """Finetune from a base checkpoint on reviewed + replay data.

    - new_rows MUST carry explicit reviewed labels (l1/l2); rows without
      labels raise immediately (predictions are never labels);
    - L1 outside the closed set raises (explicit extension policy);
    - replay keeps old knowledge: replay_n = round(len(new) * replay_ratio);
    - saves before/after fingerprints, config and log; returns paths.
    """
    import os
    torch = require_torch()
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    for r in new_rows:
        if not r.get("level1"):
            raise ValueError(f"unlabeled sample cannot train: {r.get('id')}")
        if r["level1"] not in LEVEL1:
            raise ValueError(f"L1 label space is closed in Phase 3B; "
                             f"got {r['level1']!r}")
    vocab = CharVocab.load(base_vocab)
    ck = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    hp = ck.get("hparams", {})
    model = build_model(ck["kind"], len(vocab), len(LEVEL1), len(LEVEL2),
                        emb_dim=hp.get("emb_dim", 32),
                        filters=hp.get("filters", 32),
                        dropout=hp.get("dropout", 0.3))
    model.load_state_dict(ck["state"])
    before_fp = weights_fingerprint(model)
    l2_map = LEVEL2
    if l2_labels:
        unknown_l2 = [l for l in l2_labels if l not in LEVEL2]
        if unknown_l2:
            model, l2_map = extend_l2(model, LEVEL2, l2_labels)
    replay = _stratified_replay(
        [r for r in replay_rows if r.get("level1")],
        int(round(len(new_rows) * replay_ratio)), seed)
    combo = list(new_rows) + replay
    if not combo:
        raise ValueError("nothing to train on")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    import torch.nn as nn
    ce = nn.CrossEntropyLoss()
    x, y1, y2, _ = _encode_with_maps(vocab, combo, LEVEL1, l2_map)
    n = len(x)
    log = []
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n).tolist()
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            o1, o2 = model(x[idx])
            loss = ce(o1, y1[idx]) + 0.5 * ce(o2, y2[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        log.append({"epoch": ep, "train_loss": round(tot / n, 4)})
    after_fp = weights_fingerprint(model)
    ckpt = os.path.join(out_dir, "candidate.pt")
    import shutil
    shutil.copy(base_vocab, os.path.join(out_dir, "vocab.json"))
    torch.save({"kind": ck["kind"], "seed": seed,
                "max_len": vocab.max_len, "hparams": hp,
                "level1": LEVEL1, "level2": l2_map,
                "base": base_checkpoint, "state": model.state_dict()}, ckpt)
    meta = {"base": base_checkpoint, "seed": seed, "epochs": epochs,
            "batch": batch, "lr": lr, "replay_ratio": replay_ratio,
            "new_n": len(new_rows), "replay_n": len(replay),
            "l2_extended": l2_map != LEVEL2, "before": before_fp,
            "after": after_fp, "log": log, "device": "cpu"}
    with open(os.path.join(out_dir, "incremental.json"), "w",
              encoding="utf-8") as fh:
        import json
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    meta.update({"out_dir": out_dir, "checkpoint": ckpt})
    return meta


def _encode_with_maps(vocab, rows, l1_map, l2_map):
    torch = require_torch()
    x = torch.tensor([vocab.encode(r["text"]) for r in rows], dtype=torch.long)
    y1 = torch.tensor([l1_map.index(r["level1"]) for r in rows],
                      dtype=torch.long)
    y2 = torch.tensor([l2_map.index(r.get("level2") or "none") for r in rows],
                      dtype=torch.long)
    return x, y1, y2, rows
