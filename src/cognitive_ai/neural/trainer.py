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
                        "hparams": hparams, "state": model.state_dict()}, ckpt)
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
    model = build_model(ck["kind"], len(vocab), len(LEVEL1), len(LEVEL2),
                        emb_dim=hp.get("emb_dim", 32),
                        filters=hp.get("filters", 32),
                        dropout=hp.get("dropout", 0.3))
    model.load_state_dict(ck["state"])
    model.eval()
    return model, vocab, ck
