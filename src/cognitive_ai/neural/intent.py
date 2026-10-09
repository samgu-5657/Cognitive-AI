"""Tiny intent classifier (Phase 4B, Task 4). Random init, no pretraining.

Responsibility split, enforced by design:
- rule parser (semantics.py): full frames (entities/predicates/modality).
- this net: INTENT label only. It never produces slots, entities, or facts.
Comparison harness reports rule-vs-net per-intent accuracy on the frozen
semantics test set.
"""
from __future__ import annotations

import json
import os
import random

INTENTS = ["classify", "attribute", "function", "negate", "hedge",
           "correct", "causal", "observe", "query", "unknown"]


def _torch():
    from . import require_torch
    return require_torch()


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def check_group_isolation(*splits) -> dict:
    seen: dict[str, str] = {}
    dupes = []
    for name, rows in splits:
        for r in rows:
            g = r["group"]
            if g in seen and seen[g] != name:
                dupes.append((g, seen[g], name))
            seen.setdefault(g, name)
    texts: dict[str, str] = {}
    tdupes = []
    for name, rows in splits:
        for r in rows:
            t = r["text"].strip()
            if t in texts and texts[t] != name:
                tdupes.append((t, texts[t], name))
            texts.setdefault(t, name)
    return {"group_dupes": dupes, "text_dupes": tdupes,
            "ok": not dupes and not tdupes}


def train_intent(train_rows, val_rows, out_dir, seed=21, epochs=60,
                 batch=16, lr=5e-3, emb_dim=16) -> dict:
    torch = _torch()
    from .vocabulary import CharVocab
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    vocab = CharVocab(max_len=24).build([r["text"] for r in train_rows])
    vocab.save(os.path.join(out_dir, "vocab.json"))
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(vocab), emb_dim, padding_idx=0)
            self.fc = nn.Linear(emb_dim, len(INTENTS))

        def forward(self, x):
            return self.fc(self.emb(x).mean(dim=1))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    def enc(rows):
        x = torch.tensor([vocab.encode(r["text"]) for r in rows])
        y = torch.tensor([INTENTS.index(r["intent"]) for r in rows])
        return x, y

    xtr, ytr = enc(train_rows)
    xva, yva = enc(val_rows)
    n = len(xtr)
    best = {"acc": -1.0, "epoch": -1}
    ckpt = os.path.join(out_dir, "best.pt")
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n).tolist()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = ce(net(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(xva).argmax(1) == yva).float().mean().item()
        if acc > best["acc"]:
            best = {"acc": round(acc, 4), "epoch": ep}
            torch.save({"seed": seed, "intents": INTENTS,
                        "state": net.state_dict(),
                        "emb_dim": emb_dim}, ckpt)
    meta = {"seed": seed, "epochs": epochs, "best": best,
            "train_n": n, "val_n": len(xva),
            "init": "random (no pretrained weights)"}
    with open(os.path.join(out_dir, "metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    meta.update({"out_dir": out_dir, "checkpoint": ckpt,
                 "vocab": os.path.join(out_dir, "vocab.json")})
    return meta


def load_intent(checkpoint: str, vocab_path: str):
    torch = _torch()
    from .vocabulary import CharVocab
    import torch.nn as nn
    vocab = CharVocab.load(vocab_path)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(vocab), ck.get("emb_dim", 16),
                                    padding_idx=0)
            self.fc = nn.Linear(ck.get("emb_dim", 16), len(INTENTS))

        def forward(self, x):
            return self.fc(self.emb(x).mean(dim=1))

    net = Net()
    net.load_state_dict(ck["state"])
    net.eval()
    return net, vocab


def predict_intent(net, vocab, texts: list[str]) -> list[dict]:
    torch = _torch()
    import torch.nn.functional as F
    x = torch.tensor([vocab.encode(t) for t in texts])
    net.eval()
    with torch.no_grad():
        probs = F.softmax(net(x), dim=1).tolist()
    return [{"intent": INTENTS[max(range(len(p)), key=lambda i: p[i])],
             "confidence": round(max(p), 4)} for p in probs]


def compare_rule_vs_net(test_rows, model_dir: str) -> dict:
    """Rule parser vs intent net on identical frozen rows. Reports both."""
    try:
        from cognitive_ai.semantics import parse_sentence
    except ImportError:  # source-tree use without install
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from semantics import parse_sentence
    net, vocab = load_intent(os.path.join(model_dir, "best.pt"),
                             os.path.join(model_dir, "vocab.json"))
    per = {}
    for r in test_rows:
        want = r["intent"]
        got_rule = parse_sentence(r["text"]).intent
        got_net = predict_intent(net, vocab, [r["text"]])[0]["intent"]
        d = per.setdefault(want, {"n": 0, "rule": 0, "net": 0,
                                  "rule_miss": [], "net_miss": []})
        d["n"] += 1
        for name, got in (("rule", got_rule), ("net", got_net)):
            if got == want:
                d[name] += 1
            else:
                d[name + "_miss"].append((r["text"], got))
    summary = {k: {"n": v["n"],
                   "rule_acc": round(v["rule"] / v["n"], 3),
                   "net_acc": round(v["net"] / v["n"], 3)}
               for k, v in per.items()}
    tot = sum(v["n"] for v in per.values())
    summary["_overall"] = {
        "rule_acc": round(sum(v["rule"] for v in per.values()) / tot, 3),
        "net_acc": round(sum(v["net"] for v in per.values()) / tot, 3),
        "n": tot}
    return {"per_intent": summary, "detail": per}
