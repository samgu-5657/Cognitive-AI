"""Char-level models, random init, no pretrained weights.

CharCNN: embedding -> parallel 1-D convolutions (k=2,3,4) -> max-over-time
         -> concat -> dropout -> two heads (level1, level2).
BiGRU:   embedding -> bidirectional GRU -> concat(mean-pool, max-pool)
         -> dropout -> two heads.

Complexity comparison (this data regime: ~300 short samples, ~400 chars):
- CharCNN has fewer sequential dependencies, trains faster on CPU and is
  less prone to overfitting tiny data; chosen as the DEFAULT.
- BiGRU models order explicitly but costs more per step and overfits
  faster here; kept for the honest A/B comparison the task requires.
"""
from __future__ import annotations

from . import require_torch


def _heads(torch, hidden: int, n_l1: int, n_l2: int):
    import torch.nn as nn
    return nn.Linear(hidden, n_l1), nn.Linear(hidden, n_l2)


class CharCNN:
    name = "char-cnn"

    def __init__(self, vocab_size: int, n_l1: int, n_l2: int,
                 emb_dim: int = 32, filters: int = 32,
                 kernels: tuple = (2, 3, 4), dropout: float = 0.3):
        torch = require_torch()
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
                self.convs = nn.ModuleList(
                    [nn.Conv1d(emb_dim, filters, k) for k in kernels])
                self.drop = nn.Dropout(dropout)
                self.fc1, self.fc2 = _heads(torch, filters * len(kernels),
                                            n_l1, n_l2)

            def features(self, x):
                h = self.emb(x).transpose(1, 2)  # (B, E, L)
                feats = [torch.relu(c(h)).max(dim=2).values for c in self.convs]
                return torch.cat(feats, dim=1)

            def forward(self, x):
                h = self.drop(self.features(x))
                return self.fc1(h), self.fc2(h)

        self.net = Net()

    def parameters(self):
        return self.net.parameters()

    def __call__(self, *a, **k):
        return self.net(*a, **k)

    def train(self, mode=True):
        return self.net.train(mode)

    def eval(self):
        return self.net.eval()

    def features(self, x):
        torch = require_torch()
        was_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            f = self.net.features(x)
        if was_training:
            self.net.train()
        return f

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, d):
        return self.net.load_state_dict(d)


class CharBiGRU:
    name = "char-bigru"

    def __init__(self, vocab_size: int, n_l1: int, n_l2: int,
                 emb_dim: int = 32, hidden: int = 64, dropout: float = 0.3):
        torch = require_torch()
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
                self.gru = nn.GRU(emb_dim, hidden, batch_first=True,
                                  bidirectional=True)
                self.drop = nn.Dropout(dropout)
                self.fc1, self.fc2 = _heads(torch, hidden * 4, n_l1, n_l2)

            def features(self, x):
                h = self.emb(x)
                h, _ = self.gru(h)  # (B, L, 2H)
                return torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)

            def forward(self, x):
                pooled = self.drop(self.features(x))
                return self.fc1(pooled), self.fc2(pooled)

        self.net = Net()

    def parameters(self):
        return self.net.parameters()

    def __call__(self, *a, **k):
        return self.net(*a, **k)

    def train(self, mode=True):
        return self.net.train(mode)

    def eval(self):
        return self.net.eval()

    def features(self, x):
        torch = require_torch()
        was_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            f = self.net.features(x)
        if was_training:
            self.net.train()
        return f

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, d):
        return self.net.load_state_dict(d)


def build_model(kind: str, vocab_size: int, n_l1: int, n_l2: int, **kw):
    if kind == "cnn":
        return CharCNN(vocab_size, n_l1, n_l2, **kw)
    if kind == "gru":
        return CharBiGRU(vocab_size, n_l1, n_l2,
                         emb_dim=kw.get("emb_dim", 32),
                         hidden=kw.get("filters", 64),
                         dropout=kw.get("dropout", 0.3))
    raise ValueError(f"unknown model kind: {kind}")
