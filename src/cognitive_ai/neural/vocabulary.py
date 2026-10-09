"""Character vocabulary built from training data only. No pretrained vectors."""
from __future__ import annotations

import json

PAD, UNK = "<pad>", "<unk>"


class CharVocab:
    def __init__(self, max_len: int = 24):
        self.max_len = max_len
        self.itos = [PAD, UNK]
        self.stoi = {PAD: 0, UNK: 1}

    def build(self, texts: list[str]) -> "CharVocab":
        for t in texts:
            for ch in t:
                if ch not in self.stoi:
                    self.stoi[ch] = len(self.itos)
                    self.itos.append(ch)
        return self

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi.get(ch, 1) for ch in text[:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"max_len": self.max_len, "itos": self.itos}, fh,
                      ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharVocab":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        v = cls(max_len=d["max_len"])
        v.itos = d["itos"]
        v.stoi = {s: i for i, s in enumerate(v.itos)}
        return v
