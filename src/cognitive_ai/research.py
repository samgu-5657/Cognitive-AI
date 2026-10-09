"""Research providers. Phase 1: local corpus only. No fabricated results."""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Protocol

from .policy import STOPWORDS


def _key_tokens(query: str) -> list[str]:
    """Topical tokens: length > 1, lowercased, stopwords removed, order kept."""
    out: list[str] = []
    for t in re.findall(r"\w+", query, flags=re.UNICODE):
        t = t.lower()
        if len(t) > 1 and t not in STOPWORDS and t not in out:
            out.append(t)
    return out


def _is_ascii_token(term: str) -> bool:
    try:
        term.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _count_term(low_text: str, term: str) -> int:
    if _is_ascii_token(term):
        return len(re.findall(r"\b" + re.escape(term) + r"\b", low_text))
    return low_text.count(term)


def _mentions(text: str, terms: tuple[str, ...]) -> bool:
    low = text.lower()
    for term in terms:
        if not term:
            continue
        t = term.lower()
        if _is_ascii_token(t):
            if re.search(r"\b" + re.escape(t) + r"\b", low):
                return True
        elif t in low:
            return True
    return False


def _window(low_text: str, raw_text: str, term: str,
            before: int = 120, after: int = 200) -> str:
    t = term.lower()
    if _is_ascii_token(t):
        m = re.search(r"\b" + re.escape(t) + r"\b", low_text)
        i = m.start() if m else -1
    else:
        i = low_text.find(t)
    if i < 0:
        return ""
    return raw_text[max(0, i - before):i + after].strip()


class ResearchProvider(Protocol):
    name: str

    def search(self, query: str, limit: int = 5) -> list[dict]:
        ...


class LocalCorpusProvider:
    """Keyword search over local text/markdown files in data/corpus/."""

    name = "local_corpus"

    def __init__(self, corpus_dir: str):
        self.corpus_dir = corpus_dir

    def _files(self) -> list[str]:
        if not os.path.isdir(self.corpus_dir):
            return []
        out = []
        for root, _, files in os.walk(self.corpus_dir):
            for f in files:
                if f.lower().endswith((".txt", ".md", ".markdown")):
                    out.append(os.path.join(root, f))
        return sorted(out)

    def search(self, query: str, limit: int = 5,
                 focus_terms: tuple[str, ...] = ()) -> list[dict]:
        """Keyword search with stopword removal and word-boundary matching.

        Only documents matching at least one topical (non-stopword) token
        are returned. Excerpts are built around focus_terms first (usually
        the target concept id/name) so callers can verify that the evidence
        actually discusses the target. Each hit reports `matched_terms`.
        """
        tokens = _key_tokens(query)
        focus = [f for f in focus_terms if f]
        if not tokens and not focus:
            return []
        results: list[dict] = []
        for fp in self._files():
            try:
                with open(fp, encoding="utf-8") as fh:
                    text = fh.read()
            except Exception:
                continue
            low = text.lower()
            counts = {t: _count_term(low, t) for t in tokens}
            matched = sorted([t for t, n in counts.items() if n > 0])
            focus_hit = any(_mentions(low, (f,)) for f in focus)
            if not matched and not (focus and focus_hit):
                continue
            hits = sum(counts.values())
            # excerpts: focus windows first, then topical token windows
            excerpts = []
            for f in focus:
                w = _window(low, text, f)
                if w:
                    excerpts.append(w)
                    if len(excerpts) >= 2:
                        break
            if len(excerpts) < 2:
                for t in tokens:
                    if t in matched:
                        w = _window(low, text, t)
                        if w and w not in excerpts:
                            excerpts.append(w)
                    if len(excerpts) >= 2:
                        break
            mtime = datetime.fromtimestamp(
                os.path.getmtime(fp), tz=timezone.utc).isoformat()
            # credibility: local curated corpus, modest default; longer docs slightly higher
            credibility = min(0.85, 0.55 + min(len(text) / 20000, 0.2)
                              + min(hits / 50, 0.1))
            results.append({"source": f"local:{os.path.basename(fp)}",
                            "path": fp, "hits": hits,
                            "matched_terms": matched,
                            "excerpt": "\n---\n".join(excerpts)[:1200],
                            "credibility": round(credibility, 2),
                            "retrieved_at": mtime})
        results.sort(key=lambda r: (-r["hits"], r["source"]))
        return results[:limit]


class WebSearchProvider:
    """Interface-only placeholder for a future public web-search provider.

    Not implemented in Phase 1 (would need network + policy review).
    Calling search() raises to avoid fabricating results.
    """

    name = "web_search"

    def search(self, query: str, limit: int = 5) -> list[dict]:
        raise NotImplementedError(
            "WebSearchProvider is interface-only in Phase 1; "
            "use LocalCorpusProvider.")
