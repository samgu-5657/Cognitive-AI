"""Centralized research/exploration policy constants.

Single place for all evidence thresholds, cooldowns and caps so tests and
configuration can adjust them without touching logic.
"""
from __future__ import annotations

# A knowledge-base write requires at least this many independent,
# target-relevant sources that agree on the same value.
MIN_CORROBORATING_SOURCES = 2

# Minimum per-hit credibility for a hit to count as supporting evidence.
MIN_EVIDENCE_CREDIBILITY = 0.6

# After an inconclusive attempt, the question cools down this long (seconds)
# before it becomes eligible for research again.
INCONCLUSIVE_COOLDOWN_S = 300.0

# After this many inconclusive attempts a question is left alone
# (kept as inconclusive, never force-studied).
MAX_RESEARCH_ATTEMPTS = 3

# Total independent direct sources needed to promote hypothesis -> supported.
SUPPORTED_MIN_SOURCES = 3

# Excerpt-level Jaccard similarity at/above which two hits are treated as
# copies of one origin (dependent), not independent corroboration.
# Coarse heuristic (see README limits): catches obvious copies and light
# rewordings, not paraphrases.
NEAR_DUPLICATE_SIMILARITY = 0.85

# classify(): best score at/above this is a match (or ambiguous when
# conflicts exist); below it the observation is reported as novel.
CLASSIFY_MATCH_THRESHOLD = 0.3

# Analogy acceptance bar: the MAJORITY of typed structure must align
# (goal kind + type + ordered op chain + slot keys). Tunable central
# policy, exercised on both sides by tests (accept above, refuse below).
ANALOGY_STRUCT_MIN = 0.5

# Upper bound on genuinely NEW questions persisted per generate() call.
MAX_NEW_QUESTIONS_PER_STEP = 10

# Generic English tokens that carry no topical signal. They are excluded
# from evidence ranking so that every document does not match every query.
STOPWORDS = frozenset({
    "what", "whats", "which", "who", "whom", "whose", "when", "where",
    "why", "how", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done", "the", "a", "an", "of", "for", "to",
    "in", "on", "at", "by", "with", "from", "as", "or", "and", "it",
    "its", "this", "that", "these", "those", "concept", "concepts",
    "does", "participate", "distinguish", "belong", "correct", "resolve",
    "should", "there", "their", "than", "then", "so", "such",
})
