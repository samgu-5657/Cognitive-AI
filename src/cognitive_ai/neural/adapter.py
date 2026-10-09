"""Adapter: network output -> structured CANDIDATE for the cognitive system.

Contract (stable, torch needed only to run the model, not to use types):
  predict() returns {"concept_type", "candidate_category", "confidence",
                     "model_version", "verdict", "level2", "level2_confidence"}.
  verdict is 'known' (maxprob >= threshold) or 'unknown' (abstain).
  submit() records the candidate as a lifecycle `candidate` Claim with the
  model output attached as evidence. It NEVER writes attributes/relations,
  NEVER touches verified rows, NEVER bumps the schema version. Promotion
  goes through the existing evidence + lifecycle gates only.
"""
from __future__ import annotations

from . import LEVEL1, LEVEL2

UNKNOWN_THRESHOLD = 0.6

L1_TO_KIND = {"entity": "Entity", "event": "Event", "action": "Action",
              "attribute": "Attribute", "abstract": "AbstractConcept"}


def to_candidate(text: str, l1_probs: list[float], l2_probs: list[float],
                 model_version: str,
                 threshold: float = UNKNOWN_THRESHOLD) -> dict:
    best1 = int(max(range(len(l1_probs)), key=lambda i: l1_probs[i]))
    best2 = int(max(range(len(l2_probs)), key=lambda i: l2_probs[i]))
    conf = float(l1_probs[best1])
    if conf < threshold:
        return {"concept_type": None, "candidate_category": None,
                "confidence": round(conf, 4), "model_version": model_version,
                "verdict": "unknown", "level2": None,
                "level2_confidence": 0.0, "input": text}
    return {"concept_type": L1_TO_KIND[LEVEL1[best1]],
            "candidate_category": LEVEL1[best1],
            "confidence": round(conf, 4), "model_version": model_version,
            "verdict": "known", "level2": LEVEL2[best2],
            "level2_confidence": round(float(l2_probs[best2]), 4),
            "input": text}


class NeuralAdapter:
    """Binds a trained checkpoint to the candidate contract."""

    def __init__(self, checkpoint: str, vocab_path: str,
                 threshold: float = UNKNOWN_THRESHOLD):
        import os
        from .trainer import load_model
        self._checkpoint = checkpoint
        self.model, self.vocab, ck = load_model(checkpoint, vocab_path)
        self.threshold = threshold
        tag = os.path.basename(os.path.dirname(os.path.abspath(checkpoint)))
        self._version = (f"{ck.get('kind', '?')}-seed{ck.get('seed', '?')}@{tag}")

    @property
    def version(self) -> str:
        return self._version

    def predict(self, text: str) -> dict:
        from .evaluator import predict_probs
        p1, p2 = predict_probs(self.model, self.vocab, [text])
        return to_candidate(text, p1[0], p2[0], self.version, self.threshold)

    def submit(self, memory, candidate: dict) -> dict:
        """Gated submission: candidate Claim + evidence only. Read the
        contract above: no attribute/relation writes, no version bumps."""
        v_before = memory.current_version()
        claim_id = memory.add_claim(
            f"neural candidate: '{candidate.get('input')}' -> "
            f"{candidate.get('candidate_category') or 'unknown'} "
            f"(conf={candidate.get('confidence')})",
            subject_id=None, status="candidate",
            confidence=float(candidate.get("confidence") or 0.0))
        memory.add_evidence(
            claim_id, str(candidate), candidate.get("model_version", "?"),
            credibility=float(candidate.get("confidence") or 0.0),
            relevance="direct",
            origin_note="neural-candidate; needs independent corroboration")
        memory.record_lifecycle("claim", str(claim_id), "none", "candidate",
                                "neural adapter submission", [candidate])
        memory.add_experience("observation",
                              "neural candidate submitted",
                              {"candidate": candidate, "claim_id": claim_id})
        return {"ok": True, "claim_id": claim_id,
                "version_unchanged": memory.current_version() == v_before}
