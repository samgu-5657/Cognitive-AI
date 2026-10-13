"""Continual learning: observation -> research -> KB update, all recorded."""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from .curiosity import Curiosity
from .lifecycle import contest_attribute
from .memory import Memory
from .novelty import NoveltyDetector
from .ontology import Ontology
from .policy import (INCONCLUSIVE_COOLDOWN_S, MAX_RESEARCH_ATTEMPTS,
                     MIN_CORROBORATING_SOURCES, MIN_EVIDENCE_CREDIBILITY,
                     NEAR_DUPLICATE_SIMILARITY)
from .reasoner import Reasoner
from .research import (LocalCorpusProvider, _mentions,
                       grade_relevance, text_similarity)
from .schema_evolver import SchemaEvolver


def default_regression(memory: Memory) -> tuple[bool, str]:
    """Built-in regression gate: no orphan core concepts, no cycles, cup intact."""
    onto = Ontology(memory)
    try:
        cup = memory.get_concept("cup")
        if cup is None:
            return False, "core concept 'cup' missing"
        if "container" not in onto.ancestors("cup"):
            return False, "cup lost ancestor 'container'"
        # cycle sanity: no concept is its own ancestor
        for c in memory.list_concepts():
            if c["id"] in onto.ancestors(c["id"]):
                return False, f"cycle at {c['id']}"
    except Exception as e:
        return False, str(e)
    return True, "ok"


class Learner:
    """Type-A (episodic) + Type-B (structural) learning. No neural nets here."""

    def __init__(self, memory: Memory, corpus_dir: str,
                 enable_structural: bool = True):
        self.mem = memory
        self.onto = Ontology(memory)
        self.reasoner = Reasoner(memory)
        self.detector = NoveltyDetector(memory)
        self.curiosity = Curiosity(memory)
        self.provider = LocalCorpusProvider(corpus_dir)
        self.enable_structural = enable_structural
        self.evolver = SchemaEvolver(memory,
                                     regression_check=default_regression,
                                     allow_structural=enable_structural)

    # ---- one observation ----
    def observe(self, obs: dict) -> dict:
        """obs: {"name","parents","attributes","relations","source"}."""
        report = self.detector.check_observation(obs)
        exp_id = self.mem.add_experience(
            "observation", f"observed {obs.get('name')}",
            {"observation": obs, "novelty": dict(report)})
        return {"experience_id": exp_id, "novelty": dict(report)}

    def learn_from_observation(self, obs: dict) -> dict:
        """Full mini-cycle for one structured observation."""
        result: dict[str, Any] = {"observation": obs}
        seen = self.observe(obs)
        result["experience_id"] = seen["experience_id"]
        result["novelty"] = seen["novelty"]
        name = obs.get("name", "")
        matches = self.mem.find_concepts_by_name(name)
        if not matches:
            # create provisional sub-concept under first known parent (or entity)
            parents = [p for p in obs.get("parents", [])
                       if self.mem.get_concept(p) is not None] or ["entity"]
            cid = name.lower().strip().replace(" ", "_") or "unknown_item"
            if self.mem.get_concept(cid) is None:
                r = self.evolver.direct_apply(
                    "add_concept",
                    {"id": cid, "name": name, "kind": "Entity",
                     "parents": parents,
                     "attributes": obs.get("attributes", {}),
                     "confidence": 0.5, "source": "learning"},
                    reason=f"novel observation '{name}' with no matching concept",
                    evidence=[{"observation": obs}])
                result["new_concept"] = {"id": cid, **r}
                for rel in obs.get("relations", []):
                    try:
                        self.evolver.direct_apply(
                            "add_relation",
                            {"src": cid, "rel_type": rel.get("rel_type", "related_to"),
                             "dst": rel.get("dst"), "confidence": 0.5,
                             "source": "learning"}, reason="observed relation")
                    except Exception:
                        pass
            else:
                result["new_concept"] = {"id": cid, "note": "already exists"}
        else:
            cid = matches[0]["id"]
            # add genuinely new attribute keys as hypotheses (never overwrite verified)
            inherited = self.onto.inherited_attributes(cid)
            added = []
            for k, v in obs.get("attributes", {}).items():
                if k not in inherited:
                    r = self.evolver.direct_apply(
                        "add_attribute",
                        {"concept": cid, "key": k, "value": v,
                         "confidence": 0.55, "source": "learning",
                         "status": "hypothesis"},
                        reason=f"extend concept {cid} with observed attribute {k}",
                        evidence=[{"observation": obs}])
                    added.append({k: r})
                elif inherited[k]["value"] != v:
                    # conflict: record contested claim + evidence, do NOT overwrite.
                    # Keep the verified row untouched AND store the rival
                    # observation as a second `contested` attribute row so
                    # check_conflicts() can see both sides. An identical
                    # rival row is never stored twice, and no extra
                    # lifecycle event is emitted for it.
                    same = [a for a in self.mem.get_attributes(cid)
                            if a.get("key") == k and a.get("value") == v]
                    if same:
                        added.append({k: {
                            "status": same[0].get("status", "contested"),
                            "note": "rival already recorded"}})
                        continue
                    cl = self.mem.add_claim(
                        f"{cid}.{k}: {inherited[k]['value']} vs observed {v}",
                        subject_id=cid, status="contested", confidence=0.5)
                    self.mem.add_evidence(
                        cl, f"observed {k}={v} for {name}", source="observation",
                        credibility=0.5)
                    self.evolver.direct_apply(
                        "add_attribute",
                        {"concept": cid, "key": k, "value": v,
                         "confidence": 0.5, "source": "observation",
                         "status": "contested"},
                        reason=f"contested evidence for {cid}.{k}",
                        evidence=[{"observation": obs}])
                    for row in self.mem.get_attributes(cid):
                        if row["key"] == k and row["status"] in (
                                "candidate", "hypothesis", "supported",
                                "verified"):
                            contest_attribute(
                                self.mem, int(row["id"]),
                                [{"source": "nl-confirm",
                                  "excerpt": obs.get("utterance", name)}],
                                f"nl-confirm: rival '{v}' vs "
                                f"'{inherited[k]['value']}'; row untouched")
                    added.append({k: {"status": "contested", "claim_id": cl}})
            result["updated_concept"] = {"id": cid, "added": added}
        # evaluate: did conflicts grow?
        result["conflicts"] = self.reasoner.check_conflicts()
        # outcome bookkeeping: fact vs concept learned
        result["outcome"] = "concept" if result.get("new_concept") else "fact"
        self.mem.add_experience("evaluation",
                                f"learned from {name}",
                                {"result": _jsonable(result),
                                 "outcome": result["outcome"]})
        return result

    # ---- NL candidate learning (Phase M3C) ----
    def propose_from_utterance(self, text: str) -> dict:
        """Parse + ground one Chinese statement into a pending candidate.

        Writes exactly one pending_candidates row (status proposed) and
        nothing else. The candidate records kind/observation/reason;
        confirmation happens separately in confirm_candidate().
        """
        from .semground import ground_statement
        from .semparse import parse_statement
        u = parse_statement(text, self.mem)
        v = ground_statement(text, self.mem)
        cand = {"utterance": text, "parse": u.to_dict(),
                "grounding": v.to_dict(), "kind": "unusable",
                "observation": {}, "can_confirm": False,
                "reason": ""}
        if u.ambiguity or v.verdict == "ambiguous":
            cand["reason"] = ("ambiguous input, please clarify before "
                              "anything is saved")
        elif u.predicate == "unknown":
            cand["reason"] = ("unresolvable predicate; "
                              "nothing to save")
        else:
            built = self._candidate_observation(u, v)
            if built is None:
                cand["reason"] = ("this shape is not safely storable yet; "
                                  "nothing saved")
            else:
                cand.update(built)
        pid = self.mem.add_pending_candidate(
            cand["kind"], text, cand["parse"],
            cand.get("observation") or {}, cand["reason"])
        cand["id"] = pid
        cand["status"] = "proposed"
        return cand

    def _candidate_observation(self, u, v) -> Optional[dict]:
        """Map a parsed+grounded utterance to a storable candidate."""
        subj_id = (v.subject_match or {}).get("concept_id")
        if u.predicate == "is_a":
            obj_id = (v.object_match or {}).get("concept_id")
            if u.negated:
                return {"kind": "denial",
                        "observation": {},
                        "can_confirm": True,
                        "reason": "negated category: denial claim only"}
            if subj_id is None:
                if obj_id is None:
                    return None
                return {"kind": "new_concept",
                        "observation": {"name": u.subject,
                                       "parents": [obj_id],
                                       "attributes": {}, "relations": [],
                                       "source": "nl-confirm",
                                       "utterance": u.raw},
                        "can_confirm": True,
                        "reason": f"new concept {u.subject} under {obj_id}"}
            if obj_id is None:
                return None
            if self.onto.is_a(subj_id, obj_id):
                return {"kind": "already_known",
                        "observation": {},
                        "can_confirm": True,
                        "reason": "already supported by current knowledge"}
            return {"kind": "category_claim",
                    "observation": {},
                    "can_confirm": True,
                    "reason": f"category link {subj_id}->{obj_id} kept as "
                    "text claim (no silent reparent)"}
        if u.predicate == "has":
            if u.negated:
                return {"kind": "denial",
                        "observation": {},
                        "can_confirm": True,
                        "reason": "negated possession: denial claim only"}
            if subj_id is None:
                return None
            return {"kind": "attribute",
                    "observation": {"name": u.subject,
                                   "parents": [],
                                   "attributes": {u.object: True},
                                   "relations": [], "source": "nl-confirm",
                                   "utterance": u.raw},
                    "can_confirm": True,
                    "reason": f"attribute {u.object}=True on {subj_id}"}
        if u.predicate == "capable":
            if u.negated:
                return {"kind": "denial",
                        "observation": {},
                        "can_confirm": True,
                        "reason": "negated capability: denial claim only"}
            if subj_id is None:
                return None
            return {"kind": "attribute",
                    "observation": {"name": u.subject,
                                   "parents": [],
                                   "attributes": {"function": u.object},
                                   "relations": [], "source": "nl-confirm",
                                   "utterance": u.raw},
                    "can_confirm": True,
                    "reason": f"function={u.object} on {subj_id}"}
        return None

    def confirm_candidate(self, pid: int) -> dict:
        """Apply a proposed candidate through existing gates.

        Re-checks dedup/conflicts at apply time. Writes attributes
        (hypothesis), denial/category claims (hypothesis) and audit
        experiences only. Never writes verified rows.
        """
        import json as _json
        row = self.mem.get_pending_candidate(pid)
        if row is None:
            return {"ok": False, "reason": f"no such candidate: {pid}"}
        if row.get("status") != "proposed":
            return {"ok": False,
                    "reason": f"candidate is {row.get('status')}, not proposed"}
        try:
            obs = _json.loads(row.get("observation") or "{}")
            parse = _json.loads(row.get("parse") or "{}")
        except Exception:
            return {"ok": False, "reason": "stored candidate unreadable"}
        kind = row.get("kind", "")
        if kind in ("unusable", ""):
            return {"ok": False, "reason": row.get("reason") or
                    "candidate not confirmable"}
        if kind == "already_known":
            self.mem.set_pending_status(pid, "accepted")
            self.mem.add_experience(
                "evaluation", f"candidate #{pid} already known, no writes",
                {"candidate_id": pid})
            return {"ok": True, "wrote": [], "reason": "already known"}
        wrote = []
        utterance = row.get("utterance", "")
        if kind in ("denial", "category_claim"):
            cid = self.mem.add_claim(
                f"用户确认陈述：{utterance}（{kind}，hypothesis，非独立验证事实）",
                subject_id=None, status="hypothesis", confidence=0.5)
            self.mem.add_evidence(
                cid, f"结构化语义：{parse}", source="nl-confirm",
                credibility=0.5)
            wrote.append({"claim_id": cid})
        else:
            res = self.learn_from_observation(obs)
            wrote.append({"observation": res.get("observation", {})})
        self.mem.set_pending_status(pid, "accepted")
        self.mem.add_experience(
            "evaluation", f"candidate #{pid} confirmed and applied",
            {"candidate_id": pid, "kind": kind})
        return {"ok": True, "wrote": wrote, "reason": kind}

    def reject_candidate(self, pid: int) -> dict:
        """Drop a proposed candidate. Only the status flips."""
        row = self.mem.get_pending_candidate(pid)
        if row is None:
            return {"ok": False, "reason": f"no such candidate: {pid}"}
        if row.get("status") != "proposed":
            return {"ok": False,
                    "reason": f"candidate is {row.get('status')}, not proposed"}
        self.mem.set_pending_status(pid, "rejected")
        return {"ok": True, "reason": "candidate rejected, nothing written"}

    def list_pending_candidates(self, status: Optional[str] = None) -> list:
        rows = self.mem.list_pending_candidates(status=status)
        out = []
        for r in rows:
            try:
                import json as _json
                r = dict(r)
                r["parse"] = _json.loads(r.get("parse") or "{}")
                r["observation"] = _json.loads(r.get("observation") or "{}")
            except Exception:
                pass
            out.append(r)
        return out

    # ---- research one question ----
    def research_question(self, qid: int) -> dict:
        """Research a single question. Binding preserved end to end.

        Returns a fixed-shape binding including question_id / target_id /
        key / evidence_sources / action / applied_change_id so the loop log
        always matches the actual modification (or the lack of one).

        Rules enforced:
        - inconclusive NEVER writes semantic knowledge or schema versions;
        - evidence must mention the target concept;
        - at least MIN_CORROBORATING_SOURCES independent hits must agree
          on the same cleaned value before any add_attribute proposal;
        - promoted values are always `hypothesis`, never `verified`.
        """
        q = self.mem.query_one("SELECT * FROM questions WHERE id=?", (qid,))
        if q is None:
            return {"ok": False, "error": "question not found",
                    "question_id": qid, "target_id": None, "key": None,
                    "question": None, "result": "error",
                    "evidence_sources": [], "action": "none",
                    "applied_change_id": None}
        detail = _detail_dict(q.get("detail"))
        attempts = int(detail.get("attempts", 0) or 0)
        target = q["target_id"]
        key, concept_in_text = _parse_gap_question(q["text"])
        base = {"ok": True, "question_id": qid, "target_id": target,
                "key": key, "question": q["text"]}
        if attempts >= MAX_RESEARCH_ATTEMPTS:
            self.mem.add_experience(
                "research", f"Q{qid}: skipped, max attempts reached",
                {"question": q["text"], "attempts": attempts})
            return {**base, "result": "inconclusive",
                    "evidence_sources": [], "action": "skipped_max_attempts",
                    "applied_change_id": None}
        # The question text names its target; a mismatch means the queue row
        # is stale — never write to a concept the question is not about.
        if key and concept_in_text and target and concept_in_text != target:
            return self._record_inconclusive(
                qid, q, detail, attempts, "target_mismatch",
                f"text names '{concept_in_text}' but target is '{target}'",
                [], base)
        self.mem.update_question(qid, status="researching")
        focus = _focus_terms(self.mem, target)
        hits = self.provider.search(q["text"], limit=5,
                                    focus_terms=tuple(focus))
        sources = [h["source"] for h in hits]
        # Only target-relevant, credible hits count as evidence.
        relevant = [h for h in hits
                    if h.get("credibility", 0) >= MIN_EVIDENCE_CREDIBILITY
                    and _mentions(h.get("excerpt", ""), tuple(focus))]
        rel_sources = [h["source"] for h in relevant]
        applied = None
        applied_change_id = None
        action = "none"
        value = None
        if (key and target and self.mem.get_concept(target) is not None
                and q["kind"] == "gap"):
            inh = self.onto.inherited_attributes(target)
            groups = _independent_groups(relevant, key, tuple(focus))
            agreed = [(v, g) for v, g in groups["agreed"].items()
                      if len(g) >= MIN_CORROBORATING_SOURCES]
            if key not in inh:
                if agreed:
                    agreed.sort(key=lambda kv: (-len(kv[1]), kv[0]))
                    value = agreed[0][0]
                    supporting = agreed[0][1]
                    r = self.evolver.direct_apply(
                        "add_attribute",
                        {"concept": target, "key": key, "value": value,
                         "confidence": round(
                             min(h["credibility"] for h in supporting)
                             * 0.8, 3),
                         "source": supporting[0]["source"],
                         "status": "hypothesis"},
                        reason=f"research Q{qid}: {q['text']}",
                        evidence=supporting)
                    if r.get("ok"):
                        applied = r
                        applied_change_id = r.get("change_id")
                        action = "add_attribute"
                        self.mem.record_lifecycle(
                            "attribute", f"{target}.{key}", "candidate",
                            "hypothesis",
                            f"research Q{qid}: {len(supporting)} independent "
                            f"direct sources agree on '{value}'",
                            supporting)
                elif groups["single_source"]:
                    # Exactly one independent origin: keep a candidate CLAIM
                    # (no attribute, no version) so the lead is not lost
                    # while the knowledge base stays clean.
                    v, h = groups["single_source"][0]
                    self._record_candidate(qid, q, target, key, v, h)
            else:
                # Key already known: never overwrite. A corroborated rival
                # value is recorded BESIDE the old row as contested.
                current = inh[key]["value"]
                rivals = [(v, g) for v, g in agreed if v != str(current)
                          and v != current]
                if rivals:
                    rivals.sort(key=lambda kv: (-len(kv[1]), kv[0]))
                    value = rivals[0][0]
                    supporting = rivals[0][1]
                    conflict = self._record_rival(
                        qid, q, target, key, current, value, supporting)
                    if conflict.get("ok"):
                        applied = conflict
                        applied_change_id = conflict.get("change_id")
                        action = "record_conflict"
                elif agreed:
                    # corroborated value equals what we already hold
                    self.mem.update_question(
                        qid, status="learned",
                        detail={**detail, "attempts": attempts + 1,
                                "evidence": rel_sources,
                                "note": "already_known"})
                    self.mem.add_experience(
                        "research", f"Q{qid}: learned (already known)",
                        {"question": q["text"], "target": target,
                         "key": key})
                    return {**base, "result": "learned",
                            "evidence_sources": rel_sources,
                            "action": "already_known",
                            "applied_change_id": None}
                elif groups["single_source"]:
                    v, h = groups["single_source"][0]
                    if v != current and v != str(current):
                        self._record_candidate(qid, q, target, key, v, h)
        if applied:
            self.mem.update_question(
                qid, status="learned",
                detail={**detail, "attempts": attempts + 1,
                        "evidence": rel_sources, "value": value})
            self.mem.add_experience(
                "research", f"Q{qid}: learned",
                {"question": q["text"], "target": target, "key": key,
                 "value": value, "hits": relevant, "applied": applied})
            return {**base, "result": "learned",
                    "evidence_sources": rel_sources,
                    "action": action,
                    "applied_change_id": applied_change_id}
        reason = _inconclusive_reason(hits, relevant, key, target)
        return self._record_inconclusive(qid, q, detail, attempts,
                                         reason[0], reason[1], sources, base)

    def _record_candidate(self, qid: int, q: dict, target: str,
                            key: str, value: str, hit: dict) -> int:
        """Store a single-origin lead as a candidate Claim (no KB write)."""
        claim_id = self.mem.add_claim(
            f"{target}.{key} may be '{value}' (single source, unverified)",
            subject_id=target, status="candidate", confidence=0.35)
        self.mem.add_evidence(
            claim_id, hit.get("excerpt", "")[:600], hit.get("source", ""),
            credibility=hit.get("credibility", 0.5),
            content_hash=hit.get("content_hash", ""),
            relevance="direct",
            origin_note="single independent origin; needs corroboration")
        self.mem.record_lifecycle(
            "claim", str(claim_id), "none", "candidate",
            f"research Q{qid}: single-origin lead, not promoted", [hit])
        return claim_id

    def _record_rival(self, qid: int, q: dict, target: str, key: str,
                      current: Any, value: str, supporting: list) -> dict:
        """Record corroborated rival evidence WITHOUT touching the old row.

        Creates a contested Claim + a contested rival attribute row, logs
        lifecycle contest events against the existing row(s), and returns
        the schema-change result of the rival-row write.
        """
        claim_id = self.mem.add_claim(
            f"{target}.{key}: {current} vs rival {value} "
            f"({len(supporting)} independent sources)",
            subject_id=target, status="contested", confidence=0.5)
        for h in supporting:
            self.mem.add_evidence(
                claim_id, h.get("excerpt", "")[:600], h.get("source", ""),
                credibility=h.get("credibility", 0.5),
                content_hash=h.get("content_hash", ""),
                relevance="direct",
                origin_note="independent origin backing rival value")
        for row in self.mem.get_attributes(target):
            if row["key"] == key and row["status"] in (
                    "candidate", "hypothesis", "supported", "verified"):
                contest_attribute(
                    self.mem, int(row["id"]), supporting,
                    f"research Q{qid}: rival '{value}' from "
                    f"{len(supporting)} independent sources; row untouched")
        return self.evolver.direct_apply(
            "add_attribute",
            {"concept": target, "key": key, "value": value,
             "confidence": round(
                 min(h["credibility"] for h in supporting) * 0.8, 3),
             "source": supporting[0]["source"], "status": "contested"},
            reason=f"research Q{qid}: contested rival for {target}.{key}",
            evidence=supporting)

    def classify(self, obs: dict) -> dict:
        """Rank known concepts for a structured observation.

        Score per concept: +1 per observed attr matching an inherited
        (key, value), -1 per value conflict, -0.25 per observed key the
        concept cannot explain, normalized by number of observed attrs.
        Verdicts: 'match' (best >= threshold, no conflicts), 'ambiguous'
        (best >= threshold but conflicts exist), 'novel' (nothing close AND
        positive difference evidence exists), 'insufficient' (too little
        information to tell: empty input, or no matches AND no conflicts).
        A missing database attribute is NEVER by itself proof of novelty.
        Never writes to the knowledge base.
        """
        from .policy import CLASSIFY_MATCH_THRESHOLD
        attrs: dict = obs.get("attributes", {}) or {}
        if not attrs:
            return {"verdict": "insufficient", "best": None, "ranking": [],
                    "observation": obs.get("name"),
                    "threshold": CLASSIFY_MATCH_THRESHOLD,
                    "reason": "no attributes to compare"}
        scored = []
        for c in self.mem.list_concepts():
            inh = self.onto.inherited_attributes(c["id"])
            match = conflict = unexplained = 0
            details = []
            for k, v in attrs.items():
                if k in inh:
                    if inh[k]["value"] == v:
                        match += 1
                        details.append({"key": k, "kind": "match",
                                        "concept": inh[k]["concept_id"]})
                    else:
                        conflict += 1
                        details.append({"key": k, "kind": "conflict",
                                        "expected": inh[k]["value"],
                                        "observed": v})
                else:
                    unexplained += 1
                    details.append({"key": k, "kind": "unexplained"})
            # (matches - conflicts - 0.25*unexplained) / n_attrs
            score = round((match - conflict - 0.25 * unexplained)
                          / max(1, len(attrs)), 3)
            scored.append({"concept": c["id"], "score": score,
                           "conflicts": [d for d in details
                                         if d["kind"] == "conflict"],
                           "matches": [d for d in details
                                       if d["kind"] == "match"]})
        scored.sort(key=lambda s: (-s["score"], s["concept"]))
        best = scored[0] if scored else None
        # Nothing reaches the threshold: a conflict with ANY known concept
        # is positive difference evidence (novel); with no matches AND no
        # differences there is nothing to go on (insufficient). A missing
        # database attribute is NEVER by itself proof of novelty.
        any_conflict = any(s["conflicts"] for s in scored)
        if best is None or best["score"] < CLASSIFY_MATCH_THRESHOLD:
            verdict = "novel" if any_conflict else "insufficient"
        elif best["conflicts"]:
            verdict = "ambiguous"
        else:
            verdict = "match"
        return {"verdict": verdict, "best": best,
                "ranking": scored[:3],
                "observation": obs.get("name"),
                "threshold": CLASSIFY_MATCH_THRESHOLD}

    def consolidate(self) -> dict:
        """Structural learning step (system C only).

        Finds categories whose children stably diverge on an attribute key
        and proposes a validated split. Each diverging side must be backed
        by at least one research-sourced (local:) hypothesis row so a split
        is never built on a single observation. Major changes pass through
        the regression gate; failures are rejected, never forced.
        Returns a report; never raises on validation failure.
        """
        if not self.enable_structural:
            return {"ok": True, "action": "disabled",
                    "note": "structural learning disabled (system B)"}
        report: list[dict] = []
        for parent in [c["id"] for c in self.mem.list_concepts()]:
            kids = self.mem.get_children(parent)
            if len(kids) < 2:
                continue
            divergent = self.detector._divergent_siblings(parent)
            for k in divergent:
                by_value: dict[str, list[str]] = {}
                backed: dict[str, bool] = {}
                for m in kids:
                    inh = self.onto.inherited_attributes(m)
                    if k not in inh:
                        continue
                    v = json.dumps(inh[k]["value"], sort_keys=True)
                    by_value.setdefault(v, []).append(m)
                    # backed = the value lives in stored knowledge (any
                    # non-rejected lifecycle state), so the divergence is
                    # not a single-observation fluke. Research-gated rows
                    # always qualify; seed rows qualify as prior knowledge.
                    st = inh[k].get("status", "hypothesis")
                    backed[v] = backed.get(v, False) or (st != "rejected")
                if len(by_value) >= 2 and all(backed.values()) and all(
                        len(ms) >= 1 for ms in by_value.values()):
                    groups = {f"{parent}_div{i}": ms
                              for i, ms in enumerate(
                                  sorted(by_value.values(),
                                         key=lambda ms: sorted(ms)))}
                    res = self.evolver.direct_apply(
                        "split_category",
                        {"category": parent, "groups": groups},
                        reason=(f"consolidation: children of {parent} "
                                f"stably diverge on '{k}' with stored values"),
                        evidence=[{"key": k, "distribution": by_value}])
                    report.append({"parent": parent, "key": k, **res})
                    break  # one split per parent per pass; re-run to go deeper
        self.mem.add_experience("evaluation",
                                f"consolidation: {len(report)} proposal(s)",
                                {"report": _jsonable(report)})
        return {"ok": True, "proposals": report}

    def _record_inconclusive(self, qid: int, q: dict, detail: dict,
                             attempts: int, code: str, note: str,
                             sources: list, base: dict) -> dict:
        """Mark inconclusive with attempts + cooldown. NEVER writes schema."""
        import time
        new_detail = {**detail, "attempts": attempts + 1,
                      "cooldown_until": time.time() + INCONCLUSIVE_COOLDOWN_S,
                      "last_reason": code, "last_note": note,
                      "last_evidence": sources}
        self.mem.update_question(qid, status="inconclusive",
                                 detail=new_detail)
        self.mem.add_experience("research", f"Q{qid}: inconclusive ({code})",
                                {"question": q["text"], "reason": note,
                                 "evidence_sources": sources})
        return {**base, "result": "inconclusive",
                "evidence_sources": sources, "action": "none",
                "applied_change_id": None}


def _independent_groups(relevant: list, key: str,
                        focus: tuple[str, ...]) -> dict:
    """Group relevant hits into independent origins per cleaned value.

    - identical content_hash -> same origin (exact copies collapse);
    - excerpt similarity >= NEAR_DUPLICATE_SIMILARITY -> same origin;
    - only 'direct' hits (target mention + key pattern + clean value) vote.
    Returns {"agreed": {value: [hits]}, "single_source": [(value, hit)],
             "merged_duplicates": int}.
    """
    buckets: list[dict] = []  # each: {"hash":..., "excerpt":..., "hits":[...]}
    merged = 0
    for h in relevant:
        exc = h.get("excerpt", "")
        placed = False
        for b in buckets:
            if h.get("content_hash") and h.get("content_hash") == b["hash"]:
                b["hits"].append(h)
                placed = True
                merged += 1
                break
            if text_similarity(exc, b["excerpt"]) >= NEAR_DUPLICATE_SIMILARITY:
                b["hits"].append(h)
                placed = True
                merged += 1
                break
        if not placed:
            buckets.append({"hash": h.get("content_hash", ""),
                            "excerpt": exc, "hits": [h]})
    agreed: dict[str, list] = {}
    single: list[tuple] = []
    for b in buckets:
        # one vote per origin: value from the most credible hit in the bucket
        ordered = sorted(b["hits"],
                         key=lambda h: -h.get("credibility", 0))
        v = _clean_value(_extract_value(ordered[0].get("excerpt", ""), key))
        if not v:
            continue
        grade = grade_relevance(ordered[0].get("excerpt", ""), focus, key)
        if grade != "direct":
            continue
        rep = dict(ordered[0])
        rep["relevance"] = grade
        rep["origin_note"] = ("merged duplicates" if len(b["hits"]) > 1
                              else "independent origin")
        agreed.setdefault(v, []).append(rep)
    for v, hs in list(agreed.items()):
        if len(hs) < MIN_CORROBORATING_SOURCES:
            single.extend((v, h) for h in hs)
            del agreed[v]
    return {"agreed": agreed, "single_source": single,
            "merged_duplicates": merged}


def _detail_dict(raw: Any) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def _parse_gap_question(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract (key, concept) from "What is the 'key' of concept 'c'?".

    Returns (None, None) for non-gap question kinds.
    """
    import re
    m = re.search(r"'([^']+)' of concept '([^']+)'", text or "")
    if m:
        return m.group(1), m.group(2)
    return None, None


def _focus_terms(mem: Memory, target: Optional[str]) -> list[str]:
    """Terms proving a hit is about the target: id + concept name."""
    if not target:
        return []
    terms = [target]
    c = mem.get_concept(target)
    if c and c.get("name") and c["name"] != target:
        terms.append(c["name"])
    return terms


def _inconclusive_reason(hits: list, relevant: list,
                         key: Optional[str],
                         target: Optional[str]) -> tuple[str, str]:
    if not hits:
        return ("no_evidence", "no corpus document matched the question")
    if not relevant:
        return ("no_target_relevant_evidence",
                f"hits exist but none credibly discuss target '{target}'")
    if not key or not target:
        return ("not_a_gap_question",
                "only missing-attribute gaps can propose attributes")
    return ("no_corroborated_value",
            "no value agreed upon by enough independent sources")


# Fragments that prove an extracted string is a sentence remnant, not a value.
_VALUE_POISON = (" 的 ", " 是 ", "：", ":", "；", ";", "。", ".", "/", "\\",
                 "concept", "material", "function", "location", "relation",
                 "attribute")


def _clean_value(raw: Optional[str]) -> Optional[str]:
    """Accept only short atomic values; reject sentence fragments."""
    if not raw:
        return None
    v = raw.strip().strip("'\"“”‘’").strip()
    v = re.sub(r"[。；;.\s]+$", "", v).strip()
    if not v or len(v) > 24:
        return None
    if any(p in v for p in _VALUE_POISON):
        return None
    if len(v.split()) > 2:
        return None
    return v or None


def _jsonable(o: Any) -> Any:
    try:
        json.dumps(o, ensure_ascii=False, default=str)
        return json.loads(json.dumps(o, ensure_ascii=False, default=str))
    except Exception:
        return str(o)


def _extract_value(excerpt: str, key: str) -> Optional[str]:
    """Deterministic tiny extractor: 'key: value' or 'key是value' in excerpt.

    Capture stops at sentence/field boundaries so only the adjacent value
    fragment is returned; _clean_value() validates it further.
    """
    import re
    stop = r"\n；;。.，,"
    pats = [re.compile(rf"{re.escape(key)}\s*[:：]\s*([^{stop}]{{1,40}})"),
            re.compile(rf"{re.escape(key)}\s*是\s*([^{stop}]{{1,40}})")]
    for p in pats:
        m = p.search(excerpt)
        if m:
            return m.group(1).strip()
    return None
