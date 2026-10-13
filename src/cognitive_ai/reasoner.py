"""Explainable rule reasoner. Returns structured results, never guesses unknown."""
from __future__ import annotations

import json
from typing import Optional

from .memory import Memory
from .ontology import Ontology


class ReasoningResult(dict):
    pass


class Reasoner:
    def __init__(self, memory: Memory):
        self.mem = memory
        self.onto = Ontology(memory)

    # ---- primitive queries ----
    def is_a(self, cid: str, ancestor: str) -> ReasoningResult:
        exists = self.mem.get_concept(cid) is not None
        anc_exists = self.mem.get_concept(ancestor) is not None
        if not exists or not anc_exists:
            return ReasoningResult({
                "query": {"type": "is_a", "concept": cid, "ancestor": ancestor},
                "conclusion": "unknown",
                "confidence": 0.0,
                "trace": [{"rule": "concept_lookup",
                           "detail": "one or both concepts not found"}],
            })
        val = self.onto.is_a(cid, ancestor)
        return ReasoningResult({
            "query": {"type": "is_a", "concept": cid, "ancestor": ancestor},
            "conclusion": bool(val),
            "confidence": 1.0 if val else 0.9,
            "trace": [{"rule": "transitive_is_a",
                       "facts": [f"{cid} ancestors={self.onto.ancestors(cid)}"]}],
        })

    def get_attribute(self, cid: str, key: str) -> ReasoningResult:
        if self.mem.get_concept(cid) is None:
            return ReasoningResult({
                "query": {"type": "get_attribute", "concept": cid, "key": key},
                "conclusion": "unknown", "confidence": 0.0,
                "trace": [{"rule": "concept_lookup",
                           "detail": f"concept not found: {cid}"}],
            })
        inherited = self.onto.inherited_attributes(cid)
        if key in inherited:
            a = inherited[key]
            return ReasoningResult({
                "query": {"type": "get_attribute", "concept": cid, "key": key},
                "conclusion": a["value"], "confidence": a["confidence"],
                "status": a["status"],
                "trace": [{"rule": "attribute_inheritance",
                           "facts": [f"{a['concept_id']}.{key}={a['value']} "
                                     f"status={a['status']} source={a['source']}"]}],
            })
        return ReasoningResult({
            "query": {"type": "get_attribute", "concept": cid, "key": key},
            "conclusion": "unknown", "confidence": 0.0,
            "trace": [{"rule": "attribute_inheritance",
                       "detail": f"no value for {key} on {cid} or ancestors"}],
        })

    def query_relations(self, src: str, rel_type: Optional[str] = None) -> ReasoningResult:
        rels = self.mem.get_relations(src_id=src, rel_type=rel_type)
        return ReasoningResult({
            "query": {"type": "relations", "src": src, "rel_type": rel_type},
            "conclusion": rels, "confidence": 1.0,
            "trace": [{"rule": "relation_lookup",
                       "facts": [f"found {len(rels)} relation(s)"]}],
        })

    # ---- rule engine ----
    def _cond_holds(self, cond: dict) -> tuple[bool | None, str]:
        """Returns (True/False/None-unknown, explanation)."""
        t = cond.get("type")
        if t == "is_a":
            r = self.is_a(cond["concept"], cond["ancestor"])
            if r["conclusion"] == "unknown":
                return None, f"unknown is_a {cond['concept']}->{cond['ancestor']}"
            return bool(r["conclusion"]), f"is_a {cond['concept']}->{cond['ancestor']}={r['conclusion']}"
        if t == "has_attr":
            r = self.get_attribute(cond["concept"], cond["key"])
            if r["conclusion"] == "unknown":
                return None, f"unknown attr {cond['concept']}.{cond['key']}"
            if "value" in cond:
                return (r["conclusion"] == cond["value"],
                        f"{cond['concept']}.{cond['key']}={r['conclusion']} expected={cond['value']}")
            return True, f"{cond['concept']} has {cond['key']}"
        if t == "relation":
            rels = self.mem.get_relations(src_id=cond.get("src"),
                                          rel_type=cond.get("rel_type"))
            want = cond.get("dst")
            if want is None:
                return (len(rels) > 0), f"relation exists: {len(rels)}"
            hit = any(r["dst_id"] == want for r in rels)
            return hit, f"relation {cond.get('src')} -{cond.get('rel_type')}-> {want} = {hit}"
        return None, f"unsupported condition type: {t}"

    def forward_chain(self, max_steps: int = 50) -> list[dict]:
        """Apply rules with variable binding over KB + derived facts.

        Round 0 reproduces the original single-pass entries exactly
        (skipped / unknown / conclusion+trace / not_fired); further
        rounds (up to policy.MAX_CHAIN_DEPTH) feed derived conclusions
        back as premises, so one rule's output can satisfy another
        rule's input. Derived facts live only in this run's working
        set -- nothing is written to the KB. Each derived entry records
        rule id, bindings, premises, depth and weakest status. Budget
        exhaustion appends an explicit limited marker instead of
        silently stopping.
        """
        from . import policy
        return self._chain_rounds(max_steps, policy.MAX_CHAIN_DEPTH,
                                  policy.MAX_CHAIN_WORK)

    def _chain_rounds(self, max_steps: int, max_depth: int,
                      max_work: int) -> list[dict]:
        fired: list[dict] = []
        derived: list[dict] = []  # working set, never persisted
        seen: set[str] = set()  # (rule, conclusion) dedup across rounds
        state = {"work": 0, "limited": False}

        def note_explain(entries):
            for e in entries:
                fired.append(e)
                if len(fired) >= max_steps:
                    break

        for depth in range(max_depth):
            new_facts: list[dict] = []
            for rule in self.mem.list_rules():
                if len(fired) >= max_steps or state["limited"]:
                    break
                for entry, facts in self._apply_rule_bound(
                        rule, derived, depth, state, max_work,
                        shapes=(depth == 0)):
                    novel = [f for f in facts if (f.get("rule_id"),
                             json.dumps(f.get("fact", {}), sort_keys=True,
                                        default=str)) not in seen]
                    # round 0 always records (legacy shapes); later
                    # rounds only record genuinely new conclusions --
                    # re-derivation across rounds is not re-reported.
                    if entry is not None and (depth == 0 or novel):
                        note_explain([entry])
                    for f in novel:
                        # the fact carries its own proof node so later
                        # rounds chain premises, not just values
                        f["node"] = entry
                        seen.add((f.get("rule_id"),
                                  json.dumps(f.get("fact", {}),
                                             sort_keys=True, default=str)))
                        new_facts.append(f)
                    if len(fired) >= max_steps or state["limited"]:
                        break
            if state["limited"]:
                break
            if not new_facts:
                if depth == 0:
                    # round 0 with no derivations == legacy single pass;
                    # entries already recorded above, nothing more to do
                    pass
                break
            derived.extend(new_facts)
            if len(fired) >= max_steps:
                break
        if state["limited"]:
            fired.append({"rule": "$budget", "limited": True,
                          "reason": "chaining stopped: work budget "
                          f"exhausted at depth {depth}"})
        self._report_conclusion_conflicts(fired)
        return fired

    def _report_conclusion_conflicts(self, fired: list[dict]) -> None:
        """One contested entry per rule behind conflicting values.

        Distinct derived/KB values for one (concept, key) never resolve
        silently: every involved rule gets an explicit contested entry
        while all rival conclusions stay in the trace.
        """
        import json as _json
        groups: dict = {}
        for f in fired:
            conc = f.get("conclusion")
            if not isinstance(conc, dict) or conc.get("type") != "has_attr":
                continue
            cid, key = conc.get("concept"), conc.get("key")
            if not cid or not key or self._is_var(cid) or self._is_var(key):
                continue
            val = conc.get("value")
            groups.setdefault((cid, key), {}).setdefault(
                _json.dumps(val, sort_keys=True, default=str), []).append(f)
        for (cid, key), byval in groups.items():
            if len(byval) < 2:
                continue
            kb_vals = {v for v in self._rival_values(cid, key, None)}
            involved = {f.get("rule_id", f.get("rule", "?"))
                        for vs in byval.values() for f in vs}
            detail = (f"conflicting conclusions for {cid}.{key}: " +
                      "; ".join(sorted(byval)) +
                      (f"; KB also stores {sorted(kb_vals)}"
                       if kb_vals else ""))
            for rid in sorted(involved):
                if any(x.get("rule_id") == rid and
                       x.get("conclusion") == "contested" for x in fired):
                    continue
                fired.append({"rule": rid, "rule_id": rid,
                              "conclusion": "contested",
                              "reason": detail})

    def verify_proof(self, proof: dict) -> dict:
        """Independently re-check one recorded proof node.

        Re-derives every premise from the live KB (never trusts the
        recorded trace), re-checks rule existence and exceptions,
        binding consistency and conclusion match. Returns
        {result: verified|rejected, detail}. Pure read path: no writes.
        """
        goal = proof.get("goal", {}) or {}
        bindings = dict(proof.get("bindings", {}) or {})
        if proof.get("rule_id") is None:
            res, info = self._check_ground(goal, bindings)
            if res != "ok":
                return {"result": "rejected",
                        "detail": f"KB leaf no longer holds ({res})"}
            if "value" in proof and "value" in (info or {}):
                if info["value"] != proof["value"]:
                    return {"result": "rejected",
                            "detail": "leaf value drifted"}
            return {"result": "verified", "detail": "KB fact holds"}
        rows = [r for r in self.mem.list_rules()
                if r.get("id") == proof["rule_id"]]
        if not rows:
            return {"result": "rejected",
                    "detail": "rule not in memory"}
        rule = rows[0]
        conc = rule.get("conclusion", {}) or {}
        inst = {k: self._resolve(v, bindings) for k, v in conc.items()}
        for v in inst.values():
            if self._is_var(v):
                return {"result": "rejected",
                        "detail": f"unbound variable {v} in conclusion"}
        for k, v in inst.items():
            if k in goal and not self._is_var(goal.get(k)) and v != goal[k]:
                return {"result": "rejected",
                        "detail": f"binding contradicts goal at {k}"}
        if self._rule_exc_holds(rule, bindings):
            return {"result": "rejected",
                    "detail": "rule exception holds on re-check"}
        conds = rule.get("conditions", []) or []
        premises = proof.get("premises", []) or []
        if len(premises) != len(conds):
            return {"result": "rejected",
                    "detail": f"premise count {len(premises)} != "
                    f"{len(conds)} rule conditions (missing premises)"}
        for cond, prem in zip(conds, premises):
            if not self._premise_covers(cond, prem, bindings):
                return {"result": "rejected",
                        "detail": f"premise does not cover condition "
                        f"{cond}: {prem.get('goal')}"}
            sub = self.verify_proof(prem)
            if sub["result"] != "verified":
                return {"result": "rejected",
                        "detail": f"premise failed re-check: {sub['detail']}"}
        if "value" in proof and "value" in inst and \
                proof["value"] != inst["value"]:
            return {"result": "rejected",
                    "detail": "claimed value differs from re-derivation"}
        return {"result": "verified",
                "detail": f"rule {rule.get('id')}, "
                f"{len(conds)} premise(s) re-derived"}

    def _check_ground(self, goal: dict, bindings: dict):
        """Ground fact check against live KB (verifier side)."""
        g = {k: self._resolve(v, bindings) for k, v in goal.items()}
        t = g.get("type")
        if t == "is_a":
            c, a = g.get("concept"), g.get("ancestor")
            if self._is_var(c) or self._is_var(a):
                return ("unknown", None)
            r = self.is_a(c, a)
            if r["conclusion"] == "unknown":
                return ("unknown", None)
            if r["conclusion"]:
                return ("ok", {"status": "verified"})
            return ("mismatch", None)
        if t == "has_attr":
            c, k = g.get("concept"), g.get("key")
            if self._is_var(c) or self._is_var(k):
                return ("unknown", None)
            r = self.get_attribute(c, k)
            if r["conclusion"] == "unknown":
                return ("unknown", None)
            if "value" in g and not self._is_var(g["value"]) and \
                    r["conclusion"] != g["value"]:
                return ("mismatch", r)
            return ("ok", {"status": r.get("status", "verified"),
                           "value": r["conclusion"]})
        if t == "relation":
            r = self.query_relations(g.get("src"), g.get("rel_type"))
            rels = r["conclusion"] or []
            if "dst" in g and not self._is_var(g["dst"]):
                ok = any(x.get("dst_id") == g["dst"] for x in rels)
                return ("ok", {"status": "verified"}) if ok else \
                    ("mismatch", None)
            return ("ok", {"status": "verified"}) if rels else \
                ("mismatch", None)
        return ("unknown", None)

    def _rule_exc_holds(self, rule: dict, bindings: dict) -> bool:
        for e in rule.get("exceptions", []) or []:
            g = {k: self._resolve(v, bindings) for k, v in e.items()}
            res, _ = self._check_ground(g, {})
            if res == "ok":
                return True
        return False

    def _premise_covers(self, cond: dict, prem: dict,
                        bindings: dict) -> bool:
        pg = prem.get("goal", {}) or {}
        if pg.get("type") != cond.get("type"):
            return False
        pb = prem.get("bindings", {}) or {}
        for k, v in cond.items():
            if k == "type":
                continue
            cv = self._resolve(v, bindings)
            pv = self._resolve(pg.get(k), pb)
            if not self._is_var(cv) and not self._is_var(pv) and cv != pv:
                return False
        return True

    # ---- variable binding machinery ----
    @staticmethod
    def _is_var(v) -> bool:
        return isinstance(v, str) and v.startswith("?") and len(v) > 1

    @staticmethod
    def _resolve(v, bindings: dict):
        while Reasoner._is_var(v) and v in bindings:
            v = bindings[v]
        return v

    def _kb_fact_candidates(self, cond: dict, bindings: dict,
                            state: dict, max_work: int) -> list:
        """Candidate (delta-bindings, evidence, status) rows for one
        premise against KB facts. Respects the work budget."""
        t = cond.get("type")
        out = []
        grounds = {k: self._resolve(v, bindings)
                   for k, v in cond.items() if k != "type"}
        if t == "is_a":
            c, a = grounds.get("concept"), grounds.get("ancestor")
            if not self._is_var(c) and not self._is_var(a):
                r = self.is_a(c, a)
                if r["conclusion"] == "unknown":
                    return [{"unknown": True,
                             "detail": f"unknown is_a {c}->{a}"}]
                if r["conclusion"]:
                    return [{"bind": {}, "status": "verified",
                             "evidence": [f"is_a {c}->{a}"]}]
                return []
            # enumerate one side, verify the other (budget-capped)
            cands = []
            if self._is_var(c) and not self._is_var(a):
                for row in self.mem.list_concepts():
                    state["work"] += 1
                    if state["work"] > max_work:
                        return [{"limited": True}]
                    if self.onto.is_a(row["id"], a):
                        cands.append(row["id"])
                return [{"bind": {c: x}, "status": "verified",
                         "evidence": [f"is_a {x}->{a}"]} for x in cands]
            if not self._is_var(c) and self._is_var(a):
                try:
                    chain = [c] + self.onto.ancestors(c)
                except Exception:
                    return [{"unknown": True,
                             "detail": f"unknown concept {c}"}]
                return [{"bind": {a: x}, "status": "verified",
                         "evidence": [f"is_a {c}->{x}"]} for x in chain]
            return [{"unknown": True,
                     "detail": "both sides unbound in is_a"}]
        if t == "has_attr":
            c, k = grounds.get("concept"), grounds.get("key")
            want = grounds.get("value", None)
            rows = []
            if not self._is_var(c) and not self._is_var(k):
                r = self.get_attribute(c, k)
                if r["conclusion"] == "unknown":
                    return [{"unknown": True,
                             "detail": f"unknown attr {c}.{k}"}]
                rows = [(c, k, r["conclusion"],
                         r.get("status", "verified"),
                         f"{c}.{k}={r['conclusion']}")]
                rival = self._rival_values(c, k, r["conclusion"])
                if rival:
                    return [{"contested": True,
                             "detail": f"conflicting values for {c}.{k}: "
                             f"{rival}"}]
            elif not self._is_var(k):
                for row in self.mem.list_concepts():
                    state["work"] += 1
                    if state["work"] > max_work:
                        return [{"limited": True}]
                    r = self.get_attribute(row["id"], k)
                    if r["conclusion"] != "unknown":
                        rows.append((row["id"], k, r["conclusion"],
                                     r.get("status", "verified"),
                                     f"{row['id']}.{k}={r['conclusion']}"))
            else:
                return [{"unknown": True,
                         "detail": "unbound key in has_attr"}]
            hits = []
            for (cc, kk, vv, st, ev) in rows:
                if want is not None and not self._is_var(want) and \
                        vv != want:
                    continue
                bind = {}
                if self._is_var(c):
                    bind[c] = cc
                if self._is_var(want):
                    bind[want] = vv
                hits.append({"bind": bind, "status": st, "evidence": [ev]})
            return hits
        if t == "relation":
            s, rt = grounds.get("src"), grounds.get("rel_type")
            d = grounds.get("dst")
            try:
                rels = self.mem.get_relations(
                    src_id=None if self._is_var(s) else s,
                    rel_type=None if self._is_var(rt) else rt)
            except Exception:
                return [{"unknown": True, "detail": "relation lookup failed"}]
            hits = []
            for r in rels:
                state["work"] += 1
                if state["work"] > max_work:
                    return [{"limited": True}]
                bind = {}
                if self._is_var(s):
                    bind[s] = r.get("src_id")
                if self._is_var(rt):
                    bind[rt] = r.get("rel_type")
                if d is None:
                    pass
                elif self._is_var(d):
                    bind[d] = r.get("dst_id")
                elif r.get("dst_id") != d:
                    continue
                hits.append({"bind": bind, "status": "verified",
                             "evidence": [f"{r.get('src_id')}-"
                                          f"{r.get('rel_type')}->"
                                          f"{r.get('dst_id')}"]})
            return hits
        return [{"unknown": True,
                 "detail": f"unsupported condition type: {t}"}]

    def _match_derived(self, cond: dict, bindings: dict,
                       derived: list[dict]) -> list[dict]:
        """Match one premise against working-set facts. Returns
        (delta-bindings, evidence, status, fact) rows."""
        out = []
        for f in derived:
            fact = f.get("fact", {})
            if fact.get("type") != cond.get("type"):
                continue
            delta: dict = {}
            ok = True
            for k, pv in cond.items():
                if k == "type":
                    continue
                pv = self._resolve(pv, bindings)
                fv = fact.get(k, None)
                if self._is_var(pv) and self._is_var(fv):
                    if pv != fv:
                        delta[pv] = fv
                elif self._is_var(pv):
                    if pv in delta and delta[pv] != fv:
                        ok = False
                        break
                    delta[pv] = fv
                elif self._is_var(fv):
                    continue  # derived fact still open: no constraint
                elif pv != fv:
                    ok = False
                    break
            if ok:
                out.append({"bind": delta, "status": f.get("status",
                                                           "verified"),
                            "evidence": [f"derived:{f.get('rule_id')}"],
                            "fact": fact, "node": f.get("node")})
        return out

    def _apply_rule_bound(self, rule: dict, derived: list[dict],
                          depth: int, state: dict, max_work: int,
                          shapes: bool = True):
        """Yield (entry, new-facts) for one rule in one round."""
        from . import policy
        conds = rule.get("conditions", []) or []
        excs = rule.get("exceptions", []) or []
        # join premises in order; each step extends consistent bindings
        # partial: (bindings, evidence-strings, status, proof-nodes)
        partials = [({}, [], "verified", [])]
        unknown_seen = []
        contested_seen = []
        for cond in conds:
            nxt = []
            for (b, ev, st, nodes) in partials:
                cands = self._kb_fact_candidates(cond, b, state, max_work)
                if any("limited" in c for c in cands):
                    state["limited"] = True
                    return
                cands = cands + self._match_derived(cond, b, derived)
                for c in cands:
                    if "contested" in c:
                        contested_seen.append(c.get("detail", "contested"))
                        continue
                    state["work"] += 1
                    if state["work"] > max_work:
                        state["limited"] = True
                        return
                    if c.get("unknown"):
                        unknown_seen.append(c.get("detail", ""))
                        continue
                    merged = dict(b)
                    clash = False
                    for k, v in (c.get("bind") or {}).items():
                        if k in merged and merged[k] != v:
                            clash = True
                            break
                        merged[k] = v
                    if clash:
                        continue
                    nst = st
                    for s in (c.get("status", "verified"),):
                        nst = nst if self._rank(nst) <= self._rank(s) \
                            else s
                    leaf = c.get("node")
                    if leaf is None:
                        leaf = {
                            "goal": {k: self._resolve(v, merged)
                                     for k, v in cond.items()},
                            "bindings": dict(merged),
                            "status": c.get("status", "verified"),
                            "basis": list(c.get("evidence", [])),
                            "premises": []}
                    nxt.append((merged, ev + c.get("evidence", []), nst,
                                nodes + [leaf]))
                if len(nxt) > policy.MAX_CHAIN_BINDINGS:
                    nxt = nxt[:policy.MAX_CHAIN_BINDINGS]
                    state["limited"] = True
            partials = nxt
            if not partials and not unknown_seen:
                break
        if state["limited"]:
            return
        if not partials:
            if not shapes:
                return
            if contested_seen:
                yield ({"rule": rule["id"], "rule_id": rule["id"], "conclusion": "contested",
                        "reason": "contested premises kept, no assertion: "
                        + "; ".join(dict.fromkeys(contested_seen))}, [])
                return
            if unknown_seen:
                yield ({"rule": rule["id"], "rule_id": rule["id"], "conclusion": "unknown",
                        "reason": "missing evidence: " +
                        "; ".join(dict.fromkeys(unknown_seen))}, [])
            else:
                # every premise refuted: honest non-fire, same shape as ever
                yield ({"rule": rule["id"], "rule_id": rule["id"], "conclusion": "not_fired",
                        "trace": []}, [])
            return
        for (b, ev, st, nodes) in partials:
            # exceptions share the firing bindings; an exception that can
            # be grounded to true (possibly by extending bindings) blocks
            # only this binding, never the whole rule.
            blocked, why = self._exception_blocks(rule, b, derived,
                                                  state, max_work)
            if state["limited"]:
                return
            if blocked:
                if shapes:
                    yield ({"rule": rule["id"], "rule_id": rule["id"], "skipped": True,
                            "bindings": dict(b), "reason": why}, [])
                continue
            conc = rule.get("conclusion", {}) or {}
            inst = {}
            open_var = False
            for k, v in conc.items():
                rv = self._resolve(v, b)
                if self._is_var(rv):
                    open_var = True
                    break
                inst[k] = rv
            if open_var or "type" not in inst:
                if shapes:
                    yield ({"rule": rule["id"], "rule_id": rule["id"], "conclusion": "unknown",
                            "bindings": dict(b),
                            "reason": "conclusion has unbound "
                            "variables"}, [])
                continue
            if st == "contested" or "contested" in str(st):
                if shapes:
                    yield ({"rule": rule["id"], "rule_id": rule["id"], "conclusion": "contested",
                            "bindings": dict(b), "trace": ev,
                            "reason": "contested premises kept, "
                            "no assertion"}, [])
                continue
            node = {"rule": rule["id"], "rule_id": rule["id"],
                    "conclusion": dict(inst),
                    "bindings": dict(b), "depth": depth + 1,
                    "status": st, "trace": list(ev),
                    "premises": list(nodes)}
            yield (node,
                   [{"rule_id": rule["id"], "fact": dict(inst),
                     "status": st, "depth": depth + 1, "node": node}])

    def _rival_values(self, cid: str, key: str, value) -> list:
        """Distinct stored values for the applicable rows of cid.key.

        Own rows first; if none, the nearest ancestor rows that
        inheritance would actually read (mirrors inherited_attributes).
        One distinct value -> no conflict.
        """
        import json as _json
        try:
            own = [a.get("value") for a in self.mem.get_attributes(cid)
                   if a.get("key") == key]
            pool = own if own else []
            if not pool:
                for anc in self.onto.ancestors(cid):
                    rows = [a.get("value")
                            for a in self.mem.get_attributes(anc)
                            if a.get("key") == key]
                    if rows:
                        pool = rows
                        break
            seen = sorted({_json.dumps(v, sort_keys=True, default=str)
                           for v in pool})
            return seen if len(seen) > 1 else []
        except Exception:
            return []

    @staticmethod
    def _rank(status: str) -> int:
        return {"verified": 4, "inherited(verified)": 3, "hypothesis": 2,
                "unknown": 0, "contested": -1}.get(str(status), 0)

    def _exception_blocks(self, rule: dict, bindings: dict,
                          derived: list[dict], state: dict,
                          max_work: int):
        """True (+reason) if some exception holds under these bindings
        (extending them if needed); per-binding, never whole-rule."""
        for e in rule.get("exceptions", []) or []:
            cands = self._kb_fact_candidates(e, bindings, state, max_work)
            if any("limited" in c for c in cands):
                continue
            cands = [c for c in cands if not c.get("unknown")] + \
                self._match_derived(e, bindings, derived)
            if cands:
                return True, f"exception holds: {e} via {cands[0]}"
        return False, ""

    def check_conflicts(self) -> list[dict]:
        """Find same concept+key with different values, or contested claims."""
        conflicts: list[dict] = []
        for c in self.mem.list_concepts():
            by_key: dict[str, set] = {}
            for a in self.mem.get_attributes(c["id"]):
                by_key.setdefault(a["key"], set()).add(json.dumps(a["value"], sort_keys=True))
            for k, vals in by_key.items():
                if len(vals) > 1:
                    conflicts.append({"type": "attribute_conflict",
                                      "concept": c["id"], "key": k,
                                      "values": sorted(vals)})
        for cl in self.mem.list_claims():
            if cl["status"] == "contested":
                conflicts.append({"type": "contested_claim", "claim": cl})
        return conflicts

    def answer(self, query: dict) -> ReasoningResult:
        qtype = query.get("type")
        if qtype == "is_a":
            return self.is_a(query["concept"], query["ancestor"])
        if qtype == "get_attribute":
            return self.get_attribute(query["concept"], query["key"])
        if qtype == "relations":
            return self.query_relations(query.get("src"), query.get("rel_type"))
        if qtype == "forward_chain":
            return ReasoningResult({"query": query,
                                    "conclusion": self.forward_chain(),
                                    "confidence": 1.0, "trace": []})
        if qtype == "conflicts":
            return ReasoningResult({"query": query,
                                    "conclusion": self.check_conflicts(),
                                    "confidence": 1.0, "trace": []})
        return ReasoningResult({"query": query, "conclusion": "unknown",
                                "confidence": 0.0,
                                "trace": [{"detail": "unsupported query type"}]})
