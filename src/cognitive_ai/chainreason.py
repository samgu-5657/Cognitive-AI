"""Multi-step factual chaining with variables (Phase 6H-3A).

Stored rules (memory.rules) are concrete-only and forward_chain() runs
a single pass. This module adds the missing link: rule conditions and
conclusions may contain variables ("?x"), so one rule's derived
conclusion can satisfy another rule's premise. Bounded depth-first
proof search with visited-state pruning, weakest-status propagation,
conflict detection across proofs, and full proof records. NOTHING here
writes to the KB: conclusions live in the returned proof structures
(and the executor trace) until the lifecycle explicitly promotes them.

Composability is conservative on purpose: only DECLARED rules chain.
is_a transitivity comes from the existing ontology ancestors (the same
semantics reasoner.is_a already uses); relations NEVER transfer
implicitly -- a relation conclusion needs an explicit rule.
"""
from __future__ import annotations

import json

from . import policy

_MISSING = object()

# Status lattice (rank): derived conclusions never exceed their weakest
# premise. Contested anywhere poisons that branch into a conflict.
_STATUS_RANK = {"verified": 4, "inherited(verified)": 3, "hypothesis": 2,
                "unknown": 0, "contested": -1}


def is_var(v) -> bool:
    return isinstance(v, str) and v.startswith("?") and len(v) > 1


def resolve(v, bindings: dict):
    while is_var(v) and v in bindings:
        v = bindings[v]
    return v


def _resolve_pattern(pattern: dict, bindings: dict) -> dict:
    return {k: resolve(v, bindings) for k, v in pattern.items()
            if k != "type"}


def unify(pattern: dict, fact: dict, bindings: dict) -> dict | None:
    """Extend bindings so pattern matches fact, or None. Pure."""
    out = dict(bindings)
    for k, pv in pattern.items():
        if k == "type":
            if fact.get("type") != pv:
                return None
            continue
        pv, fv = resolve(pv, out), fact.get(k, _MISSING)
        if fv is _MISSING:
            return None
        if is_var(pv) and is_var(fv):
            if pv != fv:
                out[pv] = fv
        elif is_var(pv):
            out[pv] = fv
        elif is_var(fv):
            out[fv] = pv
        elif pv != fv:
            return None
    return out


def _key(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


class _KB:
    """Fact source over the SAME stores the reasoner uses."""

    def __init__(self, memory, ontology):
        self.mem = memory
        self.onto = ontology

    def is_a(self, cid: str, ancestor: str):
        if self.mem.get_concept(cid) is None or \
                self.mem.get_concept(ancestor) is None:
            return {"found": False}
        try:
            val = self.onto.is_a(cid, ancestor)
        except Exception:
            return {"found": False}
        if not val:
            return {"found": False}
        return {"found": True, "status": "verified",
                "basis": [f"{cid} ancestors={self.onto.ancestors(cid)}"]}

    def has_attr(self, cid: str, key: str):
        if self.mem.get_concept(cid) is None:
            return {"found": False}
        try:
            inh = self.onto.inherited_attributes(cid)
        except Exception:
            return {"found": False}
        if key not in inh:
            return {"found": False}
        row = inh[key]
        return {"found": True, "status": row["status"]
                if row["concept_id"] == cid
                else f"inherited({row['status']})",
                "value": row["value"],
                "basis": [f"{row['concept_id']}.{key}={row['value']} "
                          f"[{row['status']}/{row['source']}]"]}

    def relation(self, src: str, rel_type: str, dst=None):
        try:
            rels = self.mem.get_relations(src_id=src, rel_type=rel_type)
        except Exception:
            return {"found": False}
        if dst is None:
            if not rels:
                return {"found": False}
            return {"found": True, "status": "verified",
                    "basis": [f"{len(rels)} relation(s)"]}
        hit = any(r.get("dst_id") == dst for r in rels)
        if not hit:
            return {"found": False}
        # relations carry no lifecycle status column (pre-existing gap,
        # same as reasoner semantics): reported as verified lookups.
        return {"found": True, "status": "verified",
                "basis": [f"{src} -{rel_type}-> {dst}"]}

    def rules(self) -> list[dict]:
        try:
            return self.mem.list_rules()
        except Exception:
            return []

    def fact_lookup(self, pattern: dict, bindings: dict):
        """Ground-or-bound pattern against KB. Returns
        ("ok", value/status/basis) | ("unknown",) | ("contested",)."""
        t = pattern.get("type")
        if t == "is_a":
            c, a = resolve(pattern.get("concept"), bindings), \
                resolve(pattern.get("ancestor"), bindings)
            if is_var(c) or is_var(a):
                return ("unknown", None)
            r = self.is_a(c, a)
            return ("ok", r) if r["found"] else ("unknown", None)
        if t == "has_attr":
            c, k = resolve(pattern.get("concept"), bindings), \
                resolve(pattern.get("key"), bindings)
            if is_var(c) or is_var(k):
                return ("unknown", None)
            r = self.has_attr(c, k)
            if not r["found"]:
                return ("unknown", None)
            if r["status"] == "contested" or \
                    str(r["status"]).startswith("inherited(contested"):
                return ("contested", r)
            if "value" in pattern and not is_var(
                    resolve(pattern["value"], bindings)):
                want = resolve(pattern["value"], bindings)
                if r["value"] != want:
                    return ("mismatch", r)
            return ("ok", r)
        if t == "relation":
            s, rt = resolve(pattern.get("src"), bindings), \
                resolve(pattern.get("rel_type"), bindings)
            d = resolve(pattern.get("dst"), bindings) \
                if "dst" in pattern else None
            if is_var(s) or is_var(rt) or (d is not None and is_var(d)):
                return ("unknown", None)
            r = self.relation(s, rt, d)
            return ("ok", r) if r["found"] else ("unknown", None)
        return ("unknown", None)


def _weaker(a: str, b: str) -> str:
    ra, rb = _STATUS_RANK.get(a, 0), _STATUS_RANK.get(b, 0)
    return a if ra <= rb else b


def prove(goal: dict, kb: _KB, bindings: dict | None = None,
          depth: int | None = None, visited: frozenset | None = None,
          _proofs: list | None = None) -> list[dict]:
    """Backward-chaining proof search. Returns proof nodes.

    Each proof: {goal, bindings, status, value?, premises: [proof|fact],
    rule_id?, basis[]}. Multiple proofs allowed (capped). Cycles and
    equivalent paths pruned via visited goal keys. Depth bounded by
    policy.MAX_CHAIN_DEPTH no matter what.
    """
    if depth is None:
        depth = policy.MAX_CHAIN_DEPTH
    if bindings is None:
        bindings = {}
    if visited is None:
        visited = frozenset()
    if _proofs is None:
        _proofs = []
    if depth < 0 or len(_proofs) >= policy.MAX_CHAIN_PROOFS:
        return _proofs
    gkey = _key({k: resolve(v, bindings) for k, v in goal.items()})
    if gkey in visited:
        return _proofs
    visited = visited | {gkey}
    # 1. KB facts first (ground truth before derivation)
    res, info = kb.fact_lookup(goal, bindings)
    if res == "ok":
        node = {"goal": dict(goal), "bindings": dict(bindings),
                "status": info.get("status", "verified"),
                "basis": [f"kb:{b}" for b in info.get("basis", [])],
                "premises": []}
        if "value" in (info or {}):
            node["value"] = info["value"]
        _proofs.append(node)
        if len(_proofs) >= policy.MAX_CHAIN_PROOFS:
            return _proofs
    elif res == "contested":
        _proofs.append({"goal": dict(goal), "bindings": dict(bindings),
                        "status": "contested",
                        "basis": [f"kb-contested:{info.get('basis', [])}"],
                        "premises": []})
        return _proofs
    if depth == 0:
        return _proofs
    # 2. rules whose conclusion unifies (concrete rules included: they
    # simply never introduce new bindings)
    for rule in kb.rules():
        if len(_proofs) >= policy.MAX_CHAIN_PROOFS:
            break
        conc = rule.get("conclusion", {}) or {}
        if not isinstance(conc, dict) or conc.get("type") != \
                goal.get("type"):
            continue
        gspec = {"type": goal.get("type")}
        for k, v in goal.items():
            if k != "type":
                gspec[k] = resolve(v, bindings)
        # goal-first order: the goal is PARTIAL (it may omit slots like
        # value that the conclusion provides). unify() tolerates keys
        # missing on the fact side only, so the partial goal must be
        # the pattern and the fuller conclusion the fact.
        b2 = unify(gspec, conc, dict(bindings))
        if b2 is None:
            continue
        if _exception_holds(rule, kb, b2):
            continue
        prem_proofs: list = []
        branch_status = "verified"
        failed = False
        for cond in rule.get("conditions", []) or []:
            sub: list = []
            prove(cond, kb, dict(b2), depth - 1, visited, sub)
            good = [p for p in sub if p.get("status") not in
                    ("contested",)]
            good = [p for p in good if _proof_value(p) != "unknown"]
            if not good:
                failed = True
                break
            prem_proofs.append(good[0])
            branch_status = _weaker(branch_status,
                                    good[0].get("status", "unknown"))
            b2.update(good[0].get("bindings", {}))
        if failed:
            continue
        inst = {k: resolve(v, b2) for k, v in conc.items()}
        node = {"goal": dict(goal), "bindings": dict(b2),
                "status": branch_status, "rule_id": rule.get("id"),
                "premises": prem_proofs,
                "basis": [f"rule:{rule.get('id')}"] +
                [f"{p.get('rule_id', 'kb')}:{_key(p.get('bindings', {}))}"
                 for p in prem_proofs]}
        if "value" in inst:
            node["value"] = inst["value"]
        elif inst.get("type") == "has_attr":
            node["value"] = None
        _proofs.append(node)
    return _proofs


def _proof_value(p: dict):
    if "value" in p:
        return p["value"]
    if p.get("status") in ("verified", "inherited(verified)",
                           "hypothesis"):
        return True
    return "unknown"


def _exception_holds(rule: dict, kb: _KB, bindings: dict) -> bool:
    for e in rule.get("exceptions", []) or []:
        res, _ = kb.fact_lookup(e, bindings)
        if res == "ok":
            return True
        if res == "contested":
            return True  # undecidable guard: skip the rule, honestly
    return False


def prove_attribute(memory, ontology, concept: str, key: str,
                    value=_MISSING) -> dict:
    """Prove (concept, key[=value]). Returns {status, value?, proofs[],
    conflicts[]}. Status in verified|hypothesis|inherited(verified)|
    contested|unknown. NEVER writes to the KB."""
    kb = _KB(memory, ontology)
    goal: dict = {"type": "has_attr", "concept": concept, "key": key}
    if value is not _MISSING:
        goal["value"] = value
    proofs = prove(goal, kb)
    by_value: dict[str, list] = {}
    for p in proofs:
        if p.get("status") == "contested":
            continue
        v = p.get("value", True)
        by_value.setdefault(_key(v), []).append(p)
    unknowns = [p for p in proofs if p.get("status") not in
                ("verified", "inherited(verified)", "hypothesis")]
    if len(by_value) > 1:
        return {"status": "contested", "value": None, "proofs": proofs,
                "conflicts": [
                    {"value": k, "proofs": v} for k, v in by_value.items()
                ] + ([{"value": "contested-premise", "proofs": unknowns}]
                     if unknowns else [])}
    if not by_value:
        cont = [p for p in proofs if p.get("status") == "contested"]
        if cont:
            return {"status": "contested", "value": None, "proofs": proofs,
                    "conflicts": [{"value": "contested-premise",
                                   "proofs": cont}]}
        return {"status": "unknown", "value": None, "proofs": proofs,
                "conflicts": []}
    (vkey, group), = by_value.items()
    status = "verified"
    for p in group:
        status = _weaker(status, p.get("status", "unknown"))
    val = group[0].get("value", True)
    try:
        val = json.loads(vkey)
    except Exception:
        pass
    return {"status": status, "value": val, "proofs": group,
            "conflicts": []}


def verify_proof(proof: dict, memory, ontology) -> dict:
    """Independently re-check ONE proof node. Returns
    {result: verified|rejected, detail}. Re-derives every premise from
    scratch, re-checks rule existence/exceptions, binding consistency
    and conclusion match. Trusts nothing the prover claims."""
    kb = _KB(memory, ontology)
    goal = proof.get("goal", {})
    bindings = dict(proof.get("bindings", {}) or {})
    if proof.get("rule_id") is None:
        # KB-leaf proof: the fact must hold right now
        res, info = kb.fact_lookup(goal, bindings)
        if res == "ok":
            if "value" in proof and "value" in (info or {}):
                if info["value"] != proof["value"]:
                    return {"result": "rejected",
                            "detail": "leaf value drifted"}
            return {"result": "verified", "detail": "KB fact holds"}
        return {"result": "rejected",
                "detail": f"KB leaf no longer holds ({res})"}
    rows = [r for r in kb.rules() if r.get("id") == proof["rule_id"]]
    if not rows:
        return {"result": "rejected", "detail": "rule not in memory"}
    rule = rows[0]
    conc = rule.get("conclusion", {}) or {}
    inst = {k: resolve(v, bindings) for k, v in conc.items()}
    for k, v in inst.items():
        if is_var(v):
            return {"result": "rejected",
                    "detail": f"unbound variable {v} in conclusion"}
        gv = resolve(goal.get(k), bindings) if k in goal else v
        if k in goal and not is_var(goal.get(k)) and gv != v:
            return {"result": "rejected",
                    "detail": f"binding contradicts goal at {k}"}
    if _exception_holds(rule, kb, bindings):
        return {"result": "rejected",
                "detail": "rule exception holds on re-check"}
    conds = rule.get("conditions", []) or []
    premises = proof.get("premises", []) or []
    if len(premises) != len(conds):
        return {"result": "rejected",
                "detail": f"premise count {len(premises)} != "
                f"{len(conds)} rule conditions (missing premises)"}
    for cond in conds:
        covered = False
        for prem in premises:
            pg = prem.get("goal", {}) or {}
            if pg.get("type") != cond.get("type"):
                continue
            match = True
            for k, v in cond.items():
                if k == "type":
                    continue
                cv = resolve(v, bindings)
                pv = resolve(pg.get(k), prem.get("bindings", {}) or {})
                if not is_var(cv) and not is_var(pv) and cv != pv:
                    match = False
                    break
            if match:
                covered = True
                break
        if not covered:
            return {"result": "rejected",
                    "detail": f"rule condition uncovered: {cond}"}
    for prem in premises:
        sub = verify_proof(prem, memory, ontology)
        if sub["result"] != "verified":
            return {"result": "rejected",
                    "detail": f"premise failed re-check: "
                    f"{sub['detail']}"}
    if "value" in proof and "value" in inst and \
            proof["value"] != inst["value"]:
        return {"result": "rejected",
                "detail": "claimed value differs from re-derivation"}
    return {"result": "verified",
            "detail": f"rule {rule.get('id')}, "
            f"{len(proof.get('premises', []) or [])} premise(s) re-derived"}
