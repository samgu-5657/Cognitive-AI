"""Explicit learning-cycle driver (Phase 6A, Task 5).

gap -> hypothesize (induce) -> compare +/- -> validate independently ->
activate (only with explicit confirmation) -> test unseen -> keep/reject.

Every run appends a ledger experience recording outcome kind
(fact | concept | rule | composition), source, conditions, successes,
failures and versions. User-taught rules, system candidates and active
abilities are stored in their separate statuses, never merged.
"""
from __future__ import annotations


def run_cycle(memory, ontology, name: str, domain: str,
              examples: list, negatives: list | None = None,
              heldout: list | None = None, auto_activate: bool = False,
              reviewer: str = "operator") -> dict:
    """One full, auditable learning cycle. auto_activate=False keeps the
    result at validated: only an explicit confirmation (human test harness
    passing True, or the review UI) may activate. Returns the ledger."""
    from . import rulelearn as RL
    negatives = negatives or []
    heldout = heldout or []
    ledger: dict = {"name": name, "domain": domain, "outcome": None,
                    "stages": {}}
    rep = RL.induce(examples, negatives, domain=domain)
    ledger["stages"]["hypothesize"] = {
        "candidates": len(rep["candidates"]), "ambiguous": rep["ambiguous"],
        "failures": rep.get("failures", [])}
    if not rep["candidates"]:
        ledger["outcome"] = "insufficient"
        ledger["reason"] = rep.get("failures", [])
        memory.add_experience("evaluation", f"learn cycle {name}: "
                                            "insufficient evidence",
                              {"op": "learn_cycle", "outcome": "insufficient",
                               "ledger": ledger})
        return ledger
    if rep["ambiguous"]:
        ledger["outcome"] = "needs_choice"
        ledger["reason"] = "multiple consistent candidates"
        memory.add_experience("evaluation", f"learn cycle {name}: "
                                            "ambiguous, awaiting choice",
                              {"op": "learn_cycle", "outcome": "needs_choice",
                               "ledger": ledger})
        return ledger
    cand = rep["candidates"][0]
    rid = RL.new_rule_id(name)
    # ids must stay unique even when names slug identically
    # (non-ASCII names); ASCII names keep their stable ids
    base, n = rid, 2
    while memory.get_rule(rid) is not None:
        rid = f"{base}_{n}"
        n += 1
    mem_rec = RL.make_proposal_record(rid, name, domain, [cand],
                                      evidence=[{"reviewer": reviewer}])
    mem_rec["spec"]["selected"] = 0
    memory.save_rule(mem_rec)
    ledger["stages"]["candidate"] = rid
    val = RL.validate_rule(cand["spec"], examples, heldout, negatives,
                           ontology)
    ledger["stages"]["validate"] = val["checks"]
    mem_rec["validation"] = {**val, "selected": 0}
    memory.save_rule(mem_rec)
    if not val["passed"]:
        RL.promote_record(memory, rid, "rejected")
        memory.add_experience("evaluation", f"learn cycle {name}: rejected",
                              {"op": "learn_cycle", "outcome": "rejected",
                               "ledger": ledger})
        ledger["outcome"] = "rejected"
        return ledger
    assert RL.promote_record(memory, rid, "validated",
                             validation=mem_rec["validation"])["ok"]
    ledger["outcome"] = "rule"
    if auto_activate:
        ok = RL.promote_record(memory, rid, "active")["ok"]
        ledger["stages"]["activate"] = ok
        ledger["outcome"] = "rule-active" if ok else "rule"
    memory.add_experience("evaluation", f"learn cycle {name}: "
                                        f"{ledger['outcome']}",
                          {"op": "learn_cycle", "outcome": "rule",
                           "rule_id": rid, "ledger": ledger})
    return ledger
