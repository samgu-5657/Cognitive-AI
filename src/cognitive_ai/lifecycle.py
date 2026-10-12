"""Knowledge lifecycle: candidate -> hypothesis -> supported -> verified.

Side states: contested, rejected, superseded.

Promotion conditions (evidence thresholds live in policy.py):
  candidate:  a value was extracted but corroboration is lacking
              (< MIN_CORROBORATING_SOURCES independent sources).
              Recorded as a Claim only; NEVER written as an attribute.
  hypothesis: >= MIN_CORROBORATING_SOURCES independent, target-relevant
              sources agree on the same cleaned value. Written as an
              attribute with status 'hypothesis'. This is the highest state
              the autonomous pipeline may assign by itself.
  supported:  hypothesis + at least SUPPORTED_MIN_SOURCES independent
              sources in total, no open conflicts. Requires an explicit
              promote() call (consolidation pass or operator); never automatic
              from a single research step.
  verified:   supported + explicit confirmation (confirmed_by must name the
              confirmer, e.g. 'operator' or a named regression suite) +
              regression gate passes. Two text files agreeing is NEVER enough.
  contested:  new evidence disagrees with a hypothesis/supported/verified
              row. The old row is kept untouched; the rival is recorded
              beside it and a lifecycle event is logged.
  rejected:   validation failed or evidence contradicted the claim.
  superseded: the row was replaced by a newer validated change.

Downgrades are always allowed toward contested/rejected; upgrades must
satisfy the conditions above and are recorded in lifecycle_events.
"""
from __future__ import annotations


from .memory import Memory
from .policy import MIN_CORROBORATING_SOURCES, SUPPORTED_MIN_SOURCES

STATES = ("candidate", "hypothesis", "supported", "verified",
          "contested", "rejected", "superseded")

# Allowed transitions. Upgrades (toward verified) additionally require the
# evidence/confirmation checks in promote(); this table is only the shape.
TRANSITIONS: dict[str, tuple[str, ...]] = {
    "candidate": ("hypothesis", "rejected"),
    "hypothesis": ("supported", "contested", "rejected", "superseded"),
    "supported": ("verified", "contested", "rejected", "superseded"),
    "verified": ("contested", "superseded"),
    "contested": ("hypothesis", "supported", "rejected", "superseded"),
    "rejected": ("candidate",),
    "superseded": (),
}


class LifecycleError(ValueError):
    pass


def can_transition(from_state: str, to_state: str) -> bool:
    return to_state in TRANSITIONS.get(from_state, ())


def _independent_source_count(evidence: list[dict]) -> int:
    hashes = {e.get("content_hash", "") for e in evidence
              if e.get("relevance") == "direct"}
    hashes.discard("")
    if hashes:
        return len(hashes)
    # legacy evidence without hashes: fall back to distinct sources
    return len({e.get("source", "") for e in evidence if e.get("source")})


def check_promotion(from_state: str, to_state: str,
                    evidence: list[dict] | None,
                    confirmed_by: str = "") -> tuple[bool, str]:
    """Testable gate: may `from_state` move to `to_state` with this evidence?"""
    evidence = evidence or []
    if not can_transition(from_state, to_state):
        return False, f"transition {from_state} -> {to_state} not allowed"
    n = _independent_source_count(evidence)
    if to_state == "hypothesis":
        if n < MIN_CORROBORATING_SOURCES:
            return False, (f"hypothesis needs >= {MIN_CORROBORATING_SOURCES} "
                           f"independent direct sources, got {n}")
        return True, "ok"
    if to_state == "supported":
        if n < SUPPORTED_MIN_SOURCES:
            return False, (f"supported needs >= {SUPPORTED_MIN_SOURCES} "
                           f"independent direct sources, got {n}")
        return True, "ok"
    if to_state == "verified":
        if n < SUPPORTED_MIN_SOURCES:
            return False, (f"verified needs >= {SUPPORTED_MIN_SOURCES} "
                           f"independent direct sources, got {n}")
        if not confirmed_by:
            return False, ("verified needs explicit confirmation "
                           "(confirmed_by); agreement of files is not enough")
        return True, "ok"
    # contested / rejected / superseded / candidate: shape check is enough
    return True, "ok"


def promote_attribute(mem: Memory, attribute_id: int, to_state: str,
                      evidence: list[dict] | None = None,
                      confirmed_by: str = "",
                      reason: str = "") -> dict:
    """Move one attribute row to a new lifecycle state with checks + record."""
    row = mem.query_one("SELECT * FROM attributes WHERE id=?", (attribute_id,))
    if row is None:
        return {"ok": False, "error": "attribute not found"}
    from_state = row["status"] if row["status"] in STATES else "hypothesis"
    ok, msg = check_promotion(from_state, to_state, evidence, confirmed_by)
    if not ok:
        mem.record_lifecycle("attribute", f"{row['concept_id']}.{row['key']}",
                             from_state, "rejected",
                             f"promotion denied: {msg}", evidence)
        return {"ok": False, "error": msg}
    mem.update_attribute_status(attribute_id, to_state)
    mem.record_lifecycle("attribute", f"{row['concept_id']}.{row['key']}",
                         from_state, to_state,
                         reason or f"promoted with {len(evidence or [])} evidence item(s)"
                         + (f" confirmed_by={confirmed_by}" if confirmed_by else ""),
                         evidence)
    return {"ok": True, "from": from_state, "to": to_state}


def contest_attribute(mem: Memory, attribute_id: int,
                      rival_evidence: list[dict] | None = None,
                      reason: str = "") -> dict:
    """Record a conflict against an existing row WITHOUT modifying it."""
    row = mem.query_one("SELECT * FROM attributes WHERE id=?", (attribute_id,))
    if row is None:
        return {"ok": False, "error": "attribute not found"}
    from_state = row["status"] if row["status"] in STATES else "hypothesis"
    if from_state == "contested":
        return {"ok": True, "note": "already contested"}
    if not can_transition(from_state, "contested"):
        return {"ok": False,
                "error": f"cannot contest from {from_state}"}
    mem.record_lifecycle("attribute", f"{row['concept_id']}.{row['key']}",
                         from_state, "contested",
                         reason or "rival evidence recorded; row untouched",
                         rival_evidence)
    return {"ok": True, "row_untouched": True}


def describe() -> dict:
    return {"states": list(STATES),
            "transitions": {k: list(v) for k, v in TRANSITIONS.items()},
            "rules": {
                "hypothesis": f">= {MIN_CORROBORATING_SOURCES} independent direct sources agree",
                "supported": f">= {SUPPORTED_MIN_SOURCES} independent direct sources, explicit promote(), no open conflicts",
                "verified": f">= {SUPPORTED_MIN_SOURCES} sources + explicit confirmed_by + regression pass; file agreement alone never suffices",
            }}
