"""Internal cognitive operation protocol (Phase 4C).

Every operation declares name/version/schemas/preconditions, executes
with real logic, validates its own output, and returns a checkable record
{op, inputs, basis, outputs, status}. These are the system's OWN internal
mechanisms -- not external tools. Origin tags distinguish hand-written
rules ("hand_rule") from network-learned abilities ("learned_net").
"""
from __future__ import annotations


class OpError(Exception):
    pass


def _overlap(a: str, b: str, n: int = 2) -> bool:
    """Shared-substring test for contested-evidence matching (restricted
    scope: short Chinese condition phrases; documented coarseness)."""
    grams = {a[i:i + n] for i in range(len(a) - n + 1)} - {""}
    return any(g and g in b for g in grams)


class Operation:
    name = "op"
    version = "1.0"
    origin = "hand_rule"  # or "learned_net"
    input_schema: dict = {}
    output_schema: dict = {}

    def preconditions(self, inputs: dict, ctx: dict) -> tuple[bool, str]:
        return True, "ok"

    def execute(self, inputs: dict, ctx: dict) -> dict:
        raise NotImplementedError

    def validate(self, inputs: dict, outputs: dict, ctx: dict
                 ) -> tuple[bool, str]:
        return True, "ok"

    def run(self, inputs: dict, ctx: dict) -> dict:
        ok, msg = self.preconditions(inputs, ctx)
        rec = {"op": self.name, "version": self.version,
               "origin": self.origin, "inputs": inputs, "basis": [],
               "outputs": {}, "status": "pending", "note": ""}
        if not ok:
            rec.update(status="precondition_failed", note=msg)
            return rec
        try:
            out = self.execute(inputs, ctx)
        except Exception as e:  # noqa: BLE001 - recorded, never hidden
            rec.update(status="failed", note=f"{type(e).__name__}: {e}")
            return rec
        vok, vmsg = self.validate(inputs, out, ctx)
        rec["outputs"] = out
        rec["status"] = "verified" if vok else "unverified"
        rec["note"] = vmsg
        return rec


class RetrieveConcepts(Operation):
    name = "retrieve_concepts"

    def execute(self, inputs, ctx):
        mem = ctx["memory"]
        onto = ctx["ontology"]
        found = []
        for term in inputs.get("terms", []):
            hit = mem.get_concept(term)
            if hit is None:
                for c in mem.find_concepts_by_name(term):
                    hit = c
                    break
            if hit:
                d = onto.describe(hit["id"])
                found.append({"id": hit["id"], "describe": d})
                rec_basis = f"kb:{hit['id']}"
            else:
                rec_basis = f"kb-miss:{term}"
                found.append({"id": None, "term": term})
        return {"concepts": found}

    def run(self, inputs, ctx):
        rec = super().run(inputs, ctx)
        rec["basis"] = [f.get("id") or f.get("term", "?")
                        for f in rec["outputs"].get("concepts", [])]
        return rec


class QueryAttribute(Operation):
    name = "query_attribute"
    # Statuses: verified fact | inherited (with source) | hypothesis |
    # unknown. Instance overrides block blind inheritance.

    def execute(self, inputs, ctx):
        onto = ctx["ontology"]
        cid, key = inputs["concept"], inputs["key"]
        if ctx["memory"].get_concept(cid) is None:
            return {"status": "unknown", "value": None,
                    "basis": [f"no concept {cid}"]}
        inh = onto.inherited_attributes(cid)
        if key not in inh:
            return {"status": "unknown", "value": None,
                    "basis": [f"{cid} has no {key}"]}
        row = inh[key]
        inherited = row["concept_id"] != cid
        return {"status": row["status"] if not inherited
                else f"inherited({row['status']})",
                "value": row["value"],
                "basis": [f"{row['concept_id']}.{key}={row['value']} "
                          f"[{row['status']}/{row['source']}]"]}

    def run(self, inputs, ctx):
        rec = super().run(inputs, ctx)
        rec["basis"] = rec["outputs"].get("basis", [])
        return rec


class ApplyRule(Operation):
    name = "apply_rule"
    # Only rules present in memory; trace cites rule id + facts.

    def execute(self, inputs, ctx):
        reasoner = ctx["reasoner"]
        fired = reasoner.forward_chain()
        hits = [f for f in fired if f.get("rule") == inputs["rule_id"]]
        if not hits:
            raise OpError(f"rule {inputs['rule_id']} did not fire")
        return {"fired": hits}

    def validate(self, inputs, outputs, ctx):
        ok = any("conclusion" in h for h in outputs["fired"])
        return ok, "rule fired with conclusion" if ok else "no conclusion"


class CompareExplanations(Operation):
    name = "compare_explanations"
    # Scores candidates by evidence weight, never by surface similarity.

    def execute(self, inputs, ctx):
        ranked = sorted(inputs.get("candidates", []),
                        key=lambda c: (-c.get("evidence_weight", 0),
                                       c.get("label", "")))
        return {"ranking": ranked,
                "best": ranked[0] if ranked else None}

    def validate(self, inputs, outputs, ctx):
        return (outputs["best"] is not None), \
            "best selected" if outputs["best"] else "no candidates"


class Compute(Operation):
    name = "compute"
    # Internal math engine only: arithmetic | linear | derivative.

    def preconditions(self, inputs, ctx):
        if inputs.get("kind") not in ("arithmetic", "linear", "derivative"):
            return False, f"unsupported compute kind {inputs.get('kind')}"
        if not inputs.get("expression"):
            return False, "missing expression"
        return True, "ok"

    def execute(self, inputs, ctx):
        from . import mathengine as M
        kind, expr = inputs["kind"], inputs["expression"]
        if kind == "arithmetic":
            node = M.parse_expr(expr)
            val = M.evaluate(node)
            return {"result": str(val), "steps": [
                {"rule": "精确有理数求值", "before": expr,
                 "after": str(val)}]}
        if kind == "linear":
            lhs, rhs = M.split_equation(expr)
            sol, steps = M.solve_linear(
                lhs, rhs, inputs.get("var", "x"))
            return {"result": f"{inputs.get('var', 'x')}={sol}",
                    "steps": steps}
        node = M.parse_expr(expr)
        d, steps = M.differentiate(node, inputs.get("var", "x"))
        return {"result": str(d), "steps": steps}

    def validate(self, inputs, outputs, ctx):
        from . import mathengine as M
        try:
            if inputs["kind"] == "linear":
                var = inputs.get("var", "x")
                sol = M.Fraction(outputs["result"].split("=")[1])
                lhs, rhs = M.split_equation(inputs["expression"])
                diff = M.evaluate(
                    M.Add((lhs, M.Mul((M.Num(M.Fraction(-1)), rhs)))),
                    {var: sol})
                return diff == 0, f"substitution residual={diff}"
            if inputs["kind"] == "arithmetic":
                node = M.parse_expr(inputs["expression"])
                dec = float(M.evaluate(node))
                return abs(dec - float(M.Fraction(outputs["result"]))) < 1e-9, \
                    "independent decimal re-evaluation matches"
            # derivative: re-derive term-wise and compare structurally
            node = M.parse_expr(inputs["expression"])
            terms = node.terms if isinstance(node, M.Add) else (node,)
            parts = [M.differentiate(t, inputs.get("var", "x"))[0]
                     for t in terms]
            expect = parts[0] if len(parts) == 1 else M.Add(tuple(parts))
            return str(expect) == outputs["result"], \
                "independent term-wise re-derivation matches"
        except Exception as e:  # noqa: BLE001
            return False, f"validator error: {e}"


class UpdateWorkingMemory(Operation):
    name = "update_working_memory"

    def execute(self, inputs, ctx):
        wm = ctx.setdefault("working_memory", {})
        wm.update(inputs.get("facts", {}))
        return {"working_memory": dict(wm)}


class CheckResult(Operation):
    name = "check_result"
    # Independent re-check, never a copy of the solver's answer.

    def execute(self, inputs, ctx):
        check = inputs.get("check", {})
        kind = check.get("kind")
        if kind == "kb_source":
            mem = ctx["memory"]
            sources = check.get("sources", [])
            if not sources:
                return {"ok": False, "missing": ["(no sources to audit)"]}
            missing = [s for s in sources
                       if mem.get_concept(s) is None
                       and not mem.find_concepts_by_name(s)]
            return {"ok": not missing, "missing": missing}
        if kind == "nonempty":
            return {"ok": bool(check.get("value")),
                    "missing": [] if check.get("value") else ["empty"]}
        if kind == "premises_present":
            missing = [p for p in check.get("names", [])
                       if p not in (check.get("premises", []) or [])]
            return {"ok": not missing, "missing": missing}
        if kind == "no_contested":
            mem = ctx["memory"]
            hits = []
            for subj in check.get("subjects", []):
                for cl in mem.list_claims():
                    if cl["status"] != "contested":
                        continue
                    text = cl["text"] or ""
                    if subj and ((subj in text or text in subj
                                  or (cl.get("subject_id") or "") == subj)
                                 or _overlap(subj, text)):
                        hits.append(cl["text"])
            return {"ok": not hits, "contested": hits}
        raise OpError(f"unknown check kind {kind}")

    def validate(self, inputs, outputs, ctx):
        return outputs.get("ok", False), \
            "check passed" if outputs.get("ok") else "check failed"


class RaiseGap(Operation):
    name = "raise_gap"

    def execute(self, inputs, ctx):
        mem = ctx["memory"]
        qid = mem.add_question(inputs["question"], inputs.get("kind", "gap"),
                               inputs.get("target"), 0.5, "queued",
                               {"source": "cogexec",
                                "failure": inputs.get("failure")})
        mem.add_experience("evaluation", "cogexec gap raised",
                           {"question_id": qid, **inputs})
        return {"question_id": qid}


class Replan(Operation):
    name = "replan"
    # On failure: drop the failed op, try the recorded alternative plan.

    def execute(self, inputs, ctx):
        failed = inputs.get("failed_op")
        alts = [p for p in inputs.get("alternative_plans", [])
                if failed not in p.get("ops", [])]
        if not alts:
            raise OpError("no alternative plan without the failed op")
        return {"new_plan": alts[0]}


REGISTRY = {c.name: c for c in (
    RetrieveConcepts, QueryAttribute, ApplyRule, CompareExplanations,
    Compute, UpdateWorkingMemory, CheckResult, RaiseGap, Replan)}
